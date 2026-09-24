"""The core RAG engine and LLM factory.

Ties together query rewriting, hybrid retrieval, optional reranking, context
construction and grounded answer generation with citations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import MAX_HISTORY_TURNS, get_env_api_key
from rag.prompts import FALLBACK_ANSWER, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from rag.retriever import HybridRetriever, RetrievalResult
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_CONTEXT_CHARS: int = 14_000
FALLBACK_PHRASES: tuple[str, ...] = (
    "couldn't find",
    "could not find",
    "not in the provided context",
    "no relevant information",
    "insufficient information",
)


class LLMConfigurationError(Exception):
    """Raised when the LLM cannot be constructed."""


@dataclass
class Source:
    """A single citation shown to the user."""

    index: int
    source: str
    file_type: str
    page: str
    chunk_id: str
    document_id: str
    text: str
    score: float = 0.0

    def label(self) -> str:
        """Return a short human-readable citation label."""
        return f"{self.source} — {self.page}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "index": self.index,
            "source": self.source,
            "file_type": self.file_type,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "label": self.label(),
        }


@dataclass
class RAGResponse:
    """The full result of a RAG query."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    rewritten_query: str = ""
    intent: str = "DOCUMENT_QA"
    latency: float = 0.0
    used_fallback: bool = False
    retrieved_count: int = 0
    reranked: bool = False
    used_hybrid: bool = False


def build_llm(
    api_key: str | None,
    model: str,
    temperature: float = 0.2,
) -> Any:
    """Construct a ``ChatGoogleGenerativeAI`` client.

    Args:
        api_key: Google API key. Falls back to the environment.
        model: Gemini model name.
        temperature: Sampling temperature.

    Returns:
        A configured chat model.

    Raises:
        LLMConfigurationError: If the key is missing or the model cannot load.
    """
    resolved_key = (api_key or get_env_api_key()).strip()
    if not resolved_key:
        raise LLMConfigurationError(
            "No Google API key found. Enter one in the sidebar or set GOOGLE_API_KEY in .env."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=resolved_key,
            convert_system_message_to_human=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMConfigurationError(f"Failed to initialise the Gemini model: {exc}") from exc


def _message_text(message: Any) -> str:
    """Extract plain text from a LangChain message or raw response."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def history_to_text(history: list[dict[str, Any]], limit: int = MAX_HISTORY_TURNS) -> str:
    """Render chat history as plain text for prompting.

    Args:
        history: List of ``{"role": ..., "content": ...}`` dictionaries.
        limit: Maximum number of turns to include.

    Returns:
        A newline-separated transcript.
    """
    if not history:
        return "(no previous conversation)"
    lines = [
        f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
        for turn in history[-limit:]
    ]
    return "\n".join(lines)


class RAGEngine:
    """Coordinate retrieval and grounded answer generation.

    Args:
        llm: A LangChain chat model.
        retriever: A :class:`~rag.retriever.HybridRetriever` instance.
        reranker: Optional reranker with a ``rerank`` method.
        top_k: Number of chunks passed to the LLM after reranking.
        max_context_chars: Hard cap on the context size.
    """

    def __init__(
        self,
        llm: Any,
        retriever: HybridRetriever,
        reranker: Any | None = None,
        *,
        top_k: int = 5,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------ #
    # Query processing
    # ------------------------------------------------------------------ #
    def rewrite_query(self, question: str, history: list[dict[str, Any]]) -> str:
        """Rewrite a follow-up question into a standalone search query.

        Falls back to the original question if history is empty or the LLM
        fails.
        """
        if not history:
            return question
        try:
            chain = QUERY_REWRITE_PROMPT | self.llm
            response = chain.invoke(
                {
                    "history": history_to_text(history),
                    "question": question,
                }
            )
            rewritten = _message_text(response).strip().strip('"').strip()
            if rewritten and len(rewritten) < 500:
                logger.info("Rewrote query: '%s' -> '%s'", question, rewritten)
                return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewriting failed: %s", exc)
        return question

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve candidates and optionally rerank them."""
        result = self.retriever.retrieve(query)
        if self.reranker is not None and not result.is_empty:
            reranked = self.reranker.rerank(query, result.documents, top_n=self.top_k)
            result.documents = reranked
        else:
            result.documents = result.documents[: self.top_k]
        return result

    # ------------------------------------------------------------------ #
    # Context + sources
    # ------------------------------------------------------------------ #
    def build_context(self, documents: list[Document]) -> str:
        """Build the context string injected into the system prompt.

        Each block is labelled with its source so the model can cite it.
        """
        blocks: list[str] = []
        used = 0
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            header = (
                f"[Source {index}] {metadata.get('source', 'unknown')} — "
                f"{metadata.get('page', '?')} "
                f"(type: {metadata.get('file_type', '?')}, "
                f"chunk: {metadata.get('chunk_id', '?')})"
            )
            block = f"{header}\n{document.page_content}"
            if used + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used
                if remaining > 200:
                    blocks.append(block[:remaining])
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    @staticmethod
    def build_sources(documents: list[Document], scores: list[float] | None = None) -> list[Source]:
        """Convert retrieved documents into de-duplicated citations."""
        sources: list[Source] = []
        seen: set[str] = set()
        for position, document in enumerate(documents, start=1):
            metadata = document.metadata
            chunk_id = str(metadata.get("chunk_id", f"chunk-{position}"))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            score = scores[position - 1] if scores and position - 1 < len(scores) else 0.0
            sources.append(
                Source(
                    index=len(sources) + 1,
                    source=str(metadata.get("source", "unknown")),
                    file_type=str(metadata.get("file_type", "?")),
                    page=str(metadata.get("page", "?")),
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    text=document.page_content,
                    score=float(score),
                )
            )
        return sources

    @staticmethod
    def _looks_like_fallback(answer: str) -> bool:
        """Heuristically detect that the model could not answer."""
        lowered = answer.lower()
        return any(phrase in lowered for phrase in FALLBACK_PHRASES)

    # ------------------------------------------------------------------ #
    # Answer generation
    # ------------------------------------------------------------------ #
    def answer(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        *,
        intent: str = "DOCUMENT_QA",
        rewrite: bool = True,
    ) -> RAGResponse:
        """Answer a question using only retrieved document context.

        Args:
            question: The user's question.
            history: Conversation history for follow-up resolution.
            intent: Pre-computed intent (for analytics).
            rewrite: Whether to perform query rewriting.

        Returns:
            A :class:`RAGResponse` with the grounded answer and citations.
        """
        history = history or []
        start = time.perf_counter()

        rewritten = self.rewrite_query(question, history) if rewrite else question
        retrieval = self.retrieve(rewritten)

        if retrieval.is_empty:
            return RAGResponse(
                answer=FALLBACK_ANSWER,
                intent=intent,
                latency=time.perf_counter() - start,
                used_fallback=True,
                rewritten_query=rewritten,
                used_hybrid=retrieval.used_hybrid,
            )

        context = self.build_context(retrieval.documents)
        sources = self.build_sources(retrieval.documents, retrieval.scores)

        messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue
            messages.append(
                HumanMessage(content=content) if role == "user" else AIMessage(content=content)
            )
        messages.append(HumanMessage(content=question))

        try:
            response = self.llm.invoke(messages)
            answer_text = _message_text(response).strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM invocation failed")
            raise LLMConfigurationError(
                "The Gemini API request failed. Check your API key, quota and network connection."
            ) from exc

        used_fallback = not answer_text or self._looks_like_fallback(answer_text)
        if used_fallback:
            answer_text = FALLBACK_ANSWER

        return RAGResponse(
            answer=answer_text,
            sources=[] if used_fallback else sources,
            documents=retrieval.documents,
            rewritten_query=rewritten,
            intent=intent,
            latency=time.perf_counter() - start,
            used_fallback=used_fallback,
            retrieved_count=len(retrieval.documents),
            reranked=self.reranker is not None,
            used_hybrid=retrieval.used_hybrid,
        )
