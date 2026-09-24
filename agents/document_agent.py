"""Agentic orchestration for the assistant.

The :class:`DocumentAgent` classifies each question and dispatches it to the
most appropriate capability: grounded document Q&A, deterministic data
analysis, summarisation or document comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from agents.data_analyst import DataAnalyst, Dataset
from rag.chain import RAGEngine, Source, _message_text
from rag.prompts import COMPARISON_PROMPT, FALLBACK_ANSWER
from rag.query_router import (
    COMPARISON,
    DATA_ANALYSIS,
    DOCUMENT_QA,
    SUMMARY,
    classify_intent,
)
from utils.logger import get_logger

logger = get_logger(__name__)

COMPARISON_DOCS_PER_SIDE: int = 8
COMPARISON_CONTEXT_CHARS: int = 16_000


@dataclass
class AgentResponse:
    """Unified response returned by the agent for any intent."""

    answer: str
    intent: str = DOCUMENT_QA
    sources: list[Source] = field(default_factory=list)
    latency: float = 0.0
    details: str = ""
    used_fallback: bool = False
    retrieved_count: int = 0
    used_llm: bool = True
    rewritten_query: str = ""


class DocumentAgent:
    """Route questions to the correct capability and return a uniform result.

    Args:
        llm: A LangChain chat model.
        rag_engine: The RAG engine used for document Q&A.
        data_analyst: The pandas analysis engine.
        summarizer: The map-reduce summariser.
        vectorstore: The vector store manager (for document selection).
    """

    def __init__(
        self,
        llm: Any,
        rag_engine: RAGEngine,
        data_analyst: DataAnalyst,
        summarizer: Any,
        vectorstore: Any,
    ) -> None:
        self.llm = llm
        self.rag_engine = rag_engine
        self.data_analyst = data_analyst
        self.summarizer = summarizer
        self.vectorstore = vectorstore

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _documents_for(self, document_ids: list[str], per_doc: int | None = None) -> list[Document]:
        """Return stored chunks for the given document ids.

        Args:
            document_ids: Document identifiers to include.
            per_doc: Optional maximum number of chunks per document.

        Returns:
            A list of documents in stable order.
        """
        allowed = set(document_ids)
        documents = [
            chunk
            for chunk in self.vectorstore.all_chunks()
            if str(chunk.metadata.get("document_id")) in allowed
        ]
        if per_doc is None:
            return documents

        grouped: dict[str, list[Document]] = {}
        for document in documents:
            key = str(document.metadata.get("document_id"))
            grouped.setdefault(key, []).append(document)
        limited: list[Document] = []
        for chunks in grouped.values():
            limited.extend(chunks[:per_doc])
        return limited

    def _document_names(self, document_ids: list[str]) -> dict[str, str]:
        """Map document ids to their source names."""
        names: dict[str, str] = {}
        for record in self.vectorstore.get_documents():
            names[str(record.get("document_id"))] = str(record.get("source", "document"))
        for document_id in document_ids:
            names.setdefault(document_id, document_id)
        return names

    @staticmethod
    def _summary_style(question: str) -> str:
        """Infer the requested summary style from the question."""
        lowered = question.lower()
        if "bullet" in lowered or "bullets" in lowered or "points" in lowered:
            return "bullets"
        if "short" in lowered or "brief" in lowered or "one line" in lowered:
            return "short"
        return "detailed"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def handle(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        *,
        selected_document_ids: list[str] | None = None,
        datasets: list[Dataset] | None = None,
        intent: str | None = None,
        use_llm_router: bool = False,
    ) -> AgentResponse:
        """Process a user question end-to-end.

        Args:
            question: The user's question.
            history: Conversation history.
            selected_document_ids: Documents the user restricted the query to.
            datasets: Structured datasets available for analysis.
            intent: Optional pre-computed intent.
            use_llm_router: Whether to allow LLM-based routing.

        Returns:
            An :class:`AgentResponse`.
        """
        history = history or []
        datasets = datasets or []
        selected_document_ids = selected_document_ids or []
        start = time.perf_counter()

        resolved_intent = intent or classify_intent(
            question,
            has_structured_data=bool(datasets),
            llm=self.llm,
            use_llm=use_llm_router,
        )

        try:
            if resolved_intent == DATA_ANALYSIS and datasets:
                response = self._handle_data(question, datasets, selected_document_ids)
            elif resolved_intent == SUMMARY:
                response = self._handle_summary(question, selected_document_ids)
            elif resolved_intent == COMPARISON:
                response = self._handle_comparison(
                    question, selected_document_ids, history
                )
            else:
                response = self._handle_document_qa(question, history, selected_document_ids)
        except Exception as exc:  # noqa: BLE001 - never leak tracebacks to users
            logger.exception("Agent handling failed for intent %s", resolved_intent)
            response = AgentResponse(
                answer=f"Something went wrong while processing your request: {exc}",
                intent=resolved_intent,
                used_fallback=True,
            )

        response.intent = resolved_intent
        response.latency = time.perf_counter() - start
        return response

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #
    def _handle_document_qa(
        self,
        question: str,
        history: list[dict[str, Any]],
        selected_document_ids: list[str],
    ) -> AgentResponse:
        """Answer a document question with the RAG engine."""
        if self.vectorstore.is_empty:
            return AgentResponse(
                answer="No documents are indexed yet. Upload and index documents first.",
                intent=DOCUMENT_QA,
                used_fallback=True,
            )

        if selected_document_ids:
            self.rag_engine.retriever.filter_document_ids = selected_document_ids
        else:
            self.rag_engine.retriever.filter_document_ids = None

        result = self.rag_engine.answer(question, history, intent=DOCUMENT_QA)
        return AgentResponse(
            answer=result.answer,
            intent=DOCUMENT_QA,
            sources=result.sources,
            details="",
            used_fallback=result.used_fallback,
            retrieved_count=result.retrieved_count,
            rewritten_query=result.rewritten_query,
        )

    def _handle_data(
        self,
        question: str,
        datasets: list[Dataset],
        selected_document_ids: list[str],
    ) -> AgentResponse:
        """Answer a data question using pandas."""
        scoped = (
            [dataset for dataset in datasets if dataset.document_id in set(selected_document_ids)]
            if selected_document_ids
            else datasets
        )
        if not scoped:
            scoped = datasets

        result = self.data_analyst.analyze(question, scoped)
        return AgentResponse(
            answer=result.answer,
            intent=DATA_ANALYSIS,
            details=result.details,
            used_llm=result.used_llm,
            used_fallback=False,
            retrieved_count=0,
        )

    def _handle_summary(
        self,
        question: str,
        selected_document_ids: list[str],
    ) -> AgentResponse:
        """Summarise the selected documents (or all of them)."""
        if self.vectorstore.is_empty:
            return AgentResponse(
                answer="No documents are indexed yet. Upload and index documents first.",
                intent=SUMMARY,
                used_fallback=True,
            )

        document_ids = selected_document_ids or [
            str(record.get("document_id")) for record in self.vectorstore.get_documents()
        ]
        documents = self._documents_for(document_ids)
        if not documents:
            return AgentResponse(
                answer=FALLBACK_ANSWER, intent=SUMMARY, used_fallback=True
            )

        names = self._document_names(document_ids)
        label = ", ".join(sorted(set(names.values())))
        summary = self.summarizer.summarize(
            documents,
            style=self._summary_style(question),
            source_name=label,
        )
        sources = self.rag_engine.build_sources(documents)
        return AgentResponse(
            answer=summary,
            intent=SUMMARY,
            sources=sources,
            retrieved_count=len(documents),
        )

    def _handle_comparison(
        self,
        question: str,
        selected_document_ids: list[str],
        history: list[dict[str, Any]],
    ) -> AgentResponse:
        """Compare two or more documents using grounded context."""
        document_ids = selected_document_ids or [
            str(record.get("document_id")) for record in self.vectorstore.get_documents()
        ]
        if len(document_ids) < 2:
            return AgentResponse(
                answer=(
                    "Please select at least two documents to compare "
                    "(use the document selector in the sidebar)."
                ),
                intent=COMPARISON,
                used_fallback=True,
            )

        documents = self._documents_for(document_ids, per_doc=COMPARISON_DOCS_PER_SIDE)
        if not documents:
            return AgentResponse(
                answer=FALLBACK_ANSWER, intent=COMPARISON, used_fallback=True
            )

        names = self._document_names(document_ids)
        ordered_names = [names.get(doc_id, doc_id) for doc_id in document_ids]
        left_name = ordered_names[0]
        right_name = ordered_names[1] if len(ordered_names) > 1 else "Document B"

        context = self.rag_engine.build_context(documents)
        context = context[:COMPARISON_CONTEXT_CHARS]

        try:
            chain = COMPARISON_PROMPT | self.llm
            response = chain.invoke(
                {
                    "question": question,
                    "context": context,
                    "left_name": left_name,
                    "right_name": right_name,
                }
            )
            answer = _message_text(response).strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Comparison failed")
            return AgentResponse(
                answer=f"Comparison failed: {exc}",
                intent=COMPARISON,
                used_fallback=True,
            )

        sources = self.rag_engine.build_sources(documents)
        return AgentResponse(
            answer=answer,
            intent=COMPARISON,
            sources=sources,
            retrieved_count=len(documents),
        )
