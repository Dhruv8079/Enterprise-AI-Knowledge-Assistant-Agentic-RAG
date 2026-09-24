"""Hierarchical (map-reduce) document summarisation.

Large documents are summarised section-by-section and then merged, which keeps
each LLM call within a safe context window.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chain import _message_text
from rag.prompts import SUMMARIZE_MAP_PROMPT, SUMMARIZE_REDUCE_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

MAP_CHUNK_SIZE: int = 8_000
MAP_CHUNK_OVERLAP: int = 200
MAX_REDUCE_CHARS: int = 24_000

STYLE_LABELS: dict[str, str] = {
    "short": "short, 2-3 sentence",
    "detailed": "detailed, well-structured",
    "bullets": "bullet-point",
}


class Summarizer:
    """Map-reduce summariser for one or more documents.

    Args:
        llm: A LangChain chat model.
        chunk_size: Character budget for each map step.
    """

    def __init__(self, llm: Any, chunk_size: int = MAP_CHUNK_SIZE) -> None:
        self.llm = llm
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=MAP_CHUNK_OVERLAP,
        )

    def _map(self, source: str, text: str) -> str:
        """Summarise a single section."""
        chain = SUMMARIZE_MAP_PROMPT | self.llm
        response = chain.invoke({"source": source, "text": text})
        return _message_text(response).strip()

    def _reduce(self, partials: list[str], style: str) -> str:
        """Merge partial summaries into the final summary."""
        joined = "\n\n".join(partials)
        if len(joined) > MAX_REDUCE_CHARS:
            joined = joined[:MAX_REDUCE_CHARS]
        chain = SUMMARIZE_REDUCE_PROMPT | self.llm
        response = chain.invoke(
            {
                "summaries": joined,
                "style": STYLE_LABELS.get(style, STYLE_LABELS["detailed"]),
            }
        )
        return _message_text(response).strip()

    def summarize(
        self,
        documents: list[Document],
        *,
        style: str = "detailed",
        source_name: str = "document",
    ) -> str:
        """Summarise a list of documents.

        Args:
            documents: Chunks or pages belonging to the document(s).
            style: One of ``short``, ``detailed`` or ``bullets``.
            source_name: Label used in prompts.

        Returns:
            The final summary text.

        Raises:
            ValueError: If there is no content to summarise.
        """
        if not documents:
            raise ValueError("No documents provided for summarisation.")

        combined = "\n\n".join(
            f"[{doc.metadata.get('source', source_name)} — {doc.metadata.get('page', '?')}]\n"
            f"{doc.page_content}"
            for doc in documents
        )
        combined = combined.strip()
        if not combined:
            raise ValueError("The selected documents contain no text to summarise.")

        sections = (
            self.splitter.split_text(combined)
            if len(combined) > self.chunk_size
            else [combined]
        )
        logger.info("Summarising %d section(s) for '%s'", len(sections), source_name)

        partials = [self._map(source_name, section) for section in sections]
        return self._reduce(partials, style)
