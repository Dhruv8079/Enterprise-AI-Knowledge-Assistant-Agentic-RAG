"""Hybrid retrieval: semantic (FAISS) + keyword (BM25) fusion.

The two ranked lists are combined with Reciprocal Rank Fusion (RRF), which is
robust to score-scale differences. The keyword component degrades gracefully
when ``rank-bm25`` is not installed, leaving pure semantic search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_K: int = 60


@dataclass
class RetrievalResult:
    """Outcome of a retrieval operation."""

    documents: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    vector_hits: int = 0
    keyword_hits: int = 0
    used_hybrid: bool = False

    @property
    def is_empty(self) -> bool:
        """Whether no documents were retrieved."""
        return not self.documents


def _tokenize(text: str) -> list[str]:
    """Tokenise text for BM25 indexing."""
    return _TOKEN_RE.findall(text.lower())


class _BM25Index:
    """Thin wrapper around ``rank_bm25.BM25Okapi`` with graceful fallback."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            corpus = [_tokenize(doc.page_content) for doc in documents]
            if corpus:
                self._bm25 = BM25Okapi(corpus)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.warning("BM25 keyword search unavailable: %s", exc)
            self._bm25 = None

    @property
    def available(self) -> bool:
        """Whether a BM25 index could be built."""
        return self._bm25 is not None

    def search(self, query: str, limit: int) -> list[Document]:
        """Return the top ``limit`` documents by BM25 score."""
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [doc for doc, score in ranked[:limit] if score > 0]


class HybridRetriever:
    """Combine vector and keyword search then fuse with RRF.

    Args:
        vectorstore: A :class:`~rag.vectorstore.VectorStoreManager`.
        k: Number of documents to return after fusion.
        fetch_k: Number of candidates fetched from each backend.
        use_hybrid: Whether to combine keyword search with vector search.
        filter_document_ids: Optional restriction to specific documents.
    """

    def __init__(
        self,
        vectorstore,
        *,
        k: int = 5,
        fetch_k: int = 20,
        use_hybrid: bool = True,
        filter_document_ids: list[str] | None = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.k = max(1, k)
        self.fetch_k = max(self.k, fetch_k)
        self.use_hybrid = use_hybrid
        self.filter_document_ids = filter_document_ids
        self._bm25: _BM25Index | None = None

    def _get_bm25(self) -> _BM25Index:
        """Build (or reuse) the BM25 index over all stored chunks."""
        documents = self.vectorstore.all_chunks()
        if self._bm25 is None or len(self._bm25.documents) != len(documents):
            self._bm25 = _BM25Index(documents)
        return self._bm25

    def _vector_search(self, query: str) -> list[Document]:
        results = self.vectorstore.similarity_search(
            query,
            k=self.fetch_k,
            filter_document_ids=self.filter_document_ids,
            fetch_k=max(self.fetch_k * 3, 50),
        )
        return [document for document, _score in results]

    def _keyword_search(self, query: str) -> list[Document]:
        documents = self._get_bm25().search(query, self.fetch_k)
        if self.filter_document_ids:
            allowed = set(self.filter_document_ids)
            documents = [
                doc
                for doc in documents
                if str(doc.metadata.get("document_id")) in allowed
            ]
        return documents

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[Document]],
    ) -> list[tuple[Document, float]]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
        fused: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, document in enumerate(ranked):
                key = str(document.metadata.get("chunk_id") or id(document))
                fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
                by_id.setdefault(key, document)
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [(by_id[key], score) for key, score in ordered]

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents for ``query``.

        Args:
            query: The (possibly rewritten) user query.

        Returns:
            A :class:`RetrievalResult` containing fused, de-duplicated
            documents and their RRF scores.
        """
        if self.vectorstore.is_empty:
            return RetrievalResult()

        vector_docs = self._vector_search(query)
        keyword_docs: list[Document] = []

        if self.use_hybrid and self._get_bm25().available:
            keyword_docs = self._keyword_search(query)

        if keyword_docs:
            fused = self._reciprocal_rank_fusion([vector_docs, keyword_docs])
            used_hybrid = True
        else:
            fused = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(vector_docs)]
            used_hybrid = False

        top = fused[: self.k]
        return RetrievalResult(
            documents=[document for document, _ in top],
            scores=[score for _, score in top],
            vector_hits=len(vector_docs),
            keyword_hits=len(keyword_docs),
            used_hybrid=used_hybrid,
        )
