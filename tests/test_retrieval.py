"""Tests for hybrid retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag.retriever import HybridRetriever


class FakeVectorStore:
    """Minimal duck-typed stand-in for VectorStoreManager."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    @property
    def is_empty(self) -> bool:
        return not self._documents

    def all_chunks(self) -> list[Document]:
        return list(self._documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        *,
        filter_document_ids: list[str] | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        allowed = set(filter_document_ids or [])
        results: list[tuple[Document, float]] = []
        for document in self._documents:
            if allowed and str(document.metadata.get("document_id")) not in allowed:
                continue
            results.append((document, 1.0))
        return results[:k]


def _document(chunk_id: str, text: str, document_id: str = "d1") -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source": "file.txt",
            "page": "1",
            "file_type": "TXT",
        },
    )


def test_hybrid_retrieval_returns_relevant_document() -> None:
    documents = [
        _document("c1", "The leave policy grants twenty days."),
        _document("c2", "Salary is paid monthly."),
    ]
    retriever = HybridRetriever(FakeVectorStore(documents), k=2, fetch_k=5, use_hybrid=True)

    result = retriever.retrieve("leave policy days")

    assert not result.is_empty
    assert result.documents[0].metadata["chunk_id"] == "c1"


def test_document_filter_is_applied() -> None:
    documents = [
        _document("c1", "leave policy", "d1"),
        _document("c2", "leave policy", "d2"),
    ]
    retriever = HybridRetriever(
        FakeVectorStore(documents), k=5, fetch_k=5, filter_document_ids=["d2"]
    )

    result = retriever.retrieve("leave policy")

    assert result.documents
    assert all(document.metadata["document_id"] == "d2" for document in result.documents)


def test_empty_store_returns_empty_result() -> None:
    retriever = HybridRetriever(FakeVectorStore([]))
    assert retriever.retrieve("anything").is_empty


def test_vector_only_mode_skips_keyword_search() -> None:
    documents = [_document("c1", "leave policy")]
    retriever = HybridRetriever(FakeVectorStore(documents), k=1, fetch_k=3, use_hybrid=False)

    result = retriever.retrieve("leave policy")

    assert result.keyword_hits == 0
    assert not result.used_hybrid


def test_reciprocal_rank_fusion_prefers_shared_document() -> None:
    shared = _document("shared", "shared document")
    only_left = _document("left", "left only")
    only_right = _document("right", "right only")

    fused: list[tuple[Document, Any]] = HybridRetriever._reciprocal_rank_fusion(
        [[shared, only_left], [shared, only_right]]
    )

    assert fused[0][0].metadata["chunk_id"] == "shared"
