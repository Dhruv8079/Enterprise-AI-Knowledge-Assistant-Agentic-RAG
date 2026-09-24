"""Tests for the chunking pipeline."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.chunking import ChunkingError, chunk_documents, validate_chunk_params


def _document(text: str, **metadata) -> Document:
    base = {
        "source": "file.txt",
        "file_type": "TXT",
        "page": 1,
        "document_id": "d1",
    }
    base.update(metadata)
    return Document(page_content=text, metadata=base)


def test_small_document_is_preserved_verbatim() -> None:
    chunks = chunk_documents([_document("A short paragraph.")], 1000, 100)

    assert len(chunks) == 1
    assert chunks[0].page_content == "A short paragraph."
    assert chunks[0].metadata["chunk_id"] == "d1-c00000"


def test_large_document_is_split_with_unique_ids() -> None:
    chunks = chunk_documents([_document("word " * 2000)], 500, 50)

    assert len(chunks) > 1
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)


def test_metadata_is_inherited() -> None:
    chunks = chunk_documents([_document("hello world", page=7, file_type="PDF")], 1000, 100)

    assert chunks[0].metadata["page"] == 7
    assert chunks[0].metadata["file_type"] == "PDF"
    assert chunks[0].metadata["source"] == "file.txt"


def test_chunk_ids_are_sequential_per_document() -> None:
    documents = [_document("first " * 500), _document("second " * 500)]
    chunks = chunk_documents(documents, 400, 40)
    first_document_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
        if chunk.page_content.startswith("first")
    ]
    assert first_document_ids[0].endswith("c00000")


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ChunkingError):
        validate_chunk_params(50, 10)
    with pytest.raises(ChunkingError):
        validate_chunk_params(500, 500)
    with pytest.raises(ChunkingError):
        validate_chunk_params(500, -1)
