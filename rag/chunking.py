"""Configurable, structure-aware text chunking.

Small documents (including tabular row-groups and table elements) are kept
intact; only oversized text is split using a recursive character splitter with
separators tuned for prose, technical writing and semi-structured tables.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger
from utils.metadata import make_chunk_id

logger = get_logger(__name__)

_TEXT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_STRUCTURED_SEPARATORS: list[str] = ["\n\n", "\n", " | ", "; ", ", ", " ", ""]


class ChunkingError(Exception):
    """Raised when chunking parameters are invalid."""


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunking parameters.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Raises:
        ChunkingError: If the parameters are not usable.
    """
    if chunk_size < 100:
        raise ChunkingError("Chunk size must be at least 100 characters.")
    if chunk_overlap < 0:
        raise ChunkingError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("Chunk overlap must be smaller than the chunk size.")


def get_text_splitter(
    chunk_size: int,
    chunk_overlap: int,
    *,
    structured: bool = False,
) -> RecursiveCharacterTextSplitter:
    """Create a recursive character text splitter.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        structured: When ``True`` use separators suited to tables.

    Returns:
        A configured :class:`RecursiveCharacterTextSplitter`.
    """
    separators = _STRUCTURED_SEPARATORS if structured else _TEXT_SEPARATORS
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into metadata-rich chunks.

    Documents already smaller than ``chunk_size`` are preserved verbatim so
    that tabular row-groups and table elements are not fragmented.

    Args:
        documents: Source documents produced by a loader.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of chunked :class:`Document` objects, each carrying a
        ``chunk_id`` alongside its inherited metadata.

    Raises:
        ChunkingError: If the chunking parameters are invalid.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    counters: dict[str, int] = {}

    for document in documents:
        document_id = str(document.metadata.get("document_id", "doc"))
        content = document.page_content or ""
        if not content.strip():
            continue

        if len(content) <= chunk_size:
            pieces = [content]
        else:
            pieces = splitter.split_text(content)

        for piece in pieces:
            if not piece.strip():
                continue
            index = counters.get(document_id, 0)
            counters[document_id] = index + 1
            metadata = dict(document.metadata)
            metadata["chunk_id"] = make_chunk_id(document_id, index)
            chunks.append(Document(page_content=piece, metadata=metadata))

    logger.info("Created %d chunk(s) from %d source unit(s)", len(chunks), len(documents))
    return chunks
