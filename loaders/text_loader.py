"""Plain-text loader with multi-encoding fallback."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from loaders.base import DocumentLoadError, LoadedDocument, clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_text(path: Path) -> str:
    """Read a text file trying a sequence of common encodings."""
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
    raise DocumentLoadError(f"Could not decode the text file: {last_error}")


def load_text(path: Path, document_id: str, filename: str) -> LoadedDocument:
    """Load a ``.txt`` file into a single LangChain document.

    Args:
        path: Path to the text file.
        document_id: Stable document identifier.
        filename: Original file name.

    Returns:
        A :class:`LoadedDocument` with one document.

    Raises:
        DocumentLoadError: If the file cannot be decoded or is empty.
    """
    text = clean_text(_read_text(path))
    if not text:
        raise DocumentLoadError(f"'{filename}' is empty.")

    document = Document(
        page_content=text,
        metadata={
            "source": filename,
            "file_type": "TXT",
            "page": 1,
            "document_id": document_id,
        },
    )

    logger.info("Loaded text document '%s' (%d chars)", filename, len(text))
    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        file_type="TXT",
        size_bytes=path.stat().st_size,
        num_units=1,
        documents=[document],
    )
