"""Helpers for building consistent per-chunk metadata.

Every indexed chunk carries a stable metadata dictionary so that retrieval,
citation rendering, filtering and deletion are all reliable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_chunk_id(document_id: str, index: int) -> str:
    """Build a deterministic chunk identifier for a document."""
    return f"{document_id}-c{index:05d}"


def build_chunk_metadata(
    *,
    source: str,
    file_type: str,
    page: int | str,
    chunk_id: str,
    document_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalised metadata dictionary for a single chunk.

    Args:
        source: Original file name.
        file_type: Human-readable file type label (e.g. ``PDF``).
        page: 1-based page number or a descriptive string (``row 1-25``).
        chunk_id: Unique chunk identifier.
        document_id: Parent document identifier.
        extra: Optional additional metadata merged into the result.

    Returns:
        A dictionary containing only primitive, JSON-serialisable values so
        it can be persisted by FAISS and dumped to JSON safely.
    """
    metadata: dict[str, Any] = {
        "source": source,
        "file_type": file_type,
        "page": str(page),
        "chunk_id": chunk_id,
        "document_id": document_id,
    }
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            metadata[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return metadata


def summarise_metadata(metadata: dict[str, Any]) -> str:
    """Return a compact human-readable description of chunk metadata."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page", "?")
    file_type = metadata.get("file_type", "?")
    sheet = metadata.get("sheet")
    label = f"{source} ({file_type}) — page {page}"
    if sheet:
        label += f" — sheet '{sheet}'"
    return label
