"""Shared loader types and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.documents import Document


class DocumentLoadError(Exception):
    """Raised when a document cannot be parsed."""


@dataclass
class LoadedDocument:
    """Result of loading a single uploaded file.

    Attributes:
        document_id: Stable identifier derived from the file hash.
        filename: Original file name.
        file_type: Human-readable file type label.
        size_bytes: Size of the source file in bytes.
        num_units: Number of pages (documents) or rows (tabular data).
        documents: LangChain ``Document`` objects ready for chunking.
        dataframes: Mapping of sheet/source name to a pandas DataFrame.
        extra: Additional metadata (e.g. duplicate-row counts).
    """

    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    num_units: int
    documents: list[Document] = field(default_factory=list)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tabular(self) -> bool:
        """Return ``True`` when the document contains structured data."""
        return bool(self.dataframes)

    @property
    def text_length(self) -> int:
        """Total number of characters extracted across all documents."""
        return sum(len(doc.page_content) for doc in self.documents)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary used by the UI and tracker."""
        return {
            "document_id": self.document_id,
            "source": self.filename,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "num_units": self.num_units,
            "text_length": self.text_length,
            "is_tabular": self.is_tabular,
            "sheets": list(self.dataframes.keys()),
            "extra": self.extra,
        }


def clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks.

    Args:
        text: Raw extracted text.

    Returns:
        Text with normalised line endings, collapsed spaces and at most two
        consecutive newlines.
    """
    if not text:
        return ""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    normalised = "\n".join(line.rstrip() for line in normalised.split("\n"))
    while "\n\n\n" in normalised:
        normalised = normalised.replace("\n\n\n", "\n\n")
    return normalised.strip()
