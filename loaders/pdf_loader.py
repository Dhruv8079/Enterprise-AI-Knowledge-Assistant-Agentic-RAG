"""PDF loader built on PyMuPDF (``fitz``).

Extracts text page by page so that page numbers survive into the chunk
metadata and can be shown as citations.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from loaders.base import DocumentLoadError, LoadedDocument, clean_text
from utils.logger import get_logger

logger = get_logger(__name__)


def load_pdf(path: Path, document_id: str, filename: str) -> LoadedDocument:
    """Extract text from a PDF file, preserving page numbers.

    Args:
        path: Path to the PDF on disk.
        document_id: Stable document identifier.
        filename: Original file name.

    Returns:
        A :class:`LoadedDocument` whose ``documents`` each represent one page.

    Raises:
        DocumentLoadError: If the PDF cannot be opened or is encrypted.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentLoadError(
            "PyMuPDF is not installed. Run 'pip install pymupdf' to enable PDF support."
        ) from exc

    try:
        pdf = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Could not open PDF '{filename}': {exc}") from exc

    documents: list[Document] = []
    try:
        if pdf.is_encrypted and not pdf.authenticate(""):
            raise DocumentLoadError(f"PDF '{filename}' is password protected.")

        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            raw_text = page.get_text("text")
            text = clean_text(raw_text)
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "file_type": "PDF",
                        "page": page_index + 1,
                        "document_id": document_id,
                    },
                )
            )
    finally:
        pdf.close()

    if not documents:
        raise DocumentLoadError(
            f"No extractable text found in '{filename}'. It may be a scanned PDF."
        )

    logger.info("Extracted %d page(s) from '%s'", len(documents), filename)
    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        file_type="PDF",
        size_bytes=path.stat().st_size,
        num_units=len(documents),
        documents=documents,
    )
