"""Route a file to the appropriate loader based on its extension."""

from __future__ import annotations

from pathlib import Path

from loaders.base import DocumentLoadError, LoadedDocument
from utils.file_utils import get_extension
from utils.logger import get_logger

logger = get_logger(__name__)


def load_document(path: Path, document_id: str, original_name: str | None = None) -> LoadedDocument:
    """Load a file from disk using the loader matching its extension.

    Args:
        path: Path to the stored file on disk.
        document_id: Stable document identifier.
        original_name: Original file name to record in metadata. Defaults to
            the stored file name.

    Returns:
        A :class:`LoadedDocument` with extracted content and metadata.

    Raises:
        DocumentLoadError: If the extension is unsupported or parsing fails.
    """
    name = original_name or path.name
    extension = get_extension(name)
    logger.info("Loading document '%s' (%s)", name, extension)

    try:
        if extension == ".pdf":
            from loaders.pdf_loader import load_pdf

            return load_pdf(path, document_id, name)
        if extension == ".docx":
            from loaders.docx_loader import load_docx

            return load_docx(path, document_id, name)
        if extension == ".txt":
            from loaders.text_loader import load_text

            return load_text(path, document_id, name)
        if extension == ".csv":
            from loaders.csv_loader import load_csv

            return load_csv(path, document_id, name)
        if extension in (".xlsx", ".xls"):
            from loaders.excel_loader import load_excel

            return load_excel(path, document_id, name)
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert to a user-safe error
        logger.exception("Failed to load document '%s'", name)
        raise DocumentLoadError(
            f"Failed to process '{name}'. The file may be corrupted or password protected."
        ) from exc

    raise DocumentLoadError(f"Unsupported file type for '{name}'.")
