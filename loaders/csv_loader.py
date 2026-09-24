"""CSV loader using pandas with delimiter detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import MAX_DATA_ROWS
from loaders.base import DocumentLoadError, LoadedDocument
from loaders.tabular import prepare_dataframe, row_group_documents
from utils.logger import get_logger

logger = get_logger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file, falling back to the Python engine and sniffing."""
    errors: list[str] = []
    for sep in [None, ",", ";", "\t", "|"]:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                engine="python" if sep is None else "c",
                nrows=MAX_DATA_ROWS,
                on_bad_lines="skip",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    raise DocumentLoadError(f"Could not parse the CSV file. {' | '.join(errors[-2:])}")


def load_csv(path: Path, document_id: str, filename: str) -> LoadedDocument:
    """Load a CSV file into a dataframe plus retrieval documents.

    Args:
        path: Path to the CSV on disk.
        document_id: Stable document identifier.
        filename: Original file name.

    Returns:
        A :class:`LoadedDocument` containing the dataframe and text documents.

    Raises:
        DocumentLoadError: If the CSV cannot be parsed.
    """
    dataframe = prepare_dataframe(_read_csv(path))
    if dataframe.empty:
        raise DocumentLoadError(f"'{filename}' contains no usable data.")

    documents = row_group_documents(
        dataframe,
        filename=filename,
        file_type="CSV",
        document_id=document_id,
        dataset_name="data",
    )

    logger.info("Loaded CSV '%s' with %d rows", filename, len(dataframe))
    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        file_type="CSV",
        size_bytes=path.stat().st_size,
        num_units=len(dataframe),
        documents=documents,
        dataframes={"data": dataframe},
        extra={"columns": list(map(str, dataframe.columns))},
    )
