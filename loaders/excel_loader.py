"""Excel loader using pandas + openpyxl, one dataframe per sheet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import MAX_DATA_ROWS
from loaders.base import DocumentLoadError, LoadedDocument
from loaders.tabular import prepare_dataframe, row_group_documents
from utils.logger import get_logger

logger = get_logger(__name__)


def load_excel(path: Path, document_id: str, filename: str) -> LoadedDocument:
    """Load every sheet of an Excel workbook.

    Args:
        path: Path to the workbook on disk.
        document_id: Stable document identifier.
        filename: Original file name.

    Returns:
        A :class:`LoadedDocument` with one dataframe per non-empty sheet.

    Raises:
        DocumentLoadError: If the workbook cannot be read or has no data.
    """
    try:
        sheets = pd.read_excel(path, sheet_name=None, nrows=MAX_DATA_ROWS)
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Could not read Excel file '{filename}': {exc}") from exc

    dataframes: dict[str, pd.DataFrame] = {}
    documents = []
    total_rows = 0
    all_columns: list[str] = []

    for sheet_name, raw_df in sheets.items():
        dataframe = prepare_dataframe(raw_df)
        if dataframe.empty:
            continue
        sheet_label = str(sheet_name)
        dataframes[sheet_label] = dataframe
        total_rows += len(dataframe)
        all_columns.extend(str(col) for col in dataframe.columns)
        documents.extend(
            row_group_documents(
                dataframe,
                filename=filename,
                file_type="XLSX",
                document_id=document_id,
                dataset_name=sheet_label,
            )
        )

    if not dataframes:
        raise DocumentLoadError(f"'{filename}' contains no usable data in any sheet.")

    logger.info("Loaded Excel '%s' with %d sheet(s)", filename, len(dataframes))
    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        file_type="XLSX",
        size_bytes=path.stat().st_size,
        num_units=total_rows,
        documents=documents,
        dataframes=dataframes,
        extra={"sheets": list(dataframes.keys()), "columns": sorted(set(all_columns))},
    )
