"""Shared helpers for CSV and Excel (tabular) loading.

Tabular files are represented twice:

* as a pandas ``DataFrame`` for deterministic analysis, and
* as text documents (a schema description plus row-group chunks) for semantic
  retrieval, so questions like "what does the notes column say" still work.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from loaders.base import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

ROWS_PER_CHUNK: int = 30
MAX_SAMPLE_ROWS: int = 5


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw dataframe loaded from a file.

    Column names are stripped, unnamed index columns are dropped and fully
    empty rows/columns are removed.

    Args:
        df: Raw pandas DataFrame.

    Returns:
        A cleaned copy of the dataframe.
    """
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    unnamed = [col for col in cleaned.columns if col.lower().startswith("unnamed:")]
    if unnamed:
        cleaned = cleaned.drop(columns=unnamed)
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def dataframe_schema_text(df: pd.DataFrame, name: str) -> str:
    """Build a natural-language schema summary for a dataframe.

    Args:
        df: The dataframe to describe.
        name: Human-readable dataset name (file or sheet name).

    Returns:
        A multi-line textual description including columns, dtypes, shape and
        a small sample plus descriptive statistics.
    """
    lines: list[str] = [f"Dataset: {name}"]
    lines.append(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    lines.append("Columns and types:")
    for column in df.columns:
        lines.append(f"  - {column} ({df[column].dtype})")

    if not df.empty:
        lines.append("\nSample rows:")
        sample = df.head(MAX_SAMPLE_ROWS).to_string(index=False)
        lines.append(sample)

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            lines.append("\nNumeric summary (describe):")
            lines.append(numeric.describe().to_string())

        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            lines.append("\nMissing values per column:")
            for column, count in missing.items():
                lines.append(f"  - {column}: {int(count)}")

    return "\n".join(lines)


def row_group_documents(
    df: pd.DataFrame,
    *,
    filename: str,
    file_type: str,
    document_id: str,
    dataset_name: str,
    rows_per_chunk: int = ROWS_PER_CHUNK,
) -> list[Document]:
    """Convert a dataframe into row-group text documents.

    Args:
        df: Cleaned dataframe.
        filename: Original file name.
        file_type: Human-readable file type label.
        document_id: Stable document identifier.
        dataset_name: Sheet name or ``data`` for CSVs.
        rows_per_chunk: Number of rows per generated document.

    Returns:
        A list of LangChain documents with row-range metadata.
    """
    documents: list[Document] = []
    columns = ", ".join(str(col) for col in df.columns)

    schema = dataframe_schema_text(df, f"{filename} — {dataset_name}")
    documents.append(
        Document(
            page_content=clean_text(schema),
            metadata={
                "source": filename,
                "file_type": file_type,
                "page": f"{dataset_name} schema",
                "sheet": dataset_name,
                "element": "schema",
                "columns": columns,
                "document_id": document_id,
            },
        )
    )

    if df.empty:
        return documents

    for start in range(0, len(df), rows_per_chunk):
        end = min(start + rows_per_chunk, len(df))
        subset = df.iloc[start:end]
        header = f"Rows {start + 1}-{end} of {dataset_name} ({filename}) | columns: {columns}"
        body = subset.to_string(index=False)
        documents.append(
            Document(
                page_content=clean_text(f"{header}\n{body}"),
                metadata={
                    "source": filename,
                    "file_type": file_type,
                    "page": f"rows {start + 1}-{end}",
                    "sheet": dataset_name,
                    "row_start": int(start + 1),
                    "row_end": int(end),
                    "columns": columns,
                    "document_id": document_id,
                },
            )
        )

    return documents
