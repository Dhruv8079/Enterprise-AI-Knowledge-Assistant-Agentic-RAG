"""Deterministic pandas-based data analysis.

Numerical questions about CSV/XLSX files are answered by pandas rather than by
the LLM. The LLM is only used to explain the already-computed result in plain
language, which eliminates hallucinated numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from rag.chain import _message_text
from rag.prompts import DATA_EXPLANATION_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_COMPARISON_RE = re.compile(
    r"(above|over|greater than|more than|>=|>|below|under|less than|<|<=)\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_GROUP_RE = re.compile(
    r"which\s+(.+?)\s+(?:has|have|had)\s+the\s+"
    r"(highest|lowest|largest|smallest|most|least|max|minimum|maximum|greatest)\s+(.+)",
    re.IGNORECASE,
)
_TOP_RE = re.compile(r"\btop\s+(\d+)\b(?:\s+by\s+(.+))?", re.IGNORECASE)


@dataclass
class Dataset:
    """A single analysed dataframe."""

    document_id: str
    source: str
    sheet: str
    df: pd.DataFrame

    @property
    def label(self) -> str:
        """Human-readable dataset label."""
        return self.source if self.sheet == "data" else f"{self.source} [{self.sheet}]"


@dataclass
class DataAnswer:
    """Result of a data analysis request."""

    answer: str
    details: str = ""
    operation: str = "describe"
    used_llm: bool = False
    datasets: list[str] = field(default_factory=list)


def _normalise(name: str) -> str:
    """Normalise a string for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _to_number(raw: str) -> float:
    """Convert a numeric string (possibly with commas) to a float."""
    return float(raw.replace(",", ""))


def _is_numeric(series: pd.Series) -> bool:
    """Return ``True`` when a series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


class DataAnalyst:
    """Answer questions about structured data using pandas.

    Args:
        llm: Optional chat model used solely to explain computed results.
        max_table_rows: Maximum rows rendered into result tables.
    """

    def __init__(self, llm: Any | None = None, max_table_rows: int = 20) -> None:
        self.llm = llm
        self.max_table_rows = max_table_rows

    # ------------------------------------------------------------------ #
    # Dataset selection
    # ------------------------------------------------------------------ #
    def _select_datasets(
        self,
        question: str,
        datasets: list[Dataset],
    ) -> list[Dataset]:
        """Pick the datasets referenced by the question.

        Matching is attempted against the file name and the sheet name. When
        nothing matches, every dataset is returned so that broad questions
        ("how many rows in total") still work.
        """
        if len(datasets) <= 1:
            return datasets

        lowered = question.lower()
        matched = [
            dataset
            for dataset in datasets
            if dataset.source.lower() in lowered
            or _normalise(dataset.source) in _normalise(question)
            or dataset.sheet.lower() in lowered
        ]
        return matched or datasets

    # ------------------------------------------------------------------ #
    # Column detection
    # ------------------------------------------------------------------ #
    def _find_column(
        self,
        question: str,
        df: pd.DataFrame,
        *,
        numeric_only: bool = False,
        phrase: str | None = None,
    ) -> str | None:
        """Find the dataframe column most likely referenced by the question.

        Args:
            question: Full question text.
            df: Dataframe whose columns are searched.
            numeric_only: Restrict to numeric columns.
            phrase: Optional explicit phrase to match against (takes priority).

        Returns:
            The matched column name or ``None``.
        """
        columns = [
            column
            for column in df.columns
            if not numeric_only or _is_numeric(df[column])
        ]
        if not columns:
            return None

        haystack = _normalise(phrase) if phrase else _normalise(question)
        question_lower = (phrase or question).lower()

        # Exact phrase match first (longest column name wins).
        for column in sorted(columns, key=lambda c: len(str(c)), reverse=True):
            if _normalise(column) and _normalise(column) in haystack:
                return column

        # Whole-word / substring match on the raw question.
        for column in sorted(columns, key=lambda c: len(str(c)), reverse=True):
            if str(column).lower() in question_lower:
                return column

        # Token match: every token of the column name appears.
        for column in sorted(columns, key=lambda c: len(str(c)), reverse=True):
            tokens = re.findall(r"[a-z0-9]+", str(column).lower())
            if tokens and all(token in question_lower for token in tokens):
                return column
        return None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        question: str,
        datasets: list[Dataset],
        *,
        explain: bool = True,
    ) -> DataAnswer:
        """Answer a data question deterministically.

        Args:
            question: The user's question.
            datasets: All available datasets.
            explain: Whether to ask the LLM to explain the result.

        Returns:
            A :class:`DataAnswer` with the exact computed result.
        """
        if not datasets:
            return DataAnswer(
                answer="No structured data is available. Upload a CSV or Excel file first.",
                operation="none",
            )

        selected = self._select_datasets(question, datasets)
        details, operation = self._compute(question, selected)

        answer = details
        used_llm = False
        if explain and self.llm is not None:
            try:
                chain = DATA_EXPLANATION_PROMPT | self.llm
                response = chain.invoke({"question": question, "result": details})
                explanation = _message_text(response).strip()
                if explanation:
                    answer = explanation
                    used_llm = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data explanation via LLM failed: %s", exc)

        return DataAnswer(
            answer=answer,
            details=details,
            operation=operation,
            used_llm=used_llm,
            datasets=[dataset.label for dataset in selected],
        )

    # ------------------------------------------------------------------ #
    # Computation dispatch
    # ------------------------------------------------------------------ #
    def _compute(self, question: str, datasets: list[Dataset]) -> tuple[str, str]:
        """Dispatch the question to the appropriate pandas computation."""
        lowered = question.lower()

        if re.search(r"\b(columns?|fields?|schema)\b", lowered):
            return self._list_columns(datasets), "columns"
        if re.search(r"\b(missing|null|nan|empty)\b", lowered):
            return self._missing_values(datasets), "missing"
        if re.search(r"\bduplicat", lowered):
            return self._duplicates(datasets), "duplicates"
        if re.search(r"\bcorrelat", lowered):
            return self._correlation(question, datasets), "correlation"

        group_match = _GROUP_RE.search(question)
        if group_match:
            return self._group_extreme(group_match, datasets), "group_extreme"

        top_match = _TOP_RE.search(question)
        if top_match:
            result = self._top_n(top_match, question, datasets)
            if result is not None:
                return result, "top_n"

        comparison_match = _COMPARISON_RE.search(question)
        if comparison_match:
            result = self._filter_rows(comparison_match, question, datasets)
            if result is not None:
                return result, "filter"

        aggregate = self._aggregate(question, datasets)
        if aggregate is not None:
            return aggregate, "aggregate"

        if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", lowered) and re.search(
            r"\b(rows?|records?|entries|items?|employees?|people|customers?|transactions?|"
            r"documents?|files?|products?|orders?)\b",
            lowered,
        ):
            return self._count_rows(datasets), "count"

        return self._describe(datasets), "describe"

    # ------------------------------------------------------------------ #
    # Individual operations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _list_columns(datasets: list[Dataset]) -> str:
        lines = ["Columns per dataset:"]
        for dataset in datasets:
            lines.append(f"\n{dataset.label} ({len(dataset.df)} rows):")
            for column in dataset.df.columns:
                lines.append(f"  - {column} ({dataset.df[column].dtype})")
        return "\n".join(lines)

    @staticmethod
    def _missing_values(datasets: list[Dataset]) -> str:
        lines = ["Missing values:"]
        for dataset in datasets:
            missing = dataset.df.isna().sum()
            total = int(missing.sum())
            lines.append(f"\n{dataset.label}: {total} missing cell(s) in {len(dataset.df)} rows.")
            for column, count in missing.items():
                if count:
                    lines.append(f"  - {column}: {int(count)}")
            if total == 0:
                lines.append("  - No missing values.")
        return "\n".join(lines)

    @staticmethod
    def _duplicates(datasets: list[Dataset]) -> str:
        lines = ["Duplicate records:"]
        for dataset in datasets:
            duplicated = dataset.df.duplicated()
            count = int(duplicated.sum())
            lines.append(f"\n{dataset.label}: {count} duplicate row(s).")
            if count:
                sample = dataset.df[duplicated].head(10)
                lines.append(sample.to_string(index=False))
        return "\n".join(lines)

    def _correlation(self, question: str, datasets: list[Dataset]) -> str:
        lines = ["Correlation matrix (numeric columns):"]
        for dataset in datasets:
            numeric = dataset.df.select_dtypes(include="number")
            if numeric.shape[1] < 2:
                lines.append(f"\n{dataset.label}: not enough numeric columns.")
                continue
            lines.append(f"\n{dataset.label}:")
            lines.append(numeric.corr(numeric_only=True).round(3).to_string())
        return "\n".join(lines)

    @staticmethod
    def _count_rows(datasets: list[Dataset]) -> str:
        lines = ["Row counts:"]
        total = 0
        for dataset in datasets:
            lines.append(f"  - {dataset.label}: {len(dataset.df)} row(s)")
            total += len(dataset.df)
        if len(datasets) > 1:
            lines.append(f"\nTotal across {len(datasets)} dataset(s): {total} rows.")
        return "\n".join(lines)

    def _group_extreme(
        self,
        match: re.Match[str],
        datasets: list[Dataset],
    ) -> str:
        group_phrase = match.group(1).strip()
        direction = match.group(2).lower()
        value_phrase = match.group(3).strip()
        ascending = direction in {"lowest", "smallest", "least", "min", "minimum"}
        lines = ["Grouped analysis:"]
        for dataset in datasets:
            value_column = self._find_column(
                value_phrase, dataset.df, numeric_only=True, phrase=value_phrase
            )
            group_column = self._find_column(
                group_phrase, dataset.df, phrase=group_phrase
            )
            if value_column is None or group_column is None:
                lines.append(
                    f"\n{dataset.label}: could not identify the group and value columns."
                )
                continue
            grouped = (
                dataset.df.groupby(group_column)[value_column]
                .mean()
                .sort_values(ascending=ascending)
            )
            if grouped.empty:
                lines.append(f"\n{dataset.label}: no data to group.")
                continue
            winner = grouped.index[0]
            lines.append(
                f"\n{dataset.label}: '{winner}' has the {direction} average "
                f"{value_column} ({grouped.iloc[0]:,.2f})."
            )
            lines.append(f"\nAverage {value_column} by {group_column}:")
            lines.append(grouped.round(2).to_string())
        return "\n".join(lines)

    def _top_n(
        self,
        match: re.Match[str],
        question: str,
        datasets: list[Dataset],
    ) -> str | None:
        count = int(match.group(1))
        by_phrase = (match.group(2) or "").strip()
        lines = [f"Top {count} records:"]
        produced = False
        for dataset in datasets:
            column = self._find_column(by_phrase, dataset.df, phrase=by_phrase or None)
            if column is None:
                numeric = dataset.df.select_dtypes(include="number").columns
                column = numeric[0] if len(numeric) else None
            if column is None:
                continue
            if _is_numeric(dataset.df[column]):
                subset = dataset.df.nlargest(count, column)
            else:
                subset = dataset.df.head(count)
            lines.append(f"\n{dataset.label} (top {count} by '{column}'):")
            lines.append(subset.head(self.max_table_rows).to_string(index=False))
            produced = True
        return "\n".join(lines) if produced else None

    def _filter_rows(
        self,
        match: re.Match[str],
        question: str,
        datasets: list[Dataset],
    ) -> str | None:
        operator = match.group(1).lower()
        threshold = _to_number(match.group(2))
        greater = operator in {"above", "over", "greater than", "more than", ">", ">="}
        inclusive = operator in {">=", "<="}
        lines = [f"Rows where value {operator} {threshold:g}:"]
        produced = False
        for dataset in datasets:
            column = self._find_column(question, dataset.df, numeric_only=True)
            if column is None:
                continue
            series = pd.to_numeric(dataset.df[column], errors="coerce")
            mask = series >= threshold if greater else series <= threshold
            if not inclusive:
                mask = series > threshold if greater else series < threshold
            subset = dataset.df[mask]
            lines.append(
                f"\n{dataset.label}: {len(subset)} row(s) where '{column}' "
                f"{operator} {threshold:g}."
            )
            lines.append(subset.head(self.max_table_rows).to_string(index=False))
            produced = True
        return "\n".join(lines) if produced else None

    def _aggregate(self, question: str, datasets: list[Dataset]) -> str | None:
        lowered = question.lower()
        operations = {
            "average": r"\baverage\b|\bmean\b|\bavg\b",
            "sum": r"\bsum\b|\btotal\b",
            "median": r"\bmedian\b",
            "max": r"\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b",
            "min": r"\bminimum\b|\bmin\b|\blowest\b|\bsmallest\b",
            "count": r"\bcount\b|\bnumber of\b",
        }
        requested: list[str] = [
            name for name, pattern in operations.items() if re.search(pattern, lowered)
        ]
        if not requested:
            return None

        lines: list[str] = []
        produced = False
        for dataset in datasets:
            numeric_columns = list(dataset.df.select_dtypes(include="number").columns)
            column = self._find_column(question, dataset.df, numeric_only=True)
            if column is None:
                if len(numeric_columns) == 1:
                    column = numeric_columns[0]
                else:
                    continue
            series = dataset.df[column]
            lines.append(f"\n{dataset.label} — column '{column}' ({len(series)} rows):")
            for operation in requested:
                if operation == "average":
                    lines.append(f"  - average: {series.mean():,.2f}")
                elif operation == "sum":
                    lines.append(f"  - sum: {series.sum():,.2f}")
                elif operation == "median":
                    lines.append(f"  - median: {series.median():,.2f}")
                elif operation == "max":
                    lines.append(f"  - maximum: {series.max():,.2f}")
                elif operation == "min":
                    lines.append(f"  - minimum: {series.min():,.2f}")
                elif operation == "count":
                    lines.append(f"  - count (non-null): {int(series.count())}")
            produced = True
        if not produced:
            return None
        return "Computed statistics:" + "\n".join(lines)

    def _describe(self, datasets: list[Dataset]) -> str:
        lines = ["Dataset overview:"]
        for dataset in datasets:
            df = dataset.df
            lines.append(
                f"\n{dataset.label}: {len(df)} row(s), {len(df.columns)} column(s)."
            )
            lines.append("Columns: " + ", ".join(map(str, df.columns)))
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                lines.append("\nNumeric summary:")
                lines.append(numeric.describe().round(2).to_string())
            lines.append("\nFirst rows:")
            lines.append(df.head(self.max_table_rows).to_string(index=False))
        return "\n".join(lines)


def build_datasets(dataframe_registry: dict[str, dict[str, pd.DataFrame]], sources: dict[str, str]) -> list[Dataset]:
    """Flatten a ``document_id -> {sheet: df}`` registry into datasets.

    Args:
        dataframe_registry: Mapping of document id to sheet/dataframe mapping.
        sources: Mapping of document id to original file name.

    Returns:
        A list of :class:`Dataset` objects.
    """
    datasets: list[Dataset] = []
    for document_id, sheets in dataframe_registry.items():
        for sheet_name, df in sheets.items():
            datasets.append(
                Dataset(
                    document_id=document_id,
                    source=sources.get(document_id, document_id),
                    sheet=sheet_name,
                    df=df,
                )
            )
    return datasets
