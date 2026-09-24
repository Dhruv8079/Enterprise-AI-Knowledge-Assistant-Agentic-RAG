"""Intent routing for the agentic layer.

Questions are classified into one of five intents using a fast, deterministic
rule engine. An optional LLM classifier is available as a fallback but is not
required, which keeps routing predictable and cheap.
"""

from __future__ import annotations

import re
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

DOCUMENT_QA = "DOCUMENT_QA"
DATA_ANALYSIS = "DATA_ANALYSIS"
SUMMARY = "SUMMARY"
COMPARISON = "COMPARISON"
GENERAL = "GENERAL"

ALL_INTENTS: tuple[str, ...] = (DOCUMENT_QA, DATA_ANALYSIS, SUMMARY, COMPARISON, GENERAL)

INTENT_LABELS: dict[str, str] = {
    DOCUMENT_QA: "Document Q&A",
    DATA_ANALYSIS: "Data Analysis",
    SUMMARY: "Summarisation",
    COMPARISON: "Comparison",
    GENERAL: "General",
}

_INTENT_PATTERNS: dict[str, list[str]] = {
    COMPARISON: [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifference(s)?\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bwhat changed\b",
        r"\bwhich (clauses?|sections?|terms?) (are )?(different|differ)\b",
        r"\bbetween .+ and\b",
    ],
    SUMMARY: [
        r"\bsummari[sz]e\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\btl;?dr\b",
        r"\bkey (points|takeaways|findings)\b",
        r"\bgist\b",
        r"\bmain (points|ideas|topics)\b",
    ],
    DATA_ANALYSIS: [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bmedian\b",
        r"\bmode\b",
        r"\bsum\b",
        r"\btotal\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\btop \d+\b",
        r"\bmissing values?\b",
        r"\bnull values?\b",
        r"\bduplicates?\b",
        r"\bgroup(ed)? by\b",
        r"\bcolumns?\b",
        r"\brows?\b",
        r"\bcorrelation\b",
        r"\bdistribution\b",
        r"\bshow (me )?(all )?(employees|records|rows|entries)\b",
        r"\b(salary|price|revenue|cost|age|score)\b",
        r"\bgreater than\b",
        r"\bless than\b",
        r"\babove\b",
        r"\bbelow\b",
    ],
    DOCUMENT_QA: [
        r"\bwhat does .+ say\b",
        r"\baccording to\b",
        r"\bpolicy\b",
        r"\bclause\b",
        r"\bcontract\b",
        r"\bterms\b",
        r"\bmentioned\b",
        r"\bstate(s|d)?\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bwhy\b",
        r"\bhow (do|does|should)\b",
    ],
}

_GENERAL_PATTERNS: list[str] = [
    r"^\s*(hi|hello|hey|greetings)\b",
    r"\bwho are you\b",
    r"\bwhat can you do\b",
    r"\bhelp\b",
    r"\bthank(s| you)\b",
]


def _score_patterns(question: str, patterns: list[str]) -> int:
    """Return the number of patterns that match the question."""
    return sum(1 for pattern in patterns if re.search(pattern, question, re.IGNORECASE))


def classify_intent_heuristic(
    question: str,
    *,
    has_structured_data: bool = False,
) -> str:
    """Classify a question using keyword heuristics.

    Args:
        question: The user's question.
        has_structured_data: Whether any CSV/XLSX data is loaded.

    Returns:
        One of the intent constants.
    """
    text = question.strip()
    if not text:
        return GENERAL

    scores = {
        intent: _score_patterns(text, patterns)
        for intent, patterns in _INTENT_PATTERNS.items()
    }

    # Comparison and summary are strong, unambiguous signals.
    if scores[COMPARISON] > 0:
        return COMPARISON
    if scores[SUMMARY] > 0:
        return SUMMARY

    # Data analysis only makes sense when structured data is available.
    if has_structured_data and scores[DATA_ANALYSIS] > 0:
        return DATA_ANALYSIS

    if scores[DOCUMENT_QA] > 0:
        return DOCUMENT_QA

    if _score_patterns(text, _GENERAL_PATTERNS) > 0 and not has_structured_data:
        return GENERAL

    # Default behaviour: treat as a document question when documents exist.
    if has_structured_data and scores[DATA_ANALYSIS] > 0:
        return DATA_ANALYSIS
    return DOCUMENT_QA


def classify_intent_llm(question: str, llm: Any) -> str:
    """Classify a question with an LLM, falling back to heuristics on error.

    Args:
        question: The user's question.
        llm: A LangChain chat model.

    Returns:
        One of the intent constants.
    """
    if llm is None:
        return classify_intent_heuristic(question)

    try:
        from rag.prompts import INTENT_PROMPT

        response = (INTENT_PROMPT | llm).invoke({"question": question})
        label = str(getattr(response, "content", response)).strip().upper()
        for intent in ALL_INTENTS:
            if intent in label:
                return intent
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM intent classification failed: %s", exc)

    return classify_intent_heuristic(question)


def classify_intent(
    question: str,
    *,
    has_structured_data: bool = False,
    llm: Any | None = None,
    use_llm: bool = False,
) -> str:
    """Classify a question into an intent.

    Args:
        question: The user's question.
        has_structured_data: Whether CSV/XLSX data is present.
        llm: Optional chat model for LLM-based classification.
        use_llm: When ``True`` and ``llm`` is provided, use the LLM path.

    Returns:
        One of the intent constants.
    """
    if use_llm and llm is not None:
        return classify_intent_llm(question, llm)
    return classify_intent_heuristic(question, has_structured_data=has_structured_data)
