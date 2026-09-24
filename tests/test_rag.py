"""Tests for the RAG engine, router and data analyst."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from langchain_core.documents import Document

from agents.data_analyst import DataAnalyst, Dataset
from rag.chain import RAGEngine, history_to_text
from rag.query_router import (
    COMPARISON,
    DATA_ANALYSIS,
    DOCUMENT_QA,
    SUMMARY,
    classify_intent,
)
from rag.retriever import RetrievalResult


class FakeLLM:
    """Minimal chat model returning a fixed response."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[object]] = []

    def invoke(self, messages):  # noqa: ANN001, ANN201
        self.calls.append(list(messages))
        return SimpleNamespace(content=self.content)


class FakeRetriever:
    """Duck-typed retriever returning a fixed result."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.filter_document_ids: list[str] | None = None

    def retrieve(self, query: str) -> RetrievalResult:  # noqa: ARG002
        return self._result


def _policy_document() -> Document:
    return Document(
        page_content="The leave policy grants twenty days per year.",
        metadata={
            "source": "policy.pdf",
            "file_type": "PDF",
            "page": 2,
            "chunk_id": "d1-c00000",
            "document_id": "d1",
        },
    )


def _result(documents: list[Document]) -> RetrievalResult:
    return RetrievalResult(
        documents=documents,
        scores=[1.0] * len(documents),
        vector_hits=len(documents),
        keyword_hits=0,
        used_hybrid=False,
    )


def test_rag_engine_returns_grounded_answer_with_sources() -> None:
    llm = FakeLLM("The leave policy grants twenty days. [Source: policy.pdf — page 2]")
    engine = RAGEngine(llm, FakeRetriever(_result([_policy_document()])), None, top_k=3)

    response = engine.answer("What is the leave policy?", [])

    assert not response.used_fallback
    assert response.sources
    assert response.sources[0].source == "policy.pdf"
    assert response.sources[0].page == "2"
    assert "twenty days" in response.answer


def test_rag_engine_falls_back_when_no_documents() -> None:
    engine = RAGEngine(FakeLLM("irrelevant"), FakeRetriever(_result([])), None)

    response = engine.answer("What is the leave policy?", [])

    assert response.used_fallback
    assert "couldn't find" in response.answer.lower()


def test_rag_engine_detects_model_admitting_ignorance() -> None:
    llm = FakeLLM("I couldn't find sufficient information in the uploaded documents.")
    engine = RAGEngine(llm, FakeRetriever(_result([_policy_document()])), None)

    response = engine.answer("Unanswerable question", [])

    assert response.used_fallback
    assert response.sources == []


def test_rag_engine_injects_history() -> None:
    llm = FakeLLM("Answer")
    engine = RAGEngine(llm, FakeRetriever(_result([_policy_document()])), None)
    history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]

    engine.answer("Follow-up question", history)

    assert len(llm.calls) == 1
    assert len(llm.calls[0]) >= 3


def test_history_to_text_formats_turns() -> None:
    transcript = history_to_text([{"role": "user", "content": "hi"}])
    assert "User: hi" in transcript
    assert history_to_text([]).startswith("(no previous")


def test_router_classifies_intents() -> None:
    assert classify_intent("What is the average salary?", has_structured_data=True) == DATA_ANALYSIS
    assert classify_intent("Compare these two reports") == COMPARISON
    assert classify_intent("Summarize this document") == SUMMARY
    assert classify_intent("What does the contract say about termination?") == DOCUMENT_QA


def _dataset() -> Dataset:
    frame = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"],
            "department": ["HR", "IT", "IT"],
            "salary": [100.0, 300.0, 500.0],
        }
    )
    return Dataset(document_id="d1", source="employees.csv", sheet="data", df=frame)


def test_data_analyst_average_is_deterministic() -> None:
    result = DataAnalyst(llm=None).analyze("What is the average salary?", [_dataset()])
    assert "300" in result.details


def test_data_analyst_group_extreme() -> None:
    result = DataAnalyst(llm=None).analyze(
        "Which department has the highest salary?", [_dataset()]
    )
    assert "IT" in result.details


def test_data_analyst_missing_values() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
    dataset = Dataset(document_id="d2", source="data.csv", sheet="data", df=frame)
    result = DataAnalyst(llm=None).analyze("How many missing values?", [dataset])
    assert result.operation == "missing"
    assert "3" in result.details


def test_data_analyst_duplicates() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2]})
    dataset = Dataset(document_id="d3", source="data.csv", sheet="data", df=frame)
    result = DataAnalyst(llm=None).analyze("Are there duplicate records?", [dataset])
    assert result.operation == "duplicates"
    assert "1" in result.details


def test_data_analyst_lists_columns() -> None:
    result = DataAnalyst(llm=None).analyze("What columns exist in this file?", [_dataset()])
    assert result.operation == "columns"
    assert "salary" in result.details
