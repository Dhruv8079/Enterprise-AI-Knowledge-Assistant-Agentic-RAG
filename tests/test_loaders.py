"""Tests for the document loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from loaders.base import clean_text
from loaders.csv_loader import load_csv
from loaders.docx_loader import load_docx
from loaders.pdf_loader import load_pdf
from loaders.tabular import prepare_dataframe
from loaders.text_loader import load_text


def test_clean_text_normalises_whitespace() -> None:
    assert clean_text("a\r\n\r\n\r\nb  ") == "a\n\nb"
    assert clean_text("") == ""


def test_text_loader_creates_document(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Hello world from a text file.", encoding="utf-8")

    loaded = load_text(path, "doc1", "notes.txt")

    assert loaded.num_units == 1
    assert loaded.file_type == "TXT"
    assert loaded.documents[0].metadata["document_id"] == "doc1"
    assert loaded.documents[0].metadata["page"] == 1


def test_text_loader_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    from loaders.base import DocumentLoadError

    with pytest.raises(DocumentLoadError):
        load_text(path, "doc1", "empty.txt")


def test_csv_loader_builds_dataframe_and_chunks(tmp_path) -> None:
    path = tmp_path / "employees.csv"
    path.write_text(
        "name,department,salary\nAlice,HR,100\nBob,IT,200\nCarol,IT,300\n",
        encoding="utf-8",
    )

    loaded = load_csv(path, "doc2", "employees.csv")

    assert "data" in loaded.dataframes
    assert loaded.num_units == 3
    assert loaded.documents[0].metadata["element"] == "schema"
    assert all(doc.metadata["document_id"] == "doc2" for doc in loaded.documents)
    assert "columns" in loaded.documents[-1].metadata


def test_prepare_dataframe_drops_unnamed_columns() -> None:
    frame = pd.DataFrame({"a": [1, 2], "Unnamed: 0": [3, 4]})
    cleaned = prepare_dataframe(frame)

    assert "Unnamed: 0" not in cleaned.columns
    assert list(cleaned.columns) == ["a"]


def test_docx_loader_extracts_paragraphs_and_tables(tmp_path) -> None:
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_paragraph("The leave policy grants twenty days per year.")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Employee"
    table.rows[0].cells[1].text = "Days"
    table.rows[1].cells[0].text = "Alice"
    table.rows[1].cells[1].text = "20"

    path = tmp_path / "policy.docx"
    document.save(path)

    loaded = load_docx(path, "doc3", "policy.docx")

    assert loaded.num_units >= 1
    assert any(doc.metadata.get("element") == "table" for doc in loaded.documents)


def test_pdf_loader_preserves_page_numbers(tmp_path) -> None:
    fitz = pytest.importorskip("fitz")

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The leave policy grants twenty days per year.")
    path = tmp_path / "policy.pdf"
    document.save(path)
    document.close()

    loaded = load_pdf(path, "doc4", "policy.pdf")

    assert loaded.num_units == 1
    assert loaded.documents[0].metadata["page"] == 1
    assert "twenty days" in loaded.documents[0].page_content
