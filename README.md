# Enterprise AI Knowledge Assistant

**Agentic RAG + Document Intelligence Platform**

A production-style AI Engineer portfolio project that turns a folder of business
documents into a grounded, citation-backed knowledge assistant. Unlike a basic
"PDF chatbot", it routes each question through an agent that selects the right
capability — grounded document Q&A, deterministic pandas data analysis,
hierarchical summarisation or structured document comparison.

---

## 1. Project Overview

The assistant ingests **PDF, DOCX, TXT, CSV and XLSX** files, cleans and chunks
them, generates embeddings, and stores them in a persistent **FAISS** vector
database. Questions are classified by an **intent router**, answered from
retrieved context only, and always accompanied by **citations**. Numerical
questions are answered with **pandas**, not the LLM, so numbers are never
hallucinated.

---

## 2. Features

- **Multi-format ingestion** — PDF (PyMuPDF), DOCX (python-docx), TXT, CSV and Excel (pandas + openpyxl).
- **Intelligent chunking** — structure-aware, configurable chunk size and overlap; small/tabular units are preserved intact.
- **Hybrid retrieval** — FAISS semantic search fused with BM25 keyword search via Reciprocal Rank Fusion.
- **Optional reranking** — Hugging Face cross-encoder reranks the top candidates.
- **Agentic intent router** — `DOCUMENT_QA`, `DATA_ANALYSIS`, `SUMMARY`, `COMPARISON`, `GENERAL`.
- **Grounded answers with citations** — source name, page/row range, chunk id and retrieved text in expandable sections.
- **Query rewriting** — follow-up questions become standalone retrieval queries using conversation history.
- **Conversational memory** — modern LangChain message history held in Streamlit session state (no deprecated memory APIs).
- **Excel/CSV intelligence** — row counts, averages, group-by extremes, filters, missing values, duplicates and column schemas computed deterministically.
- **Document comparison** — structured markdown comparison table grounded in retrieved context.
- **Hierarchical summarisation** — map-reduce summarisation for large documents with short/detailed/bullet styles.
- **Persistent vector store** — FAISS index + JSON manifest survive app restarts; duplicate documents and chunks are prevented.
- **Document management** — view metadata, re-index, download and delete documents.
- **Analytics dashboard** — Plotly charts for query types, document usage, latency and failed retrievals, backed by SQLite.
- **Robust error handling** — invalid keys, unsupported/corrupt files, embedding failures and empty results all produce friendly messages, never tracebacks.
- **Security-minded** — API keys only from `.env`/UI, file allow-list, size limits, sanitised file names, no arbitrary code execution.

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            Streamlit UI (app.py)            │
                    │  Dashboard · Chat · Documents · Analytics    │
                    └───────────────┬─────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
   Upload ─────────▶│   Document Processing Pipeline │
                    │ validate → extract → clean →   │
                    │ metadata → chunk → embed →     │
                    │ index                          │
                    └───────┬───────────────┬────────┘
                            │               │
              tabular  ┌────▼────┐   text   ┌▼──────────────┐
                       │ pandas  │          │ FAISS + BM25  │
                       │frames   │          │ vector store  │
                       └────┬────┘          └──────┬────────┘
                            │                      │
                    ┌───────▼──────────────────────▼────────┐
                    │        DocumentAgent (intent router)   │
                    │ DOCUMENT_QA · DATA_ANALYSIS · SUMMARY · │
                    │ COMPARISON · GENERAL                    │
                    └───────┬───────────────────────┬────────┘
                            │                       │
                     pandas results            RAGEngine
                            │             rewrite → hybrid retrieve →
                            │             rerank → context → Gemini →
                            │             grounded answer + citations
                            └───────────┬───────────┘
                                        │
                                 AnalyticsTracker (SQLite)
```

**Design principles**

- Clear separation of concerns: loaders, RAG primitives, agents, analytics, UI.
- Heavy objects (embeddings, LLM, reranker) are cached; the vector store lives in `st.session_state` and is never rebuilt on reruns.
- Every external dependency that may not be installed (FAISS, sentence-transformers, BM25) is imported lazily and degrades gracefully.

---

## 4. Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| UI | Streamlit |
| Orchestration | LangChain (`langchain`, `langchain-core`, `langchain-community`) |
| LLM | Google Gemini via `langchain-google-genai` |
| Embeddings | Google Gemini or Hugging Face `sentence-transformers` |
| Vector store | FAISS (`faiss-cpu`) |
| Keyword search | `rank-bm25` |
| Reranking | Hugging Face cross-encoder |
| PDF | PyMuPDF |
| DOCX | python-docx |
| Data | pandas + openpyxl |
| Analytics | SQLite (stdlib) + Plotly |
| Config/secrets | python-dotenv |
| Validation | Pydantic (available for extension) |

---

## 5. Installation

> **Python version:** 3.11 or 3.12 are recommended (best wheel coverage for
> `faiss-cpu` and PyTorch/sentence-transformers). 3.13 generally works.

### Windows (PowerShell)

```powershell
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Environment Variables

Copy `.env.example` to `.env` and add your key:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Get a free key from <https://aistudio.google.com/app/apikey>.

- The key can also be entered directly in the Streamlit sidebar for local testing.
- `GEMINI_API_KEY` and `GOOGLE_GENAI_API_KEY` are accepted as aliases.
- Keys are never written to disk, logged, or committed.

---

## 7. How to Run

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. Then:

1. Enter your Google API key in the sidebar.
2. Upload one or more documents.
3. Click **Process & index documents**.
4. Open the **Chat** tab and ask questions.

---

## 8. Screenshots

> _Placeholder — add your own screenshots here._

| View | File |
| --- | --- |
| Dashboard | `docs/screenshots/dashboard.png` |
| Chat with citations | `docs/screenshots/chat.png` |
| Documents | `docs/screenshots/documents.png` |
| Analytics | `docs/screenshots/analytics.png` |

---

## 9. Example Questions

**Document Q&A**
- "What does the contract say about termination?"
- "According to the employee handbook, how many vacation days are provided?"
- "What is the notice period for resignation?"

**Data analysis (CSV/XLSX)**
- "How many employees are there?"
- "What is the average salary?"
- "Which department has the highest salary?"
- "Show employees with salary above 50000."
- "How many rows contain missing values?"
- "What are the duplicate records?"
- "What columns exist in this file?"

**Summarisation**
- "Summarize this document."
- "Give me a short summary."
- "Provide a bullet-point summary of all documents."

**Comparison**
- "Compare these two reports."
- "What changed between the two contracts?"
- "Which clauses are different?"

**Follow-up (query rewriting)**
- "What does the contract say about termination?" → "What about the notice period?"

---

## 10. Project Structure

```text
enterprise_rag/
├── app.py                     # Streamlit application
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── conftest.py
│
├── config/
│   └── settings.py            # Paths, model catalogue, AppSettings
│
├── loaders/
│   ├── base.py                # LoadedDocument, clean_text
│   ├── dispatcher.py          # Extension → loader routing
│   ├── pdf_loader.py          # PyMuPDF, page-preserving
│   ├── docx_loader.py         # Paragraphs + tables
│   ├── excel_loader.py        # Sheet-per-dataframe
│   ├── csv_loader.py          # Delimiter sniffing
│   ├── text_loader.py         # Multi-encoding text
│   └── tabular.py             # Shared dataframe → text logic
│
├── rag/
│   ├── embeddings.py          # Google / Hugging Face factory
│   ├── chunking.py            # Configurable recursive splitter
│   ├── vectorstore.py         # FAISS create/load/save/update/delete
│   ├── retriever.py           # Hybrid vector + BM25 (RRF)
│   ├── reranker.py            # Cross-encoder reranker
│   ├── prompts.py             # System + task prompts
│   ├── chain.py               # RAGEngine + LLM factory
│   ├── query_router.py        # Intent classification
│   └── summarizer.py          # Map-reduce summarisation
│
├── agents/
│   ├── document_agent.py      # Intent dispatch / orchestration
│   └── data_analyst.py        # Deterministic pandas analysis
│
├── analytics/
│   └── tracker.py             # SQLite telemetry
│
├── utils/
│   ├── file_utils.py          # Validation, hashing, safe saving
│   ├── metadata.py            # Chunk metadata helpers
│   ├── persistence.py         # Dataframe + conversation storage
│   └── logger.py              # Logging configuration
│
├── data/
│   ├── uploads/               # Stored uploads
│   ├── processed/             # Cached dataframes
│   ├── vectorstore/           # FAISS index + manifest
│   ├── conversations/         # Saved conversations
│   └── logs/                  # Rotating logs
│
└── tests/
    ├── test_loaders.py
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_rag.py
```

---

## 11. RAG Pipeline

```text
User Question
      ↓
Query Rewriting (conversation-aware)
      ↓
Hybrid Retriever ── Vector Search (FAISS) ─┐
                 └─ Keyword Search (BM25) ─┤
      ↓                                    │
Reciprocal Rank Fusion ◀────────────────────┘
      ↓
Top 10–20 candidates
      ↓
Cross-encoder Reranker (optional)
      ↓
Top 3–5 chunks
      ↓
Metadata filtering + Context construction
      ↓
Gemini LLM (grounded system prompt)
      ↓
Answer + Citations
```

Each chunk carries metadata:

```python
{
    "source": "employee_policy.pdf",
    "file_type": "PDF",
    "page": "12",
    "chunk_id": "b1f...-c00012",
    "document_id": "b1f...",
    # tabular files also add:
    "sheet": "Employees",
    "row_start": 25,
    "row_end": 40,
    "columns": "name, department, salary",
}
```

If no sufficient context is found the assistant responds exactly:

> I couldn't find sufficient information in the uploaded documents.

---

## 12. Future Improvements

- OCR / scanned-PDF support (Tesseract or Gemini Vision).
- True production hybrid search (e.g. BM25 with elasticsearch/open-search) behind the existing `HybridRetriever` abstraction.
- User authentication and multi-tenant namespaces.
- Token accounting and cost dashboards.
- Streaming responses via `st.write_stream`.
- Advanced table intelligence (sheet-level SQL via DuckDB).
- Evaluation harness (RAGAS) and regression tests for retrieval quality.
- Docker + CI pipeline.

---

## 13. Troubleshooting

| Issue | Resolution |
| --- | --- |
| `No Google API key found` | Add `GOOGLE_API_KEY` to `.env` or enter it in the sidebar. |
| API key rejected / quota exceeded | Verify the key in Google AI Studio and check quota/rate limits. |
| `faiss` fails to install | Use Python 3.11/3.12; on some platforms install `faiss-cpu` from conda. |
| Reranker unavailable warning | Install `sentence-transformers`; the app continues without reranking. |
| PDF returns "No extractable text" | It is likely a scanned PDF; OCR is not yet supported. |
| "The saved vector index could not be loaded" | The embedding model changed. Use **Clear vector database** and re-index. |
| Excel/CSV question answered weakly | Ensure the file was processed and check the **Data Analysis** intent label. |
| Streamlit shows stale data | Use the sidebar **Clear chat** / **New conversation** buttons. |

Logs are written to `data/logs/app.log`.

---

## 14. Testing

```bash
pytest -q
```

The suite covers loaders, chunking, hybrid retrieval, the RAG engine, the intent
router and the pandas data analyst (28 tests).

---

## 15. Resume Description

> **Enterprise AI Knowledge Assistant — Agentic RAG + Document Intelligence Platform**
> Built an end-to-end RAG platform (Python, LangChain, Gemini, FAISS, Streamlit)
> that ingests PDF/DOCX/TXT/CSV/XLSX documents, performs hybrid semantic + BM25
> retrieval with cross-encoder reranking, and produces citation-grounded answers.
> Implemented an intent-routing agent that delegates numerical questions to
> deterministic pandas pipelines, supports conversational query rewriting,
> map-reduce summarisation and structured document comparison. Delivered a
> persistent FAISS index with duplicate prevention, a SQLite-backed analytics
> dashboard, robust error handling and a modular, tested architecture.

---

## License

Released for portfolio and educational use. Add your preferred license before
redistribution.
