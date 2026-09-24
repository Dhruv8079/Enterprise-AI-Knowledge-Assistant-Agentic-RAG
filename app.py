"""Enterprise AI Knowledge Assistant — Streamlit application.

An agentic RAG + document intelligence platform built with LangChain, Gemini,
FAISS and Streamlit. Run with::

    streamlit run app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.data_analyst import DataAnalyst, build_datasets
from agents.document_agent import DocumentAgent
from analytics.tracker import get_tracker
from config.settings import (
    AppSettings,
    EMBEDDING_PROVIDERS,
    LLM_MODELS,
    RERANKER_MODELS,
    SUPPORTED_EXTENSIONS,
    UPLOADS_DIR,
    get_env_api_key,
    mask_secret,
)
from loaders import DocumentLoadError, load_document
from rag.chain import LLMConfigurationError, RAGEngine, build_llm
from rag.chunking import ChunkingError, chunk_documents
from rag.embeddings import EmbeddingConfigurationError, get_embeddings
from rag.query_router import INTENT_LABELS
from rag.reranker import Reranker
from rag.retriever import HybridRetriever
from rag.summarizer import Summarizer
from rag.vectorstore import VectorStoreError, VectorStoreManager
from utils.file_utils import (
    FileValidationError,
    compute_file_hash,
    human_size,
    make_document_id,
    save_upload_bytes,
    validate_file,
)
from utils.logger import get_logger
from utils.metadata import utc_timestamp
from utils.persistence import (
    conversation_to_markdown,
    delete_dataframes,
    load_all_dataframes,
    save_conversation,
    save_dataframes,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_TYPES = [extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS]


# --------------------------------------------------------------------------- #
# Cached resource factories
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _cached_llm(api_key: str, model: str, temperature: float) -> Any:
    """Return a cached Gemini chat model."""
    return build_llm(api_key, model, temperature)


@st.cache_resource(show_spinner=False)
def _cached_reranker(model_name: str) -> Reranker:
    """Return a cached cross-encoder reranker."""
    return Reranker(model_name)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Initialise all Streamlit session-state keys exactly once."""
    ss = st.session_state
    ss.setdefault("settings", AppSettings())
    ss.setdefault("chat_history", [])
    ss.setdefault("documents", [])
    ss.setdefault("dataframes", load_all_dataframes())
    ss.setdefault("api_key", get_env_api_key())
    ss.setdefault("vsm", None)
    ss.setdefault("vsm_signature", None)
    ss.setdefault("conversation_id", str(uuid.uuid4()))
    ss.setdefault("last_processing", [])


def get_vector_store(settings: AppSettings, api_key: str) -> VectorStoreManager | None:
    """Return the session vector store, (re)creating it when needed."""
    signature = settings.embedding_signature()
    stored = st.session_state.get("vsm")
    if stored is not None and st.session_state.get("vsm_signature") == signature:
        return stored

    try:
        embeddings = get_embeddings(
            settings.embedding_provider, settings.embedding_model, api_key
        )
        manager = VectorStoreManager(embeddings, signature)
    except (EmbeddingConfigurationError, VectorStoreError) as exc:
        st.sidebar.error(str(exc))
        return None

    st.session_state["vsm"] = manager
    st.session_state["vsm_signature"] = signature
    if manager.signature_mismatch:
        st.sidebar.warning(
            "The embedding model changed. Re-index your documents to search them again."
        )
    return manager


def source_map() -> dict[str, str]:
    """Return a document id to source name mapping."""
    return {
        record["document_id"]: record.get("source", "document")
        for record in st.session_state.documents
    }


def sync_documents(manager: VectorStoreManager) -> None:
    """Merge persisted manifest entries into the session document list."""
    known = {record["document_id"]: record for record in st.session_state.documents}
    for meta in manager.get_documents():
        document_id = meta["document_id"]
        known[document_id] = {
            "document_id": document_id,
            "source": meta.get("source", "unknown"),
            "file_type": meta.get("file_type", "?"),
            "size_bytes": int(meta.get("size_bytes", 0) or 0),
            "num_units": int(meta.get("num_units", 0) or 0),
            "chunks": int(meta.get("chunks", 0) or 0),
            "status": "indexed",
            "indexed_at": meta.get("indexed_at", utc_timestamp()),
            "stored_file": meta.get("stored_file"),
            "file_hash": meta.get("file_hash"),
        }
    st.session_state.documents = sorted(
        known.values(), key=lambda item: item.get("source", "")
    )


def register_document(record: dict[str, Any]) -> None:
    """Insert or replace a document record in session state."""
    existing = {
        item["document_id"]: item for item in st.session_state.documents
    }
    existing[record["document_id"]] = record
    st.session_state.documents = sorted(
        existing.values(), key=lambda item: item.get("source", "")
    )


# --------------------------------------------------------------------------- #
# Document processing
# --------------------------------------------------------------------------- #
def process_uploads(files: list[Any], settings: AppSettings, api_key: str) -> None:
    """Validate, load, chunk and index uploaded files.

    Args:
        files: Streamlit ``UploadedFile`` objects.
        settings: Active application settings.
        api_key: Google API key (may be empty for HF embeddings).
    """
    if not files:
        st.sidebar.warning("Select at least one file to process.")
        return

    manager = get_vector_store(settings, api_key)
    if manager is None:
        return

    tracker = get_tracker()
    statuses: list[dict[str, Any]] = []
    total = len(files)
    progress = st.sidebar.progress(0.0, text="Processing documents...")

    for index, uploaded in enumerate(files):
        name = uploaded.name
        try:
            data = uploaded.getvalue()
            validate_file(name, len(data))
            file_hash = compute_file_hash(data)
            document_id = make_document_id(file_hash, name)

            if manager.document_exists(document_id):
                statuses.append(
                    {"File": name, "Status": "Already indexed", "Detail": "Duplicate skipped"}
                )
                progress.progress((index + 1) / total)
                continue

            stored_path = save_upload_bytes(data, name, file_hash)
            loaded = load_document(stored_path, document_id, name)
            chunks = chunk_documents(
                loaded.documents, settings.chunk_size, settings.chunk_overlap
            )
            metadata = {
                "source": name,
                "file_type": loaded.file_type,
                "size_bytes": loaded.size_bytes,
                "num_units": loaded.num_units,
                "stored_file": stored_path.name,
                "file_hash": file_hash,
            }
            result = manager.add_chunks(chunks, metadata)

            if loaded.dataframes:
                save_dataframes(document_id, loaded.dataframes)
                st.session_state.dataframes[document_id] = loaded.dataframes

            record = {
                "document_id": document_id,
                "source": name,
                "file_type": loaded.file_type,
                "size_bytes": loaded.size_bytes,
                "num_units": loaded.num_units,
                "chunks": int(result.added_chunks),
                "status": "indexed",
                "indexed_at": utc_timestamp(),
                "stored_file": stored_path.name,
                "file_hash": file_hash,
            }
            register_document(record)
            tracker.upsert_document(record)
            statuses.append(
                {
                    "File": name,
                    "Status": "Indexed",
                    "Detail": f"{result.added_chunks} chunk(s), {loaded.num_units} unit(s)",
                }
            )
        except FileValidationError as exc:
            statuses.append({"File": name, "Status": "Rejected", "Detail": str(exc)})
        except (DocumentLoadError, ChunkingError, VectorStoreError) as exc:
            statuses.append({"File": name, "Status": "Error", "Detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            logger.exception("Unexpected error processing %s", name)
            statuses.append(
                {"File": name, "Status": "Error", "Detail": f"Unexpected error: {exc}"}
            )
        progress.progress((index + 1) / total)

    progress.empty()
    st.session_state["last_processing"] = statuses


def reindex_document(
    document_id: str, settings: AppSettings, api_key: str
) -> None:
    """Re-extract and re-index a stored document."""
    manager = get_vector_store(settings, api_key)
    if manager is None:
        return

    record = next(
        (item for item in st.session_state.documents if item["document_id"] == document_id),
        None,
    )
    if record is None:
        st.error("Document not found.")
        return

    stored_file = record.get("stored_file")
    path = UPLOADS_DIR / stored_file if stored_file else None
    if path is None or not path.exists():
        st.error("The original file is no longer available for re-indexing.")
        return

    try:
        loaded = load_document(path, document_id, record["source"])
        chunks = chunk_documents(
            loaded.documents, settings.chunk_size, settings.chunk_overlap
        )
        metadata = {
            "source": record["source"],
            "file_type": loaded.file_type,
            "size_bytes": loaded.size_bytes,
            "num_units": loaded.num_units,
            "stored_file": stored_file,
            "file_hash": record.get("file_hash"),
        }
        result = manager.add_chunks(chunks, metadata, force=True)
        if loaded.dataframes:
            save_dataframes(document_id, loaded.dataframes)
            st.session_state.dataframes[document_id] = loaded.dataframes
        record["chunks"] = int(result.added_chunks)
        record["status"] = "indexed"
        record["indexed_at"] = utc_timestamp()
        register_document(record)
        get_tracker().upsert_document(record)
        st.success(f"Re-indexed '{record['source']}' with {result.added_chunks} chunk(s).")
    except (DocumentLoadError, ChunkingError, VectorStoreError) as exc:
        st.error(str(exc))


def delete_document(document_id: str, api_key: str) -> None:
    """Delete a document from the vector store and all registries."""
    settings: AppSettings = st.session_state.settings
    manager = get_vector_store(settings, api_key)
    if manager is not None:
        manager.delete_document(document_id)
    delete_dataframes(document_id)
    st.session_state.dataframes.pop(document_id, None)
    get_tracker().delete_document(document_id)
    st.session_state.documents = [
        item for item in st.session_state.documents if item["document_id"] != document_id
    ]


def clear_vector_database(api_key: str) -> None:
    """Clear the vector store, document registry and dataframe cache."""
    settings: AppSettings = st.session_state.settings
    manager = get_vector_store(settings, api_key)
    if manager is not None:
        manager.clear()
    for document_id in list(st.session_state.dataframes.keys()):
        delete_dataframes(document_id)
    st.session_state.dataframes = {}
    st.session_state.documents = []
    get_tracker().clear_documents()


# --------------------------------------------------------------------------- #
# Agent construction
# --------------------------------------------------------------------------- #
def build_agent(
    settings: AppSettings, api_key: str, manager: VectorStoreManager
) -> DocumentAgent:
    """Construct the agent graph for the current settings."""
    llm = _cached_llm(api_key, settings.llm_model, settings.temperature)
    retriever = HybridRetriever(
        manager,
        k=settings.top_k,
        fetch_k=settings.fetch_k,
        use_hybrid=settings.use_hybrid,
        filter_document_ids=settings.selected_documents or None,
    )
    reranker = _cached_reranker(settings.reranker_model) if settings.use_reranker else None
    engine = RAGEngine(llm, retriever, reranker, top_k=settings.top_k)
    analyst = DataAnalyst(llm)
    summarizer = Summarizer(llm)
    return DocumentAgent(llm, engine, analyst, summarizer, manager)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render citations as expanders."""
    if not sources:
        return
    st.markdown("**Sources**")
    for source in sources:
        label = source.get("label") or f"{source.get('source')} — {source.get('page')}"
        with st.expander(f"{source.get('index', '')}. {label}"):
            st.caption(
                f"Document: {source.get('source')} | Type: {source.get('file_type')} | "
                f"Page: {source.get('page')} | Chunk: {source.get('chunk_id')}"
            )
            st.markdown(source.get("text", "")[:2500])


def render_chat_history() -> None:
    """Render the stored conversation."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))
            if message["role"] == "assistant" and message.get("intent"):
                st.caption(
                    f"Intent: {INTENT_LABELS.get(message['intent'], message['intent'])} | "
                    f"Response time: {message.get('latency', 0):.2f}s | "
                    f"Chunks used: {message.get('retrieved', 0)}"
                )


def current_datasets() -> list[Any]:
    """Build the list of datasets available for data analysis."""
    return build_datasets(st.session_state.dataframes, source_map())


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard(manager: VectorStoreManager | None) -> None:
    """Render the dashboard overview."""
    st.subheader("Dashboard")
    summary = get_tracker().get_summary()

    column_a, column_b, column_c, column_d = st.columns(4)
    column_a.metric("Documents", len(st.session_state.documents))
    column_b.metric(
        "Indexed chunks", manager.total_chunks if manager is not None else 0
    )
    column_c.metric("Questions asked", summary["total_queries"])
    column_d.metric("Avg response time", f"{summary['avg_latency']:.2f}s")

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Recent documents")
        if st.session_state.documents:
            frame = pd.DataFrame(st.session_state.documents)[
                ["source", "file_type", "chunks", "status", "indexed_at"]
            ]
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("No documents indexed yet. Upload files from the sidebar to begin.")

    with right:
        st.markdown("### Getting started")
        st.markdown(
            "1. Enter your Google API key in the sidebar.\n"
            "2. Upload PDF, DOCX, TXT, CSV or XLSX files.\n"
            "3. Click **Process & Index documents**.\n"
            "4. Open the **Chat** tab and start asking questions."
        )
        if summary["failed_retrievals"]:
            st.warning(
                f"{summary['failed_retrievals']} question(s) could not be answered "
                "from the knowledge base."
            )


def page_chat(settings: AppSettings, api_key: str, manager: VectorStoreManager | None) -> None:
    """Render the chat experience."""
    st.subheader("Chat")
    if manager is None or manager.is_empty:
        st.info("Index at least one document to start chatting.")
        render_chat_history()
        return

    render_chat_history()

    prompt = st.chat_input("Ask a question about your documents...")
    if not prompt:
        return

    if settings.embedding_provider == "Google Gemini" and not api_key:
        st.error("Please enter your Google API key in the sidebar before asking questions.")
        return

    history = list(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tracker = get_tracker()
    response_payload: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "sources": [],
        "intent": "DOCUMENT_QA",
        "latency": 0.0,
        "retrieved": 0,
        "timestamp": utc_timestamp(),
    }

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                agent = build_agent(settings, api_key, manager)
                result = agent.handle(
                    prompt,
                    history=history,
                    selected_document_ids=settings.selected_documents,
                    datasets=current_datasets(),
                )
                response_payload.update(
                    {
                        "content": result.answer,
                        "sources": [source.as_dict() for source in result.sources],
                        "intent": result.intent,
                        "latency": result.latency,
                        "retrieved": result.retrieved_count,
                        "rewritten_query": result.rewritten_query,
                    }
                )
                tracker.log_query(
                    question=prompt,
                    intent=result.intent,
                    latency=result.latency,
                    retrieved=result.retrieved_count,
                    used_fallback=result.used_fallback,
                    sources=[source.source for source in result.sources],
                )
            except LLMConfigurationError as exc:
                response_payload["content"] = str(exc)
                response_payload["intent"] = "GENERAL"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat request failed")
                response_payload["content"] = (
                    "An unexpected error occurred while answering your question. "
                    f"Details: {exc}"
                )

        st.markdown(response_payload["content"])
        render_sources(response_payload["sources"])
        st.caption(
            f"Intent: {INTENT_LABELS.get(response_payload['intent'], response_payload['intent'])} | "
            f"Response time: {response_payload['latency']:.2f}s"
        )

    st.session_state.chat_history.append(response_payload)
    save_conversation(st.session_state.conversation_id, st.session_state.chat_history)


def page_documents(settings: AppSettings, api_key: str, manager: VectorStoreManager | None) -> None:
    """Render document management."""
    st.subheader("Documents")
    documents = st.session_state.documents
    if not documents:
        st.info("No documents yet. Upload files from the sidebar.")
        return

    table = pd.DataFrame(
        [
            {
                "Document": record["source"],
                "Type": record["file_type"],
                "Size": human_size(int(record.get("size_bytes", 0))),
                "Units": record.get("num_units", 0),
                "Chunks": record.get("chunks", 0),
                "Status": record.get("status", "indexed"),
            }
            for record in documents
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("### Manage documents")
    for record in documents:
        title = (
            f"{record['source']} | {record['file_type']} | "
            f"{record.get('chunks', 0)} chunk(s)"
        )
        with st.expander(title):
            metadata = {
                key: record.get(key)
                for key in (
                    "document_id",
                    "file_type",
                    "size_bytes",
                    "num_units",
                    "chunks",
                    "status",
                    "indexed_at",
                    "stored_file",
                )
            }
            st.json(metadata)
            has_data = record["document_id"] in st.session_state.dataframes
            if has_data:
                st.caption("Structured data available for analysis.")

            action_a, action_b, action_c = st.columns(3)
            if action_a.button("Re-index", key=f"reindex_{record['document_id']}"):
                reindex_document(record["document_id"], settings, api_key)
                st.rerun()
            if action_b.button("Delete", key=f"delete_{record['document_id']}"):
                delete_document(record["document_id"], api_key)
                st.toast(f"Deleted {record['source']}")
                st.rerun()
            stored_file = record.get("stored_file")
            path = UPLOADS_DIR / stored_file if stored_file else None
            if path is not None and path.exists():
                action_c.download_button(
                    "Download",
                    data=path.read_bytes(),
                    file_name=record["source"],
                    key=f"download_{record['document_id']}",
                )


def page_analytics(manager: VectorStoreManager | None) -> None:
    """Render the analytics dashboard."""
    st.subheader("Analytics")
    tracker = get_tracker()
    summary = tracker.get_summary()

    column_a, column_b, column_c, column_d = st.columns(4)
    column_a.metric("Documents", summary["total_documents"])
    column_b.metric("Total chunks", summary["total_chunks"])
    column_c.metric("Total questions", summary["total_queries"])
    column_d.metric("Avg chunks retrieved", f"{summary['avg_retrieved']:.1f}")

    if summary["total_queries"] == 0:
        st.info("No queries recorded yet. Ask a question in the Chat tab.")
        return

    left, right = st.columns(2)
    with left:
        query_types = summary["query_types"]
        if query_types:
            figure = px.pie(
                names=list(query_types.keys()),
                values=list(query_types.values()),
                title="Query types",
                hole=0.4,
            )
            st.plotly_chart(figure, use_container_width=True)
    with right:
        usage = summary["document_usage"]
        if usage:
            figure = px.bar(
                x=list(usage.keys()),
                y=list(usage.values()),
                labels={"x": "Document", "y": "Times cited"},
                title="Document usage",
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No cited sources recorded yet.")

    st.divider()
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Failed retrievals", summary["failed_retrievals"])
    metric_b.metric("Avg response time", f"{summary['avg_latency']:.2f}s")

    st.markdown("### Recent queries")
    recent = tracker.get_recent_queries(25)
    if recent:
        frame = pd.DataFrame(recent)[
            ["timestamp", "question", "intent", "latency", "retrieved", "used_fallback"]
        ].rename(
            columns={
                "timestamp": "Time",
                "question": "Question",
                "intent": "Intent",
                "latency": "Seconds",
                "retrieved": "Chunks",
                "used_fallback": "Fallback",
            }
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)


def page_settings(settings: AppSettings, api_key: str, manager: VectorStoreManager | None) -> None:
    """Render application settings and diagnostics."""
    st.subheader("Settings")
    st.markdown("#### Current configuration")
    st.json(settings.as_dict())

    st.markdown("#### Environment")
    st.write(
        {
            "API key": mask_secret(api_key),
            "Data directory": str(Path(UPLOADS_DIR).parent),
            "Vector store": str(manager.persist_dir) if manager else "not initialised",
            "Vector chunks": manager.total_chunks if manager else 0,
            "Reranker available": (
                _cached_reranker(settings.reranker_model).available
                if settings.use_reranker
                else False
            ),
        }
    )

    st.markdown("#### Reference")
    st.caption("Available Gemini models: " + ", ".join(LLM_MODELS))
    st.caption("Supported file types: " + ", ".join(SUPPORTED_EXTENSIONS))

    st.divider()
    st.markdown("#### Danger zone")
    if st.button("Clear vector database"):
        clear_vector_database(api_key)
        st.session_state["chat_history"] = []
        st.success("Vector database cleared.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> tuple[AppSettings, str]:
    """Render the sidebar and return the active settings and API key."""
    settings: AppSettings = st.session_state.settings

    with st.sidebar:
        st.title("Enterprise AI Knowledge Assistant")
        st.caption("Agentic RAG + Document Intelligence Platform")

        api_key = st.text_input(
            "Google API key",
            value=st.session_state.api_key,
            type="password",
            help="Used for Gemini LLM and (optionally) embeddings. Never stored on disk.",
        )
        st.session_state.api_key = api_key.strip() or get_env_api_key()

        with st.expander("Model settings", expanded=True):
            settings.llm_model = st.selectbox(
                "LLM model",
                options=LLM_MODELS,
                index=(
                    LLM_MODELS.index(settings.llm_model)
                    if settings.llm_model in LLM_MODELS
                    else 0
                ),
            )
            settings.temperature = st.slider(
                "Temperature", 0.0, 1.0, float(settings.temperature), 0.05
            )
            provider = st.selectbox(
                "Embedding provider",
                options=list(EMBEDDING_PROVIDERS.keys()),
                index=(
                    list(EMBEDDING_PROVIDERS.keys()).index(settings.embedding_provider)
                    if settings.embedding_provider in EMBEDDING_PROVIDERS
                    else 0
                ),
            )
            settings.embedding_provider = provider
            models = EMBEDDING_PROVIDERS[provider]
            default_model = (
                settings.embedding_model
                if settings.embedding_model in models
                else models[0]
            )
            settings.embedding_model = st.selectbox(
                "Embedding model",
                options=models,
                index=models.index(default_model),
            )

        with st.expander("Retrieval settings", expanded=False):
            settings.top_k = st.slider("Top-K results", 1, 15, int(settings.top_k))
            settings.fetch_k = st.slider(
                "Candidate pool (fetch-K)", 5, 50, int(settings.fetch_k)
            )
            settings.use_hybrid = st.toggle(
                "Hybrid retrieval (vector + keyword)", value=bool(settings.use_hybrid)
            )
            settings.use_reranker = st.toggle(
                "Cross-encoder reranking", value=bool(settings.use_reranker)
            )
            if settings.use_reranker:
                settings.reranker_model = st.selectbox(
                    "Reranker model",
                    options=RERANKER_MODELS,
                    index=(
                        RERANKER_MODELS.index(settings.reranker_model)
                        if settings.reranker_model in RERANKER_MODELS
                        else 0
                    ),
                )

        with st.expander("Chunking settings", expanded=False):
            settings.chunk_size = st.slider(
                "Chunk size", 200, 3000, int(settings.chunk_size), 100
            )
            settings.chunk_overlap = st.slider(
                "Chunk overlap", 0, 500, int(settings.chunk_overlap), 25
            )
            if settings.chunk_overlap >= settings.chunk_size:
                st.warning("Overlap must be smaller than the chunk size.")

        st.divider()
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, CSV, XLSX. Max 50 MB per file.",
        )
        if st.button("Process & index documents", use_container_width=True):
            process_uploads(uploaded_files or [], settings, st.session_state.api_key)

        if st.session_state.get("last_processing"):
            with st.expander("Last processing result", expanded=True):
                st.dataframe(
                    pd.DataFrame(st.session_state["last_processing"]),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()
        documents = st.session_state.documents
        if documents:
            labels = {record["document_id"]: record["source"] for record in documents}
            selected = st.multiselect(
                "Limit to documents",
                options=list(labels.keys()),
                default=[
                    value for value in settings.selected_documents if value in labels
                ],
                format_func=lambda value: labels.get(value, value),
                help="Restrict retrieval, summarisation and comparison to these documents.",
            )
            settings.selected_documents = selected

        st.divider()
        manager = get_vector_store(settings, st.session_state.api_key)
        if manager is not None:
            sync_documents(manager)

        status_a, status_b = st.columns(2)
        status_a.metric("Documents", len(st.session_state.documents))
        status_b.metric(
            "Chunks", manager.total_chunks if manager is not None else 0
        )
        key_status = "set" if settings.embedding_provider != "Google Gemini" or st.session_state.api_key else "missing"
        st.caption(
            f"API key: {key_status} | Embedding: {settings.embedding_model} | "
            f"LLM: {settings.llm_model}"
        )

        if st.button("New conversation", use_container_width=True):
            if st.session_state.chat_history:
                save_conversation(
                    st.session_state.conversation_id, st.session_state.chat_history
                )
            st.session_state.chat_history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("Clear vector database", use_container_width=True):
            clear_vector_database(st.session_state.api_key)
            st.session_state.chat_history = []
            st.toast("Vector database cleared.")
            st.rerun()

        if st.session_state.chat_history:
            st.download_button(
                "Download conversation",
                data=conversation_to_markdown(st.session_state.chat_history),
                file_name=f"conversation_{st.session_state.conversation_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    return settings, st.session_state.api_key


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Application entry point."""
    init_state()
    settings, api_key = render_sidebar()
    manager = st.session_state.get("vsm")

    if manager is not None and manager.signature_mismatch:
        st.warning(
            "The active embedding model is different from the one used to build the "
            "saved index. Re-index your documents or restore the previous model."
        )

    tab_dashboard, tab_chat, tab_documents, tab_analytics, tab_settings = st.tabs(
        ["Dashboard", "Chat", "Documents", "Analytics", "Settings"]
    )

    with tab_dashboard:
        page_dashboard(manager)
    with tab_chat:
        page_chat(settings, api_key, manager)
    with tab_documents:
        page_documents(settings, api_key, manager)
    with tab_analytics:
        page_analytics(manager)
    with tab_settings:
        page_settings(settings, api_key, manager)


if __name__ == "__main__":
    main()
