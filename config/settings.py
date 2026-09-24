"""Central application configuration.

All tunable constants, filesystem paths, supported model lists and default
runtime settings live here so that the rest of the code base never has to
hard-code values.  Secrets are always read from the environment (``.env``)
and are never written to disk or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env (if present) exactly once.
load_dotenv(BASE_DIR / ".env")

DATA_DIR: Path = BASE_DIR / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
PROCESSED_DIR: Path = DATA_DIR / "processed"
VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
CONVERSATIONS_DIR: Path = DATA_DIR / "conversations"
LOGS_DIR: Path = DATA_DIR / "logs"
DB_PATH: Path = DATA_DIR / "app.db"

for _directory in (
    DATA_DIR,
    UPLOADS_DIR,
    PROCESSED_DIR,
    VECTORSTORE_DIR,
    CONVERSATIONS_DIR,
    LOGS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls")

FILE_TYPE_LABELS: dict[str, str] = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".txt": "TXT",
    ".csv": "CSV",
    ".xlsx": "XLSX",
    ".xls": "XLS",
}

MAX_FILE_SIZE_MB: int = 50
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# Maximum number of rows materialised in memory per CSV/XLSX sheet.
MAX_DATA_ROWS: int = 200_000

# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #
LLM_MODELS: list[str] = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

GOOGLE_EMBEDDING_MODELS: list[str] = [
    "models/gemini-embedding-001",
    "models/text-embedding-004",
    "models/embedding-001",
]

HF_EMBEDDING_MODELS: list[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
]

RERANKER_MODELS: list[str] = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]

EMBEDDING_PROVIDERS: dict[str, list[str]] = {
    "Google Gemini": GOOGLE_EMBEDDING_MODELS,
    "Hugging Face (local)": HF_EMBEDDING_MODELS,
}

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_LLM_MODEL: str = "gemini-3.6-flash"
DEFAULT_EMBEDDING_PROVIDER: str = "Google Gemini"
DEFAULT_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
DEFAULT_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 150
DEFAULT_TOP_K: int = 5
DEFAULT_FETCH_K: int = 20
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_USE_RERANK: bool = True
DEFAULT_USE_HYBRID: bool = True

# Exact message shown when the knowledge base cannot answer a question.
FALLBACK_MESSAGE: str = "I couldn't find sufficient information in the uploaded documents."

# Conversation history window forwarded to the LLM / query rewriter.
MAX_HISTORY_TURNS: int = 8

# --------------------------------------------------------------------------- #
# Application settings object
# --------------------------------------------------------------------------- #
@dataclass
class AppSettings:
    """Mutable, user-configurable settings for a single session."""

    llm_model: str = DEFAULT_LLM_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    use_reranker: bool = DEFAULT_USE_RERANK
    use_hybrid: bool = DEFAULT_USE_HYBRID
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    fetch_k: int = DEFAULT_FETCH_K
    selected_documents: list[str] = field(default_factory=list)

    def embedding_signature(self) -> str:
        """Stable identifier used to detect embedding-model changes."""
        return f"{self.embedding_provider}::{self.embedding_model}"

    def as_dict(self) -> dict:
        """Return the settings as a JSON-serialisable dictionary."""
        return {
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "use_reranker": self.use_reranker,
            "use_hybrid": self.use_hybrid,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "selected_documents": list(self.selected_documents),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AppSettings":
        """Reconstruct settings from a dictionary, ignoring unknown keys."""
        valid = {f: payload[f] for f in cls.__dataclass_fields__ if f in payload}
        return cls(**valid)


def get_env_api_key() -> str:
    """Return the Google API key from the environment (never logged)."""
    return (
        os.getenv("GOOGLE_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_GENAI_API_KEY", "")
    ).strip()


def mask_secret(secret: str) -> str:
    """Return a safe, masked representation of a secret for display."""
    if not secret:
        return "(not set)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * 6}{secret[-4:]}"
