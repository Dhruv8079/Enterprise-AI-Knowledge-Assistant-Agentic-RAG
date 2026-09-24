"""Embedding model factory.

Supports Google Gemini embeddings (via ``langchain-google-genai``) and local
Hugging Face sentence-transformers embeddings. Instances are cached per
``(provider, model, api_key)`` so that re-runs do not re-create models.
"""

from __future__ import annotations

from typing import Any

from config.settings import get_env_api_key
from utils.logger import get_logger

logger = get_logger(__name__)

_EMBEDDING_CACHE: dict[tuple[str, str, str], Any] = {}


class EmbeddingConfigurationError(Exception):
    """Raised when an embedding model cannot be constructed."""


def embedding_signature(provider: str, model: str) -> str:
    """Return a stable signature used to detect embedding changes."""
    return f"{provider}::{model}"


def get_embeddings(
    provider: str,
    model: str,
    api_key: str | None = None,
    *,
    use_cache: bool = True,
) -> Any:
    """Construct (or return a cached) embeddings object.

    Args:
        provider: ``"Google Gemini"`` or ``"Hugging Face (local)"``.
        model: Model identifier for the chosen provider.
        api_key: Google API key (required for the Google provider).
        use_cache: Whether to reuse a previously constructed instance.

    Returns:
        A LangChain embeddings object.

    Raises:
        EmbeddingConfigurationError: If the provider is unknown or the model
            cannot be initialised.
    """
    resolved_key = (api_key or get_env_api_key() or "") if provider == "Google Gemini" else ""
    cache_key = (provider, model, resolved_key)

    if use_cache and cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    logger.info("Initialising embeddings: provider=%s model=%s", provider, model)

    try:
        if provider == "Google Gemini":
            if not resolved_key:
                raise EmbeddingConfigurationError(
                    "A Google API key is required to use Gemini embeddings."
                )
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=resolved_key,
            )
        elif provider == "Hugging Face (local)":
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=model,
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            raise EmbeddingConfigurationError(f"Unknown embedding provider: {provider}")
    except EmbeddingConfigurationError:
        raise
    except ImportError as exc:
        raise EmbeddingConfigurationError(
            "The required embedding library is not installed. "
            "For Hugging Face embeddings run 'pip install langchain-huggingface sentence-transformers'."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingConfigurationError(f"Failed to initialise embeddings: {exc}") from exc

    if use_cache:
        _EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


def clear_embedding_cache() -> None:
    """Clear the in-process embedding cache."""
    _EMBEDDING_CACHE.clear()
