"""Optional cross-encoder reranking layer.

Uses a lightweight Hugging Face cross-encoder to reorder the candidate chunks
returned by hybrid retrieval. If ``sentence-transformers`` is unavailable the
reranker becomes a transparent pass-through so the application still works.
"""

from __future__ import annotations

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)

_CROSS_ENCODER_CACHE: dict[str, object] = {}


class RerankerError(Exception):
    """Raised when the reranker cannot be initialised."""


class Reranker:
    """Cross-encoder reranker.

    Args:
        model_name: Hugging Face cross-encoder model id.
        max_length: Maximum sequence length passed to the encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the cross-encoder model is loaded and usable."""
        self._ensure_model()
        return self._model is not None

    def _ensure_model(self) -> None:
        """Lazily load the cross-encoder, caching it process-wide."""
        if self._model is not None or self._load_error is not None:
            return

        cached = _CROSS_ENCODER_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached
            return

        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name, max_length=self.max_length)
            _CROSS_ENCODER_CACHE[self.model_name] = model
            self._model = model
            logger.info("Loaded reranker model '%s'", self.model_name)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._load_error = str(exc)
            logger.warning("Reranker unavailable (%s); continuing without reranking.", exc)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Rerank ``documents`` against ``query``.

        Args:
            query: The user query.
            documents: Candidate documents from retrieval.
            top_n: Number of documents to keep. ``None`` keeps all.

        Returns:
            Documents ordered by cross-encoder relevance score.
        """
        if not documents:
            return []

        self._ensure_model()
        if self._model is None:
            return documents[:top_n] if top_n else documents

        pairs = [(query, document.page_content) for document in documents]
        try:
            scores = self._model.predict(pairs)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranking failed (%s); returning original order.", exc)
            return documents[:top_n] if top_n else documents

        scored = sorted(
            zip(documents, (float(score) for score in scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        reranked = [document for document, _ in scored]
        return reranked[:top_n] if top_n else reranked
