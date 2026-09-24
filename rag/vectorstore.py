"""FAISS-backed vector store manager.

Responsible for creating, loading, updating and deleting a persistent FAISS
index together with a JSON manifest that records document-level metadata.
Duplicate chunks and duplicate documents are both prevented, and the index is
never rebuilt unless the caller explicitly asks for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.settings import VECTORSTORE_DIR
from utils.logger import get_logger
from utils.metadata import utc_timestamp

logger = get_logger(__name__)

INDEX_NAME: str = "index"
MANIFEST_FILE: str = "manifest.json"
SIGNATURE_FILE: str = "embedding_signature.txt"


class VectorStoreError(Exception):
    """Raised when the vector store cannot be created or loaded."""


@dataclass
class AddResult:
    """Outcome of adding a document to the vector store."""

    document_id: str
    added_chunks: int
    skipped: bool
    message: str


class VectorStoreManager:
    """Manage a persistent FAISS index with document-level metadata.

    Args:
        embeddings: A LangChain embeddings instance.
        embedding_signature: Signature of the embeddings model, used to detect
            incompatible model changes between runs.
        persist_dir: Directory in which the index is stored.
    """

    def __init__(
        self,
        embeddings: Any,
        embedding_signature: str,
        persist_dir: Path | str = VECTORSTORE_DIR,
    ) -> None:
        self.embeddings = embeddings
        self.embedding_signature = embedding_signature
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.store: Any | None = None
        self.manifest: dict[str, dict[str, Any]] = {}
        self.signature_mismatch: bool = False
        self._bm25_documents: list[Document] | None = None

        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    @property
    def _index_file(self) -> Path:
        return self.persist_dir / f"{INDEX_NAME}.faiss"

    @property
    def _manifest_path(self) -> Path:
        return self.persist_dir / MANIFEST_FILE

    @property
    def _signature_path(self) -> Path:
        return self.persist_dir / SIGNATURE_FILE

    def _load(self) -> None:
        """Load the index and manifest from disk if they exist."""
        if not self._manifest_path.exists():
            return

        try:
            stored_signature = (
                self._signature_path.read_text(encoding="utf-8").strip()
                if self._signature_path.exists()
                else ""
            )
        except OSError:
            stored_signature = ""

        if stored_signature and stored_signature != self.embedding_signature:
            self.signature_mismatch = True
            logger.warning(
                "Embedding signature changed (%s -> %s). Re-indexing is required.",
                stored_signature,
                self.embedding_signature,
            )
            return

        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            self.manifest = payload.get("documents", {})
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not read vector store manifest: %s", exc)
            self.manifest = {}

        if not self._index_file.exists():
            return

        try:
            from langchain_community.vectorstores import FAISS

            self.store = FAISS.load_local(
                str(self.persist_dir),
                self.embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Loaded FAISS index with %d chunk(s) across %d document(s).",
                self.total_chunks,
                len(self.manifest),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load FAISS index: %s", exc)
            self.store = None
            raise VectorStoreError(
                "The saved vector index could not be loaded. "
                "It may have been created with a different embedding model. "
                "Use 'Clear vector database' and re-index your documents."
            ) from exc

    def save(self) -> None:
        """Persist the index and manifest to disk."""
        if self.store is not None:
            self.store.save_local(str(self.persist_dir), index_name=INDEX_NAME)
        self._manifest_path.write_text(
            json.dumps(
                {
                    "embedding_signature": self.embedding_signature,
                    "updated_at": utc_timestamp(),
                    "documents": self.manifest,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        self._signature_path.write_text(self.embedding_signature, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def _docstore_dict(self) -> dict[str, Document]:
        """Return the underlying FAISS docstore mapping id -> Document."""
        if self.store is None:
            return {}
        docstore = getattr(self.store, "docstore", None)
        data = getattr(docstore, "_dict", None)
        return data if isinstance(data, dict) else {}

    def _invalidate_bm25(self) -> None:
        self._bm25_documents = None

    @property
    def total_chunks(self) -> int:
        """Total number of chunks currently indexed."""
        if self.store is None:
            return 0
        return len(self._docstore_dict())

    @property
    def is_empty(self) -> bool:
        """Whether the vector store holds any chunks."""
        return self.total_chunks == 0

    def document_exists(self, document_id: str) -> bool:
        """Return ``True`` if a document id is already indexed."""
        return document_id in self.manifest

    def get_documents(self) -> list[dict[str, Any]]:
        """Return document-level metadata sorted by source name."""
        return sorted(
            (dict(meta, document_id=doc_id) for doc_id, meta in self.manifest.items()),
            key=lambda item: str(item.get("source", "")),
        )

    def all_chunks(self) -> list[Document]:
        """Return every indexed chunk (used for keyword search)."""
        if self._bm25_documents is None:
            self._bm25_documents = list(self._docstore_dict().values())
        return self._bm25_documents

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add_chunks(
        self,
        chunks: list[Document],
        document_metadata: dict[str, Any],
        *,
        force: bool = False,
    ) -> AddResult:
        """Add chunks for a document, preventing duplicates.

        Args:
            chunks: Chunked documents; each must carry ``chunk_id`` and
                ``document_id`` metadata.
            document_metadata: Document-level metadata to record.
            force: When ``True`` an already-indexed document is replaced.

        Returns:
            An :class:`AddResult` describing what happened.

        Raises:
            VectorStoreError: If the index cannot be created or updated.
        """
        if not chunks:
            return AddResult("", 0, True, "No chunks to index.")

        document_id = str(chunks[0].metadata.get("document_id", ""))
        if not document_id:
            raise VectorStoreError("Chunks are missing a document_id.")

        if self.document_exists(document_id):
            if not force:
                return AddResult(
                    document_id, 0, True, "Document already indexed (duplicate skipped)."
                )
            self.delete_document(document_id)

        from langchain_community.vectorstores import FAISS

        try:
            if self.store is None:
                self.store = FAISS.from_documents(chunks, self.embeddings)
                added = len(chunks)
            else:
                existing_ids = set(self._docstore_dict().keys())
                fresh = [
                    chunk
                    for chunk in chunks
                    if str(chunk.metadata.get("chunk_id")) not in existing_ids
                ]
                if fresh:
                    ids = [str(chunk.metadata.get("chunk_id")) for chunk in fresh]
                    self.store.add_documents(fresh, ids=ids)
                added = len(fresh)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to add chunks to FAISS index")
            raise VectorStoreError(f"Failed to index document: {exc}") from exc

        self.manifest[document_id] = {
            **document_metadata,
            "chunks": int(added + (self.manifest.get(document_id, {}).get("chunks", 0))),
            "indexed_at": utc_timestamp(),
        }
        self._invalidate_bm25()
        self.save()
        logger.info("Indexed %d chunk(s) for document %s", added, document_id)
        return AddResult(document_id, added, False, f"Indexed {added} chunk(s).")

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk belonging to a document.

        Args:
            document_id: Identifier of the document to remove.

        Returns:
            The number of chunks removed.
        """
        if self.store is None:
            self.manifest.pop(document_id, None)
            self.save()
            return 0

        ids_to_delete = [
            chunk_id
            for chunk_id, document in self._docstore_dict().items()
            if str(document.metadata.get("document_id")) == document_id
        ]

        removed = 0
        if ids_to_delete:
            try:
                self.store.delete(ids_to_delete)
                removed = len(ids_to_delete)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to delete chunks for %s: %s", document_id, exc)

        self.manifest.pop(document_id, None)
        self._invalidate_bm25()

        if self.total_chunks == 0:
            self._reset_store_files()
        else:
            self.save()
        return removed

    def clear(self) -> None:
        """Remove every chunk, the manifest and all persisted files."""
        self.store = None
        self.manifest = {}
        self._invalidate_bm25()
        self._reset_store_files()

    def _reset_store_files(self) -> None:
        """Delete persisted index files and reset in-memory state."""
        for path in (
            self._index_file,
            self.persist_dir / f"{INDEX_NAME}.pkl",
            self._manifest_path,
            self._signature_path,
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", path, exc)
        self.store = None
        self.manifest = {}

    # ------------------------------------------------------------------ #
    # Retrieval primitives
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        *,
        filter_document_ids: list[str] | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Run a vector similarity search, optionally filtered by document.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_document_ids: If provided, only chunks from these documents
                are considered.
            fetch_k: Number of candidates to fetch before filtering.

        Returns:
            A list of ``(Document, score)`` tuples ordered by relevance.
        """
        if self.store is None:
            return []

        candidate_k = fetch_k or max(k * 4, 20)
        results = self.store.similarity_search_with_score(query, k=candidate_k)

        if filter_document_ids:
            allowed = set(filter_document_ids)
            results = [
                (document, score)
                for document, score in results
                if str(document.metadata.get("document_id")) in allowed
            ]

        return results[:k]

    def as_retriever(self, k: int = 5, filter_document_ids: list[str] | None = None) -> Any:
        """Return a LangChain retriever bound to this vector store."""
        if self.store is None:
            return None

        search_kwargs: dict[str, Any] = {"k": k}
        if filter_document_ids and len(filter_document_ids) == 1:
            search_kwargs["filter"] = {"document_id": filter_document_ids[0]}
        return self.store.as_retriever(search_kwargs=search_kwargs)
