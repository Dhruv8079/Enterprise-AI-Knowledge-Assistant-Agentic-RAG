"""SQLite-backed analytics and application history.

Stores document-level metadata and per-query telemetry so the Analytics page
can present usage statistics that survive application restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from functools import lru_cache
from typing import Any

from config.settings import DB_PATH
from utils.logger import get_logger
from utils.metadata import utc_timestamp

logger = get_logger(__name__)

_DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    num_units   INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'indexed',
    indexed_at  TEXT NOT NULL
);
"""

_QUERIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    question       TEXT NOT NULL,
    intent         TEXT NOT NULL,
    latency        REAL NOT NULL DEFAULT 0,
    retrieved      INTEGER NOT NULL DEFAULT 0,
    used_fallback  INTEGER NOT NULL DEFAULT 0,
    sources        TEXT NOT NULL DEFAULT '[]'
);
"""


class AnalyticsTracker:
    """Thread-safe SQLite wrapper for documents and query telemetry."""

    def __init__(self, db_path: Any = DB_PATH) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not yet exist."""
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(_DOCUMENTS_SCHEMA)
            cursor.execute(_QUERIES_SCHEMA)
            self._connection.commit()

    # ------------------------------------------------------------------ #
    # Document records
    # ------------------------------------------------------------------ #
    def upsert_document(self, record: dict[str, Any]) -> None:
        """Insert or update a document record.

        Args:
            record: Document metadata including ``document_id`` and ``source``.
        """
        payload = {
            "document_id": str(record.get("document_id", "")),
            "source": str(record.get("source", "unknown")),
            "file_type": str(record.get("file_type", "?")),
            "size_bytes": int(record.get("size_bytes", 0) or 0),
            "num_units": int(record.get("num_units", 0) or 0),
            "chunks": int(record.get("chunks", 0) or 0),
            "status": str(record.get("status", "indexed")),
            "indexed_at": str(record.get("indexed_at", utc_timestamp())),
        }
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO documents
                    (document_id, source, file_type, size_bytes, num_units, chunks, status, indexed_at)
                VALUES
                    (:document_id, :source, :file_type, :size_bytes, :num_units, :chunks, :status, :indexed_at)
                ON CONFLICT(document_id) DO UPDATE SET
                    source=excluded.source,
                    file_type=excluded.file_type,
                    size_bytes=excluded.size_bytes,
                    num_units=excluded.num_units,
                    chunks=excluded.chunks,
                    status=excluded.status,
                    indexed_at=excluded.indexed_at
                """,
                payload,
            )
            self._connection.commit()

    def update_document_status(self, document_id: str, status: str, chunks: int | None = None) -> None:
        """Update the status (and optionally chunk count) of a document."""
        with self._lock:
            if chunks is None:
                self._connection.execute(
                    "UPDATE documents SET status = ? WHERE document_id = ?",
                    (status, document_id),
                )
            else:
                self._connection.execute(
                    "UPDATE documents SET status = ?, chunks = ? WHERE document_id = ?",
                    (status, int(chunks), document_id),
                )
            self._connection.commit()

    def delete_document(self, document_id: str) -> None:
        """Delete a document record."""
        with self._lock:
            self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            self._connection.commit()

    def clear_documents(self) -> None:
        """Delete every document record."""
        with self._lock:
            self._connection.execute("DELETE FROM documents")
            self._connection.commit()

    # ------------------------------------------------------------------ #
    # Query telemetry
    # ------------------------------------------------------------------ #
    def log_query(
        self,
        *,
        question: str,
        intent: str,
        latency: float,
        retrieved: int,
        used_fallback: bool,
        sources: list[str] | None = None,
    ) -> None:
        """Record a single query and its outcome."""
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO queries
                    (timestamp, question, intent, latency, retrieved, used_fallback, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(),
                    question,
                    intent,
                    float(latency),
                    int(retrieved),
                    1 if used_fallback else 0,
                    json.dumps(sources or []),
                ),
            )
            self._connection.commit()

    def clear_queries(self) -> None:
        """Delete all query telemetry."""
        with self._lock:
            self._connection.execute("DELETE FROM queries")
            self._connection.commit()

    # ------------------------------------------------------------------ #
    # Aggregations
    # ------------------------------------------------------------------ #
    def get_documents(self) -> list[dict[str, Any]]:
        """Return all stored document records."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM documents ORDER BY source"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_recent_queries(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return the most recent queries."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate analytics for the dashboard."""
        with self._lock:
            document_stats = self._connection.execute(
                "SELECT COUNT(*) AS count, COALESCE(SUM(chunks), 0) AS chunks FROM documents"
            ).fetchone()
            query_stats = self._connection.execute(
                """
                SELECT COUNT(*) AS count,
                       COALESCE(AVG(latency), 0) AS avg_latency,
                       COALESCE(AVG(retrieved), 0) AS avg_retrieved,
                       COALESCE(SUM(used_fallback), 0) AS failures
                FROM queries
                """
            ).fetchone()
            intent_rows = self._connection.execute(
                "SELECT intent, COUNT(*) AS count FROM queries GROUP BY intent ORDER BY count DESC"
            ).fetchall()
            source_rows = self._connection.execute(
                "SELECT sources FROM queries WHERE sources != '[]'"
            ).fetchall()
            status_rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
            ).fetchall()

        usage: dict[str, int] = {}
        for row in source_rows:
            try:
                for source in json.loads(row["sources"]):
                    usage[str(source)] = usage.get(str(source), 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        return {
            "total_documents": int(document_stats["count"]),
            "total_chunks": int(document_stats["chunks"]),
            "total_queries": int(query_stats["count"]),
            "avg_latency": float(query_stats["avg_latency"]),
            "avg_retrieved": float(query_stats["avg_retrieved"]),
            "failed_retrievals": int(query_stats["failures"]),
            "query_types": {row["intent"]: int(row["count"]) for row in intent_rows},
            "document_usage": usage,
            "document_status": {row["status"]: int(row["count"]) for row in status_rows},
        }

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self._connection.close()
        except sqlite3.Error:
            pass


@lru_cache(maxsize=1)
def get_tracker() -> AnalyticsTracker:
    """Return the process-wide analytics tracker singleton."""
    return AnalyticsTracker()
