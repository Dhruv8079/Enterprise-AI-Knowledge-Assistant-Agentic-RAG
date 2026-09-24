"""Persistence helpers for dataframes and conversations.

Tabular data extracted from CSV/XLSX files is cached to ``data/processed`` so
that deterministic data analysis keeps working after an application restart.
Conversations are stored as JSON under ``data/conversations``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import CONVERSATIONS_DIR, PROCESSED_DIR
from utils.logger import get_logger
from utils.metadata import utc_timestamp

logger = get_logger(__name__)


def save_dataframes(document_id: str, dataframes: dict[str, pd.DataFrame]) -> None:
    """Persist a document's dataframes to disk.

    Args:
        document_id: Identifier of the owning document.
        dataframes: Mapping of sheet name to dataframe.
    """
    if not dataframes:
        return
    path = PROCESSED_DIR / f"{document_id}.pkl"
    try:
        with path.open("wb") as handle:
            pickle.dump(dataframes, handle)
    except (OSError, pickle.PickleError) as exc:
        logger.warning("Could not persist dataframes for %s: %s", document_id, exc)


def load_all_dataframes() -> dict[str, dict[str, pd.DataFrame]]:
    """Load every persisted dataframe collection.

    Returns:
        A mapping of document id to ``{sheet: dataframe}``.
    """
    registry: dict[str, dict[str, pd.DataFrame]] = {}
    for path in PROCESSED_DIR.glob("*.pkl"):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                registry[path.stem] = payload
        except Exception as exc:  # noqa: BLE001 - corrupted cache should not crash
            logger.warning("Skipping unreadable dataframe cache %s: %s", path.name, exc)
    return registry


def delete_dataframes(document_id: str) -> None:
    """Remove a document's persisted dataframes."""
    path = PROCESSED_DIR / f"{document_id}.pkl"
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete dataframe cache %s: %s", path.name, exc)


def save_conversation(
    conversation_id: str,
    messages: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> Path:
    """Persist a conversation to disk as JSON.

    Args:
        conversation_id: Unique conversation identifier.
        messages: List of chat message dictionaries.
        title: Optional conversation title.

    Returns:
        The path the conversation was written to.
    """
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    payload = {
        "conversation_id": conversation_id,
        "title": title or _derive_title(messages),
        "updated_at": utc_timestamp(),
        "messages": messages,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _derive_title(messages: list[dict[str, Any]]) -> str:
    """Derive a short title from the first user message."""
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            text = str(message["content"]).strip().replace("\n", " ")
            return text[:60]
    return "Untitled conversation"


def conversation_to_markdown(messages: list[dict[str, Any]]) -> str:
    """Render a conversation as Markdown for download.

    Args:
        messages: List of chat message dictionaries.

    Returns:
        A Markdown transcript including citations when present.
    """
    lines: list[str] = ["# Enterprise AI Knowledge Assistant — Conversation", ""]
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        lines.append(f"## {role}")
        lines.append(str(message.get("content", "")))
        sources = message.get("sources") or []
        if sources:
            lines.append("")
            lines.append("**Sources:**")
            for source in sources:
                label = source.get("label") or (
                    f"{source.get('source', '?')} — {source.get('page', '?')}"
                )
                lines.append(f"- {label} (chunk {source.get('chunk_id', '?')})")
        lines.append("")
    return "\n".join(lines)
