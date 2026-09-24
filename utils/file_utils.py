"""Safe file handling utilities: validation, hashing and persistence.

Uploaded files are never trusted: they are validated against an allow-list of
extensions and a maximum size before being written to a controlled directory
under ``data/uploads`` with a sanitised name.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from pathlib import Path

from config.settings import (
    FILE_TYPE_LABELS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    UPLOADS_DIR,
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""


class UnsupportedFileTypeError(FileValidationError):
    """Raised when a file extension is not supported."""


class FileTooLargeError(FileValidationError):
    """Raised when a file exceeds the maximum allowed size."""


def get_extension(filename: str) -> str:
    """Return the lower-cased file extension including the leading dot."""
    return Path(filename).suffix.lower()


def get_file_type_label(filename: str) -> str:
    """Return a human-readable label for a file's type (e.g. ``PDF``)."""
    return FILE_TYPE_LABELS.get(get_extension(filename), "UNKNOWN")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe version of ``filename``.

    Unicode characters are normalised and any character outside
    ``[A-Za-z0-9._-]`` is replaced with an underscore. Path traversal
    sequences are removed.
    """
    name = Path(filename).name
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = _SAFE_NAME_RE.sub("_", name).strip("._-")
    return name or "file"


def validate_file(filename: str, size_bytes: int) -> None:
    """Validate a file's extension and size.

    Args:
        filename: Original uploaded filename.
        size_bytes: Size of the uploaded payload in bytes.

    Raises:
        UnsupportedFileTypeError: If the extension is not supported.
        FileTooLargeError: If the file is larger than the configured limit.
        FileValidationError: If the file is empty.
    """
    extension = get_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or 'unknown'}'. Supported types: {supported}."
        )
    if size_bytes <= 0:
        raise FileValidationError("The uploaded file is empty.")
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File '{filename}' is {human_size(size_bytes)} which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit."
        )


def compute_file_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def make_document_id(file_hash: str, filename: str) -> str:
    """Build a deterministic document identifier.

    Using the file hash makes the id stable across re-uploads, which enables
    duplicate detection and incremental indexing.
    """
    seed = f"{file_hash}::{sanitize_filename(filename)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def human_size(num_bytes: int) -> str:
    """Return a human-readable file size string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"


def save_upload_bytes(data: bytes, filename: str, file_hash: str) -> Path:
    """Persist raw uploaded bytes to the uploads directory.

    The stored name is ``<hash prefix>_<sanitised name>`` which guarantees
    uniqueness while remaining human-readable.
    """
    safe_name = sanitize_filename(filename)
    destination = UPLOADS_DIR / f"{file_hash[:12]}_{safe_name}"
    destination.write_bytes(data)
    return destination
