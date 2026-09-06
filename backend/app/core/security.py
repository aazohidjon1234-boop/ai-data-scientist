"""Upload validation and safe file handling.

- Only .csv files are accepted (extension + actual content parsing).
- Size is bounded by MAX_UPLOAD_MB.
- Uploaded bytes are stored under a random UUID directory; the original
  filename is only kept as metadata and is never used for path building.
- No code from uploaded content is ever executed (pandas.read_csv only).
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import UploadFile

from ..core.config import get_settings
from ..exceptions import ValidationError

ALLOWED_EXTENSIONS = {".csv"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    name = (name or "dataset.csv").strip()
    name = _UNSAFE_CHARS.sub("_", name)
    name = name.lstrip("._")
    return (name[:120] or "dataset.csv")


def model_slug(name: str) -> str:
    """Filesystem-safe slug for a model name (e.g. 'K-Means (k=3)' -> 'K-Means_k=3_')."""
    slug = _UNSAFE_CHARS.sub("_", (name or "model").strip())
    return slug.lstrip("._")[:80] or "model"


def safe_path(base: Path, *parts: str) -> Path:
    """Join path parts under `base`, guaranteeing no path traversal."""
    resolved = Path(base).resolve()
    target = (resolved.joinpath(*parts)).resolve()
    if not str(target).startswith(str(resolved)):
        raise ValidationError("Invalid file path requested.")
    return target


async def read_validated_csv(upload: UploadFile) -> tuple[bytes, str]:
    """Validate an uploaded file and return (raw_bytes, safe_filename)."""
    settings = get_settings()

    original = upload.filename or ""
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a .csv file."
        )

    content = await upload.read()
    if len(content) == 0:
        raise ValidationError("The uploaded file is empty.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise ValidationError(
            f"File is too large ({len(content) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {settings.max_upload_mb} MB."
        )

    # A CSV must at least look like a header row of text.
    try:
        head = content[:8192].decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError("File does not appear to be a text CSV file.")
    if not head.strip():
        raise ValidationError("The uploaded file has no content.")

    return content, sanitize_filename(original)
