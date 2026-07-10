"""Filesystem/IO helpers shared across packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_model.utils.exceptions import VisionModelError


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if it doesn't exist, and return it as a Path.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The resolved `Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_file(path: str | Path) -> Path:
    """Return `path` as a `Path`, raising if it does not exist or isn't a file.

    Args:
        path: File path expected to exist.

    Raises:
        VisionModelError: If the path does not exist or is not a file.
    """
    p = Path(path)
    if not p.is_file():
        raise VisionModelError(f"Required file not found: {p}")
    return p


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to a JSON file.

    Raises:
        VisionModelError: If the file is missing or contains invalid JSON.
    """
    p = require_file(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise VisionModelError(f"Invalid JSON in {p}: {exc}") from exc


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Write `data` to `path` as JSON, creating parent directories as needed.

    Args:
        path: Destination file path.
        data: JSON-serializable object.
        indent: Pretty-print indent width.

    Returns:
        The resolved `Path` written to.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return p
