"""
Small helper functions for working with JSON files and handling of errors. 
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from numbers import Integral

class FormatError(ValueError):
    """An error dealing with mathematical format."""

def require(condition: bool, message: str) -> None:
    """Raise an error for debugging"""
    if not condition:
        raise FormatError(message)

def check_integer(value, name: str) -> int:
    """Accept integer types without silently truncating noninteger inputs."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


# used to be called digest
def sha256(data: Any) -> str:
    """SHA-256 of dict object, independent of indentation or key order."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def read_json(path: str | Path) -> dict:
    """Read a JSON object; duplicate object keys are invalid."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=object_pairs)
    require(isinstance(result, dict), "Expected a JSON object.")
    return result

def write_json(path: str | Path, data: dict) -> Path:
    """Atomically replace a JSON file after flushing its complete contents.

    This is used for saving data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path