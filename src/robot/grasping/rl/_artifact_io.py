"""Shared artifact IO: the single neutral source (no rl/ imports) for writing and
hashing serialised policy/report artifacts. ``hash_artifact`` is re-exported per
policy as ``hash_<family>_artifact``; ``write_artifact_json`` is the one renderer
every trainer emits through."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def hash_artifact(path: str | Path) -> str:
    """Return the sha256 hex digest of a serialised policy artifact."""

    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifact_json(artifact: Mapping[str, Any], path: str | Path) -> str:
    """Write ``artifact`` as canonical JSON and return the sha256 hex of the bytes.

    Canonical means sorted keys, two-space indent, a single trailing newline, and LF on
    every platform. The write goes through ``write_bytes``: ``write_text`` translates to
    CRLF on Windows, which desyncs the returned digest from the file on disk.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(dict(artifact), sort_keys=True, indent=2) + "\n").encode("utf-8")
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
