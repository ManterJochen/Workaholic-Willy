"""Cross-platform JSON and atomic-write helpers.

Text is written as UTF-8 with LF line endings unless a caller overrides the
encoding, so a file written on Windows is byte-identical to one written on
Linux and the locale never supplies ``cp1252``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_text", "dump_json", "load_json"]


def load_json(path: os.PathLike[str] | str) -> Any:
    """Load JSON from ``path`` using UTF-8."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(
    data: Any,
    path: os.PathLike[str] | str,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> None:
    """Write ``data`` as JSON to ``path`` using UTF-8.

    With ``atomic=True``, the default, the write goes through
    ``atomic_write_text``, so a reader never sees a half-written document.
    """
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    if atomic:
        atomic_write_text(path, text)
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def atomic_write_text(
    path: os.PathLike[str] | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write ``text`` to ``path`` atomically, through a temp file and a rename.

    ``os.replace`` is atomic on Windows and Linux when the source and the
    destination sit on one filesystem, which they do here: the temp file is
    created in the parent directory of the destination.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
        os.replace(tmp, target)
    except Exception:
        # Clean up the temp file on failure without masking the original error.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


