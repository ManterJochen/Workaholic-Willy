"""The row a hand writer records in ``bundles.json`` beside the bundle it wrote.

``bundles.json`` says where every committed collision bundle came from and hashes the arrays the guard loads, and a
committed bundle with no row is refused by the repository's own checks. A customer who writes a hand body must not
have to hand-edit that index, so every hand writer records its own row through this one function: the dimensions
writer, the mesh writer and the USD bake.

``arrays_sha256`` hashes the arrays in name order, never the file: an ``.npz`` is a zip with a timestamp, and the same
arrays written twice give two files. The repository's checks keep their own copy of that hash, so the writer's row is
checked by code that did not write it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["BundleRow", "BundleRowRefused", "INDEX_NAME", "arrays_sha256", "index_bytes", "record_hand_bundle"]

#: The index beside the bundles.
INDEX_NAME = "bundles.json"
_HAND_SUFFIX = "_hand_meshes.npz"


class BundleRowRefused(ValueError):
    """A row that may not be recorded, with the sentence that says why."""


def arrays_sha256(path: str | Path) -> str:
    """One hash over a bundle's arrays, in name order: the bytes the guard loads."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=True) as bundle:
        for name in sorted(bundle.files):
            array = np.asarray(bundle[name])
            digest.update(name.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def index_bytes(index: dict[str, Any], *, crlf: bool) -> bytes:
    """``index`` in the layout the committed file uses: sorted keys, one space of indent, a final newline."""
    text = json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    return (text.replace("\n", "\r\n") if crlf else text).encode("utf-8")


@dataclass(frozen=True, slots=True)
class BundleRow:
    """What one recording put into the index."""

    bundle: str
    hand: str
    source: str
    note: str
    arrays_sha256: str
    #: Whether a row for this bundle was there before and was replaced, as a re-bake does.
    replaced: bool

    def render(self) -> str:
        verb = "replaced" if self.replaced else "recorded"
        return f"{verb} {self.bundle} in {INDEX_NAME}: source {self.source}, arrays {self.arrays_sha256[:12]}, {self.note}"

    def to_dict(self) -> dict[str, Any]:
        return {"bundle": self.bundle, "hand": self.hand, "source": self.source, "note": self.note,
                "arrays_sha256": self.arrays_sha256, "replaced": self.replaced}


def record_hand_bundle(bundle: str | Path, *, source: str, note: str) -> BundleRow:
    """Record ``bundle`` in the ``bundles.json`` beside it, or refuse and say why."""
    path = Path(bundle)
    index_path = path.parent / INDEX_NAME
    if not path.name.endswith(_HAND_SUFFIX):
        raise BundleRowRefused(f"{path.name} is not a hand bundle; a hand bundle is named <hand>{_HAND_SUFFIX}")
    if not path.is_file():
        raise BundleRowRefused(f"{path} does not exist, so there is nothing to record")
    if not note.strip():
        raise BundleRowRefused("a source with no note is a word, not a provenance: say what the body was built from")
    if not index_path.is_file():
        raise BundleRowRefused(f"{path.parent} holds no {INDEX_NAME}, so there is no index to record {path.name} in")
    raw = index_path.read_bytes()
    index = json.loads(raw.decode("utf-8"))
    declared = index.get("sources") or {}
    if source not in declared:
        raise BundleRowRefused(
            f"source {source!r} is not one {INDEX_NAME} declares; it declares {', '.join(sorted(declared))}"
        )
    bundles = index.setdefault("bundles", {})
    replaced = path.name in bundles
    digest = arrays_sha256(path)
    hand = path.name[: -len(_HAND_SUFFIX)]
    bundles[path.name] = {"arrays_sha256": digest, "hand": hand, "note": note.strip(), "source": source}
    index_path.write_bytes(index_bytes(index, crlf=b"\r\n" in raw))
    return BundleRow(bundle=path.name, hand=hand, source=source, note=note.strip(), arrays_sha256=digest,
                     replaced=replaced)
