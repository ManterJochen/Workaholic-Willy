"""Whether a robot description describes a body: it carries geometry, and its mesh references resolve.

It lives beside ``build_ur_config.py`` because that script runs in the cuRobo environment, where this repository
cannot be imported, and it is loaded by path so everything that asks reaches one implementation.

A reference resolves beside the description, one folder up, or under an asset root the caller names. Isaac writes a
UR mesh as ``package://ur_description/meshes/<model>/...``; the build rewrites that to ``meshes/<model>/...`` under
the cuRobo asset root, and that root is where the descriptor finds the file. Resolving the package reference only
beside Isaac's description refuses every UR rebuild, with all 14 ur5e meshes present under the cuRobo root.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["PACKAGE_PREFIX", "describes_a_body"]

#: The package prefix Isaac's UR descriptions put in front of every mesh.
PACKAGE_PREFIX = "package://ur_description/"


def describes_a_body(path: Path, *, asset_roots: tuple[Path, ...] = ()) -> tuple[bool, str]:
    """``(usable, why)`` for the description at ``path``.

    Usable needs geometry, and every mesh reference resolving: beside the description, one folder up, or under one
    of ``asset_roots``. A description with no geometry describes no body, and one with a mesh missing describes part
    of one: ur16e resolves 10 of its 14 meshes, and a build that accepts that writes a descriptor from half a body.
    """
    text = path.read_text(encoding="utf-8")
    refs = re.findall(r'filename="([^"]+)"', text)
    if not refs and "<collision>" not in text:
        return False, "no geometry at all: 0 mesh references and no <collision>"
    missing = [ref for ref in refs if not _resolves(path, ref, asset_roots)]
    if refs and len(missing) == len(refs):
        return False, f"all {len(refs)} mesh references are missing from disk"
    if missing:
        return False, (f"{len(missing)} of {len(refs)} mesh references are missing from disk, the first "
                       f"{missing[0]}; half a body is not a body")
    return True, f"{len(refs)} mesh refs ({len(refs)} resolvable), {text.count('<collision>')} <collision>"


def _resolves(path: Path, ref: str, asset_roots: tuple[Path, ...]) -> bool:
    relative = ref[len(PACKAGE_PREFIX):] if ref.startswith(PACKAGE_PREFIX) else ref
    stripped = relative.lstrip("./")
    candidates = [path.parent / relative, path.parent.parent / stripped]
    candidates += [root / stripped for root in asset_roots]
    return any(candidate.is_file() for candidate in candidates)
