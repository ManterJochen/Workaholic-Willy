"""What the repository's registries share: how a file is found, and how a description is compared and shown.

Two registries hold one YAML file per model: ``grippers/`` (:mod:`src.config.grippers`) and ``cameras/``
(:mod:`src.config.cameras`). Both find their files the same way on every platform, both compare a deployment
tree's copy with the repository's key by key, and both name a registry file to a person the same way, so those three
live here once and each registry keeps its own sentences.
"""

from __future__ import annotations

import re
from pathlib import Path

from .loader import _DEFAULT_DATA_DIR, ConfigError
from .schema.grippers import MODEL_NAME_PATTERN

__all__ = ["flat", "registry_files", "shown"]


def registry_files(directory: Path, *, noun: str) -> list[Path]:
    """The files in ``directory`` that describe a ``noun``, found the same way on every platform.

    A glob for ``*.yaml`` ignores ``.yml`` everywhere, matches ``.YAML`` only where the filesystem is
    case-insensitive, and reads ``robotiq_2f85.sim.yaml`` as a file named after a stem no model can
    match. So every file that looks like YAML must be named ``<model>.yaml`` exactly, and any other one
    is refused by name. A file that is not YAML at all, a README for instance, is not a description.
    """
    found: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml"):
            continue
        if path.suffix != ".yaml" or not re.fullmatch(MODEL_NAME_PATTERN, path.stem):
            raise ConfigError(
                f"{path.name} in {directory} is not a {noun} the registry can read: a {noun} is one file "
                f"named <model>.yaml, with the suffix in lower case, a registry name "
                f"({MODEL_NAME_PATTERN}) as its stem, and no profile layer"
            )
        found.append(path)
    return found


def flat(value: object, prefix: str = "") -> dict[str, object]:
    """A nested description as dotted keys, so two descriptions compare key by key."""
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(flat(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def shown(path: Path) -> str:
    """A registry path for a person: relative to the repository where it lies inside it, with forward slashes.

    The default tree is ``config/`` at the repository root, so the repository is its parent.
    """
    repository = _DEFAULT_DATA_DIR.parent
    return path.relative_to(repository).as_posix() if path.is_relative_to(repository) else str(path)
