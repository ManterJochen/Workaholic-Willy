"""Finding the scenes a run trains on, and drawing a fair sample of them.

Both functions encode corpus facts that a caller gets wrong by default, so a library caller reads
its own corpus correctly by calling them rather than by re-deriving the rules.

Neither function needs torch, so this module stays cheap to import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

__all__ = ["scene_files", "stratified_scenes"]


def scene_files(directory: str | Path) -> list[Path]:
    """Every scene `.npz` under ``directory``, recursively, sorted so a run is reproducible.

    Recursive because a scene id is `<family>_<index>` whose index counts within one dataset: two
    shards of the same corpus hold different scenes under identical names, and flattening the shards
    into one directory overwrites every scene whose name collides. Each shard therefore keeps its
    own subdirectory, and this walks them all.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"no such corpus directory: {directory}")
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"{directory} contains no .npz scene; extract one with "
                                f"`python -m datagen build-cloud-corpus`")
    return files


def stratified_scenes(files: Sequence[Path], count: int, *, seed: int = 0,
                      families: str | None = None) -> list[Path]:
    """`count` scenes drawn evenly across families, not the first `count` of a sorted list.

    A prefix is not a sample. A scene id is `<family>_<index>`, so `sorted()` groups by family and
    any prefix is one family: a held-out set drawn that way measures one family rather than the
    corpus, and "unseen scenes" taken from the front of the list are all of the same kind.

    Round robin, then a seeded shuffle within each family. Round robin alone takes the first scenes
    of every family, which is a prefix one level down; shuffling alone gives a family with more
    scenes proportionally more slots, which samples the corpus rather than the families and is the
    right answer to a different question.
    """
    import numpy as np  # noqa: PLC0415

    wanted = None if families is None else {f.strip() for f in families.split(",") if f.strip()}
    grouped: dict[str, list[Path]] = {}
    for path in files:
        family = path.stem.rsplit("_", 1)[0]
        if wanted is not None and family not in wanted:
            continue
        grouped.setdefault(family, []).append(path)
    if not grouped:
        raise FileNotFoundError(
            f"no scene matches families {sorted(wanted or ())}; the corpus holds "
            f"{sorted({f.stem.rsplit('_', 1)[0] for f in files})}")
    rng = np.random.default_rng(seed)
    order = {name: [group[i] for i in rng.permutation(len(group))]
             for name, group in grouped.items()}
    out: list[Path] = []
    for index in range(max(len(v) for v in order.values())):
        for name in sorted(order):
            if index < len(order[name]) and len(out) < count:
                out.append(order[name][index])
        if len(out) >= count:
            break
    return out
