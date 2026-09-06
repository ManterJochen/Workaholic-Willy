"""Splitting one rendered dataset into disjoint views, so the cloud extraction can use every core.

`build_cloud_corpus` walks one dataset and every scene in it. Its `scenes` parameter looks like the
obvious way to divide the work and is not: it is an even stride over the scene list, so two calls
asking for half each get the same half rather than complementary ones. The only way to spread one
rendered shard across processes is to hand each process its own dataset whose scene list is a
disjoint slice, which is what this module builds.

The scene directories are linked, not copied. A part view holds the depth and instance PNGs the
extractor reads, so copying them would be I/O for nothing and would double the corpus on disk for
the length of the run. On Windows the link is a junction (`mklink /J`), which needs no privilege
where `os.symlink` does.

Every part needs its own corpus output directory. Per-scene filenames do not make one shared
directory safe: scene ids repeat across shards, so parts pointed at one directory refuse each other
with

    packed_000001.npz already holds a scene from dataset 'v5_s2' and this run is 'v5_s2_p0'

and a directory that did not refuse would overwrite silently. The trainer walks the corpus tree with
`rglob`, so one directory per part costs nothing and keeps that guard meaningful.

Splitting a running extraction throws away what it has built. `build_cloud_corpus` does not skip a
scene it already wrote, so killing a builder to parallelise it costs its whole run so far. Check
what a process has actually produced before stopping it.

A part view is a dataset as far as every other command is concerned. It sits under the same output
root, it has a `scenes/` directory and the sidecar files, and anything that enumerates datasets will
count its scenes a second time. Each view is stamped with `SPLIT_VIEW.json` naming its source so it
can be told apart, and `clean_split` removes them once the extraction is done.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.utility.log_cfg import create_logger

from datagen.constants import CORPUS_GATE_LOG_FILE, DATAGEN_LOG_DIR

logger = create_logger("CorpusSplit", CORPUS_GATE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Written into every part view. Named rather than hidden: a directory that looks like a dataset and
#: is not needs to say so somewhere a person browsing the folder will see it.
MARKER = "SPLIT_VIEW.json"

#: Copied into each part. These are per-dataset files the extractor reads beside the scenes. The
#: grasp table matters most: it is joined by `(file, row_index)`, so every part must see the whole
#: of it, because a part's scenes carry row indices into the shared file.
_SIDECARS = ("grasps.jsonl", "index.jsonl", "provenance.json", "ATTRIBUTION",
             "grasp_label_report.json")


def _scene_names(dataset: Path) -> list[str]:
    """Every scene the dataset actually finished, sorted.

    `scene.json` is the completion marker: `datagen build` writes the directory first and the
    manifest last, so a directory without one is a scene that was interrupted mid-render. Counting
    directories instead would hand a part a scene the extractor then refuses.
    """
    scenes = dataset / "scenes"
    if not scenes.is_dir():
        raise FileNotFoundError(f"{dataset} has no scenes/ directory")
    return sorted(p.name for p in scenes.iterdir() if (p / "scene.json").is_file())


def _link_scenes(pairs: list[tuple[Path, Path]]) -> None:
    """Point each `link` at its `target` directory, without copying the contents.

    On Windows this is one batch of `mklink /J` rather than one subprocess per link: junctions are
    cheap but process creation is not, and a part view holds one link per scene.
    """
    if os.name != "nt":
        for link, target in pairs:
            os.symlink(target, link, target_is_directory=True)
        return
    lines = "\r\n".join(f'mklink /J "{link}" "{target}" >nul' for link, target in pairs)
    with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="ascii") as handle:
        handle.write(lines + "\r\n")
        script = Path(handle.name)
    try:
        subprocess.run(["cmd", "/c", str(script)], check=True, capture_output=True)
    finally:
        script.unlink(missing_ok=True)


def split_dataset(dataset: Path, parts: int, *, out_root: Path | None = None) -> dict[str, Any]:
    """Build `parts` disjoint views of `dataset`, and prove they are disjoint and complete.

    Both properties are checked rather than assumed, and against the source scene list rather than
    against each other: a mis-split corpus produces a training run on the wrong data, and that is
    discovered hours later in a trainer if it is discovered at all.
    """
    if parts < 2:
        raise ValueError(f"splitting into {parts} part(s) is not a split")
    names = _scene_names(dataset)
    if len(names) < parts:
        raise ValueError(f"{dataset} has {len(names)} finished scene(s), fewer than the {parts} "
                         f"part(s) asked for")
    root = out_root if out_root is not None else dataset.parent

    seen: set[str] = set()
    views: list[dict[str, Any]] = []
    for part in range(parts):
        # A stride, deliberately, not a contiguous block: scene ids sort by family, so contiguous
        # slices would hand one part every `bin` scene and another every `sparse` one. The families
        # differ in object count and therefore in extraction cost, and a parallel run is only as
        # fast as its slowest part.
        mine = names[part::parts]
        if seen & set(mine):
            raise ValueError(f"part {part} overlaps an earlier one")
        seen |= set(mine)

        view = root / f"{dataset.name}_p{part}"
        if view.exists():
            _remove_view(view, dataset)
        (view / "scenes").mkdir(parents=True)
        for sidecar in _SIDECARS:
            if (dataset / sidecar).is_file():
                shutil.copy(dataset / sidecar, view / sidecar)
        _link_scenes([((view / "scenes" / name).resolve(), (dataset / "scenes" / name).resolve())
                      for name in mine])
        linked = len(_scene_names(view))
        if linked != len(mine):
            raise ValueError(f"{view}: linked {linked} of {len(mine)} scene(s)")
        (view / MARKER).write_text(
            json.dumps({"source": str(dataset), "part": part, "parts": parts, "scenes": linked},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        views.append({"path": str(view), "scenes": linked})
        logger.info("split %s: part %d of %d -> %s (%d scene(s))",
                    dataset.name, part + 1, parts, view.name, linked)

    if seen != set(names):
        raise ValueError(f"the parts cover {len(seen)} of {len(names)} scene(s)")
    return {"dataset": str(dataset), "scenes": len(names), "parts": parts, "views": views}


def _remove_view(view: Path, dataset: Path) -> None:
    """Delete one part view, refusing anything that is not one.

    The marker is the whole safety of this. `_p0` is a plausible suffix for a hand-made dataset, and
    a cleanup that trusted the name could delete a real render.

    Scene entries are removed with `rmdir`, which unlinks a junction and leaves its target alone but
    refuses a real non-empty directory. That refusal is the point: if a view holds real scene data,
    something built it wrong and the right move is to stop, not to recurse.
    """
    marker = view / MARKER
    if not marker.is_file():
        raise ValueError(f"{view} has no {MARKER}, so it is not a split view of {dataset.name}")
    if json.loads(marker.read_text(encoding="utf-8")).get("source") != str(dataset):
        raise ValueError(f"{view} is a split view of a DIFFERENT dataset")
    scenes = view / "scenes"
    if scenes.is_dir():
        for scene in scenes.iterdir():
            scene.rmdir()
    shutil.rmtree(view)


def clean_split(dataset: Path, *, out_root: Path | None = None) -> list[str]:
    """Remove every part view of `dataset`. Skips, with a warning, anything that is not one."""
    root = out_root if out_root is not None else dataset.parent
    removed: list[str] = []
    for candidate in sorted(root.glob(f"{dataset.name}_p*")):
        try:
            _remove_view(candidate, dataset)
        except (ValueError, OSError) as exc:
            logger.warning("not removing %s: %s", candidate, exc)
            continue
        removed.append(str(candidate))
    return removed
