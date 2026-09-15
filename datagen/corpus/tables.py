"""Another hand's grasp table, beside the clouds a corpus already has.

One sidecar per scene per hand. A cloud carries the grasp table its extraction joined, the 2F-85's
for every corpus built so far. Labelling the same scenes for another hand (`label-grasps --jaw`)
would otherwise mean extracting every cloud again under a second identity, when the table is the
only part that depends on the hand. So this writes that part and leaves the cloud alone.

`<scene>.<model>.grasps` beside `<scene>.npz`: an npz container under its own suffix, written
through a file handle so numpy does not append `.npz`, which keeps every `rglob("*.npz")` scene
walk from reading a table as a scene. It needs the dataset's `scene.json` files and its mesh bank,
no images: the contacts, the approach corridor and the physics join come from
`clouds.scene_grasp_table` with the hand's own jaw and the label file's own join key.

A table is written for every cloud extracted from the dataset itself or from one of its split
views (`<dataset>_p<n>`, which is how `split-dataset` extracts); clouds of other datasets under the
same directory are counted and get none. The cloud's point count is stamped, so a reader can refuse
a table beside a cloud that was extracted again since.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import CORPUS_GATE_LOG_FILE, DATAGEN_LOG_DIR

__all__ = ["TABLE_SUFFIX", "TABLE_VERSION", "build_grasp_tables", "table_path"]

logger = create_logger("GraspTables", CORPUS_GATE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: The table's own suffix. Not `.npz`, so a scene walk never mistakes a table for a scene.
TABLE_SUFFIX: Final[str] = ".grasps"
#: What shape a table is. Bumped whenever a key is added, renamed or given a new meaning.
TABLE_VERSION: Final[int] = 1


def table_path(cloud: str | Path, gripper: str) -> Path:
    """Where ``gripper``'s table for ``cloud`` lives: beside it, as ``<scene>.<gripper>.grasps``."""
    cloud = Path(cloud)
    return cloud.with_name(f"{cloud.stem}.{gripper}{TABLE_SUFFIX}")


def _of_this_dataset(identity: str, dataset: str) -> bool:
    """A cloud extracted from ``dataset`` itself, or from a split view ``<dataset>_p<n>`` of it."""
    return identity == dataset or re.fullmatch(rf"{re.escape(dataset)}_p\d+", identity) is not None


def build_grasp_tables(root: str | Path, clouds: str | Path, *, labels: str,
                       physics: str | Path | None = None) -> dict[str, Any]:
    """Write ``labels``' hand's grasp table beside every cloud of ``root`` under ``clouds``.

    Returns a report. Refuses the corpus file (every cloud already carries its table), an unknown
    jaw or a report stamping another one (both before anything is written), a directory with no
    cloud of this dataset, and a cloud whose scene the dataset does not have.
    """
    from datagen.corpus.clouds import (  # noqa: PLC0415
        _gripper_of, physics_verdicts, scene_grasp_table,
    )
    from datagen.grasps.labels import (  # noqa: PLC0415 (heavy import chain)
        jaw_model_for, scene_assets,
    )

    root, clouds = Path(root), Path(clouds)
    if labels == "grasps.jsonl":
        raise ValueError("grasps.jsonl is the table every cloud of this dataset already carries; a "
                         "grasp table is for another hand's labels, grasps_jaw_<model>.jsonl")
    gripper = _gripper_of(root, labels)
    model = jaw_model_for(gripper)
    label_path = root / labels
    if not label_path.is_file():
        raise FileNotFoundError(
            f"no {labels} under {root}; label the scenes with `label-grasps --jaw {gripper}`")
    if not clouds.is_dir():
        raise FileNotFoundError(f"no such cloud directory: {clouds}")

    mine: list[tuple[Path, str, int]] = []
    foreign = 0
    for cloud in sorted(clouds.rglob("*.npz")):
        with np.load(cloud, allow_pickle=False) as handle:
            identity = (str(np.asarray(handle["source_dataset"]).reshape(-1)[0])
                        if "source_dataset" in handle.files else "")
            points = int(len(handle["points_mm"]))
        if _of_this_dataset(identity, root.name):
            mine.append((cloud, identity, points))
        else:
            foreign += 1
    if not mine:
        raise FileNotFoundError(
            f"no cloud under {clouds} was extracted from {root.name} ({foreign} belong to other "
            f"datasets); extract them with `build-cloud-corpus --name {root.name}` first")

    by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line:
            continue
        row = json.loads(line)
        row["_row_index"] = index
        by_scene[str(row["scene_id"])].append(row)
    verdicts = physics_verdicts(Path(physics) if physics else None)
    assets = scene_assets(root)
    origin = Path(labels).stem

    written = grasps = held_known = 0
    for cloud, identity, points in mine:
        scene_json = root / "scenes" / cloud.stem / "scene.json"
        if not scene_json.is_file():
            raise FileNotFoundError(
                f"{cloud} was extracted from {identity}, and {root} has no scene {cloud.stem}")
        payload = json.loads(scene_json.read_text(encoding="utf-8"))
        geometry = assets.geometry(payload, cloud.stem)
        spec_objects = (payload.get("spec") or {}).get("objects") or []
        assets_by_instance = {index: str(entry.get("asset_id") or "")
                              for index, entry in enumerate(spec_objects)}
        table = scene_grasp_table(by_scene.get(cloud.stem, []), geometry, verdicts,
                                  assets_by_instance, kinds=("jaw",), origin=origin, model=model)
        arrays: dict[str, np.ndarray] = {key: value for key, value in table.items()
                                         if key.startswith(("grasp_", "contact_"))}
        arrays.update({
            "table_version": np.asarray([TABLE_VERSION], dtype=np.int32),
            "gripper": np.asarray([gripper], dtype="<U32"),
            "label_file": np.asarray([labels], dtype="<U64"),
            "source_dataset": np.asarray([identity], dtype="<U64"),
            "cloud_points": np.asarray([points], dtype=np.int64),
            # The hand's opening, so a sample read with this table caps its jitter at the hand.
            "gripper_aperture_mm": np.asarray([float(model.aperture_mm)], dtype=np.float32),
        })
        # Through a handle: given a path, numpy appends `.npz`, and a scene walk would read the
        # table as a scene.
        with table_path(cloud, gripper).open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        written += 1
        grasps += len(table["grasp_width_mm"])
        held_known += int((table["grasp_held"] >= 0).sum())

    report: dict[str, Any] = {
        "gripper": gripper, "label_file": labels, "clouds": len(mine), "tables_written": written,
        "clouds_of_other_datasets": foreign, "grasps": grasps, "grasps_with_physics": held_known,
    }
    logger.info("grasp tables for %s: %d beside %d cloud(s) of %s, %d grasp(s), %d with physics, "
                "%d cloud(s) of other datasets skipped", gripper, written, len(mine), root.name,
                grasps, held_known, foreign)
    return report
