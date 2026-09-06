"""Reading a corpus into canonical columns: where every file format quirk is paid for, once.

A `FeatureSpec` names canonical columns; a file on disk names its own. The mapping between them is a
fact about a file format rather than about the model, so it lives here and nowhere else.

Units are converted at the edge, the way a vendor unit stays inside a driver: everything past this
module is millimetres. A corpus that reached a comparison still in metres would report a 1000x
distribution shift and read like a real finding.

``np.load`` on an ``.npz`` returns a lazy ``NpzFile``, and every ``d[key]`` re-decompresses that whole
column, so every loader here hoists its columns in one pass rather than indexing inside a per-row
comprehension. ``_load_ranker_npz`` is the shortest example.

Every source here reads a format this package produces. There is no reader for a foreign corpus, so
no foreign unit or frame convention can enter through this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np

from datagen.corpus.columns import Table

__all__ = ["load_corpus", "SOURCES"]

_M_TO_MM: Final[float] = 1000.0


def _load_grasp_labels_jsonl(path: Path) -> Table:
    """The analytic grasp labels (``grasps.jsonl``): millimetres and BASE-frame vectors.

    The pose column names carry no frame, so a corpus in another frame under the same names would
    invite exactly the comparison that has to be refused. `FeatureSpec.frame` is what carries the
    frame, and the gate checks it.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    jaw = [row for row in rows if row.get("kind") == "jaw"]
    if not jaw:
        raise ValueError(f"{path} contains no jaw labels")

    def column(name: str, index: int | None = None) -> np.ndarray:
        if index is None:
            return np.asarray([float(row[name]) for row in jaw], dtype=np.float64)
        return np.asarray([float(row[name][index]) for row in jaw], dtype=np.float64)

    return {
        "object_id": np.asarray([f"{row['scene_id']}#{row['instance_id']}" for row in jaw]),
        "width_mm": column("width_mm"),
        "centre_x_mm": column("position_mm", 0),
        "centre_y_mm": column("position_mm", 1),
        "centre_z_mm": column("position_mm", 2),
        "axis_x": column("closing_axis", 0),
        "axis_y": column("closing_axis", 1),
        "axis_z": column("closing_axis", 2),
        "approach_x": column("approach", 0),
        "approach_y": column("approach", 1),
        "approach_z": column("approach", 2),
        "approach_tilt_deg": column("approach_tilt_deg"),
        "contact_angle_deg": column("contact_angle_deg"),
    }


def _load_grasp_eval_jsonl(path: Path) -> Table:
    """The calculator's own candidates, judged (``grasp_eval*.jsonl``): the closest thing to serving.

    This is the population a deployed scorer would score: whatever the generator proposes, not what
    the analytic reference enumerates. That makes it the right second corpus for the train/serve
    check in `gate.assess_corpus`.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    jaw = [row for row in rows if row.get("kind") == "jaw" and "position_mm" in row]
    if not jaw:
        raise ValueError(f"{path} contains no jaw candidate rows")

    def column(name: str, index: int | None = None) -> np.ndarray:
        if index is None:
            return np.asarray([float(row.get(name) or 0.0) for row in jaw], dtype=np.float64)
        return np.asarray([float((row.get(name) or (0.0, 0.0, 0.0))[index]) for row in jaw],
                          dtype=np.float64)

    return {
        "object_id": np.asarray([f"{row['scene_id']}#{row['instance_id']}" for row in jaw]),
        "width_mm": column("commanded_width_mm"),
        "centre_x_mm": column("position_mm", 0),
        "centre_y_mm": column("position_mm", 1),
        "centre_z_mm": column("position_mm", 2),
        "axis_x": column("closing_axis", 0),
        "axis_y": column("closing_axis", 1),
        "axis_z": column("closing_axis", 2),
        "approach_x": column("approach", 0),
        "approach_y": column("approach", 1),
        "approach_z": column("approach", 2),
        "approach_tilt_deg": column("approach_tilt_deg"),
        "contact_angle_deg": column("contact_angle_deg"),
    }


def _load_ranker_npz(path: Path) -> Table:
    """A corpus built by `datagen.corpus.build`: already canonical columns, so this only reads it.

    Every column is hoisted in one pass, because `np.load` on an `.npz` is lazy and each `data[name]`
    would re-decompress that whole column again.
    """
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}


#: What each source name reads. Named rather than sniffed: a corpus whose format was guessed is a
#: corpus whose units and frame were guessed with it.
SOURCES: Final[dict[str, str]] = {
    "grasp_labels": "Willy analytic grasp labels (grasps.jsonl)",
    "grasp_eval": "Willy calculator candidates, judged (grasp_eval*.jsonl)",
    "ranker_npz": "a corpus built by `datagen build-ranker-corpus` (canonical columns)",
}


def load_corpus(path: str | Path, source: str) -> Table:
    """Read one corpus into canonical columns. ``source`` must name a format, never be inferred."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such corpus: {path}")
    if source == "grasp_labels":
        return _load_grasp_labels_jsonl(path)
    if source == "grasp_eval":
        return _load_grasp_eval_jsonl(path)
    if source == "ranker_npz":
        return _load_ranker_npz(path)
    known = ", ".join(sorted(SOURCES))
    raise ValueError(f"unknown corpus source {source!r}; known: {known}")
