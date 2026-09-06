"""Building a ranker training corpus from a rendered dataset: features from the cloud, label from the
reference.

A scorer whose features cannot see an object's surroundings cannot predict the failure it is asked to
predict, which is ``finger_collision``. So this corpus carries the geometry each candidate sits in,
which no file of judged candidates holds; that is why this walks the scenes and re-derives the clouds
rather than reading a column.

Every feature comes from the point cloud, never from the reference. `verdict.py` computes the contact
geometry exactly, and a feature copied from it would train a model to reproduce the referee: perfect
on every scene, and an estimator of `verdict.py` rather than of the world. The one feature most
exposed to that, `contact_normal_alignment`, is estimated from a local patch of the cloud, which is
the worse answer on purpose because it is the answer a depth image can give.

The label is the reference's too, which makes the `eval-grasps` ladder self-referential for any model
fitted here. The ladder stays a development signal and promotion is measured against the Isaac shake
only.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.robot.grasping.deep.ranker.features import (
    environment_features,
    grasp_frame_offsets,
    object_centre_mm,
)
from src.utility.log_cfg import create_logger

from datagen.constants import CORPUS_GATE_LOG_FILE, DATAGEN_LOG_DIR
from datagen.grasps.identity import grasp_identity

__all__ = ["attach_physics_labels", "build_ranker_corpus", "write_corpus"]

logger = create_logger("CorpusBuild", CORPUS_GATE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Voxel edge for thinning the obstacle cloud before the nearest-point search. The clearance feature
#: needs the distance to the nearest obstacle, and a 6 mm grid changes that by at most ~5 mm while
#: turning an O(candidates x points) search from minutes into seconds. Stated because it is an
#: approximation inside a feature, not a free optimisation.
_OBSTACLE_VOXEL_MM = 6.0



def _thin(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    """One point per occupied voxel. Deterministic: the first point encountered in index order."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_mm).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def build_ranker_corpus(
    root: str | Path, *, scenes: int | None = None, pattern: str = "grasp_eval*.jsonl",
    mask_suffix: str = "instances",
) -> dict[str, np.ndarray]:
    """Walk a rendered dataset and return a table of features + `valid` + `object_key`.

    Returns the canonical-column shape `datagen.corpus` uses everywhere, so `gate.assess_corpus` can
    judge it before anything is fitted on it.

    ``mask_suffix`` selects which segmentation the clouds are cut with: ``instances`` is the
    renderer's ground truth, ``pred_instances`` what GroundingDINO + SAM2 produced, and a cell has
    the second one. Building both and comparing isolates the features, because the candidates and
    the labels are properties of the scene geometry and stay identical between the two.
    """
    from datagen.corpus.clouds import fuse_instance_clouds  # noqa: PLC0415
    from datagen.grasps.labels import scene_assets  # noqa: PLC0415

    root = Path(root)
    rows: list[dict] = []
    for path in sorted(root.glob(pattern)):
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "jaw" and row.get("outcome") == "candidate":
                # Which file, which line. The only key a physics result, which carries a pose and
                # nothing else, can be joined back to this row's features by. Joining on the pose is
                # ambiguous: a `label` row and the `valid` row describing the same grasp carry
                # identical poses.
                row["_source_row"] = f"{path.stem}:{index}"
                rows.append(row)
    if not rows:
        raise ValueError(f"no judged jaw candidates under {root}/{pattern}")

    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)

    assets = scene_assets(root)
    scene_dirs = [p for p in sorted((root / "scenes").iterdir())
                  if (p / "scene.json").exists() and p.name in by_scene]
    if scenes is not None:
        scene_dirs = scene_dirs[:scenes]

    columns: dict[str, list[Any]] = collections.defaultdict(list)
    seen: set[tuple] = set()
    skipped_sparse = skipped_duplicate = skipped_degenerate = 0

    for scene_dir in scene_dirs:
        payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
        geometry = assets.geometry(payload, scene_dir.name)
        clouds = fuse_instance_clouds(scene_dir, payload, geometry, mask_suffix=mask_suffix)

        # The obstacle cloud for one object is everything else that was observed, which is what the
        # pick loop passes as `scene_points_mm`, so the feature is computed from what a cell would have.
        thinned = {index: _thin(np.asarray(cloud, dtype=np.float64), _OBSTACLE_VOXEL_MM)
                   for index, cloud in clouds.items() if cloud is not None and len(cloud)}

        per_unit: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
        for row in by_scene[scene_dir.name]:
            per_unit[(str(row["view"]), int(row["instance_id"]))].append(row)

        for (view, instance), candidates in sorted(per_unit.items()):
            target = clouds.get(instance)
            if target is None or len(target) < 200:
                skipped_sparse += len(candidates)
                continue
            target = np.asarray(target, dtype=np.float64)
            others = [cloud for index, cloud in thinned.items() if index != instance]
            obstacles = np.vstack(others) if others else None
            centre = object_centre_mm(target, support_z_mm=0.0)

            for row in candidates:
                key = grasp_identity(row)
                if key in seen:
                    skipped_duplicate += 1
                    continue
                seen.add(key)
                position = np.asarray(row["position_mm"], dtype=np.float64)
                approach = np.asarray(row["approach"], dtype=np.float64)
                axis = np.asarray(row["closing_axis"], dtype=np.float64)
                width = float(row["commanded_width_mm"])
                try:
                    along = grasp_frame_offsets(centre, position, approach, axis)
                except ValueError:
                    # Axis parallel to approach: the GraspPoint contract forbids it, so this is a
                    # broken candidate rather than a hard one. Counted, not silently dropped.
                    skipped_degenerate += 1
                    continue
                environment = environment_features(
                    position, approach, axis, width, target,
                    obstacle_points_mm=obstacles, support_z_mm=0.0,
                    observedness=float(row.get("visibility") or 0.0))

                columns["width_mm"].append(width)
                columns["centre_along_axis_mm"].append(along[0])
                columns["centre_along_approach_mm"].append(along[1])
                columns["centre_along_binormal_mm"].append(along[2])
                for name, value in environment.items():
                    columns[name].append(value)
                columns["valid"].append(1.0 if row.get("valid") else 0.0)
                # Grouped by object, not by (view, object): the same object seen from two views is not
                # two independent samples of it, and a split that separated them would leak.
                columns["object_key"].append(f"{row['scene_id']}#{row['instance_id']}")
                columns["reason"].append(str(row.get("reason", "")))
                # Deliberately a string. A bare integer line index in a table of floats is one
                # `FeatureSpec` typo away from becoming the strongest feature in the model.
                columns["source_row"].append(row["_source_row"])

    table = {name: np.asarray(values) for name, values in columns.items()}
    logger.info(
        "built ranker corpus from %s: %d row(s) over %d object(s) from %d scene(s); "
        "skipped %d sparse, %d duplicate, %d degenerate",
        root, len(table.get("valid", [])), len(set(columns["object_key"])), len(scene_dirs),
        skipped_sparse, skipped_duplicate, skipped_degenerate,
    )
    return table


def attach_physics_labels(
    table: dict[str, np.ndarray], physics_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Join Isaac's ``held`` onto a ranker corpus, and report what did not join.

    Stage 1 of the deep ranker trains on `valid`, the analytic verdict, which makes any ladder
    measured on it self-referential. Stage 2 trains on `held`, which is physics, and this is the seam
    between them.

    Returns `(table, report)` where the table holds only the rows physics resolved. Three things are
    dropped and all three are counted rather than filtered away quietly:

    ``refused``     the harness declined to grade the trial (a blown-up jaw, a scene that drifted).
                    A refusal is not a failure, and folding one into `held=False` records a harness
                    fault as a bad grasp.
    ``unmatched``   the trial's source row is not in this corpus. Structural for `label` trials: they
                    come from `grasps.jsonl`, which `build_ranker_corpus` does not read at all. For a
                    `grasp_eval` trial it means the row was deduped or its object's cloud was too
                    sparse, and a rising count there is a real signal about the corpus.
    ``conflicting`` the same source row shaken twice with different verdicts. PhysX is not
                    deterministic here, so this is possible rather than impossible, and a silent
                    last-one-wins would hide how possible.

    The report also carries the per-source counts. The draw is stratified, so the pooled hold rate is
    an artefact of it and must never be quoted as the corpus's rate; per-source rates are the honest
    unit, and a population rate needs re-weighting by the stratum sizes.
    """
    if "source_row" not in table:
        raise ValueError(
            "this corpus carries no `source_row` column, so nothing can be joined to it. It was "
            "built before the join key existed; rebuild it with `datagen build-ranker-corpus`.")

    verdicts: dict[str, tuple[bool, str]] = {}     # source_row -> (held, which stratum shook it)
    counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"trials": 0, "held": 0, "refused": 0, "unmatched": 0, "conflicting": 0})
    conflicts: list[str] = []
    for line in Path(physics_path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if "controls" in row or "origin" not in row:
            continue          # the controls block, or a file written before the key existed
        bucket = counts[str(row["source"])]
        if str(row.get("note", "")).startswith("refused"):
            bucket["refused"] += 1
            continue
        key = f"{row['origin']}:{row['row_index']}"
        held = bool(row["held"])
        if key in verdicts and verdicts[key][0] != held:
            bucket["conflicting"] += 1
            conflicts.append(key)
            continue
        verdicts[key] = (held, str(row["source"]))
        bucket["trials"] += 1
        bucket["held"] += int(held)

    source_rows = [str(value) for value in table["source_row"]]
    keep = [index for index, key in enumerate(source_rows) if key in verdicts]
    matched = {source_rows[index] for index in keep}
    # Charged to the stratum that shook it, not to a bucket labelled "other": which source fails to
    # join is the finding. `label` failing is structural (its file is not read here); `valid` failing
    # means the corpus dropped the row, and that is worth seeing separately.
    for row_key, (_, source) in verdicts.items():
        if row_key not in matched:
            counts[source]["unmatched"] += 1

    joined = {name: np.asarray(values)[keep] for name, values in table.items()}
    joined["held"] = np.asarray([1.0 if verdicts[source_rows[i]][0] else 0.0 for i in keep])
    # Carried so training can slice by stratum. A string, like `source_row`: a stratum is a label.
    joined["physics_source"] = np.asarray([verdicts[source_rows[i]][1] for i in keep])

    report: dict[str, Any] = {
        "rows_in": len(source_rows),
        "rows_out": len(keep),
        "physics_verdicts": len(verdicts),
        "by_source": {k: dict(v) for k, v in sorted(counts.items())},
        "conflicting_keys": sorted(set(conflicts))[:20],
        "warning": ("stratified draw; the pooled hold rate is an artefact of it; "
                    "re-weight by the stratum sizes for a population rate"),
    }
    logger.info(
        "joined physics onto %d of %d corpus row(s) from %d verdict(s); %s",
        len(keep), len(source_rows), len(verdicts),
        {k: f"{v['held']}/{v['trials']}" for k, v in sorted(counts.items()) if v["trials"]})
    return joined, report


def write_corpus(path: str | Path, table: dict[str, np.ndarray]) -> Path:
    """Write the table as a compressed ``.npz``, one array per column."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # `savez_compressed`'s stub types its second positional as `allow_pickle`, so every column has to
    # arrive as a keyword argument rather than positionally.
    np.savez_compressed(out, **dict(table))  # type: ignore[arg-type]
    return out
