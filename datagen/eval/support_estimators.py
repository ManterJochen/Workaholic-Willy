"""Where is the surface this object stands on? Four ways of answering, measured against the truth.

The generator has to know the support height or it cannot tell a grasp from one that drives a 62 mm
finger through the table. The calculator already accepts a ``support_plane``, and
``validate_grasp_collision`` in ``collision_checker.py`` skips the table check entirely when it is
``None``.

So the question is not whether to pass one. It is what to put in it, and that is a measurement rather
than a preference, because the four candidates disagree exactly where it matters:

* table: the configured constant. Free, exact on a bare table, and wrong by the height of whatever
  an object is standing on.
* footprint: the lowest point of the target's own cloud. Says what this object rests on, stacked or
  not, and needs no scene reasoning. But a camera cannot see an object's underside: from straight
  above the lowest visible point is the silhouette rim, not the base.
* annulus: a plane through the scene points in a ring around the target. Robust to a noisy target
  edge, and it finds the surface beside the object, which for something sitting on a stack is the
  wrong one.
* scene: one plane through everything. Cheapest and steadiest, and the one most likely to be wrong
  in a bin.

The truth is exact and needs no inference: an object's own lowest point is the height of whatever it
rests on. It is read from the settled solid, not from a depth image.

Errors are reported signed, and the sign is the whole point. An estimate that is too low lets a
finger under the real surface, the failure that breaks a gripper. Too high only refuses grasps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, SUPPORT_EVAL_LOG_FILE

__all__ = ["evaluate_support_estimators", "ESTIMATORS", "support_height_estimates"]

logger = create_logger("datagen.eval.support_estimators", SUPPORT_EVAL_LOG_FILE,
                       log_dir=DATAGEN_LOG_DIR)

#: Which fraction of the lowest points in a region is taken to be the support surface. A region around
#: an object contains the object too, and a plane fitted through all of it sits halfway up the objects.
_GROUND_QUANTILE = 0.25
#: The annulus, as multiples of the target's own footprint radius. Wide enough to contain surface, tight
#: enough that it is the surface near the object.
_ANNULUS_INNER = 1.2
_ANNULUS_OUTER = 3.0
#: Below this many points an estimator abstains rather than fitting a plane to noise.
_MIN_POINTS = 30

ESTIMATORS = ("table", "footprint", "annulus", "scene")


def _ground_height(points: np.ndarray, at_xy: np.ndarray) -> float | None:
    """Height of the support surface under ``at_xy``, from the lowest stratum of ``points``.

    A plane is fitted to the lowest quarter by z rather than to everything, because everything includes
    the objects standing on the surface and a plane through those sits halfway up them.
    """
    if points.shape[0] < _MIN_POINTS:
        return None
    cutoff = float(np.quantile(points[:, 2], _GROUND_QUANTILE))
    ground = points[points[:, 2] <= cutoff]
    if ground.shape[0] < _MIN_POINTS:
        return None
    a = np.column_stack([ground[:, 0], ground[:, 1], np.ones(ground.shape[0])])
    try:
        coeffs, *_ = np.linalg.lstsq(a, ground[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(coeffs[0] * at_xy[0] + coeffs[1] * at_xy[1] + coeffs[2])


def support_height_estimates(
    target_cloud: np.ndarray, scene_cloud: np.ndarray, *, table_mm: float = 0.0,
) -> dict[str, float | None]:
    """The four estimates for one object, in BASE mm. ``None`` where an estimator abstains."""
    target = np.asarray(target_cloud, dtype=np.float64).reshape(-1, 3)
    scene = np.asarray(scene_cloud, dtype=np.float64).reshape(-1, 3)
    out: dict[str, float | None] = {"table": float(table_mm), "footprint": None,
                                    "annulus": None, "scene": None}
    if target.shape[0] == 0:
        return out
    centre_xy = target[:, :2].mean(axis=0)
    out["footprint"] = float(target[:, 2].min())
    if scene.shape[0]:
        radius = float(np.linalg.norm(target[:, :2] - centre_xy, axis=1).mean()) or 1.0
        distance = np.linalg.norm(scene[:, :2] - centre_xy, axis=1)
        ring = scene[(distance >= _ANNULUS_INNER * radius) & (distance <= _ANNULUS_OUTER * radius)]
        out["annulus"] = _ground_height(ring, centre_xy)
        out["scene"] = _ground_height(scene, centre_xy)
    return out


def _bottom_z_mm(solid) -> float:  # noqa: ANN001 (Solid, kept import-light)
    """The object's own lowest point. This is the height of whatever it is standing on."""
    return float(solid.centre_mm[2]) - 0.5 * float(
        solid.support_width_mm(np.array([0.0, 0.0, 1.0])))


def evaluate_support_estimators(
    root: Path, *, limit: int | None = None, scene_step: int = 1, suffix: str = "instances",
    progress_every: int = 25, table_mm: float = 0.0,
) -> dict:
    """Score all four against the settled geometry.

    Writes ``support_report.json`` and the per-object rows behind it to ``support_estimates.jsonl``.
    """
    from PIL import Image  # noqa: PLC0415

    from datagen.grasps.labels import scene_assets  # noqa: PLC0415
    from datagen.render.camera import unproject_to_base  # noqa: PLC0415

    root = Path(root)
    assets = scene_assets(root)
    rows: list[dict] = []
    scenes = index = 0
    started = time.perf_counter()

    for scene_dir in sorted((root / "scenes").iterdir()):
        scene_json = scene_dir / "scene.json"
        if not scene_json.exists():
            continue
        if limit is not None and scenes >= limit:
            break
        index += 1
        if (index - 1) % max(1, scene_step):
            continue
        payload = json.loads(scene_json.read_text(encoding="utf-8"))
        geometry = assets.geometry(payload, scene_dir.name)
        scenes += 1
        for view in payload["views"]:
            if str(view.get("outcome")) not in ("rendered", "rendered_after_resample"):
                continue
            mask_path = scene_dir / f"{view['name']}_{suffix}.png"
            if not mask_path.exists():
                continue
            depth = np.asarray(Image.open(scene_dir / f"{view['name']}_depth.png"), dtype=np.float64)
            labels = np.asarray(Image.open(mask_path))
            camera_to_base = np.asarray(view["camera_to_base_mm"], dtype=np.float64)
            intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
            everything = unproject_to_base(depth, labels >= 0, camera_to_base, intrinsics)
            for obj in view["objects"]:
                instance_id = int(obj["instance_id"])
                solid = geometry.objects.get(instance_id)
                if solid is None:
                    continue
                mask = labels == (instance_id + 1)
                if int(mask.sum()) < 100:
                    continue
                target = unproject_to_base(depth, mask, camera_to_base, intrinsics)
                if target.shape[0] == 0:
                    continue
                # The scene without the target: an estimator must not be able to read the answer off
                # the object it is being asked about.
                others = unproject_to_base(depth, (labels > 0) & ~mask, camera_to_base, intrinsics)
                scene_cloud = others if others.shape[0] else everything
                estimates = support_height_estimates(target, scene_cloud, table_mm=table_mm)
                truth = _bottom_z_mm(solid)
                rows.append({
                    "scene_id": scene_dir.name, "view": view["name"], "instance_id": instance_id,
                    "family": geometry.family, "visibility": float(obj.get("visibility", 0.0)),
                    "truth_mm": round(truth, 3),
                    # An object whose base is above the table is standing on something else. That is
                    # the case the four estimators disagree on, so it is called out rather than
                    # averaged in.
                    "stacked": bool(truth > 5.0),
                    **{f"est_{k}": (None if v is None else round(v, 3))
                       for k, v in estimates.items()},
                })
        if progress_every and scenes % progress_every == 0:
            print(f"  {scenes} Szenen ({(time.perf_counter() - started) / scenes:.2f} s/Szene)",
                  flush=True)

    report = summarise_support(rows)
    report["scene_step"] = scene_step
    # The signed error per estimator is the result, and the sign is the decision: too low puts a
    # finger under the real surface. Reporting only the counts would hide the reason this ran.
    logger.info("support estimators over %d scene(s), %d object(s) (%d stacked) in %.1f s: %s",
                scenes, report["objects"], report["stacked_objects"],
                time.perf_counter() - started,
                {name: entry["all"].get("median_mm") for name, entry in
                 report["by_estimator"].items()})
    report_path = root / "support_report.json"
    estimates_path = root / "support_estimates.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with estimates_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    # After both writes, not between them: a line naming a file that is still unopened turns a
    # failed write into a log that says it succeeded. Sizes, because an empty file and a written
    # one are the same sentence otherwise.
    logger.info("wrote %s (%d bytes) and %s (%d bytes)",
                report_path, report_path.stat().st_size,
                estimates_path, estimates_path.stat().st_size)
    return report


def summarise_support(rows: list[dict]) -> dict:
    """Signed error per estimator, split by whether the object stands on something else."""
    report: dict = {"objects": len(rows), "by_estimator": {}}
    stacked = [r for r in rows if r["stacked"]]
    report["stacked_objects"] = len(stacked)
    for name in ESTIMATORS:
        entry: dict = {}
        for label, subset in (("all", rows), ("on_table", [r for r in rows if not r["stacked"]]),
                              ("stacked", stacked)):
            errors = np.array([r[f"est_{name}"] - r["truth_mm"] for r in subset
                               if r.get(f"est_{name}") is not None])
            if errors.size == 0:
                entry[label] = {"n": 0}
                continue
            entry[label] = {
                "n": int(errors.size),
                "abstained": len(subset) - int(errors.size),
                "median_mm": round(float(np.median(errors)), 2),
                "p90_abs_mm": round(float(np.percentile(np.abs(errors), 90)), 2),
                # The asymmetry is the decision. Too low lets a finger under the real surface.
                "too_low_5mm": round(float((errors < -5.0).mean()), 4),
                "too_high_5mm": round(float((errors > 5.0).mean()), 4),
                "within_5mm": round(float((np.abs(errors) <= 5.0).mean()), 4),
            }
        report["by_estimator"][name] = entry
    return report
