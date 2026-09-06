"""Does cross-view association pick the right object? Measured, on real masks, against ground truth.

Fusing a target cloud across cameras rests on something no camera can do alone: deciding that the
blob camera B segmented is the same physical object camera A is looking at. The one place in this
repo that fuses a target cloud today does it with ground-truth perception in every camera, which is
not a method, it is the answer.

So this pass runs the real thing. For every rendered view it takes the target as that camera
segmented it (GroundingDINO + SAM2, via ``predict-masks``), offers the other cameras' segmented
objects as candidates, and asks :mod:`~src.robot.grasping.multiview.association` which one is the
same object. The dataset knows the truth, so every decision is scored:

* correct: it chose the candidate belonging to the same instance,
* wrong: it chose a different object, which is the expensive failure, because it fabricates a
  contact face on the far side of a neighbour and the generator cannot tell that cloud from a real
  one,
* abstained: nothing cleared the threshold, which costs one view and no more.

And separately, the case that decides whether a threshold is worth anything at all: the other camera
never detected the target. No association method can win those, and the only right answer is to
abstain. A metric measured only on the answerable cases looks best when it never abstains, which is
exactly the behaviour that fabricates a neighbour's surface into the target's cloud. So those are
scored too, as held (abstained, correct) against fabricated (picked something, wrong), and the two
tables have to be read together.

Three metrics are measured on the same decisions rather than one being argued for, and the threshold
is swept rather than chosen, because the interesting quantity is not a single accuracy: it is how
much wrongness has to be accepted to buy each extra view.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.robot.grasping.multiview.association import (
    AssociationMetric,
    ViewCandidates,
    associate_target,
)

__all__ = ["evaluate_association", "SWEEP"]

#: Thresholds swept per metric. The point of the sweep is the trade, so it spans "take almost anything"
#: to "take almost nothing" rather than clustering around a value someone already liked.
SWEEP: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
#: Objects smaller than this in a view are not offered as candidates: below it a mask is a few pixels of
#: noise and would only add a lottery ticket to every decision.
MIN_CANDIDATE_PX = 100


def _clouds_for_view(
    scene_dir: Path, view: dict, suffix: str,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Every segmented object of one view as a BASE-frame cloud, keyed by the instance it really is."""
    from PIL import Image  # noqa: PLC0415

    from datagen.render.camera import unproject_to_base  # noqa: PLC0415

    depth = np.asarray(Image.open(scene_dir / f"{view['name']}_depth.png"), dtype=np.float64)
    labels = np.asarray(Image.open(scene_dir / f"{view['name']}_{suffix}.png"))
    camera_to_base = np.asarray(view["camera_to_base_mm"], dtype=np.float64)
    intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
    clouds: dict[int, np.ndarray] = {}
    for value in sorted(int(v) for v in np.unique(labels) if v):
        mask = labels == value
        if int(mask.sum()) < MIN_CANDIDATE_PX:
            continue
        points = unproject_to_base(depth, mask, camera_to_base, intrinsics)
        if points.shape[0]:
            clouds[value - 1] = points
    return clouds, camera_to_base


def evaluate_association(
    root: Path, *, limit: int | None = None, suffix: str = "pred_instances",
    metrics: Sequence[AssociationMetric] = tuple(AssociationMetric),
    progress_every: int = 25, scene_step: int = 1,
) -> dict:
    """Score every cross-view decision the fusion would make. Writes ``association_report.json``."""
    root = Path(root)
    tally: dict[tuple[str, float], dict[str, int]] = {}
    absent_tally: dict[tuple[str, float], dict[str, int]] = {}
    margins: dict[str, list[float]] = {m.value: [] for m in metrics}
    unavailable = decisions = 0
    scenes = index = 0
    started = time.perf_counter()

    for scene_dir in sorted((root / "scenes").iterdir()):
        scene_json = scene_dir / "scene.json"
        if not scene_json.exists():
            continue
        if limit is not None and scenes >= limit:
            break
        index += 1
        # Every Nth scene, never the first N. The renderer's order is alphabetical by family, so a
        # contiguous slice is one family, and `bin` is not where cross-view association is hard.
        # `pile` is.
        if (index - 1) % max(1, scene_step):
            continue
        payload = json.loads(scene_json.read_text(encoding="utf-8"))
        rendered = [v for v in payload["views"]
                    if str(v.get("outcome")) in ("rendered", "rendered_after_resample")]
        if len(rendered) < 2:
            continue
        scenes += 1
        per_view = {}
        for view in rendered:
            if not (scene_dir / f"{view['name']}_{suffix}.png").exists():
                continue
            per_view[view["name"]] = _clouds_for_view(scene_dir, view, suffix)[0]

        for primary_name, primary_clouds in per_view.items():
            others = [(name, clouds) for name, clouds in per_view.items() if name != primary_name]
            for instance_id, target in primary_clouds.items():
                # The candidate list a real cell would offer: what the other cameras segmented, in
                # their own detection order, with no hint of which is which.
                views = tuple(ViewCandidates(name, tuple(clouds[k] for k in sorted(clouds)))
                              for name, clouds in others)
                truth = tuple(sorted(clouds) for _, clouds in others)
                for metric in metrics:
                    results = associate_target(target, views, metric=metric, min_score=0.0)
                    for result, ids in zip(results, truth, strict=True):
                        if instance_id not in ids:
                            # The target is not in this view at all. Every candidate offered is a
                            # different object, so the only right answer is to take none of them.
                            if metric is metrics[0]:
                                unavailable += 1        # counted once, not once per metric
                            for threshold in SWEEP:
                                absent = absent_tally.setdefault(
                                    (metric.value, threshold), {"held": 0, "fabricated": 0})
                                absent["held" if result.score < threshold else "fabricated"] += 1
                            continue
                        if metric is metrics[0]:
                            decisions += 1
                        want = ids.index(instance_id)
                        margins[metric.value].append(result.margin)
                        for threshold in SWEEP:
                            bucket = tally.setdefault((metric.value, threshold),
                                                      {"correct": 0, "wrong": 0, "abstained": 0})
                            if result.score < threshold:
                                bucket["abstained"] += 1
                            elif result.index == want:
                                bucket["correct"] += 1
                            else:
                                bucket["wrong"] += 1
        if progress_every and scenes % progress_every == 0:
            rate = (time.perf_counter() - started) / scenes
            print(f"  {scenes} Szenen ({rate:.2f} s/Szene)", flush=True)

    report: dict = {
        "scenes": scenes,
        "scene_step": scene_step,
        "decisions": decisions,
        "unavailable": unavailable,
        "note": ("'sweep' covers the answerable decisions (the target IS in the other view). "
                 "'absent_sweep' covers the 'unavailable' ones, where the target is NOT there and the "
                 "only right answer is to abstain; read them together, because a metric that never "
                 "abstains looks perfect on the first table and fabricates surfaces on the second"),
        "by_metric": {},
    }
    for metric in metrics:
        rows = []
        for threshold in SWEEP:
            counts = tally.get((metric.value, threshold))
            if not counts:
                continue
            total = sum(counts.values())
            rows.append({
                "threshold": threshold,
                "correct": round(counts["correct"] / total, 4) if total else 0.0,
                "wrong": round(counts["wrong"] / total, 4) if total else 0.0,
                "abstained": round(counts["abstained"] / total, 4) if total else 0.0,
                "n": total,
            })
        absent_rows = []
        for threshold in SWEEP:
            counts = absent_tally.get((metric.value, threshold))
            if not counts:
                continue
            total = sum(counts.values())
            absent_rows.append({
                "threshold": threshold,
                "held": round(counts["held"] / total, 4) if total else 0.0,
                "fabricated": round(counts["fabricated"] / total, 4) if total else 0.0,
                "n": total,
            })
        got = margins[metric.value]
        report["by_metric"][metric.value] = {
            "sweep": rows,
            "absent_sweep": absent_rows,
            "median_margin": round(float(np.median(got)), 4) if got else 0.0,
        }
    (root / "association_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
