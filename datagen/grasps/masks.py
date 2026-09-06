"""The other mask source: GroundingDINO + SAM2 instead of the renderer's ground truth.

The renderer's own instance masks are the right starting point, because they isolate the grasp
generator from the perception in front of it, and they are also an advantage no real cell has. Both
sources are reported separately: predicted masks are written in exactly the encoding the
ground-truth ones use, and :mod:`datagen.eval.ladder` reads whichever it is pointed at.

The prompt deliberately does not test grounding. It names every object kind in the scene at once and
every detection is matched to the ground-truth instance it overlaps most, so a detector that finds
the right objects but attaches the wrong words to them still scores. Whether GroundingDINO picks the
object a sentence refers to is a different question, and mixing it in here would make a mask-quality
number unreadable. What this measures is narrower and is what the grasp generator consumes: given
that the right object was found, how good is its silhouette, and what does the difference cost the
grasps.

Unmatched objects are not dropped. An object the detector missed keeps its row with an empty mask,
so it lands in the evaluation as an object that produced no candidate: silence is a miss, the same
rule the ground-truth pass uses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, MASK_PREDICT_LOG_FILE

__all__ = ["predict_masks", "PRED_SUFFIX"]

logger = create_logger("datagen.grasps.masks", MASK_PREDICT_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Predicted masks are written beside the ground-truth ones under this name, in the same encoding
#: (uint16, instance_id + 1, 0 = background) so that nothing downstream needs a second code path.
PRED_SUFFIX = "pred_instances"
#: Below this IoU a detection is not the object; it is counted as a false positive rather than
#: forced onto the nearest instance. Deliberately loose: the question is mask quality given a hit,
#: and a strict threshold here would turn bad masks into missing objects.
MIN_MATCH_IOU = 0.25


def _kinds_in_scene(payload: dict) -> list[str]:
    """The object kinds present, read off the asset ids: ``proc_00007_bottle`` gives ``bottle``."""
    kinds = {str(obj["asset_id"]).rsplit("_", 1)[-1] for obj in payload["spec"]["objects"]}
    return sorted(k for k in kinds if k)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _match(predicted: list[np.ndarray], truth: dict[int, np.ndarray]) -> dict[int, int]:
    """Greedy best-IoU assignment, from predicted index to instance id.

    Greedy rather than Hungarian on purpose: with a handful of objects per view the two agree almost
    always, and where they do not it is because two detections landed on one object. There "give it
    to whichever fits best" is the honest reading, and an optimal assignment would invent a second
    object to justify the second box.
    """
    pairs = sorted(
        ((_iou(mask, truth[instance_id]), index, instance_id)
         for index, mask in enumerate(predicted) for instance_id in truth),
        reverse=True,
    )
    taken_pred: set[int] = set()
    taken_true: set[int] = set()
    assignment: dict[int, int] = {}
    for score, index, instance_id in pairs:
        if score < MIN_MATCH_IOU or index in taken_pred or instance_id in taken_true:
            continue
        assignment[index] = instance_id
        taken_pred.add(index)
        taken_true.add(instance_id)
    return assignment


def predict_masks(
    root: Path, *, limit: int | None = None, progress_every: int = 25, threshold: float | None = None,
    profile: str = "sim",
) -> dict:
    """Detect and segment every rendered view; write predicted instance maps and a report.

    Needs a GPU and downloads the two models on first use. Writes ``<view>_pred_instances.png`` next
    to each ground-truth map and ``masks_pred_report.json`` at the dataset root.

    The sim profile by default, and the choice is not cosmetic. These images come out of Isaac, and
    the sim profile is the one whose detector settings were measured against Isaac renders: it names
    a Hugging Face model rather than a local checkout that does not exist in a fresh tree, and it
    leaves ``torch_dtype`` unset. Under the default profile the detector loads in fp16, which the
    loading path itself warns can drop small-object recall. The resolved model ids and dtype land in
    the report, so no mask number is quoted without the model that produced it.
    """
    from PIL import Image  # noqa: PLC0415

    from src.config.loader import load_config  # noqa: PLC0415
    from src.models.detection.zero_shot.detector import (  # noqa: PLC0415
        GroundingDinoObjectDetector,
    )
    from src.models.segmentation.realtime.segmenter import Sam2Segmenter  # noqa: PLC0415

    root = Path(root)
    # The profile is passed, not set. Calling `set_active_profile(profile)` here and restoring
    # nothing changes the process's active profile for good. A CLI gets away with that because it
    # exits; a library caller does not, and a script that predicted masks once would find every
    # later `load_config()` returning the sim tree with nothing to connect the two.
    #
    # `profile=None` means the base tree, not "nobody said". The loader distinguishes the two with a
    # sentinel, so forwarding a None straight through would disable WILLY_PROFILE for anyone who
    # exports it. This site always names a profile, so it forwards a string.
    cfg = load_config(profile=profile)
    detector_cfg = cfg.models.objectdetector
    if threshold is not None:
        detector_cfg = detector_cfg.model_copy(update={"threshold": float(threshold)})
    # Which weights, at which precision, before a single mask exists. The dtype is logged because
    # fp16 can drop small-object recall, and a mask report that does not say which precision
    # produced it cannot be compared with the one before it.
    logger.info("loading detector %s (threshold %.2f, torch_dtype %s) and segmenter %s "
                "under profile '%s'",
                detector_cfg.model_id or detector_cfg.model_path, float(detector_cfg.threshold),
                detector_cfg.optim.torch_dtype,
                cfg.models.segmenter.model_id or cfg.models.segmenter.model_path, profile)
    load_started = time.perf_counter()
    detector = GroundingDinoObjectDetector(detector_cfg)
    segmenter = Sam2Segmenter(cfg.models.segmenter)
    logger.info("models ready in %.1f s", time.perf_counter() - load_started)

    rows: list[dict] = []
    scenes = views = 0
    started = time.perf_counter()
    for scene_dir in sorted((root / "scenes").iterdir()):
        scene_json = scene_dir / "scene.json"
        if not scene_json.exists():
            continue
        if limit is not None and scenes >= limit:
            break
        payload = json.loads(scene_json.read_text(encoding="utf-8"))
        prompt = ". ".join(_kinds_in_scene(payload)) + "."
        scenes += 1
        for view in payload["views"]:
            if str(view.get("outcome")) not in ("rendered", "rendered_after_resample"):
                continue
            rgb = np.asarray(Image.open(scene_dir / f"{view['name']}_rgb.png").convert("RGB"))
            instances = np.asarray(Image.open(scene_dir / f"{view['name']}_instances.png"))
            # The detector and the segmenter both take BGR and swap internally; handing them RGB
            # double-swaps and every colour word in the prompt grounds on the wrong colour.
            bgr = rgb[:, :, ::-1].copy()

            truth = {int(obj["instance_id"]): instances == (int(obj["instance_id"]) + 1)
                     for obj in view["objects"]}
            truth = {k: m for k, m in truth.items() if m.any()}

            detect_started = time.perf_counter()
            detections = detector.detect_all(bgr, prompt)
            detect_ms = (time.perf_counter() - detect_started) * 1000.0

            masks: list[np.ndarray] = []
            segment_ms = 0.0
            for detection in detections:
                segment_started = time.perf_counter()
                try:
                    result = segmenter.segment_detection(bgr, detection)
                except (RuntimeError, ValueError):
                    # SAM2 raises on an empty mask. That is a detection that segmented to nothing,
                    # which is a miss rather than a crash, recorded by its absence from the
                    # assignment.
                    continue
                finally:
                    segment_ms += (time.perf_counter() - segment_started) * 1000.0
                masks.append(np.asarray(result.mask, dtype=bool))

            assignment = _match(masks, truth)
            predicted = np.zeros(instances.shape, dtype=np.uint16)
            # Painted in ascending IoU so that where two predicted masks overlap, the better match wins
            # the contested pixels rather than whichever object happened to be numbered last.
            for index, instance_id in sorted(assignment.items(),
                                             key=lambda kv: _iou(masks[kv[0]], truth[kv[1]])):
                predicted[masks[index]] = np.uint16(instance_id + 1)
            Image.fromarray(predicted).save(scene_dir / f"{view['name']}_{PRED_SUFFIX}.png")

            matched = {instance_id: index for index, instance_id in assignment.items()}
            for instance_id, true_mask in sorted(truth.items()):
                hit = matched.get(instance_id)
                rows.append({
                    "scene_id": scene_dir.name, "view": view["name"], "instance_id": instance_id,
                    "found": hit is not None,
                    "iou": round(_iou(masks[hit], true_mask), 4) if hit is not None else 0.0,
                    "true_px": int(np.count_nonzero(true_mask)),
                    "pred_px": int(np.count_nonzero(masks[hit])) if hit is not None else 0,
                    "visibility": float(next(
                        (o.get("visibility", 0.0) for o in view["objects"]
                         if int(o["instance_id"]) == instance_id), 0.0)),
                    "family": str(payload["spec"]["family"]),
                })
            rows.append({
                "scene_id": scene_dir.name, "view": view["name"], "instance_id": -1,
                "detections": len(detections), "segmented": len(masks),
                "matched": len(assignment), "objects": len(truth),
                "prompt": prompt, "detect_ms": round(detect_ms, 1),
                "segment_ms": round(segment_ms, 1),
            })
            views += 1
        if progress_every and scenes % progress_every == 0:
            rate = (time.perf_counter() - started) / scenes
            print(f"  {scenes} Szenen, {views} Ansichten ({rate:.2f} s/Szene)", flush=True)

    with (root / "masks_pred.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = summarise_masks(root)
    # Aggregated: the detector and the segmenter log their own per-inference lines, so a second one
    # per view here would double the log for nothing. Recall and IoU are the pass's result.
    logger.info("predicted masks for %d view(s) over %d scene(s) in %.1f s: "
                "%d/%d objects found (%.1f%%), median IoU %.3f -> %s",
                views, scenes, time.perf_counter() - started, report["found"], report["objects"],
                report["recall"] * 100.0, report["iou"].get("median", 0.0),
                root / "masks_pred.jsonl")
    report["model"] = {"detector": detector_cfg.model_id or detector_cfg.model_path,
                       "threshold": float(detector_cfg.threshold),
                       "torch_dtype": str(detector_cfg.optim.torch_dtype),
                       "profile": profile,
                       "segmenter": cfg.models.segmenter.model_id or cfg.models.segmenter.model_path}
    (root / "masks_pred_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def summarise_masks(root: Path) -> dict:
    """Detection recall and mask IoU, from ``masks_pred.jsonl`` alone."""
    root = Path(root)
    path = root / "masks_pred.jsonl"
    if not path.exists():
        return {"objects": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    objects = [row for row in rows if row["instance_id"] >= 0]
    per_view = [row for row in rows if row["instance_id"] < 0]
    found = [row for row in objects if row["found"]]
    ious = np.array([row["iou"] for row in found]) if found else np.zeros(0)

    report: dict = {
        "objects": len(objects),
        "views": len(per_view),
        "found": len(found),
        "recall": round(len(found) / len(objects), 4) if objects else 0.0,
        # Spurious boxes matter for a real cell: each one is an object the pick loop would try.
        "detections_per_view": round(
            float(np.mean([row["detections"] for row in per_view])), 2) if per_view else 0.0,
        "unmatched_detections_per_view": round(float(np.mean(
            [row["detections"] - row["matched"] for row in per_view])), 2) if per_view else 0.0,
        "iou": {},
        "by_family": {},
        "by_visibility": {},
    }
    if ious.size:
        report["iou"] = {
            "mean": round(float(ious.mean()), 4),
            "median": round(float(np.median(ious)), 4),
            "p10": round(float(np.percentile(ious, 10)), 4),
            "p90": round(float(np.percentile(ious, 90)), 4),
        }
    for family in sorted({row["family"] for row in objects}):
        subset = [row for row in objects if row["family"] == family]
        hit = [row for row in subset if row["found"]]
        report["by_family"][family] = {
            "objects": len(subset),
            "recall": round(len(hit) / len(subset), 4) if subset else 0.0,
            "median_iou": round(float(np.median([row["iou"] for row in hit])), 4) if hit else 0.0,
        }
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        subset = [row for row in objects if low <= row["visibility"] < high]
        hit = [row for row in subset if row["found"]]
        report["by_visibility"][f"{low:.2f}-{high:.2f}"] = {
            "objects": len(subset),
            "recall": round(len(hit) / len(subset), 4) if subset else None,
            "median_iou": round(float(np.median([row["iou"] for row in hit])), 4) if hit else None,
        }
    return report
