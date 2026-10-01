"""Box mAP the way COCO computes it, in numpy: the number detector papers and toolkits report, so a model trained here
reads on the same scale as one trained anywhere else.

What it follows from pycocotools' ``COCOeval`` (bbox):

- AP per class at ten IoU thresholds, 0.50 to 0.95 in steps of 0.05, each the mean precision at 101 recall points,
  with precision first made monotone from the right (``accumulate``);
- detections taken in score order and matched to the still unmatched ground-truth box of the same class and image
  with the highest IoU at or above the threshold;
- at most 100 detections per image and class (``maxDets=100``);
- a class without a ground-truth box is left out of the mean (COCOeval writes -1 there).

Crowd regions and the small/medium/large ranges are not modelled: the datasets here drop crowd boxes, and the size
ranges are a benchmark breakdown a fixed-class cell does not need. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["DetectionScores", "IOU_THRESHOLDS", "evaluate_detections", "box_iou"]

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)

#: A ground-truth box: (label, (x1, y1, x2, y2)). A detection: (label, score, (x1, y1, x2, y2)). Pixels.
GroundTruth = tuple[int, tuple[float, float, float, float]]
Prediction = tuple[int, float, tuple[float, float, float, float]]


@dataclass(frozen=True, slots=True)
class DetectionScores:
    """mAP@0.5:0.95 (``map``), at 0.5 and 0.75, the mean recall at 100 detections, and AP per class."""

    map: float
    map50: float
    map75: float
    mar100: float
    per_class_ap: dict[str, float] = field(default_factory=dict)
    per_class_ap50: dict[str, float] = field(default_factory=dict)
    images: int = 0
    boxes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"map": self.map, "map50": self.map50, "map75": self.map75, "mar100": self.mar100,
                "per_class_ap": dict(self.per_class_ap), "per_class_ap50": dict(self.per_class_ap50),
                "images": self.images, "boxes": self.boxes}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (N, 4) with every box in ``b`` (M, 4), corners x1 y1 x2 y2; shape (N, M)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


def evaluate_detections(ground_truth: Sequence[Sequence[GroundTruth]], predictions: Sequence[Sequence[Prediction]],
                        classes: Sequence[str], *, max_dets: int = 100) -> DetectionScores:
    """Score per-image predictions against per-image ground truth; both sequences are indexed by image."""
    if len(ground_truth) != len(predictions):
        raise ValueError(f"{len(ground_truth)} image(s) of ground truth and {len(predictions)} of predictions")
    n_thr = len(IOU_THRESHOLDS)
    ap = np.full((n_thr, len(classes)), np.nan)
    recall = np.full((n_thr, len(classes)), np.nan)
    for label in range(len(classes)):
        gts = [np.array([box for lab, box in image if lab == label], dtype=float).reshape(-1, 4)
               for image in ground_truth]
        n_pos = sum(len(g) for g in gts)
        if n_pos == 0:
            continue
        scores: list[np.ndarray] = []
        matched: list[np.ndarray] = []
        for gt, image in zip(gts, predictions):
            dets = [(score, box) for lab, score, box in image if lab == label]
            dets.sort(key=lambda item: -item[0])  # stable, as COCOeval's mergesort
            dets = dets[:max_dets]
            if not dets:
                continue
            det_scores = np.array([score for score, _ in dets], dtype=float)
            ious = box_iou(np.array([box for _, box in dets], dtype=float).reshape(-1, 4), gt)
            hits = np.zeros((n_thr, len(dets)), dtype=bool)
            for t, threshold in enumerate(IOU_THRESHOLDS):
                taken = np.zeros(len(gt), dtype=bool)
                for d in range(len(dets)):
                    best, best_iou = -1, min(threshold, 1.0 - 1e-10)
                    for g in range(len(gt)):
                        if taken[g] or ious[d, g] < best_iou:
                            continue
                        best, best_iou = g, ious[d, g]
                    if best >= 0:
                        taken[best] = True
                        hits[t, d] = True
            scores.append(det_scores)
            matched.append(hits)
        if not scores:
            ap[:, label] = 0.0
            recall[:, label] = 0.0
            continue
        all_scores = np.concatenate(scores)
        order = np.argsort(-all_scores, kind="mergesort")
        all_hits = np.concatenate(matched, axis=1)[:, order]
        for t in range(n_thr):
            tp = np.cumsum(all_hits[t]).astype(float)
            fp = np.cumsum(~all_hits[t]).astype(float)
            rc = tp / n_pos
            pr = tp / np.maximum(tp + fp, np.finfo(float).eps)
            recall[t, label] = rc[-1]
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            index = np.searchsorted(rc, _RECALL_POINTS, side="left")
            sampled = np.where(index < len(pr), pr[np.minimum(index, len(pr) - 1)], 0.0)
            ap[t, label] = float(sampled.mean())
    present = ~np.isnan(ap[0])

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else float("nan")

    return DetectionScores(
        map=mean(ap[:, present]),
        map50=mean(ap[0, present]),
        map75=mean(ap[5, present]),
        mar100=mean(recall[:, present]),
        per_class_ap={classes[i]: float(np.mean(ap[:, i])) for i in range(len(classes)) if present[i]},
        per_class_ap50={classes[i]: float(ap[0, i]) for i in range(len(classes)) if present[i]},
        images=len(ground_truth),
        boxes=sum(len(image) for image in ground_truth),
    )
