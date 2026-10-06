"""What one look saw, drawn for a person: every detection's box, mask, label and score, and the grasp the pick chose.

The owner, 2026-10-05: "Debug Bilder speichern von Boundingbox und Segmentierung mit Labeln", and the grasp drawn as
research papers draw it, thin. A grasp is drawn as the grasp rectangle of the literature (Jiang et al. 2011, Lenz et
al. 2015, Morrison et al. 2018): the two edges the jaws close from in red, the two open sides in blue, a dot at the
grasp centre, all one or two pixels wide so the part stays visible under it.

Pure numpy and OpenCV; it reads nothing and moves nothing. :func:`draw_scene_debug` returns a BGR ``uint8`` image,
:func:`save_scene_debug` writes it as PNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = ["GraspRectangle", "draw_scene_debug", "grasp_rectangle", "save_scene_debug"]

#: Jaw edges red, open sides blue (BGR), the colours of the grasp-rectangle figures.
_JAW_BGR = (40, 40, 230)
_OPEN_BGR = (230, 120, 30)
#: How deep the jaws are drawn along the approach-free axis, in millimetres: a Hand-E finger's width.
FINGER_DEPTH_MM = 20.0
#: How much of a mask's own colour shows over the image.
_MASK_ALPHA = 0.35


class GraspRectangle:
    """The four image corners of a grasp rectangle, jaw edge first: ``corners[0]-corners[1]`` and
    ``corners[2]-corners[3]`` are where the jaws close from, the other two sides are the opening."""

    def __init__(self, corners: np.ndarray, centre: np.ndarray) -> None:
        self.corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        self.centre = np.asarray(centre, dtype=np.float64).reshape(2)


def _project(points_cam_mm: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    k = np.asarray(intrinsics, dtype=np.float64)
    z = np.maximum(points_cam_mm[:, 2], 1e-6)
    return np.column_stack((k[0, 0] * points_cam_mm[:, 0] / z + k[0, 2], k[1, 1] * points_cam_mm[:, 1] / z + k[1, 2]))


def grasp_rectangle(
    position_mm: Sequence[float],
    axis: Sequence[float],
    approach: Sequence[float],
    width_mm: float,
    intrinsics: np.ndarray,
    *,
    base_to_camera: np.ndarray | None = None,
    finger_depth_mm: float = FINGER_DEPTH_MM,
) -> GraspRectangle | None:
    """The grasp as a rectangle in the image: centre ``position_mm``, closing along ``axis`` over ``width_mm``, the
    jaws ``finger_depth_mm`` deep across it. The grasp is in the camera frame unless ``base_to_camera`` (4x4, BASE to
    CAMERA, millimetres) is given. ``None`` where it stands behind the camera."""
    p = np.asarray(position_mm, dtype=np.float64).reshape(3)
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.asarray(approach, dtype=np.float64).reshape(3)
    if base_to_camera is not None:
        m = np.asarray(base_to_camera, dtype=np.float64)
        p = m[:3, :3] @ p + m[:3, 3]
        a = m[:3, :3] @ a
        n = m[:3, :3] @ n
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    across = np.cross(n, a)
    if float(np.linalg.norm(across)) < 1e-9:
        return None
    across = across / float(np.linalg.norm(across))
    half_w, half_d = 0.5 * float(width_mm), 0.5 * float(finger_depth_mm)
    corners = np.array([p - half_w * a - half_d * across, p - half_w * a + half_d * across,
                        p + half_w * a + half_d * across, p + half_w * a - half_d * across])
    if np.any(corners[:, 2] <= 1.0) or p[2] <= 1.0:
        return None
    image = _project(corners, intrinsics)
    centre = _project(p.reshape(1, 3), intrinsics)[0]
    return GraspRectangle(image, centre)


def _colour(index: int) -> tuple[int, int, int]:
    """A distinct, steady colour per detection (golden-angle hues)."""
    hue = int((60 + index * 68.754) % 180)  # starts at green, never the jaws' red
    bgr = cv2.cvtColor(np.array([[[hue, 200, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


#: OpenCV's stroke font draws ASCII alone: a German label keeps its letters spelled out.
_SPELLED = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"})


def _ascii(text: str) -> str:
    return text.translate(_SPELLED).encode("ascii", "replace").decode("ascii")


def draw_scene_debug(
    rgb: np.ndarray,
    segmentations: Sequence[Any] = (),
    *,
    grasps: Sequence[GraspRectangle] = (),
    chosen: int | None = 0,
    title: str = "",
) -> np.ndarray:
    """``rgb`` (RGB, HxWx3) with every segmentation's mask tinted, its box drawn one pixel wide and ``label score`` over
    it, and each grasp rectangle; ``chosen`` is drawn two pixels wide, the others one. A segmentation is anything with
    ``mask``, ``bbox_xyxy``, ``label`` and ``score``."""
    image = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    tint = image.copy()
    for index, seg in enumerate(segmentations):
        colour = _colour(index)
        mask = getattr(seg, "mask", None)
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            if m.shape == image.shape[:2]:
                tint[m] = colour
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, colour, 1, cv2.LINE_AA)
    image = cv2.addWeighted(tint, _MASK_ALPHA, image, 1.0 - _MASK_ALPHA, 0.0)
    for index, seg in enumerate(segmentations):
        colour = _colour(index)
        box = getattr(seg, "bbox_xyxy", None)
        if box is not None:
            x0, y0, x1, y1 = (int(round(float(v))) for v in box)
            cv2.rectangle(image, (x0, y0), (x1, y1), colour, 1, cv2.LINE_AA)
            text = _ascii(f"{getattr(seg, 'label', '')} {float(getattr(seg, 'score', 0.0)):.2f}".strip())
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            top = max(y0 - th - 4, 0)
            cv2.rectangle(image, (x0, top), (x0 + tw + 4, top + th + 4), colour, -1)
            cv2.putText(image, text, (x0 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    for index, rect in enumerate(grasps):
        width = 2 if index == chosen else 1
        c = np.round(rect.corners).astype(np.int32)
        cv2.line(image, tuple(c[0]), tuple(c[1]), _JAW_BGR, width, cv2.LINE_AA)
        cv2.line(image, tuple(c[2]), tuple(c[3]), _JAW_BGR, width, cv2.LINE_AA)
        cv2.line(image, tuple(c[1]), tuple(c[2]), _OPEN_BGR, 1, cv2.LINE_AA)
        cv2.line(image, tuple(c[3]), tuple(c[0]), _OPEN_BGR, 1, cv2.LINE_AA)
        cv2.circle(image, tuple(np.round(rect.centre).astype(np.int32)), 2, _JAW_BGR, -1, cv2.LINE_AA)
    if title:
        cv2.putText(image, _ascii(title), (6, image.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return image


def save_scene_debug(path: str | Path, image_bgr: np.ndarray) -> Path:
    """Write ``image_bgr`` as PNG at ``path``, its folder made; the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image_bgr):
        raise OSError(f"the debug image could not be written to {out}")
    return out
