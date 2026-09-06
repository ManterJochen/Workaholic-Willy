"""2D debug rendering for grasp candidates.

Paints an annotated image, carrying the mask overlay, the projected gripper
boxes, the contact arrows and a score bar, which shows an operator why a grasp
was picked. The output is a plain ``np.uint8`` BGR image, so it can be served as
PNG without a frontend dependency on matplotlib or Open3D.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from src.robot.grasping.collision.gripper_model import (
    GripperGeometryStrategy,
    ParallelJawGripperModel,
)
from src.robot.grasping.constants import (
    DEBUG_DRAW_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.geometry.pointcloud import CameraIntrinsics
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = [
    "DebugDrawConfig",
    "draw_grasp_debug_image",
    "save_grasp_debug_image",
]

#: An overlay that shows fewer grasps than the pick considered is a misleading
#: artefact, and both causes are invisible in the PNG: a base-frame pose cannot be
#: projected, and ``max_grasps`` clips whatever is left. The logger names them.
logger = create_grasping_logger("DebugDraw", DEBUG_DRAW_LOG_FILE)


_BGR_GRASP_COLORS = (
    (0, 200, 255),    # amber, the best
    (0, 255, 100),    # green
    (255, 200, 0),    # cyan-ish
    (200, 100, 255),  # magenta
    (180, 180, 180),  # grey, the worst of those shown
)


@dataclass(frozen=True, slots=True)
class DebugDrawConfig:
    """Knobs for the 2D debug renderer."""

    max_grasps: int = 5
    mask_alpha: float = 0.35
    arrow_thickness_px: int = 2
    box_thickness_px: int = 2
    font_scale: float = 0.5
    show_score_bar: bool = True
    show_metadata: bool = True

    def __post_init__(self) -> None:
        if self.max_grasps < 1:
            raise ValueError("max_grasps must be >= 1")
        if not 0.0 <= self.mask_alpha <= 1.0:
            raise ValueError("mask_alpha must be in [0, 1]")
        for name in ("arrow_thickness_px", "box_thickness_px"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.font_scale <= 0.0:
            raise ValueError("font_scale must be > 0")


def _ensure_bgr_image(image: np.ndarray) -> np.ndarray:
    """Return a writable ``HxWx3`` BGR uint8 image."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be HxW or HxWx3/4, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def _project_points(
    points_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project camera-frame points to ``(N, 2)`` pixel coordinates.

    Points behind the camera (Z <= 0) become ``NaN`` so callers can skip them.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_mm must be (N, 3), got {points.shape}")
    out = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    in_front = points[:, 2] > 1e-6
    if np.any(in_front):
        z = points[in_front, 2]
        u = intrinsics.fx * points[in_front, 0] / z + intrinsics.cx
        v = intrinsics.fy * points[in_front, 1] / z + intrinsics.cy
        out[in_front, 0] = u
        out[in_front, 1] = v
    return out


def _paint_mask_overlay(
    canvas: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    if mask is None:
        return canvas
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != canvas.shape[:2]:
        raise ValueError(
            f"mask shape {mask_bool.shape} must match image {canvas.shape[:2]}"
        )
    overlay = canvas.copy()
    overlay[mask_bool] = (0, 255, 255)  # cyan tint
    return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)


def _build_gripper_local_segments(
    gripper: GripperGeometryStrategy,
    grip_width_mm: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return 12 line segments per box, so 36 segments for the three boxes."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    edge_pairs = (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )
    for box in gripper.collision_boxes(grip_width_mm):
        corners = box.corners_local_mm()
        for a, b in edge_pairs:
            segments.append((corners[a], corners[b]))
    return segments


def _draw_grasp(
    canvas: np.ndarray,
    grasp: GraspPoint,
    *,
    intrinsics: CameraIntrinsics,
    gripper: GripperGeometryStrategy,
    color: tuple[int, int, int],
    config: DebugDrawConfig,
    rank: int,
) -> None:
    R = np.column_stack(
        (grasp.axis, np.cross(grasp.approach, grasp.axis), grasp.approach)
    )
    # Re-orthogonalise the binormal to avoid tiny drift in the visualisation.
    binormal = R[:, 1]
    binormal_norm = float(np.linalg.norm(binormal))
    if binormal_norm < 1e-9:
        return
    R[:, 1] = binormal / binormal_norm

    # 1) project all corner segments
    segments = _build_gripper_local_segments(gripper, grasp.grip_width_mm)
    local = np.vstack([np.stack([a, b]) for a, b in segments])  # (2 * S, 3)
    world = local @ R.T + grasp.position
    projected = _project_points(world, intrinsics)
    H, W = canvas.shape[:2]

    for index in range(0, projected.shape[0], 2):
        p0 = projected[index]
        p1 = projected[index + 1]
        if np.any(~np.isfinite(p0)) or np.any(~np.isfinite(p1)):
            continue
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        if (x0 < 0 and x1 < 0) or (x0 >= W and x1 >= W):
            continue
        if (y0 < 0 and y1 < 0) or (y0 >= H and y1 >= H):
            continue
        cv2.line(canvas, (x0, y0), (x1, y1), color, config.box_thickness_px, cv2.LINE_AA)

    # 2) contact arrow along closing axis
    half = 0.5 * float(grasp.grip_width_mm)
    p_a = grasp.position - half * grasp.axis
    p_b = grasp.position + half * grasp.axis
    ends = _project_points(np.vstack([p_a, p_b]), intrinsics)
    if np.all(np.isfinite(ends)):
        cv2.arrowedLine(
            canvas,
            (int(round(ends[0, 0])), int(round(ends[0, 1]))),
            (int(round(ends[1, 0])), int(round(ends[1, 1]))),
            color,
            config.arrow_thickness_px,
            cv2.LINE_AA,
            tipLength=0.18,
        )

    # 3) rank label at projected centroid
    centre = _project_points(grasp.position[None, :], intrinsics)
    if np.all(np.isfinite(centre)):
        cx, cy = int(round(centre[0, 0])), int(round(centre[0, 1]))
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"#{rank} {grasp.score:.2f}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )


def _draw_score_bar(
    canvas: np.ndarray,
    grasps: Sequence[GraspPoint],
    config: DebugDrawConfig,
) -> np.ndarray:
    if not grasps or not config.show_score_bar:
        return canvas
    bar_height = 24
    bar = np.full((bar_height, canvas.shape[1], 3), 32, dtype=np.uint8)
    n = min(len(grasps), config.max_grasps)
    if n == 0:
        return canvas
    cell_w = max(1, canvas.shape[1] // n)
    for i in range(n):
        x0 = i * cell_w
        x1 = x0 + cell_w - 2
        score = float(np.clip(grasps[i].score, 0.0, 1.0))
        fill_w = int(round((x1 - x0) * score))
        color = _BGR_GRASP_COLORS[min(i, len(_BGR_GRASP_COLORS) - 1)]
        cv2.rectangle(bar, (x0, 4), (x1, bar_height - 4), color, 1, cv2.LINE_AA)
        cv2.rectangle(bar, (x0, 4), (x0 + fill_w, bar_height - 4), color, -1)
        cv2.putText(
            bar,
            f"{score:.2f}",
            (x0 + 4, bar_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale * 0.85,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([canvas, bar])


def _add_metadata_strip(
    canvas: np.ndarray,
    *,
    label: str | None,
    telemetry: dict | None,
    config: DebugDrawConfig,
) -> np.ndarray:
    if not config.show_metadata:
        return canvas
    parts: list[str] = []
    if label:
        parts.append(f"label={label}")
    if telemetry:
        parts.extend(f"{k}={v}" for k, v in telemetry.items())
    if not parts:
        return canvas
    text = "  ".join(parts)
    strip_height = 22
    strip = np.full((strip_height, canvas.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        strip,
        text[: max(1, canvas.shape[1] // 7)],
        (6, strip_height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.font_scale * 0.85,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([strip, canvas])


def draw_grasp_debug_image(
    rgb_image: np.ndarray,
    grasps: Iterable[GraspPoint],
    *,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    gripper_model: GripperGeometryStrategy | None = None,
    config: DebugDrawConfig | None = None,
    label: str | None = None,
    telemetry: dict | None = None,
) -> np.ndarray:
    """Render an annotated debug BGR image for the given grasp candidates."""
    cfg = config or DebugDrawConfig()
    canvas = _ensure_bgr_image(rgb_image)
    canvas = _paint_mask_overlay(canvas, mask, cfg.mask_alpha)

    # Base-frame candidates are skipped: projecting one needs a BASE to CAMERA
    # transform, which the debug renderer is not given.
    all_grasps = list(grasps)
    projectable = [g for g in all_grasps if g.frame == GraspFrame.CAMERA]
    grasp_list = projectable[: cfg.max_grasps]
    if len(grasp_list) != len(all_grasps):
        # A warning only when candidates were dropped as unprojectable. A plain
        # ``max_grasps`` clip is the configured behaviour rather than a defect.
        unprojectable = len(all_grasps) - len(projectable)
        log = logger.warning if unprojectable else logger.debug
        log(
            "Overlay draws %d of %d candidate(s): %d not in camera frame, "
            "%d clipped by max_grasps=%d",
            len(grasp_list),
            len(all_grasps),
            unprojectable,
            len(projectable) - len(grasp_list),
            cfg.max_grasps,
        )
    gripper = gripper_model or ParallelJawGripperModel()
    for rank, grasp in enumerate(grasp_list):
        color = _BGR_GRASP_COLORS[min(rank, len(_BGR_GRASP_COLORS) - 1)]
        _draw_grasp(
            canvas,
            grasp,
            intrinsics=intrinsics,
            gripper=gripper,
            color=color,
            config=cfg,
            rank=rank + 1,
        )

    canvas = _draw_score_bar(canvas, grasp_list, cfg)
    canvas = _add_metadata_strip(canvas, label=label, telemetry=telemetry, config=cfg)
    return canvas


def save_grasp_debug_image(
    path: str,
    rgb_image: np.ndarray,
    grasps: Iterable[GraspPoint],
    *,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    gripper_model: GripperGeometryStrategy | None = None,
    config: DebugDrawConfig | None = None,
    label: str | None = None,
    telemetry: dict | None = None,
) -> str:
    """Render and write the debug image to ``path``; return the path."""
    image = draw_grasp_debug_image(
        rgb_image,
        grasps,
        intrinsics=intrinsics,
        mask=mask,
        gripper_model=gripper_model,
        config=config,
        label=label,
        telemetry=telemetry,
    )
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise OSError(f"failed to write debug image to {path!r}")
    logger.info(
        "Wrote grasp debug overlay to %s (%dx%d, %d bytes)",
        path,
        image.shape[1],
        image.shape[0],
        os.path.getsize(path),
    )
    return path
