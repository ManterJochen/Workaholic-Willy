"""The grasp debug-overlay renderer.

A pure consumer: given an rgb image + the (camera/base-frame) candidates + intrinsics + the read-only telemetry
subset, it renders the overlay PNG. It owns no state: the facade (GraspCalculator) keeps
``last_debug_image_png`` + the ``rgb_image is not None`` guard and assigns this renderer's return value. The
frame math uses ``Rt = R.T`` (transpose, not inverse) with a per-candidate CAMERA-frame early-out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.robot.grasping.collision import GripperGeometryStrategy
from src.robot.grasping.geometry import CameraIntrinsics
from src.robot.grasping.visualization import DebugDrawConfig, draw_grasp_debug_image

from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint


class _DebugRenderer:
    """Renders the grasp debug overlay PNG. Internal collaborator of :class:`GraspCalculator`."""

    def __init__(self, logger: Any) -> None:
        # The same logger instance the facade created, injected so the exception-path message logs to the
        # identical handler/name as the facade.
        self._logger = logger

    def draw(
        self,
        *,
        rgb_image: np.ndarray,
        segmentation: Any,
        candidates: Sequence[GraspPoint],
        intrinsics: CameraIntrinsics,
        gripper_model: GripperGeometryStrategy | None,
        config: DebugDrawConfig | None,
        label: str | None,
        telemetry: dict,
        transform: np.ndarray | None,
    ) -> bytes | None:
        """Draw the debug overlay and return PNG bytes (or None on error).

        When the calculator was given ``T_cam_to_base`` the candidates are in base frame; the
        projection runs on a camera-frame copy so the overlay stays aligned with ``rgb_image``.

        The method is named ``draw`` and not ``render`` because ``render()`` has one meaning
        across this repository, fixed by ``src/contracts/reporting.py``: describe yourself to a
        person, as text, taking no arguments. This produces pixels from nine required inputs,
        which is a different operation and takes a different verb.
        """
        try:
            import cv2

            cam_candidates = self._to_camera_frame_for_render(candidates, transform)
            image = draw_grasp_debug_image(
                rgb_image=rgb_image,
                grasps=cam_candidates,
                intrinsics=intrinsics,
                mask=getattr(segmentation, "mask", None),
                gripper_model=gripper_model,
                config=config,
                label=label or getattr(segmentation, "label", None),
                telemetry={k: telemetry.get(k) for k in (
                    "candidates_silhouette",
                    "candidates_geometry",
                    "rejected_collision",
                    "rejected_table",
                    "rejected_workspace",
                    "final",
                    "best_score",
                )},
            )
            ok, buffer = cv2.imencode(".png", image)
            if not ok:
                return None
            return bytes(buffer)
        except Exception:  # noqa: BLE001 (optional debug overlay: a render failure must never break the pick)
            self._logger.exception("Failed to render grasp debug image")
            return None

    def _to_camera_frame_for_render(
        self,
        candidates: Sequence[GraspPoint],
        transform: np.ndarray | None,
    ) -> list[GraspPoint]:
        """Map base-frame candidates back to camera frame for projection.

        When no transform was applied the candidates are already in camera frame and returned unchanged.
        """
        if transform is None or not candidates:
            return list(candidates)
        R = transform[:3, :3]
        t = transform[:3, 3]
        Rt = R.T
        out: list[GraspPoint] = []
        for grasp in candidates:
            if grasp.frame == GraspFrame.CAMERA:
                out.append(grasp)
                continue
            position_cam = Rt @ (grasp.position - t)
            approach_cam = Rt @ grasp.approach
            axis_cam = Rt @ grasp.axis
            out.append(
                GraspPoint(
                    position=position_cam,
                    approach=approach_cam,
                    axis=axis_cam,
                    grip_width_mm=grasp.grip_width_mm,
                    score=grasp.score,
                    frame=GraspFrame.CAMERA,
                    label=grasp.label,
                    metadata=grasp.metadata,
                )
            )
        return out
