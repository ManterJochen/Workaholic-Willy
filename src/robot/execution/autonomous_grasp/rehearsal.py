"""A synthetic scene so the real-cell runner can be rehearsed at a desk, with no camera and no arm.

This is not a simulator. It exists so the real-cell runner can be exercised without the hardware it
is meant to de-risk: with ``--rehearse`` the same code path, config load, preflight,
``from_robot_config``, the connect order, the pick loop, safety and record logging, runs end to end
on a dummy arm against this scene, on a laptop, in seconds.

What it deliberately does not do: pretend to be perception. There is no model, no camera, no noise
model. It emits one flat box on a plane, at a known place, so what is exercised is the wiring.
Grasp quality is measured in Isaac and on the bench, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.robot.grasping.types.perception import PerceptionFrame

__all__ = ["RehearsalPerceptionSource", "rehearsal_intrinsics"]

#: A plausible 640x480 pinhole. Values are in the right ballpark for a D435 colour stream so the
#: back-projected object lands at a sane distance; they are not a calibration of anything.
_FX = _FY = 615.0
_CX, _CY = 320.0, 240.0
_W, _H = 640, 480


def rehearsal_intrinsics() -> np.ndarray:
    """The 3x3 camera matrix the rehearsal scene is rendered with."""
    return np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class _Mask:
    """Minimal ``SegmentationLike``: the protocol asks only for ``.mask``."""

    mask: np.ndarray


@dataclass
class RehearsalPerceptionSource:
    """One box on a plane, at a fixed distance. Deterministic by construction.

    Parameters
    ----------
    plane_mm
        Distance from the camera to the background plane.
    object_height_mm
        How far the box top stands above that plane.
    box_px
        Half-extent of the object mask, in pixels. The default is chosen so the object is ~46 mm
        wide at the default plane distance (2 * box_px * plane_mm / fx), which is comfortably
        inside a 2F-85's 85 mm opening. A rehearsal whose object cannot be grasped exercises only
        the failure path.
    """

    plane_mm: float = 800.0
    object_height_mm: float = 50.0
    box_px: int = 18
    #: Incremented per acquire() so a multi-run rehearsal is visibly not one cached frame.
    frames_served: int = field(default=0, init=False)

    def _object_mask(self) -> np.ndarray:
        """The box's silhouette. One definition, used by the frame and by the viewfinder alike."""
        mask = np.zeros((_H, _W), dtype=bool)
        r = self.box_px
        cy, cx = _H // 2, _W // 2
        mask[cy - r:cy + r, cx - r:cx + r] = True
        return mask

    #: This process drew these pixels. The console must not call them a camera and must not label
    #: the view "live": a rehearsal proves the wiring, and there is no camera behind it.
    colour_source_kind: str = field(default="synthetic", init=False)

    def peek_color(self) -> np.ndarray:
        """The rehearsal scene as BGR, for the viewfinder. No counter, no state, no side effect.

        It deliberately looks synthetic: a flat box on flat ground, no texture, no noise. An
        operator glancing at it must be able to tell in one look that there is no camera behind
        it.
        """
        bgr = np.zeros((_H, _W, 3), dtype=np.uint8)
        bgr[:] = (26, 22, 18)                                    # the ground plane, near-black
        bgr[self._object_mask()] = (60, 60, 200)                 # BGR: the same red the frame carries
        return bgr

    def acquire(self) -> PerceptionFrame:
        depth = np.full((_H, _W), self.plane_mm, dtype=np.float64)
        mask = self._object_mask()
        depth[mask] = self.plane_mm - self.object_height_mm  # the box stands toward the camera
        rgb = np.zeros((_H, _W, 3), dtype=np.uint8)
        rgb[mask] = (200, 60, 60)
        self.frames_served += 1
        return PerceptionFrame(
            depth_map=depth,
            intrinsics=rehearsal_intrinsics(),
            segmentations=(_Mask(mask),),
            rgb=rgb,
            timestamp=float(self.frames_served),
        )
