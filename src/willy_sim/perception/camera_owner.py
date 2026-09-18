"""An Isaac camera as the camera owner the live planner world takes.

``run_m2_pick`` is the Isaac reference, and it plans against a live world from the overhead camera.
The world is built through the one builder a real cell uses (``execution.camera_world_wiring``
through ``Robot.from_parts``) and the one depth source (``RigDepthSource``), so this module holds
only what makes an Isaac camera look like an opened rig: ``rig_id``, ``rig``, ``handle()`` and
``calibration()``.

Two facts shape it:

* The camera is placed by the extrinsic fitted from rendered depth
  (``scene.camera_to_base_ground_truth``), not by the scene's hand-built matrix, which is exact only
  on the optical axis. The grasp keeps the hand-built one; the angle between the two is kept here
  and printed at boot.
* Isaac's annotators go stale without ``step(render=True)`` and ``app.update()``, so every grab
  pumps both, and the shutter time is read before the pump, as a real owner reads it before its
  grab.

Isaac imports stay inside methods, so the module imports off the box.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["IsaacCameraOwner"]

#: How many render pumps a grab makes before it reads depth, which is what flushes the annotator.
DEFAULT_PUMPS = 2
#: The near clip a depth reading needs: Isaac's 1.0 m default hides the table from the overhead
#: camera.
NEAR_CLIP_M = 0.05


@dataclass(frozen=True)
class _Extrinsics:
    mounting_mode: str
    artifact_path: str


@dataclass(frozen=True)
class _Rig:
    """The rig facts the world plan reads for an Isaac camera.

    Enabled, RGB-D, calibrated eye_to_hand, and no body.
    """

    rig_id: str
    enabled: bool
    source: str
    extrinsics: _Extrinsics
    body: None = None


@dataclass(frozen=True)
class _Frame:
    depth: np.ndarray
    captured_at_s: float


class _Handle:
    """What ``RigDepthSource`` reads.

    ``grab()`` gives depth in millimetres with its shutter time, and ``get_intrinsics()`` the
    intrinsics.
    """

    def __init__(self, camera: Any, session: Any, *, rig_id: str, pumps: int, clock: Any = time.time) -> None:
        self.rig_id = rig_id
        self._camera = camera
        self._session = session
        self._pumps = int(pumps)
        self._clock = clock

    def grab(self) -> _Frame:
        captured = float(self._clock())
        app = getattr(self._session, "app", None)
        for _ in range(self._pumps):
            self._session.step(render=True)
            if app is not None:
                app.update()
        depth_m = np.asarray(self._camera.get_depth(), dtype=np.float64)
        depth_mm = np.where(np.isfinite(depth_m), depth_m * 1000.0, 0.0)
        return _Frame(depth=depth_mm, captured_at_s=captured)

    def get_intrinsics(self) -> np.ndarray:
        return np.asarray(self._camera.get_intrinsics_matrix(), dtype=np.float64)


@dataclass(frozen=True)
class _Calibration:
    mounting_mode: str
    transform: Any

    def camera_to_base(self) -> Any:
        return self.transform


class IsaacCameraOwner:
    """One Isaac camera, opened by the scene, answering as an eye_to_hand rig calibrated by a fit."""

    def __init__(self, camera: Any, session: Any, *, rig_id: str, camera_to_base: Any, pumps: int = DEFAULT_PUMPS,
                 fit_angle_deg: float | None = None, clock: Any = time.time) -> None:
        self.rig_id = rig_id
        self.rig = self.rig_for(rig_id)
        self.fit_angle_deg = fit_angle_deg
        self._handle = _Handle(camera, session, rig_id=rig_id, pumps=pumps, clock=clock)
        self._calibration = _Calibration(mounting_mode="eye_to_hand", transform=camera_to_base)

    @staticmethod
    def rig_for(rig_id: str) -> _Rig:
        """The rig facts of an Isaac camera, config pure, for the world plan read before the boot."""
        return _Rig(rig_id=rig_id, enabled=True, source="rgbd",
                    extrinsics=_Extrinsics(mounting_mode="eye_to_hand", artifact_path=f"isaac:{rig_id} (fitted)"))

    @classmethod
    def from_scene(cls, session: Any, handles: Any, *, rig_id: str, pumps: int = DEFAULT_PUMPS,
                   warmup_steps: int = 25, attempts: int = 6) -> "IsaacCameraOwner":
        """The overhead camera of a booted scene, its depth annotator attached and its extrinsic fitted.

        Refuses a camera whose annotator or clip cannot be set, and one whose depth never becomes
        readable within ``attempts`` rounds of pumps: a world placed without a fit would misplace
        every point off the optical axis.
        """
        from src.willy_sim.scene import camera_to_base_ground_truth

        camera = handles.camera
        try:
            camera.add_distance_to_image_plane_to_frame()
            camera.set_clipping_range(NEAR_CLIP_M, 1.0e6)
        except Exception as exc:  # noqa: BLE001 (a camera that cannot give depth cannot feed a world)
            raise ValueError(f"the Isaac camera {rig_id!r} cannot give depth to a live world: {exc}") from exc
        app = getattr(session, "app", None)
        fitted = None
        last: Exception | None = None
        for attempt in range(int(attempts)):
            for _ in range(int(warmup_steps) if attempt == 0 else 12):
                session.step(render=True)
                if app is not None:
                    app.update()
            try:
                fitted, _ = camera_to_base_ground_truth(camera)
                break
            except Exception as exc:  # noqa: BLE001 (depth not ready yet: pump and try again)
                last = exc
        if fitted is None:
            raise ValueError(f"the Isaac camera {rig_id!r} gave no depth to fit its extrinsic from: {last}")
        return cls(camera, session, rig_id=rig_id, camera_to_base=fitted, pumps=pumps,
                   fit_angle_deg=_angle_deg(fitted, getattr(handles, "camera_to_base", None)))

    def handle(self) -> _Handle:
        return self._handle

    def calibration(self) -> _Calibration:
        return self._calibration


def _angle_deg(fitted: Any, scene: Any) -> float | None:
    """The rotation between the fitted CAMERA to BASE and the scene's hand-built one, in degrees.

    ``None`` when the scene holds no hand-built one.
    """
    if scene is None:
        return None
    a = np.asarray(fitted.to_matrix(), dtype=np.float64)[:3, :3]
    b = np.asarray(scene.to_matrix(), dtype=np.float64)[:3, :3]
    cos = (float(np.trace(a @ b.T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
