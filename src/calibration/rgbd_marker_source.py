"""An RGB-D calibration-target source for the hand-eye routine's ``marker_source`` seam.

``CalibrationRoutine`` (``robot/execution/calibration.py``) takes an injected
``MarkerPoseProvider = Callable[[], Optional[np.ndarray]]`` in place of its default stereo-rectified
path, and consumes it fail-closed. That default assumes a stereo rig, and ``FrameProvider`` raises for
an RGB-D one, so an eye-to-hand D435 needs a source of its own: grab a colour frame, pose the target in
it with an estimator, return ``T_cam_to_marker`` (4x4) or ``None``. The estimator is the generic mono
``ArucoPoseEstimator`` (detect plus ``SOLVEPNP_IPPE_SQUARE``) for one marker, or a
``CharucoPoseEstimator`` for a board (``src/calibration/targets.py``). The two sim runners fill the same
seam with an Isaac source.

What the judged frame showed stays on the source: ``last_observation`` is the
:class:`~src.calibration.targets.Observation` of the last call, whose ``why_not`` says why a pose was not
returned, and ``on_observation(bgr, observation, K, dist)`` is called with the exact frame that was judged.
The hook is display-only: whatever it does or raises, the returned pose is the estimator's.

The streamer is duck-typed (``grab`` / ``get_intrinsics`` / ``get_distortion``), so this module imports
no camera code and no ``pyrealsense2``; the bring-up runner injects the real streamer.

Honesty bucket (2): exercised offline against a rendered marker and a rendered board, never run against a
physical D435. The distortion path (decision D1) runs, but an aligned stream reports coefficients near
zero, so the residual is unconfirmed on a real bench.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from src.calibration.exceptions import CalibrationError
from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator
from src.calibration.targets import Observation

__all__ = ["ObservationHook", "RGBDArucoMarkerSource"]

logger = logging.getLogger(__name__)

#: Called with the judged frame (BGR, as grabbed and not copied), what the estimator made of it, and the camera
#: matrix and distortion it was posed with. A hook that keeps the frame must copy it: a RealSense colour frame
#: is a view into the device's buffer.
ObservationHook = Callable[[np.ndarray, Observation, np.ndarray, np.ndarray], None]


class RGBDArucoMarkerSource:
    """Callable ``MarkerPoseProvider``: one grab -> one ``T_cam_to_marker`` (4x4) or ``None``.

    Parameters
    ----------
    streamer
        Anything with ``grab() -> RGBDFrame`` (``.color`` BGR uint8), ``get_intrinsics() -> 3x3 K`` and
        ``get_distortion() -> dist | None``. In production the ``RealSenseRGBDStreamer``.
    marker_length_mm, dict_name, target_id
        One ArUco marker: its square edge in mm, its dictionary and its id. A wrong edge scales every
        sample uniformly, so the solve converges and is uniformly wrong. Measure the printed edge. Read only
        when no ``estimator`` is given.
    estimator
        What poses the target in a frame: anything with ``observe(bgr, K, dist) -> Observation``, such as
        ``estimator_for(target)`` returns for a marker or a ChArUco board. ``None`` builds an
        ``ArucoPoseEstimator`` from the three values above.
    intrinsics, distortion
        Decision D1: ``None`` uses the streamer's factory K/dist, and a bench-calibrated pair (for
        example from :func:`camera.setup.image_taking.intrinsics.load_intrinsics`) overrides them.
        Which of the two applied is recorded on ``intrinsics_source``.
    warmup_grabs
        Throwaway grabs so a real camera's auto-exposure settles before the pose read. A negative
        count clamps to zero.
    on_observation
        Called after every judged frame, see :data:`ObservationHook`. An exception in it is logged and
        swallowed, so a preview cannot abort a sweep or change a pose.
    """

    def __init__(
        self,
        *,
        streamer: Any,
        marker_length_mm: float = 50.0,
        dict_name: str = "DICT_5X5_100",
        target_id: int = 0,
        estimator: Any = None,
        intrinsics: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
        warmup_grabs: int = 3,
        on_observation: ObservationHook | None = None,
    ) -> None:
        self._streamer = streamer
        self._estimator = estimator if estimator is not None else ArucoPoseEstimator(
            marker_length_mm=marker_length_mm, dict_name=dict_name, target_id=int(target_id))
        self._k_override = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._dist_override = None if distortion is None else np.asarray(distortion, dtype=np.float64)
        self._warmup = max(0, int(warmup_grabs))
        self.intrinsics_source = "override" if intrinsics is not None else "factory"
        #: What the last call's frame showed; ``None`` before the first call and after a call that raised.
        self.last_observation: Observation | None = None
        self.on_observation: ObservationHook | None = on_observation

    @property
    def estimator(self) -> Any:
        return self._estimator

    def __call__(self) -> Optional[np.ndarray]:
        self.last_observation = None
        for _ in range(self._warmup):
            self._streamer.grab()
        frame = self._streamer.grab()
        bgr = np.ascontiguousarray(np.asarray(frame.color))

        if self._k_override is not None:
            k = self._k_override
        else:
            k = self._streamer.get_intrinsics()
            if k is None:
                raise RuntimeError(
                    "streamer.get_intrinsics() is None; open() the streamer first, or pass a "
                    "calibrated intrinsics= (the D1 override)."
                )
        if self._dist_override is not None:
            dist = self._dist_override
        else:
            dist = self._streamer.get_distortion()
            if dist is None:
                dist = np.zeros(5)  # device reported no coefficients: zeros, not a fabricated value
        K = np.asarray(k, dtype=np.float64)
        dist = np.asarray(dist, dtype=np.float64)

        try:
            observation = self._estimator.observe(bgr, K, dist)
        except CalibrationError as exc:
            # The routine skips a pose on RuntimeError and would abort the whole sweep on a CalibrationError, so
            # a frame the estimator refuses is this pose's failure, not the sweep's.
            raise RuntimeError(f"the target estimator refused this frame: {exc}") from exc
        self.last_observation = observation
        self._notify(bgr, observation, K, dist)
        pose = observation.T_cam_to_target
        return None if pose is None else np.asarray(pose, dtype=np.float64)

    def _notify(self, bgr: np.ndarray, observation: Observation, K: np.ndarray, dist: np.ndarray) -> None:
        hook = self.on_observation
        if hook is None:
            return
        try:
            hook(bgr, observation, K, dist)
        except Exception as exc:  # noqa: BLE001 (display-only: a hook must never change the pose or stop the sweep)
            logger.warning("on_observation raised %s: %s; the pose is unaffected", type(exc).__name__, exc)
