"""ArUco marker pose estimation in rectified left frames."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.targets import (
    Observation,
    canonical_aruco_dict_name,
    checked_camera,
    ids_text,
    other_dictionaries_clause,
    reprojection_rms_px,
    rvec_tvec_to_matrix,
)

logger = logging.getLogger(__name__)

__all__ = ["ArucoPoseEstimator", "resolve_aruco_dictionary"]


def resolve_aruco_dictionary(name: str):
    """Resolve a dictionary name to a cv2 ArUco dictionary; case does not matter and the ``DICT_`` prefix is optional."""
    try:
        canonical = canonical_aruco_dict_name(name)
    except ValueError as exc:
        raise StereoCalibrationError(f"Unknown ArUco dictionary: {name}") from exc
    return cv.aruco.getPredefinedDictionary(getattr(cv.aruco, canonical))


def _rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return rvec_tvec_to_matrix(rvec, tvec)


class ArucoPoseEstimator:
    """Detects ArUco markers and returns ``T_cam_to_marker`` 4x4 transforms.

    The translations carry the unit of ``marker_length_mm``, so millimetres. ``target_id`` is the marker
    :meth:`observe` poses when it is not given one; :meth:`estimate` takes its own.
    """

    kind = "aruco"

    def __init__(
        self,
        marker_length_mm: float = 50.0,
        dict_name: str = "DICT_5X5_100",
        target_id: Optional[int] = None,
    ) -> None:
        if not hasattr(cv, "aruco"):
            raise StereoCalibrationError(
                "cv2.aruco not found. Please install opencv-contrib-python."
            )

        self.marker_length = float(marker_length_mm)
        if self.marker_length <= 0.0:
            raise StereoCalibrationError("marker_length_mm must be > 0")
        self.aruco_dict = resolve_aruco_dictionary(dict_name)
        self.dict_name = canonical_aruco_dict_name(dict_name)
        self.target_id = None if target_id is None else int(target_id)
        self.aruco_params = cv.aruco.DetectorParameters()
        # Sub-pixel corner refinement improves solvePnP pose accuracy on fiducials.
        self.aruco_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Grayscale buffer reused across frames; _to_gray reallocates it only
        # when the frame size changes.
        self._gray: Optional[np.ndarray] = None

    def _object_points(self) -> np.ndarray:
        half = self.marker_length / 2.0
        # cv2.aruco returns corners TL, TR, BR, BL; obj follows that order, marker +Z out of plane.
        return np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )

    def estimate(
        self,
        rect_left_bgr: np.ndarray,
        K_rect: Optional[np.ndarray],
        dist_rect: np.ndarray,
        target_id: Optional[int] = None,
    ):
        """Detect markers in a rectified left frame and pose each one.

        ``K_rect`` is the rectified 3x3 intrinsics and is required; ``dist_rect``
        is the matching distortion, zeros for input that is already rectified.

        Returns:
            {marker id: 4x4 ``T_cam_to_marker``} when ``target_id`` is None,
            otherwise that one marker's matrix, or ``None`` if it was not
            detected.
        """
        if K_rect is None:
            raise StereoCalibrationError("K_rect is not available (set rectified intrinsics).")
        K = np.asarray(K_rect, dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise StereoCalibrationError("K_rect must be a finite 3x3 matrix")
        dist = np.asarray(dist_rect, dtype=np.float64).reshape(-1)

        gray = self._to_gray(rect_left_bgr)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            logger.debug("ArUco: no markers detected.")
            return {} if target_id is None else None

        # OpenCV >= 4.7 has no cv.aruco.estimatePoseSingleMarkers (4.13 is pinned), so each
        # marker is posed by cv.solvePnP over its square corners, as willy_sim/hand_eye.py does.
        obj = self._object_points()
        result: Dict[int, np.ndarray] = {}
        for i, mid in enumerate(ids.flatten()):
            img_pts = np.asarray(corners[i], dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv.solvePnP(obj, img_pts, K, dist, flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                logger.debug("ArUco: solvePnP failed for marker %d.", int(mid))
                continue
            result[int(mid)] = _rvec_tvec_to_matrix(rvec, tvec)

        if target_id is None:
            return result
        return result.get(int(target_id))

    def observe(
        self,
        bgr: np.ndarray,
        K: Optional[np.ndarray],
        dist: Optional[np.ndarray],
        target_id: Optional[int] = None,
    ) -> Observation:
        """Pose one marker in one frame, or say why not, with what the frame showed.

        The marker is ``target_id``, else the one this estimator was built for. The pose is the one
        :meth:`estimate` returns for it. When the marker is posed and other markers of the dictionary are in view
        too, ``hint`` asks whether the target is a ChArUco board: a board's marker 0 posed as a single marker is
        scaled by square over marker edge, and nothing else in the sweep would show it. A frame of the wrong shape,
        a camera matrix no pose can be solved with and an OpenCV error raise ``StereoCalibrationError``.
        """
        wanted = self.target_id if target_id is None else int(target_id)
        if wanted is None:
            raise ValueError("observe() poses one marker: pass target_id, or build the estimator with one")
        matrix, coefficients = checked_camera(K, dist)
        frame = np.asarray(bgr)
        shape = tuple(frame.shape)
        try:
            gray = self._to_gray(frame)
            corners, ids, rejected = self._detector.detectMarkers(gray)
        except cv.error as exc:
            raise StereoCalibrationError(f"ArUco detection failed: {exc}") from exc
        found_ids = () if ids is None else tuple(int(i) for i in np.asarray(ids).ravel())
        quads = tuple(np.asarray(c, dtype=np.float64).reshape(4, 2) for c in corners) if found_ids else ()
        seen: dict[str, Any] = dict(kind=self.kind, marker_ids=found_ids, marker_corners=quads,
                                    frame_shape=shape)
        if not found_ids:
            others = other_dictionaries_clause(gray, self.dict_name)
            why = f"no {self.dict_name} marker in view{others}"
            if not others and rejected is not None and len(rejected):
                why += (f"; {len(rejected)} square candidates did not decode (glare, blur, a marker too small in "
                        "the frame, or one of a dictionary this check does not scan)")
            return Observation(**seen, why_not=why)
        distinct = sorted(set(found_ids))
        if wanted not in found_ids:
            why = f"marker {wanted} not in view; {self.dict_name} ids in view: {ids_text(distinct)}"
            if len(distinct) > 1:
                why += "; several markers in view: is this a ChArUco board? Then name it as a charuco target"
            return Observation(**seen, why_not=why)
        img = quads[found_ids.index(wanted)]
        obj = self._object_points()
        try:
            ok, rvec, tvec = cv.solvePnP(obj, img, matrix, coefficients, flags=cv.SOLVEPNP_IPPE_SQUARE)
        except cv.error as exc:
            raise StereoCalibrationError(f"solvePnP failed on marker {wanted}: {exc}") from exc
        if not ok:
            return Observation(**seen, why_not=f"solvePnP found no pose for marker {wanted}", points_px=img)
        hint = ""
        if len(distinct) > 1:
            hint = (f"{len(distinct)} {self.dict_name} markers visible: is this a ChArUco board? A single-marker "
                    f"target poses marker {wanted} at its {self.marker_length:.1f} mm edge, and on a board the "
                    "marker edge is not the square, so every sample would be scaled")
        return Observation(
            **seen, T_cam_to_target=_rvec_tvec_to_matrix(rvec, tvec), hint=hint, points_px=img,
            reprojection_px=reprojection_rms_px(obj, img, rvec, tvec, matrix, coefficients),
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3), tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
        )

    def _to_gray(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 2:
            return bgr
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise StereoCalibrationError(
                f"rect_left_bgr must be grayscale or BGR, got shape {bgr.shape}"
            )
        h, w = bgr.shape[:2]
        if self._gray is None or self._gray.shape != (h, w):
            self._gray = np.empty((h, w), dtype=np.uint8)
        cv.cvtColor(bgr, cv.COLOR_BGR2GRAY, dst=self._gray)
        return self._gray
