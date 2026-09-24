"""Sim-native hand-eye calibration for the Isaac validation platform.

Drives the real eye-hand calibrators (``EyeInHandCalibrator`` and ``EyeToHandCalibrator`` in
``src.calibration.eye_hand``) inside Isaac: the arm is moved through declared, diverse, reachable
viewpoints, a ``(T_base_to_tool, T_cam_to_marker)`` sample is recorded per view, AX=XB is solved,
and the result is validated against the empirical ground-truth oracle
(:func:`src.willy_sim.scene.camera_to_tcp_ground_truth`).

The pieces are separated so they swap freely:

* :class:`MarkerPoseSource` says how ``T_cam_to_marker`` is obtained per view. Two
  implementations: :class:`GroundTruthMarkerPoseSource` reads the marker prim's true pose and is
  deterministic, which exercises the collection and AX=XB plumbing, and
  :class:`ArucoMarkerPoseSource` solves the pose from a rendered board, which is genuine
  perception. Both yield the same
  :class:`~src.calibration.eye_hand.dataset.EyeHandSample`.
* :func:`~src.willy_sim.calibration.paths.declared_stations` names the stations file a sim sweep runs. The stations are declared, one
  file per mounting and robot model under ``stations/``, never generated: ``eih_<model>.json``
  holds TCP poses whose wrist camera looks at the scene marker from a ring of distances,
  elevations and azimuths (rotating the wrist about at least 2 axes, as the AX=XB observability
  check requires), and ``eth_<model>.json`` tool-down poses tilted up to 30 degrees in a column
  under the overhead camera. They were frozen from the generators this repository used to carry,
  the eye-in-hand ring through the wrist mount the sim authors in place of the runtime oracle;
  an Isaac run has not been repeated on them since.

The collection and AX=XB loop itself is not here: it is the vendor-neutral
:class:`src.robot.execution.CalibrationRoutine`, which the sim runner drives through its
injected ``marker_source`` seam (one of the :class:`MarkerPoseSource` implementations below) and
``run_with_poses`` over the declared stations.

``isaacsim`` is not imported at module top: the arm, camera and session are injected, and marker
prims are read through a lazily-imported wrapper. The module therefore imports on a machine with
no Isaac installed, and the pure-math helpers run there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from src.geometry import Frame, Transform
from src.geometry.matrix import invert_homogeneous
from src.utility.log_cfg import create_logger

from src.willy_sim.constants import HAND_EYE_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.scene import camera_to_base_ground_truth

#: A calibration run is a long sweep whose interesting content is which views the marker source
#: refused and why: AX=XB is solved from the views that survived, so a residual over few accepted
#: views is a different claim from the same residual over many. The collector counts what it kept;
#: only the source knows why it dropped the rest.
_LOG = create_logger("SimHandEye", log_file=HAND_EYE_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

__all__ = [
    "MarkerPoseSource",
    "GroundTruthMarkerPoseSource",
    "ArucoMarkerPoseSource",
    "transform_delta",
]


# ---------------------------------------------------------------------------
# Marker pose sources (the swappable perception seam)
# ---------------------------------------------------------------------------


@runtime_checkable
class MarkerPoseSource(Protocol):
    """Obtain ``T_cam_to_marker`` (``Transform(CAMERA -> MARKER)``) for the current camera frame.

    Returns ``None`` when the marker is not detected (the collector skips that view). The camera
    is injected; implementations must not move the arm.
    """

    def marker_in_camera(self, camera: object) -> Optional[Transform]: ...


@dataclass(frozen=True, slots=True)
class GroundTruthMarkerPoseSource:
    """Deterministic ``T_cam_to_marker`` from the marker prim's true world pose.

    Computes ``cam_T_marker = inv(base_T_cam) @ base_T_marker``, where ``base_T_cam`` is the
    empirical ground-truth oracle (:func:`scene.camera_to_base_ground_truth`, world == base) and
    ``base_T_marker`` is the marker prim's world pose. It exercises the collection, AX=XB and
    save/load machinery deterministically, with no flaky detection. Because the inputs derive from
    the oracle, the recovered ``T_cam_to_tool`` matches the oracle near-exactly: this validates the
    plumbing, and :class:`ArucoMarkerPoseSource` provides the perception-independent check.
    """

    marker_prim_path: str

    def marker_in_camera(self, camera: object) -> Optional[Transform]:
        from isaacsim.core.prims import SingleXFormPrim  # type: ignore[import-not-found]

        t_cam_to_base, _ = camera_to_base_ground_truth(camera)
        pos_m, quat_wxyz = SingleXFormPrim(self.marker_prim_path).get_world_pose()
        base_t_marker = _pose_wxyz_to_matrix_mm(np.asarray(pos_m), np.asarray(quat_wxyz))
        cam_t_marker = invert_homogeneous(np.asarray(t_cam_to_base.to_matrix())) @ base_t_marker
        # Logged at debug: one line per view, and a sweep is dozens. This source cannot fail to
        # detect, so at info it would say nothing the viewpoint count does not already say.
        _LOG.debug(
            "ground-truth marker %s at %s mm in the camera frame",
            self.marker_prim_path, np.round(cam_t_marker[:3, 3], 1).tolist(),
        )
        return Transform.from_matrix(cam_t_marker, from_frame=Frame.CAMERA, to_frame=Frame.MARKER)


@dataclass(frozen=True, slots=True)
class ArucoMarkerPoseSource:
    """Perception-based ``T_cam_to_marker`` from rendered-ArUco detection and PnP, with no oracle.

    Reads the camera RGB, detects the ArUco marker (``cv2.aruco``), and solves the marker pose in
    the CAMERA frame with ``cv2.solvePnP`` over the known square corners. cv2's camera frame is the
    CV optical convention, the same frame the empirical oracle fits, so the recovered
    ``T_cam_to_tool`` is directly comparable. Returns ``None`` when the marker is not detected,
    occluded or out of frame, and the collector then skips that view. The ``cv2`` import is lazy.
    """

    marker_id: int
    marker_length_mm: float
    dict_name: str = "DICT_4X4_50"

    def marker_in_camera(self, camera: object) -> Optional[Transform]:
        import cv2  # type: ignore[import-not-found]

        frame = camera.get_current_frame()  # type: ignore[attr-defined]
        rgba = frame.get("rgb") if isinstance(frame, dict) else None
        if rgba is None:
            # Every refusal below returns None and the collector then silently skips the view. Four
            # causes collapse into one "fewer samples than viewpoints", and telling them apart is
            # what separates a bad rig from a bad marker.
            _LOG.warning("no RGB in the camera frame: skipping this calibration view")
            return None
        rgb = np.asarray(rgba)[..., :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel corner accuracy
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name)), params
        )
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            # Expected at a station that sees the marker at a grazing angle or not at all, so this is
            # logged at debug rather than warning: the routine rejects that station as marker_not_found
            # and says so in its own per-pose line.
            _LOG.debug("no ArUco marker detected in this view (%s)", self.dict_name)
            return None
        id_list = ids.ravel().tolist()
        if self.marker_id not in id_list:
            # A different id is not the same non-event as seeing none: it means the board in the
            # scene is not the board this source was configured for.
            _LOG.warning(
                "ArUco ids %s detected but the configured id %d is not among them: skipping view",
                id_list, self.marker_id,
            )
            return None
        idx = id_list.index(self.marker_id)
        half = float(self.marker_length_mm) / 2.0
        # cv2.aruco corner order is TL, TR, BR, BL; the object points match it (marker +Z out of plane).
        obj = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float32,
        )
        k = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)  # type: ignore[attr-defined]
        # IPPE_SQUARE is the planar-square-fiducial solver, more accurate here than the default.
        ok, rvec, tvec = cv2.solvePnP(
            obj, corners[idx].reshape(4, 2).astype(np.float32), k, np.zeros(5),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            _LOG.warning("solvePnP (IPPE_SQUARE) did not converge for marker %d: skipping view",
                         self.marker_id)
            return None
        rot, _ = cv2.Rodrigues(rvec)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        _LOG.debug(
            "ArUco %d solved at %s mm (range %.1f mm) in the camera frame",
            self.marker_id, np.round(mat[:3, 3], 1).tolist(), float(np.linalg.norm(mat[:3, 3])),
        )
        return Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.MARKER)


def _pose_wxyz_to_matrix_mm(pos_m: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert an Isaac world pose (metres, WXYZ) to a 4x4 homogeneous matrix in mm."""
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(pos_m, dtype=np.float64).reshape(3) * 1000.0
    return mat


# ---------------------------------------------------------------------------
# Validation helper (sim-only: compare a calibrated transform to the GT oracle)
# ---------------------------------------------------------------------------


def transform_delta(a: Transform, b: Transform) -> tuple[float, float]:
    """(translation mm, rotation deg) between two Transforms via their matrices."""
    ma, mb = np.asarray(a.to_matrix()), np.asarray(b.to_matrix())
    d_t = float(np.linalg.norm(ma[:3, 3] - mb[:3, 3]))
    r = ma[:3, :3].T @ mb[:3, :3]
    cos = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return d_t, float(np.degrees(np.arccos(cos)))
