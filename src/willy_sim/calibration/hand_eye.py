"""Sim-native hand-eye calibration for the Isaac validation platform.

Drives the real eye-hand calibrators (``EyeInHandCalibrator`` and ``EyeToHandCalibrator`` in
``src.calibration.eye_hand``) inside Isaac: the arm is moved through diverse, reachable
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
* :func:`generate_hemisphere_viewpoints` produces diverse poses that look at the marker from many
  angles, rotating the wrist about at least 2 axes as the AX=XB observability check requires,
  expressed as TCP targets through the camera-on-tool transform, oracle or rough.

The collection and AX=XB loop itself is not here: it is the vendor-neutral
:class:`src.robot.execution.CalibrationRoutine`, which the sim runner drives through its
injected ``marker_source`` seam (one of the :class:`MarkerPoseSource` implementations below) and
``run_with_poses`` (the look-at viewpoints from :func:`generate_hemisphere_viewpoints`).

``isaacsim`` is not imported at module top: the arm, camera and session are injected, and marker
prims are read through a lazily-imported wrapper. The module therefore imports on a machine with
no Isaac installed, and the pure-math helpers run there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from src.geometry import Frame, Pose, Transform
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
    "CalibrationViewpoint",
    "look_at_world_matrix",
    "generate_hemisphere_viewpoints",
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
            # Expected on a hemisphere sweep at grazing angles or out of frame, so this is logged at
            # debug rather than warning: the sweep over-generates views by design.
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
# Viewpoint generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationViewpoint:
    """A planned calibration view: the TCP target plus where the camera should look."""

    tcp_pose: Pose            # Frame.BASE TCP target to move the arm to
    cam_pos_mm: np.ndarray    # intended camera position in base (for diagnostics)


def look_at_world_matrix(
    cam_pos_mm: np.ndarray,
    target_mm: np.ndarray,
    up_hint: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """4x4 ``CAMERA -> BASE`` pose whose optical +Z (forward) points at ``target_mm``.

    Forward +Z into the scene is the property this function exists for and the only one its caller
    uses: :func:`generate_hemisphere_viewpoints` aims the optical axis at a marker, and the marker
    is then found by detection, not by projection.

    Do not use this to project world points to pixels. The X and Y axes here are rolled 180 deg
    from the image-right and image-down convention, so a projection through this matrix lands every
    point at ``(W - u, H - v)``: a perfectly consistent, perfectly wrong answer that looks like a
    permuted label set. The roll is harmless for aiming, since any roll still points the axis at
    the target, and it is left exactly as it is because it feeds the wrist poses the eye-in-hand
    calibration depends on. Project with the CV optical convention instead: +Z forward, +X
    image-right, +Y image-down.

    The returned matrix's columns are the camera axes in base; its translation is ``cam_pos_mm``.
    """
    cam = np.asarray(cam_pos_mm, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_mm, dtype=np.float64).reshape(3)
    up = np.asarray(up_hint, dtype=np.float64).reshape(3)
    z = tgt - cam  # CV optical +Z points toward the target
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        raise ValueError("look_at_world_matrix: camera coincides with target")
    z = z / nz
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:  # up parallel to z, so pick another up axis
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.column_stack([x, y, z])
    mat[:3, 3] = cam
    return mat


def generate_hemisphere_viewpoints(
    marker_pos_mm: Sequence[float] | np.ndarray,
    t_cam_to_tool: Transform,
    *,
    radii_mm: Sequence[float] = (235.0, 275.0),
    elevations_deg: Sequence[float] = (16.0, 30.0),
    azimuths_deg: Sequence[float] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
    up_hint: Sequence[float] = (1.0, 0.0, 0.0),
) -> list[CalibrationViewpoint]:
    """Diverse TCP viewpoints whose wrist camera looks at the marker from many angles.

    For each (radius, elevation, azimuth) the camera is placed on a hemisphere above the marker,
    its optical axis is aimed at the marker (:func:`look_at_world_matrix`), and the TCP target is
    backed out through ``base_T_tool = base_T_cam @ inv(T_cam_to_tool)``, the inverse of the
    EyeInHandFrameResolver composition. Varying azimuth and elevation rotates the wrist about at
    least 2 independent axes, which the AX=XB observability check (``_assert_ready``) requires.
    Reachability is filtered later by the collector, which attempts the move. ``elevation`` is
    measured from the vertical, so 0 is straight above.
    """
    marker = np.asarray(marker_pos_mm, dtype=np.float64).reshape(3)
    # T_cam_to_tool.to_matrix() = tool_T_cam (maps cam->tool); its inverse maps tool->cam, so
    # base_T_tool = base_T_cam @ inv(tool_T_cam), the inverse of the resolver's compose.
    cam_t_tool = invert_homogeneous(np.asarray(t_cam_to_tool.to_matrix(), dtype=np.float64))
    views: list[CalibrationViewpoint] = []
    for r in radii_mm:
        for el in elevations_deg:
            el_rad = np.radians(el)
            for az in azimuths_deg:
                az_rad = np.radians(az)
                direction = np.array(
                    [
                        np.sin(el_rad) * np.cos(az_rad),
                        np.sin(el_rad) * np.sin(az_rad),
                        np.cos(el_rad),
                    ],
                    dtype=np.float64,
                )
                cam_pos = marker + r * direction
                base_t_cam = look_at_world_matrix(cam_pos, marker, up_hint)
                base_t_tool = base_t_cam @ cam_t_tool
                pose = Pose.from_matrix(base_t_tool, frame=Frame.BASE, label=f"cal-r{int(r)}-el{int(el)}-az{int(az)}")
                views.append(CalibrationViewpoint(tcp_pose=pose, cam_pos_mm=cam_pos))
    # The planned count, against which the collector's accepted count is read. AX=XB needs rotation
    # about >= 2 independent axes, so the elevation by azimuth spread is the observability claim.
    _LOG.info(
        "generated %d calibration viewpoint(s) around marker %s: radii=%s mm elevations=%s deg "
        "azimuths=%s deg",
        len(views), np.round(marker, 1).tolist(), list(radii_mm), list(elevations_deg),
        list(azimuths_deg),
    )
    return views


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
