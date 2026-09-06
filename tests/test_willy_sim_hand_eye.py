"""Pure-math tests for the sim-native eye-in-hand calibration helpers (no Isaac required).

Covers the viewpoint geometry that the on-box calibration relies on: that a generated TCP
viewpoint, fed back through the production EyeInHandFrameResolver, reconstructs a camera that
looks exactly at the marker -- the algebraic contract between pose generation and the resolver.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.willy_sim.calibration.hand_eye import (
    generate_hemisphere_viewpoints,
    look_at_world_matrix,
    transform_delta,
)
from src.geometry import Frame, Pose, Transform
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver


def _t_cam_to_tool() -> Transform:
    """A realistic CAMERA->TOOL (mirrors the on-box oracle: [130,0,-112] mm + a 35 deg tilt)."""
    ang = np.radians(35.0)
    rot = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(ang), -np.sin(ang)], [0.0, np.sin(ang), np.cos(ang)]]
    )
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [130.0, 0.0, -112.0]
    return Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.TOOL)


class _FakeArm:
    """Minimal RobotArm stand-in: returns a fixed TCP pose in Frame.BASE."""

    def __init__(self, pose: Pose) -> None:
        self._pose = pose

    def get_tcp_pose(self) -> Pose:
        return self._pose


def test_look_at_world_matrix_points_optical_axis_at_target() -> None:
    cam = np.array([100.0, -50.0, 300.0])
    target = np.array([450.0, -150.0, 5.0])
    mat = look_at_world_matrix(cam, target)
    assert np.allclose(mat[:3, 3], cam)
    rot = mat[:3, :3]
    # right-handed, orthonormal
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)
    # CV optical +Z (third column) points from the camera toward the target
    optical_z = rot[:, 2]
    aim = (target - cam) / np.linalg.norm(target - cam)
    assert np.allclose(optical_z, aim, atol=1e-9)


def test_look_at_world_matrix_handles_up_parallel_to_axis() -> None:
    # camera directly above the target -> forward == up_hint default (0,0,1); must not blow up.
    cam = np.array([450.0, -150.0, 305.0])
    target = np.array([450.0, -150.0, 5.0])
    mat = look_at_world_matrix(cam, target)  # default up_hint (0,0,1) is parallel to forward
    rot = mat[:3, :3]
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)
    assert np.allclose(rot[:, 2], np.array([0.0, 0.0, -1.0]), atol=1e-9)  # looks straight down


def test_generate_hemisphere_viewpoints_count_default() -> None:
    vps = generate_hemisphere_viewpoints([450.0, -150.0, 5.0], _t_cam_to_tool())
    # defaults: 2 radii * 2 elevations * 6 azimuths
    assert len(vps) == 24
    for vp in vps:
        assert vp.tcp_pose.frame is Frame.BASE


def test_generated_viewpoints_resolve_to_camera_looking_at_marker() -> None:
    """The core contract: TCP target -> EyeInHandFrameResolver -> camera optical axis hits marker."""
    marker = np.array([450.0, -150.0, 5.0])
    t_cam_to_tool = _t_cam_to_tool()
    resolver = EyeInHandFrameResolver(t_cam_to_tool=t_cam_to_tool)
    vps = generate_hemisphere_viewpoints(marker, t_cam_to_tool)
    worst_aim_deg = 0.0
    worst_pos_mm = 0.0
    for vp in vps:
        arm = _FakeArm(vp.tcp_pose)
        t_cam_to_base = resolver.camera_to_base_for_frame(None, arm=arm)  # type: ignore[arg-type]
        base = np.asarray(t_cam_to_base.to_matrix())
        cam_pos, optical_z = base[:3, 3], base[:3, 2]
        aim = (marker - cam_pos) / np.linalg.norm(marker - cam_pos)
        worst_aim_deg = max(worst_aim_deg, float(np.degrees(np.arccos(np.clip(optical_z @ aim, -1, 1)))))
        worst_pos_mm = max(worst_pos_mm, float(np.linalg.norm(cam_pos - vp.cam_pos_mm)))
    assert worst_aim_deg < 1e-3   # optical axis points at the marker (arccos numerical noise only)
    assert worst_pos_mm < 1e-6    # camera position reconstructs exactly


def test_aruco_marker_pose_source_detects_and_solves() -> None:
    """ArucoMarkerPoseSource: synthetic frontal marker -> detect + PnP -> CAMERA->MARKER pose."""
    cv2 = pytest.importorskip("cv2")
    from src.willy_sim.calibration.hand_eye import ArucoMarkerPoseSource

    length_mm, fx, z_mm = 40.0, 800.0, 300.0
    side = int(round(length_mm * fx / z_mm))  # apparent marker size (px) of a frontal marker at z
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, side)
    canvas = np.full((480, 640), 255, dtype=np.uint8)  # white -> quiet zone
    y0, x0 = (480 - side) // 2, (640 - side) // 2
    canvas[y0:y0 + side, x0:x0 + side] = marker
    rgb = np.stack([canvas] * 3, axis=-1)
    k = np.array([[fx, 0.0, 320.0], [0.0, fx, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    class _Cam:
        def get_current_frame(self):
            return {"rgb": rgb}

        def get_intrinsics_matrix(self):
            return k

    t = ArucoMarkerPoseSource(marker_id=0, marker_length_mm=length_mm).marker_in_camera(_Cam())
    assert t is not None
    assert t.from_frame is Frame.CAMERA and t.to_frame is Frame.MARKER
    tvec = np.asarray(t.to_matrix())[:3, 3]
    assert tvec[2] == pytest.approx(z_mm, abs=15.0)  # recovered depth from apparent size
    assert abs(tvec[0]) < 8.0 and abs(tvec[1]) < 8.0  # marker is centred


def test_aruco_marker_pose_source_none_when_absent() -> None:
    pytest.importorskip("cv2")
    from src.willy_sim.calibration.hand_eye import ArucoMarkerPoseSource

    rgb = np.full((480, 640, 3), 255, dtype=np.uint8)  # blank: no marker
    k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    class _Cam:
        def get_current_frame(self):
            return {"rgb": rgb}

        def get_intrinsics_matrix(self):
            return k

    assert ArucoMarkerPoseSource(marker_id=0, marker_length_mm=40.0).marker_in_camera(_Cam()) is None


def test_transform_delta_identity_and_known() -> None:
    a = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
    assert transform_delta(a, a) == pytest.approx((0.0, 0.0), abs=1e-9)
    mat = np.eye(4)
    mat[:3, 3] = [3.0, 4.0, 0.0]  # 5 mm translation
    b = Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
    d_t, d_r = transform_delta(a, b)
    assert d_t == pytest.approx(5.0, abs=1e-9)
    assert d_r == pytest.approx(0.0, abs=1e-9)
