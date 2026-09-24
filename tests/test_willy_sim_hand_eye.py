"""Pure-math tests for the sim-native eye-in-hand calibration helpers (no Isaac required).

Covers the declared stations the on-box calibration runs: that each shipped stations file reads as the real cell's
``--fixed-poses`` reads one, and that an eye-in-hand station, fed back through the production
EyeInHandFrameResolver with the wrist mount the sim authors, reconstructs a camera that looks at the scene marker.
Nothing generates a sim station any more (the owner, 2026-09-24): the files were frozen from the generators this
repository used to carry, and these tests hold them to what those generators promised.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import Frame, Pose, Transform
from src.robot.execution.pose_provider import load_stations
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.willy_sim.calibration.hand_eye import transform_delta
from src.willy_sim.calibration.paths import STATIONS_DIR, declared_stations

#: The scene marker each model's eye-in-hand stations look at (robot.sim.yaml, and robot.ur3e.yaml's own).
_MARKERS = {"ur5e": (450.0, -150.0, 6.0), "ur3e": (150.0, -40.0, 6.0), "ur10e": (450.0, -150.0, 6.0)}


def _mount() -> Transform:
    """CAMERA->TOOL as the sim authors the wrist camera: 130 mm along the tool's +X, 112 mm behind the TCP, looking
    along the tool's +Z (``mount_offset_mm`` and ``mount_aim_mm`` through the ``willy`` tool frame)."""
    mat = np.eye(4)
    mat[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mat[:3, 3] = [130.0, 0.0, -112.0]
    return Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.TOOL)


class _FakeArm:
    """Minimal RobotArm stand-in: returns a fixed TCP pose in Frame.BASE."""

    def __init__(self, pose: Pose) -> None:
        self._pose = pose

    def get_tcp_pose(self) -> Pose:
        return self._pose


def test_every_sim_model_declares_both_station_files_and_each_reads_as_a_fixed_poses_file() -> None:
    """The three models the sim cell drives (robot.sim.robot_model: ur5e, ur3e, ur10e)."""
    for model in _MARKERS:
        for mounting in ("eth", "eih"):
            path = declared_stations(mounting, model)
            assert path.parent == STATIONS_DIR
            stations = load_stations(path)
            assert len(stations) >= 20, (path, len(stations))
            assert all(isinstance(station, Pose) and station.frame is Frame.BASE for station in stations)


def test_a_model_with_no_declared_file_is_refused_with_the_way_out() -> None:
    with pytest.raises(SystemExit, match="pass --stations PATH"):
        declared_stations("eih", "ur99")
    assert declared_stations("eih", "ur99", "mine.json").name == "mine.json"


def test_the_declared_eih_stations_look_at_the_marker() -> None:
    """The core contract: TCP station -> EyeInHandFrameResolver -> the camera's optical axis hits the marker."""
    resolver = EyeInHandFrameResolver(t_cam_to_tool=_mount())
    for model, marker_mm in _MARKERS.items():
        marker = np.asarray(marker_mm)
        worst_aim_deg = 0.0
        for station in load_stations(declared_stations("eih", model)):
            assert isinstance(station, Pose)
            base = np.asarray(resolver.camera_to_base_for_frame(None, arm=_FakeArm(station)).to_matrix())  # type: ignore[arg-type]
            cam_pos, optical_z = base[:3, 3], base[:3, 2]
            aim = (marker - cam_pos) / np.linalg.norm(marker - cam_pos)
            worst_aim_deg = max(worst_aim_deg, float(np.degrees(np.arccos(np.clip(optical_z @ aim, -1, 1)))))
            assert 230.0 < float(np.linalg.norm(marker - cam_pos)) < 280.0, (model, station.label)
        assert worst_aim_deg < 0.05, (model, worst_aim_deg)  # the rounding of the file only


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
