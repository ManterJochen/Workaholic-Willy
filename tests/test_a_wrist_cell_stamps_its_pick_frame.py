"""A real wrist cell's pick frame carries the tool pose read at its shutter.

`EyeInHandFrameResolver` places a wrist camera's grasp by the TCP. With an unstamped frame it reads the arm when
the grasp is resolved, and every millimetre the tool travelled since the shutter goes into the grasp.
`build_real_cell` hands the arm's reader to every pick source whose resolver composes the TCP, the primary and each
fused camera, and to no other, and refuses a wrist rig whose calibration was not solved against the declared flange
to TCP.

Never run on hardware. What is proven is the wiring, against a fake streamer, a dummy arm and a counting arm.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose, Transform
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.autonomous_grasp import cells
from src.robot.execution.autonomous_grasp.service import AutonomousGraspService
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.perception import RealSenseVisionPerceptionSource

_K = np.array([[600.0, 0.0, 4.0], [0.0, 600.0, 4.0], [0.0, 0.0, 1.0]])
_POSE = Pose(position_mm=np.array([420.0, -35.0, 310.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
             frame=Frame.BASE)


def _wrist_resolver() -> EyeInHandFrameResolver:
    return EyeInHandFrameResolver(t_cam_to_tool=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL))


def _calibration(*, record: "object" = "declared") -> object:
    """A wrist rig's calibration: CAMERA to TOOL, its shutter tolerances and the flange to TCP it was solved against."""
    from src.calibration.rig_calibration import RigCalibration
    from src.calibration.serialization import FlangeToTcp
    from src.contracts import UNSET

    if record == "declared":
        flange = FlangeToTcp.from_matrix("willy", np.eye(4))
    elif record is None:
        flange = UNSET
    else:
        flange = record
    return RigCalibration(rig_id="wrist", mounting_mode="eye_in_hand", artifact_path="wrist.json",
                          transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL),
                          shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5, flange_to_tcp=flange)


class _Streamer:
    """One RGB-D frame on every grab, shaped like a rig handle, with the camera that owns it."""

    def __init__(self, calibration: object | None = None) -> None:
        self.camera = SimpleNamespace(calibration=lambda: calibration if calibration is not None else _calibration())

    def grab(self) -> RGBDFrame:
        return RGBDFrame(color=np.zeros((8, 8, 3), dtype=np.uint8), depth=np.full((8, 8), 500, dtype=np.uint16))

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


class _CountingArm:
    """An arm whose TCP read is counted, so a test can say how often it was asked."""

    def __init__(self) -> None:
        self.reads = 0

    def get_tcp_pose(self) -> Pose:
        self.reads += 1
        return _POSE


def _source(calibration: object | None = None) -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(
        streamer=_Streamer(calibration), backend=SimpleNamespace(perceive=lambda _bgr, _prompt: []), prompt="a box",
        warmup_grabs=0,
    )


#: The tree the wiring reads: the cell's fresh-frame attempts and its declared tool frame, at the flange.
_TREE = SimpleNamespace(
    safety=SimpleNamespace(planning_world=SimpleNamespace(perceived=SimpleNamespace(fresh_frame_attempts=2))),
    gripper=SimpleNamespace(tool_frame=SimpleNamespace(source="willy", offset_mm=(0.0, 0.0, 0.0),
                                                       rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                                                       verify_tolerance_mm=1.0)),
)


def _build_real_cell(*, resolver: object, arm: object, perception: RealSenseVisionPerceptionSource,
                     resolvers: "dict | None" = None, fused: "dict | None" = None) -> None:
    """`build_real_cell` with the device half and the root replaced, so what runs is its own wiring."""
    multi = None if fused is None else SimpleNamespace(sources=fused)
    service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
        frame_resolver=resolver, arm=arm, camera_frame_resolvers=resolvers or {}, multi_camera_perception=multi)))
    app_cfg = SimpleNamespace(camera=SimpleNamespace(cameras=SimpleNamespace(primary_rig_id="wrist")))
    with mock.patch.object(cells, "build_real_components", return_value=(None, perception, None, multi, None)), \
         mock.patch.object(AutonomousGraspService, "from_robot_config", return_value=service):
        cells.build_real_cell(_TREE, app_config=app_cfg)  # type: ignore[arg-type]


class AWristCellStampsItsPickFrameTests(unittest.TestCase):

    def test_a_real_wrist_cell_stamps_the_pick_frame(self) -> None:
        arm = DummyRobotArm(initial_pose=_POSE)
        arm.connect()
        perception = _source()

        _build_real_cell(resolver=_wrist_resolver(), arm=arm, perception=perception)

        self.assertEqual(perception.acquire().tool_pose, _POSE,
                         "a wrist cell's pick frame carries no tool pose, so its grasp reads the arm when resolved")

    def test_the_stamped_frame_is_resolved_without_reading_the_arm_again(self) -> None:
        stamping, resolving = _CountingArm(), _CountingArm()
        perception, resolver = _source(), _wrist_resolver()
        _build_real_cell(resolver=resolver, arm=stamping, perception=perception)

        resolver.camera_to_base_for_frame(perception.acquire(), arm=resolving)  # type: ignore[arg-type]

        # Read before and after the grab, to show the tool held still; never again when the grasp is resolved.
        self.assertEqual((stamping.reads, resolving.reads), (2, 0))

    def test_a_real_cell_binds_every_wrist_source(self) -> None:
        arm = DummyRobotArm(initial_pose=_POSE)
        arm.connect()
        primary, fused_wrist, fused_fixed = _source(), _source(), _source()
        fixed = StaticCameraToBaseResolver(transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE))
        _build_real_cell(resolver=_wrist_resolver(), arm=arm, perception=primary,
                         resolvers={"side_wrist": _wrist_resolver(), "top": fixed},
                         fused={"side_wrist": fused_wrist, "top": fused_fixed})

        self.assertEqual(_POSE, fused_wrist.acquire().tool_pose, "a fused wrist camera's frames go unstamped")
        self.assertIsNone(fused_fixed.acquire().tool_pose)

    def test_a_real_wrist_cell_refuses_a_missing_or_stale_flange_to_tcp_record(self) -> None:
        from src.calibration.serialization import FlangeToTcp

        moved = np.eye(4)
        moved[:3, 3] = (0.0, 0.0, 12.0)
        for label, calibration, says in (
            ("no record", _calibration(record=None), "records none"),
            ("a stale record", _calibration(record=FlangeToTcp.from_matrix("willy", moved)), "stale"),
            ("a record from another tool frame source",
             _calibration(record=FlangeToTcp.from_matrix("polyscope", np.eye(4))), "tool frame source"),
        ):
            with self.subTest(label):
                arm = DummyRobotArm(initial_pose=_POSE)
                arm.connect()
                with self.assertRaises(cells.CellBuildRefused) as caught:
                    _build_real_cell(resolver=_wrist_resolver(), arm=arm, perception=_source(calibration))
                self.assertIn(says, str(caught.exception))
                self.assertIn("calibrate", str(caught.exception))


class AFixedCellIsLeftAsItWasTests(unittest.TestCase):
    """The controls: a fixed cell and an unstamped frame behave as they would with no stamping at all."""

    def test_a_fixed_camera_cell_stamps_nothing_and_reads_no_arm(self) -> None:
        arm, perception = _CountingArm(), _source()
        _build_real_cell(
            resolver=StaticCameraToBaseResolver(
                transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)),
            arm=arm, perception=perception,
        )

        self.assertIsNone(perception.acquire().tool_pose)
        self.assertEqual(arm.reads, 0)

    def test_an_unstamped_frame_still_reads_the_arm_when_resolved(self) -> None:
        """The fallback that keeps every existing source working."""
        arm = _CountingArm()
        _wrist_resolver().camera_to_base_for_frame(_source().acquire(), arm=arm)  # type: ignore[arg-type]
        self.assertEqual(arm.reads, 1)


if __name__ == "__main__":
    unittest.main()
