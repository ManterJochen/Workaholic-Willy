"""`RigDepthSource`: a camera rig asked for depth and nothing else, beside the world it feeds.

It lives in `safety.planning.depth_source` rather than in `execution.autonomous_grasp`, because the world is to
be handed to `Robot`, and `Robot` may load neither `autonomous_grasp` nor `src.camera`. The rows below are
literals rather than a second copy of the class, so they check what the class does.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.safety.planning.depth_source import RigDepthSource

_MODULE = "src.robot.safety.planning.depth_source"
_ROOT = Path(__file__).resolve().parents[1]
_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


class _Handle:
    """A rig handle double: `grab` and `get_intrinsics`, the two calls a depth source makes."""

    def __init__(self, *, frame=None, intrinsics=None, raises: Exception | None = None,  # noqa: ANN001
                 rig_id: str = "overhead") -> None:
        self.rig_id = rig_id
        self._frame = frame
        self._intrinsics = intrinsics
        self._raises = raises
        self.events: list[str] = []

    def grab(self):  # noqa: ANN201
        self.events.append("grab")
        if self._raises is not None:
            raise self._raises
        return self._frame

    def get_intrinsics(self):  # noqa: ANN201
        return self._intrinsics


class EveryReasonARigCannotAnswerIsNoneTests(unittest.TestCase):

    def test_each_row_gives_none(self) -> None:
        rows = {
            "the grab raises": _Handle(raises=RuntimeError("device gone"), intrinsics=_K),
            "the frame has no depth": _Handle(
                frame=SimpleNamespace(left=np.zeros((2, 2)), right=np.zeros((2, 2))), intrinsics=_K),
            "the rig has no matrix": _Handle(frame=SimpleNamespace(depth=np.ones((2, 2))), intrinsics=None),
            "the depth is one dimensional": _Handle(frame=SimpleNamespace(depth=np.ones(4)), intrinsics=_K),
            "the depth is empty": _Handle(frame=SimpleNamespace(depth=np.zeros((0, 0))), intrinsics=_K),
        }
        for why, handle in rows.items():
            with self.subTest(why):
                self.assertIsNone(RigDepthSource(handle).grab_surface_depth())


class AReadingTests(unittest.TestCase):

    def test_a_reading_carries_the_depth_the_matrix_and_the_time_read_before_the_grab(self) -> None:
        depth = np.array([[1000, 0], [1200, 1500]], dtype=np.uint16)
        handle = _Handle(frame=SimpleNamespace(depth=depth), intrinsics=_K)

        def clock() -> float:
            handle.events.append("clock")
            return 1700000000.25

        with mock.patch(f"{_MODULE}.time", SimpleNamespace(time=clock)):
            reading = RigDepthSource(handle).grab_surface_depth()
        assert reading is not None
        self.assertEqual(handle.events, ["clock", "grab"])
        self.assertEqual(reading.timestamp, 1700000000.25)
        self.assertEqual(reading.depth_mm.dtype, np.float64)
        np.testing.assert_array_equal(reading.depth_mm, [[1000.0, 0.0], [1200.0, 1500.0]])
        np.testing.assert_array_equal(reading.intrinsics, _K)

    def test_its_name_is_the_rig_id_unless_one_is_given(self) -> None:
        self.assertEqual(RigDepthSource(_Handle(rig_id="wrist")).name, "wrist")
        self.assertEqual(RigDepthSource(_Handle(rig_id="wrist"), name="left").name, "left")
        self.assertEqual(repr(RigDepthSource(_Handle(rig_id="wrist"))), "RigDepthSource('wrist')")


class ImportBoundaryTests(unittest.TestCase):
    """`Robot` is to take the world, and its guard forbids the camera package and the execution layer."""

    _PROBE = ("import importlib, json, sys; importlib.import_module({module!r}); "
              "print(json.dumps(sorted(m for m in sys.modules "
              "if m.startswith(('src.camera', 'src.robot.execution')))))")

    def _loaded_after(self, module: str) -> list[str]:
        out = subprocess.run([sys.executable, "-c", self._PROBE.format(module=module)], cwd=_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])

    def test_the_depth_source_loads_no_camera_and_no_execution(self) -> None:
        self.assertEqual(self._loaded_after(_MODULE), [])

    def test_the_probe_sees_the_execution_layer_where_it_is_loaded(self) -> None:
        """The control: the same probe over the cell builders sees the execution layer arrive."""
        self.assertIn("src.robot.execution",
                      self._loaded_after("src.robot.execution.autonomous_grasp.cells"))


def _pose(x_mm: float = 400.0, *, yaw_deg: float = 0.0) -> Pose:
    half = float(np.deg2rad(yaw_deg)) / 2.0
    return Pose(position_mm=np.array([x_mm, 0.0, 300.0]),
                quaternion_xyzw=np.array([0.0, 0.0, np.sin(half), np.cos(half)]), frame=Frame.BASE)


class _Reader:
    """The arm's TCP reader: answers its poses in turn, the last one from then on, and says when it was asked."""

    def __init__(self, handle: _Handle, *poses: Pose) -> None:
        self._handle = handle
        self._poses = list(poses)

    def __call__(self) -> Pose:
        self._handle.events.append("pose")
        return self._poses.pop(0) if len(self._poses) > 1 else self._poses[0]


def _depth_handle() -> _Handle:
    return _Handle(frame=SimpleNamespace(depth=np.full((2, 2), 900.0)), intrinsics=_K, rig_id="wrist")


class AWristRigTests(unittest.TestCase):
    """A wrist rig reads the TCP either side of its grab, and a frame taken while the tool moved is none."""

    def test_an_arm_that_moved_during_the_grab_gives_no_frame(self) -> None:
        rows = {
            "5 mm against a 1 mm tolerance": (_pose(400.0), _pose(405.0)),
            "2 degrees against a 0.5 degree tolerance": (_pose(yaw_deg=0.0), _pose(yaw_deg=2.0)),
        }
        for why, (before, after) in rows.items():
            with self.subTest(why):
                handle = _depth_handle()
                source = RigDepthSource(handle, tool_pose=_Reader(handle, before, after), motion_tolerance=(1.0, 0.5))
                self.assertIsNone(source.grab_surface_depth())
                self.assertEqual(handle.events, ["pose", "grab", "pose"])

    def test_a_resting_arm_stamps_its_pose(self) -> None:
        """Inside both tolerances, the frame carries the pose read before the grab, the one nearest the shutter."""
        handle = _depth_handle()
        before, after = _pose(400.0), _pose(400.4, yaw_deg=0.2)
        reading = RigDepthSource(
            handle, tool_pose=_Reader(handle, before, after), motion_tolerance=(1.0, 0.5),
        ).grab_surface_depth()
        assert reading is not None
        np.testing.assert_array_equal(reading.tool_to_base_mm, before.to_matrix())

    def test_a_reader_that_raises_gives_no_frame(self) -> None:
        def broken() -> Pose:
            raise RuntimeError("the controller dropped the connection")

        self.assertIsNone(
            RigDepthSource(_depth_handle(), tool_pose=broken, motion_tolerance=(1.0, 0.5)).grab_surface_depth())

    def test_the_reader_and_the_tolerance_come_together(self) -> None:
        with self.assertRaises(ValueError):
            RigDepthSource(_depth_handle(), tool_pose=lambda: _pose())
        with self.assertRaises(ValueError):
            RigDepthSource(_depth_handle(), motion_tolerance=(1.0, 0.5))

    def test_a_fixed_rig_stamps_no_pose(self) -> None:
        """The control: a rig built without a reader reads no arm and stamps nothing."""
        handle = _depth_handle()
        reading = RigDepthSource(handle).grab_surface_depth()
        assert reading is not None
        self.assertIsNone(reading.tool_to_base_mm)
        self.assertEqual(handle.events, ["grab"])


class TheCaptureTimeTests(unittest.TestCase):

    def test_the_depth_snapshot_carries_the_frames_capture_time(self) -> None:
        """The owner stamps a frame before its grab, under the rig's lock, and that stamp is the one kept."""
        handle = _Handle(frame=SimpleNamespace(depth=np.ones((2, 2)), captured_at_s=123.0), intrinsics=_K)
        with mock.patch(f"{_MODULE}.time", SimpleNamespace(time=lambda: 999.0)):
            reading = RigDepthSource(handle).grab_surface_depth()
        assert reading is not None
        self.assertEqual(reading.timestamp, 123.0)


class TheReaderIsTheArmsOwnMethodTests(unittest.TestCase):
    """The reader is `RobotArm.get_tcp_pose`, so it is measured on arms that are not a UR."""

    def test_a_dummy_arm_and_a_sim_arm_stamp_through_get_tcp_pose(self) -> None:
        dummy = DummyRobotArm(initial_pose=_pose(250.0))
        dummy.connect()
        sim = IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True))
        sim.connect()
        for name, arm in (("dummy", dummy), ("sim", sim)):
            with self.subTest(name):
                reading = RigDepthSource(
                    _depth_handle(), tool_pose=arm.get_tcp_pose, motion_tolerance=(1.0, 0.5),
                ).grab_surface_depth()
                assert reading is not None
                np.testing.assert_array_equal(reading.tool_to_base_mm, arm.get_tcp_pose().to_matrix())


if __name__ == "__main__":
    unittest.main()
