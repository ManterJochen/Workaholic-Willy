"""Sweep stations may be joint angles: read from JSON, checked, boxed on the arm's own FK, and run as one joint move.

Issue 2 of the calibration chain, part A (owner, 2026-09-23): the user asked for MoveJ stations from a JSON file, so a
station taught on the pendant goes to exactly the configuration taught, instead of whichever of up to 8 x 32 joint
configurations a planner picks for a pose (the analysis measured cuRobo taking 8 of 22 legs of a wrist sweep the long
way round). Before this change a joint record in a stations file was refused as unrecognised, ``CalibrationRoutine``
raised ``TypeError`` for anything but a ``Pose``, the calibrate CLI had no ``--fixed-poses``, ``check()`` reported a
JSON file as ``-1`` poses, and the JSON pre-filter dropped a pose with one log line and nothing on the report.

What must hold, and is held here:

* a record is checked when the file is read: one unit key, six finite numbers, and units that can be what they say;
* a joint move skips the Cartesian box, so the sweep boxes the grasp centre the arm's own forward kinematics puts at
  the joints, and a station outside the box is reported and never moved to;
* the joints the arm is sent are exactly the joints the file wrote (the same ``JointPositions``), through
  ``move_to_joints``, so the move the arm judged is the move that runs;
* the guard's history takes the TCP the arm reports after the move, not the record;
* every station a screen drops is on the events and the report, with its label and reason.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from src.calibration import MountingMode
from src.calibration.exceptions import CalibrationDataError
from src.config.schema.camera import HandEyeConfig
from src.config.schema.robot import MotionLimitsConfig, RobotConfig, WorkspaceLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions
from src.robot.core.camera_world import active_decline
from src.robot.core.motion_result import MotionCommand, MotionResult, MotionStatus
from src.robot.events import RobotCalibrationEvent
from src.robot.execution.calibration import CalibrationRoutine
from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions, render_sweep_event
from src.robot.execution.pose_provider import (
    JointStation,
    load_stations,
)
from src.robot.execution.real_cell import calibrate
from tests.test_camera_boundaries import _rgbd_rig
from tests.test_robot_boundaries import _inverse, _synthetic_eye_to_hand_data

_ROOT = Path(__file__).resolve().parents[1]
_STATION = Pose.tool_down(400.0, 0.0, 350.0, label="down")
_WIDE = WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=-2000.0, z_max=2000.0)
_HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


def _write(folder: Path, records: Any, name: str = "stations.json") -> Path:
    path = folder / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _joints(index: int) -> JointPositions:
    return JointPositions([0.1 * index, -1.2, 1.4, -1.6, -1.5, 0.05 * index])


# ---------------------------------------------------------------------------------------------------------------------
# Reading a stations file
# ---------------------------------------------------------------------------------------------------------------------


class AStationsFileIsReadAndCheckedTests(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_joint_records_read_as_radians_and_mix_with_poses_in_file_order(self) -> None:
        path = _write(self.folder, [
            {"label": "taught_0", "joints_deg": _HOME_DEG},
            {"x": 400.0, "y": 0.0, "z": 350.0, "rx": 0.0, "ry": 3.14159265, "rz": 0.0, "label": "down"},
            {"joints_rad": [0.5, -1.0, 1.0, -1.5, -1.5708, 0.25]},
            {"schema": "willy.geometry.pose/1", "position_mm": [300.0, 10.0, 400.0],
             "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0], "frame": "base", "label": "flip"},
        ])
        stations = load_stations(path)
        self.assertEqual([type(one).__name__ for one in stations], ["JointStation", "Pose", "JointStation", "Pose"])
        taught, _, written, _ = stations
        self.assertEqual((taught.label, taught.written_as), ("taught_0", "joints_deg"))
        np.testing.assert_array_equal(taught.joints.values, np.radians(_HOME_DEG))
        self.assertEqual((written.label, written.written_as), ("pose_2", "joints_rad"))
        np.testing.assert_array_equal(written.joints.values, [0.5, -1.0, 1.0, -1.5, -1.5708, 0.25])
        self.assertEqual(taught.degrees, tuple(_HOME_DEG))

    def test_each_bad_record_is_refused_naming_its_place_and_label(self) -> None:
        cases = [
            ({"label": "both", "joints_deg": _HOME_DEG, "joints_rad": [0.0] * 6}, "both joints_deg and joints_rad"),
            ({"label": "mixed", "joints_deg": _HOME_DEG, "x": 1.0}, "a record is a pose or a joint station"),
            ({"label": "five", "joints_deg": _HOME_DEG[:5]}, "holds 5 values; a joint station holds 6"),
            ({"label": "seven", "joints_deg": [*_HOME_DEG, 0.0]}, "holds 7 values"),
            ({"label": "scalar", "joints_deg": 90.0}, "holds float values"),
            ({"label": "word", "joints_deg": [0.0, "-90", 90.0, -90.0, -90.0, 0.0]}, "shoulder as '-90'"),
            ({"label": "flag", "joints_rad": [True, 0.0, 0.0, 0.0, 0.0, 0.0]}, "base as True"),
            ({"label": "nan", "joints_rad": [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]}, "not a finite number"),
            ({"label": "inf", "joints_deg": [0.0, -90.0, float("inf"), -90.0, -90.0, 0.0]}, "not a finite number"),
            ({"label": "degrees", "joints_rad": _HOME_DEG}, "shoulder as -90, beyond one full turn (2*pi) in radians"),
            ({"label": "wound", "joints_deg": [0.0, -90.0, 90.0, -90.0, -90.0, 400.0]}, "wrist 3 as 400, beyond one "
                                                                                         "full turn (360)"),
            ({"label": "radians", "joints_deg": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}, "which reads as radians"),
            ({"label": "camera", "schema": "willy.geometry.pose/1", "position_mm": [0.0, 0.0, 500.0],
              "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "frame": "camera"}, "a station is in the robot's base frame"),
            ({"label": "unversioned", "position_mm": [0.0, 0.0, 500.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
              "frame": "base"}, "expected schema 'willy.geometry.pose/1'"),
            ({"label": "nothing", "z": 3.0}, "Unrecognised pose JSON record"),
        ]
        for record, said in cases:
            with self.subTest(record["label"]):
                path = _write(self.folder, [{"joints_deg": _HOME_DEG}, record])
                with self.assertRaises(ValueError) as caught:
                    load_stations(path)
                text = str(caught.exception)
                self.assertIn(f"record 1 ({record['label']!r})", text)
                self.assertIn(said, text)

    def test_a_file_that_is_not_a_list_or_is_empty_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a top-level list"):
            load_stations(_write(self.folder, {"joints_deg": _HOME_DEG}))
        with self.assertRaisesRegex(ValueError, "holds no stations"):
            load_stations(_write(self.folder, []))

    def test_a_value_at_a_full_turn_is_not_refused_for_its_rounding(self) -> None:
        (station,) = load_stations(_write(self.folder, [{"joints_deg": [360.0, -90.0, 90.0, -90.0, -90.0, -360.0]}]))
        self.assertEqual(station.joints.values[0], 2.0 * math.pi)

    def test_a_joint_station_built_in_code_is_radians_within_one_turn(self) -> None:
        with self.assertRaisesRegex(ValueError, "beyond one full turn"):
            JointStation(JointPositions([0.0, -90.0, 0.0, 0.0, 0.0, 0.0]), label="deg")
        with self.assertRaises(TypeError):
            JointStation([0.0] * 6)  # type: ignore[arg-type]

    def test_a_wound_joint_is_read_as_written(self) -> None:
        """Nothing here rewrites a value: the arm turns a joint onto the full turn nearest where it stands, inside its
        cable window, when it moves (the owner, 2026-09-24); the file keeps what it says."""
        wound = [0.0, -90.0, 90.0, -90.0, -90.0, 270.0]
        path = _write(self.folder, [{"label": "a", "joints_deg": _HOME_DEG}, {"label": "b", "joints_deg": wound}])
        stations = load_stations(path)
        np.testing.assert_array_equal(stations[1].joints.values, np.radians(wound))

    def test_a_joint_station_may_carry_the_tcp_it_was_taught_at_as_a_note(self) -> None:
        """The record a hand-guided run writes: the joints run, the TCP beside them is for the person reading it."""
        path = _write(self.folder, [{"label": "hand_01", "joints_deg": _HOME_DEG,
                                     "tcp_pose": {"x": 1.0, "y": 2.0, "z": 3.0, "rx": 0.0, "ry": 3.14, "rz": 0.0}}])
        (station,) = load_stations(path)
        self.assertIsInstance(station, JointStation)
        np.testing.assert_array_equal(station.joints.values, np.radians(_HOME_DEG))

    def test_the_template_file_reads_as_four_joint_stations_and_two_poses(self) -> None:
        stations = load_stations(_ROOT / "examples" / "real_robot" / "eih_fixed_stations.json")
        self.assertEqual([isinstance(one, JointStation) for one in stations], [True] * 4 + [False] * 2)
        self.assertEqual([one.label for one in stations], [f"look_{index}" for index in range(6)])


# ---------------------------------------------------------------------------------------------------------------------
# The routine: boxed on the arm's FK, moved with exactly the record, accepted on the TCP read
# ---------------------------------------------------------------------------------------------------------------------


class _Arm:
    """An arm double. ``tcp_at`` is its forward kinematics (joints tuple to a BASE matrix); after a joint move it
    reports that TCP shifted by ``drift_mm``, so a test can tell the FK from the TCP read. A label in ``refuse``
    answers with that status and message and does not move."""

    def __init__(self, tcp_at: dict[tuple[float, ...], np.ndarray], *, drift_mm: tuple[float, float, float] = (0, 0, 0),
                 refuse: dict[str, tuple[MotionStatus, str]] | None = None) -> None:
        self.tcp_at = tcp_at
        self.drift = np.asarray(drift_mm, dtype=np.float64)
        self.refuse = refuse or {}
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.declines: list[Any] = []
        self.at: Pose | None = None
        self.joints = JointPositions(np.radians(_HOME_DEG))

    def _pose(self, joints: JointPositions, *, label: str | None = None) -> Pose:
        return Pose.from_matrix(self.tcp_at[tuple(joints.tolist())], frame=Frame.BASE, label=label)

    def fk(self, joints: JointPositions) -> Pose:
        self.calls.append(("fk", joints, {}))
        return self._pose(joints, label="fk")

    def move(self, pose: Pose, **keywords: Any) -> MotionResult:
        self.calls.append(("move", pose, dict(keywords)))
        self.declines.append(active_decline(self))
        if pose.label in self.refuse:
            status, message = self.refuse[pose.label]
            return MotionResult.failed(status, MotionCommand.MOVE_TO, target_pose=pose, message=message)
        self.at = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        self.calls.append(("move_to_joints", joints, dict(keywords)))
        self.declines.append(active_decline(self))
        reached = self._pose(joints)
        self.at = Pose(position_mm=reached.position_mm + self.drift, quaternion_xyzw=reached.quaternion_xyzw,
                       frame=Frame.BASE, label="actual")
        self.joints = joints
        return MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints)

    def get_tcp_pose(self) -> Pose:
        assert self.at is not None
        return self.at

    def get_joint_positions(self) -> JointPositions:
        return self.joints

    def commanded(self) -> list[tuple[str, Any]]:
        return [(verb, what) for verb, what, _ in self.calls if verb != "fk"]


class _Source:
    """The marker seen from wherever the arm actually stands, for a fixed camera and a board on the flange."""

    def __init__(self, arm: _Arm) -> None:
        self.arm = arm
        self.T_cam_to_base, self.T_tool_to_marker, _ = _synthetic_eye_to_hand_data()
        self.calls = 0

    def __call__(self) -> np.ndarray:
        self.calls += 1
        return _inverse(self.T_cam_to_base) @ self.arm.get_tcp_pose().to_matrix() @ self.T_tool_to_marker


def _routine(arm: _Arm, *, limits: WorkspaceLimitsConfig = _WIDE, min_samples: int = 4,
             motion_limits: MotionLimitsConfig | None = None) -> tuple[CalibrationRoutine, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    routine = CalibrationRoutine(
        arm=arm, marker_source=_Source(arm), workspace_limits=limits,  # type: ignore[arg-type]
        calibration_mode="eye_to_hand", settle_time_s=0.0, motion_limits=motion_limits,
        eth_settings=SimpleNamespace(mode="eye_to_hand", min_samples=min_samples, min_distance_mm=10.0,
                                     min_angle=2.0, min_angle_deg=2.0),  # type: ignore[arg-type]
        on_event=lambda kind, data: events.append((kind, dict(data))),
    )
    return routine, events


def _tool() -> list[np.ndarray]:
    return _synthetic_eye_to_hand_data()[2]


class JointStationsRunAsJudgedJointMovesTests(unittest.TestCase):

    def test_each_joint_station_is_sent_as_exactly_the_joints_written_and_a_pose_is_planned(self) -> None:
        tool = _tool()
        stations: list[Any] = [JointStation(_joints(i), label=f"j{i}") for i in range(5)]
        stations.insert(2, Pose.from_matrix(tool[5], frame=Frame.BASE, label="pose"))
        arm = _Arm({tuple(_joints(i).tolist()): tool[i] for i in range(5)})
        routine, _ = _routine(arm, motion_limits=MotionLimitsConfig(max_velocity=0.4, max_acceleration=0.3))
        result = routine.run_with_poses(stations)

        commanded = arm.commanded()
        self.assertEqual([verb for verb, _ in commanded],
                         ["move_to_joints", "move_to_joints", "move", "move_to_joints", "move_to_joints",
                          "move_to_joints"])
        joint_moves = [what for verb, what in commanded if verb == "move_to_joints"]
        for sent, station in zip(joint_moves, [one for one in stations if isinstance(one, JointStation)]):
            self.assertIs(sent, station.joints)  # the configuration boxed is the configuration sent
        keywords = [kw for verb, _, kw in arm.calls if verb == "move_to_joints"]
        self.assertEqual(keywords[0], {"velocity": 0.4, "acceleration": 0.3})
        self.assertEqual([kw for verb, _, kw in arm.calls if verb == "move"],
                         [{"register": False, "vel": 0.4, "acc": 0.3}])
        self.assertTrue(all(decline is not None for decline in arm.declines), "a move ran outside the decline")
        self.assertEqual(len(result.camera_worlds), 6)
        self.assertEqual(result.num_samples, 6)

    def test_a_joint_station_whose_grasp_centre_is_outside_the_box_is_reported_and_never_moved_to(self) -> None:
        tool = _tool()
        box = WorkspaceLimitsConfig(x_min=-100.0, x_max=650.0, y_min=-300.0, y_max=300.0, z_min=0.0, z_max=400.0)
        stations = [JointStation(_joints(i), label=f"j{i}") for i in range(7)]  # j6 stands at x = 720 mm
        arm = _Arm({tuple(_joints(i).tolist()): tool[i] for i in range(7)})
        routine, events = _routine(arm, limits=box)
        result = routine.run_with_poses(stations)

        self.assertNotIn(stations[6].joints, [what for _, what in arm.commanded()])
        self.assertEqual(len(arm.commanded()), 6)
        verdict = result.pose_log[6]
        self.assertEqual((verdict.index, verdict.label, verdict.counted, verdict.reason),
                         (7, "j6", False, "outside_workspace"))
        self.assertTrue(verdict.detail.startswith("the grasp centre at these joints (720.0, 240.0, 280.0) mm is "
                                                  "outside workspace_limits (x -100.0 to 650.0"), verdict.detail)
        self.assertTrue(verdict.detail.endswith("so nothing moved"), verdict.detail)
        moving = [data["index"] for kind, data in events if kind == RobotCalibrationEvent.MOVING_TO_POSE]
        self.assertEqual(moving, [1, 2, 3, 4, 5, 6], "a station that never moves announced a move")
        (rejected,) = [data for kind, data in events if kind == RobotCalibrationEvent.POSE_REJECTED]
        self.assertEqual((rejected["label"], rejected["reason"]), ("j6", "outside_workspace"))

    def test_a_joint_station_nobody_can_place_is_not_moved_to(self) -> None:
        arm = _Arm({})

        def unread(_joints: JointPositions) -> Pose:
            raise RuntimeError("FK failed: controller busy")

        arm.fk = unread  # type: ignore[method-assign]
        routine, _ = _routine(arm)
        with self.assertRaises(CalibrationDataError):  # nothing counted, so the solve refuses; the log is held
            routine.run_with_poses([JointStation(_joints(0), label="blind")])
        self.assertEqual(arm.commanded(), [])
        (verdict,) = routine.pose_log
        self.assertEqual(verdict.reason, "not_boxed")
        self.assertIn("could not be read (fk raised RuntimeError: FK failed: controller busy)", verdict.detail)
        self.assertIn("a joint station the workspace box never saw is not moved to", verdict.detail)

    def test_the_arm_kinematic_chain_places_a_station_when_fk_cannot(self) -> None:
        tool = _tool()
        keys = [tuple(_joints(i).tolist()) for i in range(6)]
        arm = _Arm(dict(zip(keys, tool)))

        def unread(_joints: JointPositions) -> Pose:
            raise RuntimeError("no connection")

        arm.fk = unread  # type: ignore[method-assign]
        arm._tcp_at_joints_mm = lambda joints: tool[keys.index(tuple(joints.tolist()))]  # type: ignore[attr-defined]
        box = WorkspaceLimitsConfig(x_min=-100.0, x_max=500.0, y_min=-300.0, y_max=300.0, z_min=0.0, z_max=400.0)
        routine, _ = _routine(arm, limits=box)
        routine.run_with_poses([JointStation(_joints(i), label=f"j{i}") for i in range(6)])
        self.assertEqual([verdict.reason for verdict in routine.pose_log], ["", "", "", "", "", "outside_workspace"])
        self.assertIn("(600.0, -200.0, 250.0) mm is outside", routine.pose_log[5].detail)

    def test_a_joint_station_too_close_to_one_kept_before_it_is_skipped(self) -> None:
        tool = _tool()
        twin = JointPositions([*_joints(1).tolist()[:5], 0.05 + 1e-3])
        arm = _Arm({**{tuple(_joints(i).tolist()): tool[i] for i in range(5)}, tuple(twin.tolist()): tool[1]})
        routine, _ = _routine(arm)
        stations = [JointStation(_joints(i), label=f"j{i}") for i in range(5)]
        stations.insert(2, JointStation(twin, label="twin"))
        result = routine.run_with_poses(stations)
        verdict = result.pose_log[2]
        self.assertEqual((verdict.label, verdict.reason), ("twin", "too_similar"))
        self.assertIn("is 0.0 mm and 0.0 deg from 'j1', kept before it", verdict.detail)
        self.assertNotIn(twin, [what for _, what in arm.commanded()])

    def test_the_guard_keeps_the_tcp_the_arm_reports_not_the_record(self) -> None:
        tool = _tool()
        arm = _Arm({tuple(_joints(i).tolist()): tool[i] for i in range(4)}, drift_mm=(1.5, -2.0, 0.5))
        routine, _ = _routine(arm)
        routine.run_with_poses([JointStation(_joints(i), label=f"j{i}") for i in range(4)])
        kept = routine.guard.accepted_poses
        self.assertEqual(len(kept), 4)
        for pose, matrix in zip(kept, tool):
            np.testing.assert_allclose(pose.position_mm, matrix[:3, 3] + [1.5, -2.0, 0.5])
        self.assertEqual({pose.label for pose in kept}, {"actual"})

    def test_a_refused_joint_move_skips_its_station_and_the_sweep_goes_on(self) -> None:
        tool = _tool()
        arm = _Arm({tuple(_joints(i).tolist()): tool[i] for i in range(5)})
        refused = _joints(2)
        original = arm.move_to_joints

        def refusing(joints: JointPositions, **keywords: Any) -> MotionResult:
            if joints is refused:
                arm.calls.append(("move_to_joints", joints, dict(keywords)))
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_JOINTS,
                                           target_joints=joints, message="the planner refused this joint path")
            return original(joints, **keywords)

        arm.move_to_joints = refusing  # type: ignore[method-assign]
        routine, _ = _routine(arm)
        stations = [JointStation(_joints(i), label=f"j{i}") for i in range(5)]
        stations[2] = JointStation(refused, label="j2")
        result = routine.run_with_poses(stations)
        self.assertEqual([verdict.reason for verdict in result.pose_log], ["", "", "move_rejected", "", ""])
        self.assertEqual(result.pose_log[2].detail, "self_collision_rejected: the planner refused this joint path")

    def test_a_joint_station_says_its_joints_and_is_handed_to_the_arm_as_written(self) -> None:
        """Which full turn a joint takes is the arm's to choose, the one nearest where it stands inside its cable
        window (the owner, 2026-09-24): the sweep neither rewrites nor wraps a station, and warns of no hop."""
        tool = _tool()
        far = JointPositions([0.0, -1.2, 1.4, -1.6, -1.5, 4.0])
        arm = _Arm({**{tuple(_joints(i).tolist()): tool[i] for i in range(4)}, tuple(far.tolist()): tool[5]})
        arm.joints = JointPositions([0.0, -1.2, 1.4, -1.6, -1.5, -0.5])
        routine, events = _routine(arm)
        stations = [JointStation(far, label="far"), *(JointStation(_joints(i), label=f"j{i}") for i in range(4))]
        routine.run_with_poses(stations)
        moving = [data for kind, data in events if kind == RobotCalibrationEvent.MOVING_TO_POSE]
        self.assertEqual(moving[0]["joints_deg"], [0.0, -68.755, 80.214, -91.673, -85.944, 229.183])
        self.assertNotIn("hop", moving[0])
        self.assertEqual(render_sweep_event(RobotCalibrationEvent.MOVING_TO_POSE, moving[0]),
                         "pose  1/5 'far'  moving, 0 counted so far, need 4")
        self.assertIs(arm.commanded()[0][1], far)

    def test_anything_but_a_pose_or_a_joint_station_is_a_programming_error(self) -> None:
        routine, _ = _routine(_Arm({}))
        with self.assertRaisesRegex(TypeError, "requires Pose or JointStation targets; got list"):
            routine.run_with_poses([[0.0] * 6])  # type: ignore[list-item]


class OnACuroboUrTheJudgedLineIsTheLineThatRunsTests(unittest.TestCase):
    """The sweep's joint station through the real UR driver, its controller and planner doubles: the straight joint
    line from where the arm stands is judged by the path gate and by the planner, and one ``moveJ`` to exactly the
    joints written runs it."""

    _HERE = [0.0, -1.2, 1.4, -1.6, -1.5, -0.1]

    def _ur(self, *, refuse_at: int | None = None) -> tuple[Any, Any, list[Any], list[Any]]:
        """The UR driver on cuRobo, its controller a double whose moveJ returns False for station ``refuse_at``."""
        from src.robot.safety.planning import JointCheckVerdict
        from tests.test_ur_arm import _arm as _ur_arm

        tool = _tool()
        keys = [tuple(_joints(i).tolist()) for i in range(5)]
        arm = _ur_arm("curobo")
        conn = arm._conn  # noqa: SLF001 (the controller double)
        standing = {"joints": list(self._HERE)}

        def fk(joints: list[float]) -> list[float]:
            pose = Pose.from_matrix(tool[keys.index(tuple(joints))], frame=Frame.BASE)
            return [*(float(v) / 1000.0 for v in pose.position_mm), *(float(v) for v in pose.axis_angle_rad())]

        def move_j(joints: list[float], **_keywords: Any) -> bool:
            if refuse_at is not None and tuple(joints) == keys[refuse_at]:
                return False  # a protective stop part of the way along
            standing["joints"] = list(joints)
            return True

        conn.fk.side_effect = fk
        conn.moveJ.side_effect = move_j
        conn.get_joint_positions.side_effect = lambda: list(standing["joints"])
        judged: list[list[list[float]]] = []
        checked: list[list[list[float]]] = []
        arm._preflight = SimpleNamespace(  # noqa: SLF001
            gate_joint_target=lambda joints, *, arm=None: None,
            gate_planned_path=lambda waypoints, *, arm=None, command=None: judged.append(
                [list(map(float, w)) for w in waypoints]),
            path_step_mm=5.0, joint_radii_mm=lambda _arm: [100.0] * 6)

        class _Planner:
            last_world_refresh = None

            def check_joint_path(self, configs: Any, *, refresh: bool = True,
                                 clearance_mm: float = 0.0) -> JointCheckVerdict:
                checked.append([list(map(float, c)) for c in configs])
                return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="accepts")

        arm._curobo_ur = _Planner()  # noqa: SLF001
        arm.get_tcp_pose = lambda: arm.fk(JointPositions(standing["joints"]))
        return arm, conn, judged, checked

    def test_one_judged_move_j_per_joint_station_to_exactly_the_joints_written(self) -> None:
        arm, conn, judged, checked = self._ur()
        routine, _ = _routine(arm)
        stations = [JointStation(_joints(i), label=f"j{i}") for i in range(5)]
        result = routine.run_with_poses(stations)

        self.assertEqual(result.num_samples, 5)
        sent = [call.args[0] for call in conn.moveJ.call_args_list]
        self.assertEqual(sent, [station.joints.tolist() for station in stations])  # one moveJ each, as written
        starts = [self._HERE, *(station.joints.tolist() for station in stations[:-1])]
        self.assertEqual(judged, [[start, station.joints.tolist()] for start, station in zip(starts, stations)])
        for line, start, station in zip(checked, starts, stations):
            self.assertEqual((line[0], line[-1]), (start, station.joints.tolist()))  # the planner saw the same line
        self.assertEqual([stamp.use.value for stamp in result.camera_worlds], ["declined"] * 5)

    def test_a_move_j_the_controller_did_not_complete_stops_the_sweep_through_the_real_driver(self) -> None:
        """The contract between the UR driver's refusal and the routine's stop, held on the driver's own words."""
        from src.robot.execution.calibration import SweepStopped

        arm, conn, _, _ = self._ur(refuse_at=2)
        routine, _ = _routine(arm)
        with self.assertRaises(SweepStopped) as caught:
            routine.run_with_poses([JointStation(_joints(i), label=f"j{i}") for i in range(5)])
        self.assertEqual(len(conn.moveJ.call_args_list), 3, "a moveJ was sent after the controller refused one")
        self.assertEqual((caught.exception.label, caught.exception.status), ("j2", "controller_rejected"))
        self.assertIn("moveJ was sent", routine.pose_log[-1].detail)


class AJsonFileReportsEveryStationItDropsTests(unittest.TestCase):
    """Red before: the JSON pre-filter dropped a pose with one log line; the events and the report never said so."""

    def test_a_dropped_pose_is_on_the_events_and_the_pose_log_in_file_order(self) -> None:
        tool = _tool()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        records: list[dict[str, Any]] = []
        for index, matrix in enumerate(tool[:5]):
            pose = Pose.from_matrix(matrix, frame=Frame.BASE)
            rx, ry, rz = (float(v) for v in pose.axis_angle_rad())
            x, y, z = (float(v) for v in pose.position_mm)
            records.append({"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz, "label": f"p{index}"})
        records.insert(2, {**records[1], "label": "again"})
        records.insert(4, {**records[0], "z": 5000.0, "label": "sky"})
        taught_deg = [float(v) for v in np.degrees(_joints(5).values)]
        records.append({"label": "taught", "joints_deg": taught_deg})
        path = _write(folder, records)
        arm = _Arm({tuple(math.radians(v) for v in taught_deg): tool[5]})
        routine, events = _routine(arm)
        result = routine.run_from_json(path)

        self.assertEqual([(v.index, v.label, v.reason) for v in result.pose_log], [
            (1, "p0", ""), (2, "p1", ""), (3, "again", "too_similar"), (4, "p2", ""), (5, "sky", "outside_workspace"),
            (6, "p3", ""), (7, "p4", ""), (8, "taught", "")])
        rejected = {data["label"]: data for kind, data in events if kind == RobotCalibrationEvent.POSE_REJECTED}
        self.assertEqual(set(rejected), {"again", "sky"})
        self.assertEqual((rejected["sky"]["index"], rejected["sky"]["total"]), (5, 8))
        self.assertTrue(rejected["sky"]["detail"].startswith("the pose (0.0, 0.0, 5000.0) mm is outside"))
        self.assertEqual(len(arm.commanded()), 6)
        self.assertIs(arm.commanded()[-1][0], "move_to_joints")


# ---------------------------------------------------------------------------------------------------------------------
# check() and the CLI
# ---------------------------------------------------------------------------------------------------------------------


def _app() -> SimpleNamespace:
    return SimpleNamespace(
        robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.41"}}),
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")]), hand_eye=HandEyeConfig()))


def _noun(**options: Any) -> HandEyeCalibration:
    return HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_to_hand",
                                          options=SweepOptions(**options))


class TheCheckCountsTheStationsTests(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_a_file_is_read_counted_and_its_joint_stations_named(self) -> None:
        path = _write(self.folder, [
            {"label": "a", "joints_deg": _HOME_DEG},
            {"label": "b", "joints_deg": [0.0, -90.0, 90.0, -90.0, -90.0, 200.0]},
            {"x": 400.0, "y": 0.0, "z": 350.0, "rx": 0.0, "ry": 3.14159265, "rz": 0.0},
        ])
        check = _noun(fixed_poses=path).check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual((check.poses, check.joint_stations, check.poses_file), (3, 2, str(path)))
        lines = check.render().splitlines()
        self.assertEqual(lines[4], f"  poses      3 from {path}, 2 of them joint stations")
        self.assertEqual(lines[5], "  artifact   calibration/real/eth_overhead.json")
        self.assertEqual(check.to_dict()["joint_stations"], 2)

    def test_a_bad_or_missing_file_is_refused_at_check(self) -> None:
        bad = _write(self.folder, [{"label": "degrees", "joints_rad": _HOME_DEG}])
        refusal = _noun(fixed_poses=bad).check().refusal
        self.assertTrue(refusal.startswith(f"fixed_poses: Pose JSON at '{bad}', record 0 ('degrees'): joints_rad"),
                        refusal)
        self.assertIn("reads as degrees", refusal)
        self.assertIn("No such file", _noun(fixed_poses=str(self.folder / "absent.json")).check().refusal)

    def test_a_list_is_counted_and_checked(self) -> None:
        stations = [JointStation(_joints(0), label="j0"), Pose.tool_down(400.0, 0.0, 350.0, label="down")]
        check = _noun(fixed_poses=stations).check()
        self.assertEqual(check.render().splitlines()[4], "  poses      2, 1 of them joint stations")
        self.assertIn("fixed_poses is an empty list", _noun(fixed_poses=[]).check().refusal)
        self.assertEqual(_noun(fixed_poses=[stations[1], [0.0] * 6]).check().refusal,
                         "fixed_poses[1] is a list; a station is a Pose (BASE) or a JointStation")

    def test_no_stations_and_no_hands_is_refused_because_nothing_generates_them(self) -> None:
        self.assertIn("Nothing generates stations", _noun().check().refusal)

    def test_a_file_beside_freedrive_is_the_targets_it_shows_the_way_to(self) -> None:
        path = _ROOT / "examples" / "real_robot" / "eih_fixed_stations.json"
        check = _noun(fixed_poses=str(path), freedrive=True, samples=8).check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual(check.render().splitlines()[4], f"  poses      8 guided by hand, toward the 6 stations of {path}")
        self.assertEqual((check.by_hand, check.targets), ("freedrive", 6))

    def test_adjust_needs_fixed_stations_and_is_not_freedrive(self) -> None:
        self.assertIn("adjust fine-tunes fixed stations by hand, and none were given", _noun(adjust=True).check().refusal)
        self.assertIn("Give one of them", _noun(adjust=True, freedrive=True, fixed_poses=[_STATION]).check().refusal)
        check = _noun(adjust=True, fixed_poses=[_STATION]).check()
        self.assertEqual(check.render().splitlines()[4], "  poses      1, each adjusted by hand")


class TheCliTakesAStationsFileTests(unittest.TestCase):

    def _main(self, *argv: str) -> tuple[Any, str]:
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=_app()), redirect_stdout(printed):
            try:
                code: Any = calibrate.main(["--rig", "overhead", *argv])
            except SystemExit as exc:
                code = f"exit {exc.code}"
        return code, printed.getvalue()

    def test_check_reads_the_file_and_prints_its_count(self) -> None:
        path = _ROOT / "examples" / "real_robot" / "eih_fixed_stations.json"
        code, printed = self._main("--fixed-poses", str(path), "--check")
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertIn(f"  poses      6 from {path}, 4 of them joint stations\n", printed)
        self.assertNotIn("-1", printed)

    def test_a_bad_file_exits_one_at_check(self) -> None:
        path = _write(Path(self.enterContext(tempfile.TemporaryDirectory())), [{"joints_deg": [1.0, 2.0]}])
        code, printed = self._main("--fixed-poses", str(path), "--check")
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        self.assertIn("[config] REFUSED: fixed_poses: Pose JSON at", printed)

    def test_the_file_reaches_the_sweep_options(self) -> None:
        seen: list[Any] = []
        real = HandEyeCalibration.from_config

        def spy(*args: Any, **keywords: Any) -> HandEyeCalibration:
            seen.append(keywords["options"])
            return real(*args, **keywords)

        with patch.object(HandEyeCalibration, "from_config", side_effect=spy):
            self._main("--fixed-poses", "stations.json", "--check")
        self.assertEqual(seen[0].fixed_poses, "stations.json")

    def test_the_generated_and_the_aimed_sweep_flags_are_gone(self) -> None:
        """The owner, 2026-09-24: every automatic station generator was deleted, their flags with them."""
        for flags in (("--poses", "12"), ("--aim-at=-130,-700,50",), ("--aim-distance-mm", "500"),
                      ("--closing-axis", "-y")):
            with self.subTest(flags[0]):
                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code, _ = self._main(*flags, "--freedrive", "--check")
                self.assertEqual(code, "exit 2")
                self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_freedrive_and_adjust_are_one_or_the_other(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            code, _ = self._main("--freedrive", "--adjust", "--fixed-poses", "stations.json", "--check")
        self.assertEqual(code, "exit 2")
        self.assertIn("not allowed with argument", stderr.getvalue())


class TheSweepRunsTheFileAndReportsItsCountTests(unittest.TestCase):

    def test_a_file_runs_through_run_from_json_and_the_result_counts_against_the_file(self) -> None:
        from src.camera.orchestration.camera import Camera
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.execution.calibration import CalibrationResult, PoseVerdict
        from src.robot.execution.robot import Robot
        from tests.test_calibration_sweep_events import _Streamer

        path = _ROOT / "examples" / "real_robot" / "eih_fixed_stations.json"
        camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=_Streamer())
        self.addCleanup(camera.release)
        out = self.enterContext(tempfile.TemporaryDirectory())
        calibration = HandEyeCalibration.from_parts(
            robot=Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None), camera=camera,
            robot_config=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.41"}}),
            mode="eye_to_hand", options=SweepOptions(out_dir=out, fixed_poses=str(path)))
        log = tuple(PoseVerdict(i + 1, f"look_{i}", i != 3, reason="" if i != 3 else "outside_workspace")
                    for i in range(6))
        solved = CalibrationResult(T_cam_to_base=None, rmse_mm=9.0, max_error_mm=9.0, num_samples=5,
                                   mode=MountingMode.EYE_TO_HAND, pose_log=log)
        with patch.object(CalibrationRoutine, "run_from_json", autospec=True, return_value=solved) as ran:
            report = calibration.run()
        self.assertEqual(ran.call_args.args[1], str(path))
        self.assertIn("  accepted samples  5/6\n", report.summary())
        self.assertIn("REJECTED  outside_workspace", report.summary())


# ---------------------------------------------------------------------------------------------------------------------
# Example 10
# ---------------------------------------------------------------------------------------------------------------------


class ExampleTenRollsWithTheBaseTests(unittest.TestCase):

    def test_the_ring_is_built_with_a_tangential_closing_axis_and_names_the_joint_file(self) -> None:
        (path,) = sorted((_ROOT / "examples" / "real_robot").glob("10_*.py"))
        source = path.read_text(encoding="utf-8")
        self.assertIn('closing_axis="tangential"', source)
        self.assertIn("examples/real_robot/eih_fixed_stations.json", source)
        self.assertLessEqual(len(source.splitlines()), 70)

    def test_the_ring_s_tool_heading_stays_in_one_band_where_the_default_winds(self) -> None:
        """The ring's numbers as example 10 writes them: the tool +X heading per station, in degrees."""
        board = (500.0, 0.0, 0.0)
        ring = ((0.0, 0.0, 380.0), (0.0, 120.0, 340.0), (90.0, 140.0, 300.0), (180.0, 120.0, 360.0),
                (270.0, 140.0, 320.0), (45.0, 170.0, 280.0))

        def headings(**keywords: Any) -> list[float]:
            out = []
            for bearing, radius, height in ring:
                pose = Pose.aimed_at(board[0] + radius * math.cos(math.radians(bearing)),
                                     board[1] + radius * math.sin(math.radians(bearing)), board[2] + height,
                                     target_mm=board, **keywords)
                x_axis = pose.to_matrix()[:3, 0]
                out.append(math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))))
            return out

        default, tangential = headings(), headings(closing_axis="tangential")
        self.assertEqual([round(value) for value in default], [0, -90, 0, 90, 180, -45])
        self.assertLess(max(tangential) - min(tangential), 40.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
