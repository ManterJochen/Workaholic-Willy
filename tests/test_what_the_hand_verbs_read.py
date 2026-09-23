"""What the hand verbs read before they command anything, from core and from the drivers.

``Robot.grasp``, ``release``, ``pick`` and ``place`` refuse before their first command wherever the answer is
already known: a camera world a motion would be refused for, a line the arm cannot keep, a gripper that cannot say what
it holds. None of that may cost a planner start, a connection or a motion, and none of it changes what a driver does.
So each reading is a capability an arm or a gripper opts into, read here against what the driver's own verbs do.
"""

from __future__ import annotations

import unittest

from src.contracts import UNSET
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldUse
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.ur.arm import URRobotArm
from tests.test_camera_world_on_every_driver import (
    _JOINTS,
    _sim_mock,
    _sim_unconnected,
    _ur_curobo,
    _ur_ik,
)
from tests.test_ur_arm import _arm as _ur_arm
from tests.test_ur_arm import _pose

_BENCH = "bench check, no cameras mounted"


class TheCameraWorldIsReadFromCoreTests(unittest.TestCase):
    def test_the_weakest_stamp_is_read_from_core(self) -> None:
        from src.robot.core.camera_world import CameraWorldStamp, weakest_camera_world
        from src.robot.grasping.motion import execution_policy

        planned = CameraWorldStamp.planned(cameras=("overhead",), captured_at_s=1.0)
        missing = CameraWorldStamp.missing("no world")
        self.assertIs(missing, weakest_camera_world((planned, missing, planned)))
        self.assertIs(planned, weakest_camera_world((planned,)))
        self.assertIsNone(weakest_camera_world(()))
        self.assertIs(weakest_camera_world, execution_policy.weakest_camera_world)
        self.assertIn("weakest_camera_world", execution_policy.__all__)

    def test_the_camera_world_a_move_would_carry_is_read_before_moving(self) -> None:
        from src.robot.core.camera_world import ReadsCameraWorld

        bare = _ur_arm("curobo")
        read = bare.camera_world_for_move()
        self.assertIsInstance(bare, ReadsCameraWorld)
        self.assertIs(CameraWorldUse.MISSING, read.use)
        self.assertFalse(bare.live_camera_world_wired())
        self.assertIsNone(bare._curobo_ur, "reading the camera world started a planner")

        cases: tuple[tuple[str, object, object], ...] = (
            ("ur curobo, no world", _ur_curobo, UNSET),
            ("ur curobo, declined", _ur_curobo, CameraWorldDecline(_BENCH)),
            ("ur ik", _ur_ik, UNSET),
            ("sim curobo, no world", _sim_unconnected, UNSET),
            ("sim mock", _sim_mock, UNSET),
        )
        for label, build, keyword in cases:
            with self.subTest(label):
                arm = build()  # type: ignore[operator]
                assert isinstance(arm, (URRobotArm, IsaacRobotArm))
                self.assertIsInstance(arm, ReadsCameraWorld)
                before = arm.camera_world_for_move(keyword)  # type: ignore[arg-type]
                if keyword is UNSET:
                    moved = arm.move(_pose())
                else:
                    moved = arm.move(_pose(), camera_world=keyword)  # type: ignore[arg-type]
                self.assertEqual(moved.camera_world.use, before.use, moved.message)
                self.assertEqual(moved.camera_world.reason, before.reason)

    def test_a_block_decline_is_read_as_declined(self) -> None:
        arm = _ur_curobo()
        with arm.without_camera_world(_BENCH):
            read = arm.camera_world_for_move()
            moved = arm.move_to_joints(_JOINTS)
        self.assertIs(CameraWorldUse.DECLINED, read.use)
        self.assertEqual(moved.camera_world, read)

    def test_a_wired_world_is_reported_without_asking_it(self) -> None:
        wired = _ur_curobo(live_world=object())
        self.assertTrue(wired.live_camera_world_wired())
        self.assertFalse(_sim_mock().live_camera_world_wired())


# ---------------------------------------------------------------------------------------------------
# Hold evidence and measured widths
# ---------------------------------------------------------------------------------------------------


def _hold():  # noqa: ANN202
    from src.robot.core.gripper import HoldEvidence

    return HoldEvidence


def _classes_that_detect_without_evidence(source: str) -> list[str]:
    """Every class in ``source`` that defines ``is_object_detected`` and no ``hold_evidence``."""
    import ast

    missing = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            if "is_object_detected" in methods and "hold_evidence" not in methods:
                missing.append(node.name)
    return missing


class HoldEvidenceTests(unittest.TestCase):
    def test_a_robotiq_reads_gobj_as_evidence(self) -> None:
        from src.config.schema.robot import GripperConfig
        from src.robot.grippers.robotiq import GripperController
        from tests.test_gripper_commands_the_number_it_promised import _gripper

        HoldEvidence = _hold()
        g, driver = _gripper(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
        for status, expected in ((0, HoldEvidence.UNMEASURED), (1, HoldEvidence.HELD), (2, HoldEvidence.HELD),
                                 (3, HoldEvidence.EMPTY)):
            with self.subTest(gobj=status):
                driver.obj = status
                self.assertIs(expected, g.hold_evidence())

        class _Mute:
            def connect(self, ip: str, port: int) -> None:
                return None

            def activate_if_needed(self) -> None:
                return None

            def disconnect(self) -> None:
                return None

        mute = GripperController(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0),
                                 ip="127.0.0.1", driver_factory=lambda: _Mute())
        mute.connect()
        self.assertIs(HoldEvidence.UNMEASURED, mute.hold_evidence())

    def test_a_vacuum_without_a_switch_is_unmeasured_even_while_engaged(self) -> None:
        from tests.test_vacuum_gripper import FakeIO
        from tests.test_vacuum_gripper import _gripper as _vacuum

        HoldEvidence = _hold()
        blind = _vacuum(FakeIO())
        blind.connect()
        blind.set_width_mm(0.0)
        self.assertTrue(blind.is_object_detected(), "the command answers the old question")
        self.assertIs(HoldEvidence.UNMEASURED, blind.hold_evidence())

        io = FakeIO({3: True})
        switched = _vacuum(io, vacuum_ok_input_pin=3)
        switched.connect()
        switched.set_width_mm(0.0)
        self.assertIs(HoldEvidence.HELD, switched.hold_evidence())
        io.inputs[3] = False
        self.assertIs(HoldEvidence.EMPTY, switched.hold_evidence())

    def test_open_jaws_are_unmeasured_even_with_a_part_pin_reading_high(self) -> None:
        from src.robot.grippers.jaw_io import JawIOGripper
        from tests.test_jaw_io_gripper import CLOSE_PIN, CLOSED_SW, OPEN_SW, PART_SW, FakeIO

        HoldEvidence = _hold()
        io = FakeIO({PART_SW: True})
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, part_present_input_pin=PART_SW, sleep=lambda _s: None)
        g.connect()
        g.set_width_mm(g.max_width_mm)
        self.assertIs(HoldEvidence.UNMEASURED, g.hold_evidence())
        g.set_width_mm(0.0)
        self.assertIs(HoldEvidence.HELD, g.hold_evidence())
        io.inputs[PART_SW] = False
        self.assertIs(HoldEvidence.EMPTY, g.hold_evidence())

        reed_io = FakeIO()
        reed = JawIOGripper(reed_io, close_output_pin=CLOSE_PIN, closed_confirm_input_pin=CLOSED_SW,
                            open_confirm_input_pin=OPEN_SW, sleep=lambda _s: None)
        reed.connect()
        reed.set_width_mm(0.0)
        for closed, opened, expected in ((False, False, HoldEvidence.HELD), (True, False, HoldEvidence.EMPTY)):
            with self.subTest(closed=closed, opened=opened):
                reed_io.inputs[CLOSED_SW], reed_io.inputs[OPEN_SW] = closed, opened
                self.assertIs(expected, reed.hold_evidence())

    def test_an_onrobot_reads_its_status_word(self) -> None:
        from src.robot.grippers.onrobot_modbus import RGStatus
        from tests.test_onrobot_gripper import _gripper as _onrobot

        HoldEvidence = _hold()
        held, _ = _onrobot(status=RGStatus.GRIP_DETECTED)
        held.connect()
        self.assertIs(HoldEvidence.HELD, held.hold_evidence())
        empty, _ = _onrobot(status=RGStatus.AT_POSITION)
        empty.connect()
        self.assertIs(HoldEvidence.EMPTY, empty.hold_evidence())

    def test_a_suction_gripper_without_a_view_is_unmeasured(self) -> None:
        from src.robot.grippers.sim.suction_gripper import IsaacSuctionGripper

        HoldEvidence = _hold()
        mock = IsaacSuctionGripper(session=object(), gripper_prim_path=None, mock_mode=True)
        mock.connect()
        mock.close()
        self.assertTrue(mock.is_object_detected(), "the command answers the old question")
        self.assertIs(HoldEvidence.UNMEASURED, mock.hold_evidence())

        viewed = IsaacSuctionGripper(session=object(), gripper_prim_path="/World/Cup", mock_mode=False)
        viewed._connected = True
        self.assertIs(HoldEvidence.UNMEASURED, viewed.hold_evidence())
        viewed._view = object()  # type: ignore[assignment]
        for bonded, expected in ((True, HoldEvidence.HELD), (False, HoldEvidence.EMPTY)):
            with self.subTest(bonded=bonded):
                viewed._is_closed = lambda bonded=bonded: bonded  # type: ignore[method-assign]
                self.assertIs(expected, viewed.hold_evidence())

    def test_a_gripper_without_the_capability_reads_unmeasured(self) -> None:
        from src.robot.core.gripper import hold_evidence_of
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.robot.drivers.sim.session import IsaacSimSession
        from src.robot.grippers.dummy import DummyGripper
        from src.robot.grippers.sim.gripper import IsaacGripper

        HoldEvidence = _hold()
        mock = IsaacGripper(session=IsaacSimSession(SimRobotConfig(mock_mode=True)), gripper_prim_path="/World/G",
                            mock_mode=True)
        self.assertIs(HoldEvidence.UNMEASURED, hold_evidence_of(DummyGripper()))
        self.assertIs(HoldEvidence.UNMEASURED, hold_evidence_of(mock))

    def test_every_detecting_gripper_reports_evidence(self) -> None:
        import pathlib

        control = "class Old:\n    def is_object_detected(self):\n        return True\n"
        self.assertEqual(["Old"], _classes_that_detect_without_evidence(control), "the scan finds nothing")
        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "robot" / "grippers"
        files = sorted(root.rglob("*.py"))
        self.assertGreater(len(files), 5)
        missing = [f"{path.name}:{name}" for path in files
                   for name in _classes_that_detect_without_evidence(path.read_text(encoding="utf-8"))]
        self.assertEqual([], missing)

    def test_which_widths_are_measured(self) -> None:
        from src.config.schema.robot import GripperConfig
        from src.robot.core.gripper import width_is_measured_of
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.robot.drivers.sim.session import IsaacSimSession
        from src.robot.grippers.dummy import DummyGripper
        from src.robot.grippers.jaw_io import JawIOGripper
        from src.robot.grippers.sim.gripper import IsaacGripper
        from tests.test_gripper_commands_the_number_it_promised import _gripper as _robotiq
        from tests.test_jaw_io_gripper import CLOSE_PIN, FakeIO
        from tests.test_onrobot_gripper import _gripper as _onrobot

        robotiq, _ = _robotiq(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
        onrobot, _ = _onrobot()
        rows = (
            ("robotiq", robotiq, True),
            ("onrobot", onrobot, True),
            ("isaac off mock", IsaacGripper(session=IsaacSimSession(SimRobotConfig()), gripper_prim_path="/World/G",
                                            mock_mode=False), True),
            ("isaac mock", IsaacGripper(session=IsaacSimSession(SimRobotConfig(mock_mode=True)),
                                        gripper_prim_path="/World/G", mock_mode=True), False),
            ("jaw_io", JawIOGripper(FakeIO(), close_output_pin=CLOSE_PIN, sleep=lambda _s: None), False),
            ("dummy", DummyGripper(), False),
        )
        for label, gripper, expected in rows:
            with self.subTest(label):
                self.assertIs(expected, width_is_measured_of(gripper))

    def test_stoppable_gripper_is_exported_from_core(self) -> None:
        from src.robot import core
        from src.robot.core import StoppableGripper
        from src.robot.core.gripper import __all__ as gripper_all
        from src.robot.grippers.robotiq import GripperController

        self.assertIn("StoppableGripper", gripper_all)
        self.assertIs(StoppableGripper, core.StoppableGripper)
        self.assertTrue(issubclass(GripperController, StoppableGripper))

# ---------------------------------------------------------------------------------------------------
# The payload model
# ---------------------------------------------------------------------------------------------------


def running_normally(conn: object) -> None:
    """Make a MagicMock UR connection read a controller that can move: RUNNING, NORMAL safety, no stop.

    The hand verbs and the grasp policy ask the controller before any gripper command (2026-09-23), and a bare
    MagicMock answers with mocks, which read as a controller in an unknown mode that cannot move.
    """
    conn.get_robot_mode.return_value = 7  # type: ignore[attr-defined]  # UR RUNNING
    conn.get_safety_mode.return_value = 1  # type: ignore[attr-defined]  # UR NORMAL
    conn.is_protective_stopped.return_value = False  # type: ignore[attr-defined]
    conn.is_emergency_stopped.return_value = False  # type: ignore[attr-defined]
    conn.dashboard_safety_status.return_value = ""  # type: ignore[attr-defined]


def _payload_ur(*, enabled: bool = True, planner_answer: bool = True, planner_kind: str = "curobo") -> URRobotArm:
    """The UR of tests/test_self_filter_envelope.py that attaches a part, its planner a double answering ``planner_answer``."""
    from unittest.mock import MagicMock

    from src.config.schema.robot import RobotConfig

    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": planner_kind},
        "gripper": {"model": "robotiq_2f85"},
        "safety": {
            "payload": {"enforce": False},
            "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                "payload": {"enabled": enabled, "length_mm": 120.0, "lateral_margin_mm": 10.0},
            },
        },
    })
    arm = URRobotArm(config)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = [0.0, -1.5, 1.5, 0.0, 1.5, 0.0]
    running_normally(arm._conn)
    if enabled and planner_kind == "curobo":
        planner = MagicMock()
        planner.attach_payload.return_value = planner_answer
        planner.detach_payload.return_value = True
        arm._curobo_ur = planner
    return arm


class ThePayloadModelTests(unittest.TestCase):
    def test_the_ur_says_why_it_models_no_payload_without_starting_a_planner(self) -> None:
        from src.robot.core.arm_capabilities import CarriesPayload, PayloadModel

        arm = _payload_ur(enabled=False)
        self.assertIsInstance(arm, CarriesPayload)
        reason = arm.payload_declined_reason()
        assert reason is not None
        self.assertIn("safety.planning_world.payload.enabled", reason)
        self.assertIs(PayloadModel.NONE, arm.payload_model())
        self.assertIsNone(arm._curobo_ur, "asking why started a planner")

        ik = _payload_ur(planner_kind="ik")
        ik_reason = ik.payload_declined_reason()
        assert ik_reason is not None
        self.assertIn("robot.ur.motion_planner", ik_reason)

        self.assertIsNone(_payload_ur().payload_declined_reason())

    def test_a_declined_attach_is_filter_only(self) -> None:
        from src.robot.core.arm_capabilities import PayloadModel

        declined = _payload_ur(planner_answer=False)
        self.assertFalse(declined.attach_payload(60.0))
        self.assertIs(PayloadModel.FILTER_ONLY, declined.payload_model())
        declined.detach_payload()
        self.assertIs(PayloadModel.NONE, declined.payload_model())

        carried = _payload_ur(planner_answer=True)
        self.assertTrue(carried.attach_payload(60.0))
        self.assertIs(PayloadModel.PLANNER_AND_FILTER, carried.payload_model())
        carried.detach_payload()
        self.assertIs(PayloadModel.NONE, carried.payload_model())


# ---------------------------------------------------------------------------------------------------
# Line motion
# ---------------------------------------------------------------------------------------------------


class LineMotionTests(unittest.TestCase):
    def test_line_motion_per_driver(self) -> None:
        from unittest.mock import MagicMock, patch

        from src.robot.core.arm_capabilities import LineMotion, line_motion_of
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.robot.safety.preflight import SafetyPreflight
        from tests.test_ur_checked_linear import _arm as _line_ur
        from tests.test_ur_checked_linear import _mesh_backend_available

        def sim(planner: str, *, judge: "str | None | bool" = False) -> IsaacRobotArm:
            arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                                               motion_planner=planner))  # type: ignore[arg-type]
            if judge is not False:
                guard = MagicMock()
                guard.path_judge_refusal.return_value = judge
                arm._preflight = guard
            return arm

        sentence = "no self_collision guard is wired, so nothing here can judge a path"
        rows: list[tuple[str, object, LineMotion]] = [
            ("ur ik", _ur_ik(), LineMotion.CONTROLLER_LINE),
            ("sim mock", _sim_mock(), LineMotion.TELEPORT),
            ("sim curobo, judged", sim("curobo", judge=None), LineMotion.CHECKED),
            ("sim curobo, no preflight", sim("curobo"), LineMotion.NOT_KEPT),
            ("sim curobo, a refusal", sim("curobo", judge=sentence), LineMotion.NOT_KEPT),
            ("sim ik", sim("ik", judge=None), LineMotion.NOT_KEPT),
            ("sim rmpflow", sim("rmpflow", judge=None), LineMotion.NOT_KEPT),
            ("dummy", DummyRobotArm(), LineMotion.TELEPORT),
        ]
        refused_ur = _line_ur()
        with patch.object(SafetyPreflight, "path_judge_refusal", return_value=sentence):
            reading = line_motion_of(refused_ur)
            assert reading is not None
            self.assertIs(LineMotion.NOT_KEPT, reading.motion)
            self.assertEqual(sentence, reading.reason)
        if _mesh_backend_available():
            rows.append(("ur curobo, judged", _line_ur(), LineMotion.CHECKED))
        else:  # pragma: no cover (a box without Coal)
            self.skipTest("no exact mesh backend on this box: the CHECKED row of the UR cannot be read")
        for label, arm, expected in rows:
            with self.subTest(label):
                read = line_motion_of(arm)
                assert read is not None, label
                self.assertIs(expected, read.motion, read.reason)
                self.assertTrue(read.reason, "a reading says why")
        sim_refused = line_motion_of(sim("curobo", judge=sentence))
        assert sim_refused is not None
        self.assertEqual(sentence, sim_refused.reason)
        self.assertIsNone(line_motion_of(object()))

    def test_the_path_judge_refusal_is_the_gate_sentence(self) -> None:
        from unittest.mock import MagicMock

        from src.robot.core import MotionCommand
        from src.robot.safety.preflight import (
            SafetyPreflight,
            exact_mesh_path_refusal,
            no_path_guard_refusal,
        )

        empty = SafetyPreflight([])
        self.assertEqual(no_path_guard_refusal(), empty.path_judge_refusal(None))
        gated = empty.gate_planned_path([[0.0] * 6, [0.1] * 6], arm=None, command=MotionCommand.MOVE_JOINTS)
        assert gated is not None
        self.assertIn(no_path_guard_refusal(), gated.message or "")

        guard = MagicMock()
        guard.exact_mesh_engine.return_value = None
        blind = SafetyPreflight([])
        blind._path_authority = lambda arm: guard  # type: ignore[method-assign]
        self.assertEqual(exact_mesh_path_refusal(), blind.path_judge_refusal(None))
        blind_gated = blind.gate_planned_path([[0.0] * 6, [0.1] * 6], arm=None, command=MotionCommand.MOVE_JOINTS)
        assert blind_gated is not None
        self.assertIn(exact_mesh_path_refusal(), blind_gated.message or "")

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
