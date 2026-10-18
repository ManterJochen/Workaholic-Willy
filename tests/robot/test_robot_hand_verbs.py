"""A robot closes its hand on a part, opens it off one, and says what it measured.

``Robot.grasp(width_mm)`` commands the width and reads back the width, whether it was measured, and what the gripper
measured about the hold. ``Robot.release()`` opens to the hand's width; a part still measured in the jaws says the
release was not confirmed. No part in the jaws is modelled (every cell since 2026-10-17): the arm is told nothing of a
grasp or a release, and its self filter holds the hand alone. Force is not a keyword. A gripper that raises is stopped
once where it can be, and the report carries the fault. Nothing is commanded on a robot with no gripper, a gripper that
holds nothing, or a link that is not open.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import numpy as np

from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.robot import Robot
from src.robot.grippers.dummy import DummyGripper


def _handling():  # noqa: ANN202
    from src.robot.execution import handling

    return handling


class _Hand:
    """A connected gripper under the test's control: what it measures, what it holds, whether it raises."""

    def __init__(self, *, hold: HoldEvidence, width_mm: float = 38.2, measured: bool = True,
                 raises: BaseException | None = None, max_width_mm: float = 85.0) -> None:
        self.hold = hold
        self.width_mm = width_mm
        self.measured = measured
        self.raises = raises
        self.max = max_width_mm
        self.commands: list[tuple[str, Any]] = []
        self.stops = 0

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def min_width_mm(self) -> float:
        return 0.0

    @property
    def max_width_mm(self) -> float:
        return self.max

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def activate(self) -> None:
        return None

    def open(self, **_: Any) -> None:
        self.set_width_mm(self.max)

    def close(self, **_: Any) -> None:
        self.set_width_mm(0.0)

    def set_width_mm(self, width_mm: float, *, speed: float | None = None, force: float | None = None) -> None:
        self.commands.append(("set_width_mm", float(width_mm)))
        if self.raises is not None:
            raise self.raises

    def get_width_mm(self) -> float:
        return self.width_mm

    def hold_evidence(self) -> HoldEvidence:
        return self.hold

    def width_is_measured(self) -> bool:
        return self.measured

    def stop(self) -> None:
        self.stops += 1


def _dummy_robot(gripper: Any) -> Robot:
    arm = DummyRobotArm()
    arm.connect()
    return Robot.from_parts(arm=arm, gripper=gripper, lock_key=None)


def _planned_arm() -> Any:
    """A cuRobo arm over the fake transport, built as a vendor builds it."""
    from tests._arm_kit import HERE, kit_tree, planned_arm
    from tests._fake_transport import fake_transport

    return planned_arm(kit_tree("curobo"), transport=fake_transport(joints=HERE))


class GraspTests(unittest.TestCase):
    def test_a_detected_grasp_reads_back_the_measured_width(self) -> None:
        handling = _handling()
        robot = Robot.from_parts(arm=_planned_arm(), gripper=_Hand(hold=HoldEvidence.HELD, width_mm=38.2),
                                 lock_key=None)

        report = robot.grasp(40.0)

        self.assertIs(handling.HandOutcome.GRASPED, report.outcome, report.render())
        self.assertIs(HoldEvidence.HELD, report.hold)
        self.assertEqual(38.2, report.reported_width_mm)
        self.assertTrue(report.width_measured)
        self.assertIn("38.20 mm measured", report.render())
        self.assertIn("hold HELD", report.render())

    def test_an_empty_grasp_holds_nothing(self) -> None:
        handling = _handling()
        robot = Robot.from_parts(arm=_planned_arm(), gripper=_Hand(hold=HoldEvidence.EMPTY), lock_key=None)

        report = robot.grasp(40.0)

        self.assertIs(handling.HandOutcome.NOTHING_HELD, report.outcome)
        self.assertFalse(report.ok)

    def test_a_dummy_gripper_grasp_says_nothing_was_measured(self) -> None:
        handling = _handling()
        gripper = DummyGripper(max_width_mm=85.0)
        gripper.connect()
        robot = Robot.from_parts(arm=_planned_arm(), gripper=gripper, lock_key=None)

        report = robot.grasp(40.0)

        self.assertIs(handling.HandOutcome.GRASPED, report.outcome, report.render())
        self.assertIs(HoldEvidence.UNMEASURED, report.hold)
        self.assertFalse(report.width_measured)
        self.assertIn("hold not measured", report.render())
        self.assertIn("width not measured", report.render())


class ReleaseTests(unittest.TestCase):
    def test_a_release_reading_empty_opens_to_the_hand_width(self) -> None:
        handling = _handling()
        hand = _Hand(hold=HoldEvidence.EMPTY, max_width_mm=49.99)
        robot = Robot.from_parts(arm=_planned_arm(), gripper=hand, lock_key=None)

        report = robot.release()

        self.assertIs(handling.HandOutcome.RELEASED, report.outcome, report.render())
        self.assertEqual([("set_width_mm", 49.99)], hand.commands)

    def test_a_release_still_held_is_not_confirmed(self) -> None:
        from src.config.schema.robot import GripperConfig
        from tests.robot.grippers.test_gripper_commands_the_number_it_promised import _gripper

        handling = _handling()
        robotiq, driver = _gripper(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
        driver.obj = 1  # STOPPED_OPENING: something is still in the way of the fingers
        robot = Robot.from_parts(arm=_planned_arm(), gripper=robotiq, lock_key=None)

        report = robot.release()

        self.assertIs(handling.HandOutcome.RELEASE_NOT_CONFIRMED, report.outcome, report.render())
        self.assertIs(HoldEvidence.HELD, report.hold)

    def test_a_jaw_io_release_with_its_pin_high_is_unmeasured_and_says_so(self) -> None:
        from src.robot.grippers.jaw_io import JawIOGripper
        from tests.robot.grippers.test_jaw_io_gripper import CLOSE_PIN, PART_SW, FakeIO

        handling = _handling()
        jaws = JawIOGripper(FakeIO({PART_SW: True}), close_output_pin=CLOSE_PIN, part_present_input_pin=PART_SW,
                            sleep=lambda _s: None)
        jaws.connect()
        robot = Robot.from_parts(arm=_planned_arm(), gripper=jaws, lock_key=None)

        report = robot.release()

        self.assertIs(handling.HandOutcome.RELEASED, report.outcome, report.render())
        self.assertIs(HoldEvidence.UNMEASURED, report.hold)
        self.assertIn("release not checked", report.render())


class TheSelfFilterHoldsTheHandAloneTests(unittest.TestCase):
    def _point_past_the_tips(self, arm: Any) -> np.ndarray:
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm
        from src.robot.safety.planning.hand import HAND_APPROACH_IN_TOOL0
        from tests.robot.safety.planning.test_self_filter_envelope import _hand, _place, _tip_mm

        joints = np.asarray(arm.get_joint_positions().values, dtype=np.float64)
        # UR-specific: the kit arm is a UR5e description, so its links are the UR5e's DH chain.
        frames = ur_link_transforms_mm("ur5e", joints)
        assert frames is not None
        return _place(frames[6], (np.asarray(HAND_APPROACH_IN_TOOL0) * (_tip_mm(_hand()) + 100.0))[None, :])

    @staticmethod
    def _filtered(arm: Any, point: np.ndarray) -> bool:
        from src.robot.safety.planning.perceived import SelfBody
        from tests.robot.safety.planning.test_self_filter_envelope import _PADDING_MM

        envelope = arm._self_envelope()
        assert envelope is not None
        return bool(SelfBody.from_frames(envelope.frames_mm, envelope.capsules, padding_mm=_PADDING_MM)
                    .contains(point)[0])

    def test_a_point_past_the_fingertips_is_filtered_neither_before_nor_after_a_grasp(self) -> None:
        """No part in the jaws is modelled, so the self filter grows nothing past the fingertips when the jaws close on
        one: what the hand holds is the camera's to see, as on the owner's cell, which modelled none."""
        arm = _planned_arm()
        robot = Robot.from_parts(arm=arm, gripper=_Hand(hold=HoldEvidence.HELD, width_mm=60.0), lock_key=None)
        point = self._point_past_the_tips(arm)
        self.assertFalse(self._filtered(arm, point))

        robot.grasp(60.0)
        self.assertFalse(self._filtered(arm, point), "a grasp grew the self filter past the fingertips")
        robot.gripper.hold = HoldEvidence.EMPTY  # type: ignore[union-attr]
        robot.release()
        self.assertFalse(self._filtered(arm, point))


class ArmsThatCarryNoPlannerTests(unittest.TestCase):
    def test_a_sim_mock_and_a_dummy_grasp_as_every_arm_does(self) -> None:
        from tests.robot.drivers.test_camera_world_on_every_driver import _sim_mock

        handling = _handling()
        for label, arm in (("sim mock", _sim_mock()), ("dummy", DummyRobotArm())):
            with self.subTest(label):
                arm.connect()
                robot = Robot.from_parts(arm=arm, gripper=_Hand(hold=HoldEvidence.HELD), lock_key=None)
                report = robot.grasp(40.0)
                self.assertIs(handling.HandOutcome.GRASPED, report.outcome, report.render())


class FaultsAndRefusalsTests(unittest.TestCase):
    def test_a_gripper_that_raises_is_stopped_once_and_the_fault_is_reported(self) -> None:
        handling = _handling()
        hand = _Hand(hold=HoldEvidence.HELD, raises=RuntimeError("the fingers did not reach 40.0 mm"))
        robot = _dummy_robot(hand)

        report = robot.grasp(40.0)

        self.assertIs(handling.HandOutcome.GRIPPER_FAULT, report.outcome)
        self.assertEqual(1, hand.stops)
        self.assertIn("the fingers did not reach 40.0 mm", report.error)

    def test_nothing_is_commanded_on_a_robot_that_cannot_hold(self) -> None:
        from src.robot.core import GripperVendor
        from src.robot.grippers import create_gripper

        handling = _handling()
        idle = _Hand(hold=HoldEvidence.HELD)
        not_connected = DummyRobotArm()
        null = create_gripper(GripperVendor.NONE)
        cases = (
            ("no gripper", _dummy_robot(None)),
            ("a gripper that holds nothing", _dummy_robot(null)),
            ("an arm not connected", Robot.from_parts(arm=not_connected, gripper=idle, lock_key=None)),
        )
        for label, robot in cases:
            with self.subTest(label):
                for verb in (lambda r: r.grasp(40.0), lambda r: r.release()):
                    report = verb(robot)
                    self.assertIs(handling.HandOutcome.REFUSED, report.outcome, report.render())
                    self.assertTrue(report.error)
        self.assertEqual([], idle.commands)

    def test_is_holding_commands_nothing(self) -> None:
        hand = _Hand(hold=HoldEvidence.HELD)
        robot = _dummy_robot(hand)
        self.assertIs(HoldEvidence.HELD, robot.is_holding())
        self.assertIs(HoldEvidence.UNMEASURED, _dummy_robot(None).is_holding())
        self.assertEqual([], hand.commands)

    def test_the_report_describes_itself(self) -> None:
        robot = _dummy_robot(_Hand(hold=HoldEvidence.HELD))
        for report in (robot.grasp(40.0), robot.release(), _dummy_robot(None).grasp(40.0)):
            with self.subTest(report.outcome.value):
                text = report.render()
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))
                data = json.loads(json.dumps(report.to_dict()))
                self.assertEqual(report.outcome.value, data["outcome"])
                self.assertEqual(report.verb.value, data["verb"])
                self.assertNotIn("payload", data, "no part in the jaws is modelled")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
