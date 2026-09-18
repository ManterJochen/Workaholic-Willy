"""A pick built from config opens the jaws to the hand's width before its first motion.

Without a pre-open the arm descends with the jaws wherever the previous close left them, and a close onto a wider part
arrives at the gripper as an opening. ``from_components`` opens them before the approach when it builds its own policy,
and ``from_robot_config``, the path a real cell and the rehearsal build, does the same for its default policy. The width
is the hand's own ``max_width_mm``, which a named hand fills at load, so the 2F-85 opens 85 mm and the Hand-E 49.99. A
policy the caller passes is left as given.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.config import load_robot_config
from src.config.schema.robot import RobotConfig
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.autonomous_grasp import AutonomousGraspService, build_rehearsal_cell
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grippers.dummy import DummyGripper
from tests.test_grasping_config_wiring import (
    _calc_and_perception,
    _FakePerception,
    _perception_frame,
    _ScriptedCalculator,
    _success_result,
    _TypedFakeArm,
)


def _dummy_tree(**gripper: Any) -> RobotConfig:
    return RobotConfig.model_validate({
        "vendor": "dummy",
        "gripper": {"vendor": "dummy", **gripper},
        "grasping": {"default_mode": "easy", "max_attempts": 1},
    })


def _policy(service: AutonomousGraspService) -> GraspExecutionPolicy:
    policy = service.runtime.orchestrator.policy
    assert policy is not None
    return policy


class _RecordingArm(DummyRobotArm):
    """The dummy arm, writing each motion onto a log it shares with the gripper."""

    def __init__(self, log: list[Any]) -> None:
        super().__init__()
        self.log = log

    def move(self, pose: Any, **kwargs: Any) -> Any:
        self.log.append("move")
        return super().move(pose, **kwargs)

    def move_to(self, pose: Any, *args: Any, **kwargs: Any) -> Any:
        self.log.append("move")
        return super().move_to(pose, *args, **kwargs)


class _RecordingGripper(DummyGripper):
    """The dummy gripper, writing each width command onto the arm's log."""

    def __init__(self, log: list[Any], *, max_width_mm: float) -> None:
        super().__init__(min_width_mm=0.0, max_width_mm=max_width_mm, initial_width_mm=10.0)
        self.log = log

    def set_width_mm(self, width_mm: float, **kwargs: Any) -> Any:
        self.log.append(("set_width_mm", float(width_mm)))
        return super().set_width_mm(width_mm, **kwargs)


class TheConfigPickOpensTheJawsTests(unittest.TestCase):
    def test_the_rehearsal_cell_opens_to_the_hand_width(self) -> None:
        service = build_rehearsal_cell(load_robot_config(profile="console_dummy"))

        self.assertEqual(85.0, _policy(service).pre_open_width_mm)

    def test_a_stated_width_is_what_opens(self) -> None:
        calc, perception = _calc_and_perception()

        service = AutonomousGraspService.from_robot_config(
            _dummy_tree(max_width_mm=49.99), calculator=calc, perception=perception,  # type: ignore[arg-type]
        )

        self.assertEqual(49.99, _policy(service).pre_open_width_mm)

    def test_the_jaws_open_before_the_first_motion(self) -> None:
        log: list[Any] = []
        arm, gripper = _RecordingArm(log), _RecordingGripper(log, max_width_mm=85.0)
        arm.connect()
        gripper.connect()
        calc, perception = _calc_and_perception()
        service = AutonomousGraspService.from_robot_config(
            _dummy_tree(max_width_mm=85.0), calculator=calc, perception=perception,  # type: ignore[arg-type]
            arm=arm, gripper=gripper,
        )

        service.pick()

        self.assertIn("move", log, log)
        first_move = log.index("move")
        widths = [entry for entry in log[:first_move] if isinstance(entry, tuple)]
        self.assertEqual([("set_width_mm", 85.0)], widths, log)

    def test_a_dummy_config_pick_with_no_resolver_still_opens(self) -> None:
        calc, perception = _calc_and_perception()

        service = AutonomousGraspService.from_robot_config(
            _dummy_tree(), calculator=calc, perception=perception,  # type: ignore[arg-type]
        )

        self.assertIsNone(service.runtime.orchestrator.frame_resolver)
        gripper = service.runtime.orchestrator.gripper
        self.assertEqual(float(gripper.max_width_mm), _policy(service).pre_open_width_mm)


class TheOtherDoorsStayAsTheyWereTests(unittest.TestCase):
    """Controls, green before and after."""

    def test_from_components_keeps_its_pre_open(self) -> None:
        from src.robot.grasping.motion.frame_resolver import IdentityFrameResolver

        gripper = DummyGripper(min_width_mm=0.0, max_width_mm=49.99)
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            gripper=gripper,
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            frame_resolver=IdentityFrameResolver(),
        )

        self.assertEqual(49.99, _policy(service).pre_open_width_mm)

    def test_an_explicit_policy_is_left_as_it_was_given(self) -> None:
        # On the service's own arm and hand: a policy around any other is refused while `policy=` is
        # accepted at all (tests/test_grasp_motion.py).
        calc, perception = _calc_and_perception()
        arm = DummyRobotArm()
        hand = DummyGripper(min_width_mm=0.0, max_width_mm=85.0)
        given = GraspExecutionPolicy(arm=arm, gripper=hand)

        service = AutonomousGraspService.from_robot_config(
            _dummy_tree(), calculator=calc, perception=perception, policy=given,  # type: ignore[arg-type]
            arm=arm, gripper=hand,
        )

        self.assertIs(given, _policy(service))
        self.assertIsNone(_policy(service).pre_open_width_mm)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
