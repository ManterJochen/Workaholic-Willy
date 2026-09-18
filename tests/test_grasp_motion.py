"""A caller tunes how a pick moves without losing a guard.

``GraspMotion`` carries what a caller may choose about the approach, the close and the lift, and nothing that keeps a
pick safe. Both service factories build the one policy from it on the arm and hand they resolved, with the base frame
guard, the dwell gate and the jaws opened before the approach. A policy a caller built by hand drives whatever arm it
holds and carries only the guards its builder set; while ``policy=`` is accepted, one whose arm or hand is not the
service's is refused.

The cell half: ``Cell.from_tree`` reads both halves from one load, ``Cell(motion=)`` reaches the service, and
``Cell.robot`` hands the built arm and hand to the robot verbs.
"""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

from src.config import ConfigTree
from src.config.loader import ConfigError
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, AutonomousGraspService
from src.robot.execution.autonomous_grasp.cells import build_rehearsal_cell
from src.robot.execution.cell import Cell, CellNotBuilt
from src.robot.execution.motion import MotionOutcome
from src.robot.execution.robot import Robot
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.motion.grasp_motion import (
    GraspMotion,
    build_execution_policy,
    foreign_policy_refusal,
)


class _Arm:
    """An arm the factories only hold: nothing here is moved."""

    def __init__(self, dwell: object = None) -> None:
        if dwell is not None:
            self.config = SimpleNamespace(safety=SimpleNamespace(dwell=dwell))


class _Hand:
    max_width_mm = 85.0
    min_width_mm = 0.0


class _Unused:
    """A calculator or a perception source no test here calls."""

    def compute_result(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("not used")

    def acquire(self) -> None:  # pragma: no cover
        raise AssertionError("not used")


class _Resolver:
    """A frame resolver the factories only wire."""


def _dwell(on: bool = True, timeout: float = 2.5) -> SimpleNamespace:
    return SimpleNamespace(require_steady_before_motion=on, steady_timeout_s=timeout)


def _service(**kwargs: object) -> AutonomousGraspService:
    arm = kwargs.pop("arm", _Arm())
    return AutonomousGraspService.from_components(
        arm=arm, calculator=_Unused(), perception=_Unused(), mode="easy", **kwargs)  # type: ignore[arg-type]


def _console_dummy():
    return ConfigTree.from_directory(profile="console_dummy").load()


# ---------------------------------------------------------------------------------------------------


class AMotionIsWhatACallerMayChooseTests(unittest.TestCase):

    def test_unset_chooses_nothing_and_a_chosen_field_is_named(self) -> None:
        self.assertEqual({}, GraspMotion().to_dict())
        motion = GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0)
        self.assertEqual({"standoff_mm": 60.0, "close_squeeze_mm": 5.0}, motion.to_dict())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            motion.standoff_mm = 10.0  # type: ignore[misc]

    def test_skipping_the_pre_open_is_refused_because_it_is_a_guard(self) -> None:
        with self.assertRaisesRegex(TypeError, "not None"):
            GraspMotion(pre_open_width_mm=None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "opens nothing"):
            GraspMotion(pre_open_width_mm=0.0)

    def test_a_value_that_describes_no_motion_is_refused(self) -> None:
        for kwargs, error in (
            ({"standoff_mm": -1.0}, ValueError),
            ({"retreat_mm": float("nan")}, ValueError),
            ({"close_force_n": float("inf")}, ValueError),
            ({"close_speed": 1.5}, ValueError),
            ({"approach_steps": 1}, ValueError),
            ({"retreat_steps": 0}, ValueError),
            ({"standoff_mm": True}, TypeError),
            ({"standoff_mm": "60"}, TypeError),
            ({"approach_steps": 4.0}, TypeError),
            ({"align_closing_to_base_x": 1}, TypeError),
        ):
            with self.subTest(**kwargs), self.assertRaises(error):
                GraspMotion(**kwargs)  # type: ignore[arg-type]

    def test_its_standoff_and_retreat_stand_over_the_services_own(self) -> None:
        self.assertEqual((80.0, 100.0), GraspMotion().standoff_and_retreat(80.0, 100.0))
        self.assertEqual((60.0, 100.0), GraspMotion(standoff_mm=60.0).standoff_and_retreat(80.0, 100.0))


class TheOneBuilderPutsEveryGuardInTests(unittest.TestCase):

    def test_an_unset_motion_opens_the_jaws_to_the_hands_widest_and_keeps_the_gates(self) -> None:
        arm, hand = _Arm(), _Hand()
        policy = build_execution_policy(
            GraspMotion(), arm=arm, gripper=hand, standoff_mm=80.0, retreat_mm=100.0,  # type: ignore[arg-type]
            base_frame_required=True, dwell=_dwell(True, 2.5))

        self.assertIs(arm, policy.arm)
        self.assertIs(hand, policy.gripper)
        self.assertEqual(85.0, policy.pre_open_width_mm)
        self.assertTrue(policy.require_base_frame_grasp)
        self.assertTrue(policy.require_steady_before_motion)
        self.assertEqual(2.5, policy.steady_timeout_s)
        self.assertEqual((80.0, 100.0), (policy.standoff_mm, policy.retreat_mm))

    def test_what_the_caller_chose_reaches_the_policy(self) -> None:
        motion = GraspMotion(standoff_mm=60.0, retreat_mm=40.0, approach_steps=6, retreat_steps=3,
                             pre_open_width_mm=70.0, close_squeeze_mm=5.0, close_speed=0.5, close_force_n=40.0,
                             align_closing_to_base_x=True)
        policy = build_execution_policy(
            motion, arm=_Arm(), gripper=_Hand(), standoff_mm=80.0, retreat_mm=100.0,  # type: ignore[arg-type]
            base_frame_required=False)

        self.assertEqual(
            (60.0, 40.0, 6, 3, 70.0, 5.0, 0.5, 40.0, True, False),
            (policy.standoff_mm, policy.retreat_mm, policy.approach_steps, policy.retreat_steps,
             policy.pre_open_width_mm, policy.close_squeeze_mm, policy.close_speed, policy.close_force_n,
             policy.align_closing_to_base_x, policy.require_base_frame_grasp))

    def test_a_pre_open_the_hand_cannot_give_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "opens 85.0 mm at most"):
            build_execution_policy(GraspMotion(pre_open_width_mm=90.0), arm=_Arm(), gripper=_Hand(),  # type: ignore[arg-type]
                                   standoff_mm=80.0, retreat_mm=100.0, base_frame_required=False)
        with self.assertRaisesRegex(ValueError, "drives no hand"):
            build_execution_policy(GraspMotion(pre_open_width_mm=40.0), arm=_Arm(), gripper=None,  # type: ignore[arg-type]
                                   standoff_mm=80.0, retreat_mm=100.0, base_frame_required=False)


class FromComponentsTests(unittest.TestCase):

    def test_a_motion_is_built_on_the_services_arm_and_hand_with_every_guard(self) -> None:
        arm, hand = _Arm(dwell=_dwell(True, 3.0)), _Hand()
        service = _service(arm=arm, gripper=hand, frame_resolver=_Resolver(),
                           motion=GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0))
        policy = service.runtime.orchestrator.policy

        self.assertIs(arm, policy.arm)
        self.assertIs(hand, policy.gripper)
        self.assertEqual((60.0, 5.0), (policy.standoff_mm, policy.close_squeeze_mm))
        self.assertEqual(85.0, policy.pre_open_width_mm)
        self.assertTrue(policy.require_base_frame_grasp)
        self.assertTrue(policy.require_steady_before_motion)
        self.assertEqual(3.0, policy.steady_timeout_s)

    def test_without_a_resolver_the_motion_keeps_the_pre_open_and_asks_for_no_base_frame(self) -> None:
        policy = _service(gripper=_Hand(), motion=GraspMotion()).runtime.orchestrator.policy

        self.assertEqual(85.0, policy.pre_open_width_mm)
        self.assertFalse(policy.require_base_frame_grasp)
        self.assertFalse(policy.require_steady_before_motion)

    def test_a_policy_on_another_arm_or_hand_is_refused_and_names_the_motion(self) -> None:
        arm, hand = _Arm(), _Hand()
        with self.assertRaisesRegex(ValueError, r"its own arm .*motion=GraspMotion"):
            _service(arm=arm, gripper=hand, policy=GraspExecutionPolicy(arm=_Arm(), gripper=hand))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"its own hand .*motion=GraspMotion"):
            _service(arm=arm, gripper=hand, policy=GraspExecutionPolicy(arm=arm, gripper=_Hand()))  # type: ignore[arg-type]

    def test_a_policy_on_the_services_own_arm_and_hand_still_drives_verbatim(self) -> None:
        arm, hand = _Arm(), _Hand()
        own = GraspExecutionPolicy(arm=arm, gripper=hand, standoff_mm=5.0)  # type: ignore[arg-type]

        service = _service(arm=arm, gripper=hand, frame_resolver=_Resolver(), policy=own)

        self.assertIs(own, service.runtime.orchestrator.policy)
        self.assertFalse(own.require_base_frame_grasp)

    def test_both_at_once_is_a_programmers_error(self) -> None:
        arm = _Arm()
        with self.assertRaisesRegex(TypeError, "not both"):
            _service(arm=arm, policy=GraspExecutionPolicy(arm=arm), motion=GraspMotion())  # type: ignore[arg-type]

    def test_the_refusal_is_empty_for_the_services_own_handles(self) -> None:
        arm, hand = _Arm(), _Hand()
        self.assertEqual("", foreign_policy_refusal(GraspExecutionPolicy(arm=arm, gripper=hand),  # type: ignore[arg-type]
                                                    arm=arm, gripper=hand))


class FromRobotConfigTests(unittest.TestCase):
    """Through the rehearsal of ``console_dummy``, the cell the real path builds with a dummy arm and hand."""

    def test_without_a_motion_the_policy_is_the_guarded_default_it_always_was(self) -> None:
        tree = _console_dummy()
        service = build_rehearsal_cell(tree.robot)
        orchestrator = service.runtime.orchestrator
        policy = orchestrator.policy
        dwell = tree.robot.safety.dwell

        self.assertIs(orchestrator.arm, policy.arm)
        self.assertIs(orchestrator.gripper, policy.gripper)
        self.assertEqual((80.0, 100.0, 4, 1, 1.0), (policy.standoff_mm, policy.retreat_mm, policy.approach_steps,
                                                    policy.retreat_steps, policy.close_squeeze_mm))
        self.assertEqual(float(orchestrator.gripper.max_width_mm), policy.pre_open_width_mm)
        self.assertEqual(orchestrator.frame_resolver is not None, policy.require_base_frame_grasp)
        self.assertEqual(bool(dwell.require_steady_before_motion), policy.require_steady_before_motion)
        self.assertEqual(float(dwell.steady_timeout_s), policy.steady_timeout_s)

    def test_a_motion_reaches_the_policy_and_the_guards_stay(self) -> None:
        tree = _console_dummy()
        service = build_rehearsal_cell(tree.robot, motion=GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0))
        orchestrator = service.runtime.orchestrator
        policy = orchestrator.policy

        self.assertEqual((60.0, 5.0), (policy.standoff_mm, policy.close_squeeze_mm))
        self.assertEqual(float(orchestrator.gripper.max_width_mm), policy.pre_open_width_mm)
        self.assertEqual(bool(tree.robot.safety.dwell.require_steady_before_motion),
                         policy.require_steady_before_motion)

    def test_a_policy_the_caller_built_around_another_arm_is_refused(self) -> None:
        tree = _console_dummy()
        with self.assertRaisesRegex(ValueError, "motion=GraspMotion"):
            build_rehearsal_cell(tree.robot, policy=GraspExecutionPolicy(arm=_Arm()))  # type: ignore[arg-type]

    def test_both_at_once_is_a_programmers_error(self) -> None:
        tree = _console_dummy()
        with self.assertRaisesRegex(TypeError, "not both"):
            build_rehearsal_cell(tree.robot, policy=GraspExecutionPolicy(arm=_Arm()),  # type: ignore[arg-type]
                                 motion=GraspMotion())


class TheCellTests(unittest.TestCase):

    def test_a_cell_from_a_tree_reads_both_halves_and_the_root_from_the_one_load(self) -> None:
        tree = _console_dummy()
        cell = Cell.from_tree(tree, motion=GraspMotion(standoff_mm=60.0))

        self.assertIs(tree.app_config, cell.app_config)
        self.assertEqual(tree.robot, cell.robot_config)
        self.assertEqual(tree.root, cell.data_dir)
        self.assertEqual(GraspMotion(standoff_mm=60.0), cell.motion)

    def test_a_tree_that_did_not_load_and_a_non_tree_are_refused(self) -> None:
        refused = _console_dummy().with_values({"robot.grasping.max_attempts": -1})
        self.assertFalse(refused.ok)
        with self.assertRaises(ConfigError):
            Cell.from_tree(refused)
        with self.assertRaisesRegex(TypeError, r"\.load\(\)"):
            Cell.from_tree(ConfigTree.from_directory(profile="console_dummy"))  # type: ignore[arg-type]

    def test_a_rehearsal_picks_with_the_motion_it_was_given(self) -> None:
        cell = Cell.rehearsal(_console_dummy().robot, motion=GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0))
        cell.build()
        policy = cell.service.runtime.orchestrator.policy
        self.assertEqual((60.0, 5.0), (policy.standoff_mm, policy.close_squeeze_mm))

        with cell.connected() as live:
            report = live.service.pick()

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())

    def test_the_cells_robot_is_its_built_arm_and_hand_and_moves_inside_the_connection(self) -> None:
        cell = Cell.rehearsal(_console_dummy().robot)
        with self.assertRaises(CellNotBuilt):
            _ = cell.robot
        cell.build()

        robot = cell.robot

        self.assertIsInstance(robot, Robot)
        self.assertIs(cell.arm, robot.arm)
        self.assertIs(cell.gripper, robot.gripper)
        self.assertIs(MotionOutcome.REFUSED, robot.home().outcome)       # the link is not open yet
        with cell.connected():
            report = cell.robot.home()
        self.assertIs(MotionOutcome.EXECUTED, report.outcome, report.render())


if __name__ == "__main__":
    unittest.main()
