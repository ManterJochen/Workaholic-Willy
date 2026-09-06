"""Tests for Phase S7 swept-volume validators."""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision.gripper_model import (
    ParallelJawGripperModel,
)
from src.robot.grasping.planning import GraspPose
from src.robot.grasping.motion.trajectory_safety import (
    AcceptAllTrajectorySafetyCheck,
    ApproachPathOutcome,
    ApproachPathPolicy,
    TrajectorySafetyCheck,
    validate_approach_and_retreat,
    validate_approach_path,
    validate_retreat_path,
)


# Approach axis is +Z (third column). Pre-grasp lands at -Z offset.
_IDENTITY_ROT = np.eye(3, dtype=np.float64)


def _grasp(position_mm=(0.0, 0.0, 0.0), grip_width_mm: float = 40.0) -> GraspPose:
    return GraspPose(
        position_mm=np.asarray(position_mm, dtype=np.float64),
        rotation_matrix=_IDENTITY_ROT.copy(),
        grip_width_mm=grip_width_mm,
        score=0.5,
        confidence=0.5,
        contacts=(
            np.array([-grip_width_mm / 2.0, 0.0, 0.0]),
            np.array([+grip_width_mm / 2.0, 0.0, 0.0]),
        ),
        frame=Frame.BASE,
        metadata={},
    )


def _reject_pose_at_fraction(threshold: float):
    """Build a TrajectorySafetyCheck that rejects poses past ``threshold``.

    threshold is along +Z so it works for the approach (which moves
    from negative-Z standoff to grasp at z=0).
    """

    class _Check:
        def is_safe(self, pose: GraspPose) -> bool:
            return float(pose.position_mm[2]) <= threshold

    return _Check()


class ApproachPathPolicyTests(unittest.TestCase):
    def test_defaults_are_safe(self) -> None:
        p = ApproachPathPolicy()
        self.assertGreaterEqual(p.num_approach_samples, 2)

    def test_rejects_negative_standoff(self) -> None:
        with self.assertRaises(ValueError):
            ApproachPathPolicy(standoff_mm=-1.0)

    def test_rejects_invalid_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            ApproachPathPolicy(num_approach_samples=1)
        with self.assertRaises(ValueError):
            ApproachPathPolicy(num_retreat_samples=1)

    def test_rejects_zero_retreat_direction(self) -> None:
        with self.assertRaises(ValueError):
            ApproachPathPolicy(retreat_direction=(0.0, 0.0, 0.0))

    def test_rejects_negative_margin(self) -> None:
        with self.assertRaises(ValueError):
            ApproachPathPolicy(collision_margin_mm=-1.0)

    def test_zero_samples_disables_validator(self) -> None:
        policy = ApproachPathPolicy(num_approach_samples=0)
        report = validate_approach_path(_grasp(), policy=policy)
        self.assertIs(report.outcome, ApproachPathOutcome.SKIPPED)


class ValidateApproachPathTests(unittest.TestCase):
    def test_no_obstacles_yields_no_obstacles_outcome(self) -> None:
        report = validate_approach_path(_grasp())
        self.assertIs(report.outcome, ApproachPathOutcome.NO_OBSTACLES)
        self.assertTrue(report.is_clear)
        # Every sampled step should have collision_count == 0.
        for step in report.steps:
            self.assertEqual(step.collision_count, 0)

    def test_clear_path_with_distant_obstacles(self) -> None:
        # Points well outside the gripper envelope.
        far_points = np.array(
            [
                [500.0, 0.0, 0.0],
                [-500.0, 0.0, 0.0],
                [0.0, 500.0, 0.0],
            ]
        )
        report = validate_approach_path(_grasp(), obstacle_points_mm=far_points)
        self.assertIs(report.outcome, ApproachPathOutcome.CLEAR)
        self.assertIsNone(report.first_blocking_index)

    def test_blocked_when_obstacle_intersects_pre_grasp(self) -> None:
        # The pre-grasp sits at world z = -standoff_mm = -80 with
        # identity rotation. The negative-x finger box at that
        # pose spans world x in [-32, -18] (default model). Place
        # an obstacle inside it.
        policy = ApproachPathPolicy(standoff_mm=80.0)
        obstacle = np.array([[-25.0, 0.0, -80.0]])
        report = validate_approach_path(
            _grasp(), obstacle_points_mm=obstacle, policy=policy
        )
        self.assertIs(report.outcome, ApproachPathOutcome.BLOCKED)
        self.assertEqual(report.first_blocking_index, 0)
        self.assertGreater(report.steps[0].collision_count, 0)

    def test_blocked_when_obstacle_intersects_final_grasp(self) -> None:
        # A point inside the negative-x finger box at the final grasp
        # pose (world x in [-32, -18]).
        obstacle = np.array([[-25.0, 0.0, 0.0]])
        report = validate_approach_path(_grasp(), obstacle_points_mm=obstacle)
        self.assertIs(report.outcome, ApproachPathOutcome.BLOCKED)
        assert report.first_blocking_index is not None
        self.assertGreaterEqual(report.first_blocking_index, 0)

    def test_safety_check_rejection_blocks_sweep(self) -> None:
        # Reject any sample with z > -40 -> half the sweep is rejected.
        report = validate_approach_path(
            _grasp(),
            safety_check=_reject_pose_at_fraction(-40.0),
        )
        self.assertIs(report.outcome, ApproachPathOutcome.BLOCKED)
        assert report.first_blocking_index is not None
        self.assertTrue(report.steps[report.first_blocking_index].safety_rejected)

    def test_non_default_safety_check_marks_outcome_clear(self) -> None:
        # No obstacles + a real (passing) safety check -> CLEAR, not
        # NO_OBSTACLES.
        class _AlwaysSafe:
            def is_safe(self, pose: GraspPose) -> bool:  # noqa: ARG002
                return True

        report = validate_approach_path(_grasp(), safety_check=_AlwaysSafe())
        self.assertIs(report.outcome, ApproachPathOutcome.CLEAR)

    def test_rejects_non_grasp_pose(self) -> None:
        with self.assertRaises(TypeError):
            validate_approach_path("not a grasp")  # type: ignore[arg-type]


class ValidateRetreatPathTests(unittest.TestCase):
    def test_retreat_clear_when_obstacles_below(self) -> None:
        # Retreat moves along +Z by default; obstacles way below
        # cannot block.
        report = validate_retreat_path(
            _grasp(),
            obstacle_points_mm=np.array([[0.0, 0.0, -500.0]]),
        )
        self.assertIs(report.outcome, ApproachPathOutcome.CLEAR)

    def test_retreat_blocked_by_obstacle_above(self) -> None:
        policy = ApproachPathPolicy(retreat_mm=100.0, num_retreat_samples=4)
        # Place an obstacle inside the negative-x finger box at
        # world (-25, 0, 50). As the gripper retreats along +Z it
        # will pass through this point with its finger volume.
        report = validate_retreat_path(
            _grasp(),
            obstacle_points_mm=np.array([[-25.0, 0.0, 50.0]]),
            policy=policy,
        )
        self.assertIs(report.outcome, ApproachPathOutcome.BLOCKED)

    def test_zero_retreat_samples_skips(self) -> None:
        report = validate_retreat_path(
            _grasp(),
            policy=ApproachPathPolicy(num_retreat_samples=0),
        )
        self.assertIs(report.outcome, ApproachPathOutcome.SKIPPED)


class ValidateApproachAndRetreatTests(unittest.TestCase):
    def test_returns_both_reports(self) -> None:
        approach, retreat = validate_approach_and_retreat(_grasp())
        self.assertIs(approach.outcome, ApproachPathOutcome.NO_OBSTACLES)
        self.assertIs(retreat.outcome, ApproachPathOutcome.NO_OBSTACLES)

    def test_protocol_compatibility(self) -> None:
        check = AcceptAllTrajectorySafetyCheck()
        self.assertIsInstance(check, TrajectorySafetyCheck)


class CustomGripperModelTests(unittest.TestCase):
    def test_custom_gripper_model_is_forwarded(self) -> None:
        # The default gripper's negative-x finger box at world
        # (-25, 0, -80) collides during pre-grasp. A tiny gripper
        # has a much smaller envelope and should miss the point.
        tiny = ParallelJawGripperModel(
            finger_length_mm=5.0,
            finger_thickness_mm=1.0,
            finger_width_mm=1.0,
            fingertip_depth_mm=1.0,
            palm_depth_mm=1.0,
            palm_width_mm=1.0,
        )
        obstacle = np.array([[-25.0, 0.0, -80.0]])
        default_report = validate_approach_path(
            _grasp(),
            obstacle_points_mm=obstacle,
            policy=ApproachPathPolicy(standoff_mm=80.0),
        )
        tiny_report = validate_approach_path(
            _grasp(),
            obstacle_points_mm=obstacle,
            policy=ApproachPathPolicy(standoff_mm=80.0),
            gripper_model=tiny,
        )
        self.assertIs(default_report.outcome, ApproachPathOutcome.BLOCKED)
        self.assertIs(tiny_report.outcome, ApproachPathOutcome.CLEAR)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
