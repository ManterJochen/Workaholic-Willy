"""A path's step bound holds for what the flange carries, not only for the arm (Step 8h).

Every judged path, joint, line or planned, is sampled so that between two neighbouring configurations no point of the
robot moves further than the bound the gate is told: the sum of each joint's turn times its radius. The radii came from
the bare DH table, which ends at the flange, so a wrist turn that the gate counted as 10 mm moved a 132 mm tool's grasp
centre 11.6 mm and its fingertips further (found by the Step 8f and 8g review, 2026-09-18). A radius is now the DH radius
plus the furthest the carried geometry reaches from the flange: the hand's sphere map as the guard places it, a wrist
camera's fill, and a carried part where the cell declares its length.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety._ur_kinematics import ur_joint_radii_mm, ur_link_transforms_mm
from src.robot.safety.planning.self_envelope import hand_spheres, payload_capsule


def _arm(*, payload: "dict[str, Any] | None" = None) -> URRobotArm:
    safety: dict[str, Any] = {
        "payload": {"enforce": False},
        "self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0},
    }
    if payload is not None:
        safety["planning_world"] = {"payload": payload}
    return URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
        "gripper": {"model": "robotiq_2f85", "coupling_plates": [], "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 0.0, 132.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": safety,
    }))


def _carried_points(arm: URRobotArm, *, with_part: bool = False) -> np.ndarray:
    """The extreme points of every sphere and capsule the flange carries, in the flange frame."""
    preflight = arm.safety_preflight
    assert preflight is not None
    hand = preflight.planner_hand(arm)
    spheres = hand_spheres(hand, "ur5e")  # type: ignore[arg-type]
    assert spheres is not None
    capsules = list(spheres)
    if with_part:
        cfg = arm.config.safety.planning_world.payload
        capsules.append(payload_capsule(hand, spheres, length_mm=float(cfg.length_mm),  # type: ignore[arg-type]
                                        lateral_margin_mm=float(cfg.lateral_margin_mm)))
    points = []
    for capsule in capsules:
        for end in (capsule.start_mm, capsule.end_mm):
            for axis in np.vstack([np.eye(3), -np.eye(3)]):
                points.append(np.asarray(end, dtype=np.float64) + axis * float(capsule.radius_mm))
    return np.asarray(points)


def _largest_move_mm(points: np.ndarray, before: np.ndarray, after: np.ndarray) -> float:
    frames = [np.asarray(ur_link_transforms_mm("ur5e", q)[-1]) for q in (before, after)]  # type: ignore[index]
    placed = [(frame[:3, :3] @ points.T).T + frame[:3, 3] for frame in frames]
    return float(np.max(np.linalg.norm(placed[1] - placed[0], axis=1)))


class TheStepBoundHoldsForTheHandTests(unittest.TestCase):
    def _assert_bound_holds(self, arm: URRobotArm, points: np.ndarray) -> None:
        preflight = arm.safety_preflight
        assert preflight is not None
        radii = preflight.joint_radii_mm(arm)
        assert radii is not None
        rng = np.random.default_rng(20260918)
        worst = 0.0
        for _ in range(300):
            before = rng.uniform(-math.pi, math.pi, 6)
            # One joint at a time: with every joint turning, the shoulder's radius dominates the sum and hides a wrist
            # radius that misses the tool, which is the case the bound has to hold in.
            turn = np.zeros(6)
            turn[int(rng.integers(6))] = rng.normal(0.0, 0.05)
            bound = float(np.sum(np.abs(turn) * np.asarray(radii)))
            moved = _largest_move_mm(points, before, before + turn)
            worst = max(worst, moved / bound)
            self.assertLessEqual(moved, bound + 1e-6, f"a carried point moved {moved:.2f} mm for a bound of {bound:.2f}")
        self.assertGreater(worst, 0.2, "the samples never came near the bound, so they prove nothing about it")

    def test_no_point_of_the_hand_moves_further_than_the_bound_the_gate_is_told(self) -> None:
        arm = _arm()
        self._assert_bound_holds(arm, _carried_points(arm))

    def test_a_wrist_turn_alone_keeps_the_bound_at_the_fingertips(self) -> None:
        """The reviewer's case: only wrist 2 turns, where the bare radius of 199.3 mm misses the tool."""
        arm = _arm()
        preflight = arm.safety_preflight
        assert preflight is not None
        radii = preflight.joint_radii_mm(arm)
        assert radii is not None
        before = np.array([0.3, -1.2, 1.4, -1.7, -1.5708, 0.2])
        turn = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0])
        moved = _largest_move_mm(_carried_points(arm), before, before + turn)
        self.assertLessEqual(moved, 0.05 * radii[4] + 1e-6)

    def test_a_declared_carried_part_is_in_the_bound_too(self) -> None:
        arm = _arm(payload={"enabled": True, "length_mm": 120.0})
        self._assert_bound_holds(arm, _carried_points(arm, with_part=True))

    def test_the_bare_arm_is_the_floor(self) -> None:
        """The control: nothing carried shrinks a radius below the arm's own."""
        arm = _arm()
        preflight = arm.safety_preflight
        assert preflight is not None
        radii = preflight.joint_radii_mm(arm)
        bare = ur_joint_radii_mm("ur5e")
        assert radii is not None and bare is not None
        for joint, (carried, own) in enumerate(zip(radii, bare)):
            with self.subTest(joint=joint + 1):
                self.assertGreaterEqual(carried, own)


class AHandWhoseReachCannotBeReadTests(unittest.TestCase):
    def test_a_path_is_refused_with_the_reason_when_the_hand_bundle_is_missing(self) -> None:
        """No radius leaves the hand out: where its reach cannot be read, the path judge says so and names why."""
        from unittest.mock import patch

        arm = _arm()
        preflight = arm.safety_preflight
        assert preflight is not None
        with patch("src.robot.safety.planning.self_envelope.hand_mesh_bundle",
                   return_value="no/such/bundle.npz"):
            self.assertIsNone(preflight.joint_radii_mm(arm))
            refusal = preflight.path_judge_refusal(arm)
        assert refusal is not None
        self.assertIn("no mesh bundle to read its reach past the flange", refusal)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
