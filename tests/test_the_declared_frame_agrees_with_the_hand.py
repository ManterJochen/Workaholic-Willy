"""The tool frame a cell declares and the hand model its guard and planner check against approach along one axis.

Step 4g, as the owner decided after the probe 4g.0 (step4-plan, 4g). On the Isaac cell DH frame 6, Lula's tool0
and cuRobo's tool0 are one frame, and the physical 2F-85 lies along its +Y; the committed bundles, the sphere maps
and the sim's tool frame all agree with it. A hand bolted to a real UR flange approaches along +Z by the UR
convention, which is what the ursim bed declares, and nothing in the tree measures a real flange. So nothing is
rotated: a cell whose guard reads hand geometry refuses to build while the approach its tool frame declares and
the approach of the hand model disagree, and says both.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ConfigError, load_robot_config
from src.config.grippers import available_grippers
from src.config.schema.robot import RobotConfig
from src.robot.execution.real_cell.preflight import CheckStatus

_REAL_UR = {"source": "willy", "offset_mm": (0.0, 0.0, 132.0), "rotation_quat_xyzw": (0.0, 0.0, 0.0, 1.0)}
_ISAAC = {
    "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
    "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
}
_MAPS = Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "planning" / "robot"


def _ur(frame: dict[str, Any] | None, **safety: Any) -> RobotConfig:
    gripper: dict[str, Any] = {"model": "robotiq_2f85"}
    if frame is not None:
        gripper["tool_frame"] = frame
    return RobotConfig.model_validate(
        {"vendor": "ur", "safety": {"payload": {"enforce": False}, **safety}, "gripper": gripper}
    )


def _angle_deg(a: np.ndarray, b: Any) -> float:
    b = np.asarray(b, dtype=np.float64)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _hand_row(robot: RobotConfig) -> Any:
    from src.robot.execution.real_cell.preflight import run_config_preflight

    (row,) = [c for c in run_config_preflight(robot, curobo_available=False).checks if c.name == "hand"]
    return row


class TheCommittedHandsApproachAlongTheModelAxisTests(unittest.TestCase):
    """The constant the check compares against is a property of the files, held for every registry hand."""

    def test_every_bundle_the_guard_reads_holds_its_fingers_along_the_model_axis(self) -> None:
        from src.robot.safety.planning.environment import collision_mesh_bundle
        from src.robot.safety.planning.hand import (
            APPROACH_TOLERANCE_DEG,
            HAND_APPROACH_IN_TOOL0,
            guard_variant_for,
        )

        hands = available_grippers()
        self.assertGreaterEqual(len(hands), 3, hands)
        for name in hands:
            with self.subTest(hand=name):
                bundle = collision_mesh_bundle("ur5e", guard_variant_for(name))
                with np.load(bundle) as data:
                    fingers = (np.asarray(data["lfinger__v"]).mean(axis=0) + np.asarray(data["rfinger__v"]).mean(axis=0)) / 2
                self.assertLess(
                    _angle_deg(fingers, HAND_APPROACH_IN_TOOL0), APPROACH_TOLERANCE_DEG,
                    f"{bundle.name} holds the fingers of {name} at {np.round(fingers, 2).tolist()} mm",
                )


class ACellWhoseFrameDisagreesRefusesTests(unittest.TestCase):
    def test_a_ur_declaring_the_real_flange_axis_refuses_naming_both_axes(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        with self.assertRaises(ConfigError) as caught:
            URRobotArm(_ur(_REAL_UR))
        message = str(caught.exception)
        for part in ("robot.gripper.tool_frame", "+Z", "+Y", "robotiq_2f85"):
            self.assertIn(part, message)

    def test_the_hand_says_how_far_apart_the_two_axes_are(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        self.assertAlmostEqual(planner_hand(_ur(_REAL_UR)).approach_disagreement_deg, 90.0, places=3)
        self.assertAlmostEqual(planner_hand(_ur(_ISAAC)).approach_disagreement_deg, 0.0, places=3)

    def test_the_ursim_bed_refuses_at_build(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        with self.assertRaises(ConfigError):
            URRobotArm(load_robot_config(profile="ursim"))

    def test_the_real_cell_checklist_blocks_it(self) -> None:
        row = _hand_row(load_robot_config(profile="ursim"))
        self.assertEqual(row.status, CheckStatus.BLOCK)
        self.assertIn("+Z", row.detail)


class WhatAgreesOrReadsNoHandBuildsTests(unittest.TestCase):
    """Controls, green before and after."""

    def test_a_ur_declaring_the_isaac_frame_builds(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        URRobotArm(_ur(_ISAAC))

    def test_the_sim_cell_builds(self) -> None:
        from src.willy_sim.config import load_sim_config, sim_safety_preflight

        sim_safety_preflight(load_sim_config())

    def test_an_undeclared_frame_builds(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        URRobotArm(_ur(None))

    def test_a_capsule_guard_reads_no_hand_and_builds(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        URRobotArm(_ur(_REAL_UR, self_collision={"backend": "capsule"}))

    def test_a_dummy_cell_declaring_plus_z_is_not_blocked(self) -> None:
        self.assertEqual(_hand_row(load_robot_config(profile="console_dummy")).status, CheckStatus.OK)


class AnUndeclaredFrameIsNotComparedTests(unittest.TestCase):
    def test_an_undeclared_frame_has_no_disagreement(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        self.assertIsNone(planner_hand(_ur(None)).approach_disagreement_deg)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
