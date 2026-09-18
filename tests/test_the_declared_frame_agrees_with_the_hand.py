"""A declared tool frame places the hand model wherever the derivation admits it, and refuses only where it does not.

Step 4g, as the owner decided after the probe 4g.0 (step4-plan, 4g), and narrowed in UM lane S23. On the Isaac cell
DH frame 6, Lula's tool0 and cuRobo's tool0 are one frame, and the physical 2F-85 lies along its +Y; every committed
hand model holds the hand that way. A hand bolted to a real UR flange approaches along +Z by the UR convention, which
is what the ursim bed declares. Until UM lane B2 nothing could turn a model, so a +Z cell refused to build.

Since B2 ``_hand_placement`` derives the one rotation that puts the model where the frame says, and every reader
places the hand by it: the guard, the planner's body link and the self filter. So the build refuses exactly what that
derivation refuses (a mirror, an oblique approach, -Z, an axis nobody measured, a clocking that is not a quarter
turn), and a planner starts on a placement only with a committed evidence file for it (S22).

What the checklist says, as the owner decided it (2026-09-16):

* a real UR cell declaring +Z (or any quarter turn of it) reads OK and says the placement is declared and not
  measured: the first real cell measures approach, clocking and plate at bring up (UM7);
* a real UR cell declaring the Isaac +Y frame reads WARN and names +Y as the Isaac asset's axis; it builds;
* a frame that places no hand reads BLOCK, with the refusal sentence and the placements that are admitted.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config import ConfigError, load_robot_config
from src.config.grippers import available_grippers
from src.config.schema.robot import RobotConfig
from src.contracts import UNSET
from src.robot.execution.real_cell.preflight import CheckStatus

_REAL_UR = {"source": "willy", "offset_mm": (0.0, 0.0, 132.0), "rotation_quat_xyzw": (0.0, 0.0, 0.0, 1.0)}
_ISAAC = {
    "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
    "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
}
#: Half a turn about flange X: the approach points into the wrist, -Z.
_INTO_THE_WRIST = {"source": "willy", "offset_mm": (0.0, 0.0, -132.0), "rotation_quat_xyzw": (1.0, 0.0, 0.0, 0.0)}
#: The real UR frame leaning 30 degrees about flange X: no axis, so no placement.
_OBLIQUE = {
    "source": "willy", "offset_mm": (0.0, -66.0, 114.3),
    "rotation_quat_xyzw": (math.sin(math.radians(15.0)), 0.0, 0.0, math.cos(math.radians(15.0))),
}
#: The real UR frame leaning half a degree about flange X: inside the tolerance, snapped to +Z, and said.
_HALF_A_DEGREE = {
    "source": "willy", "offset_mm": (0.0, -1.2, 132.0),
    "rotation_quat_xyzw": (math.sin(math.radians(0.25)), 0.0, 0.0, math.cos(math.radians(0.25))),
}
_MAPS = Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "planning" / "robot"
_HERE = (0.0, -1.2, 1.4, -1.8, -1.57, 0.3)


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
    """The axis every placement turns FROM is a property of the files, held for every registry hand."""

    def test_every_bundle_the_guard_reads_holds_its_fingers_along_the_model_axis(self) -> None:
        from src.robot.safety.planning.environment import hand_mesh_bundle
        from src.robot.safety.planning.hand import (
            APPROACH_TOLERANCE_DEG,
            HAND_APPROACH_IN_TOOL0,
        )

        hands = available_grippers()
        self.assertGreaterEqual(len(hands), 3, hands)
        for name in hands:
            with self.subTest(hand=name):
                # The hand's own bundle, the one the guard composes onto every arm (UM lane S08).
                bundle = hand_mesh_bundle(name)
                with np.load(bundle) as data:
                    fingers = (np.asarray(data["lfinger__v"]).mean(axis=0) + np.asarray(data["rfinger__v"]).mean(axis=0)) / 2
                self.assertLess(
                    _angle_deg(fingers, HAND_APPROACH_IN_TOOL0), APPROACH_TOLERANCE_DEG,
                    f"{bundle.name} holds the fingers of {name} at {np.round(fingers, 2).tolist()} mm",
                )


class AUrDeclaringItsOwnFlangeAxisBuildsTests(unittest.TestCase):
    """⭐ S23. Refused from Step 4g until B2 could place a hand; the placement is derived, so the refusal is gone."""

    def test_a_ur_declaring_the_real_flange_axis_builds(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        URRobotArm(_ur(_REAL_UR))

    def test_its_hand_is_placed_along_z(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(_ur(_REAL_UR))
        self.assertEqual((hand.placement.approach, hand.placement.closing), ("+Z", "+X"))
        self.assertIsNone(hand.placement_refusal)

    def test_the_ursim_bed_builds(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        URRobotArm(load_robot_config(profile="ursim"))

    def test_its_self_filter_describes_the_hand_along_z(self) -> None:
        """Until S23 a +Z cell had no self envelope at all, so a camera world refused. Its hand now lies along the
        flange's +Z: the farthest sphere is far out along z and near the axis, where a +Y model would put it along y."""
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(_ur(_REAL_UR))
        arm._conn = MagicMock()
        arm._conn.get_joint_positions.return_value = list(_HERE)
        envelope = arm._self_envelope()
        self.assertIsNotNone(envelope, "a +Z UR cell must describe its own body")
        assert envelope is not None
        hand = [c for c in envelope.capsules if c.frame == 6 and c.start_mm == c.end_mm]
        self.assertTrue(hand, "the hand's spheres ride on the flange frame")
        tip = max(hand, key=lambda c: c.start_mm[2])
        self.assertGreater(tip.start_mm[2], 100.0)
        self.assertLess(abs(tip.start_mm[1]), 30.0)


class TheChecklistSaysWhatWasDeclaredTests(unittest.TestCase):
    def test_a_real_ur_declaring_plus_z_says_it_is_declared_and_not_measured(self) -> None:
        row = _hand_row(_ur(_REAL_UR))
        self.assertEqual(row.status, CheckStatus.OK, row.detail)
        for part in ("approach +Z", "closing +X", "declared", "not measured"):
            self.assertIn(part, row.detail)

    def test_the_ursim_bed_reads_ok(self) -> None:
        row = _hand_row(load_robot_config(profile="ursim"))
        self.assertEqual(row.status, CheckStatus.OK, row.detail)
        self.assertIn("approach +Z", row.detail)

    def test_a_real_ur_declaring_the_isaac_frame_is_warned_and_builds(self) -> None:
        """Owner, 2026-09-16: WARN, not BLOCK, and construction unchanged."""
        from src.robot.drivers.ur.arm import URRobotArm

        row = _hand_row(_ur(_ISAAC))
        self.assertEqual(row.status, CheckStatus.WARN, row.detail)
        for part in ("approach +Y", "Isaac", "+Z"):
            self.assertIn(part, row.detail + " " + row.fix)
        URRobotArm(_ur(_ISAAC))

    def test_a_snapped_residual_is_said(self) -> None:
        row = _hand_row(_ur(_HALF_A_DEGREE))
        self.assertEqual(row.status, CheckStatus.OK, row.detail)
        self.assertIn("0.5 degrees", row.detail)

    def test_an_exact_frame_says_no_residual(self) -> None:
        self.assertNotIn("degrees", _hand_row(_ur(_REAL_UR)).detail)


class WhatTheDerivationRefusesStillRefusesTests(unittest.TestCase):
    """The narrowing must not open a placement nobody can model."""

    _REFUSED = {"-Z": (_INTO_THE_WRIST, "-Z"), "oblique": (_OBLIQUE, "30 degrees")}

    def test_a_refused_placement_refuses_at_build_with_its_sentence(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        for name, (frame, said) in self._REFUSED.items():
            with self.subTest(frame=name), self.assertRaises(ConfigError) as caught:
                URRobotArm(_ur(frame))
            self.assertIn("robot.gripper.tool_frame places no hand model", str(caught.exception))
            self.assertIn(said, str(caught.exception))

    def test_the_checklist_blocks_it_and_names_what_is_admitted(self) -> None:
        for name, (frame, said) in self._REFUSED.items():
            with self.subTest(frame=name):
                row = _hand_row(_ur(frame))
                self.assertEqual(row.status, CheckStatus.BLOCK, row.detail)
                self.assertIn(said, row.detail)
                self.assertIn("+Y or +Z", row.fix)

    def test_a_refused_placement_has_no_hand_spheres_to_filter(self) -> None:
        """No cell with it builds a guard (the guard refuses an unplaceable hand on every backend), so the self
        filter is held where it reads the hand: a body nobody can place is not filtered. +Z is the control."""
        from src.robot.safety.planning.hand import planner_hand
        from src.robot.safety.planning.self_envelope import hand_spheres

        self.assertIsNone(hand_spheres(planner_hand(_ur(_INTO_THE_WRIST)), "ur5e"))
        self.assertIsNotNone(hand_spheres(planner_hand(_ur(_REAL_UR)), "ur5e"))


class WhatReadsNoHandOrNoFlangeTests(unittest.TestCase):
    """Controls, green before and after."""

    def test_the_sim_cell_builds(self) -> None:
        from src.willy_sim.config import load_sim_config, sim_safety_preflight

        sim_safety_preflight(load_sim_config())

    def test_an_undeclared_frame_builds(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        URRobotArm(_ur(None))

    def test_an_undeclared_frame_is_the_hand_models_own_axes(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(_ur(None))
        self.assertIsNot(hand.placement, UNSET)
        self.assertEqual(hand.placement.source, "undeclared")
        self.assertIn("tool frame undeclared", _hand_row(_ur(None)).detail)

    def test_a_dummy_cell_declaring_the_isaac_frame_is_not_warned(self) -> None:
        """No UR flange: the +Y warning is about a real UR's convention, and a dummy arm has none."""
        self.assertEqual(_hand_row(load_robot_config(profile="console_dummy")).status, CheckStatus.OK)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
