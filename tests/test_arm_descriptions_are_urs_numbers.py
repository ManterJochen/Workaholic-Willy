"""Every number in the arm registry is one Universal Robots published, and every reader takes it from there.

From D2 to D3 this file was the bridge: it held each description to the hand-kept tables it replaced at exact
equality, so a reader moved in D3 saw what it saw before. D4 deleted the tables, and with them the half of this
file that compared against them; the exact floats are held now by ``tests/test_inlined_dh_tables.py``, against the
copies the committed collision bundles were baked through. What remains here is the other half: the registry
against Universal Robots' own pinned description, and each reader against the registry.

Two axes differ from UR's file on purpose, and the test names both. An absence that nobody states reads
exactly like an agreement, so the exceptions are listed with their reasons rather than being skipped.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

from src.config.arms import admitted_arms, available_arms, load_arm

_ROOT = Path(__file__).resolve().parents[1]
#: UR's own numbers agree with ours to float noise, not bit for bit: `dh_rows` derives them from joint origins.
_UR_TOLERANCE = 1e-12


def _ur_description():
    """Universal Robots' pinned config, loaded by path as `test_ur_description_tables.py` loads it."""
    name = "_ur_description_for_the_arm_registry"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "arms" / "_ur_description.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class WhatTheDescriptionsHoldTests(unittest.TestCase):
    """What a description says about its own arm, with no table left to compare it to."""

    def test_the_one_length_a_millimetre_round_trip_would_change_survived(self) -> None:
        """The positive control for the unit choice: ur3e d6 is 0.0921, and 92.1/1000 is not."""
        self.assertEqual(0.0921, load_arm("ur3e").joints[5].d_m)
        self.assertNotEqual(92.1 / 1000.0, 0.0921, "the trap this control guards has gone, so it proves nothing")

    def test_every_home_is_the_pose_the_driver_sent_before_it_read_the_description(self) -> None:
        """Since P0e the UR driver reads each description's home (tests/test_the_arm_says_who_it_is.py), and
        `HOME_JOINTS_DEFAULT` is deleted. Its value stays here as a literal, so no description can move a home by
        accident: shoulder lift and wrist 1 at -pi/2, the rest at zero, the pose authored for a UR5e.
        """
        half = -math.pi / 2
        for model in available_arms():
            with self.subTest(model=model):
                self.assertEqual([0.0, half, 0.0, half, 0.0, 0.0], [joint.home_rad for joint in load_arm(model).joints])

    def test_exactly_the_arms_a_cell_may_name_carry_what_a_cell_needs(self) -> None:
        """Admission is stated, and today it agrees with what the descriptions bring: a sim block and a reach."""
        complete = {model for model in available_arms()
                    if load_arm(model).sim is not None and load_arm(model).reach_mm is not None}
        self.assertEqual(set(admitted_arms()), complete)


class TheKinematicsReadTheRegistryTests(unittest.TestCase):
    """P0d D3b. The guard, the closed-form IK, the sim's reach helper and datagen take a UR's DH rows from the registry.

    The floats themselves are held by `tests/test_inlined_dh_tables.py`, exactly, against the copies the committed
    collision bundles were baked through; the hand-typed table this move read beside was deleted in D4. What is
    here is the behaviour of the readers.
    """

    def test_an_arm_described_from_universal_robots_alone_has_kinematics(self) -> None:
        """ur20 never had a hand-typed row, and has joints from UR's pinned description in the registry.

        This was the case that proved D3b moved the source rather than copying it, while the old table stood
        beside the registry. The table is gone; what it pins now is that a described arm is answered at all.
        """
        from src.robot.safety._ur_kinematics import ur_dh_rows, ur_link_transforms_mm

        self.assertIsNotNone(ur_dh_rows("ur20"))
        self.assertIsNotNone(ur_link_transforms_mm("ur20", _zeros()))

    def test_an_arm_with_known_links_and_no_bundle_is_called_that(self) -> None:
        """ur20's status moves from ``unknown_model`` to ``no_bundle``, which is what is true of it now."""
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        self.assertEqual("no_bundle", mesh_backend_status("ur20"))
        self.assertEqual("unknown_model", mesh_backend_status("ur99"), "an arm nobody describes is still unknown")

    def test_the_refusal_for_an_unknown_arm_names_where_to_describe_it(self) -> None:
        """The hint said to add a row to UR_DH_TABLES_M. A row there reaches nothing any more."""
        from src.robot.safety._fcl_self_collision import _STATUS_HINTS

        self.assertIn("config/arms", _STATUS_HINTS["unknown_model"])
        self.assertNotIn("UR_DH_TABLES_M", _STATUS_HINTS["unknown_model"])

    def test_a_non_ur_description_is_not_answered_as_a_ur(self) -> None:
        """``ur_dh_rows`` is the UR DH form. A later Franka description must not be read through it."""
        from unittest import mock

        from src.robot.safety import _ur_kinematics

        franka = load_arm("ur5e").model_copy(update={"vendor": "franka"})
        _ur_kinematics._described_rows.cache_clear()
        try:
            with mock.patch("src.config.arms.load_arm", return_value=franka):
                self.assertIsNone(_ur_kinematics._described_rows("fr3"))
        finally:
            _ur_kinematics._described_rows.cache_clear()

    def test_the_lookup_is_cached_because_its_readers_are_hot(self) -> None:
        """The continuous monitor asks on every sample of a path; an uncached read parses eight files each time."""
        from src.robot.safety._ur_kinematics import _described_rows, ur_dh_rows

        ur_dh_rows("ur5e")
        before = _described_rows.cache_info().hits
        for _ in range(10):
            ur_dh_rows("ur5e")
        self.assertEqual(before + 10, _described_rows.cache_info().hits)

    def test_the_closed_form_ik_reads_the_registry_too(self) -> None:
        """Willy's own reader of the rows: the closed-form IK answers ur20, which only the registry describes."""
        from src.robot.safety._ur_ik import ur_branch

        self.assertIsNotNone(ur_branch("ur20", [0.0, -1.0, 1.0, -1.0, 1.0, 0.0]))


class TheSimSpecIsBuiltFromTheRegistryTests(unittest.TestCase):
    """P0d D3d. ``ur_model_spec`` builds its spec from a description's ``sim`` block, reach and payload.

    Seven readers go through that one function, so none of them moved and all of them read the registry now.
    """

    def test_the_reference_arm_is_the_patch_the_sim_runs_were_measured_on(self) -> None:
        """An oracle the registry did not write. With the table gone, a spec compared to its description only
        restates the builder. These are the ur5e numbers the sim runners were validated with, written out: every
        derived patch in the registry is scaled from this one, so a drift here moves every arm's scene.
        """
        from src.robot.drivers.sim.robot_models import URModelSpec, ur_model_spec

        self.assertEqual(
            URModelSpec(
                "ur5e", "UR5e", "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
                max_reach_mm=850.0, max_payload_kg=5.0,
                workspace_center_mm=(450.0, 0.0), workspace_half_extents_mm=(150.0, 150.0),
                baked_gripper_variant="Robotiq_2f_85", baked_hand="robotiq_2f85"),
            ur_model_spec("ur5e"),
        )

    def test_the_lookup_takes_a_model_in_any_case(self) -> None:
        from src.robot.drivers.sim.robot_models import simulable_models, ur_model_spec

        for model in simulable_models():
            with self.subTest(model=model):
                self.assertEqual(ur_model_spec(model), ur_model_spec(model.upper()))

    def test_a_described_arm_with_no_sim_block_is_refused_as_it_was(self) -> None:
        """ur16e and ur20 are described, carry no Isaac asset, and the sim refuses them the way it always did."""
        from src.robot.drivers.sim.robot_models import ur_model_spec

        for model in ("ur16e", "ur20"):
            with self.subTest(model=model), self.assertRaises(ValueError) as caught:
                ur_model_spec(model)
            self.assertIn("config/arms", str(caught.exception), "the fix names where to describe it")

    def test_the_spec_comes_from_the_registry_now(self) -> None:
        """The discriminating case. A description whose patch moved gives a spec whose patch moved.

        Written while the hand-kept table still stood beside the registry and agreed with it on every model, so no
        real model could tell the two sources apart. It stays because it is still the one test here that cannot
        pass unless the spec is read from the description when it is asked for.
        """
        from unittest import mock

        from src.robot.drivers.sim import robot_models

        real = load_arm("ur5e")
        assert real.sim is not None
        before = real.sim.workspace_center_mm
        moved = real.model_copy(update={"sim": real.sim.model_copy(update={"workspace_center_mm": (400.0, 25.0)})})
        robot_models._described_spec.cache_clear()
        try:
            with mock.patch("src.config.arms.load_arm", return_value=moved):
                spec = robot_models.ur_model_spec("ur5e")
        finally:
            robot_models._described_spec.cache_clear()
        self.assertEqual((400.0, 25.0), spec.workspace_center_mm)
        self.assertNotEqual(before, spec.workspace_center_mm, "the control: the patch actually moved")


def _zeros():
    import numpy as np

    return np.zeros(6)


class TheNumbersAreStillUniversalRobotsOwnTests(unittest.TestCase):
    """The derived half, held against the pinned description that `fetch.py` keeps byte for byte."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ur = _ur_description()

    def test_every_described_arm_is_one_universal_robots_pinned(self) -> None:
        pinned = set(self.ur.models())
        self.assertLessEqual(set(available_arms()), pinned, "an arm was described that UR's pinned config has not")

    def test_every_dh_row_is_the_row_universal_robots_published(self) -> None:
        for model in available_arms():
            theirs = self.ur.dh_rows(model)
            arm = load_arm(model)
            with self.subTest(model=model):
                self.assertEqual(len(theirs), len(arm.joints))
                for index, (row, joint) in enumerate(zip(theirs, arm.joints, strict=True)):
                    a_m, d_m, alpha_rad = row
                    self.assertAlmostEqual(a_m, joint.a_m, delta=_UR_TOLERANCE, msg=f"{model} joint {index} a")
                    self.assertAlmostEqual(d_m, joint.d_m, delta=_UR_TOLERANCE, msg=f"{model} joint {index} d")
                    self.assertAlmostEqual(alpha_rad, joint.alpha_rad, delta=_UR_TOLERANCE,
                                           msg=f"{model} joint {index} alpha")

    def test_every_joint_moves_the_link_universal_robots_names_as_its_child(self) -> None:
        """The permanent half of F6c's oracle: each row's ``link`` is its joint's child in UR's rendered URDF."""
        import xml.etree.ElementTree as ET

        for model in available_arms():
            children = {joint.get("name"): joint.find("child").get("link")  # type: ignore[union-attr]
                        for joint in ET.fromstring(self.ur.render_urdf(model)).iter("joint")}
            with self.subTest(model=model):
                self.assertEqual({joint.name: children[joint.name] for joint in load_arm(model).joints},
                                 {joint.name: joint.link for joint in load_arm(model).joints})

    def test_the_chain_starts_where_universal_robots_turns_it(self) -> None:
        """F6d's permanent oracle: ``base.link`` is the URDF's root, and the joint from it into ``base_link_inertia``,
        where the DH chain starts, turns by exactly ``base.chain_yaw_deg`` about z and does nothing else."""
        import xml.etree.ElementTree as ET

        for model in available_arms():
            joints = {joint.get("name"): joint for joint in ET.fromstring(self.ur.render_urdf(model)).iter("joint")}
            parents = {joint.find("parent").get("link") for joint in joints.values()}  # type: ignore[union-attr]
            children = {joint.find("child").get("link") for joint in joints.values()}  # type: ignore[union-attr]
            turn = joints["base_link-base_link_inertia"]
            origin = turn.find("origin")
            assert origin is not None
            roll, pitch, yaw = (float(value) for value in origin.get("rpy", "").split())
            base = load_arm(model).base
            with self.subTest(model=model):
                self.assertEqual({base.link}, parents - children)
                self.assertEqual(base.link, turn.find("parent").get("link"))  # type: ignore[union-attr]
                self.assertEqual([0.0, 0.0, 0.0], [float(value) for value in origin.get("xyz", "").split()])
                self.assertEqual((0.0, 0.0), (roll, pitch))
                self.assertEqual(float(base.chain_yaw_deg), math.degrees(yaw))
                self.assertEqual("base_link_inertia",
                                 joints["shoulder_pan_joint"].find("parent").get("link"))  # type: ignore[union-attr]

    def test_the_controller_reports_in_the_frame_universal_robots_turns(self) -> None:
        """F6e's permanent oracle: UR's ``base``, the controller's Base, hangs from ``base.link`` turned by exactly
        ``base.controller_yaw_deg`` about z ("needed to correctly align 'base' with the 'Base' coordinate system of the
        UR controller", ur_macro.xacro)."""
        import xml.etree.ElementTree as ET

        for model in available_arms():
            joints = {joint.get("name"): joint for joint in ET.fromstring(self.ur.render_urdf(model)).iter("joint")}
            turn = joints["base_link-base_fixed_joint"]
            origin = turn.find("origin")
            assert origin is not None
            roll, pitch, yaw = (float(value) for value in origin.get("rpy", "").split())
            base = load_arm(model).base
            with self.subTest(model=model):
                self.assertEqual((base.link, "base"), (turn.find("parent").get("link"),  # type: ignore[union-attr]
                                                       turn.find("child").get("link")))  # type: ignore[union-attr]
                self.assertEqual([0.0, 0.0, 0.0], [float(value) for value in origin.get("xyz", "").split()])
                self.assertEqual((0.0, 0.0), (roll, pitch))
                self.assertEqual(float(base.controller_yaw_deg), math.degrees(yaw))

    def test_the_joint_names_are_universal_robots_own(self) -> None:
        for model in available_arms():
            with self.subTest(model=model):
                theirs = set(self.ur.load(model).joint_limits)
                self.assertEqual(theirs, {joint.name for joint in load_arm(model).joints})

    def test_every_envelope_is_urs_except_the_two_this_repository_means_to_differ_on(self) -> None:
        """THE Two exceptions, named rather than skipped.

        `elbow_joint`: UR writes +/-180 and says why in its own file, "we artificially limit this joint to half
        its actual joint position limit to avoid (MoveIt/OMPL) planning problems". That is a planner's
        convenience. The guard decides whether a commanded angle leaves Python at all, so it holds the arm's
        envelope, which is the +/-360 of UR's user manual.

        `wrist_3_joint` on the ur3 and ur3e: UR writes `has_position_limits: false`, a continuous joint with no
        bounds. This repository bounds it at +/-360, which is the safe direction (tighter than the robot) and
        does stop a rotation a UR3e can perform. Recorded here so the choice is visible; whether the guard
        adopts the continuous joint is a question for the slice that moves the guard.
        """
        differed: list[str] = []
        for model in available_arms():
            theirs = self.ur.load(model).joint_limits
            for joint in load_arm(model).joints:
                limit = theirs[joint.name]
                low, high = joint.limit_deg
                if limit.lower_rad is None or limit.upper_rad is None:
                    differed.append(f"{model}.{joint.name} continuous there, {low:g}..{high:g} here")
                    continue
                if (abs(math.degrees(limit.lower_rad) - low) > 1e-9
                        or abs(math.degrees(limit.upper_rad) - high) > 1e-9):
                    differed.append(f"{model}.{joint.name} {math.degrees(limit.lower_rad):g}.."
                                    f"{math.degrees(limit.upper_rad):g} there, {low:g}..{high:g} here")

        expected = [f"{model}.elbow_joint -180..180 there, -360..360 here" for model in available_arms()]
        expected += [f"{model}.wrist_3_joint continuous there, -360..360 here" for model in ("ur3", "ur3e")]
        self.assertEqual(sorted(expected), sorted(differed))

    def test_every_planner_envelope_is_urs_except_the_continuous_wrist(self) -> None:
        """The permanent half of F6f's oracle: ``planner_limit_deg`` is UR's planning file, the elbow's +/-180
        included, except on the two joints UR leaves continuous, where this repository writes the guard's +/-360."""
        differed: list[str] = []
        for model in available_arms():
            theirs = self.ur.load(model).joint_limits
            for joint in load_arm(model).joints:
                limit = theirs[joint.name]
                low, high = joint.planner_limit_deg
                if limit.lower_rad is None or limit.upper_rad is None:
                    differed.append(f"{model}.{joint.name} continuous there, {low:g}..{high:g} here")
                elif (abs(math.degrees(limit.lower_rad) - low) > 1e-9
                        or abs(math.degrees(limit.upper_rad) - high) > 1e-9):
                    differed.append(f"{model}.{joint.name} {math.degrees(limit.lower_rad):g}.."
                                    f"{math.degrees(limit.upper_rad):g} there, {low:g}..{high:g} here")
        self.assertEqual(sorted(f"{model}.wrist_3_joint continuous there, -360..360 here" for model in ("ur3", "ur3e")),
                         sorted(differed))

    def test_the_elbow_exception_is_real_and_not_an_artefact_of_this_test(self) -> None:
        """The control: if UR ever widened the elbow, the list above would still pass by being empty."""
        for model in available_arms():
            with self.subTest(model=model):
                elbow = self.ur.load(model).joint_limits["elbow_joint"]
                self.assertAlmostEqual(-math.pi, elbow.lower_rad, delta=1e-12)
                self.assertAlmostEqual(math.pi, elbow.upper_rad, delta=1e-12)

    def test_the_continuous_wrist_is_real_and_only_on_those_two_arms(self) -> None:
        continuous = {model for model in available_arms()
                      if not self.ur.load(model).joint_limits["wrist_3_joint"].has_position_limits}
        self.assertEqual({"ur3", "ur3e"}, continuous)


if __name__ == "__main__":
    unittest.main()
