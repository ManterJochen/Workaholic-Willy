"""The committed retract table is a geometry authority, and it says what it measured (UM5, B4).

⭐ ONE ROW PER ARM, HAND, PLATE AND PLANNER MARGIN since 2026-09-16 (owner decision). The retract used to be a
property of the arm alone, judged on the exact meshes alone, and that was wrong twice: measured over the rule's
own candidate family, at ur3's committed pose the meshes clear the Hand-E by 13.4 mm while the planner's sphere
model calls the same pose a self collision, so the sidecar never becomes ready and the cell does not come up.
The rule now takes the first pose BOTH models accept, and the pose that works is different for each hand.

``src/robot/safety/planning/robot/ur_retract.yaml`` holds one retract per arm, chosen by the rule on the exact
meshes with every registry hand. The descriptor builder reads it in the cuRobo environment, where neither Coal nor the
bundles can be loaded, so nothing there can check the numbers: these are the checks.

What this file can check off the box is the table against itself and against the committed bundles. What it cannot is
the distances, which only Coal can measure; ``tests/test_retract_clears_the_guard.py`` does that where an engine is
installed, and ``choose_ur_retract.py --check`` does it on the box.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

from src.config.grippers import available_grippers
from src.robot.safety.planning.environment import collision_mesh_bundle, hand_mesh_bundle

_ROOT = Path(__file__).resolve().parents[1]
_TABLE = _ROOT / "src" / "robot" / "safety" / "planning" / "robot" / "ur_retract.yaml"


def _module(name: str):
    path = _ROOT / "scripts" / "curobo" / name
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TheCommittedTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(_TABLE.is_file(), f"no committed retract table at {_TABLE}")
        self.table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))
        self.rule = self.table["rule"]
        self.rows = self.table["retracts"]
        self.arms = sorted({row["arm"] for row in self.rows})

    def test_it_judges_exactly_the_arms_that_have_a_bundle(self) -> None:
        anchors = _module("_retract_rule.py").ANCHORS
        with_bundle = sorted(arm for arm in anchors if collision_mesh_bundle(arm).is_file())
        judged = sorted(set(self.arms) | {row["arm"] for row in self.rule["pairs_with_no_retract"]})
        self.assertEqual(judged, with_bundle,
                         "an arm with a bundle was neither judged nor recorded as having no retract")
        self.assertEqual(sorted(self.rule["arms_without_a_bundle"]),
                         sorted(arm for arm in anchors if not collision_mesh_bundle(arm).is_file()))

    def test_the_record_of_an_arm_with_no_bundle_can_still_be_written(self) -> None:
        """⭐ THE CONTROL, rebuilt because it rotted by growth.

        It used to read "an arm without a bundle must exist, or this assertion cannot fail", which was true while
        ur16e had none. B5 baked that bundle, so the list above is empty and the assertion that reads it has nothing
        left to compare. The rule is proven by taking a bundle away instead of by waiting for one to be missing.
        """
        import tempfile
        from unittest import mock

        anchors = _module("_retract_rule.py").ANCHORS
        with tempfile.TemporaryDirectory() as empty, mock.patch(
            "src.robot.safety.planning.environment.COLLISION_MESH_DIR", Path(empty)
        ):
            self.assertEqual(sorted(arm for arm in anchors if not collision_mesh_bundle(arm).is_file()),
                             sorted(anchors))

    def test_every_arm_was_asked_about_every_hand_the_registry_holds(self) -> None:
        """Judged or refused, but never simply absent: a pair nobody asked about is the state this replaced."""
        asked: dict = {}
        for row in self.rows:
            asked.setdefault(row["arm"], set()).add(row["hand"])
        for row in self.rule["pairs_with_no_retract"]:
            asked.setdefault(row["arm"], set()).add(row["hand"])
        for arm, hands in sorted(asked.items()):
            with self.subTest(arm=arm):
                self.assertEqual(sorted(hands), sorted(available_grippers()))

    def test_every_recorded_clearance_clears_the_guard_with_its_reserve(self) -> None:
        needed = float(self.rule["guard_margin_mm"]) + float(self.rule["reserve_mm"])
        for row in self.rows:
            with self.subTest(arm=row["arm"], hand=row["hand"],
                              plate=row["plate_mm"]):
                self.assertGreaterEqual(row["exact"]["clearance_mm"], needed)
                self.assertGreaterEqual(row["exact"]["lowest_z_mm"],
                                        float(self.rule["table_clearance_mm"]))

    def test_every_committed_pose_is_one_the_planner_admits(self) -> None:
        """⭐ THE SECOND HALF OF THE RULE, recorded. A pose in the sphere model's own collision is a sidecar
        that never becomes ready, whatever the meshes say about it.
        """
        for row in self.rows:
            with self.subTest(arm=row["arm"], hand=row["hand"]):
                self.assertIsNone(row["spheres"]["depth_mm"],
                                  f"the planner overlaps by {row['spheres']['depth_mm']} mm at the "
                                  f"pose this table commits")

    def test_every_retract_lies_inside_the_planner_envelope(self) -> None:
        limits = _module("_planner_limits.py")
        lower, upper = limits.planner_envelope_rad()
        for row in self.rows:
            for axis, value in enumerate(row["retract"]):
                with self.subTest(arm=row["arm"], hand=row["hand"], axis=axis):
                    self.assertGreaterEqual(value, lower[axis])
                    self.assertLessEqual(value, upper[axis])

    def test_a_moved_retract_is_the_anchor_plus_its_steps(self) -> None:
        """The steps are what a reader checks the pose against, so they have to reproduce it."""
        step = float(self.rule["step_rad"])
        for row in self.rows:
            with self.subTest(arm=row["arm"], hand=row["hand"]):
                expected = list(row["anchor"])
                for joint, count in zip((1, 2, 3, 4, 5), row["steps"]):
                    expected[joint] = round(expected[joint] + count * step, 6)
                self.assertEqual([round(v, 6) for v in row["retract"]], [round(v, 6) for v in expected])

    def test_the_geometry_it_was_measured_on_is_the_geometry_that_ships(self) -> None:
        """⭐ A re-bake of any bundle invalidates the table, and this is what says so without an engine."""
        for row in self.rows:
            bundles = dict(row["bundles"])
            with self.subTest(arm=row["arm"], bundle="arm"):
                self.assertEqual(
                    hashlib.sha256(collision_mesh_bundle(row["arm"]).read_bytes()).hexdigest(),
                    bundles.pop("arm"))
            for hand, digest in bundles.items():
                with self.subTest(arm=row["arm"], bundle=hand):
                    self.assertEqual(hashlib.sha256(hand_mesh_bundle(hand).read_bytes()).hexdigest(), digest)

    def test_the_hash_check_can_fail(self) -> None:
        """⭐ THE CONTROL for the check above: one changed byte in a recorded digest must be caught."""
        row = self.rows[0]
        digest = row["bundles"]["arm"]
        altered = ("0" if digest[0] != "0" else "1") + digest[1:]
        self.assertNotEqual(
            hashlib.sha256(collision_mesh_bundle(row["arm"]).read_bytes()).hexdigest(), altered)

    def test_the_table_says_which_placement_and_which_plates_it_judged(self) -> None:
        self.assertEqual(self.rule["placement"], "+Y+X")
        self.assertIn(20.0, self.rule["plates_mm_judged_for_mounting_face_hands"])
        self.assertIn(0.0, self.rule["plates_mm_judged_for_mounting_face_hands"])
        self.assertEqual(self.rule["engine"], "coal")


class TheAnchorsAreIsaacsPosesTests(unittest.TestCase):
    def test_every_anchor_is_the_lula_default_q_where_isaac_is_installed(self) -> None:
        anchors = _module("_retract_rule.py").ANCHORS
        isaac = Path("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts/"
                     "isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots")
        if not isaac.is_dir():
            self.skipTest(f"no Isaac motion policy configs at {isaac}")
        for arm, anchor in anchors.items():
            description = isaac / arm / "rmpflow" / f"{arm}_robot_description.yaml"
            with self.subTest(arm=arm):
                if not description.is_file():
                    self.skipTest(f"{description} is absent")
                lula = yaml.safe_load(description.read_text(encoding="utf-8"))
                self.assertEqual([round(float(v), 6) for v in lula["default_q"]],
                                 [round(float(v), 6) for v in anchor])

    def test_a_pair_that_moved_moved_because_a_judge_refused_the_anchor(self) -> None:
        """⭐ The evidence that a move was necessary, per pair, from the row itself.

        Each row records what the ANCHOR measured in both models. A pair only moves when one of them refused it:
        either the exact meshes kept less than the guard margin plus the reserve, or the planner's own sphere model
        overlapped there. A row that moved while both judges cleared the anchor would mean the rule walked for
        nothing, and nothing else in this file would notice.
        """
        table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))
        needed = float(table["rule"]["guard_margin_mm"]) + float(table["rule"]["reserve_mm"])
        moved = [row for row in table["retracts"] if any(row["steps"])]
        self.assertTrue(moved, "no pair moved, so this assertion can no longer fail")
        for row in moved:
            with self.subTest(arm=row["arm"], hand=row["hand"], plate=row["plate_mm"]):
                meshes_refused = row["exact"]["anchor_clearance_mm"] < needed
                planner_refused = row["spheres"]["anchor_depth_mm"] is not None
                self.assertTrue(meshes_refused or planner_refused,
                                "this pair moved although both judges cleared the anchor")

    def test_a_pair_that_kept_the_anchor_was_cleared_by_both(self) -> None:
        """⭐ THE CONTROL on the one above: the other half of the rule, read from the same two fields."""
        table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))
        needed = float(table["rule"]["guard_margin_mm"]) + float(table["rule"]["reserve_mm"])
        kept = [row for row in table["retracts"] if not any(row["steps"])]
        self.assertTrue(kept, "no pair kept its anchor, so this assertion can no longer fail")
        for row in kept:
            with self.subTest(arm=row["arm"], hand=row["hand"], plate=row["plate_mm"]):
                self.assertGreaterEqual(row["exact"]["anchor_clearance_mm"], needed)
                self.assertIsNone(row["spheres"]["anchor_depth_mm"])


if __name__ == "__main__":
    unittest.main()
