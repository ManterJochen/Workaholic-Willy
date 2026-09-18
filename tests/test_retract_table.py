"""The committed retract table is a geometry authority, and it says what it measured (UM5, B4).

⭐ ONE ROW PER ARM, HAND, PLATE AND PLANNER MARGIN since 2026-09-16 (owner decision). The retract used to be a
property of the arm alone, judged on the exact meshes alone, and that was wrong twice: measured over the rule's
own candidate family, at ur3's committed pose the meshes clear the Hand-E by 13.4 mm while the planner's sphere
model calls the same pose a self collision, so the sidecar never becomes ready and the cell does not come up.
The rule now takes the first pose BOTH models accept, and the pose that works is different for each hand.

``src/robot/safety/planning/robot/ur_retract.yaml`` holds one retract per arm, hand, plate, planner margin and
placement, chosen by the rule on the exact meshes and the planner's spheres. Every shipped hand is asked about on every
arm, and a customer's hand on the arms its profiles name (owner, 2026-09-17). The descriptor builder reads the table in
the cuRobo environment, where neither Coal nor the bundles can be loaded, so nothing there can check the numbers: these
are the checks.

What this file can check off the box is the table against itself and against the committed bundles. What it cannot is
the distances, which only Coal can measure; ``choose_ur_retract.py --check`` does that on the box.
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


#: The hands this repository ships. Each is judged on every arm the table judges (owner decision 4 of 2026-09-17), so a
#: cell can put any of them on any arm; any other registry hand is a customer's, judged on the arms its profiles name.
_SHIPPED_HANDS = frozenset({"robotiq_2f85", "robotiq_hande", "schunk_egu50"})


def _arms_a_profile_names(hand: str) -> set[str]:
    """Every arm a robot profile layer that names ``hand`` states, read from the layer itself.

    Read rather than loaded: a cell layer chains on its arm's layer (``--profile ur10e,acme_2f_ur10e``), so loaded alone
    it inherits the base tree's arm, and MEASURED 2026-09-17 in the customer chain trial that asked for a ur5e row
    nobody needs. A layer that names the hand and states no arm raises, naming itself: a customer layer states
    ``safety.self_collision.kinematics_model``, the arm its guard models.
    """
    arms: set[str] = set()
    for path in sorted((_ROOT / "config" / "robot").glob("robot.*.yaml")):
        robot = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("robot") or {}
        if (robot.get("gripper") or {}).get("model") != hand:
            continue
        stated = (((robot.get("safety") or {}).get("self_collision") or {}).get("kinematics_model")
                  or (robot.get("ur") or {}).get("model"))
        if not stated:
            raise AssertionError(f"{path.name} names the hand {hand} and no arm: state "
                                 f"safety.self_collision.kinematics_model, the arm its guard models")
        arms.add(str(stated))
    return arms


def _pairs_nobody_asked(table: dict, *, hands, profile_arms) -> list[str]:
    """One sentence per (arm, hand) the table should have asked about and did not, ending in the command that asks."""
    from src.robot.safety.planning.robot.retract_table import retract_command

    rule = table["rule"]
    rows, refused = table["retracts"], rule["pairs_with_no_retract"]
    asked = {(row["arm"], row["hand"]) for row in (*rows, *refused)}
    judged_arms = sorted({arm for arm, _ in asked})
    missing = []
    for hand in sorted(hands):
        arms = judged_arms if hand in _SHIPPED_HANDS else sorted(profile_arms(hand))
        for arm in arms:
            if (arm, hand) in asked:
                continue
            example = next((row for row in (*rows, *refused) if row["hand"] == hand), {})
            placement = str(example.get("placement") or rule["placement"])
            command = retract_command(arm, hand, float(example.get("plate_mm", 0.0)),
                                      float(rule["planner_margin_mm"]), placement)
            whose = "a shipped hand, judged on every arm" if hand in _SHIPPED_HANDS else "named on this arm by a profile"
            missing.append(f"{arm} was never asked about {hand} ({whose}): run {command} on a box with the cuRobo "
                           f"environment, and commit the table it writes")
    return missing


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
        """Judged or refused, but never simply absent: a pair nobody asked about is the state this replaced.

        Scoped by the owner's decision 4 of 2026-09-17: a shipped hand on every arm the table judges, a customer's
        hand on the arms its profiles name. A missing pair is named with the chooser command that asks it.
        """
        missing = _pairs_nobody_asked(self.table, hands=available_grippers(), profile_arms=_arms_a_profile_names)
        self.assertEqual(missing, [], "\n" + "\n".join(missing))

    def test_a_pair_nobody_asked_about_is_named_with_the_run_that_asks_it(self) -> None:
        """Customer chain lane C1e: a customer who judged one arm learns which pair is missing and what to run."""
        table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))

        def other(row: dict) -> bool:
            return not (row["arm"] == "ur3" and row["hand"] == "schunk_egu50")

        table["retracts"] = [row for row in table["retracts"] if other(row)]
        table["rule"]["pairs_with_no_retract"] = [row for row in table["rule"]["pairs_with_no_retract"] if other(row)]
        missing = _pairs_nobody_asked(table, hands=available_grippers(), profile_arms=_arms_a_profile_names)
        self.assertEqual(len(missing), 1, missing)
        for part in ("ur3", "schunk_egu50", "choose_ur_retract.py ur3 --hand schunk_egu50", "--tool-rotation-xyzw"):
            self.assertIn(part, missing[0])

    def test_the_committed_table_passes_the_same_method(self) -> None:
        """⭐ THE CONTROL: the unmodified table through the same method, so the removal is what fails the one above."""
        self.assertEqual(_pairs_nobody_asked(self.table, hands=available_grippers(),
                                             profile_arms=_arms_a_profile_names), [])

    def test_a_customer_hand_is_asked_only_on_the_arms_its_profiles_name(self) -> None:
        """Owner decision 4: a shipped hand on every arm, a customer's on the arms its profiles name."""
        hands = [*available_grippers(), "acme_2f"]
        missing = _pairs_nobody_asked(self.table, hands=hands, profile_arms=lambda hand: {"ur10e"})
        self.assertEqual(len(missing), 1, missing)
        self.assertIn("ur10e was never asked about acme_2f", missing[0])
        self.table["retracts"].append({"arm": "ur10e", "hand": "acme_2f", "plate_mm": 0.0})
        self.assertEqual(_pairs_nobody_asked(self.table, hands=hands, profile_arms=lambda hand: {"ur10e"}), [])

    def test_the_shipped_hands_are_the_registry_s_shipped_hands(self) -> None:
        """⭐ THE CONTROL on the scope: a shipped hand dropped from the set would quietly stop being asked everywhere."""
        self.assertLessEqual(_SHIPPED_HANDS, set(available_grippers()))

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
