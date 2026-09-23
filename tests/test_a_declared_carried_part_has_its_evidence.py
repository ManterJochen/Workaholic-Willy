"""A cell that declares its carried part's length can start its planner on the owner's arm and hand.

The audit of the owner's cell, 2026-09-23 (lens "world", and "commissioning"): leaving
``safety.planning_world.payload.length_mm`` undeclared plans every carry as an empty hand, and declaring it, as the
desk's carried-part row says to, reserves 16 attach slots, and every committed evidence file was measured with 0. The
desk and the planner start then refused ``ur10_robotiq_hande_c20mm_+Z+X_m4mm_a16.json is not there``, so a cell that
followed the checklist could not plan at all.

The files were measured on this box's GPU on 2026-09-23 with ``scripts/curobo/matrix_gate.py --attach 16`` for the UR10
with the Hand-E (0 and 20 mm plates) and the 2F-85, at both placements, about 65 s each. Every one reached b1 on the
same 1,482 poses with the same counts as its a0 sibling, and differs from it only in the attach slots and the composed
hash, which covers the attach link. A real planner started on the ur10,hande profile with a 60 mm carried part at
+Y+X and at +Z+X, and was admitted by these files (25.5 s and 26.2 s).
"""

from __future__ import annotations

import json
import unittest

from src.config import ConfigTree
from src.robot.safety.planning.evidence import EVIDENCE_DIR, desk_evidence_refusal

_CARRIED = {"robot.safety.self_collision.planner_margin_mm": 4.0,
            "robot.safety.planning_world.payload.length_mm": 60.0}
_FLANGE = {"source": "willy", "offset_mm": [0.0, 0.0, 156.2], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _tree(profile: str, **extra: object) -> object:
    tree = ConfigTree.from_directory(profile=profile).load().with_values({**_CARRIED, **extra})
    assert tree.ok, tree.error
    return tree


class TheOwnersCombinationIsAdmittedTests(unittest.TestCase):

    def test_a_declared_carried_part_on_the_ur10_with_the_hand_e_is_admitted_at_both_placements(self) -> None:
        """⛔ The finding: 'nothing has measured ur10 with robotiq_hande ... 16 attach slots'."""
        rows = {"the hand's own frame, +Y+X": ({}, "ur10_robotiq_hande_c20mm_+Y+X_m4mm_a16.json"),
                "a real flange's tool frame, +Z+X": ({"robot.gripper.tool_frame": _FLANGE},
                                                     "ur10_robotiq_hande_c20mm_+Z+X_m4mm_a16.json")}
        for label, (extra, name) in rows.items():
            with self.subTest(label):
                said, evidence = desk_evidence_refusal(_tree("ur10,hande", **extra).robot)  # type: ignore[attr-defined]
                self.assertIsNone(said)
                assert evidence is not None
                self.assertEqual(evidence.path.name, name)

    def test_every_measured_attach_file_proves_the_same_robot_as_its_empty_hand_sibling(self) -> None:
        files = sorted(EVIDENCE_DIR.glob("ur10_*_a16.json"))
        self.assertEqual(len(files), 6)
        for path in files:
            with self.subTest(file=path.name):
                carried = json.loads(path.read_text(encoding="utf-8"))
                empty = json.loads(path.with_name(path.name.replace("_a16.json", "_a0.json")).read_text(encoding="utf-8"))
                self.assertEqual(carried.pop("attach_spheres"), 16)
                self.assertEqual(empty.pop("attach_spheres"), 0)
                self.assertNotEqual(carried.pop("composed_sha256"), empty.pop("composed_sha256"),
                                    "the attach link is in the planner's config, so the composed hash moves")
                self.assertEqual(carried, empty)


class ARefusalNamesWhereTheSlotsComeFromTests(unittest.TestCase):

    def test_an_unmeasured_carried_part_says_it_is_the_carried_part(self) -> None:
        said, evidence = desk_evidence_refusal(_tree("ur5e,hande").robot)  # type: ignore[attr-defined]
        self.assertIsNone(evidence)
        assert said is not None
        self.assertIn("ur5e_robotiq_hande_c20mm_+Y+X_m4mm_a16.json is not there", said)
        self.assertIn("--attach 16", said)
        self.assertIn("safety.planning_world.payload declares a carried part's length_mm", said)
        self.assertIn("ur5e_robotiq_hande_c20mm_+Y+X_m4mm_a0.json measured this combination without them", said)
        self.assertIn("plans every carry as an empty hand", said)

    def test_a_missing_file_without_attach_slots_says_nothing_about_them(self) -> None:
        """The control: a margin nobody measured is refused as before, with no word about a carried part."""
        tree = ConfigTree.from_directory(profile="ur5e,hande").load().with_values(
            {"robot.safety.self_collision.planner_margin_mm": 7.0})
        said, _ = desk_evidence_refusal(tree.robot)
        assert said is not None
        self.assertIn("m7mm_a0.json is not there", said)
        self.assertNotIn("carried part", said)


if __name__ == "__main__":
    unittest.main()
