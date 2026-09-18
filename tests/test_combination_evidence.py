"""One arm and one hand are admitted by a file somebody measured, not by a table somebody typed (S19).

The arc goal is every UR arm planning collision free with every gripper it can carry. Which pairs actually do is a
MEASUREMENT, and until now it lived in two places that could not both be right: `hand__admitted_arms`, a list baked
into each hand bundle, and the retract table, which knows a pose but not whether the planner and the guard agree
about the rest of the space. This step designs the record and its key; S20 writes them, S21 measures them on the
box, S22 makes a planner refuse without one.

⭐ **THE KEY IS THE COMBINATION, NOT THE PAIR.** The same arm and hand disagree at different plates, different
placements and different planner margins, so all of them are in the name:
`{arm}_{hand}_c{coupling}mm_{approach}{closing}_m{margin}mm_a{attach}.json`. A ur3e at 4 mm and the same ur3e at
10 mm are two files, because they are two measurements, and B6 measured exactly that: at 10 mm the UR5 and the UR10
have no pose at all.

⛔ **AND A HASH THAT WAS NOT REPORTED IS NOT A HASH THAT MATCHED.** A sidecar too old to say what it loaded reports
UNSET, and evidence keyed on hashes has to refuse that rather than skip the comparison. Skipping is what an
inherited `hand__admitted_arms` did for a year.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.contracts import UNSET
from src.robot.safety.planning.evidence import (
    REQUIRED_CRITERION,
    CombinationEvidence,
    Criterion,
    Measured,
    evidence_path,
    evidence_refusal,
)

#: The combination the sim cell runs, as its file is named. The margin is 4, not the 10 this design was drafted
#: against: B6 measured the family at 4 mm and the sim layer ships it, so the drafted name was already stale.
_SIM_KEY = "ur5e_robotiq_2f85_c0mm_+Y+X_m4mm_a0.json"


def _measured(**over: object) -> Measured:
    """What the gate counted: a pass unless a caller asks for something else."""
    numbers: dict[str, object] = {
        "poses": 1483, "poses_sha256": "p" * 64, "retract_clear": True, "refused": 0, "false_clears": 0,
        "attribution_disagreements": 0, "control_mm": 0.0,
    }
    numbers.update(over)
    return Measured(**numbers)  # type: ignore[arg-type]


def _evidence(**over: object) -> CombinationEvidence:
    fields: dict[str, object] = {
        "arm": "ur5e", "hand": "robotiq_2f85", "coupling_mm": 0.0, "approach": "+Y", "closing": "+X",
        "planner_margin_mm": 4.0, "attach_spheres": 0, "guard_margin_mm": 10.0,
        "composed_sha256": "c" * 64, "urdf_sha256": "u" * 64, "arm_descriptor_sha256": "a" * 64,
        "hand_map_sha256": "h" * 64, "guard_sha256": "g" * 64,
        "measured": _measured(),
    }
    fields.update(over)
    return CombinationEvidence(**fields)  # type: ignore[arg-type]


class TheKeyNamesEverythingThatCanDisagree(unittest.TestCase):

    def test_the_sim_cell_is_the_name_the_gate_will_write(self) -> None:
        self.assertEqual(
            evidence_path(arm="ur5e", hand="robotiq_2f85", coupling_mm=0.0, approach="+Y", closing="+X",
                          planner_margin_mm=4.0, attach_spheres=0).name,
            _SIM_KEY,
        )

    def test_two_margins_are_two_files(self) -> None:
        """⭐ Not cosmetic: B6 measured the UR5 as having no pose at all at 10 mm and one at 4 mm."""
        four = evidence_path(arm="ur3e", hand="robotiq_hande", coupling_mm=20.0, approach="+Y", closing="+X",
                             planner_margin_mm=4.0, attach_spheres=0)
        ten = evidence_path(arm="ur3e", hand="robotiq_hande", coupling_mm=20.0, approach="+Y", closing="+X",
                            planner_margin_mm=10.0, attach_spheres=0)
        self.assertNotEqual(four.name, ten.name)
        self.assertIn("m4mm", four.name)
        self.assertIn("m10mm", ten.name)

    def test_two_placements_are_two_files(self) -> None:
        plus_y = evidence_path(arm="ur5e", hand="robotiq_hande", coupling_mm=20.0, approach="+Y", closing="+X",
                               planner_margin_mm=4.0, attach_spheres=0)
        plus_z = evidence_path(arm="ur5e", hand="robotiq_hande", coupling_mm=20.0, approach="+Z", closing="+X",
                               planner_margin_mm=4.0, attach_spheres=0)
        self.assertNotEqual(plus_y.name, plus_z.name)

    def test_two_plates_are_two_files(self) -> None:
        bare = evidence_path(arm="ur5e", hand="robotiq_hande", coupling_mm=0.0, approach="+Y", closing="+X",
                             planner_margin_mm=4.0, attach_spheres=0)
        plated = evidence_path(arm="ur5e", hand="robotiq_hande", coupling_mm=20.0, approach="+Y", closing="+X",
                               planner_margin_mm=4.0, attach_spheres=0)
        self.assertNotEqual(bare.name, plated.name)

    def test_a_placement_nobody_declared_has_no_file_to_look_for(self) -> None:
        """A cell whose tool frame places no hand cannot be measured, and a path would suggest it could."""
        with self.assertRaises(ValueError) as refused:
            evidence_path(arm="ur5e", hand="robotiq_hande", coupling_mm=20.0, approach="", closing="+X",
                          planner_margin_mm=4.0, attach_spheres=0)
        self.assertIn("places", str(refused.exception))


class WhatTheFirstRungAsksFor(unittest.TestCase):
    """b1 is narrow on purpose, and what it leaves out is as much a decision as what it holds."""

    def test_a_clean_measurement_meets_it(self) -> None:
        self.assertEqual(_evidence().criterion, REQUIRED_CRITERION)

    def test_a_retract_the_planner_refuses_does_not(self) -> None:
        """The sidecar never becomes ready from a pose its own spheres call a self collision."""
        self.assertLess(_evidence(measured=_measured(retract_clear=False)).criterion, REQUIRED_CRITERION)

    def test_a_model_that_contradicts_itself_does_not(self) -> None:
        """A pose reported as colliding with no pair to name is the planner disagreeing with its own report."""
        self.assertLess(_evidence(measured=_measured(attribution_disagreements=1)).criterion, REQUIRED_CRITERION)

    def test_a_control_that_did_not_come_out_at_zero_does_not(self) -> None:
        self.assertLess(_evidence(measured=_measured(control_mm=0.4)).criterion, REQUIRED_CRITERION)

    def test_false_clears_do_not_stop_it_and_that_is_the_decision(self) -> None:
        """⚠ b1 does NOT ask the sphere model and the exact meshes to agree everywhere.

        They do not, by construction: the fit reaches past the body and the guard judges the body. What b1 asks is
        that the planner can start and that its own reporting is consistent. A rung that asks for agreement
        everywhere is a later one, and inventing it here would record a standard nothing measures.
        """
        self.assertEqual(_evidence(measured=_measured(false_clears=34)).criterion, REQUIRED_CRITERION)

    def test_a_measured_refusal_keeps_its_counts_and_is_not_a_pass(self) -> None:
        refused = _evidence(measured=_measured(retract_clear=False, refused=371293))
        self.assertLess(refused.criterion, REQUIRED_CRITERION)
        self.assertEqual(refused.measured.refused, 371293, "a refusal that forgets what it counted proves nothing")

    def test_the_rungs_are_ordered_so_raising_the_bar_is_a_value_and_not_a_mechanism(self) -> None:
        self.assertLess(Criterion.NONE, Criterion.B1)
        self.assertEqual(REQUIRED_CRITERION, Criterion.B1)


class WhatARefusalRefuses(unittest.TestCase):
    """Every way a file can fail to admit the cell in front of it, each naming what it found."""

    def _written(self, folder: str, **over: object) -> Path:
        evidence = _evidence(**over)
        path = Path(folder) / evidence.path.name
        path.write_text(json.dumps(evidence.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _asked(self, folder: str, **over: object) -> "str | None":
        asked: dict[str, object] = {
            "arm": "ur5e", "hand": "robotiq_2f85", "coupling_mm": 0.0, "approach": "+Y", "closing": "+X",
            "planner_margin_mm": 4.0, "attach_spheres": 0, "guard_margin_mm": 10.0,
            "composed_sha256": "c" * 64, "urdf_sha256": "u" * 64, "arm_descriptor_sha256": "a" * 64,
            "hand_map_sha256": "h" * 64, "guard_sha256": "g" * 64, "evidence_dir": Path(folder),
        }
        asked.update(over)
        return evidence_refusal(**asked)  # type: ignore[arg-type]

    def test_a_pair_with_a_matching_file_is_admitted(self) -> None:
        with TemporaryDirectory() as folder:
            self._written(folder)
            self.assertIsNone(self._asked(folder))

    def test_no_file_names_the_path_and_what_writes_it(self) -> None:
        with TemporaryDirectory() as folder:
            said = self._asked(folder)
            assert said is not None
            self.assertIn(_SIM_KEY, said)
            self.assertIn("matrix_gate.py", said)

    def test_a_config_that_is_not_the_one_measured_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            self._written(folder)
            said = self._asked(folder, composed_sha256="d" * 64)
            assert said is not None
            self.assertIn("composed_sha256", said)

    def test_a_urdf_that_is_not_the_one_measured_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            self._written(folder)
            said = self._asked(folder, urdf_sha256="v" * 64)
            assert said is not None
            self.assertIn("urdf_sha256", said)

    def test_a_sidecar_that_reported_nothing_is_refused_rather_than_skipped(self) -> None:
        """⛔ The whole point. An unreported hash is not a hash that matched."""
        with TemporaryDirectory() as folder:
            self._written(folder)
            said = self._asked(folder, composed_sha256=UNSET)
            assert said is not None
            self.assertIn("did not say", said)

    def test_a_guard_at_another_margin_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            self._written(folder)
            said = self._asked(folder, guard_margin_mm=8.0)
            assert said is not None
            self.assertIn("guard", said)

    def test_guard_geometry_that_moved_is_refused(self) -> None:
        """A plate that became a body changes this, which is how UM8 reaches the evidence at all."""
        with TemporaryDirectory() as folder:
            self._written(folder)
            said = self._asked(folder, guard_sha256="z" * 64)
            assert said is not None
            self.assertIn("guard_sha256", said)

    def test_a_file_below_the_required_rung_is_refused_and_says_what_it_measured(self) -> None:
        with TemporaryDirectory() as folder:
            self._written(folder, measured=_measured(retract_clear=False, refused=371293))
            said = self._asked(folder)
            assert said is not None
            self.assertIn("371,293", said)

    def test_a_file_that_is_not_json_is_refused_by_name(self) -> None:
        with TemporaryDirectory() as folder:
            (Path(folder) / _SIM_KEY).write_text("not json at all", encoding="utf-8")
            said = self._asked(folder)
            assert said is not None
            self.assertIn(_SIM_KEY, said)

    def test_a_file_whose_name_does_not_match_its_contents_is_refused(self) -> None:
        """⛔ Otherwise a file could be copied to admit a combination nobody measured."""
        with TemporaryDirectory() as folder:
            evidence = _evidence(arm="ur3e")
            (Path(folder) / _SIM_KEY).write_text(json.dumps(evidence.to_dict()), encoding="utf-8")
            said = self._asked(folder)
            assert said is not None
            self.assertIn("names", said)


class TheGuardHashMovesWithTheGeometry(unittest.TestCase):
    """⭐ It is over the PLACED arrays, which is the only way it can see what moved.

    Three things move a cell's guard geometry without touching a bundle file: a declared tool frame turning the
    hand, a plate shifting a mounting face map, and a coupling body UM8 adds. A hash over the files would read
    identical through all three, and the evidence would admit a robot that is not the one in front of it.
    """

    def _parts(self, **over: object) -> dict:
        from src.robot.safety._fcl_self_collision import composed_parts

        asked: dict[str, object] = {"model": "ur5e", "mesh_name": "robotiq_hande", "coupling_mm": 20.0}
        asked.update(over)
        return composed_parts(**asked)  # type: ignore[arg-type]

    def test_the_same_cell_hashes_the_same_twice(self) -> None:
        from src.robot.safety.planning.evidence import guard_sha256

        self.assertEqual(guard_sha256(self._parts()), guard_sha256(self._parts()))

    def test_a_different_plate_is_a_different_hash(self) -> None:
        from src.robot.safety.planning.evidence import guard_sha256

        self.assertNotEqual(guard_sha256(self._parts()), guard_sha256(self._parts(coupling_mm=0.0)))

    def test_a_plate_that_became_a_body_is_a_different_hash(self) -> None:
        from src.robot.safety.planning._declared_body import stacked_boxes
        from src.robot.safety.planning.evidence import guard_sha256

        boxes, _ = stacked_boxes([("hande_adapter", 20.0, (31.5, 31.5))], axis=1)
        self.assertNotEqual(guard_sha256(self._parts()), guard_sha256(self._parts(coupling_boxes=tuple(boxes))))

    def test_a_different_hand_is_a_different_hash(self) -> None:
        from src.robot.safety.planning.evidence import guard_sha256

        self.assertNotEqual(guard_sha256(self._parts()),
                            guard_sha256(self._parts(mesh_name="schunk_egu50", coupling_mm=0.0)))

    def test_it_is_a_sha256(self) -> None:
        from src.robot.safety.planning.evidence import guard_sha256

        said = guard_sha256(self._parts())
        self.assertEqual(len(said), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in said))


class TheRecordReadsBackAsItself(unittest.TestCase):

    def test_the_wire_half_round_trips(self) -> None:
        evidence = _evidence()
        self.assertEqual(CombinationEvidence.from_dict(evidence.to_dict()), evidence)

    def test_it_describes_itself_to_a_person(self) -> None:
        said = _evidence().render()
        self.assertIn("ur5e", said)
        self.assertIn("robotiq_2f85", said)
        self.assertIn("b1", said)
        self.assertNotIn("\n", said)

    def test_the_bytes_are_stable_so_a_second_run_writes_the_same_file(self) -> None:
        first = _evidence().evidence_bytes()
        self.assertEqual(first, _evidence().evidence_bytes())
        self.assertTrue(first.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
