"""Every committed evidence file says what it measured, and the cells this repository ships are in there (S21).

The files under `src/robot/safety/planning/robot/evidence/` are what admits an arm to a hand from S22 on.
This holds the properties a reader has to be able to rely on without re-running a box: that a file's NAME is its own
contents, that the gate's calibration came out at zero, and that the shipped sim profile resolves to a file that
passes rather than to no file at all.

⛔ **AN EMPTY DIRECTORY MUST FAIL HERE.** Otherwise the whole file passes on a tree where the measurement was never
run, and the one thing it exists to prove is the one thing it would stop proving.
"""

from __future__ import annotations

import json
import unittest

from src.robot.safety.planning.evidence import (
    EVIDENCE_DIR,
    REQUIRED_CRITERION,
    CombinationEvidence,
    _CONTROL_TOLERANCE_MM,
)

_FILES = sorted(EVIDENCE_DIR.glob("*.json"))


class TheDirectoryHoldsMeasurements(unittest.TestCase):

    def test_there_are_files_at_all(self) -> None:
        """⛔ The negative control for this whole file: an unmeasured tree fails here and nowhere else."""
        self.assertTrue(
            _FILES,
            f"{EVIDENCE_DIR} holds no combination evidence. Measure the matrix with "
            f"scripts/curobo/matrix_sweep.py --write and commit what it writes; until then no cell may plan.",
        )

    def test_every_file_reads_back_as_a_record(self) -> None:
        for path in _FILES:
            with self.subTest(file=path.name):
                CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def test_every_file_is_named_after_its_own_contents(self) -> None:
        """⛔ Otherwise a file could be copied to admit a combination nobody measured."""
        for path in _FILES:
            with self.subTest(file=path.name):
                evidence = CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.assertEqual(evidence.path.name, path.name)

    def test_every_file_carries_the_hashes_it_was_measured_against(self) -> None:
        """An empty hash is a measurement against a sidecar that said nothing, and it would admit anything."""
        for path in _FILES:
            with self.subTest(file=path.name):
                held = json.loads(path.read_text(encoding="utf-8"))
                for key in ("composed_sha256", "urdf_sha256", "arm_descriptor_sha256", "hand_map_sha256",
                            "guard_sha256"):
                    self.assertEqual(len(held[key]), 64, f"{path.name} has no {key}")

    def test_the_gate_was_calibrated_on_every_one_of_them(self) -> None:
        """The control compares the exact clearance now against what the retract table recorded. It is the cheapest
        check that the gate measured the robot it thinks it did, and it applies to a refusal as much as to a pass."""
        for path in _FILES:
            with self.subTest(file=path.name):
                evidence = CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.assertLessEqual(abs(evidence.measured.control_mm), _CONTROL_TOLERANCE_MM)

    def test_no_file_records_a_model_that_contradicts_itself(self) -> None:
        for path in _FILES:
            with self.subTest(file=path.name):
                evidence = CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.assertEqual(evidence.measured.attribution_disagreements, 0)

    def test_every_file_judged_the_same_number_of_poses(self) -> None:
        """One pose set, one rule. A file with a different count was measured by something else."""
        counts = {json.loads(p.read_text(encoding="utf-8"))["measured"]["poses"] for p in _FILES}
        self.assertEqual(len(counts), 1, f"the files judged {sorted(counts)} poses, which is more than one rule")


class TheCellsThisRepositoryShipsAreMeasured(unittest.TestCase):
    """A profile that names a hand has to resolve to a file, or S22 turns it into a cell that cannot start."""

    def _combination(self, profile: str) -> dict:
        from src.config.loader import load_robot_config
        from src.contracts import chosen
        from src.robot.safety.planning.hand import planner_hand

        cfg = load_robot_config(profile=profile)
        hand = planner_hand(cfg)
        assert chosen(hand), f"the {profile} profile names no hand"
        placement = hand.placement
        assert chosen(placement), f"the {profile} profile places no hand: {hand.placement_refusal}"
        return {
            "arm": str(getattr(cfg.safety.self_collision, "kinematics_model", None) or cfg.ur.model),
            "hand": hand.model, "coupling_mm": hand.coupling_mm,
            "approach": placement.approach, "closing": placement.closing,
            "planner_margin_mm": float(cfg.safety.self_collision.planner_margin_mm or 0.0),
            "attach_spheres": 0,
        }

    def test_the_sim_profile_resolves_to_a_file_that_passes(self) -> None:
        from src.robot.safety.planning.evidence import evidence_path

        path = evidence_path(**self._combination("sim"))  # type: ignore[arg-type]
        self.assertTrue(path.is_file(), f"the sim profile needs {path.name} and it is not there")
        evidence = CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(evidence.criterion, REQUIRED_CRITERION, evidence.render())

    def test_a_real_ur_flange_resolves_to_a_file_that_passes(self) -> None:
        """⭐ UM lane S23. The ursim bed with cuRobo declares +Z, the UR tool axis, which is what every real UR cell
        declares. Until S23 it refused to build, and no file measured a hand along +Z; the +Z matrix is committed now."""
        from src.robot.safety.planning.evidence import evidence_path

        asked = self._combination("ursim,ursim_curobo")
        self.assertEqual((asked["approach"], asked["closing"]), ("+Z", "+X"))
        path = evidence_path(**asked)  # type: ignore[arg-type]
        self.assertTrue(path.is_file(), f"a real UR flange needs {path.name} and it is not there")
        evidence = CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(evidence.criterion, REQUIRED_CRITERION, evidence.render())

    def test_every_committed_file_is_at_a_placement_the_derivation_admits(self) -> None:
        """The two placements a cell can declare today and a file can therefore answer for: the Isaac cell's +Y and a
        real flange's +Z, both closing along +X."""
        placements = {path.name.split("_")[-3] for path in _FILES}
        self.assertEqual(placements, {"+Y+X", "+Z+X"}, sorted(placements))

    def test_a_margin_the_sim_profile_does_not_ship_resolves_to_no_file(self) -> None:
        """⭐ The control: the margin is IN the key, so a file cannot answer for a cell at another one."""
        from src.robot.safety.planning.evidence import evidence_path

        asked = self._combination("sim")
        asked["planner_margin_mm"] = 10.0
        self.assertFalse(evidence_path(**asked).is_file())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
