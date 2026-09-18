"""The gate that measures one combination through the sidecar, and the counts it writes down (S20).

S19 designed the record; this writes it. The parts that need a GPU are the box's (S21), and what is held here is
everything that can be wrong without one: which poses were judged, how the two models are compared, and that a
second run over an unchanged tree writes the same bytes.

⭐ **ONE POSE SET, SHARED.** The exact judge, the fidelity probe, the hand link equivalence probe and this gate all
call `scripts/curobo/_matrix_gate.pose_set`. A gate that generated its own would compare two measurements of
different things, and the difference would read as a finding.

⚠ **A COUNT IS NOT A MEASUREMENT WITHOUT ITS POSES.** Two runs at different seeds judge different space, so the
record carries `poses_sha256` beside the count. Without it a file saying 1,483 poses cannot be told from another
file saying 1,483 poses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_ROOT = Path(__file__).resolve().parents[1]
_CUROBO = _ROOT / "scripts" / "curobo"


def _pure():
    """`_matrix_gate.py` by path: it lives beside the cuRobo scripts and must load without the repository."""
    path = _CUROBO / "_matrix_gate.py"
    spec = importlib.util.spec_from_file_location("_matrix_gate_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ThePoseSetIsOneRuleAndSaysWhichPoses(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = _pure()
        cls.retract = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

    def test_the_retract_is_the_first_pose(self) -> None:
        """The pose a sidecar starts from is judged first, because a cell that cannot start judges nothing else."""
        poses, kinds = self.gate.pose_set(self.retract)
        self.assertEqual(poses[0], self.retract)
        self.assertEqual(kinds[0], "retract")

    def test_the_count_is_one_plus_the_draws_plus_the_sweep(self) -> None:
        poses, _ = self.gate.pose_set(self.retract)
        self.assertEqual(len(poses), 1 + self.gate.RANDOM_POSES + 481)

    def test_two_calls_are_the_same_set(self) -> None:
        self.assertEqual(self.gate.poses_sha256(self.gate.pose_set(self.retract)[0]),
                         self.gate.poses_sha256(self.gate.pose_set(self.retract)[0]))

    def test_another_seed_is_another_set_and_the_hash_says_so(self) -> None:
        """⭐ The control that makes the hash worth carrying: the count is identical and the hash is not."""
        first, _ = self.gate.pose_set(self.retract)
        second, _ = self.gate.pose_set(self.retract, seed=self.gate.SEED + 1)
        self.assertEqual(len(first), len(second))
        self.assertNotEqual(self.gate.poses_sha256(first), self.gate.poses_sha256(second))

    def test_another_retract_is_another_set(self) -> None:
        other = [0.1, -1.57, 1.57, 0.0, 0.0, 0.0]
        self.assertNotEqual(self.gate.poses_sha256(self.gate.pose_set(self.retract)[0]),
                            self.gate.poses_sha256(self.gate.pose_set(other)[0]))


class HowTheTwoModelsAreCompared(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = _pure()

    def test_a_verdict_with_its_pair_named_agrees_with_itself(self) -> None:
        said = self.gate.attribution_disagreements(
            [True, False, True], [("hand", "wrist_1_link"), None, ("forearm", "wrist_2")], [3.1, None, 0.4])
        self.assertEqual(said, 0)

    def test_a_collision_with_nothing_to_name_is_one_disagreement(self) -> None:
        """⛔ The control from the design: one depth set to None where the term says collides counts 1."""
        said = self.gate.attribution_disagreements(
            [True, False], [("hand", "wrist_1_link"), None], [None, None])
        self.assertEqual(said, 1)

    def test_a_pair_named_where_the_term_says_clear_is_one_too(self) -> None:
        said = self.gate.attribution_disagreements([False, False], [("hand", "wrist_1_link"), None], [2.0, None])
        self.assertEqual(said, 1)

    def test_rows_that_do_not_line_up_are_refused_rather_than_counted(self) -> None:
        with self.assertRaises(ValueError) as refused:
            self.gate.attribution_disagreements([True, False], [None], [None])
        self.assertIn("one row per pose", str(refused.exception))

    def test_a_false_clear_is_the_planner_clear_and_the_meshes_not(self) -> None:
        said = self.gate.false_clears([False, False, True, True], [True, False, False, True])
        self.assertEqual(said, 1, "only the pose the planner cleared and the meshes refused counts")

    def test_the_two_counts_are_different_questions(self) -> None:
        """A model can be perfectly self consistent and still disagree with the meshes; b1 asks only the first."""
        collides = [False, False]
        self.assertEqual(self.gate.attribution_disagreements(collides, [None, None], [None, None]), 0)
        self.assertEqual(self.gate.false_clears(collides, [False, False]), 2)


class WhatTheGateWrites(unittest.TestCase):
    """The file, end to end, from counts a box would have produced."""

    def _evidence(self, **over: object):
        from src.robot.safety.planning.evidence import CombinationEvidence, Measured

        gate = _pure()
        poses, _ = gate.pose_set([0.0, -1.57, 1.57, 0.0, 0.0, 0.0])
        measured = Measured(poses=len(poses), poses_sha256=gate.poses_sha256(poses), retract_clear=True,
                            refused=12, false_clears=34, attribution_disagreements=0, control_mm=0.0)
        fields: dict[str, object] = {
            "arm": "ur5e", "hand": "robotiq_2f85", "coupling_mm": 0.0, "approach": "+Y", "closing": "+X",
            "planner_margin_mm": 4.0, "attach_spheres": 0, "guard_margin_mm": 10.0,
            "composed_sha256": "c" * 64, "urdf_sha256": "u" * 64, "arm_descriptor_sha256": "a" * 64,
            "hand_map_sha256": "h" * 64, "guard_sha256": "g" * 64, "measured": measured,
        }
        fields.update(over)
        return CombinationEvidence(**fields)  # type: ignore[arg-type]

    def test_a_second_run_over_an_unchanged_tree_writes_the_same_bytes(self) -> None:
        """No timestamps, sorted keys: a diff in this folder is a change in what was measured."""
        self.assertEqual(self._evidence().evidence_bytes(), self._evidence().evidence_bytes())

    def test_the_file_is_json_a_person_can_read(self) -> None:
        held = json.loads(self._evidence().evidence_bytes().decode("utf-8"))
        self.assertEqual(held["arm"], "ur5e")
        self.assertEqual(held["measured"]["poses"], 1482)
        self.assertEqual(len(held["measured"]["poses_sha256"]), 64)

    def test_it_lands_under_the_name_of_its_own_combination(self) -> None:
        with TemporaryDirectory() as folder:
            evidence = self._evidence()
            path = Path(folder) / evidence.path.name
            path.write_bytes(evidence.evidence_bytes())
            self.assertEqual(path.name, "ur5e_robotiq_2f85_c0mm_+Y+X_m4mm_a0.json")

    def test_false_clears_do_not_stop_it_reaching_the_bar(self) -> None:
        from src.robot.safety.planning.evidence import REQUIRED_CRITERION

        self.assertEqual(self._evidence().criterion, REQUIRED_CRITERION)


class EverythingTheGateDoesBeforeItNeedsAGpu(unittest.TestCase):
    """⭐ The half that runs off box, exercised for real rather than read as source.

    A gate whose only proof is a box run is a gate nobody can fix on a laptop, and the parts below are where the
    mistakes of this arc have actually lived: a hand resolved to the wrong placement, a guard composed without its
    plate, a control that was never zero because it compared two different things.
    """

    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("_gate_cli_under_test", _CUROBO / "matrix_gate.py")
        assert spec is not None and spec.loader is not None
        cls.cli = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.cli
        sys.path.insert(0, str(_CUROBO))
        spec.loader.exec_module(cls.cli)

    def _resolved(self, arm: str, hand: str, plate: float):
        from src.contracts import chosen
        from src.robot.safety.planning.hand import planner_hand

        resolved = planner_hand(self.cli._robot_config(arm, hand, plate, 4.0, None))
        assert chosen(resolved)
        return resolved

    def test_the_cell_it_builds_is_the_cell_a_profile_would_declare(self) -> None:
        resolved = self._resolved("ur3e", "robotiq_hande", 20.0)
        self.assertEqual(resolved.model, "robotiq_hande")
        self.assertEqual(resolved.coupling_mm, 20.0)
        self.assertEqual(resolved.origin, "mounting_face")

    def test_the_control_comes_out_at_zero_on_the_arms_this_repository_ships(self) -> None:
        """⭐ The real calibration, on real committed data: the ruler now against the table then.

        Not exactly zero, and it cannot be: the table records three decimals, so up to 5e-4 mm of its own rounding
        is inside the comparison. Held against the tolerance the criterion uses, not against a literal zero.
        """
        from src.robot.safety.planning.evidence import _CONTROL_TOLERANCE_MM

        for arm, hand, plate in (("ur5e", "robotiq_2f85", 0.0), ("ur3e", "robotiq_hande", 20.0)):
            with self.subTest(arm=arm, hand=hand):
                resolved = self._resolved(arm, hand, plate)
                from src.robot.safety.planning.robot.retract_table import read_retract

                retract = read_retract(arm, hand, plate, 4.0, placement="+Y+X")
                judge = self.cli._exact_judge().ExactJudge(arm, hand, plate, resolved.placement)
                recorded = self.cli._recorded_clearance(arm, hand, plate, 4.0, placement="+Y+X")
                assert recorded is not None, f"{arm} with {hand} has no recorded clearance to calibrate against"
                self.assertLessEqual(abs(judge.nearest(retract)[0] - recorded), _CONTROL_TOLERANCE_MM)

    def test_a_combination_the_table_never_judged_has_nothing_to_calibrate_against(self) -> None:
        """The control has to be able to be absent, or its presence says nothing."""
        self.assertIsNone(self.cli._recorded_clearance("ur5e", "robotiq_2f85", 999.0, 4.0, placement="+Y+X"))

    def test_a_declared_frame_is_the_placement_the_gate_measures(self) -> None:
        """⛔ UM lane S23. This branch wrote a key the schema does not know and had never run: S21 measured every
        combination with the frame undeclared. A real UR declares +Z, and that is the combination S23 measures."""
        from src.robot.safety.planning.hand import planner_hand

        sim = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
        for rotation, expected in (((0.0, 0.0, 0.0, 1.0), ("+Z", "+X")), (sim, ("+Y", "+X")), (None, ("+Y", "+X"))):
            with self.subTest(rotation=rotation):
                placement = planner_hand(self.cli._robot_config("ur5e", "robotiq_2f85", 0.0, 4.0, rotation)).placement
                self.assertEqual((placement.approach, placement.closing), expected)

    def test_a_placement_the_table_never_judged_has_nothing_to_calibrate_against(self) -> None:
        self.assertIsNone(self.cli._recorded_clearance("ur5e", "robotiq_2f85", 0.0, 4.0, placement="+Z+Y"))

    def test_the_guard_it_composes_carries_the_hand_and_its_plate(self) -> None:
        from src.robot.safety._fcl_self_collision import composed_parts
        from src.robot.safety.planning.evidence import guard_sha256

        resolved = self._resolved("ur3e", "robotiq_hande", 20.0)
        parts = composed_parts("ur3e", None, resolved.guard_variant, 20.0,
                               placement=resolved.placement, coupling_boxes=resolved.coupling_boxes)
        self.assertEqual(len(parts), 9, "six arm links and three hand parts")
        bare = composed_parts("ur3e", None, resolved.guard_variant, 0.0, placement=resolved.placement)
        self.assertNotEqual(guard_sha256(parts), guard_sha256(bare), "the plate has to be inside the hash")

    def test_the_shipped_cells_send_no_coupling_body_and_that_is_the_measured_answer(self) -> None:
        from src.robot.safety.planning.body_link import coupling_bodies

        self.assertEqual(coupling_bodies(self._resolved("ur3e", "robotiq_hande", 20.0)), [])


class TheGateItselfLoadsAndSaysWhatItNeeds(unittest.TestCase):
    """The CLI is read as source here: running it needs a sidecar, which is S21's box run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (_CUROBO / "matrix_gate.py").read_text(encoding="utf-8")

    def test_it_measures_rather_than_plans(self) -> None:
        """⭐ A sidecar that refuses its own start pose must stay UP, or a refused combination cannot be measured."""
        self.assertIn("measure_only=True", self.source)

    def test_it_asks_for_the_pair_names_because_the_count_needs_them(self) -> None:
        self.assertIn("name_pairs=True", self.source)

    def test_it_reads_the_pose_set_rather_than_making_one(self) -> None:
        self.assertIn("pose_set", self.source)
        self.assertNotIn("default_rng", self.source, "a second generator here would be a second pose set")

    def test_it_writes_nothing_without_being_asked(self) -> None:
        self.assertIn("--write", self.source)

    def test_it_refuses_a_descriptor_that_can_name_no_pair_at_all(self) -> None:
        """Otherwise every colliding pose reads as an attribution disagreement and the file records a fiction."""
        self.assertIn("names no pair", self.source)

    def test_every_client_call_it_makes_is_one_the_client_has(self) -> None:
        """⛔ The first draft called `client.stop()`, which does not exist: the client closes.

        A method name that is wrong only shows up when a sidecar actually starts, which is a box run away, and by
        then the gate is holding a GPU. So the calls are read out of the source and checked against the class.
        """
        import ast

        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        tree = ast.parse(self.source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "client"
        }
        self.assertTrue(called, "this test proves nothing if the gate calls the client through another name")
        for name in sorted(called):
            with self.subTest(call=name):
                self.assertTrue(hasattr(CuroboPlanClient, name), f"CuroboPlanClient has no {name}()")

    def test_it_closes_the_sidecar_it_opens(self) -> None:
        self.assertIn("with CuroboPlanClient(", self.source)


if __name__ == "__main__":
    unittest.main()
