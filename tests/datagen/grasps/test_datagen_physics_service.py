"""Asking the simulator whether a grasp holds, from Python, and the promotion gate that had no route.

⭐⭐ **THE PAIRED COMPARISON IS THE ARC'S PROMOTION GATE AND NOTHING COULD REACH IT.**
`paired_trials` has existed since the reference work and is the only instrument here that can compare
two generators without object difficulty sitting inside the comparison: `sample_trials` gives each
arm its own strata, so the two land on DIFFERENT objects, measured on `v1_proof` as 104 against 153.
In a corpus where `bin` objects fail for reasons that have nothing to do with which generator
proposed the grasp, that variance is not small. **Every generator comparison this repository has
published is an unpaired one.**

⛔ **THREE QUESTIONS, THREE FILES.** A proposal verdict and a label verdict answer different
questions, and a shared file lets one be quoted as the other.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from datagen.grasps.service import PhysicsReport, PhysicsSampling, physics_output_name


def _raw(**over: object) -> dict:
    base = {"drawn": 200, "trials": 180, "refused": 20,
            "by_source": {"gso": {"held": 60, "trials": 120, "hold_rate": 0.5, "refused": 4},
                          "ycb": {"held": 20, "trials": 60, "hold_rate": 1 / 3, "refused": 2}}}
    base.update(over)
    return base


class EachQuestionGetsItsOwnFileTests(unittest.TestCase):

    def test_a_label_run_and_a_proposal_run_do_not_share_one(self) -> None:
        """⛔ THEY ANSWER DIFFERENT QUESTIONS. One asks "did the model's own proposal hold", the
        other "did a grasp the geometry calls valid hold". Mixing them lets one be quoted as the
        other, and the first referee success rate this repository ever computed came out of exactly
        this family of file."""
        self.assertNotEqual(physics_output_name(), physics_output_name(proposals=True))

    def test_a_paired_comparison_names_its_arms(self) -> None:
        """So two comparisons of DIFFERENT arms cannot merge into one file."""
        a = physics_output_name(paired_arms=["sfe", "deep"])
        b = physics_output_name(paired_arms=["sfe", "geometric"])
        self.assertNotEqual(a, b)
        self.assertIn("sfe_vs_deep", a)

    def test_the_paired_name_wins_over_the_proposal_flag(self) -> None:
        """A paired run of proposals is still a paired run; one file, named for what varies."""
        self.assertIn("paired", physics_output_name(proposals=True, paired_arms=["a", "b"]))

    def test_every_name_is_a_grasp_physics_jsonl(self) -> None:
        for kwargs in ({}, {"proposals": True}, {"paired_arms": ["a", "b"]}):
            name = physics_output_name(**kwargs)               # type: ignore[arg-type]
            self.assertTrue(name.startswith("grasp_physics"), name)
            self.assertTrue(name.endswith(".jsonl"), name)


class TheComparisonRefusesRatherThanReportingATieTests(unittest.TestCase):
    """⛔ AN ARM THAT PROPOSED NOTHING IS A REFUSAL, NOT A TIE."""

    def test_one_arm_is_not_a_comparison(self) -> None:
        with self.assertRaises(ValueError) as caught:
            PhysicsSampling.from_dataset("d").compare(["sfe"])
        self.assertIn("at least two", str(caught.exception))

    def test_an_empty_string_does_not_count_as_an_arm(self) -> None:
        """`--configs sfe,` splits into two, and one of them is nothing."""
        with self.assertRaises(ValueError):
            PhysicsSampling.from_dataset("d").compare(["sfe", "  "])

    def test_no_shared_object_refuses_rather_than_running(self) -> None:
        with mock.patch("datagen.grasps.physics.paired_trials", return_value=[]), \
                mock.patch("datagen.grasps.physics.run_physics_sample") as run, \
                self.assertRaises(ValueError) as caught:
            PhysicsSampling.from_dataset("d").compare(["a", "b"])
        run.assert_not_called()
        self.assertIn("not a tie", str(caught.exception))

    def test_a_real_pairing_runs_and_is_marked_paired(self) -> None:
        """The control. Without it the refusals above pass for a method that never runs."""
        with mock.patch("datagen.grasps.physics.paired_trials", return_value=[1, 2]), \
                mock.patch("datagen.grasps.physics.run_physics_sample", return_value=_raw()):
            report = PhysicsSampling.from_dataset("d").compare(["sfe", "deep"])
        self.assertEqual(("sfe", "deep"), report.paired_arms)
        self.assertIn("PAIRED", report.render())

    def test_an_unpaired_run_is_NOT_marked_paired(self) -> None:
        """⚠ THE MARK IS A CLAIM ABOUT COMPARABILITY. An unpaired rate that carried it would invite
        exactly the comparison the pairing exists to make safe."""
        with mock.patch("datagen.grasps.physics.run_physics_sample", return_value=_raw()):
            report = PhysicsSampling.from_dataset("d").sample()
        self.assertEqual((), report.paired_arms)
        self.assertNotIn("PAIRED", report.render())


class TheSamplingReachesTheRefereeTests(unittest.TestCase):

    def test_the_engine_and_the_display_arrive(self) -> None:
        seen: dict = {}

        def capture(root, **kwargs):                   # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return _raw()

        with mock.patch("datagen.grasps.physics.run_physics_sample", side_effect=capture):
            PhysicsSampling.from_dataset("d", engine="mujoco", headless=False).sample()
        self.assertEqual("mujoco", seen["engine"])
        self.assertIs(False, seen["headless"])

    def test_the_computed_file_name_arrives_too(self) -> None:
        """⚠ A NAMING RULE THAT IS RIGHT AND NOT WIRED is the inert-switch shape this repository
        fences everywhere else."""
        seen: dict = {}

        def capture(root, **kwargs):                   # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return _raw()

        with mock.patch("datagen.grasps.physics.run_physics_sample", side_effect=capture), \
                mock.patch("datagen.grasps.physics.trials_from_proposals", return_value=[1]):
            PhysicsSampling.from_dataset("d").sample(proposals="p.jsonl")
        self.assertEqual("grasp_physics_proposals.jsonl", seen["out_name"])

    def test_without_proposals_no_proposal_file_is_read(self) -> None:
        with mock.patch("datagen.grasps.physics.run_physics_sample", return_value=_raw()), \
                mock.patch("datagen.grasps.physics.trials_from_proposals") as loader:
            PhysicsSampling.from_dataset("d").sample()
        loader.assert_not_called()


class TheReportKeepsBothHalvesTests(unittest.TestCase):
    """⚠ PER SOURCE, NEVER POOLED. The draw is stratified, so a pooled hold rate describes the
    sampler rather than the data."""

    def test_the_strata_survive(self) -> None:
        report = PhysicsReport(raw=_raw(), out_name="x.jsonl")
        self.assertIn("gso", report.render())
        self.assertIn("ycb", report.render())

    def test_held_sums_the_strata_rather_than_reading_a_pooled_key(self) -> None:
        self.assertEqual(80, PhysicsReport(raw=_raw(), out_name="x").held)

    def test_the_refused_count_is_reported_as_not_a_failed_grasp(self) -> None:
        """A refused trial is a scene that was not the labelled scene. Reading it as a failure would
        deflate every hold rate this repository has published."""
        self.assertIn("not a grasp", PhysicsReport(raw=_raw(), out_name="x").render())

    def test_an_empty_run_does_not_divide_by_zero(self) -> None:
        empty = PhysicsReport(raw={"trials": 0, "drawn": 0, "refused": 0, "by_source": {}},
                              out_name="x")
        self.assertEqual(0.0, empty.hold_rate)

    def test_the_rendered_text_is_ascii_and_names_its_file(self) -> None:
        text = PhysicsReport(raw=_raw(), out_name="grasp_physics.jsonl").render()
        text.encode("ascii")
        self.assertIn("grasp_physics.jsonl", text)

    def test_as_dict_survives_json(self) -> None:
        json.dumps(PhysicsReport(raw=_raw(), out_name="x", paired_arms=("a", "b")).as_dict())


if __name__ == "__main__":
    unittest.main()
