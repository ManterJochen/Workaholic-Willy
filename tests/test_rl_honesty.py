"""P4.8 — the shared RL honesty/provenance helper (rl/honesty.py)."""

from __future__ import annotations

import unittest

from src.robot.grasping.rl import honesty


class DatasetProvenanceTests(unittest.TestCase):
    def test_canonical_keys_and_forwarding(self) -> None:
        p = honesty.make_dataset_provenance(
            input_source=honesty.INPUT_SIM,
            dataset_id="p4_4_cubes",
            dataset_hash="abc",
            source_pack_paths=["logs/demo/p4_4_cubes.jsonl"],
            leakage_proven=True,
            fit_for="V2 candidate reordering (off-policy shadow)",
            not_fit_for=["real-hardware lift", "geometry generalization"],
        )
        self.assertEqual(
            set(p),
            {
                "input_source",
                "dataset_id",
                "dataset_hash",
                "source_pack_paths",
                "leakage_proven",
                "fit_for",
                "not_fit_for",
            },
        )
        self.assertEqual(p["input_source"], "sim")
        self.assertIs(p["leakage_proven"], True)
        self.assertEqual(p["not_fit_for"], ["real-hardware lift", "geometry generalization"])

    def test_optional_fields_default_none(self) -> None:
        p = honesty.make_dataset_provenance(
            input_source=honesty.INPUT_REAL_HARDWARE, fit_for="x", not_fit_for=[]
        )
        self.assertIsNone(p["dataset_id"])
        self.assertIsNone(p["leakage_proven"])
        self.assertIsNone(p["source_pack_paths"])

    def test_unknown_input_source_rejected(self) -> None:
        with self.assertRaises(ValueError):
            honesty.make_dataset_provenance(input_source="bogus", fit_for="x", not_fit_for=[])


class DegeneracyNoteTests(unittest.TestCase):
    def test_zero_tuples_is_degenerate(self) -> None:
        note = honesty.single_action_degeneracy_note({"a": 0, "b": 0})
        assert note is not None
        self.assertIn("zero training tuples", note)

    def test_single_action_dominance_is_degenerate(self) -> None:
        # The committed V6 shape: one action carries every tuple.
        note = honesty.single_action_degeneracy_note(
            {"reobserve": 418, "rescan": 0, "next_viewpoint": 0}, fallback_rate=1.0
        )
        assert note is not None
        self.assertIn("DEGENERATE", note)
        self.assertIn("reobserve", note)
        self.assertIn("fallback_rate=1.000", note)

    def test_high_share_is_degenerate(self) -> None:
        note = honesty.single_action_degeneracy_note({"a": 99, "b": 1})
        self.assertIsNotNone(note)

    def test_balanced_is_not_degenerate(self) -> None:
        note = honesty.single_action_degeneracy_note({"a": 50, "b": 50})
        self.assertIsNone(note)

    def test_reward_models_contains_constants(self) -> None:
        self.assertIn(honesty.REWARD_LOGGED_OUTCOMES, honesty.REWARD_MODELS)
        self.assertIn(honesty.REWARD_SYNTHETIC_HARDCODED, honesty.REWARD_MODELS)
        self.assertIn(honesty.REWARD_NOT_APPLICABLE_RAW, honesty.REWARD_MODELS)


class PerceptionHonestyDegeneracyTests(unittest.TestCase):
    """C: build_perception_honesty must flag a single-action perception fit (the committed v5 is 600/0)."""

    def test_single_action_perception_is_flagged(self) -> None:
        block = honesty.build_perception_honesty(
            dataset_id="x", dataset_hash="y", num_records_kept=600,
            num_per_action={"stop": 0, "continue": 600},
        )
        self.assertIn("degeneracy_note", block)
        self.assertIn("single action dominates", block["degeneracy_note"])
        self.assertIn("continue", block["degeneracy_note"])

    def test_balanced_perception_is_not_flagged(self) -> None:
        block = honesty.build_perception_honesty(
            dataset_id="x", dataset_hash="y", num_records_kept=100,
            num_per_action={"stop": 45, "continue": 55},
        )
        self.assertNotIn("degeneracy_note", block)

    def test_zero_records_note_takes_precedence(self) -> None:
        block = honesty.build_perception_honesty(
            dataset_id="x", dataset_hash="y", num_records_kept=0,
            num_per_action={"stop": 0, "continue": 0},
        )
        self.assertIn("zero in-scope records", block["degeneracy_note"])

    def test_backcompat_without_action_counts(self) -> None:
        # Callers that pass only num_records_kept (kept>0) keep their exact prior behaviour: no note.
        block = honesty.build_perception_honesty(
            dataset_id="x", dataset_hash="y", num_records_kept=600,
        )
        self.assertNotIn("degeneracy_note", block)


if __name__ == "__main__":
    unittest.main()
