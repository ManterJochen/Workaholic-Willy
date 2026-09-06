"""The seam between our own analytic verdict and physics — `attach_physics_labels`.

Stage 1 of the deep ranker trains on `valid`, which is our reference judging itself. Stage 2 trains on
`held`, which is Isaac. This joins the second onto the first, and everything it drops it counts.

No Isaac and no rendering: the join is a dictionary lookup, and what it must get right is arithmetic
about which rows survive it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datagen.corpus.build import attach_physics_labels


def _corpus(keys: list[str]) -> dict[str, np.ndarray]:
    return {
        "width_mm": np.arange(len(keys), dtype=np.float64),
        "object_key": np.asarray([f"scene#{i}" for i in range(len(keys))]),
        "source_row": np.asarray(keys),
    }


def _physics(path: Path, rows: list[dict]) -> Path:
    path.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


class AttachPhysicsTests(unittest.TestCase):
    def _run(self, corpus_keys: list[str], rows: list[dict]) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory() as name:
            path = _physics(Path(name) / "p.jsonl", rows)
            return attach_physics_labels(_corpus(corpus_keys), path)

    def test_only_rows_physics_resolved_come_back(self) -> None:
        table, report = self._run(
            ["grasp_eval:0", "grasp_eval:1", "grasp_eval:2"],
            [{"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""},
             {"origin": "grasp_eval", "row_index": 2, "source": "valid", "held": False, "note": ""}])
        self.assertEqual(report["rows_in"], 3)
        self.assertEqual(report["rows_out"], 2)
        np.testing.assert_allclose(table["held"], [1.0, 0.0])
        np.testing.assert_allclose(table["width_mm"], [0.0, 2.0], err_msg="row 1 had no verdict")

    def test_a_refusal_is_not_a_failure(self) -> None:
        """The defect that made 26 of 36 trials look like bad grasps. It must not survive the join."""
        table, report = self._run(
            ["grasp_eval:0", "grasp_eval:1"],
            [{"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""},
             {"origin": "grasp_eval", "row_index": 1, "source": "valid", "held": False,
              "note": "refused: the jaw joint blew up (9.59e+12 rad)"}])
        self.assertEqual(report["rows_out"], 1, "the refused row carries no label at all")
        np.testing.assert_allclose(table["held"], [1.0])
        self.assertEqual(report["by_source"]["valid"]["refused"], 1)
        self.assertEqual(report["by_source"]["valid"]["trials"], 1,
                         "a refusal is not a trial either -- it is not in the denominator")

    def test_unmatched_verdicts_are_charged_to_their_own_stratum(self) -> None:
        """`label` trials read a file this corpus does not; that has to be visible, not pooled."""
        _table, report = self._run(
            ["grasp_eval:0"],
            [{"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""},
             {"origin": "grasps", "row_index": 7, "source": "label", "held": True, "note": ""},
             {"origin": "grasps", "row_index": 9, "source": "label", "held": False, "note": ""}])
        self.assertEqual(report["by_source"]["label"]["unmatched"], 2)
        self.assertEqual(report["by_source"]["valid"]["unmatched"], 0)
        self.assertEqual(report["rows_out"], 1)

    def test_a_disagreeing_repeat_is_counted_not_overwritten(self) -> None:
        """PhysX is not deterministic here. Last-one-wins would hide exactly how non-deterministic."""
        table, report = self._run(
            ["grasp_eval:0"],
            [{"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""},
             {"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": False, "note": ""}])
        self.assertEqual(report["by_source"]["valid"]["conflicting"], 1)
        self.assertEqual(report["conflicting_keys"], ["grasp_eval:0"])
        np.testing.assert_allclose(table["held"], [1.0], err_msg="the first verdict stands")

    def test_an_agreeing_repeat_is_not_a_conflict(self) -> None:
        _table, report = self._run(
            ["grasp_eval:0"],
            [{"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""},
             {"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""}])
        self.assertEqual(report["by_source"]["valid"]["conflicting"], 0)

    def test_the_stratum_is_carried_so_training_can_slice_by_it(self) -> None:
        table, _report = self._run(
            ["grasp_eval:0", "grasp_eval:1"],
            [{"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""},
             {"origin": "grasp_eval", "row_index": 1, "source": "rejected:too_wide",
              "held": False, "note": ""}])
        self.assertEqual(list(table["physics_source"]), ["valid", "rejected:too_wide"])

    def test_the_controls_block_is_not_a_trial(self) -> None:
        _table, report = self._run(
            ["grasp_eval:0"],
            [{"controls": {"positive_held": True}},
             {"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""}])
        self.assertEqual(report["physics_verdicts"], 1)

    def test_a_corpus_without_the_key_refuses_rather_than_joining_nothing(self) -> None:
        """Silently returning zero rows would read as "physics found nothing", which is a lie."""
        with tempfile.TemporaryDirectory() as name:
            path = _physics(Path(name) / "p.jsonl", [
                {"origin": "grasp_eval", "row_index": 0, "source": "valid", "held": True, "note": ""}])
            with self.assertRaisesRegex(ValueError, "source_row"):
                attach_physics_labels({"width_mm": np.zeros(1)}, path)


if __name__ == "__main__":
    unittest.main()
