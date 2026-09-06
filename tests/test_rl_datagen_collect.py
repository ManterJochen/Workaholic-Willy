"""Joining physics outcomes onto candidate rows: the contract, and the three refusals inside it.

This is the seam where a measured `held` becomes a `GraspAttemptRecord` the RL trainers can read, and
three of its decisions are the kind a later tidy-up would quietly undo:

1. **``extra.scene_id`` is set explicitly.** ``train_ranking._group_key`` looks for ``scene_family_id``
   then ``scene_id``; a provenance key named anything else (``datagen_scene_id``) falls through to the
   attempt-id prefix and NO PAIRS FORM. A pairwise ranker with no pairs trains on nothing and reports
   it as a converged model over zero pairs.
2. **A refused trial is dropped, not recorded as a failure.** The harness refuses to grade a trial
   whose scene drifted, and says so in ``note``. Counting that as `held=False` would put the previous
   trial's explosion into this candidate's label.
3. **A failure carries no taxonomy token.** The tokens name CAUSES (``empty_air_grasp``,
   ``slip_after_grasp``, ...) and physics measured "did not hold", not why. ``derive_outcome_class``
   maps the absence to not-success, so pairs still form; inventing a cause would put a guess into a
   field that is read as a measurement.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.robot.grasping.rl.dataset import (
    OUTCOME_CLASS_SUCCESS,
    derive_outcome_class,
)
from src.robot.grasping.rl.train_ranking import _group_key, _is_success
from datagen.rl.collect import write_records


def _index(scene: str, n: int) -> dict:
    return {
        f"{scene}_c{i}": {
            "scene_id": scene, "family": "bin", "instance_id": 6, "rank": i,
            "executed_by_policy": i == 0,
            "features": {"geometric_score": 0.9 - 0.1 * i},
            "geometry": {"grip_width_mm": 30.0 + i, "approach_tilt_deg": 5.0 * i},
        }
        for i in range(n)
    }


def _physics(rows: list[dict]) -> str:
    head = json.dumps({"controls": {"jaw_is_solid": True, "positive_held": True,
                                    "negative_held": False, "repeat_held": True}})
    return "\n".join([head] + [json.dumps(r) for r in rows]) + "\n"


class RecordJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, index: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
        physics = self.root / "phys.jsonl"
        physics.write_text(_physics(rows), encoding="utf-8")
        out = self.root / "records.jsonl"
        summary = write_records(index, physics, out)
        records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        return summary, records

    def test_held_becomes_a_success_and_not_held_a_failure(self) -> None:
        summary, records = self._write(_index("bin_1", 2), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": -0.4, "note": ""},
            {"source": "rl:bin_1_c1", "held": False, "rise_mm": -812.0, "note": ""},
        ])
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["held"], 1)
        by_id = {r["attempt_id"]: r for r in records}
        self.assertEqual(by_id["bin_1_c0"]["final_outcome"], "succeeded")
        self.assertEqual(by_id["bin_1_c1"]["final_outcome"], "failed")
        self.assertTrue(_is_success(by_id["bin_1_c0"]))
        self.assertFalse(_is_success(by_id["bin_1_c1"]))

    def test_the_group_key_finds_the_scene_so_pairs_can_form(self) -> None:
        # The failure this guards: with no `scene_id`, every record lands in its own group and the
        # pairwise trainer builds ZERO pairs while reporting a converged fit.
        _summary, records = self._write(_index("bin_1", 3), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
            {"source": "rl:bin_1_c1", "held": False, "rise_mm": -900.0, "note": ""},
            {"source": "rl:bin_1_c2", "held": False, "rise_mm": -880.0, "note": ""},
        ])
        keys = {_group_key(r) for r in records}
        self.assertEqual(len(keys), 1, f"all three candidates must share one group; got {keys}")
        wins = [r for r in records if _is_success(r)]
        losses = [r for r in records if not _is_success(r)]
        self.assertEqual((len(wins), len(losses)), (1, 2))  # 2 pairs form

    def test_a_refused_trial_is_dropped_not_scored(self) -> None:
        summary, records = self._write(_index("bin_1", 2), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
            {"source": "rl:bin_1_c1", "held": False, "rise_mm": 0.0,
             "note": "refused: the scene is 212.6 mm out of place"},
        ])
        self.assertEqual(summary["refused"], 1)
        self.assertEqual(summary["records"], 1)
        self.assertEqual([r["attempt_id"] for r in records], ["bin_1_c0"])

    def test_a_failure_carries_no_invented_cause(self) -> None:
        _summary, records = self._write(_index("bin_1", 1), [
            {"source": "rl:bin_1_c0", "held": False, "rise_mm": -900.0, "note": ""},
        ])
        extra = records[0]["extra"]
        self.assertNotIn("failure_taxonomy_class", extra)
        self.assertNotIn("expected_root_cause", extra)
        # ... and it still counts as a failure for pair building, which is the only thing needed.
        self.assertNotEqual(derive_outcome_class(records[0]), OUTCOME_CLASS_SUCCESS)

    def test_a_candidate_with_no_physics_row_is_simply_absent(self) -> None:
        summary, records = self._write(_index("bin_1", 3), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
        ])
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["candidates_indexed"], 3)
        self.assertEqual(len(records), 1)

    def test_rows_from_another_producer_are_ignored(self) -> None:
        # The physics file also holds the harness controls line and, in a shared run, trials from the
        # label/rejected sampler. Only `rl:` rows belong to this dataset.
        summary, _records = self._write(_index("bin_1", 1), [
            {"source": "label", "held": True, "rise_mm": 0.0, "note": ""},
            {"source": "rejected:too_wide", "held": False, "rise_mm": -900.0, "note": ""},
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
        ])
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["physics_rows"], 1)

    def test_the_features_and_geometry_land_where_the_trainer_reads_them(self) -> None:
        _summary, records = self._write(_index("bin_1", 1), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
        ])
        extra = records[0]["extra"]
        self.assertAlmostEqual(extra["geometric_score"], 0.9)      # flat, as `_record_features` wants
        self.assertAlmostEqual(extra["geom_grip_width_mm"], 30.0)  # prefixed, so it cannot collide
        self.assertAlmostEqual(extra["geom_approach_tilt_deg"], 0.0)

    def test_every_record_says_what_produced_its_reward(self) -> None:
        # A corpus whose reward model is not stamped is a corpus that will one day be compared against
        # a real-hardware one without anybody noticing.
        _summary, records = self._write(_index("bin_1", 1), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
        ])
        extra = records[0]["extra"]
        self.assertEqual(extra["reward_model"], "sim_physics_held")
        self.assertIn("not a real cell", extra["reward_interpretation"])
        self.assertEqual(extra["dataset_origin"], "datagen_rl_collect")

    def test_the_harness_controls_are_carried_into_the_summary(self) -> None:
        # The run's own proof that its gripper could grip. A dataset built by a harness that failed its
        # controls is not a weaker dataset, it is not a dataset.
        summary, _records = self._write(_index("bin_1", 1), [
            {"source": "rl:bin_1_c0", "held": True, "rise_mm": 0.0, "note": ""},
        ])
        self.assertTrue(summary["harness_controls"]["jaw_is_solid"])
        self.assertTrue(summary["harness_controls"]["positive_held"])
        self.assertFalse(summary["harness_controls"]["negative_held"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
