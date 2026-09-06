"""Is this record log trainable? Asked before a training run, or before a month of collecting.

A customer's cell logs `GraspAttemptRecord` lines for weeks and someone eventually trains on them.
Three failures make that corpus worthless, all silent, and all cheap to detect up front:

* **no pairs** -- the pairwise ranker forms success/fail pairs WITHIN a group; a log where every group
  is all-success, all-failure or a single row yields zero, and the trainer reports a converged fit
  over nothing;
* **dead features** -- `_project_feature` maps a missing key to `0.0` without a word, which is exactly
  how the 2026-07-17 ranker sat at 0.0 for a day;
* **no per-candidate rows** -- written ONLY when the ranking shadow ran. A cell whose `rl.mode` is not
  `rl_shadow` produces records that look perfectly healthy and contain no per-candidate features at
  all. That is the failure a customer is most likely to hit, and "your features are dead" would send
  them to the wrong place, so it is named separately.

Verified on the two real logs this arc produced:

    rl_records.jsonl (physics outcomes)  TRAINABLE      23 records, 30 pairs across 3 groups
    rl_occupancy_records.jsonl           NOT TRAINABLE  0 pairs, and no positive class at all
"""

from __future__ import annotations

import unittest

from src.robot.grasping.rl.readiness import assess_records, format_readiness


def _record(attempt: str, scene: str, *, ok: bool, **features) -> dict:
    return {
        "timestamp": 1.0, "attempt_id": attempt, "mode": "auto",
        "final_outcome": "succeeded" if ok else "failed",
        "extra": {"scene_id": scene, **features},
    }


def _with_candidates(record: dict, rows: list[dict]) -> dict:
    record["extra"]["rl_candidate_features"] = rows
    return record


class PairFormationTests(unittest.TestCase):
    def test_a_healthy_log_is_trainable(self) -> None:
        records = [
            _record("a", "s1", ok=True, geometric_score=0.9),
            _record("b", "s1", ok=False, geometric_score=0.5),
            _record("c", "s2", ok=True, geometric_score=0.8),
            _record("d", "s2", ok=False, geometric_score=0.4),
        ]
        readiness = assess_records(records)
        self.assertTrue(readiness.trainable)
        self.assertEqual(readiness.pairs, 2)
        self.assertEqual(readiness.groups_with_pairs, 2)

    def test_all_success_yields_no_pairs_and_blocks(self) -> None:
        records = [_record(f"a{i}", "s1", ok=True, geometric_score=0.9) for i in range(4)]
        readiness = assess_records(records)
        self.assertFalse(readiness.trainable)
        self.assertEqual(readiness.pairs, 0)
        self.assertTrue(any("no negative class" in b for b in readiness.blocking))

    def test_all_failure_blocks_too(self) -> None:
        records = [_record(f"a{i}", "s1", ok=False, geometric_score=0.9) for i in range(4)]
        readiness = assess_records(records)
        self.assertFalse(readiness.trainable)
        self.assertTrue(any("no positive class" in b for b in readiness.blocking))

    def test_a_shared_attempt_prefix_still_clusters(self) -> None:
        # `_group_key` deliberately strips a numeric suffix so consecutive attempts of one scene
        # cluster when no scene_id was recorded. Pinned because it is a REAL rescue, not a bug: these
        # six pair up despite carrying no scene identity at all.
        records = [
            {"timestamp": 1.0, "attempt_id": f"pick-{i}", "mode": "auto",
             "final_outcome": "succeeded" if i % 2 else "failed",
             "extra": {"geometric_score": 0.1 * i}}
            for i in range(6)
        ]
        readiness = assess_records(records)
        self.assertEqual(readiness.groups, 1)
        self.assertEqual(readiness.pairs, 9)

    def test_unrelated_attempt_ids_yield_no_pairs_and_name_the_likely_cause(self) -> None:
        # The real trap: ids with no shared prefix and no scene identity -- a uuid or a timestamp per
        # attempt, which is what a cell that never stamped `scene_id` produces. Every record becomes
        # its own group and not one pair forms.
        names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        records = [
            {"timestamp": 1.0, "attempt_id": name, "mode": "auto",
             "final_outcome": "succeeded" if i % 2 else "failed", "extra": {"geometric_score": 0.5}}
            for i, name in enumerate(names)
        ]
        readiness = assess_records(records)
        self.assertFalse(readiness.trainable)
        self.assertEqual(readiness.pairs, 0)
        blocking = " ".join(readiness.blocking)
        self.assertIn("scene_family_id", blocking)
        self.assertIn("own group", " ".join(readiness.warnings))

    def test_an_empty_log_blocks_rather_than_crashing(self) -> None:
        readiness = assess_records([])
        self.assertFalse(readiness.trainable)
        self.assertEqual(readiness.records, 0)


class PerCandidateTests(unittest.TestCase):
    def test_a_missing_shadow_is_named_as_a_shadow_problem(self) -> None:
        # NOT "your features are dead" -- that sends the reader to the wrong place entirely.
        records = [
            _record("a", "s1", ok=True, geometric_score=0.9),
            _record("b", "s1", ok=False, geometric_score=0.5),
        ]
        readiness = assess_records(records)
        self.assertEqual(readiness.candidate_rows, 0)
        warnings = " ".join(readiness.warnings)
        self.assertIn("rl_candidate_features", warnings)
        self.assertIn("rl_shadow", warnings)

    def test_per_candidate_rows_are_measured_separately(self) -> None:
        rows = [
            {"candidate_id": "c0", "rank": 0, "executed": True,
             "features": {"geometric_score": 0.9}, "geometry": {"grip_width_mm": 30.0}},
            {"candidate_id": "c1", "rank": 1, "executed": False,
             "features": {"geometric_score": 0.4}, "geometry": {"grip_width_mm": 55.0}},
            {"tail_count": 3, "tail_mean_geometric_score": 0.2},   # the aggregate row, not a candidate
        ]
        records = [
            _with_candidates(_record("a", "s1", ok=True, geometric_score=0.9), rows),
            _with_candidates(_record("b", "s1", ok=False, geometric_score=0.4), rows),
        ]
        readiness = assess_records(records)
        self.assertEqual(readiness.candidate_rows, 4)  # 2 per record, the tail row excluded
        keys = {c.key for c in readiness.per_candidate}
        self.assertIn("geometric_score", keys)
        self.assertIn("geom.grip_width_mm", keys)
        self.assertIn("geom.grip_width_mm", readiness.live_per_candidate)

    def test_all_constant_per_candidate_features_block(self) -> None:
        # A pairwise model differences within a group. Constant columns leave it nothing to separate by.
        rows = [
            {"candidate_id": f"c{i}", "rank": i, "executed": i == 0,
             "features": {"geometric_score": 0.5}, "geometry": {"grip_width_mm": 40.0}}
            for i in range(3)
        ]
        records = [
            _with_candidates(_record("a", "s1", ok=True, geometric_score=0.5), rows),
            _with_candidates(_record("b", "s1", ok=False, geometric_score=0.5), rows),
        ]
        readiness = assess_records(records)
        self.assertFalse(readiness.trainable)
        self.assertTrue(any("nothing for it to separate" in b for b in readiness.blocking))

    def test_a_single_varying_feature_is_a_warning_not_a_pass(self) -> None:
        rows = [
            {"candidate_id": "c0", "rank": 0, "executed": True,
             "features": {"geometric_score": 0.9}, "geometry": {"grip_width_mm": 40.0}},
            {"candidate_id": "c1", "rank": 1, "executed": False,
             "features": {"geometric_score": 0.3}, "geometry": {"grip_width_mm": 40.0}},
        ]
        records = [
            _with_candidates(_record("a", "s1", ok=True, geometric_score=0.9), rows),
            _with_candidates(_record("b", "s1", ok=False, geometric_score=0.3), rows),
        ]
        readiness = assess_records(records)
        self.assertTrue(readiness.trainable)
        self.assertTrue(any("a threshold" in w for w in readiness.warnings))


class DeadFeatureTests(unittest.TestCase):
    def test_a_log_with_no_declared_features_at_all_blocks(self) -> None:
        records = [
            _record("a", "s1", ok=True),
            _record("b", "s1", ok=False),
        ]
        readiness = assess_records(records)
        self.assertFalse(readiness.trainable)
        self.assertTrue(any("SILENTLY" in b for b in readiness.blocking))

    def test_some_dead_features_are_only_a_warning(self) -> None:
        records = [
            _record("a", "s1", ok=True, geometric_score=0.9),
            _record("b", "s1", ok=False, geometric_score=0.2),
        ]
        readiness = assess_records(records)
        self.assertTrue(readiness.trainable)
        self.assertTrue(any("constant or absent" in w for w in readiness.warnings))


class ReportTests(unittest.TestCase):
    def test_the_verdict_leads(self) -> None:
        records = [
            _record("a", "s1", ok=True, geometric_score=0.9),
            _record("b", "s1", ok=False, geometric_score=0.2),
        ]
        text = format_readiness(assess_records(records))
        self.assertIn("TRAINABLE", text.splitlines()[0])

    def test_a_refusal_says_so_first(self) -> None:
        text = format_readiness(assess_records([_record("a", "s1", ok=True, geometric_score=0.9)]))
        self.assertIn("NOT TRAINABLE", text.splitlines()[0])

    def test_it_always_states_the_within_group_rule(self) -> None:
        # The one thing no dashboard shows, and the reason most of the declared contract cannot help.
        records = [
            _record("a", "s1", ok=True, geometric_score=0.9),
            _record("b", "s1", ok=False, geometric_score=0.2),
        ]
        self.assertIn("constant WITHIN a group", format_readiness(assess_records(records)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
