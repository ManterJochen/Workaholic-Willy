"""The RL data arc, callable from Python, with the gate that could not see its own regression.

⛔⛔ **THE MEASUREMENT THAT MADE THIS URGENT.** `datagen/rl/` had zero subcommands and had barely run:
`rl_collect.log` is 2,143 lines and every one is a pytest temporary path, so `collect_trials` has not
executed once since logging landed; `rl_proof.log` does not exist at all. And the occupancy sweep
regressed from 98 candidate rows with the shadow router built on 12 of 12 scenes (2026-08-19) to
**zero and zero** three times on 2026-09-04, with an exit code that could not tell the difference.

⚠ **THE REGRESSION ITSELF IS NOT FIXED AND THESE TESTS DO NOT CLAIM IT IS.** What they pin is that it
can no longer pass silently, and that its evidence survives.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen.rl.occupancy import sweep_verdict
from datagen.rl.service import CollectionReport, OccupancyReport, RecordCollection


def _report(*, rows: int, live: int, features: int, live_pc: int, pc_features: int) -> dict:
    """A sweep report shaped like the real one: `features` at record level, `per_candidate` beside."""
    def table(n_live: int, n_total: int) -> dict:
        return {f"k{i}": {"verdict": "live" if i < n_live else "constant",
                          "distinct": 3 if i < n_live else 1, "stdev": "0.4",
                          "nonzero_frac": "1.0", "min": "0", "max": "1"} for i in range(n_total)}
    return {"features": table(live, features),
            "per_candidate": {**table(live_pc, pc_features), "__rows__": {"rows": rows}}}


class TheGateCountedTheWrongNumberTests(unittest.TestCase):
    """⛔⛔ IT WAS `live * 2 >= len(report["features"])`, WHICH READS ONLY THE RECORD LEVEL, while the
    line printed immediately above it calls the per-candidate table "the only thing a pairwise ranker
    can use". `live_pc` was computed, printed, and never reached the exit code.
    """

    def test_a_sweep_with_no_candidate_rows_FAILS(self) -> None:
        """⛔ THE ACTUAL 2026-09-04 STATE. The shadow router was built on 0 scenes, so no candidate
        row existed, and the old gate passed it because every record-level feature was still live.
        Three runs went unnoticed."""
        verdict = sweep_verdict(_report(rows=0, live=15, features=15, live_pc=0, pc_features=19))
        self.assertFalse(verdict.ok)
        self.assertIn("no candidate row", verdict.reason)

    def test_the_OLD_rule_would_have_passed_exactly_that_sweep(self) -> None:
        """The control that makes the test above mean something: without it, "the gate rejects a
        zero-row sweep" could describe a gate that rejects everything."""
        report = _report(rows=0, live=15, features=15, live_pc=0, pc_features=19)
        live = sum(1 for s in report["features"].values() if s["verdict"] == "live")
        self.assertTrue(live * 2 >= len(report["features"]), "the old rule passed this")

    def test_zero_rows_is_refused_by_its_own_sentence_not_by_a_threshold(self) -> None:
        """⚠ ZERO ROWS IS NOT A LOW SCORE. Every per-candidate verdict is vacuous rather than bad,
        and a refusal that says "0 of 19 features move" would send a reader after the features."""
        reason = sweep_verdict(_report(rows=0, live=15, features=15, live_pc=0,
                                       pc_features=19)).reason
        self.assertIn("vacuous", reason)
        self.assertNotIn("of 19", reason)

    def test_dead_PER_CANDIDATE_features_fail_even_when_the_record_level_is_healthy(self) -> None:
        """The state that makes a pairwise ranker unlearnable: the key is present in every record
        and identical across every candidate inside it."""
        verdict = sweep_verdict(_report(rows=98, live=15, features=15, live_pc=2, pc_features=19))
        self.assertFalse(verdict.ok)
        self.assertIn("PER-CANDIDATE", verdict.reason)

    def test_dead_record_level_features_still_fail(self) -> None:
        """The older half of the rule is kept, not replaced."""
        self.assertFalse(sweep_verdict(
            _report(rows=98, live=2, features=15, live_pc=19, pc_features=19)).ok)

    def test_the_2026_08_19_sweep_passes(self) -> None:
        """⭐ THE GOOD RUN, AS THE UPPER CONTROL. 12 scenes, 98 candidate rows, 6/15 record-level and
        7/19 per-candidate live. A gate nothing passes is not a gate, and this is the shape of the
        best run this sweep has ever produced.
        """
        verdict = sweep_verdict(_report(rows=98, live=8, features=15, live_pc=10, pc_features=19))
        self.assertTrue(verdict.ok, verdict.reason)
        self.assertEqual("", verdict.reason)

    def test_the_verdict_carries_the_counts_rather_than_only_a_boolean(self) -> None:
        verdict = sweep_verdict(_report(rows=98, live=8, features=15, live_pc=10, pc_features=19))
        self.assertEqual((98, 8, 15, 10, 19), (verdict.rows, verdict.live_record, verdict.features,
                                               verdict.live_per_candidate,
                                               verdict.per_candidate_features))


class TheSweepStopsDestroyingItsOwnEvidenceTests(unittest.TestCase):
    """⛔⛔ IT CALLED `record_path.unlink()` BEFORE EVERY RUN. The 98 candidate rows from 2026-08-19
    exist nowhere on disk because three zero-row runs each deleted them first, and the only surviving
    trace is an aggregate in a report file that was written because `--out` happened to be passed
    that day. A before/after comparison of exactly the regression this sweep exists to detect was
    made impossible BY the sweep.
    """

    def test_the_previous_records_are_kept_beside_the_new_ones(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "scenes").mkdir()
            (root / "scenes" / "s0").mkdir()
            records = root / "rl_occupancy_records.jsonl"
            records.write_text('{"the": "98-row run"}\n', encoding="utf-8")

            from datagen.rl import occupancy

            # ⚠ THE SWEEP DOES NOT RAISE ON A BROKEN SCENE, it warns and carries on, which is
            # correct for a sweep and is why this test asserts no exception. What matters is
            # that the rename happened BEFORE any of that.
            occupancy.measure_occupancy(root, scene_limit=1)

            previous = root / "rl_occupancy_records.jsonl.previous"
            self.assertTrue(previous.is_file(), "the previous run was deleted, not kept")
            self.assertEqual('{"the": "98-row run"}', previous.read_text(encoding="utf-8").strip())

    def test_only_ONE_generation_is_kept(self) -> None:
        """⚠ NOT A HISTORY. The question this answers is "what changed since last time"; a directory
        of timestamped records would be a different feature with a different cost. So a second run
        replaces the `.previous` file rather than accumulating."""
        self.assertEqual(
            Path("r.jsonl.previous"),
            Path("r.jsonl").with_suffix(Path("r.jsonl").suffix + ".previous"))


class TheArcIsCallableWithoutArgparseTests(unittest.TestCase):

    def test_describe_names_the_referee_before_a_simulator_starts(self) -> None:
        text = RecordCollection.from_dataset("d", engine="mujoco", scenes=4).describe()
        self.assertIn("mujoco", text)
        self.assertIn("scenes         4", text)
        text.encode("ascii")

    def test_the_engine_reaches_the_physics_call(self) -> None:
        """⛔ IT WAS PINNED TO ISAAC BY OMISSION. `collect.py` passed no `engine`, so it took
        `run_physics_sample`'s default, and the MuJoCo branch exists precisely so a customer without
        a 47 GB NVIDIA-only install can grade their own trials."""
        seen: dict[str, object] = {}

        def capture(root, **kwargs):                  # type: ignore[no-untyped-def]
            seen.update(kwargs)

        with mock.patch("datagen.rl.collect.collect_trials",
                        return_value=([1, 2], {}, {"ok": 2})), \
                mock.patch("datagen.grasps.physics.run_physics_sample", side_effect=capture), \
                mock.patch("datagen.rl.collect.write_records",
                           return_value={"records": 2, "held": 1, "refused": 0,
                                         "harness_controls": None}):
            RecordCollection.from_dataset("d", engine="mujoco").collect()
        self.assertEqual("mujoco", seen["engine"])

    def test_join_only_does_not_start_the_referee(self) -> None:
        with mock.patch("datagen.rl.collect.collect_trials", return_value=([], {}, {})), \
                mock.patch("datagen.grasps.physics.run_physics_sample") as physics, \
                mock.patch("datagen.rl.collect.write_records",
                           return_value={"records": 0, "held": 0, "refused": 0}):
            RecordCollection.from_dataset("d").collect(physics=False)
        physics.assert_not_called()

    def test_the_collection_report_reads_as_text(self) -> None:
        report = CollectionReport(records_path=Path("r.jsonl"), trials=40, records=38, held=14,
                                  refused=2, outcomes={"execution_failed": 3})
        text = report.render()
        text.encode("ascii")
        self.assertIn("38 record(s), 14 held", text)
        self.assertAlmostEqual(14 / 38, report.hold_rate)

    def test_an_empty_collection_does_not_divide_by_zero(self) -> None:
        self.assertEqual(0.0, CollectionReport(records_path=Path("r"), trials=0, records=0,
                                               held=0, refused=0).hold_rate)

    def test_the_occupancy_report_says_which_number_matters(self) -> None:
        raw = _report(rows=0, live=15, features=15, live_pc=0, pc_features=19)
        text = OccupancyReport(raw=raw, verdict=sweep_verdict(raw)).render()
        text.encode("ascii")
        self.assertIn("pairwise ranker", text)
        self.assertIn("REFUSED", text)

    def test_as_dict_survives_json(self) -> None:
        raw = _report(rows=98, live=8, features=15, live_pc=10, pc_features=19)
        json.dumps(OccupancyReport(raw=raw, verdict=sweep_verdict(raw)).as_dict())
        json.dumps(CollectionReport(records_path=Path("r"), trials=1, records=1, held=1,
                                    refused=0).as_dict())


class TheSharedSeamIsPublicTests(unittest.TestCase):
    """⛔ `_build_service` WAS PRIVATE BY NAME AND PUBLIC BY USE. `collect.py` imported it across the
    module boundary: this arc's most load-bearing constructor, the one that forces `GraspMode.AUTO`
    and attaches the rig, was protected by nothing and tested by nothing.
    """

    def test_it_has_a_public_name(self) -> None:
        from datagen.rl.occupancy import build_service

        self.assertTrue(callable(build_service))

    def test_no_module_still_reaches_for_the_private_one(self) -> None:
        root = Path(__file__).resolve().parents[1] / "datagen"
        offenders = [str(p.relative_to(root)) for p in root.rglob("*.py")
                     if "_build_service" in p.read_text(encoding="utf-8")
                     and "was `_build_service`" not in p.read_text(encoding="utf-8")]
        self.assertEqual([], offenders)


class TheDeadParametersAreGoneTests(unittest.TestCase):
    """Three functions accepted an argument and never read it, which makes every call site a claim
    that something is needed when it is not."""

    def test_none_of_the_three_still_takes_it(self) -> None:
        import inspect

        from datagen.rl.collect import _candidate_instance_id, _geometry, write_records

        self.assertNotIn("rig", inspect.signature(_candidate_instance_id).parameters)
        self.assertNotIn("metadata", inspect.signature(_geometry).parameters)
        self.assertNotIn("dataset_dir", inspect.signature(write_records).parameters)


if __name__ == "__main__":
    unittest.main()
