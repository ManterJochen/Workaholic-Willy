"""The parallel grasp evaluation must produce the same MEASUREMENT as the sequential one.

A faster measurement that is not the same measurement is worthless, and the failure mode here is
quiet: ``ProcessPoolExecutor`` finishes tasks out of order, so a writer that appends on completion
reorders the file without changing any single row. Nothing downstream would raise -- ``summarise``
aggregates, so the totals would even match -- and only a row-by-row comparison catches it.

⚠ "The same" cannot mean byte-identical, and finding out why is the reason this file says so
explicitly. Each row carries ``elapsed_ms``, a wall-clock measurement of the call that produced it,
so **two sequential runs of the same code on the same box already differ**: measured 2026-08-16 over
4 scenes, 2644 rows, 2369 cells apart -- every one of them ``elapsed_ms`` and nothing else. A
byte-comparison here fails for a reason that has nothing to do with parallelism, which is exactly how
a green-looking guard ends up being deleted as flaky.

So the comparison excludes timing, and a SEQUENTIAL CONTROL runs alongside: if the control ever fails,
the evaluator has become non-deterministic on its own and the parallel comparison means nothing. That
control is not hypothetical insurance -- the dense sampler carries a wall-clock BUDGET and falls back
to silhouette geometry when it overruns, plus a cooldown that changes the calls after it. On this
subset it fires identically in both runs and changes no verdict, but it is the mechanism by which a
slower box could measure something different, and the control is what would notice.

Runs the real evaluator over a few scenes of the proof dataset when present, skips otherwise, so CI
without the dataset stays green while the box that has it gets the guard.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "logs" / "p5" / "datasets" / "v1_proof"

#: ⚠ THESE TESTS WRITE INTO A REAL, SHARED DATASET DIRECTORY, and two pytest sessions running at once
#: therefore corrupt each other's row files. MEASURED 2026-08-31: a second session started while one
#: was still running produced four failures that looked exactly like code defects -- a
#: `JSONDecodeError: Extra data: line 1 column 2` from a half-written `.jsonl`, plus `WinError 32` on
#: the log rotation, which is the same shared-file class this repo already recorded once when pool
#: workers held three log files open. The session that won the race was entirely green.
#:
#: There is no lock here on purpose: a lock would serialise and hide it. The guard below simply says
#: what happened, so the next reader does not spend an hour debugging a race as a regression.
_SHARED_DIRECTORY_WARNING = (
    "these tests write into the shared dataset at logs/p5/datasets/v1_proof; do not run two pytest "
    "sessions over this file at once, the row files will interleave and the failures will look like "
    "code defects"
)

#: Wall-clock measurements of the run itself. Excluded from the comparison because they are not part
#: of the measurement's VERDICT -- every other field is.
TIMING_KEYS = frozenset({"elapsed_ms"})


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _without_timing(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in TIMING_KEYS} for row in rows]


@unittest.skipUnless((DATASET / "scenes").is_dir() and (DATASET / "grasps.jsonl").exists(),
                     f"proof dataset not present at {DATASET}")
class MaskCompletionLeverTests(unittest.TestCase):
    """The mask-completion lever must be wired, and its default must change nothing.

    Two failure modes, and only one of them is loud. A default that shifted would silently move every
    number this ladder has ever produced. A lever that reached the evaluator but never the WORKER --
    the parameter is threaded through a ProcessPoolExecutor ``initargs`` tuple, where a positional
    slip passes the wrong value with no error -- would be worse: it would look measured and be inert.

    So this asserts BOTH directions on the real evaluator: the default reproduces itself, and a
    non-default policy demonstrably changes the rows. The second is the one that catches a dead wire.
    """

    LIMIT = 2

    @classmethod
    def _run(cls, policy: str, out_name: str, *, jobs: int = 1) -> "list[dict]":
        from datagen.eval.ladder import CONFIGURATIONS, evaluate_dataset

        path = DATASET / out_name
        rung = tuple(c for c in CONFIGURATIONS if c.name == "neighbours")
        try:
            evaluate_dataset(DATASET, limit=cls.LIMIT, jobs=jobs, out_name=out_name, configs=rung,
                             with_suction=False, progress_every=0, mask_source="gt",
                             mask_completion=policy)
            return _rows(path)
        finally:
            path.unlink(missing_ok=True)

    def test_the_default_changes_nothing(self) -> None:
        """Byte-identical on every substantive field -- verified end to end 2026-08-20 by running the
        pre-change code from a stash: 185 rows, 156 of them differing, every difference ``elapsed_ms``
        and nothing else."""
        self.assertEqual(_without_timing(self._run("none", "_mc_a.jsonl")),
                         _without_timing(self._run("none", "_mc_b.jsonl")))

    def test_a_non_default_policy_actually_reaches_the_calculator(self) -> None:
        """The dead-wire guard. `axis_aligned_box` fires on 78.7 % of masks, so on any real scene it
        MUST move rows. Identical output here means the policy never reached the mask table."""
        default = _without_timing(self._run("none", "_mc_c.jsonl"))
        filled = _without_timing(self._run("axis_aligned_box", "_mc_d.jsonl"))
        self.assertNotEqual(default, filled, "the mask-completion policy never reached the evaluator")

    def test_filling_moves_objects_ACROSS_the_visibility_threshold(self) -> None:
        """The row count changes, and that is the finding -- not a flaw in the comparison.

        This test was written asserting the opposite, that the policy could not change WHICH rows
        exist, and it failed at 54 != 56. The mechanism: a row carries ``visible_px = mask.sum()`` and
        an object under ``MIN_VISIBLE_PX`` (100) is recorded ``not_visible`` and never evaluated.
        Filling a mask to its box inflates that count -- measured on bin_000009/wrist, instance 1 goes
        from 31 px to 192 px -- so objects the evaluation used to skip become grasp targets.

        Two consequences, both worth pinning. On a real cell, a barely-visible object is promoted to a
        target whose silhouette is a rectangle covering mostly its neighbours. And any comparison of
        rates across policies is a DENOMINATOR TRAP: the populations differ, so only a PAIRED
        comparison over the objects present in both is honest.
        """
        default = self._run("none", "_mc_g.jsonl")
        filled = self._run("axis_aligned_box", "_mc_h.jsonl")
        self.assertGreater(len(filled), len(default),
                           "filling must admit objects the visibility floor used to reject")

        def _skipped(rows):
            return {(r["scene_id"], r["view"], r["instance_id"])
                    for r in rows if r.get("outcome") == "not_visible"}

        promoted = _skipped(default) - _skipped(filled)
        self.assertTrue(promoted, "no object crossed MIN_VISIBLE_PX; the mechanism has changed")

    def test_the_worker_path_carries_the_policy_too(self) -> None:
        """``initargs`` is positional; a slip there is silent. Parallel must equal sequential AT a
        non-default policy, which is the only run that would notice."""
        self.assertEqual(_without_timing(self._run("axis_aligned_box", "_mc_e.jsonl", jobs=1)),
                         _without_timing(self._run("axis_aligned_box", "_mc_f.jsonl", jobs=2)))


@unittest.skipUnless((DATASET / "scenes").is_dir() and (DATASET / "grasps.jsonl").exists(),
                     f"proof dataset not present at {DATASET}")
class ParallelEvaluationMatchesSequentialTests(unittest.TestCase):
    #: Two scenes: the ordering bug shows up the moment two tasks can finish out of order. The
    #: suction pass is the expensive part and is not what this test is about.
    LIMIT = 2

    #: THREE evaluations, run ONCE and shared by both tests. Each is a real generator pass over every
    #: object in every view, so re-running them per test method took 266 s and would have roughly
    #: doubled the local suite -- the cost at which a guard stops being run.
    _first: "list[dict]"
    _control: "list[dict]"
    _parallel: "list[dict]"

    @classmethod
    def setUpClass(cls) -> None:
        # ⛔⛔ THE NAMES CARRY THE PROCESS ID, AND WITHOUT IT THIS TEST IS NOT SAFE TO RUN TWICE AT
        # ONCE. `evaluate_dataset` writes `out_name` into the DATASET directory, which is the shipped
        # `v1_proof` tree, and it APPENDS. Two full suites running side by side on this box therefore
        # appended into one another's file and one unlinked it under the other.
        #
        # ⚠ IT DOES NOT SURFACE AS A RACE. MEASURED on 2026-09-05, by a peer session running a suite
        # at the same time as mine: `_eval_par_a.jsonl:147 is not a JSON row (Expecting value). 8
        # chars, starts '_px": 0}'. The file holds 146 good row(s) before it.` A torn line 147 rows
        # deep reads as corrupt data, and nobody looking at that message goes hunting for test
        # isolation.
        #
        # The dataset directory is still the destination because `evaluate_dataset` puts `out_name`
        # beside the scenes it read and a tmpdir would need the whole 270-scene tree copied into it.
        # A unique name is the cheap half of that fix and closes the same hole.
        cls._first = cls._evaluate(jobs=1, out_name=cls._scratch_name("a"))
        cls._control = cls._evaluate(jobs=1, out_name=cls._scratch_name("b"))
        cls._parallel = cls._evaluate(jobs=2, out_name=cls._scratch_name("p"))

    @staticmethod
    def _scratch_name(tag: str) -> str:
        """A per-process output name, so two suites on one box cannot share a file."""
        import os

        return f"_eval_par_{tag}_{os.getpid()}.jsonl"

    @classmethod
    def _evaluate(cls, *, jobs: int, out_name: str) -> "list[dict]":
        from datagen.eval.ladder import evaluate_dataset

        path = DATASET / out_name
        try:
            evaluate_dataset(DATASET, limit=cls.LIMIT, jobs=jobs, out_name=out_name,
                             configs=cls._configs(), with_suction=False, progress_every=0)
            return _rows(path)
        finally:
            path.unlink(missing_ok=True)

    @classmethod
    def _configs(cls):  # noqa: ANN206
        """Two rungs, not the whole ladder. This test is about the PLUMBING -- whether a worker
        reproduces what the sequential loop produced and in what order -- so paying for 13 rungs buys
        nothing and costs ~20 minutes, which is how a correctness guard turns into one people skip.
        ``sfe_fused`` is kept deliberately: it is the rung with per-scene shared state (the fused
        cloud), so it is the one that would break if a worker rebuilt that state differently."""
        from datagen.eval.ladder import CONFIGURATIONS

        wanted = {"sfe", "sfe_fused"}
        return tuple(c for c in CONFIGURATIONS if c.name in wanted)

    def test_the_evaluator_reproduces_itself(self) -> None:
        """The CONTROL. If sequential does not reproduce itself, the parallel comparison below is
        uninterpretable -- and that is not hypothetical: the dense sampler carries a wall-clock
        budget and falls back to silhouette geometry when it overruns, plus a cooldown that changes
        the calls after it. Measured 2026-08-16 it fires identically in both runs and moves no
        verdict, so this passes; a box slow enough to change that would fail HERE, which is the
        point."""
        self.assertEqual(_without_timing(self._first), _without_timing(self._control))

    def test_parallel_reproduces_sequential_row_for_row(self) -> None:
        self.assertEqual(
            _without_timing(self._first), _without_timing(self._parallel),
            "the parallel evaluation produced different rows than the sequential one; if the rows "
            "match but the order does not, results are being consumed on completion instead of in "
            "submission order")

    def test_timing_is_the_only_thing_that_moves(self) -> None:
        """Pins WHY the comparison excludes fields -- so the exclusion list cannot quietly grow."""
        moved = {k for a, b in zip(self._first, self._control)
                 for k in set(a) | set(b) if a.get(k) != b.get(k)}
        self.assertTrue(moved <= TIMING_KEYS,
                        f"fields other than timing differ between two identical runs: "
                        f"{sorted(moved - TIMING_KEYS)}")


class ResolveJobsTests(unittest.TestCase):
    def test_zero_means_auto_and_leaves_headroom(self) -> None:
        import os

        from datagen.__main__ import _resolve_jobs

        self.assertEqual(_resolve_jobs(0), max(1, (os.cpu_count() or 2) - 2))
        self.assertEqual(_resolve_jobs(4), 4)
        self.assertEqual(_resolve_jobs(-3), 1)  # never a pool of zero or a negative width


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TheLimitFlagMustActuallyLimitTest(unittest.TestCase):
    """⛔ IT DID NOT, FOR AS LONG AS IT HAS EXISTED ON THE COMMAND LINE.

    `evaluate_dataset` has taken `limit` since the parallel path landed, and only these tests ever
    passed it. `_cmd_eval_grasps` never did, so `--limit 3` was accepted, warned about nothing, and
    evaluated the whole dataset. MEASURED 2026-08-31: a run launched as a three-scene smoke test came
    back with 454 candidates over 310 objects, which is the full 25-scene figure and identical to the
    previous run's -- the only reason it was noticed at all.

    ⚠ This is the shape the repo fences everywhere else and the wiring guard cannot see, because that
    guard enumerates BOOLEAN config flags and this is a CLI integer. A differential test is what
    catches it: same dataset, two limits, different scene counts.
    """

    def test_the_HANDLER_hands_its_limit_to_the_evaluation(self) -> None:
        """Behavioural, not a source grep: what did `evaluate_dataset` actually receive?

        ⚠ The first draft of this test DID grep the source, for `cli.main` -- a function that does not
        exist, since the dispatch is `_dispatch`. It failed for the right reason by accident, which is
        not a reason to keep an instrument that reads a name instead of an effect.
        """
        from unittest import mock

        from datagen import __main__ as cli
        from datagen.config import DatagenConfig

        with mock.patch("datagen.eval.ladder.evaluate_dataset") as fake:
            fake.return_value = {"rows": 0, "by_config": {}}
            cli._cmd_eval_grasps(  # noqa: SLF001 - the wiring IS the subject
                DatagenConfig(), name="whatever", out_root=str(DATASET.parent.parent),
                summary_only=False, limit=7)
        self.assertTrue(fake.called, "evaluate_dataset was never reached")
        self.assertEqual(fake.call_args.kwargs.get("limit"), 7,
                         f"the handler dropped its limit: {fake.call_args.kwargs}")

    def test_the_DISPATCH_hands_the_parsed_flag_to_the_handler(self) -> None:
        """The other half of the same wire, and the half that was actually broken."""
        import argparse
        from unittest import mock

        from datagen import __main__ as cli
        from datagen.config import DatagenConfig

        args = argparse.Namespace(
            command="eval-grasps", name="whatever", out=None, summary_only=False,
            masks="gt", mask_completion="none", rungs=None, jobs=1, limit=5)
        with mock.patch.object(cli, "_cmd_eval_grasps", return_value=0) as fake:
            cli._dispatch(args, DatagenConfig())  # noqa: SLF001 - the wiring IS the subject
        self.assertTrue(fake.called, "the dispatch never reached the handler")
        self.assertEqual(fake.call_args.kwargs.get("limit"), 5,
                         f"the dispatch dropped --limit: {fake.call_args.kwargs}")

    #: Written into a REAL dataset directory, so they are removed again. `summarise(root)` globs every
    #: `grasp_eval*.jsonl` it finds, and this directory is the repo's own example of what that costs:
    #: fourteen files spanning weeks, reporting 60.33 % where the run that produced it measured
    #: 59.28 %. A test that leaves two more behind makes the trap worse for the next reader.
    ARTEFACTS = ("grasp_eval_limit1.jsonl", "grasp_eval_limitall.jsonl")

    def tearDown(self) -> None:
        for name in self.ARTEFACTS:
            (DATASET / name).unlink(missing_ok=True)

    def test_a_limit_changes_the_scene_count(self) -> None:
        """The differential: a flag that is read but inert produces the same run either way."""
        from datagen.eval.ladder import CONFIGURATIONS, evaluate_dataset

        rung = tuple(c for c in CONFIGURATIONS if c.name == "neighbours")
        scenes = sorted(d for d in (DATASET / "scenes").iterdir() if (d / "scene.json").exists())
        if len(scenes) < 2:
            self.skipTest("needs at least two scenes to tell a limit from no limit")
        # THREE scenes, not `len(scenes)`. The differential only needs two runs that must differ,
        # and the wide arm was costing the suite 18 minutes and 22364 rows to prove the same
        # inequality three scenes prove in ten seconds (measured 2026-09-01: 1096.8 s for 270
        # scenes against 5.2 s for one). A test nobody wants to run is a test that stops running.
        few_limit = min(3, len(scenes))
        one = evaluate_dataset(DATASET, limit=1, jobs=1, out_name=self.ARTEFACTS[0], configs=rung)
        few = evaluate_dataset(DATASET, limit=few_limit, jobs=1, out_name=self.ARTEFACTS[1],
                               configs=rung)
        self.assertLess(one["rows"], few["rows"],
                        f"one scene produced as many rows as {few_limit}")
