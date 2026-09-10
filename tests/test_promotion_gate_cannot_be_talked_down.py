"""The gate could be talked out of its own exam, three different ways, from the shipped CLI.

⛔⛔ **MEASURED 2026-09-05.** A copy of the committed artifact with every logistic coefficient
multiplied by 0.35 -- a real regression -- is refused at the locked slice and was accepted by all
three of these, each with `verify` answering ``ok: true``, exit 0:

    promote                                   -> fail (brier 0.0901 > 0.0900), verify exit 2
    promote --validation-seed 20260401        -> PASS (brier 0.0844),          verify exit 0
    promote --n-attempts 200                  -> PASS (brier 0.0656),          verify exit 0
    promote --brier-max 0.5 --log-loss-max 2.0 -> PASS,                        verify exit 0

The last one is the sharpest: ``verify_promotion``'s ``thresholds`` parameter is the bar its CALLER
demands, and the shipped ``verify`` verb hands that parameter to the operator. The wall's height was
an argument.

⚠ **WHAT THESE TESTS DO NOT CLAIM.** Every field compared is written by whoever wrote the file. A
hand-written ``promotion.json`` carrying the locked slice, plausible metrics and a chain SHA
recomputed over degraded bytes still verifies -- before and after. What closed is the ACCIDENT: a
flag, a script, a default. Refusing a forgery needs the signature to cover the report's own bytes,
and today it covers the chain SHA only.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.calibration import model_promotion as mp

_REPO = Path(__file__).resolve().parents[1]
_COMMITTED = _REPO / "assets" / "models" / "success_probability" / "v1"


def _committed() -> mp.PromotionReport:
    return mp.load_promotion_report(_COMMITTED)


class TheCommittedArtifactStillPassesTests(unittest.TestCase):
    """⭐ THE FIRST QUESTION, AND THE ONE THAT DECIDES WHETHER THE FIX MAY SHIP AT ALL."""

    def test_the_committed_slice_is_exactly_the_plan_locked_one(self) -> None:
        v = _committed().validation
        self.assertEqual(
            (v.kind, v.seed, v.n_attempts),
            ("synthetic_split", mp.PROMOTION_VALIDATION_SEED, mp.PROMOTION_VALIDATION_ATTEMPTS),
        )

    def test_the_committed_report_is_accepted_with_no_reasons(self) -> None:
        self.assertEqual(mp.runtime_gate_reasons(_committed()), [])
        ok, reasons = mp.verify_promotion(_COMMITTED)
        self.assertTrue(ok, reasons)

    def test_no_field_was_added_to_the_on_disk_payload(self) -> None:
        """⛔ THE SCHEMA VERSION DID NOT MOVE, because nothing new is STORED. Whether a report is
        acceptable is DERIVED from fields it already carries, so `promotion.json` is byte-identical
        and every existing reader keeps working."""
        payload = json.loads((_COMMITTED / "promotion.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], mp.PROMOTION_SCHEMA_VERSION)
        self.assertEqual(
            set(payload),
            {"schema_version", "artifact_basename", "model_json_sha256", "manifest_json_sha256",
             "artifact_sha256", "validation", "metrics", "gate_thresholds", "verdict",
             "promoted_at", "promoted_by", "tooling_version", "signature"},
        )


class TheThreeWaysDownTests(unittest.TestCase):
    def test_a_shrunken_slice_is_refused(self) -> None:
        r = _committed()
        weak = dataclasses.replace(r, validation=dataclasses.replace(r.validation, n_attempts=200))
        self.assertIn("n_attempts=200", " ".join(mp.runtime_gate_reasons(weak)))

    def test_a_moved_seed_is_refused_and_the_training_seed_is_named(self) -> None:
        """⚠ 20260517 is `DEFAULT_DATASET_SEED` -- the trainer's own draw. The five lines of prose at
        `PROMOTION_VALIDATION_SEED` about independence were enforced by a default value."""
        from src.robot.grasping.calibration.success_model_calibration import (
            DEFAULT_DATASET_SEED,
        )

        r = _committed()
        for seed in (DEFAULT_DATASET_SEED, 20260401):
            with self.subTest(seed=seed):
                moved = dataclasses.replace(r, validation=dataclasses.replace(r.validation, seed=seed))
                self.assertIn(f"seed={seed}", " ".join(mp.runtime_gate_reasons(moved)))

    def test_a_bigger_slice_is_refused_too_because_it_is_a_different_draw(self) -> None:
        """⛔ THE MEASUREMENT THAT RULES OUT `n >= locked`, WHICH IS THE OBVIOUS RELAXATION. A larger
        draw looks like a superset and is not one: the first row already differs."""
        from src.robot.grasping.calibration.success_model_calibration import (
            DatasetSpec, build_synthetic_dataset,
        )

        seed = mp.PROMOTION_VALIDATION_SEED
        small_x = build_synthetic_dataset(DatasetSpec(seed=seed, n_attempts=200))[0]
        big_x = build_synthetic_dataset(DatasetSpec(seed=seed, n_attempts=400))[0]
        self.assertFalse(
            bool((big_x[:200] == small_x).all()),
            "a larger n turned out to be a superset after all -- then `>=` would be defensible",
        )

        r = _committed()
        bigger = dataclasses.replace(
            r, validation=dataclasses.replace(r.validation, n_attempts=20000))
        self.assertTrue(mp.runtime_gate_reasons(bigger))

    def test_a_weaker_recorded_bar_is_refused_even_when_the_caller_asks_for_it(self) -> None:
        """⛔⛔ THE WALL'S HEIGHT WAS AN ARGUMENT. `verify --brier-max 0.5 --log-loss-max 2.0` on an
        artifact whose report RECORDS 0.5/2.0 answered ok=true, exit 0. The effective bar is now the
        stricter of the caller's and the plan's, so the parameter means "additionally demand"."""
        weak = dataclasses.replace(
            _committed(), gate_thresholds=mp.PromotionThresholds(0.5, 2.0))
        self.assertTrue(mp.runtime_gate_reasons(weak))
        self.assertTrue(
            mp.runtime_gate_reasons(weak, thresholds=mp.PromotionThresholds(0.5, 2.0)),
            "a caller asking for less than the plan must not be able to lower the wall",
        )

    def test_a_stricter_caller_still_tightens_it(self) -> None:
        """⭐ THE CONTROL ON THE TEST ABOVE. If `min` had been written the other way round, or the
        parameter simply ignored, this would go green while the guard did nothing."""
        self.assertEqual(mp.runtime_gate_reasons(_committed()), [])
        self.assertTrue(
            mp.runtime_gate_reasons(_committed(), thresholds=mp.PromotionThresholds(0.05, 0.2)),
            "a caller demanding MORE than the plan must still be able to refuse",
        )


class AReportMustAgreeWithItselfTests(unittest.TestCase):
    def test_a_pass_beside_metrics_that_violate_its_own_bounds_is_refused(self) -> None:
        """⛔ MEASURED: a hand-edited `promotion.json` with brier 0.5, log_loss 2.0 and
        verdict 'pass' verified clean, exit 0. `verify` held both numbers and never compared them."""
        lying = dataclasses.replace(
            _committed(), metrics={"brier": 0.5, "log_loss": 2.0, "ece": 0.01, "auroc": 0.7})
        reasons = " ".join(mp.runtime_gate_reasons(lying))
        self.assertIn("contradicts its own recorded metrics", reasons)
        self.assertIn("brier=0.5000", reasons)
        self.assertIn("log_loss=2.0000", reasons)

    def test_a_nan_metric_does_not_get_a_free_pass(self) -> None:
        """⛔ `json.loads` accepts a bare NaN literal and every comparison against NaN is False, so
        `value > limit` passes it. The rule is written `not (value <= limit)` for exactly this."""
        self.assertTrue(math.isnan(json.loads('{"brier": NaN}')["brier"]))
        nan = dataclasses.replace(_committed(), metrics={"brier": float("nan"), "log_loss": 0.1})
        self.assertTrue(mp.runtime_gate_reasons(nan))

    def test_a_missing_bounded_metric_is_a_refusal_not_a_pass(self) -> None:
        """⚠ A report that omits `log_loss` cannot be SHOWN to satisfy the log_loss bound, and
        silence is the one answer a gate may never give."""
        thin = dataclasses.replace(_committed(), metrics={"brier": 0.01})
        self.assertIn("log_loss is missing", " ".join(mp.runtime_gate_reasons(thin)))


class OneRuleTwoReadersTests(unittest.TestCase):
    def test_the_metric_to_bound_pairing_is_derived_from_the_threshold_names(self) -> None:
        """⛔ A HAND-KEPT `(("brier","brier_max"), ("log_loss","log_loss_max"))` WAS THE OBVIOUS WAY
        AND IT ROTS ON FIRST CONTACT. `render()` already derived the pairing from the `_max`/`_min`
        suffix; a second, hand-written copy would let a new bound be displayed and not enforced."""

        thresholds = mp.PromotionThresholds()
        pairs = mp.bounded_metrics({"brier": 0.1, "ece": 0.9, "auroc": 0.5}, thresholds)
        self.assertEqual([p[0] for p in pairs], ["brier"], "ece and auroc are recorded and UNGATED")

        # And when a bound exists, the metric is both displayed and enforced -- the same source.
        class _Extra:
            def to_dict(self) -> dict[str, float]:
                return {"brier_max": 0.09, "log_loss_max": 0.31, "ece_max": 0.02}

        extra = _Extra()
        shown = {p[0] for p in mp.bounded_metrics({"brier": 0.01, "ece": 0.99}, extra)}  # type: ignore[arg-type]
        enforced = {r.split("=")[0] for r in mp.gate_reasons({"brier": 0.01, "ece": 0.99}, extra)}  # type: ignore[arg-type]
        self.assertIn("ece", shown)
        self.assertIn("ece", enforced, "a bound that is displayed must also be decided on")

    def test_evaluate_and_verify_apply_the_same_rule(self) -> None:
        """The producer's verdict and the verifier's consistency check must not be two rules."""
        metrics = {"brier": 0.5, "log_loss": 2.0}
        self.assertTrue(mp.gate_reasons(metrics, mp.PromotionThresholds()))
        report = dataclasses.replace(_committed(), metrics=metrics)
        self.assertTrue(mp.runtime_gate_reasons(report))


class TheCliStillSaysSoTests(unittest.TestCase):
    """⚠ DRIVEN THROUGH THE REAL CLI, ON A COPY. `--artifact-dir` DEFAULTS TO THE COMMITTED ARTIFACT
    and `promote` writes `promotion.json` even on a fail, so a forgotten flag here would rewrite the
    SHA-chained file this whole commit promises not to touch."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="gate-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.artifact = self.tmp / "v1"
        shutil.copytree(_COMMITTED, self.artifact)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m",
             "src.robot.grasping.calibration.success_model_calibration", *args],
            cwd=_REPO, capture_output=True, text=True,
        )

    def test_promote_with_a_moved_slice_still_exits_zero_and_says_verify_will_refuse(self) -> None:
        """⭐ THE EXIT CODES DO NOT MOVE, DELIBERATELY. `verdict` means "the metrics cleared the
        thresholds" and that is still true on a 200-sample slice; making one field mean two things is
        how the `policy_id` defect happened. What changes is that the operator is TOLD, at the moment
        the file is written, instead of one command later."""
        done = self._run("promote", "--artifact-dir", str(self.artifact),
                         "--promoted-by", "t", "--promoted-at", "2030-01-01T00:00:00Z",
                         "--n-attempts", "200")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["verdict"], "pass")
        self.assertIn("WILL REFUSE it", done.stderr)
        self.assertIn("n_attempts=200", done.stderr)

        verified = self._run("verify", "--artifact-dir", str(self.artifact))
        self.assertEqual(verified.returncode, 2)
        self.assertFalse(json.loads(verified.stdout)["ok"])

    def test_the_default_promote_still_passes_and_verifies(self) -> None:
        """The control: the fix must not refuse the honest path."""
        done = self._run("promote", "--artifact-dir", str(self.artifact),
                         "--promoted-by", "t", "--promoted-at", "2030-01-01T00:00:00Z")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("WILL REFUSE it", done.stderr)
        self.assertEqual(self._run("verify", "--artifact-dir", str(self.artifact)).returncode, 0)

    def test_the_committed_artifact_on_disk_was_not_touched(self) -> None:
        """⚠ Belt and braces: this file drives `promote`, and `promote` writes."""
        import hashlib

        blob = (_COMMITTED / "promotion.json").read_bytes()
        self.assertEqual(len(blob), 900)
        self.assertEqual(
            hashlib.sha256(blob).hexdigest(),
            "deda43c5fcd5d555db354bfc6e5a2b5e9cbce0acde6cc57a522e21ab36fae814",
        )


class TheDeterminismLockIsNotInertTests(unittest.TestCase):
    """⛔ FIVE OF THE 27 PLATFORM-LOCKED NODE IDs NAMED FILES THAT DO NOT EXIST.

    `pytest_collection_modifyitems` matches on the exact nodeid, so the committed-promotion drift
    test and three verify/canary goldens ran UNGUARDED on every box -- the cross-platform ULP
    exposure `tests/conftest.py` exists to prevent, on the very artifact this commit touches. A stale
    entry is silent in both directions; this makes it loud.

    ⚠ MEASURED 2026-09-10 and this class was right about something worse. The list it scans had
    never gated anything on any box, because the environment variable it waited for is set nowhere
    in the tree: this class was checking that a list of 27 dead entries was spelled correctly. The
    list is now 3 entries, each paired with the measurement that has to agree before it stands down,
    and the tests below moved to `tests._determinism` with it. The lower bound went from 20 to 1 for
    that reason; `tests/test_determinism_lock.py` owns the UPPER bound, which is the direction that
    now needs watching.
    """

    def test_every_locked_nodeid_names_a_file_that_exists(self) -> None:
        from tests._determinism import PLATFORM_FLOAT_LOCKED_NODEIDS

        missing = sorted(
            n for n in PLATFORM_FLOAT_LOCKED_NODEIDS if not (_REPO / n.split("::")[0]).is_file()
        )
        self.assertEqual(missing, [], "a locked node ID names a file that does not exist")

    def test_every_locked_nodeid_names_a_test_that_exists(self) -> None:
        """The other half: a renamed CLASS is just as silent as a renamed file."""
        import ast

        from tests._determinism import PLATFORM_FLOAT_LOCKED_NODEIDS

        missing: list[str] = []
        for node in sorted(PLATFORM_FLOAT_LOCKED_NODEIDS):
            rel, cls, func = node.split("::")
            tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
            klass = next((n for n in ast.walk(tree)
                          if isinstance(n, ast.ClassDef) and n.name == cls), None)
            if klass is None or not any(
                isinstance(f, ast.FunctionDef) and f.name == func for f in klass.body
            ):
                missing.append(node)
        self.assertEqual(missing, [], "a locked node ID names a test that does not exist")

    def test_the_scan_is_not_over_an_empty_set(self) -> None:
        from tests._determinism import PLATFORM_FLOAT_LOCKED_NODEIDS

        self.assertGreaterEqual(len(PLATFORM_FLOAT_LOCKED_NODEIDS), 1)

    def test_every_locked_nodeid_names_a_probe_that_exists(self) -> None:
        """The lock's new failure mode: an entry pointing at a measurement nobody registered.

        A missing probe would raise a KeyError deep inside collection, which reads as a broken
        conftest rather than as a stale lock. Naming it here keeps the diagnosis where it belongs.
        """

        from tests._determinism import DRIFT_PROBES, PLATFORM_FLOAT_LOCKED_NODEIDS

        unknown = sorted(
            f"{node} -> {probe}"
            for node, probe in PLATFORM_FLOAT_LOCKED_NODEIDS.items()
            if probe not in DRIFT_PROBES
        )
        self.assertEqual(unknown, [], "a locked node ID names an unregistered drift probe")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
