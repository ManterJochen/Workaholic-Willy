"""Promotion gate tests.

Covers:

* :class:`PromotionThresholds` defaults + freeze.
* :class:`PromotionVerdict` shape.
* :func:`evaluate_for_promotion` determinism + pass on committed.
* :func:`build_promotion_report` / :func:`write_promotion_report` /
  :func:`load_promotion_report` round-trip + byte stability.
* :func:`verify_promotion` pass / missing / verdict-fail / SHA-tamper /
  thresholds-weakened.
* Loader enforcement: shadow ungated, canary/active refuse on missing /
  tampered / verdict-fail.
* Committed artifact drift check: re-evaluation matches recorded
  metrics bit-for-bit.
* CLI ``promote`` + ``verify`` subcommands.

Runs in the existing
``unittest discover`` flow without any extra deps.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.robot.grasping.calibration import (
    model_promotion as mp,
)
from src.robot.grasping.calibration.success_model_calibration import (
    main as calibration_main,
)
from src.robot.grasping.scoring.success_probability import (
    try_load_shadow_success_context,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = REPO_ROOT / "assets" / "models" / "success_probability" / "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_artifact(dst_root: Path) -> Path:
    dst = dst_root / "v1"
    shutil.copytree(COMMITTED_ARTIFACT, dst)
    return dst


# ---------------------------------------------------------------------------
# Section 1: PromotionThresholds
# ---------------------------------------------------------------------------


class PromotionThresholdsTests(unittest.TestCase):
    def test_defaults_match_plan_honest_values(self) -> None:
        t = mp.PromotionThresholds()
        self.assertEqual(t.brier_max, 0.09)
        self.assertEqual(t.log_loss_max, 0.31)

    def test_frozen(self) -> None:
        t = mp.PromotionThresholds()
        with self.assertRaises(FrozenInstanceError):
            t.brier_max = 0.5  # type: ignore[misc]

    def test_round_trip_dict(self) -> None:
        t = mp.PromotionThresholds(brier_max=0.05, log_loss_max=0.20)
        self.assertEqual(
            mp.PromotionThresholds.from_dict(t.to_dict()),
            t,
        )


# ---------------------------------------------------------------------------
# Section 2: PromotionVerdict
# ---------------------------------------------------------------------------


class PromotionVerdictShapeTests(unittest.TestCase):
    def test_verdict_passed_flag(self) -> None:
        ok = mp.PromotionVerdict(
            verdict="pass",
            metrics={"brier": 0.05, "log_loss": 0.2},
            thresholds=mp.PromotionThresholds(),
            reasons=(),
        )
        bad = mp.PromotionVerdict(
            verdict="fail",
            metrics={"brier": 0.5, "log_loss": 1.0},
            thresholds=mp.PromotionThresholds(),
            reasons=("brier too high",),
        )
        self.assertTrue(ok.passed())
        self.assertFalse(bad.passed())


# ---------------------------------------------------------------------------
# Section 3 + 4: evaluate_for_promotion — pass + determinism
# ---------------------------------------------------------------------------


class EvaluateForPromotionTests(unittest.TestCase):
    def test_committed_artifact_passes_gate(self) -> None:
        verdict, metrics, validation = mp.evaluate_for_promotion(COMMITTED_ARTIFACT)
        self.assertEqual(verdict.verdict, "pass", msg=str(verdict.reasons))
        self.assertLessEqual(metrics["brier"], 0.09)
        self.assertLessEqual(metrics["log_loss"], 0.31)
        self.assertEqual(validation.kind, "synthetic_split")
        self.assertEqual(validation.seed, mp.PROMOTION_VALIDATION_SEED)
        self.assertEqual(validation.n_attempts, mp.PROMOTION_VALIDATION_ATTEMPTS)

    def test_deterministic_across_runs(self) -> None:
        v1, m1, s1 = mp.evaluate_for_promotion(COMMITTED_ARTIFACT)
        v2, m2, s2 = mp.evaluate_for_promotion(COMMITTED_ARTIFACT)
        self.assertEqual(m1, m2)
        self.assertEqual(s1.dataset_sha256, s2.dataset_sha256)
        self.assertEqual(v1.verdict, v2.verdict)

    def test_strict_thresholds_fail_current_artifact(self) -> None:
        # Sanity: the *original* plan-locked thresholds (0.08 / 0.28)
        # are documented to fail this artifact. Pin that here so any
        # future model uplift that crosses the strict bar surfaces as
        # a deliberate test update.
        strict = mp.PromotionThresholds(brier_max=0.08, log_loss_max=0.28)
        verdict, _, _ = mp.evaluate_for_promotion(
            COMMITTED_ARTIFACT, thresholds=strict
        )
        self.assertEqual(verdict.verdict, "fail")
        self.assertTrue(any("brier" in r for r in verdict.reasons))
        self.assertTrue(any("log_loss" in r for r in verdict.reasons))


# ---------------------------------------------------------------------------
# Section 5: round-trip
# ---------------------------------------------------------------------------


class PromotionReportRoundTripTests(unittest.TestCase):
    def test_build_write_load_roundtrip(self) -> None:
        verdict, _m, validation = mp.evaluate_for_promotion(COMMITTED_ARTIFACT)
        report = mp.build_promotion_report(
            COMMITTED_ARTIFACT,
            verdict,
            validation,
            promoted_at="2030-01-01T00:00:00Z",
            promoted_by="unit-test",
        )
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            mp.write_promotion_report(report, tmp)
            loaded = mp.load_promotion_report(tmp)
        self.assertEqual(loaded.verdict, report.verdict)
        self.assertEqual(loaded.model_json_sha256, report.model_json_sha256)
        self.assertEqual(loaded.manifest_json_sha256, report.manifest_json_sha256)
        self.assertEqual(loaded.artifact_sha256, report.artifact_sha256)
        self.assertEqual(loaded.promoted_by, "unit-test")
        self.assertEqual(loaded.promoted_at, "2030-01-01T00:00:00Z")
        self.assertEqual(loaded.tooling_version, mp.PROMOTION_TOOLING_VERSION)
        self.assertEqual(loaded.signature, "none")

    def test_schema_version_locked(self) -> None:
        self.assertEqual(mp.PROMOTION_SCHEMA_VERSION, 1)

    def test_load_rejects_unknown_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            payload = json.loads((COMMITTED_ARTIFACT / "promotion.json").read_text(encoding="utf-8"))
            payload["schema_version"] = 999
            (tmp / "promotion.json").write_text(json.dumps(payload))
            with self.assertRaises(mp.PromotionLoadError):
                mp.load_promotion_report(tmp)


# ---------------------------------------------------------------------------
# Section 6 - 10: verify_promotion
# ---------------------------------------------------------------------------


class VerifyPromotionTests(unittest.TestCase):
    def test_passes_on_committed_artifact(self) -> None:
        ok, reasons = mp.verify_promotion(COMMITTED_ARTIFACT)
        self.assertTrue(ok, msg=str(reasons))
        self.assertEqual(reasons, [])

    def test_fails_when_promotion_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            (tmp / "promotion.json").unlink()
            ok, reasons = mp.verify_promotion(tmp)
        self.assertFalse(ok)
        self.assertTrue(any("promotion_load_failed" in r for r in reasons))

    def test_fails_when_verdict_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            payload = json.loads((tmp / "promotion.json").read_text(encoding="utf-8"))
            payload["verdict"] = "fail"
            (tmp / "promotion.json").write_text(json.dumps(payload, sort_keys=True))
            ok, reasons = mp.verify_promotion(tmp)
        self.assertFalse(ok)
        self.assertTrue(any("verdict" in r for r in reasons))

    def test_fails_when_artifact_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            # Flip one byte of model.json (preserve length to be safe).
            raw = (tmp / "model.json").read_bytes()
            tampered = b" " + raw[1:] if raw[:1] != b" " else b"\t" + raw[1:]
            (tmp / "model.json").write_bytes(tampered)
            ok, reasons = mp.verify_promotion(tmp)
        self.assertFalse(ok)
        self.assertTrue(any("model.json SHA mismatch" in r for r in reasons))
        self.assertTrue(any("chained artifact_sha256" in r for r in reasons))

    def test_fails_when_thresholds_weaker_than_plan(self) -> None:
        # Hand-craft a promotion.json whose recorded gate is far weaker
        # than the runtime's plan-locked bound and confirm verify
        # refuses it even though everything else matches.
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            payload = json.loads((tmp / "promotion.json").read_text(encoding="utf-8"))
            payload["gate_thresholds"] = {"brier_max": 0.5, "log_loss_max": 2.0}
            (tmp / "promotion.json").write_text(json.dumps(payload, sort_keys=True))
            ok, reasons = mp.verify_promotion(tmp)
        self.assertFalse(ok)
        self.assertTrue(any("brier_max" in r and "weaker" in r for r in reasons))
        self.assertTrue(any("log_loss_max" in r and "weaker" in r for r in reasons))


# ---------------------------------------------------------------------------
# Section 11 - 15: loader enforcement
# ---------------------------------------------------------------------------


class _SilentLogger:
    """Captures warnings instead of writing to stderr during tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, fmt: str, *args: object) -> None:  # pragma: no cover - trivial
        self.messages.append(fmt % args if args else fmt)

    def info(self, *_: object, **__: object) -> None:
        pass

    def debug(self, *_: object, **__: object) -> None:
        pass


class LoaderEnforcementTests(unittest.TestCase):
    def test_shadow_loads_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            (tmp / "promotion.json").unlink()
            log = _SilentLogger()
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir=tmp,
                mode_label="vacuum",
                lifecycle_phase="shadow",
                logger=log,  # type: ignore[arg-type]
            )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.lifecycle_phase, "shadow")  # type: ignore[union-attr]
        self.assertEqual(log.messages, [])

    def test_canary_refuses_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            (tmp / "promotion.json").unlink()
            log = _SilentLogger()
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir=tmp,
                mode_label="vacuum",
                lifecycle_phase="canary",
                logger=log,  # type: ignore[arg-type]
            )
        self.assertIsNone(ctx)
        self.assertTrue(
            any("promotion gate failed" in m for m in log.messages),
            msg=log.messages,
        )

    def test_canary_loads_with_valid_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            log = _SilentLogger()
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir=tmp,
                mode_label="vacuum",
                lifecycle_phase="canary",
                logger=log,  # type: ignore[arg-type]
            )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.lifecycle_phase, "canary")  # type: ignore[union-attr]
        self.assertEqual(log.messages, [])

    def test_active_refuses_when_verdict_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            payload = json.loads((tmp / "promotion.json").read_text(encoding="utf-8"))
            payload["verdict"] = "fail"
            (tmp / "promotion.json").write_text(json.dumps(payload, sort_keys=True))
            log = _SilentLogger()
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir=tmp,
                mode_label="vacuum",
                lifecycle_phase="active",
                logger=log,  # type: ignore[arg-type]
            )
        self.assertIsNone(ctx)
        self.assertTrue(any("promotion gate failed" in m for m in log.messages))

    def test_canary_refuses_when_artifact_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            raw = (tmp / "manifest.json").read_bytes()
            (tmp / "manifest.json").write_bytes(b" " + raw[1:])
            log = _SilentLogger()
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir=tmp,
                mode_label="vacuum",
                lifecycle_phase="canary",
                logger=log,  # type: ignore[arg-type]
            )
        self.assertIsNone(ctx)
        self.assertTrue(any("SHA mismatch" in m for m in log.messages))

    def test_disabled_returns_none_silently(self) -> None:
        log = _SilentLogger()
        ctx = try_load_shadow_success_context(
            enabled=False,
            artifact_dir=COMMITTED_ARTIFACT,
            mode_label="vacuum",
            lifecycle_phase="canary",
            logger=log,  # type: ignore[arg-type]
        )
        self.assertIsNone(ctx)
        self.assertEqual(log.messages, [])


# ---------------------------------------------------------------------------
# Section 16: committed promotion.json drift check
# ---------------------------------------------------------------------------


class CommittedPromotionDriftTests(unittest.TestCase):
    """If the artifact changes underneath us the committed promotion
    file becomes a lie. This test reads the committed report's metrics
    and re-evaluates from scratch — they must match bit-for-bit."""

    def test_metrics_match_recorded(self) -> None:
        report = mp.load_promotion_report(COMMITTED_ARTIFACT)
        _verdict, metrics, validation = mp.evaluate_for_promotion(
            COMMITTED_ARTIFACT,
            validation_seed=report.validation.seed,
            n_attempts=report.validation.n_attempts,
        )
        for key, recorded in report.metrics.items():
            self.assertAlmostEqual(
                metrics[key],
                recorded,
                places=12,
                msg=f"{key}: recorded={recorded}, recomputed={metrics[key]}",
            )
        self.assertEqual(validation.dataset_sha256, report.validation.dataset_sha256)


# ---------------------------------------------------------------------------
# Section 17: lifecycle_phase_requires_promotion
# ---------------------------------------------------------------------------


class LifecyclePromotionPredicateTests(unittest.TestCase):
    def test_shadow_excluded(self) -> None:
        self.assertFalse(mp.lifecycle_phase_requires_promotion("shadow"))

    def test_canary_active_included(self) -> None:
        self.assertTrue(mp.lifecycle_phase_requires_promotion("canary"))
        self.assertTrue(mp.lifecycle_phase_requires_promotion("active"))

    def test_unknown_excluded(self) -> None:
        # Unknown phases are caught upstream by ``_VALID_LIFECYCLE_PHASES``;
        # this predicate just answers the "do I gate?" question. Defaulting
        # unknowns to False keeps the loader's existing
        # ``invalid lifecycle_phase`` warning path intact.
        self.assertFalse(mp.lifecycle_phase_requires_promotion("bogus"))


# ---------------------------------------------------------------------------
# Section 18: CLI smoke
# ---------------------------------------------------------------------------


class PromotionCLITests(unittest.TestCase):
    def test_promote_subcommand_writes_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            (tmp / "promotion.json").unlink()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = calibration_main(
                    [
                        "promote",
                        "--artifact-dir",
                        str(tmp),
                        "--promoted-by",
                        "cli-test",
                        "--promoted-at",
                        "2030-01-01T00:00:00Z",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue((tmp / "promotion.json").is_file())
            payload = json.loads((tmp / "promotion.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], "pass")
            self.assertEqual(payload["promoted_by"], "cli-test")
            self.assertEqual(payload["promoted_at"], "2030-01-01T00:00:00Z")
            # stdout payload matches what's on disk.
            stdout_payload = json.loads(buf.getvalue())
            self.assertEqual(stdout_payload["verdict"], "pass")

    def test_verify_subcommand_returns_zero_on_committed(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = calibration_main(
                ["verify", "--artifact-dir", str(COMMITTED_ARTIFACT)]
            )
        self.assertEqual(rc, 0)
        result = json.loads(buf.getvalue())
        self.assertTrue(result["ok"])

    def test_verify_subcommand_returns_nonzero_on_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            raw = (tmp / "model.json").read_bytes()
            (tmp / "model.json").write_bytes(b" " + raw[1:])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = calibration_main(["verify", "--artifact-dir", str(tmp)])
            self.assertNotEqual(rc, 0)
            result = json.loads(buf.getvalue())
            self.assertFalse(result["ok"])

    def test_promote_subcommand_nonzero_when_strict_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = _copy_artifact(Path(td))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = calibration_main(
                    [
                        "promote",
                        "--artifact-dir",
                        str(tmp),
                        "--brier-max",
                        "0.08",
                        "--log-loss-max",
                        "0.28",
                        "--promoted-by",
                        "strict-test",
                        "--promoted-at",
                        "2030-01-01T00:00:00Z",
                    ]
                )
            self.assertEqual(rc, 1)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["verdict"], "fail")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
