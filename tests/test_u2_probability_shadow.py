"""Phase U2 — shadow-mode runtime adapter for the success-probability model.

U2 scope (locked at handoff):

* The runtime *computes and logs* a predicted success probability for
  every executed candidate set, using the U1-validated artifact.
* The annotator never mutates ``GraspPoint.score`` — selection/ranking
  remains byte-identical to the pre-U2 path.
* Missing or invalid artifacts are *fail-safe*: the loader returns
  ``None`` and emits a single ``WARNING``; the runtime continues
  exactly as if ``success_model.enabled=False``.
* The seam stays tiny: only ``pick_loop.py``,
  ``runtime_pick.py``, and ``autonomous_grasp.py`` import the predictor
  (the U1 isolation guard enforces this).

These tests pin all of the above. They are intentionally additive — the
U1 + U0 invariants are validated by their own suites and must not
regress as part of U2 acceptance.
"""

from __future__ import annotations

import logging
import re
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspService,
    GraspMode,
)
from src.robot.grasping import GraspCalculator
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionFrame,
    PickOutcome,
    PickReport,
)
from src.robot.grasping.scoring.success_probability import (
    FEATURE_NAMES,
    SHADOW_METADATA_KEY_LIFECYCLE_PHASE,
    SHADOW_METADATA_KEY_MODEL_VERSION,
    SHADOW_METADATA_KEY_PROBABILITY,
    ShadowSuccessContext,
    ShadowSuccessTelemetry,
    annotate_grasp_points_with_shadow_probability,
    extract_features_from_grasp_point,
    load_success_probability_model,
    try_load_shadow_success_context,
)
from tests._helpers import (
    _FakeArm,
    _FakePerception,
    _ScriptedCalculator,
    _perception_frame,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "assets/models/success_probability/v1"


def _load_ctx(
    *,
    mode_label: str = "auto",
    version_label: str = "v1",
    lifecycle_phase: str = "shadow",
) -> ShadowSuccessContext:
    """Build a context backed by the committed v1 artifact."""

    model = load_success_probability_model(ARTIFACT_DIR)
    return ShadowSuccessContext(
        model=model,
        version_label=version_label,
        lifecycle_phase=lifecycle_phase,
        mode_label=mode_label,
    )


def _make_grasp_point(
    *,
    score: float = 0.9,
    geometric: float = 0.8,
    stability: float = 0.7,
    reachability: float = 0.9,
    feasibility: float = 0.85,
    confidence: float = 0.95,
    grip_width_mm: float = 40.0,
) -> GraspPoint:
    """Build a GraspPoint with metadata populated like ``_to_grasp_point``."""

    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=grip_width_mm,
        score=score,
        frame=GraspFrame.BASE,
        label="test",
        metadata={
            "geometric_score": geometric,
            "stability_score": stability,
            "reachability_score": reachability,
            "feasibility_score": feasibility,
            "pose_confidence": confidence,
            "score_components": {
                "geometric": geometric,
                "stability": stability,
                "reachability": reachability,
            },
        },
    )


def _success_result(points: tuple[GraspPoint, ...] | None = None) -> GraspResult:
    if points is None:
        points = (_make_grasp_point(score=0.9), _make_grasp_point(score=0.6))
    return GraspResult(
        candidates=points,
        reasons=(),
        top_score=max(p.score for p in points),
    )


# ---------------------------------------------------------------------------
# Section 1 — Carrier dataclasses are frozen, slotted, additive
# ---------------------------------------------------------------------------


class CarrierTypeTests(unittest.TestCase):
    def test_shadow_success_context_is_frozen(self) -> None:
        ctx = _load_ctx()
        with self.assertRaises(FrozenInstanceError):
            ctx.mode_label = "easy"  # type: ignore[misc]

    def test_shadow_success_telemetry_is_frozen(self) -> None:
        t = ShadowSuccessTelemetry(
            predicted_success_probability=0.5,
            success_probability_model_version="v1",
            model_lifecycle_phase="shadow",
        )
        with self.assertRaises(FrozenInstanceError):
            t.predicted_success_probability = 0.7  # type: ignore[misc]

    def test_metadata_keys_are_locked_strings(self) -> None:
        # These names are part of the U2 wire contract. Renaming them
        # silently would break downstream U3/U4 consumers.
        self.assertEqual(
            SHADOW_METADATA_KEY_PROBABILITY,
            "shadow_predicted_success_probability",
        )
        self.assertEqual(
            SHADOW_METADATA_KEY_MODEL_VERSION,
            "shadow_success_probability_model_version",
        )
        self.assertEqual(
            SHADOW_METADATA_KEY_LIFECYCLE_PHASE,
            "shadow_model_lifecycle_phase",
        )


# ---------------------------------------------------------------------------
# Section 2 — extract_features_from_grasp_point: schema parity
# ---------------------------------------------------------------------------


class ExtractFeaturesFromGraspPointTests(unittest.TestCase):
    def test_returns_locked_feature_vector_length(self) -> None:
        v = extract_features_from_grasp_point(_make_grasp_point(), mode="auto")
        self.assertEqual(v.shape, (len(FEATURE_NAMES),))
        self.assertEqual(v.dtype, np.float64)

    def test_mode_one_hot_changes_with_mode_label(self) -> None:
        p = _make_grasp_point()
        v_easy = extract_features_from_grasp_point(p, mode="easy")
        v_auto = extract_features_from_grasp_point(p, mode="auto")
        self.assertFalse(
            np.allclose(v_easy, v_auto),
            "Mode bucket must produce different one-hot tail.",
        )

    def test_missing_pose_confidence_does_not_raise(self) -> None:
        p = _make_grasp_point()
        del p.metadata["pose_confidence"]
        # Must not raise even though confidence is missing — the predictor
        # tolerates NaN/missing via ``_safe_float``.
        v = extract_features_from_grasp_point(p, mode="auto")
        self.assertEqual(v.shape, (len(FEATURE_NAMES),))


# ---------------------------------------------------------------------------
# Section 3 — try_load_shadow_success_context: fail-safe contract
# ---------------------------------------------------------------------------


class TryLoadShadowSuccessContextTests(unittest.TestCase):
    def test_disabled_returns_none_silently(self) -> None:
        logger = logging.getLogger("u2.test.disabled")
        with self.assertLogs(logger, level="WARNING") as cap:
            ctx = try_load_shadow_success_context(
                enabled=False,
                artifact_dir=str(ARTIFACT_DIR),
                mode_label="auto",
                logger=logger,
            )
            # ``assertLogs`` requires at least one record; emit a dummy
            # to confirm no WARNING was emitted by the loader itself.
            logger.warning("sentinel")
        self.assertIsNone(ctx)
        self.assertEqual(len(cap.records), 1)
        self.assertEqual(cap.records[0].getMessage(), "sentinel")

    def test_enabled_with_valid_artifact_loads(self) -> None:
        ctx = try_load_shadow_success_context(
            enabled=True,
            artifact_dir=str(ARTIFACT_DIR),
            mode_label="auto",
        )
        self.assertIsNotNone(ctx)
        assert ctx is not None  # for type checkers
        self.assertEqual(ctx.mode_label, "auto")
        self.assertEqual(ctx.lifecycle_phase, "shadow")
        self.assertEqual(ctx.version_label, "v1")

    def test_missing_artifact_dir_warns_and_returns_none(self) -> None:
        logger = logging.getLogger("u2.test.missing")
        with self.assertLogs(logger, level="WARNING") as cap:
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir="/tmp/__definitely_not_a_real_path__/u2_missing",
                mode_label="auto",
                logger=logger,
            )
        self.assertIsNone(ctx)
        self.assertTrue(
            any("shadow_success_model load failed" in r.getMessage() for r in cap.records),
            f"expected fail-safe warning, got {[r.getMessage() for r in cap.records]}",
        )

    def test_corrupt_json_warns_and_returns_none(self) -> None:
        logger = logging.getLogger("u2.test.corrupt")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "model.json").write_text("{not valid json")
            (d / "manifest.json").write_text("{}")
            with self.assertLogs(logger, level="WARNING") as cap:
                ctx = try_load_shadow_success_context(
                    enabled=True,
                    artifact_dir=str(d),
                    mode_label="auto",
                    logger=logger,
                )
        self.assertIsNone(ctx)
        self.assertTrue(
            any("shadow_success_model load failed" in r.getMessage() for r in cap.records)
        )

    def test_invalid_lifecycle_phase_warns_and_returns_none(self) -> None:
        logger = logging.getLogger("u2.test.lifecycle")
        with self.assertLogs(logger, level="WARNING") as cap:
            ctx = try_load_shadow_success_context(
                enabled=True,
                artifact_dir=str(ARTIFACT_DIR),
                mode_label="auto",
                lifecycle_phase="not_a_phase",
                logger=logger,
            )
        self.assertIsNone(ctx)
        self.assertTrue(
            any("invalid lifecycle_phase" in r.getMessage() for r in cap.records)
        )


# ---------------------------------------------------------------------------
# Section 4 — annotate_grasp_points_with_shadow_probability
# ---------------------------------------------------------------------------


class AnnotateGraspPointsTests(unittest.TestCase):
    def test_no_op_when_ctx_is_none(self) -> None:
        points = [_make_grasp_point()]
        telemetry = annotate_grasp_points_with_shadow_probability(points, ctx=None)
        self.assertIsNone(telemetry)
        self.assertNotIn(SHADOW_METADATA_KEY_PROBABILITY, points[0].metadata)

    def test_no_op_when_points_empty(self) -> None:
        telemetry = annotate_grasp_points_with_shadow_probability([], ctx=_load_ctx())
        self.assertIsNone(telemetry)

    def test_mutates_all_candidates_and_returns_winner_telemetry(self) -> None:
        ctx = _load_ctx(mode_label="auto")
        points = [_make_grasp_point(score=0.9), _make_grasp_point(score=0.6)]
        telemetry = annotate_grasp_points_with_shadow_probability(points, ctx=ctx)
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        for p in points:
            self.assertIn(SHADOW_METADATA_KEY_PROBABILITY, p.metadata)
            self.assertIn(SHADOW_METADATA_KEY_MODEL_VERSION, p.metadata)
            self.assertIn(SHADOW_METADATA_KEY_LIFECYCLE_PHASE, p.metadata)
            prob = p.metadata[SHADOW_METADATA_KEY_PROBABILITY]
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertEqual(p.metadata[SHADOW_METADATA_KEY_MODEL_VERSION], "v1")
            self.assertEqual(p.metadata[SHADOW_METADATA_KEY_LIFECYCLE_PHASE], "shadow")
        # Telemetry == winner (points[0]).
        self.assertEqual(
            telemetry.predicted_success_probability,
            points[0].metadata[SHADOW_METADATA_KEY_PROBABILITY],
        )
        self.assertEqual(telemetry.success_probability_model_version, "v1")
        self.assertEqual(telemetry.model_lifecycle_phase, "shadow")

    def test_does_not_mutate_ranking_score(self) -> None:
        ctx = _load_ctx()
        points = [_make_grasp_point(score=0.9), _make_grasp_point(score=0.6)]
        original_scores = [p.score for p in points]
        annotate_grasp_points_with_shadow_probability(points, ctx=ctx)
        self.assertEqual([p.score for p in points], original_scores)


# ---------------------------------------------------------------------------
# Section 5 — Orchestrator integration: ranking-preserving, opt-in
# ---------------------------------------------------------------------------


class OrchestratorShadowIntegrationTests(unittest.TestCase):
    def _build(self, *, ctx: Optional[ShadowSuccessContext]) -> tuple[
        BinPickingOrchestrator,
        _FakeArm,
        _ScriptedCalculator,
    ]:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_success_result()])
        perception = _FakePerception([_perception_frame()])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            max_attempts=3,
            shadow_success_context=ctx,
        )
        return orch, arm, calc

    def test_default_is_byte_identical_to_pre_u2(self) -> None:
        orch, _arm, _calc = self._build(ctx=None)
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertIsNone(report.shadow_success_telemetry)
        # No GraspPoint should carry the shadow keys.
        executed = report.executed_grasp
        self.assertIsNotNone(executed)
        assert executed is not None
        for c in executed.candidates:
            self.assertNotIn(SHADOW_METADATA_KEY_PROBABILITY, c.metadata)

    def test_enabled_populates_telemetry_and_metadata(self) -> None:
        ctx = _load_ctx(mode_label="auto")
        orch, _arm, _calc = self._build(ctx=ctx)
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        tel = report.shadow_success_telemetry
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertEqual(tel.success_probability_model_version, "v1")
        self.assertEqual(tel.model_lifecycle_phase, "shadow")
        prob = tel.predicted_success_probability
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        # Per-candidate metadata is populated on the executed set.
        executed = report.executed_grasp
        assert executed is not None
        for c in executed.candidates:
            self.assertIn(SHADOW_METADATA_KEY_PROBABILITY, c.metadata)
            self.assertEqual(c.metadata[SHADOW_METADATA_KEY_MODEL_VERSION], "v1")
            self.assertEqual(c.metadata[SHADOW_METADATA_KEY_LIFECYCLE_PHASE], "shadow")
        # Winner telemetry matches winner candidate metadata.
        self.assertEqual(
            tel.predicted_success_probability,
            executed.candidates[0].metadata[SHADOW_METADATA_KEY_PROBABILITY],
        )

    def test_ranking_unchanged_by_shadow_context(self) -> None:
        # Run twice — once with ctx, once without — and confirm executed
        # candidate ordering (by ``score``) is identical. Shadow scoring
        # must never move a candidate.
        baseline_orch, _b_arm, _b_calc = self._build(ctx=None)
        shadow_orch, _s_arm, _s_calc = self._build(ctx=_load_ctx())
        baseline_report = baseline_orch.run()
        shadow_report = shadow_orch.run()
        b_exec = baseline_report.executed_grasp
        s_exec = shadow_report.executed_grasp
        assert b_exec is not None and s_exec is not None
        self.assertEqual(
            [round(c.score, 6) for c in b_exec.candidates],
            [round(c.score, 6) for c in s_exec.candidates],
        )

    def test_no_perception_path_leaves_telemetry_none(self) -> None:
        arm = _FakeArm()
        calc = _ScriptedCalculator([_success_result()])
        empty = PerceptionFrame(
            depth_map=np.full((16, 16), 500.0, dtype=np.float64),
            intrinsics=np.eye(3),
            segmentations=(),
        )
        perception = _FakePerception([empty])
        orch = BinPickingOrchestrator(
            arm=arm,  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
            shadow_success_context=_load_ctx(),
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.NO_PERCEPTION)
        self.assertIsNone(report.shadow_success_telemetry)

    def test_pickreport_default_field_is_none(self) -> None:
        # Pre-U2 callers that construct PickReport without the new field
        # must keep working byte-identically.
        rep = PickReport(outcome=PickOutcome.EXECUTED, attempts=())
        self.assertIsNone(rep.shadow_success_telemetry)


# ---------------------------------------------------------------------------
# Section 6 — Service-level wiring via ``from_robot_config``
# ---------------------------------------------------------------------------


def _cfg(
    *,
    enabled: bool,
    default_mode: str = "auto",
    artifact_dir: Optional[str] = None,
) -> RobotConfig:
    return RobotConfig(
        vendor="dummy",
        gripper={"vendor": "none"},
        grasping={
            "default_mode": default_mode,
            "max_attempts": 3,
            "success_model": {
                "enabled": enabled,
                "artifact_dir": artifact_dir or str(ARTIFACT_DIR),
            },
        },
    )


def _real_calc_and_perception():
    calc = GraspCalculator(
        min_grip_width_mm=1.0,
        max_grip_width_mm=200.0,
        max_candidates=3,
    )
    perception = _FakePerception([_perception_frame()])
    return calc, perception


class ServiceWiringTests(unittest.TestCase):
    def test_disabled_leaves_orchestrator_context_none(self) -> None:
        cfg = _cfg(enabled=False)
        calc, perception = _real_calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=calc, perception=perception
        )
        self.assertIsNone(svc.runtime.orchestrator.shadow_success_context)

    def test_enabled_loads_context_with_resolved_mode_label(self) -> None:
        cfg = _cfg(enabled=True, default_mode="dense_clutter")
        calc, perception = _real_calc_and_perception()
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=calc, perception=perception
        )
        ctx = svc.runtime.orchestrator.shadow_success_context
        self.assertIsNotNone(ctx)
        assert ctx is not None
        # The service plumbs the *raw* resolved mode value so the
        # predictor's bucketing sees operator intent.
        self.assertEqual(ctx.mode_label, GraspMode.DENSE_CLUTTER.value)
        self.assertEqual(ctx.lifecycle_phase, "shadow")
        self.assertEqual(ctx.version_label, "v1")

    def test_enabled_with_missing_artifact_is_fail_safe(self) -> None:
        cfg = _cfg(enabled=True, artifact_dir="/tmp/__u2_missing_artifact__")
        calc, perception = _real_calc_and_perception()
        # Must construct without raising; context stays None.
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=calc, perception=perception
        )
        self.assertIsNone(svc.runtime.orchestrator.shadow_success_context)

    def test_enabled_with_corrupt_artifact_is_fail_safe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "model.json").write_text("{not valid json")
            (Path(td) / "manifest.json").write_text("{}")
            cfg = _cfg(enabled=True, artifact_dir=td)
            calc, perception = _real_calc_and_perception()
            svc = AutonomousGraspService.from_robot_config(
                cfg, calculator=calc, perception=perception
            )
        self.assertIsNone(svc.runtime.orchestrator.shadow_success_context)

    def test_enabled_in_each_canonical_mode(self) -> None:
        # Q3=A: shadow logs regardless of ``apply_modes``; the loader
        # itself does not gate on apply_modes (that gate is reserved for
        # later phases). We at least confirm the context loads for each
        # canonical mode label and bucketing succeeds.
        for mode_str, mode_enum in (
            ("easy", GraspMode.EASY),
            ("auto", GraspMode.AUTO),
            ("dense_clutter", GraspMode.DENSE_CLUTTER),
            ("dense_autonomous", GraspMode.DENSE_AUTONOMOUS),
        ):
            with self.subTest(mode=mode_str):
                cfg = _cfg(enabled=True, default_mode=mode_str)
                calc, perception = _real_calc_and_perception()
                svc = AutonomousGraspService.from_robot_config(
                    cfg, calculator=calc, perception=perception
                )
                ctx = svc.runtime.orchestrator.shadow_success_context
                self.assertIsNotNone(ctx)
                assert ctx is not None
                self.assertEqual(ctx.mode_label, mode_enum.value)


# ---------------------------------------------------------------------------
# Section 7 — Positive wiring contract (greppable imports)
# ---------------------------------------------------------------------------


class PositiveWiringContractTests(unittest.TestCase):
    """Inverse of the U1 RuntimeIsolation guard.

    The U1 test pins the *negative* contract (no other file imports the
    predictor). Here we pin the *positive* contract: the three U2-
    sanctioned files MUST import it, so a future refactor that removes
    the seam silently is caught.
    """

    PATTERN = re.compile(
        r"from\s+src\.robot\.grasping\.scoring\.success_probability\s+import"
    )

    def test_pick_loop_imports_predictor(self) -> None:
        src = (REPO_ROOT / "src/robot/grasping/loop/pick_loop.py").read_text(encoding="utf-8")
        self.assertTrue(
            self.PATTERN.search(src),
            "pick_loop.py must import from success_probability (U2 seam).",
        )

    def test_autonomous_grasp_imports_predictor(self) -> None:
        # The U2 seam now lives in the decomposed ``autonomous_grasp`` package:
        # ``builders.py`` runs the runtime loader wiring and ``report.py``
        # carries the typed telemetry on ``AutonomousGraspReport``. Assert the
        # package as a whole still wires the predictor so a refactor that drops
        # the seam silently is caught.
        pkg = REPO_ROOT / "src/robot/execution/autonomous_grasp"
        sources = [path.read_text(encoding="utf-8") for path in pkg.rglob("*.py")]
        self.assertTrue(
            any(self.PATTERN.search(src) for src in sources),
            "autonomous_grasp package must import from success_probability "
            "(U2 seam).",
        )

    def test_runtime_pick_imports_predictor_for_type_checking(self) -> None:
        # runtime_pick.py forwards the carrier on PickSessionReport; the
        # import is TYPE_CHECKING-guarded but the textual reference is
        # still part of the U2 wire contract.
        src = (REPO_ROOT / "src/robot/execution/runtime_pick.py").read_text(encoding="utf-8")
        self.assertTrue(
            self.PATTERN.search(src),
            "runtime_pick.py must reference success_probability (U2 type-only seam).",
        )


if __name__ == "__main__":
    unittest.main()
