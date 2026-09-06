"""Phase U10 — focused tests for the runtime latency / SLO gate stack.

These tests gate the Phase U10 subphases:

* U10-A: :class:`LatencyTracker` contract — span/record contracts,
  exception-safe span commit, snapshot omits unentered stages.
* U10-B: :class:`GraspingPerformanceConfig` schema validators
  (timeout > slo rejected; fallback policy enums) and snapshot wiring
  through :class:`EffectiveGraspingConfig`.
* U10-D: :class:`RobotWatchdogEvent.SLO_BREACH` namespace + rising-edge
  emission in :meth:`AutonomousGraspService._update_latency_history_and_emit`.
* U10-E: canonical packs carry decision/ranking/(fusion)
  latencies; EASY omits fusion (Q2=A null contract); regeneration is
  byte-identical.
* U10-F: :mod:`slo_eval` — :class:`LatencyKPI` percentile math, gate
  decisions, null-fusion handling.
* U10-G: ``--slo-gate`` CLI and the baseline-report fold-in.

The full-suite (1298 tests prior to U10-H) keeps the legacy default
byte-identical: ``performance.enabled=False`` and no listener wired
means no SLO emission and no behavioural drift.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from typing import Any
from unittest.mock import MagicMock

from src.config.schema.robot.robot_schema import (
    GraspingPerformanceConfig,
)
from src.robot.events import (
    RobotWatchdogEvent,
    RobotWatchdogEventListener,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspService,
    EffectiveGraspingConfig,
    EffectivePerformanceConfig,
    GraspMode,
)
from src.robot.grasping.telemetry.latency_tracker import (
    STAGE_FIELD_NAMES,
    LatencySpan,
    LatencyStage,
    LatencyTracker,
)
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.replay.baseline_report import (
    build_baseline_report,
)
from src.robot.grasping.replay.canonical_datasets import (
    CANONICAL_PACKS,
    CanonicalPackSpec,
    render_pack_jsonl,
    repo_root_from_module,
)
from src.robot.grasping.replay.slo_eval import (
    DECISION_P95_MAX_MS,
    FUSION_P95_MAX_MS,
    RANKING_P95_MAX_MS,
    STAGE_GATES_MS,
    LatencyKPI,
    SLOGateReport,
    evaluate_slo_pack_path,
)


REPO_ROOT = repo_root_from_module()


def _easy_pack() -> CanonicalPackSpec:
    for p in CANONICAL_PACKS:
        if p.name == "replay_easy_canonical_v1":
            return p
    raise RuntimeError("EASY canonical pack missing from CANONICAL_PACKS")


def _dense_pack() -> CanonicalPackSpec:
    for p in CANONICAL_PACKS:
        if p.name == "replay_dense_canonical_v1":
            return p
    raise RuntimeError("DENSE canonical pack missing from CANONICAL_PACKS")


# ---------------------------------------------------------------------------
# U10-A — LatencyTracker contract
# ---------------------------------------------------------------------------


class LatencyTrackerTests(unittest.TestCase):
    def test_record_then_get(self) -> None:
        t = LatencyTracker()
        t.record(LatencyStage.DECISION, 12.5)
        self.assertEqual(t.get(LatencyStage.DECISION), 12.5)
        self.assertIsNone(t.get(LatencyStage.RANKING))

    def test_span_commits_on_success(self) -> None:
        t = LatencyTracker()
        with t.span(LatencyStage.RANKING):
            pass
        v = t.get(LatencyStage.RANKING)
        self.assertIsNotNone(v)
        self.assertGreaterEqual(float(v), 0.0)

    def test_span_commits_on_exception(self) -> None:
        t = LatencyTracker()
        with self.assertRaises(RuntimeError):
            with t.span(LatencyStage.FUSION):
                raise RuntimeError("boom")
        v = t.get(LatencyStage.FUSION)
        self.assertIsNotNone(v)
        self.assertGreaterEqual(float(v), 0.0)

    def test_snapshot_omits_unentered_stages(self) -> None:
        t = LatencyTracker()
        t.record(LatencyStage.DECISION, 4.0)
        snap = t.snapshot()
        # Mapping should expose the field-name keys for entered stages
        # only — fusion was never entered.
        self.assertIn("decision_latency_ms", snap)
        self.assertNotIn("fusion_latency_ms", snap)
        self.assertNotIn("ranking_latency_ms", snap)
        # Snapshot is a fresh dict; mutating it must not poison state.
        snap["decision_latency_ms"] = 999.0
        self.assertEqual(t.get(LatencyStage.DECISION), 4.0)

    def test_record_rejects_negative_or_non_finite(self) -> None:
        with self.assertRaises((ValueError, TypeError)):
            LatencySpan(stage=LatencyStage.DECISION, elapsed_ms=-1.0)
        with self.assertRaises((ValueError, TypeError)):
            LatencySpan(
                stage=LatencyStage.DECISION, elapsed_ms=float("inf")
            )

    def test_stage_field_names_lock(self) -> None:
        self.assertEqual(
            STAGE_FIELD_NAMES[LatencyStage.DECISION],
            "decision_latency_ms",
        )
        self.assertEqual(
            STAGE_FIELD_NAMES[LatencyStage.RANKING],
            "ranking_latency_ms",
        )
        self.assertEqual(
            STAGE_FIELD_NAMES[LatencyStage.FUSION],
            "fusion_latency_ms",
        )


# ---------------------------------------------------------------------------
# U10-B — GraspingPerformanceConfig schema + EffectiveGraspingConfig wiring
# ---------------------------------------------------------------------------


class GraspingPerformanceConfigTests(unittest.TestCase):
    def test_defaults_pass(self) -> None:
        cfg = GraspingPerformanceConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.decision_latency_slo_ms, 60.0)
        self.assertEqual(cfg.ranking_latency_slo_ms, 80.0)
        self.assertEqual(cfg.fusion_latency_slo_ms, 220.0)

    def test_unknown_model_fallback_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingPerformanceConfig(model_fallback_policy="warp_drive")

    def test_unknown_fusion_fallback_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingPerformanceConfig(fusion_fallback_policy="warp_drive")

    def test_decision_timeout_exceeding_slo_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingPerformanceConfig(
                decision_latency_slo_ms=60.0,
                decision_timeout_ms=120.0,
            )


class EffectiveGraspingConfigPerformanceFieldsTests(unittest.TestCase):
    def test_performance_fields_snapshot_round_trip(self) -> None:
        snap = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=3,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
            performance=EffectivePerformanceConfig(
                enabled=True,
                decision_latency_slo_ms=60.0,
                ranking_latency_slo_ms=80.0,
                fusion_latency_slo_ms=220.0,
                emit_breach_events=True,
                breach_window_size=8,
            ),
        )
        d = snap.to_dict()
        self.assertTrue(d["performance_enabled"])
        self.assertEqual(d["performance_decision_latency_slo_ms"], 60.0)
        self.assertEqual(d["performance_breach_window_size"], 8)


# ---------------------------------------------------------------------------
# U10-D — RobotWatchdogEvent.SLO_BREACH namespace + rising-edge emission
# ---------------------------------------------------------------------------


class SLOBreachNamespaceTests(unittest.TestCase):
    def test_slo_breach_event_value(self) -> None:
        self.assertEqual(
            RobotWatchdogEvent.SLO_BREACH,
            "robot_watchdog_slo_breach",
        )

    def test_slo_breach_in_all_tuple(self) -> None:
        self.assertIn(RobotWatchdogEvent.SLO_BREACH, RobotWatchdogEvent.ALL)


class _ListenerSink:
    """Minimal :class:`RobotWatchdogEventListener` capturing events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, dict(data)))


def _build_perf_effective_config(
    *,
    enabled: bool,
    emit: bool,
    window: int = 4,
) -> EffectiveGraspingConfig:
    return EffectiveGraspingConfig(
        default_mode=GraspMode.AUTO,
        max_attempts=1,
        closed_loop_enabled=False,
        verification_enabled=False,
        dense_recovery_enabled=False,
        dense_recovery_allowed_actions=(),
        performance=EffectivePerformanceConfig(
            enabled=enabled,
            decision_latency_slo_ms=60.0,
            ranking_latency_slo_ms=80.0,
            fusion_latency_slo_ms=220.0,
            emit_breach_events=emit,
            breach_window_size=window,
        ),
    )


def _build_service(
    *,
    cfg: EffectiveGraspingConfig,
    listener: RobotWatchdogEventListener | None,
) -> AutonomousGraspService:
    # ``runtime`` is irrelevant for the breach-emission helpers, so we
    # plug in a MagicMock to satisfy the dataclass field type.
    return AutonomousGraspService(
        runtime=MagicMock(),
        effective_config=cfg,
        watchdog_event_listener=listener,
    )


class SLOBreachEmissionTests(unittest.TestCase):
    def test_no_emit_until_window_full(self) -> None:
        cfg = _build_perf_effective_config(enabled=True, emit=True, window=4)
        listener = _ListenerSink()
        svc = _build_service(cfg=cfg, listener=listener)
        for _ in range(3):
            t = LatencyTracker()
            t.record(LatencyStage.DECISION, 500.0)  # well over 60 ms
            svc._latency.update_and_emit(config=svc.effective_config, event_listener=svc.watchdog_event_listener, 
                tracker=t, attempt_id="a", effective_mode=GraspMode.AUTO
            )
        self.assertEqual(listener.events, [])

    def test_rising_edge_emits_once(self) -> None:
        cfg = _build_perf_effective_config(enabled=True, emit=True, window=4)
        listener = _ListenerSink()
        svc = _build_service(cfg=cfg, listener=listener)
        # Fill window with breaching decision latencies.
        for _ in range(4):
            t = LatencyTracker()
            t.record(LatencyStage.DECISION, 500.0)
            svc._latency.update_and_emit(config=svc.effective_config, event_listener=svc.watchdog_event_listener, 
                tracker=t, attempt_id="a", effective_mode=GraspMode.AUTO
            )
        decision_breaches = [
            e for e in listener.events if e[1]["stage"] == "decision"
        ]
        self.assertEqual(len(decision_breaches), 1)
        evt_type, payload = decision_breaches[0]
        self.assertEqual(evt_type, RobotWatchdogEvent.SLO_BREACH)
        self.assertEqual(payload["stage"], "decision")
        self.assertEqual(payload["slo_ms"], 60.0)
        self.assertGreater(payload["p95_ms"], 60.0)
        self.assertEqual(payload["attempt_id"], "a")
        self.assertEqual(payload["grasp_mode"], "auto")
        # Subsequent breaching ticks must not re-fire (still breached).
        t = LatencyTracker()
        t.record(LatencyStage.DECISION, 500.0)
        svc._latency.update_and_emit(config=svc.effective_config, event_listener=svc.watchdog_event_listener, 
            tracker=t, attempt_id="b", effective_mode=GraspMode.AUTO
        )
        decision_breaches = [
            e for e in listener.events if e[1]["stage"] == "decision"
        ]
        self.assertEqual(len(decision_breaches), 1)

    def test_no_emit_when_disabled(self) -> None:
        cfg = _build_perf_effective_config(enabled=False, emit=True, window=4)
        listener = _ListenerSink()
        svc = _build_service(cfg=cfg, listener=listener)
        for _ in range(4):
            t = LatencyTracker()
            t.record(LatencyStage.DECISION, 500.0)
            svc._latency.update_and_emit(config=svc.effective_config, event_listener=svc.watchdog_event_listener, 
                tracker=t, attempt_id="a", effective_mode=GraspMode.AUTO
            )
        self.assertEqual(listener.events, [])

    def test_no_emit_when_emit_flag_off(self) -> None:
        cfg = _build_perf_effective_config(
            enabled=True, emit=False, window=4
        )
        listener = _ListenerSink()
        svc = _build_service(cfg=cfg, listener=listener)
        for _ in range(4):
            t = LatencyTracker()
            t.record(LatencyStage.DECISION, 500.0)
            svc._latency.update_and_emit(config=svc.effective_config, event_listener=svc.watchdog_event_listener, 
                tracker=t, attempt_id="a", effective_mode=GraspMode.AUTO
            )
        self.assertEqual(listener.events, [])

    def test_missing_fusion_span_does_not_count(self) -> None:
        """Q2=A null contract: absent fusion spans never poison p95."""

        cfg = _build_perf_effective_config(enabled=True, emit=True, window=4)
        listener = _ListenerSink()
        svc = _build_service(cfg=cfg, listener=listener)
        for _ in range(8):
            t = LatencyTracker()
            t.record(LatencyStage.DECISION, 10.0)  # well under SLO
            # No fusion span recorded.
            svc._latency.update_and_emit(config=svc.effective_config, event_listener=svc.watchdog_event_listener, 
                tracker=t, attempt_id="x", effective_mode=GraspMode.AUTO
            )
        # No breaches expected.
        self.assertEqual(listener.events, [])


# ---------------------------------------------------------------------------
# U10-E — canonical packs carry latencies; EASY omits fusion
# ---------------------------------------------------------------------------


class CanonicalPackLatencyEnrichmentTests(unittest.TestCase):
    def _records(self, pack: CanonicalPackSpec):
        return tuple(iter_jsonl(REPO_ROOT / pack.relative_path))

    def test_easy_pack_omits_fusion(self) -> None:
        for rec in self._records(_easy_pack()):
            extra = rec.extra or {}
            self.assertIn("decision_latency_ms", extra)
            self.assertIn("ranking_latency_ms", extra)
            self.assertNotIn("fusion_latency_ms", extra)
            self.assertIn("attempt_wall_time_s", extra)

    def test_dense_pack_includes_fusion(self) -> None:
        for rec in self._records(_dense_pack()):
            extra = rec.extra or {}
            self.assertIn("decision_latency_ms", extra)
            self.assertIn("ranking_latency_ms", extra)
            self.assertIn("fusion_latency_ms", extra)
            self.assertIn("attempt_wall_time_s", extra)

    def test_enrichment_is_deterministic(self) -> None:
        # Re-rendering the EASY pack from spec must produce
        # byte-identical output across two calls — exercises the
        # ``_LATENCY_ENRICH_SEED`` determinism contract.
        easy = _easy_pack()
        self.assertEqual(render_pack_jsonl(easy), render_pack_jsonl(easy))


# ---------------------------------------------------------------------------
# U10-F — slo_eval math + gate logic
# ---------------------------------------------------------------------------


class LatencyKPITests(unittest.TestCase):
    def test_empty_samples_returns_nones(self) -> None:
        kpi = LatencyKPI.from_samples([])
        self.assertEqual(kpi.count, 0)
        self.assertIsNone(kpi.p50_ms)
        self.assertIsNone(kpi.p95_ms)
        self.assertIsNone(kpi.p99_ms)

    def test_single_sample_all_percentiles_match(self) -> None:
        kpi = LatencyKPI.from_samples([42.0])
        self.assertEqual(kpi.count, 1)
        self.assertEqual(kpi.p50_ms, 42.0)
        self.assertEqual(kpi.p95_ms, 42.0)
        self.assertEqual(kpi.p99_ms, 42.0)

    def test_p95_nearest_rank_known_sequence(self) -> None:
        # 100 samples 1..100 → p95 == 95 under round-based nearest-rank.
        kpi = LatencyKPI.from_samples([float(i) for i in range(1, 101)])
        self.assertEqual(kpi.p95_ms, 95.0)

    def test_to_dict_round_trip(self) -> None:
        kpi = LatencyKPI.from_samples([1.0, 2.0, 3.0, 4.0])
        d = kpi.to_dict()
        self.assertEqual(d["count"], 4)
        for key in ("p50_ms", "p95_ms", "p99_ms"):
            self.assertIsInstance(d[key], float)


class SLOGateLogicTests(unittest.TestCase):
    def test_required_stage_empty_stream_fails(self) -> None:
        kpi = LatencyKPI.from_samples([])
        rep = SLOGateReport(
            stage="decision_latency_ms",
            kpi=kpi,
            p95_gate_ms=60.0,
            required=True,
        )
        self.assertFalse(rep.passes_gate)

    def test_optional_stage_empty_stream_passes(self) -> None:
        kpi = LatencyKPI.from_samples([])
        rep = SLOGateReport(
            stage="fusion_latency_ms",
            kpi=kpi,
            p95_gate_ms=220.0,
            required=False,
        )
        self.assertTrue(rep.passes_gate)

    def test_stage_gates_lock(self) -> None:
        # The locked U10 gate triple — stage name, p95 budget, required flag.
        gates = {name: (gate, req) for (name, gate, req) in STAGE_GATES_MS}
        self.assertEqual(
            gates["decision_latency_ms"], (DECISION_P95_MAX_MS, True)
        )
        self.assertEqual(
            gates["ranking_latency_ms"], (RANKING_P95_MAX_MS, True)
        )
        self.assertEqual(
            gates["fusion_latency_ms"], (FUSION_P95_MAX_MS, False)
        )


class SLOPackEvaluationTests(unittest.TestCase):
    def test_easy_pack_passes_with_null_fusion(self) -> None:
        path = REPO_ROOT / _easy_pack().relative_path
        report = evaluate_slo_pack_path(path)
        self.assertTrue(report.passes_gate)
        fusion = report.stage("fusion_latency_ms")
        self.assertEqual(fusion.kpi.count, 0)
        self.assertTrue(fusion.passes_gate)  # optional

    def test_dense_pack_passes_with_fusion(self) -> None:
        path = REPO_ROOT / _dense_pack().relative_path
        report = evaluate_slo_pack_path(path)
        self.assertTrue(report.passes_gate)
        self.assertGreater(report.stage("fusion_latency_ms").kpi.count, 0)

    def test_to_dict_shape(self) -> None:
        path = REPO_ROOT / _dense_pack().relative_path
        d = evaluate_slo_pack_path(path).to_dict()
        self.assertEqual(d["pack_name"], "replay_dense_canonical_v1")
        self.assertIn("stages", d)
        stage_names = {s["stage"] for s in d["stages"]}
        self.assertEqual(
            stage_names,
            {
                "decision_latency_ms",
                "ranking_latency_ms",
                "fusion_latency_ms",
            },
        )
        self.assertIn("passes_gate", d)


# ---------------------------------------------------------------------------
# U10-G — CLI smoke + baseline-report fold-in
# ---------------------------------------------------------------------------


class SLOGateCLITests(unittest.TestCase):
    def test_slo_gate_cli_exits_zero_and_reports_every_pack(self) -> None:
        from src.robot.grasping.replay.__main__ import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--slo-gate"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["mode"], "slo-gate")
        self.assertTrue(payload["passes_gate"])
        names = {p["pack_name"] for p in payload["packs"]}
        self.assertEqual(
            names, {p.name for p in CANONICAL_PACKS}
        )


class BaselineReportSLOFoldInTests(unittest.TestCase):
    def test_per_pack_slo_block_present(self) -> None:
        report = build_baseline_report(REPO_ROOT)
        for pack in report["packs"]:  # type: ignore[index]
            self.assertIn("slo", pack)
            slo = pack["slo"]
            self.assertIn("stages", slo)
            self.assertIn("passes_gate", slo)

    def test_runtime_slo_aggregate_is_finite_and_passes(self) -> None:
        report = build_baseline_report(REPO_ROOT)
        slo = report["runtime_slo"]  # type: ignore[index]
        self.assertEqual(slo["capability_group"], "latency")
        for key in (
            "p95_decision_latency_ms",
            "p95_ranking_latency_ms",
            "p95_fusion_latency_ms",
        ):
            value = slo[key]
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(float(value), 0.0)
        self.assertTrue(slo["passes_gate"])
        self.assertEqual(slo["p95_decision_latency_ms_gate"], 60.0)
        self.assertEqual(slo["p95_ranking_latency_ms_gate"], 80.0)
        self.assertEqual(slo["p95_fusion_latency_ms_gate"], 220.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
