"""Tests for Phase S8 outcome logging (JSONL replay)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.robot.grasping import GraspFrame, GraspPoint
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    append_jsonl,
    frame_metadata_from,
    grasp_metadata_from,
    iter_jsonl,
    json_safe,
    profile_metadata_from,
    recovery_metadata_from,
    refinement_metadata_from,
    target_metadata_from,
    verification_metadata_from,
)


# ---------------------------------------------------------------------------
# Lightweight duck-typed stand-ins so we test the recorder, not the
# production types. The recorder is intentionally duck-typed.
# ---------------------------------------------------------------------------


@dataclass
class _Seg:
    mask: np.ndarray


@dataclass
class _PerceptionFrame:
    depth_map: np.ndarray
    intrinsics: np.ndarray
    segmentations: tuple
    rgb: Optional[np.ndarray] = None
    timestamp: Optional[float] = None


@dataclass
class _Profile:
    mode: str = "auto"
    sampling_mode: str = "antipodal"
    refinement_enabled: bool = True
    verification_enabled: bool = True
    recovery_allowed_actions: tuple = ("next_viewpoint", "next_target")


@dataclass
class _Target:
    mask: np.ndarray
    centroid_xy: tuple
    area_px: int
    label: Optional[str] = "cup"


@dataclass
class _RefinementReport:
    outcome: str = "accepted"
    matched_segmentation_index: int = 0
    match_iou: float = 0.92
    position_delta_mm: float = 1.5
    orientation_delta_deg: float = 2.0
    grip_width_delta_mm: float = 0.5
    failure_reason: Optional[str] = None
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _VerificationReport:
    outcome: str = "passed"
    reason: str = "object_detected"
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _RecoveryPlan:
    action: str = "next_viewpoint"
    reason: str = "no_grasp_found"
    nudge_offset_mm: Optional[tuple] = None
    agitate_amplitude_mm: float = 0.0
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _RecoveryReport:
    plan: _RecoveryPlan
    executed: bool = True
    outcome: str = "completed"
    telemetry: Mapping[str, Any] = field(default_factory=dict)


def _grasp_point() -> GraspPoint:
    return GraspPoint(
        position=np.array([10.0, 20.0, 30.0]),
        approach=np.array([0.0, 0.0, -1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=0.75,
        frame=GraspFrame.BASE,
        label="cup",
        metadata={"source": "test"},
    )


def _frame() -> _PerceptionFrame:
    return _PerceptionFrame(
        depth_map=np.zeros((4, 5), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=(_Seg(np.zeros((4, 5), dtype=bool)),),
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        timestamp=12345.5,
    )


# ---------------------------------------------------------------------------
# json_safe
# ---------------------------------------------------------------------------


class JsonSafeTests(unittest.TestCase):
    def test_primitives_pass_through(self) -> None:
        self.assertEqual(json_safe(None), None)
        self.assertEqual(json_safe(True), True)
        self.assertEqual(json_safe(3), 3)
        self.assertEqual(json_safe(3.5), 3.5)
        self.assertEqual(json_safe("x"), "x")

    def test_nonfinite_floats_become_none(self) -> None:
        self.assertIsNone(json_safe(float("nan")))
        self.assertIsNone(json_safe(float("inf")))
        self.assertIsNone(json_safe(float("-inf")))

    def test_numpy_scalars(self) -> None:
        self.assertEqual(json_safe(np.int32(7)), 7)
        self.assertEqual(json_safe(np.float32(1.5)), 1.5)
        self.assertEqual(json_safe(np.bool_(True)), True)

    def test_numpy_array_recursion(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.int16)
        self.assertEqual(json_safe(arr), [[1, 2], [3, 4]])

    def test_mapping_and_sequence(self) -> None:
        result = json_safe({"a": (1, 2, np.float64(3))})
        self.assertEqual(result, {"a": [1, 2, 3.0]})

    def test_enum_values(self) -> None:
        from enum import StrEnum

        class _E(StrEnum):
            A = "alpha"

        self.assertEqual(json_safe(_E.A), "alpha")

    def test_fallback_uses_str(self) -> None:
        class _Opaque:
            def __str__(self) -> str:
                return "opaque"

        self.assertEqual(json_safe(_Opaque()), "opaque")

    def test_object_with_to_dict_is_recursed(self) -> None:
        class _Box:
            def to_dict(self) -> dict:
                return {"x": np.float32(1.25)}

        self.assertEqual(json_safe(_Box()), {"x": 1.25})

    def test_output_round_trips_through_json(self) -> None:
        payload = json_safe(
            {
                "arr": np.array([1, 2, 3]),
                "score": np.float32(0.5),
                "nested": {"t": (np.int64(7),)},
            }
        )
        # Must be serialisable with stdlib json + reproduce same shape.
        round_tripped = json.loads(json.dumps(payload))
        self.assertEqual(
            round_tripped,
            {"arr": [1, 2, 3], "score": 0.5, "nested": {"t": [7]}},
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class BuilderTests(unittest.TestCase):
    def test_profile_metadata(self) -> None:
        meta = profile_metadata_from(_Profile())
        self.assertEqual(meta["mode"], "auto")
        self.assertEqual(meta["sampling_mode"], "antipodal")
        self.assertTrue(meta["refinement_enabled"])
        self.assertTrue(meta["verification_enabled"])
        self.assertEqual(
            meta["recovery_allowed_actions"],
            ["next_viewpoint", "next_target"],
        )

    def test_frame_metadata_records_shapes(self) -> None:
        meta = frame_metadata_from(_frame())
        self.assertEqual(meta["depth_shape"], [4, 5])
        self.assertEqual(meta["rgb_shape"], [4, 5, 3])
        self.assertTrue(meta["has_rgb"])
        self.assertEqual(meta["segmentation_count"], 1)
        self.assertEqual(meta["timestamp"], 12345.5)
        self.assertEqual(meta["intrinsics"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_frame_metadata_handles_missing_rgb(self) -> None:
        frame = _PerceptionFrame(
            depth_map=np.zeros((2, 2)),
            intrinsics=np.eye(3),
            segmentations=(),
        )
        meta = frame_metadata_from(frame)
        self.assertIsNone(meta["rgb_shape"])
        self.assertFalse(meta["has_rgb"])
        self.assertIsNone(meta["timestamp"])

    def test_target_metadata_none_in_none_out(self) -> None:
        self.assertIsNone(target_metadata_from(None))

    def test_target_metadata(self) -> None:
        t = _Target(
            mask=np.zeros((8, 8), dtype=bool),
            centroid_xy=(3.0, 4.0),
            area_px=12,
            label="cup",
        )
        meta = target_metadata_from(t)
        assert meta is not None
        self.assertEqual(meta["mask_shape"], [8, 8])
        self.assertEqual(meta["centroid_xy"], [3.0, 4.0])
        self.assertEqual(meta["area_px"], 12)
        self.assertEqual(meta["label"], "cup")

    def test_grasp_metadata_uses_to_dict(self) -> None:
        meta = grasp_metadata_from(_grasp_point())
        assert meta is not None
        self.assertEqual(meta["position"], [10.0, 20.0, 30.0])
        self.assertEqual(meta["grip_width_mm"], 40.0)
        self.assertEqual(meta["score"], 0.75)
        self.assertEqual(meta["frame"], "base")
        self.assertEqual(meta["label"], "cup")

    def test_refinement_metadata(self) -> None:
        meta = refinement_metadata_from(_RefinementReport())
        assert meta is not None
        self.assertEqual(meta["outcome"], "accepted")
        self.assertEqual(meta["matched_segmentation_index"], 0)
        self.assertAlmostEqual(meta["match_iou"], 0.92)

    def test_verification_metadata(self) -> None:
        meta = verification_metadata_from(_VerificationReport())
        assert meta is not None
        self.assertEqual(meta["outcome"], "passed")
        self.assertEqual(meta["reason"], "object_detected")

    def test_recovery_metadata(self) -> None:
        report = _RecoveryReport(plan=_RecoveryPlan(action="next_viewpoint"))
        meta = recovery_metadata_from(report)
        assert meta is not None
        self.assertTrue(meta["executed"])
        self.assertEqual(meta["outcome"], "completed")
        self.assertEqual(meta["plan"]["action"], "next_viewpoint")


# ---------------------------------------------------------------------------
# GraspAttemptRecord
# ---------------------------------------------------------------------------


class GraspAttemptRecordTests(unittest.TestCase):
    def test_validation_rejects_empty_attempt_id(self) -> None:
        with self.assertRaises(ValueError):
            GraspAttemptRecord(
                timestamp=0.0,
                attempt_id="",
                mode="auto",
                final_outcome="executed",
            )

    def test_validation_rejects_empty_mode(self) -> None:
        with self.assertRaises(ValueError):
            GraspAttemptRecord(
                timestamp=0.0,
                attempt_id="a",
                mode="",
                final_outcome="executed",
            )

    def test_validation_rejects_empty_outcome(self) -> None:
        with self.assertRaises(ValueError):
            GraspAttemptRecord(
                timestamp=0.0,
                attempt_id="a",
                mode="auto",
                final_outcome="",
            )

    def test_validation_rejects_negative_timestamp(self) -> None:
        with self.assertRaises(ValueError):
            GraspAttemptRecord(
                timestamp=-1.0,
                attempt_id="a",
                mode="auto",
                final_outcome="executed",
            )

    def test_validation_rejects_non_numeric_timestamp(self) -> None:
        with self.assertRaises(TypeError):
            GraspAttemptRecord(
                timestamp="now",  # type: ignore[arg-type]
                attempt_id="a",
                mode="auto",
                final_outcome="executed",
            )

    def test_new_stamps_default_timestamp(self) -> None:
        rec = GraspAttemptRecord.new(
            attempt_id="a", mode="auto", final_outcome="executed"
        )
        self.assertGreater(rec.timestamp, 0.0)

    def test_to_dict_schema_version_and_keys(self) -> None:
        rec = GraspAttemptRecord.new(
            attempt_id="a", mode="auto", final_outcome="executed"
        )
        d = rec.to_dict()
        self.assertEqual(d["schema_version"], 1)
        for key in (
            "timestamp",
            "attempt_id",
            "mode",
            "final_outcome",
            "profile",
            "frame",
            "target",
            "initial_grasp",
            "initial_telemetry",
            "refined_grasp",
            "refinement",
            "selected_grasp",
            "execution",
            "verification",
            "recovery_actions",
            "extra",
        ):
            self.assertIn(key, d)

    def test_to_json_is_single_line(self) -> None:
        rec = GraspAttemptRecord.new(
            attempt_id="a",
            mode="auto",
            final_outcome="executed",
            initial_telemetry={"score": np.float32(0.5)},
            extra={"note": "hello"},
        )
        line = rec.to_json()
        self.assertNotIn("\n", line)
        # Stable schema_version + parsable.
        parsed = json.loads(line)
        self.assertEqual(parsed["initial_telemetry"]["score"], 0.5)

    def test_round_trip_preserves_payload(self) -> None:
        rec = GraspAttemptRecord.new(
            attempt_id="abc",
            mode="auto",
            final_outcome="executed",
            timestamp=42.0,
            profile=profile_metadata_from(_Profile()),
            frame=frame_metadata_from(_frame()),
            target=target_metadata_from(
                _Target(
                    mask=np.zeros((3, 3), dtype=bool),
                    centroid_xy=(1.0, 2.0),
                    area_px=5,
                )
            ),
            initial_grasp=grasp_metadata_from(_grasp_point()),
            initial_telemetry={"candidates": 7},
            refinement=refinement_metadata_from(_RefinementReport()),
            verification=verification_metadata_from(_VerificationReport()),
            recovery_actions=[
                recovery_metadata_from(
                    _RecoveryReport(plan=_RecoveryPlan(action="next_target"))
                ),
            ],
            extra={"operator": "tim"},
        )
        round_tripped = GraspAttemptRecord.from_json(rec.to_json())
        self.assertEqual(round_tripped.to_dict(), rec.to_dict())

    def test_round_trip_preserves_per_candidate_log(self) -> None:
        # P4.1: the per-candidate feature log (list of dicts) + behavior id round-trip through the free-form
        # ``extra`` bag (the offline RL / OPE source) -- exactly what shadow.py splices on a V3-shadow pick.
        candidate_log = [
            {"candidate_id": "a0_c0", "rank": 0, "executed": True,
             "features": {"geometric_score": 0.91, "uncertainty_score": 0.12}},
            {"candidate_id": "a0_c1", "rank": 1, "executed": False,
             "features": {"geometric_score": 0.74, "uncertainty_score": 0.30}},
            {"tail_count": 3, "tail_mean_geometric_score": 0.40},
        ]
        rec = GraspAttemptRecord.new(
            attempt_id="cand-log",
            mode="auto",
            final_outcome="executed",
            extra={
                "rl_candidate_features": candidate_log,
                "rl_behavior_candidate_id": "a0_c0",
            },
        )
        round_tripped = GraspAttemptRecord.from_json(rec.to_json())
        self.assertEqual(round_tripped.to_dict(), rec.to_dict())
        rt_extra = round_tripped.extra
        self.assertEqual(rt_extra["rl_behavior_candidate_id"], "a0_c0")
        feats = rt_extra["rl_candidate_features"]
        self.assertEqual(len(feats), 3)
        self.assertTrue(feats[0]["executed"])
        self.assertEqual(feats[0]["features"]["geometric_score"], 0.91)
        self.assertEqual(feats[-1]["tail_count"], 3)

    def test_from_dict_rejects_missing_keys(self) -> None:
        with self.assertRaises(ValueError):
            GraspAttemptRecord.from_dict({"attempt_id": "a"})

    def test_from_dict_rejects_invalid_optional_dict(self) -> None:
        with self.assertRaises(ValueError):
            GraspAttemptRecord.from_dict(
                {
                    "timestamp": 0.0,
                    "attempt_id": "a",
                    "mode": "auto",
                    "final_outcome": "executed",
                    "profile": [1, 2, 3],
                }
            )

    def test_to_json_handles_numpy_scalars_in_extra(self) -> None:
        rec = GraspAttemptRecord.new(
            attempt_id="a",
            mode="auto",
            final_outcome="executed",
            extra={"arr": np.array([1, 2, 3]), "f": np.float64(2.5)},
        )
        parsed = json.loads(rec.to_json())
        self.assertEqual(parsed["extra"]["arr"], [1, 2, 3])
        self.assertEqual(parsed["extra"]["f"], 2.5)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


class JsonlIOTests(unittest.TestCase):
    def test_append_and_iter_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "attempts.jsonl"
            records = [
                GraspAttemptRecord.new(
                    attempt_id=f"a{i}", mode="auto", final_outcome="executed"
                )
                for i in range(3)
            ]
            for rec in records:
                append_jsonl(rec, path)
            loaded = list(iter_jsonl(path))
            self.assertEqual(len(loaded), 3)
            for original, replayed in zip(records, loaded):
                self.assertEqual(original.to_dict(), replayed.to_dict())

    def test_append_accepts_plain_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            append_jsonl(
                {
                    "schema_version": 1,
                    "timestamp": 0.0,
                    "attempt_id": "a",
                    "mode": "auto",
                    "final_outcome": "executed",
                },
                path,
            )
            loaded = list(iter_jsonl(path))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].attempt_id, "a")

    def test_iter_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            rec = GraspAttemptRecord.new(
                attempt_id="a", mode="auto", final_outcome="executed"
            )
            with path.open("w", encoding="utf-8") as fh:
                fh.write("\n")
                fh.write(rec.to_json() + "\n")
                fh.write("   \n")
            loaded = list(iter_jsonl(path))
            self.assertEqual(len(loaded), 1)

    def test_iter_reports_line_number_on_malformed_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                fh.write('{"not": "valid"}\n')
            with self.assertRaises(ValueError) as ctx:
                list(iter_jsonl(path))
            self.assertIn("line 1", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
