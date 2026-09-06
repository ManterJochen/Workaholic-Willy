"""Real-data path for the success-probability trainer: the record feature-logging + the loader.

The runtime stamps the executed grasp's 23-d feature vector into ``record.extra["success_model_features"]``
(via the shadow success telemetry); the trainer consumes those (features, outcome) pairs. Here we lock the
loader + the serializer stamp with no runtime/Isaac needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.robot.execution.autonomous_grasp.record_logging import to_attempt_record
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.grasping.calibration.success_model_calibration import (
    RECORD_FEATURES_KEY,
    build_dataset_from_records,
    train_and_export,
)
from src.robot.grasping.scoring.success_probability import FEATURE_NAMES

_N = len(FEATURE_NAMES)  # 23


def _record(features: list[float] | None, outcome: str = "succeeded") -> dict:
    extra = {} if features is None else {RECORD_FEATURES_KEY: features}
    return {"attempt_id": "a", "mode": "dense", "final_outcome": outcome, "extra": extra}


def test_build_dataset_from_records_maps_features_and_labels() -> None:
    records = [
        _record([0.1] * _N, "succeeded"),
        _record([0.2] * _N, "execution_failed"),
        _record(None, "succeeded"),          # no feature vector -> skipped
        _record([0.3] * (_N - 1), "succeeded"),  # wrong length -> skipped
    ]
    x, y, meta = build_dataset_from_records(records)

    assert x.shape == (2, _N)
    assert list(y) == [1, 0]
    assert meta["kind"] == "real_records"
    assert meta["n_attempts"] == 2
    assert meta["n_skipped_missing_features"] == 2
    assert meta["positive_prevalence"] == pytest.approx(0.5)


def test_build_dataset_from_records_empty_raises() -> None:
    with pytest.raises(ValueError, match="success_model_features"):
        build_dataset_from_records([_record(None), _record(None)])


def test_build_dataset_from_records_imputes_null_features() -> None:
    # A feature that was unavailable at collection time (e.g. the feasibility signals when feasibility
    # scoring isn't wired) is logged as ``null`` -- the extractor emits NaN, JSON serialises it to null.
    # Such rows must be mean-imputed (exactly as predict_proba imputes at inference), NOT dropped whole.
    import numpy as np

    row_ok = [0.4] * _N
    row_null = [0.6] * _N
    row_null[14] = None  # feas_ik_quality missing on this attempt
    row_null[15] = None  # feas_joint_margin missing on this attempt
    x, y, meta = build_dataset_from_records(
        [_record(row_ok, "succeeded"), _record(row_null, "execution_failed")]
    )
    assert x.shape == (2, _N)                       # BOTH kept -- the null row is not skipped
    assert meta["n_skipped_missing_features"] == 0
    assert np.isfinite(x).all()                     # nulls imputed -> finite
    # the null cells impute to the per-column mean over the finite values (only row_ok=0.4 here)
    assert x[1, 14] == pytest.approx(0.4)
    assert x[1, 15] == pytest.approx(0.4)
    assert list(y) == [1, 0]


def test_train_and_export_consumes_a_prebuilt_real_dataset(tmp_path) -> None:
    # A tiny real-shaped dataset (both classes present) trains + exports end-to-end via the records path.
    rng = np.random.default_rng(0)
    x = rng.random((200, _N))
    y = (x[:, 0] > 0.5).astype(np.int64)
    meta = {"kind": "real_records", "n_attempts": 200}
    report = train_and_export(tmp_path, dataset=(x, y, meta))

    assert report["dataset_metadata"]["kind"] == "real_records"
    assert (tmp_path / "model.json").is_file()
    assert (tmp_path / "manifest.json").is_file()


def test_serializer_stamps_executed_grasp_features_from_shadow_telemetry() -> None:
    features = tuple(float(i) / _N for i in range(_N))
    report = SimpleNamespace(
        outcome=AutonomousGraspOutcome.SUCCEEDED,
        mode="dense",
        telemetry={},
        profile=None,
        recovery_actions=(),
        verification=None,
        pick_report=None,
        shadow_success_telemetry=SimpleNamespace(features=features),
    )
    record = to_attempt_record(report, attempt_id="a1")
    assert record.extra[RECORD_FEATURES_KEY] == [float(v) for v in features]


def test_serializer_no_features_when_shadow_absent() -> None:
    report = SimpleNamespace(
        outcome=AutonomousGraspOutcome.SUCCEEDED,
        mode="dense",
        telemetry={},
        profile=None,
        recovery_actions=(),
        verification=None,
        pick_report=None,
        shadow_success_telemetry=None,
    )
    record = to_attempt_record(report, attempt_id="a2")
    assert RECORD_FEATURES_KEY not in record.extra
