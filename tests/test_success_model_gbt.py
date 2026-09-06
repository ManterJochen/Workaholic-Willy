"""Gradient-boosted-trees success model: numpy runtime == sklearn, JSON round-trip, both families load.

The GBT family is the more expressive model intended for a real grasp-outcome corpus. On the
logistic-structured synthetic bootstrap it does NOT beat logistic (measured), so the shipped bootstrap
stays logistic; these tests lock that the GBT trainer + the numpy runtime are correct + deterministic,
no Isaac / real data needed.
"""

from __future__ import annotations

import json

import numpy as np

from src.robot.grasping.calibration.success_model_calibration import (
    DatasetSpec,
    TrainConfig,
    build_synthetic_dataset,
    train_and_export,
    train_gradient_boosted,
)
from src.robot.grasping.scoring.success_probability import (
    GBT_ARTIFACT_VERSION,
    MODEL_FAMILY_GBT,
    MODEL_FAMILY_LOGISTIC,
    load_success_probability_model,
    predict_proba,
)
from src.robot.grasping.scoring.success_probability._model import _model_from_payload

# A LIGHT config — these are correctness tests, not benchmarks; keep training fast for CI.
_CFG = TrainConfig(gbt_n_estimators=40, gbt_max_depth=2)


def _data(n: int = 800):
    return build_synthetic_dataset(DatasetSpec(n_attempts=n))


def test_gbt_trains_and_numpy_matches_sklearn() -> None:
    # train_gradient_boosted asserts numpy==sklearn end-to-end internally; here we lock the shape/family.
    x, y, meta = _data()
    result = train_gradient_boosted(x, y, dataset_metadata=meta, config=_CFG)
    assert result.model.model_family == MODEL_FAMILY_GBT
    assert result.model.artifact_version == GBT_ARTIFACT_VERSION
    assert result.model.gbt is not None
    assert len(result.model.gbt.trees) == _CFG.gbt_n_estimators
    p = predict_proba(result.model, x)
    assert p.shape == (x.shape[0],)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_gbt_json_roundtrip_predicts_identically() -> None:
    x, y, meta = _data()
    model = train_gradient_boosted(x, y, dataset_metadata=meta, config=_CFG).model
    reloaded = _model_from_payload(json.loads(json.dumps(model.to_dict())), model.manifest)
    assert np.array_equal(predict_proba(model, x), predict_proba(reloaded, x))


def test_gbt_is_byte_deterministic_across_runs() -> None:
    x, y, meta = _data()
    a = train_gradient_boosted(x, y, dataset_metadata=meta, config=_CFG).model.to_dict()
    b = train_gradient_boosted(x, y, dataset_metadata=meta, config=_CFG).model.to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_gbt_handles_nan_features_via_mean_imputation() -> None:
    x, y, meta = _data()
    model = train_gradient_boosted(x, y, dataset_metadata=meta, config=_CFG).model
    corrupted = x.copy()
    corrupted[0, 3] = np.nan
    corrupted[1, :5] = np.nan
    p = predict_proba(model, corrupted)  # must not raise; NaNs -> training means
    assert np.all(np.isfinite(p))


def test_train_and_export_gbt_writes_loadable_v2_artifact(tmp_path) -> None:
    train_and_export(
        tmp_path, dataset_spec=DatasetSpec(n_attempts=800), train_config=_CFG, model_family=MODEL_FAMILY_GBT
    )
    model = load_success_probability_model(tmp_path)
    assert model.model_family == MODEL_FAMILY_GBT
    assert model.artifact_version == GBT_ARTIFACT_VERSION


def test_loader_accepts_both_families(tmp_path) -> None:
    x, y, meta = _data()
    train_and_export(tmp_path / "log", dataset=(x, y, meta), train_config=_CFG, model_family=MODEL_FAMILY_LOGISTIC)
    train_and_export(tmp_path / "gbt", dataset=(x, y, meta), train_config=_CFG, model_family=MODEL_FAMILY_GBT)
    assert load_success_probability_model(tmp_path / "log").model_family == MODEL_FAMILY_LOGISTIC
    assert load_success_probability_model(tmp_path / "gbt").model_family == MODEL_FAMILY_GBT
