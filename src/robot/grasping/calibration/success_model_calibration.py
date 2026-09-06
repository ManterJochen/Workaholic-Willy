"""Offline trainer + exporter for the success-probability model.

This module is the only place scikit-learn is imported in the training stack.
After training, the model is exported as a pure-JSON artifact under
``assets/models/success_probability/v<N>/``, and runtime code, in
:mod:`src.robot.grasping.scoring.success_probability`, consumes that artifact
with numpy alone.

The CLI entry points are invoked through ``python -m`` directly, which bypasses
the ``__main__.py`` shim of the calibration package, since that shim belongs to
uncertainty calibration:

* ``python -m src.robot.grasping.calibration.success_model_calibration train``
* ``python -m src.robot.grasping.calibration.success_model_calibration eval``

Both commands are fully deterministic given the locked seeds: running one twice
on the same machine produces byte-identical ``model.json`` and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.robot.grasping.scoring.success_probability import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    GBT_ARTIFACT_VERSION,
    MODE_BUCKETS,
    MODEL_ARTIFACT_VERSION,
    MODEL_FAMILY_GBT,
    MODEL_FAMILY_LOGISTIC,
    SuccessProbabilityModel,
    _model_from_payload,
    predict_proba,
)
from src.robot.grasping.constants import (
    SUCCESS_MODEL_CALIBRATION_LOG_FILE,
    create_grasping_logger,
)

#: A training run is reproducible only if what went in (corpus size, skipped rows,
#: seeds) and what came out (CV metrics, artifact SHAs) are both recoverable.
logger = create_grasping_logger(
    "SuccessModelCalibration", SUCCESS_MODEL_CALIBRATION_LOG_FILE
)


__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_DATASET_SEED",
    "DEFAULT_N_ATTEMPTS",
    "RECORD_FEATURES_KEY",
    "DatasetSpec",
    "TrainConfig",
    "TrainResult",
    "build_dataset_from_records",
    "build_synthetic_dataset",
    "build_train_config",
    "cross_validate_metrics",
    "evaluate_metrics",
    "load_records_jsonl",
    "main",
    "train_and_export",
    "train_gradient_boosted",
    "train_logistic_isotonic",
]


# Locked defaults (do not change without bumping artifact dir version)

DEFAULT_DATASET_SEED: int = 20260517
DEFAULT_TRAIN_SEED: int = 20260518
DEFAULT_N_ATTEMPTS: int = 5000

#: Repo-relative default storage.
DEFAULT_ARTIFACT_DIR: str = "assets/models/success_probability/v1"

#: Per-mode share of the synthetic dataset (sums to 1.0). Tuned so the
#: positive-class prevalence of each mode resembles a real replay, where an
#: easy pick succeeds far more often than an autonomous one.
_MODE_MIX: tuple[tuple[str, float], ...] = (
    ("easy", 0.30),
    ("dense", 0.45),
    ("auto", 0.25),
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Frozen description of the synthetic bootstrap dataset (synthetic by design; the manifest records this so no consumer mistakes it for production evidence)."""

    seed: int = DEFAULT_DATASET_SEED
    n_attempts: int = DEFAULT_N_ATTEMPTS
    feature_schema_version: int = FEATURE_SCHEMA_VERSION


def _mode_prevalence(mode: str) -> float:
    if mode == "easy":
        return 0.985
    if mode == "dense":
        return 0.892
    return 0.800


def _per_mode_count(mode: str, total: int) -> int:
    for name, share in _MODE_MIX:
        if name == mode:
            return int(round(total * share))
    raise ValueError(f"unknown mode bucket: {mode}")


def _draw_mode_block(
    rng: np.random.Generator,
    mode: str,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` synthetic feature rows + labels for a mode bucket (distributions are mode-dependent so the model has something non-trivial to separate on)."""
    n_features = len(FEATURE_NAMES)
    x = np.zeros((n, n_features), dtype=np.float64)

    # Top-level scalars + subscores: Beta-distributed in [0,1]; (a,b) per mode.
    if mode == "easy":
        score_pairs = (8.0, 2.0)
        sub_pairs = (6.0, 2.0)
        width_loc, width_scale = 35.0, 6.0
        conf_pair = (8.0, 2.0)
    elif mode == "dense":
        score_pairs = (5.0, 3.0)
        sub_pairs = (4.0, 3.0)
        width_loc, width_scale = 40.0, 10.0
        conf_pair = (5.0, 3.0)
    else:  # auto
        score_pairs = (4.0, 3.5)
        sub_pairs = (3.5, 3.5)
        width_loc, width_scale = 45.0, 12.0
        conf_pair = (4.0, 3.0)

    # Indices: 0..3 top scalars, 4..17 subscores, 18 width, 19 confidence,
    # 20..22 mode one-hot.
    x[:, 0:4] = rng.beta(*score_pairs, size=(n, 4))
    x[:, 4:18] = rng.beta(*sub_pairs, size=(n, 14))
    x[:, 18] = np.clip(rng.normal(width_loc, width_scale, size=n), 5.0, 90.0)
    x[:, 19] = rng.beta(*conf_pair, size=n)

    one_hot = {"easy": 20, "dense": 21, "auto": 22}[mode]
    x[:, 20:23] = 0.0
    x[:, one_hot] = 1.0

    # Label generation: weighted sum of a handful of physically-meaningful
    # features, mapped through a logistic with a mode-specific bias so the
    # positive-class prevalence matches the canonical packs.
    logits = (
        2.2 * (x[:, 0] - 0.5)              # geometric_score
        + 2.0 * (x[:, 1] - 0.5)             # stability_score
        + 1.0 * (x[:, 2] - 0.5)             # reachability_score
        + 1.2 * (x[:, 3] - 0.5)             # feasibility_score
        + 1.5 * (x[:, 10] - 0.5)            # stab_collision_free
        + 0.8 * (x[:, 19] - 0.5)            # pose_confidence
    )
    # Calibrate bias so mean(sigmoid(logits + bias)) ~= target prevalence.
    target = _mode_prevalence(mode)
    bias = _solve_bias_for_prevalence(logits, target)
    probs = 1.0 / (1.0 + np.exp(-(logits + bias)))
    y = (rng.random(size=n) < probs).astype(np.int64)
    return x, y


def _solve_bias_for_prevalence(
    logits: np.ndarray,
    target: float,
    *,
    iters: int = 60,
) -> float:
    """Bisection on bias so ``mean(sigmoid(logits + bias)) ~= target``."""
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(logits + mid)))))
        if mean < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_synthetic_dataset(
    spec: DatasetSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate a deterministic synthetic training dataset as ``(X, y, metadata)`` (X ``(n_attempts, 23)`` float64, y int64 in ``{0, 1}``)."""
    cfg = spec or DatasetSpec()
    rng = np.random.default_rng(cfg.seed)

    blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
    remaining = cfg.n_attempts
    total_assigned = 0
    for name, _ in _MODE_MIX[:-1]:
        n = _per_mode_count(name, cfg.n_attempts)
        x, y = _draw_mode_block(rng, name, n)
        blocks.append((name, x, y))
        total_assigned += n
        remaining -= n
    # Give the residual to the last mode so totals exactly add up.
    last_name, _ = _MODE_MIX[-1]
    x_last, y_last = _draw_mode_block(rng, last_name, remaining)
    blocks.append((last_name, x_last, y_last))

    x = np.vstack([b[1] for b in blocks])
    y = np.concatenate([b[2] for b in blocks])

    # Deterministic shuffle so the model never sees pure mode-sorted batches.
    perm = rng.permutation(x.shape[0])
    x = x[perm]
    y = y[perm]

    hasher = hashlib.sha256()
    hasher.update(x.tobytes())
    hasher.update(y.tobytes())
    metadata = {
        "kind": "synthetic_bootstrap",
        "seed": cfg.seed,
        "n_attempts": int(cfg.n_attempts),
        "feature_schema_version": cfg.feature_schema_version,
        "mode_counts": {b[0]: int(b[1].shape[0]) for b in blocks},
        "positive_prevalence": float(y.mean()),
        "dataset_sha256": hasher.hexdigest(),
        "note": (
            "Synthetic bootstrap dataset; exercises the training/export pipeline. Prefer a real "
            "grasp-outcome corpus via build_dataset_from_records once one is collected."
        ),
    }
    return x, y, metadata


#: Key under a ``GraspAttemptRecord``'s ``extra`` bag holding the executed grasp's 23-d feature vector
#: (stamped by the record serializer when the shadow success predictor scored the grasp).
RECORD_FEATURES_KEY: str = "success_model_features"

#: The ``final_outcome`` value that counts as a positive (success) label.
_SUCCESS_OUTCOME: str = "succeeded"


def load_records_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL corpus of logged ``GraspAttemptRecord`` entries (one JSON object per line)."""
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    logger.info("Loaded %d records from %s", len(records), Path(path))
    return records


def build_dataset_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a real training dataset ``(X, y, metadata)`` from logged ``GraspAttemptRecord`` entries.

    A record contributes one row exactly when its ``extra[RECORD_FEATURES_KEY]`` carries the locked
    23-dimensional feature vector, and a record without one is skipped. The label is ``1`` exactly
    when ``final_outcome == "succeeded"``. This is the real-data counterpart of
    :func:`build_synthetic_dataset`: point ``train --records <jsonl>`` at a corpus of real pick logs,
    from simulation or from hardware, to train on measured grasp outcomes instead of the synthetic
    bootstrap.
    """
    xs: list[list[float]] = []
    ys: list[int] = []
    skipped = 0
    for record in records:
        extra = record.get("extra") if isinstance(record, Mapping) else None
        feats = extra.get(RECORD_FEATURES_KEY) if isinstance(extra, Mapping) else None
        if not isinstance(feats, (list, tuple)) or len(feats) != len(FEATURE_NAMES):
            skipped += 1
            continue
        # A feature unavailable at collection time is logged as ``null``: the feature extractor emits
        # NaN for a missing signal, the feasibility features when feasibility scoring is not wired for
        # instance, and JSON serialises NaN to ``null``. Those read back as NaN, which is the
        # convention of the live scorer, rather than costing the whole record; the NaN columns are
        # mean-imputed below exactly as ``predict_proba`` imputes at inference. A value that is
        # neither numeric nor null is genuinely bad, so that record is skipped.
        try:
            xs.append([float(v) if v is not None else float("nan") for v in feats])
        except (TypeError, ValueError):
            skipped += 1
            continue
        ys.append(1 if record.get("final_outcome") == _SUCCESS_OUTCOME else 0)

    if not xs:
        raise ValueError(
            f"no records carried a length-{len(FEATURE_NAMES)} '{RECORD_FEATURES_KEY}' vector; "
            "feature-logging must be active (shadow success predictor on) when the corpus is collected"
        )

    x = np.asarray(xs, dtype=np.float64)
    # Mean-impute NaN with the per-feature training mean, so the downstream fit and the exported
    # ``feature_means`` are finite, matching how ``predict_proba`` imputes at inference. A column that
    # is entirely NaN becomes 0.0.
    if np.isnan(x).any():
        with np.errstate(invalid="ignore"):  # an all-NaN column -> nanmean warns; handled as 0.0 below
            col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_rows, nan_cols = np.where(np.isnan(x))
        x[nan_rows, nan_cols] = np.take(col_means, nan_cols)
    y = np.asarray(ys, dtype=np.int64)
    hasher = hashlib.sha256()
    hasher.update(x.tobytes())
    hasher.update(y.tobytes())
    metadata: dict[str, Any] = {
        "kind": "real_records",
        "n_attempts": len(xs),
        "n_skipped_missing_features": skipped,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "positive_prevalence": float(y.mean()),
        "dataset_sha256": hasher.hexdigest(),
        "note": "Real logged grasp-outcome records (extra.success_model_features + final_outcome).",
    }
    # Skipped rows are the silent failure here: a corpus collected without
    # feature logging yields a dataset that looks plausible and is tiny.
    logger.info(
        "Real-record dataset: %d rows, %d skipped (no/!=%d-d feature vector), prevalence=%.3f",
        len(xs),
        skipped,
        len(FEATURE_NAMES),
        float(y.mean()),
    )
    if skipped:
        logger.warning(
            "%d of %d records carried no usable '%s' vector",
            skipped,
            skipped + len(xs),
            RECORD_FEATURES_KEY,
        )
    return x, y, metadata


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """All knobs for one reproducible training run."""

    train_seed: int = DEFAULT_TRAIN_SEED
    n_cv_folds: int = 5
    # Logistic-regression family.
    logistic_C: float = 1.0
    logistic_max_iter: int = 200
    # The gradient-boosted-trees family, the more expressive default. A bigger model means more and
    # deeper trees. The ensemble is serialized to JSON and evaluated at runtime by a numpy traversal,
    # with no sklearn.
    gbt_n_estimators: int = 300
    gbt_max_depth: int = 3
    gbt_learning_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Container for everything :func:`train_logistic_isotonic` returns."""

    model: SuccessProbabilityModel
    holdout_metrics: dict[str, float]
    cv_metrics: dict[str, dict[str, Any]]
    feature_schema_version: int


def build_train_config() -> TrainConfig:
    """Return the locked default :class:`TrainConfig` for v1."""
    return TrainConfig()


def _safe_sklearn_imports() -> tuple[Any, Any, Any, Any]:
    """Late, scoped import so the runtime predictor stays sklearn-free."""
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-not-found]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-not-found]
        import sklearn  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover (defensive)
        raise RuntimeError(
            "scikit-learn is required for training but is not installed"
        ) from exc
    return LogisticRegression, IsotonicRegression, StratifiedKFold, sklearn


def _fit_logistic(
    LogisticRegression: Any,
    x: np.ndarray,
    y: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    cfg: TrainConfig,
) -> tuple[np.ndarray, float]:
    z = (x - means) / np.where(stds > 1e-12, stds, 1.0)
    model = LogisticRegression(
        C=cfg.logistic_C,
        max_iter=cfg.logistic_max_iter,
        solver="lbfgs",
        random_state=cfg.train_seed,
    )
    model.fit(z, y)
    return model.coef_.ravel().astype(np.float64), float(model.intercept_[0])


def _fit_isotonic(
    IsotonicRegression: Any,
    raw_proba: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_proba, y)
    x_th = np.asarray(iso.X_thresholds_, dtype=np.float64)
    y_th = np.asarray(iso.y_thresholds_, dtype=np.float64)
    return x_th, y_th


def train_logistic_isotonic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dataset_metadata: Mapping[str, Any],
    config: TrainConfig | None = None,
) -> TrainResult:
    """Fit a standardised logistic + isotonic calibrator and export weights."""
    cfg = config or build_train_config()
    started = time.perf_counter()
    LogisticRegression, IsotonicRegression, StratifiedKFold, sklearn_mod = (
        _safe_sklearn_imports()
    )

    means = x.mean(axis=0)
    stds = x.std(axis=0)

    coefs, intercept = _fit_logistic(LogisticRegression, x, y, means, stds, cfg)
    z_full = (x - means) / np.where(stds > 1e-12, stds, 1.0)
    raw_full = 1.0 / (1.0 + np.exp(-(z_full @ coefs + intercept)))
    # The isotonic fit is in-sample and therefore slightly optimistic. The 5-fold
    # CV metrics below are the honest generalisation signal.
    iso_x, iso_y = _fit_isotonic(IsotonicRegression, raw_full, y)

    cv_metrics = cross_validate_metrics(
        x,
        y,
        cfg=cfg,
        LogisticRegression=LogisticRegression,
        IsotonicRegression=IsotonicRegression,
        StratifiedKFold=StratifiedKFold,
    )

    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model_family": MODEL_FAMILY_LOGISTIC,
        "feature_names": list(FEATURE_NAMES),
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "logistic_coefficients": coefs.tolist(),
        "logistic_intercept": intercept,
        "isotonic_x": iso_x.tolist(),
        "isotonic_y": iso_y.tolist(),
    }
    manifest = _build_manifest(
        model_family=MODEL_FAMILY_LOGISTIC,
        artifact_version=MODEL_ARTIFACT_VERSION,
        training={
            "train_seed": cfg.train_seed,
            "n_cv_folds": cfg.n_cv_folds,
            "logistic_C": cfg.logistic_C,
            "logistic_max_iter": cfg.logistic_max_iter,
        },
        dataset_metadata=dataset_metadata,
        cv_metrics=cv_metrics,
        sklearn_version=str(getattr(sklearn_mod, "__version__", "unknown")),
        numpy_version=np.__version__,
    )
    model = _model_from_payload(payload, manifest)

    holdout_preds = predict_proba(model, x)
    holdout_metrics = evaluate_metrics(y, holdout_preds)
    logger.info(
        "Trained logistic+isotonic on %d x %d in %.0f ms: cv_brier=%.4f holdout_brier=%.4f",
        x.shape[0],
        x.shape[1],
        (time.perf_counter() - started) * 1000.0,
        float(cv_metrics["brier"]["mean"]),
        float(holdout_metrics["brier"]),
    )

    return TrainResult(
        model=model,
        holdout_metrics=holdout_metrics,
        cv_metrics=cv_metrics,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def cross_validate_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cfg: TrainConfig,
    LogisticRegression: Any,
    IsotonicRegression: Any,
    StratifiedKFold: Any,
) -> dict[str, dict[str, Any]]:
    """5-fold stratified CV; reports per-fold + mean/std for each metric."""
    skf = StratifiedKFold(
        n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.train_seed
    )
    brier: list[float] = []
    ece: list[float] = []
    auroc: list[float] = []
    for train_idx, test_idx in skf.split(x, y):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        means = x_tr.mean(axis=0)
        stds = x_tr.std(axis=0)
        coefs, intercept = _fit_logistic(
            LogisticRegression, x_tr, y_tr, means, stds, cfg
        )
        z_tr = (x_tr - means) / np.where(stds > 1e-12, stds, 1.0)
        raw_tr = 1.0 / (1.0 + np.exp(-(z_tr @ coefs + intercept)))
        iso_x, iso_y = _fit_isotonic(IsotonicRegression, raw_tr, y_tr)

        z_te = (x_te - means) / np.where(stds > 1e-12, stds, 1.0)
        raw_te = 1.0 / (1.0 + np.exp(-(z_te @ coefs + intercept)))
        calibrated = np.interp(raw_te, iso_x, iso_y)
        calibrated = np.clip(calibrated, 0.0, 1.0)

        metrics = evaluate_metrics(y_te, calibrated)
        brier.append(metrics["brier"])
        ece.append(metrics["ece"])
        auroc.append(metrics["auroc"])

    return {
        "brier": {
            "mean": float(np.mean(brier)),
            "std": float(np.std(brier)),
            "per_fold": [float(v) for v in brier],
        },
        "ece": {
            "mean": float(np.mean(ece)),
            "std": float(np.std(ece)),
            "per_fold": [float(v) for v in ece],
        },
        "auroc": {
            "mean": float(np.mean(auroc)),
            "std": float(np.std(auroc)),
            "per_fold": [float(v) for v in auroc],
        },
    }


# Metrics: numpy-only, no sklearn dependency at eval time.


def evaluate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_ece_bins: int = 10,
) -> dict[str, float]:
    """Return Brier score, ECE (equal-width bins), and AUROC, all in pure numpy."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_proba, dtype=np.float64), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError("shape mismatch between y_true and y_proba")

    brier = float(np.mean((p - y) ** 2))

    edges = np.linspace(0.0, 1.0, n_ece_bins + 1)
    ece_sum = 0.0
    n = y.size
    for i in range(n_ece_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_ece_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        bin_p = float(np.mean(p[mask]))
        bin_y = float(np.mean(y[mask]))
        weight = float(np.sum(mask)) / float(n)
        ece_sum += weight * abs(bin_p - bin_y)
    ece = float(ece_sum)

    auroc = _auroc(y, p)
    return {"brier": brier, "ece": ece, "auroc": auroc}


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """The Mann-Whitney U formulation of AUROC. It is ``nan`` when a class is missing."""
    pos = p[y > 0.5]
    neg = p[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    # Tie correction: average ranks of equal-probability groups.
    sorted_p = p[order]
    i = 0
    while i < p.size:
        j = i
        while j + 1 < p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(np.sum(ranks[y > 0.5]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _build_manifest(
    *,
    model_family: str,
    artifact_version: int,
    training: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    cv_metrics: Mapping[str, Mapping[str, Any]],
    sklearn_version: str,
    numpy_version: str,
) -> dict[str, Any]:
    return {
        "artifact_version": artifact_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_family": f"{model_family}+isotonic_calibration",
        "dataset": dict(dataset_metadata),
        "training": dict(training),
        "metrics_cv_5fold": {
            metric: {
                "mean": float(values["mean"]),
                "std": float(values["std"]),
                "per_fold": list(values["per_fold"]),
            }
            for metric, values in cv_metrics.items()
        },
        "feature_names": list(FEATURE_NAMES),
        "mode_buckets": list(MODE_BUCKETS),
        "environment": {
            "sklearn_version": sklearn_version,
            "numpy_version": numpy_version,
            "python_version": sys.version.split()[0],
            "platform": platform.system().lower(),
        },
        "runtime_predictor": (
            "src.robot.grasping.scoring.success_probability"
            ":load_success_probability_model"
        ),
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _write_artifact(
    model: SuccessProbabilityModel,
    artifact_dir: Path,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.json"
    manifest_path = artifact_dir / "manifest.json"
    model_bytes = _canonical_json_bytes(model.to_dict())
    manifest_bytes = _canonical_json_bytes(model.manifest)
    model_path.write_bytes(model_bytes)
    manifest_path.write_bytes(manifest_bytes)
    logger.info(
        "Exported artifact to %s: model.json %d bytes (sha %s...), manifest.json %d bytes",
        artifact_dir,
        len(model_bytes),
        hashlib.sha256(model_bytes).hexdigest()[:12],
        len(manifest_bytes),
    )
    return {
        "model_json": str(model_path),
        "manifest_json": str(manifest_path),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _safe_gbt_imports() -> tuple[Any, Any, Any, Any]:
    """Late, scoped import so the runtime predictor stays sklearn-free."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-not-found]
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-not-found]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-not-found]
        import sklearn  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover (defensive)
        raise RuntimeError("scikit-learn is required for training but is not installed") from exc
    return GradientBoostingClassifier, IsotonicRegression, StratifiedKFold, sklearn


def _fit_gbc(gbc_cls: Any, x: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> Any:
    model = gbc_cls(
        n_estimators=cfg.gbt_n_estimators,
        max_depth=cfg.gbt_max_depth,
        learning_rate=cfg.gbt_learning_rate,
        subsample=1.0,  # deterministic: no stochastic row subsampling
        random_state=cfg.train_seed,
    )
    model.fit(x, y)
    return model


def _serialize_gbc(gbc: Any, x: np.ndarray, learning_rate: float) -> dict[str, Any]:
    """Serialize a fitted GradientBoostingClassifier to the numpy-evaluable ``gbt`` payload.

    The ``decision_function`` of sklearn is ``init_score + learning_rate * sum(tree leaf values)``,
    so the trees and the init score are stored and a numpy traversal reproduces it. The caller
    validates that end to end.
    """
    trees: list[dict[str, Any]] = []
    tree_sum = np.zeros(x.shape[0], dtype=np.float64)
    for estimator in gbc.estimators_[:, 0]:
        tree = estimator.tree_
        feature = tree.feature.astype(np.int64).copy()
        feature[tree.children_left == -1] = -1  # normalise sklearn's leaf marker (-2) to -1
        trees.append(
            {
                "feature": feature.tolist(),
                "threshold": tree.threshold.astype(np.float64).tolist(),
                "left": tree.children_left.astype(np.int64).tolist(),
                "right": tree.children_right.astype(np.int64).tolist(),
                "value": tree.value.reshape(-1).astype(np.float64).tolist(),
            }
        )
        tree_sum += np.asarray(estimator.predict(x), dtype=np.float64)
    decision = np.asarray(gbc.decision_function(x), dtype=np.float64).ravel()
    init_score = float(np.mean(decision - learning_rate * tree_sum))
    if not np.allclose(decision, init_score + learning_rate * tree_sum, atol=1e-9):
        raise RuntimeError("GBT decision_function != init + lr*sum(trees); cannot serialize losslessly")
    return {"learning_rate": float(learning_rate), "init_score": init_score, "trees": trees}


def gbt_cross_validate_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cfg: TrainConfig,
    gbc_cls: Any,
    IsotonicRegression: Any,
    StratifiedKFold: Any,
) -> dict[str, dict[str, Any]]:
    """5-fold stratified CV for the GBT family; per-fold + mean/std brier/ece/auroc."""
    skf = StratifiedKFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.train_seed)
    scores: dict[str, list[float]] = {"brier": [], "ece": [], "auroc": []}
    for train_idx, test_idx in skf.split(x, y):
        gbc = _fit_gbc(gbc_cls, x[train_idx], y[train_idx], cfg)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(gbc.predict_proba(x[train_idx])[:, 1], y[train_idx])
        cal = np.clip(iso.transform(gbc.predict_proba(x[test_idx])[:, 1]), 0.0, 1.0)
        metrics = evaluate_metrics(y[test_idx], cal)
        for key in scores:
            scores[key].append(metrics[key])
    return {
        key: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_fold": [float(f) for f in v]}
        for key, v in scores.items()
    }


def train_gradient_boosted(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dataset_metadata: Mapping[str, Any],
    config: TrainConfig | None = None,
) -> TrainResult:
    """Fit a gradient-boosted-tree classifier + isotonic calibrator and export a numpy-evaluable artifact."""
    cfg = config or build_train_config()
    started = time.perf_counter()
    gbc_cls, IsotonicRegression, StratifiedKFold, sklearn_mod = _safe_gbt_imports()

    gbc = _fit_gbc(gbc_cls, x, y, cfg)
    gbt_payload = _serialize_gbc(gbc, x, cfg.gbt_learning_rate)
    raw_proba = gbc.predict_proba(x)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_proba, y)

    cv_metrics = gbt_cross_validate_metrics(
        x, y, cfg=cfg, gbc_cls=gbc_cls, IsotonicRegression=IsotonicRegression, StratifiedKFold=StratifiedKFold
    )
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "artifact_version": GBT_ARTIFACT_VERSION,
        "model_family": MODEL_FAMILY_GBT,
        "feature_names": list(FEATURE_NAMES),
        "feature_means": x.mean(axis=0).tolist(),  # used only for NaN imputation at inference
        "isotonic_x": np.asarray(iso.X_thresholds_, dtype=np.float64).tolist(),
        "isotonic_y": np.asarray(iso.y_thresholds_, dtype=np.float64).tolist(),
        "gbt": gbt_payload,
    }
    manifest = _build_manifest(
        model_family=MODEL_FAMILY_GBT,
        artifact_version=GBT_ARTIFACT_VERSION,
        training={
            "train_seed": cfg.train_seed,
            "n_cv_folds": cfg.n_cv_folds,
            "gbt_n_estimators": cfg.gbt_n_estimators,
            "gbt_max_depth": cfg.gbt_max_depth,
            "gbt_learning_rate": cfg.gbt_learning_rate,
        },
        dataset_metadata=dataset_metadata,
        cv_metrics=cv_metrics,
        sklearn_version=str(getattr(sklearn_mod, "__version__", "unknown")),
        numpy_version=np.__version__,
    )
    model = _model_from_payload(payload, manifest)

    # The exported artifact must predict identically without scikit-learn, so the numpy runtime, the
    # tree traversal and the isotonic map, is validated against the sklearn model before shipping.
    reference = np.clip(iso.transform(raw_proba), 0.0, 1.0)
    numpy_preds = predict_proba(model, x)
    if not np.allclose(numpy_preds, reference, atol=1e-9):
        raise RuntimeError("numpy GBT runtime does not reproduce the sklearn model")

    holdout_metrics = evaluate_metrics(y, numpy_preds)
    logger.info(
        "Trained GBT (%d trees, depth %d) on %d x %d in %.0f ms: "
        "cv_brier=%.4f holdout_brier=%.4f, numpy runtime reproduces sklearn",
        cfg.gbt_n_estimators,
        cfg.gbt_max_depth,
        x.shape[0],
        x.shape[1],
        (time.perf_counter() - started) * 1000.0,
        float(cv_metrics["brier"]["mean"]),
        float(holdout_metrics["brier"]),
    )

    return TrainResult(
        model=model,
        holdout_metrics=holdout_metrics,
        cv_metrics=cv_metrics,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def train_and_export(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    dataset_spec: DatasetSpec | None = None,
    train_config: TrainConfig | None = None,
    dataset: tuple[np.ndarray, np.ndarray, Mapping[str, Any]] | None = None,
    model_family: str = MODEL_FAMILY_LOGISTIC,
) -> dict[str, Any]:
    """The whole path: a dataset, then training, then the CV gates, then the JSON export.

    ``dataset`` is a pre-built ``(X, y, metadata)``, from :func:`build_dataset_from_records` for real
    logs for instance. When it is omitted the synthetic bootstrap dataset is generated from
    ``dataset_spec``. ``model_family`` selects the trainer, either gradient-boosted trees, which are
    the more expressive default, or logistic regression.
    """
    cfg = train_config or build_train_config()
    if dataset is not None:
        x, y, metadata = dataset[0], dataset[1], dict(dataset[2])
    else:
        x, y, metadata = build_synthetic_dataset(dataset_spec or DatasetSpec())
    if model_family == MODEL_FAMILY_LOGISTIC:
        result = train_logistic_isotonic(x, y, dataset_metadata=metadata, config=cfg)
    else:
        result = train_gradient_boosted(x, y, dataset_metadata=metadata, config=cfg)
    files = _write_artifact(result.model, Path(artifact_dir))
    return {
        "artifact_dir": str(artifact_dir),
        "files": files,
        "cv_metrics": result.cv_metrics,
        "holdout_metrics": result.holdout_metrics,
        "dataset_metadata": metadata,
    }


# Eval helpers: re-load + score a freshly generated synthetic test slice.


def eval_artifact_against_fresh_split(
    artifact_dir: str | Path,
    *,
    eval_seed: int,
    n_attempts: int = 1000,
) -> dict[str, float]:
    """Load an on-disk artifact and score it on an independently-seeded synthetic slice."""
    from src.robot.grasping.scoring.success_probability import (
        load_success_probability_model,
    )

    spec = DatasetSpec(seed=eval_seed, n_attempts=n_attempts)
    x, y, _ = build_synthetic_dataset(spec)
    model = load_success_probability_model(artifact_dir)
    p = predict_proba(model, x)
    metrics = evaluate_metrics(y, p)
    logger.info(
        "Eval of %s on fresh slice (seed=%d, n=%d): brier=%.4f ece=%.4f auroc=%.4f",
        Path(artifact_dir).name,
        eval_seed,
        n_attempts,
        metrics["brier"],
        metrics["ece"],
        metrics["auroc"],
    )
    return metrics


def _cmd_train(args: argparse.Namespace) -> int:
    dataset = None
    if args.records:
        dataset = build_dataset_from_records(load_records_jsonl(args.records))
    report = train_and_export(
        artifact_dir=args.artifact_dir,
        dataset_spec=DatasetSpec(seed=args.dataset_seed, n_attempts=args.n_attempts),
        train_config=TrainConfig(train_seed=args.train_seed),
        dataset=dataset,
        model_family=args.model_family,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    metrics = eval_artifact_against_fresh_split(
        args.artifact_dir,
        eval_seed=args.eval_seed,
        n_attempts=args.n_attempts,
    )
    print(json.dumps({"eval_metrics": metrics}, indent=2, sort_keys=True))
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    # A local import, which keeps the trainer importable on a machine without the
    # promotion module wired in and avoids a circular import with the predictor
    # module that ``model_promotion`` loads from.
    from src.robot.grasping.calibration.model_promotion import (
        PromotionThresholds,
        promote_artifact,
    )

    thresholds = PromotionThresholds(
        brier_max=args.brier_max,
        log_loss_max=args.log_loss_max,
    )
    report = promote_artifact(
        args.artifact_dir,
        promoted_at=args.promoted_at,
        promoted_by=args.promoted_by,
        thresholds=thresholds,
        validation_seed=args.validation_seed,
        n_attempts=args.n_attempts,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    # The exit code is a property of the report rather than a rule this handler restates: 0 for
    # `pass` and 1 for anything else.
    return report.exit_code


def _cmd_verify(args: argparse.Namespace) -> int:
    from src.robot.grasping.calibration.model_promotion import (
        PromotionThresholds,
        verify_promotion,
    )

    thresholds = PromotionThresholds(
        brier_max=args.brier_max,
        log_loss_max=args.log_loss_max,
    )
    ok, reasons = verify_promotion(args.artifact_dir, thresholds=thresholds)
    print(json.dumps({"ok": ok, "reasons": reasons}, indent=2, sort_keys=True))
    return 0 if ok else 2


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="success_model_calibration",
        description="Offline train/eval/export for the success-probability model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train + export the artifact.")
    train.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    train.add_argument("--dataset-seed", type=int, default=DEFAULT_DATASET_SEED)
    train.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    train.add_argument("--n-attempts", type=int, default=DEFAULT_N_ATTEMPTS)
    train.add_argument(
        "--model-family",
        choices=(MODEL_FAMILY_LOGISTIC, MODEL_FAMILY_GBT),
        default=MODEL_FAMILY_LOGISTIC,
        help="Model to train. Default 'logistic_regression' fits the synthetic bootstrap; use "
        "'gradient_boosted_trees' (more expressive) with --records on a real grasp-outcome corpus.",
    )
    train.add_argument(
        "--records",
        default=None,
        help="JSONL of logged GraspAttemptRecords to train on (REAL data); "
        "default = the synthetic bootstrap dataset.",
    )
    train.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="Score an artifact on a fresh synthetic slice.")
    ev.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    ev.add_argument("--eval-seed", type=int, default=DEFAULT_DATASET_SEED + 1)
    ev.add_argument("--n-attempts", type=int, default=1000)
    ev.set_defaults(func=_cmd_eval)

    # The promotion subcommands, imported lazily so this CLI still exposes
    # train and eval if the promotion module is removed.
    from src.robot.grasping.calibration.model_promotion import (
        PROMOTION_VALIDATION_ATTEMPTS as _VAL_ATTEMPTS,
        PROMOTION_VALIDATION_SEED as _VAL_SEED,
        PromotionThresholds as _PT,
    )
    _defaults = _PT()

    pr = sub.add_parser(
        "promote",
        help="Evaluate an artifact against the promotion gate and write promotion.json.",
    )
    pr.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    pr.add_argument("--validation-seed", type=int, default=_VAL_SEED)
    pr.add_argument("--n-attempts", type=int, default=_VAL_ATTEMPTS)
    pr.add_argument("--brier-max", type=float, default=_defaults.brier_max)
    pr.add_argument("--log-loss-max", type=float, default=_defaults.log_loss_max)
    pr.add_argument(
        "--promoted-by",
        default=None,
        help="Override the recorded operator (default: $USER).",
    )
    pr.add_argument(
        "--promoted-at",
        default=None,
        help="Override the recorded UTC timestamp (default: now). "
        "Use a frozen value to produce a byte-deterministic promotion.json.",
    )
    pr.set_defaults(func=_cmd_promote)

    vf = sub.add_parser(
        "verify",
        help="Verify an existing promotion.json against the artifact bytes.",
    )
    vf.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    vf.add_argument("--brier-max", type=float, default=_defaults.brier_max)
    vf.add_argument("--log-loss-max", type=float, default=_defaults.log_loss_max)
    vf.set_defaults(func=_cmd_verify)

    ns = parser.parse_args(list(argv) if argv is not None else None)
    return int(ns.func(ns))


if __name__ == "__main__":  # pragma: no cover (module CLI guard)
    raise SystemExit(main())
