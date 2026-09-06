"""The stronger success-model family, a ReLU MLP, and the A/B gate that selects between families.

A training-only module. sklearn is imported lazily so the shipped runtime stays numpy-only: the
fitted network is serialised into the numpy-evaluable :class:`MlpNetwork` the predictor already
understands, and the two must agree.

The A/B gate mirrors the RL promotion gate. A stronger family never ships merely because it is
stronger. A challenger is promoted only if it

1. improves the Brier score by at least ``min_brier_improvement``, a positive floor, so float noise
   does not win,
2. does not regress the ECE, which is calibration, beyond tolerance, and
3. does not regress the AUROC, which is ranking, beyond tolerance.

If the AUROC cannot be graded, because a class is missing from the eval split, the challenger
abstains rather than passing. A tie, a regression and an abstention all keep the incumbent. That is
why the shipped bootstrap stays logistic until a real corpus proves a stronger family wins.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..scoring.success_probability._model import (
    MlpLayer,
    MlpNetwork,
    SuccessProbabilityModel,
    _mlp_raw_predict,
    predict_proba,
)
from ..scoring.success_probability._schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MLP_ARTIFACT_VERSION,
    MODEL_FAMILY_GBT,
    MODEL_FAMILY_LOGISTIC,
    MODEL_FAMILY_MLP,
)
from .success_model_calibration import (
    evaluate_metrics,
    train_gradient_boosted,
    train_logistic_isotonic,
)

#: Default gate thresholds. The Brier floor is deliberately a positive margin.
DEFAULT_MIN_BRIER_IMPROVEMENT: float = 0.005
#: A small absolute tolerance on an ECE regression. A zero tolerance blocks on calibration noise,
#: which must not veto a real ranking win, and every family is isotonic-recalibrated anyway. Only a
#: meaningful calibration regression, above 0.5%, blocks.
DEFAULT_MAX_ECE_REGRESSION: float = 0.005
DEFAULT_MAX_AUROC_REGRESSION: float = 0.0
#: A challenger may also earn promotion on a genuine ranking gain, measured by AUROC, which is the
#: operational metric for grasp selection because the model ranks candidates and the top one is
#: taken. On a heavily imbalanced task the Brier score barely moves even when ranking improves a
#: lot, so a Brier-only gate under-rewards a real ranker and can miss a family that ranks better at
#: a flat Brier score. The AUROC path requires this gain and no meaningful Brier or ECE regression.
DEFAULT_MIN_AUROC_IMPROVEMENT: float = 0.02
DEFAULT_MAX_BRIER_REGRESSION: float = 0.002


# ---------------------------------------------------------------------------
# The stronger family: a ReLU MLP fitted, then serialised to a numpy-evaluable network.
# ---------------------------------------------------------------------------


def fit_mlp_network(
    z: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layer_sizes: Sequence[int] = (16, 8),
    seed: int = 0,
    max_iter: int = 500,
) -> MlpNetwork:
    """Fit a ReLU MLP on standardised features and serialise it to :class:`MlpNetwork`.

    The caller standardises (the model stores ``feature_stds``), exactly as for the logistic family.
    ``sigmoid(_mlp_raw_predict(net, z))`` reproduces sklearn's ``predict_proba[:, 1]``."""

    try:
        from sklearn.neural_network import (  # type: ignore[import-not-found]
            MLPClassifier,
        )
    except ImportError as exc:  # pragma: no cover (training-only dependency)
        raise RuntimeError(
            "training the 'mlp' success-model family requires scikit-learn"
        ) from exc

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(int(h) for h in hidden_layer_sizes),
        activation="relu",
        random_state=seed,
        max_iter=max_iter,
    )
    clf.fit(np.asarray(z, dtype=np.float64), np.asarray(y).astype(int))
    layers = tuple(
        MlpLayer(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64))
        for w, b in zip(clf.coefs_, clf.intercepts_)
    )
    return MlpNetwork(layers=layers, activation="relu")


# ---------------------------------------------------------------------------
# A/B family selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyResult:
    """One family's held-out metrics."""

    family: str
    metrics: dict[str, float]  # brier, ece, auroc
    n_eval: int


@dataclass(frozen=True)
class ABDecision:
    """Outcome of the A/B gate; ``promoted=False`` means the incumbent is kept."""

    winner: str
    promoted: bool
    reasons: tuple[str, ...]
    table: dict[str, dict[str, float]]


def evaluate_family(
    model: SuccessProbabilityModel, features: np.ndarray, y_true: np.ndarray
) -> FamilyResult:
    """Score a model on a held-out split, returning its Brier score, ECE and AUROC."""

    y = np.asarray(y_true, dtype=np.float64)
    proba = predict_proba(model, np.asarray(features, dtype=np.float64))
    return FamilyResult(
        family=model.model_family, metrics=evaluate_metrics(y, proba), n_eval=int(y.size)
    )


def select_winner(
    incumbent: FamilyResult,
    challengers: Sequence[FamilyResult],
    *,
    min_brier_improvement: float = DEFAULT_MIN_BRIER_IMPROVEMENT,
    max_ece_regression: float = DEFAULT_MAX_ECE_REGRESSION,
    max_auroc_regression: float = DEFAULT_MAX_AUROC_REGRESSION,
    min_auroc_improvement: float = DEFAULT_MIN_AUROC_IMPROVEMENT,
    max_brier_regression: float = DEFAULT_MAX_BRIER_REGRESSION,
) -> ABDecision:
    """Promote the best admissible challenger, and otherwise keep the incumbent, which is fail-closed.

    A challenger is admissible by either of two paths, and neither may regress the ECE:

    * the calibration path, a positive Brier gain of ``>= min_brier_improvement`` with no AUROC
      regression;
    * the ranking path, a real AUROC gain of ``>= min_auroc_improvement`` with no meaningful Brier
      regression, meaning ``>= -max_brier_regression``.

    Among the admissible challengers the best ranker wins, on the highest AUROC with the Brier score
    as the tiebreak, because ranking is the operational metric for grasp selection. A tie, a
    regression and a non-gradeable AUROC all keep the incumbent.
    """

    reasons: list[str] = []
    table: dict[str, dict[str, float]] = {incumbent.family: dict(incumbent.metrics)}
    admissible: list[FamilyResult] = []

    inc_brier = incumbent.metrics["brier"]
    inc_ece = incumbent.metrics["ece"]
    inc_auroc = incumbent.metrics["auroc"]

    for challenger in challengers:
        table[challenger.family] = dict(challenger.metrics)
        ch_brier = challenger.metrics["brier"]
        ch_ece = challenger.metrics["ece"]
        ch_auroc = challenger.metrics["auroc"]
        if not math.isfinite(inc_auroc) or not math.isfinite(ch_auroc):
            reasons.append(
                f"{challenger.family}: AUROC not gradeable (a class is missing) -> abstain"
            )
            continue
        if ch_ece > inc_ece + max_ece_regression:
            reasons.append(
                f"{challenger.family}: ECE regressed ({ch_ece:.6f} > {inc_ece:.6f})"
            )
            continue
        brier_gain = inc_brier - ch_brier
        auroc_gain = ch_auroc - inc_auroc
        calibration_path = (
            brier_gain >= min_brier_improvement and ch_auroc >= inc_auroc - max_auroc_regression
        )
        ranking_path = (
            auroc_gain >= min_auroc_improvement and brier_gain >= -max_brier_regression
        )
        if calibration_path or ranking_path:
            admissible.append(challenger)
        else:
            reasons.append(
                f"{challenger.family}: neither path cleared "
                f"(Brier gain {brier_gain:+.6f} < floor {min_brier_improvement}; "
                f"AUROC gain {auroc_gain:+.6f} < floor {min_auroc_improvement})"
            )

    if not admissible:
        reasons.append(f"no challenger cleared the gate; keeping incumbent {incumbent.family!r}")
        return ABDecision(
            winner=incumbent.family, promoted=False, reasons=tuple(reasons), table=table
        )

    best = max(admissible, key=lambda c: (c.metrics["auroc"], -c.metrics["brier"]))
    reasons.append(
        f"{best.family!r} promoted: AUROC {best.metrics['auroc']:.6f} / Brier {best.metrics['brier']:.6f} "
        f"beats incumbent {incumbent.family!r} (AUROC {inc_auroc:.6f} / Brier {inc_brier:.6f})"
    )
    return ABDecision(winner=best.family, promoted=True, reasons=tuple(reasons), table=table)


def build_mlp_success_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layer_sizes: Sequence[int] = (32, 16),
    seed: int = 0,
    max_iter: int = 300,
) -> SuccessProbabilityModel:
    """Fit the MLP family end to end, standardising, then the MLP, then isotonic, into a predictable artifact.

    Mirrors what ``train_logistic_isotonic`` / ``train_gradient_boosted`` return (a
    :class:`SuccessProbabilityModel`), so all three families slot into :func:`run_family_ab`."""

    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-not-found]

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    means = x.mean(axis=0)
    stds = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
    z = (x - means) / stds
    net = fit_mlp_network(z, y, hidden_layer_sizes=hidden_layer_sizes, seed=seed, max_iter=max_iter)
    raw = 1.0 / (1.0 + np.exp(-_mlp_raw_predict(net, z)))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    return SuccessProbabilityModel(
        schema_version=FEATURE_SCHEMA_VERSION,
        artifact_version=MLP_ARTIFACT_VERSION,
        model_family=MODEL_FAMILY_MLP,
        feature_names=FEATURE_NAMES,
        feature_means=means,
        isotonic_x=np.asarray(iso.X_thresholds_, dtype=np.float64),
        isotonic_y=np.asarray(iso.y_thresholds_, dtype=np.float64),
        manifest={},
        feature_stds=stds,
        mlp=net,
    )


def run_family_ab(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    val_fraction: float = 0.2,
    families: Sequence[str] = (MODEL_FAMILY_LOGISTIC, MODEL_FAMILY_GBT, MODEL_FAMILY_MLP),
    hidden_layer_sizes: Sequence[int] = (32, 16),
) -> tuple[dict[str, FamilyResult], ABDecision]:
    """Train each requested family on a held-out split and run the A/B gate.

    Logistic is always the incumbent, since it is the shipped bootstrap family. Returns the held-out
    metrics per family together with the :class:`ABDecision`. The split is seeded, so a run
    reproduces."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    x, y = x[perm], y[perm]
    n_tr = int((1.0 - val_fraction) * x.shape[0])
    x_tr, y_tr, x_va, y_va = x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:]
    meta = {"dataset_id": "family_ab", "input_source": "sim"}

    results: dict[str, FamilyResult] = {}
    if MODEL_FAMILY_LOGISTIC in families:
        model = train_logistic_isotonic(x_tr, y_tr, dataset_metadata=meta).model
        results[MODEL_FAMILY_LOGISTIC] = evaluate_family(model, x_va, y_va)
    if MODEL_FAMILY_GBT in families:
        model = train_gradient_boosted(x_tr, y_tr, dataset_metadata=meta).model
        results[MODEL_FAMILY_GBT] = evaluate_family(model, x_va, y_va)
    if MODEL_FAMILY_MLP in families:
        mlp = build_mlp_success_model(
            x_tr, y_tr, hidden_layer_sizes=hidden_layer_sizes, seed=seed
        )
        results[MODEL_FAMILY_MLP] = evaluate_family(mlp, x_va, y_va)

    incumbent = results[MODEL_FAMILY_LOGISTIC]
    challengers = [r for fam, r in results.items() if fam != MODEL_FAMILY_LOGISTIC]
    decision = select_winner(incumbent, challengers)
    return results, decision


__all__ = (
    "DEFAULT_MAX_AUROC_REGRESSION",
    "DEFAULT_MAX_ECE_REGRESSION",
    "DEFAULT_MIN_BRIER_IMPROVEMENT",
    "ABDecision",
    "FamilyResult",
    "build_mlp_success_model",
    "evaluate_family",
    "fit_mlp_network",
    "run_family_ab",
    "select_winner",
)
