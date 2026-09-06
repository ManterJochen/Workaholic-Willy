"""Stronger success-model family (MLP) + the honest A/B family-selection gate.

Two things must hold: (1) the shipped numpy-only runtime reproduces what sklearn trained
(``sigmoid(mlp_raw) == clf.predict_proba[:,1]``) and the artifact round-trips; (2) the A/B gate is
fail-closed — a stronger family ships only on a positive Brier margin with no ECE/AUROC regression,
and abstains when AUROC is not gradeable.
"""

from __future__ import annotations

import unittest
import warnings
from contextlib import contextmanager

import numpy as np

from src.robot.grasping.calibration.model_families import (
    ABDecision,
    FamilyResult,
    fit_mlp_network,
    select_winner,
)
from src.robot.grasping.scoring.success_probability._model import (
    MlpLayer,
    MlpNetwork,
    SuccessProbabilityModel,
    _mlp_raw_predict,
    _model_from_payload,
    predict_proba,
)
from src.robot.grasping.scoring.success_probability._schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MLP_ARTIFACT_VERSION,
    MODEL_FAMILY_MLP,
    ArtifactSchemaError,
)

_N = len(FEATURE_NAMES)


@contextmanager
def _quiet_fit():
    """These tests assert the numpy runtime reproduces what sklearn fitted — convergence on the tiny
    toy fixture is beside the point, so its ConvergenceWarning is noise HERE (never in training)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _toy_data(n: int = 120, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, _N))
    # a learnable signal on the first two features
    y = ((z[:, 0] + 0.8 * z[:, 1]) > 0.0).astype(np.float64)
    return z, y


def _mlp_model(net: MlpNetwork) -> SuccessProbabilityModel:
    return SuccessProbabilityModel(
        schema_version=FEATURE_SCHEMA_VERSION,
        artifact_version=MLP_ARTIFACT_VERSION,
        model_family=MODEL_FAMILY_MLP,
        feature_names=FEATURE_NAMES,
        feature_means=np.zeros(_N),
        isotonic_x=np.array([0.0, 1.0]),
        isotonic_y=np.array([0.0, 1.0]),
        manifest={},
        feature_stds=np.ones(_N),
        mlp=net,
    )


class MlpRuntimeTests(unittest.TestCase):
    def test_numpy_forward_matches_sklearn(self) -> None:
        try:
            from sklearn.neural_network import MLPClassifier  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("scikit-learn not installed")
        from sklearn.neural_network import MLPClassifier

        z, y = _toy_data()
        clf = MLPClassifier(
            hidden_layer_sizes=(8, 4), activation="relu", random_state=0, max_iter=400
        )
        with _quiet_fit():
            clf.fit(z, y.astype(int))
        net = MlpNetwork(
            layers=tuple(
                MlpLayer(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64))
                for w, b in zip(clf.coefs_, clf.intercepts_)
            )
        )
        ours = _sigmoid(_mlp_raw_predict(net, z))
        theirs = clf.predict_proba(z)[:, 1]
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)

    def test_fit_mlp_network_shapes_and_agreement(self) -> None:
        z, y = _toy_data()
        with _quiet_fit():
            net = fit_mlp_network(z, y, hidden_layer_sizes=(6,), seed=1, max_iter=300)
        self.assertEqual(len(net.layers), 2)  # 1 hidden + output
        self.assertEqual(net.layers[0].weight.shape, (_N, 6))
        self.assertEqual(net.layers[-1].bias.shape, (1,))
        raw = _mlp_raw_predict(net, z)
        self.assertEqual(raw.shape, (z.shape[0],))

    def test_predict_proba_uses_mlp_branch_and_is_bounded(self) -> None:
        z, y = _toy_data(n=60)
        with _quiet_fit():
            net = fit_mlp_network(z, y, hidden_layer_sizes=(6,), seed=1, max_iter=300)
        p = predict_proba(_mlp_model(net), z)
        self.assertEqual(p.shape, (60,))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))

    def test_artifact_round_trip(self) -> None:
        z, y = _toy_data(n=40)
        with _quiet_fit():
            net = fit_mlp_network(z, y, hidden_layer_sizes=(5,), seed=2, max_iter=200)
        model = _mlp_model(net)
        payload = model.to_dict()
        self.assertEqual(payload["model_family"], MODEL_FAMILY_MLP)
        restored = _model_from_payload(payload, {})
        np.testing.assert_allclose(predict_proba(model, z), predict_proba(restored, z))

    def test_bad_mlp_payload_rejected(self) -> None:
        model = _mlp_model(
            MlpNetwork(layers=(MlpLayer(np.zeros((_N, 1)), np.zeros(1)),))
        )
        payload = model.to_dict()
        payload["mlp"]["activation"] = "tanh"
        with self.assertRaises(ArtifactSchemaError):
            _model_from_payload(payload, {})

    def test_output_layer_must_be_one_unit(self) -> None:
        model = _mlp_model(
            MlpNetwork(layers=(MlpLayer(np.zeros((_N, 2)), np.zeros(2)),))
        )
        with self.assertRaises(ArtifactSchemaError):
            _model_from_payload(model.to_dict(), {})


def _res(family: str, brier: float, ece: float, auroc: float) -> FamilyResult:
    return FamilyResult(
        family=family, metrics={"brier": brier, "ece": ece, "auroc": auroc}, n_eval=100
    )


class ABGateTests(unittest.TestCase):
    def test_challenger_promoted_on_clear_win(self) -> None:
        inc = _res("logistic_regression", brier=0.20, ece=0.05, auroc=0.70)
        ch = _res("mlp", brier=0.15, ece=0.04, auroc=0.80)
        decision = select_winner(inc, [ch])
        self.assertIsInstance(decision, ABDecision)
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.winner, "mlp")

    def test_tie_keeps_incumbent(self) -> None:
        inc = _res("logistic_regression", brier=0.20, ece=0.05, auroc=0.70)
        ch = _res("mlp", brier=0.20, ece=0.05, auroc=0.70)
        decision = select_winner(inc, [ch])
        self.assertFalse(decision.promoted)
        self.assertEqual(decision.winner, "logistic_regression")

    def test_brier_gain_below_floor_keeps_incumbent(self) -> None:
        inc = _res("logistic_regression", brier=0.2000, ece=0.05, auroc=0.70)
        ch = _res("mlp", brier=0.1990, ece=0.05, auroc=0.70)  # gain 0.001 < 0.005 floor
        self.assertFalse(select_winner(inc, [ch]).promoted)

    def test_ece_regression_blocks_promotion(self) -> None:
        inc = _res("logistic_regression", brier=0.20, ece=0.03, auroc=0.70)
        ch = _res("mlp", brier=0.10, ece=0.09, auroc=0.85)  # better Brier, worse calibration
        decision = select_winner(inc, [ch])
        self.assertFalse(decision.promoted)
        self.assertTrue(any("ECE regressed" in r for r in decision.reasons))

    def test_auroc_regression_blocks_calibration_path(self) -> None:
        # a big Brier gain does NOT promote if AUROC (ranking) regressed
        inc = _res("logistic_regression", brier=0.20, ece=0.05, auroc=0.80)
        ch = _res("mlp", brier=0.10, ece=0.04, auroc=0.60)
        decision = select_winner(inc, [ch])
        self.assertFalse(decision.promoted)
        self.assertEqual(decision.winner, "logistic_regression")

    def test_ranking_path_promotes_on_auroc_gain_at_flat_brier(self) -> None:
        # the aletheia geometry-only case: flat Brier, a real AUROC gain -> promote via the ranking path
        inc = _res("logistic_regression", brier=0.0221, ece=0.02, auroc=0.664)
        ch = _res("mlp", brier=0.0220, ece=0.02, auroc=0.739)
        decision = select_winner(inc, [ch])
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.winner, "mlp")

    def test_ranking_path_blocked_by_brier_regression(self) -> None:
        # a big AUROC gain does NOT promote if Brier regressed meaningfully
        inc = _res("logistic_regression", brier=0.020, ece=0.02, auroc=0.66)
        ch = _res("mlp", brier=0.030, ece=0.02, auroc=0.80)  # +0.14 AUROC but +0.010 Brier
        decision = select_winner(inc, [ch])
        self.assertFalse(decision.promoted)

    def test_non_gradeable_auroc_abstains(self) -> None:
        inc = _res("logistic_regression", brier=0.20, ece=0.05, auroc=0.80)
        ch = _res("mlp", brier=0.05, ece=0.01, auroc=float("nan"))
        decision = select_winner(inc, [ch])
        self.assertFalse(decision.promoted)
        self.assertTrue(any("abstain" in r for r in decision.reasons))

    def test_best_admissible_challenger_wins(self) -> None:
        inc = _res("logistic_regression", brier=0.30, ece=0.05, auroc=0.60)
        gbt = _res("gradient_boosted_trees", brier=0.22, ece=0.04, auroc=0.75)
        mlp = _res("mlp", brier=0.18, ece=0.04, auroc=0.78)
        decision = select_winner(inc, [gbt, mlp])
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.winner, "mlp")
        self.assertEqual(set(decision.table), {"logistic_regression", "gradient_boosted_trees", "mlp"})


if __name__ == "__main__":
    unittest.main()
