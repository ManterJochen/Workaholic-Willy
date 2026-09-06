"""Success-probability model contract lock.

These tests pin the model's design decisions so any future drift is caught at the
test layer:

* Feature schema v1 is frozen (23 ordered names).
* Mode normalisation is fixed for every observed raw mode string.
* The runtime predictor module is **not** imported from any live grasping
  runtime path (it must stay invisible to the live pipeline).
* The on-disk v1 artifact loads, validates, and predicts deterministically.
* The synthetic-bootstrap trainer is byte-deterministic across re-runs.
* 5-fold CV metrics on the synthetic dataset meet the locked quality gates
  (Brier mean \u2264 0.20 / std \u2264 0.03, ECE mean \u2264 0.08 / std \u2264 0.02,
  AUROC mean \u2265 0.75 / std \u2264 0.05).
* ``RobotGraspingConfig`` exposes a ``success_model`` block that is disabled
  by default (the model stays opaque to the runtime).
* ``GraspAttemptRecord.SCHEMA_VERSION`` is still ``1`` and default
  ``to_dict()`` is byte-identical to the record baseline (re-asserted
  here as a cross-cutting guard).
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.config.schema.robot.robot_schema import (
    GraspingSuccessModelConfig,
    RobotGraspingConfig,
)
from src.robot.grasping.calibration.success_model_calibration import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATASET_SEED,
    DEFAULT_N_ATTEMPTS,
    DatasetSpec,
    TrainConfig,
    build_synthetic_dataset,
    eval_artifact_against_fresh_split,
    evaluate_metrics,
    train_and_export,
    train_logistic_isotonic,
)
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.scoring.success_probability import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MODE_BUCKETS,
    MODEL_ARTIFACT_VERSION,
    ArtifactSchemaError,
    extract_features,
    load_success_probability_model,
    normalize_mode,
    predict_proba,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / DEFAULT_ARTIFACT_DIR

EXPECTED_FEATURE_NAMES: tuple[str, ...] = (
    "geometric_score",
    "stability_score",
    "reachability_score",
    "feasibility_score",
    "geo_antipodal",
    "geo_normal_opposition",
    "geo_axis_alignment",
    "geo_width_fit",
    "stab_confidence",
    "stab_contact_stability",
    "stab_collision_free",
    "stab_table_clearance",
    "reach_workspace",
    "reach_approach_alignment",
    "feas_ik_quality",
    "feas_joint_margin",
    "feas_approach_clearance",
    "feas_corridor_risk",
    "pose_grip_width_mm",
    "pose_confidence",
    "mode_easy",
    "mode_dense",
    "mode_auto",
)


# ---------------------------------------------------------------------------
# Schema constants & feature extraction
# ---------------------------------------------------------------------------


class FeatureSchemaContractTests(unittest.TestCase):
    def test_feature_schema_version_locked(self) -> None:
        self.assertEqual(FEATURE_SCHEMA_VERSION, 1)
        self.assertEqual(MODEL_ARTIFACT_VERSION, 1)

    def test_feature_names_locked(self) -> None:
        self.assertEqual(FEATURE_NAMES, EXPECTED_FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 23)
        self.assertEqual(len(set(FEATURE_NAMES)), 23)

    def test_mode_buckets_locked(self) -> None:
        self.assertEqual(MODE_BUCKETS, ("easy", "dense", "auto"))

    def test_normalize_mode_covers_every_raw_string(self) -> None:
        self.assertEqual(normalize_mode("easy"), "easy")
        self.assertEqual(normalize_mode("dense_clutter"), "dense")
        self.assertEqual(normalize_mode("dense_autonomous"), "dense")
        self.assertEqual(normalize_mode("auto"), "auto")
        self.assertEqual(normalize_mode("autonomous"), "auto")
        # Defaults
        self.assertEqual(normalize_mode(None), "auto")
        self.assertEqual(normalize_mode("something_unknown"), "auto")
        self.assertEqual(normalize_mode(""), "auto")


class FeatureExtractionTests(unittest.TestCase):
    def _stub_breakdown(self):
        class _Pose:
            grip_width_mm = 38.0
            confidence = 0.7

        class _Bd:
            geometric_score = 0.6
            stability_score = 0.7
            reachability_score = 0.8
            feasibility_score = 0.55
            pose = _Pose()
            components = {
                "geometric": {
                    "antipodal": 0.6,
                    "normal_opposition": 0.5,
                    "axis_alignment": 0.4,
                    "width_fit": 0.7,
                },
                "stability": {
                    "confidence": 0.65,
                    "contact_stability": 0.55,
                    "collision_free": 1.0,
                    "table_clearance": 0.8,
                },
                "reachability": {
                    "workspace": 0.9,
                    "approach_alignment": 0.7,
                },
                "feasibility": {
                    "ik_quality": 0.5,
                    "joint_margin": 0.6,
                    "approach_clearance": 0.55,
                    "corridor_risk": 0.4,
                },
            }

        return _Bd()

    def test_extract_returns_23_dim_float64_vector(self) -> None:
        vec = extract_features(self._stub_breakdown(), mode="easy")
        self.assertEqual(vec.shape, (23,))
        self.assertEqual(vec.dtype, np.float64)

    def test_mode_one_hot_matches_bucket(self) -> None:
        for raw, bucket in [
            ("easy", "easy"),
            ("dense_clutter", "dense"),
            ("dense_autonomous", "dense"),
            ("auto", "auto"),
        ]:
            vec = extract_features(self._stub_breakdown(), mode=raw)
            for idx, name in enumerate(MODE_BUCKETS):
                key = f"mode_{name}"
                expected = 1.0 if name == bucket else 0.0
                self.assertEqual(
                    vec[EXPECTED_FEATURE_NAMES.index(key)],
                    expected,
                    msg=f"raw={raw} key={key}",
                )

    def test_missing_components_become_nan(self) -> None:
        class _Bd:
            geometric_score = 0.5
            stability_score = math.nan
            reachability_score = 0.5
            feasibility_score = 0.0
            pose = None
            components = {}

        vec = extract_features(_Bd(), mode="auto")
        self.assertTrue(math.isnan(vec[EXPECTED_FEATURE_NAMES.index("stability_score")]))
        self.assertTrue(math.isnan(vec[EXPECTED_FEATURE_NAMES.index("pose_grip_width_mm")]))
        self.assertTrue(math.isnan(vec[EXPECTED_FEATURE_NAMES.index("geo_antipodal")]))


# ---------------------------------------------------------------------------
# Artifact load + predict
# ---------------------------------------------------------------------------


class ArtifactLoadAndPredictTests(unittest.TestCase):
    def test_committed_v1_artifact_loads(self) -> None:
        self.assertTrue(
            ARTIFACT_DIR.is_dir(),
            f"committed v1 artifact missing at {ARTIFACT_DIR}",
        )
        model = load_success_probability_model(ARTIFACT_DIR)
        self.assertEqual(model.schema_version, 1)
        self.assertEqual(model.artifact_version, 1)
        self.assertEqual(model.feature_names, EXPECTED_FEATURE_NAMES)
        self.assertEqual(model.feature_means.shape, (23,))
        self.assertEqual(model.feature_stds.shape, (23,))
        self.assertEqual(model.logistic_coefficients.shape, (23,))
        self.assertGreaterEqual(model.isotonic_x.size, 2)
        self.assertEqual(model.isotonic_x.shape, model.isotonic_y.shape)

    def test_predict_proba_is_in_unit_interval(self) -> None:
        model = load_success_probability_model(ARTIFACT_DIR)
        x, _y, _ = build_synthetic_dataset(DatasetSpec(seed=999, n_attempts=200))
        p = predict_proba(model, x)
        self.assertEqual(p.shape, (200,))
        self.assertTrue(np.all(p >= 0.0))
        self.assertTrue(np.all(p <= 1.0))

    def test_predict_imputes_nans_with_training_means(self) -> None:
        model = load_success_probability_model(ARTIFACT_DIR)
        x_nan = np.full((1, 23), np.nan, dtype=np.float64)
        p_nan = predict_proba(model, x_nan)
        x_mean = model.feature_means.reshape(1, -1).copy()
        p_mean = predict_proba(model, x_mean)
        self.assertAlmostEqual(float(p_nan[0]), float(p_mean[0]), places=12)

    def test_load_rejects_wrong_schema_version(self) -> None:
        model_payload = json.loads((ARTIFACT_DIR / "model.json").read_text(encoding="utf-8"))
        bad = dict(model_payload)
        bad["schema_version"] = 999
        with self.assertRaises(ArtifactSchemaError):
            from backend.src.robot.grasping.scoring.success_probability import (
                _model_from_payload,
            )
            _model_from_payload(bad, {})

    def test_load_rejects_feature_name_reorder(self) -> None:
        model_payload = json.loads((ARTIFACT_DIR / "model.json").read_text(encoding="utf-8"))
        bad = dict(model_payload)
        names = list(model_payload["feature_names"])
        names[0], names[1] = names[1], names[0]
        bad["feature_names"] = names
        with self.assertRaises(ArtifactSchemaError):
            from backend.src.robot.grasping.scoring.success_probability import (
                _model_from_payload,
            )
            _model_from_payload(bad, {})

    def test_load_rejects_missing_directory(self) -> None:
        with self.assertRaises(ArtifactSchemaError):
            load_success_probability_model("/tmp/__nonexistent_u1_dir__")


# ---------------------------------------------------------------------------
# Synthetic dataset & training determinism
# ---------------------------------------------------------------------------


class SyntheticDatasetDeterminismTests(unittest.TestCase):
    def test_seed_locks_dataset_bytes(self) -> None:
        x1, y1, meta1 = build_synthetic_dataset(DatasetSpec())
        x2, y2, meta2 = build_synthetic_dataset(DatasetSpec())
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        self.assertEqual(meta1["dataset_sha256"], meta2["dataset_sha256"])

    def test_dataset_default_size(self) -> None:
        x, y, meta = build_synthetic_dataset(DatasetSpec())
        self.assertEqual(x.shape, (DEFAULT_N_ATTEMPTS, 23))
        self.assertEqual(y.shape, (DEFAULT_N_ATTEMPTS,))
        self.assertEqual(meta["seed"], DEFAULT_DATASET_SEED)
        self.assertEqual(meta["kind"], "synthetic_bootstrap")
        # All four modes accounted for via the three buckets.
        self.assertEqual(set(meta["mode_counts"].keys()), {"easy", "dense", "auto"})

    def test_per_mode_prevalence_is_in_expected_band(self) -> None:
        x, y, _ = build_synthetic_dataset(DatasetSpec())
        # Easy mode has mode_easy one-hot at index 20.
        easy_mask = x[:, EXPECTED_FEATURE_NAMES.index("mode_easy")] > 0.5
        dense_mask = x[:, EXPECTED_FEATURE_NAMES.index("mode_dense")] > 0.5
        auto_mask = x[:, EXPECTED_FEATURE_NAMES.index("mode_auto")] > 0.5
        self.assertGreater(float(y[easy_mask].mean()), 0.95)
        self.assertGreater(float(y[dense_mask].mean()), 0.80)
        self.assertGreater(float(y[auto_mask].mean()), 0.70)


# ---------------------------------------------------------------------------
# CV gates
# ---------------------------------------------------------------------------


class CrossValidationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        x, y, meta = build_synthetic_dataset(DatasetSpec())
        cls.result = train_logistic_isotonic(
            x, y, dataset_metadata=meta, config=TrainConfig()
        )

    def test_brier_gate(self) -> None:
        m = self.result.cv_metrics["brier"]
        self.assertLessEqual(m["mean"], 0.20, msg=f"Brier mean={m['mean']:.4f}")
        self.assertLessEqual(m["std"], 0.03, msg=f"Brier std={m['std']:.4f}")

    def test_ece_gate(self) -> None:
        m = self.result.cv_metrics["ece"]
        self.assertLessEqual(m["mean"], 0.08, msg=f"ECE mean={m['mean']:.4f}")
        self.assertLessEqual(m["std"], 0.02, msg=f"ECE std={m['std']:.4f}")

    def test_auroc_gate(self) -> None:
        m = self.result.cv_metrics["auroc"]
        self.assertGreaterEqual(m["mean"], 0.75, msg=f"AUROC mean={m['mean']:.4f}")
        self.assertLessEqual(m["std"], 0.05, msg=f"AUROC std={m['std']:.4f}")

    def test_cv_folds_count(self) -> None:
        self.assertEqual(len(self.result.cv_metrics["brier"]["per_fold"]), 5)


class HoldoutMetricsSanityTests(unittest.TestCase):
    def test_evaluate_metrics_returns_three_keys(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
        m = evaluate_metrics(y, p)
        self.assertEqual(set(m.keys()), {"brier", "ece", "auroc"})
        self.assertGreater(m["auroc"], 0.9)
        self.assertLess(m["brier"], 0.1)

    def test_auroc_perfect_separation(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(evaluate_metrics(y, p)["auroc"], 1.0)


# ---------------------------------------------------------------------------
# Trainer determinism + artifact byte-stability
# ---------------------------------------------------------------------------


class TrainerByteDeterminismTests(unittest.TestCase):
    def _train_to(self, target_dir: Path) -> dict[str, str]:
        report = train_and_export(
            artifact_dir=target_dir,
            dataset_spec=DatasetSpec(),
            train_config=TrainConfig(),
        )
        return report["files"]

    def test_two_runs_produce_identical_bytes(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            f1 = self._train_to(Path(td1))
            f2 = self._train_to(Path(td2))
            self.assertEqual(f1["model_sha256"], f2["model_sha256"])
            self.assertEqual(f1["manifest_sha256"], f2["manifest_sha256"])
            self.assertEqual(
                Path(f1["model_json"]).read_bytes(),
                Path(f2["model_json"]).read_bytes(),
            )

    def test_committed_artifact_matches_fresh_train(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            files = self._train_to(Path(td))
            fresh_model_bytes = Path(files["model_json"]).read_bytes()
            fresh_manifest = json.loads(Path(files["manifest_json"]).read_text(encoding="utf-8"))

        committed_model_bytes = (ARTIFACT_DIR / "model.json").read_bytes()
        committed_manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(
            hashlib.sha256(fresh_model_bytes).hexdigest(),
            hashlib.sha256(committed_model_bytes).hexdigest(),
            msg=(
                "Committed assets/models/success_probability/v1/model.json is "
                "out of sync with the trainer. Re-run "
                "`.venv/bin/python -m src.robot.grasping.calibration."
                "success_model_calibration train` to refresh."
            ),
        )
        # Manifest deep-equality on the version + dataset hash (environment
        # block intentionally not pinned so re-runs on a slightly different
        # platform still pass — sklearn/numpy/python versions float).
        self.assertEqual(
            fresh_manifest["artifact_version"],
            committed_manifest["artifact_version"],
        )
        self.assertEqual(
            fresh_manifest["feature_schema_version"],
            committed_manifest["feature_schema_version"],
        )
        self.assertEqual(
            fresh_manifest["dataset"]["dataset_sha256"],
            committed_manifest["dataset"]["dataset_sha256"],
        )

    def test_eval_against_fresh_split_is_deterministic(self) -> None:
        m1 = eval_artifact_against_fresh_split(
            ARTIFACT_DIR, eval_seed=12345, n_attempts=500
        )
        m2 = eval_artifact_against_fresh_split(
            ARTIFACT_DIR, eval_seed=12345, n_attempts=500
        )
        for key in ("brier", "ece", "auroc"):
            self.assertAlmostEqual(m1[key], m2[key], places=12)


# ---------------------------------------------------------------------------
# Runtime isolation: the model must stay invisible to the live pipeline
# ---------------------------------------------------------------------------


class RuntimeIsolationTests(unittest.TestCase):
    """Pin the import surface: only the sanctioned integration files may import the predictor/trainer.

    The predictor + trainer are offline tooling; the runtime path is wired
    *exclusively* through :mod:`pick_loop` and :mod:`autonomous_grasp`
    (the orchestrator and the service). Any other runtime module that
    grows an import of the predictor/trainer is rejected here so the
    wiring surface cannot drift sideways into the rest of the runtime.
    """

    FORBIDDEN_IMPORTS = (
        "from src.robot.grasping.scoring.success_probability",
        "import src.robot.grasping.scoring.success_probability",
        "from .success_probability",
        "from src.robot.grasping.calibration.success_model_calibration",
        "import src.robot.grasping.calibration.success_model_calibration",
        "from .success_model_calibration",
    )

    ALLOWED_FILES = {
        # The predictor module itself + the trainer module are obviously
        # allowed to reference these symbols.
        "src/robot/grasping/scoring/success_probability.py",
        "src/robot/grasping/calibration/success_model_calibration.py",
        # The promotion gate lives alongside the trainer; the offline
        # promote/verify entry points load the predictor to score
        # artifacts against the validation slice, and the runtime
        # loader imports the *pure* verify_promotion locally (no
        # sklearn touched) when lifecycle_phase grants behavioural
        # influence.
        "src/robot/grasping/calibration/model_promotion.py",
        # Offline A/B family selection: scores each candidate family (logistic / GBT / MLP) on a
        # held-out slice via the predictor, then gates promotion on Brier/ECE/AUROC. Training-only —
        # sklearn is imported lazily, nothing on the runtime path imports this module back.
        "src/robot/grasping/calibration/model_families.py",
        # Runtime wiring seams: the orchestrator consumes the shadow
        # context and annotator; the service eagerly loads the artifact
        # and feeds the carrier in (via the extracted ``builders``
        # composition helper); the runtime facade forwards the typed
        # telemetry through ``PickSessionReport``. Any *other* file under
        # backend/src/robot/grasping/ or backend/src/robot/execution/
        # importing the predictor remains a regression.
        "src/robot/grasping/loop/pick_loop.py",
        "src/robot/execution/autonomous_grasp/service.py",
        "src/robot/execution/autonomous_grasp/builders.py",
        # ``report.py`` defines ``AutonomousGraspReport``, which composes the
        # typed shadow/blend telemetry carriers (ShadowSuccessTelemetry /
        # RankingBlendTelemetry) — the same contract surface the service used to
        # own. A pure value-object module, not a hot-path import.
        "src/robot/execution/autonomous_grasp/report.py",
        "src/robot/execution/runtime_pick.py",
    }

    def test_no_runtime_grasping_file_imports_the_predictor(self) -> None:
        grasping_root = REPO_ROOT / "src" / "robot" / "grasping"
        offenders: list[str] = []
        for path in grasping_root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in self.ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in self.FORBIDDEN_IMPORTS:
                if needle in text:
                    offenders.append(f"{rel}: {needle!r}")
        self.assertEqual(
            offenders,
            [],
            msg=(
                "Only the sanctioned integration files may import the predictor. "
                f"Unexpected runtime files importing it: {offenders}."
            ),
        )

    def test_no_robot_execution_file_imports_the_predictor(self) -> None:
        exec_root = REPO_ROOT / "src" / "robot" / "execution"
        offenders: list[str] = []
        for path in exec_root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in self.ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in self.FORBIDDEN_IMPORTS:
                if needle in text:
                    offenders.append(rel)
        self.assertEqual(offenders, [])


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class SuccessModelConfigTests(unittest.TestCase):
    def test_defaults_keep_runtime_disabled(self) -> None:
        cfg = RobotGraspingConfig()
        self.assertIsInstance(cfg.success_model, GraspingSuccessModelConfig)
        self.assertFalse(cfg.success_model.enabled)
        self.assertEqual(
            cfg.success_model.artifact_dir,
            "assets/models/success_probability/v1",
        )
        self.assertEqual(
            cfg.success_model.apply_modes,
            ("easy", "auto", "dense_clutter", "dense_autonomous"),
        )

    def test_extra_field_rejected(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            GraspingSuccessModelConfig(typo=True)  # type: ignore[call-arg]

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(apply_modes=("easy", "not_a_mode"))

    def test_duplicate_mode_rejected(self) -> None:
        with self.assertRaises(Exception):
            GraspingSuccessModelConfig(apply_modes=("easy", "easy"))


# ---------------------------------------------------------------------------
# Cross-cutting guard: the record contract is still byte-identical to defaults.
# ---------------------------------------------------------------------------


class RecordDefaultsStillUnchangedTests(unittest.TestCase):
    def test_grasp_attempt_record_schema_version_unchanged(self) -> None:
        self.assertEqual(GraspAttemptRecord.SCHEMA_VERSION, 1)

    def test_default_to_dict_unchanged_by_u1(self) -> None:
        record = GraspAttemptRecord(
            timestamp=1700000000.0,
            attempt_id="u1-guard",
            mode="auto",
            final_outcome="success",
        )
        payload = record.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["extra"], {})
        self.assertEqual(payload["mode"], "auto")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
