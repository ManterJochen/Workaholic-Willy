"""Dataset and replay-env contract tests.

Contracts under test:

* Both formats: JSONL is always emitted; parquet is best-effort.
* Stratified by 7-class outcome: every class with >=3
  records appears in every split.
* Deterministic replay env: both ``RecordedObservationReplayEnv``
  and ``GeometricRerunReplayEnv`` produce identical fingerprints
  across two builds.
* CLI entry-point ``python -m backend.src.robot.grasping.rl``
  exits 0 on success.
* Split ratios: default 60/20/20 ratios match.
* Leakage: defaults reject attempt_id overlap; soft warn vs
  hard reject for object_set.
* One stable SAR extractor: name=="v1_baseline", version==1,
  state keys frozen.
* Storage: manifest committed under
  ``docs/baselines/rl_datasets/<dataset_id>.json``; split files under
  gitignored ``logs/rl/datasets/<dataset_id>/splits/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl import (
    RL_TOOLING_VERSION,
)
from src.robot.grasping.rl.dataset import (
    CANONICAL_BOOTSTRAP_SOURCES,
    OUTCOME_CLASSES,
    RL_DATASET_SCHEMA_VERSION,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    build_dataset,
    canonical_record_hash,
    derive_outcome_class,
    load_jsonl,
    stratified_split_by_outcome,
    write_manifest,
)
from src.robot.grasping.rl.leakage import (
    SEVERITY_NOTE,
    SEVERITY_REJECT,
    SEVERITY_WARN,
    audit_attempt_id_overlap,
    audit_object_set_overlap,
    audit_time_bleed,
    run_leakage_audits,
)
from src.robot.grasping.rl.replay_env import (
    GeometricRerunReplayEnv,
    RecordedObservationReplayEnv,
    build_default_envs,
)
from src.robot.grasping.rl.sar import (
    ACTION_GRASP_PREFIX,
    ACTION_NOOP,
    ACTION_RECOVERY_PREFIX,
    SAR_EXTRACTOR_VERSION,
    STATE_FEATURE_KEYS,
    BaselineSARExtractor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = REPO_ROOT / "docs" / "baselines" / "rl_datasets" / "v1_bootstrap.json"


# ---------------------------------------------------------------------------
# Constants and contract values
# ---------------------------------------------------------------------------


class ContractConstantsTests(unittest.TestCase):
    def test_eight_outcome_classes(self) -> None:
        # 7 failure-taxonomy classes + unclassified.
        self.assertEqual(len(OUTCOME_CLASSES), 8)
        self.assertIn("success", OUTCOME_CLASSES)
        self.assertIn("unclassified", OUTCOME_CLASSES)

    def test_default_ratios_are_60_20_20(self) -> None:
        self.assertEqual(SPLIT_RATIOS, (0.60, 0.20, 0.20))
        self.assertAlmostEqual(sum(SPLIT_RATIOS), 1.0)

    def test_split_names_locked(self) -> None:
        self.assertEqual(SPLIT_NAMES, ("train", "val", "test"))

    def test_schema_version_is_2(self) -> None:
        self.assertEqual(RL_DATASET_SCHEMA_VERSION, 2)

    def test_canonical_sources_exist(self) -> None:
        for rel in CANONICAL_BOOTSTRAP_SOURCES:
            self.assertTrue(
                (REPO_ROOT / rel).is_file(), msg=f"canonical source missing: {rel}"
            )


# ---------------------------------------------------------------------------
# Outcome derivation
# ---------------------------------------------------------------------------


class OutcomeClassDerivationTests(unittest.TestCase):
    def test_taxonomy_class_takes_precedence(self) -> None:
        rec = {
            "final_outcome": "succeeded",
            "extra": {"failure_taxonomy_class": "slip_after_grasp"},
        }
        self.assertEqual(derive_outcome_class(rec), "slip_after_grasp")

    def test_succeeded_default(self) -> None:
        self.assertEqual(
            derive_outcome_class({"final_outcome": "succeeded", "extra": {}}),
            "success",
        )

    def test_expected_root_cause_fallback(self) -> None:
        rec = {
            "final_outcome": "execution_failed",
            "extra": {"expected_root_cause": "empty_air_grasp"},
        }
        self.assertEqual(derive_outcome_class(rec), "empty_air_grasp")

    def test_unclassified_when_no_signal(self) -> None:
        rec = {"final_outcome": "execution_failed", "extra": {}}
        self.assertEqual(derive_outcome_class(rec), "unclassified")

    def test_unknown_root_cause_token_falls_through(self) -> None:
        rec = {
            "final_outcome": "execution_failed",
            "extra": {"expected_root_cause": "something_made_up"},
        }
        self.assertEqual(derive_outcome_class(rec), "unclassified")


class TheTwoReadersOfOneRecordAgreeTests(unittest.TestCase):
    """⛔ THE DATASET AND THE TAXONOMY KEYED ON DISJOINT FIELDS. `derive_outcome_class` reads
    `extra.failure_taxonomy_class` (written by the sim runners) and `extra.expected_root_cause`
    (a label on the committed test pack). `replay.failure_taxonomy.classify_record` reads
    `extra.*_evidence` + `final_outcome` (written by the PRODUCTION serializer,
    `record_logging._stamp_taxonomy_evidence`). Neither set overlapped the other, so MEASURED
    2026-09-10 the same record was `collision_rejection` to the taxonomy report and `unclassified`
    to the dataset that stratifies on it, and the RL side's whole failure axis collapses into one
    bucket exactly for records produced by a real cell."""

    def _both(self, final_outcome: str, extra: dict) -> tuple[str, str]:
        from src.robot.grasping.replay.failure_taxonomy import classify_record
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        record = GraspAttemptRecord.new(
            attempt_id="agree-1", mode="auto", final_outcome=final_outcome, extra=extra)
        return derive_outcome_class(record.to_dict()), classify_record(record).primary.value

    def test_a_production_collision_record_is_the_same_class_to_both(self) -> None:
        dataset_class, taxonomy_class = self._both(
            "execution_failed", {"collision_evidence": True})
        self.assertEqual(taxonomy_class, "collision_rejection")
        self.assertEqual(dataset_class, taxonomy_class)

    def test_the_precedence_the_classifier_uses_survives_the_handover(self) -> None:
        """Two flags on one record resolve by enum precedence, and a second reader that resolved
        them differently would be a second taxonomy."""
        dataset_class, taxonomy_class = self._both(
            "verification_failed", {"slip_evidence": True, "empty_air_evidence": True})
        self.assertEqual(taxonomy_class, "slip_after_grasp")
        self.assertEqual(dataset_class, taxonomy_class)

    def test_an_outcome_the_classifier_will_not_name_stays_unclassified(self) -> None:
        """The classifier refuses to force-classify, and borrowing it must not smuggle in a guess."""
        dataset_class, taxonomy_class = self._both("execution_failed", {})
        self.assertEqual(taxonomy_class, "unclassified")
        self.assertEqual(dataset_class, "unclassified")

    def test_a_stray_flag_on_a_SUCCEEDED_record_is_still_a_success(self) -> None:
        """`_FAILURE_GATING_OUTCOMES` is what keeps a stray flag from relabelling a success. The
        dataset's own `succeeded` rule must stay in front of the borrowed classifier."""
        dataset_class, _ = self._both("succeeded", {"slip_evidence": True})
        self.assertEqual(dataset_class, "success")

    def test_the_committed_bootstrap_histogram_is_UNCHANGED(self) -> None:
        """⚠ THE BYTE-IDENTICAL GUARD. The three canonical sources carry no `*_evidence` flag at
        all, so borrowing the classifier must move no row of the committed manifest. Measured
        before the change and pinned here after it."""
        from collections import Counter

        repo = Path(__file__).resolve().parents[1]
        counts: Counter = Counter()
        for rel in CANONICAL_BOOTSTRAP_SOURCES:
            for line in (repo / rel).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[derive_outcome_class(json.loads(line))] += 1
        manifest = json.loads(
            (repo / "docs/baselines/rl_datasets/v1_bootstrap.json").read_text(encoding="utf-8"))
        self.assertEqual(dict(counts), manifest["class_counts"])


# ---------------------------------------------------------------------------
# Split determinism + stratification
# ---------------------------------------------------------------------------


def _records(class_counts: dict[str, int]) -> list[dict]:
    out: list[dict] = []
    for cls, n in class_counts.items():
        for i in range(n):
            attempt = f"{cls}-{i:04d}"
            if cls == "success":
                out.append(
                    {
                        "attempt_id": attempt,
                        "final_outcome": "succeeded",
                        "mode": "easy",
                        "extra": {},
                    }
                )
            else:
                out.append(
                    {
                        "attempt_id": attempt,
                        "final_outcome": "execution_failed",
                        "mode": "auto",
                        "extra": {"expected_root_cause": cls},
                    }
                )
    return out


class StratifiedSplitTests(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        records = _records({"success": 100, "slip_after_grasp": 30, "occlusion_misread": 20})
        a = stratified_split_by_outcome(records, seed=7)
        b = stratified_split_by_outcome(records, seed=7)
        self.assertEqual(
            [r["attempt_id"] for r in a["train"]],
            [r["attempt_id"] for r in b["train"]],
        )
        self.assertEqual(
            [r["attempt_id"] for r in a["test"]],
            [r["attempt_id"] for r in b["test"]],
        )

    def test_each_class_in_every_split_when_sufficient(self) -> None:
        records = _records(
            {"success": 100, "slip_after_grasp": 30, "occlusion_misread": 20}
        )
        splits = stratified_split_by_outcome(records, seed=1)
        for cls in ("success", "slip_after_grasp", "occlusion_misread"):
            for split_name in SPLIT_NAMES:
                count = sum(
                    1
                    for r in splits[split_name]
                    if derive_outcome_class(r) == cls
                )
                self.assertGreater(
                    count, 0, msg=f"{cls} missing from split {split_name}"
                )

    def test_tiny_class_routed_to_train(self) -> None:
        records = _records({"success": 100, "slip_after_grasp": 2})
        splits = stratified_split_by_outcome(records, seed=1)
        train_slip = sum(1 for r in splits["train"] if derive_outcome_class(r) == "slip_after_grasp")
        self.assertEqual(train_slip, 2)
        for split_name in ("val", "test"):
            self.assertEqual(
                sum(1 for r in splits[split_name] if derive_outcome_class(r) == "slip_after_grasp"),
                0,
            )

    def test_no_attempt_id_overlap_between_splits(self) -> None:
        records = _records({"success": 100, "slip_after_grasp": 30, "occlusion_misread": 20})
        splits = stratified_split_by_outcome(records, seed=1)
        ids = {name: {r["attempt_id"] for r in recs} for name, recs in splits.items()}
        self.assertEqual(ids["train"] & ids["val"], set())
        self.assertEqual(ids["train"] & ids["test"], set())
        self.assertEqual(ids["val"] & ids["test"], set())

    def test_record_hash_stable(self) -> None:
        rec = {"a": 1, "b": [2, 3], "c": {"x": "y"}}
        self.assertEqual(canonical_record_hash(rec), canonical_record_hash(dict(reversed(list(rec.items())))))


# ---------------------------------------------------------------------------
# Build dataset (end to end)
# ---------------------------------------------------------------------------


class BuildDatasetTests(unittest.TestCase):
    def test_build_default_bootstrap_against_temp_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            splits_root = Path(td) / "splits"
            manifest = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="unit_test_v1",
                seed=1,
                splits_root=splits_root,
                emit_parquet=False,
            )
            self.assertEqual(manifest.schema_version, 2)
            # The manifest stamps the tooling version that built it.
            self.assertEqual(manifest.rl_tooling_version, RL_TOOLING_VERSION)
            self.assertEqual(manifest.seed, 1)
            self.assertEqual(manifest.ratios, SPLIT_RATIOS)
            self.assertEqual(
                set(manifest.split_counts.keys()), set(SPLIT_NAMES)
            )
            self.assertEqual(
                sum(manifest.split_counts.values()), manifest.record_count
            )
            self.assertFalse(manifest.parquet_emitted)
            for split in SPLIT_NAMES:
                self.assertTrue((splits_root / f"{split}.jsonl").is_file())

    def test_build_is_deterministic_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r1 = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="det_a",
                seed=42,
                splits_root=Path(td) / "a",
                emit_parquet=False,
            )
            r2 = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="det_b",
                seed=42,
                splits_root=Path(td) / "b",
                emit_parquet=False,
            )
            self.assertEqual(r1.split_hashes, r2.split_hashes)
            self.assertEqual(r1.split_counts, r2.split_counts)
            self.assertEqual(r1.class_counts, r2.class_counts)

    def test_different_seed_changes_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r1 = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="seed_1",
                seed=1,
                splits_root=Path(td) / "a",
                emit_parquet=False,
            )
            r2 = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="seed_2",
                seed=2,
                splits_root=Path(td) / "b",
                emit_parquet=False,
            )
            self.assertNotEqual(r1.split_hashes, r2.split_hashes)

    def test_parquet_soft_optional(self) -> None:
        # When pyarrow is unavailable, manifest records the
        # skip reason and JSONL is still emitted.
        with tempfile.TemporaryDirectory() as td:
            splits_root = Path(td) / "splits"
            manifest = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="parquet_check",
                seed=1,
                splits_root=splits_root,
                emit_parquet=True,
            )
            try:
                import pyarrow  # type: ignore[import-not-found]  # noqa: F401
                pyarrow_present = True
            except Exception:
                pyarrow_present = False
            if pyarrow_present:
                self.assertTrue(manifest.parquet_emitted)
            else:
                self.assertFalse(manifest.parquet_emitted)
                self.assertIsNotNone(manifest.parquet_skip_reason)
            for split in SPLIT_NAMES:
                self.assertTrue((splits_root / f"{split}.jsonl").is_file())


# ---------------------------------------------------------------------------
# Leakage audits
# ---------------------------------------------------------------------------


class LeakageTests(unittest.TestCase):
    def test_attempt_id_overlap_rejects(self) -> None:
        a = [{"attempt_id": "x"}, {"attempt_id": "y"}]
        b = [{"attempt_id": "y"}, {"attempt_id": "z"}]
        findings = audit_attempt_id_overlap({"train": a, "test": b})
        self.assertTrue(any(f.severity == SEVERITY_REJECT for f in findings))

    def test_object_set_soft_then_hard(self) -> None:
        # 12% of test records have an object_set seen in train → hard reject.
        train = [{"extra": {"object_set": ["bolt_a", "nut_b"]}}]
        test = [
            {"extra": {"object_set": ["bolt_a", "nut_b"]}}
            for _ in range(12)
        ] + [
            {"extra": {"object_set": ["foreign_x"]}}
            for _ in range(88)
        ]
        findings = audit_object_set_overlap({"train": train, "test": test})
        self.assertTrue(any(f.severity == SEVERITY_REJECT for f in findings))

        # 7% overlap → soft warn (not reject).
        test2 = [
            {"extra": {"object_set": ["bolt_a", "nut_b"]}}
            for _ in range(7)
        ] + [
            {"extra": {"object_set": ["foreign_x"]}}
            for _ in range(93)
        ]
        findings2 = audit_object_set_overlap({"train": train, "test": test2})
        self.assertTrue(any(f.severity == SEVERITY_WARN for f in findings2))
        self.assertFalse(any(f.severity == SEVERITY_REJECT for f in findings2))

    def test_synthetic_timestamps_downgrade_to_note(self) -> None:
        train = [{"extra": {"timestamp": float(i), "attempt_wall_time_s": 0.1}} for i in range(50)]
        test = [{"extra": {"timestamp": float(i), "attempt_wall_time_s": 0.1}} for i in range(50)]
        findings = audit_time_bleed({"train": train, "test": test})
        self.assertTrue(all(f.severity == SEVERITY_NOTE for f in findings))

    def test_real_time_bleed_rejects(self) -> None:
        # Wall-clock-style timestamps with overlapping ranges → reject.
        train = [
            {"extra": {"timestamp": 1_700_000_000.0 + i * 0.001, "attempt_wall_time_s": 0.001}}
            for i in range(20)
        ]
        test = [
            {"extra": {"timestamp": 1_700_000_000.0 + i * 0.001, "attempt_wall_time_s": 0.001}}
            for i in range(20)
        ]
        findings = audit_time_bleed({"train": train, "test": test})
        self.assertTrue(any(f.severity == SEVERITY_REJECT for f in findings))

    def test_run_leakage_audits_strict_promotes_warn_to_failure(self) -> None:
        train = [{"attempt_id": "a", "extra": {"object_set": ["x"]}}]
        test = [
            {"attempt_id": f"t{i}", "extra": {"object_set": ["x"]}}
            for i in range(7)
        ] + [
            {"attempt_id": f"t{i}", "extra": {"object_set": ["y"]}}
            for i in range(93)
        ]
        report_lenient = run_leakage_audits({"train": train, "val": [], "test": test}, strict=False)
        report_strict = run_leakage_audits({"train": train, "val": [], "test": test}, strict=True)
        self.assertTrue(report_lenient.passed)
        self.assertFalse(report_strict.passed)

    def test_run_leakage_audits_strict_fails_on_skipped_audits(self) -> None:
        # Records carry only attempt_id, so the scene / object_set / time-bleed audits all SKIP
        # (NOTE) and only attempt_id_overlap runs. No warns, no rejects. Lenient passes; strict must FAIL
        # rather than green-light a VACUOUS audit (3/4 checks silently could not run).
        train = [{"attempt_id": "a0"}]
        test = [{"attempt_id": f"t{i}"} for i in range(5)]
        splits = {"train": train, "val": [], "test": test}
        report_lenient = run_leakage_audits(splits, strict=False)
        report_strict = run_leakage_audits(splits, strict=True)
        self.assertTrue(report_lenient.passed)
        self.assertFalse(report_strict.passed)
        self.assertTrue(report_strict.has_notes)  # the SKIPS drove the strict failure
        self.assertFalse(report_strict.has_warns)  # not a warn -- a genuine skip


# ---------------------------------------------------------------------------
# SAR extractor
# ---------------------------------------------------------------------------


class SARExtractorTests(unittest.TestCase):
    def test_baseline_name_and_version_locked(self) -> None:
        ex = BaselineSARExtractor()
        self.assertEqual(ex.name, "v1_baseline")
        self.assertEqual(ex.version, SAR_EXTRACTOR_VERSION)
        self.assertEqual(ex.version, 1)

    def test_state_feature_keys_are_frozen(self) -> None:
        # Reordering or removing keys would break replay determinism.
        self.assertEqual(STATE_FEATURE_KEYS[0], "uncertainty_score")
        self.assertIn("fused_view_count", STATE_FEATURE_KEYS)
        self.assertEqual(len(STATE_FEATURE_KEYS), 12)

    def test_success_reward_is_one(self) -> None:
        ex = BaselineSARExtractor()
        sar = ex.extract({"attempt_id": "a", "mode": "easy", "final_outcome": "succeeded"})
        self.assertEqual(sar.reward, 1.0)
        self.assertEqual(set(sar.state.keys()), set(STATE_FEATURE_KEYS))

    def test_failure_reward_is_zero(self) -> None:
        ex = BaselineSARExtractor()
        sar = ex.extract(
            {
                "attempt_id": "b",
                "mode": "auto",
                "final_outcome": "execution_failed",
                "extra": {"uncertainty_score": 0.5},
            }
        )
        self.assertEqual(sar.reward, 0.0)
        self.assertEqual(sar.state["uncertainty_score"], 0.5)

    def test_action_token_grasp_then_recovery_then_noop(self) -> None:
        ex = BaselineSARExtractor()
        grasp = ex.extract(
            {
                "attempt_id": "g",
                "mode": "easy",
                "final_outcome": "succeeded",
                "selected_grasp": {"x": 1.0, "y": 2.0},
            }
        )
        self.assertTrue(grasp.action.startswith(f"{ACTION_GRASP_PREFIX}:"))
        recovery = ex.extract(
            {
                "attempt_id": "r",
                "mode": "auto",
                "final_outcome": "execution_failed",
                "recovery_actions": [{"kind": "reobserve"}],
            }
        )
        self.assertEqual(recovery.action, f"{ACTION_RECOVERY_PREFIX}:reobserve")
        none = ex.extract({"attempt_id": "n", "mode": "easy", "final_outcome": "succeeded"})
        self.assertEqual(none.action, ACTION_NOOP)

    def test_missing_attempt_id_raises(self) -> None:
        ex = BaselineSARExtractor()
        with self.assertRaises(ValueError):
            ex.extract({"mode": "easy"})


# ---------------------------------------------------------------------------
# Replay env determinism
# ---------------------------------------------------------------------------


class ReplayEnvTests(unittest.TestCase):
    def test_recorded_and_geometric_envs_construct(self) -> None:
        rec, geo = build_default_envs()
        self.assertIsInstance(rec, RecordedObservationReplayEnv)
        self.assertIsInstance(geo, GeometricRerunReplayEnv)
        self.assertNotEqual(rec.name, geo.name)

    def test_fingerprint_is_deterministic_across_two_runs(self) -> None:
        records = load_jsonl(REPO_ROOT / CANONICAL_BOOTSTRAP_SOURCES[0])[:50]
        rec1, geo1 = build_default_envs()
        rec2, geo2 = build_default_envs()
        self.assertEqual(rec1.fingerprint(records), rec2.fingerprint(records))
        self.assertEqual(geo1.fingerprint(records), geo2.fingerprint(records))

    def test_envs_agree_on_fingerprint(self) -> None:
        # Record-replay and geometric-rerun share the same
        # baseline extractor, so their fingerprints must agree.
        # When the recovery layer overrides geometric_rerun this test
        # will be updated to assert distinct fingerprints under specific
        # transforms.
        records = load_jsonl(REPO_ROOT / CANONICAL_BOOTSTRAP_SOURCES[0])[:50]
        rec, geo = build_default_envs()
        self.assertEqual(rec.fingerprint(records), geo.fingerprint(records))


# ---------------------------------------------------------------------------
# Committed manifest
# ---------------------------------------------------------------------------


class CommittedManifestTests(unittest.TestCase):
    def test_committed_manifest_exists(self) -> None:
        self.assertTrue(
            COMMITTED_MANIFEST.is_file(),
            msg=(
                f"manifest missing at {COMMITTED_MANIFEST}. Run: "
                "python -m src.robot.grasping.rl build-dataset "
                "--dataset-id v1_bootstrap"
            ),
        )

    def test_committed_manifest_matches_rebuild(self) -> None:
        on_disk = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            manifest = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="v1_bootstrap",
                seed=on_disk["seed"],
                ratios=tuple(on_disk["ratios"]),
                splits_root=Path(td) / "splits",
                emit_parquet=False,
            )
            from src.robot.grasping.rl.leakage import run_leakage_audits as _audit
            splits_for_audit = {
                name: load_jsonl(Path(td) / "splits" / f"{name}.jsonl")
                for name in SPLIT_NAMES
            }
            leakage = _audit(splits_for_audit, strict=False)
            rebuilt = manifest.to_json()
            rebuilt["leakage"] = leakage.to_json()

        # Compare structurally; split_files paths differ between the
        # committed (logs/...) build and the temp rebuild, so we drop
        # them and compare hashes/counts/leakage.
        for k in ("split_files",):
            on_disk.pop(k, None)
            rebuilt.pop(k, None)
        self.assertEqual(on_disk["split_hashes"], rebuilt["split_hashes"], msg="committed manifest is stale")
        self.assertEqual(on_disk["split_counts"], rebuilt["split_counts"])
        self.assertEqual(on_disk["class_counts"], rebuilt["class_counts"])
        self.assertEqual(on_disk["sources"], rebuilt["sources"])
        self.assertEqual(on_disk["leakage"]["passed"], rebuilt["leakage"]["passed"])

    def test_committed_manifest_passes_leakage(self) -> None:
        payload = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
        self.assertTrue(payload["leakage"]["passed"], msg=payload["leakage"]["findings"])


# ---------------------------------------------------------------------------
# CLI subprocess
# ---------------------------------------------------------------------------


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The committed manifest records its split files under the GITIGNORED
        # logs/rl/datasets/v1_bootstrap/splits/ (generated, not committed — see
        # .gitignore). On a fresh checkout (CI / a new workstation) they are absent, so
        # the subprocess CLIs below otherwise fail with FileNotFoundError. Materialize
        # them once to the canonical location (idempotent), exactly as
        # `python -m backend.src.robot.grasping.rl build-dataset --dataset-id v1_bootstrap`.
        splits_root = REPO_ROOT / "logs" / "rl" / "datasets" / "v1_bootstrap" / "splits"
        if all((splits_root / f"{name}.jsonl").is_file() for name in SPLIT_NAMES):
            return
        payload = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
        build_dataset(
            repo_root=REPO_ROOT,
            dataset_id="v1_bootstrap",
            seed=payload["seed"],
            ratios=tuple(payload["ratios"]),
            splits_root=splits_root,
            emit_parquet=False,
        )

    def test_replay_env_check_subprocess(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.robot.grasping.rl",
                "replay-env-check",
                "--manifest",
                "docs/baselines/rl_datasets/v1_bootstrap.json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        for split in SPLIT_NAMES:
            self.assertIn("recorded_observation_fingerprint", payload[split])
            self.assertIn("geometric_rerun_fingerprint", payload[split])

    def test_audit_leakage_subprocess(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.robot.grasping.rl",
                "audit-leakage",
                "--manifest",
                "docs/baselines/rl_datasets/v1_bootstrap.json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["passed"])


# ---------------------------------------------------------------------------
# Manifest write helper
# ---------------------------------------------------------------------------


class WriteManifestTests(unittest.TestCase):
    def test_write_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest = build_dataset(
                repo_root=REPO_ROOT,
                dataset_id="rt",
                seed=1,
                splits_root=Path(td) / "s",
                emit_parquet=False,
            )
            out = Path(td) / "manifest.json"
            digest = write_manifest(manifest, out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_id"], "rt")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(len(digest), 64)


class GroupAwareSplitTests(unittest.TestCase):
    """group_aware_split_by_family keeps every scene_family_id entirely in ONE split (no object
    identity leaks train<->test) so the object_set audit passes rigidly + ranking pairs stay within a split."""

    @staticmethod
    def _recs(families: dict[str, list[str]]) -> list[dict]:
        recs: list[dict] = []
        i = 0
        for fam, outcomes in families.items():
            for oc in outcomes:
                recs.append({"attempt_id": f"a{i}", "final_outcome": oc,
                             "extra": {"scene_family_id": fam}})
                i += 1
        return recs

    def test_families_never_span_splits(self) -> None:
        from src.robot.grasping.rl.dataset import group_aware_split_by_family

        recs = self._recs({
            "ycb:sugar_box": ["succeeded"] * 30 + ["execution_failed"] * 20,
            "cube:red_cube": ["succeeded"] * 40,
            "ycb:pudding_box": ["succeeded"] * 30,
            "ycb:gelatin_box": ["succeeded"] * 20,
        })
        splits = group_aware_split_by_family(recs, seed=1)
        fam_to_splits: dict[str, set[str]] = {}
        for sname, srecs in splits.items():
            for r in srecs:
                fam_to_splits.setdefault(r["extra"]["scene_family_id"], set()).add(sname)
        for fam, sset in fam_to_splits.items():
            self.assertEqual(len(sset), 1, f"family {fam} leaked across splits {sset}")
        self.assertEqual(sum(len(v) for v in splits.values()), 140)  # all records preserved
        self.assertEqual(len(fam_to_splits), 4)

    def test_deterministic(self) -> None:
        from src.robot.grasping.rl.dataset import group_aware_split_by_family

        recs = self._recs({f"fam{i}": ["succeeded"] * 10 for i in range(6)})
        a = group_aware_split_by_family(recs, seed=3)
        b = group_aware_split_by_family(recs, seed=3)
        self.assertEqual({k: [r["attempt_id"] for r in v] for k, v in a.items()},
                         {k: [r["attempt_id"] for r in v] for k, v in b.items()})

    def test_fallback_singleton_without_field(self) -> None:
        from src.robot.grasping.rl.dataset import group_aware_split_by_family

        recs = [{"attempt_id": f"a{i}", "final_outcome": "succeeded", "extra": {}} for i in range(12)]
        splits = group_aware_split_by_family(recs, seed=1)  # no scene_family_id -> per-record singletons
        self.assertEqual(sum(len(v) for v in splits.values()), 12)
        self.assertTrue(all(len(v) > 0 for v in splits.values()))  # 60/20/20 non-degenerate


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
