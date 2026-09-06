"""Top-level pytest configuration for the Workaholic-Willy suite.

Platform-gates the **byte-exact determinism / artifact-SHA tests**. These assert that
regenerated canonical replay packs, trained model artifacts, and promotion reports
reproduce the *committed* bytes (or their SHA-256) exactly. That identity is
**floating-point- and platform-sensitive**: the committed goldens are rendered on one
origin platform (the macOS dev laptop / the canonical determinism CI job), and on a
different BLAS/libm (e.g. the Windows RTX workstation) the values drift in the last ULP,
so the serialized bytes differ even though the computation is correct.

They are therefore **skipped by default** and run only when ``WILLY_DETERMINISM_NATIVE=1``
is set — do that on the artifact-origin platform / the canonical determinism job (where the
goldens are (re)generated and committed), NOT on developer or cross-platform boxes. This
keeps ``pytest tests`` honestly green everywhere while preserving the byte-identity gate
where it is meaningful.

Scope is an **explicit allow-list of node IDs** (below) so nothing else is ever skipped.
If you add a new byte-identity / SHA artifact test, add its node ID here. (The integration
``isaac`` marker is gated separately in ``tests/integration/conftest.py``.)
"""

from __future__ import annotations

import os

import pytest

# Exact byte-identity / artifact-SHA tests — locked to the artifact-origin platform.
# Empirically the set that fails purely on cross-platform float drift (verified on the
# Windows RTX 5080 box, 2026-06-04). NOT included on purpose: the two
# ``test_dataset_and_replay_env …_subprocess`` tests (they fail on a *missing generated dataset*, a separate
# fixture gap, not determinism) and ``test_log_cfg_threadsafe`` (Windows file-lock teardown).
# ⛔ FIVE OF THESE NAMED FILES THAT DO NOT EXIST, AND THE GUARD WAS THEREFORE INERT FOR THEM.
# MEASURED 2026-09-05: `tests/test_u1_success_probability_model.py` and
# `tests/test_u3_model_promotion_gate.py` were renamed to `test_success_probability_model.py` and
# `test_model_promotion_gate.py`; `pytest_collection_modifyitems` matches on the EXACT nodeid, so the
# committed-promotion drift test and the three verify/canary goldens ran unguarded on every box --
# the precise cross-platform ULP exposure this file exists to prevent. They were green here only
# because this box happens to agree. Fixed by renaming the prefixes; every class::test name below
# still resolves, checked one by one.
#
# ⚠ A stale entry is silent in BOTH directions, which is why `test_every_locked_nodeid_exists`
# in tests/test_determinism_lock.py now fails on one rather than leaving it to be noticed.
_DETERMINISM_NATIVE_NODEIDS = frozenset(
    {
        "tests/test_canonical_determinism.py::CanonicalPackDeterminismTests::test_each_pack_regenerates_to_committed_bytes",
        "tests/test_canonical_determinism.py::CanonicalPackDeterminismTests::test_manifest_matches_on_disk_packs",
        "tests/test_canonical_determinism.py::CanonicalPackDeterminismTests::test_manifest_sha256_matches_pack_bytes",
        "tests/test_u0_telemetry_contract.py::BaselineReportTests::test_report_committed_on_disk",
        "tests/test_u0_telemetry_contract.py::CanonicalPacksTests::test_manifest_matches_packs_on_disk",
        "tests/test_u0_telemetry_contract.py::CanonicalPacksTests::test_pack_bytes_match_spec_render",
        "tests/test_u0_telemetry_contract.py::RegenerateIdempotenceTests::test_regenerate_all_is_byte_idempotent",
        "tests/test_success_probability_model.py::TrainerByteDeterminismTests::test_committed_artifact_matches_fresh_train",
        "tests/test_model_promotion_gate.py::CommittedPromotionDriftTests::test_metrics_match_recorded",
        "tests/test_model_promotion_gate.py::LoaderEnforcementTests::test_canary_loads_with_valid_promotion",
        "tests/test_model_promotion_gate.py::PromotionCLITests::test_verify_subcommand_returns_zero_on_committed",
        "tests/test_model_promotion_gate.py::VerifyPromotionTests::test_passes_on_committed_artifact",
        "tests/test_u7_failure_taxonomy.py::LabeledPackTests::test_manifest_sha_matches_disk",
        "tests/test_dataset_and_replay_env.py::CommittedManifestTests::test_committed_manifest_matches_rebuild",
        "tests/test_shadow_router_and_candidate_policy.py::TrainCandidatePolicyCLITests::test_cli_reproduces_committed_artifact",
        "tests/test_ranking_shadow_and_pairwise_logistic.py::CommittedArtifactSha256Tests::test_cli_regenerates_byte_identical_artifact",
        "tests/test_ranking_shadow_and_pairwise_logistic.py::CommittedArtifactSha256Tests::test_committed_sha256_locked",
        "tests/test_sequencing_shadow_and_lookup.py::CommittedArtifactSha256Tests::test_committed_artifact_byte_identity_via_cli",
        "tests/test_ope_harness.py::CommittedArtifactTests::test_committed_report_byte_identity",
        "tests/test_perception_budget_policy.py::CommittedArtifactSha256Tests::test_committed_artifact_matches",
        "tests/test_recovery_policy.py::CommittedArtifactTests::test_committed_artifact_reproduces_from_packs",
        "tests/test_recovery_policy.py::CommittedArtifactTests::test_committed_artifact_sha256",
        "tests/test_promotion_pipeline.py::CommittedPromotionReportTests::test_committed_report_is_reproducible",
        "tests/test_promotion_pipeline.py::CommittedPromotionReportTests::test_committed_report_sha256_matches",
        "tests/test_promotion_pipeline.py::CommittedSequencingPromotionReportTests::test_committed_report_is_reproducible",
        "tests/test_promotion_pipeline.py::CommittedSequencingPromotionReportTests::test_committed_report_sha256_matches",
        "tests/test_u12_docs_and_soak_gate.py::SoakReportCLITests::test_committed_report_is_regen_stable",
    }
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the platform-locked byte-identity tests unless explicitly opted in."""

    if os.environ.get("WILLY_DETERMINISM_NATIVE"):
        return
    skip_determinism = pytest.mark.skip(
        reason="byte-exact determinism gate is platform-locked (float/BLAS-sensitive); "
        "set WILLY_DETERMINISM_NATIVE=1 on the artifact-origin platform / canonical CI "
        "job to run it. See tests/conftest.py."
    )
    for item in items:
        if item.nodeid.replace("\\", "/") in _DETERMINISM_NATIVE_NODEIDS:
            item.add_marker(skip_determinism)
