"""R3 import-surface guard — asserts every rl/ module imports cleanly AND that each R3-relocated private stays
re-exported (as the SAME object) from its original module.

The R3 refactor moves shared/duplicated logic into new leaf modules behind a stable re-exporting facade. The
by-name import sites in the RL stack + the per-phase tests (e.g. `from .candidate_policy import _project_feature`)
must keep resolving. This test fails the moment an R3 facade move drops or copies a re-export instead of
re-exporting the moved symbol. It GROWS with each R3 step (R3.1a features → R3.1b artifact_io → R3.3/R3.4 …).
"""

from __future__ import annotations

import importlib
import unittest

_RL = "src.robot.grasping.rl"

# Every rl/ module must import without error (catches a broken move / circular import introduced by a facade).
_RL_MODULES = (
    "_features", "_artifact_io", "_linucb", "_router_telemetry", "_io",
    "_promotion_extractors", "_promotion_estimators",
    "_cli_common", "_commands_dataset", "_commands_train", "_commands_eval", "_commands_hardening",
    "candidate_policy", "ranking_policy", "sequencing_policy",
    "perception_budget_policy", "recovery_policy", "router", "promotion", "ope",
    "dataset", "leakage", "train_candidate", "train_ranking", "train_sequencing",
    "train_perception_budget", "train_recovery", "honesty", "action_mask",
)


class ImportSurfaceTests(unittest.TestCase):
    def test_every_rl_module_imports_clean(self) -> None:
        importlib.import_module(_RL)  # the package itself
        for mod in _RL_MODULES:
            with self.subTest(module=mod):
                importlib.import_module(f"{_RL}.{mod}")

    def test_r3_1a_features_reexport(self) -> None:
        # R3.1a: _project_feature/_sigmoid live in ._features and are re-exported (SAME object) from
        # candidate_policy — ranking_policy + train_ranking import them by-name from candidate_policy.
        from src.robot.grasping.rl._features import _project_feature, _sigmoid
        from src.robot.grasping.rl.candidate_policy import (
            _project_feature as cp_project_feature,
        )
        from src.robot.grasping.rl.candidate_policy import (
            _sigmoid as cp_sigmoid,
        )
        self.assertIs(_project_feature, cp_project_feature, "candidate_policy no longer re-exports _features._project_feature")
        self.assertIs(_sigmoid, cp_sigmoid, "candidate_policy no longer re-exports _features._sigmoid")

    def test_r3_1b_hash_artifact_reexport(self) -> None:
        # R3.1b: the chunked-sha256 file hash lives in ._artifact_io and is re-exported (SAME object) under each
        # family's committed hash_<family>_artifact name (recovery gained one additively).
        from src.robot.grasping.rl._artifact_io import hash_artifact
        from src.robot.grasping.rl.candidate_policy import hash_logistic_artifact
        from src.robot.grasping.rl.perception_budget_policy import hash_perception_budget_artifact
        from src.robot.grasping.rl.ranking_policy import hash_ranking_artifact
        from src.robot.grasping.rl.recovery_policy import hash_recovery_artifact
        from src.robot.grasping.rl.sequencing_policy import hash_sequencing_artifact
        for name, fn in {
            "hash_logistic_artifact": hash_logistic_artifact,
            "hash_ranking_artifact": hash_ranking_artifact,
            "hash_sequencing_artifact": hash_sequencing_artifact,
            "hash_perception_budget_artifact": hash_perception_budget_artifact,
            "hash_recovery_artifact": hash_recovery_artifact,
        }.items():
            with self.subTest(alias=name):
                self.assertIs(fn, hash_artifact, f"{name} is not the shared _artifact_io.hash_artifact")

    def test_r3_3a_router_telemetry_reexport(self) -> None:
        # R3.3a: the pure telemetry/extras projection helpers live in ._router_telemetry and are re-exported
        # (SAME object) from router (shadow.py + the per-phase tests import emit_* / build_candidate_breakdown
        # + BREAKDOWN_TOP_N / KENDALL_TAU_BOUNDS by-name from router).
        from src.robot.grasping.rl import _router_telemetry as rt
        from src.robot.grasping.rl import router
        for name in (
            "BREAKDOWN_TOP_N", "KENDALL_TAU_BOUNDS", "build_candidate_breakdown",
            "emit_candidate_shadow_extras", "emit_ranking_extras", "emit_sequencing_extras",
            "emit_perception_extras", "emit_recovery_extras",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(router, name), getattr(rt, name), f"router.{name} is not the _router_telemetry one")

    def test_r3_3b_promotion_reexport(self) -> None:
        # R3.3b: the OPE extractors + estimators live in ._promotion_extractors / ._promotion_estimators and are
        # re-exported (SAME object) from promotion (the orchestration + the __all__ + test_v7's _per_record_weight).
        from src.robot.grasping.rl import _promotion_estimators as est
        from src.robot.grasping.rl import _promotion_extractors as ext
        from src.robot.grasping.rl import promotion
        for name in ("EvaluationTriple", "TripleExtractionResult", "extract_sequencing_triples", "extract_perception_triples", "extract_recovery_triples"):
            with self.subTest(name=name):
                self.assertIs(getattr(promotion, name), getattr(ext, name), f"promotion.{name} is not the _promotion_extractors one")
        for name in ("WISLiftEstimate", "DMEstimate", "IndependentDMEstimate", "_per_record_weight", "compute_wis_lift", "compute_dm", "fit_tabular_dm"):
            with self.subTest(name=name):
                self.assertIs(getattr(promotion, name), getattr(est, name), f"promotion.{name} is not the _promotion_estimators one")

    def test_r3_4a_io_and_hash_reexport(self) -> None:
        # R3.4a: the fail-closed JSONL loader is shared (dataset.load_jsonl re-exports _io.load_jsonl), and
        # promotion.hash_file is now the shared _artifact_io.hash_artifact (the 6th file-hash copy, deduped).
        from src.robot.grasping.rl._artifact_io import hash_artifact
        from src.robot.grasping.rl._io import load_jsonl
        from src.robot.grasping.rl import promotion
        from src.robot.grasping.rl.dataset import load_jsonl as dataset_load_jsonl
        self.assertIs(dataset_load_jsonl, load_jsonl, "dataset.load_jsonl is not the shared _io.load_jsonl")
        self.assertIs(promotion.hash_file, hash_artifact, "promotion.hash_file is not the shared _artifact_io.hash_artifact")


if __name__ == "__main__":
    unittest.main()
