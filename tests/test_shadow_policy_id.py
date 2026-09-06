"""Two kinds of identifier met under one key name, and the correct config lost.

⛔⛔ **MEASURED 2026-09-05: `rl_shadow` built NO router on the only shipped profile that enables it.**

    RLShadowWiring - WARNING - rl.policy_id mismatch: config declares
    'v2_candidate_baseline_v1' but the artifact is 'v2_candidate_baseline_v1@1'
    -- running with NO shadow router

The artifact stores ``policy_name`` and ``policy_version`` separately;
``LogisticCandidatePolicy.policy_id`` composes them as ``name@version``. The YAML declares the NAME,
which is also the artifact's own filename. The comparison was ``!=`` on the raw strings, so the
operator was told that THEIR declaration was wrong.

⚠ ``rl_schema.py:118`` makes ``policy_id`` MANDATORY for every RL-active mode, so an operator has to
write something, and the only value a person would reach for was the failing one.

⚠ **AND THE GUARD IS CORRECT WHERE IT CAME FROM.** ``canary_router.py:254`` compares a promotion
report against its artifact, and there both sides are machine-written -- ``promotion.py:537`` writes
``report.policy_id`` straight out of ``policy.policy_id``. The copy was dragged across the one
boundary that breaks it: the single place where one side is typed by a human.

Found by the parallel session while chasing an occupancy-sweep regression -- 98 candidate rows with a
router on 2026-08-19, three runs of 0 and 0 on 2026-09-04. The sweep was doing the right thing; it
just never got a router.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.shadow import maybe_build_shadow_router

_REPO = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPO / "docs" / "baselines" / "rl_policies" / "v2_candidate_baseline_v1.json"


def _rl(policy_id: str, artifact: str | Path = _ARTIFACT) -> SimpleNamespace:
    return SimpleNamespace(rl=SimpleNamespace(
        mode="rl_shadow", policy_id=policy_id, artifact_path=str(artifact),
    ))


class TheShippedProfileTests(unittest.TestCase):
    def test_the_artifact_and_the_yaml_carry_two_different_kinds_of_identifier(self) -> None:
        """⛔ THE ROOT, AS DATA. The file stores a name and a version in separate keys; the property
        composes them. The YAML has always declared the name."""
        payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(payload["policy_name"], "v2_candidate_baseline_v1")
        self.assertEqual(payload["policy_version"], 1)
        self.assertNotIn("policy_id", payload, "the composed form is not stored, it is derived")

        from src.robot.grasping.rl.candidate_policy import load_logistic_candidate_policy

        self.assertEqual(load_logistic_candidate_policy(_ARTIFACT).policy_id,
                         "v2_candidate_baseline_v1@1")

    def test_the_shipped_declaration_now_builds_a_router(self) -> None:
        """⭐ THE REGRESSION THAT WAS LIVE. This is the exact value in
        `backend/config/data/robot/robot.rl_datagen.yaml:108`, and before the fix it produced None."""
        declared = "v2_candidate_baseline_v1"
        yaml = (_REPO / "config" / "data" / "robot" / "robot.rl_datagen.yaml")
        self.assertIn(f'policy_id: "{declared}"', yaml.read_text(encoding="utf-8"),
                      "the shipped value moved; this test is asserting about the wrong string")

        router = maybe_build_shadow_router(_rl(declared))
        self.assertIsNotNone(router, "the shipped profile must wire a shadow router")

    def test_the_comparison_that_shipped_would_still_refuse_this(self) -> None:
        """⭑ THE SELF-FAILING CONTROL, and this file needed one: every test here passed on its
        first run, which is exactly when a guard deserves suspicion. So the defect is stated as an
        EXPRESSION over today's shipped values rather than as a memory of what the code used to say.
        `declared != actual` -- the comparison that shipped -- is still true, and would still refuse.
        """
        from src.robot.grasping.rl.candidate_policy import load_logistic_candidate_policy

        declared = "v2_candidate_baseline_v1"
        actual = load_logistic_candidate_policy(_ARTIFACT).policy_id
        self.assertNotEqual(declared, actual, "the old raw comparison refuses the shipped config")
        self.assertEqual(declared, actual.split("@", 1)[0], "the new reading accepts it")

    def test_the_full_shipped_profile_wires_end_to_end(self) -> None:
        """The control on the test above: a hand-built namespace could agree while the real config
        chain disagrees."""
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"WILLY_PROFILE": "rl_datagen"}):
            from backend.config.loader import load_config

            cfg = load_config()
        self.assertEqual(cfg.robot.rl.mode, "rl_shadow")
        self.assertIsNotNone(maybe_build_shadow_router(cfg.robot))


class TheGuardKeepsItsStrengthTests(unittest.TestCase):
    """⭐ THE HALF THAT MUST NOT HAVE BEEN WEAKENED. The paragraph in `shadow.py` argues that a
    swapped artifact must be refused because shadow annotations become training data. Every test here
    is that claim, still true."""

    def test_a_swapped_artifact_is_still_refused(self) -> None:
        self.assertIsNone(
            maybe_build_shadow_router(_rl("v6_recovery_baseline_v1")),
            "declaring one policy and loading another must still refuse",
        )

    def test_a_prefix_is_not_a_match(self) -> None:
        """⛔ THE REASON THIS IS `split("@", 1)[0]` AND NOT `startswith`. A declaration of
        `v2_candidate` naming an artifact called `v2_candidate_baseline_v1` is a DIFFERENT policy
        whose name happens to begin the same way, and a prefix test would have accepted it."""
        self.assertIsNone(maybe_build_shadow_router(_rl("v2_candidate")))
        self.assertIsNone(maybe_build_shadow_router(_rl("v2_candidate_baseline")))

    def test_a_declared_version_that_disagrees_is_refused(self) -> None:
        """⭐ THE POINT OF KEEPING THE EXACT FORM: an operator who writes `@2` is making a claim about
        the version, and a version bump under them must fail."""
        self.assertIsNone(maybe_build_shadow_router(_rl("v2_candidate_baseline_v1@2")))

    def test_a_declared_version_that_agrees_is_accepted(self) -> None:
        self.assertIsNotNone(maybe_build_shadow_router(_rl("v2_candidate_baseline_v1@1")))

    def test_declaring_nothing_stays_allowed(self) -> None:
        """`policy_id` is optional on the schema for non-RL modes, and an empty declaration is not a
        claim to contradict. Unchanged behaviour, asserted so the fix did not quietly require one."""
        self.assertIsNotNone(maybe_build_shadow_router(_rl("")))


class TheLogSaysWhichReadingAppliedTests(unittest.TestCase):
    def test_a_name_only_match_is_announced_and_an_exact_one_is_not(self) -> None:
        """⚠ "accepted" and "accepted loosely" are different facts about a run whose annotations
        later become training data, so the looser one leaves a line. INFO rather than WARNING: the run
        is admissible, the operator simply pinned less than they could have."""
        with self.assertLogs("RLShadowWiring", level="INFO") as loose:
            maybe_build_shadow_router(_rl("v2_candidate_baseline_v1"))
        text = "\n".join(loose.output)
        self.assertIn("by NAME", text)
        self.assertIn("v2_candidate_baseline_v1@1", text, "it names the value that would pin it")

        with self.assertLogs("RLShadowWiring", level="INFO") as exact:
            maybe_build_shadow_router(_rl("v2_candidate_baseline_v1@1"))
        self.assertNotIn("by NAME", "\n".join(exact.output))

    def test_a_refusal_says_the_declaration_carried_no_version(self) -> None:
        """Because the most likely reason a name-only declaration fails is that the operator meant a
        different policy, and the message should not read as if the version were the problem."""
        with self.assertLogs("RLShadowWiring", level="WARNING") as caught:
            maybe_build_shadow_router(_rl("v6_recovery_baseline_v1"))
        self.assertIn("names no version", "\n".join(caught.output))


class ItStillFailsSafeTests(unittest.TestCase):
    def test_a_missing_artifact_is_still_no_router_rather_than_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            self.assertIsNone(maybe_build_shadow_router(_rl("anything", missing)))

    def test_a_non_shadow_mode_builds_nothing(self) -> None:
        cfg = SimpleNamespace(rl=SimpleNamespace(
            mode="hybrid_ml", policy_id="v2_candidate_baseline_v1", artifact_path=str(_ARTIFACT),
        ))
        self.assertIsNone(maybe_build_shadow_router(cfg))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
