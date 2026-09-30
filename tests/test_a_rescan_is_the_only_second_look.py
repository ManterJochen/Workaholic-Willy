"""A rescan is the only second look: ``next_viewpoint`` is merged into it, and ``dense_recovery`` is gone.

Removed on 2026-09-29 (cleanup phase 4, owner-approved 2026-09-28): ``SceneRecoveryAction.NEXT_VIEWPOINT``, merged
into ``RESCAN`` (with the viewpoint planners gone nothing moved the camera for it, so it re-perceived exactly as
``RESCAN`` does); the ``robot.grasping.dense_recovery`` block with the policy and the strategy it built on the
service (``recovery_policy``, ``recovery_strategy``), which no pick consulted; the strategies only it built
(``ActivePerceptionRecoveryStrategy``, ``NextTargetRecoveryStrategy``, ``NoRecoveryStrategy``); the
``recovery_fixture`` argument; and the effective keys ``dense_recovery_enabled`` and
``dense_recovery_allowed_actions``. The recovery a pick runs is ``robot.grasping.recovery``, unchanged.

What must hold after the removal, and is pinned here:

* a tree or a preset that still names ``next_viewpoint`` is refused with ``removed on purpose: use rescan``, and so
  is an effective config built by hand, when the service builds its recovery policy from it;
* a tree, a preset or ``explain`` that still names the ``dense_recovery`` block or one of its keys is answered
  with the sentence that says what to do;
* ``auto`` allows ``rescan`` and ``dense_clutter`` allows ``rescan`` and ``nudge_target`` (and both allow
  ``next_target`` since the owner's recovery decisions of the same day), and every failure class that tried
  ``NEXT_VIEWPOINT`` tries ``RESCAN``;
* a record logged before the merge still audits and round-trips, the offline recovery trainer still reads its
  ``next_viewpoint``, and the canonical replay packs, synthesised in that shape, still regenerate as committed;
* nothing still builds or accepts a dense-recovery policy or strategy.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from src.config import ConfigError
from src.config.loader import load_robot_section
from src.config.schema._removed import REMOVED_KEYS, REMOVED_RECOVERY_ACTIONS
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from tests.test_section_loaders import _ScratchTree

_USE_RESCAN = "removed on purpose: use rescan"

#: Every key the removed block held, with a value an old file wrote.
_DENSE_KEYS: dict[str, Any] = {
    "enabled": True,
    "max_recovery_actions": 2,
    "strategy": "active_perception",
    "allowed_actions": ["next_viewpoint"],
}
_DENSE = "robot.grasping.dense_recovery"


def _write_grasping(root: Path, **blocks: dict) -> None:
    path = root / "robot" / "robot.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["robot"].setdefault("grasping", {}).update(blocks)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class NextViewpointIsRefusedWhereverItIsNamedTests(_ScratchTree):
    """Loads a copy of the shipped tree that still names it, as an operator's old file would."""

    def _refusal(self, recovery: dict) -> str:
        _write_grasping(self.root, recovery=recovery)
        with self.assertRaises(ConfigError) as caught:
            load_robot_section(self.root, profile=None)
        return " ".join(str(caught.exception).split())

    def test_an_allow_list_that_names_it_is_refused_with_the_sentence(self) -> None:
        said = self._refusal({"enabled": True, "allowed_actions": ["rescan", "next_viewpoint"]})
        self.assertIn("recovery.allowed_actions names the recovery action 'next_viewpoint'", said)
        self.assertIn(_USE_RESCAN, said)
        self.assertIn(REMOVED_RECOVERY_ACTIONS["next_viewpoint"], said)

    def test_a_budget_that_names_it_is_refused_with_the_sentence(self) -> None:
        said = self._refusal({"enabled": True, "allowed_actions": ["rescan"],
                              "per_action_budget": [["next_viewpoint", 1]]})
        self.assertIn("recovery.per_action_budget names the recovery action 'next_viewpoint'", said)
        self.assertIn(_USE_RESCAN, said)

    def test_the_tree_without_it_still_loads(self) -> None:
        self._assert_loads(load_robot_section)


class NextViewpointIsRefusedByTheSchemaAndThePresetTests(unittest.TestCase):

    def test_the_schema_refuses_it_in_any_spelling_and_names_rescan(self) -> None:
        for written in ("next_viewpoint", "NEXT_VIEWPOINT", " next_viewpoint "):
            with self.subTest(written=written):
                with self.assertRaises(ValueError) as caught:
                    RobotGraspingConfig.model_validate({"recovery": {"allowed_actions": [written]}})
                self.assertIn(_USE_RESCAN, " ".join(str(caught.exception).split()))
        self.assertEqual(
            RobotGraspingConfig.model_validate({"recovery": {"allowed_actions": ["rescan"]}}).recovery.allowed_actions,
            ("rescan",),
        )

    def test_the_sentence_is_one_sentence_that_names_rescan(self) -> None:
        said = REMOVED_RECOVERY_ACTIONS["next_viewpoint"]
        self.assertTrue(said.startswith("use rescan"))
        self.assertIn("2026-09-29", said)
        self.assertTrue(said.endswith("."))
        self.assertEqual(said.rstrip(".").count(". "), 0, said)

    def test_a_preset_that_names_it_is_refused_by_its_validation(self) -> None:
        from pydantic import ValidationError

        from src.robot.grasping.replay import presets

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "old.yaml").write_text(yaml.safe_dump(
                {"grasping": {"default_mode": "dense_clutter",
                              "recovery": {"enabled": True, "allowed_actions": ["next_viewpoint"]}}}),
                encoding="utf-8")
            with mock.patch.object(presets, "_PRESETS_DIR", Path(tmp)):
                with self.assertRaises(ValidationError) as caught:
                    presets.validate_preset("old", base={"vendor": "dummy", "gripper": {"vendor": "none"}})
        self.assertIn(_USE_RESCAN, " ".join(str(caught.exception).split()))

    def test_the_shipped_dense_clutter_preset_allows_rescan_and_validates(self) -> None:
        from src.robot.grasping.replay.presets import apply_preset, validate_preset

        validate_preset("dense_clutter", base={"vendor": "dummy", "gripper": {"vendor": "none"}})
        merged = apply_preset({"vendor": "dummy", "gripper": {"vendor": "none"}}, "dense_clutter")
        self.assertEqual(merged["grasping"]["recovery"]["allowed_actions"], ["rescan"])


class AHandBuiltSnapshotThatNamesItIsRefusedTests(unittest.TestCase):
    """A simulator runner builds the effective config with ``dataclasses.replace`` and so skips the schema; the
    service answers it with the same sentence, not as an unknown enum value."""

    def _policy(self, **recovery: Any) -> Any:
        from types import SimpleNamespace

        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspService,
            EffectiveGraspingConfig,
            EffectiveRecoveryOrchestratorConfig,
            GraspMode,
        )

        effective = EffectiveGraspingConfig(
            default_mode=GraspMode.DENSE_CLUTTER, max_attempts=1,
            recovery_orchestrator=EffectiveRecoveryOrchestratorConfig(enabled=True, **recovery),
        )
        return AutonomousGraspService._build_recovery_orchestrator_policy(SimpleNamespace(effective_config=effective))

    def test_an_allow_list_or_a_budget_that_names_it_is_refused_with_the_sentence(self) -> None:
        for recovery in ({"allowed_actions": ("rescan", "next_viewpoint")},
                         {"allowed_actions": ("rescan",), "per_action_budget": (("next_viewpoint", 1),)}):
            with self.subTest(recovery=recovery):
                with self.assertRaises(ValueError) as caught:
                    self._policy(**recovery)
                said = " ".join(str(caught.exception).split())
                self.assertIn("names the recovery action 'next_viewpoint'", said)
                self.assertIn(_USE_RESCAN, said)
                self.assertIn(REMOVED_RECOVERY_ACTIONS["next_viewpoint"], said)

    def test_a_name_that_was_never_an_action_keeps_the_enum_refusal(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._policy(allowed_actions=("teleport",))
        self.assertNotIn("removed on purpose", str(caught.exception))

    def test_rescan_still_builds(self) -> None:
        from src.robot.grasping.recovery.policy import SceneRecoveryAction

        policy = self._policy(allowed_actions=("rescan",), per_action_budget=(("rescan", 1),))
        self.assertEqual(policy.allowed_actions, (SceneRecoveryAction.RESCAN,))
        self.assertEqual(policy.per_action_budget[SceneRecoveryAction.RESCAN], 1)


class TheDenseRecoveryBlockIsRefusedTests(_ScratchTree):

    def test_each_key_is_refused_at_the_block_with_its_sentence(self) -> None:
        for key, value in _DENSE_KEYS.items():
            with self.subTest(key=key):
                _write_grasping(self.root, dense_recovery={key: value})
                with self.assertRaises(ConfigError) as caught:
                    load_robot_section(self.root, profile=None)
                message = " ".join(str(caught.exception).split())
                self.assertIn(_DENSE, message)
                self.assertEqual(message.count("removed on purpose"), 1)
                self.assertIn(REMOVED_KEYS[f"{_DENSE}.{key}"], message)


class TheDenseRecoverySentenceSaysWhatToDoTests(unittest.TestCase):

    def test_the_block_and_every_key_have_one_sentence_that_names_the_recovery_block(self) -> None:
        for dotted in (_DENSE, *(f"{_DENSE}.{key}" for key in _DENSE_KEYS)):
            with self.subTest(key=dotted):
                said = REMOVED_KEYS[dotted]
                self.assertIn("delete this block", said)
                self.assertIn("robot.grasping.recovery", said)
                self.assertIn("2026-09-29", said)
                self.assertTrue(said.endswith("."))
                self.assertEqual(said.rstrip(".").count(". "), 0, said)

    def test_explain_answers_each_key_with_the_sentence(self) -> None:
        from src.config.explain import explain

        root = Path(__file__).resolve().parents[1] / "config"
        for dotted in (_DENSE, *(f"{_DENSE}.{key}" for key in _DENSE_KEYS)):
            with self.subTest(key=dotted):
                answer = explain(dotted, root, ())
                self.assertFalse(answer.known)
                self.assertEqual(answer.removed, REMOVED_KEYS[dotted])

    def test_a_preset_that_still_writes_it_is_refused_with_the_sentence(self) -> None:
        from pydantic import ValidationError

        from src.robot.grasping.replay import presets

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "old.yaml").write_text(
                yaml.safe_dump({"grasping": {"dense_recovery": {"enabled": True}}}), encoding="utf-8")
            with mock.patch.object(presets, "_PRESETS_DIR", Path(tmp)):
                with self.assertRaises(ValidationError) as caught:
                    presets.validate_preset("old", base={"vendor": "dummy", "gripper": {"vendor": "none"}})
        said = " ".join(str(caught.exception).split())
        self.assertEqual(said.count("removed on purpose"), 1)
        self.assertIn(REMOVED_KEYS[_DENSE], said)


class RescanTakesNextViewpointsPlaceTests(unittest.TestCase):

    def test_the_profiles(self) -> None:
        from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for

        self.assertEqual(_profile_for(GraspMode.EASY).recovery_allowed_actions, ())
        self.assertEqual(_profile_for(GraspMode.AUTO).recovery_allowed_actions, ("rescan", "next_target"))
        self.assertEqual(_profile_for(GraspMode.DENSE_CLUTTER).recovery_allowed_actions,
                         ("rescan", "next_target", "nudge_target"))

    def test_the_action_is_gone_from_the_vocabulary(self) -> None:
        from src.robot.grasping.recovery.policy import SceneRecoveryAction, SceneRecoveryPolicy

        self.assertNotIn("next_viewpoint", {action.value for action in SceneRecoveryAction})
        with self.assertRaises(ValueError):
            SceneRecoveryAction("next_viewpoint")
        self.assertEqual(SceneRecoveryPolicy().allowed_actions, (SceneRecoveryAction.RESCAN,))

    def test_every_failure_class_that_moved_the_view_now_rescans(self) -> None:
        from src.robot.grasping.recovery.orchestrator import RecoveryDispatcher
        from src.robot.grasping.recovery.policy import SceneRecoveryAction as A
        from src.robot.grasping.types.feedback import GraspFailureReason as R

        dispatcher = RecoveryDispatcher()
        expected = {
            R.RESCAN_RECOMMENDED: (A.RESCAN,), R.LOW_DEPTH_CONFIDENCE: (A.RESCAN,),
            R.LOW_MASK_CONFIDENCE: (A.RESCAN,), R.EMPTY_MASK: (A.RESCAN,), R.MASK_TOO_SMALL: (A.RESCAN,),
            R.NO_VALID_DEPTH: (A.RESCAN,), R.ACTIVE_PERCEPTION_RECOMMENDED: (A.RESCAN,),
            R.HEAVY_OCCLUSION: (A.RESCAN,),
            # The push left the no-candidate rows and NEXT_TARGET joined the refusal row the same day (owner).
            R.NO_CANDIDATES_GENERATED: (A.NEXT_TARGET, A.RESCAN),
            R.ALL_COLLIDED: (A.NEXT_TARGET, A.NUDGE_TARGET, A.CONTAINER_AGITATE, A.RESCAN),
            R.NO_VALID_GRASP: (A.NEXT_TARGET, A.RESCAN),
            R.MOTION_PLAN_REFUSED: (A.NEXT_TARGET, A.RESCAN),
        }
        for reason, actions in expected.items():
            with self.subTest(reason=reason.value):
                self.assertEqual(dispatcher.actions_for(reason), actions)

    def test_the_loop_rescans_where_it_moved_the_view(self) -> None:
        """Through the service's own loop driver: an occluded attempt is rescanned, and nothing moves."""
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
        from src.robot.grasping.recovery.orchestrator import (
            RecoveryDispatcher,
            RecoveryOrchestrator,
            run_recovery_loop,
        )
        from src.robot.grasping.recovery.policy import SceneRecoveryAction, SceneRecoveryPolicy
        from src.robot.grasping.types.feedback import GraspFailureReason

        reports = iter((
            SimpleReport(AutonomousGraspOutcome.NO_VALID_GRASP, (GraspFailureReason.HEAVY_OCCLUSION,)),
            SimpleReport(AutonomousGraspOutcome.SUCCEEDED, ()),
        ))
        looked: list[int] = []
        policy = SceneRecoveryPolicy(enabled=True, allowed_actions=(SceneRecoveryAction.RESCAN,),
                                     max_recovery_actions=2)
        final, trail = run_recovery_loop(
            pick=lambda: next(reports), profile=_profile_for(GraspMode.AUTO), policy=policy,
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), strategies={},
                                              bypass_strategies=True),
            frame_acquirer=lambda: looked.append(1), arm=None,
        )
        self.assertIs(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(trail.terminal_reason, "recovered_success")
        self.assertEqual([entry.plan_action for entry in trail.entries], [SceneRecoveryAction.RESCAN])
        self.assertEqual(looked, [1])


@dataclasses.dataclass
class SimpleReport:
    outcome: Any
    failure_reasons: tuple


class OldRecordsStillReadTests(unittest.TestCase):
    """A record logged before the merge carries ``next_viewpoint``; the replay and RL layers must still read it."""

    def test_a_record_with_the_retired_action_audits_and_round_trips(self) -> None:
        from src.robot.grasping.replay.telemetry_catalog import audit_record
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        record = GraspAttemptRecord(
            timestamp=1.0, attempt_id="before-2026-09-29", mode="dense_clutter", final_outcome="recovery_exhausted",
            recovery_actions=({"action": "next_viewpoint", "outcome": "no_change"},),
            profile={"mode": "dense_clutter", "recovery_allowed_actions": ["rescan", "next_viewpoint", "nudge_target"],
                     "verification_enabled": False, "refinement_enabled": False},
            extra={"dense_recovery_enabled": False, "dense_recovery_allowed_actions": ["next_viewpoint"]},
        )
        self.assertEqual(audit_record(record), ())
        self.assertEqual(GraspAttemptRecord.from_json(record.to_json()), record)

    def test_the_recovery_trainer_still_reads_a_logged_next_viewpoint(self) -> None:
        from src.robot.grasping.rl.recovery_policy import RECOVERY_ACTION_REOBSERVE
        from src.robot.grasping.rl.train_recovery import RECOVERY_TOKEN_MAP

        self.assertEqual(RECOVERY_TOKEN_MAP["next_viewpoint"], RECOVERY_ACTION_REOBSERVE)

    def test_the_canonical_packs_still_regenerate_as_committed(self) -> None:
        """The packs were synthesised with ``next_viewpoint`` and a ``verification`` block, and their bytes are a
        signed-off contract. Compared as values, so a platform's decimal spelling of a float does not decide it."""
        from src.robot.grasping.replay.canonical_datasets import (
            CANONICAL_PACKS,
            render_pack_jsonl,
            repo_root_from_module,
        )

        def same(a: Any, b: Any) -> bool:
            if isinstance(a, float) or isinstance(b, float):
                return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                        and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
            if isinstance(a, dict):
                return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
            if isinstance(a, list):
                return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            return bool(a == b)

        root = repo_root_from_module()
        saw_next_viewpoint = False
        for pack in CANONICAL_PACKS:
            with self.subTest(pack=pack.name):
                disk = (root / pack.relative_path).read_text(encoding="utf-8").splitlines()
                made = render_pack_jsonl(pack).splitlines()
                self.assertEqual(len(disk), len(made))
                for index, (a, b) in enumerate(zip(disk, made)):
                    self.assertTrue(same(json.loads(a), json.loads(b)), f"{pack.name} line {index}")
                saw_next_viewpoint = saw_next_viewpoint or '"next_viewpoint"' in "".join(made)
        self.assertTrue(saw_next_viewpoint, "the dense pack no longer carries the retired action it was made with")


class NothingBuildsADenseRecoveryPolicyTests(unittest.TestCase):

    def test_the_strategies_only_it_built_are_gone(self) -> None:
        policy = importlib.import_module("src.robot.grasping.recovery.policy")
        grasping = importlib.import_module("src.robot.grasping")
        schema = importlib.import_module("src.config.schema.robot")
        for module, name in (
            (policy, "ActivePerceptionRecoveryStrategy"), (policy, "NextTargetRecoveryStrategy"),
            (policy, "NoRecoveryStrategy"), (grasping, "ActivePerceptionRecoveryStrategy"),
            (grasping, "NextTargetRecoveryStrategy"), (grasping, "NoRecoveryStrategy"),
            (schema, "GraspingDenseRecoveryConfig"),
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(module, name))
        # The agitate strategy stays. The scene-blind nudge left too (2026-09-29): the push runs inside the pick
        # attempt, planned from what the camera saw.
        self.assertFalse(hasattr(policy, "SmallNudgeStrategy"))
        self.assertFalse(hasattr(grasping, "SmallNudgeStrategy"))
        self.assertTrue(hasattr(policy, "ContainerAgitateStrategy"))

    def test_no_service_takes_or_carries_one(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService

        slots = {"recovery_policy", "recovery_strategy"}
        self.assertFalse(slots & {f.name for f in dataclasses.fields(AutonomousGraspService)})
        for factory in (AutonomousGraspService.from_components, AutonomousGraspService.from_robot_config):
            with self.subTest(factory=factory.__qualname__):
                parameters = set(inspect.signature(factory).parameters)
                self.assertFalse((slots | {"recovery_fixture"}) & parameters)

    def test_the_snapshot_forgot_it(self) -> None:
        from src.robot.execution.autonomous_grasp import EffectiveGraspingConfig, GraspMode

        self.assertNotIn("dense_recovery", RobotGraspingConfig.model_fields)
        fields = {f.name for f in dataclasses.fields(EffectiveGraspingConfig)}
        keys = EffectiveGraspingConfig(default_mode=GraspMode.AUTO, max_attempts=5).to_dict()
        for gone in ("dense_recovery_enabled", "dense_recovery_allowed_actions"):
            with self.subTest(key=gone):
                self.assertNotIn(gone, fields)
                self.assertNotIn(gone, keys)


if __name__ == "__main__":
    unittest.main()
