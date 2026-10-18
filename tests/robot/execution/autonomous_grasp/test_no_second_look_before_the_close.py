"""No second look before the close: the two-scan refinement, its two modes and the post-lift vision check are gone.

Removed on 2026-09-29 (cleanup phase 3, owner-approved 2026-09-28): the two-scan pre-grasp refinement
(`closed_loop/refinement.py`, the service's `_pick_with_refinement` with its `S3`/`S4` readiness gates), the target
trackers it re-identified the target with (`closed_loop/target_tracking.py`: `TargetIdentity`,
`IoUCentroidTargetTracker`, `WorldSpacePoseTracker`), the `closed_loop` and `dense_autonomous` grasp modes built on
it, the `robot.grasping.closed_loop` block, the effective key `closed_loop_enabled`, the report's `refinement`
field, and the post-lift vision check (`VisionTargetDisplacementVerifier`, `verification.post_lift_vision_check`,
`verification.vision_displacement_iou_max`), whose target identity came from that refinement. No shipped config
switched any of it on, and it never ran on a physical arm. Owner's decision with it: `nudge_target` moved into the
`dense_clutter` profile, which was the only other dense mode, so the one physical recovery a built-in profile
allowed stays reachable.

What must hold after the removal, and is pinned here:

* a tree that still writes one of the removed keys is refused at load with the sentence that replaces it, and
  `python -m src.config explain` says the same sentence;
* a tree or a caller that still names a removed mode is refused with the mode to name instead;
* `dense_clutter` allows `nudge_target`, and the push gate (`push_permitted`) lets it through there and nowhere else:
  the push runs inside the pick attempt;
* a record logged before the removal still audits and still classifies, whatever retired string it carries;
* nothing that took a refiner, a tracker or a post-lift frame still accepts one.

The ``verification`` block the two post-lift keys sat in left whole on 2026-09-29 as well (cleanup phase 4,
``tests/robot/execution/autonomous_grasp/test_the_gripper_says_whether_it_holds.py``), so a tree that still writes one of them is refused at the
block, with the block's sentence; ``explain`` still answers each key with its own.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import unittest
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.config import ConfigError
from src.config.loader import load_robot_section
from src.config.schema._removed import REMOVED_GRASP_MODES, REMOVED_KEYS
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from tests.models.test_section_loaders import _ScratchTree

#: Each removed key, as the whole tree names it, with the block it sat in and a value an old file wrote.
_REMOVED = {
    "robot.grasping.closed_loop": (None, "closed_loop", {"enabled": True, "pregrasp_rescan": False}),
    "robot.grasping.verification.post_lift_vision_check": ("verification", "post_lift_vision_check", True),
    "robot.grasping.verification.vision_displacement_iou_max": (
        "verification", "vision_displacement_iou_max", 0.2),
}

#: What a loader names when a key is refused: the key itself, or the block it sat in when
#: that block left too. ``verification`` left whole on 2026-09-29, so its keys are refused as the block.
_REFUSED_AS = {
    "robot.grasping.closed_loop": "robot.grasping.closed_loop",
    "robot.grasping.verification.post_lift_vision_check": "robot.grasping.verification",
    "robot.grasping.verification.vision_displacement_iou_max": "robot.grasping.verification",
}

#: Each removed mode, by every spelling it was accepted in, with the words that name its replacement.
_MODES = {
    "closed_loop": "name auto instead",
    "closedloop": "name auto instead",
    "dense_autonomous": "name dense_clutter instead",
    "autonomous": "name dense_clutter instead",
}


def _write_grasping(root: Path, edit) -> None:  # noqa: ANN001
    path = root / "robot" / "robot.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data["robot"].setdefault("grasping", {}))
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class TheRemovedKeysAreRefusedAtLoadTests(_ScratchTree):
    """Loads a copy of the shipped tree that still writes the key, as an operator's old file would."""

    def _refusal(self, edit) -> str:  # noqa: ANN001
        _write_grasping(self.root, edit)
        with self.assertRaises(ConfigError) as caught:
            load_robot_section(self.root, profile=None)
        return str(caught.exception)

    def _assert_refused_with_its_sentence(self, dotted: str) -> None:
        block, key, value = _REMOVED[dotted]

        def edit(grasping: dict) -> None:
            (grasping.setdefault(block, {}) if block else grasping)[key] = value

        message = self._refusal(edit)
        self.assertIn(_REFUSED_AS[dotted], message)
        self.assertEqual(message.count("removed on purpose"), 1)
        self.assertIn(REMOVED_KEYS[_REFUSED_AS[dotted]], " ".join(message.split()))

    def test_the_closed_loop_block_is_refused_with_its_sentence(self) -> None:
        self._assert_refused_with_its_sentence("robot.grasping.closed_loop")

    def test_the_post_lift_vision_check_is_refused_with_its_sentence(self) -> None:
        self._assert_refused_with_its_sentence("robot.grasping.verification.post_lift_vision_check")

    def test_its_threshold_is_refused_with_its_sentence(self) -> None:
        self._assert_refused_with_its_sentence("robot.grasping.verification.vision_displacement_iou_max")

    def test_all_three_at_once_are_all_named(self) -> None:
        def edit(grasping: dict) -> None:
            grasping["closed_loop"] = {"enabled": True}
            grasping.setdefault("verification", {}).update(
                {"post_lift_vision_check": True, "vision_displacement_iou_max": 0.3})

        message = self._refusal(edit)
        # Two refusals: the closed_loop block, and the verification block both keys sat in.
        self.assertEqual(message.count("removed on purpose"), 2)
        for dotted in _REMOVED:
            self.assertIn(_REFUSED_AS[dotted], message)

    def test_the_tree_without_them_still_loads(self) -> None:
        self._assert_loads(load_robot_section)


class TheSentencesSayWhatToDoTests(unittest.TestCase):

    def test_each_key_is_named_and_gone_from_the_schema(self) -> None:
        models: dict[str | None, type[BaseModel]] = {None: RobotGraspingConfig}
        for dotted, (block, key, _value) in _REMOVED.items():
            with self.subTest(key=dotted):
                self.assertIn(dotted, REMOVED_KEYS)
                if block is None:
                    self.assertNotIn(key, models[block].model_fields)
                else:
                    # The block the key sat in is gone as a whole.
                    self.assertNotIn(block, RobotGraspingConfig.model_fields)

    def test_each_sentence_is_one_sentence_that_says_delete_it(self) -> None:
        for dotted in _REMOVED:
            with self.subTest(key=dotted):
                said = REMOVED_KEYS[dotted]
                self.assertRegex(said, r"delete this (key|block)")
                self.assertIn("2026-09-29", said)
                self.assertTrue(said.endswith("."))
                self.assertEqual(said.rstrip(".").count(". "), 0, said)

    def test_each_mode_sentence_is_one_sentence_that_names_the_replacement(self) -> None:
        for mode, replacement in _MODES.items():
            with self.subTest(mode=mode):
                said = REMOVED_GRASP_MODES[mode]
                self.assertIn(replacement, said)
                self.assertIn("2026-09-29", said)
                self.assertTrue(said.endswith("."))
                self.assertEqual(said.rstrip(".").count(". "), 0, said)


class TheSameSentenceWhereverTheKeyIsWrittenTests(unittest.TestCase):
    """`explain` answers without loading: it says the sentence too."""

    def test_explain_answers_with_the_sentence_rather_than_a_near_miss(self) -> None:
        from src.config.explain import explain

        root = Path(__file__).resolve().parents[4] / "config"
        asked = {dotted: dotted for dotted in _REMOVED}
        asked["robot.grasping.closed_loop.enabled"] = "robot.grasping.closed_loop"
        asked["robot.grasping.fusion.cameras.cam0.mounting_mode"] = "robot.grasping.fusion.cameras.*.mounting_mode"
        for path, dotted in asked.items():
            with self.subTest(key=path):
                answer = explain(path, root, ())
                self.assertFalse(answer.known)
                self.assertEqual(answer.removed, REMOVED_KEYS[dotted])
                self.assertEqual(answer.suggestions, ())
                text = " ".join(answer.render().split())
                self.assertIn("REMOVED ON PURPOSE", text)
                self.assertIn(REMOVED_KEYS[dotted], text)
        typo = explain("robot.grasping.clsed_loop", root, ())
        self.assertEqual(typo.removed, "")
        self.assertIn("NOT A KNOWN KEY", typo.render())


class ARemovedModeIsRefusedWithItsReplacementTests(_ScratchTree):
    """Wherever a mode is named: the tree's `default_mode` and a caller. The one mode list of the schema,
    `recovery.apply_modes`, left with the recovery block on 2026-10-16."""

    def _refusal(self, edit) -> str:  # noqa: ANN001
        _write_grasping(self.root, edit)
        with self.assertRaises(ConfigError) as caught:
            load_robot_section(self.root, profile=None)
        return " ".join(str(caught.exception).split())

    def test_a_tree_whose_default_mode_left_is_refused_at_load(self) -> None:
        for mode in ("closed_loop", "dense_autonomous"):
            with self.subTest(mode=mode):
                message = self._refusal(lambda grasping, mode=mode: grasping.update(default_mode=mode))
                self.assertIn("robot.grasping.default_mode", message)
                self.assertIn("removed on purpose", message)
                self.assertIn(REMOVED_GRASP_MODES[mode], message)
                self.assertIn(_MODES[mode], message)

    def test_a_caller_that_names_one_is_refused_before_anything_is_built(self) -> None:
        from src.robot.execution.autonomous_grasp import GraspMode, resolve_grasp_mode

        for spelling, replacement in _MODES.items():
            for written in (spelling, f"  {spelling.upper()} "):
                with self.subTest(mode=written):
                    with self.assertRaises(ValueError) as caught:
                        resolve_grasp_mode(written)
                    said = str(caught.exception)
                    self.assertIn("removed on purpose", said)
                    self.assertIn(replacement, said)
        self.assertEqual({mode.value for mode in GraspMode}, {"easy", "auto", "dense_clutter"})

    def test_the_shipped_tree_names_none(self) -> None:
        self._assert_loads(load_robot_section)


class NudgeTargetBelongsToDenseClutterTests(unittest.TestCase):
    """The owner's decision: the push `dense_autonomous` alone allowed moved to the dense mode that stays."""

    def test_the_dense_clutter_profile_allows_it(self) -> None:
        from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for

        # `next_viewpoint` left it on 2026-09-29, merged into `rescan` (cleanup phase 4); `next_target` joined
        # the same day (owner) and left on 2026-10-16 (the owner's cleanup).
        self.assertEqual(_profile_for(GraspMode.DENSE_CLUTTER).recovery_allowed_actions, ("rescan", "nudge_target"))

    def test_the_push_gate_lets_it_through_in_dense_clutter_only(self) -> None:
        from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
        from src.robot.grasping.recovery.policy import SceneRecoveryAction, SceneRecoveryPolicy, push_permitted

        policy = SceneRecoveryPolicy(enabled=True, allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
                                     max_recovery_actions=1)
        self.assertTrue(push_permitted(_profile_for(GraspMode.DENSE_CLUTTER), policy))
        for mode in (GraspMode.AUTO, GraspMode.EASY):
            with self.subTest(mode=mode):
                self.assertFalse(push_permitted(_profile_for(mode), policy),
                                 f"{mode.value} let a push through its profile gate")


class OldRecordsStillAuditTests(unittest.TestCase):
    """A record logged before the removal carries strings no enum names any more; the replay layer must read it."""

    @staticmethod
    def _record(final_outcome: str, *, mode: str = "closed_loop", refinement: "dict | None" = None,
                extra: "dict | None" = None):  # noqa: ANN205
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        return GraspAttemptRecord(timestamp=1.0, attempt_id="before-2026-09-29", mode=mode,
                                  final_outcome=final_outcome, refinement=refinement, extra=extra or {})

    def test_the_three_refine_outcomes_audit_with_their_block_and_flag_without_it(self) -> None:
        from src.robot.grasping.replay.telemetry_catalog import audit_record

        block = {"outcome": "diverged", "position_delta_mm": 41.0, "telemetry": {"stage": "refiner"}}
        for outcome in ("refinement_failed", "target_lost_during_refine", "refinement_diverged"):
            with self.subTest(outcome=outcome):
                self.assertEqual(audit_record(self._record(outcome, refinement=block)), ())
                self.assertEqual(audit_record(self._record(outcome)), ("refinement",))

    def test_a_record_in_a_retired_mode_round_trips_and_classifies(self) -> None:
        from src.robot.grasping.replay.failure_taxonomy import FailureRootCause, classify_record
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        record = self._record("refinement_diverged", mode="dense_autonomous", refinement={"outcome": "diverged"},
                              extra={"collision_evidence": True})
        again = GraspAttemptRecord.from_json(record.to_json())
        self.assertEqual(again, record)
        self.assertIs(classify_record(again).primary, FailureRootCause.COLLISION_REJECTION)

    def test_no_writer_fills_the_refinement_block_any_more(self) -> None:
        from src.robot.execution.autonomous_grasp.record_logging import to_attempt_record

        class _Report:
            outcome = "succeeded"
            mode = "dense_clutter"
            profile = None
            pick_report = None
            telemetry: dict = {}
            refinement = {"outcome": "accepted"}  # a stray attribute is not a verdict any more

        record = to_attempt_record(_Report(), attempt_id="after")
        self.assertIsNone(record.refinement)
        assert record.profile is not None
        self.assertIs(record.profile["refinement_enabled"], False)


class NothingRefinesTests(unittest.TestCase):

    def test_the_modules_and_their_names_are_gone(self) -> None:
        # The verification module, which held the post-lift check, left whole on 2026-09-29 as well.
        for dotted in ("src.robot.grasping.closed_loop.refinement", "src.robot.grasping.closed_loop.target_tracking",
                       "src.robot.grasping.closed_loop.verification"):
            with self.subTest(module=dotted):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(dotted)
        grasping = importlib.import_module("src.robot.grasping")
        for module, name in (
            (grasping, "DefaultPreGraspRefiner"), (grasping, "RefinementPolicy"), (grasping, "TargetIdentity"),
            (grasping, "IoUCentroidTargetTracker"), (grasping, "WorldSpacePoseTracker"),
            (grasping, "VisionTargetDisplacementVerifier"),
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(module, name))

    def test_no_constructor_accepts_a_refiner(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService

        self.assertNotIn("refiner", {f.name for f in dataclasses.fields(AutonomousGraspService)})
        self.assertNotIn("refinement_policy", {f.name for f in dataclasses.fields(AutonomousGraspService)})
        for factory in (AutonomousGraspService.from_components, AutonomousGraspService.from_robot_config):
            with self.subTest(factory=factory.__qualname__):
                parameters = inspect.signature(factory).parameters
                self.assertNotIn("refiner", parameters)
                self.assertNotIn("refinement_policy", parameters)

    def test_the_report_the_outcomes_and_the_snapshot_forgot_it(self) -> None:
        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspOutcome,
            AutonomousGraspReport,
            EffectiveGraspingConfig,
            GraspBehaviorProfile,
        )

        self.assertNotIn("refinement", {f.name for f in dataclasses.fields(AutonomousGraspReport)})
        self.assertFalse({"refinement_failed", "target_lost_during_refine", "refinement_diverged"}
                         & {outcome.value for outcome in AutonomousGraspOutcome})
        self.assertNotIn("closed_loop_enabled", {f.name for f in dataclasses.fields(EffectiveGraspingConfig)})
        self.assertNotIn("refinement_enabled", {f.name for f in dataclasses.fields(GraspBehaviorProfile)})


if __name__ == "__main__":
    unittest.main()
