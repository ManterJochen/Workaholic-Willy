"""The gripper says whether it holds: the post-grasp verification stage is gone, and its block with it.

Removed on 2026-09-29 (cleanup phase 4, owner-approved 2026-09-29): the ``robot.grasping.verification`` block
(``GraspingVerificationConfig``), the verifiers it built (``closed_loop/verification.py``: ``GraspVerificationPolicy``,
``NoOpVerifier``, ``ObjectDetectingGripperVerifier``, ``WidthDeltaGripperVerifier``, ``CompositeGraspVerifier``), the
builders that assembled them, the service's ``verification_policy`` / ``verifier`` slots and its verification stage,
``GraspBehaviorProfile.verification_enabled`` and the effective key ``verification_enabled``, and the stopgap that
refused ``verification.enabled`` as an unwired switch. No pick path ran the stage once the two-scan refinement left.
The hold is verified where it always was on the open-loop path: :class:`GraspExecutionPolicy` reads the gripper's own
``is_object_detected`` and ``hold_evidence`` right after every close, and the real Robotiq socket driver reports its
gOBJ register there.

What must hold after the removal, and is pinned here:

* a tree, a preset or ``explain`` that still names the block or one of its keys is answered with the sentence that
  says what to do, and the unwired-switch list keeps its one other entry;
* a pick on Robotiq-like hold evidence still reports held and empty through the execution policy, up to the
  service's outcome and the record it logs;
* a record logged before the removal, verification block and all, still audits and classifies; one logged after it
  needs no such block;
* nothing still builds, accepts or exports a verifier.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from src.config import ConfigError
from src.config.loader import load_robot_section
from src.config.schema._removed import REMOVED_KEYS
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from tests.test_section_loaders import _ScratchTree

#: Every key the removed block held, with a value an old file wrote.
_KEYS: dict[str, Any] = {
    "enabled": False,
    "require_object_detected": True,
    "width_delta_min_mm": 2.0,
    "width_delta_max_mm": 10.0,
    "fail_closed": True,
    "require_all_conclusive": False,
}
_BLOCK = "robot.grasping.verification"


def _write_grasping(root: Path, block: dict) -> None:
    path = root / "robot" / "robot.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["robot"].setdefault("grasping", {})["verification"] = block
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class TheBlockIsRefusedWhereverItIsWrittenTests(_ScratchTree):
    """Loads a copy of the shipped tree that still writes the block, as an operator's old file would."""

    def test_each_key_is_refused_at_the_block_with_its_sentence(self) -> None:
        for key, value in _KEYS.items():
            with self.subTest(key=key):
                _write_grasping(self.root, {key: value})
                with self.assertRaises(ConfigError) as caught:
                    load_robot_section(self.root, profile=None)
                message = " ".join(str(caught.exception).split())
                self.assertIn(_BLOCK, message)
                self.assertEqual(message.count("removed on purpose"), 1)
                self.assertIn(REMOVED_KEYS[f"{_BLOCK}.{key}"], message)

    def test_the_tree_without_it_still_loads(self) -> None:
        self._assert_loads(load_robot_section)


class TheSentenceSaysWhatToDoTests(unittest.TestCase):

    def test_the_block_and_every_key_have_one_sentence_that_says_delete_it(self) -> None:
        for dotted in (_BLOCK, *(f"{_BLOCK}.{key}" for key in _KEYS)):
            with self.subTest(key=dotted):
                said = REMOVED_KEYS[dotted]
                self.assertIn("delete this block", said)
                self.assertIn("2026-09-29", said)
                self.assertIn("hold_evidence", said)
                self.assertTrue(said.endswith("."))
                self.assertEqual(said.rstrip(".").count(". "), 0, said)

    def test_a_preset_that_still_writes_it_is_refused_with_the_sentence(self) -> None:
        from pydantic import ValidationError

        from src.robot.grasping.replay import presets

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "old.yaml").write_text(
                yaml.safe_dump({"grasping": {"verification": {"enabled": True}}}), encoding="utf-8")
            with mock.patch.object(presets, "_PRESETS_DIR", Path(tmp)):
                with self.assertRaises(ValidationError) as caught:
                    presets.validate_preset("old", base={"vendor": "dummy", "gripper": {"vendor": "none"}})
        said = " ".join(str(caught.exception).split())
        self.assertEqual(said.count("removed on purpose"), 1)
        self.assertIn(REMOVED_KEYS[_BLOCK], said)

    def test_explain_answers_each_key_with_the_sentence(self) -> None:
        from src.config.explain import explain

        root = Path(__file__).resolve().parents[1] / "config"
        for dotted in (_BLOCK, *(f"{_BLOCK}.{key}" for key in _KEYS)):
            with self.subTest(key=dotted):
                answer = explain(dotted, root, ())
                self.assertFalse(answer.known)
                self.assertEqual(answer.removed, REMOVED_KEYS[dotted])
                self.assertEqual(answer.suggestions, ())

    def test_the_unwired_list_keeps_its_other_entry_and_refuses_it_as_before(self) -> None:
        self.assertEqual(RobotGraspingConfig.UNWIRED_SWITCHES, {"occlusion.hard_reject_enabled": "occlusion"})
        self.assertEqual(set(RobotGraspingConfig.UNWIRED_REASONS), {"occlusion"})
        with self.assertRaises(ValueError) as caught:
            RobotGraspingConfig.model_validate({"occlusion": {"hard_reject_enabled": True}})
        self.assertIn("occlusion.hard_reject_enabled is set", str(caught.exception))
        self.assertNotIn("verification", str(caught.exception))


# ---------------------------------------------------------------------------------------------------
# The replacement path: the execution policy reads the gripper after its close
# ---------------------------------------------------------------------------------------------------


def _robotiq(obj: int) -> Any:
    """The real Robotiq ``GripperController`` on a socket seam that reports gOBJ ``obj``.

    1 or 2 is a stall on something (held), 3 is the fingers at their target (empty).
    """
    from src.config.schema.robot import GripperConfig
    from tests.test_gripper_commands_the_number_it_promised import _gripper

    gripper, driver = _gripper(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
    driver.obj = obj
    return gripper


def _service(gripper: Any) -> Any:
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Calculator, _Frames

    arm = _Arm([])
    return AutonomousGraspService.from_components(
        arm=arm,  # type: ignore[arg-type]
        calculator=_Calculator(),  # type: ignore[arg-type]
        perception=_Frames(),
        mode=GraspMode.EASY,
        gripper=gripper,
        policy=GraspExecutionPolicy(arm=arm, gripper=gripper, pre_open_width_mm=50.0),  # type: ignore[arg-type]
    )


class ARobotiqHoldIsReadAfterTheCloseTests(unittest.TestCase):
    """The one check a pick runs on a hold, pinned on the driver the owner's cell ships with."""

    def test_the_policy_reports_a_stall_as_held_and_an_arrival_as_empty(self) -> None:
        from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _grasp

        for obj in (1, 2):
            with self.subTest(gOBJ=obj):
                held = GraspExecutionPolicy(arm=_Arm([]), gripper=_robotiq(obj)).execute(_grasp())
                self.assertIs(PolicyOutcome.EXECUTED, held.outcome)
                self.assertIs(True, held.object_detected)
        empty = GraspExecutionPolicy(arm=_Arm([]), gripper=_robotiq(3)).execute(_grasp())
        self.assertIs(PolicyOutcome.OBJECT_NOT_DETECTED, empty.outcome)
        self.assertIs(False, empty.object_detected)

    def test_the_service_reports_a_held_part_as_a_measured_success(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome

        report = _service(_robotiq(2)).pick()

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome)
        self.assertIs(True, report.pick_report.object_detected)
        self.assertIs(True, report.hold_measured)

    def test_the_service_reports_an_empty_close_as_verification_failed(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from src.robot.grasping.loop.pick_loop import PickOutcome

        report = _service(_robotiq(3)).pick()

        self.assertIs(AutonomousGraspOutcome.VERIFICATION_FAILED, report.outcome)
        self.assertIs(PickOutcome.OBJECT_NOT_DETECTED, report.pick_report.outcome)
        self.assertIs(False, report.pick_report.object_detected)

    def test_both_records_audit_with_no_verification_block(self) -> None:
        from src.robot.execution.autonomous_grasp.record_logging import to_attempt_record
        from src.robot.grasping.replay.telemetry_catalog import audit_record

        for obj, outcome in ((2, "succeeded"), (3, "verification_failed")):
            with self.subTest(outcome=outcome):
                record = to_attempt_record(_service(_robotiq(obj)).pick(), attempt_id=f"after-{outcome}")
                self.assertEqual(record.final_outcome, outcome)
                self.assertIsNone(record.verification)
                self.assertEqual(audit_record(record), ())
                assert record.profile is not None
                self.assertIs(record.profile["verification_enabled"], False)


class OldRecordsStillAuditTests(unittest.TestCase):
    """A record logged before the removal carries the verifier's block; the replay layer must still read it."""

    _BLOCK = {"outcome": "failed", "reason": "child_failed:jaws_collapsed_to_minimum",
              "telemetry": {"verifier": "composite", "rule": "all_must_pass"}}

    def _record(self, final_outcome: str, *, verification: "dict | None", extra: "dict | None" = None) -> Any:
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        return GraspAttemptRecord(timestamp=1.0, attempt_id="before-2026-09-29", mode="auto",
                                  final_outcome=final_outcome, execution={"outcome": "executed"},
                                  verification=verification, extra=extra or {})

    def test_a_record_with_the_block_audits_round_trips_and_classifies(self) -> None:
        from src.robot.grasping.replay.failure_taxonomy import FailureRootCause, classify_record
        from src.robot.grasping.replay.telemetry_catalog import audit_record
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        record = self._record("verification_failed", verification=dict(self._BLOCK),
                              extra={"empty_air_evidence": True})
        self.assertEqual(audit_record(record), ())
        again = GraspAttemptRecord.from_dict(record.to_dict())
        self.assertEqual(again, record)
        self.assertEqual(again.verification, self._BLOCK)
        self.assertIs(classify_record(again).primary, FailureRootCause.EMPTY_AIR_GRASP)

    def test_the_block_is_no_longer_required(self) -> None:
        from src.robot.grasping.replay.telemetry_catalog import TELEMETRY_CATALOG, audit_record

        for outcome in ("succeeded", "verification_failed"):
            with self.subTest(outcome=outcome):
                self.assertNotIn("verification", TELEMETRY_CATALOG[outcome])
                self.assertEqual(audit_record(self._record(outcome, verification=None)), ())
                self.assertEqual(audit_record(self._record(outcome, verification=dict(self._BLOCK))), ())


class NothingVerifiesAfterThePolicyTests(unittest.TestCase):

    def test_the_package_and_its_names_are_gone(self) -> None:
        for dotted in ("src.robot.grasping.closed_loop", "src.robot.grasping.closed_loop.verification"):
            with self.subTest(module=dotted):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(dotted)
        grasping = importlib.import_module("src.robot.grasping")
        builders = importlib.import_module("src.robot.execution.autonomous_grasp.builders")
        outcome_logging = importlib.import_module("src.robot.grasping.telemetry.outcome_logging")
        schema = importlib.import_module("src.config.schema.robot")
        for module, name in (
            (grasping, "GraspVerifier"), (grasping, "GraspVerificationPolicy"), (grasping, "NoOpVerifier"),
            (grasping, "ObjectDetectingGripperVerifier"), (grasping, "WidthDeltaGripperVerifier"),
            (grasping, "CompositeGraspVerifier"), (grasping, "VerificationOutcome"),
            (grasping, "verification_metadata_from"), (outcome_logging, "verification_metadata_from"),
            (builders, "build_subpolicies"), (builders, "build_closed_loop_actors"),
            (builders, "assert_closed_loop_actors_wired"), (schema, "GraspingVerificationConfig"),
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(module, name))

    def test_no_service_takes_or_carries_a_verifier(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService

        fields = {f.name for f in dataclasses.fields(AutonomousGraspService)}
        self.assertFalse({"verifier", "verification_policy"} & fields)
        self.assertFalse(hasattr(AutonomousGraspService, "_run_post_grasp_verification"))
        for factory in (AutonomousGraspService.from_components, AutonomousGraspService.from_robot_config):
            with self.subTest(factory=factory.__qualname__):
                self.assertFalse({"verifier", "verification_policy"} & set(inspect.signature(factory).parameters))

    def test_the_profile_and_the_snapshot_forgot_it(self) -> None:
        from src.robot.execution.autonomous_grasp import EffectiveGraspingConfig, GraspBehaviorProfile

        self.assertNotIn("verification", RobotGraspingConfig.model_fields)
        self.assertNotIn("verification_enabled", {f.name for f in dataclasses.fields(GraspBehaviorProfile)})
        self.assertNotIn("verification_enabled", {f.name for f in dataclasses.fields(EffectiveGraspingConfig)})


if __name__ == "__main__":
    unittest.main()
