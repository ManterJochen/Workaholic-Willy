"""No viewpoint is planned: the viewpoint planners, the relocate path and MOVE_CAMERA are gone.

Removed on 2026-09-29 (cleanup phase 2, owner-approved 2026-09-28): the `ViewpointPlanner` Protocol, the
`LateralOffsetViewpointPlanner` and `ScoringViewpointPlanner` with the rest of `closed_loop/active_perception.py`,
the pick loop's relocate path (`PickOutcome.RELOCATED_EXHAUSTED`, `PickReport.viewpoints_visited`), and the
decision gate's `MOVE_CAMERA` with its budget and the two reason codes only it used (`low_confidence` beside it,
`reobserve_budget_exhausted`). No config had built a planner since the multi-view commit gate left on 2026-09-28,
so a config-built cell already rescanned where it would have relocated and fail-closed, with
`reobserve_planner_unavailable`, where it would have re-observed. That verdict carries `low_confidence` since
2026-09-29 (owner decision), and `reobserve_planner_unavailable` is retired with the planner it named. Another view
comes from the look poses a program hands a pick (`src/robot/execution/looks.py`) or from a second camera.

What must hold after the removal, and is pinned here:

* a tree that still writes one of the two removed keys is refused at load with the sentence that replaces it;
* a record logged before the removal still audits, whatever retired string it carries;
* the two reasons that asked for a camera move are rescanned, without motion;
* nothing that took a viewpoint planner still accepts one.
"""

from __future__ import annotations

import importlib
import inspect
import unittest

import yaml

from src.config import ConfigError
from src.config.loader import load_robot_section
from src.config.schema._removed import REMOVED_KEYS
from src.config.schema.robot.grasping_schema import GraspingDecisionConfig, RobotGraspingFusionConfig
from tests.test_section_loaders import _ScratchTree

_REMOVED = {
    "robot.grasping.decision.max_reobservations": ("decision", "max_reobservations", 2),
    "robot.grasping.fusion.active_perception_use_fusion": ("fusion", "active_perception_use_fusion", False),
}


class TheRemovedKeysAreRefusedAtLoadTests(_ScratchTree):
    """Loads a copy of the shipped tree that still writes the key, as an operator's old file would."""

    def _refusal(self, **blocks: dict) -> str:
        path = self.root / "robot" / "robot.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        grasping = data["robot"].setdefault("grasping", {})
        for block, values in blocks.items():
            grasping.setdefault(block, {}).update(values)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_robot_section(self.root, profile=None)
        return str(caught.exception)

    def _assert_refused_with_its_sentence(self, dotted: str) -> None:
        block, key, value = _REMOVED[dotted]
        message = self._refusal(**{block: {key: value}})
        self.assertIn(dotted, message)
        self.assertEqual(message.count("removed on purpose"), 1)
        self.assertIn(REMOVED_KEYS[dotted], " ".join(message.split()))

    def test_the_reobservation_budget_is_refused_with_its_sentence(self) -> None:
        self._assert_refused_with_its_sentence("robot.grasping.decision.max_reobservations")

    def test_the_planner_fusion_switch_is_refused_with_its_sentence(self) -> None:
        self._assert_refused_with_its_sentence("robot.grasping.fusion.active_perception_use_fusion")

    def test_both_at_once_are_both_named(self) -> None:
        message = self._refusal(decision={"max_reobservations": 1}, fusion={"active_perception_use_fusion": True})
        self.assertEqual(message.count("removed on purpose"), 2)
        self.assertIn("robot.grasping.decision.max_reobservations", message)
        self.assertIn("robot.grasping.fusion.active_perception_use_fusion", message)

    def test_the_tree_without_them_still_loads(self) -> None:
        self._assert_loads(load_robot_section)


class TheSentencesSayWhatToDoTests(unittest.TestCase):

    def test_each_key_is_named_and_gone_from_the_schema(self) -> None:
        models = {"decision": GraspingDecisionConfig, "fusion": RobotGraspingFusionConfig}
        for dotted, (block, key, _value) in _REMOVED.items():
            with self.subTest(key=dotted):
                self.assertIn(dotted, REMOVED_KEYS)
                self.assertNotIn(key, models[block].model_fields)

    def test_each_sentence_is_one_sentence_that_says_delete_the_key(self) -> None:
        for dotted in _REMOVED:
            with self.subTest(key=dotted):
                said = REMOVED_KEYS[dotted]
                self.assertIn("delete this key", said)
                self.assertIn("2026-09-29", said)
                self.assertTrue(said.endswith("."))
                self.assertEqual(said.rstrip(".").count(". "), 0, said)


class OldRecordsStillAuditTests(unittest.TestCase):
    """A record logged before the removal carries strings no enum names any more; the replay layer must read it."""

    @staticmethod
    def _record(final_outcome: str, extra: dict):  # noqa: ANN205
        from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

        return GraspAttemptRecord(timestamp=1.0, attempt_id="before-2026-09-29", mode="auto",
                                  final_outcome=final_outcome, extra=extra)

    def test_a_fail_closure_with_a_retired_reason_code_audits(self) -> None:
        from src.robot.grasping.replay.telemetry_catalog import audit_extra_record, audit_record

        for reason in ("reobserve_budget_exhausted", "reobserve_planner_unavailable"):
            with self.subTest(reason=reason):
                record = self._record("decision_fail_closed", {
                    "decision_action": "fail_closed", "decision_reason_code": reason,
                    "uncertainty_score": 0.7, "threshold_used": 0.4, "reobservation_count": 2,
                    "decision_max_reobservations": 2,
                })
                self.assertEqual(audit_record(record), ())
                self.assertEqual(audit_extra_record(record), ())

    def test_a_pick_the_relocate_path_ended_audits_and_classifies(self) -> None:
        from src.robot.grasping.replay.failure_taxonomy import FailureRootCause, classify_record
        from src.robot.grasping.replay.telemetry_catalog import audit_record

        record = self._record("no_valid_grasp", {
            "low_level_outcome": "PickOutcome.RELOCATED_EXHAUSTED", "occlusion_misread_evidence": True,
        })
        self.assertEqual(audit_record(record), ())
        self.assertIs(classify_record(record).primary, FailureRootCause.OCCLUSION_MISREAD)


class ThePickLoopRescansWhereItRelocatedTests(unittest.TestCase):

    def test_the_two_camera_move_reasons_are_rescan_reasons(self) -> None:
        from src.robot.grasping.loop.pick_loop import _RESCAN_REASONS
        from src.robot.grasping.types.feedback import GraspFailureReason

        self.assertIn(GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED, _RESCAN_REASONS)
        self.assertIn(GraspFailureReason.ALL_OUT_OF_WORKSPACE, _RESCAN_REASONS)

    def test_the_relocate_outcome_and_report_field_are_gone(self) -> None:
        from src.robot.grasping.loop.pick_loop import PickOutcome, PickReport

        self.assertNotIn("relocated_exhausted", {o.value for o in PickOutcome})
        self.assertNotIn("viewpoints_visited", PickReport.__dataclass_fields__)


class NothingTakesAViewpointPlannerTests(unittest.TestCase):

    def test_the_planner_module_and_its_names_are_gone(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("src.robot.grasping.closed_loop.active_perception")
        grasping = importlib.import_module("src.robot.grasping")
        pick_loop = importlib.import_module("src.robot.grasping.loop.pick_loop")
        builders = importlib.import_module("src.robot.execution.autonomous_grasp.builders")
        for module, name in (
            (grasping, "ScoringViewpointPlanner"), (grasping, "ViewScoringPolicy"), (grasping, "ViewpointHistory"),
            (pick_loop, "ViewpointPlanner"), (pick_loop, "LateralOffsetViewpointPlanner"),
            (pick_loop, "_RELOCATE_REASONS"), (builders, "build_config_viewpoint_planner"),
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(module, name))

    def test_no_constructor_accepts_one(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService
        from src.robot.execution.autonomous_grasp.builders import build_runtime
        from src.robot.execution.runtime_pick import RuntimePickService
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        self.assertNotIn("viewpoint_planner", BinPickingOrchestrator.__dataclass_fields__)
        for factory in (
            AutonomousGraspService.from_components, AutonomousGraspService.from_robot_config,
            RuntimePickService.from_components, RuntimePickService.from_robot_config, build_runtime,
        ):
            with self.subTest(factory=factory.__qualname__):
                self.assertNotIn("viewpoint_planner", inspect.signature(factory).parameters)

    def test_the_decision_engine_takes_no_planner_and_no_count(self) -> None:
        from src.robot.grasping.decision import DecisionEngine

        parameters = inspect.signature(DecisionEngine.decide).parameters
        self.assertNotIn("viewpoint_planner_available", parameters)
        self.assertNotIn("reobservation_count", parameters)


if __name__ == "__main__":
    unittest.main()
