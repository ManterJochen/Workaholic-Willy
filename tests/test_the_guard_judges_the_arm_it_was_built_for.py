"""The safety preflight names the arm it was built for, and its guard judges no other (P0e E6).

The hand refusal is decided when the pipeline is built, for one arm: whether a hand is named, where it sits, which hand
geometry the exact mesh guard reads. Until P0e a pipeline built with no arm model decided nothing, and its guard, handed
an arm that reports a UR, judged the arm bundle's own hand, with no "robot.gripper.model is unset" anywhere. Every
builder now names its arm (``arm_model`` is required, ``None`` for an arm the registry does not describe), and the self
collision guard refuses an arm that reports another UR than the one it was built for.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.config.arms import admitted_arms
from src.config.schema.robot import RobotConfig, WorkspaceLimitsConfig
from src.contracts import UNSET
from src.robot.core import JointPositions, MotionCommand, RobotCapabilities
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety import SafetyReason
from src.robot.safety.planning.hand import planner_hand
from src.robot.safety.preflight import SafetyPreflight

_JOINTS = JointPositions([0.1, -1.4, 1.6, -1.77, -1.57, 0.2])


def _cell(model: str = "ur5e") -> RobotConfig:
    return RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.1.2.3", "model": model},
                                       "gripper": {"model": "robotiq_2f85"}})


def _reporting(model: str) -> SimpleNamespace:
    """An arm double that reports ``model`` and nothing else."""
    return SimpleNamespace(capabilities=RobotCapabilities(vendor="ur", model=model, dof=6, controller=None,
                                                          owns_a_controller=True))


def _built_for(model: str | None) -> SafetyPreflight:
    cell = _cell()
    return SafetyPreflight.from_safety_config(cell.safety, cell.workspace_limits, hand=planner_hand(cell),
                                              arm_model=model)


def _self_collision(preflight: SafetyPreflight, arm: object):  # noqa: ANN202
    ctx = preflight.context_for_joints(_JOINTS, command=MotionCommand.MOVE_JOINTS, arm=arm)  # type: ignore[arg-type]
    guard = next(g for g in preflight.guards if g.name == "self_collision")
    return guard.evaluate(ctx)


class EveryArmIsJudgedByThePipelineBuiltForItTests(unittest.TestCase):
    def test_every_admitted_ur_builds_its_preflight_for_the_model_it_reports(self) -> None:
        """The agreement, from the registry: the arm and its pipeline name one arm, and the guard judges it."""
        for model in admitted_arms():
            with self.subTest(model=model):
                arm = URRobotArm(_cell(model))
                preflight = arm.safety_preflight
                assert preflight is not None
                self.assertEqual(preflight.arm_model, arm.capabilities.model)
                guard = next(g for g in preflight.guards if g.name == "self_collision")
                self.assertIsNone(guard.refusal_for(arm))  # type: ignore[attr-defined]

    def test_the_sim_builder_names_the_arm_the_isaac_arm_reports(self) -> None:
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.willy_sim.config import load_sim_config, sim_driver_config, sim_safety_preflight

        for model in ("ur5e", "ur3e"):
            with self.subTest(model=model):
                cfg = load_sim_config(robot_model=model)
                self.assertEqual(sim_safety_preflight(cfg).arm_model, IsaacRobotArm(sim_driver_config(cfg)).capabilities.model)


class AnArmThatReportsAnotherUrIsRefusedTests(unittest.TestCase):
    def test_a_ur5e_pipeline_refuses_to_judge_a_ur3e(self) -> None:
        """The control: built for one UR and handed another, the guard refuses rather than judging the other."""
        decision = _self_collision(_built_for("ur5e"), _reporting("ur3e"))
        self.assertFalse(decision.accepted)
        self.assertIs(decision.reason, SafetyReason.UNAVAILABLE)
        self.assertIn("built for ur5e and the arm reports ur3e", decision.message)

    def test_the_same_pipeline_judges_the_arm_it_was_built_for(self) -> None:
        self.assertIsNot(_self_collision(_built_for("ur5e"), _reporting("ur5e")).reason, SafetyReason.UNAVAILABLE)

    def test_a_pipeline_built_for_no_ur_refuses_an_arm_that_reports_one(self) -> None:
        """The latent gap itself: no arm named at build, a UR reported at the motion."""
        cell = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.1.2.3"}})
        preflight = SafetyPreflight.from_safety_config(cell.safety, cell.workspace_limits, arm_model=None)
        decision = _self_collision(preflight, _reporting("ur5e"))
        self.assertIs(decision.reason, SafetyReason.UNAVAILABLE)
        self.assertIn("no arm the arm registry describes as a UR", decision.message)

    def test_an_arm_that_reports_no_ur_is_not_refused_here(self) -> None:
        """It gets no UR geometry at all, as before: the dummy's model is nobody's UR."""
        guard = next(g for g in _built_for("ur5e").guards if g.name == "self_collision")
        dummy = SimpleNamespace(capabilities=RobotCapabilities(vendor="dummy", model="dummy-6dof", dof=6,
                                                               controller=None, owns_a_controller=False))
        self.assertIsNone(guard.refusal_for(dummy))  # type: ignore[attr-defined]

    def test_a_path_on_the_wrong_arm_is_refused_before_its_first_sample_with_the_reason(self) -> None:
        refusal = _built_for("ur5e").path_judge_refusal(_reporting("ur3e"))  # type: ignore[arg-type]
        self.assertIsNotNone(refusal)
        self.assertIn("arm reports ur3e", str(refusal))

    def test_a_pipeline_assembled_from_guards_alone_is_not_built_for_an_arm(self) -> None:
        self.assertIs(SafetyPreflight([]).arm_model, UNSET)


class TheModelIsRequiredTests(unittest.TestCase):
    def test_a_builder_must_name_the_arm(self) -> None:
        with self.assertRaises(TypeError):
            SafetyPreflight.from_safety_config(RobotConfig().safety, WorkspaceLimitsConfig())  # type: ignore[call-arg]


class ADriverReadsTheModelOffItsOwnIdentityTests(unittest.TestCase):
    """Rule, by AST: inside ``src/robot/drivers`` a pipeline is built for ``capabilities.model``, the arm's own
    identity, so the pipeline and the arm cannot be built from two different sources."""

    def test_every_driver_builds_its_preflight_for_its_capabilities_model(self) -> None:
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        found, wrong = 0, []
        for path in sorted((root / "src" / "robot" / "drivers").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "from_safety_config"):
                    found += 1
                    passed = [ast.unparse(k.value) for k in node.keywords if k.arg == "arm_model"]
                    if not passed or "capabilities.model" not in passed[0]:
                        wrong.append(f"{path.relative_to(root).as_posix()}:{node.lineno} arm_model={passed}")
        self.assertGreaterEqual(found, 1, "the walk found no driver building a preflight")
        self.assertEqual([], wrong)

    def test_the_rule_sees_a_driver_that_reads_the_tree(self) -> None:
        """The positive control: the UR arm before P0e E4 read ``config.ur.model``, and the rule must say so."""
        import ast

        call = ast.parse("SafetyPreflight.from_safety_config(s, w, arm_model=config.ur.model)").body[0].value
        assert isinstance(call, ast.Call)
        self.assertNotIn("capabilities.model", ast.unparse(call.keywords[0].value))


if __name__ == "__main__":
    unittest.main()
