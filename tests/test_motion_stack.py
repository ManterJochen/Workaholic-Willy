"""The motion-stack reading, and the robot it is about.

Every assertion here corresponds to a measured defect, not to a shape I liked.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.robot.safety.planning import stack as stack_mod
from src.robot.safety.planning.environment import (
    CollisionEngineStatus,
    CuroboStatus,
    PlanningEnvironment,
)
from src.robot.safety.planning.stack import ModelSource, MotionStack

ANCHORED = PlanningEnvironment(
    CuroboStatus("/py", True, "ur5e.yml"), CollisionEngineStatus("coal", True, None)
)


class _Model:
    def __init__(self, value: object) -> None:
        self.model = value
        self.kinematics_model = value


class _Safety:
    def __init__(self, kinematics_model: object = None) -> None:
        self.self_collision = _Model(kinematics_model) if kinematics_model else None


class _Robot:
    """The three fields the ladder reads, and nothing else."""

    def __init__(self, *, kinematics_model: object = None, ur_model: object = None) -> None:
        self.safety = _Safety(kinematics_model)
        self.ur = _Model(ur_model) if ur_model else None


class ModelResolutionTests(unittest.TestCase):
    def test_the_self_collision_model_outranks_the_vendor_block(self) -> None:
        """The guard's own model is authoritative, because it is what the guard keys its bundle on."""
        s = MotionStack.from_robot_config(_Robot(kinematics_model="ur3e", ur_model="ur5e"))
        self.assertEqual(s.model, "ur3e")
        self.assertIs(s.source, ModelSource.SELF_COLLISION)
        self.assertEqual(s.model_source, "robot.safety.self_collision.kinematics_model")

    def test_the_vendor_block_answers_when_the_guard_does_not(self) -> None:
        s = MotionStack.from_robot_config(_Robot(ur_model="ur3e"))
        self.assertEqual(s.model, "ur3e")
        self.assertIs(s.source, ModelSource.VENDOR_BLOCK)

    def test_a_config_declaring_nothing_says_so_instead_of_pretending(self) -> None:
        """⛔⛔ THE DEFECT THIS MODULE EXISTS FOR. `probe_planning_environment()` with no arguments
        returns a green reading computed against ur5e and cannot say that it did. Falling back is
        correct; falling back SILENTLY is the false green.
        """
        s = MotionStack.from_robot_config(_Robot())
        self.assertEqual(s.model, "ur5e")
        self.assertIs(s.source, ModelSource.FALLBACK)
        self.assertEqual(s.model_source, "default (no model declared in this config)")

    def test_an_explicit_model_wins_over_the_config(self) -> None:
        s = MotionStack.from_robot_config(_Robot(ur_model="ur5e"), model="ur10e")
        self.assertEqual(s.model, "ur10e")
        self.assertIs(s.source, ModelSource.CALLER)


class ReadingTests(unittest.TestCase):
    def test_the_reading_probes_for_the_resolved_model_not_for_ur5e(self) -> None:
        """⛔ THE ASSERTION THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT. Both engine probes are
        per-robot: the bundle ships as `{model}_collision_meshes.npz` and the descriptor as
        `{model}.yml`, so a reading that quietly probes ur5e for a UR3e cell is green about the
        wrong robot.
        """
        with mock.patch.object(
            stack_mod, "probe_planning_environment", return_value=ANCHORED
        ) as probe:
            MotionStack.from_robot_config(_Robot(ur_model="ur3e")).probe()
        probe.assert_called_once_with(robot_config="ur3e.yml", kinematics_model="ur3e")

    def test_render_is_the_whole_object_including_the_model_line(self) -> None:
        """⭐ The CLI used to append the model line itself, so `render()` returned a fragment and the
        one fact that makes the reading actionable lived outside the object.
        """
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=ANCHORED):
            text = MotionStack.from_robot_config(_Robot(ur_model="ur3e")).probe().render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertIn("(model: ur3e, from robot.ur.model)", text)
        self.assertIn("fully anchored", text)

    def test_exit_code_is_on_the_report(self) -> None:
        partial = PlanningEnvironment(
            CuroboStatus("/py", False, "ur5e.yml"), CollisionEngineStatus(None, False, None)
        )
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=ANCHORED):
            self.assertEqual(MotionStack.from_robot_config(_Robot(ur_model="ur3e")).probe().exit_code, 0)
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=partial):
            self.assertEqual(MotionStack.from_robot_config(_Robot(ur_model="ur3e")).probe().exit_code, 1)

    def test_to_dict_carries_the_caveat_that_available_is_not_working(self) -> None:
        """A consumer reading `available` as "the planner will work" is wrong, and the sentence that
        says so used to live in two CLI bodies where no consumer could reach it."""
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=ANCHORED):
            payload = MotionStack.from_robot_config(_Robot(ur_model="ur3e")).probe().to_dict()
        self.assertEqual(payload["model"], "ur3e")
        self.assertEqual(payload["model_source"], "robot.ur.model")
        self.assertIn(
            "the robot descriptor inside it is not verified from here",
            payload["curobo"]["available_means"],
        )
        import json

        json.dumps(payload)  # no custom encoder, or this raises


class OneLadderTests(unittest.TestCase):
    def test_neither_the_cli_nor_the_console_resolves_the_model_itself(self) -> None:
        """⛔ THE DUPLICATE IS WHAT DIVERGED. The three-step ladder was written out in
        `safety/planning/__main__.py` and again in `api/routers/diagnostics.py`, and the two already
        disagreed: bare "default" against "default (no model declared in this config)". This asserts
        neither file names the config keys any more, which is the cheapest way to see a third copy
        appear.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in (
            "src/robot/safety/planning/__main__.py",
            "api/routers/diagnostics.py",
        ):
            body = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn(
                'getattr(getattr(robot, "ur", None), "model", None)',
                body,
                f"{rel} resolves the model itself again; MotionStack owns that ladder",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
