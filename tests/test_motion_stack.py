"""The motion-stack reading, and the robot it is about.

Every assertion here corresponds to a measured defect, not to a shape I liked.
"""

from __future__ import annotations

import contextlib
import tempfile
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


class ConfigThatDidNotLoadTests(unittest.TestCase):
    """⛔⛔ "FULLY ANCHORED", EXIT 0, ABOUT A CELL NOTHING IS KNOWN ABOUT.

    `for_this_box` tolerates a config tree that does not load, deliberately: "the config does not
    load" is one of the things an operator is standing there to diagnose. What it hands back is the
    ur5e FALLBACK, and `exit_code` reads only `fully_anchored`, which on any box with the engines
    installed is True for ur5e. So `--check --data <a tree that does not load>` prints
    "=> fully anchored" and exits 0, and the one line that says otherwise is a parenthesis at the
    end of the last line.
    """

    @contextlib.contextmanager
    def _reading_for_an_unreadable_tree(self):
        """The reading `--check --data <tree>` takes when that tree does not load."""
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(stack_mod, "probe_planning_environment", return_value=ANCHORED):
            # An empty directory: `load_config` raises `ConfigError` on it, which is the same door a
            # syntactically broken YAML tree comes through.
            yield MotionStack.for_this_box(data_dir=tmp)

    def test_a_tree_that_does_not_load_is_not_an_anchored_cell(self) -> None:
        with self._reading_for_an_unreadable_tree() as stack:
            self.assertIs(stack.source, ModelSource.FALLBACK)
            report = stack.probe()
            self.assertEqual(report.exit_code, 1,
                             f"exit 0 for a cell nothing is known about: {report.render()}")

    def test_the_reading_says_out_loud_that_it_is_about_the_fallback(self) -> None:
        """The LAST line is what an operator is left with, and it was `=> fully anchored` plus a
        parenthesis. The engines line stays true (they ARE present, for ur5e); what it may not be is
        the last word, so the retraction has to come after it and name the fallback."""
        with self._reading_for_an_unreadable_tree() as stack:
            text = stack.probe().render()
        last = text.splitlines()[-1]
        self.assertIn("not a reading about this cell", last)
        self.assertIn("ConfigError", last)
        self.assertIn("ur5e", last)
        self.assertLess(text.index("=> fully anchored"), text.index("not a reading"))

    def test_the_payload_carries_the_refusal_for_the_console(self) -> None:
        with self._reading_for_an_unreadable_tree() as stack:
            payload = stack.probe().to_dict()
        self.assertTrue(payload["config_error"])

    def test_a_tree_that_loads_is_untouched_by_any_of_this(self) -> None:
        """The shipped tree loads and declares a robot; that reading stays exactly what it was."""
        s = MotionStack.for_this_box()
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=ANCHORED):
            report = s.probe()
        self.assertEqual(report.exit_code, 0)
        self.assertIn("=> fully anchored", report.render())
        self.assertEqual(report.to_dict()["config_error"], "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
