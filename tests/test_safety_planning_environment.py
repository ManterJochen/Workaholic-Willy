"""Tests for the external-engine anchor: safety.planning.environment + the --check CLI.

The engines themselves (cuRobo GPU env, Coal) are not present in CI, so these lock the pure
logic: env-var resolution, the typed status snapshots, the fully-anchored predicate, and the CLI
exit code — with the probes / interpreter presence patched so the outcome is deterministic.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.robot.safety.planning import (
    CollisionEngineStatus,
    CuroboStatus,
    PlanningEnvironment,
    probe_planning_environment,
)
from src.robot.safety.planning import environment as env
from src.robot.safety.planning import __main__ as cli
from src.robot.safety.planning import stack as stack_mod


class EnvVarResolutionTests(unittest.TestCase):
    def test_env_var_names_are_stable(self) -> None:
        # These names are the public contract with operators + the sidecar; keep them stable.
        self.assertEqual(env.ENV_CUROBO_PYTHON, "WILLY_CUROBO_PYTHON")
        self.assertEqual(env.ENV_CUROBO_ROBOT, "WILLY_CUROBO_ROBOT")
        self.assertEqual(env.ENV_COAL_PREFIX, "WILLY_COAL_PREFIX")

    def test_robot_config_default_and_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env.curobo_robot_config(), "ur5e.yml")
        with mock.patch.dict(os.environ, {"WILLY_CUROBO_ROBOT": "custom.yml"}):
            self.assertEqual(env.curobo_robot_config(), "custom.yml")

    def test_curobo_env_available_tracks_python_presence(self) -> None:
        with mock.patch.dict(os.environ, {"WILLY_CUROBO_PYTHON": "/no/such/python.exe"}):
            self.assertFalse(env.curobo_env_available())
        # Point at a file that certainly exists (this test module) -> available.
        with mock.patch.dict(os.environ, {"WILLY_CUROBO_PYTHON": __file__}):
            self.assertTrue(env.curobo_env_available())

    def test_collision_mesh_bundle_names(self) -> None:
        self.assertEqual(env.collision_mesh_bundle().name, "ur5e_collision_meshes.npz")
        self.assertEqual(
            env.collision_mesh_bundle("ur5e", "schunk_egu50").name,
            "schunk_egu50_collision_meshes.npz",
        )


class StatusSnapshotTests(unittest.TestCase):
    def test_collision_available_requires_engine_and_bundle(self) -> None:
        self.assertTrue(CollisionEngineStatus("coal", True, None).available)
        self.assertFalse(CollisionEngineStatus("coal", False, None).available)  # no bundle
        self.assertFalse(CollisionEngineStatus(None, True, None).available)      # no engine

    def test_fully_anchored_is_the_and_of_both_engines(self) -> None:
        up = CuroboStatus("/py", True, "ur5e.yml")
        down = CuroboStatus("/py", False, "ur5e.yml")
        coal = CollisionEngineStatus("coal", True, None)
        nofcl = CollisionEngineStatus(None, False, None)
        self.assertTrue(PlanningEnvironment(up, coal).fully_anchored)
        self.assertFalse(PlanningEnvironment(down, coal).fully_anchored)
        self.assertFalse(PlanningEnvironment(up, nofcl).fully_anchored)

    def test_render_names_both_engines(self) -> None:
        """⚠ THIS USED TO CALL `.report()`, the deprecated alias, which is now gone. The alias
        outlived its purpose the moment its last caller moved: it existed for
        `willy_sim/harness/bootstrap.py:86` while that tree was out of scope, and a re-export whose
        expiry nobody writes down is how one name quietly becomes two addresses.
        """
        text = PlanningEnvironment(
            CuroboStatus("/py", True, "ur5e.yml"), CollisionEngineStatus("coal", True, None)
        ).render()
        self.assertIn("cuRobo planner", text)
        self.assertIn("collision engine", text)
        self.assertIn("fully anchored", text)
        self.assertFalse(
            hasattr(PlanningEnvironment, "report"),
            "the deprecated alias came back; render() is the only name",
        )

    def test_probe_returns_typed_snapshots(self) -> None:
        result = probe_planning_environment()
        self.assertIsInstance(result, PlanningEnvironment)
        self.assertIsInstance(result.curobo, CuroboStatus)
        self.assertIsInstance(result.collision, CollisionEngineStatus)
        self.assertIsInstance(result.fully_anchored, bool)


class CheckCliTests(unittest.TestCase):
    """The CLI's exit-code rule, exercised through a faked probe.

    ⚠ THE PATCH TARGET MOVED FROM THE CLI TO `stack`, AND THAT IS THE POINT OF THE MOVE. These tests
    used to patch `cli.probe_planning_environment`, which pinned the CLI's own import list rather
    than its behaviour: the day the resolution moved into `MotionStack` they failed with an
    `AttributeError` about a name, not about an exit code. `MotionStack.probe` is where the reading
    is taken now, so that is where a fake belongs, and the assertions below are unchanged.
    """

    def test_exit_zero_when_fully_anchored(self) -> None:
        anchored = PlanningEnvironment(
            CuroboStatus("/py", True, "ur5e.yml"), CollisionEngineStatus("coal", True, None)
        )
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=anchored):
            self.assertEqual(cli.main(["--check"]), 0)

    def test_exit_one_when_partially_anchored(self) -> None:
        partial = PlanningEnvironment(
            CuroboStatus("/py", False, "ur5e.yml"), CollisionEngineStatus(None, False, None)
        )
        with mock.patch.object(stack_mod, "probe_planning_environment", return_value=partial):
            self.assertEqual(cli.main([]), 1)  # --check is the default action


if __name__ == "__main__":
    unittest.main()
