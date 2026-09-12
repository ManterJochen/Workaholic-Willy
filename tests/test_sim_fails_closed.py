"""A sim cell configured for cuRobo either uses it or refuses, and never quietly runs something else.

The driver used to degrade. A cuRobo arm whose sidecar could not start latched a flag, logged one
warning and ran the blind IK path for the rest of its life. Measured 2026-08-10: an OS policy blocked
the sidecar's CUDA extension, the cell kept running, blind IK proposed configurations the self
collision guard correctly refused, and the run reported a low rate with nothing tying it back. The
diagnosis took an hour.

The runner did the same thing one layer up: `run_dense_pick` checked at build time whether the
environment was there and quietly built the `ik` path when it was not.

Both are gone. The environment is either there, or the run says so and stops. Asking for `ik`
explicitly is still allowed, because that is somebody deciding rather than something failing.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.robot.safety.planning import CuroboUnavailableError
from src.willy_sim.harness.bootstrap import (
    DegradedMotionStackError,
    resolve_runner_planner,
)


class TheRunnerPlannerIsDecidedNotDiscoveredTests(unittest.TestCase):
    """`resolve_runner_planner` is the one place a runner turns a request into a planner."""

    def test_curobo_without_the_environment_raises(self) -> None:
        with self.assertRaises(DegradedMotionStackError) as caught:
            resolve_runner_planner("curobo", env_available=False)
        message = str(caught.exception)
        self.assertIn("curobo", message)
        self.assertIn("ik", message, "the refusal does not say what the alternative would be")

    def test_curobo_with_the_environment_is_curobo(self) -> None:
        self.assertEqual("curobo", resolve_runner_planner("curobo", env_available=True))

    def test_ik_is_returned_whether_or_not_curobo_is_there(self) -> None:
        """Asking for ik is a decision, and a decision does not depend on what is installed."""
        for available in (True, False):
            with self.subTest(env_available=available):
                self.assertEqual("ik", resolve_runner_planner("ik", env_available=available))

    def test_another_planner_passes_through(self) -> None:
        self.assertEqual("rmpflow", resolve_runner_planner("rmpflow", env_available=False))


class TheBootstrapTakesThePlannerItIsGivenTests(unittest.TestCase):
    """A runner that resolved a planner hands it down, rather than leaving the config to disagree."""

    def test_the_gate_is_asked_about_the_planner_the_caller_chose(self) -> None:
        import src.willy_sim.harness.bootstrap as boot

        seen: list[str] = []

        def fake_require(planner, **kwargs):
            seen.append(planner)
            raise _StopBoot

        with patch.object(boot, "require_motion_stack", fake_require):
            with self.assertRaises(_StopBoot):
                boot.bootstrap_sim_cell(None, headless=True, motion_planner="ik")
        self.assertEqual(["ik"], seen, "the bootstrap read the config instead of the caller")

    def test_without_one_it_reads_the_config(self) -> None:
        """The control. Unset means the tree decides, which is what every runner did before."""
        import src.willy_sim.harness.bootstrap as boot

        seen: list[str] = []

        def fake_require(planner, **kwargs):
            seen.append(planner)
            raise _StopBoot

        with patch.object(boot, "require_motion_stack", fake_require):
            with self.assertRaises(_StopBoot):
                boot.bootstrap_sim_cell(None, headless=True)
        self.assertEqual(["curobo"], seen, "the shipped sim tree no longer asks for curobo")


class _StopBoot(RuntimeError):
    """Raised by the patched gate so the test never reaches Isaac."""


class TheDriverRefusesRatherThanDegradeTests(unittest.TestCase):
    """The driver half, on a bare arm with no Isaac behind it."""

    @staticmethod
    def _arm():
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig

        arm = IsaacRobotArm(SimRobotConfig(motion_planner="curobo", mock_mode=False))
        arm._get_curobo_client = MagicMock(  # type: ignore[method-assign]
            side_effect=CuroboUnavailableError("no cuRobo environment on this host")
        )
        return arm

    def test_the_planner_resolution_raises_rather_than_answering_ik(self) -> None:
        arm = self._arm()
        with self.assertRaises(CuroboUnavailableError):
            arm._resolve_motion_planner()

    def test_it_raises_again_on_the_second_call(self) -> None:
        """There is no latch, so a transient failure is not a permanent downgrade either."""
        arm = self._arm()
        for attempt in range(2):
            with self.subTest(attempt=attempt), self.assertRaises(CuroboUnavailableError):
                arm._resolve_motion_planner()
        self.assertEqual(2, arm._get_curobo_client.call_count)

    def test_an_ik_arm_never_touches_the_client(self) -> None:
        """The control: the refusal above is the configured planner, not the arm."""
        arm = self._arm()
        arm._motion_planner = "ik"
        self.assertEqual("ik", arm._resolve_motion_planner())
        self.assertEqual(0, arm._get_curobo_client.call_count)

    def test_mock_mode_is_untouched(self) -> None:
        """The mock has no sidecar and drives nothing, so there is nothing there to fail closed on."""
        from unittest.mock import MagicMock as _MagicMock

        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig

        arm = IsaacRobotArm(SimRobotConfig(motion_planner="curobo", mock_mode=True))
        arm._get_curobo_client = _MagicMock(  # type: ignore[method-assign]
            side_effect=CuroboUnavailableError("no cuRobo environment on this host")
        )
        self.assertEqual("curobo", arm._resolve_motion_planner())
        self.assertEqual(0, arm._get_curobo_client.call_count)


if __name__ == "__main__":
    unittest.main()
