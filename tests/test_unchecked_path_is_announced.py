"""A planned path executing between unexamined waypoints says so, once, where it becomes true.

⛔⛔ **FOUR SHIPPED DEFAULTS COMPOSE INTO AN UNGUARDED PATH.** `motion_planner: curobo` makes a real
move a multi-waypoint trajectory; `safety.trajectory_check.enabled: false` makes the trajectory gate
return before examining anything; `safety.planning_world.enabled: false` leaves the planner its own
generic 1.6 m table; and `safety.self_collision.fixtures: []` leaves the endpoint guard no bench, bin
or fixture. A plan that grazes a bin wall halfway and lands clear passes every check in the stack.

⚠ **THE FIX IS A SENTENCE, NOT A RULE, AND THE REASON IS MEASURED.** All three shipped profiles
(default, `sim`, `rl_datagen`) carry exactly that combination, so a config-time refusal would refuse
the tree this repository ships. That is the rule-too-sharp shape: a guard that fires on the default is
a bug. The YAML already states the consequence in prose, word for word, and nobody reads a YAML
comment at two in the morning with an arm in front of them. So it is said at runtime, at the moment a
path is actually about to execute unchecked.
"""

from __future__ import annotations

import logging
import unittest

from src.robot.safety.preflight import SafetyPreflight


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _preflight(*, check: bool) -> SafetyPreflight:
    return SafetyPreflight([], check_trajectories=check)


def _waypoints(n: int) -> list[tuple[float, ...]]:
    return [tuple(0.01 * i for _ in range(6)) for i in range(n)]


class AnUncheckedPathIsAnnouncedTests(unittest.TestCase):

    def _run(self, preflight: SafetyPreflight, waypoints: object, times: int = 1) -> list[str]:
        recorder = _Recorder()
        logger = preflight._logger
        logger.addHandler(recorder)
        previous = logger.level
        logger.setLevel(logging.WARNING)
        try:
            for _ in range(times):
                preflight.gate_trajectory(waypoints, arm=None)   # type: ignore[arg-type]
        finally:
            logger.removeHandler(recorder)
            logger.setLevel(previous)
        # De-shouted by the prose rewrite, same sentence: `preflight.py` emitted
        # "a %d-waypoint PLANNED PATH is executing UNEXAMINED between its endpoints:" before the
        # migration and emits "a %d-waypoint planned path is executing unexamined between its
        # endpoints:" now. The word that identifies the line is still the word the line is about.
        return [line for line in recorder.lines if "unexamined" in line]

    def test_a_planned_path_with_the_check_off_is_announced(self) -> None:
        said = self._run(_preflight(check=False), _waypoints(40))
        self.assertEqual(1, len(said), said)
        self.assertIn("40-waypoint", said[0])

    def test_it_names_all_three_keys_that_would_close_it(self) -> None:
        """⚠ ONE KEY WOULD BE HALF AN INSTRUCTION. Turning the check on judges the path; the
        fixtures and the planning world are what give the guard and the planner a cell to judge it
        against. An operator who flips only the first gets a path checked against nothing."""
        said = self._run(_preflight(check=False), _waypoints(12))[0]
        for key in ("trajectory_check.enabled",
                    "safety.self_collision.fixtures",
                    "safety.planning_world"):
            self.assertIn(key, said, f"the warning does not name {key}")

    def test_it_is_said_ONCE_however_many_paths_run(self) -> None:
        """⚠ A PICK LOOP PLANS CONTINUOUSLY. A line per motion is noise, and noise teaches an
        operator to filter the channel this appears in."""
        self.assertEqual(1, len(self._run(_preflight(check=False), _waypoints(9), times=25)))

    def test_a_SECOND_preflight_is_told_too(self) -> None:
        """The control on the latch. A module-level flag would silence every cell built after the
        first one in the same process, which is exactly the sim runners' shape."""
        first, second = _preflight(check=False), _preflight(check=False)
        self.assertEqual(1, len(self._run(first, _waypoints(5))))
        self.assertEqual(1, len(self._run(second, _waypoints(5))))

    def test_a_preflight_WITH_the_check_on_says_nothing(self) -> None:
        """The control. Without it the tests above pass for a preflight that always warns."""
        self.assertEqual([], self._run(_preflight(check=True), _waypoints(6)))

    def test_an_empty_path_says_nothing(self) -> None:
        """No waypoints is not an unexamined path, it is no path. Warning there would fire on every
        single-pose move in the repository."""
        self.assertEqual([], self._run(_preflight(check=False), []))


class TheShippedTreeIsWhyThisIsAWarningTests(unittest.TestCase):
    """⚠ MEASURED RATHER THAN ASSUMED, because it is the whole argument for not refusing. If a
    profile ever ships with the check ON, this test says so and the decision can be revisited."""

    def test_every_shipped_profile_carries_the_combination(self) -> None:
        from src.config.loader import load_robot_config

        for profile in (None, "sim", "rl_datagen"):
            with self.subTest(profile=profile):
                config = (load_robot_config(profile=profile) if profile
                          else load_robot_config())
                safety = config.safety
                self.assertFalse(safety.trajectory_check.enabled)
                self.assertFalse(safety.planning_world.enabled)
                self.assertEqual([], list(safety.self_collision.fixtures))

    def test_the_yaml_still_states_the_consequence_in_prose(self) -> None:
        """The warning replaces nothing: the comment is where an operator reading the config learns
        it, and the log line is where an operator watching a cell does."""
        from pathlib import Path

        yaml = (Path(__file__).resolve().parents[1]
                / "config/robot/robot.yaml").read_text(encoding="utf-8")
        self.assertIn("executes unexamined", yaml)


if __name__ == "__main__":
    unittest.main()
