"""A cuRobo joint plan runs cuRobo's own loop with no time-optimal pass after its first, on every cell.

cuRobo's ``MotionPlanner.plan_cspace`` runs, in every attempt, one optimiser pass that finds a trajectory and three more
that only shrink its time step towards cuRobo's own limits. The UR driver runs the positions of a plan with ``moveJ`` at
its own speed and never that timing, so every cell runs none of the three since 2026-10-17
(``_curobo_cspace.JOINT_FINETUNE_PASSES``; the owner's cell since the desk A/B of 2026-10-09, when
``safety.planned_motion.finetune_passes`` was 0 there). This file holds that on the CPU:

* ``plan_cspace`` is cuRobo's loop: cuRobo's own ``plan_cspace`` source, read from the pinned checkout and run against
  the same planner of this file's, makes the same calls in the same order with the same arguments at three passes;
  at any other number only ``finetune_attempts`` differs;
* the sidecar's ``plan_js`` branch, read as source: the copy at no pass after the first, after the seed is reset, and
  no variable that could ask for another number; ``WILLY_CUROBO_TIMING`` prints and changes nothing else;
* the config: the old key is refused, naming what every cell runs.
"""

from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path
from typing import Any

from src.robot.safety.planning._curobo_cspace import (
    CUROBO_FINETUNE_DT_SCALE,
    CUROBO_FINETUNE_PASSES,
    ENV_TIMING,
    JOINT_FINETUNE_PASSES,
    PlannedAttempts,
    plan_cspace,
    timing_asked,
)

_ROOT = Path(__file__).resolve().parents[4]
_SERVER = _ROOT / "src" / "robot" / "safety" / "planning" / "curobo_planner_server.py"
_PINNED = _ROOT / "ext_deps" / "curobo" / "curobo" / "_src" / "motion" / "motion_planner.py"


class _State:
    """A joint state as the loop handles it: cloned per attempt, its position viewed and repeated per seed."""

    def __init__(self, name: str, log: list) -> None:
        self.name, self.log = name, log
        self.ndim = 2
        self.position = self

    def clone(self) -> "_State":
        self.log.append(("clone", self.name))
        return _State(self.name, self.log)

    def view(self, *shape: int) -> "_State":
        self.log.append(("view", self.name, shape))
        return self

    def repeat(self, *times: int) -> str:
        self.log.append(("repeat", self.name, times))
        return f"{self.name} x {times[1]}"


class _Result:
    def __init__(self, success: bool) -> None:
        self.success = [success]
        self.total_time = 0.25
        self.solve_time = 0.125


class _Planner:
    """A planner that answers like cuRobo's: a graph seed or none per attempt, a result that succeeds or not."""

    def __init__(self, *, seeds: "list[bool]", successes: "list[bool]", graph: bool = True) -> None:
        self.log: list = []
        self._seeds, self._successes = list(seeds), list(successes)
        self.trajopt_solver = types.SimpleNamespace(config=types.SimpleNamespace(num_seeds=4),
                                                    solve_cspace=self._solve)
        self.graph_planner = object() if graph else None

    def _get_graph_seed_trajectories(self, state: _State, goal_configs: str) -> "str | None":
        self.log.append(("graph", state.name, goal_configs))
        return "graph seed" if self._seeds.pop(0) else None

    def _solve(self, goal: _State, state: _State, **kwargs: Any) -> _Result:
        self.log.append(("solve", goal.name, state.name, tuple(sorted(kwargs.items()))))
        return _Result(self._successes.pop(0))


def _curobo_loop() -> "types.FunctionType | None":
    """cuRobo's own ``MotionPlanner.plan_cspace``, read from the pinned checkout and compiled against a torch of one
    function; ``None`` where the tree carries no checkout."""
    if not _PINNED.is_file():
        return None
    source = _PINNED.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "MotionPlanner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "plan_cspace":
                    function = ast.Module(body=[item], type_ignores=[])
                    item.returns = None
                    for argument in item.args.args:
                        argument.annotation = None
                    namespace: dict[str, Any] = {
                        "torch": types.SimpleNamespace(count_nonzero=lambda success: sum(bool(v) for v in success)),
                        "log_and_raise": lambda message: (_ for _ in ()).throw(ValueError(message)),
                    }
                    exec(compile(function, str(_PINNED), "exec"), namespace)  # noqa: S102 (the pinned file's own code)
                    return namespace["plan_cspace"]
    return None


def _succeeded(result: _Result) -> bool:
    return any(result.success)


class TheLoopIsCurobosOwnTests(unittest.TestCase):
    def _both(self, *, seeds: "list[bool]", successes: "list[bool]", attempts: int = 4, graph_from: int = 1):
        loop = _curobo_loop()
        if loop is None:
            self.skipTest(f"no pinned cuRobo checkout at {_PINNED}")
        theirs = _Planner(seeds=seeds, successes=successes)
        their_log: list = []
        their_result = loop(theirs, _State("goal", their_log), _State("start", their_log), max_attempts=attempts,
                            enable_graph_attempt=graph_from)
        ours = _Planner(seeds=seeds, successes=successes)
        our_log: list = []
        our_result = plan_cspace(ours, _State("goal", our_log), _State("start", our_log), max_attempts=attempts,
                                 enable_graph_attempt=graph_from, finetune_attempts=CUROBO_FINETUNE_PASSES,
                                 succeeded=_succeeded)
        return (theirs.log, their_log, their_result), (ours.log, our_log, our_result)

    def test_at_three_passes_the_copy_makes_curobos_calls_in_curobos_order(self) -> None:
        """⭐ THE PROOF: cuRobo's own source and the copy, the same planner, the same calls, for every way it can go."""
        cases = {
            "the line succeeds at once": ([True, True, True], [True]),
            "the graph plans the second attempt": ([True, True, True], [False, True]),
            "the graph gives no seed once": ([False, True, True], [False, True]),
            "nothing plans": ([True, True, True], [False, False, False, False]),
            "the graph gives nothing at all": ([False, False, False], [False]),
        }
        for name, (seeds, successes) in cases.items():
            with self.subTest(case=name):
                (their_calls, their_states, theirs), (our_calls, our_states, ours) = self._both(
                    seeds=seeds, successes=successes)
                self.assertEqual(their_calls, our_calls)
                self.assertEqual(their_states, our_states)
                self.assertEqual((theirs.success, theirs.total_time, theirs.solve_time),
                                 (ours.success, ours.total_time, ours.solve_time))

    def test_curobo_still_asks_three_passes_at_the_scale_the_copy_names(self) -> None:
        """Pinned, so a cuRobo that changes either number is noticed here before the copy runs with stale ones."""
        if not _PINNED.is_file():
            self.skipTest(f"no pinned cuRobo checkout at {_PINNED}")
        loop = _curobo_loop()
        planner = _Planner(seeds=[], successes=[True])
        loop(planner, _State("goal", []), _State("start", []), max_attempts=1, enable_graph_attempt=1)
        solved = dict(planner.log[-1][3])
        self.assertEqual(CUROBO_FINETUNE_PASSES, solved["finetune_attempts"])
        self.assertEqual(CUROBO_FINETUNE_DT_SCALE, solved["finetune_dt_scale"])

    def test_curobos_later_passes_never_decide_whether_a_plan_is_found(self) -> None:
        """What lets one pass find a plan exactly where four do, read in the pinned trajectory optimiser: an attempt
        whose first pass finds nothing stops there, and a later pass only replaces a trajectory with a faster one."""
        solver = _ROOT / "ext_deps" / "curobo" / "curobo" / "_src" / "solver" / "solver_trajopt.py"
        if not solver.is_file():
            self.skipTest(f"no pinned cuRobo checkout at {solver}")
        source = solver.read_text(encoding="utf-8")
        self.assertIn("if i == 0 and torch.count_nonzero(best_trajopt_result.success) == 0:", source)
        self.assertIn("update_mask = torch.logical_and(trajopt_result.success, dt_mask)", source)
        self.assertIn("dt_mask = new_dt <= best_dt", source)

    def test_fewer_passes_change_nothing_but_the_passes(self) -> None:
        planner = _Planner(seeds=[True, True], successes=[False, True])
        attempts = PlannedAttempts()
        result = plan_cspace(planner, _State("goal", []), _State("start", []), max_attempts=4, enable_graph_attempt=1,
                             finetune_attempts=0, succeeded=_succeeded, attempts=attempts)
        solves = [dict(entry[3]) for entry in planner.log if entry[0] == "solve"]
        self.assertEqual([0, 0], [solve["finetune_attempts"] for solve in solves])
        self.assertEqual([CUROBO_FINETUNE_DT_SCALE] * 2, [solve["finetune_dt_scale"] for solve in solves])
        self.assertEqual([None, "graph seed"], [solve["seed_traj"] for solve in solves])
        self.assertTrue(result.success[0])
        self.assertEqual((2, 1, 0), (attempts.ran, attempts.graph_seeded, attempts.skipped))
        self.assertAlmostEqual(0.5, result.total_time)

    def test_nothing_ran_is_none(self) -> None:
        planner = _Planner(seeds=[False, False], successes=[])
        self.assertIsNone(plan_cspace(planner, _State("goal", []), _State("start", []), max_attempts=2,
                                      enable_graph_attempt=0, finetune_attempts=0, succeeded=_succeeded))


class TheNumbersTests(unittest.TestCase):
    def test_every_cell_runs_no_pass_after_the_first_where_curobo_runs_three(self) -> None:
        self.assertEqual((0, 3), (JOINT_FINETUNE_PASSES, CUROBO_FINETUNE_PASSES))

    def test_timing_is_asked_with_a_one_alone(self) -> None:
        self.assertTrue(timing_asked({ENV_TIMING: "1"}))
        self.assertFalse(timing_asked({ENV_TIMING: "0"}))
        self.assertFalse(timing_asked({}))


def _block(source: str, head: str) -> str:
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == head), None)
    if start is None:
        return ""
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return "\n".join(block)


class TheSidecarPlansWithNoPassAfterTheFirstTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")
        self.plan_js = _block(self.source, 'if cmd == "plan_js":')

    def test_the_copy_runs_at_no_pass_after_the_first_and_curobos_own_loop_never(self) -> None:
        self.assertTrue(self.plan_js, "the plan_js branch is gone")
        self.assertIn("result = plan_cspace(", self.plan_js)
        self.assertIn("finetune_attempts=JOINT_FINETUNE_PASSES", self.plan_js)
        self.assertNotIn("_planner.plan_cspace(", self.plan_js, "cuRobo's own three-pass loop runs")

    def test_no_variable_asks_for_another_number(self) -> None:
        """A shell's leftover cannot change how a cell plans: no side reads a number of passes from the environment."""
        cspace = (_ROOT / "src" / "robot" / "safety" / "planning" / "_curobo_cspace.py").read_text(encoding="utf-8")
        for source in (self.source, cspace):
            self.assertNotIn("WILLY_CUROBO_JOINT_FINETUNE", source)
            self.assertNotIn("finetune_passes(os.environ)", source)

    def test_it_runs_after_the_seed_is_reset(self) -> None:
        reset = self.plan_js.find("_planner.reset_seed()")
        self.assertTrue(0 <= reset < self.plan_js.find("result = plan_cspace("))

    def test_timing_is_printed_and_nothing_else(self) -> None:
        """The clock waits for the GPU only where timing was asked; a [timing] line is the whole of what it adds."""
        clock = _block(self.source, "def _clock() -> float:")
        self.assertIn("if _TIMING:", clock)
        self.assertIn("torch.cuda.synchronize()", clock)
        said = _block(self.source, "def _say_timing(what: str, *parts: tuple, note: str = \"\") -> None:")
        self.assertIn("if _TIMING:", said)
        self.assertNotIn("_emit(", said)


class TheConfigTests(unittest.TestCase):
    def test_the_old_key_is_refused_naming_what_every_cell_runs(self) -> None:
        from pydantic import ValidationError

        from src.config.schema._removed import removed_key_sentence
        from src.config.schema.robot import RobotConfig

        with self.assertRaises(ValidationError) as caught:
            RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"finetune_passes": 3}}})
        self.assertIn("planned_motion", str(caught.exception))
        said = removed_key_sentence("robot.safety.planned_motion.finetune_passes")
        self.assertIn("no time-optimal pass after its first", said or "")


if __name__ == "__main__":
    unittest.main()
