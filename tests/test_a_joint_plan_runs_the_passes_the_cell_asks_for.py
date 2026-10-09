"""A cuRobo joint plan runs cuRobo's own loop, with as many time-optimal passes as the cell asks for (3, today's).

cuRobo's ``MotionPlanner.plan_cspace`` runs, in every attempt, one optimiser pass that finds a trajectory and three more
that only shrink its time step towards cuRobo's own limits. The UR driver runs the positions of a plan with ``moveJ`` at
its own speed and never that timing, so ``safety.planned_motion.finetune_passes`` (3 by default) may ask for fewer
(``_curobo_cspace``). This file holds that on the CPU:

* ``plan_cspace`` is cuRobo's loop: cuRobo's own ``plan_cspace`` source, read from the pinned checkout and run against
  the same planner of this file's, makes the same calls in the same order with the same arguments at three passes;
  at any other number only ``finetune_attempts`` differs;
* the variable: unset is cuRobo's three, anything but 0 to 3 refuses the sidecar's start;
* the sidecar's ``plan_js`` branch, read as source: cuRobo's own call at three passes, the copy otherwise, both after
  the seed is reset; ``WILLY_CUROBO_TIMING`` prints and changes nothing else;
* the client and the config: the variable written only where the cell asks, a shell's leftover never inherited.

That a plan at three passes is today's plan to the bit, and what one pass saves, is the GPU's to say:
``scripts/curobo/ab_planning_spheres.py --finetune 0`` against ``--finetune 3``.
"""

from __future__ import annotations

import ast
import os
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from src.robot.safety.planning._curobo_cspace import (
    CUROBO_FINETUNE_DT_SCALE,
    CUROBO_FINETUNE_PASSES,
    ENV_JOINT_FINETUNE,
    ENV_TIMING,
    PlannedAttempts,
    finetune_passes,
    plan_cspace,
    timing_asked,
)

_ROOT = Path(__file__).resolve().parents[1]
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


class TheVariablesTests(unittest.TestCase):
    def test_unset_is_curobos_three(self) -> None:
        self.assertEqual(3, finetune_passes({}))
        self.assertEqual(3, finetune_passes({ENV_JOINT_FINETUNE: ""}))

    def test_a_whole_number_from_0_to_3_is_read(self) -> None:
        for value in range(4):
            with self.subTest(value=value):
                self.assertEqual(value, finetune_passes({ENV_JOINT_FINETUNE: str(value)}))

    def test_anything_else_refuses_the_start(self) -> None:
        for value in ("4", "-1", "1.5", "one", "+1", "01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finetune_passes({ENV_JOINT_FINETUNE: value})

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


class TheSidecarPlansTodaysWayUnlessAskedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")
        self.plan_js = _block(self.source, 'if cmd == "plan_js":')

    def test_three_passes_call_curobos_own_loop_and_any_other_number_the_copy(self) -> None:
        self.assertTrue(self.plan_js, "the plan_js branch is gone")
        branch = self.plan_js.find("if _JOINT_FINETUNE == CUROBO_FINETUNE_PASSES:")
        theirs = self.plan_js.find("result = _planner.plan_cspace(")
        ours = self.plan_js.find("result = plan_cspace(")
        self.assertTrue(0 <= branch < theirs < ours, "cuRobo's own call is not what three passes run")
        self.assertIn("finetune_attempts=_JOINT_FINETUNE", self.plan_js)
        self.assertIn("_JOINT_FINETUNE = finetune_passes(os.environ)", self.source)

    def test_both_run_after_the_seed_is_reset(self) -> None:
        reset = self.plan_js.find("_planner.reset_seed()")
        self.assertTrue(0 <= reset < self.plan_js.find("_planner.plan_cspace("))
        self.assertTrue(0 <= reset < self.plan_js.find("plan_cspace(\n                    _planner,"))

    def test_timing_is_printed_and_nothing_else(self) -> None:
        """The clock waits for the GPU only where timing was asked; a [timing] line is the whole of what it adds."""
        clock = _block(self.source, "def _clock() -> float:")
        self.assertIn("if _TIMING:", clock)
        self.assertIn("torch.cuda.synchronize()", clock)
        said = _block(self.source, "def _say_timing(what: str, *parts: tuple, note: str = \"\") -> None:")
        self.assertIn("if _TIMING:", said)
        self.assertNotIn("_emit(", said)


class TheClientAndTheConfigTests(unittest.TestCase):
    def test_the_variable_is_written_only_where_the_cell_asks_and_a_leftover_is_never_inherited(self) -> None:
        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        with mock.patch.dict(os.environ, {ENV_JOINT_FINETUNE: "0"}):
            self.assertNotIn(ENV_JOINT_FINETUNE, CuroboPlanClient()._sidecar_env())  # noqa: SLF001
            env = CuroboPlanClient(joint_finetune_passes=0)._sidecar_env()  # noqa: SLF001
        self.assertEqual("0", env[ENV_JOINT_FINETUNE])

    def test_the_key_is_todays_three_and_reaches_the_client_through_the_reservation(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.curobo_client import CuroboPlanClient
        from src.robot.safety.planning.reservation import PlannerReservation

        today = RobotConfig.model_validate({"vendor": "ur"})
        self.assertEqual(3, today.safety.planned_motion.finetune_passes)
        self.assertEqual(3, PlannerReservation.from_config(robot_cfg=today).finetune_passes)
        for refused in (-1, 4):
            with self.subTest(refused=refused), self.assertRaises(ValidationError):
                RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"finetune_passes": refused}}})
        one = RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"finetune_passes": 0}}})
        client = CuroboPlanClient()
        client.reserve_world(PlannerReservation.from_config(robot_cfg=one))
        self.assertEqual("0", client._sidecar_env()[ENV_JOINT_FINETUNE])  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
