"""One joint plan the way cuRobo's own ``MotionPlanner.plan_cspace`` runs it, with its time-optimal passes a parameter.

cuRobo plans a joint move in attempts: the first optimises a straight line from the start to the goal, the later ones
start from the graph planner's roadmap, and the first attempt that succeeds is the plan. Inside each attempt the
trajectory optimiser runs one pass that finds a collision-free trajectory, then up to ``finetune_attempts`` more passes
that only shrink its time step towards cuRobo's own acceleration and jerk limits. ``plan_cspace`` hard-codes three of
them (the pinned checkout, ``motion/motion_planner.py``).

The UR driver keeps the positions of a plan and nothing else: it shortens them to the waypoints both authorities judge
and runs those with ``moveJ`` at its own speed (``CuroboPlanner.execute``). So the three passes after the first shape
a timing nobody runs. Nor do they decide whether a plan is found: an attempt whose first pass finds no trajectory stops
there, and a later pass only replaces a trajectory with a faster one (``TrajOptSolver._solve_impl``), so an attempt
succeeds with one pass exactly where it succeeds with four. What one pass changes is which positions come back, and
those are judged sample by sample as every plan's are.

:func:`plan_cspace` is cuRobo's loop as it stands, with the number of those passes a parameter: the attempts, the
attempt the graph seeds from, the graph seed per attempt and the step scale are cuRobo's, call for call; at
:data:`CUROBO_FINETUNE_PASSES` it makes cuRobo's own calls. The sidecar runs it at :data:`JOINT_FINETUNE_PASSES`, 0:
a plan is the first pass's, which every later pass starts from, about a quarter of the optimiser's work per attempt
(the owner's cell after the desk A/B of 2026-10-09, every cell since 2026-10-17, when
``robot.safety.planned_motion.finetune_passes`` left). ``WILLY_CUROBO_TIMING=1`` makes the sidecar print
one ``[timing]`` line per request it answers (:func:`timing_asked`): what it judged first, what the plan took on the
wall clock and on the optimiser's own GPU timer, and what the reply took. It changes nothing else.

Imported from both sides of the process boundary, like ``_curobo_plan_policy``: as part of the Willy package under
python 3.11, where the CPU suite runs the loop against a planner of its own, and as a plain sibling module by
``curobo_planner_server.py`` under python 3.10. Standard library only: the planner and the result are duck typed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "CUROBO_FINETUNE_DT_SCALE",
    "CUROBO_FINETUNE_PASSES",
    "ENV_TIMING",
    "JOINT_FINETUNE_PASSES",
    "PlannedAttempts",
    "plan_cspace",
    "timing_asked",
]

#: The time-optimal passes after the first that cuRobo's ``MotionPlanner.plan_cspace`` asks of every attempt.
CUROBO_FINETUNE_PASSES = 3
#: The step scale between those passes that ``plan_cspace`` asks, for an attempt seeded by a line or by the graph.
CUROBO_FINETUNE_DT_SCALE = 0.75

#: How many time-optimal passes a joint plan runs after its first on every cell: none. The arm runs a plan's positions
#: with ``moveJ`` at its own speed and never its timing, and a later pass never decides whether a plan is found.
JOINT_FINETUNE_PASSES = 0
#: ``1`` prints a ``[timing]`` line per request on the sidecar's stderr; unset prints none.
ENV_TIMING = "WILLY_CUROBO_TIMING"


def timing_asked(environ: Mapping[str, str]) -> bool:
    """Whether the sidecar prints a ``[timing]`` line per request: ``WILLY_CUROBO_TIMING`` is exactly ``1``."""
    return str(environ.get(ENV_TIMING, "") or "").strip() == "1"


class PlannedAttempts:
    """What one :func:`plan_cspace` call did: the attempts it ran, the ones the graph seeded, the ones it skipped."""

    __slots__ = ("graph_seeded", "ran", "skipped")

    def __init__(self) -> None:
        self.ran = 0
        self.graph_seeded = 0
        #: Attempts the graph planner gave no seed for, which cuRobo skips without optimising anything.
        self.skipped = 0

    def render(self) -> str:
        return f"{self.ran} attempt(s), {self.graph_seeded} seeded by the graph, {self.skipped} with no graph seed"


def plan_cspace(
    planner: Any,
    goal_state: Any,
    current_state: Any,
    *,
    max_attempts: int,
    enable_graph_attempt: int,
    finetune_attempts: int,
    succeeded: "Callable[[Any], bool]",
    attempts: "PlannedAttempts | None" = None,
) -> Any:
    """cuRobo's ``MotionPlanner.plan_cspace`` with ``finetune_attempts`` a parameter; ``None`` where nothing ran.

    The loop is cuRobo's: per attempt a fresh copy of the start, from attempt ``enable_graph_attempt`` on a seed from
    the graph planner towards the goal repeated once per trajectory seed (an attempt the graph gives no seed for is
    skipped), then one ``trajopt_solver.solve_cspace`` with the step scale cuRobo uses, and the first attempt whose
    result ``succeeded`` ends it. The last result is returned, a failed one included, with the optimiser's times summed
    over the attempts that ran, as cuRobo returns them. ``succeeded`` is ``torch.count_nonzero(result.success) > 0`` in
    the sidecar; it is a parameter so this module stays free of torch.
    """
    result = None
    total_time = 0.0
    solve_time = 0.0
    start = current_state.clone()
    seeds = planner.trajopt_solver.config.num_seeds
    for attempt in range(int(max_attempts)):
        state = start.clone()
        seed_traj = None
        if attempt >= int(enable_graph_attempt) and planner.graph_planner is not None:
            goal_configs = goal_state.position.view(1, 1, -1).repeat(1, seeds, 1)
            graph_seed = planner._get_graph_seed_trajectories(state, goal_configs)  # noqa: SLF001 (cuRobo's own loop)
            if graph_seed is None:
                if attempts is not None:
                    attempts.skipped += 1
                continue
            seed_traj = graph_seed
            if attempts is not None:
                attempts.graph_seeded += 1
        if attempts is not None:
            attempts.ran += 1
        result = planner.trajopt_solver.solve_cspace(
            goal_state, state, seed_traj=seed_traj,
            finetune_attempts=int(finetune_attempts), finetune_dt_scale=CUROBO_FINETUNE_DT_SCALE,
        )
        total_time += result.total_time
        solve_time += result.solve_time
        if succeeded(result):
            break
    if result is not None:
        result.total_time = total_time
        result.solve_time = solve_time
    return result
