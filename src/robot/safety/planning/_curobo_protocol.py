"""The words the two interpreters say about a state the planner refuses. Standard library only, on purpose.

The python 3.11 client reads these strings out of the JSON of the sidecar and turns them into enums; the python 3.10
sidecar writes them. They are spelled once here so the two halves cannot drift apart into a parser that raises on a
refusal its own sidecar sent. The sidecar loads this as a plain sibling, with no package around it, so nothing here may
import anything of Willy's.
"""

from __future__ import annotations

__all__ = [
    "ENV_MEASURE_ONLY",
    "KINDS",
    "KIND_JOINT_LIMIT",
    "KIND_SELF_COLLISION",
    "KIND_WORLD",
    "WHERES",
    "WHERE_DEFAULT_Q",
    "WHERE_GOAL",
    "WHERE_START",
]

#: Set by a client that only wants to measure: the sidecar then reports a refused ``default_q`` in its ready line
#: instead of exiting, and refuses every planning command. Only the matrix gate asks for it, and a driver never does,
#: because a planner that cannot plan is not a planner.
ENV_MEASURE_ONLY = "WILLY_CUROBO_MEASURE_ONLY"

#: Which configuration the sidecar judged: the descriptor's own retract, the start of a move, or its goal.
WHERE_DEFAULT_Q = "default_q"
WHERE_START = "start"
WHERE_GOAL = "goal"
WHERES = (WHERE_DEFAULT_Q, WHERE_START, WHERE_GOAL)

#: What it found there. ``world`` is the planner's obstacles, not the robot.
KIND_SELF_COLLISION = "self_collision"
KIND_JOINT_LIMIT = "joint_limit"
KIND_WORLD = "world"
KINDS = (KIND_SELF_COLLISION, KIND_JOINT_LIMIT, KIND_WORLD)
