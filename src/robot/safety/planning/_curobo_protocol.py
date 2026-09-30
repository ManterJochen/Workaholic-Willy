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
    "NAME_PAIRS_KEY",
    "PAIRS_NAMED_KEY",
    "REFUSED_KEY",
    "REPORT_REFUSED_KEY",
    "WHERES",
    "WHERE_DEFAULT_Q",
    "WHERE_GOAL",
    "WHERE_PATH",
    "WHERE_START",
]

#: Set by a client that only wants to measure: the sidecar then reports a refused ``default_q`` in its ready line
#: instead of exiting, and refuses every planning command. Only the matrix gate asks for it, and a driver never does,
#: because a planner that cannot plan is not a planner.
ENV_MEASURE_ONLY = "WILLY_CUROBO_MEASURE_ONLY"

#: Which configuration the sidecar judged: the descriptor's own retract, the start of a move, its goal, or a
#: configuration of a path it was handed to judge (check_js), which it cannot tell a start, a screened goal or a sample
#: between them apart by.
WHERE_DEFAULT_Q = "default_q"
WHERE_START = "start"
WHERE_GOAL = "goal"
WHERE_PATH = "path"
WHERES = (WHERE_DEFAULT_Q, WHERE_START, WHERE_GOAL, WHERE_PATH)

#: What it found there. ``world`` is the planner's obstacles, not the robot.
KIND_SELF_COLLISION = "self_collision"
KIND_JOINT_LIMIT = "joint_limit"
KIND_WORLD = "world"
KINDS = (KIND_SELF_COLLISION, KIND_JOINT_LIMIT, KIND_WORLD)

#: A check_js request key: true asks for every configuration the sidecar refuses, not only the first, each as a row
#: under ``REFUSED_KEY``: ``{"index", "bound_ok", "self_ok", "world_ok", "pairs"}``, where ``pairs`` lists every pair
#: of links whose spheres overlap there, ``[link_a, link_b, depth_mm, unpadded_depth_mm]`` deepest first
#: (``_curobo_pairs.overlapping_pairs``), or is null where no names were asked (``NAME_PAIRS_KEY`` false) or the loaded
#: descriptor carries no ownership; ``PAIRS_NAMED_KEY`` says which. A request without it, and its reply, are the ones
#: check_js always had. The UR driver asks it where the exact guard decides the planner's self pairs (the owner,
#: 2026-09-30).
REPORT_REFUSED_KEY = "report_refused"
REFUSED_KEY = "refused"
NAME_PAIRS_KEY = "name_pairs"
PAIRS_NAMED_KEY = "pairs_named"
