"""Two rules of how the sidecar plans and judges, beyond the robot it loaded. Standard library only, on purpose.

Imported from both sides of the process boundary, like ``_curobo_margin``: as part of this package under python 3.11,
where the client and the tests read them, and as a plain sibling module by ``curobo_planner_server.py`` under python
3.10, which cannot import this package. So nothing here imports anything of Willy's, and a test without a GPU reads the
very rules the sidecar runs.

The graph planner goes straight from the start to the goal. cuRobo's shipped graph planner config
(``graph_planner/exact_graph_planner.yml``) sets ``use_default_position_heuristic``, which links the start and the goal
through the descriptor's retract configuration in every roadmap it seeds: every graph-seeded plan, which is every
attempt from the second on, is offered the route start, retract, goal. The committed retract of a UR10 with a Hand-E
(``robot/ur_retract.yaml``) is [-1.57, -1.57, -1.57, -1.57, 1.57, 0.5], a quarter turn of the base from most of the
bench, so a plan between two stations a few centimetres apart could be seeded through it. :func:`graph_planner_without_retract` switches that link
off and changes nothing else.

A straight line is judged with room to spare. ``check_js`` passes a configuration whose spheres do not penetrate the
world, which is right for a path cuRobo planned, because cuRobo shaped it to keep clear. A straight joint line nobody
shaped is judged at a clearance instead (:data:`CLEARANCE_KEY`), so a line that grazes an obstacle only the camera saw is
planned around rather than driven along. :func:`requested_clearance_m` reads it off a request, and the reply says the
clearance it judged at, so a client can refuse a sidecar that judged at none.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "CLEARANCE_KEY",
    "GRAPH_PLANNER_CONFIG",
    "MAX_CLEARANCE_M",
    "graph_planner_without_retract",
    "requested_clearance_m",
]

#: The graph planner config the sidecar starts from, relative to cuRobo's task config folder: cuRobo's own default.
GRAPH_PLANNER_CONFIG = "graph_planner/exact_graph_planner.yml"

#: The ``check_js`` request key, and the key of its reply, for the world clearance a judgement was asked at, in metres.
CLEARANCE_KEY = "clearance_m"

#: The most clearance a judgement may ask for, metres. cuRobo answers a mesh query within ``SceneCollisionCfg.
#: max_distance``, 0.1 m unless a planner is built otherwise, and a clearance past it would pass a sphere the query
#: never looked far enough to find.
MAX_CLEARANCE_M = 0.1


def graph_planner_without_retract(config: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a graph planner config whose roadmap links the start to the goal directly, never through the retract.

    ``config`` is the YAML cuRobo loads, ``{"graph_planner": {...}}``. A config without the key is refused rather than
    passed on: a cuRobo that no longer reads ``use_default_position_heuristic`` might route through something else the
    sidecar never switched off, and a planner started on a guess is the planner this rule exists to avoid.
    """
    block = config.get("graph_planner") if isinstance(config, Mapping) else None
    if not isinstance(block, Mapping) or "use_default_position_heuristic" not in block:
        raise ValueError(
            "the graph planner config holds no graph_planner.use_default_position_heuristic, so whether this cuRobo "
            "seeds its plans through the retract cannot be switched off; the planner is not started on a guess"
        )
    turned = copy.deepcopy(dict(config))
    turned["graph_planner"] = dict(turned["graph_planner"], use_default_position_heuristic=False)
    return turned


def requested_clearance_m(request: Mapping[str, Any]) -> float:
    """The world clearance a ``check_js`` request asks for, metres: 0.0 when it names none, which is today's check.

    Raises ``ValueError`` for anything that is not a number from 0 to :data:`MAX_CLEARANCE_M`, which the sidecar answers
    as a failed call rather than as a verdict at some other clearance.
    """
    value = request.get(CLEARANCE_KEY, 0.0)
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{CLEARANCE_KEY} has to be a number of metres, got {value!r}")
    clearance = float(value)
    if not math.isfinite(clearance) or not 0.0 <= clearance <= MAX_CLEARANCE_M:
        raise ValueError(f"{CLEARANCE_KEY} has to lie from 0 to {MAX_CLEARANCE_M} m, got {value!r}")
    return clearance
