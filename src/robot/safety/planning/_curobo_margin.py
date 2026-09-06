"""Teaching the cuRobo planner the clearance the safety guard will demand.

Two engines decide whether an arm configuration is acceptable, and they do not use the
same geometry:

  * cuRobo plans against a sphere model of the links and returns the first path it
    believes is collision-free;
  * ``SelfCollisionGuard`` then re-checks the final configuration of that path against
    the exact link meshes and rejects anything closer than
    ``safety.self_collision.min_distance_mm``.

A planner that knows nothing of that margin returns configurations the guard was
always going to refuse. Measured on-box on a UR5e with radial closing, three of ten
picks were lost to ``forearm|wrist_2`` clearances of 9.44 to 9.47 mm against a
10.000 mm margin: a plan good by the rules of cuRobo and refused by the rules here,
differing by half a millimetre. That is neither a bad grasp nor a bad planner, but two
engines that were never introduced to each other.

So this inflates the cuRobo self-collision buffers by the margin of the guard, which
leaves the planner offering only paths the guard can accept. cuRobo measures pairwise
clearance as ``sphere_distance - (buffer_a + buffer_b)``, so half the margin goes on
each link and a pair gets the whole of it.

The limit is real: this reduces the disagreement and does not remove it. Spheres are
not meshes, so the two models still differ locally, and the buffer is a cushion rather
than a proof. The guard and never the planner decides what executes.

The module is imported from both sides of the process boundary: as part of this
package under python 3.11, and as a plain sibling module by
``curobo_planner_server.py`` under python 3.10, which cannot import this package. It
therefore uses the standard library and PyYAML alone and holds no relative imports.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

__all__ = [
    "ENV_SELF_COLLISION_MARGIN_MM",
    "apply_self_collision_margin",
    "derive_margin_config_file",
]

#: The environment variable the client uses to hand the guard margin to the sidecar.
#: Unset, or ``0``, leaves the config untouched.
ENV_SELF_COLLISION_MARGIN_MM = "WILLY_CUROBO_SELF_COLLISION_MARGIN_MM"

_BUFFER_PATH = ("robot_cfg", "kinematics", "self_collision_buffer")


def apply_self_collision_margin(config: dict[str, Any], margin_mm: float) -> tuple[dict[str, Any], int]:
    """Return ``(config_copy, links_adjusted)`` with every self-collision buffer raised by half ``margin_mm``.

    It is half per link because cuRobo subtracts the buffers of both links from the
    sphere distance of a pair, so half each demands exactly ``margin_mm`` of pairwise
    clearance.

    A margin at or below zero returns an unchanged copy with ``links_adjusted`` at 0,
    which is the default and leaves the planner unchanged. A config without the buffer
    block is also returned unchanged rather than having one invented, because adding a
    key to a third-party robot description in silence would be a worse failure than
    doing nothing visibly.
    """
    out = copy.deepcopy(config)
    if margin_mm <= 0.0:
        return out, 0
    node: Any = out
    for key in _BUFFER_PATH[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return out, 0
    buffers = node.get(_BUFFER_PATH[-1]) if isinstance(node, dict) else None
    if not isinstance(buffers, dict) or not buffers:
        return out, 0

    half_m = float(margin_mm) / 2000.0  # mm to m, and split across the pair
    adjusted = 0
    for link, value in list(buffers.items()):
        if isinstance(value, (int, float)):
            buffers[link] = float(value) + half_m
            adjusted += 1
    return out, adjusted


def derive_margin_config_file(source_path: str, dest_path: str, margin_mm: float) -> int:
    """Write the robot config at ``source_path`` to ``dest_path`` with the margin applied.

    It returns the number of links adjusted. It is separate from
    :func:`apply_self_collision_margin` so that the transform stays a pure function and
    only this wrapper touches the filesystem.
    """
    with open(source_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    adjusted_config, adjusted = apply_self_collision_margin(config, margin_mm)
    with open(dest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(adjusted_config, handle, sort_keys=True)
    return adjusted
