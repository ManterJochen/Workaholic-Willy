"""A cuRobo descriptor for an arm, with no hand in it. Standard library only, on purpose.

``build_ur_config.py`` runs under the cuRobo sidecar's interpreter, where this repository cannot be imported, so this
module lives beside it and is loaded by path rather than imported.

The template every UR build reads, a copy of cuRobo's own ``ur10e.yml``, guards ``tool0`` as a collision link. The
hand is a body link the sidecar adds when it starts (``src/robot/safety/planning/_curobo_body_links.py``), so an arm
descriptor guards no tool frame at all: cuRobo raises on a collision link with no spheres, and a tool0 link with
spheres would count the hand twice. :func:`arm_only` takes the tool frame out of the collision links and the buffer
table and keeps the ignore entries that name it, which then cost nothing. It also drops the template's
``camera_mount`` ignore key, which names no link of any UR description: a camera body of that name would inherit the
table, and the body link composition refuses a name that is already a key. The arm spheres come from the committed
cover fit, whose bodies are the six arm links, and Isaac's Lula descriptions, the fallback for an arm with no fit, name
no tool0, so a build never has tool0 spheres to refuse.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["ArmDescriptorError", "arm_only"]

#: Keys of the template's ignore table that name no link of any UR description.
_STALE_IGNORE_KEYS = ("camera_mount",)


class ArmDescriptorError(ValueError):
    """The config cannot be made an arm descriptor as it stands."""


def arm_only(cfg: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``cfg`` that guards no tool frame and says ``_provenance.carries_hand: false``; ``cfg`` is untouched."""
    out = copy.deepcopy(cfg)
    robot_cfg = out.get("robot_cfg")
    kinematics = robot_cfg.get("kinematics") if isinstance(robot_cfg, dict) else None
    if not isinstance(kinematics, dict):
        raise ArmDescriptorError("the config has no robot_cfg.kinematics block, so there is no arm to describe")
    frames = kinematics.get("tool_frames") or []
    if not frames:
        raise ArmDescriptorError("the config names no tool_frames, so there is no tool frame to keep the hand off")
    tool = str(frames[0])
    provenance = out.get("_provenance")
    provenance = dict(provenance) if isinstance(provenance, dict) else {}
    if provenance.get("gripper_key"):
        raise ArmDescriptorError(
            f"the config was built for the hand {provenance['gripper_key']!r}, and an arm descriptor carries none"
        )
    spheres = kinematics.get("collision_spheres")
    if isinstance(spheres, dict) and spheres.get(tool):
        raise ArmDescriptorError(
            f"the config still has {len(spheres[tool])} {tool} sphere(s): that is a hand, and an arm descriptor carries "
            "none. The sidecar adds the hand the cell names as a body link when it starts"
        )
    if isinstance(spheres, dict):
        spheres.pop(tool, None)
    kinematics["collision_link_names"] = [
        link for link in kinematics.get("collision_link_names") or () if str(link) != tool
    ]
    buffer = kinematics.get("self_collision_buffer")
    if isinstance(buffer, dict):
        buffer.pop(tool, None)
    ignore = kinematics.get("self_collision_ignore")
    if isinstance(ignore, dict):
        for key in _STALE_IGNORE_KEYS:
            ignore.pop(key, None)
    out["_provenance"] = {**provenance, "carries_hand": False}
    return out
