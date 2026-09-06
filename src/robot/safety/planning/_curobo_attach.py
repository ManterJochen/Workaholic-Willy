"""Teaching the cuRobo planner that the gripper is carrying something.

The planner collision model ends at the gripper. Without the grasped part attached to
it, every transit, lift, place and retreat after a successful close is planned as if
the hand were empty. On a real cell carrying a part out of a bin, that part is the
geometry most likely to meet a wall.

cuRobo can carry an attached body, but only on a robot whose config declares a link to
hang it from. ``franka.yml`` ships one and ``ur5e.yml`` and ``ur3e.yml`` do not, so the
link is derived here, exactly as :mod:`_curobo_margin` derives the guard clearance into
a temporary config.

Measured on this box against the real cuRobo build with ``ur5e.yml``: the derived
config builds a planner, ``attachment_manager.attach`` hangs an 80 x 80 x 120 mm box on
``tool0``, and ``detach`` removes it again. The sphere budget is the one detail that has
to be right. Fitting that box automatically wants 13 spheres, so a link declared with
the franka-style 4 slots raises ``ValueError: Fitted 13 spheres but link
'attached_object' only has 4 sphere slots``. Sixteen slots accommodate the automatic
fit, and a caller can pin ``num_spheres`` to whatever the link has.

The limit is real: a sphere-fitted bounding box is not the part. It over-claims in the
corners and under-claims on a concave shape, so this makes the planner aware of the
payload without making it exact. Nothing here measures what was grasped either; the
dimensions come from the caller.

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
    "ATTACHED_LINK_NAME",
    "DEFAULT_ATTACH_SPHERES",
    "ENV_ATTACH_SPHERES",
    "apply_attached_object_link",
    "derive_attach_config_file",
]

#: The environment variable the client uses to tell the sidecar how many collision
#: spheres to reserve for a carried payload. Unset, or ``0``, leaves the config
#: untouched and the sidecar unable to attach anything, which is the unchanged path.
ENV_ATTACH_SPHERES = "WILLY_CUROBO_ATTACH_SPHERES"

#: The link the cuRobo attachment manager hangs a payload from. It is that manager own
#: default, kept explicit here because both sides of the boundary agree on the string.
ATTACHED_LINK_NAME = "attached_object"

#: Sixteen rather than the franka four. The automatic sphere fit for an
#: 80 x 80 x 120 mm box is measured to want 13, and a link with fewer slots than the fit
#: raises rather than degrading.
DEFAULT_ATTACH_SPHERES = 16

_KINEMATICS_PATH = ("robot_cfg", "kinematics")


def apply_attached_object_link(
    config: dict[str, Any], *, spheres: int = DEFAULT_ATTACH_SPHERES, parent_link: str | None = None
) -> tuple[dict[str, Any], bool]:
    """Return ``(config_copy, added)`` with a payload link declared on the robot tool frame.

    ``added`` is ``False`` where the config already declares the link, so a robot that
    ships one, as franka does, is left exactly as its author wrote it.

    ``parent_link`` defaults to the first ``tool_frames`` entry of the config, which is
    the frame the rest of this stack treats as the tool, ``tool0`` on every UR. Passing
    it explicitly is for a robot whose payload does not hang from its tool frame.
    """
    if spheres <= 0:
        return config, False
    kinematics = config.get(_KINEMATICS_PATH[0], {}).get(_KINEMATICS_PATH[1])
    if not isinstance(kinematics, dict):
        return config, False
    if ATTACHED_LINK_NAME in (kinematics.get("collision_link_names") or ()):
        return config, False

    parent = parent_link or next(iter(kinematics.get("tool_frames") or ()), None)
    if not parent:
        return config, False

    out = copy.deepcopy(config)
    kin = out[_KINEMATICS_PATH[0]][_KINEMATICS_PATH[1]]

    links = kin.get("collision_link_names")
    kin["collision_link_names"] = ([*links] if isinstance(links, list) else []) + [ATTACHED_LINK_NAME]

    # Extended only where it already exists, because a robot that declares no grasp
    # contacts does not acquire the concept here.
    contacts = kin.get("grasp_contact_link_names")
    if isinstance(contacts, list):
        kin["grasp_contact_link_names"] = [*contacts, ATTACHED_LINK_NAME]

    extra_spheres = kin.get("extra_collision_spheres")
    kin["extra_collision_spheres"] = {
        **(extra_spheres if isinstance(extra_spheres, dict) else {}),
        ATTACHED_LINK_NAME: int(spheres),
    }

    extra_links = kin.get("extra_links")
    kin["extra_links"] = {
        **(extra_links if isinstance(extra_links, dict) else {}),
        ATTACHED_LINK_NAME: {
            # The identity transform, in the cuRobo [x, y, z, qw, qx, qy, qz] order. The
            # payload frame is the tool frame, and where the part sits inside the gripper
            # is the offset of the attach call rather than a property of the robot.
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_name": "attach_joint",
            "joint_type": "FIXED",
            "link_name": ATTACHED_LINK_NAME,
            "parent_link_name": str(parent),
        },
    }

    # Zero, deliberately. The guard clearance margin belongs on the robot links, which is
    # what _curobo_margin inflates, and inflating the payload as well would refuse grasps
    # of anything that fits snugly, which is most of what a bin holds.
    buffer = kin.get("self_collision_buffer")
    if isinstance(buffer, dict):
        kin["self_collision_buffer"] = {**buffer, ATTACHED_LINK_NAME: 0.0}

    # The part is inside the hand that holds it. Without this the payload self-collides
    # with the gripper permanently and the planner refuses everything: measured on
    # ur5e.yml, a 10 mm cube attached to tool0 turns a 61-waypoint plan into no plan at
    # all. `franka.yml` lists `attached_object` under its hand and both fingers for the
    # same reason.
    kin["self_collision_ignore"] = _with_payload_ignored(
        kin.get("self_collision_ignore"), parent=str(parent)
    )

    return out, True


def _with_payload_ignored(
    ignore: Any, *, parent: str
) -> dict[str, list[str]]:
    """Add the payload to the ignore list of the parent link and everything adjacent to it.

    ``tool0`` on a UR is a frame at the flange, so ignoring the payload against it alone
    leaves the wrist links behind the gripper still colliding with a part the gripper
    holds. The config already states which links are adjacent to the tool, in its own
    ignore entries, and this reuses that statement rather than hard-coding a vendor link
    names here.
    """
    table: dict[str, list[str]] = {
        str(k): list(v) for k, v in (ignore or {}).items() if isinstance(v, (list, tuple))
    }
    neighbours = {parent}
    for link, ignored in table.items():
        if parent in ignored:
            neighbours.add(link)
    for link in neighbours:
        entries = table.setdefault(link, [])
        if ATTACHED_LINK_NAME not in entries:
            entries.append(ATTACHED_LINK_NAME)
    return table


def derive_attach_config_file(
    source_path: str, dest_path: str, *, spheres: int = DEFAULT_ATTACH_SPHERES
) -> bool:
    """Write ``source_path`` to ``dest_path`` with a payload link declared, and say whether one was added.

    It returns ``False`` without writing where the source already declares the link, so
    the caller keeps using the original file and nothing is copied for nothing.
    """
    with open(source_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        return False
    derived, added = apply_attached_object_link(config, spheres=spheres)
    if not added:
        return False
    with open(dest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(derived, handle, sort_keys=False)
    return True
