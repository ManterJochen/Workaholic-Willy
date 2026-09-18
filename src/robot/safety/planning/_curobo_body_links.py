"""The one robot config the cuRobo sidecar loads, composed in memory from the descriptor it was started with.

The descriptor is read once and composed here, so that one set of bytes is what the planner, ``Kinematics`` and the
check_js checker all hold. The order is load bearing:

  1. body links (:func:`apply_body_links`): the hand, a plate, a wrist camera, each a fixed link under the tool frame
     with its own spheres, buffer and ignore list;
  2. the guard's margin (``_curobo_margin``), which raises only buffers already in the table, so a body added after it
     would plan without the margin;
  3. the payload link (``_curobo_attach``), which is ignored against every link whose ignore list names the tool frame,
     so a hand added after it would collide with the part it holds; a wrist camera stays checked against it.

A wrist camera's body is composed on top of the config the combination evidence names. :func:`compose_for_cell`
composes both, and refuses unless removing the camera's links from the loaded config gives back the evidence config
exactly, so ``composed_sha256`` can keep naming the evidence config while the planner loads the camera too.

:func:`canonical_sha256` names a config by its content: sorted keys and compact separators, so key order and line
endings in the file a dict came from do not change it, and any changed value or list order does.

cuRobo's side of a body link, read in its source: ``extra_links[L]`` with ``joint_type`` FIXED and ``fixed_transform``
``[x, y, z, qw, qx, qy, qz]`` in metres in the parent's frame (``link_params.py``, ``pose.py``); spheres are built only
for ``collision_link_names``, from ``collision_spheres[L]`` in L's own frame (``kinematics_loader.py``); a pair's
padding is the two buffers summed, and an ignore entry is a membership test against ``collision_link_names``
(``self_collision_params.py``), so an entry naming a link without spheres, tool0, is harmless and one direction is
enough.

Imported from both sides of the process boundary, like its two siblings: as part of the Willy package under python
3.11, and as a plain sibling module by ``curobo_planner_server.py`` under python 3.10. It uses nothing but the standard
library and those two.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from ._curobo_attach import apply_attached_object_link
    from ._curobo_margin import apply_self_collision_margin
except ImportError:  # loaded by the sidecar as a plain sibling, with no package around it
    from _curobo_attach import apply_attached_object_link  # type: ignore[import-not-found,no-redef]
    from _curobo_margin import apply_self_collision_margin  # type: ignore[import-not-found,no-redef]

__all__ = [
    "COUPLING_JOINT",
    "COUPLING_LINK",
    "COUPLING_REACH_MM",
    "ENV_BODY_LINKS",
    "ENV_DEFAULT_Q",
    "ENV_WRIST_BODY_LINKS",
    "HAND_IGNORE",
    "HAND_JOINT",
    "HAND_LINK",
    "HAND_PARENT",
    "WRIST_BODY_IGNORE",
    "WRIST_BODY_PREFIX",
    "WRIST_BODY_REACH_MM",
    "BodyLinkError",
    "apply_body_links",
    "apply_default_q",
    "body_report",
    "canonical_sha256",
    "compose_for_cell",
    "compose_sidecar_config",
    "compose_with_counts",
    "coupling_body_link",
    "fixed_transform_wxyz",
    "hand_body_link",
    "rotation_to_wxyz",
    "without_links",
    "wrist_body_link",
]

#: Env var the client hands the body links to the sidecar in, as a JSON list. Unset or empty adds none.
ENV_BODY_LINKS = "WILLY_CUROBO_BODY_LINKS"

#: The retract this arm and hand were judged at, as a JSON list of joint values. Set by the client from the committed
#: table (`robot/ur_retract.yaml`), because the descriptor is per arm and its own retract can only be right about a
#: bare arm: at the ur3's own retract the exact meshes clear the Hand-E by 13.4 mm and the planner's sphere model reads
#: the same pose as a self collision, so the sidecar would never become ready.
ENV_DEFAULT_Q = "WILLY_CUROBO_DEFAULT_Q"

#: The link a hand is added as, the frame it hangs from on a UR flange, and its joint.
HAND_LINK = "hand"
HAND_PARENT = "tool0"
HAND_JOINT = "hand_joint"
#: The links a hand is never checked against: the flange frame, and the two wrist links the exact guard skips too.
HAND_IGNORE = ("tool0", "wrist_3_link", "wrist_2_link")

#: The link the declared coupling plates are added as. One link for the whole stack, because the plates are bolted to
#: each other and to the flange and never move relative to it. It ignores exactly what the hand ignores, for the same
#: reason: it is bolted to those links.
COUPLING_LINK = "coupling"
COUPLING_JOINT = "coupling_joint"

#: How far the plates' spheres may reach past the plates, in millimetres. Not a free choice: the one arm link a
#: coupling is ever checked against is wrist_1 (it is nearer the flange than the hand, and the hand's own binding
#: partner is wrist_1 in every pair the retract table judged), and the cover fit of the arm gives wrist_1 8 mm out of
#: the same table. So the plate is covered at the budget of the link it meets, rather than at a number picked for it.
COUPLING_REACH_MM = 8.0

#: Env var the client hands a wrist camera's body links in, as a JSON list. It is apart from ``ENV_BODY_LINKS`` so
#: the bytes of that one, and the config the evidence names, stay the same for a cell with a camera and without one.
ENV_WRIST_BODY_LINKS = "WILLY_CUROBO_WRIST_BODY_LINKS"

#: A wrist camera's link is ``wrist_camera_<rig id>``. The prefix is how the composition tells a camera from the hand
#: holding the part, and how the evidence config is recovered from the loaded one.
WRIST_BODY_PREFIX = "wrist_camera_"

#: How far a camera's spheres may reach past its grown boxes, in millimetres: the budget the cover fit of the arm
#: gives wrist_1 and wrist_2, the two arm links a camera on the flange can meet.
WRIST_BODY_REACH_MM = 8.0

#: The links a camera is never checked against: the flange frame and wrist_3, which it is bolted to, and the hand and
#: the plates, which are bolted to the same flange. wrist_1 and wrist_2 stay checked, as does a carried part.
WRIST_BODY_IGNORE = ("tool0", "wrist_3_link", HAND_LINK, COUPLING_LINK)

#: The arm links a camera body may never ignore, because they are the ones it can hit.
_WRIST_BODY_NEVER_IGNORES = ("wrist_1_link", "wrist_2_link")

#: The one arm link a hand on a UR flange is always checked against. The hand ignores wrist_2 and wrist_3, as the exact
#: guard skips them, so wrist_1 is where a hand in the wrong place shows.
_HAND_NEVER_IGNORES = "wrist_1_link"

#: Where a hand map's numbers start (``robot/gripper_spheres.py``, which this module cannot import).
_FLANGE = "flange"
_MOUNTING_FACE = "mounting_face"

#: cuRobo's empty sphere slot: a radius no pair distance can ever overcome.
_EMPTY_SLOT = {"center": [0.0, 0.0, 0.0], "radius": -100.0}


class BodyLinkError(ValueError):
    """A body link that cannot be composed into this config as described."""


def hand_body_link(
    *,
    spheres: Sequence[Mapping[str, Any]],
    rotation: Sequence[Sequence[float]],
    approach_in_tool0: Sequence[float],
    origin: str,
    coupling_mm: float,
) -> dict[str, Any]:
    """The hand as a body under tool0, in the shape :func:`apply_body_links` reads.

    ``spheres`` are the hand map's, in the hand model's frame. ``rotation`` is where that frame sits on tool0 (R_TM) and
    ``approach_in_tool0`` its approach, both from the placement the cell's declared tool frame gives. A mounting face
    map sits one plate out along that approach and a flange map where it is, as the exact guard places the hand's
    vertices. One derivation, for ``body_link.HandLink`` and for the scripts that run where the repository cannot be
    imported.
    """
    if origin not in (_FLANGE, _MOUNTING_FACE):
        raise BodyLinkError(f"a hand map starts at {_FLANGE!r} or at {_MOUNTING_FACE!r}, and this one says {origin!r}")
    plate_m = float(coupling_mm) / 1000.0 if origin == _MOUNTING_FACE else 0.0
    return {
        "link": HAND_LINK,
        "parent": HAND_PARENT,
        "joint": HAND_JOINT,
        "fixed_transform": fixed_transform_wxyz(rotation, [plate_m * float(axis) for axis in approach_in_tool0]),
        "spheres": [
            {"center": [float(v) for v in sphere["center"]], "radius": float(sphere["radius"])} for sphere in spheres
        ],
        "slots": 0,
        "buffer_m": 0.0,
        "ignore": list(HAND_IGNORE),
    }


def coupling_body_link(
    *,
    boxes: Sequence[Any],
    rotation: Sequence[Sequence[float]],
    reach_mm: float = COUPLING_REACH_MM,
) -> "dict[str, Any] | None":
    """The declared coupling plates as one body under tool0, or ``None`` where a cell declared none.

    ``boxes`` are :class:`_declared_body.Box` in the hand model's own axes, stacked from the flange along the
    approach; ``rotation`` is where that frame sits on tool0, the same R_TM the hand link carries. The plates do not
    move relative to the flange, so the link's transform is that rotation with no translation.

    ``None`` is the answer for every shipped cell: the one plate this repository declares is the Hand-E's 20 mm
    adapter, whose cross section nobody measured, so it holds the hand out and becomes no body. A body the planner
    cannot see is a false clear, and a body invented from a guessed width is a false collide nobody can tell from a
    measured one, so the plate is named instead.
    """
    from ._declared_body import box_spheres

    listed = list(boxes)
    if not listed:
        return None
    spheres: list[dict[str, Any]] = []
    for box in listed:
        spheres.extend(box_spheres(box, reach_mm=float(reach_mm)))
    return {
        "link": COUPLING_LINK,
        "parent": HAND_PARENT,
        "joint": COUPLING_JOINT,
        "fixed_transform": fixed_transform_wxyz(rotation, [0.0, 0.0, 0.0]),
        "spheres": spheres,
        "slots": 0,
        "buffer_m": 0.0,
        "ignore": list(HAND_IGNORE),
    }


def wrist_body_link(
    *,
    rig_id: str,
    boxes: Sequence[Any],
    rotation: Sequence[Sequence[float]],
    translation_m: Sequence[float],
    reach_mm: float = WRIST_BODY_REACH_MM,
) -> dict[str, Any]:
    """A wrist camera's grown boxes as one body under tool0, in the shape :func:`apply_body_links` reads.

    ``boxes`` are :class:`_declared_body.Box` in the colour camera's optical frame, already grown by the rig's margin.
    ``rotation`` and ``translation_m`` place that frame on tool0: the recorded flange to TCP times the calibrated
    CAMERA to TOOL. The spheres stay in the optical frame, which is the link's own.
    """
    from ._declared_body import box_spheres

    listed = list(boxes)
    if not listed:
        raise BodyLinkError(f"the wrist camera of rig {rig_id!r} has no boxes, so it has no body to plan with")
    spheres: list[dict[str, Any]] = []
    for box in listed:
        spheres.extend(box_spheres(box, reach_mm=float(reach_mm)))
    link = f"{WRIST_BODY_PREFIX}{rig_id}"
    return {
        "link": link,
        "parent": HAND_PARENT,
        "joint": f"{link}_joint",
        "fixed_transform": fixed_transform_wxyz(rotation, translation_m),
        "spheres": spheres,
        "slots": 0,
        "buffer_m": 0.0,
        "ignore": list(WRIST_BODY_IGNORE),
    }


def canonical_sha256(value: Any) -> str:
    """The sha256 hex digest of ``value`` as JSON with sorted keys and compact separators."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rotation_to_wxyz(rotation: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """The unit quaternion ``(w, x, y, z)`` of a proper rotation matrix, with ``w`` non negative and no ``-0.0``."""
    m = [[float(v) for v in row] for row in rotation]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s
    elif m[0][0] >= m[1][1] and m[0][0] >= m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] >= m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (w + 0.0, x + 0.0, y + 0.0, z + 0.0)


def fixed_transform_wxyz(rotation: Sequence[Sequence[float]], translation_m: Sequence[float]) -> list[float]:
    """cuRobo's ``fixed_transform`` for a body: ``[x, y, z, qw, qx, qy, qz]``, metres, in the parent's frame."""
    translation = [float(v) + 0.0 for v in translation_m]
    if len(translation) != 3:
        raise BodyLinkError(f"a translation has three components, and this one has {len(translation)}")
    return [*translation, *rotation_to_wxyz(rotation)]


def apply_body_links(
    config: Mapping[str, Any], bodies: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(config_copy, links_added)`` with every body declared as a fixed link, in order.

    A body is ``{link, parent, joint, fixed_transform, spheres, slots, buffer_m, ignore}``: ``spheres`` in the body's own
    frame, in metres; ``slots`` empty sphere slots after them; ``ignore`` the links it is never checked against. Other
    keys are ignored. Every refusal is a :class:`BodyLinkError`, and ``config`` is never mutated.
    """
    out = copy.deepcopy(dict(config))
    if not bodies:
        return out, []
    robot_cfg = out.get("robot_cfg")
    kinematics = robot_cfg.get("kinematics") if isinstance(robot_cfg, dict) else None
    if not isinstance(kinematics, dict):
        raise BodyLinkError("the config has no robot_cfg.kinematics block, so there is no robot to add a body link to")
    spheres = kinematics.get("collision_spheres")
    if not isinstance(spheres, dict):
        raise BodyLinkError(
            f"collision_spheres is a {type(spheres).__name__}, not an inline table, so a body's spheres have nowhere to "
            "go: compose onto a descriptor that writes its spheres out, as build_ur_config.py does"
        )
    tool_frames = kinematics.get("tool_frames") or []
    if not tool_frames:
        raise BodyLinkError("the config names no tool_frames, so no body has a frame to hang from")
    tool_frame = str(tool_frames[0])
    provenance = out.get("_provenance")
    carried = provenance.get("gripper_key") if isinstance(provenance, dict) else None
    if tool_frame in spheres or carried:
        what = f"gripper_key {carried!r}" if carried else f"{tool_frame} spheres"
        raise BodyLinkError(
            f"this descriptor already models a hand ({what}), so a body link on it would count the hand twice: start "
            "the planner on the arm's own descriptor, which carries no hand"
        )
    if tool_frame in (kinematics.get("collision_link_names") or ()):
        raise BodyLinkError(
            f"this config guards the tool frame {tool_frame!r} as a collision link and holds no spheres for it, which "
            "cuRobo raises on: it is a template, not an arm descriptor. Build the arm with build_ur_config.py"
        )

    added: list[str] = []
    for body in bodies:
        link = str(body.get("link") or "")
        parent = str(body.get("parent") or "")
        if not link:
            raise BodyLinkError("a body link has no name")
        existing = (
            set(kinematics.get("collision_link_names") or ()) | set(spheres)
            | set(kinematics.get("extra_links") or {}) | set(kinematics.get("self_collision_ignore") or {})
            | set(kinematics.get("self_collision_buffer") or {})
        )
        if link in existing:
            raise BodyLinkError(
                f"a link named {link!r} already exists in this config, as a link or as a key of one of its tables, so a "
                "body of that name would inherit or overwrite what that entry says"
            )
        if parent != tool_frame and parent not in added:
            raise BodyLinkError(
                f"body {link!r} hangs from {parent!r}, and a body hangs from the tool frame {tool_frame!r} or from a body "
                "added before it"
            )
        ignore = [str(name) for name in body.get("ignore") or ()]
        if parent not in ignore:
            raise BodyLinkError(
                f"body {link!r} is not ignored against its parent {parent!r}, which it touches by construction, so the "
                "planner would refuse every configuration"
            )
        if link == HAND_LINK and _HAND_NEVER_IGNORES in ignore:
            raise BodyLinkError(
                f"the hand ignores {_HAND_NEVER_IGNORES}, the one link whose distance to the hand shows a wrong "
                "placement, so a hand in the wrong place would plan with nothing noticing"
            )
        if link.startswith(WRIST_BODY_PREFIX):
            if parent != tool_frame:
                raise BodyLinkError(
                    f"wrist camera {link!r} hangs from {parent!r}, and a camera is placed on the flange by its "
                    f"calibration, so it hangs from the tool frame {tool_frame!r}"
                )
            blind = [name for name in _WRIST_BODY_NEVER_IGNORES if name in ignore]
            if blind:
                raise BodyLinkError(
                    f"wrist camera {link!r} ignores {blind}, the arm links a camera on the flange can hit, so a "
                    "housing driven into the wrist would plan with nothing noticing"
                )
        body_spheres = [
            {"center": [float(v) for v in sphere["center"]], "radius": float(sphere["radius"])}
            for sphere in body.get("spheres") or ()
        ]
        slots = int(body.get("slots") or 0)
        if not body_spheres and slots <= 0:
            raise BodyLinkError(
                f"body {link!r} has no spheres and no slots, and cuRobo refuses a collision link with nothing in it"
            )
        transform = [float(v) for v in body.get("fixed_transform") or ()]
        if (len(transform) != 7 or not all(math.isfinite(v) for v in transform)
                or abs(math.sqrt(sum(v * v for v in transform[3:])) - 1.0) > 1e-6):
            raise BodyLinkError(
                f"body {link!r} has fixed_transform {transform}, and cuRobo reads [x, y, z, qw, qx, qy, qz] with a unit "
                "quaternion"
            )
        kinematics["extra_links"] = {
            **(kinematics.get("extra_links") or {}),
            link: {
                "link_name": link,
                "parent_link_name": parent,
                "joint_name": str(body.get("joint") or f"{link}_joint"),
                "joint_type": "FIXED",
                "fixed_transform": transform,
            },
        }
        kinematics["collision_link_names"] = [*(kinematics.get("collision_link_names") or ()), link]
        spheres[link] = body_spheres + [copy.deepcopy(_EMPTY_SLOT) for _ in range(slots)]
        buffer = kinematics.get("self_collision_buffer")
        if isinstance(buffer, dict):
            buffer[link] = float(body.get("buffer_m") or 0.0)
        kinematics["self_collision_ignore"] = {**(kinematics.get("self_collision_ignore") or {}), link: ignore}
        added.append(link)
    return out, added


def body_report(config: Mapping[str, Any], links: Sequence[str]) -> list[dict[str, Any]]:
    """What a composed config holds for each body: where it hangs, its spheres and their hash, its buffer and ignores."""
    kinematics = config["robot_cfg"]["kinematics"]
    report = []
    for link in links:
        declared = kinematics["extra_links"][link]
        entries = kinematics["collision_spheres"][link]
        real = [sphere for sphere in entries if float(sphere["radius"]) > 0.0]
        report.append({
            "link": link,
            "parent": declared["parent_link_name"],
            "fixed_transform": list(declared["fixed_transform"]),
            "spheres": len(real),
            "slots": len(entries) - len(real),
            "spheres_sha256": canonical_sha256(real),
            "buffer_m": (kinematics.get("self_collision_buffer") or {}).get(link),
            "ignore": list(kinematics["self_collision_ignore"][link]),
        })
    return report


def apply_default_q(config: dict[str, Any], default_q: Sequence[float]) -> None:
    """Write the retract this arm and hand were judged at into the config the planner loads.

    A descriptor is per arm and the hand arrives as a body link, so the retract in it can only ever be right about a
    bare arm. At the ur3's own retract the exact meshes clear the Hand-E by 13.4 mm and the planner's sphere model calls
    the same pose a self collision, so the sidecar never becomes ready. The pose that works is a property of the pair,
    and this is where the pair exists.

    Only the keys this config already uses. cuRobo parses the cspace block into ``CSpaceParams``, which refuses a
    keyword it does not know, so writing a field a descriptor does not carry takes the sidecar down at load with a
    TypeError and no plan at all. The descriptors in this repository carry ``default_joint_position`` and no
    ``retract_config``.
    """
    cspace = config.get("robot_cfg", config).get("kinematics", {}).get("cspace")
    if not isinstance(cspace, dict):
        raise BodyLinkError(
            "this config has no cspace block, so there is nowhere to put a retract: it is not a robot "
            "config this planner can start from"
        )
    joints = cspace.get("joint_names") or ()
    pose = list(default_q)
    if len(pose) != len(joints):
        raise BodyLinkError(
            f"this retract has {len(pose)} joint(s) and the arm has {len(joints)}: a pose for another "
            f"robot is not a retract for this one"
        )
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in pose):
        raise BodyLinkError(f"this retract is not six numbers: {pose!r}")
    written = [key for key in ("retract_config", "default_joint_position") if key in cspace]
    if not written:
        raise BodyLinkError(
            "this cspace names neither retract_config nor default_joint_position, so there is no field to put a "
            "retract in that the planner would read"
        )
    for key in written:
        cspace[key] = [float(value) for value in pose]


def compose_with_counts(
    config: Mapping[str, Any],
    *,
    bodies: Sequence[Mapping[str, Any]] = (),
    margin_mm: float = 0.0,
    attach_spheres: int = 0,
    default_q: "Sequence[float] | None" = None,
) -> tuple[dict[str, Any], int, bool]:
    """The composed config, how many buffers took the margin, and whether a payload link was added.

    The counts are what the sidecar says on stderr: a margin that found no buffer to raise is the planner running
    without the guard's clearance, and has to be loud. ``config`` is never mutated, and the result shares nothing with
    it.
    """
    composed, _ = apply_body_links(config, bodies)
    # The retract goes in with the hand, before the margin and the payload: it is a property of the arm and the hand
    # together, and the two arrive together or not at all.
    if default_q is not None:
        apply_default_q(composed, default_q)
    composed, margin_links = apply_self_collision_margin(composed, float(margin_mm))
    composed, attach_added = apply_attached_object_link(composed, spheres=int(attach_spheres),
                                                        checked_prefixes=(WRIST_BODY_PREFIX,))
    return composed, margin_links, attach_added


def without_links(config: Mapping[str, Any], links: Sequence[str]) -> dict[str, Any]:
    """``config`` with every trace of ``links`` removed from the kinematics tables, and nothing else touched.

    Removed are the link's ``extra_links`` entry, its place in ``collision_link_names``, and its own key in
    ``collision_spheres``, ``self_collision_buffer`` and ``self_collision_ignore``. What another link's entries say
    about it stays, so a composition that wrote outside the link's own keys shows up as a difference rather than being
    cleaned away.
    """
    out = copy.deepcopy(dict(config))
    names = set(links)
    kinematics = out["robot_cfg"]["kinematics"]
    kinematics["collision_link_names"] = [name for name in kinematics.get("collision_link_names") or ()
                                          if name not in names]
    for table in ("extra_links", "collision_spheres", "self_collision_buffer", "self_collision_ignore"):
        held = kinematics.get(table)
        if isinstance(held, dict):
            kinematics[table] = {key: value for key, value in held.items() if key not in names}
    return out


def _first_difference(left: Any, right: Any, path: str = "") -> str:
    """The first key path at which two JSON values differ, for a refusal a person can follow."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}".lstrip(".")
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return ""
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} (length {len(left)} against {len(right)})".lstrip(".")
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return ""
    return "" if left == right else (path.lstrip(".") or "the whole config")


def compose_for_cell(
    config: Mapping[str, Any],
    *,
    bodies: Sequence[Mapping[str, Any]] = (),
    wrist_bodies: Sequence[Mapping[str, Any]] = (),
    margin_mm: float = 0.0,
    attach_spheres: int = 0,
    default_q: "Sequence[float] | None" = None,
) -> tuple[dict[str, Any], dict[str, Any], int, bool, list[str]]:
    """``(loaded, evidence, margin_links, attach_added, wrist_links)``: what the planner loads, and what evidence names.

    The combination evidence is keyed by arm, hand, plate, placement and margin, and formed without a wrist camera.
    ``evidence`` is :func:`compose_with_counts` over ``bodies`` alone, byte for byte what a cell without a camera
    composes; ``loaded`` adds ``wrist_bodies`` on top. This refuses unless ``loaded`` without the camera's links is
    ``evidence`` exactly, which turns the reading of every step into a check: each writes only its own link's keys,
    the margin raises buffers link by link, and the payload's ignore entries skip the camera. Without a wrist body the
    two are the same config.
    """
    for body in bodies:
        if str(body.get("link") or "").startswith(WRIST_BODY_PREFIX):
            raise BodyLinkError(
                f"{body.get('link')!r} is a wrist camera sent among the bodies the evidence names; a camera is sent "
                "as a wrist body, so the evidence stays keyed without it"
            )
    for body in wrist_bodies:
        if not str(body.get("link") or "").startswith(WRIST_BODY_PREFIX):
            raise BodyLinkError(
                f"{body.get('link')!r} was sent as a wrist body and is not named {WRIST_BODY_PREFIX}<rig>, so the "
                "evidence config could not be told apart from it"
            )
    evidence, margin_links, attach_added = compose_with_counts(
        config, bodies=bodies, margin_mm=margin_mm, attach_spheres=attach_spheres, default_q=default_q)
    if not wrist_bodies:
        return evidence, evidence, margin_links, attach_added, []
    loaded, _, _ = compose_with_counts(
        config, bodies=[*bodies, *wrist_bodies], margin_mm=margin_mm, attach_spheres=attach_spheres,
        default_q=default_q)
    wrist_links = [str(body.get("link")) for body in wrist_bodies]
    stripped = without_links(loaded, wrist_links)
    if stripped != evidence:
        raise BodyLinkError(
            f"adding the wrist camera {wrist_links} changed the config outside its own links, at "
            f"{_first_difference(stripped, evidence)}, so the combination evidence would no longer name the robot "
            "the planner loads"
        )
    return loaded, evidence, margin_links, attach_added, wrist_links


def compose_sidecar_config(
    config: Mapping[str, Any],
    *,
    bodies: Sequence[Mapping[str, Any]] = (),
    margin_mm: float = 0.0,
    attach_spheres: int = 0,
    default_q: "Sequence[float] | None" = None,
) -> dict[str, Any]:
    """The config the planner loads: ``config`` with body links, the retract, the margin, the payload link."""
    return compose_with_counts(config, bodies=bodies, margin_mm=margin_mm, attach_spheres=attach_spheres,
                               default_q=default_q)[0]
