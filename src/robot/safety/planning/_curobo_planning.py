"""The robot cuRobo plans with, where it is leaner than the robot the sidecar judges with. Standard library only.

The sidecar holds one robot today: the arm's cover fit (747 spheres on a UR10, every surface sample inside one), the
hand's (70 on the Hand-E) and the wrist camera's box fill, composed with the margin and the retract
(``_curobo_body_links.compose_for_cell``). The planner, its IK, its graph planner, the check behind ``check_js`` and
the ready gate are all built on it, and the combination evidence names it (``composed_sha256``). On the owner's cell
that is 845 spheres and 172,414 sphere pairs (841 and 169,642 with the 111 mm enclosure of 2026-10-08), about 150
times the collision arithmetic of NVIDIA's own UR10e (``ur10e.yml``: 20 spheres, 83 pairs) in every iteration of
every plan.

A planning model is a second robot for the planner alone (the owner, 2026-10-08: "CuRobo richtig verwenden"): the same
URDF, joints, links, bodies, retract and margin, with spheres the way NVIDIA's UR configs place them, a few along each
link's axis, and with the self pairs the exact guard decides (``band.py``) left to that guard. The planner, its IK
and its graph planner are built from it; the check behind ``check_js`` and ``explain_js``, the ready gate and every
evidence hash stay on the evidence model, so what is judged, and how, does not change: every plan is judged sample by
sample as before, by the evidence model's ``check_js`` and by the exact guard.

What the client sends (``WILLY_CUROBO_PLANNING_MODEL``, JSON, :func:`read_payload`):

* ``model``: its name (``lean``);
* ``dh_m``: the arm's six DH rows ``[a, d, alpha]`` in metres and radians, the rows its committed bundle was baked in;
* ``links``: per arm link, the DH frame its spheres are written in and the spheres, metres, in that frame;
* ``bodies``, ``wrist_bodies``: the hand, its plates and the wrist cameras, as ``_curobo_body_links`` reads a body,
  hanging exactly where the evidence model's bodies hang, with the planning model's spheres;
* ``ignore``: link pairs the planning model does not check against one another.

The sidecar places the arm's spheres from their DH frames into the URDF link frames of the URDF its kinematics
resolved (:func:`dh_to_link`), the placement ``scripts/curobo/build_ur_config.py`` writes the evidence descriptor's
spheres with, and composes the bodies over it with ``compose_for_cell`` as it composes the evidence model
(:func:`planning_config`). Anything it cannot place or compose raises :class:`PlanningModelError`, and the sidecar
plans on the evidence model and says why, as it does on a cell that models a carried part: only the planner's own
kinematics carries that part, and the check would no longer see it.

Imported from both sides of the process boundary, like ``_curobo_body_links``: as part of the Willy package under
python 3.11, and as a plain sibling module by ``curobo_planner_server.py`` under python 3.10.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    from ._curobo_body_links import canonical_sha256, compose_for_cell
except ImportError:  # loaded by the sidecar as a plain sibling, with no package around it
    from _curobo_body_links import canonical_sha256, compose_for_cell  # type: ignore[import-not-found,no-redef]

__all__ = [
    "ENV_PLANNING_MODEL",
    "PlanningModelError",
    "dh_frames",
    "dh_to_link",
    "planning_config",
    "planning_row",
    "read_payload",
    "self_pair_count",
    "sphere_count",
    "urdf_link_frames",
    "with_ignored_pairs",
]

#: The planning model the client asks for, as JSON; unset plans on the evidence model, as always.
ENV_PLANNING_MODEL = "WILLY_CUROBO_PLANNING_MODEL"

Matrix = list[list[float]]


class PlanningModelError(ValueError):
    """A planning model the sidecar cannot build as sent: it plans on the evidence model and says why."""


def _identity() -> Matrix:
    return [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]


def _product(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)] for row in range(4)]


def _rigid_inverse(matrix: Matrix) -> Matrix:
    """The inverse of a rigid transform: the transposed rotation, and the translation turned back through it."""
    rotation = [[matrix[column][row] for column in range(3)] for row in range(3)]
    translation = [-sum(rotation[row][k] * matrix[k][3] for k in range(3)) for row in range(3)]
    return [[*rotation[0], translation[0]], [*rotation[1], translation[1]], [*rotation[2], translation[2]],
            [0.0, 0.0, 0.0, 1.0]]


def _axis_angle(axis: Sequence[float], theta: float) -> "list[list[float]]":
    norm = math.sqrt(sum(float(v) * float(v) for v in axis)) or 1.0
    x, y, z = (float(v) / norm for v in axis)
    c, s = math.cos(theta), math.sin(theta)
    return [[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]]


def _rotation_product(left: "list[list[float]]", right: "list[list[float]]") -> "list[list[float]]":
    return [[sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)] for row in range(3)]


def urdf_link_frames(urdf_text: str) -> "dict[str, Matrix]":
    """Every link of a URDF at the zero configuration, as a 4x4 in its root link's frame, metres.

    The joint origins alone, ``Rz(yaw) Ry(pitch) Rx(roll)`` and ``xyz``, as ``_arm_spheres.link_frames`` reads them
    for the descriptor builder: a revolute joint at zero adds nothing to its origin.
    """
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError as exc:
        raise PlanningModelError(f"the URDF does not parse: {exc}") from exc
    parents: dict[str, tuple[str, Matrix]] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
        child, parent = joint.find("child"), joint.find("parent")
        if child is None or parent is None or len(rpy) != 3 or len(xyz) != 3:
            raise PlanningModelError(f"joint {joint.attrib.get('name', '?')!r} names no child or no parent, or no "
                                     "whole origin")
        turn = _rotation_product(_rotation_product(_axis_angle((0, 0, 1), rpy[2]), _axis_angle((0, 1, 0), rpy[1])),
                                 _axis_angle((1, 0, 0), rpy[0]))
        parents[child.attrib["link"]] = (parent.attrib["link"], [[*turn[0], xyz[0]], [*turn[1], xyz[1]],
                                                                  [*turn[2], xyz[2]], [0.0, 0.0, 0.0, 1.0]])
    if not parents:
        raise PlanningModelError("the URDF declares no joints, so no link has a frame")
    roots = {parent for parent, _ in parents.values()} - set(parents)
    if len(roots) != 1:
        raise PlanningModelError(f"the URDF has {len(roots)} root links ({sorted(roots)}), and a robot has one")
    frames: dict[str, Matrix] = {next(iter(roots)): _identity()}

    def frame(link: str, depth: int = 0) -> Matrix:
        if link not in frames:
            if depth > len(parents):
                raise PlanningModelError(f"the URDF's joints form a loop at {link!r}")
            parent, step = parents[link]
            frames[link] = _product(frame(parent, depth + 1), step)
        return frames[link]

    for link in parents:
        frame(link)
    return frames


def dh_frames(rows: Sequence[Sequence[float]]) -> "list[Matrix]":
    """``T_dh[0..n]`` of a DH chain at the zero configuration: frame ``k`` after ``k`` rows of ``[a, d, alpha]``."""
    out = [_identity()]
    for row in rows:
        a, d, alpha = (float(v) for v in row)
        ca, sa = math.cos(alpha), math.sin(alpha)
        out.append(_product(out[-1], [[1.0, 0.0, 0.0, a], [0.0, ca, -sa, 0.0], [0.0, sa, ca, d],
                                      [0.0, 0.0, 0.0, 1.0]]))
    return out


def dh_to_link(urdf_text: str, rows: Sequence[Sequence[float]]) -> "Callable[[str, int], Matrix]":
    """``place(link, frame)``: the 4x4 that carries a point from DH frame ``frame`` into ``link``'s URDF frame.

    ``inv(urdf_link) @ Rz(pi) @ T_dh[frame]``: UR's description mounts ``base_link_inertia`` yawed half a turn, so
    the DH base is the URDF base turned by pi. The same map ``scripts/curobo/_arm_spheres.link_frames`` gives the
    descriptor builder, which placed the evidence model's spheres with it.
    """
    links = urdf_link_frames(urdf_text)
    chain = dh_frames(rows)
    turned = [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    def place(link: str, frame: int) -> Matrix:
        if link not in links:
            raise PlanningModelError(f"the URDF holds no link {link!r}")
        if not 0 <= int(frame) < len(chain):
            raise PlanningModelError(f"DH frame {frame} lies outside a chain of {len(chain) - 1} rows")
        return _product(_rigid_inverse(links[link]), _product(turned, chain[int(frame)]))

    return place


def _placed(spheres: Sequence[Mapping[str, Any]], matrix: Matrix) -> "list[dict[str, Any]]":
    out = []
    for sphere in spheres:
        x, y, z = (float(v) for v in sphere["center"])
        centre = [matrix[row][0] * x + matrix[row][1] * y + matrix[row][2] * z + matrix[row][3] for row in range(3)]
        out.append({"center": [value + 0.0 for value in centre], "radius": float(sphere["radius"])})
    return out


def _spheres(value: Any, where: str) -> "list[dict[str, Any]]":
    if not isinstance(value, list) or not value:
        raise PlanningModelError(f"{where} holds no spheres")
    out = []
    for sphere in value:
        centre = sphere.get("center") if isinstance(sphere, Mapping) else None
        radius = sphere.get("radius") if isinstance(sphere, Mapping) else None
        if (not isinstance(centre, list) or len(centre) != 3 or not isinstance(radius, (int, float))
                or isinstance(radius, bool) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in centre)
                or not math.isfinite(radius) or radius <= 0.0):
            raise PlanningModelError(f"{where} holds a sphere that is not a centre of three numbers and a radius "
                                     f"above 0: {sphere!r}")
        out.append({"center": [float(v) for v in centre], "radius": float(radius)})
    return out


def read_payload(text: str) -> "dict[str, Any]":
    """The planning model a client sent, checked field by field, or :class:`PlanningModelError` naming the field."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise PlanningModelError(f"the planning model is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlanningModelError("the planning model is not a JSON object")
    name = payload.get("model")
    if not isinstance(name, str) or not name:
        raise PlanningModelError("the planning model has no name")
    rows = payload.get("dh_m")
    if (not isinstance(rows, list) or len(rows) != 6
            or not all(isinstance(row, list) and len(row) == 3
                       and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row) for row in rows)):
        raise PlanningModelError("the planning model's dh_m is not six rows of [a, d, alpha]")
    links = payload.get("links")
    if not isinstance(links, dict) or not links:
        raise PlanningModelError("the planning model names no arm link")
    for link, block in links.items():
        frame = block.get("frame") if isinstance(block, dict) else None
        if not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame <= 6:
            raise PlanningModelError(f"the planning model's {link} has no DH frame from 0 to 6")
        block["spheres"] = _spheres(block.get("spheres"), f"the planning model's {link}")
    for key in ("bodies", "wrist_bodies"):
        listed = payload.get(key, [])
        if not isinstance(listed, list) or not all(isinstance(body, dict) for body in listed):
            raise PlanningModelError(f"the planning model's {key} is not a list of bodies")
        for body in listed:
            body["spheres"] = _spheres(body.get("spheres"), f"the planning model's body {body.get('link')!r}")
        payload[key] = listed
    ignore = payload.get("ignore", [])
    if not isinstance(ignore, list) or not all(
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, str) and v for v in pair) for pair in ignore):
        raise PlanningModelError("the planning model's ignore is not a list of link pairs")
    return payload


def with_ignored_pairs(config: Mapping[str, Any], pairs: Sequence[Sequence[str]]) -> "dict[str, Any]":
    """``config`` with every pair of ``pairs`` taken out of self collision, each written once, under its first link.

    cuRobo's ignore entry is a membership test either way round (``self_collision_params.py``), so one direction is
    enough, and a pair already ignored either way is left as it is.
    """
    out = copy.deepcopy(dict(config))
    kinematics = out["robot_cfg"]["kinematics"]
    table = {str(link): [str(other) for other in others or ()]
             for link, others in (kinematics.get("self_collision_ignore") or {}).items()}
    for pair in pairs:
        a, b = str(pair[0]), str(pair[1])
        if a == b or b in table.get(a, ()) or a in table.get(b, ()):
            continue
        table.setdefault(a, []).append(b)
    kinematics["self_collision_ignore"] = table
    return out


def _kinematics(config: Mapping[str, Any]) -> "Mapping[str, Any]":
    return (config.get("robot_cfg") or config).get("kinematics") or {}


def sphere_count(config: Mapping[str, Any]) -> "dict[str, int]":
    """The sphere slots of every collision link, in ``collision_link_names`` order, as cuRobo allocates them."""
    kinematics = _kinematics(config)
    spheres = kinematics.get("collision_spheres") or {}
    extra = kinematics.get("extra_collision_spheres") or {}
    return {str(link): int(extra[link]) if link in extra else len(spheres.get(link) or ())
            for link in kinematics.get("collision_link_names") or ()}


def self_pair_count(config: Mapping[str, Any]) -> int:
    """How many sphere pairs cuRobo's self collision checks in ``config``: two links that do not ignore one another,
    every slot of one against every slot of the other (``_curobo_pairs.collision_pair_rows`` lists them on load)."""
    counts = sphere_count(config)
    ignored = {frozenset((str(a), str(b))) for a, others in (_kinematics(config).get("self_collision_ignore") or {}).items()
               for b in others or ()}
    links = list(counts)
    return sum(counts[a] * counts[b] for index, a in enumerate(links) for b in links[index + 1:]
               if frozenset((a, b)) not in ignored)


def planning_config(
    descriptor: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    urdf_text: str,
    evidence: Mapping[str, Any],
    margin_mm: float = 0.0,
    default_q: "Sequence[float] | None" = None,
) -> "dict[str, Any]":
    """The robot the planner loads for ``payload``: ``descriptor`` with the planning model's arm spheres, its bodies
    composed over them as the evidence model's are, and its ignored pairs.

    ``evidence`` is the evidence model the sidecar composed from the same descriptor, and the planning model has to be
    that robot in everything but its spheres and its ignore table: the same collision links in the same order, every
    body under the same parent at the same fixed transform, the same cspace and retract. :class:`PlanningModelError`
    otherwise, and for an arm link the descriptor does not check.
    """
    planned = copy.deepcopy(dict(descriptor))
    kinematics = planned.get("robot_cfg", {}).get("kinematics")
    if not isinstance(kinematics, dict) or not isinstance(kinematics.get("collision_spheres"), dict):
        raise PlanningModelError("the descriptor writes no collision spheres out, so there are none to replace")
    place = dh_to_link(urdf_text, payload["dh_m"])
    checked = set(kinematics.get("collision_link_names") or ())
    for link, block in payload["links"].items():
        if link not in checked:
            raise PlanningModelError(f"the planning model gives {link!r} spheres, and the descriptor checks no such link")
        kinematics["collision_spheres"][link] = _placed(block["spheres"], place(link, int(block["frame"])))
    left = sorted(link for link in checked if link not in payload["links"] and kinematics["collision_spheres"].get(link))
    if left:
        raise PlanningModelError(f"the planning model gives no spheres for the arm links {left}")
    try:
        loaded, _, _, _, _ = compose_for_cell(
            planned, bodies=payload.get("bodies", []), wrist_bodies=payload.get("wrist_bodies", []),
            margin_mm=float(margin_mm), attach_spheres=0, default_q=default_q,
        )
    except ValueError as exc:
        raise PlanningModelError(f"the planning model's bodies do not compose: {exc}") from exc
    loaded = with_ignored_pairs(loaded, payload.get("ignore", []))
    ours, theirs = _kinematics(loaded), _kinematics(evidence)
    if list(ours.get("collision_link_names") or ()) != list(theirs.get("collision_link_names") or ()):
        raise PlanningModelError(
            f"the planning model checks the links {list(ours.get('collision_link_names') or ())} and the evidence model "
            f"{list(theirs.get('collision_link_names') or ())}: they are not one robot")
    for link, declared in (theirs.get("extra_links") or {}).items():
        held = (ours.get("extra_links") or {}).get(link)
        if held is None or held.get("parent_link_name") != declared.get("parent_link_name") or any(
                abs(float(a) - float(b)) > 1e-12 for a, b in zip(held.get("fixed_transform") or (),
                                                                  declared.get("fixed_transform") or ())):
            raise PlanningModelError(f"the planning model hangs {link!r} elsewhere than the evidence model does")
    for key in ("cspace", "tool_frames", "base_link", "urdf_path", "self_collision_buffer"):
        if ours.get(key) != theirs.get(key):
            raise PlanningModelError(f"the planning model's {key} is not the evidence model's")
    return loaded


def planning_row(name: str, planned: Mapping[str, Any], evidence: Mapping[str, Any]) -> "dict[str, Any]":
    """What the ready line says of a planning model: its name, spheres and self pairs beside the evidence model's, and
    the hash of the config the planner loaded."""
    counts = sphere_count(planned)
    return {
        "model": str(name),
        "spheres": sum(counts.values()),
        "links": counts,
        "self_pairs": self_pair_count(planned),
        "evidence_spheres": sum(sphere_count(evidence).values()),
        "evidence_self_pairs": self_pair_count(evidence),
        "sha256": canonical_sha256(planned),
    }
