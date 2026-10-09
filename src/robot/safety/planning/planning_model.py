"""The planning model a cell asks for, composed on the client: committed spheres for the arm and the hand, a coarse fill
for a wrist camera's boxes, and the pairs the exact guard decides.

``safety.planned_motion.planning_spheres`` names it: ``""``, the default, plans on the robot the sidecar judges with,
as always; ``lean`` plans on the spheres NVIDIA's own UR configs are made of, a few along each link's axis
(:data:`LEAN`). The sidecar builds its planner, IK and graph planner from it and keeps judging every sample with the
evidence model and the exact guard (``_curobo_planning``), so the model decides what cuRobo proposes and nothing that
is judged.

What a lean model is made of, per part:

* the arm: ``robot/{arm}_arm_lean_spheres.yml``, written by ``scripts/curobo/fit_planning_spheres.py`` from the arm's
  committed bundle, in the bundle's DH frames, with the DH rows the sidecar places them with;
* the hand: ``robot/{hand}_gripper_lean_spheres.yml``, from the hand's bundle by the same script, in the hand map's own
  frame, and hung where the evidence model's hand hangs (the same fixed transform);
* a wrist camera: its grown boxes filled at :data:`LEAN_BOX_REACH_MM` (``_declared_body.box_spheres``), a complete
  cover with a few spheres where the evidence model fills them at 8 mm;
* a coupling plate with a declared cross section, which no shipped cell has: as the evidence model holds it;
* the ignored pairs: every pair of links that both stand for parts the exact guard holds (``band.guard_parts``): the
  arm from the upper arm to wrist_3, the hand and the wrist cameras. Those the guard judges at every sample, or they
  are the neighbours every descriptor ignores; the shoulder link, the planner's only model of the robot's base, stays
  checked against all of them.

An arm or a hand with no committed lean file, a carried part the planner models, or a name this module does not know
gives a sentence instead (:func:`planning_payload`), and the client plans on the evidence model and logs it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ._curobo_planning import ENV_PLANNING_MODEL
from ._declared_body import Box, box_spheres
from .band import guard_parts

__all__ = [
    "ENV_PLANNING_MODEL",
    "LEAN",
    "LEAN_BOX_REACH_MM",
    "PLANNING_MODELS",
    "PlanningPayload",
    "arm_planning_spheres_path",
    "hand_planning_spheres_path",
    "planning_payload",
]

#: NVIDIA's way: a few spheres along each link's axis, sized to the CB3's own links (``fit_planning_spheres.py``).
LEAN = "lean"
#: The names ``safety.planned_motion.planning_spheres`` takes; ``""`` plans on the evidence model, as always.
PLANNING_MODELS = ("", LEAN)
#: How far a lean fill of a wrist camera's boxes may reach past them, millimetres: the box's complete cover with cells
#: of 20 / (sqrt(3) - 1) * 2 = 54.6 mm, three spheres along a D415 in its enclosure where 8 mm takes 28.
LEAN_BOX_REACH_MM = 20.0

_MAPS = Path(__file__).resolve().parent / "robot"
#: The DH frame each arm link's spheres are written in, as the committed bundles and cover fits hold them.
_ARM_FRAMES = {"shoulder": 1, "upper_arm": 2, "forearm": 3, "wrist_1": 4, "wrist_2": 5, "wrist_3": 6}


def arm_planning_spheres_path(arm: str, model: str = LEAN) -> Path:
    """The committed planning spheres of ``arm`` for ``model``, named after both, so no table maps names."""
    return _MAPS / f"{arm.lower()}_arm_{model}_spheres.yml"


def hand_planning_spheres_path(hand: str, model: str = LEAN) -> Path:
    """The committed planning spheres of the hand ``hand`` for ``model``."""
    return _MAPS / f"{hand}_gripper_{model}_spheres.yml"


@dataclass(frozen=True)
class PlanningPayload:
    """A planning model as the sidecar reads it (``WILLY_CUROBO_PLANNING_MODEL``), and what it holds."""

    model: str
    payload: "dict[str, Any]"

    @property
    def spheres(self) -> int:
        """Every sphere of the model: the arm's and the bodies'."""
        arm = sum(len(block["spheres"]) for block in self.payload["links"].values())
        return arm + sum(len(body["spheres"]) for key in ("bodies", "wrist_bodies") for body in self.payload[key])

    def to_json(self) -> str:
        """The payload as compact JSON, the form the sidecar's environment carries, its links in the arm's order."""
        return json.dumps(self.payload, separators=(",", ":"))

    def render(self) -> str:
        bodies = ", ".join(f"{body['link']} {len(body['spheres'])}"
                           for key in ("bodies", "wrist_bodies") for body in self.payload[key])
        arm = sum(len(block["spheres"]) for block in self.payload["links"].values())
        return (f"planning model {self.model}: {self.spheres} spheres (arm {arm}{', ' + bodies if bodies else ''}), "
                f"{len(self.payload['ignore'])} link pair(s) left to the exact guard")


def _read(path: Path) -> "dict[str, Any]":
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} holds no mapping")
    return document


def _rounded(spheres: Sequence[Mapping[str, Any]]) -> "list[dict[str, Any]]":
    return [{"center": [round(float(v), 6) for v in sphere["center"]], "radius": round(float(sphere["radius"]), 6)}
            for sphere in spheres]


def _arm_links(arm: str, model: str) -> "tuple[dict[str, Any], list[list[float]]] | str":
    path = arm_planning_spheres_path(arm, model)
    if not path.is_file():
        return f"no {model} planning spheres are committed for the {arm} ({path.name})"
    document = _read(path)
    rows = (document.get("_provenance") or {}).get("dh_m")
    blocks = document.get("collision_spheres") or {}
    if sorted(blocks) != sorted(_ARM_FRAMES):
        return f"{path.name} holds the bodies {sorted(blocks)}, and an arm has {sorted(_ARM_FRAMES)}"
    links: dict[str, Any] = {}
    for body, frame in _ARM_FRAMES.items():
        block = blocks[body]
        if int(block.get("frame", -1)) != frame or not block.get("spheres"):
            return f"{path.name}: {body} is not written in DH frame {frame}, or holds no spheres"
        links[f"{body}_link"] = {"frame": frame, "spheres": _rounded(block["spheres"])}
    if not isinstance(rows, list) or len(rows) != 6:
        return f"{path.name} records no six DH rows to place its spheres with"
    return links, [[float(v) for v in row] for row in rows]


def _hand_spheres(body: Mapping[str, Any], model: str) -> "list[dict[str, Any]] | str":
    hand = str(body.get("model") or "")
    if not hand:
        return "the hand body names no model"
    path = hand_planning_spheres_path(hand, model)
    if not path.is_file():
        return f"no {model} planning spheres are committed for the {hand} ({path.name})"
    document = _read(path)
    origin = (document.get("_provenance") or {}).get("origin")
    if origin != body.get("origin"):
        return (f"{path.name} starts at {origin!r} and the hand's map at {body.get('origin')!r}: its spheres would "
                "hang a plate away from the hand")
    spheres = (document.get("collision_spheres") or {}).get("tool0") or []
    return _rounded(spheres) if spheres else f"{path.name} holds no spheres under collision_spheres.tool0"


def _camera_spheres(body: Mapping[str, Any]) -> "list[dict[str, Any]] | str":
    boxes = body.get("boxes") or []
    if not boxes:
        return f"the wrist camera {body.get('link')!r} was sent without its boxes, so it cannot be filled anew"
    spheres: list[dict[str, Any]] = []
    for box in boxes:
        spheres.extend(box_spheres(Box(name=str(box["name"]), centre_mm=tuple(float(v) for v in box["centre_mm"]),  # type: ignore[arg-type]
                                       half_extents_mm=tuple(float(v) for v in box["half_extents_mm"])),  # type: ignore[arg-type]
                                   reach_mm=LEAN_BOX_REACH_MM))
    return _rounded(spheres)


def _body(body: Mapping[str, Any], spheres: "list[dict[str, Any]]") -> "dict[str, Any]":
    """``body`` as ``apply_body_links`` reads it, with ``spheres`` in place of its own and nothing else changed."""
    keys = ("link", "parent", "joint", "fixed_transform", "slots", "buffer_m", "ignore")
    return {**{key: body[key] for key in keys if key in body}, "spheres": spheres}


def planning_payload(
    model: str,
    *,
    arm: "str | None",
    body_links: Sequence[Mapping[str, Any]],
    wrist_body_links: Sequence[Mapping[str, Any]],
    attach_spheres: int = 0,
) -> "PlanningPayload | str":
    """The planning model ``model`` for this arm and these bodies, or the sentence why this cell plans on the evidence
    model: an unknown name, no arm the bundled DH tables know, a part without committed spheres, a carried part."""
    if model not in PLANNING_MODELS or not model:
        return f"{model!r} names no planning model; the names are {[name for name in PLANNING_MODELS if name]}"
    if int(attach_spheres) > 0:
        return ("the planner models a carried part, which only its own kinematics carries, so the check that judges "
                "every plan would no longer see it")
    if arm is None:
        return "the descriptor names no arm this repository has planning spheres for"
    read = _arm_links(arm, model)
    if isinstance(read, str):
        return read
    links, rows = read
    bodies: list[dict[str, Any]] = []
    for body in body_links:
        if body.get("link") == "hand":
            spheres = _hand_spheres(body, model)
            if isinstance(spheres, str):
                return spheres
            bodies.append(_body(body, spheres))
        else:
            bodies.append(_body(body, _rounded(body.get("spheres") or [])))
    cameras: list[dict[str, Any]] = []
    for body in wrist_body_links:
        spheres = _camera_spheres(body)
        if isinstance(spheres, str):
            return spheres
        cameras.append(_body(body, spheres))
    names = [*links, *(body["link"] for body in bodies), *(body["link"] for body in cameras)]
    guarded = [name for name in names if guard_parts(str(name))]
    ignore = [[a, b] for index, a in enumerate(guarded) for b in guarded[index + 1:]]
    return PlanningPayload(model=model, payload={
        "model": model, "dh_m": rows, "links": links, "bodies": bodies, "wrist_bodies": cameras, "ignore": ignore,
    })
