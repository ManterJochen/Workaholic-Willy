"""The hand as its own link of the planner's robot, derived from the resolved hand and from where it sits.

The hand stays its committed map, in the hand model's own frame M (x closing, y approach out of the mounting face), and
the placement the cell's declared tool frame gives (:class:`~._hand_placement.HandPlacement`) becomes the fixed
transform of a link named ``hand`` under ``tool0``:

    p_tool0 = R_TM (p_model + plate along model y) = R_TM p_model + plate * (R_TM e_y),

so the rotation is R_TM and the translation is the plate along the placed approach. That is where the exact mesh guard
puts the hand's vertices (``_fcl_self_collision.place_hand_vertices``), and like the guard a flange map takes no plate.
The arithmetic is ``_curobo_body_links.hand_body_link``, shared with the scripts in the cuRobo environment.

The placement is required. A hand whose declared frame places it nowhere has no link, and the refusal says why: UNSET
read as the identity would model a real UR hand a quarter turn from where it is, so it is refused and never implied.

The hand ignores ``tool0``, ``wrist_3_link`` and ``wrist_2_link``: the frame a descriptor holds an ignore table for, and
the pairs the guard skips. ``wrist_1_link`` stays checked. Its buffer is 0, and the margin raises it like every link's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from src.contracts import chosen

from ._curobo_body_links import (
    HAND_IGNORE,
    HAND_JOINT,
    HAND_LINK,
    HAND_PARENT,
    BodyLinkError,
    canonical_sha256,
    hand_body_link,
)
from ._hand_placement import HandPlacement
from .hand import PlannerHand

__all__ = ["HAND_IGNORE", "HAND_JOINT", "HAND_PARENT", "HandLink"]


@dataclass(frozen=True)
class HandLink:
    """One hand as a fixed link under tool0: its map's spheres in the hand's own frame, and where that frame sits."""

    #: The registry name, ``robot.gripper.model``.
    model: str
    placement: HandPlacement
    #: Where the map's numbers start: ``flange`` or ``mounting_face``.
    origin: str
    #: The plates between the flange and the hand, summed; only a mounting face map is moved by them.
    coupling_mm: float
    #: The map's spheres, verbatim: (centre, radius) in metres, in the hand model's frame.
    spheres: tuple[tuple[tuple[float, ...], float], ...]
    #: cuRobo's ``fixed_transform``: ``(x, y, z, qw, qx, qy, qz)``, metres, in tool0.
    fixed_transform: tuple[float, ...]
    #: :func:`~._curobo_body_links.canonical_sha256` of the spheres as they are sent.
    spheres_sha256: str

    @classmethod
    def from_hand(cls, hand: PlannerHand) -> HandLink:
        """The link ``hand`` is, or :class:`BodyLinkError` where its placement or its map gives none."""
        if not chosen(hand.placement):
            raise BodyLinkError(
                f"{hand.model} cannot be a link of the planner's robot, because its declared tool frame places it "
                f"nowhere: {hand.placement_refusal}"
            )
        document = yaml.safe_load(hand.sphere_map.read_text(encoding="utf-8")) or {}
        entries = (document.get("collision_spheres") or {}).get("tool0") or []
        if not entries:
            raise BodyLinkError(
                f"{hand.sphere_map.name} holds no spheres under collision_spheres.tool0, so {hand.model} has no body to "
                "plan with"
            )
        placement = hand.placement
        body = hand_body_link(
            spheres=entries, rotation=placement.rotation, approach_in_tool0=placement.approach_in_tool0,
            origin=hand.origin, coupling_mm=hand.coupling_mm,
        )
        return cls(
            model=hand.model,
            placement=placement,
            origin=hand.origin,
            coupling_mm=float(hand.coupling_mm),
            spheres=tuple((tuple(sphere["center"]), sphere["radius"]) for sphere in body["spheres"]),
            fixed_transform=tuple(body["fixed_transform"]),
            spheres_sha256=canonical_sha256(body["spheres"]),
        )

    def render(self) -> str:
        x, y, z = (1000.0 * v for v in self.fixed_transform[:3])
        return (
            f"hand link {self.model}: {len(self.spheres)} spheres under {HAND_PARENT} at ({x:g}, {y:g}, {z:g}) mm, "
            f"approach {self.placement.approach}, closing {self.placement.closing}, ignoring {', '.join(HAND_IGNORE)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """The body in the shape ``apply_body_links`` reads, and what it was derived from."""
        return {
            "link": HAND_LINK,
            "parent": HAND_PARENT,
            "joint": HAND_JOINT,
            "fixed_transform": list(self.fixed_transform),
            "spheres": [{"center": list(centre), "radius": radius} for centre, radius in self.spheres],
            "slots": 0,
            "buffer_m": 0.0,
            "ignore": list(HAND_IGNORE),
            "model": self.model,
            "origin": self.origin,
            "coupling_mm": self.coupling_mm,
            "spheres_sha256": self.spheres_sha256,
            "placement": self.placement.to_dict(),
        }
