"""Another arm of the cell as the boxes this arm keeps its distance from (owner, 2026-10-15).

Each arm's guard describes its own body where it stands (``SelfCollisionGuard.body_hulls``): every part's convex hull,
its base among them, in that arm's base. :func:`partner_bodies` stands those hulls in this arm's base, by where the
two arms stand in the cell (``arm_registration.base_in_cell``), as boxes that carry their hulls
(``AxisAlignedBox.hull_mm``), named ``<arm>:<part>``. ``SelfCollisionGuard.set_partner_bodies`` takes them: the exact
mesh guard judges every sample against the hulls, at the guard's own ``min_distance_mm``, and the capsule proxy
against their enclosures.

A body handed so is the other arm where it stands now. Only a cell that keeps that arm standing while this one moves
may hand it so; an arm that moves meanwhile is no obstacle anybody judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ._capsule import AxisAlignedBox

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .planning.perceived import SelfEnvelope

__all__ = ["PartnerWorld", "hull_gap_mm", "own_from_other", "partner_bodies", "partner_cuboids", "partner_world"]


def own_from_other(own_in_cell: np.ndarray, other_in_cell: np.ndarray) -> np.ndarray:
    """The other arm's base in this arm's base, from where both stand in the cell.

    Args:
        own_in_cell (np.ndarray): 4x4, millimetres: this arm's base in the cell's frame.
        other_in_cell (np.ndarray): 4x4, millimetres: the other arm's base in the cell's frame.

    Returns:
        np.ndarray: 4x4, millimetres: a point in the other arm's base, in this arm's.
    """
    return np.linalg.inv(np.asarray(own_in_cell, dtype=np.float64)) @ np.asarray(other_in_cell, dtype=np.float64)


def partner_bodies(arm: str, hulls: Mapping[str, np.ndarray], own_from_other_mm: np.ndarray
                   ) -> tuple[AxisAlignedBox, ...]:
    """The other arm's parts as boxes carrying their hulls, in this arm's base.

    Args:
        arm (str): The other arm's name, the first half of every box's name.
        hulls (Mapping[str, np.ndarray]): Its parts' hull vertices in its own base (``SelfCollisionGuard.body_hulls``).
        own_from_other_mm (np.ndarray): 4x4: a point in its base, in this arm's (:func:`own_from_other`).

    Returns:
        tuple[AxisAlignedBox, ...]: One box per part, sorted by name: the hull in this arm's base, and the axis-aligned
            box that encloses it.
    """
    turn = np.asarray(own_from_other_mm, dtype=np.float64)[:3, :3]
    shift = np.asarray(own_from_other_mm, dtype=np.float64)[:3, 3]
    bodies = []
    for part, vertices in sorted(hulls.items()):
        placed = np.asarray(vertices, dtype=np.float64) @ turn.T + shift
        low, high = placed.min(axis=0), placed.max(axis=0)
        bodies.append(AxisAlignedBox(center_mm=(low + high) / 2.0, half_extents_mm=(high - low) / 2.0,
                                     name=f"{arm}:{part}", hull_mm=placed))
    return tuple(bodies)


def partner_cuboids(bodies: Sequence[AxisAlignedBox]) -> tuple[dict[str, Any], ...]:
    """Each part as the planner takes it: the box along the hull's own axes that holds every vertex, in the planner's
    wire format (``world.planner_cuboid``), named ``arm_<arm>_<part>`` so it is a declared box the planner keeps, never a
    seen one the exact guard decides.

    Args:
        bodies (Sequence[AxisAlignedBox]): What :func:`partner_bodies` gave, in this arm's base.

    Returns:
        tuple[dict[str, Any], ...]: One box per part, metres, WXYZ.
    """
    from .planning.world import planner_cuboid  # noqa: PLC0415

    boxes = []
    for body in bodies:
        points = np.asarray(body.hull_mm if body.hull_mm is not None else _corners(body), dtype=np.float64)
        centre = points.mean(axis=0)
        _values, axes = np.linalg.eigh(np.cov((points - centre).T))
        if np.linalg.det(axes) < 0.0:
            axes[:, 0] = -axes[:, 0]
        local = (points - centre) @ axes
        low, high = local.min(axis=0), local.max(axis=0)
        name = "arm_" + body.name.replace(":", "_")
        boxes.append(planner_cuboid(name, centre + axes @ ((low + high) / 2.0), high - low,
                                    rotation=[float(v) for v in axes.reshape(-1)]))
    return tuple(boxes)


def _corners(body: AxisAlignedBox) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return np.asarray(body.center_mm, dtype=np.float64) + signs * np.asarray(body.half_extents_mm, dtype=np.float64)


@dataclass(frozen=True)
class PartnerWorld:
    """The other arms of the cell where they stand, in one arm's base: what its guard, its planner and its cameras'
    self filter hold of them (owner, 2026-10-15). Empty holds none, which is a cell of one arm."""

    #: Each part as a box carrying its hull, for the guard (``SelfCollisionGuard.set_partner_bodies``).
    bodies: tuple[AxisAlignedBox, ...] = ()
    #: Each part as the planner's box (:func:`partner_cuboids`).
    cuboids: tuple[dict[str, Any], ...] = ()
    #: Each other arm's body for the self filter, already in this arm's base (``SelfEnvelope.moved``).
    envelopes: "tuple[SelfEnvelope, ...]" = ()

    @property
    def arms(self) -> tuple[str, ...]:
        """The other arms' names, sorted."""
        return tuple(sorted({body.name.split(":", 1)[0] for body in self.bodies}))


def partner_world(
    others: "Sequence[tuple[str, Mapping[str, np.ndarray], SelfEnvelope | None, np.ndarray]]",
) -> PartnerWorld:
    """One arm's view of every other arm of the cell.

    Args:
        others (Sequence[tuple[str, Mapping[str, np.ndarray], SelfEnvelope | None, np.ndarray]]): For each other arm,
            its name, its parts' hulls in its own base (``SelfCollisionGuard.body_hulls``), its body for the self
            filter in its own base (``self_envelope.self_envelope``; ``None`` where none is built) and its base in this
            arm's (:func:`own_from_other`).

    Returns:
        PartnerWorld: The bodies, the planner's boxes and the envelopes, in this arm's base.
    """
    bodies: list[AxisAlignedBox] = []
    envelopes: list[SelfEnvelope] = []
    for name, hulls, envelope, matrix in others:
        bodies.extend(partner_bodies(name, hulls, matrix))
        if envelope is not None:
            envelopes.append(envelope.moved(matrix))
    return PartnerWorld(bodies=tuple(bodies), cuboids=partner_cuboids(bodies), envelopes=tuple(envelopes))


def hull_gap_mm(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """The distance between the convex hulls of two point sets, millimetres, 0 where they overlap.

    Measured by the exact guard's engine where it builds hulls (Coal); where it builds none, the distance between the
    two sets' axis-aligned boxes, which is never more than the hulls' and so refuses where the hulls might not: the safe
    side.

    Args:
        points_a (np.ndarray): ``(N, 3)``, millimetres.
        points_b (np.ndarray): ``(M, 3)``, millimetres, in the same frame.

    Returns:
        float: The gap, millimetres.
    """
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 3)
    engine = _hull_engine()
    if engine is not None:
        try:
            hull_a, hull_b = engine.convex_object(a), engine.convex_object(b)
            if hull_a is not None and hull_b is not None:
                return max(0.0, float(engine.distance(hull_a, hull_b)))
        except Exception:  # noqa: BLE001 - a flat set has no hull; its box answers, the safe side
            pass
    separation = np.maximum(0.0, np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0)))
    return float(np.linalg.norm(separation))


@lru_cache(maxsize=1)
def _hull_engine() -> Any:
    """The exact guard's engine adapter where it can build hulls, else ``None``."""
    from ._fcl_self_collision import _EngineAdapter  # noqa: PLC0415
    from .planning.environment import import_collision_engine  # noqa: PLC0415

    module, kind = import_collision_engine()
    return None if module is None or kind != "coal" else _EngineAdapter(module, kind)
