"""Which arm of a cell may move now, and through which space: a reservation for every send (owner, 2026-10-15).

A cell of one arm has none of this. In a cell of several arms :meth:`CellMotion.install` puts every arm's sends behind
the cell (``SharesItsCell.join_cell``, ``core.cell_gate``), so nothing reaches a controller without its grant:

* A joint path (a planned route, a joint move) reserves the space it sweeps from where its arm stands: every leg
  sampled, every part the hull of its box at the samples of a stretch (``SelfCollisionGuard.swept_corners``), a part in
  the jaws among them. It is granted where that space keeps the guard's distance from every other arm, standing where
  it stands or sweeping what it reserved itself, so two arms move at once wherever their spaces do not meet.
* A line, which the controller draws, and a person guiding an arm hold the whole cell: a line from its judgement to its
  end, so no other arm moves between the two. Every other arm waits.
* A send not granted within the wait is refused with nothing sent (``cell_gate.CellOccupied``), naming the arm in the
  way.

Every arm's guard, planner and cameras hold the others as the cell knows them (``PlannedArm.hold_partners``), handed
again at every grant and release that changed them: a standing arm where it stands, a moving one as the whole space it
reserved. Each arm of such a cell judges every sample of its lines and paths (cuRobo) and its guard describes its
body (the exact mesh guard): an arm whose path nobody judges, or whose body nobody can describe, could not keep its
distance from another.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from src.robot.core.arm_capabilities import LineMotion, SharesItsCell, line_motion_of
from src.robot.core.cell_gate import CellOccupied
from src.robot.core.motion_result import MotionStatus

__all__ = ["CellMotion", "MovingArm", "WAIT_S"]

#: How long a send waits for the space it needs before it is refused, seconds.
WAIT_S = 30.0
#: The largest turn of any joint between two samples of a reserved leg, radians.
LEG_STEP_RAD = 0.05
#: How many samples one hull of a part's sweep holds: the sweep of a long route is a row of hulls, not one about it all.
STRETCH = 6


@dataclass(frozen=True)
class MovingArm:
    """One arm of the cell as the cell moves it.

    Attributes:
        name (str): The arm's name in the cell file.
        arm (Any): The planned arm.
        in_cell (np.ndarray): 4x4, millimetres: its base in the cell's frame (``arm_registration.base_in_cell``).
    """

    name: str
    arm: Any
    in_cell: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass(frozen=True)
class _Held:
    """What a span holds: the whole cell (``swept`` ``None``), or the space a route sweeps in the cell's frame, part by
    stretch; ``serial`` tells two spans apart."""

    swept: "dict[str, np.ndarray] | None"
    serial: int


class CellMotion:
    """The reservations of a cell of several arms.

    ```python
    cell = CellMotion([MovingArm("halter", halter), MovingArm("werker", werker, in_cell=base_in_cell(loaded, "werker"))])
    cell.install()          # before the first planned move: every send passes the cell from here on
    ```

    Args:
        arms (Sequence[MovingArm]): The cell's arms, connected; the first stands at the cell's frame.
        margin_mm (float | None): The distance a reserved space keeps from another arm; ``None`` the largest of the
            arms' guards' ``min_distance_mm`` (default: None).
        wait_s (float): How long a send waits for its space (default: :data:`WAIT_S`).
        clock (Callable[[], float]): Seconds, monotonic (default: time.monotonic).
    """

    def __init__(self, arms: Sequence[MovingArm], *, margin_mm: float | None = None, wait_s: float = WAIT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if len({member.name for member in arms}) != len(arms) or len(arms) < 2:
            raise ValueError("a cell of several arms names two arms or more, each once")
        self._arms = tuple(arms)
        self._guards = {member.name: _body_guard(member.arm) for member in arms}
        self._margin = float(margin_mm) if margin_mm is not None else max(
            float(getattr(guard, "min_distance_mm", 0.0)) for guard in self._guards.values())
        self._wait_s = float(wait_s)
        self._clock = clock
        self._cond = threading.Condition()
        self._held: dict[str, _Held] = {}
        self._serials = itertools.count()
        self._refreshing = threading.Lock()
        #: What each arm was last handed of the others, so an unchanged view is not handed again.
        self._handed: dict[str, tuple[Any, ...]] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """The arms' names, the frame's first."""
        return tuple(member.name for member in self._arms)

    def held(self) -> dict[str, str]:
        """What each arm holds now: ``"cell"`` or ``"path"``; an arm that stands holds nothing.

        Returns:
            dict[str, str]: By arm name, the arms that hold anything.
        """
        with self._cond:
            return {name: "cell" if held.swept is None else "path" for name, held in self._held.items()}

    def install(self) -> None:
        """Put every arm's sends behind the cell and hand every arm the others. Moves nothing.

        Raises:
            ConfigError: An arm cannot share a cell, does not judge every sample of its lines, or its guard cannot
                describe its body.
        """
        from src.config import ConfigError  # noqa: PLC0415

        parts: dict[str, int] = {}
        for member in self._arms:
            if not isinstance(member.arm, SharesItsCell):
                raise ConfigError(f"{member.name}: this arm cannot share a cell with other arms")
            lines = line_motion_of(member.arm)
            if lines is None or lines.motion is not LineMotion.CHECKED:
                raise ConfigError(f"{member.name}: an arm whose path nobody judges cannot keep its distance from "
                                  f"another arm of its cell ({lines.reason if lines else 'it says nothing of its lines'})")
            hulls = self._hulls(member)
            if hulls is None:
                raise ConfigError(f"{member.name}: its guard cannot describe its body (safety.self_collision.backend "
                                  "fcl, an arm model with a mesh bundle), so no other arm could keep its distance from it")
            # Its parts, its hand among them; a part in its jaws is modelled on no cell (2026-10-17), so the other arms
            # keep their distance from its hand while it carries one.
            parts[member.name] = len(hulls)
        try:
            for member in self._arms:
                member.arm.reserve_partner_slots(sum(count for name, count in parts.items() if name != member.name))
                member.arm.join_cell(_Gate(self, member.name))
            self.refresh()
        except BaseException:
            self.uninstall()
            raise

    def uninstall(self) -> None:
        """Take every arm out of the cell again (``SharesItsCell.leave_cell``): its sends pass no gate, and it holds no
        other arm. What the arms hold is forgotten."""
        for member in self._arms:
            member.arm.leave_cell()
        with self._cond:
            self._held.clear()
            self._cond.notify_all()
        self._handed.clear()

    # --- the reservations ------------------------------------------------------------------------------------------

    def reserve(self, name: str, route: "Sequence[Sequence[float]] | None") -> None:
        """Hold the space ``route`` sweeps, or the whole cell for ``None``, for ``name``, waiting while another arm
        stands or moves in it.

        Args:
            name (str): The arm.
            route (Sequence[Sequence[float]] | None): The joints its sends run through, where it stands first; ``None``
                for the whole cell.

        Raises:
            CellOccupied: The space did not clear within the wait, or the cell cannot describe it; nothing may be sent.
        """
        member = self._member(name)
        swept = None if route is None else self._swept(member, route)
        deadline = self._clock() + self._wait_s
        with self._cond:
            while True:
                blocker = self._blocker(member, route is None, swept)
                if blocker is None:
                    self._held[name] = _Held(swept=swept, serial=next(self._serials))
                    break
                status, why = blocker
                if self._clock() >= deadline:
                    raise CellOccupied(f"{name}: {why}, and that did not clear within {self._wait_s:g} s; nothing was "
                                       "sent", status=status)
                self._cond.wait(timeout=min(max(deadline - self._clock(), 0.0), 0.2))
        try:
            self.refresh()
        except BaseException:
            self._drop(name)
            raise

    def release(self, name: str) -> None:
        """End what ``name`` holds; the others then hold it where it stands.

        Args:
            name (str): The arm.
        """
        self._drop(name)
        self.refresh()

    def _drop(self, name: str) -> None:
        with self._cond:
            self._held.pop(name, None)
            self._cond.notify_all()

    def refresh(self) -> None:
        """Hand every arm the others as the cell knows them now, where that changed since it was last handed: a standing
        arm where it stands, a moving one as the space it reserved (each part one hull over its stretches)."""
        from src.robot.safety.partner_arm import own_from_other, partner_world  # noqa: PLC0415

        with self._refreshing:
            with self._cond:
                held = dict(self._held)
            seen: dict[str, tuple[Any, ...]] = {}
            joints: dict[str, list[float] | None] = {}
            for member in self._arms:
                reserved = held.get(member.name)
                if reserved is not None and reserved.swept is not None:
                    seen[member.name] = ("path", reserved.serial)
                else:
                    joints[member.name] = _joints(member.arm)
                    here = joints[member.name]
                    seen[member.name] = ("stands", None if here is None else tuple(round(v, 4) for v in here))
            bodies: dict[str, tuple[Any, Any]] = {}

            def body(other: MovingArm) -> tuple[Any, Any]:
                """Its parts as the others hold them, and its body for their cameras, read once a refresh."""
                if other.name not in bodies:
                    reserved = held.get(other.name)
                    parts = (_from_cell(_whole_parts(reserved.swept), other.in_cell)
                             if reserved is not None and reserved.swept is not None
                             else self._hulls(other, joints[other.name]) if joints[other.name] is not None else None)
                    bodies[other.name] = (parts, other.arm.own_envelope())
                return bodies[other.name]

            for member in self._arms:
                others = [other for other in self._arms if other is not member]
                key = tuple(seen[other.name] for other in others)
                if self._handed.get(member.name) == key:
                    continue
                member.arm.hold_partners(partner_world([
                    (other.name, body(other)[0], body(other)[1], own_from_other(member.in_cell, other.in_cell))
                    for other in others if body(other)[0] is not None]))
                self._handed[member.name] = key

    # --- what stands in the way -------------------------------------------------------------------------------------

    def _blocker(self, member: MovingArm, whole: bool, swept: "dict[str, np.ndarray] | None"
                 ) -> "tuple[MotionStatus, str] | None":
        """Why ``member`` may not hold what it asks now, or ``None``."""
        if not whole and swept is None:
            return MotionStatus.UNSUPPORTED, "the cell cannot describe the space its route sweeps"
        for other in self._arms:
            if other is member:
                continue
            held = self._held.get(other.name)
            if whole:
                if held is not None:
                    return MotionStatus.TIMEOUT, f"it needs the whole cell, and {other.name} moves"
                continue
            assert swept is not None  # noqa: S101 (a route's space, refused above where it is not described)
            if held is not None and held.swept is None:
                return MotionStatus.TIMEOUT, f"{other.name} holds the whole cell (a line, or a person guiding it)"
            if held is not None and held.swept is not None:
                gap = _gap(swept, held.swept)
                if gap < self._margin:
                    return MotionStatus.TIMEOUT, (f"the space it would sweep comes {gap:.1f} mm from the space "
                                                  f"{other.name} sweeps, where {self._margin:g} mm are kept")
                continue
            hulls = self._hulls(other)
            if hulls is None:
                return MotionStatus.CONNECTION_ERROR, f"{other.name} cannot say where it stands"
            gap = _gap(swept, {part: _to_cell(points, other.in_cell) for part, points in hulls.items()})
            if gap < self._margin:
                return MotionStatus.SELF_COLLISION_REJECTED, (f"{other.name} stands {gap:.1f} mm from the space it "
                                                              f"would sweep, where {self._margin:g} mm are kept")
        return None

    def _swept(self, member: MovingArm, route: "Sequence[Sequence[float]]") -> "dict[str, np.ndarray] | None":
        """Every part's sweep over ``route`` in the cell's frame, a hull per stretch of :data:`STRETCH` samples at most
        :data:`LEG_STEP_RAD` apart; ``None`` where the arm's guard cannot describe it."""
        configs = [np.asarray(route[0], dtype=np.float64)]
        for start, end in zip(route, route[1:]):
            a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
            steps = max(1, math.ceil(float(np.max(np.abs(b - a), initial=0.0)) / LEG_STEP_RAD))
            configs.extend(a + (b - a) * (k / steps) for k in range(1, steps + 1))
        guard = self._guards[member.name]
        swept: dict[str, np.ndarray] = {}
        # Stretches share their end samples, so every step between two samples lies inside one stretch's hulls.
        for index, first in enumerate(range(0, max(1, len(configs) - 1), STRETCH - 1)):
            corners = None if guard is None else guard.swept_corners(member.arm, configs[first:first + STRETCH])
            if corners is None:
                return None
            for part, points in corners.items():
                swept[part if part == "base" else f"{part}#{index}"] = _to_cell(points, member.in_cell)
        return swept

    def _hulls(self, member: MovingArm, joints: "Sequence[float] | None" = None) -> "dict[str, np.ndarray] | None":
        """The arm's parts at ``joints``, or where it stands now, in its own base."""
        guard = self._guards[member.name]
        if joints is None:
            joints = _joints(member.arm)
        return None if guard is None or joints is None else guard.body_hulls(member.arm, joints)

    def _member(self, name: str) -> MovingArm:
        for member in self._arms:
            if member.name == name:
                return member
        raise KeyError(f"no arm {name!r} in this cell: its arms are {', '.join(self.names)}")


class _Gate:
    """One arm's door to the cell (``cell_gate.CellGate``)."""

    def __init__(self, cell: CellMotion, name: str) -> None:
        self._cell, self._name = cell, name

    def reserve(self, route: "Sequence[Sequence[float]] | None") -> None:
        self._cell.reserve(self._name, route)

    def release(self) -> None:
        self._cell.release(self._name)


def _body_guard(arm: Any) -> Any:
    preflight = arm.safety_preflight
    return next((guard for guard in (preflight.guards if preflight is not None else ())
                 if guard.name == "self_collision"), None)


def _joints(arm: Any) -> "list[float] | None":
    try:
        return [float(v) for v in arm.get_joint_positions()]
    except Exception:  # noqa: BLE001 - an arm that cannot say where it stands is said so where it matters
        return None


def _to_cell(points: np.ndarray, in_cell: np.ndarray) -> np.ndarray:
    matrix = np.asarray(in_cell, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _from_cell(parts: "dict[str, np.ndarray]", in_cell: np.ndarray) -> "dict[str, np.ndarray]":
    inverse = np.linalg.inv(np.asarray(in_cell, dtype=np.float64))
    return {part: np.asarray(points) @ inverse[:3, :3].T + inverse[:3, 3] for part, points in parts.items()}


def _whole_parts(swept: "dict[str, np.ndarray]") -> "dict[str, np.ndarray]":
    """Each part's stretches as one point set: the guard and the planner hold one box per part of a moving arm."""
    merged: dict[str, list[np.ndarray]] = {}
    for key, points in swept.items():
        merged.setdefault(key.split("#", 1)[0], []).append(points)
    return {part: np.concatenate(stretches) for part, stretches in merged.items()}


def _gap(a: "dict[str, np.ndarray]", b: "dict[str, np.ndarray]") -> float:
    """The smallest distance between any hull of ``a`` and any hull of ``b``, both in the cell's frame."""
    from src.robot.safety.partner_arm import hull_gap_mm  # noqa: PLC0415

    boxes_b = [(points, points.min(axis=0), points.max(axis=0)) for points in b.values()]
    best = math.inf
    for points_a in a.values():
        low_a, high_a = points_a.min(axis=0), points_a.max(axis=0)
        for points_b, low_b, high_b in boxes_b:
            # Two hulls come no nearer than the boxes about them: a pair whose boxes keep ``best`` is no nearer.
            if float(np.linalg.norm(np.maximum(0.0, np.maximum(low_a - high_b, low_b - high_a)))) >= best:
                continue
            best = min(best, hull_gap_mm(points_a, points_b))
    return best
