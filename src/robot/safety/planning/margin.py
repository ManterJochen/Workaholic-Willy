"""The clearance a cell's planner keeps, read from that cell and never derived from the guard's.

Two margins live next to one another and are not the same number. The guard's ``min_distance_mm`` is what the exact
mesh check demands of a configuration before the arm moves. The planner's margin is padding inside the descriptor the
sidecar plans with, so cuRobo stops proposing configurations the guard was always going to refuse.

Neither can be derived from the other. A UR5e plans fine at 10 mm; a UR3e finds no plan at all at 10 mm, because its
thinner links make the inflated spheres read as permanent self collision, and a UR3e whose margin is set from
``min_distance_mm`` picks 0 of 10 where it picks 10 of 10 at 4 mm. How much margin a planner can absorb is a property
of how tightly its spheres fit that arm with that hand, which is what the matrix evidence measures.

So there is no default. A UR cell that plans with cuRobo and declares nothing gets a refusal naming the key, not a
planner with no margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["MarginReading", "MarginStatus", "declared_planner_margin", "planner_margin_refusal"]


def declared_planner_margin(self_collision: Any) -> Maybe[float]:
    """What this cell declared, or :data:`UNSET` where it declared nothing. ``0.0`` is a declaration."""
    value = getattr(self_collision, "planner_margin_mm", None)
    return float(value) if isinstance(value, (int, float)) else UNSET


def planner_margin_refusal(robot_cfg: "RobotConfig") -> "tuple[str, str] | None":
    """What is wrong with a UR cuRobo cell that declares no planner margin, and the fix, or ``None``.

    Only UR cells that plan with cuRobo: an ``ik`` cell starts no sidecar, and the sim declares its own margin in its
    own layer. The sentence names the key, this guard's own margin and why the two are separate numbers.
    """
    if str(getattr(robot_cfg, "vendor", "")) != "ur":
        return None
    if str(getattr(getattr(robot_cfg, "ur", None), "motion_planner", "")) != "curobo":
        return None
    self_collision = getattr(getattr(robot_cfg, "safety", None), "self_collision", None)
    if chosen(declared_planner_margin(self_collision)):
        return None
    guard_mm = float(getattr(self_collision, "min_distance_mm", 10.0))
    what = (
        "robot.safety.self_collision.planner_margin_mm is undeclared, and this cell plans with cuRobo: the planner "
        f"would then keep no clearance while the guard demands {guard_mm:g} mm, so it proposes paths the guard "
        "refuses just before the arm moves."
    )
    fix = (
        "Declare it in the cell profile: robot.safety.self_collision.planner_margin_mm. It is not this guard's "
        f"{guard_mm:g} mm and is not derived from it: a UR5e plans fine at 10, a UR3e finds no plan at all at 10 and "
        "is fine at 4. Measure the pair with scripts/curobo/matrix_gate.py and declare what it measured."
    )
    return what, fix


class MarginStatus(StrEnum):
    """How a cell's two margins stand to one another."""

    UNSET = "unset"
    OK = "ok"
    BELOW_GUARD = "below_guard"
    ABOVE_GUARD = "above_guard"


@dataclass(frozen=True, slots=True)
class MarginReading:
    """The planner margin a cell declared beside the clearance its guard demands.

    ``BELOW_GUARD`` is not a defect by itself: the UR3e ships 4 mm against a 10 mm guard on purpose, because it plans
    nothing at 10. It does mean the planner can hand back a path the guard refuses, which is a late refusal rather than
    a collision, and it is the reason the checklist says both numbers out loud.
    """

    planner_mm: Maybe[float]
    guard_mm: float

    @classmethod
    def of(cls, *, planner_mm: "float | None", guard_mm: float) -> MarginReading:
        """A reading from plain numbers, where ``None`` means the cell declared nothing."""
        return cls(planner_mm=float(planner_mm) if planner_mm is not None else UNSET, guard_mm=float(guard_mm))

    @property
    def status(self) -> MarginStatus:
        if not chosen(self.planner_mm):
            return MarginStatus.UNSET
        if self.planner_mm == self.guard_mm:
            return MarginStatus.OK
        return MarginStatus.BELOW_GUARD if self.planner_mm < self.guard_mm else MarginStatus.ABOVE_GUARD

    def render(self) -> str:
        if not chosen(self.planner_mm):
            return f"planner margin undeclared, guard margin {self.guard_mm:g} mm"
        if self.status is MarginStatus.OK:
            return f"planner and guard both keep {self.guard_mm:g} mm"
        if self.status is MarginStatus.BELOW_GUARD:
            consequence = "the planner can hand back a path the guard then refuses"
            side = "below"
        else:
            consequence = "the planner refuses more than the guard would"
            side = "above"
        return (f"planner margin {self.planner_mm:g} mm, {side} the guard's {self.guard_mm:g} mm: {consequence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_mm": self.planner_mm if chosen(self.planner_mm) else None,
            "guard_mm": self.guard_mm,
            "status": str(self.status),
        }
