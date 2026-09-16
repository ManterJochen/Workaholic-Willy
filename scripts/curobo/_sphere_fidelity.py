"""How often the planner's spheres and the exact meshes disagree about a pose, counted the way the guard judges.

The guard refuses a pose whose exact meshes come closer than ``safety.self_collision.min_distance_mm``
(``src/robot/safety/_fcl_self_collision.py``, ``d < min_distance_mm``), so a pose exactly at the margin passes. A false
clear is a pose the spheres pass and the guard refuses: a path offered and refused late. A false collide is a pose the
spheres refuse and the guard passes: a path never offered. A refit of the arm spheres is held to the condition that the
first never gets worse.

Loaded by path from both halves of the measurement, the exact one in the project venv and the sphere one in the cuRobo
environment, so both count with one implementation. Standard library only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Agreement"]


@dataclass(frozen=True)
class Agreement:
    """The four ways the spheres and the exact meshes can judge one pose, summed over a pose set."""

    poses: int
    false_clear: int
    false_collide: int
    both_collide: int
    both_clear: int
    margin_mm: float

    @classmethod
    def from_verdicts(
        cls, *, sphere_collides: Sequence[bool], exact_distance_mm: Sequence[float], margin_mm: float,
    ) -> Agreement:
        """Count ``sphere_collides`` against the exact distance of the same poses, in the same order."""
        if len(sphere_collides) != len(exact_distance_mm):
            raise ValueError(
                f"{len(sphere_collides)} sphere verdicts and {len(exact_distance_mm)} exact distances: the two halves "
                "judged different pose sets, so nothing can be counted")
        false_clear = false_collide = both_collide = both_clear = 0
        for spheres, distance in zip(sphere_collides, exact_distance_mm, strict=True):
            exact = float(distance) < float(margin_mm)
            if spheres and exact:
                both_collide += 1
            elif spheres:
                false_collide += 1
            elif exact:
                false_clear += 1
            else:
                both_clear += 1
        return cls(len(sphere_collides), false_clear, false_collide, both_collide, both_clear, float(margin_mm))

    @property
    def false_clear_rate(self) -> float:
        return self.false_clear / self.poses if self.poses else 0.0

    @property
    def false_collide_rate(self) -> float:
        return self.false_collide / self.poses if self.poses else 0.0

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        return (f"{self.poses} poses at a {self.margin_mm:g} mm margin: false clear {self.false_clear} of {self.poses}, "
                f"false collide {self.false_collide} of {self.poses}, both collide {self.both_collide}, "
                f"both clear {self.both_clear}")

    def to_dict(self) -> dict[str, Any]:
        """The wire view."""
        return {"poses": self.poses, "false_clear": self.false_clear, "false_collide": self.false_collide,
                "both_collide": self.both_collide, "both_clear": self.both_clear, "margin_mm": self.margin_mm}
