"""Space a planner world leaves empty for one motion, and the offer that carries it there.

A perceived world registers everything the cameras see, and during a pick that includes the one object the arm is
deliberately driving into. A goal inside an obstacle has no plan by construction, and a retreat that starts with the
part between the fingers starts inside one. So a pick hands the world its target, and the world leaves that space out:
as pixel masks for the camera that took them, and as a box in BASE that no camera, fixed or on the wrist, keeps an
obstacle in.

The box is fitted by the world, with the plane, limits, clustering, margin and floor it fits every other box with
(``safety.planning.perceived.target_keep_out_box``), so a target leaves exactly the space it would have filled. The
offer carries the target's points in BASE and never a box of its own making.

Value types only, with no import above ``robot.core``: the pick loop, the locator and the safety layer all read them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen

__all__ = ["GoalKeepOut", "KeepOutBox", "KeepOutScope", "KeepOutSummary", "SegmentationOffer", "keeping_out"]


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class KeepOutBox:
    """A box in BASE whose inside is not an obstacle for the motion it was handed to.

    ``box_to_base_mm`` places the box's own frame in BASE as a rigid 4x4 in millimetres, and ``half_extents_mm`` is
    its half size along that frame's X, Y and Z. A point on a face is inside.
    """

    name: str
    box_to_base_mm: tuple[tuple[float, float, float, float], ...]
    half_extents_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a keep-out box needs a name a refusal can print")
        matrix = np.asarray(self.box_to_base_mm, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"keep-out box {self.name!r}: box_to_base_mm must be a finite 4x4, got {matrix.shape}")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.allclose(matrix[3], (0, 0, 0, 1)):
            raise ValueError(f"keep-out box {self.name!r}: box_to_base_mm must be a rigid transform")
        half = tuple(float(v) for v in self.half_extents_mm)
        if len(half) != 3 or not all(math.isfinite(v) and v >= 0.0 for v in half):
            raise ValueError(
                f"keep-out box {self.name!r}: half_extents_mm must be three finite non-negative numbers, got "
                f"{self.half_extents_mm!r}"
            )
        object.__setattr__(self, "box_to_base_mm", tuple(tuple(float(v) for v in row) for row in matrix))
        object.__setattr__(self, "half_extents_mm", half)

    @classmethod
    def from_matrix(cls, name: str, box_to_base_mm: Any, half_extents_mm: Sequence[float]) -> "KeepOutBox":
        """A box from a 4x4 array and its half extents."""
        matrix = np.asarray(box_to_base_mm, dtype=np.float64)
        return cls(name=name, box_to_base_mm=tuple(tuple(row) for row in matrix.tolist()),
                   half_extents_mm=tuple(float(v) for v in half_extents_mm))  # type: ignore[arg-type]

    @classmethod
    def from_jaw(
        cls,
        *,
        aperture_mm: float,
        finger_width_mm: float,
        pad_ahead_mm: float,
        pad_behind_mm: float,
        tcp_to_base_mm: Any,
        name: str = "goal",
    ) -> "KeepOutBox":
        """The space between a parallel jaw's pads at a goal, laid out in the TCP's own axes, with no padding.

        The TCP frame is X closing, Y binormal, Z approach. The region runs across the open aperture on X, across the
        finger's width on Y, and along the pad on Z, from ``pad_behind_mm`` behind the grasp centre to ``pad_ahead_mm``
        past it. Nothing grows it: a point outside it stays an obstacle, so a finger closing on a wall is still checked.
        """
        tcp = np.asarray(tcp_to_base_mm, dtype=np.float64)
        if tcp.shape != (4, 4) or not np.all(np.isfinite(tcp)):
            raise ValueError(f"keep-out box {name!r}: the goal's TCP must be a finite 4x4, got {tcp.shape}")
        along = (float(pad_ahead_mm) - float(pad_behind_mm)) / 2.0
        box_to_base = tcp.copy()
        box_to_base[:3, 3] = tcp[:3, 3] + tcp[:3, :3] @ np.array([0.0, 0.0, along])
        half = (float(aperture_mm) / 2.0, float(finger_width_mm) / 2.0,
                (float(pad_ahead_mm) + float(pad_behind_mm)) / 2.0)
        return cls.from_matrix(name, box_to_base, half)

    def matrix(self) -> np.ndarray:
        """``box_to_base_mm`` as a 4x4 array."""
        return np.asarray(self.box_to_base_mm, dtype=np.float64)

    def contains(self, points_base_mm: Any) -> np.ndarray:
        """Which of the ``(N, 3)`` BASE points lie inside or on the box."""
        points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
        matrix = self.matrix()
        local = (points - matrix[:3, 3]) @ matrix[:3, :3]
        return np.all(np.abs(local) <= np.asarray(self.half_extents_mm) + 1e-9, axis=1)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        centre = self.matrix()[:3, 3]
        size = [2.0 * v for v in self.half_extents_mm]
        return (
            f"keep-out {self.name}: centre ({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}) mm, "
            f"size ({size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f}) mm"
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "name": self.name,
            "box_to_base_mm": [list(row) for row in self.box_to_base_mm],
            "half_extents_mm": list(self.half_extents_mm),
        }


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class SegmentationOffer:
    """What one perception frame hands a planner world: its masks for one camera, and its target's points in BASE.

    ``camera`` names the camera the masks belong to, because a pixel means nothing in another camera's image. An offer
    with masks or labels has to name it; an offer with only target points may leave it unset, because a box in BASE
    belongs to no camera. ``captured_at_s`` is the shutter time of the frame, on the ``time.time()`` clock, and is
    required: masks and a box age like the image they came from.
    """

    captured_at_s: float
    camera: Maybe[str] = UNSET
    labelled_masks: tuple[tuple[str, np.ndarray], ...] = ()
    exclude_masks: tuple[np.ndarray, ...] = ()
    target_points_base_mm: np.ndarray | None = None
    target_label: str = ""

    def __post_init__(self) -> None:
        if chosen(self.camera) and (not isinstance(self.camera, str) or not self.camera.strip()):
            raise ValueError(f"a segmentation offer names its camera with a non-blank name, not {self.camera!r}")
        labelled = tuple((str(label), np.asarray(mask).astype(bool)) for label, mask in self.labelled_masks)
        exclude = tuple(np.asarray(mask).astype(bool) for mask in self.exclude_masks)
        if (labelled or exclude) and not chosen(self.camera):
            raise ValueError(
                "a segmentation offer with masks names the camera that took them: a pixel means nothing in another "
                "camera's image"
            )
        if not _finite(self.captured_at_s) or float(self.captured_at_s) < 0.0:
            raise ValueError(
                f"a segmentation offer carries the finite shutter time of its frame, not {self.captured_at_s!r}: "
                "its masks and its box age like the image they came from"
            )
        points = self.target_points_base_mm
        if points is not None:
            array = np.asarray(points, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"target_points_base_mm are finite (N, 3) BASE millimetres, got shape {array.shape}"
                )
            object.__setattr__(self, "target_points_base_mm", array)
        object.__setattr__(self, "labelled_masks", labelled)
        object.__setattr__(self, "exclude_masks", exclude)
        object.__setattr__(self, "target_label", str(self.target_label))
        object.__setattr__(self, "captured_at_s", float(self.captured_at_s))

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        camera = self.camera if chosen(self.camera) else "no camera (box only)"
        points = 0 if self.target_points_base_mm is None else int(self.target_points_base_mm.shape[0])
        target = f"target {self.target_label or 'unnamed'} with {points} point(s)" if points else "no target points"
        return (
            f"segmentation offer from {camera} at {self.captured_at_s:.3f} s: {len(self.labelled_masks)} labelled "
            f"mask(s), {len(self.exclude_masks)} exclude mask(s), {target}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe: names and counts, never the masks or the points."""
        return {
            "camera": self.camera if chosen(self.camera) else None,
            "labels": [label for label, _ in self.labelled_masks],
            "exclude_masks": len(self.exclude_masks),
            "target_label": self.target_label,
            "target_points": 0 if self.target_points_base_mm is None else int(self.target_points_base_mm.shape[0]),
            "captured_at_s": self.captured_at_s,
        }


@dataclass(frozen=True, slots=True, eq=False)
class KeepOutScope:
    """What a :func:`keeping_out` block holds: whether an arm's live world took the offer, and the offer."""

    world_wired: bool
    offer: SegmentationOffer | None = None


def keeping_out(arm: object, offer: SegmentationOffer) -> AbstractContextManager[KeepOutScope]:
    """Hold ``offer`` in ``arm``'s live planner world for every motion inside the block, and forget it after.

    The offer is held whatever its age, because a pick's target stays out of the world for the approach, the close
    and the retreat alike, and forgotten when the block ends, also when its body raises, a ``CameraWorldUnavailable``
    included, so the next motion plans against the cell with the target back in it. An arm with no live world opens
    an empty scope and costs one attribute lookup.
    """
    return _keeping_out(arm, offer)


@contextmanager
def _keeping_out(arm: object, offer: SegmentationOffer) -> Iterator[KeepOutScope]:
    world = getattr(arm, "live_planner_world", None)
    if world is None:
        yield KeepOutScope(world_wired=False, offer=offer)
        return
    world.offer_segmentation(
        camera=offer.camera, labelled_masks=offer.labelled_masks, exclude_masks=offer.exclude_masks,
        timestamp=offer.captured_at_s, target_points_base_mm=offer.target_points_base_mm,
        target_label=offer.target_label, hold=True,
    )
    try:
        yield KeepOutScope(world_wired=True, offer=offer)
    finally:
        world.forget_segmentation()


@dataclass(frozen=True, slots=True)
class GoalKeepOut:
    """The region a motion's goal leaves out of the world it plans against, or the reason there is none."""

    #: The space between the jaws at the goal, or ``None``.
    region: KeepOutBox | None
    #: Why there is no region; empty when there is one.
    reason: str = ""


@dataclass(frozen=True, slots=True)
class KeepOutSummary:
    """What a refresh left out of its world, in numbers a stamp and a report can carry. Hashable.

    ``goal_points`` is how many points the goal region took out, ``None`` when no region was in force, and
    ``goal_reason`` then says why. ``held`` names every held or fresh offer whose box was in force: the camera its masks
    belong to or ``box only``, the shutter time it ages by, and the points its box took out.
    """

    goal_points: int | None
    goal_reason: str = ""
    held: tuple[tuple[str, float, int], ...] = ()

    @property
    def in_force(self) -> bool:
        """Whether a goal region or a held box took anything out of the world's reach."""
        return self.goal_points is not None or bool(self.held)

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        goal = (f"goal region left out {self.goal_points} point(s)" if self.goal_points is not None
                else f"no goal region ({self.goal_reason or 'none asked'})")
        if not self.held:
            return goal
        held = ", ".join(f"{camera} at {stamp:.3f} s left out {points} point(s)" for camera, stamp, points in self.held)
        return f"{goal}; held: {held}"

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "goal_points": self.goal_points,
            "goal_reason": self.goal_reason,
            "held": [{"camera": camera, "captured_at_s": stamp, "points": points}
                     for camera, stamp, points in self.held],
        }
