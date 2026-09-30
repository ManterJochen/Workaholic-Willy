"""What lets a pick push its failed part, what a campaign keeps across its picks, and what a pick's push came to.

The push (``nudge_target``) runs inside the pick attempt, where the part, its neighbours, the table the looks saw and
the part's keep-out are known (owner, 2026-09-29). The pick loop never decides on its own that it may push. The
service hands it a :class:`PushGate` for one pick, built from the recovery policy
(:func:`~src.robot.grasping.recovery.policy.push_permitted`: dense_clutter's profile, ``recovery.allowed_actions``,
a declared fixture, a budget that is not zero), from the cell (:class:`PushCell`: the open hand from the gripper
registry, the workspace, the clearance the arm's line judge keeps, a declared container) and from the campaign
(:class:`PushCampaign`: the push budgets and the distance its pushes run). Without a gate nothing pushes.

A campaign (a ``PickRun``, a console run) keeps two things across its picks: the push budgets (1 per part, 2 per
pick, 5 per campaign) and the parts ``next_target`` skips (:class:`~src.robot.grasping.recovery.exclusion_zones.
ExclusionZones`). The service holds one and calls :meth:`PushCampaign.start_pick` as every pick starts.

:class:`PickPush` is what one push the pick considered came to: refused before anything moved (the pick falls
through to its next action), pushed (the arm went back to the look and looked again), or stopped where the arm is
(a person decides; the campaign ends). A push refused because the controller cannot move ends the pick instead, as a
stopped controller ends it anywhere, and so does one refused because nobody can say where a toggle hand's jaws stand,
as a gripper fault. :class:`FailedPart` is the part a pick failed on, which ``next_target`` skips.
:func:`support_points_of_views` is the table the looks saw, the points a push's landing is bounded by.

Nothing here moves the arm, asks a person or touches the jaws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from src.robot.grasping.recovery.exclusion_zones import ExclusionZones
from src.robot.grasping.recovery.push_budgets import PushBudgets
from src.robot.grasping.recovery.push_planner import (
    AxisBox,
    PushHand,
    PushRefusal,
    table_points_from_cloud,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.recovery.push_motion import PushOutcome

__all__ = [
    "REFUSED_PUSH_CELL_UNKNOWN",
    "REFUSED_STOP_REQUESTED",
    "TRIGGER_ALL_COLLIDED",
    "TRIGGER_APPROACH_BLOCKED",
    "FailedPart",
    "PickPush",
    "PushCampaign",
    "PushCell",
    "PushGate",
    "support_points_of_views",
]

#: A push is considered because every grasp candidate of the part collided (``ALL_COLLIDED``, at least one did).
TRIGGER_ALL_COLLIDED = "all_collided"
#: A push is considered because every ranked candidate's approach sweep was blocked (approach validation on).
TRIGGER_APPROACH_BLOCKED = "approach_path_blocked"
#: The cell's push inputs could not be read from its config.
REFUSED_PUSH_CELL_UNKNOWN = "push_cell_unknown"
#: A stop was asked for (the console's Stop, ``should_cancel``) after the pick's looks and before the push began:
#: nothing is commanded, and the next attempt's start ends the pick, as a stop between two attempts does.
REFUSED_STOP_REQUESTED = "refused_stop_requested"

#: About this many rows of table points are read from each view: a view's depth image is strided to give them, and one
#: with fewer rows is read whole. The planner keeps the table in 5 mm cells, and a D415 at the owner's working range
#: puts neighbouring pixels about 0.5 mm apart, so the stride of 4 a 720-row image gets still lands a point in every
#: cell.
_SUPPORT_ROWS = 180

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PushCell:
    """What the cell is, as a push needs it: the open hand, the workspace, the hand's clearance and a container.

    ``hand`` is the open hand the planner sweeps (its fingers from the grasping geometry, its housing from the gripper
    registry, :mod:`~src.robot.grasping.recovery.push_hand`). ``workspace`` is ``robot.workspace_limits``: every TCP
    point of a push stays 20 mm inside it, ``z_min`` included. ``hand_clearance_mm`` is how far the open hand keeps
    from every neighbour point: the camera world's ``margin_mm`` plus the arm's ``line_clearance_mm`` (25 mm shipped),
    so the planner plans no push the arm's line judge would predictably refuse. ``container_interior`` is a declared
    container's interior box (``grasping.support.container``), the push box in place of the table the looks saw.
    """

    hand: PushHand
    workspace: AxisBox
    hand_clearance_mm: float
    container_interior: Optional[AxisBox] = None

    def __post_init__(self) -> None:
        clearance = float(self.hand_clearance_mm)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(f"PushCell.hand_clearance_mm must be a non-negative number of mm, got {clearance!r}")
        object.__setattr__(self, "hand_clearance_mm", clearance)

    @classmethod
    def from_robot_config(cls, robot: Any) -> "Union[PushCell, PushRefusal]":
        """The cell's push inputs, read from a loaded robot config; a refusal with a sentence where they cannot be.

        The hand is :func:`~src.robot.grasping.recovery.push_hand.push_hand_from_robot_config` (a cell whose hand the
        registry does not hold, or that carries a suction cup, is refused there). The clearance is
        ``robot.safety.planning_world.perceived.margin_mm`` plus ``robot.safety.planned_motion.line_clearance_mm``.
        """
        from src.robot.grasping.recovery.push_hand import push_hand_from_robot_config  # noqa: PLC0415

        hand = push_hand_from_robot_config(robot)
        if isinstance(hand, PushRefusal):
            return hand
        try:
            limits = robot.workspace_limits
            workspace = AxisBox((float(limits.x_min), float(limits.y_min), float(limits.z_min)),
                                (float(limits.x_max), float(limits.y_max), float(limits.z_max)))
            safety = robot.safety
            clearance = (float(safety.planning_world.perceived.margin_mm)
                         + float(safety.planned_motion.line_clearance_mm))
            container = getattr(getattr(getattr(robot, "grasping", None), "support", None), "container", None)
            interior = None
            if container is not None and bool(getattr(container, "interior_declared", False)):
                interior = AxisBox(tuple(float(v) for v in container.interior_min_mm),  # type: ignore[arg-type]
                                   tuple(float(v) for v in container.interior_max_mm))  # type: ignore[arg-type]
            return cls(hand=hand, workspace=workspace, hand_clearance_mm=clearance, container_interior=interior)
        except (AttributeError, TypeError, ValueError) as exc:
            return PushRefusal(
                code=REFUSED_PUSH_CELL_UNKNOWN,
                sentence=(f"The robot config does not say where the workspace is or how far a line keeps from the "
                          f"camera world ({type(exc).__name__}: {exc}), so no push is planned."),
            )


class PushCampaign:
    """The pushes and the skipped parts of one campaign: its budgets, its exclusion zones and its push distance.

    ``distance_mm`` is the push every pick of the campaign makes (``push_mm`` of a ``PickRun`` or a console run,
    settled against the config by ``resolve_push_distance``); ``None`` for a campaign that pushes nothing, whose
    distance the config refused. One campaign is driven from one thread.
    """

    def __init__(self, *, distance_mm: Optional[float], budgets: Optional[PushBudgets] = None,
                 zones: Optional[ExclusionZones] = None) -> None:
        distance = None if distance_mm is None else float(distance_mm)
        if distance is not None and (not math.isfinite(distance) or distance <= 0.0):
            raise ValueError(f"a campaign's push distance is a positive number of mm, got {distance_mm!r}")
        self.distance_mm = distance
        self.budgets = budgets if budgets is not None else PushBudgets()
        self.zones = zones if zones is not None else ExclusionZones()

    def start_pick(self) -> int:
        """A pick of this campaign starts: its own two pushes are available again, and zones past their lifetime
        are dropped. Returns the pick's index, 1 for the first."""
        self.zones.start_pick()
        return self.budgets.start_pick()


@dataclass(frozen=True, slots=True)
class PushGate:
    """What lets one pick push: built by the service from the policy for that pick, and nothing pushes without it.

    ``budgets`` and ``distance_mm`` are the campaign's. ``cell`` is the cell's push inputs. ``operator_box`` is the
    declared fixture box (``recovery.fixture``), which only narrows the push box. ``on_approach_blocked`` is whether a
    pick whose every approach sweep was blocked is a trigger too, which it is where approach validation runs.
    """

    budgets: PushBudgets
    distance_mm: float
    cell: PushCell
    operator_box: Optional[AxisBox] = None
    on_approach_blocked: bool = False


@dataclass(frozen=True, slots=True)
class PickPush:
    """What one push a pick considered came to.

    ``trigger`` is why it was considered (:data:`TRIGGER_ALL_COLLIDED`, :data:`TRIGGER_APPROACH_BLOCKED`). ``code``
    is what it came to: a ``PushOutcomeCode`` value where the push was driven or refused by
    :func:`~src.robot.grasping.recovery.push_motion.execute_push`; the planner's or the budgets' code, or a
    ``refused_*`` of the pick's own, where it was refused before that. ``sentence`` says it in words.

    ``motion_started`` is whether the push may have moved the part: something beyond a refusal before motion was
    commanded (``PushOutcome.motion_started``). ``arm_moved`` is whether the arm left the look for the push at all,
    P0 in the air above a part nothing touched included (``refused_down_not_sent``): the budgets count every such push,
    as ``PushBudgets.record`` asks (as soon as a push commanded any motion), and the record's ``recovery_actions`` carry
    it. ``stopped`` is whether the push ended where the arm stands, a person to decide (``controller_stopped`` where
    the controller said it cannot move): no further motion follows, and the campaign ends. ``looked_again`` is the look
    the pick perceived from after the push, where the arm went back to it.
    """

    trigger: str
    code: str
    sentence: str
    motion_started: bool = False
    arm_moved: bool = False
    stopped: bool = False
    controller_stopped: bool = False
    distance_mm: Optional[float] = None
    part_centre_mm: Optional[Vec3] = None
    landing_mm: Optional[Vec3] = None
    direction: Optional[Vec3] = None
    leg: Optional[str] = None
    legs_done: tuple[str, ...] = ()
    looked_again: str = ""
    outcome: Optional["PushOutcome"] = field(default=None, repr=False, compare=False)

    @property
    def pushed(self) -> bool:
        """Whether the whole push ran and the arm went back to the look."""
        return self.motion_started and not self.stopped

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-safe values, for telemetry."""

        def _r(values: Optional[Sequence[float]]) -> Optional[list[float]]:
            return None if values is None else [round(float(v), 2) for v in values]

        return {
            "trigger": self.trigger,
            "code": self.code,
            "sentence": self.sentence,
            "motion_started": self.motion_started,
            "arm_moved": self.arm_moved,
            "stopped": self.stopped,
            "controller_stopped": self.controller_stopped,
            "distance_mm": None if self.distance_mm is None else round(float(self.distance_mm), 2),
            "part_centre_mm": _r(self.part_centre_mm),
            "landing_mm": _r(self.landing_mm),
            "direction": _r(self.direction),
            "leg": self.leg,
            "legs_done": list(self.legs_done),
            "looked_again": self.looked_again,
        }


@dataclass(frozen=True, slots=True)
class FailedPart:
    """The part a pick failed on, as ``next_target`` skips it: its label, where it stood in BASE, its footprint."""

    label: str
    centre_mm: Vec3
    footprint_diagonal_mm: Optional[float] = None


def support_points_of_views(
    views: Sequence[tuple[Any, Any, Any]],
    *,
    support_normal: ArrayLike,
    support_offset_mm: float,
) -> np.ndarray:
    """The support points every view saw, in BASE mm: the table a push's landing is bounded by.

    ``views`` are ``(depth_mm, intrinsics, camera_to_base)`` of every look of the pick (and every fixed camera fused
    with it): the whole depth image of each, not its masks, since nothing segments the table. Each is placed by its own
    CAMERA to BASE and the points within the support band are kept
    (:func:`~src.robot.grasping.recovery.push_planner.table_points_from_cloud`), so what one look saw in a part's
    shadow another fills in. A view with no depth or no placement gives nothing.
    """
    kept: list[np.ndarray] = []
    for depth_mm, intrinsics, camera_to_base in views:
        if depth_mm is None or intrinsics is None or camera_to_base is None:
            continue
        depth = np.asarray(depth_mm, dtype=np.float64)
        lens = np.asarray(intrinsics, dtype=np.float64)
        placed = np.asarray(camera_to_base, dtype=np.float64)
        if depth.ndim != 2 or lens.shape != (3, 3) or placed.shape != (4, 4) or not np.all(np.isfinite(placed)):
            continue
        rows, cols = depth.shape
        stride = max(1, int(round(rows / _SUPPORT_ROWS)))
        v, u = np.mgrid[0:rows:stride, 0:cols:stride]
        z = depth[v, u]
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            continue
        z, u, v = z[valid], u[valid].astype(np.float64), v[valid].astype(np.float64)
        camera = np.column_stack([(u - lens[0, 2]) * z / lens[0, 0], (v - lens[1, 2]) * z / lens[1, 1], z])
        base = camera @ placed[:3, :3].T + placed[:3, 3]
        table = table_points_from_cloud(base, support_normal=support_normal, support_offset_mm=support_offset_mm)
        if len(table):
            kept.append(table)
    return np.vstack(kept) if kept else np.zeros((0, 3), dtype=np.float64)
