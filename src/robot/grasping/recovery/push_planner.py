"""Plan one push of a failed part along the gripper's closing axis, from what the camera saw.

Pure numpy and no robot import. The pick attempt passes in BASE-frame points it already has. The answer is a
:class:`PushPlan` (where the hand goes, how far the part travels, where it is predicted to land) or a
:class:`PushRefusal` (a reason code and one sentence). Nothing here moves the arm, asks a person or reads config.

The owner's rules (2026-09-29), each one a named constant below:

* **The pusher.** The jaws stay open and the outer face of the leading finger pushes the part along the closing
  axis. The tool closes along the push direction, and either finger can lead. No DO0 change is needed for that;
  the caller checks the jaw count. Only the finger touches the part: a part that reaches up to the palm's
  underside (less :data:`PALM_PART_CLEARANCE_MM`) is not pushed, since the housing would push it too.
* **The direction.** The push goes toward free space and away from the neighbours that blocked the grasp. Each
  direction the caller offers (:data:`DEFAULT_PUSH_AXES_XY`, each axis both ways) must pass these tests, and the
  planner keeps the one that gains the most clearance:

  1. The open hand, swept from where it comes down to where it stops, keeps the caller's hand clearance
     (``hand_clearance_mm``, never under :data:`HAND_NEIGHBOUR_CLEARANCE_MM`) from every neighbour point. The
     pick passes the camera world's ``margin_mm`` plus the arm's ``line_clearance_mm`` (25 mm on the shipped
     cell): the box the world grows around a neighbour, and the distance every judged line keeps from it, so
     the planner does not plan a push the arm's line judge would predictably refuse. The fingers are checked
     at every height, the palm only above its own underside. Along the push the palm reaches the larger of the
     fingers' outer faces and half the housing's thickness (``PushHand.palm_thickness_mm``, 75 mm on the
     Hand-E).
  2. The part's own path keeps :data:`PATH_NEIGHBOUR_CLEARANCE_MM` from every neighbour point that is not
     behind it.
  3. The part's minimum clearance to its neighbours grows by at least :data:`MIN_CLEARANCE_GAIN_MM`.
  4. The landing and the hand stay inside the push box (below), over table the camera saw.
  5. Every TCP point of the push stays inside the workspace by :data:`WORKSPACE_MARGIN_MM`.

  The finger comes down on the side facing away from the push direction. So a neighbour within 25 mm can
  seldom be pushed straight away from: the hand would have to fit into the gap the grasp could not use.
  The planner then slides the part sideways out of that neighbour's reach, or it refuses.
* **The height.** The part must rest on the support: the low end of its heights (:data:`PART_BASE_PERCENTILE`,
  depth noise within the support band kept) lies at most :data:`PART_BASE_MAX_MM` above it. A part resting on
  something nobody segmented, or seen only from above, is not pushed: the finger could pass under it. The
  part's own height is its top less that base. The fingertip rides at support + clamp(own height / 2,
  :data:`FINGER_HEIGHT_MIN_MM`, :data:`FINGER_HEIGHT_MAX_MM`). No push is planned for a part whose own height is
  under :data:`MIN_PUSHABLE_PART_HEIGHT_MM`. Every TCP point stays :data:`WORKSPACE_MARGIN_MM` above the
  workspace ``z_min``, and the same distance inside every other face.
* **The push box.** Without a container, the box is the workspace box intersected with the table region the
  camera saw, shrunk by (push distance + :data:`AUTO_BOX_EXTRA_SHRINK_MM`). With a declared container, the box
  is the container's interior. The predicted landing plus :data:`LANDING_MARGIN_MM` must stay inside the box.
  The hand comes down only inside the unshrunk region: over table the camera saw, or inside the container.

  The table the camera saw is kept cell by cell (:data:`TABLE_CELL_MM` cells in BASE XY), not as the box around
  its points. A cell is seen when a table point falls in it, or when the part or a segmented neighbour stands
  on it: they hide the table they stand on. So the shrunk box is tested around every landing point: every
  cell within (push distance + :data:`AUTO_BOX_EXTRA_SHRINK_MM` + :data:`LANDING_MARGIN_MM`) of it must be
  seen, and so must every cell under the hand's swept footprint. A table edge that is not parallel to BASE X
  or Y, a hole, a shadow behind a tall object, or a gap between two surfaces refuses the direction. From one
  45 deg wrist view, the shadow behind a part usually lies inside that window, so the fused table points of
  several views are what let a push through.
* **The distance.** The default is :data:`DEFAULT_PUSH_DISTANCE_MM`. Above :data:`PUSH_DISTANCE_CAP_MM` the
  push is refused, never clamped, and so is a push shorter than :data:`MIN_CLEARANCE_GAIN_MM`, which could not
  open that much room. :func:`resolve_push_distance` settles a request against the config:
  ``recovery.fixture.push_distance_mm`` when nobody asks, and never more than ``recovery.fixture.max_nudge_mm``,
  the longest push the cell allows. A cell that declares no fixture takes the defaults, 30 and 50 mm.
* **The legs.** :attr:`PushPlan.legs` carries each judged line's speed and acceleration in m/s and m/s^2, the
  units ``arm.move(linear=True, vel=..., acc=...)`` takes, and nothing in mm/s.

The planner cannot see what lies behind the part, below the camera's view. Unsegmented clutter is not in the
neighbour points either. Both are left to the motion layer: every leg is judged against the live camera world
when it runs. The landing is a prediction, not a guard. A pushed part may rotate, tip or roll.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "AUTO_BOX_EXTRA_SHRINK_MM",
    "BACK_OFF_MM",
    "BACK_OFF_SPEED_MM_S",
    "CONTACT_GAP_MM",
    "DEFAULT_PUSH_AXES_XY",
    "DEFAULT_PUSH_DISTANCE_MM",
    "DESCENT_SPEED_MM_S",
    "DIRECTION_HAND_BLOCKED",
    "DIRECTION_HAND_OUTSIDE_REGION",
    "DIRECTION_HAND_OVER_UNSEEN_TABLE",
    "DIRECTION_LANDING_OUTSIDE_BOX",
    "DIRECTION_LANDING_OVER_UNSEEN_TABLE",
    "DIRECTION_NO_CLEARANCE_GAIN",
    "DIRECTION_OK",
    "DIRECTION_PATH_BLOCKED",
    "DIRECTION_TCP_OUTSIDE_WORKSPACE",
    "FINGER_HEIGHT_MAX_MM",
    "FINGER_HEIGHT_MIN_MM",
    "HAND_NEIGHBOUR_CLEARANCE_MM",
    "LANDING_MARGIN_MM",
    "LEG_ACCEL_M_S2",
    "LIFT_SPEED_MM_S",
    "MAX_SUPPORT_TILT_DEG",
    "MIN_CLEARANCE_GAIN_MM",
    "MIN_PUSHABLE_PART_HEIGHT_MM",
    "MIN_SEEN_TABLE_CELLS",
    "NEIGHBOUR_EVIDENCE_RADIUS_MM",
    "PALM_PART_CLEARANCE_MM",
    "PART_BASE_MAX_MM",
    "PART_BASE_PERCENTILE",
    "PATH_NEIGHBOUR_CLEARANCE_MM",
    "PUSH_ACCEL_M_S2",
    "PUSH_APPROACH_RISE_MM",
    "PUSH_DISTANCE_CAP_MM",
    "PUSH_SPEED_MM_S",
    "REFUSED_ABOVE_Z_MAX",
    "REFUSED_BELOW_Z_MIN",
    "REFUSED_NO_BLOCKING_NEIGHBOUR",
    "REFUSED_NO_FREE_DIRECTION",
    "REFUSED_NO_PUSH_AXIS",
    "REFUSED_NO_TABLE_SEEN",
    "REFUSED_NO_TARGET_POINTS",
    "REFUSED_PART_NOT_ON_SUPPORT",
    "REFUSED_PART_REACHES_THE_PALM",
    "REFUSED_PART_TOO_FLAT",
    "REFUSED_PUSH_BOX_EMPTY",
    "REFUSED_PUSH_DISTANCE_ABOVE_CAP",
    "REFUSED_PUSH_DISTANCE_ABOVE_CONFIG",
    "REFUSED_PUSH_DISTANCE_INVALID",
    "REFUSED_PUSH_DISTANCE_TOO_SHORT",
    "REFUSED_SUPPORT_TILTED",
    "SUPPORT_BAND_MM",
    "TABLE_CELL_MM",
    "WORKSPACE_MARGIN_MM",
    "AxisBox",
    "DirectionVerdict",
    "NeighbourEvidence",
    "PushHand",
    "PushLeg",
    "PushPlan",
    "PushRefusal",
    "neighbour_evidence",
    "plan_push",
    "resolve_push_distance",
    "swept_target_points_mm",
    "table_points_from_cloud",
]

# --- the owner's numbers (2026-09-29) --------------------------------------------------------------------------

#: How far the part is pushed when nobody says otherwise, in mm.
DEFAULT_PUSH_DISTANCE_MM = 30.0
#: The hard cap on a push, in mm. A longer push is refused, never shortened.
PUSH_DISTANCE_CAP_MM = 50.0
#: A part lower than this (its own height) is not pushed: the finger would ride over it or scrape the table.
MIN_PUSHABLE_PART_HEIGHT_MM = 15.0
#: The fingertip height above the support is half the part's height, clamped to this range. The 10 mm floor is
#: the 5 mm plane band the camera world drops as bench plus the 5 mm clearance a grasp keeps above the support.
FINGER_HEIGHT_MIN_MM = 10.0
FINGER_HEIGHT_MAX_MM = 20.0
#: Every TCP point of the push stays this far inside every face of the workspace box. For ``z_min`` this is the
#: owner's rule. The other faces get the same margin because the arm's own workspace guard applies it there too
#: (``robot.safety.limits.workspace_margin_mm``, 20 mm).
WORKSPACE_MARGIN_MM = 20.0
#: A push is planned only when a neighbour point lies this close to the part (in the support plane): the
#: evidence that a neighbour, and not the part itself, left the fingers no room.
NEIGHBOUR_EVIDENCE_RADIUS_MM = 25.0
#: The automatic push box is shrunk by the push distance plus this, so a part pushed from the edge of the seen
#: table still lands well inside it.
AUTO_BOX_EXTRA_SHRINK_MM = 30.0
#: The predicted landing footprint grown by this must stay inside the push box.
LANDING_MARGIN_MM = 15.0
#: Push speed along the contact line, and its acceleration.
PUSH_SPEED_MM_S = 25.0
PUSH_ACCEL_M_S2 = 0.1
#: Speed of the line down beside the part, the 5 mm back-off and the lift. The owner fixed 50 mm/s for down
#: and back; the lift uses the same. Their acceleration is the push's: nothing asked for faster.
DESCENT_SPEED_MM_S = 50.0
BACK_OFF_SPEED_MM_S = 50.0
LIFT_SPEED_MM_S = 50.0
LEG_ACCEL_M_S2 = PUSH_ACCEL_M_S2
#: How far the finger backs off along the push line before it lifts.
BACK_OFF_MM = 5.0

# --- geometry choices, each one named ------------------------------------------------------------------------

#: The leading finger's outer face comes down this far behind the part's trailing edge, then travels this gap
#: plus the push distance. Nothing touches the part while the finger descends.
CONTACT_GAP_MM = 10.0
#: P0, in the air above the contact start, and the lift point are this far above the contact height: the grasp
#: standoff (``execution_policy``'s 80 mm).
PUSH_APPROACH_RISE_MM = 80.0
#: The open hand, swept along the push, keeps at least this far from every neighbour point: the arm's own line
#: clearance (``robot.safety.planned_motion.line_clearance_mm``). It is the floor of ``plan_push``'s
#: ``hand_clearance_mm``, which the pick raises to the camera world's ``margin_mm`` plus that line clearance, since
#: the world grows every neighbour's box by the margin before the line judge keeps its clearance from it.
HAND_NEIGHBOUR_CLEARANCE_MM = 10.0
#: The pushed part's swept footprint keeps this far from every neighbour point not behind it.
PATH_NEIGHBOUR_CLEARANCE_MM = 5.0
#: A push must grow the part's minimum clearance to its neighbours by at least this, about one finger's
#: thickness. A push that opens less room is not worth the contact. A push can open at most its own length,
#: so a push shorter than this is refused outright.
MIN_CLEARANCE_GAIN_MM = 10.0
#: Points within this of the support plane are table, not part and not neighbour. It is the camera world's
#: plane band (``plane_clearance_mm``).
SUPPORT_BAND_MM = 5.0
#: A push is planned only on a support within this many degrees of level.
MAX_SUPPORT_TILT_DEG = 5.0
#: Where the part rests is this percentile of its heights above the support (points down to the support band
#: kept, since the bottom of a side face lies within it): a few stray low points do not decide it.
PART_BASE_PERCENTILE = 2.0
#: A part resting higher than this above the support is not pushed. It is the finger's lowest ride
#: (:data:`FINGER_HEIGHT_MIN_MM`): a part resting higher stands on something, and the finger could pass under
#: it and push that instead.
PART_BASE_MAX_MM = FINGER_HEIGHT_MIN_MM
#: The part's top stays this far below the palm's underside, so the finger, and not the housing, touches it.
PALM_PART_CLEARANCE_MM = 5.0
#: The seen table region is the box of the table points between this percentile and its complement on each
#: axis. That box is slightly smaller than all the points, so one stray point cannot widen it. The cells below
#: decide what inside it was seen.
TABLE_REGION_PERCENTILE = 0.5
#: The table the camera saw is kept as cells of this size in BASE XY. The table points should be no sparser
#: than this, or seen table between them reads as holes (which only refuses more).
TABLE_CELL_MM = 5.0
#: Fewer seen cells than this (a 50 x 50 mm patch) near the part is not a table region.
MIN_SEEN_TABLE_CELLS = 100
#: Fewer part points above the support than this is not a part.
MIN_TARGET_POINTS = 10
#: The part's top is this percentile of its points' heights above the support: one depth spike does not
#: make a flat part tall.
PART_HEIGHT_PERCENTILE = 95.0
#: Pairwise distances are taken between cells of this size in the support plane, not between raw points.
PLANNER_GRID_MM = 2.0
#: Neighbour points farther than this (in the support plane) from the part's footprint box cannot reach the
#: hand, the path or the landing of a push of at most 50 mm, and are dropped before any distance is taken. The
#: seen-table cells reach as far around the part; beyond that nothing counts as seen.
NEIGHBOUR_WINDOW_MM = 250.0
#: Step of :func:`swept_target_points_mm`.
SWEEP_STEP_MM = 5.0
#: The axes a push may run along when the caller offers none, in BASE XY. Each is tried both ways, in this
#: order, and on a tie in clearance gain the first one tried wins.
_DIAGONAL = math.sqrt(0.5)
DEFAULT_PUSH_AXES_XY: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (_DIAGONAL, _DIAGONAL),
    (0.0, 1.0),
    (-_DIAGONAL, _DIAGONAL),
)

_EPS_MM = 1e-6
#: Pairwise distances are taken in blocks of at most this many pairs, so a large part cannot make the pick
#: attempt allocate hundreds of megabytes at once (about 20 MB per block).
_DISTANCE_PAIRS_PER_BLOCK = 500_000
_MM_PER_M = 1000.0

# --- reason codes --------------------------------------------------------------------------------------------

REFUSED_PUSH_DISTANCE_INVALID = "push_distance_invalid"
REFUSED_PUSH_DISTANCE_ABOVE_CAP = "push_distance_above_cap"
REFUSED_PUSH_DISTANCE_ABOVE_CONFIG = "push_distance_above_config"
REFUSED_PUSH_DISTANCE_TOO_SHORT = "push_distance_too_short"
REFUSED_SUPPORT_TILTED = "support_tilted"
REFUSED_NO_TARGET_POINTS = "no_target_points"
REFUSED_PART_NOT_ON_SUPPORT = "part_not_on_support"
REFUSED_PART_TOO_FLAT = "part_too_flat"
REFUSED_PART_REACHES_THE_PALM = "part_reaches_the_palm"
REFUSED_NO_BLOCKING_NEIGHBOUR = "no_blocking_neighbour"
REFUSED_NO_TABLE_SEEN = "no_table_seen"
REFUSED_PUSH_BOX_EMPTY = "push_box_empty"
REFUSED_BELOW_Z_MIN = "below_z_min"
REFUSED_ABOVE_Z_MAX = "above_z_max"
REFUSED_NO_PUSH_AXIS = "no_push_axis"
REFUSED_NO_FREE_DIRECTION = "no_free_direction"

DIRECTION_OK = "ok"
DIRECTION_HAND_BLOCKED = "hand_blocked"
DIRECTION_PATH_BLOCKED = "path_blocked"
DIRECTION_NO_CLEARANCE_GAIN = "no_clearance_gain"
DIRECTION_LANDING_OUTSIDE_BOX = "landing_outside_box"
DIRECTION_HAND_OUTSIDE_REGION = "hand_outside_region"
DIRECTION_LANDING_OVER_UNSEEN_TABLE = "landing_over_unseen_table"
DIRECTION_HAND_OVER_UNSEEN_TABLE = "hand_over_unseen_table"
DIRECTION_TCP_OUTSIDE_WORKSPACE = "tcp_outside_workspace"

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


# --- value types ---------------------------------------------------------------------------------------------


def _vec3(value: Sequence[float], name: str) -> Vec3:
    out = tuple(float(c) for c in value)
    if len(out) != 3 or not all(math.isfinite(c) for c in out):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return (out[0], out[1], out[2])


@dataclass(frozen=True, slots=True)
class AxisBox:
    """An axis-aligned box in BASE, in mm: the workspace, a container's interior or an operator's box."""

    min_mm: Vec3
    max_mm: Vec3

    def __post_init__(self) -> None:
        low = _vec3(self.min_mm, "AxisBox.min_mm")
        high = _vec3(self.max_mm, "AxisBox.max_mm")
        for axis in range(3):
            if not high[axis] > low[axis]:
                raise ValueError(
                    f"AxisBox.max_mm[{axis}] ({high[axis]}) must exceed min_mm[{axis}] ({low[axis]})"
                )
        object.__setattr__(self, "min_mm", low)
        object.__setattr__(self, "max_mm", high)

    @property
    def xy(self) -> tuple[Vec2, Vec2]:
        return (self.min_mm[0], self.min_mm[1]), (self.max_mm[0], self.max_mm[1])


@dataclass(frozen=True, slots=True)
class PushHand:
    """The open hand as a pusher, in mm, in the grasp frame of :class:`ParallelJawGripperModel`.

    ``finger_thickness_mm`` runs along the closing axis and ``finger_width_mm`` across it.
    ``fingertip_depth_mm`` is how far the fingertip reaches past the TCP along the approach.
    ``finger_length_mm`` is how far the finger runs back from the TCP toward the palm. The palm
    (``palm_width_mm`` across the closing axis) starts where the finger ends. ``open_width_mm`` is the
    gap between the finger pads with the jaws open. ``palm_thickness_mm`` is the housing along the closing
    axis (the gripper registry's field, 75 mm on the Hand-E); unset, the housing is taken as wide as the open
    fingers' outer faces.
    """

    finger_thickness_mm: float
    finger_width_mm: float
    fingertip_depth_mm: float
    finger_length_mm: float
    palm_width_mm: float
    open_width_mm: float
    palm_thickness_mm: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("finger_thickness_mm", "finger_width_mm", "fingertip_depth_mm", "finger_length_mm",
                     "palm_width_mm", "open_width_mm", "palm_thickness_mm"):
            raw = getattr(self, name)
            if raw is None and name == "palm_thickness_mm":
                continue
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"PushHand.{name} must be a positive number of mm, got {value!r}")
            object.__setattr__(self, name, value)

    @classmethod
    def from_gripper_model(
        cls, model: Any, *, open_width_mm: float, palm_thickness_mm: Optional[float] = None,
    ) -> "PushHand":
        """The hand from a :class:`ParallelJawGripperModel` (read by name, so nothing is imported).

        ``palm_thickness_mm`` is the registry's housing thickness along the closing axis. The model does not
        carry it, so pass it from the gripper config; a model that does carry it is read.
        """

        thickness = palm_thickness_mm if palm_thickness_mm is not None else getattr(model, "palm_thickness_mm", None)
        return cls(
            finger_thickness_mm=float(model.finger_thickness_mm),
            finger_width_mm=float(model.finger_width_mm),
            fingertip_depth_mm=float(model.fingertip_depth_mm),
            finger_length_mm=float(model.finger_length_mm),
            palm_width_mm=float(model.palm_width_mm),
            open_width_mm=float(open_width_mm),
            palm_thickness_mm=None if thickness is None else float(thickness),
        )

    @property
    def half_outer_mm(self) -> float:
        """From the TCP to either finger's outer face, along the closing axis, with the jaws open."""

        return 0.5 * self.open_width_mm + self.finger_thickness_mm

    @property
    def palm_half_along_mm(self) -> float:
        """From the TCP to the housing's face along the closing axis: never less than the fingers' outer face."""

        if self.palm_thickness_mm is None:
            return self.half_outer_mm
        return max(self.half_outer_mm, 0.5 * self.palm_thickness_mm)

    @property
    def palm_underside_above_tip_mm(self) -> float:
        """How far above the fingertip the palm's underside is."""

        return self.fingertip_depth_mm + self.finger_length_mm


@dataclass(frozen=True, slots=True)
class PushLeg:
    """One judged straight line of the push, TCP to TCP in BASE mm, with the speed (m/s) and acceleration
    (m/s^2) it runs at: the units ``arm.move(linear=True, vel=..., acc=...)`` takes."""

    name: str
    start_tcp_mm: Vec3
    end_tcp_mm: Vec3
    vel_m_s: float
    acc_m_s2: float


def _leg(name: str, start: Vec3, end: Vec3, speed_mm_s: float, accel_m_s2: float) -> PushLeg:
    return PushLeg(name, start, end, speed_mm_s / _MM_PER_M, accel_m_s2)


@dataclass(frozen=True, slots=True)
class DirectionVerdict:
    """What one candidate direction met. ``clearance_gain_mm`` is None when it failed before the gain was taken."""

    direction_xy: Vec2
    verdict: str
    clearance_gain_mm: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_xy": [round(c, 4) for c in self.direction_xy],
            "verdict": self.verdict,
            "clearance_gain_mm": None if self.clearance_gain_mm is None else round(self.clearance_gain_mm, 2),
        }


@dataclass(frozen=True, slots=True)
class PushPlan:
    """One planned push. Every position is the TCP in BASE mm. The tool approaches along ``approach`` and
    closes along ``direction`` (either sign: either finger may lead).

    The motion is: a planned move to ``approach_tcp_mm`` (P0, in the air), then :attr:`legs`, each a judged
    straight line. The lines go down beside the part, push it, back off 5 mm, and lift.
    ``part_height_mm`` is the part's own height (its top less ``part_base_mm``, where it rests above the
    support).
    """

    direction: Vec3
    approach: Vec3
    approach_tcp_mm: Vec3
    contact_start_tcp_mm: Vec3
    end_tcp_mm: Vec3
    back_off_tcp_mm: Vec3
    lift_tcp_mm: Vec3
    finger_height_mm: float
    part_height_mm: float
    part_base_mm: float
    push_distance_mm: float
    target_centre_mm: Vec3
    predicted_landing_centre_mm: Vec3
    landing_box_xy_mm: tuple[Vec2, Vec2]
    clearance_before_mm: float
    clearance_after_mm: float
    candidates: tuple[DirectionVerdict, ...] = ()

    @property
    def clearance_gain_mm(self) -> float:
        return self.clearance_after_mm - self.clearance_before_mm

    @property
    def legs(self) -> tuple[PushLeg, ...]:
        """The four contact legs in order: ``down``, ``push``, ``back``, ``up``."""

        return (
            _leg("down", self.approach_tcp_mm, self.contact_start_tcp_mm, DESCENT_SPEED_MM_S, LEG_ACCEL_M_S2),
            _leg("push", self.contact_start_tcp_mm, self.end_tcp_mm, PUSH_SPEED_MM_S, PUSH_ACCEL_M_S2),
            _leg("back", self.end_tcp_mm, self.back_off_tcp_mm, BACK_OFF_SPEED_MM_S, LEG_ACCEL_M_S2),
            _leg("up", self.back_off_tcp_mm, self.lift_tcp_mm, LIFT_SPEED_MM_S, LEG_ACCEL_M_S2),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable values, for telemetry."""

        def _r(values: Sequence[float]) -> list[float]:
            return [round(float(c), 3) for c in values]

        return {
            "direction": _r(self.direction),
            "approach": _r(self.approach),
            "approach_tcp_mm": _r(self.approach_tcp_mm),
            "contact_start_tcp_mm": _r(self.contact_start_tcp_mm),
            "end_tcp_mm": _r(self.end_tcp_mm),
            "back_off_tcp_mm": _r(self.back_off_tcp_mm),
            "lift_tcp_mm": _r(self.lift_tcp_mm),
            "finger_height_mm": round(self.finger_height_mm, 2),
            "part_height_mm": round(self.part_height_mm, 2),
            "part_base_mm": round(self.part_base_mm, 2),
            "push_distance_mm": round(self.push_distance_mm, 2),
            "target_centre_mm": _r(self.target_centre_mm),
            "predicted_landing_centre_mm": _r(self.predicted_landing_centre_mm),
            "landing_box_xy_mm": [_r(self.landing_box_xy_mm[0]), _r(self.landing_box_xy_mm[1])],
            "clearance_before_mm": round(self.clearance_before_mm, 2),
            "clearance_after_mm": round(self.clearance_after_mm, 2),
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class PushRefusal:
    """No push: ``code`` is one of the ``REFUSED_*`` strings, ``sentence`` says why in words."""

    code: str
    sentence: str
    candidates: tuple[DirectionVerdict, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "sentence": self.sentence,
                "candidates": [c.to_dict() for c in self.candidates]}


@dataclass(frozen=True, slots=True)
class NeighbourEvidence:
    """Whether a neighbour stands within ``radius_mm`` of the part, measured in the support plane."""

    found: bool
    nearest_gap_mm: float
    radius_mm: float


# --- the support plane ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SupportFrame:
    """Orthonormal frame on the support plane: ``e1``, ``e2`` span it, ``n`` is its unit normal.

    A point p has plane coordinates ``(p.e1, p.e2)`` and height ``p.n - offset`` above the support,
    the same signed clearance as :class:`SupportPlane`.
    """

    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray
    offset_mm: float

    @classmethod
    def build(cls, normal: ArrayLike, offset_mm: float) -> "_SupportFrame":
        n = np.asarray(normal, dtype=np.float64).reshape(-1)
        if n.shape != (3,) or not np.all(np.isfinite(n)) or float(np.linalg.norm(n)) < 1e-9:
            raise ValueError(f"support_normal must be a non-zero 3-vector, got {normal!r}")
        offset = float(offset_mm)
        if not math.isfinite(offset):
            raise ValueError(f"support_offset_mm must be finite, got {offset_mm!r}")
        n = n / float(np.linalg.norm(n))
        if n[2] < 0.0:
            raise ValueError("support_normal must point up (positive BASE z), away from the support")
        x = np.array([1.0, 0.0, 0.0])
        e1 = x - float(x @ n) * n
        e1 = e1 / float(np.linalg.norm(e1))
        e2 = np.cross(n, e1)
        return cls(e1=e1, e2=e2, n=n, offset_mm=offset)

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.n[2])))))

    def plane_xy(self, points: np.ndarray) -> np.ndarray:
        return np.column_stack([points @ self.e1, points @ self.e2])

    def height(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points @ self.n - self.offset_mm, dtype=np.float64)

    def to_base(self, a: float, b: float, height_mm: float) -> np.ndarray:
        return np.asarray(a * self.e1 + b * self.e2 + (self.offset_mm + height_mm) * self.n, dtype=np.float64)

    def to_base_many(self, a: np.ndarray, b: np.ndarray, height_mm: float) -> np.ndarray:
        """:meth:`to_base` for arrays of plane coordinates at one height, as (N, 3) BASE rows."""

        return np.asarray(a[:, None] * self.e1[None, :] + b[:, None] * self.e2[None, :]
                          + (self.offset_mm + height_mm) * self.n[None, :], dtype=np.float64)


# --- the table the camera saw, cell by cell ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SeenTable:
    """The table the camera saw, as :data:`TABLE_CELL_MM` cells in BASE XY around the part.

    A cell is seen when a table point falls in it, or when the part or a neighbour stands on it. ``integral``
    is the summed-area table of the seen cells, so any rectangle of cells is tested in constant time. Cells
    outside the grid count as unseen.
    """

    origin_xy: Vec2
    shape: tuple[int, int]
    integral: np.ndarray
    table_cells: int

    @classmethod
    def build(cls, *, table_xy: np.ndarray, standing_xy: Sequence[np.ndarray], around_xy: np.ndarray,
              reach_mm: float) -> "_SeenTable":
        cell = TABLE_CELL_MM
        low = np.floor((around_xy.min(axis=0) - reach_mm) / cell) * cell
        high = around_xy.max(axis=0) + reach_mm
        nx = int(math.ceil((float(high[0]) - float(low[0])) / cell)) + 1
        ny = int(math.ceil((float(high[1]) - float(low[1])) / cell)) + 1
        origin: Vec2 = (float(low[0]), float(low[1]))
        grid = np.zeros((nx, ny), dtype=bool)
        cls._mark(grid, table_xy, origin)
        table_cells = int(grid.sum())
        for points in standing_xy:
            cls._mark(grid, points, origin)
        integral = np.zeros((nx + 1, ny + 1), dtype=np.int64)
        integral[1:, 1:] = grid.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
        return cls(origin_xy=origin, shape=(nx, ny), integral=integral, table_cells=table_cells)

    @staticmethod
    def _mark(grid: np.ndarray, points_xy: np.ndarray, origin: Vec2) -> None:
        if len(points_xy) == 0:
            return
        ix = np.floor((points_xy[:, 0] - origin[0]) / TABLE_CELL_MM).astype(np.int64)
        iy = np.floor((points_xy[:, 1] - origin[1]) / TABLE_CELL_MM).astype(np.int64)
        inside = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
        grid[ix[inside], iy[inside]] = True

    def windows_seen(self, points_xy: np.ndarray, half_mm: float) -> bool:
        """Whether every cell within ``half_mm`` (a square window) of every point is seen. ``half_mm = 0``
        asks about the cell each point lies in."""

        if len(points_xy) == 0:
            return True
        cell = TABLE_CELL_MM
        ox, oy = self.origin_xy
        ix0 = np.floor((points_xy[:, 0] - half_mm - ox) / cell).astype(np.int64)
        ix1 = np.floor((points_xy[:, 0] + half_mm - ox) / cell).astype(np.int64)
        iy0 = np.floor((points_xy[:, 1] - half_mm - oy) / cell).astype(np.int64)
        iy1 = np.floor((points_xy[:, 1] + half_mm - oy) / cell).astype(np.int64)
        nx, ny = self.shape
        if bool(np.any(ix0 < 0) or np.any(iy0 < 0) or np.any(ix1 >= nx) or np.any(iy1 >= ny)):
            return False
        s = self.integral
        seen = s[ix1 + 1, iy1 + 1] - s[ix0, iy1 + 1] - s[ix1 + 1, iy0] + s[ix0, iy0]
        return bool(np.all(seen == (ix1 - ix0 + 1) * (iy1 - iy0 + 1)))


# --- small numpy helpers -------------------------------------------------------------------------------------


def _points(value: ArrayLike, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    points = points.reshape(-1, 3) if points.ndim == 1 and points.size % 3 == 0 else points
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be shaped (N, 3), got {points.shape}")
    return points[np.all(np.isfinite(points), axis=1)]


def _cells(plane_xy: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points reduced to :data:`PLANNER_GRID_MM` cells: the mean position in each cell and its highest point."""

    if len(plane_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    keys = np.floor(plane_xy / PLANNER_GRID_MM).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    count = np.bincount(inverse).astype(np.float64)
    mean_a = np.bincount(inverse, weights=plane_xy[:, 0]) / count
    mean_b = np.bincount(inverse, weights=plane_xy[:, 1]) / count
    top = np.full(len(count), -np.inf)
    np.maximum.at(top, inverse, heights)
    return np.column_stack([mean_a, mean_b]), top


def _blocks(n_a: int, n_b: int) -> tuple[int, int]:
    """Block sizes over two point sets so one block holds at most :data:`_DISTANCE_PAIRS_PER_BLOCK` pairs."""

    a_step = max(1, min(n_a, _DISTANCE_PAIRS_PER_BLOCK))
    return a_step, max(1, _DISTANCE_PAIRS_PER_BLOCK // a_step)


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two 2-D point sets, ``inf`` when either is empty."""

    if len(a) == 0 or len(b) == 0:
        return math.inf
    a_step, b_step = _blocks(len(a), len(b))
    best = math.inf
    for i in range(0, len(a), a_step):
        a_block = a[i:i + a_step]
        for j in range(0, len(b), b_step):
            b_block = b[j:j + b_step]
            dx = a_block[:, None, 0] - b_block[None, :, 0]
            dy = a_block[:, None, 1] - b_block[None, :, 1]
            best = min(best, float((dx * dx + dy * dy).min()))
    return math.sqrt(best)


def _min_swept_distance(target_uv: np.ndarray, points_uv: np.ndarray, distance: float) -> float:
    """Smallest distance from any point to the part's footprint swept ``distance`` along +u."""

    if len(target_uv) == 0 or len(points_uv) == 0:
        return math.inf
    t_step, p_step = _blocks(len(target_uv), len(points_uv))
    best = math.inf
    for i in range(0, len(target_uv), t_step):
        t_block = target_uv[i:i + t_step]
        for j in range(0, len(points_uv), p_step):
            p_block = points_uv[j:j + p_step]
            du = p_block[None, :, 0] - t_block[:, None, 0]
            dv = p_block[None, :, 1] - t_block[:, None, 1]
            along = np.where(du < 0.0, du, np.where(du > distance, du - distance, 0.0))
            best = min(best, float((along * along + dv * dv).min()))
    return math.sqrt(best)


def _inside_xy(points_xy: np.ndarray, box: tuple[Vec2, Vec2], margin: float = 0.0) -> bool:
    (lo_x, lo_y), (hi_x, hi_y) = box
    return bool(
        np.all(points_xy[:, 0] - margin >= lo_x - _EPS_MM) and np.all(points_xy[:, 0] + margin <= hi_x + _EPS_MM)
        and np.all(points_xy[:, 1] - margin >= lo_y - _EPS_MM)
        and np.all(points_xy[:, 1] + margin <= hi_y + _EPS_MM)
    )


def _intersect(a: tuple[Vec2, Vec2], b: tuple[Vec2, Vec2]) -> tuple[Vec2, Vec2]:
    return ((max(a[0][0], b[0][0]), max(a[0][1], b[0][1])), (min(a[1][0], b[1][0]), min(a[1][1], b[1][1])))


def _shrink(box: tuple[Vec2, Vec2], by: float) -> tuple[Vec2, Vec2]:
    return ((box[0][0] + by, box[0][1] + by), (box[1][0] - by, box[1][1] - by))


def _empty(box: tuple[Vec2, Vec2]) -> bool:
    return box[1][0] <= box[0][0] or box[1][1] <= box[0][1]


def _tuple3(value: np.ndarray) -> Vec3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _span(low: float, high: float, step: float) -> np.ndarray:
    """``low`` to ``high`` inclusive, at most ``step`` apart."""

    return np.linspace(low, high, max(1, int(math.ceil((high - low) / step))) + 1)


# --- public helpers ------------------------------------------------------------------------------------------


def resolve_push_distance(
    requested_mm: Optional[float], *, default_mm: float, ceiling_mm: float,
) -> Union[float, PushRefusal]:
    """The push distance for one push, or a refusal: a request (API or ``PickRun`` keyword) settled against the
    config (owner, 2026-09-29).

    ``default_mm`` is ``recovery.fixture.push_distance_mm`` (30 mm unless the cell says otherwise), the push a
    pick makes when nobody asks for another. ``ceiling_mm`` is ``recovery.fixture.max_nudge_mm`` (50 mm unless
    the cell says less), the longest push the cell allows; :data:`PUSH_DISTANCE_CAP_MM` caps it whatever it says.
    A cell that declares no fixture passes the defaults, :data:`DEFAULT_PUSH_DISTANCE_MM` and the cap.

    * Nothing requested: the config's push distance.
    * A request up to the ceiling is taken as asked. Above the ceiling, or above :data:`PUSH_DISTANCE_CAP_MM`,
      it is refused with a sentence, never shortened.
    * A distance shorter than :data:`MIN_CLEARANCE_GAIN_MM` is refused: a push opens at most its own length,
      so it could never open the room a finger needs.
    * What is not a positive number, in the request or the config, is refused, and so is a config whose push
      distance lies above its own ceiling (the schema refuses that at load; this says it again for a caller
      that did not load one).
    """

    ceiling = float(ceiling_mm)
    if not math.isfinite(ceiling) or ceiling <= 0.0:
        return _refuse(REFUSED_PUSH_DISTANCE_INVALID,
                       f"The config's longest push (recovery.fixture.max_nudge_mm) of {ceiling_mm!r} mm is not a "
                       "positive number of mm, so no push is planned.")
    ceiling = min(ceiling, PUSH_DISTANCE_CAP_MM)
    if requested_mm is None:
        distance = float(default_mm)
        if not math.isfinite(distance) or distance <= 0.0:
            return _refuse(REFUSED_PUSH_DISTANCE_INVALID,
                           f"The config's push distance (recovery.fixture.push_distance_mm) of {default_mm!r} mm is "
                           "not a positive number of mm, so no push is planned.")
        if distance > ceiling + _EPS_MM:
            return _refuse(REFUSED_PUSH_DISTANCE_ABOVE_CONFIG,
                           f"The config's push distance of {distance:g} mm (recovery.fixture.push_distance_mm) is "
                           f"longer than the {ceiling:g} mm the cell allows (recovery.fixture.max_nudge_mm), so no "
                           "push is planned.")
        said = f"the config's push distance of {distance:g} mm (recovery.fixture.push_distance_mm)"
    else:
        distance = float(requested_mm)
        if not math.isfinite(distance) or distance <= 0.0:
            return _refuse(REFUSED_PUSH_DISTANCE_INVALID,
                           f"A push distance of {requested_mm!r} mm is not a positive number, so no push is planned.")
        if distance > PUSH_DISTANCE_CAP_MM + _EPS_MM:
            return _refuse(REFUSED_PUSH_DISTANCE_ABOVE_CAP,
                           f"A push of {distance:g} mm was asked for and the hard cap is "
                           f"{PUSH_DISTANCE_CAP_MM:g} mm, so no push is planned.")
        if distance > ceiling + _EPS_MM:
            return _refuse(REFUSED_PUSH_DISTANCE_ABOVE_CONFIG,
                           f"A push of {distance:g} mm was asked for and the cell allows {ceiling:g} mm "
                           "(recovery.fixture.max_nudge_mm), so no push is planned.")
        said = f"a push of {distance:g} mm"
    if distance < MIN_CLEARANCE_GAIN_MM - _EPS_MM:
        return _refuse(REFUSED_PUSH_DISTANCE_TOO_SHORT,
                       f"{said[0].upper()}{said[1:]} cannot open the {MIN_CLEARANCE_GAIN_MM:g} mm a finger needs, "
                       "so no push is planned.")
    return distance


def neighbour_evidence(
    *,
    target_points_mm: ArrayLike,
    neighbour_points_mm: ArrayLike,
    support_normal: ArrayLike = (0.0, 0.0, 1.0),
    support_offset_mm: float = 0.0,
    radius_mm: float = NEIGHBOUR_EVIDENCE_RADIUS_MM,
) -> NeighbourEvidence:
    """Whether a neighbour stands within ``radius_mm`` of the part: the trigger's evidence for a push.

    Distances are taken in the support plane between points more than :data:`SUPPORT_BAND_MM` above it, so
    a mask that bled onto the table is not a neighbour. The pick loop calls this before it asks for a plan,
    and :func:`plan_push` asks it again.
    """

    frame = _SupportFrame.build(support_normal, support_offset_mm)
    target = _points(target_points_mm, "target_points_mm")
    neighbours = _points(neighbour_points_mm, "neighbour_points_mm")
    target = target[frame.height(target) > SUPPORT_BAND_MM]
    neighbours = neighbours[frame.height(neighbours) > SUPPORT_BAND_MM]
    target_cells, _ = _cells(frame.plane_xy(target), frame.height(target))
    neighbour_cells, _ = _cells(frame.plane_xy(neighbours), frame.height(neighbours))
    gap = _min_distance(target_cells, neighbour_cells)
    return NeighbourEvidence(found=gap <= float(radius_mm), nearest_gap_mm=gap, radius_mm=float(radius_mm))


def table_points_from_cloud(
    cloud_mm: ArrayLike,
    *,
    support_normal: ArrayLike = (0.0, 0.0, 1.0),
    support_offset_mm: float = 0.0,
    band_mm: float = SUPPORT_BAND_MM,
) -> np.ndarray:
    """The points of a BASE cloud that lie on the support: within ``band_mm`` of the plane, either side."""

    frame = _SupportFrame.build(support_normal, support_offset_mm)
    cloud = _points(cloud_mm, "cloud_mm")
    return cloud[np.abs(frame.height(cloud)) <= float(band_mm)]


def swept_target_points_mm(
    target_points_mm: ArrayLike,
    direction: Sequence[float],
    distance_mm: float,
    *,
    step_mm: float = SWEEP_STEP_MM,
) -> np.ndarray:
    """The part's points repeated every ``step_mm`` along ``direction`` up to ``distance_mm``: the region to keep
    out of the planner world for the whole push, so the part is not an obstacle anywhere along its path."""

    points = _points(target_points_mm, "target_points_mm")
    unit = np.asarray(_vec3(direction, "direction"), dtype=np.float64)
    unit = unit / float(np.linalg.norm(unit))
    steps = max(1, int(math.ceil(float(distance_mm) / float(step_mm))))
    offsets = np.linspace(0.0, float(distance_mm), steps + 1)
    return np.vstack([points + t * unit for t in offsets])


# --- the planner ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Scene:
    frame: _SupportFrame
    target: np.ndarray            # raw BASE points above the band
    target_ab: np.ndarray         # their plane coordinates
    target_cells: np.ndarray      # plane cells
    neighbour_cells: np.ndarray   # plane cells
    neighbour_tops: np.ndarray    # highest point per neighbour cell, above the support
    centre_ab: np.ndarray
    part_height: float            # own height: top less base
    part_base: float              # where it rests above the support
    finger_height: float
    tcp_height: float
    distance: float
    clearance_before: float
    landing_box: tuple[Vec2, Vec2]
    hand_region: tuple[Vec2, Vec2]
    seen: Optional[_SeenTable]    # None inside a declared container
    workspace: AxisBox
    hand: PushHand
    hand_clearance: float         # the hand's clearance to every neighbour point, never under HAND_NEIGHBOUR_CLEARANCE_MM


def _refuse(code: str, sentence: str, candidates: tuple[DirectionVerdict, ...] = ()) -> PushRefusal:
    return PushRefusal(code=code, sentence=sentence, candidates=candidates)


def plan_push(
    *,
    target_points_mm: ArrayLike,
    neighbour_points_mm: ArrayLike,
    support_normal: ArrayLike,
    support_offset_mm: float,
    workspace: AxisBox,
    hand: PushHand,
    table_points_mm: Optional[ArrayLike] = None,
    push_distance_mm: float = DEFAULT_PUSH_DISTANCE_MM,
    container_interior: Optional[AxisBox] = None,
    operator_box: Optional[AxisBox] = None,
    push_axes_xy: Sequence[Sequence[float]] = DEFAULT_PUSH_AXES_XY,
    hand_clearance_mm: float = HAND_NEIGHBOUR_CLEARANCE_MM,
) -> Union[PushPlan, PushRefusal]:
    """Plan one push of the failed part, or refuse with a reason.

    ``target_points_mm`` are the part's BASE points and ``neighbour_points_mm`` every other segmented object's,
    merged. ``support_normal`` / ``support_offset_mm`` are the resolved support plane in BASE
    (``dot(normal, p) - offset`` is the height above it). ``workspace`` is the cell's workspace box; its
    ``z_min`` is the floor every TCP point stays 20 mm above. ``table_points_mm`` are the support points the
    camera saw (:func:`table_points_from_cloud`), from every view of the pick when there are several; they are
    not needed when ``container_interior`` is given. ``operator_box``, when given, only narrows both the landing
    box and the region the hand may come down in. ``push_axes_xy`` are the closing-axis directions the wrist may
    take, in BASE XY; each is tried both ways. ``push_distance_mm`` is best settled first with
    :func:`resolve_push_distance`. ``hand_clearance_mm`` is how far the open hand keeps from every neighbour
    point; a value under :data:`HAND_NEIGHBOUR_CLEARANCE_MM` is raised to it (a clearance only narrows), and one
    that is not a finite number of mm raises ``ValueError``.
    """

    clearance = float(hand_clearance_mm)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError(f"hand_clearance_mm must be a non-negative number of mm, got {hand_clearance_mm!r}")
    clearance = max(clearance, HAND_NEIGHBOUR_CLEARANCE_MM)
    distance = float(push_distance_mm)
    if not math.isfinite(distance) or distance <= 0.0:
        return _refuse(REFUSED_PUSH_DISTANCE_INVALID,
                       f"A push distance of {push_distance_mm!r} mm is not a positive number, so no push is planned.")
    if distance > PUSH_DISTANCE_CAP_MM + _EPS_MM:
        return _refuse(REFUSED_PUSH_DISTANCE_ABOVE_CAP,
                       f"A push of {distance:g} mm was asked for and the hard cap is "
                       f"{PUSH_DISTANCE_CAP_MM:g} mm, so no push is planned.")
    if distance < MIN_CLEARANCE_GAIN_MM - _EPS_MM:
        return _refuse(REFUSED_PUSH_DISTANCE_TOO_SHORT,
                       f"A push of {distance:g} mm cannot open the {MIN_CLEARANCE_GAIN_MM:g} mm a finger needs, "
                       "so no push is planned.")

    frame = _SupportFrame.build(support_normal, support_offset_mm)
    if frame.tilt_deg > MAX_SUPPORT_TILT_DEG:
        return _refuse(REFUSED_SUPPORT_TILTED,
                       f"The support tilts {frame.tilt_deg:.1f} deg from level and a push is planned only on a "
                       f"support within {MAX_SUPPORT_TILT_DEG:g} deg of level.")

    raw = _points(target_points_mm, "target_points_mm")
    raw_h = frame.height(raw)
    above = raw_h > SUPPORT_BAND_MM
    target, target_h = raw[above], raw_h[above]
    if len(target) < MIN_TARGET_POINTS:
        return _refuse(REFUSED_NO_TARGET_POINTS,
                       f"The part has {len(target)} points more than {SUPPORT_BAND_MM:g} mm above the support, "
                       "so there is nothing to push.")
    part_base = float(np.percentile(raw_h[raw_h >= -SUPPORT_BAND_MM], PART_BASE_PERCENTILE))
    if part_base > PART_BASE_MAX_MM + _EPS_MM:
        return _refuse(REFUSED_PART_NOT_ON_SUPPORT,
                       f"The part's lowest points seen are {part_base:.0f} mm above the support, more than the "
                       f"{PART_BASE_MAX_MM:g} mm the finger rides at its lowest: it rests on something, or only "
                       "its top was seen, and the finger could pass under it, so it is not pushed.")
    part_top = float(np.percentile(target_h, PART_HEIGHT_PERCENTILE))
    part_height = part_top - max(part_base, 0.0)
    if part_height < MIN_PUSHABLE_PART_HEIGHT_MM:
        return _refuse(REFUSED_PART_TOO_FLAT,
                       f"The part is {part_height:.0f} mm tall and a push needs at least "
                       f"{MIN_PUSHABLE_PART_HEIGHT_MM:g} mm, or the finger rides over it or scrapes the table.")

    target_ab = frame.plane_xy(target)
    target_cells, _ = _cells(target_ab, target_h)
    neighbours = _points(neighbour_points_mm, "neighbour_points_mm")
    neighbour_h = frame.height(neighbours)
    keep = neighbour_h > SUPPORT_BAND_MM
    neighbours, neighbour_h = neighbours[keep], neighbour_h[keep]
    neighbour_ab = frame.plane_xy(neighbours)
    lo_ab = target_ab.min(axis=0) - NEIGHBOUR_WINDOW_MM
    hi_ab = target_ab.max(axis=0) + NEIGHBOUR_WINDOW_MM
    near = np.all((neighbour_ab >= lo_ab) & (neighbour_ab <= hi_ab), axis=1) if len(neighbour_ab) else keep[:0]
    neighbour_cells, neighbour_tops = _cells(neighbour_ab[near], neighbour_h[near])
    gap = _min_distance(target_cells, neighbour_cells)
    if not gap <= NEIGHBOUR_EVIDENCE_RADIUS_MM:
        said = "no neighbour was seen" if math.isinf(gap) else f"the nearest is {gap:.0f} mm away"
        return _refuse(REFUSED_NO_BLOCKING_NEIGHBOUR,
                       f"No neighbour stands within {NEIGHBOUR_EVIDENCE_RADIUS_MM:g} mm of the part ({said}), "
                       "so nothing blocked the fingers and a push would not help.")

    boxes = _push_boxes(workspace=workspace, frame=frame, table_points_mm=table_points_mm, distance=distance,
                        container_interior=container_interior, operator_box=operator_box,
                        target_xy=target[:, :2], neighbour_xy=neighbours[near][:, :2])
    if isinstance(boxes, PushRefusal):
        return boxes
    landing_box, hand_region, seen = boxes

    finger_height = min(FINGER_HEIGHT_MAX_MM, max(FINGER_HEIGHT_MIN_MM, 0.5 * part_height))
    palm_underside = finger_height + hand.palm_underside_above_tip_mm
    if part_top > palm_underside - PALM_PART_CLEARANCE_MM + _EPS_MM:
        return _refuse(REFUSED_PART_REACHES_THE_PALM,
                       f"The part's top is {part_top:.0f} mm above the support and the palm's underside would pass "
                       f"at {palm_underside:.0f} mm, less than {PALM_PART_CLEARANCE_MM:g} mm above it: the housing, "
                       "not the finger, would push it, so it is not pushed.")
    tcp_height = finger_height + hand.fingertip_depth_mm
    centre_ab = target_cells.mean(axis=0)
    contact_z = float(frame.to_base(float(centre_ab[0]), float(centre_ab[1]), tcp_height)[2])
    floor = workspace.min_mm[2] + WORKSPACE_MARGIN_MM
    if contact_z < floor - _EPS_MM:
        return _refuse(REFUSED_BELOW_Z_MIN,
                       f"The push would run the TCP at z = {contact_z:.1f} mm, less than {WORKSPACE_MARGIN_MM:g} mm "
                       f"above the workspace z_min of {workspace.min_mm[2]:g} mm, so no push is planned.")
    top_z = float(frame.to_base(float(centre_ab[0]), float(centre_ab[1]), tcp_height + PUSH_APPROACH_RISE_MM)[2])
    ceiling = workspace.max_mm[2] - WORKSPACE_MARGIN_MM
    if top_z > ceiling + _EPS_MM:
        return _refuse(REFUSED_ABOVE_Z_MAX,
                       f"The push would lift the TCP to z = {top_z:.1f} mm, less than {WORKSPACE_MARGIN_MM:g} mm "
                       f"below the workspace z_max of {workspace.max_mm[2]:g} mm, so no push is planned.")

    directions = _directions(frame, push_axes_xy)
    if not directions:
        return _refuse(REFUSED_NO_PUSH_AXIS, "No push axis was offered, so no push is planned.")

    scene = _Scene(frame=frame, target=target, target_ab=target_ab, target_cells=target_cells,
                   neighbour_cells=neighbour_cells, neighbour_tops=neighbour_tops, centre_ab=centre_ab,
                   part_height=part_height, part_base=part_base, finger_height=finger_height,
                   tcp_height=tcp_height, distance=distance, clearance_before=gap, landing_box=landing_box,
                   hand_region=hand_region, seen=seen, workspace=workspace, hand=hand, hand_clearance=clearance)
    verdicts: list[DirectionVerdict] = []
    best: Optional[tuple[float, np.ndarray, float]] = None
    for direction_ab in directions:
        verdict, after = _judge(scene, direction_ab)
        verdicts.append(verdict)
        if verdict.verdict == DIRECTION_OK and after is not None:
            gain = after - gap
            if best is None or gain > best[0] + _EPS_MM:
                best = (gain, direction_ab, after)
    candidates = tuple(verdicts)
    if best is None:
        counts: dict[str, int] = {}
        for v in candidates:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        summary = ", ".join(f"{n} {name}" for name, n in counts.items())
        return _refuse(REFUSED_NO_FREE_DIRECTION,
                       f"None of the {len(candidates)} push directions is free ({summary}), so the part is not "
                       "pushed.", candidates)
    _, direction_ab, after = best
    return _plan(scene, direction_ab, after, candidates)


def _push_boxes(
    *,
    workspace: AxisBox,
    frame: _SupportFrame,
    table_points_mm: Optional[ArrayLike],
    distance: float,
    container_interior: Optional[AxisBox],
    operator_box: Optional[AxisBox],
    target_xy: np.ndarray,
    neighbour_xy: np.ndarray,
) -> Union[tuple[tuple[Vec2, Vec2], tuple[Vec2, Vec2], Optional[_SeenTable]], PushRefusal]:
    """The landing box and the region the hand may come down in, both in BASE XY, and the seen table's cells
    (``None`` inside a declared container, whose interior is the whole answer)."""

    if container_interior is not None:
        region = _intersect(container_interior.xy, workspace.xy)
        if operator_box is not None:
            region = _intersect(region, operator_box.xy)
        if _empty(region):
            return _refuse(REFUSED_PUSH_BOX_EMPTY,
                           "The declared container's interior does not overlap the workspace, so there is nowhere "
                           "for the part to land.")
        return region, region, None
    table = None if table_points_mm is None else table_points_from_cloud(
        table_points_mm, support_normal=frame.n, support_offset_mm=frame.offset_mm)
    seen = None if table is None else _SeenTable.build(
        table_xy=table[:, :2], standing_xy=(target_xy, neighbour_xy), around_xy=target_xy,
        reach_mm=NEIGHBOUR_WINDOW_MM)
    if table is None or seen is None or seen.table_cells < MIN_SEEN_TABLE_CELLS:
        return _refuse(REFUSED_NO_TABLE_SEEN,
                       "The camera saw too little of the table near the part to bound where it may land, and no "
                       "container is declared, so no push is planned.")
    low = np.percentile(table[:, :2], TABLE_REGION_PERCENTILE, axis=0)
    high = np.percentile(table[:, :2], 100.0 - TABLE_REGION_PERCENTILE, axis=0)
    box: tuple[Vec2, Vec2] = ((float(low[0]), float(low[1])), (float(high[0]), float(high[1])))
    region = _intersect(box, workspace.xy)
    if operator_box is not None:
        region = _intersect(region, operator_box.xy)
    shrink = distance + AUTO_BOX_EXTRA_SHRINK_MM
    landing = _shrink(region, shrink)
    if _empty(region) or _empty(landing):
        return _refuse(REFUSED_PUSH_BOX_EMPTY,
                       f"The table the camera saw inside the workspace, shrunk by {shrink:g} mm, leaves no room for "
                       "the part to land, so no push is planned.")
    return landing, region, seen


def _directions(frame: _SupportFrame, axes: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Each offered axis both ways, as unit vectors in the support plane's coordinates, duplicates dropped."""

    out: list[np.ndarray] = []
    for axis in axes:
        xy = [float(c) for c in axis]
        if len(xy) != 2 or not all(math.isfinite(c) for c in xy):
            raise ValueError(f"a push axis must be two finite numbers in BASE XY, got {axis!r}")
        w = np.array([xy[0], xy[1], 0.0])
        w = w - float(w @ frame.n) * frame.n
        ab = np.array([float(w @ frame.e1), float(w @ frame.e2)])
        norm = float(np.linalg.norm(ab))
        if norm < 1e-9:
            continue
        ab = ab / norm
        for signed in (ab, -ab):
            if not any(float(signed @ seen) > 1.0 - 1e-9 for seen in out):
                out.append(signed)
    return out


def _uv(points_ab: np.ndarray, direction_ab: np.ndarray) -> np.ndarray:
    """Plane coordinates rotated so +u is the push direction and +v is n x direction."""

    da, db = float(direction_ab[0]), float(direction_ab[1])
    return np.column_stack([points_ab[:, 0] * da + points_ab[:, 1] * db, -points_ab[:, 0] * db + points_ab[:, 1] * da])


def _ab(u: float, v: float, direction_ab: np.ndarray) -> tuple[float, float]:
    da, db = float(direction_ab[0]), float(direction_ab[1])
    return u * da - v * db, u * db + v * da


def _direction_xyz(scene: _Scene, direction_ab: np.ndarray) -> np.ndarray:
    return np.asarray(float(direction_ab[0]) * scene.frame.e1 + float(direction_ab[1]) * scene.frame.e2,
                      dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Stations:
    """The push line in (u, v) for one direction: the TCP stations and the hand's swept extent along u, for the
    fingers (their outer faces) and for the housing (the wider of those and half its thickness)."""

    u_trail: float
    v_line: float
    u_start: float
    u_end: float
    u_back: float
    hand_u_low: float
    hand_u_high: float
    palm_u_low: float
    palm_u_high: float


def _stations(scene: _Scene, direction_ab: np.ndarray) -> _Stations:
    target_uv = _uv(scene.target_ab, direction_ab)
    u_trail = float(target_uv[:, 0].min())
    v_line = float(_uv(scene.centre_ab[None, :], direction_ab)[0, 1])
    half_outer = scene.hand.half_outer_mm
    palm_half = scene.hand.palm_half_along_mm
    u_start = u_trail - CONTACT_GAP_MM - half_outer
    u_end = u_start + CONTACT_GAP_MM + scene.distance
    return _Stations(u_trail=u_trail, v_line=v_line, u_start=u_start, u_end=u_end, u_back=u_end - BACK_OFF_MM,
                     hand_u_low=u_start - half_outer, hand_u_high=u_end + half_outer,
                     palm_u_low=u_start - palm_half, palm_u_high=u_end + palm_half)


def _hand_footprint_xy(scene: _Scene, direction_ab: np.ndarray, st: _Stations, band: float) -> np.ndarray:
    """BASE XY samples, half a seen-table cell apart, over the rectangle the hand sweeps at finger height."""

    step = 0.5 * TABLE_CELL_MM
    uu, vv = np.meshgrid(_span(st.palm_u_low, st.palm_u_high, step), _span(-band, band, step), indexing="ij")
    u = uu.reshape(-1)
    v = st.v_line + vv.reshape(-1)
    da, db = float(direction_ab[0]), float(direction_ab[1])
    base = scene.frame.to_base_many(u * da - v * db, u * db + v * da, scene.finger_height)
    return base[:, :2]


def _judge(scene: _Scene, direction_ab: np.ndarray) -> tuple[DirectionVerdict, Optional[float]]:
    """One direction against every test; the clearance after the push when it passes them all."""

    d3 = _direction_xyz(scene, direction_ab)
    xy = np.array([d3[0], d3[1]])
    xy = xy / max(float(np.linalg.norm(xy)), 1e-12)
    label: Vec2 = (float(xy[0]), float(xy[1]))
    st = _stations(scene, direction_ab)
    hand = scene.hand
    clear = scene.hand_clearance

    n_uv = _uv(scene.neighbour_cells, direction_ab)
    lateral = np.abs(n_uv[:, 1] - st.v_line)
    finger_u = (n_uv[:, 0] >= st.hand_u_low - clear) & (n_uv[:, 0] <= st.hand_u_high + clear)
    palm_u = (n_uv[:, 0] >= st.palm_u_low - clear) & (n_uv[:, 0] <= st.palm_u_high + clear)
    fingers = finger_u & (lateral <= 0.5 * hand.finger_width_mm + clear)
    palm_underside = scene.finger_height + hand.palm_underside_above_tip_mm
    palm = palm_u & (lateral <= 0.5 * hand.palm_width_mm + clear) & (scene.neighbour_tops >= palm_underside - clear)
    if bool(np.any(fingers | palm)):
        return DirectionVerdict(label, DIRECTION_HAND_BLOCKED), None

    target_uv = _uv(scene.target_cells, direction_ab)
    ahead = n_uv[n_uv[:, 0] >= st.u_trail - _EPS_MM]
    if _min_swept_distance(target_uv, ahead, scene.distance) < PATH_NEIGHBOUR_CLEARANCE_MM:
        return DirectionVerdict(label, DIRECTION_PATH_BLOCKED), None

    shift_ab = direction_ab * scene.distance
    after = _min_distance(scene.target_cells + shift_ab[None, :], scene.neighbour_cells)
    gain = after - scene.clearance_before
    if gain < MIN_CLEARANCE_GAIN_MM:
        return DirectionVerdict(label, DIRECTION_NO_CLEARANCE_GAIN, gain), None

    landing_xy = (scene.target + scene.distance * d3[None, :])[:, :2]
    if not _inside_xy(landing_xy, scene.landing_box, LANDING_MARGIN_MM):
        return DirectionVerdict(label, DIRECTION_LANDING_OUTSIDE_BOX, gain), None

    band = 0.5 * max(hand.finger_width_mm, hand.palm_width_mm)
    corners = np.array([
        scene.frame.to_base(*_ab(u, st.v_line + v, direction_ab), scene.finger_height)[:2]
        for u in (st.palm_u_low, st.palm_u_high) for v in (-band, band)
    ])
    if not _inside_xy(corners, scene.hand_region):
        return DirectionVerdict(label, DIRECTION_HAND_OUTSIDE_REGION, gain), None

    if scene.seen is not None:
        window = scene.distance + AUTO_BOX_EXTRA_SHRINK_MM + LANDING_MARGIN_MM
        if not scene.seen.windows_seen(landing_xy, window):
            return DirectionVerdict(label, DIRECTION_LANDING_OVER_UNSEEN_TABLE, gain), None
        if not scene.seen.windows_seen(_hand_footprint_xy(scene, direction_ab, st, band), 0.0):
            return DirectionVerdict(label, DIRECTION_HAND_OVER_UNSEEN_TABLE, gain), None

    tcp = _tcp_points(scene, direction_ab, st)
    low = np.asarray(scene.workspace.min_mm) + WORKSPACE_MARGIN_MM - _EPS_MM
    high = np.asarray(scene.workspace.max_mm) - WORKSPACE_MARGIN_MM + _EPS_MM
    if not bool(np.all((tcp >= low) & (tcp <= high))):
        return DirectionVerdict(label, DIRECTION_TCP_OUTSIDE_WORKSPACE, gain), None
    return DirectionVerdict(label, DIRECTION_OK, gain), after


def _tcp_points(scene: _Scene, direction_ab: np.ndarray, st: _Stations) -> np.ndarray:
    """P0, contact start, end, back-off and lift, in that order, as BASE rows."""

    rise = scene.tcp_height + PUSH_APPROACH_RISE_MM
    stations = ((st.u_start, rise), (st.u_start, scene.tcp_height), (st.u_end, scene.tcp_height),
                (st.u_back, scene.tcp_height), (st.u_back, rise))
    return np.array([scene.frame.to_base(*_ab(u, st.v_line, direction_ab), h) for u, h in stations])


def _plan(scene: _Scene, direction_ab: np.ndarray, after: float,
          candidates: tuple[DirectionVerdict, ...]) -> PushPlan:
    st = _stations(scene, direction_ab)
    p0, start, end, back, lift = _tcp_points(scene, direction_ab, st)
    d3 = _direction_xyz(scene, direction_ab)
    mid_height = max(scene.part_base, 0.0) + 0.5 * scene.part_height
    centre = scene.frame.to_base(float(scene.centre_ab[0]), float(scene.centre_ab[1]), mid_height)
    return PushPlan(
        direction=_tuple3(d3),
        approach=_tuple3(-scene.frame.n),
        approach_tcp_mm=_tuple3(p0),
        contact_start_tcp_mm=_tuple3(start),
        end_tcp_mm=_tuple3(end),
        back_off_tcp_mm=_tuple3(back),
        lift_tcp_mm=_tuple3(lift),
        finger_height_mm=float(scene.finger_height),
        part_height_mm=float(scene.part_height),
        part_base_mm=float(scene.part_base),
        push_distance_mm=float(scene.distance),
        target_centre_mm=_tuple3(centre),
        predicted_landing_centre_mm=_tuple3(centre + scene.distance * d3),
        landing_box_xy_mm=scene.landing_box,
        clearance_before_mm=float(scene.clearance_before),
        clearance_after_mm=float(after),
        candidates=candidates,
    )
