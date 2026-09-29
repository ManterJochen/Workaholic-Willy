"""Were both faces a jaw closes on seen, and where would the camera stand to see the one that was not?

A parallel jaw closes on two opposite faces, and one depth view sees one side of a part. The generator
still places a grasp on a one-sided cloud: it infers the far contact from the silhouette and from the
assumption that the part is as deep as it looks wide. That inference is right for a cube and wrong for
anything that is not, and nothing downstream can tell a grasp whose far contact was measured from one
whose far contact was guessed. On a wrist camera the question has a cheap answer, because the arm can
move the camera, and this module supplies both halves of it.

:func:`jaw_faces_seen` answers the first half. It asks, per side, whether enough of the patch the pad
would touch was measured: the contact patch the gripper model already describes (``pad_boxes``), cut
into cells of about 4 mm, clipped to where the part actually is, and counted against the points of the
target cloud that lie on that face within the error a face is measured with and do not belong to a
surface running on through the part between the faces. It needs no normals and no surface
reconstruction, only the grasp and the cloud the generator was given.

:func:`orbit_views` answers the second. The wrist that took the judged look is turned as one rigid body
about the vertical through the target, so the camera keeps its distance, its tilt and its aim, and the
only thing that changes is which side of the part faces it. The candidates are the turns that bring a
missing face within the incidence the camera can measure at, smallest turn first. Whether the arm can
reach a candidate is not asked here: that is the arm's ``nearest_configuration``, before anything
moves.

Pure numpy and deterministic, BASE millimetres throughout. No config, no arm and no logger: every
number arrives as an argument or is a named constant below with its reasoning, and the callers (the
pick loop's judgement of a look and ``Locator.look_around``) log what the answers mean.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.geometry import Frame, Pose

__all__ = [
    "FACE_BAND_MM",
    "FACE_CELL_MIN_POINTS",
    "FACE_CELL_MM",
    "FACE_MIN_CELLS",
    "FACE_MIN_COVERAGE",
    "MAX_INCIDENCE_DEG",
    "ORBIT_MAX_DEG",
    "ORBIT_MAX_MOTIONS",
    "ORBIT_STEP_DEG",
    "SUPPORT_MARGIN_MM",
    "JawFace",
    "JawFacesSeen",
    "OrbitView",
    "away_from_views",
    "jaw_faces_seen",
    "orbit_target",
    "orbit_views",
]

# --------------------------------------------------------------------------------------------------
# Was the face seen?
# --------------------------------------------------------------------------------------------------

#: How far off the face a point may lie along the closing axis and still be read as the face, in mm,
#: either side of it. The owner's 3 to 5 mm of end-to-end error, hand-eye and robot together, plus
#: about 2 mm of D415 depth noise at the 0.45 m a wrist look works from. It is also the height below
#: the part's top within which a cell counts for neither face: the top is measured with the same
#: error, and within it a point on the face and a point on the rim of the top are one measurement.
FACE_BAND_MM = 6.0
#: The side of a patch cell, at most, in mm. About 9 D415 pixels across at 0.45 mm per pixel on a
#: face seen square, about 6 at the 0.65 mm a 45 degree view gives. A cell of a face the camera did
#: measure holds tens of points, so the cell is small enough to say where on the patch the evidence
#: is and large enough that the count in it is not one pixel's luck.
FACE_CELL_MM = 4.0
#: The points a cell needs to count as seen. A measured cell holds tens; three rejects the flying
#: pixels a stereo camera throws off an edge, which arrive alone.
FACE_CELL_MIN_POINTS = 3
#: The share of the touchable cells a face needs seen. A third of the contact area is enough to
#: confirm where the face is and which way it turns, and tolerant of a neighbour hiding part of it.
#: Asked of the touchable cells, not of the whole pad, so a narrow part is judged on the patch it can
#: touch.
FACE_MIN_COVERAGE = 0.30
#: The fewest seen cells a face needs, whatever the share: three cells are the fewest that say where a
#: plane is and which way it faces, and below it a small patch's two cells could be one streak of
#: noise.
FACE_MIN_CELLS = 3
#: How far above the support a point still counts as the support, in mm. The figure the scene plans
#: with, ``grasping.scene.SUPPORT_READ_ERROR_MM``, restated rather than imported so this module stays
#: free of the scene's imports; a test holds the two equal. A table point counted as part would put
#: the far face where the table is.
SUPPORT_MARGIN_MM = 5.0

#: The percentiles the part's extent is read at: the top of the part over the support, and its width
#: across the binormal near the grasp. Two percent at either end is a few hundred points on a wrist
#: cloud, enough that one stray row of flying pixels does not move the extent the patch is clipped to.
_EXTENT_LOW_PCT = 2.0
_EXTENT_HIGH_PCT = 98.0
#: The band never reaches nearer the grasp centre than this, so the bands of the two faces never meet
#: on a thin part and one point never counts for both.
_BAND_GAP_MM = 1.0
#: A surface that runs along the closing axis, through the part, leaves as many points per millimetre
#: of that axis between the two bands as in a band it reaches into, twice as many where the part ends
#: at the face; a face across the axis leaves its points in its band alone. A cell counts no evidence
#: when its points between the bands are at least this share, per millimetre, of its points in the
#: band. Four fifths leaves room for the counting noise of the few points between the bands of a short
#: grip, and stays far over what the tail of a neighbouring side leaves in a cell a face fills.
_THROUGH_SHARE = 0.8
#: Below this length a direction is the zero vector.
_ZERO_MM = 1e-9


@dataclass(frozen=True, slots=True)
class JawFace:
    """One of the two faces a jaw closes on, and how much of the patch its pad would touch was seen.

    ``side`` is +1 for the face the closing axis points to and -1 for the other. ``centre_mm`` is
    where the pad would touch, ``grasp centre + side x width / 2 x closing axis``, and ``normal`` its
    outward normal, ``side x closing axis``, both BASE. ``points`` counts the measured points on the
    patch cells the pad can touch and no surface runs through, the evidence the verdict was reached
    on. ``cells_touchable`` is how many cells the pad can touch, ``cells_seen`` how many of them hold
    enough of that evidence. A cell a surface runs through stays touchable: the pad can still close
    there, and the share a face needs is asked of every cell it could touch.
    """

    side: int
    centre_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    points: int
    cells_seen: int
    cells_touchable: int

    @property
    def cells_needed(self) -> int:
        """The seen cells this face needs: its share of the touchable cells, never under the floor."""
        return max(FACE_MIN_CELLS, math.ceil(FACE_MIN_COVERAGE * self.cells_touchable))

    @property
    def seen(self) -> bool:
        """Whether this face was measured. A face with no cell the pad can touch never was."""
        return self.cells_touchable > 0 and self.cells_seen >= self.cells_needed


@dataclass(frozen=True, slots=True)
class JawFacesSeen:
    """Both faces of one grasp, and whether each was measured."""

    negative: JawFace
    positive: JawFace

    @property
    def both(self) -> bool:
        return self.negative.seen and self.positive.seen

    @property
    def missing(self) -> tuple[JawFace, ...]:
        """The faces not seen, negative first. What a generated view has to show."""
        return tuple(face for face in (self.negative, self.positive) if not face.seen)

    def as_pair(self) -> tuple[bool, bool]:
        """(negative seen, positive seen), the shape a report carries."""
        return self.negative.seen, self.positive.seen


def _vec3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return vector


def _unit(value: Any, name: str) -> np.ndarray:
    vector = _vec3(value, name)
    length = float(np.linalg.norm(vector))
    if length < _ZERO_MM:
        raise ValueError(f"{name} is the zero vector and names no direction")
    return vector / length


def _length(value: float, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be a finite number {bound}, got {value!r}")
    return number


def _cloud(points_base_mm: Any) -> np.ndarray:
    """An ``(N, 3)`` float cloud with every row that is not finite dropped. Empty stays empty."""
    cloud = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    return cloud[np.all(np.isfinite(cloud), axis=1)]


def _grasp_frame(approach: Any, closing_axis: Any) -> np.ndarray:
    """The grasp frame's axes as columns: X the closing axis, Z the approach, Y = Z x X.

    The closing axis is made perpendicular to the approach rather than trusted to be, so a grasp
    that carries both to three decimals is not judged in a frame that is slightly sheared. A closing
    axis along the approach names no pair of faces and is refused.
    """
    z = _unit(approach, "approach")
    x = _vec3(closing_axis, "closing_axis")
    x = x - float(x @ z) * z
    length = float(np.linalg.norm(x))
    if length < 1e-6:
        raise ValueError("closing_axis runs along the approach, so it names no pair of faces")
    x = x / length
    return np.column_stack([x, np.cross(z, x), z])


def jaw_faces_seen(
    points_base_mm: np.ndarray,
    *,
    centre_mm: Any,
    approach: Any,
    closing_axis: Any,
    grip_width_mm: float,
    pad_ahead_mm: float,
    pad_behind_mm: float,
    finger_width_mm: float,
    support_height_mm: float | None = None,
) -> JawFacesSeen:
    """Whether the target cloud holds both faces this grasp's pads would close on.

    ``points_base_mm`` is the cloud the generator was given for the target, fused or single-view.
    ``centre_mm``, ``approach``, ``closing_axis`` and ``grip_width_mm`` are the grasp's, BASE, with
    the width the expected pad separation at contact. ``pad_ahead_mm``, ``pad_behind_mm`` and
    ``finger_width_mm`` are the hand's contact patch, the numbers its registry file and its gripper
    model carry. ``support_height_mm`` is the BASE height of what the part stands on, or ``None``
    when it is not known, in which case nothing is ruled out as the table.

    In the grasp frame (X the closing axis, Z the approach, Y = Z x X) the patch is the box the
    gripper model's ``pad_boxes`` describes: Y within half the finger width, Z from ``-pad_behind``
    to ``pad_ahead``. It is cut into the fewest equal cells no larger than :data:`FACE_CELL_MM` a
    side, 8 x 6 for the Hand-E. For each side ``s`` a cell, placed on the face at
    ``centre + s x width / 2 x X``, is touchable when all of these hold:

    * it stands at least :data:`SUPPORT_MARGIN_MM` over the support, where the support is known;
    * it lies wholly at least :data:`FACE_BAND_MM` below the part's top, the high percentile of the
      part's heights, so the rim of the top is never read as the face below it;
    * its centre lies across the binormal within the part's width near the grasp, the percentile
      span of Y over the points within a band of either face.

    That clips the patch to where the part actually is using only what every view measures well,
    its top and its width across the binormal, and never its extent along the closing axis, which is
    the very thing one view cannot see. The points below the support margin are dropped before any
    of it is read: they are the table, and a mask that bleeds onto the table would otherwise widen
    the part. A point counts for side ``s`` when ``s x X`` lies within the band about the face,
    reaching :data:`FACE_BAND_MM` outward and as far inward, but never nearer the centre than a
    millimetre, and inside the patch in Y and Z.

    A band point is not yet the face. A part narrower than the pad shows a look from off the closing
    axis one of its sides, and that side runs the length of the part: its strip within the band of
    the far face lies in the edge column of the patch on every row, where the far pad would close, and
    one look, or two from the same side, would read a face no camera faced. A margin off the part's
    width does not separate the two: the depth noise spreads the side across it, and a second look's
    misalignment puts a second copy inside it. What does is that the side runs *through* the part,
    between the two bands, where a jaw face never does. So every cell also counts the points between
    the bands at its place on the patch, and its band points are no evidence when those are, per
    millimetre along the closing axis, at least :data:`_THROUGH_SHARE` of its band points: the side of
    a narrow part, the rim of a top, the wall of a slot. The comparison is of densities, so it holds
    however noisy the side and however many looks saw it, and a face that fills a cell keeps it
    against the tail of a side that reaches in.

    A cell is seen when it is touchable and holds at least :data:`FACE_CELL_MIN_POINTS` points that
    are evidence; a face is seen when it has touchable cells and at least
    :attr:`JawFace.cells_needed` of them are seen.

    When nothing of the part is left to judge by, an empty cloud or one that is all table, no cell
    is touchable and neither face is seen: a verdict of "not measured" is the fail-closed one.
    """
    centre = _vec3(centre_mm, "centre_mm")
    frame = _grasp_frame(approach, closing_axis)
    half_width = 0.5 * _length(grip_width_mm, "grip_width_mm")
    ahead = _length(pad_ahead_mm, "pad_ahead_mm")
    behind = _length(pad_behind_mm, "pad_behind_mm")
    half_finger = 0.5 * _length(finger_width_mm, "finger_width_mm", positive=True)
    if ahead + behind <= 0.0:
        raise ValueError("the contact patch has no length: pad_ahead_mm + pad_behind_mm is 0")

    cloud = _cloud(points_base_mm)
    floor: float | None = None
    if support_height_mm is not None:
        floor = float(support_height_mm) + SUPPORT_MARGIN_MM
        cloud = cloud[cloud[:, 2] >= floor]
    local = (cloud - centre) @ frame

    count_y = max(1, math.ceil(2.0 * half_finger / FACE_CELL_MM))
    count_z = max(1, math.ceil((ahead + behind) / FACE_CELL_MM))
    cell_y = 2.0 * half_finger / count_y
    cell_z = (ahead + behind) / count_z
    centres_y = -half_finger + cell_y * (np.arange(count_y) + 0.5)
    centres_z = -behind + cell_z * (np.arange(count_z) + 0.5)
    x_axis, y_axis, z_axis = frame[:, 0], frame[:, 1], frame[:, 2]
    # How far a cell reaches up or down from its centre in BASE, for a patch tilted any way.
    cell_rise = 0.5 * (abs(cell_y * y_axis[2]) + abs(cell_z * z_axis[2]))

    near = local[np.abs(local[:, 0]) <= half_width + FACE_BAND_MM]
    if len(near) == 0:
        across = np.zeros(count_y, dtype=bool)
        ceiling = -math.inf
    else:
        low, high = np.percentile(near[:, 1], [_EXTENT_LOW_PCT, _EXTENT_HIGH_PCT])
        across = (centres_y >= low) & (centres_y <= high)
        ceiling = float(np.percentile(cloud[:, 2], _EXTENT_HIGH_PCT)) - FACE_BAND_MM

    inward = max(0.0, min(FACE_BAND_MM, half_width - _BAND_GAP_MM))
    in_patch = (
        (np.abs(local[:, 1]) <= half_finger) & (local[:, 2] >= -behind) & (local[:, 2] <= ahead)
    )
    index_y = np.floor((local[:, 1] + half_finger) / cell_y).astype(np.int64)
    index_z = np.floor((local[:, 2] + behind) / cell_z).astype(np.int64)
    index_y = np.clip(index_y, 0, count_y - 1)
    index_z = np.clip(index_z, 0, count_z - 1)
    # What lies between the two bands, per cell, and how long each stretch of the closing axis is: the
    # comparison is per millimetre, so a short grip's two millimetres between the bands weigh as much
    # as a long grip's twenty-eight.
    between = in_patch & (np.abs(local[:, 0]) < half_width - inward)
    through = np.zeros((count_y, count_z), dtype=np.int64)
    np.add.at(through, (index_y[between], index_z[between]), 1)
    through_length = 2.0 * (half_width - inward)
    band_length = inward + FACE_BAND_MM

    def face(side: int) -> JawFace:
        on_face = centre + side * half_width * x_axis
        heights = on_face[2] + centres_y[:, None] * y_axis[2] + centres_z[None, :] * z_axis[2]
        touchable = across[:, None] & (heights + cell_rise <= ceiling)
        if floor is not None:
            touchable &= heights >= floor
        reach = side * local[:, 0]
        band = in_patch & (reach >= half_width - inward) & (reach <= half_width + FACE_BAND_MM)
        counts = np.zeros((count_y, count_z), dtype=np.int64)
        np.add.at(counts, (index_y[band], index_z[band]), 1)
        crossed = (through > 0) & (through * band_length >= _THROUGH_SHARE * counts * through_length)
        counts = np.where(touchable & ~crossed, counts, 0)
        normal = side * x_axis
        return JawFace(
            side=side,
            centre_mm=(float(on_face[0]), float(on_face[1]), float(on_face[2])),
            normal=(float(normal[0]), float(normal[1]), float(normal[2])),
            points=int(counts.sum()),
            cells_seen=int(np.count_nonzero(counts >= FACE_CELL_MIN_POINTS)),
            cells_touchable=int(np.count_nonzero(touchable)),
        )

    return JawFacesSeen(negative=face(-1), positive=face(1))


# --------------------------------------------------------------------------------------------------
# Where would the camera have to stand?
# --------------------------------------------------------------------------------------------------

#: The steepest incidence at which a face counts as shown, in degrees from its normal. About where the
#: D415's active stereo stops filling a surface: past it the projected pattern stretches across the
#: face and the depth fill falls away, so a view that shows a face more obliquely than this does not
#: measure it. It must be met at the look's own elevation, since the orbit keeps it: seen from 45
#: degrees up a vertical face comes within 60 degrees only within 45 degrees of its own normal, and a
#: look steeper than 60 degrees can show no vertical face at all.
MAX_INCIDENCE_DEG = 60.0
#: The turn between candidates, in degrees. Fine enough that the first candidate is within 5 degrees
#: of the least turn that shows the face, coarse enough that the 48 candidates of a full sweep are
#: cheap to screen for reach before anything moves.
ORBIT_STEP_DEG = 5.0
#: The largest turn either way, in degrees: half a turn, the owner's decision of 2026-09-29. The face
#: directly behind a 45 degree look needs 135 degrees (the incidence limit leaves 45 degrees either
#: side of its normal), and a generated view exists to show the contact face no look showed, so the
#: bound reaches it. The wrist turns about the vertical through the part and the base joint turns by
#: about as much, so a large turn is a large motion: the safeguards past 120 degrees are the caller's
#: (``src/robot/execution/generated_view.py``: every candidate screened by the arm before it moves, a
#: travel cap per joint, the cable window about home, a warning, the straight joint line only). Past
#: half a turn a turn is the other way round's, which is why a larger bound is refused.
ORBIT_MAX_DEG = 180.0
#: The most pre-screened candidates a pick moves toward before it gives up on a generated view. Each
#: runs on its straight joint line alone and is given up when the line is not clear, so three bound
#: how long a pick spends on one view it can do without, and how often the arm is asked to go far.
ORBIT_MAX_MOTIONS = 3

#: A face whose normal lies within this of the vertical is a top or a bottom face to the orbit.
#: Turning about the vertical moves its incidence by no more than this either side of what the look's
#: elevation gives it, so a top face is shown by the look as well as by any turn of it and a bottom
#: face by none. One orbit step: less than a step of change is no reason to move the arm.
_VERTICAL_FACE_DEG = 5.0
#: Below this horizontal distance a camera stands over the target and names no bearing.
_BEARING_MIN_MM = 1e-6
#: The largest bound a sweep may be asked for, in degrees. Plus and minus half a turn are one pose,
#: and a turn past it is the other way round's turn, so a bound past it would offer turns twice.
_HALF_TURN_DEG = 180.0
#: How far a cosine may fall short of the incidence limit and still meet it: a face exactly at the
#: limit is at it, not beyond it by the last bit of a product of cosines.
_COS_SLACK = 1e-9
#: How far a bound may fall short of a whole number of steps and still reach the last one: 0.3 / 0.1
#: is 2.9999999999999996, which would drop the third step of a sweep asked to reach 0.3.
_STEP_SLACK = 1e-9


def orbit_target(points_base_mm: np.ndarray, grasp_centre_mm: Any = None) -> np.ndarray:
    """The point the wrist turns about, BASE mm.

    The grasp's centre when there is a grasp: it lies between the two contact faces, so turning
    about it keeps both in view. Otherwise the middle of the target cloud: across the table, halfway
    between its 5th and 95th percentiles, and in height its median. A cloud from one side is dense on
    that side, so its median leans toward the side the looks already saw; the midpoint of the
    percentile span does not, and a few stray points move it no more than they move the percentiles.
    """
    if grasp_centre_mm is not None:
        return _vec3(grasp_centre_mm, "grasp_centre_mm").copy()
    cloud = _cloud(points_base_mm)
    if len(cloud) == 0:
        raise ValueError("orbit_target: no grasp centre and no finite point to find a middle by")
    low, high = np.percentile(cloud[:, :2], [5.0, 95.0], axis=0)
    middle = 0.5 * (low + high)
    return np.array([float(middle[0]), float(middle[1]), float(np.median(cloud[:, 2]))])


def away_from_views(target_mm: Any, camera_centres_mm: Sequence[Any]) -> np.ndarray:
    """The horizontal direction from the target that no look so far has faced, as a unit vector.

    With no grasp there is no face to show, so the side to look at is the one furthest from every
    look: the bisector of the widest gap between the looks' bearings as seen from the target, which
    for one look is its opposite. Looks from one bearing, at another height or tilt or the same pose
    twice, face one side and are one bearing here: the gap between them is no gap. A tie between
    gaps goes to the one that starts at the lowest bearing, so the answer does not hang on the order
    the looks came in. A camera straight over the target names no bearing and is left out; when no
    camera names one, every side is equally unseen and the question has no answer, which is refused.
    """
    target = _vec3(target_mm, "target_mm")
    bearings = []
    for centre in camera_centres_mm:
        offset = _vec3(centre, "camera centre") - target
        if math.hypot(float(offset[0]), float(offset[1])) >= _BEARING_MIN_MM:
            bearings.append(math.atan2(float(offset[1]), float(offset[0])) % math.tau)
    if not bearings:
        raise ValueError("away_from_views: no camera stands to one side of the target")
    # Equal bearings are one. Two a rounding apart leave a gap that is never the widest, so only the
    # equal ones need folding.
    distinct = sorted(set(bearings))
    if len(distinct) == 1:
        heading = distinct[0] + math.pi
    else:
        gaps = [(distinct[(i + 1) % len(distinct)] - distinct[i]) % math.tau
                for i in range(len(distinct))]
        widest = max(range(len(gaps)), key=lambda i: (gaps[i], -i))   # a tie goes to the first
        heading = distinct[widest] + 0.5 * gaps[widest]
    return np.array([math.cos(heading), math.sin(heading), 0.0])


@dataclass(frozen=True, slots=True, eq=False)
class OrbitView:
    """One candidate: the judged look turned by ``azimuth_deg`` about the vertical through the target.

    ``tool_to_base_mm`` and ``camera_to_base_mm`` are 4x4 BASE transforms of the turned tool and
    camera, read-only. ``incidence_deg`` is the squarest incidence at which it shows a missing face.
    Compared by identity: it carries arrays, and two candidates are the same only as one object.
    """

    azimuth_deg: float
    tool_to_base_mm: np.ndarray
    camera_to_base_mm: np.ndarray
    incidence_deg: float

    def __post_init__(self) -> None:
        for name in ("tool_to_base_mm", "camera_to_base_mm"):
            matrix = np.array(getattr(self, name), dtype=np.float64)
            matrix.setflags(write=False)
            object.__setattr__(self, name, matrix)

    @property
    def label(self) -> str:
        return f"orbit {self.azimuth_deg:+g} deg"

    def pose(self) -> Pose:
        """The TCP pose this view is taken from, BASE, labelled with its turn."""
        return Pose.from_matrix(self.tool_to_base_mm, frame=Frame.BASE, label=self.label)


def _rigid(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return matrix


def _orbit(target: np.ndarray, azimuth_deg: float) -> np.ndarray:
    """``Tr(t) . Rz(phi) . Tr(-t)``: a turn by ``phi`` about the vertical through ``t``."""
    c, s = math.cos(math.radians(azimuth_deg)), math.sin(math.radians(azimuth_deg))
    turn = np.eye(4)
    turn[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    turn[:3, 3] = target - turn[:3, :3] @ target
    return turn


def orbit_views(
    *,
    target_mm: Any,
    camera_to_base_mm: Any,
    tool_to_base_mm: Any,
    faces: Sequence[tuple[Any, Any]],
    intrinsics: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    max_incidence_deg: float = MAX_INCIDENCE_DEG,
    step_deg: float = ORBIT_STEP_DEG,
    max_deg: float = ORBIT_MAX_DEG,
) -> tuple[OrbitView, ...]:
    """The turns of the judged look that would show a missing face, smallest turn first.

    ``camera_to_base_mm`` is the judged frame's camera pose and ``tool_to_base_mm`` the TCP stamped
    at its shutter; the turn ``M(phi) = Tr(t) . Rz(phi) . Tr(-t)`` about the vertical through
    ``target_mm`` (:func:`orbit_target`) is applied to both, so camera and tool stay one rigid body
    and the hand-eye needs no restating. By construction the camera keeps its distance to the target,
    its tilt, and the target at the same pixel.

    ``faces`` are ``(centre, outward normal)`` pairs, BASE: each missing :class:`JawFace` as
    ``(face.centre_mm, face.normal)``, or with no grasp one face at the target facing
    :func:`away_from_views`. A face within :data:`_VERTICAL_FACE_DEG` of vertical is never shown:
    the orbit cannot turn it.

    The candidates are ``+-step, +-2 step, ... +-max_deg``, with ``max_deg`` at most half a turn:
    plus and minus 180 degrees are one pose, offered once as the positive turn, and a bound past
    half a turn is refused. One is kept when some face is shown by
    it: in front of the turned camera, at an incidence no steeper than ``max_incidence_deg``, and,
    given ``intrinsics`` (3x3) and ``image_size`` (width, height) of the judged frame, projecting
    inside its image. They are ordered by the size of the turn, then by incidence, squarest first,
    then the positive turn first, so the order is total. An empty tuple means no turn within
    ``max_deg`` shows any face: the look is steeper than the incidence limit, the faces are tops and
    bottoms, or the face lies further round than the bound.
    """
    target = _vec3(target_mm, "target_mm")
    camera0 = _rigid(camera_to_base_mm, "camera_to_base_mm")
    tool0 = _rigid(tool_to_base_mm, "tool_to_base_mm")
    if (intrinsics is None) != (image_size is None):
        raise ValueError("orbit_views: intrinsics and image_size come together, lens and image")
    lens: np.ndarray | None = None
    width = height = 0.0
    if intrinsics is not None and image_size is not None:
        lens = np.asarray(intrinsics, dtype=np.float64)
        if lens.shape != (3, 3) or not np.all(np.isfinite(lens)):
            raise ValueError("orbit_views: intrinsics must be a finite 3x3 lens matrix")
        width, height = (_length(size, "image_size", positive=True) for size in image_size)
    limit = float(max_incidence_deg)
    if not 0.0 < limit <= 90.0:
        raise ValueError(f"orbit_views: max_incidence_deg must lie in (0, 90], got {limit!r}")
    step = _length(step_deg, "step_deg", positive=True)
    reach = _length(max_deg, "max_deg")
    if reach > _HALF_TURN_DEG:
        raise ValueError(f"orbit_views: max_deg must be at most {_HALF_TURN_DEG:g}, half a turn, "
                         f"got {max_deg!r}")

    horizontal = math.sin(math.radians(_VERTICAL_FACE_DEG))
    shown: list[tuple[np.ndarray, np.ndarray]] = []
    for centre, normal in faces:
        unit = _unit(normal, "face normal")
        if math.hypot(float(unit[0]), float(unit[1])) >= horizontal:
            shown.append((_vec3(centre, "face centre"), unit))
    if not shown:
        return ()

    cos_limit = math.cos(math.radians(limit)) - _COS_SLACK
    views: list[OrbitView] = []
    for k in range(1, int(math.floor(reach / step + _STEP_SLACK)) + 1):
        for sign in (1.0, -1.0):
            azimuth = sign * k * step
            if sign < 0.0 and k * step >= _HALF_TURN_DEG - _STEP_SLACK:
                continue   # -180 is +180, already asked
            turn = _orbit(target, azimuth)
            camera = turn @ camera0
            to_camera = np.linalg.inv(camera)
            squarest: float | None = None
            for centre, normal in shown:
                sight = camera[:3, 3] - centre
                distance = float(np.linalg.norm(sight))
                if distance < _ZERO_MM:
                    continue
                cosine = float(normal @ sight) / distance
                if cosine < cos_limit:
                    continue
                local = to_camera[:3, :3] @ centre + to_camera[:3, 3]
                if local[2] <= 0.0:
                    continue
                if lens is not None:
                    u, v, w = lens @ (local / local[2])
                    if not (0.0 <= u / w < width and 0.0 <= v / w < height):
                        continue
                incidence = math.degrees(math.acos(min(1.0, cosine)))
                squarest = incidence if squarest is None else min(squarest, incidence)
            if squarest is not None:
                views.append(OrbitView(azimuth, turn @ tool0, camera, squarest))
    views.sort(key=lambda view: (abs(view.azimuth_deg), view.incidence_deg, -view.azimuth_deg))
    return tuple(views)
