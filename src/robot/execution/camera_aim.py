"""Aim a wrist camera, not the tool: where the camera sits on the tool, and tool poses that point it at a marker.

An eye-in-hand sweep has to keep its target in the camera's view from every station, and a camera beside the hand,
tilted away from the tool axis, does not look where the tool points. ``Pose.aimed_at`` aims the tool's +Z; this
module aims the camera's optical axis, from an estimate of where the camera sits on the tool that the camera itself
provides, before anything is calibrated:

* :func:`estimate_camera_in_tool`: the camera's pose in the TOOL frame from ONE view of a marker lying flat, face up,
  at a known base position, and the TCP the arm reported at the shutter.
* :func:`refine_camera_in_tool`: the same from every view so far, once a second one exists; this is what a sweep
  does from its second view on.
* :func:`nominal_camera_in_tool`: a mount stated by hand (a tilt toward a named tool axis and an offset), for a
  caller who wants a sweep aimed although the first look saw no marker. Nothing here guesses one.
* :func:`aim_camera`: the tool pose that puts the camera at a point looking straight at the marker, turned as little
  as it can be from ``Pose.tool_down`` with a caller's ``closing_axis``, so the tool keeps its heading and only tilts;
  then, where the view asks for it, the camera rolled about its own line of sight.
* :func:`aimed_stations`: a cone of such views round the marker, each rolled by -45, 0 or +45 degrees so the solve
  sees the camera turn about its own axis (:data:`DEFAULT_VIEWS`), screened (workspace box, the arm's reach and its
  joint window) and ordered by joint travel. A roll the screen refuses gives way to the view unrolled, and so does one
  the arm's planning refuses (:meth:`AimedSweep.unrolled`). :class:`AimedSweep` rebuilds the stations not yet visited
  each time a view refines the estimate; ``CalibrationRoutine.run_aimed`` drives it.
* :func:`choose_heading`: the heading a sweep or a viewing pose keeps. A camera tilted toward the tool's -X under a
  heading written for one tilted toward +X stands every view beyond the marker and outside the box; where the heading
  asked for admits too few views, the one of :data:`HEADINGS` that admits the most is kept, and said
  (:class:`HeadingChoice`).
* :func:`viewing_pose`: after the calibration, where a wrist camera looks at the work from before anything is
  located, aimed from the CALIBRATED camera-in-tool and screened as a station is; a fixed camera needs none
  (:class:`ViewingPose`).

What this is NOT: a calibration. The estimate is good enough to aim (a few degrees, see below), and the sweep's
AX=XB solve is what measures the camera. Nor is an aimed pose a judged one: every station still goes to
``RobotArm.move``, which plans and judges it as any move, and the screen here only drops stations the arm would refuse
anyway, so they are reported with their reason instead of being tried.

Frames. BASE is the robot's base, TOOL the frame ``get_tcp_pose`` reports (the grasp centre on a cell with a declared
tool frame), CAMERA the colour optical frame (+Z along the optical axis, +X right, +Y down in the image), MARKER the
frame the pose estimator reports (+Z out of the printed face, towards a camera that sees it). A "camera in tool" is the
4x4 CAMERA->TOOL, millimetres: the camera's pose in the tool frame, the transform an eye-in-hand solve produces.

Accuracy of one view, and why it is one view's. A view fixes the camera's pose relative to the marker, and the marker
is known in BASE up to its rotation about its own normal, so every camera pose on the circle round the marker's
vertical axis explains the view equally well. The estimate takes the point of that circle nearest where the camera is
first taken to stand (the flange, by default): two direction pairs, the marker's normal (base +Z) and the line of
sight to the marker's centre, fix the rotation, and the view then fixes the position. Refining that from the same view
changes nothing, because the position the view gives puts the line of sight exactly where the rotation already has it;
a second view from elsewhere is what fixes the circle, which :func:`refine_camera_in_tool` does. So, from one view:

* the turn about the vertical is off by about ``atan(d / r)``, ``d`` the camera's offset from the flange across the line
  from the flange to the marker (seen from above), ``r`` the camera's horizontal distance from the marker (about 350 mm
  at 500 mm and 45 degrees). A camera whose bracket runs the way it looks, as on a D415 tilted out beside a Hand-E,
  has ``d`` near zero; one 80 mm to the side of that line is off by 11 to 13 degrees.
* the tilt is the marker normal's error, a degree or two for a 150 mm marker at 500 mm seen obliquely (the planar
  pose's flip ambiguity grows toward a frontal view, so a view within :data:`MIN_OBLIQUE_DEG` of frontal is refused);
* the position is off by about ``d``, plus the error of the marker position as measured, millimetre for millimetre.

What an aimed station inherits is much less than that. The estimate's error is one the look cannot see: a camera
turned about the marker's vertical and moved round it to match, which explains the look exactly, so a station near the
look is aimed almost as well as the look itself, and the miss grows with how far the station's tool is turned from the
look's. Measured on synthetic geometry on 2026-09-23 (scratch probe, 20 draws per case, each view's marker pose
disturbed by 1.5 degrees and 3 mm, the 17 rolled default views at 500 mm, closing axis -y): a camera 70 mm out along its
view missed the marker by at most 1.1 degrees from one view; 80 mm to the side, 1.2 degrees (its estimate itself 11
degrees and 78 mm off); 120 mm to the side, 3.2; 80 mm to the side seen from a look yawed 30 degrees, 3.6; with the
marker stated 28 mm from where it lies, 3.7 (1.9 with the heading kept: a roll turns the camera about a line of sight
aimed at the marker as stated). After one more view, at most 1.2 degrees in every case (1.0 with the heading kept).
The D415's colour image is 69 by 42 degrees, so the marker stays well inside it: it left the image at none of the 340
stations of each case.

Measured before the views were rolled, on the first set of 11 views (5 at the ring's elevation, 3 each 15 degrees above
and below it, azimuths up to 40 degrees, the heading kept at each): run through ``CalibrationRoutine.run_aimed`` with
the real UR10 driver and the real cuRobo sidecar (2026-09-23, scratch probe; Hand-E on a 20 mm plate, the half-turn
window about a home at the first look, a simulated controller and a synthetic camera on the owner's geometry), 9 of
those 11 ran from an elbow-up home and 10 from an elbow-down one; the rest, lower-ring views, were refused by the arm's
own planning and gates, and reported. :data:`DEFAULT_VIEWS` says what the 17 that replaced them counted on the same
sidecar, with the heading kept and rolled.

Driven through the same driver against the CB3 URSim at 127.0.0.1 (PolyScope 3.15.8, 2026-09-23, scratch probe; the
Hand-E as ``jaw_io`` on tool DO0, the stated 45-degree mount, an elbow-up home facing the work, no camera, so nothing
was captured), 8 of that first set of 11 moved, each as 2 ``moveJ``; no joint sample left the half-turn window and no
move turned a joint more than 113 degrees. The 3 views above the ring were refused by the arm with nothing moved (the
nearest configuration folds the forearm onto wrist 2 inside the planner's margin). The viewing pose then moved from
where the sweep ended (64 degrees at most) and from home (6 degrees). The 17 views, rolled or not, have not been driven
against URSim.

Pure numpy apart from :func:`reach_of`, which reads a connected arm best effort, and :func:`viewing_pose`, which
reads a camera's calibration and an arm the same way. Nothing here moves anything.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.pose import CLOSING_AXES
from src.robot.execution.pose_provider import SkippedStation

__all__ = [
    "AIM_DISTANCE_RANGE_MM",
    "DEFAULT_VIEWS",
    "HEADINGS",
    "MIN_OBLIQUE_DEG",
    "AimedStation",
    "AimedSweep",
    "ArmReach",
    "HeadingChoice",
    "Look",
    "MarkerAim",
    "MountEstimate",
    "NotAimed",
    "VIEWING_VIEWS",
    "View",
    "ViewingPose",
    "aim_camera",
    "aimed_stations",
    "choose_heading",
    "estimate_camera_in_tool",
    "flange_in_tool_mm",
    "nominal_camera_in_tool",
    "reach_of",
    "refine_camera_in_tool",
    "viewing_pose",
]

logger = logging.getLogger(__name__)

#: The views of an aimed sweep round its marker, as (elevation, azimuth, roll) in degrees. The elevation and azimuth
#: are offsets from the view the camera has with the tool pointing straight down; the roll then turns the camera about
#: its own line of sight, right-handed about the way it looks, so the marker stays where it is in the image and the
#: image turns about it. The places are a ring of seven at that elevation (-60 to +60), five 20 degrees higher, three
#: 12 degrees lower and two steep ones 30 degrees higher; the rolls run 0, +45, -45 in the order written, so the
#: camera's own view with the tool down keeps the heading.
#: The places were MEASURED 2026-09-23 on the real cuRobo sidecar and UR10 driver with a synthetic D415 carrying
#: PnP-like noise (the owner's marker at (-130, -700, 50), camera 45 degrees toward the tool +X, 20 sweeps per set),
#: the heading kept at every view: these 17 counted 11.7 and 15.0 samples on average from two homes against 8.3 and
#: 9.0 for the earlier 11 (5 at 0, 3 at +-15), and the solve's rotation error fell from 1.1 and 1.0 degrees to 0.8 and
#: 0.6; no configuration left the half-turn window.
#: The rolls came after review 4 (2026-09-23): with the heading kept, every turn between two views is about an axis
#: across the line of sight, and AX=XB reads the camera's turn about that line from almost nothing (over every pair of
#: the 17, the weakest axis of the relative turns carried 1/70 of the strongest, 6 degrees from the optical axis).
#: MEASURED on synthetic geometry, not on the sidecar (scratch probe; the UR10 box less 20 mm, its reach and a
#: half-turn window about an elbow-up home at the first look; each view's marker pose disturbed by 0.5 degrees and
#: 1 mm, depth by 0.4 %; 40 seeds): a point 500 mm down the optical axis landed 5.0 mm off at the median (7.8 at the
#: 90th percentile) with the heading kept and 2.0 (4.7) rolled, and the rotation error fell from 0.56 to 0.31 degrees.
#: For a camera tilted toward each of the four tool axes, all 17 rolled views passed that screen under the heading that
#: admits them, as the heading-kept ones did; the tool stands at most 41 degrees from straight down instead of 31.
DEFAULT_VIEWS: tuple[tuple[float, float, float], ...] = (
    (0.0, -60.0, 0.0), (0.0, -40.0, 45.0), (0.0, -20.0, -45.0), (0.0, 0.0, 0.0), (0.0, 20.0, 45.0),
    (0.0, 40.0, -45.0), (0.0, 60.0, 0.0),
    (20.0, -50.0, 45.0), (20.0, -25.0, -45.0), (20.0, 0.0, 0.0), (20.0, 25.0, 45.0), (20.0, 50.0, -45.0),
    (-12.0, -35.0, 0.0), (-12.0, 0.0, 45.0), (-12.0, 35.0, -45.0),
    (30.0, -20.0, 0.0), (30.0, 20.0, 45.0),
)
#: The headings an aimed sweep and a viewing pose fall back to when the one asked for admits too few views, in the
#: order a tie between them is settled; the one asked for wins any tie (:func:`choose_heading`).
HEADINGS: tuple[str, ...] = ("-y", "y", "x", "-x")
#: How far from the marker an aimed camera may be asked to stand. Below it a marker fills the frame; above it a
#: marker of a usual size is a few pixels across.
AIM_DISTANCE_RANGE_MM = (150.0, 2000.0)
#: The narrowest angle between the marker's normal and the line of sight a first look is estimated from. A planar
#: marker seen nearly frontally has an ambiguous normal, and the line of sight then names no direction round it.
MIN_OBLIQUE_DEG = 10.0
#: The narrowest horizontal angle, seen from the marker, between straight down and where the camera is first taken to
#: stand, in degrees. Nearer the vertical the first look cannot tell which way round the marker the camera stands.
_MIN_PRIOR_ELEVATION_OFF_VERTICAL_DEG = 10.0
#: Where the elevation of the cone's middle ring is kept, degrees below the horizontal: a camera that looks straight
#: down with the tool down is aimed at 65 degrees, one that looks nearly level at 30.
_BASE_ELEVATION_RANGE_DEG = (30.0, 65.0)
#: The elevations a view may have at all, degrees below the horizontal.
_ELEVATION_RANGE_DEG = (10.0, 85.0)
#: Below this cosine of its elevation the camera's view with the tool down names no azimuth, and the cone faces away
#: from the base instead.
_MIN_AZIMUTH_COS = 0.2
#: The joints of a UR, for a window an ``ArmReach`` leaves open.
_UR_JOINTS = 6

#: The weights of a refinement: how far a view's marker position and rotation may miss, the marker position as
#: measured, and where the camera was first taken to stand (read only with one view).
_SIGMA_VIEW_MM = 3.0
_SIGMA_VIEW_RAD = math.radians(2.0)
_SIGMA_MARKER_MM = 100.0
_SIGMA_PRIOR_MM = 250.0


class NotAimed(RuntimeError):
    """A sweep could not be aimed, and nothing moved: the first look saw no marker, or saw it from where no aim can
    be estimated. The message says what to do."""


# ---------------------------------------------------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _exp(rotvec: np.ndarray) -> np.ndarray:
    """The rotation turning by ``|rotvec|`` about ``rotvec``."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3) + _skew(rotvec)
    k = _skew(np.asarray(rotvec, dtype=np.float64) / angle)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _log(rotation: np.ndarray) -> np.ndarray:
    """The rotation vector of ``rotation``, the inverse of :func:`_exp`, including near half a turn."""
    cos = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]]) / 2.0
    if math.pi - angle < 1e-6:
        # Near half a turn the antisymmetric part vanishes; the axis is the column of R + I with the most in it.
        plus = rotation + np.eye(3)
        axis = plus[:, int(np.argmax(np.linalg.norm(plus, axis=0)))]
        return angle * axis / np.linalg.norm(axis)
    vee = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return angle * vee / (2.0 * math.sin(angle))


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / float(np.linalg.norm(v))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_unit(a), _unit(b)))))))


def _swing(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """The smallest rotation taking unit ``a`` onto unit ``b``; ``None`` where they point opposite ways."""
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    sin, cos = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if sin < 1e-9:
        return np.eye(3) if cos > 0.0 else None
    return _exp(axis / sin * math.atan2(sin, cos))


def _matrix(value: Any, what: str) -> np.ndarray:
    """``value`` as a finite 4x4 rigid transform: a ``Pose``, a matrix, or anything ``np.asarray`` reads as one."""
    matrix = value.to_matrix() if isinstance(value, Pose) else np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{what} is a finite 4x4 transform, got shape {matrix.shape}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or float(np.linalg.det(rotation)) < 0.0:
        raise ValueError(f"{what} is not a rigid transform: its rotation block is not a rotation")
    return matrix


def _rigid(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = position
    return out


def _as_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in np.asarray(matrix, dtype=np.float64))


def _point(value: "Sequence[float] | np.ndarray", what: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{what} is three finite numbers in millimetres, got {value!r}")
    return point


# ---------------------------------------------------------------------------------------------------------------------
# Where the camera sits on the tool
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class View:
    """One frame in which the marker was posed: the TCP in BASE at the shutter and the marker in the CAMERA, 4x4, mm."""

    tool_in_base: np.ndarray
    marker_in_camera: np.ndarray

    @classmethod
    def of(cls, tool_in_base: Any, marker_in_camera: Any) -> "View":
        return cls(_matrix(tool_in_base, "the tool's pose in the base"),
                   _matrix(marker_in_camera, "the marker's pose in the camera"))


@dataclass(frozen=True, slots=True, eq=False)
class Look:
    """What the camera saw from where the arm stood, with nothing moved: the first thing an aimed sweep does.

    ``tool_in_base`` is the TCP read just before the frame was taken, ``joints`` the configuration (``None`` where
    the arm could not say), ``marker_in_camera`` the marker's pose in that frame or ``None``, and ``why_not`` the
    marker source's sentence for a frame in which it found none.
    """

    tool_in_base: Pose
    joints: tuple[float, ...] | None
    marker_in_camera: np.ndarray | None
    why_not: str = ""

    @property
    def view(self) -> View | None:
        return None if self.marker_in_camera is None else View.of(self.tool_in_base, self.marker_in_camera)


@dataclass(frozen=True, slots=True)
class MountEstimate:
    """Where the camera sits on the tool, well enough to aim it: CAMERA->TOOL as a 4x4 in millimetres.

    ``source`` says what it rests on (``"one look"``, ``"N views"``, ``"a stated mount ..."``), ``views`` how many
    views (0 for a stated mount), and
    ``marker_mm`` the marker position it places (the one measured, or refined from two views or more). ``rms_mm`` and
    ``rms_deg`` are how far the views miss the marker it places, ``None`` for one view, which fits exactly.
    """

    camera_in_tool: tuple[tuple[float, ...], ...]
    source: str
    views: int = 0
    marker_mm: tuple[float, float, float] | None = None
    rms_mm: float | None = None
    rms_deg: float | None = None

    def matrix(self) -> np.ndarray:
        return np.array(self.camera_in_tool, dtype=np.float64)

    @property
    def optical_axis_in_tool(self) -> np.ndarray:
        """Where the camera looks, as a unit vector in the tool frame."""
        return self.matrix()[:3, 2].copy()

    @property
    def position_in_tool_mm(self) -> np.ndarray:
        return self.matrix()[:3, 3].copy()

    @property
    def tilt_deg(self) -> float:
        """How far the optical axis is turned from the tool's +Z, the way the tool points."""
        return _angle_deg(self.optical_axis_in_tool, np.array([0.0, 0.0, 1.0]))

    def line(self) -> str:
        """The estimate in one line: where the camera is, how far it is tilted, and what it rests on."""
        x, y, z = (float(v) for v in self.position_in_tool_mm)
        ax, ay, az = (float(v) for v in self.optical_axis_in_tool)
        said = (f"camera at ({x:.0f}, {y:.0f}, {z:.0f}) mm in the tool frame, looking along ({ax:+.2f}, {ay:+.2f}, "
                f"{az:+.2f}), {self.tilt_deg:.1f} deg from the tool axis; from {self.source}")
        if self.rms_mm is not None and self.rms_deg is not None:
            said += f", the views missing the marker by {self.rms_mm:.1f} mm and {self.rms_deg:.1f} deg rms"
        return said


def estimate_camera_in_tool(
    tool_in_base: Any,
    marker_in_camera: Any,
    marker_mm: "Sequence[float] | np.ndarray",
    *,
    camera_prior_in_tool_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> MountEstimate:
    """The camera's pose in the tool frame from ONE view of a marker lying flat, face up, at ``marker_mm`` (BASE).

    ``tool_in_base`` is the TCP the arm reported at the shutter (a ``Pose`` or a 4x4), ``marker_in_camera`` the
    marker's pose in the camera as the estimator reported it (``Observation.T_cam_to_target``). The camera is first
    taken to stand at ``camera_prior_in_tool_mm``, the flange's place in the tool frame when the caller knows the tool
    frame (``(0, 0, -L)`` for a TCP ``L`` mm past the flange), the TCP otherwise.

    Two direction pairs fix the rotation: the marker's normal (its +Z in the camera, base +Z in the world) exactly, and
    the line of sight to the marker's centre (in the camera, and from the prior in the world) for the turn about that
    normal. The view then fixes the position. The module docstring says how far off that is and why a second view is
    what improves it.

    Raises ``ValueError`` for a view seen within :data:`MIN_OBLIQUE_DEG` of frontal, or from a prior within 10 degrees
    of straight above the marker: either names no turn about the marker's normal.
    """
    tool = _matrix(tool_in_base, "the tool's pose in the base")
    seen = _matrix(marker_in_camera, "the marker's pose in the camera")
    marker = _point(marker_mm, "the marker position")
    prior = _point(camera_prior_in_tool_mm, "the camera's first place in the tool")
    r_bt, p_bt = tool[:3, :3], tool[:3, 3]
    normal_cam, sight_cam = seen[:3, 2], _unit(seen[:3, 3])
    oblique = _angle_deg(normal_cam, -sight_cam)
    if oblique < MIN_OBLIQUE_DEG:
        raise ValueError(
            f"the camera sees the marker {oblique:.1f} deg from face on, within {MIN_OBLIQUE_DEG:g} deg: a flat marker "
            "seen that squarely names no turn about its own normal. Look at it from further to one side")
    normal_tool = r_bt.T @ np.array([0.0, 0.0, 1.0])
    towards = marker - (p_bt + r_bt @ prior)
    off_vertical = _angle_deg(-towards, np.array([0.0, 0.0, 1.0]))
    if off_vertical < _MIN_PRIOR_ELEVATION_OFF_VERTICAL_DEG:
        raise ValueError(
            f"the tool stands {off_vertical:.1f} deg from straight above the marker, within "
            f"{_MIN_PRIOR_ELEVATION_OFF_VERTICAL_DEG:g} deg: from there one look cannot tell which way round the "
            "marker the camera stands. Stand the arm off to one side of the marker")
    sight_tool = _unit(r_bt.T @ towards)
    rotation = _triad(normal_cam, sight_cam, normal_tool, sight_tool)
    position = r_bt.T @ (marker - p_bt) - rotation @ seen[:3, 3]
    return MountEstimate(camera_in_tool=_as_tuple(_rigid(rotation, position)), source="one look", views=1,
                         marker_mm=(float(marker[0]), float(marker[1]), float(marker[2])))


def _triad(first_a: np.ndarray, second_a: np.ndarray, first_b: np.ndarray, second_b: np.ndarray) -> np.ndarray:
    """The rotation taking ``first_a`` exactly onto ``first_b`` and ``second_a`` as near ``second_b`` as it can."""

    def frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        e1 = _unit(first)
        e2 = _unit(np.cross(e1, second))
        return np.column_stack([e1, e2, np.cross(e1, e2)])

    return frame(first_b, second_b) @ frame(first_a, second_a).T


def refine_camera_in_tool(
    views: Sequence[View],
    marker_mm: "Sequence[float] | np.ndarray",
    *,
    start: MountEstimate,
    camera_prior_in_tool_mm: Sequence[float] = (0.0, 0.0, 0.0),
    iterations: int = 40,
) -> MountEstimate:
    """The camera's pose in the tool frame from every view so far, starting from ``start``.

    A small least-squares fit (Levenberg-Marquardt) of the mount, the marker's turn about its normal and the marker's
    position, holding the marker flat: every view's tool pose, times the mount, times the marker's pose in that
    camera, must land on one marker lying flat. The measured marker position is a weak prior (100 mm), so what the
    views can fix they fix, and what they cannot (a translation along the axis every view so far turned about) stays
    where it was measured. The camera's first place is read only with one view, where it is the only thing that fixes
    the turn about the normal. Two views from places that are not turned about the marker's own vertical fix the
    turn the first look left open.
    """
    if not views:
        raise ValueError("a refinement needs at least one view")
    measured = _point(marker_mm, "the marker position")
    prior = _point(camera_prior_in_tool_mm, "the camera's first place in the tool")
    mount = start.matrix()
    rotation, position = mount[:3, :3].copy(), mount[:3, 3].copy()
    placed = np.array(start.marker_mm, dtype=np.float64) if start.marker_mm is not None else measured.copy()
    first = views[0].tool_in_base[:3, :3] @ rotation @ views[0].marker_in_camera[:3, :3]
    yaw = math.atan2(float(first[1, 0]), float(first[0, 0]))
    use_prior = len(views) == 1

    def residuals(delta: np.ndarray, r0: np.ndarray, p: np.ndarray, psi: float, m: np.ndarray) -> np.ndarray:
        r_tc = _exp(delta) @ r0
        rz_t = _rz(psi).T
        out: list[np.ndarray] = []
        for view in views:
            r_bt, p_bt = view.tool_in_base[:3, :3], view.tool_in_base[:3, 3]
            r_cm, t_cm = view.marker_in_camera[:3, :3], view.marker_in_camera[:3, 3]
            out.append((r_bt @ (r_tc @ t_cm + p) + p_bt - m) / _SIGMA_VIEW_MM)
            out.append(_log(rz_t @ r_bt @ r_tc @ r_cm) / _SIGMA_VIEW_RAD)
        out.append((m - measured) / _SIGMA_MARKER_MM)
        if use_prior:
            out.append((p - prior) / _SIGMA_PRIOR_MM)
        return np.concatenate(out)

    # Ten unknowns about the current values: a turn of the mount (3), its position (3), the marker's turn about its
    # normal (1) and its position (3). Each pass folds the step into the current values and starts again at zero.
    damping = 1e-3
    steps = np.array([1e-6] * 3 + [1e-3] * 3 + [1e-6] + [1e-3] * 3)

    def at(x: np.ndarray) -> np.ndarray:
        return residuals(x[:3], rotation, position + x[3:6], yaw + float(x[6]), placed + x[7:10])

    for _ in range(max(1, int(iterations))):
        r0 = at(np.zeros(10))
        jacobian = np.column_stack([(at(np.eye(10)[k] * steps[k]) - r0) / steps[k] for k in range(10)])
        normal, gradient, cost = jacobian.T @ jacobian, jacobian.T @ r0, float(r0 @ r0)
        while True:
            try:
                step = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
            except np.linalg.LinAlgError:
                step = np.zeros(10)
            trial = at(step)
            if float(trial @ trial) <= cost or damping > 1e8:
                break
            damping *= 10.0
        if float(trial @ trial) > cost:
            break
        damping = max(damping / 10.0, 1e-9)
        rotation = _exp(step[:3]) @ rotation
        position, yaw, placed = position + step[3:6], yaw + float(step[6]), placed + step[7:10]
        if float(np.max(np.abs(step / steps))) < 1e-2:
            break

    misses_mm: list[float] = []
    misses_deg: list[float] = []
    for view in views:
        r_bt, p_bt = view.tool_in_base[:3, :3], view.tool_in_base[:3, 3]
        r_cm, t_cm = view.marker_in_camera[:3, :3], view.marker_in_camera[:3, 3]
        misses_mm.append(float(np.linalg.norm(r_bt @ (rotation @ t_cm + position) + p_bt - placed)))
        misses_deg.append(math.degrees(float(np.linalg.norm(_log(_rz(yaw).T @ r_bt @ rotation @ r_cm)))))
    many = len(views) > 1
    return MountEstimate(
        camera_in_tool=_as_tuple(_rigid(rotation, position)),
        source=f"{len(views)} views" if many else start.source, views=len(views),
        marker_mm=(float(placed[0]), float(placed[1]), float(placed[2])),
        rms_mm=float(np.sqrt(np.mean(np.square(misses_mm)))) if many else None,
        rms_deg=float(np.sqrt(np.mean(np.square(misses_deg)))) if many else None,
    )


def nominal_camera_in_tool(
    tilt_deg: float = 45.0,
    *,
    toward: str = "+x",
    offset_mm: Sequence[float] = (0.0, 0.0, 0.0),
) -> MountEstimate:
    """A mount stated by hand: the optical axis turned ``tilt_deg`` from the tool's +Z toward the tool axis ``toward``.

    ``toward`` is ``"+x"``, ``"-x"``, ``"+y"`` or ``"-y"``, a tool axis: ``+x`` is the axis the jaws close along, so on
    a pose built with ``closing_axis="-y"`` (the tool's +X along base -Y) a camera tilted to look along base -Y as well
    is ``"+x"``, and one tilted the other way ``"-x"``. The turn is about the tool axis at right angles to both: toward
    ``+x`` it is a turn about +Y by ``+tilt_deg``, toward ``+y`` a turn about +X by ``-tilt_deg``; the sign of
    ``toward`` flips it. The image's own roll is left as the tool's, which aiming does not read. ``offset_mm`` is where
    the camera stands in the tool frame, ``(0, 0, -L)`` at a flange ``L`` mm behind the TCP.

    Nothing reads this unless a caller hands it in (``MarkerAim.mount_if_unseen``): which way a camera is tilted on a
    tool is a fact of the cell, and a stated one is aimed from until the first station that sees the marker replaces it.
    """
    axis = str(toward).strip().lower()
    if axis not in ("+x", "-x", "+y", "-y", "x", "y"):
        raise ValueError(f"toward is a tool axis, '+x', '-x', '+y' or '-y', not {toward!r}")
    tilt = math.radians(float(tilt_deg))
    if not math.isfinite(tilt) or not 0.0 <= float(tilt_deg) < 90.0:
        raise ValueError(f"tilt_deg is at least 0 and under 90 degrees, got {tilt_deg!r}")
    sign = -1.0 if axis.startswith("-") else 1.0
    rotation = (_exp(np.array([0.0, sign * tilt, 0.0])) if axis.endswith("x")
                else _exp(np.array([-sign * tilt, 0.0, 0.0])))
    position = _point(offset_mm, "offset_mm")
    named = axis if axis[0] in "+-" else "+" + axis
    return MountEstimate(camera_in_tool=_as_tuple(_rigid(rotation, position)),
                         source=f"a stated mount, {float(tilt_deg):g} deg toward the tool's {named}")


# ---------------------------------------------------------------------------------------------------------------------
# Aiming
# ---------------------------------------------------------------------------------------------------------------------


def _reference_rotation(closing_axis: str, at_mm: np.ndarray) -> np.ndarray:
    """The tool pointing straight down, its +X along ``closing_axis``, as ``Pose.tool_down`` builds it at ``at_mm``."""
    down = Pose.tool_down(float(at_mm[0]), float(at_mm[1]), float(at_mm[2]), closing_axis=closing_axis)
    return down.to_matrix()[:3, :3]


def aim_camera(
    camera_mm: "Sequence[float] | np.ndarray",
    marker_mm: "Sequence[float] | np.ndarray",
    mount: MountEstimate,
    *,
    closing_axis: str = "x",
    roll_deg: float = 0.0,
    label: str | None = None,
) -> tuple[Pose, float]:
    """The TCP pose that puts the camera at ``camera_mm`` looking straight at ``marker_mm``, and its tilt in degrees.

    The tool is turned as little as it can be from ``Pose.tool_down`` with ``closing_axis`` (at the camera's position,
    for ``radial`` and ``tangential``): the smallest rotation that swings the optical axis from where it points with the
    tool down onto the marker. The tilt returned is that rotation's angle. ``roll_deg`` then turns the camera about its
    own line of sight, right-handed about the way it looks, with the camera standing where it stood: the marker stays
    where it is in the image and the image turns about it, and the tool turns with the camera, away from its heading.
    The tilt does not count the roll. Aiming says nothing about reach, the box or a collision; the arm judges the pose
    when it moves.

    Raises ``ValueError`` where the camera would have to turn half a turn from where it looks with the tool down.
    """
    camera = _point(camera_mm, "the camera position")
    target = _point(marker_mm, "the marker position")
    if float(np.linalg.norm(target - camera)) < 1.0:
        raise ValueError("the camera would stand on the marker, which names no direction to look along")
    mount_matrix = mount.matrix()
    reference = _reference_rotation(closing_axis, camera)
    looks = reference @ mount_matrix[:3, 2]
    wanted = _unit(target - camera)
    swing = _swing(looks, wanted)
    if swing is None:
        raise ValueError("the camera would have to turn half a turn from where it looks with the tool down")
    rotation = _exp(wanted * math.radians(float(roll_deg))) @ swing @ reference
    tool = _rigid(rotation, camera - rotation @ mount_matrix[:3, 3])
    return Pose.from_matrix(tool, frame=Frame.BASE, label=label), _angle_deg(looks, wanted)


@dataclass(frozen=True, slots=True, eq=False)
class ArmReach:
    """What an arm reaches, for screening aimed stations before they are commanded.

    ``model`` names a UR with a DH table (``_ur_ik``), ``flange_to_tcp`` is the 4x4 the TCP stands at from the flange
    (identity where the tool frame is the flange), and ``lower_rad`` / ``upper_rad`` the joint window the arm chooses a
    Cartesian goal's configuration inside (the planner's, narrowed to what the joint-limit guard admits, half a turn
    about home included where the cell keeps it). ``None`` bounds screen reach only.
    """

    model: str
    flange_to_tcp: np.ndarray = field(default_factory=lambda: np.eye(4))
    lower_rad: tuple[float, ...] | None = None
    upper_rad: tuple[float, ...] | None = None


def reach_of(arm: Any) -> ArmReach | None:
    """The reach of a connected arm, read best effort: ``None`` where it offers no model to solve on.

    A UR driver answers with its model (``config.ur.model``), the tool frame its TCP applies (``active_tool_frame``)
    and the window its nearest-goal choice uses (``_goal_joint_window``), so the stations screened here are screened
    against the same numbers the move is chosen within.
    """
    from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

    model = getattr(getattr(getattr(arm, "config", None), "ur", None), "model", None)
    if not isinstance(model, str) or model.lower() not in UR_DH_TABLES_M:
        return None
    try:
        tool = getattr(arm, "active_tool_frame", None)
    except Exception:  # noqa: BLE001 (best effort: a frame that cannot be read is the flange's)
        tool = None
    flange_to_tcp = np.eye(4) if tool is None else np.asarray(tool, dtype=np.float64)
    lower = upper = None
    window = getattr(arm, "_goal_joint_window", None)
    if callable(window):
        try:
            low, high = window()
            lower, upper = tuple(float(v) for v in low), tuple(float(v) for v in high)
        except Exception:  # noqa: BLE001 (no window read: reach only)
            logger.debug("the arm's joint window could not be read", exc_info=True)
    return ArmReach(model=model.lower(), flange_to_tcp=flange_to_tcp, lower_rad=lower, upper_rad=upper)


def flange_in_tool_mm(flange_to_tcp: Any) -> tuple[float, float, float]:
    """Where the flange stands in the tool frame, in millimetres: the camera's first place for a first look.

    ``flange_to_tcp`` is the 4x4 the TCP stands at from the flange (``ArmReach.flange_to_tcp``,
    ``active_tool_frame``); ``None`` is a tool frame that is the flange, and gives the origin. A wrist camera beside
    the coupling stands nearer the flange than the grasp centre, which is why the flange is the better first guess.
    """
    if flange_to_tcp is None:
        return (0.0, 0.0, 0.0)
    inverse = np.linalg.inv(_matrix(flange_to_tcp, "the flange to TCP"))
    return (float(inverse[0, 3]), float(inverse[1, 3]), float(inverse[2, 3]))


@dataclass(frozen=True, slots=True)
class MarkerAim:
    """An eye-in-hand sweep aimed at one marker lying flat, face up, at ``marker_mm`` in BASE millimetres.

    The sweep looks once from where the arm stands, estimates where the camera sits on the tool
    (:func:`estimate_camera_in_tool`), and visits ``views`` round the marker with the camera ``distance_mm`` from it,
    each station tilting the tool as little as it can from tool-down with ``closing_axis`` and then rolling the camera
    about its line of sight by the view's roll. From its second view on it refines the estimate and re-aims the
    stations still to come. ``views`` are (elevation, azimuth) offsets in degrees from the view the camera has with the
    tool down, each with a roll in degrees as a third number, at most 90 either way, or none for no roll
    (:data:`DEFAULT_VIEWS`); two views at one elevation and azimuth are refused. A station that would tilt the tool
    more than ``max_tilt_deg`` is reported and not moved to. ``closing_axis`` is where the heading starts: where it
    admits fewer views than the solve needs, the sweep takes the one of :data:`HEADINGS` that admits the most
    (:meth:`AimedSweep.settle_heading`).

    ``mount_if_unseen`` is a mount to aim from when the first look sees no marker (:func:`nominal_camera_in_tool`).
    Left ``None``, such a sweep is refused with nothing moved, which is the default because which way a camera is
    tilted on a tool is a fact of the cell and not something to guess.
    """

    marker_mm: tuple[float, float, float]
    distance_mm: float = 500.0
    closing_axis: str = "x"
    views: tuple[tuple[float, ...], ...] = DEFAULT_VIEWS
    max_tilt_deg: float = 60.0
    mount_if_unseen: MountEstimate | None = None

    def __post_init__(self) -> None:
        marker = _point(self.marker_mm, "MarkerAim.marker_mm")
        object.__setattr__(self, "marker_mm", (float(marker[0]), float(marker[1]), float(marker[2])))
        low, high = AIM_DISTANCE_RANGE_MM
        distance = float(self.distance_mm)
        if not math.isfinite(distance) or not low <= distance <= high:
            raise ValueError(f"MarkerAim.distance_mm is between {low:g} and {high:g} mm, got {self.distance_mm!r}")
        object.__setattr__(self, "distance_mm", distance)
        axis = str(self.closing_axis)
        if axis.lstrip("+") not in CLOSING_AXES:
            raise ValueError(f"MarkerAim.closing_axis is one of {', '.join(CLOSING_AXES)}, got {self.closing_axis!r}")
        views = tuple(tuple(float(v) for v in view) for view in self.views)
        if not views:
            raise ValueError("MarkerAim.views holds at least one (elevation, azimuth) offset")
        for view in views:
            if len(view) not in (2, 3):
                raise ValueError(f"MarkerAim.views: {view!r} is not (elevation, azimuth) or (elevation, azimuth, roll) "
                                 "in degrees")
            elevation, azimuth, roll = view[0], view[1], _roll(view)
            finite = all(math.isfinite(v) for v in view)
            if not finite or abs(elevation) > 45.0 or abs(azimuth) > 90.0 or abs(roll) > 90.0:
                raise ValueError(f"MarkerAim.views: {view!r} is not an offset of at most 45 deg in elevation and 90 "
                                 "deg in azimuth, with a roll of at most 90 deg")
        if len({(round(view[0], 6), round(view[1], 6)) for view in views}) != len(views):
            raise ValueError("MarkerAim.views names one view twice: two views at one elevation and azimuth")
        object.__setattr__(self, "views", views)
        tilt = float(self.max_tilt_deg)
        if not math.isfinite(tilt) or not 0.0 < tilt <= 90.0:
            raise ValueError(f"MarkerAim.max_tilt_deg is above 0 and at most 90, got {self.max_tilt_deg!r}")
        if self.mount_if_unseen is not None and not isinstance(self.mount_if_unseen, MountEstimate):
            raise TypeError("MarkerAim.mount_if_unseen is a MountEstimate, such as nominal_camera_in_tool() returns")

    def line(self) -> str:
        x, y, z = self.marker_mm
        return (f"aimed at the marker at ({x:.1f}, {y:.1f}, {z:.1f}) mm from {self.distance_mm:.0f} mm, "
                f"{len(self.views)} views, heading {self.closing_axis!r} unless it admits too few, stations built "
                "after a first look")


@dataclass(frozen=True, slots=True)
class AimedStation:
    """One view of an aimed sweep: its pose, or why it is not moved to, and what reaching it costs.

    ``view`` is the view as the aim names it; ``roll_deg`` the roll about the line of sight the pose was aimed with,
    the view's own, or 0 where the rolled pose was refused and the view is visited with the heading instead.
    ``joints`` is the configuration the arm's nearest-goal choice would take for it from the station before, where
    the arm's reach was known; ``largest_rad`` the largest turn of any joint on the way there.
    """

    label: str
    view: tuple[float, ...]
    pose: Pose | None
    tilt_deg: float = 0.0
    reason: str = ""
    detail: str = ""
    joints: tuple[float, ...] | None = None
    largest_rad: float = 0.0
    roll_deg: float = 0.0

    def as_station(self) -> "Pose | SkippedStation":
        """What the sweep visits: the pose, or a station it reports with its reason and never moves to."""
        if self.pose is not None:
            return self.pose
        return SkippedStation(label=self.label, reason=self.reason, detail=self.detail)


def _natural_view(mount: MountEstimate, marker: np.ndarray, closing_axis: str) -> tuple[float, float]:
    """The elevation and azimuth, in degrees, the camera looks along with the tool pointing down."""
    looks = _reference_rotation(closing_axis, marker + np.array([0.0, 0.0, 300.0])) @ mount.optical_axis_in_tool
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, -float(looks[2])))))
    if math.hypot(float(looks[0]), float(looks[1])) < _MIN_AZIMUTH_COS:
        # A camera that looks straight down names no side; the cone then stands between the base and the marker.
        azimuth = math.degrees(math.atan2(float(marker[1]), float(marker[0])))
    else:
        azimuth = math.degrees(math.atan2(float(looks[1]), float(looks[0])))
    low, high = _BASE_ELEVATION_RANGE_DEG
    return min(high, max(low, elevation)), azimuth


def _inside(box: Any, point: np.ndarray, margin: float) -> bool:
    x, y, z = (float(v) for v in point)
    return (box.x_min + margin <= x <= box.x_max - margin and box.y_min + margin <= y <= box.y_max - margin
            and box.z_min + margin <= z <= box.z_max - margin)


def _box_text(box: Any, margin: float) -> str:
    return (f"x {box.x_min + margin:.1f} to {box.x_max - margin:.1f}, y {box.y_min + margin:.1f} to "
            f"{box.y_max - margin:.1f}, z {box.z_min + margin:.1f} to {box.z_max - margin:.1f} mm")


def _view_label(view: tuple[float, ...]) -> str:
    """A view's label, by its place alone: a view is visited once whatever its roll, so the label names it."""
    return f"aim_e{view[0]:+g}_a{view[1]:+g}"


def _roll(view: Sequence[float]) -> float:
    """A view's roll about the line of sight in degrees: its third number, or 0 for a view of two."""
    return float(view[2]) if len(view) > 2 else 0.0


def aimed_stations(
    aim: MarkerAim,
    mount: MountEstimate,
    *,
    box: Any = None,
    margin_mm: float = 0.0,
    reach: ArmReach | None = None,
    start_joints: Sequence[float] | None = None,
    start_tool_mm: Sequence[float] | None = None,
    skip_labels: Sequence[str] = (),
) -> list[AimedStation]:
    """Every view of ``aim`` not in ``skip_labels``, aimed from ``mount``, screened and ordered.

    Each view puts the camera ``aim.distance_mm`` from the marker at its elevation and azimuth (offsets from the view
    the camera has with the tool down) and aims it with :func:`aim_camera`, rolled by the view's roll. A view is
    reported, with its reason and not moved to, when its tool would tilt more than ``aim.max_tilt_deg``
    (``too_tilted``), when its TCP or its flange lies outside ``box`` less ``margin_mm`` (``outside_workspace``; the
    arm's own gate boxes the TCP, less the same margin), and, with ``reach``, when no joint configuration puts the
    flange there (``out_of_reach``) or none lies inside the arm's joint window (``outside_joint_window``, the window
    half a turn about home keeps a cable within). A roll never costs a view: where the rolled pose is refused, the view
    is screened again unrolled, the tool keeping its heading, and taken so where that is admitted (its ``roll_deg`` is
    then 0); the reason reported for a view refused both ways is the unrolled one's.

    With ``reach`` every view is screened so; with ``start_joints`` as well the stations are ordered greedily by joint
    travel from ``start_joints``, each next the one whose nearest configuration in the window is the smallest
    largest-joint turn from the last, as the arm's nearest-goal choice would take it. Without them, greedily by the
    TCP's travel from ``start_tool_mm`` (where the tool stands, BASE mm), or from the first of ``aim.views`` without
    that. The stations reported come last, in the order of ``aim.views``.
    """
    from src.robot.safety._ur_ik import nearest_goals, ur_flange_ik

    marker = np.array(aim.marker_mm, dtype=np.float64)
    if mount.marker_mm is not None:
        marker = np.array(mount.marker_mm, dtype=np.float64)
    base_elevation, base_azimuth = _natural_view(mount, marker, aim.closing_axis)
    low_e, high_e = _ELEVATION_RANGE_DEG
    margin = float(margin_mm)
    flange_from_tcp = None if reach is None else np.linalg.inv(np.asarray(reach.flange_to_tcp, dtype=np.float64))
    skipped = set(skip_labels)
    # Whether a view has a configuration inside the window does not depend on where the arm comes from, so each is
    # screened once, from any start; which of its configurations is nearest does, so the order is chosen one station
    # at a time, from where the arm stands.
    here = [float(v) for v in start_joints] if start_joints is not None else None
    window = (reach.lower_rad, reach.upper_rad) if reach is not None else (None, None)
    lower = list(window[0]) if window[0] is not None else [-2.0 * math.pi] * _UR_JOINTS
    upper = list(window[1]) if window[1] is not None else [2.0 * math.pi] * _UR_JOINTS
    seed = here if here is not None else [0.0] * len(lower)

    def screened(view: tuple[float, ...], label: str, camera: np.ndarray,
                 roll: float) -> tuple[AimedStation, tuple[tuple[float, ...], ...]]:
        """The station ``view`` aims at ``camera`` rolled by ``roll``, or why it is refused, and its configurations."""
        try:
            pose, tilt = aim_camera(camera, marker, mount, closing_axis=aim.closing_axis, roll_deg=roll, label=label)
        except ValueError as exc:
            return AimedStation(label, view, None, reason="not_aimed", detail=f"{exc}, so nothing moved"), ()
        if tilt > aim.max_tilt_deg:
            return AimedStation(label, view, None, tilt_deg=tilt, reason="too_tilted", detail=(
                f"the tool would tilt {tilt:.0f} deg from tool-down to aim the camera, over the {aim.max_tilt_deg:g} "
                "deg this sweep allows, so nothing moved")), ()
        outside = _outside_box(box, margin, pose.to_matrix(), flange_from_tcp) if box is not None else ""
        if outside:
            return AimedStation(label, view, None, tilt_deg=tilt, reason="outside_workspace", detail=outside), ()
        station = AimedStation(label, view, pose, tilt_deg=tilt, roll_deg=roll)
        if reach is None:
            return station, ()
        assert flange_from_tcp is not None
        solutions = ur_flange_ik(reach.model, pose.to_matrix() @ flange_from_tcp, q6_if_singular=seed[-1])
        if not solutions:
            return dataclasses.replace(station, pose=None, reason="out_of_reach", detail=(
                "no joint configuration of the arm puts its flange where this view needs it, so nothing moved")), ()
        if not nearest_goals(solutions, current=seed, lower=lower, upper=upper, velocity=1.0):
            return dataclasses.replace(station, pose=None, reason="outside_joint_window", detail=(
                f"none of its {len(solutions)} joint configuration(s) lies inside the arm's joint window (half a turn "
                "either side of home, where the cell keeps it), so nothing moved")), ()
        return station, tuple(tuple(float(q) for q in solution) for solution in solutions)

    reachable: list[tuple[AimedStation, tuple[tuple[float, ...], ...]]] = []
    refused: list[AimedStation] = []
    for view in aim.views:
        label = _view_label(view)
        if label in skipped:
            continue
        elevation, azimuth = base_elevation + view[0], base_azimuth + view[1]
        if not low_e <= elevation <= high_e:
            refused.append(AimedStation(label, view, None, reason="not_aimed", detail=(
                f"the view would look {elevation:.0f} deg below the horizontal, outside {low_e:g} to {high_e:g}, so "
                "nothing moved")))
            continue
        e, a = math.radians(elevation), math.radians(azimuth)
        looks = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), -math.sin(e)])
        camera = marker - aim.distance_mm * looks
        station, solutions = screened(view, label, camera, _roll(view))
        if station.pose is None and _roll(view) != 0.0:
            # A roll never costs a view: refused rolled, the view is screened again with the tool's heading.
            station, solutions = screened(view, label, camera, 0.0)
        if station.pose is None:
            refused.append(station)
        else:
            reachable.append((station, solutions))

    order = {view: index for index, view in enumerate(aim.views)}
    if reach is None or here is None:
        return (_by_tool_travel([station for station, _ in reachable], start_tool_mm, aim)
                + sorted(refused, key=lambda station: order[station.view]))

    ordered: list[AimedStation] = []
    while reachable:
        best_index, best_goal = 0, None
        for index, (_, solutions) in enumerate(reachable):
            goal = nearest_goals(solutions, current=here, lower=lower, upper=upper, velocity=1.0)[0]
            if best_goal is None or (goal.largest_rad, goal.total_rad) < (best_goal.largest_rad, best_goal.total_rad):
                best_index, best_goal = index, goal
        assert best_goal is not None
        station, _ = reachable.pop(best_index)
        ordered.append(dataclasses.replace(station, joints=tuple(best_goal.joints),
                                           largest_rad=float(best_goal.largest_rad)))
        here = list(best_goal.joints)
    return ordered + sorted(refused, key=lambda station: order[station.view])


def _outside_box(box: Any, margin: float, tool: np.ndarray, flange_from_tcp: np.ndarray | None) -> str:
    """Why a station's TCP or flange lies outside ``box`` less ``margin``, in one sentence; ``""`` where both are in."""
    points = [("TCP", tool[:3, 3])]
    if flange_from_tcp is not None:
        points.append(("flange", (tool @ flange_from_tcp)[:3, 3]))
    for what, point in points:
        if not _inside(box, point, margin):
            x, y, z = (float(v) for v in point)
            return (f"its {what} ({x:.1f}, {y:.1f}, {z:.1f}) mm is outside the workspace box less its margin "
                    f"({_box_text(box, margin)}), so nothing moved")
    return ""


def _by_tool_travel(stations: list[AimedStation], start_mm: Sequence[float] | None,
                    aim: MarkerAim) -> list[AimedStation]:
    """``stations`` ordered greedily by how far the TCP travels: from ``start_mm``, or from the first of ``aim.views``
    that remains where the tool's place is not known."""
    order = {view: index for index, view in enumerate(aim.views)}
    left = sorted(stations, key=lambda station: order[station.view])
    ordered: list[AimedStation] = []
    here = None if start_mm is None else np.asarray(start_mm, dtype=np.float64).reshape(3)
    while left:
        if here is None:
            nearest = 0
        else:
            nearest = min(range(len(left)), key=lambda i: float(np.linalg.norm(_position(left[i]) - here)))
        ordered.append(left.pop(nearest))
        here = _position(ordered[-1])
    return ordered


def _position(station: AimedStation) -> np.ndarray:
    assert station.pose is not None
    return np.asarray(station.pose.position_mm, dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# Which heading the tool keeps
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeadingChoice:
    """The heading an aimed sweep or a viewing pose keeps: the one asked for, or the one of :data:`HEADINGS` that
    admits the most views where the one asked for admits fewer than ``need``.

    Which way a camera is tilted on the tool decides where it has to stand to look at a point, and a heading that
    stands it beyond the workspace box admits no view at all: a camera tilted toward the tool's -X under a heading made
    for one tilted toward +X. ``admitted`` is each heading tried with the number of views it admits, the one asked for
    first; ``views`` is how many views there were, ``need`` how many ``what`` needs (the solve's ``min_samples``, or
    one for a viewing pose).
    """

    asked: str
    chosen: str
    admitted: tuple[tuple[str, int], ...]
    views: int
    need: int
    what: str = "the solve"

    @property
    def changed(self) -> bool:
        """Whether the heading kept is another than the one asked for."""
        return self.chosen != self.asked

    def line(self) -> str:
        """The choice in one line: what the heading asked for admits and, where that is too few, what was taken."""
        count = dict(self.admitted)[self.asked]
        said = f"{self.asked!r} admits {count} of the {self.views} views"
        if count >= self.need:
            return said
        short = f"fewer than the {self.need} {self.what} needs" if self.need > 1 else f"and {self.what} needs one"
        counts = ", ".join(f"{heading} {admitted}" for heading, admitted in self.admitted)
        if self.changed:
            return (f"{self.chosen!r}, not the {self.asked!r} asked: {said}, {short}; {self.chosen!r} admits the most "
                    f"({counts})")
        return f"{said}, {short}; no other heading admits more ({counts})"

    def to_dict(self) -> dict[str, Any]:
        return {"asked": self.asked, "chosen": self.chosen, "changed": self.changed, "admitted": dict(self.admitted),
                "views": self.views, "need": self.need, "what": self.what}


def choose_heading(
    asked: str,
    admitted_by: Callable[[str], int],
    *,
    need: int,
    views: int,
    what: str = "the solve",
) -> HeadingChoice:
    """The heading to keep: ``asked`` while it admits ``need`` views, else the one of :data:`HEADINGS` that admits most.

    ``admitted_by`` counts the views a heading admits, screened as the stations are. ``asked`` is counted first and
    alone while it admits enough; otherwise every heading of :data:`HEADINGS` other than it is counted too, and the
    most wins, ``asked`` winning a tie and the order of :data:`HEADINGS` settling the others. Where none admits
    ``need``, the most is still taken: the sweep then moves to what can be reached and its solve says how few.
    """
    counts = {asked: int(admitted_by(asked))}
    if counts[asked] < need:
        same = str(asked).lstrip("+")
        for heading in HEADINGS:
            if heading != same:
                counts[heading] = int(admitted_by(heading))
    # max keeps the first of the most: the one asked for, then the order of HEADINGS.
    chosen = max(counts, key=counts.__getitem__)
    return HeadingChoice(asked=asked, chosen=chosen, admitted=tuple(counts.items()), views=int(views), need=int(need),
                         what=what)


# ---------------------------------------------------------------------------------------------------------------------
# A sweep that re-aims as it sees
# ---------------------------------------------------------------------------------------------------------------------


class AimedSweep:
    """The stations of one aimed sweep, rebuilt from every view as the sweep sees the marker.

    Built from the first look's estimate (or the mount stated for a look that saw nothing). :meth:`stations` gives
    every view not yet visited, aimed and screened; :meth:`replan`, handed to ``CalibrationRoutine`` as its re-plan
    hook, refines the estimate from all the views so far and returns the stations still to come, aimed afresh. The
    first view that sees the marker after a stated mount replaces it with a one-view estimate; every later one refines.
    A refinement whose views miss the marker they place by more than ``25 mm`` or ``8 deg`` rms is not taken: the
    stations stay aimed from the estimate before it, and the log says why. :meth:`settle_heading`, before the first
    station, keeps the aim's heading or takes one that admits more views.
    """

    _MAX_RMS_MM = 25.0
    _MAX_RMS_DEG = 8.0

    def __init__(
        self,
        aim: MarkerAim,
        mount: MountEstimate,
        *,
        box: Any = None,
        margin_mm: float = 0.0,
        reach: ArmReach | None = None,
        camera_prior_in_tool_mm: Sequence[float] = (0.0, 0.0, 0.0),
        joints_now: Callable[[], Sequence[float] | None] | None = None,
        tool_now: Callable[[], Sequence[float] | None] | None = None,
    ) -> None:
        self.aim = aim
        self.mount = mount
        self.box = box
        self.margin_mm = float(margin_mm)
        self.reach = reach
        self.camera_prior_in_tool_mm = tuple(float(v) for v in camera_prior_in_tool_mm)
        self._joints_now = joints_now
        self._tool_now = tool_now
        #: Each estimate the sweep aimed from, in order, the first included.
        self.estimates: list[MountEstimate] = [mount]
        #: The heading the stations keep and why, once :meth:`settle_heading` has chosen it; ``None`` before.
        self.heading: HeadingChoice | None = None
        #: The roll each station last handed out was aimed with, by label, and the views already tried unrolled.
        self._rolls: dict[str, float] = {}
        self._unrolled: set[str] = set()

    def settle_heading(
        self, need: int, *, start_joints: Sequence[float] | None = None,
        start_tool_mm: Sequence[float] | None = None,
    ) -> HeadingChoice:
        """Keep the aim's heading where it admits ``need`` views, else take the one of :data:`HEADINGS` that admits
        the most (:func:`choose_heading`), each counted from the current estimate with the screen the stations get.

        The heading taken replaces ``aim.closing_axis`` for every station after, re-aimed ones included, and is kept on
        :attr:`heading`. Nothing moves; which heading admits a view is a question of the box, the reach and the window
        alone, and every station is still judged by the arm when it moves.
        """
        joints = start_joints if start_joints is not None else _read(self._joints_now)
        tool = start_tool_mm if start_tool_mm is not None else _read(self._tool_now)

        def admitted(heading: str) -> int:
            aim = dataclasses.replace(self.aim, closing_axis=heading)
            return sum(1 for station in aimed_stations(aim, self.mount, box=self.box, margin_mm=self.margin_mm,
                                                       reach=self.reach, start_joints=joints, start_tool_mm=tool)
                       if station.pose is not None)

        choice = choose_heading(self.aim.closing_axis, admitted, need=need, views=len(self.aim.views))
        if choice.changed:
            self.aim = dataclasses.replace(self.aim, closing_axis=choice.chosen)
        self.heading = choice
        return choice

    def stations(
        self, *, visited: Sequence[str] = (), start_joints: Sequence[float] | None = None,
        start_tool_mm: Sequence[float] | None = None,
    ) -> list[AimedStation]:
        """Every view not in ``visited``, aimed from the current estimate, screened and ordered from where the arm
        stands (``start_joints``, ``start_tool_mm``, or read through the callables the sweep was built with)."""
        joints = start_joints if start_joints is not None else _read(self._joints_now)
        tool = start_tool_mm if start_tool_mm is not None else _read(self._tool_now)
        stations = aimed_stations(self.aim, self.mount, box=self.box, margin_mm=self.margin_mm, reach=self.reach,
                                  start_joints=joints, start_tool_mm=tool, skip_labels=visited)
        self._rolls.update({station.label: station.roll_deg for station in stations if station.pose is not None})
        return stations

    def unrolled(self, label: str) -> Pose | None:
        """The view ``label`` names once more with the tool's heading, after the arm refused it rolled.

        The routine's retry hook (``CalibrationRoutine.run_aimed``): called with the label of a station the arm refused
        with nothing moved. A roll the screen refuses already gives way to the view unrolled (:func:`aimed_stations`);
        this does the same for one the arm's planning refuses, once per view, aimed from the current estimate and
        screened from where the arm stands. ``None`` for a station that was not rolled, a view tried so already, and
        one the screen refuses unrolled. The pose is labelled ``<label>_r0``, and the arm judges it as any move.
        """
        view = next((view for view in self.aim.views if _view_label(view) == label), None)
        if view is None or self._rolls.get(label, 0.0) == 0.0 or label in self._unrolled:
            return None
        self._unrolled.add(label)
        (station,) = aimed_stations(dataclasses.replace(self.aim, views=(view[:2],)), self.mount, box=self.box,
                                    margin_mm=self.margin_mm, reach=self.reach, start_joints=_read(self._joints_now),
                                    start_tool_mm=_read(self._tool_now))
        if station.pose is None:
            logger.info("View %s was refused rolled, and unrolled the screen refuses it too: %s", label,
                        station.detail)
            return None
        return Pose.from_matrix(station.pose.to_matrix(), frame=Frame.BASE, label=f"{label}_r0")

    def replan(self, views: Sequence[View], visited: Sequence[str]) -> "list[Pose | SkippedStation] | None":
        """The stations still to come, aimed from an estimate refined by ``views``; ``None`` to keep those planned."""
        if not views:
            return None
        try:
            start = self.mount
            if start.views == 0:
                # A stated mount: the first view the marker was seen in replaces it, as a first look would have.
                first = views[0]
                start = estimate_camera_in_tool(first.tool_in_base, first.marker_in_camera, self.aim.marker_mm,
                                                camera_prior_in_tool_mm=self.camera_prior_in_tool_mm)
            estimate = start if len(views) == 1 else refine_camera_in_tool(
                views, self.aim.marker_mm, start=start, camera_prior_in_tool_mm=self.camera_prior_in_tool_mm)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("The aim was not refined (%s); the stations stay aimed as they were.", exc)
            return None
        if ((estimate.rms_mm or 0.0) > self._MAX_RMS_MM) or ((estimate.rms_deg or 0.0) > self._MAX_RMS_DEG):
            logger.warning("The aim was not refined: %s, more than %g mm or %g deg; is the marker where MarkerAim "
                           "says, lying flat and still? The stations stay aimed as they were.",
                           estimate.line(), self._MAX_RMS_MM, self._MAX_RMS_DEG)
            return None
        self.mount = estimate
        self.estimates.append(estimate)
        logger.info("Aim refined: %s.", estimate.line())
        return [station.as_station() for station in self.stations(visited=visited)]



def _read(where: Callable[[], Sequence[float] | None] | None) -> Sequence[float] | None:
    """What ``where`` answers, or ``None`` where it is not given or raises: it orders stations, it gates nothing."""
    if where is None:
        return None
    try:
        return where()
    except Exception:  # noqa: BLE001 (order only)
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Where a calibrated wrist camera looks at the work from
# ---------------------------------------------------------------------------------------------------------------------

#: The views a viewing pose is chosen from, as (elevation, azimuth) offsets in degrees from the view the camera has
#: with the tool pointing straight down, written as :data:`DEFAULT_VIEWS` writes them: that view first, which tilts the
#: tool not at all, then views turned further from it. Of those the arm's box, reach and joint window admit, the one
#: that tilts the tool least is taken.
VIEWING_VIEWS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (0.0, -20.0), (0.0, 20.0), (15.0, 0.0), (-15.0, 0.0),
    (0.0, -40.0), (0.0, 40.0), (15.0, -30.0), (15.0, 30.0), (-15.0, -30.0), (-15.0, 30.0),
)


@dataclass(frozen=True, slots=True)
class ViewingPose:
    """Where a camera looks at the work from before anything is located, or why it needs no such pose, or has none.

    A wrist camera (``eye_in_hand``) sees what the arm points it at, so a locate from wherever the arm stands sees
    whatever that is: after a connect, where the last program left the arm; in a campaign, the retreat above the last
    grasp, too near for its depth to measure and turned by that grasp's yaw. ``pose`` is the TCP pose that puts the
    CALIBRATED camera ``distance_mm`` from ``work_mm`` looking straight at it, the tool turned as little as that allows
    from ``Pose.tool_down`` with ``closing_axis``, so it keeps that heading and only tilts (``tilt_deg``).
    ``closing_axis`` is the caller's, or, where that admits no view, the one of :data:`HEADINGS` that admits the most;
    ``heading`` says which and why (:class:`HeadingChoice`), and the text says so where it is not the caller's.
    ``view`` is which of :data:`VIEWING_VIEWS` it is, and ``joints`` the configuration the arm's nearest-goal choice
    would take for it from where the arm stood, where the arm could say; ``largest_rad`` is the largest joint turn to it.

    A fixed camera (``eye_to_hand``) sees the work from where it is mounted and needs none: ``needed`` is False and
    ``pose`` None. A wrist camera no view could be found for has ``needed`` True, ``pose`` None, and ``reason`` and
    ``detail`` saying why; ``ok`` is then False, and nothing should be located from where the arm happens to stand.

    ``nearest_mm`` is the nearest depth the camera measures at its depth mode where it is known, and ``depth_note``
    where the number comes from; ``screened`` says what the pose was screened against. Screening only drops a pose the
    arm would refuse: the pose is not a judged one, and the arm plans and judges it when it is handed to
    ``Robot.move``, as any move.
    """

    rig_id: str
    needed: bool
    work_mm: tuple[float, float, float]
    pose: Pose | None = None
    distance_mm: float = 0.0
    nearest_mm: float | None = None
    depth_note: str = ""
    tilt_deg: float = 0.0
    view: tuple[float, ...] | None = None
    joints: tuple[float, ...] | None = None
    largest_rad: float = 0.0
    screened: str = ""
    reason: str = ""
    detail: str = ""
    closing_axis: str = ""
    heading: HeadingChoice | None = None

    @property
    def ok(self) -> bool:
        """Whether a locate may go ahead: from ``pose`` once the arm stands there, or at once for a fixed camera."""
        return self.pose is not None or not self.needed

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        x, y, z = self.work_mm
        work = f"the work at ({x:.0f}, {y:.0f}, {z:.0f}) mm"
        if not self.needed:
            return (f"viewing pose  NOT NEEDED: rig {self.rig_id!r} is a fixed camera (eye_to_hand) and sees {work} "
                    "from where it is mounted, so it locates from wherever the arm stands")
        if self.pose is None:
            return (f"viewing pose  REFUSED ({self.reason}): no viewing pose for rig {self.rig_id!r} and {work}: "
                    f"{self.detail}. Nothing should be located from where the arm happens to stand")
        px, py, pz = (float(v) for v in self.pose.position_mm)
        lines = [f"viewing pose  FOUND for rig {self.rig_id!r}: TCP ({px:.1f}, {py:.1f}, {pz:.1f}) mm, the camera "
                 f"{self.distance_mm:.0f} mm from {work} and looking straight at it, the tool tilted "
                 f"{self.tilt_deg:.1f} deg from tool-down"]
        if self.heading is not None and self.heading.changed:
            lines.append(f"  heading: {self.heading.line()}")
        if self.nearest_mm is not None:
            lines.append(f"  depth: measured from {self.nearest_mm:.0f} mm on ({self.depth_note})")
        else:
            lines.append(f"  depth: {self.depth_note}")
        lines.append(f"  screened against {self.screened}")
        if self.joints is not None:
            lines.append(f"  the arm's nearest configuration there turns a joint {math.degrees(self.largest_rad):.1f} "
                         "deg at most from where it stood")
        lines.append("  the arm plans and judges the move there as any move")
        return "\n".join(lines)


def viewing_pose(
    camera: Any,
    work_mm: "Sequence[float] | np.ndarray",
    *,
    arm: Any = None,
    distance_mm: float = 500.0,
    closing_axis: str = "x",
    nearest_mm: float | None = None,
    views: Sequence[tuple[float, float]] = VIEWING_VIEWS,
) -> ViewingPose:
    """Where ``camera`` looks at ``work_mm`` (BASE millimetres) from before a locate: a :class:`ViewingPose`.

    ``camera`` is a camera owner (``Camera``, open or not, anything answering ``calibration()``, and ``rig`` for the
    depth mode) or the ``RigCalibration`` one gives. The calibration is read through the one loader, so a camera that
    declares none is refused here as everywhere (``RigNotCalibrated``, ``RigCalibrationError``). A fixed camera needs
    no viewing pose and gets none. For a wrist camera, the CAMERA->TOOL its eye-in-hand sweep measured is aimed as a
    calibration sweep aims a station (:func:`aimed_stations`): each of ``views``, the camera ``distance_mm`` from the
    work, is screened, and of those admitted the one that tilts the tool least from ``Pose.tool_down`` with
    ``closing_axis`` is taken. ``closing_axis`` is the heading the tool keeps; the one the camera was calibrated with
    keeps its image the way up it was then. Where it admits no view, as for a camera tilted the other way on the tool
    than the heading was written for, the one of :data:`HEADINGS` that admits the most is kept instead, the log and the
    text say so, and ``closing_axis`` and ``heading`` on the result say which (:func:`choose_heading`); where none
    admits one, the refusal names each.

    ``arm``, where handed in, is read best effort as the sweep reads it: its workspace box less the gate's margin
    (``safety.limits.workspace_margin_mm``, TCP and flange), its reach and the joint window its nearest-goal choice
    keeps (:func:`reach_of`: half a turn about home where the cell keeps it, the cable window), and the joints it stands
    at, which the reported ``joints`` are nearest to. Without an arm nothing is screened, and the text says so.

    ``nearest_mm`` is the nearest depth the camera measures; left ``None`` it is read from the rig, a RealSense its
    ``body.model`` names at its ``depth_resolution`` (Intel's figures: a D415 about 450 mm at 1280 x 720 and 310 mm at
    848 x 480, not measured here), and not checked where neither says. A ``distance_mm`` nearer than it is refused
    (``too_near``), because the work would be a hole in the depth. ``work_mm`` is the point that must be measured; a
    part's top stands nearer the camera than the table it lies on, by up to its height.

    Raises ``ValueError`` for a distance, closing axis or view :class:`MarkerAim` refuses, and ``TypeError`` for a
    ``camera`` that is neither a camera nor a calibration.
    """
    calibration = camera.calibration() if callable(getattr(camera, "calibration", None)) else camera
    mode = getattr(calibration, "mounting_mode", None)
    mode = str(getattr(mode, "value", mode))
    rig_id = str(getattr(calibration, "rig_id", "") or getattr(camera, "rig_id", "") or "")
    work = _point(work_mm, "the work point")
    work_tuple = (float(work[0]), float(work[1]), float(work[2]))
    if mode == "eye_to_hand":
        return ViewingPose(rig_id=rig_id, needed=False, work_mm=work_tuple)
    if mode != "eye_in_hand":
        raise TypeError(f"viewing_pose takes a camera or its calibration (a mounting_mode of eye_in_hand or "
                        f"eye_to_hand), not {type(camera).__name__} with mounting_mode {mode!r}")
    aim = MarkerAim(marker_mm=work_tuple, distance_mm=distance_mm, closing_axis=closing_axis, views=tuple(views))
    nearest: float | None
    if nearest_mm is not None:
        stated = float(nearest_mm)
        if not math.isfinite(stated) or stated <= 0.0:
            raise ValueError(f"nearest_mm is a depth above 0 mm, got {nearest_mm!r}")
        nearest, depth_note = stated, "as stated by the caller"
    else:
        nearest, depth_note = _nearest_depth(getattr(camera, "rig", None))
    if nearest is not None and aim.distance_mm < nearest:
        return ViewingPose(
            rig_id=rig_id, needed=True, work_mm=work_tuple, distance_mm=aim.distance_mm, nearest_mm=nearest,
            depth_note=depth_note, reason="too_near", detail=(
                f"the camera would stand {aim.distance_mm:.0f} mm from the work, nearer than the {nearest:.0f} mm its "
                f"depth is measured from ({depth_note}), so the work would read as a hole. Stand it further off "
                "(distance_mm), or stream depth at a mode that measures nearer (a D415's depth_resolution [848, 480], "
                "colour kept at [1280, 720])"))

    raw = calibration.camera_to_tool()
    raw = raw.to_matrix() if callable(getattr(raw, "to_matrix", None)) else raw
    mount = MountEstimate(camera_in_tool=_as_tuple(_matrix(raw, "the camera's calibrated pose in the tool")),
                          source=f"the calibration at {getattr(calibration, 'artifact_path', '?')}")
    box, margin, reach, joints, tool = _arm_screen(arm)
    screened = _screened(box, margin, reach)
    by_heading: dict[str, tuple[list[AimedStation], list[AimedStation]]] = {}

    def admitted_by(heading: str) -> int:
        admitted: list[AimedStation] = []
        turned_away: list[AimedStation] = []
        for view in aim.views:
            # One view at a time, each from where the arm stands, so the joints reported are the ones a move from here
            # would take and not the ones a sweep through the views before it would.
            (station,) = aimed_stations(dataclasses.replace(aim, closing_axis=heading, views=(view,)), mount, box=box,
                                        margin_mm=margin, reach=reach, start_joints=joints, start_tool_mm=tool)
            (admitted if station.pose is not None else turned_away).append(station)
        by_heading[heading] = (admitted, turned_away)
        return len(admitted)

    choice = choose_heading(aim.closing_axis, admitted_by, need=1, views=len(aim.views), what="a viewing pose")
    admitted, _ = by_heading[choice.chosen]
    if not admitted:
        turned_away = by_heading[aim.closing_axis][1]
        first = turned_away[0]
        others = sorted({station.reason for station in turned_away[1:]})
        detail = f"the camera's own view with the tool down is not admitted: {first.detail}"
        if others:
            detail += f"; nor is any of the {len(turned_away) - 1} views turned from it ({', '.join(others)})"
        tried = [f"{heading} {count}" for heading, count in choice.admitted[1:]]
        if tried:
            detail += (f"; and no other heading admits one either ({', '.join(tried)} of the {len(aim.views)} "
                       "views)")
        return ViewingPose(rig_id=rig_id, needed=True, work_mm=work_tuple, distance_mm=aim.distance_mm,
                           nearest_mm=nearest, depth_note=depth_note, screened=screened, reason=first.reason,
                           detail=detail, closing_axis=choice.chosen, heading=choice)
    if choice.changed:
        logger.warning("Viewing pose for rig %r: heading %s.", rig_id, choice.line())
    best = min(admitted, key=lambda station: (round(station.tilt_deg, 1), station.largest_rad))
    assert best.pose is not None
    pose = Pose.from_matrix(best.pose.to_matrix(), frame=Frame.BASE, label="viewing_pose")
    return ViewingPose(rig_id=rig_id, needed=True, work_mm=work_tuple, pose=pose, distance_mm=aim.distance_mm,
                       nearest_mm=nearest, depth_note=depth_note, tilt_deg=best.tilt_deg, view=best.view,
                       joints=best.joints, largest_rad=best.largest_rad, screened=screened,
                       closing_axis=choice.chosen, heading=choice)


def _arm_screen(arm: Any) -> tuple[Any, float, ArmReach | None, Sequence[float] | None, Sequence[float] | None]:
    """What ``arm`` says to screen a viewing pose against, best effort: its box, its gate's margin, its reach and
    window, the joints it stands at and where its tool is. ``None`` for anything it cannot say."""
    if arm is None:
        return None, 0.0, None, None, None
    config = getattr(arm, "config", None)
    box = getattr(config, "workspace_limits", None)
    if box is not None and not all(hasattr(box, key) for key in ("x_min", "x_max", "y_min", "y_max", "z_min",
                                                                  "z_max")):
        box = None
    margin = getattr(getattr(getattr(config, "safety", None), "limits", None), "workspace_margin_mm", 0.0)
    margin = float(margin) if isinstance(margin, (int, float)) else 0.0
    joints = _read(lambda: [float(v) for v in arm.get_joint_positions()])
    tool = _read(lambda: [float(v) for v in arm.get_tcp_pose().position_mm])
    return box, margin, reach_of(arm), joints, tool


def _screened(box: Any, margin: float, reach: ArmReach | None) -> str:
    """What a viewing pose was screened against, in words."""
    parts: list[str] = []
    if box is not None:
        points = "TCP and flange" if reach is not None else "TCP"
        parts.append(f"the workspace box less {margin:g} mm ({points})")
    if reach is not None:
        parts.append(f"the reach of a {reach.model}" + (" and the joint window its moves are chosen in"
                                                       if reach.lower_rad is not None else ""))
    return ", ".join(parts) if parts else "nothing: no arm said what to screen it against"


def _nearest_depth(rig: Any) -> tuple[float | None, str]:
    """The nearest depth ``rig``'s camera measures at its depth mode, in millimetres, and where that number comes from.

    Known for a RealSense the rig's body names (``body.model``, ``realsense_d415``) at the rig's ``depth_resolution``:
    Intel's figures, the ones the RealSense driver logs when it opens, not a measurement. ``None`` otherwise.
    """
    unknown = "its nearest measured depth is not known here, so the distance is not checked against it"
    model = getattr(getattr(rig, "body", None), "model", None)
    resolution = getattr(rig, "depth_resolution", None)
    if not isinstance(model, str) or "realsense" not in model.lower() or resolution is None:
        return None, unknown
    named = re.search(r"d\d{3}i?", model.lower())
    if named is None:
        return None, unknown
    from src.camera.setup.image_taking.rgbd import realsense_min_depth_mm

    width, height = int(resolution[0]), int(resolution[1])
    nearest = realsense_min_depth_mm(named.group(0).upper(), width)
    if nearest is None:
        return None, unknown
    return float(nearest), (f"Intel's figure for a {named.group(0).upper()} at a {width} x {height} depth mode, not "
                            "measured here")
