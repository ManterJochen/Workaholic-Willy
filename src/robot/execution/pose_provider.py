"""
PoseProvider: vendor-neutral source of sweep stations.

Loads calibration / sampling stations from JSON files or generates poses
automatically inside the configured workspace. A pose is a
:class:`src.geometry.Pose` in :attr:`Frame.BASE` (millimetres + XYZW
quaternion). A joint station is a :class:`JointStation`: joint angles the arm
is sent to with one judged joint move, for a station an operator taught on the
pendant rather than one a planner has to find a configuration for.

Accepted record layouts
-----------------------
A file is a top-level list, read in order, and the three shapes may be mixed:

* URPose-shaped records (``{x, y, z, rx, ry, rz, label}``, mm +
  axis-angle rad), detected via the ``rx``/``ry``/``rz`` keys,
* Pose-shaped records (via :func:`pose_from_dict`), and
* joint stations, ``{"label": ..., "joints_deg": [six values]}`` or
  ``{"joints_rad": [six values]}``, one value per joint from the base to the
  last wrist joint (on a UR: base, shoulder, elbow, wrist 1, wrist 2, wrist 3,
  the order the pendant lists them in).

A joint station is checked when the file is read, with no hardware: one unit
key and no pose keys beside it, six finite numbers, and units that can be what
they say. A ``joints_rad`` value beyond one full turn (2*pi) is refused as
probably degrees, a ``joints_deg`` value beyond 360 as beyond any joint, and a
``joints_deg`` record whose every value lies within 2*pi as probably radians.
Nothing rewrites a value: a station runs as written, or is refused.

Stations the sweep commands
---------------------------
:meth:`PoseProvider.screen` is the box and diversity screen a station passes
before a sweep commands it. For a joint station it judges the grasp centre the
arm's own forward kinematics puts at those joints, because the joint move
itself skips the Cartesian box. :func:`order_by_bearing` orders generated
poses round the base, and :func:`joint_hops` names the legs between joint
stations that turn a joint more than half a turn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.exceptions import GeometryError
from src.geometry.quaternion import from_axis_angle
from src.geometry.serialization import pose_from_dict
from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from src.robot.core.joint_positions import JointPositions
from src.robot.drivers.ur.pose_adapter import urpose_to_pose
from src.robot.safety.workspace import WorkspaceGuard
from src.utility.io import load_json
from src.utility.log_cfg import create_logger

__all__ = [
    "JointStation",
    "PoseProvider",
    "SkippedStation",
    "Station",
    "StationScreen",
    "joint_hops",
    "load_stations",
    "order_by_bearing",
]


# Default base orientation: TCP pointing straight down (Z = -base.Z).
# Equivalent to URPose default ``[0, pi, 0]`` axis-angle.
_DEFAULT_DOWN_AXIS_ANGLE_RAD: list[float] = [0.0, float(np.pi), 0.0]

#: The joints of a joint station, in the order a record writes them.
JOINT_NAMES: tuple[str, ...] = ("base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3")
#: The keys that make a record a pose. A joint station carries none of them.
_POSE_KEYS = frozenset({"x", "y", "z", "rx", "ry", "rz", "position_mm", "quaternion_xyzw"})
#: One full turn, the furthest any joint of the arms this reads goes either way.
_FULL_TURN_RAD = 2.0 * math.pi
#: Rounding slack at a full turn: 360 degrees converts to 2*pi and must not be refused for the last bit.
_TURN_SLACK_RAD = 1e-9
#: Half a turn. A joint that turns further between two stations has a twin a full turn away that is nearer.
_HALF_TURN_DEG = 180.0


@dataclass(frozen=True, slots=True)
class JointStation:
    """A sweep station given as joint angles, in radians, from the base to the last wrist joint.

    The sweep sends the arm there with one joint move (``RobotArm.move_to_joints``), which the arm
    judges as it judges any joint move, and boxes the grasp centre the arm's forward kinematics puts
    at these joints before it does. ``written_as`` is the key the file used (``joints_deg`` or
    ``joints_rad``), empty for a station built in code.

    Every value lies within one full turn: a value beyond it is refused here, because no joint of
    the arms this drives reaches it, and in a file it is how degrees written as radians read.
    """

    joints: JointPositions
    label: str | None = None
    written_as: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.joints, JointPositions):
            raise TypeError(f"JointStation.joints is a JointPositions, not {type(self.joints).__name__}")
        beyond = [float(v) for v in self.joints.values if abs(float(v)) > _FULL_TURN_RAD + _TURN_SLACK_RAD]
        if beyond:
            raise ValueError(
                f"JointStation {self.label or '<unlabeled>'!r}: {beyond[0]:.4g} rad is beyond one full turn "
                "(2*pi), which no joint reaches; joint angles are radians")

    @property
    def degrees(self) -> tuple[float, ...]:
        """The joints in degrees, as a pendant shows them."""
        return tuple(float(math.degrees(float(v))) for v in self.joints.values)


@dataclass(frozen=True, slots=True)
class SkippedStation:
    """A station a sweep reports and never moves to: a view that was planned and could not be aimed or reached.

    An aimed sweep (``camera_aim``) keeps every view it planned in the list it runs, so the pose log accounts for
    each of them in order: the ones it moves to, and these, with the ``reason`` and one sentence of ``detail`` that
    kept them from being commanded. Nothing is sent for one.
    """

    label: str
    reason: str
    detail: str = ""


#: What a sweep visits: a pose to plan to, or joint angles to move to.
Station: TypeAlias = Pose | JointStation


def _axis_angle_to_quat(axis_angle_rad: Sequence[float]) -> np.ndarray:
    """Build a unit XYZW quaternion from an axis-angle 3-vector (rad)."""
    return from_axis_angle(np.asarray(axis_angle_rad, dtype=np.float64))


def _record_to_pose(record: dict, *, default_label: str = "") -> Pose:
    """Convert one JSON record into a :class:`Pose` (Frame.BASE)."""
    if isinstance(record, dict) and {"rx", "ry", "rz"} <= record.keys():
        # URPose-shaped record. Coerce via URPose to Pose to reuse the
        # canonical conversion path (axis-angle rad to quaternion).
        from src.robot.drivers.ur.pose import URPose

        urpose = URPose(
            x=float(record["x"]),
            y=float(record["y"]),
            z=float(record["z"]),
            rx=float(record["rx"]),
            ry=float(record["ry"]),
            rz=float(record["rz"]),
            label=str(record.get("label", default_label)),
        )
        return urpose_to_pose(urpose, frame=Frame.BASE, label=urpose.label or None)
    if isinstance(record, dict) and "position_mm" in record and "quaternion_xyzw" in record:
        return pose_from_dict(record)
    raise ValueError(
        "Unrecognised pose JSON record. Expected URPose-shaped "
        "{x,y,z,rx,ry,rz} or Pose-shaped {position_mm, quaternion_xyzw, ...}, or a joint station "
        '{"joints_deg": [six values]} or {"joints_rad": [six values]}; '
        f"got keys: {sorted(record.keys()) if isinstance(record, dict) else type(record).__name__}"
    )


def _record_to_joint_station(record: dict, *, default_label: str) -> JointStation:
    """One joint-station record, checked for its keys, its count, its numbers and its units."""
    units = [key for key in ("joints_deg", "joints_rad") if key in record]
    if len(units) > 1:
        raise ValueError("carries both joints_deg and joints_rad; write one, in the unit the numbers are in")
    (key,) = units
    mixed = sorted(_POSE_KEYS & record.keys())
    if mixed:
        raise ValueError(f"carries {key} and the pose keys {mixed}; a record is a pose or a joint station, not both")
    written = record[key]
    if not isinstance(written, (list, tuple)) or len(written) != len(JOINT_NAMES):
        count = len(written) if isinstance(written, (list, tuple)) else type(written).__name__
        raise ValueError(f"{key} holds {count} values; a joint station holds {len(JOINT_NAMES)}, one per joint "
                         f"from the base to the last wrist joint ({', '.join(JOINT_NAMES)})")
    values: list[float] = []
    for name, value in zip(JOINT_NAMES, written):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} gives {name} as {value!r}, which is not a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{key} gives {name} as {value!r}, which is not a finite number")
        values.append(float(value))
    if key == "joints_rad":
        for name, value in zip(JOINT_NAMES, values):
            if abs(value) > _FULL_TURN_RAD + _TURN_SLACK_RAD:
                raise ValueError(
                    f"joints_rad gives {name} as {value:g}, beyond one full turn (2*pi) in radians, which no joint "
                    "reaches and which reads as degrees; write the record as joints_deg if it is")
        radians = values
    else:
        for name, value in zip(JOINT_NAMES, values):
            if abs(value) > 360.0:
                raise ValueError(f"joints_deg gives {name} as {value:g}, beyond one full turn (360), which no "
                                 "joint reaches")
        if max(abs(value) for value in values) > 0.0 and all(abs(value) <= _FULL_TURN_RAD for value in values):
            raise ValueError(
                f"joints_deg holds {values}, every value within 2*pi, which reads as radians: a station with every "
                "joint within 6.3 degrees of zero is not one a sweep looks from; write the record as joints_rad if "
                "it is")
        radians = [math.radians(value) for value in values]
    label = record.get("label", default_label)
    return JointStation(JointPositions(radians), label=str(label) if label is not None else default_label,
                        written_as=key)


def _record_to_station(record: Any, *, default_label: str) -> Station:
    """One record of a stations file: a joint station when it names joints, a pose in BASE otherwise."""
    if isinstance(record, dict) and ({"joints_deg", "joints_rad"} & record.keys()):
        return _record_to_joint_station(record, default_label=default_label)
    pose = _record_to_pose(record, default_label=default_label)
    if pose.frame is not Frame.BASE:
        raise ValueError(f"is a pose in {pose.frame.value!r}; a station is in the robot's base frame")
    return pose


def load_stations(path: str | Path) -> list[Station]:
    """Every record of a stations file, in file order, as a :class:`Pose` or a :class:`JointStation`.

    Reads the file and checks each record, and filters nothing: the box and the diversity rule
    are the sweep's (:meth:`PoseProvider.screen`), because a joint station's grasp centre is known
    only from the arm's forward kinematics. A record's label defaults to ``pose_<i>``, its 0-based
    place in the file.

    Raises ``ValueError`` naming the record for a file that is not a list, is empty, or holds a
    record that is neither shape or fails its checks, and ``OSError`` for a file that cannot be read.
    """
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(
            f"Pose JSON at '{path}' must be a top-level list; got {type(records).__name__}"
        )
    if not records:
        raise ValueError(f"Pose JSON at '{path}' holds no stations")
    stations: list[Station] = []
    for index, record in enumerate(records):
        default = f"pose_{index}"
        try:
            stations.append(_record_to_station(record, default_label=default))
        except (ValueError, TypeError, KeyError, GeometryError) as exc:
            named = record.get("label", default) if isinstance(record, dict) else default
            raise ValueError(f"Pose JSON at '{path}', record {index} ({str(named)!r}): {exc}") from None
    return stations


def order_by_bearing(poses: Sequence[Pose], *, start_mm: Sequence[float] | None = None) -> list[Pose]:
    """``poses`` in the order a base turning one way round visits them: by bearing, ``atan2(y, x)``.

    The order starts after the widest gap between neighbouring bearings, so a set that straddles
    the line behind the base is not cut in two. With ``start_mm`` (where the tool stands now) it
    runs from whichever end lies nearer that bearing. The poses themselves are unchanged.

    A sweep of random positions in the order they were drawn swings the base back and forth
    across the box. Measured on a 22-pose auto sweep planned with cuRobo on a UR10 descriptor and
    its table, as the driver plans them (the calibration chain's GPU probe of 2026-09-23, not a
    physical arm): 188.8 and 201.2 rad of joint travel as drawn, 77.1 and 80.0 rad in bearing
    order.
    """
    ordered = sorted(poses, key=_bearing)
    if len(ordered) < 2:
        return list(ordered)
    bearings = [_bearing(pose) for pose in ordered]
    gaps = [(bearings[(i + 1) % len(bearings)] - bearings[i]) % _FULL_TURN_RAD for i in range(len(bearings))]
    cut = int(np.argmax(gaps)) + 1
    ordered = ordered[cut:] + ordered[:cut]
    if start_mm is not None:
        here = math.atan2(float(start_mm[1]), float(start_mm[0]))
        if _apart(here, _bearing(ordered[-1])) < _apart(here, _bearing(ordered[0])):
            ordered.reverse()
    return ordered


def _bearing(pose: Pose) -> float:
    position = pose.position_mm
    return math.atan2(float(position[1]), float(position[0]))


def _apart(a: float, b: float) -> float:
    """How far apart two bearings are, the short way round, in radians."""
    return abs((a - b + math.pi) % _FULL_TURN_RAD - math.pi)


def joint_hop(before: JointPositions, after: JointPositions) -> str:
    """The joints that turn more than half a turn from ``before`` to ``after``, in one clause; ``""`` for none.

    A joint that turns further than half a turn has, a full turn away, a value that is nearer; the
    station still runs as written, because the value a caller wrote is the value that was judged.
    """
    turns = [(name, math.degrees(float(b) - float(a)))
             for name, a, b in zip(JOINT_NAMES, before.values, after.values)]
    large = [f"{name} turns {degrees:+.0f} deg" for name, degrees in turns if abs(degrees) > _HALF_TURN_DEG]
    return ", ".join(large)


def joint_hops(stations: Sequence[Station]) -> list[str]:
    """One sentence per leg between two neighbouring joint stations that turns a joint more than half a turn.

    Only neighbouring joint stations are compared: the configuration a pose ends in is the planner's,
    and not known before it plans.
    """
    said: list[str] = []
    for before, after in zip(stations, stations[1:]):
        if isinstance(before, JointStation) and isinstance(after, JointStation):
            hop = joint_hop(before.joints, after.joints)
            if hop:
                said.append(f"{before.label or '<unlabeled>'!r} to {after.label or '<unlabeled>'!r}: {hop}, more "
                            "than half a turn; it runs as written, the long way round")
    return said


class StationScreen:
    """The box and diversity screen a sweep's stations pass before it commands them, in order.

    Built by :meth:`PoseProvider.screen` with an empty history. :meth:`refusal` says why a station
    would be skipped; :meth:`keep` records one that was not, so the next is compared with it.
    """

    def __init__(self, guard: WorkspaceGuard) -> None:
        self._guard = guard

    def refusal(self, pose: Pose, *, what: str = "the pose") -> "tuple[str, str] | None":
        """``("outside_workspace", why)``, ``("too_similar", why)``, or ``None`` where ``pose`` may be commanded.

        ``what`` names the point in the sentence, such as ``"the grasp centre at these joints"``.
        """
        local_guard = self._guard
        x, y, z = (float(v) for v in pose.position_mm)
        if not local_guard.is_inside_workspace(pose):
            box = local_guard.limits
            return "outside_workspace", (
                f"{what} ({x:.1f}, {y:.1f}, {z:.1f}) mm is outside workspace_limits (x {box.x_min:.1f} to "
                f"{box.x_max:.1f}, y {box.y_min:.1f} to {box.y_max:.1f}, z {box.z_min:.1f} to {box.z_max:.1f} mm), "
                "so nothing moved")
        if local_guard.validate(pose):
            return None
        for before in local_guard.accepted_poses:
            distance = float(np.linalg.norm(pose.position_mm - before.position_mm))
            angle = float(np.degrees(before.angle_to(pose)))
            if distance < local_guard.min_distance_mm and angle < local_guard.min_angle_deg:
                return "too_similar", (
                    f"{what} is {distance:.1f} mm and {angle:.1f} deg from {before.label or '<unlabeled>'!r}, kept "
                    f"before it, and a station needs {local_guard.min_distance_mm:g} mm or "
                    f"{local_guard.min_angle_deg:g} deg from every one kept, so nothing moved")
        return "too_similar", f"{what} is too close to a station kept before it, so nothing moved"

    def keep(self, pose: Pose) -> None:
        """Record ``pose`` as commanded, for the diversity of the stations after it."""
        self._guard.accept(pose)


class PoseProvider:
    """Source of validated calibration / sampling poses.

    Parameters
    ----------
    guard
        :class:`WorkspaceGuard` used to validate every emitted pose.
    """

    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard
        self.logger = create_logger("PoseProvider", ROBOT_LOG_FILE, log_dir=ROBOT_LOG_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_json(self, path: str | Path) -> list[Pose]:
        """Load poses from a JSON file and validate each one.

        A file of poses only: a joint station's grasp centre needs the arm's forward kinematics,
        so a file that holds one is refused here and runs through
        :meth:`CalibrationRoutine.run_from_json`, which screens every station as it comes.

        Returns
        -------
        list of Pose
            Poses (Frame.BASE) that passed the workspace + diversity
            checks, in input order.

        Raises
        ------
        ValueError
            If the file is empty, malformed, holds a joint station, or none of the loaded poses
            pass validation.
        """
        stations = load_stations(path)
        joints = [station.label for station in stations if isinstance(station, JointStation)]
        if joints:
            raise ValueError(
                f"Pose JSON at '{path}' holds joint stations ({', '.join(repr(label) for label in joints)}), whose "
                "grasp centre only the arm's forward kinematics knows; run it with CalibrationRoutine.run_from_json")
        poses = [station for station in stations if isinstance(station, Pose)]
        self.logger.info("Loaded %d poses from %s.", len(poses), path)

        valid = self._filter_valid(poses)
        if not valid:
            raise ValueError(
                f"None of the {len(poses)} poses from '{path}' "
                "passed the workspace + diversity check."
            )
        return valid

    def screen(self) -> StationScreen:
        """A box and diversity screen with an empty history, holding this provider's limits and thresholds."""
        return StationScreen(self._fresh_local_guard())

    def generate(
        self,
        n: int,
        *,
        base_orientation: list[float] | None = None,
        orientation_spread_deg: float = 15.0,
        max_attempts_per_pose: int = 200,
        seed: int | None = None,
        start_mm: Sequence[float] | None = None,
    ) -> list[Pose]:
        """Generate ``n`` diverse poses inside the workspace.

        Parameters
        ----------
        n
            Number of poses to generate.
        base_orientation
            Reference TCP orientation as an axis-angle 3-vector in
            radians (e.g. ``[0, pi, 0]`` for "tool pointing down").
            Per-pose perturbations are added on top. ``None`` uses
            ``_DEFAULT_DOWN_AXIS_ANGLE_RAD``, which points the tool down.
        orientation_spread_deg
            Half-amplitude (deg) of the per-axis uniform perturbation
            applied around ``base_orientation``.
        max_attempts_per_pose
            How many random samples to try per generated pose before
            giving up and raising :class:`RuntimeError`.
        seed
            RNG seed for reproducibility.
        start_mm
            Where the tool stands now, in BASE millimetres, or ``None``. The order starts at the
            end of the round nearer it (:func:`order_by_bearing`).

        Returns
        -------
        list of Pose
            ``n`` poses, all inside the workspace, sufficiently diverse
            from each other, in the order a base turning one way round
            visits them, labelled ``auto_0`` on in that order. Diversity
            is between every pair, so the order changes none of it.
        """
        rng = np.random.default_rng(seed)
        limits = self.guard.limits

        base_axis_angle = (
            list(base_orientation)
            if base_orientation is not None
            else list(_DEFAULT_DOWN_AXIS_ANGLE_RAD)
        )

        spread_rad = np.radians(orientation_spread_deg)
        poses: list[Pose] = []
        local_guard = self._fresh_local_guard()

        for i in range(n):
            found = False
            for _ in range(max_attempts_per_pose):
                x = rng.uniform(limits.x_min, limits.x_max)
                y = rng.uniform(limits.y_min, limits.y_max)
                z = rng.uniform(limits.z_min, limits.z_max)

                rx = base_axis_angle[0] + rng.uniform(-spread_rad, spread_rad)
                ry = base_axis_angle[1] + rng.uniform(-spread_rad, spread_rad)
                rz = base_axis_angle[2] + rng.uniform(-spread_rad, spread_rad)

                quat = _axis_angle_to_quat([rx, ry, rz])
                candidate = Pose(
                    position_mm=np.array([x, y, z], dtype=np.float64),
                    quaternion_xyzw=quat,
                    frame=Frame.BASE,
                    label=f"drawn_{i}",
                )

                if local_guard.validate(candidate):
                    local_guard.accept(candidate)
                    poses.append(candidate)
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Could not generate pose #{i + 1}/{n} after "
                    f"{max_attempts_per_pose} attempts. "
                    f"Try widening the workspace or lowering diversity thresholds."
                )

        ordered = [pose.with_label(f"auto_{index}")
                   for index, pose in enumerate(order_by_bearing(poses, start_mm=start_mm))]
        self.logger.info("Generated %d diverse poses, ordered by bearing round the base.", len(ordered))
        return ordered

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fresh_local_guard(self) -> WorkspaceGuard:
        """Return a guard with the same limits/thresholds but empty state."""
        return WorkspaceGuard(
            limits=self.guard.limits,
            min_distance_mm=self.guard.min_distance_mm,
            min_angle_deg=self.guard.min_angle_deg,
        )

    def _filter_valid(self, poses: list[Pose]) -> list[Pose]:
        """Return poses that pass workspace + diversity checks, in order."""
        valid: list[Pose] = []
        local_guard = self._fresh_local_guard()
        for pose in poses:
            if local_guard.validate(pose):
                local_guard.accept(pose)
                valid.append(pose)
            else:
                self.logger.warning(
                    "Pose '%s' rejected (workspace or diversity).",
                    pose.label or "<unlabeled>",
                )
        self.logger.info("%d / %d poses accepted.", len(valid), len(poses))
        return valid
