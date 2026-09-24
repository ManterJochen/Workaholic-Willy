"""
PoseProvider: vendor-neutral source of sweep stations.

Loads calibration / sampling stations from JSON files. A pose is a
:class:`src.geometry.Pose` in :attr:`Frame.BASE` (millimetres + XYZW
quaternion). A joint station is a :class:`JointStation`: joint angles the arm
is sent to with one judged joint move, for a station an operator taught on the
pendant, or by hand in a hand-guided calibration, rather than one a planner has
to find a configuration for. Nothing here generates a station: every station a
sweep visits is one somebody wrote down or guided the arm to.

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

A joint station may also carry ``tcp_pose``, ``{x, y, z, rx, ry, rz}``: where
the TCP stood when the station was taught, which a hand-guided calibration
writes beside the joints so a person can read the file. It is a note: the
joints are what runs, and it is never read as a pose.

A joint station is checked when the file is read, with no hardware: one unit
key and no pose keys beside it, six finite numbers, and units that can be what
they say. A ``joints_rad`` value beyond one full turn (2*pi) is refused as
probably degrees, a ``joints_deg`` value beyond 360 as beyond any joint, and a
``joints_deg`` record whose every value lies within 2*pi as probably radians.
Nothing rewrites a value: a station runs as written, or is refused, and the arm
itself turns each joint onto the full turn nearest where it stands inside its
joint window when it moves.

Stations the sweep commands
---------------------------
:meth:`PoseProvider.screen` is the box and diversity screen a station passes
before a sweep commands it. For a joint station it judges the grasp centre the
arm's own forward kinematics puts at those joints, because the joint move
itself skips the Cartesian box.
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
from src.geometry.serialization import pose_from_dict
from src.robot.core.joint_positions import JointPositions
from src.robot.drivers.ur.pose_adapter import urpose_to_pose
from src.robot.safety.workspace import WorkspaceGuard
from src.utility.io import load_json

__all__ = [
    "JointStation",
    "PoseProvider",
    "Station",
    "StationScreen",
    "load_stations",
    "station_record",
    "write_stations",
]

#: The joints of a joint station, in the order a record writes them.
JOINT_NAMES: tuple[str, ...] = ("base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3")
#: The keys that make a record a pose. A joint station carries none of them.
_POSE_KEYS = frozenset({"x", "y", "z", "rx", "ry", "rz", "position_mm", "quaternion_xyzw"})
#: One full turn, the furthest any joint of the arms this reads goes either way.
_FULL_TURN_RAD = 2.0 * math.pi
#: Rounding slack at a full turn: 360 degrees converts to 2*pi and must not be refused for the last bit.
_TURN_SLACK_RAD = 1e-9


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


#: What a sweep visits: a pose to plan to, or joint angles to move to.
Station: TypeAlias = Pose | JointStation


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


def station_record(label: str, joints: JointPositions, tcp: Pose) -> dict[str, Any]:
    """One joint station as a stations file writes it: the joints in degrees, and where they put the TCP as a note.

    What :func:`load_stations` reads back as a :class:`JointStation` with this label and these joints; ``tcp_pose``
    (``x, y, z`` mm, ``rx, ry, rz`` axis-angle rad, BASE) is for the person reading the file and is never run.
    """
    position = [float(v) for v in tcp.position_mm]
    rotation = [float(v) for v in tcp.axis_angle_rad()]
    return {
        "label": label,
        "joints_deg": [round(math.degrees(float(v)), 4) for v in joints.values],
        "tcp_pose": {**{k: round(v, 2) for k, v in zip(("x", "y", "z"), position)},
                     **{k: round(v, 6) for k, v in zip(("rx", "ry", "rz"), rotation)}},
    }


def write_stations(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    """Write a stations file whole, through a file beside it that then replaces it, so no reader meets half a file."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    partial.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")
    partial.replace(target)
    return target


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
    """The box and diversity screen a sweep's stations pass, holding its limits and thresholds. It generates nothing.

    Parameters
    ----------
    guard
        :class:`WorkspaceGuard` whose limits and diversity thresholds every screen copies.
    """

    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard

    def screen(self) -> StationScreen:
        """A box and diversity screen with an empty history, holding this provider's limits and thresholds."""
        return StationScreen(WorkspaceGuard(
            limits=self.guard.limits,
            min_distance_mm=self.guard.min_distance_mm,
            min_angle_deg=self.guard.min_angle_deg,
        ))
