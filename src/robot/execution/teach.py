"""Joint poses taught by hand: a person stands the arm where a pose should be, and Enter takes the line down.

    from src.config import load_tree
    from src.robot.execution.robot import Robot
    from src.robot.execution.teach import teach_poses

    tree = load_tree()
    robot = Robot.from_config(tree.robot, gripper=None)        # the arm alone, as a calibration builds it
    with robot.connected():                                    # the cell's lock, then the arm
        taught = teach_poses(robot, tree=tree)                 # for the primary rig: the payload, then a name per pose
    for pose in taught:
        print(pose.line())            # LOOK_1 = JointPositions.deg(-45.0, -100.2, ...)  # rig 'wrist', eye in hand

    with Camera.from_tree(tree, rig_id="wrist") as camera, robot.connected():
        teach_poses(robot, tree=tree, camera=camera)           # for that camera's rig, with a window of what it sees

A program declares the joints it looks from, and any joint pose it goes to, as ``JointPositions.deg(...)`` lines
(``PickRun.from_cell(cell, look=[...])``). Those numbers used to be copied off the pendant or off
``python -m src.robot.drivers.ur --where``; here a person stands the arm there by hand, and the line is printed, one
per pose, to paste as it is. What happens, in order:

1. The arm has to offer hand guiding (:class:`~src.robot.core.freedrive.SupportsFreedrive`; the UR teach mode is
   one) and be connected. An arm that offers none is refused with a sentence naming its vendor, and one that is not
   connected is refused too, both before anything is asked. The file the poses go to is read next: one that is not a
   list of taught poses is refused and left as it is, because writing it would lose what it holds.
2. The payload the controller compensates for is shown and has to be confirmed (``HandGuide.confirm_payload``)
   before the arm is first freed: a wrong one makes the freed arm sink or rise in a person's hands.
3. One hand-guiding session for the whole run. For each pose the console asks for a name while the arm holds (Enter
   takes ``LOOK_1``, ``LOOK_2``, ..., the first not taken in this run; ``q`` finishes), then frees the arm, and
   ``HandGuide.wait`` samples it until the person presses Enter and the arm has stood still for half a second, the
   stillness gate. Then the arm is held, its joints and its TCP are read, the line to paste is printed, and the pose
   is added to the file. ``s`` holds the arm, teaches nothing and asks for the name again.
4. Leaving the session holds the arm: after ``q``, on Ctrl-C and on an error alike. Every pose taught before is
   printed and in the file by then.

Nothing here moves the arm by itself. It moves only while a person moves it, and between poses it holds where it was
left, so no hands-off countdown comes before anything: ``HandGuide.hands_off`` guards an arm that is about to drive
by itself, and nothing drives here.

Boundaries are said, never enforced (:class:`~src.robot.execution.hand_guiding.HandGuidingLimits`). A pose outside
the cable window, the joint-limit guard's window (half a turn either side of home where the cell keeps
``safety.joint_limits.within_half_turn_of_home``), or outside the workspace box is said once per crossing on the
console, and in red in the camera's window where one is shown, and Enter does not capture there. The arm is never
held, locked or stopped for it: an arm that locks while a person pushes it is how a hand gets caught.

A pose belongs to one camera (the owner, 2026-09-24 evening): an eye-in-hand camera and an eye-to-hand camera are
calibrated differently, and each camera gets poses of its own, so every run teaches for one rig of
``camera.cameras.rigs``, resolved before anything is asked: the rig of the one camera handed in, or else the one
``for_rig`` names in the tree, its primary by default, whether or not a window shows. A camera of another rig than
``for_rig`` is refused, and so is a run that can name no rig. Every record keeps the rig and the mounting the tree
declares for it (``eye_in_hand`` or ``eye_to_hand``; none for a rig not calibrated and carrying no body), and every
line to paste says both in a comment: ``LOOK_1 = JointPositions.deg(...)  # rig 'wrist', eye in hand``.
:func:`read_taught_poses` keeps one rig's poses where asked; a pose taught before a pose had a rig reads as rig unknown,
and its record is written back as it was.

What the camera sees, optionally (``camera=``, the one open :class:`~src.camera.Camera` the poses are taught for): a
window beside the console, the cell's live view (``src/camera/live_view.py``) with that camera alone, shows what it
sees, live, with the guide: the pose being taught and how many are, red with the reason outside a boundary, the way a
calibration guided by hand shows it. Look poses are what it is for: a wrist camera is guided by hand to where it sees
the work, and a fixed camera shows where the arm stands in the cell. Enter or Space in the window captures, ``q`` or
ESC finishes, and closing it finishes; the arm is held on every way out. Only the caller's thread calls the arm: the
view's thread reads the camera, through its display path, and draws, nothing else. The window opens once the payload
is confirmed at the console and is closed on every way out, before the caller gives the camera back. Where no window
can show, or the camera gives no frame, one console line says why and the poses are taught at the console, still for
that rig: teaching never needs the camera.

A name is pasted as Python, so it is an ASCII Python name and no keyword. A name given again in the same run takes the
place of the pose taught under it before in what the run returns, and the file keeps both.

The file, ``logs/taught_poses.json`` unless the caller names another, is a JSON list of one record per pose, in the
order taught::

    {"name": "LOOK_1",
     "joints_deg": [-45.0, -100.2, -110.0, -60.0, 90.0, 0.0],
     "tcp": {"frame": "base", "x_mm": 450.0, "y_mm": -120.0, "z_mm": 300.0,
             "rotation_vector_deg": [0.0, 170.0, 0.0]},
     "taught_at": "2026-09-24T21:03:12+02:00",
     "rig": "wrist",
     "mounting": "eye_in_hand"}

``joints_deg`` are the joints from the base to the last wrist joint as the arm reported them once it held, to a
ten-thousandth of a degree (the printed line rounds them to a tenth); ``tcp`` is where that put the TCP in the robot's
base frame, to a hundredth of a millimetre, its rotation as a rotation vector in degrees; ``taught_at`` is the local
time with its offset; ``rig`` is the rig the pose was taught for and ``mounting`` how the tree mounted it then
(``null`` where it declared none). A record written before poses had a rig carries neither and still reads. Each pose
is added as it is taught: the file is read, the pose appended, and the whole written to a file beside it that then
replaces it, so no pose taught before is lost, none is written back in another shape, and no reader meets half a
file.

Threads: everything that touches the arm runs on the caller's thread, as in :mod:`.hand_guiding`. Nothing here has
run beside a physical arm; the UR teach mode behind ``SupportsFreedrive`` is the freedrive workstream's.
"""

from __future__ import annotations

import json
import keyword
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle
from src.robot.core import JointPositions, RobotCapabilities
from src.robot.core.freedrive import FreedriveSample, SupportsFreedrive
from src.robot.drivers.ur.bench import where_lines
from src.robot.execution.hand_guiding import (
    CAPTURE,
    EOF,
    FINISH,
    HandGuide,
    HandGuidingLimits,
    HandGuidingRefused,
    OperatorConsole,
    sample_pose,
)
from src.robot.execution.pose_provider import write_stations

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.camera.live_view import LiveView
    from src.robot.core import RobotArm
    from src.robot.execution.robot import Robot

__all__ = [
    "DEFAULT_PREFIX",
    "DEFAULT_STORE",
    "TaughtPose",
    "add_taught_pose",
    "read_taught_poses",
    "teach_poses",
]

#: Where the poses go unless the caller names another file, relative to the directory the program runs in.
DEFAULT_STORE = "logs/taught_poses.json"
#: What Enter calls a pose: ``LOOK_1``, ``LOOK_2``, ..., the look poses a camera pick declares.
DEFAULT_PREFIX = "LOOK"

#: What finishes the run at a name prompt, beside input that has ended.
_FINISH_WORDS = ("q", "quit", "finish")
#: What a name has to be, said where one is not.
_NAME_RULE = ("a name is pasted as Python, so it is letters, digits and underscores, and it does not start with a "
              "digit (LOOK_1, look_left)")
#: The six faces of a workspace box, as ``HandGuidingLimits`` reads them.
_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
#: What the camera window's lines say about teaching: after an error switched it off, and after the person closed it.
_GOES_ON = "teaching goes on at the console"
_ON_CLOSE = "teaching finishes, and the arm is held"
#: The mountings a rig's calibration declares, as a record keeps them and a line says them.
_MOUNTINGS = {"eye_in_hand": "eye in hand", "eye_to_hand": "eye to hand"}


@dataclass(frozen=True, slots=True)
class TaughtPose:
    """One pose taught by hand: its name, the joints the arm held at, where they put the TCP, and the camera it is for.

    ``joints`` are radians, from the base to the last wrist joint, as the arm reported them once it held; ``tcp`` is
    the TCP there, in the robot's base frame, in millimetres. ``taught_at`` is when, local time in ISO 8601 with its
    offset, empty where nobody said. ``rig`` is the rig of ``camera.cameras.rigs`` the pose was taught for, and
    ``mounting`` how the tree mounted it then, ``eye_in_hand`` or ``eye_to_hand``; ``None`` where the tree declared no
    mounting, and both ``None`` for a pose taught before poses had a rig. :meth:`line` is what a program pastes,
    :meth:`to_dict` the record the file keeps.
    """

    name: str
    joints: JointPositions
    tcp: Pose
    taught_at: str = ""
    rig: str | None = None
    mounting: str | None = None

    @classmethod
    def from_sample(cls, name: str, sample: FreedriveSample, *, rig: str | None = None,
                    mounting: str | None = None) -> "TaughtPose":
        """The pose one reading of the held arm gives, taught now for ``rig``, mounted as ``mounting``."""
        return cls(name=name, joints=JointPositions(sample.joints_rad), tcp=sample_pose(sample), taught_at=_now(),
                   rig=rig, mounting=mounting)

    @classmethod
    def from_dict(cls, record: Any) -> "TaughtPose":
        """A record :meth:`to_dict` wrote, read back; ``ValueError`` says what a record that is none lacks.

        Keys the record carries beside its own are left alone, so a note written into the file by hand reads. A record
        with no ``rig``, one taught before poses had a rig, reads as a pose of no known rig; a ``rig`` that names no rig
        and a ``mounting`` that is neither ``eye_in_hand`` nor ``eye_to_hand`` are refused.
        """
        if not isinstance(record, dict):
            raise ValueError(f"it is a JSON {_json_kind(record)}, and a taught pose is an object")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("it has no name")
        joints = _numbers(record.get("joints_deg"), "joints_deg")
        tcp = record.get("tcp")
        if not isinstance(tcp, dict):
            raise ValueError(f"{name!r} has no tcp")
        position = [_number(tcp.get(key), f"tcp.{key}") for key in ("x_mm", "y_mm", "z_mm")]
        rotation = _numbers(tcp.get("rotation_vector_deg"), "tcp.rotation_vector_deg", size=3)
        try:
            frame = Frame(tcp.get("frame", Frame.BASE.value))
        except ValueError:
            raise ValueError(f"{name!r} has tcp.frame {tcp.get('frame')!r}, which is no frame") from None
        rig = record.get("rig")
        if rig is not None and (not isinstance(rig, str) or not rig):
            raise ValueError(f"{name!r} has rig {rig!r}, which names no rig")
        mounting = record.get("mounting")
        if mounting is not None and mounting not in _MOUNTINGS:
            raise ValueError(f"{name!r} has mounting {mounting!r}, which is neither eye_in_hand nor eye_to_hand")
        taught_at = record.get("taught_at")
        return cls(
            name=name,
            joints=JointPositions(np.radians(np.asarray(joints, dtype=np.float64))),
            tcp=Pose(position_mm=np.asarray(position, dtype=np.float64),
                     quaternion_xyzw=from_axis_angle(np.radians(np.asarray(rotation, dtype=np.float64))), frame=frame),
            taught_at=str(taught_at) if taught_at is not None else "",
            rig=rig,
            mounting=mounting,
        )

    @property
    def joints_deg(self) -> tuple[float, ...]:
        """The joints in degrees, as the arm reported them."""
        return tuple(math.degrees(float(value)) for value in self.joints.values)

    def taught_for(self) -> str:
        """The camera the pose belongs to, in words: ``rig 'wrist', eye in hand``, ``rig 'side', mounting not
        declared``, or ``rig unknown`` for a pose taught before poses had a rig."""
        if self.rig is None:
            return "rig unknown"
        return f"rig {self.rig!r}, {_mounted(self.mounting)}"

    def line(self) -> str:
        """The line to paste: ``LOOK_1 = JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)  # rig 'wrist',
        eye in hand``.

        One decimal of a degree per joint and never ``-0.0``: the line ``python -m src.robot.drivers.ur --where``
        prints (``where_lines``), with the name in front and the camera the pose belongs to in a comment after it
        (:meth:`taught_for`), so a pasted look says which camera it is a look of.
        """
        return f"{self.name} = {where_lines(self.joints.tolist(), self.tcp)[0]}  # {self.taught_for()}"

    def tcp_line(self) -> str:
        """Where the TCP stood, as ``--where`` says it: ``TCP (base)  x 450.0  y -120.0  z 300.0 mm  rx ...``."""
        return where_lines(self.joints.tolist(), self.tcp)[1]

    def to_dict(self) -> dict[str, Any]:
        """The record the file keeps (the module shows one), ``json.dumps`` safe, with no ``-0.0`` in it."""
        x, y, z = (float(value) for value in self.tcp.position_mm)
        rotation = [_rounded(math.degrees(float(value)), 4) for value in self.tcp.axis_angle_rad()]
        return {
            "name": self.name,
            "joints_deg": [_rounded(value, 4) for value in self.joints_deg],
            "tcp": {"frame": self.tcp.frame.value, "x_mm": _rounded(x, 2), "y_mm": _rounded(y, 2),
                    "z_mm": _rounded(z, 2), "rotation_vector_deg": rotation},
            "taught_at": self.taught_at,
            "rig": self.rig,
            "mounting": self.mounting,
        }


# --- the file ----------------------------------------------------------------------------------------------------


def read_taught_poses(path: str | Path = DEFAULT_STORE, *, rig: str | None = None) -> tuple[TaughtPose, ...]:
    """Every pose a file of taught poses holds, in the order taught; ``()`` where there is no file yet.

    ``rig`` keeps the poses taught for that rig alone: a program that looks through one camera reads that camera's
    looks. A pose taught before poses had a rig belongs to no rig it could be asked for, and is kept only where
    ``rig`` is ``None``. ``ValueError`` names the file, and the record, for a file that is not JSON, holds no list, or
    holds a record that is not a taught pose; ``OSError`` is a file that cannot be read.
    """
    poses = tuple(pose for _, pose in _read(Path(path)))
    return poses if rig is None else tuple(pose for pose in poses if pose.rig == rig)


def add_taught_pose(path: str | Path, pose: TaughtPose) -> int:
    """Append ``pose`` to a file of taught poses, keeping every record before it; how many the file holds now.

    The file is read and checked as :func:`read_taught_poses` reads it, and one that is not taught poses is refused
    and left as it is. The records before are written back as they were read and the pose after them, the whole
    through a file beside it that then replaces it, so nothing taught before is lost and no reader meets half a file.
    A missing folder is made.
    """
    target = Path(path)
    records = [record for record, _ in _read(target)]
    records.append(pose.to_dict())
    # The whole-file write a hand-guided calibration's stations go through: a file beside it, then a replace.
    write_stations(target, records)
    return len(records)


def _read(path: Path) -> list[tuple[Any, TaughtPose]]:
    """Each record of the file as it was read, with the pose it holds; ``[]`` for no file and for an empty one."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        records = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not JSON ({exc})") from None
    if not isinstance(records, list):
        raise ValueError(f"{path} holds a JSON {_json_kind(records)}, not a list of taught poses")
    read: list[tuple[Any, TaughtPose]] = []
    for index, record in enumerate(records):
        try:
            read.append((record, TaughtPose.from_dict(record)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: record {index} is not a taught pose: {exc}") from None
    return read


# --- the run -----------------------------------------------------------------------------------------------------


def teach_poses(
    robot: "Robot | RobotArm",
    *,
    tree: Any = None,
    for_rig: str | None = None,
    camera: Any = None,
    store: str | Path = DEFAULT_STORE,
    prefix: str = DEFAULT_PREFIX,
    guide: HandGuide | None = None,
    box: Any = None,
    out: Callable[[str], None] | None = None,
) -> tuple[TaughtPose, ...]:
    """Teach joint poses by hand on a connected arm until the person finishes; the poses, in the order first taught.

    ``robot`` is a connected :class:`~src.robot.execution.robot.Robot`, built alone (``gripper=None``) as a
    calibration builds it, or its arm. ``store`` is the file each pose is added to as it is taught. ``prefix`` names
    the poses the person leaves unnamed: ``LOOK`` gives ``LOOK_1``, ``LOOK_2`` and so on. ``guide`` is the console
    side, the terminal with no window when ``None``; ``HandGuide(console, view)`` shows a view of the caller's beside
    it, red outside a boundary. ``box`` is the workspace box a pose is said to be outside of, the robot's own
    ``workspace_limits`` when ``None``. ``out`` receives each line to paste as it is taught, unindented so it pastes as
    printed; ``None`` prints it.

    Every pose is taught for one rig, the camera it belongs to: the rig of ``camera`` where one is handed in, else the
    rig ``for_rig`` names in ``tree`` (a loaded tree; the robot's own where it kept one, as ``Robot.from_tree``
    does), the tree's primary when ``for_rig`` is ``None``. Its mounting is the tree's, read off the rig's calibration
    (a declared body counts as the wrist); where the tree declares none, the pose says so. Each line printed and each
    record kept carries the rig and the mounting.

    ``camera`` is the one open camera the poses are taught for, to look through while the arm is guided: a
    :class:`~src.camera.Camera` owner, or anything with ``peek()`` or ``grab()`` and a ``rig_id``. Given one, a window
    beside the console shows what it sees, live (the cell's ``LiveView``, that camera alone), with the guide: the pose
    being taught and how many are taught, red with the reason outside the cable window or the box. Enter or Space in it
    captures, ``q`` or ESC finishes, and closing it finishes, the arm held on every way out. It is display only: the
    view's own thread reads the camera through its display path and draws, and never calls the arm, which only the
    caller's thread does. It opens once the payload is confirmed, takes the place of ``guide``'s own view for the run,
    and is closed on every way out of this call, so the caller gives the camera back after it. Where no window can show
    (``preview_unavailable``, or stdout is no terminal) or the camera gives no frame, one console line says why and the
    poses are taught at the console, still for that rig. Teaching never needs the camera.

    Before anything is asked, ``ValueError`` refuses a ``prefix`` that makes no Python name, a ``camera`` of another
    rig than ``for_rig``, a rig the tree does not configure, and a run that can name no rig (no camera and no tree:
    a pose always belongs to one camera); ``TypeError`` refuses a ``camera`` that is no camera or names no rig. Before
    anything is freed, :class:`~src.robot.execution.hand_guiding.HandGuidingRefused` refuses an arm that offers no hand
    guiding (naming its vendor), an arm that is not connected, a file that is not taught poses (left as it is), a robot
    whose workspace box is not known and a payload the person does not confirm. Leaving the session holds the arm on
    every way out: Ctrl-C and an error raised while the arm is guided are raised on once it holds, and every pose
    taught before them is printed and in the file.
    """
    if _not_a_name(f"{prefix}_1"):
        raise ValueError(f"prefix {prefix!r} makes no Python name ({prefix}_1): {_NAME_RULE}")
    rig, mounting = _taught_for(robot, tree, for_rig, camera)
    arm = _guidable(robot)
    path = Path(store)
    kept = _kept(path)
    watched = _workspace_box(robot, arm, box)
    guide = guide if guide is not None else HandGuide()
    emit = out if out is not None else _print
    window = None if camera is None else _camera_window(camera, rig, guide.console)
    own_view = guide.view
    try:
        guide.confirm_payload(arm)
        limits = HandGuidingLimits.of(arm, watched)
        guide.console.say(f"Watched while you guide the arm: {limits.window}. Outside either you are told, and Enter "
                          "does not capture there; the arm is never held or stopped for it.")
        already = f"{kept} there already" if kept else "none there yet"
        guide.console.say(f"Nothing moves by itself: the arm moves only while you move it, and it holds where you leave "
                          f"it between poses. Every pose is added to {path} ({already}).")
        guide.console.say(f"Every pose taught here belongs to rig {rig!r}, {_mounted(mounting)}: its line to paste and "
                          "its record say so.")
        if window is not None:
            # Opened once the payload is confirmed, which is asked at the console alone, with a guide's words on it
            # from its first frame, and the rig it teaches for as its first word.
            window.guide(["TEACHING POSES BY HAND", "the console asks for the name of each pose"], "guide")
            window.note(f"teaching poses by hand for rig {rig!r}, {_mounted(mounting)}")
            guide.view = window
            window.start()
        taught = _session(arm, guide, limits, path, prefix, emit, rig, mounting)
    finally:
        # The session has held the arm by now, on every way out; the window closes after it and before the caller
        # gives the camera back.
        guide.view = own_view
        if window is not None:
            window.close()
    guide.console.say(_summary(len(taught), path, rig))
    return tuple(taught)


def _session(arm: SupportsFreedrive, guide: HandGuide, limits: HandGuidingLimits, path: Path, prefix: str,
             emit: Callable[[str], None], rig: str, mounting: str | None) -> list[TaughtPose]:
    """The one hand-guiding session: a name while the arm holds, the arm freed, Enter, held and read, until the person
    finishes, every pose for ``rig``. Leaving it holds the arm, whatever ended it."""
    taught: list[TaughtPose] = []

    def status(_sample: FreedriveSample) -> tuple[list[str], float | None]:
        return [_so_far(taught)], None

    with arm.freedrive() as session:
        while (name := _next_name(guide, taught, prefix)) is not None:
            session.free()
            choice = guide.wait(session, limits, banner=f"MOVE THE ARM BY HAND to where {name} should be, then Enter",
                                status=status)
            _show(guide, ["HOLDING THE ARM", f"{name} is read once it holds" if choice == CAPTURE
                          else "teaching finishes" if choice == FINISH else f"{name} skipped: nothing taught"])
            # Held on every answer, before anything else is asked: a question waits on the person, and nothing
            # samples a free arm while it does.
            session.hold()
            if choice == FINISH:
                break
            if choice != CAPTURE:
                guide.console.say(f"{name} skipped: nothing taught, and the arm holds where it stands.")
                continue
            pose = TaughtPose.from_sample(name, session.sample(), rig=rig, mounting=mounting)
            emit(pose.line())
            held = add_taught_pose(path, pose)
            _keep(taught, pose)
            guide.console.say(f"{name} taught: {pose.tcp_line()}. Added to {path}, which holds "
                              f"{held} {'pose' if held == 1 else 'poses'} now; the arm holds where it stands.")
    return taught


def _taught_for(robot: Any, tree: Any, for_rig: str | None, camera: Any) -> tuple[str, str | None]:
    """The rig the poses are taught for and its mounting, or the refusal of a run that names none or two.

    The rig of ``camera`` where one is handed in (refused where ``for_rig`` names another), else the rig ``for_rig``
    names in the tree, the tree's primary when it names none. The tree is ``tree``, or the robot's own where it kept
    one; the mounting is the one the tree declares for the rig, or the camera's own rig's where there is no tree.
    """
    named: str | None = None
    if camera is not None:
        if not (callable(getattr(camera, "peek", None)) or callable(getattr(camera, "grab", None))):
            raise TypeError(f"camera= is one open Camera, the one the poses are taught for, or an object with peek() "
                            f"or grab(); not {type(camera).__name__}")
        rig_id = getattr(camera, "rig_id", None)
        if not isinstance(rig_id, str) or not rig_id:
            raise TypeError(f"camera= names no rig (rig_id), so the poses could belong to no camera: "
                            f"{type(camera).__name__}")
        if for_rig is not None and for_rig != rig_id:
            raise ValueError(f"camera= is rig {rig_id!r}, and the poses are taught for rig {for_rig!r} (for_rig=): a "
                             "pose belongs to one camera, so open the camera of the rig the poses are for, or name that "
                             "rig")
        named = rig_id
    section = _camera_section(tree if tree is not None else getattr(robot, "tree", None))
    if section is None:
        if named is not None:
            return named, _mounting_of(getattr(camera, "rig", None))
        if for_rig is not None:
            raise ValueError(f"for_rig={for_rig!r} names a rig, and without the loaded tree (tree=) nothing says how rig "
                             f"{for_rig!r} is mounted: pass tree=")
        raise ValueError("teach_poses needs the rig the poses are taught for: the loaded tree (tree=), whose primary "
                         "rig it is unless for_rig= names another, or the camera (camera=); a pose always belongs to "
                         "one camera")
    rigs = {str(getattr(rig, "rig_id", "")): rig for rig in section.rigs}
    if named is not None:
        if named not in rigs:
            raise ValueError(f"camera= is rig {named!r}, which camera.cameras.rigs of the tree does not configure "
                             f"({sorted(rigs)}): the poses would belong to a camera the cell does not know")
        return named, _mounting_of(rigs[named])
    wanted = for_rig if for_rig is not None else str(section.primary_rig_id)
    if wanted not in rigs:
        raise ValueError(f"for_rig={wanted!r} is no rig of camera.cameras.rigs {sorted(rigs)}: name the rig whose poses "
                         "you teach")
    return wanted, _mounting_of(rigs[wanted])


def _camera_section(tree: Any) -> Any:
    """The camera section (``camera.cameras``) of a loaded tree, or ``None`` where there is no tree. A tree that did
    not load refuses with its own refusal, as every door that reads one does."""
    if tree is None:
        return None
    return getattr(getattr(getattr(tree, "app_config", None), "camera", None), "cameras", None)


def _mounting_of(rig: Any) -> str | None:
    """How a rig is mounted, as its tree declares it: its calibration's mounting, else the wrist where it declares a
    body the arm carries, else ``None``."""
    mode = getattr(getattr(rig, "extrinsics", None), "mounting_mode", None)
    if mode in _MOUNTINGS:
        return str(mode)
    return "eye_in_hand" if getattr(rig, "body", None) is not None else None


def _mounted(mounting: str | None) -> str:
    """A mounting in words: ``eye in hand``, ``eye to hand``, or ``mounting not declared``."""
    return _MOUNTINGS.get(mounting or "", "mounting not declared")


def _camera_window(camera: Any, rig: str, console: OperatorConsole) -> "LiveView | None":
    """The window that shows what ``camera`` sees, built and not started; ``None`` once one line said why there is none.

    No window where none can show (``preview_unavailable``, or stdout is no terminal) or where the camera gives no frame:
    one grab is taken here, on the caller's thread, before the view's thread reads the camera. The window is the cell's
    live view with this camera alone and teaching's words for its lines; its thread only reads the camera through its
    display path and draws. Without it the poses are taught at the console, still for ``rig``.
    """
    from src.calibration import preview  # noqa: PLC0415 (only where a camera is shown)

    reason = preview.preview_unavailable()
    if reason is None and not _is_terminal(sys.stdout):
        reason = "stdout is not a terminal"
    grab = getattr(camera, "grab", None)
    if reason is None and callable(grab):
        try:
            grab()
        except Exception as exc:  # noqa: BLE001 (said, and the poses are taught at the console: no camera needed)
            reason = f"the camera gives no frame ({type(exc).__name__}: {exc})"
    if reason is None:
        try:
            from src.camera import live_view  # noqa: PLC0415 (the robot loads no camera package unless it shows one)

            return live_view.LiveView([camera], title="willy teach poses", goes_on=_GOES_ON, on_close=_ON_CLOSE,
                                      say=console.say)
        except Exception as exc:  # noqa: BLE001 (a window is display only: teaching goes on without it)
            reason = f"{type(exc).__name__}: {exc}"
    console.say(f"No camera window: {reason}. The poses are taught at the console, still for rig {rig!r}.")
    return None


def _guidable(robot: Any) -> SupportsFreedrive:
    """The arm of ``robot`` (a ``Robot``, or an arm) as one a person can guide and that is connected.

    Refused before anything is asked: an arm that offers no hand guiding, naming its vendor, and one not connected.
    """
    arm = getattr(robot, "arm", robot)
    if not isinstance(arm, SupportsFreedrive):
        raise HandGuidingRefused(
            f"{_named(arm)} offers no hand guiding (SupportsFreedrive; the UR teach mode is one), so no pose can be "
            "taught by hand on it: read its joints off its own pendant instead. Nothing was asked and nothing was "
            "freed")
    if not bool(getattr(arm, "is_connected", True)):
        raise HandGuidingRefused(
            f"{type(arm).__name__} is not connected, so it cannot be freed: teach inside `with robot.connected():`, "
            "which takes the cell's lock and connects the arm. Nothing was asked and nothing was freed")
    return arm


def _kept(path: Path) -> int:
    """How many poses the file holds before this run, its folder made; refused where it cannot take a pose."""
    try:
        kept = len(_read(path))
        path.parent.mkdir(parents=True, exist_ok=True)
    except ValueError as exc:
        raise HandGuidingRefused(f"{exc}. It is left as it is and nothing was freed: move it aside, or name another "
                                 "file with store=") from None
    except OSError as exc:
        raise HandGuidingRefused(f"{path} cannot be read or its folder made ({type(exc).__name__}: {exc}), so no pose "
                                 "could be kept. Nothing was freed: name another file with store=") from None
    return kept


def _workspace_box(robot: Any, arm: Any, box: Any) -> Any:
    """``box``, or the workspace box of the robot section the robot or its arm keeps; refused where none is known."""
    if box is None:
        for section in (getattr(robot, "robot_config", None), getattr(arm, "config", None)):
            box = getattr(section, "workspace_limits", None)
            if box is not None:
                break
    if box is None or not all(isinstance(getattr(box, face, None), (int, float)) for face in _FACES):
        raise HandGuidingRefused(
            "the workspace box is not known (the robot keeps no robot section with workspace_limits, and no box= was "
            "given), so a pose outside it could not be said: build the robot from the cell's tree, "
            "Robot.from_config(tree.robot, gripper=None). Nothing was asked and nothing was freed")
    return box


def _next_name(guide: HandGuide, taught: Sequence[TaughtPose], prefix: str) -> str | None:
    """The next pose's name, asked while the arm holds, or ``None`` when the person finishes (``q``, or no input).

    Enter takes ``<prefix>_<n>``, the smallest ``n`` from 1 no pose of this run is called. A name that cannot be
    pasted as Python is said not to be one and asked again; a name taken in this run is said to replace that pose.
    """
    taken = {pose.name for pose in taught}
    default = next(name for name in (f"{prefix}_{n}" for n in count(1)) if name not in taken)
    while True:
        answer = guide.ask(f"Name of the next pose? Enter = {default}, q finishes")
        word = answer.strip()
        if EOF in (answer, word) or word.lower() in _FINISH_WORDS:
            return None
        name = word or default
        why = _not_a_name(name)
        if why:
            guide.console.say(f"{word!r} is not a name here: {why}.")
            continue
        if name in taken:
            guide.console.say(f"{name} was taught already in this run: this pose takes its place, and the file keeps "
                              "both, each with its time.")
        return name


def _not_a_name(name: str) -> str:
    """Why ``name`` cannot be pasted as a Python name, or ``""`` where it can."""
    if not (name.isascii() and name.isidentifier()):
        return _NAME_RULE
    if keyword.iskeyword(name):
        return f"{name} is a Python keyword"
    return ""


def _keep(taught: list[TaughtPose], pose: TaughtPose) -> None:
    """``pose`` in its place: where the pose of its name stood, which it replaces, or after the others."""
    for index, before in enumerate(taught):
        if before.name == pose.name:
            taught[index] = pose
            return
    taught.append(pose)


def _show(guide: HandGuide, lines: list[str]) -> None:
    """``lines`` in the guide's view, where there is one. ``HandGuide.wait`` clears its view as it returns, and a
    cleared sweep window speaks of a sweep (``ESC closes this window only``); this keeps a guide's words on it."""
    if guide.view is not None:
        guide.view.guide(lines, "guide")


def _so_far(taught: Sequence[TaughtPose]) -> str:
    """How many poses this run has taught, and their names, for the window."""
    if not taught:
        return "none taught yet"
    return f"{len(taught)} taught: {', '.join(pose.name for pose in taught)}"


def _is_terminal(stream: Any) -> bool:
    """Whether ``stream`` is a terminal. A stream that cannot say is not one."""
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except Exception:  # noqa: BLE001 (a closed or odd stream is no terminal)
        return False


def _summary(taught: int, path: Path, rig: str) -> str:
    """The run's last line, said once the session is left."""
    if taught == 0:
        return "No pose taught. The arm holds where it stands."
    if taught == 1:
        return f"1 pose taught for rig {rig!r}, added to {path}. The arm holds where it stands."
    return f"{taught} poses taught for rig {rig!r}, each added to {path}. The arm holds where it stands."


def _named(arm: Any) -> str:
    """The arm's class and the vendor it reports, as a refusal names them."""
    capabilities = getattr(arm, "capabilities", None)
    if isinstance(capabilities, RobotCapabilities):
        return f"{type(arm).__name__} (vendor {capabilities.vendor!r})"
    return f"{type(arm).__name__} (which reports no vendor)"


def _print(line: str) -> None:
    """A line to paste, unindented, on stdout, now."""
    print(line, flush=True)


def _now() -> str:
    """The local time, to the second, with its offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rounded(value: float, digits: int) -> float:
    """``value`` to ``digits`` places, and ``0.0`` rather than ``-0.0``."""
    return round(float(value), digits) + 0.0


def _number(value: Any, key: str) -> float:
    """``value`` as a finite number, or ``ValueError`` naming ``key``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{key} is {value!r}, not a finite number")
    return float(value)


def _numbers(value: Any, key: str, *, size: int | None = None) -> list[float]:
    """``value`` as a non-empty list of finite numbers (``size`` of them where given), or ``ValueError``."""
    if not isinstance(value, list) or not value or (size is not None and len(value) != size):
        wanted = f"a list of {size} numbers" if size is not None else "a list of numbers"
        raise ValueError(f"{key} is {value!r}, not {wanted}")
    return [_number(item, f"{key}[{index}]") for index, item in enumerate(value)]


def _json_kind(value: Any) -> str:
    """What JSON calls the kind of ``value``."""
    return {dict: "object", list: "array", str: "string", bool: "boolean", type(None): "null"}.get(type(value),
                                                                                                  "number")
