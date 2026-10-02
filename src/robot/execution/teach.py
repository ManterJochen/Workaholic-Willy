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

Each pose is screened once it is held, where the arm screens (``URRobotArm.screen_configuration``, the owner,
2026-09-30): clear, in the planner's cushion band, or refused (an ERROR line), with a pose nearby both clear. Only on
an arm that carries the housing of every wrist camera the tree hangs on it (``Robot.from_tree`` hands them,
``Robot.from_config`` none): a screen without a housing that is there reads clear what the housing meets, so on any
other arm nothing is screened and one line says so.

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

One pose from the console (:func:`teach_one`, the owner, decision 15 and Q10, Q14, 2026-09-30): a pose a task places
at or returns to is taught one per session, over the same ``HandGuide.wait`` and ``TaughtPose``, with the browser as the
console. Everything that would make the pose unwritable is refused before the arm is freed (a name that is no pose name,
an arm that screens nothing or was not handed its wrist camera's housing, a planner that is not ready, a store that
names no file of the cell's own), by one rule the console can ask without freeing anything (:func:`teach_refusal`).
The pose is screened the moment it is held, and only a pose both authorities clear, or one in the planner's band,
reaches the store (:class:`ProfilePoseStore`: the cell's own layer, through the pose door alone). Save, Hold, Cancel,
a halt and a disconnect hold the arm at once, also while a Save still waits for the arm to stand still, and a pose
cancelled then is never written. The console's own holds, a lapsed heartbeat and the time limit, hold the arm only once
it stands still (:class:`StillnessGate`, hand_guiding's rule), never while it moves in a person's hands.
"""

from __future__ import annotations

import json
import keyword
import logging
import math
import sys
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from src.config.schema.robot.robot_schema import pose_label_refusal, pose_name_refusal, pose_words
from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle
from src.robot.core import JointPositions, RobotCapabilities
from src.robot.core.freedrive import FreedriveSample, SupportsFreedrive
from src.robot.drivers.ur.bench import where_lines
from src.robot.execution.hand_guiding import (
    CAPTURE,
    EOF,
    FINISH,
    SKIP,
    STILL_FOR_S,
    STILL_RAD_S,
    GuideView,
    HandGuide,
    HandGuidingLimits,
    HandGuidingRefused,
    OperatorConsole,
    sample_pose,
)
from src.robot.execution.pose_provider import write_stations

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.camera.live_view import LiveView
    from src.config.tree import ConfigTree
    from src.robot.core import RobotArm
    from src.robot.core.freedrive import FreedriveSession
    from src.robot.execution.robot import Robot
    from src.robot.safety.planning.band import PoseScreen

__all__ = [
    "DEFAULT_PREFIX",
    "DEFAULT_STORE",
    "HELD_WHEN_STILL",
    "PoseStore",
    "ProfilePoseStore",
    "StillnessGate",
    "TaughtOne",
    "TaughtPose",
    "TeachRefused",
    "add_taught_pose",
    "name_refusal",
    "read_taught_poses",
    "teach_one",
    "teach_poses",
    "teach_refusal",
]

logger = logging.getLogger(__name__)

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
#: What :func:`teach_one` ended on where a hold the caller asked for (a lapsed heartbeat, the time limit) held the arm
#: once it stood still: no person saved or cancelled, and nothing was taught.
HELD_WHEN_STILL = "held_when_still"
#: The screens a taught pose is written with (``src.config.edit.WRITTEN_SCREENS``): both authorities clear it, or it
#: lies in the planner's band. An unscreened or refused pose never is.
_WRITTEN = ("clear", "band")
#: What a person sends that holds the arm at once in :func:`teach_one`, as ``HandGuide`` reads it: from the console a
#: finish (``q``, input that ended) or a skip (``s``), from the view its finish, a closed window, or its skip.
_CONSOLE_HOLDS = {"q": FINISH, "quit": FINISH, "finish": FINISH, EOF: FINISH, "s": SKIP, "skip": SKIP}
_VIEW_HOLDS = {"finish": FINISH, "closed": FINISH, "skip": SKIP}


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
        screen = _screens(arm)
        unscreened = _unscreened_because(robot, arm, tree) if screen else None
        if unscreened is not None:
            guide.console.say(unscreened)
            screen = False
        elif screen:
            guide.console.say("Every pose is screened once it is taught, by the exact mesh guard and by the planner, "
                              "which starts with the first pose (about a minute): a pose the planner's padded spheres "
                              "alone refuse is said, with a pose nearby both clear. Nothing moves for it.")
        if window is not None:
            # Opened once the payload is confirmed, which is asked at the console alone, with a guide's words on it
            # from its first frame, and the rig it teaches for as its first word.
            window.guide(["TEACHING POSES BY HAND", "the console asks for the name of each pose"], "guide")
            window.note(f"teaching poses by hand for rig {rig!r}, {_mounted(mounting)}")
            guide.view = window
            window.start()
        taught = _session(arm, guide, limits, path, prefix, emit, rig, mounting, screen=screen)
    finally:
        # The session has held the arm by now, on every way out; the window closes after it and before the caller
        # gives the camera back.
        guide.view = own_view
        if window is not None:
            window.close()
    guide.console.say(_summary(len(taught), path, rig))
    return tuple(taught)


def _session(arm: SupportsFreedrive, guide: HandGuide, limits: HandGuidingLimits, path: Path, prefix: str,
             emit: Callable[[str], None], rig: str, mounting: str | None, *, screen: bool) -> list[TaughtPose]:
    """The one hand-guiding session: a name while the arm holds, the arm freed, Enter, held and read, until the person
    finishes, every pose for ``rig``, each screened once it is held where ``screen``. Leaving it holds the arm,
    whatever ended it."""
    taught: list[TaughtPose] = []
    ask_planner = True

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
            if screen:
                ask_planner = _say_screen(arm, pose, guide, ask_planner)
    return taught


def _screens(arm: Any) -> bool:
    """Whether ``arm`` screens a configuration with the exact guard and the planner (``URRobotArm.screen_configuration``)."""
    return callable(getattr(type(arm), "screen_configuration", None))


def _unscreened_because(robot: Any, arm: Any, tree: Any) -> str | None:
    """Why no pose is screened on ``arm``, in one line; ``None`` where it carries every wrist camera the tree hangs on it.

    A screen asks both models, and both hold a wrist camera's housing only where the arm was handed it
    (``execution.wrist_bodies``): ``Robot.from_tree`` hands every one the tree declares, ``Robot.from_config`` none. A
    screen without a housing that is there would read clear a pose the housing meets, and building the planner and the
    guard for it would refuse the housing handed in after them. So a camera the tree hangs on the arm (a declared body,
    or an enabled eye_in_hand rig) that the arm does not hold, or a run with no tree to say which hang there, screens
    nothing (review of F1, 2026-09-30).
    """
    section = _camera_section(tree if tree is not None else getattr(robot, "tree", None))
    where = ("The looks are screened where a campaign starts and at the desk: python -m "
             "src.robot.execution.real_cell --start-planner.")
    if section is None:
        return ("Poses are not screened here: no tree says which wrist cameras hang on the arm, and a screen without a "
                f"housing that is there would miss what it meets. {where}")
    hung = [str(getattr(rig, "rig_id", "")) for rig in getattr(section, "rigs", None) or ()
            if getattr(rig, "body", None) is not None
            or (getattr(rig, "enabled", False)
                and getattr(getattr(rig, "extrinsics", None), "mounting_mode", None) == "eye_in_hand")]
    bodies = getattr(getattr(arm, "safety_preflight", None), "wrist_bodies", None)
    held = {str(getattr(body, "rig_id", "")) for body in (bodies(arm) if callable(bodies) else ())}
    missing = [rig for rig in hung if rig not in held]
    if not missing:
        return None
    return (f"Poses are not screened here: the tree hangs wrist camera(s) {', '.join(repr(r) for r in missing)} on the "
            "arm, and this arm was not handed the housing (Robot.from_config hands none, Robot.from_tree every one the "
            f"tree declares), so both models would judge a pose without it. {where}")


def _say_screen(arm: Any, pose: TaughtPose, guide: HandGuide, ask_planner: bool) -> bool:
    """Screen a pose just taught and say the verdict; whether the planner is still worth asking for the next one.

    The arm holds while it is screened, and nothing moves for it. The pose is taught and kept whatever the verdict: the
    screen says what a pick would meet there (``planning.band``), and a pose the exact guard refuses is said in an
    ERROR line with a pose nearby both clear. A planner that could not start is not asked again in this run, and a
    screen that raises is said and teaching goes on.
    """
    if not _screens(arm):
        return ask_planner
    try:
        screen = arm.screen_configuration(pose.joints, ask_planner=ask_planner)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 (a screen that fails never stops a person teaching)
        guide.console.say(f"{pose.name} was not screened: {type(exc).__name__}: {exc}")
        return ask_planner
    guide.console.say(screen.line(pose.name))
    return ask_planner and not bool(getattr(screen, "planner_unavailable", False))


# --- one pose, from the console ----------------------------------------------------------------------------------


class TeachRefused(HandGuidingRefused):
    """A teach refused before anything was freed, with the code the console answers it with.

    ``code`` is one of ``invalid_name``, ``invalid_label``, ``no_hand_guiding``, ``not_connected``,
    ``screen_unavailable``, ``planner_not_ready`` and ``no_layer``, the refusals of ``POST /v1/teach``. A
    :class:`~src.robot.execution.hand_guiding.HandGuidingRefused`, so a caller that catches those catches this too.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class StillnessGate:
    """Hand guiding's stillness gate, fed one sample at a time: still once the fastest joint has stayed below
    ``still_rad_s`` for ``still_for_s`` (``STILL_RAD_S`` and ``STILL_FOR_S`` of :mod:`.hand_guiding`, about 1.1 deg/s
    for half a second), the rule ``HandGuide.wait`` captures by.

    The console holds a hand-guided arm on its own initiative, a lapsed heartbeat or the time limit, only once this
    says still, never while a person moves the arm (the owner, Q14): an arm that locks in a person's hands is the
    injury the freedrive design avoids. A sample at or above the threshold starts the half second again, and
    :meth:`reset` forgets every sample.
    """

    def __init__(self, still_rad_s: float = STILL_RAD_S, still_for_s: float = STILL_FOR_S) -> None:
        self.still_rad_s = float(still_rad_s)
        self.still_for_s = float(still_for_s)
        self._since: float | None = None
        self._still = False

    @property
    def still(self) -> bool:
        """What the last sample said: the arm has stood still long enough."""
        return self._still

    def feed(self, sample: FreedriveSample, now: float) -> bool:
        """Take one sample, read at ``now`` on a clock that only runs forward; whether the arm stands still."""
        if sample.peak_joint_speed_rad_s < self.still_rad_s:
            self._since = now if self._since is None else self._since
            self._still = now - self._since >= self.still_for_s
        else:
            self._since, self._still = None, False
        return self._still

    def reset(self) -> None:
        """Forget every sample: the next ones start the half second again."""
        self._since, self._still = None, False


@dataclass(frozen=True, slots=True)
class TaughtOne:
    """What one session of :func:`teach_one` ended with.

    ``choice`` is how the session ended: ``capture`` (Save, once the arm stood still), ``skip`` or ``finish`` (Hold,
    Cancel, a halt, a disconnect: held at once, nothing taught), or :data:`HELD_WHEN_STILL` (a hold the caller asked
    for, held once the arm stood still; ``because`` says what asked for it). ``pose`` is the pose read off the held arm,
    for a capture alone; ``screen`` what the exact guard and the planner said about it, ``None`` where the screen
    raised (``screen_error`` says how). ``written`` is the file the pose was written to, ``""`` where it was not, and
    ``not_written`` says why a captured pose was not. :attr:`outcome` sums it up for the console.
    """

    name: str
    choice: str
    pose: TaughtPose | None = None
    screen: "PoseScreen | None" = None
    screen_error: str = ""
    because: str = ""
    written: str = ""
    not_written: str = ""

    @property
    def outcome(self) -> str:
        """``saved``; ``refused``, an ERROR verdict; ``not_saved``, unscreened, a screen that raised or a write that did
        not land; ``held_when_still``; or ``cancelled``."""
        if self.choice == HELD_WHEN_STILL:
            return "held_when_still"
        if self.choice != CAPTURE:
            return "cancelled"
        if self.written:
            return "saved"
        if self.screen is not None and self.screen.is_error:
            return "refused"
        return "not_saved"

    def render(self) -> str:
        """One line: the pose, how the session ended, and what was written or why nothing was."""
        if self.choice == HELD_WHEN_STILL:
            return f"{self.name}: held once the arm stood still ({self.because}); nothing taught."
        if self.choice != CAPTURE:
            return f"{self.name}: held at once ({self.choice}); nothing taught."
        if self.written:
            verdict = _verdict(self.screen)
            return f"{self.name}: taught, screened {verdict}, written to {self.written}."
        return f"{self.name}: taught, not written: {self.not_written}"

    def __str__(self) -> str:
        return self.render()

    def to_dict(self) -> dict[str, Any]:
        """Plain data for the console's events and its log, ``json.dumps`` safe."""
        pose = self.pose
        screen = self.screen
        nearby = getattr(screen, "nearby", None)
        return {
            "name": self.name,
            "choice": self.choice,
            "outcome": self.outcome,
            "because": self.because,
            "joints_deg": None if pose is None else [_rounded(value, 4) for value in pose.joints_deg],
            "tcp_mm": None if pose is None else [_rounded(float(value), 2) for value in pose.tcp.position_mm],
            "verdict": None if screen is None else _verdict(screen),
            "detail": "" if screen is None else str(screen.detail),
            "nearby_deg": None if nearby is None else [_rounded(math.degrees(float(v)), 4) for v in nearby],
            "screen_error": self.screen_error,
            "written": self.written,
            "not_written": self.not_written,
        }


class PoseStore(Protocol):
    """Where :func:`teach_one` keeps a pose: named before the arm is freed, written only once it is screened clear or
    in the planner's band."""

    @property
    def target(self) -> str:
        """The file a pose is written to, said before the arm is freed; :class:`TeachRefused` where there is none."""
        ...

    def write(self, pose: TaughtPose, screen: "PoseScreen") -> str:
        """Write ``pose``, which ``screen`` cleared or found in the band: ``""`` once written, else why not, with
        nothing written."""
        ...


@dataclass(frozen=True)
class ProfilePoseStore:
    """The cell profile's own layer, the last of its chain (the owner, Q10), written through the pose door alone.

    ``tree`` is the chain the console runs (``ConfigTree``), ``name`` the pose it keeps, ``label`` what the chat and the
    cards call it, ``make_default_place`` makes it the default place in the same transaction, and ``replace`` lets it
    take the place of a pose of that name. The pose goes in with its joints in degrees, its label, when it was taught,
    its screen, and the screen's own words for a band pose (``src.config.edit.set_named_pose``).

    Refused as it is made, before anything is freed, so nobody guides the arm for a pose that cannot be written: a name
    or a label that is none (``invalid_name``, ``invalid_label``), a chain with no layer of the cell's own
    (``no_layer``: none at all, whose pose would land in the shared ``robot.yaml``, or a last layer git does not keep
    out, whose pose would land in the repository: ``ConfigTree.pose_layer_refusal``), a name the tree already holds
    unless ``replace`` (``name_taken``), and a word another pose already answers to (``robot_schema.pose_words``: a
    pose answers to its name and to its label, an unlabelled one to its name; ``invalid_name`` or ``invalid_label``
    for the one that clashes), since the command reader hands a pose on by the word a person said. The tree is read
    from its own folder under its own chain, as the pose door validates it.
    """

    tree: "ConfigTree"
    name: str
    label: str
    make_default_place: bool = False
    replace: bool = False

    def __post_init__(self) -> None:
        refused = name_refusal(self.name)
        if refused:
            raise TeachRefused("invalid_name", f"{refused}. Nothing was freed")
        refused = pose_label_refusal(self.label)
        if refused:
            raise TeachRefused("invalid_label", f"{refused}. Nothing was freed")
        target = self.target  # the file is named before anything is freed, or the store is refused
        poses = _taught_poses_of(self.tree)
        if self.name in poses and not self.replace:
            raise TeachRefused("name_taken", (
                f"a pose is taught under {self.name!r} already, in the tree written to {target}: teach it again only to "
                "replace it, or choose another name. Nothing was freed"))
        mine, by_name = pose_words(self.name, self.label), pose_words(self.name)
        for other, pose in poses.items():
            shared = sorted(mine & pose_words(other, pose.label)) if other != self.name else []
            if shared:
                word = shared[0]
                code, what = ("invalid_name", "name") if word in by_name else ("invalid_label", "label")
                how = "its name" if word in pose_words(other) else "its label"
                raise TeachRefused(code, (
                    f"{other!r} answers to {word!r} already, by {how}, and the command reader hands a pose on by the "
                    "word a person said, so two poses answering to one word could send a part to either: choose "
                    f"another {what}. Nothing was freed"))

    @property
    def target(self) -> str:
        refused = self.tree.pose_layer_refusal()
        if refused:
            raise TeachRefused("no_layer", f"{refused} Nothing was freed.")
        file = self.tree.pose_file()
        if file is None:
            raise TeachRefused("no_layer", (
                f"no file backs robot.named_poses under the config root {self.tree.root}, so the pose could be written "
                "nowhere. Nothing was freed."))
        return str(file)

    def write(self, pose: TaughtPose, screen: "PoseScreen") -> str:
        if pose.name != self.name:
            return f"this store keeps {self.name!r}, and the pose handed to it is {pose.name!r}: nothing was written"
        verdict = _verdict(screen)
        try:
            result = self.tree.write_named_pose(
                pose.name,
                joints_deg=[_rounded(value, 4) for value in pose.joints_deg],
                label=self.label,
                screen=verdict,
                taught_at=pose.taught_at or _now(),
                note=screen.render() if verdict == "band" else "",
                make_default_place=self.make_default_place,
            )
        except Exception as exc:  # noqa: BLE001 (a tree that did not load before the write refuses it here too)
            return f"the pose door could not write it ({type(exc).__name__}: {exc})"
        if result.applied:
            return ""
        return result.message or f"the pose door refused it ({result.refused})"


def _taught_poses_of(tree: "ConfigTree") -> dict[str, Any]:
    """The named poses ``tree`` holds now, read from its own folder under its own chain; ``{}`` where it does not load
    (the pose door then refuses the write with the loader's own words)."""
    from src.config.loader import ConfigError, load_config, reload_config  # noqa: PLC0415 (only before a teach)

    reload_config()  # the files as they are now, as the pose door reads them when it writes
    try:
        robot = load_config(tree.root, profile=tree.profile).robot
    except ConfigError:
        return {}
    return dict(getattr(robot, "named_poses", None) or {})


def name_refusal(name: str) -> str:
    """Why ``name`` cannot name a pose taught in the console, or ``""`` where it can: the schema's own rule
    (``robot_schema.pose_name_refusal``), an ASCII identifier of at most 32 characters, no keyword, never ``home``, no
    word YAML reads as true, false or null, and none of the loader's own words (``__null__``)."""
    return pose_name_refusal(name)


def teach_refusal(
    robot: "Robot | RobotArm",
    *,
    tree: Any = None,
    name: str | None = None,
    store: PoseStore | None = None,
) -> TeachRefused | None:
    """Why no pose could be taught on ``robot`` now, as :func:`teach_one` would refuse it before anything is freed, or
    ``None`` where one could.

    The console asks it before anyone asks to teach (``GET /v1/poses``: ``teachable`` and ``why_not``), so the reason a
    teach is not offered is the reason the teach itself would refuse with, by one rule: ``name`` where one is chosen
    (``invalid_name``); an arm that offers no hand guiding (``no_hand_guiding``) or is not connected
    (``not_connected``); an arm on which no pose could be screened honestly (``screen_unavailable``: it screens
    nothing, plans with no planner, was not handed the housing of a wrist camera ``tree`` hangs on it, or comes with no
    tree to say which hang there); a planner it reports not ready (``planner_not_ready``); and the store's own
    refusal of its file where a store is handed (``no_layer``). Nothing is freed, asked, moved or written: the arm's
    connection and planner state, the tree's cameras and the store's file are read.
    """
    try:
        _ready_to_teach(robot, tree, name, store)
    except TeachRefused as refused:
        return refused
    return None


def _ready_to_teach(robot: Any, tree: Any, name: str | None, store: PoseStore | None) -> tuple[SupportsFreedrive, str]:
    """The checks made before anything is freed, in the order :func:`teach_one` makes them: the arm a person can guide,
    and the file the store names (``""`` without a store); :class:`TeachRefused` with its code otherwise."""
    if name is not None:
        refused = name_refusal(name)
        if refused:
            raise TeachRefused("invalid_name", f"{refused}. Nothing was asked and nothing was freed")
    arm = _guidable_for_one(robot)
    _screened_here_or_refused(robot, arm, tree)
    _planner_ready_or_refused(arm)
    return arm, (store.target if store is not None else "")


def teach_one(
    robot: "Robot | RobotArm",
    guide: HandGuide,
    limits: HandGuidingLimits,
    name: str,
    *,
    store: PoseStore,
    tree: Any = None,
    rig: str | None = None,
    mounting: str | None = None,
    ask_planner: bool = True,
    hold_when_still: Callable[[], str] | None = None,
    watch: Callable[[FreedriveSample], None] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> TaughtOne:
    """Teach ONE pose by hand on a connected arm: freed, captured once still, held, screened at once, written only if
    clear or in the planner's band.

    The console's teach (the owner, decision 15, Q10, Q14): ``robot`` is a connected robot or its arm, ``guide`` the
    console side (the browser's console and view; ``payload_confirmed`` set where the console confirmed the payload,
    otherwise it is asked first), ``limits`` the cable window and the box, ``name`` the pose's name, and ``store`` where
    it is written (:class:`ProfilePoseStore`). ``tree`` says which wrist cameras hang on the arm (a loaded tree or its
    ``AppConfig``; the robot's own where it kept one). ``rig`` and ``mounting`` go onto the pose read off the arm.
    ``ask_planner`` is the build plan's spelling and only ``True`` is taken: the planner is always asked, since only a
    pose both authorities clear, or one in the planner's band, is written, and a screen without it could write nothing
    (``ValueError``, a caller's mistake, before anything is asked).

    Refused before anything is freed, with :class:`TeachRefused` and its code (:func:`teach_refusal` says the same
    without freeing anything): a name that is no pose name (``invalid_name``); an arm that offers no hand guiding or is
    not connected; an arm that screens nothing, plans with no planner, was not handed the housing of a wrist camera the
    tree hangs on it, or comes with no tree to say which hang there (``screen_unavailable``: a screen without a housing
    that is there would read clear what the housing meets); a planner the arm reports not ready
    (``planner_not_ready``: a screen would start it, about a minute, while a person stands at the arm); a store that
    names no file of the cell's own. The file is said before the arm is freed.

    Then one hand-guiding session: the arm is freed, and ``HandGuide.wait`` samples it at 50 Hz. Save captures once the
    arm has stood still for half a second, never outside the cable window or the box; Hold and Cancel, and a halt or a
    disconnect the console sends as a key, hold at once wherever the arm is. Every sample reads what the person sent,
    so a key that holds wins at once also while a Save still waits for the arm to stand still (``HandGuide`` reads no
    key during that wait, up to 5 s): the arm is held within one sample and the pose is never written. Every other key
    read early is handed on to the wait in order. ``HandGuide.wait`` forgets a console line sent before it waits, as at
    the terminal, and keeps a finish from the view, so the console sends those holds as the view's ``finish``: none of
    them is lost however early it comes. ``hold_when_still`` is the console's own hold: what it names (a lapsed
    heartbeat, the time limit) holds the arm only once it stands still (:class:`StillnessGate`), never while it moves in
    a person's hands, and one named before the arm is freed frees nothing; a request that cannot be read counts as one.
    A Save the person sent first is left to finish. ``watch`` gets every sample the session takes, for the console's
    live readout, and ``on_step`` each step as it happens: ``free`` once the arm is freed, ``holding`` once it holds,
    ``screening`` before the screen; either one that raises is logged and never ends the session.

    The arm is held on every answer before anything else, and on every way out of the session. A capture is read off
    the held arm and screened at once, while it holds (``screen_configuration``, the planner asked): CLEAR and BAND
    are handed to ``store``; an ERROR (the exact guard or the planner refuses it), an UNSCREENED verdict and a screen
    that raised are never written, and the result says why. Nothing here moves the arm by itself.
    """
    if ask_planner is not True:
        raise ValueError(
            f"teach_one(ask_planner={ask_planner!r}): every pose is screened with the planner asked, since only a pose "
            "the exact guard and the planner both clear, or one in the planner's band, is written; a screen without "
            "the planner could write nothing. Leave ask_planner out")
    arm, target = _ready_to_teach(robot, tree, name, store)
    guide.confirm_payload(arm)
    guide.console.say(f"Teaching {name}: it is written to {target} once the exact guard and the planner clear it or "
                      "find it in the planner's band; a pose either refuses, or one that could not be screened, is "
                      "never written.")
    guide.console.say(f"Watched while you guide the arm: {limits.window}. Outside either you are told, and Save does "
                      "not capture there; the arm is never held or stopped for it.")
    guide.console.say("Nothing moves by itself: the arm moves only while you move it. Save holds it once it stands "
                      "still, and Cancel holds it at once.")
    asked = _asked_to_hold(hold_when_still)
    if asked:
        guide.console.say(f"Not freed: the console asked for a hold before the arm was freed ({asked}). Nothing "
                          "was taught.")
        return TaughtOne(name=name, choice=HELD_WHEN_STILL, because=asked)
    gate = StillnessGate(guide.still_rad_s, guide.still_for_s)
    console = _HoldOnceStill(guide.console, gate, hold_when_still)
    own_console, own_view = guide.console, guide.view
    view = None if own_view is None else _ViewKeys(own_view)
    pose: TaughtPose | None = None
    screen: PoseScreen | None = None
    screen_error, choice = "", ""
    guide.console, guide.view = console, view
    try:
        with arm.freedrive() as session:
            watched = _Watched(session, gate, guide.clock, watch, look=lambda: console.look() or _looked(view))
            watched.free()
            _step(on_step, "free")
            try:
                choice = guide.wait(watched, limits, status=lambda _sample: ([f"teaching {name}"], None),
                                    banner=f"MOVE THE ARM BY HAND to where {name} should be, then save it")
            except _HoldNow as now:
                # A key that holds, read off a sample: it wins over a Save still waiting for the arm to stand still.
                choice = now.choice
                _show(guide, [])
            # Held on every answer, before anything else: nothing samples a free arm while the rest runs.
            session.hold()
            _step(on_step, "holding")
            if console.because:
                choice = HELD_WHEN_STILL
            elif choice == CAPTURE:
                _show(guide, ["HOLDING THE ARM", f"{name} is read, then screened"])
                pose = TaughtPose.from_sample(name, session.sample(), rig=rig, mounting=mounting)
                _step(on_step, "screening")
                screen, screen_error = _screen_held(arm, pose, guide)
    finally:
        guide.console, guide.view = own_console, own_view
    if choice == HELD_WHEN_STILL:
        guide.console.say(f"Held once it stood still ({console.because}): {name} was not taught, and the arm holds "
                          "where it stands.")
        return TaughtOne(name=name, choice=HELD_WHEN_STILL, because=console.because)
    if choice != CAPTURE or pose is None:
        guide.console.say(f"{name} was not taught, and the arm holds where it stands.")
        return TaughtOne(name=name, choice=choice)
    written, not_written = "", _not_written(name, screen, screen_error)
    if not not_written:
        assert screen is not None  # a pose with no screen has a reason it was not written
        try:
            said = store.write(pose, screen)
        except Exception as exc:  # noqa: BLE001 (the arm holds, and the person is told: nothing is lost but the pose)
            logger.exception("the pose store raised writing %s", name)
            said = f"the pose store raised {type(exc).__name__}: {exc}"
        if said:
            not_written = f"{name} was not written: {said}"
        else:
            written = target
    guide.console.say(f"{name} written to {written}; the arm holds where it stands." if written else
                      f"{not_written} The arm holds where it stands.")
    return TaughtOne(name=name, choice=CAPTURE, pose=pose, screen=screen, screen_error=screen_error, written=written,
                     not_written=not_written)


def _guidable_for_one(robot: Any) -> SupportsFreedrive:
    """The arm of ``robot`` as one a person can guide and that is connected, or :class:`TeachRefused` with its code."""
    try:
        return _guidable(robot)
    except HandGuidingRefused as refused:
        arm = getattr(robot, "arm", robot)
        code = "not_connected" if isinstance(arm, SupportsFreedrive) else "no_hand_guiding"
        raise TeachRefused(code, str(refused)) from None


def _screened_here_or_refused(robot: Any, arm: Any, tree: Any) -> None:
    """Refuse an arm on which no pose could be screened honestly: no pose taught there could ever be written."""
    if not _screens(arm):
        raise TeachRefused("screen_unavailable", (
            f"{_named(arm)} screens no pose (it offers no screen_configuration), so no pose taught on it could be "
            "written: an unscreened pose never is. Nothing was freed"))
    unscreened = _unscreened_because(robot, arm, tree)
    if unscreened is not None:
        raise TeachRefused("screen_unavailable", f"{unscreened} An unscreened pose is never written, so nothing was "
                                                 "freed.")


def _planner_ready_or_refused(arm: Any) -> None:
    """Refuse a planner the arm reports not ready (``planner_state``, a property or a method): every pose is screened the
    moment it is held, and a screen would start it, about a minute, while a person stands at the arm. An arm that
    reports no planner state leaves it to the screen: a planner that cannot be asked gives an UNSCREENED verdict, which
    is never written."""
    try:
        state = getattr(arm, "planner_state", None)
        if callable(state):
            state = state()
    except Exception as exc:  # noqa: BLE001 (a state nobody can read is no ready planner)
        raise TeachRefused("planner_not_ready", (
            f"the planner's state could not be read ({type(exc).__name__}: {exc}), so it cannot be said to be ready: "
            "start the planner first, then teach. Nothing was freed")) from None
    if state is None or state == "ready":
        return
    if state == "not_used":
        raise TeachRefused("screen_unavailable", (
            "this arm plans with no planner (robot.ur.motion_planner is not curobo), so no pose taught on it could be "
            "screened, and an unscreened pose is never written. Nothing was freed"))
    raise TeachRefused("planner_not_ready", (
        f"the planner is {state!r}, not ready: every pose is screened the moment it is held, and a screen would start "
        "the planner (about a minute) while a person stands at the arm. Start the planner first, then teach. Nothing "
        "was freed"))


def _step(on_step: Callable[[str], None] | None, step: str) -> None:
    """Tell the caller the session reached ``step``. Display only: a caller that raises never ends a session."""
    if on_step is None:
        return
    try:
        on_step(step)
    except Exception:  # noqa: BLE001 (display only: the arm is never left free for it)
        logger.exception("the teach session's step listener raised at %r; the session goes on", step)


def _asked_to_hold(hold_when_still: Callable[[], str] | None) -> str:
    """What the console's hold request names now, ``""`` for none; one that cannot be read counts as one."""
    if hold_when_still is None:
        return ""
    try:
        return " ".join(str(hold_when_still() or "").split())
    except Exception as exc:  # noqa: BLE001 (a request nobody can read holds, never frees)
        return f"the hold request could not be read ({type(exc).__name__}: {exc})"


def _screen_held(arm: Any, pose: TaughtPose, guide: HandGuide) -> "tuple[PoseScreen | None, str]":
    """Screen a pose just read off the held arm, the planner asked, and say the verdict; ``(None, why)`` where it raised.

    Nothing moves for it, and a screen that fails never stops the session: the pose is then simply not written.
    """
    guide.console.say(f"{pose.name} read: {pose.tcp_line()}. It is screened now by the exact guard and the planner; "
                      "the arm holds where it stands.")
    try:
        screen = arm.screen_configuration(pose.joints, ask_planner=True)
    except Exception as exc:  # noqa: BLE001 (said, and the pose is not written: an unscreened pose never is)
        why = f"{type(exc).__name__}: {exc}"
        guide.console.say(f"{pose.name} was not screened: {why}")
        return None, why
    guide.console.say(screen.line(pose.name))
    return screen, ""


def _not_written(name: str, screen: "PoseScreen | None", screen_error: str) -> str:
    """Why a captured pose is not written, or ``""`` where its screen lets it be: CLEAR or BAND."""
    if screen is None:
        return f"{name} was not written: it could not be screened ({screen_error}), and an unscreened pose never is."
    if screen.is_error:
        return f"{name} was not written: {screen.render()}"
    if _verdict(screen) not in _WRITTEN:
        return f"{name} was not written: it was not screened ({screen.detail}), and an unscreened pose never is."
    return ""


def _verdict(screen: Any) -> str:
    """A screen's verdict as its plain value (``clear``, ``band``, ...), ``""`` for none."""
    verdict = getattr(screen, "verdict", "")
    return str(getattr(verdict, "value", verdict) or "")


class _HoldNow(Exception):
    """A key that holds the arm at once, read off a sample of :func:`teach_one`'s session: raised out of
    ``HandGuide.wait``, also out of its stillness wait, which reads no key itself, and caught in the session, which
    holds at once. ``choice`` is what the key asked for: :data:`FINISH` or :data:`SKIP`."""

    def __init__(self, choice: str) -> None:
        super().__init__(choice)
        self.choice = choice


class _HoldOnceStill:
    """The console ``HandGuide.wait`` polls during :func:`teach_one`, with the console's own hold in it: once the hold
    request names a reason and the stillness gate says still, it answers a finish, exactly once, so the wait ends and
    the arm is held. A line the person sent always comes first, a line :meth:`look` read early included."""

    def __init__(self, console: OperatorConsole, gate: StillnessGate,
                 hold_when_still: Callable[[], str] | None) -> None:
        self._console = console
        self._gate = gate
        self._hold_when_still = hold_when_still
        #: Lines :meth:`look` read before the wait asked for them, handed on in order.
        self._kept: deque[str] = deque()
        #: What asked for the hold, once the finish went out; ``""`` before.
        self.because = ""

    def say(self, line: str) -> None:
        self._console.say(line)

    def bell(self) -> None:
        self._console.bell()

    def poll(self) -> str | None:
        line = self._kept.popleft() if self._kept else self._console.poll()
        if line is not None or self.because or self._hold_when_still is None or not self._gate.still:
            return line
        because = _asked_to_hold(self._hold_when_still)
        if not because:
            return None
        self.because = because
        return "q"

    def look(self) -> str | None:
        """Read what the person typed since the last look: the choice of a line that holds (:data:`FINISH`,
        :data:`SKIP`), or ``None``, every other line kept for :meth:`poll` in order."""
        while (line := self._console.poll()) is not None:
            held = _CONSOLE_HOLDS.get(line.strip().lower())
            if held is not None:
                return held
            self._kept.append(line)
        return None


class _ViewKeys:
    """The view ``HandGuide.wait`` reads during :func:`teach_one`: the guide's own view, with the keys :meth:`look` read
    early handed on in order. Everything else is the view's own."""

    def __init__(self, view: GuideView) -> None:
        self._view = view
        self._kept: deque[str] = deque()

    def guide(self, lines: Sequence[str], tone: str, bar: float | None = None) -> None:
        self._view.guide(lines, tone, bar)

    def poll_key(self) -> str | None:
        return self._kept.popleft() if self._kept else self._view.poll_key()

    def look(self) -> str | None:
        """Read the keys sent since the last look: the choice of one that holds (a finish, a closed window, a skip), or
        ``None``, every other key kept for :meth:`poll_key` in order."""
        while (key := self._view.poll_key()) is not None:
            held = _VIEW_HOLDS.get(key)
            if held is not None:
                return held
            self._kept.append(key)
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # its own slots, before __init__ set them: never the view's
            raise AttributeError(name)
        return getattr(self._view, name)


def _looked(view: _ViewKeys | None) -> str | None:
    """What a look at ``view`` read that holds, ``None`` without a view."""
    return None if view is None else view.look()


class _Watched:
    """The freedrive session ``HandGuide.wait`` samples during :func:`teach_one`: every sample also feeds the stillness
    gate and the caller's watcher, and then reads what the person sent (``look``): a key that holds raises
    :class:`_HoldNow` out of the wait at once, also out of its stillness wait, which reads no key itself. The watcher
    is display only: one that raises is logged once and never ends the session."""

    def __init__(self, session: "FreedriveSession", gate: StillnessGate, clock: Callable[[], float],
                 watch: Callable[[FreedriveSample], None] | None,
                 look: Callable[[], str | None] | None = None) -> None:
        self._session = session
        self._gate = gate
        self._clock = clock
        self._watch = watch
        self._look = look
        self._said = False

    @property
    def is_free(self) -> bool:
        return bool(self._session.is_free)

    def free(self) -> None:
        self._gate.reset()
        self._session.free()

    def hold(self) -> None:
        self._session.hold()

    def sample(self) -> FreedriveSample:
        sample = self._session.sample()
        self._gate.feed(sample, self._clock())
        if self._watch is not None:
            try:
                self._watch(sample)
            except Exception:  # noqa: BLE001 (display only: a person guiding the arm is never stopped for it)
                if not self._said:
                    logger.exception("the teach session's watcher raised; the session goes on")
                    self._said = True
        held = self._look() if self._look is not None else None
        if held:
            raise _HoldNow(held)
        return sample

    def __enter__(self) -> "_Watched":
        self._session.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._session.__exit__(*exc)


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
    """The camera section (``camera.cameras``) of a loaded tree, or of the ``AppConfig`` handed in itself (the console
    holds one), or ``None`` where there is no tree. A tree that did not load refuses with its own refusal, as every door
    that reads one does."""
    if tree is None:
        return None
    config = getattr(tree, "app_config", None)
    return getattr(getattr(tree if config is None else config, "camera", None), "cameras", None)


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
