"""A pose taught by hand: a person stands the arm where a pose should be, and Enter prints the line to paste.

The owner, 2026-09-24: the joint poses a program declares, the look poses of a camera pick above all, are taught by
guiding the arm by hand instead of copying numbers off the pendant or ``--where``. The arm is built alone and has to
offer hand guiding; the payload is confirmed before it is first freed; then, pose by pose, the console asks a name
(Enter takes ``LOOK_<n>``, q finishes) while the arm holds, frees the arm, and Enter captures once the arm has stood
still: the arm is held, its joints and its TCP are read, one line to paste is printed,
``LOOK_1 = JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)``, and the pose is appended to a JSON file that
keeps every pose taught before. What is held here, over a scripted hand-guided arm, a scripted console and a clock of
the test's own:

* the payload is shown and confirmed once, before the arm is first freed, and unconfirmed nothing is freed;
* Enter captures only once the arm has stood still; then the arm is held and the exact line is printed;
* the file gains every pose and keeps those taught before as they were; a file that is not taught poses is refused
  before anything is freed, and left as it was;
* outside the cable window or the workspace box the person is told (in red in a preview), Enter does not capture
  there, and the arm is never held because of it;
* q finishes and leaves the arm held, at the name prompt and while the arm is free; Ctrl-C leaves it held too, and
  nothing asks for the hands off the arm, because nothing moves by itself;
* a name the person types is used, one that is no Python name is asked again, and a name taught again takes the
  place of the earlier pose while the file keeps both;
* an arm that offers no hand guiding is refused naming its vendor, and one not connected is refused, before anything
  is asked;
* a pose belongs to one camera (the owner, 2026-09-24 evening: an eye-in-hand and an eye-to-hand camera are calibrated
  differently, and each gets poses of its own): the rig is the one camera handed in, or the one the tree names
  (``for_rig=``, the primary by default), window or none; its record keeps the rig and the mounting the tree declares,
  the line to paste says both in a comment, a camera of another rig is refused before anything is asked, poses taught
  before a pose had a rig still read, as rig unknown, and ``read_taught_poses`` keeps one rig's poses where asked;
* given the camera, a window beside the console shows what it sees, the pose and the count, red with the reason outside
  a boundary, opened once the payload is confirmed; Enter in it captures, q or closing it finishes with the arm held,
  and it is closed before the camera is given back on every way out; its thread draws and reads the camera through
  its display path, and never calls the arm; where no window can show, or the camera gives no frame, one line says why
  and the poses are taught at the console, still for that rig;
* example 11 opens the one camera its poses are taught for, the primary unless it names another, and with no window
  still hands the rig on.

Nothing here has run beside a physical arm; the UR teach mode behind ``SupportsFreedrive`` is the freedrive
workstream's, and its own tests hold it.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.calibration import preview as preview_module
from src.camera import live_view as live_view_module
from src.camera.live_view import LiveView
from src.camera.orchestration.camera import Camera
from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, RobotCapabilities
from src.robot.core.freedrive import ControllerPayload, FreedriveSample, SupportsFreedrive
from src.robot.execution.hand_guiding import EOF, HandGuide, HandGuidingRefused
from src.robot.execution.teach import TaughtPose, read_taught_poses, teach_poses
from tests.test_calibration_sweep_preview import _Gui, _until
from tests.test_camera_boundaries import _rgbd_rig
from tests.test_every_camera_of_the_cell_can_be_watched_live import _Desk

#: Where the tests' person stands the arm, in degrees: the owner's own line, with a wrist 3 a hair below zero that must
#: not print as -0.0, and a second look.
LOOK_1_DEG = (-45.0, -100.2, -110.0, -60.0, 90.0, -0.00001)
LOOK_2_DEG = (-70.0, -95.5, -112.3, -58.0, 90.0, 12.0)
#: The lines to paste, taught for the tests' primary rig, a camera on the wrist.
LOOK_1_LINE = "LOOK_1 = JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)  # rig 'wrist', eye in hand"
LOOK_2_LINE = "LOOK_2 = JointPositions.deg(-70.0, -95.5, -112.3, -58.0, 90.0, 12.0)  # rig 'wrist', eye in hand"
#: The TCP at every stand unless a test says otherwise: inside the box, the tool turned 170 degrees about base y.
TCP_MM = (450.0, -120.0, 300.0)
TCP_ROTVEC = (0.0, math.radians(170.0), 0.0)
#: The tests' workspace box, and the cable window of the tests' arm: half a turn about a home at zero, less 5 degrees.
BOX = WorkspaceLimitsConfig(x_min=-1000.0, x_max=1000.0, y_min=-1000.0, y_max=1000.0, z_min=50.0, z_max=1200.0)
WINDOW = ([-180.0] * 6, [180.0] * 6)
#: The question before the arm is first freed.
PAYLOAD_QUESTION = "Is this payload right (hand + camera + bracket)? Enter = yes, n = no"


def _rig(rig_id: str, mounting: str | None = None, *, body: bool = False) -> SimpleNamespace:
    """A rig of a camera section, as teaching reads it: its id, whether the arm carries it, and its calibration."""
    return SimpleNamespace(rig_id=rig_id, enabled=True, source="rgbd", body=object() if body else None,
                           extrinsics=None if mounting is None else SimpleNamespace(mounting_mode=mounting))


def _tree(primary: str = "wrist") -> SimpleNamespace:
    """A loaded tree as teaching reads it: a camera on the wrist (the primary), a fixed one over the table, and one
    whose mounting is not declared yet (not calibrated, no body)."""
    rigs = [_rig("wrist", "eye_in_hand", body=True), _rig("overhead", "eye_to_hand"), _rig("side")]
    return SimpleNamespace(app_config=SimpleNamespace(camera=SimpleNamespace(
        cameras=SimpleNamespace(primary_rig_id=primary, rigs=rigs))))


#: The tree every run is taught against unless a test says otherwise.
TREE = _tree()


def _sample(joints_deg: tuple[float, ...], *, xyz: tuple[float, float, float] = TCP_MM,
            speed: float = 0.0) -> FreedriveSample:
    return FreedriveSample(joints_rad=tuple(math.radians(v) for v in joints_deg),
                           joint_speeds_rad_s=(speed, 0.0, 0.0, 0.0, 0.0, 0.0),
                           tcp_xyz_mm=xyz, tcp_rotvec_rad=TCP_ROTVEC, t_s=0.0)


def _stand(joints_deg: tuple[float, ...], *, moving: int = 3) -> list[FreedriveSample]:
    """A person moving the arm to ``joints_deg``: a few samples on the way, then the arm standing still there."""
    return [_sample(joints_deg, speed=0.3)] * moving + [_sample(joints_deg)]


def _answers(*typed: str | None) -> list[str | None]:
    """The console for prompts answered in turn: a quiet poll that ends each prompt's drain, then what was typed."""
    return [item for answer in typed for item in (None, answer)]


class _Clock:
    """The time the guide runs on. A guide that waits for more than ``budget_s`` fails the test instead of hanging."""

    def __init__(self, budget_s: float = 600.0) -> None:
        self.now = 0.0
        self.budget_s = budget_s
        self.on_sleep: Any = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        self.now += seconds
        if self.now > self.budget_s:
            raise AssertionError("the guide waited for longer than the test allows: a prompt nobody answers")


class _Console:
    """A scripted person at the terminal: ``poll`` returns the next item (``None``: nothing typed), then nothing."""

    def __init__(self, lines: list[str | None], log: list[str]) -> None:
        self.lines = list(lines)
        self.log = log
        self.said: list[str] = []

    def say(self, line: str) -> None:
        self.said.append(line)
        self.log.append(f"say: {line}")

    def bell(self) -> None:
        pass

    def poll(self) -> str | None:
        return self.lines.pop(0) if self.lines else None


class _View:
    """A preview double: keeps what the guide showed, and hands back scripted keys."""

    def __init__(self, keys: list[str | None] | None = None) -> None:
        self.keys = list(keys or [])
        self.shown: list[tuple[list[str], str]] = []

    def guide(self, lines: Any, tone: str, bar: float | None = None) -> None:
        self.shown.append((list(lines), tone))

    def poll_key(self) -> str | None:
        return self.keys.pop(0) if self.keys else None


class _Session:
    """A hand-guiding session over :class:`_GuidedArm`: each ``free`` starts the next scripted stand."""

    def __init__(self, arm: "_GuidedArm") -> None:
        self.arm = arm
        self._free = False

    @property
    def is_free(self) -> bool:
        return self._free

    def free(self) -> None:
        self.arm.threads.add(threading.get_ident())
        self.arm.log.append("free")
        self._free = True
        self.arm.stand += 1

    def hold(self) -> None:
        self.arm.threads.add(threading.get_ident())
        self.arm.log.append(f"hold at {self.arm.clock():.2f}")
        self._free = False

    def sample(self) -> FreedriveSample:
        self.arm.threads.add(threading.get_ident())
        return self.arm.next_sample()

    def __enter__(self) -> "_Session":
        self.arm.log.append("session")
        self.arm.open_sessions += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self._free = False
        self.arm.open_sessions -= 1
        self.arm.log.append("left, held")


class _GuidedArm:
    """A connected arm a person can guide (``SupportsFreedrive``), scripted stand by stand. It never moves by itself:
    every motion verb fails the test."""

    def __init__(self, stands: list[list[FreedriveSample]], clock: _Clock, *, vendor: str = "ur",
                 payload: ControllerPayload | None = ControllerPayload(1.9, (0.0, 12.0, 61.0))) -> None:
        self.stands = [list(stand) for stand in stands]
        self.clock = clock
        self.payload = payload
        self.capabilities = RobotCapabilities(vendor=vendor)
        self.is_connected = True
        self.config = SimpleNamespace(workspace_limits=BOX)
        self.safety_preflight = SimpleNamespace(guards=(SimpleNamespace(
            name="joint_limit", margin_deg=5.0, within_half_turn_of_home=True, limits_for_arm=lambda _arm: WINDOW),))
        self.log: list[str] = []
        self.stand = -1
        self.sessions = 0
        self.open_sessions = 0
        #: When each stand first served a still sample, on ``clock``.
        self.still_since: dict[int, float] = {}
        #: Every thread that called the arm or its session.
        self.threads: set[int] = set()

    # --- SupportsFreedrive ---
    def freedrive(self) -> _Session:
        self.threads.add(threading.get_ident())
        self.sessions += 1
        return _Session(self)

    def controller_payload(self) -> ControllerPayload | None:
        self.threads.add(threading.get_ident())
        self.log.append("payload read")
        return self.payload

    # --- the script ---
    def next_sample(self) -> FreedriveSample:
        stand = self.stands[min(self.stand, len(self.stands) - 1)]
        sample = stand.pop(0) if len(stand) > 1 else stand[0]
        if sample.peak_joint_speed_rad_s == 0.0:
            self.still_since.setdefault(self.stand, self.clock())
        return sample

    # --- nothing here may move the arm ---
    def move(self, *_args: Any, **_keywords: Any) -> None:
        raise AssertionError("the arm was commanded to move while poses were taught by hand")

    move_to_joints = move_joint = move_linear = move_to = move_home = move


class _PlainArm(_GuidedArm):
    """The same arm with no hand guiding."""

    freedrive = None  # type: ignore[assignment]
    controller_payload = None  # type: ignore[assignment]


class _Case(unittest.TestCase):
    """A fresh file, a fresh clock, and one teaching run at a time over a scripted arm and console."""

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = self.folder / "logs" / "taught_poses.json"
        self.clock = _Clock()
        self.printed: list[str] = []

    def arm(self, *stands: list[FreedriveSample], plain: bool = False, **keywords: Any) -> _GuidedArm:
        self.guided = (_PlainArm if plain else _GuidedArm)(list(stands), self.clock, **keywords)
        return self.guided

    def teach(self, answers: list[str | None], *, view: Any = None, robot: Any = None,
              **keywords: Any) -> tuple[TaughtPose, ...]:
        """Teach on :attr:`guided` (or ``robot``), answered by ``answers``, for the rig ``TREE`` names unless the
        test says otherwise; what the person saw is on ``console``."""
        self.console = _Console(answers, self.guided.log)
        self.guide = HandGuide(self.console, view, clock=self.clock, sleep=self.clock.sleep)
        keywords.setdefault("tree", TREE)
        return teach_poses(self.guided if robot is None else robot, store=self.store, guide=self.guide,
                           out=self.printed.append, **keywords)

    def steps(self) -> list[str]:
        """What the arm was asked and did, in order, without the console's lines, a window's life and the hold
        times."""
        return [line.split(" at ")[0] for line in self.guided.log if not line.startswith(("say", "window.", "camera."))]

    def records(self) -> list[dict[str, Any]]:
        return json.loads(self.store.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------------
# Two poses, the names Enter gives, then q
# ---------------------------------------------------------------------------------------------------------------------


class APersonTeachesTwoPosesTests(_Case):

    def setUp(self) -> None:
        super().setUp()
        self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG))
        # The payload; the name of pose 1 and its capture; the name of pose 2 and its capture; q at the third name.
        self.taught = self.teach(_answers("", "", "", "", "", "q"))

    def test_the_payload_is_shown_and_confirmed_once_before_the_arm_is_first_freed(self) -> None:
        asked = [line for line in self.guided.log if line == f"say: {PAYLOAD_QUESTION}"]
        self.assertEqual(len(asked), 1)
        self.assertIn("1.90 kg", " ".join(self.console.said))
        self.assertLess(self.guided.log.index(asked[0]), self.guided.log.index("free"))
        self.assertLess(self.guided.log.index("payload read"), self.guided.log.index("session"))

    def test_enter_captures_once_the_arm_stood_still_then_holds_it_and_prints_the_line_to_paste(self) -> None:
        self.assertEqual(self.printed, [LOOK_1_LINE, LOOK_2_LINE])
        self.assertEqual(self.steps(), ["payload read", "session", "free", "hold", "free", "hold", "left, held"])
        holds = [float(line.split(" at ")[1]) for line in self.guided.log if line.startswith("hold at")]
        for stand, held_at in enumerate(holds):
            # Enter came on the first sample, while the arm still moved: the hold waited out half a second of stillness.
            self.assertGreaterEqual(round(held_at - self.guided.still_since[stand], 6), 0.5, stand)

    def test_each_pose_is_the_joints_and_the_tcp_the_held_arm_reported(self) -> None:
        self.assertEqual([pose.name for pose in self.taught], ["LOOK_1", "LOOK_2"])
        np.testing.assert_allclose(self.taught[1].joints_deg, LOOK_2_DEG, atol=1e-9)
        np.testing.assert_allclose(self.taught[0].tcp.position_mm, TCP_MM)
        self.assertIs(self.taught[0].tcp.frame, Frame.BASE)
        self.assertEqual([pose.line() for pose in self.taught], [LOOK_1_LINE, LOOK_2_LINE])
        said = " ".join(self.console.said)
        self.assertIn("LOOK_1 taught: TCP (base)  x 450.0  y -120.0  z 300.0 mm  rx 0.0  ry 170.0  rz 0.0 deg", said)

    def test_the_file_holds_every_pose_as_a_record_that_reads_back(self) -> None:
        records = self.records()
        self.assertEqual([record["name"] for record in records], ["LOOK_1", "LOOK_2"])
        self.assertEqual(records[0]["joints_deg"], [-45.0, -100.2, -110.0, -60.0, 90.0, 0.0])
        self.assertEqual(records[0]["tcp"], {"frame": "base", "x_mm": 450.0, "y_mm": -120.0, "z_mm": 300.0,
                                             "rotation_vector_deg": [0.0, 170.0, 0.0]})
        taught_at = datetime.fromisoformat(records[1]["taught_at"])
        self.assertIsNotNone(taught_at.tzinfo, "the time says its offset")
        self.assertEqual([(record["rig"], record["mounting"]) for record in records],
                         [("wrist", "eye_in_hand"), ("wrist", "eye_in_hand")], "the rig taught for, as the tree mounts it")
        read = read_taught_poses(self.store)
        self.assertEqual([(pose.rig, pose.mounting) for pose in read], [("wrist", "eye_in_hand")] * 2)
        self.assertEqual([pose.name for pose in read], ["LOOK_1", "LOOK_2"])
        self.assertEqual([pose.line() for pose in read], [LOOK_1_LINE, LOOK_2_LINE])
        np.testing.assert_allclose(read[1].tcp.position_mm, TCP_MM)

    def test_q_at_the_name_prompt_finishes_leaves_the_arm_held_and_nothing_asked_for_the_hands_off(self) -> None:
        self.assertEqual(self.guided.log[-1], "say: 2 poses taught for rig 'wrist', each added to "
                                              f"{self.store}. The arm holds where it stands.")
        self.assertEqual(self.steps()[-1], "left, held")
        self.assertEqual(self.guided.open_sessions, 0)
        self.assertEqual(self.guided.sessions, 1, "one session for the whole run")
        said = " ".join(self.console.said)
        self.assertNotIn("Hands off", said)
        self.assertNotIn("moves by itself in", said)

    def test_each_name_is_asked_while_the_arm_holds_with_the_next_free_default(self) -> None:
        asked = [i for i, line in enumerate(self.guided.log) if line.startswith("say: Name of the next pose?")]
        self.assertEqual([self.guided.log[i] for i in asked],
                         [f"say: Name of the next pose? Enter = LOOK_{n}, q finishes" for n in (1, 2, 3)])
        for index in asked:
            last_step = [line for line in self.guided.log[:index] if not line.startswith("say")][-1]
            self.assertTrue(last_step == "session" or last_step.startswith("hold"), last_step)


# ---------------------------------------------------------------------------------------------------------------------
# The file of taught poses
# ---------------------------------------------------------------------------------------------------------------------


class TheFileKeepsWhatWasTaughtBeforeTests(_Case):

    EARLIER = {"name": "HOME_VIEW", "joints_deg": [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
               "tcp": {"frame": "base", "x_mm": 400.0, "y_mm": 0.0, "z_mm": 600.0,
                       "rotation_vector_deg": [180.0, 0.0, 0.0]},
               "taught_at": "2026-09-23T10:00:00+02:00", "note": "kept as it was written"}

    def test_the_file_gains_the_pose_and_keeps_the_earlier_ones_as_they_were(self) -> None:
        self.store.parent.mkdir(parents=True)
        self.store.write_text(json.dumps([self.EARLIER]), encoding="utf-8")
        self.arm(_stand(LOOK_1_DEG))
        self.teach(_answers("", "", "", "q"))
        records = self.records()
        self.assertEqual(records[0], self.EARLIER, "a record taught before a pose had a rig is kept as it was")
        self.assertEqual([record["name"] for record in records], ["HOME_VIEW", "LOOK_1"])
        self.assertIn(f"Every pose is added to {self.store} (1 there already).", " ".join(self.console.said))
        self.assertFalse(list(self.store.parent.glob("*.partial")), "the file beside it was renamed onto it")
        earlier, taught = read_taught_poses(self.store)
        self.assertEqual((earlier.rig, earlier.mounting), (None, None))
        self.assertEqual(earlier.line(), "HOME_VIEW = JointPositions.deg(0.0, -90.0, 90.0, -90.0, -90.0, 0.0)  # rig unknown")
        self.assertEqual(taught.line(), LOOK_1_LINE)

    def test_one_rigs_poses_are_read_on_request_and_a_pose_of_no_known_rig_is_none_of_them(self) -> None:
        wrist = TaughtPose(name="LOOK_1", joints=JointPositions.deg(*LOOK_1_DEG), tcp=Pose.tool_down(450.0, 0.0, 300.0),
                           rig="wrist", mounting="eye_in_hand")
        overhead = TaughtPose(name="CLEAR", joints=JointPositions.deg(*LOOK_2_DEG),
                              tcp=Pose.tool_down(300.0, 0.0, 500.0), rig="overhead", mounting="eye_to_hand")
        self.store.parent.mkdir(parents=True)
        self.store.write_text(json.dumps([self.EARLIER, wrist.to_dict(), overhead.to_dict()]), encoding="utf-8")
        self.assertEqual([pose.name for pose in read_taught_poses(self.store)], ["HOME_VIEW", "LOOK_1", "CLEAR"])
        self.assertEqual([pose.name for pose in read_taught_poses(self.store, rig="wrist")], ["LOOK_1"])
        self.assertEqual([pose.name for pose in read_taught_poses(self.store, rig="overhead")], ["CLEAR"])
        self.assertEqual(read_taught_poses(self.store, rig="side"), ())

    def test_a_record_whose_rig_or_mounting_is_none_is_refused_naming_it(self) -> None:
        for field, value, why in (("rig", 5, "rig 5, which names no rig"),
                                  ("rig", "", "rig '', which names no rig"),
                                  ("mounting", "on the ceiling", "neither eye_in_hand nor eye_to_hand")):
            with self.subTest(f"{field}={value!r}"):
                self.store.parent.mkdir(parents=True, exist_ok=True)
                self.store.write_text(json.dumps([dict(self.EARLIER, **{field: value})]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, why):
                    read_taught_poses(self.store)

    def test_a_file_that_is_not_taught_poses_is_refused_before_anything_is_freed_and_left_as_it_was(self) -> None:
        stations = json.dumps([{"label": "s0", "joints_deg": [10.0, -90.0, 90.0, -90.0, -90.0, 0.0]}])
        for kind, text, why in (("an object", '{"poses": []}', "not a list of taught poses"),
                                ("not JSON", "[{", "is not JSON"),
                                ("a stations file", stations, "record 0 is not a taught pose")):
            with self.subTest(kind):
                self.store.parent.mkdir(parents=True, exist_ok=True)
                self.store.write_text(text, encoding="utf-8")
                self.arm(_stand(LOOK_1_DEG))
                with self.assertRaisesRegex(HandGuidingRefused, why) as raised:
                    self.teach(_answers("", "", "", "q"))
                self.assertIn(str(self.store), str(raised.exception))
                self.assertIn("nothing was freed", str(raised.exception))
                self.assertEqual(self.store.read_text(encoding="utf-8"), text)
                self.assertEqual((self.guided.log, self.guided.sessions), ([], 0), "nothing was asked or freed")


# ---------------------------------------------------------------------------------------------------------------------
# A pose belongs to one camera
# ---------------------------------------------------------------------------------------------------------------------


class APoseBelongsToOneCameraTests(_Case):
    """The owner, 2026-09-24 evening: an eye-in-hand and an eye-to-hand camera are calibrated differently, and each
    camera gets its own poses, so a pose always belongs to one camera: the rig is resolved before anything is asked,
    window or none, and every line and every record says it."""

    def _taught_for(self, **keywords: Any) -> list[tuple[str | None, str | None]]:
        self.arm(_stand(LOOK_1_DEG))
        self.teach(_answers("", "", "", "q"), **keywords)
        return [(record["rig"], record["mounting"]) for record in self.records()]

    def test_the_default_is_the_trees_primary_rig(self) -> None:
        self.assertEqual(self._taught_for(), [("wrist", "eye_in_hand")])
        self.assertEqual(self.printed, [LOOK_1_LINE])
        self.assertIn("Every pose taught here belongs to rig 'wrist', eye in hand: its line to paste and its record say "
                      "so.", self.console.said)
        self.assertEqual(self._taught_for(tree=_tree(primary="overhead"))[-1], ("overhead", "eye_to_hand"))

    def test_the_rig_the_tree_names_is_taught_for_with_its_mounting(self) -> None:
        self.assertEqual(self._taught_for(for_rig="overhead"), [("overhead", "eye_to_hand")])
        self.assertTrue(self.printed[0].endswith("  # rig 'overhead', eye to hand"), self.printed[0])
        self.assertEqual(self._taught_for(for_rig="side")[-1], ("side", None), "a mounting not declared is none")
        self.assertTrue(self.printed[-1].endswith("  # rig 'side', mounting not declared"), self.printed[-1])

    def test_without_a_window_the_poses_still_belong_to_the_rig(self) -> None:
        """What example 11 does with SHOW_CAMERA false: no camera handed in, the rig named from the tree."""
        self.assertEqual(self._taught_for(for_rig="overhead", camera=None), [("overhead", "eye_to_hand")])
        self.assertFalse([line for line in self.console.said if "camera window" in line.lower()])

    def test_given_a_camera_the_rig_is_the_cameras(self) -> None:
        with patch.object(preview_module, "preview_unavailable", lambda: "no display: DISPLAY and WAYLAND_DISPLAY "
                                                                          "are unset"):
            self.assertEqual(self._taught_for(camera=_Camera("overhead")), [("overhead", "eye_to_hand")])
            self.assertEqual(self._taught_for(camera=_Camera("overhead"), for_rig="overhead")[-1],
                             ("overhead", "eye_to_hand"))

    def test_a_robot_that_kept_its_tree_names_the_rig_by_itself(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        self.teach(_answers("", "", "", "q"), robot=SimpleNamespace(arm=self.guided, tree=TREE), tree=None)
        self.assertEqual([(record["rig"], record["mounting"]) for record in self.records()], [("wrist", "eye_in_hand")])

    def test_what_names_no_rig_or_another_one_is_refused_before_anything_is_asked(self) -> None:
        ways: dict[str, tuple[dict[str, Any], type[Exception], str]] = {
            "a camera for another rig": (dict(camera=_Camera("overhead"), for_rig="wrist"), ValueError,
                                         "camera= is rig 'overhead', and the poses are taught for rig 'wrist'"),
            "a rig the tree does not configure": (dict(for_rig="ceiling"), ValueError,
                                                  r"for_rig='ceiling' is no rig of camera.cameras.rigs \['overhead', "
                                                  r"'side', 'wrist'\]"),
            "a camera of a rig the tree does not configure": (dict(camera=_Camera("ceiling")), ValueError,
                                                              "camera= is rig 'ceiling', which camera.cameras.rigs"),
            "neither a tree nor a camera": (dict(tree=None), ValueError, "a pose always belongs to one camera"),
            "a rig named without its tree": (dict(tree=None, for_rig="wrist"), ValueError,
                                             "how rig 'wrist' is mounted"),
            "a camera that names no rig": (dict(camera=_Handle()), TypeError, "camera= names no rig"),
        }
        for way, (keywords, raised, why) in ways.items():
            with self.subTest(way):
                self.arm(_stand(LOOK_1_DEG))
                with self.assertRaisesRegex(raised, why):
                    self.teach(_answers("", "q"), **keywords)
                self.assertEqual((self.guided.log, self.guided.sessions), ([], 0), "something was asked or freed")
                self.assertFalse(self.store.exists())


# ---------------------------------------------------------------------------------------------------------------------
# Boundaries: said, never enforced
# ---------------------------------------------------------------------------------------------------------------------


class ABoundaryIsSaidNeverEnforcedTests(_Case):

    def _outside_then_back(self, outside: FreedriveSample) -> tuple[tuple[TaughtPose, ...], _View]:
        """Enter while the arm stands outside, quiet while the person brings it back, Enter inside, then q."""
        self.arm([outside] * 5 + [_sample(LOOK_1_DEG)])
        view = _View()
        taught = self.teach(_answers("", "", "") + [None] * 6 + ["", None, "q"], view=view)
        return taught, view

    def _held_only_once_inside(self) -> None:
        holds = [i for i, line in enumerate(self.guided.log) if line.startswith("hold")]
        self.assertEqual(len(holds), 1, "the only hold is the capture: a boundary never held the arm")
        back = next(i for i, line in enumerate(self.guided.log) if line == "say: Back inside.")
        self.assertLess(back, holds[0])
        self.assertEqual(sum("OUTSIDE:" in line for line in self.console.said), 1, "said once per crossing")

    def test_outside_the_cable_window_enter_does_not_capture_and_the_arm_is_never_held_for_it(self) -> None:
        wound = (*LOOK_1_DEG[:5], 200.0)
        taught, view = self._outside_then_back(_sample(wound, speed=0.3))
        said = " ".join(self.console.said)
        self.assertIn("OUTSIDE: wrist 3 stands at 200.0 deg, outside the cable window -175.0 to 175.0 deg: turn it "
                      "back down inside it. Enter does not capture here.", said)
        self.assertIn("Not captured: wrist 3 stands at 200.0 deg", said)
        self._held_only_once_inside()
        red = [lines for lines, tone in view.shown if tone == "outside"]
        self.assertTrue(red and red[0][0].startswith("OUTSIDE - NOT CAPTURED: wrist 3 stands at 200.0 deg"))
        self.assertEqual(self.printed, [LOOK_1_LINE], "the pose taught is the one inside")
        self.assertEqual(len(taught), 1)

    def test_below_the_box_enter_does_not_capture_and_the_arm_is_never_held_for_it(self) -> None:
        taught, view = self._outside_then_back(_sample(LOOK_1_DEG, xyz=(450.0, -120.0, 20.0), speed=0.3))
        said = " ".join(self.console.said)
        self.assertIn("OUTSIDE: the TCP stands 30 mm past the workspace box's z_min face (z 20.0 mm, the box starts "
                      "at 50.0 mm): bring it back inside. Enter does not capture here.", said)
        self._held_only_once_inside()
        self.assertTrue(any(tone == "outside" for _, tone in view.shown))
        np.testing.assert_allclose(taught[0].tcp.position_mm, TCP_MM)

    def test_what_is_watched_is_said_before_the_arm_is_freed(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        self.teach(_answers("", "q"))
        said = " ".join(self.console.said)
        self.assertIn("Watched while you guide the arm: the cable window, half a turn either side of home less the "
                      "5 deg margin, and the workspace box.", said)
        self.assertIn("the arm is never held or stopped for it", said)
        self.assertIn("Nothing moves by itself", said)


# ---------------------------------------------------------------------------------------------------------------------
# Every way out holds the arm
# ---------------------------------------------------------------------------------------------------------------------


class TheWaysOutHoldTheArmTests(_Case):

    def test_q_while_the_arm_is_free_holds_it_and_teaches_nothing(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        taught = self.teach(_answers("", "", "q"))
        self.assertEqual(taught, ())
        self.assertEqual(self.steps(), ["payload read", "session", "free", "hold", "left, held"])
        self.assertEqual(self.printed, [])
        self.assertFalse(self.store.exists(), "nothing taught, nothing written")

    def test_ctrl_c_while_the_arm_is_free_leaves_it_held_and_keeps_the_poses_taught_before(self) -> None:
        self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG))

        def interrupt(_seconds: float) -> None:
            if self.guided.log.count("free") == 2:
                raise KeyboardInterrupt

        self.clock.on_sleep = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.teach(_answers("", "", "", "") + [None])
        self.assertEqual((self.steps()[-2:], self.guided.open_sessions), (["free", "left, held"], 0))
        self.assertEqual(self.printed, [LOOK_1_LINE])
        self.assertEqual([record["name"] for record in self.records()], ["LOOK_1"])

    def test_a_payload_nobody_confirms_frees_nothing(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        with self.assertRaisesRegex(HandGuidingRefused, "the controller payload was not confirmed"):
            self.teach(_answers("n"))
        self.assertEqual(self.guided.sessions, 0)
        self.assertNotIn("free", self.guided.log)

    def test_s_holds_the_arm_teaches_nothing_and_asks_the_same_name_again(self) -> None:
        self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG))
        # The payload; LOOK_1, s; LOOK_1 again and its capture; q.
        taught = self.teach(_answers("", "", "s", "", "", "q"))
        self.assertEqual([pose.name for pose in taught], ["LOOK_1"])
        self.assertIn("LOOK_1 skipped: nothing taught, and the arm holds where it stands.", self.console.said)
        self.assertEqual(self.steps(), ["payload read", "session", "free", "hold", "free", "hold", "left, held"])
        self.assertEqual(self.printed, [LOOK_2_LINE.replace("LOOK_2", "LOOK_1")])

    def test_input_that_has_ended_at_the_name_prompt_finishes(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        self.assertEqual(self.teach(_answers("", EOF)), ())
        self.assertEqual(self.steps(), ["payload read", "session", "left, held"])


# ---------------------------------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------------------------------


class NamesTests(_Case):

    def test_a_name_the_person_types_is_used_in_the_line_and_the_file(self) -> None:
        self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG))
        taught = self.teach(_answers("", "  look_left ", "", "", "", "q"))
        self.assertEqual(self.printed, [LOOK_1_LINE.replace("LOOK_1", "look_left"),
                                        LOOK_2_LINE.replace("LOOK_2", "LOOK_1")])
        self.assertEqual([record["name"] for record in self.records()], ["look_left", "LOOK_1"])
        self.assertEqual([pose.name for pose in taught], ["look_left", "LOOK_1"])

    def test_a_name_that_is_no_python_name_is_asked_again(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        taught = self.teach(_answers("", "look left", "1st", "class", "look_left", "", "q"))
        self.assertEqual([pose.name for pose in taught], ["look_left"])
        for typed in ("look left", "1st"):
            self.assertIn(f"{typed!r} is not a name here: a name is pasted as Python, so it is letters, digits and "
                          "underscores, and it does not start with a digit (LOOK_1, look_left).", self.console.said)
        self.assertIn("'class' is not a name here: class is a Python keyword.", self.console.said)
        self.assertEqual(self.steps().count("free"), 1, "the arm was freed once, for the name it took")

    def test_a_name_taught_again_takes_the_place_of_the_earlier_pose_and_the_file_keeps_both(self) -> None:
        retaught = (-80.0, -90.0, -100.0, -70.0, 90.0, 5.0)
        self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG), _stand(retaught))
        # LOOK_1, LOOK_2, then LOOK_1 again somewhere else.
        taught = self.teach(_answers("", "", "", "", "", "LOOK_1", "", "q"))
        self.assertEqual([pose.name for pose in taught], ["LOOK_1", "LOOK_2"], "in the order first taught")
        self.assertEqual(taught[0].line(),
                         "LOOK_1 = JointPositions.deg(-80.0, -90.0, -100.0, -70.0, 90.0, 5.0)  # rig 'wrist', eye in hand")
        self.assertIn("LOOK_1 was taught already in this run: this pose takes its place, and the file keeps both, "
                      "each with its time.", self.console.said)
        self.assertEqual([record["name"] for record in self.records()], ["LOOK_1", "LOOK_2", "LOOK_1"])
        self.assertEqual(self.records()[0]["joints_deg"][0], -45.0, "the earlier record is kept as it was")

    def test_the_default_is_the_first_name_not_taken_in_this_run(self) -> None:
        self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG))
        # The person calls the first pose LOOK_2; Enter then takes LOOK_1, and after it LOOK_3 is offered.
        taught = self.teach(_answers("", "LOOK_2", "", "", "", "q"), prefix="LOOK")
        self.assertEqual([pose.name for pose in taught], ["LOOK_2", "LOOK_1"])
        offered = [line for line in self.console.said if line.startswith("Name of the next pose?")]
        self.assertEqual(offered, [f"Name of the next pose? Enter = LOOK_{n}, q finishes" for n in (1, 1, 3)])

    def test_a_prefix_that_makes_no_python_name_is_refused_before_anything_is_asked(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        with self.assertRaisesRegex(ValueError, "prefix 'look pose' makes no Python name"):
            self.teach(_answers("", "q"), prefix="look pose")
        self.assertEqual(self.guided.log, [])


# ---------------------------------------------------------------------------------------------------------------------
# An arm nobody can guide, or one not connected
# ---------------------------------------------------------------------------------------------------------------------


class AnArmThatCannotBeGuidedTests(_Case):

    def test_an_arm_without_hand_guiding_is_refused_naming_its_vendor_before_anything_is_asked(self) -> None:
        self.arm(_stand(LOOK_1_DEG), plain=True, vendor="kuka")
        self.assertNotIsInstance(self.guided, SupportsFreedrive)
        with self.assertRaisesRegex(HandGuidingRefused, r"_PlainArm \(vendor 'kuka'\) offers no hand guiding "
                                                        r"\(SupportsFreedrive; the UR teach mode is one\)"):
            self.teach(_answers("", "", "", "q"))
        self.assertEqual((self.console.said, self.guided.log, self.guided.sessions), ([], [], 0))

    def test_a_robot_whose_arm_offers_none_is_refused_the_same_way(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.execution.robot import Robot

        robot = Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None)
        self.arm(_stand(LOOK_1_DEG))
        with self.assertRaisesRegex(HandGuidingRefused, r"DummyRobotArm \(vendor 'dummy'\) offers no hand guiding"):
            self.teach(_answers("", "q"), robot=robot)

    def test_an_arm_that_is_not_connected_is_refused_before_anything_is_asked(self) -> None:
        self.arm(_stand(LOOK_1_DEG))
        self.guided.is_connected = False
        with self.assertRaisesRegex(HandGuidingRefused, "is not connected.*with robot.connected"):
            self.teach(_answers("", "", "", "q"))
        self.assertEqual((self.console.said, self.guided.log, self.guided.sessions), ([], [], 0))


# ---------------------------------------------------------------------------------------------------------------------
# The value, and the question the guide asks
# ---------------------------------------------------------------------------------------------------------------------


class TheTaughtPoseTests(unittest.TestCase):

    def test_the_line_is_one_decimal_per_joint_and_never_minus_zero(self) -> None:
        pose = TaughtPose(name="LOOK_1", joints=JointPositions.deg(*LOOK_1_DEG),
                          tcp=Pose.tool_down(450.0, -120.0, 300.0), rig="wrist", mounting="eye_in_hand")
        self.assertEqual(pose.line(), LOOK_1_LINE)
        self.assertNotIn("-0.0", pose.line())
        self.assertEqual(pose.to_dict()["joints_deg"][5], 0.0)
        self.assertEqual(str(pose.to_dict()["joints_deg"][5]), "0.0", "no -0.0 in the file either")

    def test_the_line_says_the_rig_and_how_it_is_mounted_in_a_comment(self) -> None:
        joints, tcp = JointPositions.deg(*LOOK_1_DEG), Pose.tool_down(450.0, -120.0, 300.0)
        head = "LOOK_1 = JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)  # "
        for rig, mounting, said in (("wrist", "eye_in_hand", "rig 'wrist', eye in hand"),
                                    ("overhead", "eye_to_hand", "rig 'overhead', eye to hand"),
                                    ("side", None, "rig 'side', mounting not declared"),
                                    (None, None, "rig unknown")):
            with self.subTest(said):
                self.assertEqual(TaughtPose(name="LOOK_1", joints=joints, tcp=tcp, rig=rig, mounting=mounting).line(),
                                 head + said)

    def test_a_record_reads_back_as_the_pose_it_was_written_from(self) -> None:
        pose = TaughtPose(name="look_left", joints=JointPositions.deg(*LOOK_2_DEG),
                          tcp=Pose.tool_down(300.0, 200.0, 400.0, yaw_deg=30.0), taught_at="2026-09-24T21:03:12+02:00",
                          rig="overhead", mounting="eye_to_hand")
        record = json.loads(json.dumps(pose.to_dict()))
        self.assertEqual((record["rig"], record["mounting"]), ("overhead", "eye_to_hand"))
        again = TaughtPose.from_dict(record)
        self.assertEqual((again.name, again.taught_at, again.line(), again.rig, again.mounting),
                         (pose.name, pose.taught_at, pose.line(), "overhead", "eye_to_hand"))
        # The file keeps a ten-thousandth of a degree and a hundredth of a millimetre.
        np.testing.assert_allclose(again.tcp.to_matrix(), pose.tcp.to_matrix(), atol=1e-5)
        np.testing.assert_allclose(again.joints_deg, LOOK_2_DEG, atol=1e-4)


class TheGuideAsksAQuestionTests(unittest.TestCase):

    def test_ask_says_the_question_and_returns_the_line_typed_or_what_the_window_answered(self) -> None:
        clock = _Clock()
        console = _Console(["stale Enter", None, None, "LOOK_7"], [])
        guide = HandGuide(console, None, clock=clock, sleep=clock.sleep)
        self.assertEqual(guide.ask("Name of the next pose?"), "LOOK_7")
        self.assertEqual(console.said, ["Name of the next pose?"], "what was typed before the question is dropped")
        for key, answer in (("enter", ""), ("finish", "q"), ("closed", "q")):
            with self.subTest(key):
                view = _View([None, key])
                windowed = HandGuide(_Console([], []), view, clock=clock, sleep=clock.sleep)
                self.assertEqual(windowed.ask("Name?"), answer)
                self.assertEqual(view.shown[0], (["Name?"], "guide"))


# ---------------------------------------------------------------------------------------------------------------------
# A window beside the console: what the one camera sees while the arm is guided
# ---------------------------------------------------------------------------------------------------------------------


class _Terminal:
    """A stdout that is a terminal, so a window may open where the test says one can."""

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


class _Handle:
    """The device behind a real ``Camera`` owner, or a camera itself: frames, noting the thread that asked; a handle
    that fails gives none."""

    def __init__(self, log: list[str] | None = None, *, fail: bool = False) -> None:
        self.log = log
        self.fail = fail
        self.grabs: list[int] = []

    def grab(self) -> Any:
        self.grabs.append(threading.get_ident())
        if self.log is not None:
            self.log.append("camera.grab")
        if self.fail:
            raise RuntimeError("Frame didn't arrive within 5000")
        return SimpleNamespace(color=np.zeros((48, 64, 3), dtype=np.uint8))

    def get_intrinsics(self) -> None:
        return None

    def get_distortion(self) -> None:
        return None


class _Camera(_Handle):
    """An open camera as teaching meets it: a rig id, the grab that checks it gives frames, and the display path the
    window looks through, each noting the thread that asked."""

    def __init__(self, rig_id: str = "wrist", *, fail: bool = False) -> None:
        super().__init__(fail=fail)
        self.rig_id = rig_id
        self.looks: list[int] = []

    def peek(self) -> Any:
        self.looks.append(threading.get_ident())
        return SimpleNamespace(color=np.full((48, 64, 3), 90, dtype=np.uint8), captured_at_s=time.time())


class _Device(_Handle):
    """The device behind a real ``Camera`` owner: its opens and releases on the shared log."""

    def open(self) -> None:
        assert self.log is not None
        self.log.append("camera.open")

    def release(self) -> None:
        assert self.log is not None
        self.log.append("camera.release")


class _RecordingWindow:
    """Stands in for ``LiveView``: keeps the cameras it was given and what it was shown, hands back scripted keys, and
    logs its life."""

    built: list["_RecordingWindow"] = []
    keys: list[str | None] = []
    log: list[str] = []

    def __init__(self, cameras: Any = (), **keywords: Any) -> None:
        self.cameras = list(cameras)
        self.keywords = keywords
        self.shown: list[tuple[list[str], str]] = []
        self.notes: list[str] = []
        self.keys = list(type(self).keys)
        self.started = self.closed = False
        type(self).built.append(self)
        type(self).log.append("window.built")

    def start(self) -> None:
        self.started = True
        type(self).log.append("window.start")

    def close(self, timeout_s: float = 2.0) -> None:
        self.closed = True
        type(self).log.append("window.close")

    def guide(self, lines: Any, tone: str, bar: float | None = None) -> None:
        self.shown.append((list(lines), tone))

    def poll_key(self) -> str | None:
        return self.keys.pop(0) if self.keys else None

    def note(self, text: str, rig_id: str | None = None) -> None:
        self.notes.append(text)


class _WindowCase(_Case):
    """A run with the camera beside the console, on a desk the test says can or cannot show a window."""

    def setUp(self) -> None:
        super().setUp()
        _RecordingWindow.built, _RecordingWindow.keys = [], []

    def arm(self, *stands: list[FreedriveSample], plain: bool = False, **keywords: Any) -> _GuidedArm:
        """The arm, with the window's life written to its log beside everything else."""
        arm = super().arm(*stands, plain=plain, **keywords)
        _RecordingWindow.log = arm.log
        _RecordingWindow.built = []
        return arm

    def window_here(self, *, unavailable: str | None = None, terminal: bool = True,
                    window: Any = _RecordingWindow) -> ExitStack:
        """While it is entered: the view ``window`` builds, and why none can show (``None``: one can)."""
        stack = ExitStack()
        stack.enter_context(patch.object(live_view_module, "LiveView", window))
        stack.enter_context(patch.object(preview_module, "preview_unavailable", lambda: unavailable))
        stack.enter_context(patch("sys.stdout", _Terminal() if terminal else io.StringIO()))
        return stack


class TheCameraWindowBesideTheConsoleTests(_WindowCase):

    def test_the_window_shows_the_pose_and_the_count_and_opens_only_once_the_payload_is_confirmed(self) -> None:
        camera = _Camera()
        with self.window_here():
            self.arm(_stand(LOOK_1_DEG), _stand(LOOK_2_DEG))
            self.teach(_answers("", "", "", "", "", "q"), camera=camera)
        (window,) = _RecordingWindow.built
        self.assertEqual(window.cameras, [camera], "one window, for the one camera the poses are taught for")
        self.assertEqual(window.keywords["title"], "willy teach poses")
        self.assertEqual(window.keywords["on_close"], "teaching finishes, and the arm is held")
        self.assertEqual(window.notes, ["teaching poses by hand for rig 'wrist', eye in hand"])
        self.assertTrue(window.started and window.closed)
        self.assertIn((["MOVE THE ARM BY HAND to where LOOK_1 should be, then Enter", "none taught yet"], "guide"),
                      window.shown)
        self.assertIn((["MOVE THE ARM BY HAND to where LOOK_2 should be, then Enter", "1 taught: LOOK_1"], "guide"),
                      window.shown)
        self.assertIn((["Name of the next pose? Enter = LOOK_2, q finishes"], "guide"), window.shown)
        self.assertIn((["HOLDING THE ARM", "LOOK_1 is read once it holds"], "guide"), window.shown)
        self.assertEqual(self.printed, [LOOK_1_LINE, LOOK_2_LINE])
        log = self.guided.log
        self.assertNotIn(([PAYLOAD_QUESTION], "guide"), window.shown, "the payload is asked at the console alone")
        self.assertLess(log.index(f"say: {PAYLOAD_QUESTION}"), log.index("window.start"))
        self.assertLess(log.index("window.start"), log.index("free"))
        self.assertLess(log.index("left, held"), log.index("window.close"), "the arm holds before the window closes")
        self.assertIsNone(self.guide.view, "the guide's own view is given back")

    def test_enter_in_the_window_captures(self) -> None:
        with self.window_here():
            self.arm(_stand(LOOK_1_DEG))
            # The window's keys: the name prompt's drain, the guiding's drain, then Enter.
            _RecordingWindow.keys = [None, None, "enter"]
            taught = self.teach(_answers("", "") + [None, None] + _answers("q"), camera=_Camera())
        self.assertEqual([pose.name for pose in taught], ["LOOK_1"])
        self.assertEqual(self.printed, [LOOK_1_LINE])
        self.assertEqual(self.steps(), ["payload read", "session", "free", "hold", "left, held"])

    def test_closing_the_window_or_q_in_it_finishes_and_the_arm_is_held(self) -> None:
        guiding = ["payload read", "session", "free", "hold", "left, held"]
        ways = {
            "closed while the arm is free": ([None, None, "closed"], _answers("", "") + [None, None], guiding),
            "q in the window while the arm is free": ([None, None, "finish"], _answers("", "") + [None, None], guiding),
            "closed at the name prompt": ([None, "closed"], _answers("") + [None, None],
                                          ["payload read", "session", "left, held"]),
        }
        for way, (keys, answers, steps) in ways.items():
            with self.subTest(way), self.window_here():
                self.arm(_stand(LOOK_1_DEG))
                _RecordingWindow.keys = keys
                self.assertEqual(self.teach(answers, camera=_Camera()), ())
                self.assertEqual(self.steps(), steps)
                self.assertEqual(self.guided.open_sessions, 0)
                self.assertTrue(_RecordingWindow.built[0].closed)
                self.assertEqual(self.printed, [])

    def test_where_no_window_can_show_or_the_camera_gives_no_frame_one_line_says_why_and_teaching_goes_on(
            self) -> None:
        ways = {
            "no display": (dict(unavailable="no display: DISPLAY and WAYLAND_DISPLAY are unset"), False,
                           "no display: DISPLAY and WAYLAND_DISPLAY are unset"),
            "not a terminal": (dict(terminal=False), False, "stdout is not a terminal"),
            "no frame": ({}, True, "the camera gives no frame (RuntimeError: Frame didn't arrive within 5000)"),
        }
        for way, (desk, fails, why) in ways.items():
            with self.subTest(way):
                camera = _Camera(fail=fails)
                with self.window_here(**desk):
                    self.arm(_stand(LOOK_1_DEG))
                    taught = self.teach(_answers("", "", "", "q"), camera=camera)
                said = [line for line in self.console.said if line.startswith("No camera window")]
                self.assertEqual(said, [f"No camera window: {why}. The poses are taught at the console, still for rig "
                                        "'wrist'."])
                self.assertLess(self.console.said.index(said[0]), self.console.said.index(PAYLOAD_QUESTION))
                self.assertEqual(_RecordingWindow.built, [], "no window was built")
                self.assertEqual([(pose.name, pose.rig) for pose in taught], [("LOOK_1", "wrist")])
                self.assertEqual(len(camera.grabs), 1 if fails else 0, "the camera is read only where a window can show")

    def test_the_window_is_closed_before_the_camera_is_given_back_on_every_way_out(self) -> None:
        ways: dict[str, tuple[list[str | None], Any]] = {
            "q": (_answers("", "", "", "q"), None),
            "a payload not confirmed": (_answers("n"), HandGuidingRefused),
            "Ctrl-C while the arm is free": (_answers("", "") + [None], KeyboardInterrupt),
        }
        for way, (answers, raised) in ways.items():
            with self.subTest(way), self.window_here():
                self.clock = _Clock()
                log = self.arm(_stand(LOOK_1_DEG)).log

                def interrupt(_seconds: float, log: list[str] = log) -> None:
                    if "free" in log:
                        raise KeyboardInterrupt

                self.clock.on_sleep = interrupt if raised is KeyboardInterrupt else None
                camera = Camera.from_rig(_rgbd_rig("wrist"), streamer=_Device(log))
                with self.assertRaises(raised) if raised else nullcontext():
                    with camera:  # as example 11 opens it
                        self.teach(answers, camera=camera)
                life = [line for line in log if line.startswith(("camera.", "window."))]
                started = [] if raised is HandGuidingRefused else ["window.start"]
                self.assertEqual(life, ["camera.open", "camera.grab", "window.built", *started, "window.close",
                                        "camera.release"])

    def test_a_camera_that_is_no_camera_is_refused_before_anything_is_asked(self) -> None:
        for camera, why in ((object(), "camera= is one open Camera"), ([_Camera()], "camera= is one open Camera")):
            with self.subTest(type(camera).__name__), self.window_here():
                self.arm(_stand(LOOK_1_DEG))
                with self.assertRaisesRegex(TypeError, why):
                    self.teach(_answers("", "q"), camera=camera)
                self.assertEqual(self.guided.log, [])


class TheRealWindowTests(_WindowCase):
    """The cell's live view over a HighGUI double, one window: its thread draws and reads the camera, the caller's the
    arm."""

    _TITLE = "willy teach poses: wrist"

    def _real_window(self, gui: _Gui) -> ExitStack:
        stack = self.window_here(window=LiveView)
        stack.enter_context(patch.object(preview_module, "_highgui", lambda: gui))
        return stack

    def test_the_window_draws_and_reads_the_camera_on_its_own_thread_and_never_calls_the_arm(self) -> None:
        gui, camera = _Desk(), _Camera("wrist")
        drawn = lambda: len(gui.windows.get(self._TITLE, [])) >= 2  # noqa: E731
        with self._real_window(gui):
            self.arm(_stand(LOOK_1_DEG))
            # Real time passes while the arm is free: the first sleep waits until the window has drawn twice.
            self.clock.on_sleep = lambda _s: None if drawn() else _until(drawn)
            taught = self.teach(_answers("", "", "", "q"), camera=camera)
        main = threading.get_ident()
        self.assertEqual([pose.name for pose in taught], ["LOOK_1"])
        self.assertEqual(self.guided.threads, {main}, "the arm was called from another thread")
        self.assertEqual(len(gui.threads()), 1)
        self.assertNotIn(main, gui.threads())
        self.assertEqual(gui.opened, [self._TITLE], "one window, the camera's")
        self.assertEqual(camera.grabs, [main], "the one grab that checks the camera, before the view runs")
        self.assertGreaterEqual(len(camera.looks), 1)
        self.assertEqual(set(camera.looks), gui.threads(), "every look on the window's thread")
        self.assertEqual(gui.destroyed, [self._TITLE], "the window was closed")

    def test_the_guide_and_its_red_boundary_show_in_the_window(self) -> None:
        gui = _Desk()

        def red() -> bool:
            return any(tuple(int(v) for v in image[2, 10]) == (0, 0, 230) for image in gui.windows.get(self._TITLE, []))

        def wait_for_red(_seconds: float) -> None:
            view = self.guide.view
            if view is not None and view._guiding[1] == "outside" and not red():
                _until(red)

        wound = (*LOOK_1_DEG[:5], 200.0)
        with self._real_window(gui):
            self.arm([_sample(wound, speed=0.3)] * 5 + [_sample(LOOK_1_DEG)])
            self.clock.on_sleep = wait_for_red
            taught = self.teach(_answers("", "", "") + [None] * 6 + ["", None, "q"], camera=_Camera("wrist"))
        self.assertEqual(len(taught), 1)
        self.assertTrue(red(), "the guide's red boundary is not in the window")

    def test_esc_in_the_window_finishes_in_teachings_words_and_the_arm_is_held(self) -> None:
        with self._real_window(_Desk(keys=(27,))):
            self.arm(_stand(LOOK_1_DEG))
            self.clock.on_sleep = lambda _s: time.sleep(0.002)  # the window's thread runs while the name is asked
            taught = self.teach(_answers("") + [None] * 5, camera=_Camera())
        self.assertEqual(taught, ())
        self.assertEqual(self.steps(), ["payload read", "session", "left, held"])
        self.assertIn("camera windows closed (ESC): teaching finishes, and the arm is held. The pendant stops the robot",
                      self.console.said)
        self.assertFalse([line for line in self.console.said if "sweep" in line], "a sweep's words while teaching")

    def test_a_window_that_cannot_open_says_so_once_and_teaching_goes_on_at_the_console(self) -> None:
        with self._real_window(_Desk(fail="namedWindow")):
            self.arm(_stand(LOOK_1_DEG))
            self.clock.on_sleep = lambda _s: time.sleep(0.001)
            taught = self.teach(_answers("", "", "", "q"), camera=_Camera())
        self.assertEqual([(pose.name, pose.rig) for pose in taught], [("LOOK_1", "wrist")])
        self.assertEqual([line for line in self.console.said if line.startswith("camera windows")],
                         ["camera windows off after RuntimeError: the display went away; teaching goes on at the "
                          "console"])

    def test_the_library_builds_no_window_unasked(self) -> None:
        explode = MagicMock(side_effect=AssertionError("a window was built"))
        with self.window_here(window=explode):
            self.arm(_stand(LOOK_1_DEG))
            self.assertEqual([pose.name for pose in self.teach(_answers("", "", "", "q"))], ["LOOK_1"])
        self.assertFalse([line for line in self.console.said if "camera window" in line.lower()])


# ---------------------------------------------------------------------------------------------------------------------
# Example 11: the one camera the poses are taught for
# ---------------------------------------------------------------------------------------------------------------------

_EXAMPLE_11 = Path(__file__).resolve().parents[1] / "examples" / "real_robot" / "11_teach_poses_by_hand.py"


class _Opened:
    """A camera example 11 opened: its rig, and its opening and giving back on the run's log."""

    def __init__(self, rig_id: str, log: list[str]) -> None:
        self.rig_id = rig_id
        self.log = log

    def __enter__(self) -> "_Opened":
        self.log.append(f"camera {self.rig_id} opened")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.log.append(f"camera {self.rig_id} given back")


class Example11TeachesForOneCameraTests(unittest.TestCase):
    """Example 11 run on doubles of its willy names: which camera it opens, and what it hands ``teach_poses``."""

    def _run(self, *, camera: str | None = None, show: bool = True, dark: bool = False) -> dict[str, Any]:
        source = _EXAMPLE_11.read_text(encoding="utf-8")
        switches = ('CAMERA = None  # the rig whose poses you teach, e.g. "wrist"; None: the primary',
                    "SHOW_CAMERA = True  # False: no window")
        for switch in switches:
            self.assertEqual(source.count(switch), 1, f"example 11 lost {switch!r}")
        source = source.replace(switches[0], f"CAMERA = {camera!r}", 1).replace(switches[1], f"SHOW_CAMERA = {show}", 1)
        log: list[str] = []
        taught: list[dict[str, Any]] = []
        tree = SimpleNamespace(robot="robot section", app_config=TREE.app_config)

        def from_tree(given: Any, *, rig_id: str) -> _Opened:
            self.assertIs(given, tree)
            if dark:
                raise RuntimeError(f"RealSense rig {rig_id!r} could not start: no device connected")
            return _Opened(rig_id, log)

        def teach(robot: Any, **keywords: Any) -> tuple[TaughtPose, ...]:
            taught.append(keywords)
            log.append("taught")
            return ()

        robot = SimpleNamespace(connected=lambda: nullcontext())
        doubles = {"load_tree": lambda: tree, "Camera": SimpleNamespace(from_tree=from_tree), "teach_poses": teach,
                   "Robot": SimpleNamespace(from_config=lambda _section, gripper: robot)}
        out = io.StringIO()
        import willy

        with patch.dict(vars(willy), doubles), patch("sys.stdout", out):
            exec(compile(source, str(_EXAMPLE_11), "exec"), {"__name__": "__main__"})  # noqa: S102 (the example itself)
        (keywords,) = taught
        return {"log": log, "keywords": keywords, "printed": out.getvalue(), "tree": tree}

    def test_the_switches_are_the_first_two_lines_after_the_imports(self) -> None:
        import ast

        source = _EXAMPLE_11.read_text(encoding="utf-8")
        tree = ast.parse(source)
        first = [node for node in tree.body if not isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))][:2]
        self.assertEqual([ast.get_source_segment(source, node) for node in first],
                         ["CAMERA = None", "SHOW_CAMERA = True"])
        self.assertIn("SHOW_CAMERA", ast.get_docstring(tree) or "")

    def test_only_the_primary_is_opened_and_shown_when_no_camera_is_named(self) -> None:
        run = self._run()
        self.assertEqual(run["log"], ["camera wrist opened", "taught", "camera wrist given back"])
        keywords = run["keywords"]
        self.assertEqual((keywords["for_rig"], keywords["camera"].rig_id), ("wrist", "wrist"))
        self.assertIs(keywords["tree"], run["tree"])

    def test_only_the_camera_named_is_opened_and_shown(self) -> None:
        run = self._run(camera="overhead")
        self.assertEqual(run["log"], ["camera overhead opened", "taught", "camera overhead given back"])
        self.assertEqual((run["keywords"]["for_rig"], run["keywords"]["camera"].rig_id), ("overhead", "overhead"))

    def test_with_no_window_nothing_is_opened_and_the_poses_still_belong_to_the_rig(self) -> None:
        for camera, rig in ((None, "wrist"), ("overhead", "overhead")):
            with self.subTest(rig):
                run = self._run(camera=camera, show=False)
                self.assertEqual(run["log"], ["taught"])
                self.assertEqual((run["keywords"]["for_rig"], run["keywords"]["camera"]), (rig, None))

    def test_a_camera_that_cannot_open_is_said_and_teaching_goes_on_for_its_rig(self) -> None:
        run = self._run(camera="overhead", dark=True)
        self.assertEqual(run["log"], ["taught"])
        self.assertEqual((run["keywords"]["for_rig"], run["keywords"]["camera"]), ("overhead", None))
        self.assertIn("No camera window, the poses are still taught for rig 'overhead': RealSense rig 'overhead' could "
                      "not start: no device connected", run["printed"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
