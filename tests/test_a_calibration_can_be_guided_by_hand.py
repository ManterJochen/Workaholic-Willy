"""A calibration a person guides by hand, and fixed stations a person fine-tunes: the arm only moves when it is moved.

The owner, 2026-09-24: every automatic station generator is gone, and what is left is (1) fixed stations, poses or
joints from a file or a list, with an optional ``adjust`` that frees the arm at each reached station for a person to
fine-tune it, Enter capturing; and (2) freedrive: the person moves the arm to a free pose, the software checks the
board is seen well, and stores the pose and the image, and each pose it counts goes to a stations file that replays
the run without hands. What is held here, over a scripted hand-guided arm, a scripted console and a clock of the
test's own:

* Enter captures, from the console or from the preview window, only once the arm has stood still;
* a pose outside the cable window or the workspace box is shown in red and Enter does not capture there, and the
  arm is never held because of it (an arm that locks while a person pushes it is how a hand gets caught);
* the controller payload is shown and asked about once, before anything is freed;
* before the arm drives by itself again the console asks for the hands off it and counts down 3, 2, 1;
* leaving the session always holds the arm, on a finish, a crash and Ctrl-C alike;
* every pose counted by hand lands in a stations file that ``run_from_json`` replays without hands;
* an arm that offers no hand guiding is refused before anything moves.

Nothing here has run beside a physical arm; the UR teach mode behind ``SupportsFreedrive`` is the freedrive
workstream's, and its own tests hold it.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.calibration import MountingMode
from src.calibration import preview as preview_module
from src.calibration.exceptions import CalibrationDataError
from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions
from src.robot.core.freedrive import ControllerPayload, FreedriveSample, SupportsFreedrive
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.events import RobotCalibrationEvent
from src.robot.execution.calibration import CalibrationRoutine
from src.robot.execution.hand_guiding import (
    HandGuide,
    HandGuidingLimits,
    HandGuidingRefused,
    offset_in_tool,
    sample_pose,
)
from src.robot.execution.pose_provider import JointStation, load_stations
from tests.test_calibration_sweep_preview import _Gui, _preview, _until
from tests.test_robot_boundaries import _inverse, _synthetic_eye_to_hand_data

_WIDE = WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=-2000.0, z_max=2000.0)
T_CAM_TO_BASE, T_TOOL_TO_MARKER, TOOL = _synthetic_eye_to_hand_data()
#: The joints each synthetic tool pose stands at: made up, inside the cable window, one set per pose.
JOINTS_DEG = [[10.0 * i, -90.0 + 5.0 * i, 90.0, -90.0, -90.0, 5.0 * i] for i in range(len(TOOL))]
#: The cable window of the tests' arm: half a turn about a home at zero, less a 5 degree margin.
_WINDOW = ([-180.0] * 6, [180.0] * 6)


def _sample(T: np.ndarray, joints_deg: list[float], speed: float = 0.0) -> FreedriveSample:
    pose = Pose.from_matrix(T, frame=Frame.BASE)
    return FreedriveSample(joints_rad=tuple(math.radians(v) for v in joints_deg),
                           joint_speeds_rad_s=(speed, 0.0, 0.0, 0.0, 0.0, 0.0),
                           tcp_xyz_mm=tuple(float(v) for v in pose.position_mm),  # type: ignore[arg-type]
                           tcp_rotvec_rad=tuple(float(v) for v in pose.axis_angle_rad()),  # type: ignore[arg-type]
                           t_s=0.0)


def _stand(index: int, *, moving: int = 3) -> list[FreedriveSample]:
    """A person moving the arm to tool pose ``index``: a few samples on the way, then the arm standing still."""
    return [_sample(TOOL[index], JOINTS_DEG[index], speed=0.3)] * moving + [_sample(TOOL[index], JOINTS_DEG[index])]


class _Clock:
    """The time the guide runs on. A guide that waits for more than ``budget_s`` fails the test instead of hanging."""

    def __init__(self, budget_s: float = 600.0) -> None:
        self.now = 0.0
        self.budget_s = budget_s
        self.sleeps: list[float] = []
        self.on_sleep: Any = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        self.now += seconds
        if self.now > self.budget_s:
            raise AssertionError("the guide waited for longer than the test allows: a prompt nobody answers")


class _Console:
    """A scripted operator: ``poll`` returns the next item (``None`` for nothing typed), then nothing.

    While the last line said is a countdown's (the arm moves by itself in n s), ``poll`` answers from
    ``counting_down`` instead, then nothing: what the person types while the countdown runs, kept apart from the
    answers to the prompts, so a countdown that reads the console many times a second takes none of them.
    """

    def __init__(self, lines: list[str | None], log: list[str], *,
                 counting_down: list[str | None] | None = None) -> None:
        self.lines = list(lines)
        self.counting_down = list(counting_down or [])
        self.log = log
        self.said: list[str] = []
        #: When each line was said, on ``clock`` when a test gives one.
        self.said_at: list[float] = []
        self.clock: _Clock | None = None
        self.bells = 0
        self.polls_counting_down = 0

    def say(self, line: str) -> None:
        self.said.append(line)
        self.said_at.append(self.clock() if self.clock is not None else 0.0)
        self.log.append(f"say: {line}")

    def bell(self) -> None:
        self.bells += 1

    def poll(self) -> str | None:
        if self.said and "moves by itself in" in self.said[-1]:
            self.polls_counting_down += 1
            return self.counting_down.pop(0) if self.counting_down else None
        return self.lines.pop(0) if self.lines else None


class _View:
    """A preview double: keeps what the guide showed, and hands back scripted keys."""

    def __init__(self, keys: list[str | None] | None = None) -> None:
        self.keys = list(keys or [])
        self.shown: list[tuple[list[str], str, float | None]] = []

    def guide(self, lines: Any, tone: str, bar: float | None = None) -> None:
        self.shown.append((list(lines), tone, bar))

    def poll_key(self) -> str | None:
        return self.keys.pop(0) if self.keys else None


class _CountdownView(_View):
    """A window in which ``key`` is typed once, while a countdown shows in it, after ``after`` quiet looks."""

    def __init__(self, key: str, *, after: int = 0) -> None:
        super().__init__()
        self.key = key
        self.after = after
        self.sent = 0

    def poll_key(self) -> str | None:
        counting = bool(self.shown and self.shown[-1][0] and "MOVES BY ITSELF IN" in self.shown[-1][0][0])
        if not counting or self.sent:
            return None
        if self.after > 0:
            self.after -= 1
            return None
        self.sent += 1
        return self.key


class _Session:
    """A hand-guiding session over :class:`_GuidedArm`: each ``free`` after the first starts the next scripted stand."""

    def __init__(self, arm: "_GuidedArm") -> None:
        self.arm = arm
        self._free = False

    @property
    def is_free(self) -> bool:
        return self._free

    def free(self) -> None:
        self.arm.log.append("free")
        self._free = True
        self.arm.next_stand()

    def hold(self) -> None:
        self.arm.log.append(f"hold at {self.arm.clock():.2f}")
        self._free = False

    def sample(self) -> FreedriveSample:
        self.arm.samples += 1
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
    """An arm a person can guide (``SupportsFreedrive``), scripted stand by stand. It reports the TCP and the joints
    of the last sample it gave, or of the pose it last moved to, and knows the joints of every synthetic pose."""

    def __init__(self, stands: list[list[FreedriveSample]], clock: _Clock, *,
                 payload: ControllerPayload | None = ControllerPayload(1.9, (0.0, 12.0, 61.0))) -> None:
        self.stands = [list(stand) for stand in stands]
        self.clock = clock
        self.payload = payload
        self.log: list[str] = []
        #: When each move was commanded, on ``clock``.
        self.moved_at: list[float] = []
        self.stand = -1
        self.samples = 0
        self.open_sessions = 0
        self.sessions = 0
        self.last: FreedriveSample | None = None
        self.at = TOOL[0]
        self.safety_preflight = SimpleNamespace(guards=(SimpleNamespace(
            name="joint_limit", margin_deg=5.0, within_half_turn_of_home=True,
            limits_for_arm=lambda _arm: _WINDOW),))

    # --- SupportsFreedrive ---
    def freedrive(self) -> _Session:
        self.sessions += 1
        return _Session(self)

    def controller_payload(self) -> ControllerPayload | None:
        self.log.append("payload read")
        return self.payload

    # --- the script ---
    def next_stand(self) -> None:
        self.stand += 1

    def next_sample(self) -> FreedriveSample:
        stand = self.stands[min(self.stand, len(self.stands) - 1)]
        sample = stand.pop(0) if len(stand) > 1 else stand[0]
        self.last = sample
        self.at = sample_pose(sample).to_matrix()
        return sample

    # --- what the routine reads and commands ---
    def move(self, pose: Pose, **_keywords: Any) -> MotionResult:
        assert self.open_sessions == 0, "a move was commanded inside a hand-guiding session"
        self.log.append(f"move {pose.label}")
        self.moved_at.append(self.clock())
        self.at = pose.to_matrix()
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def move_to_joints(self, joints: JointPositions, **_keywords: Any) -> MotionResult:
        assert self.open_sessions == 0, "a move was commanded inside a hand-guiding session"
        index = self._index(joints)
        self.log.append(f"move_to_joints {index}")
        self.moved_at.append(self.clock())
        self.at = TOOL[index]
        return MotionResult.executed(MotionCommand.MOVE_TO, target_joints=joints)

    def fk(self, joints: JointPositions) -> Pose:
        return Pose.from_matrix(TOOL[self._index(joints)], frame=Frame.BASE)

    def get_tcp_pose(self) -> Pose:
        return Pose.from_matrix(self.at, frame=Frame.BASE)

    def get_joint_positions(self) -> JointPositions:
        assert self.last is not None
        return JointPositions(list(self.last.joints_rad))

    @staticmethod
    def _index(joints: JointPositions) -> int:
        degrees = np.degrees(np.asarray(joints.values, dtype=np.float64))
        return int(np.argmin([np.abs(degrees - np.asarray(row)).max() for row in JOINTS_DEG]))


class _PlainArm(_GuidedArm):
    """The same arm with no hand guiding: an arm of fixed stations only."""

    freedrive = None  # type: ignore[assignment]
    controller_payload = None  # type: ignore[assignment]


class _Source:
    """The marker the fixed camera sees on the flange at wherever the arm stands, and the frame it judged."""

    def __init__(self, arm: _GuidedArm) -> None:
        self.arm = arm
        self.last_observation = None
        self.last_frame: np.ndarray | None = None

    def __call__(self) -> np.ndarray:
        self.last_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return _inverse(T_CAM_TO_BASE) @ self.arm.at @ T_TOOL_TO_MARKER


def _routine(arm: _GuidedArm, console: _Console, clock: _Clock, *, view: Any = None,
             images: str | None = None) -> tuple[CalibrationRoutine, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    console.clock = clock
    guide = HandGuide(console, view, clock=clock, sleep=clock.sleep)
    routine = CalibrationRoutine(
        arm=arm, marker_source=_Source(arm), workspace_limits=_WIDE, calibration_mode="eye_to_hand",  # type: ignore[arg-type]
        settle_time_s=0.0, on_event=lambda kind, data: events.append((kind, dict(data))),
        eth_settings=SimpleNamespace(mode="eye_to_hand", min_samples=4, min_distance_mm=40.0, min_angle=10.0,
                                     min_angle_deg=10.0),  # type: ignore[arg-type]
        hand_guide=guide, images_dir=images)
    return routine, events


def _enter(times: int) -> list[str | None]:
    """The console for ``times`` prompts answered with Enter: a quiet poll that ends the drain, then the Enter."""
    return [None, ""] * times


# ---------------------------------------------------------------------------------------------------------------------
# Freedrive: the person moves the arm, Enter captures
# ---------------------------------------------------------------------------------------------------------------------


class APersonGuidesTheArmToEveryPoseTests(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.clock = _Clock()
        self.arm = _GuidedArm([_stand(i) for i in range(4)], self.clock)
        self.console = _Console(_enter(5), self.arm.log)  # the payload, then four poses
        routine, self.events = _routine(self.arm, self.console, self.clock, images=str(self.folder / "images"))
        self.stations = self.folder / "stations.json"
        self.result = routine.run_freedrive(samples=4, dataset_save_path=str(self.folder / "dataset.json"),
                                            stations_save_path=self.stations)

    def test_every_enter_counts_a_pose_and_the_solve_finds_the_camera(self) -> None:
        self.assertEqual(self.result.num_samples, 4)
        assert self.result.T_cam_to_base is not None
        np.testing.assert_allclose(self.result.T_cam_to_base, T_CAM_TO_BASE, atol=1e-6)
        self.assertEqual([verdict.counted for verdict in self.result.pose_log], [True] * 4)
        self.assertEqual([verdict.label for verdict in self.result.pose_log], [f"hand_0{i}" for i in range(1, 5)])

    def test_nothing_moves_by_itself_and_every_capture_holds_first_then_frees_again(self) -> None:
        motions = [line for line in self.arm.log if line.startswith("move")]
        self.assertEqual(motions, [])
        steps = [line.split(" at ")[0] for line in self.arm.log if not line.startswith("say")]
        self.assertEqual(steps, ["payload read", "session", "free", *(["hold", "free"] * 4), "left, held"])
        self.assertEqual(self.arm.open_sessions, 0)

    def test_the_payload_is_shown_and_confirmed_once_before_the_arm_is_freed(self) -> None:
        said = [line for line in self.arm.log if line.startswith("say")]
        asked = [line for line in said if "Is this payload right (hand + camera + bracket)? Enter = yes" in line]
        self.assertEqual(len(asked), 1)
        self.assertIn("1.90 kg", " ".join(said))
        self.assertLess(self.arm.log.index(asked[0]), self.arm.log.index("free"))

    def test_the_person_is_called_by_the_console_and_the_bell(self) -> None:
        calls = [line for line in self.console.said if line.startswith("MOVE THE ARM BY HAND")]
        self.assertEqual(len(calls), 4)
        self.assertIn("to a new pose where the camera sees the board, then Enter", calls[0])
        self.assertGreaterEqual(self.console.bells, 5)

    def test_every_counted_pose_is_written_to_a_stations_file_run_from_json_replays(self) -> None:
        records = json.loads(self.stations.read_text(encoding="utf-8"))
        self.assertEqual([record["label"] for record in records], ["hand_01", "hand_02", "hand_03", "hand_04"])
        self.assertEqual(records[1]["joints_deg"], JOINTS_DEG[1])
        self.assertEqual(sorted(records[1]["tcp_pose"]), ["rx", "ry", "rz", "x", "y", "z"])
        self.assertEqual(self.result.stations_path, str(self.stations))
        stations = load_stations(self.stations)
        self.assertTrue(all(isinstance(station, JointStation) for station in stations))
        # The replay: a fresh routine drives the arm to exactly those joints, nobody's hands on it.
        clock = _Clock()
        arm = _GuidedArm([[_sample(TOOL[0], JOINTS_DEG[0])]], clock)
        replay, _ = _routine(arm, _Console([], arm.log), clock)
        result = replay.run_from_json(self.stations)
        self.assertEqual([line for line in arm.log if line.startswith("move")],
                         [f"move_to_joints {i}" for i in range(4)])
        self.assertEqual(arm.sessions, 0)
        self.assertEqual(result.num_samples, 4)

    def test_the_dataset_and_each_counted_frame_are_kept(self) -> None:
        self.assertTrue((self.folder / "dataset.json").is_file())
        self.assertEqual(sorted(path.name for path in (self.folder / "images").iterdir()),
                         [f"0{i}_hand_0{i}.png" for i in range(1, 5)])

    def test_the_stillness_gate_held_the_arm_only_after_it_stood_still_for_half_a_second(self) -> None:
        holds = [float(line.split(" at ")[1]) for line in self.arm.log if line.startswith("hold")]
        # Each stand: three samples moving, then still; Enter comes on the first sample, so each hold is at least
        # 0.5 s after the arm stopped.
        gaps = [later - earlier for earlier, later in zip([0.0, *holds], holds)]
        self.assertTrue(all(gap >= 0.5 for gap in gaps), gaps)


class EnterFromThePreviewTests(unittest.TestCase):

    def test_enter_in_the_window_captures_as_the_console_does_and_q_there_finishes(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(4)], clock)
        console = _Console([None, ""], arm.log)  # the payload only; everything else comes from the window
        view = _View([None] + [None, "enter"] * 4 + [None, "finish"])  # the payload prompt drains the first
        routine, _ = _routine(arm, console, clock, view=view)
        result = routine.run_freedrive(samples=10)
        self.assertEqual(result.num_samples, 4)
        self.assertEqual(arm.log[-1], "left, held")
        banners = [lines[0] for lines, tone, _ in view.shown if tone == "guide" and lines]
        self.assertTrue(any(line.startswith("MOVE THE ARM BY HAND") for line in banners))
        self.assertTrue(any("HOLD THE ARM STILL" in line for line in banners))

    def test_a_closed_window_finishes_at_the_next_prompt(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(3)], clock)
        view = _View(["closed"])
        routine, _ = _routine(arm, _Console([None, ""], arm.log), clock, view=view)
        with self.assertRaisesRegex(HandGuidingRefused, "payload was not confirmed"):  # closed before it was
            routine.run_freedrive(samples=10)
        self.assertEqual(arm.sessions, 0, "the arm was freed after the person closed the window")


class TheStillnessGateTests(unittest.TestCase):

    def test_an_arm_still_moving_is_not_captured_and_the_person_is_told(self) -> None:
        clock = _Clock()
        restless = [_sample(TOOL[0], JOINTS_DEG[0], speed=0.2)]
        arm = _GuidedArm([restless], clock)
        console = _Console([None, "", None, ""] + [None] * 5 + ["q"], arm.log)
        routine, _ = _routine(arm, console, clock)
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):  # nothing counted
            routine.run_freedrive(samples=3)
        self.assertFalse(any(line.startswith("hold") for line in arm.log), arm.log)
        self.assertIn("the arm still moves", " ".join(console.said))
        self.assertEqual(arm.log[-1], "left, held")


# ---------------------------------------------------------------------------------------------------------------------
# Boundaries: red, no capture, and never a hold
# ---------------------------------------------------------------------------------------------------------------------


class ABoundaryIsShownNeverEnforcedTests(unittest.TestCase):

    def test_outside_the_cable_window_turns_red_refuses_enter_and_never_holds_the_arm(self) -> None:
        clock = _Clock()
        wound = [*JOINTS_DEG[0][:5], 200.0]
        outside = [_sample(TOOL[0], wound, speed=0.3)] * 5
        arm = _GuidedArm([outside + [_sample(TOOL[0], JOINTS_DEG[0])], *(_stand(i) for i in (1, 2, 3))], clock)
        # The payload; Enter on the first sample, out past the window; quiet while the person turns wrist 3 back;
        # Enter inside; then three more poses.
        console = _Console([None, "", None, ""] + [None] * 6 + ["", *_enter(3)], arm.log)
        view = _View()
        routine, _ = _routine(arm, console, clock, view=view)
        result = routine.run_freedrive(samples=4)
        self.assertEqual(result.num_samples, 4)
        said = " ".join(console.said)
        self.assertIn("OUTSIDE: wrist 3 stands at 200.0 deg, outside the cable window -175.0 to 175.0 deg: turn it "
                      "back down inside it. Enter does not capture here.", said)
        self.assertIn("Not captured: wrist 3 stands at 200.0 deg", said)
        self.assertIn("Back inside.", said)
        self.assertEqual(sum("OUTSIDE:" in line for line in console.said), 1, "said once per crossing, not per sample")
        red = [lines for lines, tone, _ in view.shown if tone == "outside"]
        self.assertTrue(red and red[0][0].startswith("OUTSIDE - NOT CAPTURED: wrist 3 stands at 200.0 deg"))
        # The only holds are the four captures, each inside: a boundary never held the arm.
        holds = [line for line in arm.log if line.startswith("hold")]
        self.assertEqual(len(holds), 4)
        first_hold = next(i for i, line in enumerate(arm.log) if line.startswith("hold"))
        back_inside = next(i for i, line in enumerate(arm.log) if "Back inside." in line)
        self.assertLess(back_inside, first_hold)

    def test_a_box_face_is_named_with_the_way_back(self) -> None:
        box = WorkspaceLimitsConfig(x_min=-100.0, x_max=500.0, y_min=-500.0, y_max=500.0, z_min=150.0, z_max=900.0)
        limits = HandGuidingLimits(box=box)
        low = _sample(TOOL[0], JOINTS_DEG[0])  # z 100
        self.assertEqual(limits.crossed(low)[0], "box z_min")
        self.assertEqual(limits.outside(low), "the TCP stands 50 mm past the workspace box's z_min face (z 100.0 mm, "
                                              "the box starts at 150.0 mm): bring it back inside")
        self.assertEqual(limits.outside(_sample(TOOL[3], JOINTS_DEG[3])), "")

    def test_the_window_is_the_arms_joint_limit_guard_less_its_margin(self) -> None:
        clock = _Clock()
        limits = HandGuidingLimits.of(_GuidedArm([], clock), _WIDE)
        self.assertEqual((limits.lower_deg, limits.upper_deg), ((-175.0,) * 6, (175.0,) * 6))
        self.assertIn("half a turn either side of home", limits.window)
        bare = HandGuidingLimits.of(SimpleNamespace(), _WIDE)
        self.assertEqual(bare.lower_deg, ())
        self.assertIn("only the workspace box is watched", bare.window)


# ---------------------------------------------------------------------------------------------------------------------
# The payload, and leaving the session
# ---------------------------------------------------------------------------------------------------------------------


class ThePayloadAndTheWayOutTests(unittest.TestCase):

    def test_a_payload_nobody_confirms_frees_nothing(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(0)], clock)
        routine, _ = _routine(arm, _Console([None, "n"], arm.log), clock)
        with self.assertRaisesRegex(HandGuidingRefused, "the controller payload was not confirmed"):
            routine.run_freedrive()
        self.assertEqual(arm.sessions, 0)

    def test_a_payload_the_arm_cannot_read_is_said_and_still_asked(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(0)], clock, payload=None)
        console = _Console([None, "n"], arm.log)
        routine, _ = _routine(arm, console, clock)
        with self.assertRaises(HandGuidingRefused):
            routine.run_freedrive()
        self.assertIn("cannot be read from this arm", " ".join(console.said))

    def test_a_terminal_whose_input_has_ended_refuses_rather_than_taking_silence_for_a_yes(self) -> None:
        import io

        from src.robot.execution.hand_guiding import TerminalConsole

        printed = io.StringIO()
        guide = HandGuide(TerminalConsole(stdin=io.StringIO(""), stdout=printed))
        # Whether the end of input is read before the question or as its answer, nobody confirmed anything.
        with self.assertRaisesRegex(HandGuidingRefused, "input has ended|payload was not confirmed"):
            guide.confirm_payload(_GuidedArm([], _Clock()))
        self.assertFalse(guide.payload_confirmed)

    def test_a_crash_inside_the_session_leaves_it_and_holds_the_arm(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(0)], clock)
        routine, _ = _routine(arm, _Console(_enter(2), arm.log), clock)

        def interrupted() -> None:
            raise KeyboardInterrupt

        routine._marker_source = interrupted  # type: ignore[assignment]  # noqa: SLF001
        with self.assertRaises(KeyboardInterrupt):
            routine.run_freedrive(samples=3)
        self.assertEqual((arm.log[-1], arm.open_sessions), ("left, held", 0))


# ---------------------------------------------------------------------------------------------------------------------
# Adjust: fixed stations fine-tuned by hand
# ---------------------------------------------------------------------------------------------------------------------


class AdjustTests(unittest.TestCase):

    def _run(self, *, on_sleep: Any = None) -> tuple[_GuidedArm, _Console, _Clock, Any]:
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(4)], clock)
        # The payload, then per station: Enter to capture; and before stations 2 to 4, Enter for hands off.
        console = _Console(_enter(1) + _enter(1) + _enter(2) * 3, arm.log)
        clock.on_sleep = (lambda seconds: on_sleep(seconds, console)) if on_sleep is not None else None
        self.arm, self.console = arm, console
        routine, events = _routine(arm, console, clock)
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        poses = [Pose.from_matrix(TOOL[i], frame=Frame.BASE, label=f"s{i}") for i in range(4)]
        result = routine.run_with_poses(poses, adjust=True, stations_save_path=self.folder / "adjusted.json")
        return arm, console, clock, (result, events)

    def test_each_reached_station_is_freed_captured_on_enter_and_left_holding(self) -> None:
        arm, _, _, (result, _) = self._run()
        steps = [line.split(" at ")[0] for line in arm.log if not line.startswith("say") and line != "payload read"]
        per_station = ["session", "free", "hold", "left, held"]
        self.assertEqual(steps, [step for i in range(4) for step in (f"move s{i}", *per_station)])
        self.assertEqual(result.num_samples, 4)
        self.assertEqual(len(json.loads((self.folder / "adjusted.json").read_text(encoding="utf-8"))), 4)

    def test_the_hands_off_prompt_and_a_visible_countdown_come_before_every_automatic_move_after_a_touch(self) -> None:
        arm, console, clock, _ = self._run()
        log = arm.log
        first_move = log.index("move s0")
        self.assertFalse(any("Hands off" in line for line in log[:first_move]), "nobody had touched the arm yet")
        for label in ("s1", "s2", "s3"):
            before = log[:log.index(f"move {label}")]
            last_leave = max(i for i, line in enumerate(before) if line == "left, held")
            tail = before[last_leave:]
            self.assertIn("say: Hands off the arm - Enter to continue (q finishes here)", tail)
            self.assertEqual([line for line in tail if "moves by itself in" in line],
                             [f"say: Hands off: the arm moves by itself in {n} s (q finishes, Ctrl-C stops)"
                              for n in (3, 2, 1)])
        # Each countdown lasts its three seconds, one line a second, and the move comes only after the last.
        for label, moved_at in zip(("s1", "s2", "s3"), arm.moved_at[1:]):
            said = [at for at, line in zip(console.said_at, console.said) if "moves by itself in" in line
                    and at <= moved_at][-3:]
            self.assertEqual([round(b - a, 6) for a, b in zip(said, said[1:])], [1.0, 1.0], label)
            self.assertGreaterEqual(round(moved_at - said[0], 6), 3.0, label)
        # And the console was read all through them, not only between them.
        self.assertGreaterEqual(console.polls_counting_down, 3 * 3 * 10)

    def test_ctrl_c_during_the_countdown_moves_nothing_more(self) -> None:
        def interrupt(seconds: float, console: _Console) -> None:
            if console.said and "moves by itself in" in console.said[-1]:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self._run(on_sleep=interrupt)
        self.assertEqual([line for line in self.arm.log if line.startswith("move")], ["move s0"])

    def _two_stations(self, console: _Console, *, view: Any = None) -> tuple[_GuidedArm, Any]:
        """Two stations adjusted by hand over ``console``: what moved, and what the solve of too few samples raised."""
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(2)], clock)
        console.log = arm.log
        routine, _ = _routine(arm, console, clock, view=view)
        poses = [Pose.from_matrix(TOOL[i], frame=Frame.BASE, label=f"s{i}") for i in range(2)]
        with self.assertRaisesRegex(CalibrationDataError, "too few samples") as raised:
            routine.run_with_poses(poses, adjust=True)
        return arm, raised.exception

    def test_an_answer_that_is_no_yes_is_asked_again_and_never_starts_the_countdown(self) -> None:
        # The payload, the capture at s0; then at the hands-off prompt four answers that are none, and finish.
        console = _Console(_enter(2) + [None, "stop", None, "n", None, "s", None, "wait", None, "finish"], [])
        arm, _ = self._two_stations(console)
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0"])
        self.assertFalse(any("moves by itself in" in line for line in console.said), console.said)
        for answer in ("stop", "n", "s", "wait"):
            self.assertIn(f"{answer!r} is not an answer here: Enter continues, q finishes.", console.said)
        self.assertEqual(console.said.count("Hands off the arm - Enter to continue (q finishes here)"), 5)

    def test_q_typed_in_the_console_during_the_countdown_moves_nothing_more(self) -> None:
        console = _Console(_enter(3), [], counting_down=[None] * 7 + ["q"])
        arm, _ = self._two_stations(console)
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0"])
        self.assertEqual([line for line in console.said if "moves by itself in" in line],
                         ["Hands off: the arm moves by itself in 3 s (q finishes, Ctrl-C stops)"])

    def test_q_or_a_closed_window_during_the_countdown_moves_nothing_more(self) -> None:
        for key in ("finish", "closed"):
            with self.subTest(key=key):
                view = _CountdownView(key, after=5)
                console = _Console(_enter(3), [])
                arm, _ = self._two_stations(console, view=view)
                self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0"])
                self.assertEqual(view.sent, 1)
                self.assertEqual([line for line in console.said if "moves by itself in" in line],
                                 ["Hands off: the arm moves by itself in 3 s (q finishes, Ctrl-C stops)"])
                self.assertEqual(view.shown[-1], ([], "guide", None), "the window was given back")

    def test_the_window_names_a_stop_that_works_during_the_countdown(self) -> None:
        view = _View()
        console = _Console(_enter(3) + _enter(1), [])
        arm, _ = self._two_stations(console, view=view)
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0", "move s1"])
        counting = [lines for lines, tone, _ in view.shown if lines and "MOVES BY ITSELF IN" in lines[0]]
        self.assertEqual([lines[0] for lines in counting],
                         [f"HANDS OFF: THE ARM MOVES BY ITSELF IN {n} S" for n in (3, 2, 1)])
        for lines in counting:
            self.assertIn("q or ESC here finishes before it moves, Ctrl-C in the console stops", lines)
            self.assertNotIn("CTRL-C STOPS", " ".join(lines), "Ctrl-C typed into the window stops nothing")

    def test_anything_else_typed_during_the_countdown_stops_it_and_asks_again(self) -> None:
        # The payload, the capture at s0, Enter for the hands off; "x" during the countdown; Enter for the hands off
        # again, then the capture at s1.
        console = _Console(_enter(3) + _enter(1) + _enter(1), [], counting_down=[None, "x"])
        arm, _ = self._two_stations(console)
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0", "move s1"])
        before = console.said[:[i for i, line in enumerate(console.said) if "moves by itself in 1 s" in line][0]]
        self.assertIn("'x' stopped the countdown: Enter starts it again, q finishes.", before)
        self.assertEqual(before.count("Hands off the arm - Enter to continue (q finishes here)"), 2)
        self.assertEqual(sum("moves by itself in 3 s" in line for line in console.said), 2)

    def test_a_key_in_the_window_other_than_q_stops_the_countdown_and_asks_again(self) -> None:
        console = _Console(_enter(3) + _enter(1) + _enter(1), [])
        arm, _ = self._two_stations(console, view=_CountdownView("enter", after=2))
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0", "move s1"])
        self.assertIn("Enter in the window stopped the countdown: Enter starts it again, q finishes.", console.said)
        self.assertEqual(console.said.count("Hands off the arm - Enter to continue (q finishes here)"), 2)

    def test_yes_starts_the_countdown_as_enter_does(self) -> None:
        clock = _Clock()
        console = _Console([None, "yes"], [])
        console.clock = clock
        guide = HandGuide(console, None, clock=clock, sleep=clock.sleep)
        guide.touched = True
        self.assertTrue(guide.hands_off())
        self.assertEqual(sum("moves by itself in" in line for line in console.said), 3)
        self.assertFalse(guide.touched)
        self.assertAlmostEqual(clock.now, 3.0, places=6)  # the answer at the first look, then three seconds

    def test_q_at_the_hands_off_prompt_finishes_before_the_arm_moves_again(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(3)], clock)
        console = _Console(_enter(2) + [None, "q"], arm.log)
        routine, _ = _routine(arm, console, clock)
        poses = [Pose.from_matrix(TOOL[i], frame=Frame.BASE, label=f"s{i}") for i in range(3)]
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):  # one sample
            routine.run_with_poses(poses, adjust=True)
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move s0"])


def _stations_file(folder: Path, indices: Any, name: str = "stations.json") -> Path:
    """A stations file of joint stations at the synthetic poses ``indices``, as a run guided by hand writes one."""
    path = folder / name
    path.write_text(json.dumps([{"label": f"s{i}", "joints_deg": JOINTS_DEG[i]} for i in indices], indent=2),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# The stations file a run replays is never written over by it
# ---------------------------------------------------------------------------------------------------------------------


class TheFileAReplayReadsIsNeverWrittenOverTests(unittest.TestCase):

    def test_a_replay_with_adjust_that_saves_to_its_own_file_keeps_it_whole_when_finished_partway(self) -> None:
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        stations = _stations_file(folder, range(4))
        taught = stations.read_bytes()
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(4)], clock)
        # The payload, the capture at the first station, then q at the hands-off prompt: finished partway.
        console = _Console(_enter(2) + [None, "q"], arm.log)
        routine, _ = _routine(arm, console, clock)
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):
            routine.run_from_json(stations, adjust=True, stations_save_path=stations)
        self.assertEqual([line for line in arm.log if line.startswith("move")], ["move_to_joints 0"])
        self.assertEqual(stations.read_bytes(), taught, "the taught stations were written over")
        adjusted = folder / "stations.adjusted.json"
        self.assertEqual(routine.stations_path, str(adjusted))
        self.assertEqual([record["label"] for record in json.loads(adjusted.read_text(encoding="utf-8"))], ["s0"])
        said = " ".join(console.said)
        self.assertIn(f"{stations} is the file the stations were read from, so it is not written over: the stations "
                      f"counted by hand go to {adjusted}", said)

    def test_the_same_file_by_another_spelling_is_the_same_file(self) -> None:
        from src.robot.execution.calibration import stations_file_to_write

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        stations = _stations_file(folder, range(2))
        other = folder / "sub" / ".." / "stations.json"
        (folder / "sub").mkdir()
        written = stations_file_to_write(stations, other)
        assert written is not None
        self.assertEqual(Path(os.path.normpath(written)), folder / "stations.adjusted.json")
        self.assertEqual(stations_file_to_write(stations, folder / "new.json"), str(folder / "new.json"))
        self.assertEqual(stations_file_to_write(None, folder / "new.json"), str(folder / "new.json"))
        self.assertIsNone(stations_file_to_write(stations, None))


# ---------------------------------------------------------------------------------------------------------------------
# A routine a person has guided asks for the hands off before it drives again, adjust or not
# ---------------------------------------------------------------------------------------------------------------------


class AGuidedRoutineAsksForTheHandsOffBeforeAnyReplayTests(unittest.TestCase):

    def _guided_then_replayed(self, answer: str) -> tuple[_GuidedArm, _Console, Any]:
        """One capture by hand, then q; then a replay of three joint stations on the same routine, no adjust."""
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        stations = _stations_file(folder, range(1, 4))
        clock = _Clock()
        arm = _GuidedArm([_stand(0)], clock)
        # The payload, one capture, q; then the hands-off prompt of the replay.
        console = _Console(_enter(2) + [None, "q"] + [None, answer], arm.log)
        routine, _ = _routine(arm, console, clock)
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):
            routine.run_freedrive(samples=4)
        self.assertEqual(arm.log[-1], "left, held")
        try:
            outcome: Any = routine.run_from_json(stations)
        except CalibrationDataError as exc:
            outcome = exc
        return arm, console, outcome

    def test_the_hands_off_prompt_and_the_countdown_come_before_the_first_joint_move(self) -> None:
        arm, console, _ = self._guided_then_replayed("")
        moves = [i for i, line in enumerate(arm.log) if line.startswith("move")]
        self.assertEqual([arm.log[i] for i in moves], [f"move_to_joints {i}" for i in range(1, 4)])
        before = arm.log[:moves[0]]
        self.assertIn("say: Hands off the arm - Enter to continue (q finishes here)", before)
        self.assertEqual([line for line in before if "moves by itself in" in line],
                         [f"say: Hands off: the arm moves by itself in {n} s (q finishes, Ctrl-C stops)"
                          for n in (3, 2, 1)])
        # Asked once: nobody touched the arm between the replay's own moves.
        self.assertEqual(console.said.count("Hands off the arm - Enter to continue (q finishes here)"), 1)
        self.assertEqual(arm.sessions, 1, "the replay freed nothing")

    def test_q_at_that_prompt_moves_nothing(self) -> None:
        arm, _, outcome = self._guided_then_replayed("q")
        self.assertIsInstance(outcome, CalibrationDataError)
        self.assertEqual([line for line in arm.log if line.startswith("move")], [])

    def test_a_routine_nobody_guided_asks_nothing_before_a_replay(self) -> None:
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        clock = _Clock()
        arm = _GuidedArm([_stand(0)], clock)
        console = _Console([], arm.log)
        routine, _ = _routine(arm, console, clock)
        result = routine.run_from_json(_stations_file(folder, range(4)))
        self.assertEqual(result.num_samples, 4)
        self.assertEqual(console.said, [])


# ---------------------------------------------------------------------------------------------------------------------
# An arm that offers no hand guiding
# ---------------------------------------------------------------------------------------------------------------------


class AnArmWithoutHandGuidingTests(unittest.TestCase):

    def test_freedrive_and_adjust_are_refused_naming_the_capability_and_nothing_moves(self) -> None:
        clock = _Clock()
        arm = _PlainArm([_stand(0)], clock)
        self.assertNotIsInstance(arm, SupportsFreedrive)
        routine, _ = _routine(arm, _Console([], arm.log), clock)
        with self.assertRaisesRegex(HandGuidingRefused, "offers no hand guiding \\(SupportsFreedrive"):
            routine.run_freedrive()
        with self.assertRaisesRegex(HandGuidingRefused, "Run the fixed stations without it; nothing moved"):
            routine.run_with_poses([Pose.from_matrix(TOOL[0], frame=Frame.BASE, label="s0")], adjust=True)
        self.assertEqual([line for line in arm.log if line.startswith("move")], [])

    def test_the_same_fixed_stations_run_without_adjust(self) -> None:
        clock = _Clock()
        arm = _PlainArm([_stand(0)], clock)
        routine, _ = _routine(arm, _Console([], arm.log), clock)
        result = routine.run_with_poses([Pose.from_matrix(TOOL[i], frame=Frame.BASE, label=f"s{i}") for i in range(4)])
        self.assertEqual((result.num_samples, result.stations_path), (4, None))


# ---------------------------------------------------------------------------------------------------------------------
# The way to a target, in the tool's own terms
# ---------------------------------------------------------------------------------------------------------------------


class TheWayToATargetTests(unittest.TestCase):

    def test_the_offset_is_along_the_tools_axes_not_the_bases(self) -> None:
        down = Pose.tool_down(400.0, 0.0, 300.0)
        here = FreedriveSample(joints_rad=(0.0,) * 6, joint_speeds_rad_s=(0.0,) * 6,
                               tcp_xyz_mm=(400.0, 0.0, 300.0),
                               tcp_rotvec_rad=tuple(float(v) for v in down.axis_angle_rad()),  # type: ignore[arg-type]
                               t_s=0.0)
        above = Pose.tool_down(400.0, 0.0, 350.0)
        offset = offset_in_tool(here, above)
        # 50 mm up in the base is 50 mm back along a tool that points down.
        np.testing.assert_allclose(offset.along_mm, (0.0, 0.0, -50.0), atol=1e-9)
        np.testing.assert_allclose(offset.about_deg, (0.0, 0.0, 0.0), atol=1e-9)
        self.assertAlmostEqual(offset.closeness, 1.0 - 50.0 / 150.0)
        turned = offset_in_tool(here, Pose.tool_down(400.0, 0.0, 300.0, yaw_deg=20.0))
        self.assertAlmostEqual(turned.angle_deg, 20.0, places=6)
        self.assertIn("turn", turned.line())

    def test_the_preview_shows_the_way_to_each_target_and_moves_on_after_a_capture(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(i) for i in range(4)], clock)
        view = _View()
        routine, _ = _routine(arm, _Console(_enter(5), arm.log), clock, view=view)
        targets = [Pose.from_matrix(TOOL[i], frame=Frame.BASE, label=f"t{i}") for i in range(4)]
        result = routine.run_freedrive(stations=targets, samples=4)
        self.assertEqual([verdict.label for verdict in result.pose_log], ["t0", "t1", "t2", "t3"])
        banners = [lines[0] for lines, tone, _ in view.shown if lines and lines[0].startswith("MOVE THE ARM")]
        self.assertIn("MOVE THE ARM BY HAND to target 't1' (2 of 4), then Enter", banners)
        guided = [(lines, bar) for lines, tone, bar in view.shown if len(lines) > 1 and lines[1].startswith("to 't0'")]
        self.assertTrue(guided)
        lines, bar = guided[-1]
        self.assertIn("mm along tool x / y / z", lines[1])
        self.assertEqual(bar, 1.0, "the person stands at the target")
        self.assertIn("counted", lines[-1])


# ---------------------------------------------------------------------------------------------------------------------
# The preview as the guide's window
# ---------------------------------------------------------------------------------------------------------------------


class ThePreviewIsTheGuidesWindowTests(unittest.TestCase):

    def test_keys_typed_while_a_guide_shows_come_back_and_none_otherwise(self) -> None:
        gui = _Gui(keys=(13, 32, ord("s"), ord("q")))
        preview, _ = _preview(gui=gui, live_hz=0)
        self.addCleanup(preview.close)
        preview.guide(["MOVE THE ARM BY HAND"], "guide")
        preview.start()
        self.assertTrue(_until(lambda: not gui.keys))
        self.assertTrue(_until(lambda: len(gui.calls) > 10))
        self.assertEqual([preview.poll_key() for _ in range(5)], ["enter", "enter", "skip", "finish", None])
        quiet = _Gui(keys=(13,))
        unguided, _ = _preview(gui=quiet, live_hz=0)
        self.addCleanup(unguided.close)
        unguided.start()
        self.assertTrue(_until(lambda: not quiet.keys))
        self.assertIsNone(unguided.poll_key(), "a key typed into a sweep's window is not a capture")

    def test_outside_is_drawn_red_with_a_border_and_the_closeness_bar_is_drawn(self) -> None:
        frame = np.full((90, 640, 3), 128, dtype=np.uint8)
        lines = ["OUTSIDE - NOT CAPTURED: wrist 3"]
        red = preview_module.annotate(frame, lines=lines, tone="outside", bar=0.5)
        barless = preview_module.annotate(frame, lines=lines, tone="outside")
        guide = preview_module.annotate(frame, lines=lines, tone="guide")
        self.assertEqual(tuple(int(v) for v in red[2, 10]), (0, 0, 230))
        self.assertEqual(red.shape[0], barless.shape[0] + 16, "the bar is one row under the frame")
        strip = barless.shape[0] - frame.shape[0]
        self.assertEqual(tuple(int(v) for v in red[strip + 2, 320]), (0, 0, 230), "the frame has a red border")
        self.assertEqual(tuple(int(v) for v in guide[strip + 2, 320]), (128, 128, 128), "no border off the boundary")

    def test_the_live_line_says_how_far_how_tilted_and_how_well_the_board_is_seen(self) -> None:
        from src.calibration.targets import Observation

        T = np.eye(4)
        T[:3, :3] = _rotation_x(150.0)  # the board's face 30 degrees off facing the camera
        T[:3, 3] = (0.0, 0.0, 500.0)
        seen = Observation(kind="aruco", T_cam_to_target=T, points_px=np.zeros((4, 2)), reprojection_px=0.3)
        line = preview_module.PreviewState().guided_lines(["MOVE"], seen)[1]
        self.assertEqual(line, "in view: 4 corners, 0.30 px, 500 mm away, tilted 30 deg")
        unseen = Observation(kind="aruco", why_not="no DICT_4X4_100 marker in view")
        self.assertEqual(preview_module.PreviewState().guided_lines(["MOVE"], unseen)[1],
                         "board not seen: no DICT_4X4_100 marker in view")


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])


class TheEventsOfAGuidedRunTests(unittest.TestCase):

    def test_a_guided_capture_is_on_the_events_like_a_station_and_moves_nothing(self) -> None:
        clock = _Clock()
        arm = _GuidedArm([_stand(0), _stand(1)], clock)
        routine, events = _routine(arm, _Console(_enter(3), arm.log), clock)
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):  # two of four: the events stand
            routine.run_freedrive(samples=2)
        kinds = [kind for kind, _ in events]
        self.assertNotIn(RobotCalibrationEvent.MOVING_TO_POSE, kinds)
        self.assertEqual(kinds.count(RobotCalibrationEvent.POSE_ACCEPTED), 2)
        self.assertIs(routine.calibration_mode, MountingMode.EYE_TO_HAND)


# ---------------------------------------------------------------------------------------------------------------------
# The noun and the CLI hand the run to the routine
# ---------------------------------------------------------------------------------------------------------------------


class TheNounRunsTheWayTheOptionsSayTests(unittest.TestCase):

    def _noun(self, arm: Any, console: Any = None, **options: Any) -> Any:
        from unittest.mock import MagicMock

        from src.camera.orchestration.camera import Camera
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.hand_eye import HandEyeCalibration, SweepOptions
        from src.robot.execution.robot import Robot
        from tests.test_camera_boundaries import _rgbd_rig
        from tests.test_hand_eye_calibration import _Streamer, _eth_result

        camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=_Streamer())
        self.addCleanup(camera.release)
        if not getattr(self, "out", None):
            self.out = self.enterContext(tempfile.TemporaryDirectory())
        self.solved = MagicMock(return_value=_eth_result())
        return HandEyeCalibration.from_parts(
            robot=Robot.from_parts(arm=arm, gripper=None, lock_key=None), camera=camera,
            robot_config=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.51"}}),
            mode="eye_to_hand", options=SweepOptions(out_dir=self.out, **options), console=console)

    @staticmethod
    def _guided_arm() -> Any:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        class _Guided(DummyRobotArm):
            def freedrive(self) -> Any:
                raise AssertionError("nothing is freed here")

            def controller_payload(self) -> None:
                return None

        return _Guided()

    def test_a_replay_of_the_file_this_run_would_write_writes_beside_it_and_says_so(self) -> None:
        from unittest.mock import patch

        for way in ("adjust", "freedrive"):
            with self.subTest(way=way):
                self.out = self.enterContext(tempfile.TemporaryDirectory())
                # The file 09 wrote, named back as 09 and 10 suggest: the input and this run's output by default.
                stations = _stations_file(Path(self.out), range(4), name="eye_to_hand_overhead_stations.json")
                console = _Console([], [])
                calibration = self._noun(self._guided_arm(), console=console, fixed_poses=str(stations),
                                         **{way: True})
                verb = "run_from_json" if way == "adjust" else "run_freedrive"
                with patch.object(CalibrationRoutine, verb, autospec=True, return_value=self.solved()) as ran:
                    calibration.run()
                adjusted = ran.call_args.kwargs["stations_save_path"]
                self.assertEqual(Path(adjusted), Path(self.out) / "eye_to_hand_overhead_stations.adjusted.json")
                self.assertIn(f"{stations} is the file the stations were read from, so it is not written over: the "
                              f"stations counted by hand go to {adjusted}", " ".join(console.said))

    def test_a_failed_run_that_counted_nothing_names_no_stations_file_even_when_an_old_one_is_there(self) -> None:
        from src.robot.execution.hand_eye import CalibrationOutcome

        self.out = self.enterContext(tempfile.TemporaryDirectory())
        old = _stations_file(Path(self.out), range(4), name="eye_to_hand_overhead_stations.json")
        console = _Console([None, "n"], [])  # the payload is not confirmed: nothing is freed, nothing counted
        report = self._noun(self._guided_arm(), console=console, freedrive=True, samples=6).run()
        self.assertIs(report.outcome, CalibrationOutcome.SWEEP_FAILED)
        self.assertIn("the controller payload was not confirmed", report.failure)
        self.assertTrue(old.is_file())
        self.assertEqual(report.stations_path, "")
        self.assertNotIn("replay without hands", report.render())

    def test_a_failed_run_that_counted_a_pose_names_its_stations_file(self) -> None:
        from unittest.mock import patch

        from src.robot.execution.hand_eye import CalibrationOutcome
        from src.robot.execution.pose_provider import station_record

        self.out = self.enterContext(tempfile.TemporaryDirectory())
        calibration = self._noun(self._guided_arm(), freedrive=True, samples=6)
        stations = Path(self.out) / "eye_to_hand_overhead_stations.json"

        def counted_one_then_failed(routine: CalibrationRoutine, targets: Any, **keywords: Any) -> Any:
            routine._begin(keywords["stations_save_path"])  # noqa: SLF001
            routine._stations = [station_record("hand_01", JointPositions([0.0] * 6),  # noqa: SLF001
                                                Pose.from_matrix(TOOL[0], frame=Frame.BASE))]
            stations.write_text("[]", encoding="utf-8")
            raise CalibrationDataError("too few samples")

        with patch.object(CalibrationRoutine, "run_freedrive", autospec=True, side_effect=counted_one_then_failed):
            report = calibration.run()
        self.assertIs(report.outcome, CalibrationOutcome.SWEEP_FAILED)
        self.assertEqual(Path(report.stations_path), stations)
        self.assertIn(f"replay without hands: --fixed-poses {report.stations_path}", report.render())

    def test_freedrive_runs_the_guided_routine_and_the_report_names_the_stations_file(self) -> None:
        from unittest.mock import patch

        from src.robot.drivers.dummy.arm import DummyRobotArm

        class _Guided(DummyRobotArm):
            def freedrive(self) -> Any:
                raise AssertionError("the routine is patched; nothing is freed here")

            def controller_payload(self) -> None:
                return None

        calibration = self._noun(_Guided(), freedrive=True, samples=6)
        stations = Path(self.out) / "eye_to_hand_overhead_stations.json"

        def guided(routine: CalibrationRoutine, targets: Any, **keywords: Any) -> Any:
            stations.write_text("[]", encoding="utf-8")
            return self.solved()

        with patch.object(CalibrationRoutine, "run_freedrive", autospec=True, side_effect=guided) as ran:
            self.solved.return_value.stations_path = str(stations)
            report = calibration.run()
        self.assertEqual(ran.call_args.args[1], None)
        self.assertEqual(ran.call_args.kwargs["samples"], 6)
        self.assertEqual(Path(ran.call_args.kwargs["stations_save_path"]), stations)
        self.assertIn(f"  stations          {stations}", report.render())
        self.assertIn(f"replay without hands: --fixed-poses {stations}", report.render())
        self.assertIn("=== 3. SWEEP === guided by hand: the arm moves only when you move it", report.render())

    def test_adjust_on_an_arm_nobody_can_guide_is_a_build_refusal(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.execution.hand_eye import CalibrationOutcome

        report = self._noun(DummyRobotArm(), adjust=True,
                            fixed_poses=[Pose.from_matrix(TOOL[0], frame=Frame.BASE)]).run()
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED)
        assert report.build is not None
        self.assertIn("HandGuidingRefused: adjust frees the arm at each station by hand, and DummyRobotArm offers no "
                      "hand guiding", report.build.refusal)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
