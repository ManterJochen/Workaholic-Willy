"""A task picks the grey parts and ends without a long search: the owner's scene, on the real chain (2026-10-08).

The presentation's two defects on one mat. Three grey cubes and two red ones, and the task "every grey cube onto the
drop" (until empty). The detector boxed every part with the prompt's words, as Qwen3-VL did on the cell, so the red
cubes were targets too and the task placed all five; and after the last part it searched two empty picks of four looks
each. Now the camera source maps "each separate grey cube" onto the object only where the label holds its words, and
judges every part's colour on its pixels (``src/robot/perception/colour_check.py``): a red cube keeps its mask as a
neighbour under a label no object carries, and the calculator never computes it. An empty pick counts the looks it
perceived from, and where the part placed was the last target its pick's first look counted, the next pick looks from
home alone: seeing nothing there ends the task at once.

Real: the task, the pick service with its pick loop and grasp policy, the camera source (label mapping and colour check)
over a ``TwoStageBackend``, and the owner's toggle on tool DO0. Stand-ins: the arm (``tests/_task_fakes.TaskArm``), the
wrist camera, which ray-casts the mat from where the tool stands (``tests/_wrist_views.render``, 200 mm behind the TCP
and looking along the tool), a grounder that boxes every part it is shown, a segmenter that cuts each box's part, and a
calculator that grasps the part it is handed straight down at its middle. A part the jaws close on leaves the mat.
Four looks, as on the cell: home and three taught joints, which this arm reaches with the tool where home has it.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Transform
from src.models.detection.types import Detection
from src.robot.core import JointPositions
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grippers.jaw_io import JawIOGripper
from tests._task_fakes import POSES, RecordingHooks, TaskArm, _ArmIO, do0_changes, motions
from tests._wrist_views import K, Box, render

#: The wrist camera: 200 mm behind the TCP along the tool, looking along it; a tool pointing down looks straight down.
_MOUNT = np.eye(4)
_MOUNT[2, 3] = -200.0
#: The owner's black mat (L* 16) and the parts on it, BGR.
_MAT_BGR = (34, 40, 42)
_BGR = {"grey": (156, 160, 162), "red": (40, 40, 200)}
#: The four looks: home, then three taught joints.
LOOKS = ("home", JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0),
         JointPositions.deg(-10.0, -90.0, -110.0, -60.0, 90.0, 0.0),
         JointPositions.deg(0.0, -80.0, -120.0, -60.0, 90.0, 0.0))
_OTHER_LOOKS = {tuple(round(value, 1) for value in look.degrees()) for look in LOOKS[1:]}


@dataclass(eq=False)
class _Part:
    """A 40 mm cube on the mat: its name, its colour and where its middle stands (BASE mm)."""

    name: str
    colour: str
    xy: tuple[float, float]

    @property
    def box(self) -> Box:
        x, y = self.xy
        return Box((x - 20.0, y - 20.0, 0.0), (x + 20.0, y + 20.0, 40.0), self.name)


def _owner_s_scene() -> list[_Part]:
    """Three grey cubes and two red ones, 100 mm and more apart, in the camera's view from home."""
    return [_Part("grey 1", "grey", (-200.0, -560.0)), _Part("red 1", "red", (-200.0, -440.0)),
            _Part("grey 2", "grey", (-50.0, -440.0)), _Part("red 2", "red", (100.0, -440.0)),
            _Part("grey 3", "grey", (100.0, -600.0))]


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str


class _Mat:
    """The mat, the parts on it, and the wrist camera over it: each frame rendered from where the tool stands."""

    rig_id = "wrist"

    def __init__(self, arm: TaskArm, parts: list[_Part]) -> None:
        self.arm = arm
        self.parts = list(parts)
        self.gripped: list[_Part] = []
        self.hit = np.zeros((1, 1), dtype=np.int64)
        self.shown: tuple[_Part, ...] = ()

    def grab(self) -> RGBDFrame:
        camera_to_base = np.asarray(self.arm.get_tcp_pose().to_matrix(), dtype=np.float64) @ _MOUNT
        self.shown = tuple(self.parts)
        depth, self.hit = render(camera_to_base, tuple(part.box for part in self.shown))
        colour = np.empty((*self.hit.shape, 3), dtype=np.uint8)
        colour[...] = _MAT_BGR
        for index, part in enumerate(self.shown):
            colour[self.hit == index] = _BGR[part.colour]
        return RGBDFrame(color=colour, depth=np.round(depth).astype(np.uint16))

    def get_intrinsics(self) -> np.ndarray:
        return K.copy()

    def part_of(self, mask: np.ndarray) -> "_Part | None":
        """The part most of ``mask`` shows in the last frame, ``None`` for none."""
        under = self.hit[np.asarray(mask).astype(bool)]
        under = under[under >= 0]
        if not under.size:
            return None
        return self.shown[int(np.bincount(under).argmax())]

    def grip_at(self, xy: tuple[float, float]) -> None:
        """The jaws closed with the tool at ``xy``: the part there leaves the mat with them."""
        near = [part for part in self.parts if np.hypot(part.xy[0] - xy[0], part.xy[1] - xy[1]) < 25.0]
        for part in near[:1]:
            self.parts.remove(part)
            self.gripped.append(part)


class _Grounder:
    """Boxes every part the frame shows, each labelled with the prompt's words, as Qwen3-VL labelled them on the cell."""

    def __init__(self, mat: _Mat, log: list[Any]) -> None:
        self.mat = mat
        self.log = log
        self.asked: list[str] = []

    def detect_all(self, bgr: Any, prompt: str) -> list[Detection]:
        self.asked.append(prompt)
        self.log.append(("ground", prompt))
        found = []
        for index in range(len(self.mat.shown)):
            rows, cols = np.nonzero(self.mat.hit == index)
            if cols.size:
                box = [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)]
                found.append(Detection(box=box, x_center=(box[0] + box[2]) / 2.0, y_center=(box[1] + box[3]) / 2.0,
                                       label=prompt, score=1.0))
        return found


class _Segmenter:
    """Cuts the part a box holds: the pixels of the part most of the box shows."""

    def __init__(self, mat: _Mat) -> None:
        self.mat = mat

    def segment_detection(self, bgr: Any, det: Detection) -> _Seg:
        x0, y0, x1, y1 = (int(round(value)) for value in det.box)
        window = self.mat.hit[y0:y1, x0:x1]
        index = int(np.bincount(window[window >= 0]).argmax())
        return _Seg(mask=(self.mat.hit == index).astype(np.uint8), label=det.label)


class _Calculator:
    """Grasps the part a mask shows straight down at its middle, closing along base x, the parts earlier on the mat's list
    scoring higher; and writes down every part it computed, with its label, and the parts it was handed as neighbours."""

    def __init__(self, mat: _Mat) -> None:
        self.mat = mat
        self.computed: list[tuple[str, str]] = []
        self.neighbours: list[str] = []

    def compute_result(self, seg: Any, depth: Any, *_: Any, other_object_masks: Any = (), **__: Any) -> GraspResult:
        part = self.mat.part_of(seg.mask)
        self.computed.append(("" if part is None else part.name, str(seg.label)))
        self.neighbours.extend(found.name for found in (self.mat.part_of(mask) for mask in other_object_masks or ())
                               if found is not None)
        if part is None:
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,))
        x, y = part.xy
        grasp = GraspPoint(position=np.array([x, y, 20.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                           label=part.name)
        return GraspResult(candidates=(grasp,), reasons=(), top_score=0.9 - 0.01 * self.mat.parts.index(part))


class _GrippingIO(_ArmIO):
    """The tool I/O of the owner's toggle: every change of DO0 moves the jaws, and a close takes the part at the tool."""

    def __init__(self, log: list[Any], arm: TaskArm, mat: _Mat) -> None:
        super().__init__(log, arm)
        self.mat = mat
        self.closed = False

    def set_digital_output(self, pin: int, value: bool, **keywords: Any) -> None:
        before = self.do.get(pin, False)
        super().set_digital_output(pin, value, **keywords)
        if pin == 0 and self.do.get(pin, False) != before:
            self.closed = not self.closed
            if self.closed:
                tool = self.arm.get_tcp_pose()
                self.mat.grip_at((float(tool.position_mm[0]), float(tool.position_mm[1])))


@dataclass
class _Ran:
    report: Any
    log: list[Any]
    mat: _Mat
    grounder: _Grounder
    calculator: _Calculator
    hooks: RecordingHooks

    def after_place(self, number: int) -> list[Any]:
        """The log after the ``number``-th part was placed."""
        marks = [index for index, entry in enumerate(self.log) if entry == ("event", "task.placed")]
        return self.log[marks[number - 1] + 1:]


def _task(parts: list[_Part], **options: Any) -> _Ran:
    """The owner's task on the real chain: every grey cube onto the drop, until empty, from four wrist looks."""
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
    from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, run_task
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
    from src.robot.perception import RealSenseVisionPerceptionSource

    colour_check = options.pop("colour_check", "on")
    log: list[Any] = []
    arm = TaskArm(log)
    mat = _Mat(arm, parts)
    io = _GrippingIO(log, arm, mat)
    jaws = JawIOGripper(io, actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=lambda _question: "open", sleep=lambda _s: None)
    jaws.connect()
    io.closed = False
    del log[:]
    grounder = _Grounder(mat, log)
    calculator = _Calculator(mat)
    source = RealSenseVisionPerceptionSource(streamer=mat, detector=grounder, segmenter=_Segmenter(mat),
                                             prompt="object", warmup_grabs=0, colour_check=colour_check)
    policy = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99,  # type: ignore[arg-type]
                                  require_steady_before_motion=True)
    resolver = EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(_MOUNT, from_frame=Frame.CAMERA,
                                                                          to_frame=Frame.TOOL))
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=calculator, perception=source, mode=GraspMode.EASY,  # type: ignore[arg-type]
        gripper=jaws, policy=policy, frame_resolver=resolver, max_attempts=1)
    service.configured_looks = LOOKS
    hooks = RecordingHooks(on_event=lambda name, _data: log.append(("event", name)))
    report = run_task(service, TaskPlan(object="grey cube", place=PlaceAt(pose="drop_left"), scope="until_empty",
                                        options=TaskOptions(**options)), hooks=hooks, poses=POSES)
    return _Ran(report=report, log=log, mat=mat, grounder=grounder, calculator=calculator, hooks=hooks)


def _groundings(log: list[Any]) -> int:
    return sum(1 for entry in log if isinstance(entry, tuple) and entry[:1] == ("ground",))


def _moves_to_other_looks(log: list[Any]) -> int:
    """The moves to a look other than home, where the arm already stands when a pick starts."""
    return sum(1 for entry in motions(log) if entry[0] == "joints" and entry[1] in _OTHER_LOOKS)


class TheOwnersSceneTests(unittest.TestCase):
    def test_the_grey_cubes_are_placed_and_the_task_ends_at_home_with_one_look(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task(_owner_s_scene())

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.parts_placed)
        self.assertEqual(["grey", "grey", "grey"], [part.colour for part in ran.mat.gripped])
        self.assertTrue(ran.report.at_return)
        self.assertEqual(("home",), motions(ran.log)[-1])
        self.assertEqual(6, do0_changes(ran.log), "DO0 changes twice per part: three closes, three releases")
        # The red cubes: never computed as the part, and still on the mat, handed to the calculator as neighbours.
        self.assertEqual({"grey 1", "grey 2", "grey 3"}, {name for name, _ in ran.calculator.computed})
        self.assertEqual({"grey cube"}, {label for _, label in ran.calculator.computed})
        self.assertEqual(["red 1", "red 2"], [part.name for part in ran.mat.parts])
        self.assertIn("red 1", ran.calculator.neighbours)
        self.assertIn("red 2", ran.calculator.neighbours)
        # After the third part: the check look from home, one grounding, no move to another look.
        after = ran.after_place(3)
        self.assertEqual(1, _groundings(after))
        self.assertEqual(0, _moves_to_other_looks(after))
        self.assertEqual(4, ran.report.picks)
        [found] = ran.hooks.of("task.nothing_found")
        self.assertTrue(found["check_look"])

    def test_the_pick_s_first_look_counts_the_grey_cubes_alone(self) -> None:
        ran = _task(_owner_s_scene())

        counted = [report.telemetry.get("targets_by_look") for _, _, report in ran.hooks.picks]
        self.assertEqual([[3], [2], [1], [0]], counted)

    def test_without_the_check_look_one_empty_pass_over_the_four_looks_ends_it(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task(_owner_s_scene(), check_look=False)

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.parts_placed)
        after = ran.after_place(3)
        self.assertEqual(4, _groundings(after), "one empty pass: a grounding at each of the four looks")
        self.assertEqual(3, _moves_to_other_looks(after))
        [found] = ran.hooks.of("task.nothing_found")
        self.assertEqual(4, found["looks"])

    def test_only_red_cubes_end_the_task_after_one_pass_with_no_grip(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task([part for part in _owner_s_scene() if part.colour == "red"])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual((0, 1), (ran.report.parts_placed, ran.report.picks))
        self.assertEqual(0, do0_changes(ran.log))
        self.assertEqual([], ran.calculator.computed)
        self.assertEqual(4, _groundings(ran.log))
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_with_the_colour_check_off_the_red_cubes_are_picked_too(self) -> None:
        """What the check takes away: the detector's words alone make every cube a grey one."""
        from src.robot.execution.task import TaskStop

        ran = _task(_owner_s_scene(), colour_check="off")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(5, ran.report.parts_placed)
        self.assertEqual(10, do0_changes(ran.log))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
