"""The console's one choice of looks: the first look only, when needed, or every look (the owner, 2026-10-08 night).

"Multi-View" and "Alle Posen" became one choice in the console. "Nur erster Blick" is multi-view off (Q11): the first
look alone and no generated view. "Bei Bedarf" is the looks as they always ran: the looking stops at the first look whose
grasp is safe enough (and, where the cell turns its trigger on, a weak look is not). "Alle Posen" is ``every_look``: a
wrist pick visits every look it was handed, however safe an earlier look's grasp, and goes on with the judgement of the
last look, fused over every look (the map's change A). A fixed camera never looks around and ignores it, and so does a
pick handed no looks.

The switch is the pick's, as ``both_faces`` is: the service hands it to the pick loop for one pick and takes it back
when the pick ends, whatever ended it; a task hands it to each of its picks but its check look, which is always its first
look alone; the console's plan carries it (``TaskOptionsIn.every_look``) and the library refuses it with multi-view off.

The scene is ``tests/_wrist_views.py``: a 40 mm cube on the bench, ray cast through the wrist D415 from the tool pose each
look puts the arm at. The pick loop and the service are the real ones; the calculator grasps the cube on every frame;
the policy records what it was asked to execute and moves nothing.
"""

from __future__ import annotations

import unittest
from typing import Any

import pydantic

from src.geometry import Frame, Transform
from src.robot.core import JointPositions
from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from tests._task_fakes import Pick, run
from tests._wrist_views import LookCalculator, LookingArm, WristCamera, camera_looking_at, camera_to_tool, tool_for

#: The looks a program declares: the cube from its +x side, its -x side and its +y side, 45 degrees up.
LOOK_PLUS_X = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_MINUS_X = JointPositions.deg(180.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_PLUS_Y = JointPositions.deg(90.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOKS = (LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y)
#: The Hand-E's contact patch (config/grippers/robotiq_hande.yaml), the owner's hand.
HAND_E = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)


def _label(joints: JointPositions) -> str:
    return "(" + ", ".join(f"{v:.1f}" for v in joints.degrees()) + ") deg"


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _poses() -> dict[JointPositions, Any]:
    return {
        LOOK_PLUS_X: tool_for(camera_looking_at(bearing_deg=0.0)),
        LOOK_MINUS_X: tool_for(camera_looking_at(bearing_deg=180.0)),
        LOOK_PLUS_Y: tool_for(camera_looking_at(bearing_deg=90.0)),
    }


class _Policy:
    """Executes nothing and says it executed: what the loop asked of the arm is what these tests read."""

    def __init__(self, arm: Any = None) -> None:
        self.arm = arm
        self.gripper = None
        self.executed: list[Any] = []

    def execute(self, grasp: Any) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


class _Watching(LookCalculator):
    """The scene's calculator, which also writes down whether the pick loop drove every look at each call."""

    orchestrator: Any = None

    def __init__(self, camera: WristCamera, **keywords: Any) -> None:
        super().__init__(camera, **keywords)
        self.every_look: list[bool] = []
        self.raise_on: int | None = None

    def compute_result(self, seg: Any, *args: Any, **kwargs: Any) -> Any:
        if self.orchestrator is not None:
            self.every_look.append(bool(self.orchestrator.every_look))
        if self.raise_on is not None and len(self.camera.taken) - 1 == self.raise_on:
            raise TypeError("a programmer's error")
        return super().compute_result(seg, *args, **kwargs)


def _loop(*, looks: tuple[JointPositions, ...], fixed: bool = False, **wiring: Any) -> tuple[Any, Any, Any, _Policy]:
    """The real pick loop on the looking arm, its wrist camera (or a fixed one), the scene's calculator and the
    recording policy."""
    arm = LookingArm(_poses())
    camera = WristCamera(arm)
    calculator = LookCalculator(camera)
    policy = _Policy(arm)
    resolver: Any = (StaticCameraToBaseResolver(transform=Transform.from_matrix(
        camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE)) if fixed
        else EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()))
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=calculator, perception=camera,  # type: ignore[arg-type]
        frame_resolver=resolver, policy=policy,  # type: ignore[arg-type]
        max_attempts=1, primary_camera_id="wrist", gripper_model=HAND_E, looks=looks, **wiring,
    )
    return orchestrator, arm, camera, policy


def _joints_moved_to(arm: LookingArm) -> list[tuple[float, ...]]:
    return [key for kind, key in arm.motions if kind == "joints"]


class EveryLookIsVisitedWhereThePickAsksTests(unittest.TestCase):
    def test_every_look_drives_every_look_after_a_safe_grasp(self) -> None:
        orchestrator, arm, camera, policy = _loop(looks=LOOKS, every_look=True)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([_key(look) for look in LOOKS], _joints_moved_to(arm), "a look was left out")
        self.assertEqual(3, len(camera.taken))
        looked = orchestrator.looked_around
        self.assertEqual(tuple(_label(look) for look in LOOKS), looked.visited)
        self.assertTrue(looked.every_look)
        self.assertTrue(looked.judged.good)
        self.assertEqual(1, len(policy.executed), "every look visited, and the part gripped once")

    def test_the_pick_goes_on_with_the_last_looks_grasp_fused_over_every_look(self) -> None:
        orchestrator, _, _, _ = _loop(looks=LOOKS, every_look=True)

        orchestrator.run()

        looked = orchestrator.looked_around
        self.assertEqual(_label(LOOK_PLUS_Y), looked.judged.look.label, "the grasp is not the last look's")
        self.assertEqual({_label(look) for look in LOOKS}, set(looked.looks_fused))
        self.assertEqual("", looked.moved_back, "the arm stands at the look its grasp was ranked on")

    def test_without_it_a_safe_grasp_at_the_first_look_ends_the_looking_as_before(self) -> None:
        orchestrator, arm, camera, _ = _loop(looks=LOOKS)

        orchestrator.run()

        self.assertEqual([_key(LOOK_PLUS_X)], _joints_moved_to(arm))
        self.assertEqual(1, len(camera.taken))
        self.assertFalse(orchestrator.looked_around.every_look)
        self.assertFalse(orchestrator.every_look, "the switch is off unless a pick asks for it")

    def test_a_pick_handed_no_looks_looks_once_whatever_it_asks(self) -> None:
        orchestrator, arm, camera, policy = _loop(looks=(), every_look=True)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([], _joints_moved_to(arm))
        self.assertEqual(1, len(camera.taken))
        self.assertFalse(orchestrator.looked_around.every_look, "a pick handed no looks drove none")

    def test_a_fixed_camera_ignores_it(self) -> None:
        orchestrator, arm, camera, policy = _loop(looks=LOOKS, fixed=True, every_look=True)

        report = orchestrator.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([], _joints_moved_to(arm), "a fixed camera's pick loop moved the arm to look")
        self.assertEqual(1, len(camera.taken))
        self.assertIsNone(orchestrator.looked_around)


class _ServiceCell:
    """The pick service on the looking arm, its wrist camera (or a fixed one), the watching calculator and the policy."""

    def __init__(self, *, fixed: bool = False) -> None:
        self.arm = LookingArm(_poses())
        self.camera = WristCamera(self.arm)
        self.calculator = _Watching(self.camera)
        self.policy = _Policy(self.arm)
        resolver: Any = (StaticCameraToBaseResolver(transform=Transform.from_matrix(
            camera_looking_at(bearing_deg=0.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE)) if fixed
            else EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()))
        self.service = AutonomousGraspService.from_components(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            mode=GraspMode.EASY, policy=self.policy,  # type: ignore[arg-type]
            frame_resolver=resolver, max_attempts=1,
        )
        self.orchestrator = self.service.runtime.orchestrator
        self.orchestrator.gripper_model = HAND_E
        self.orchestrator.primary_camera_id = "wrist"
        self.calculator.orchestrator = self.orchestrator


class TheServiceHandsItToOnePickTests(unittest.TestCase):
    def test_every_look_is_handed_to_one_pick_and_taken_back(self) -> None:
        cell = _ServiceCell()

        first = cell.service.pick(look=list(LOOKS), every_look=True)

        self.assertTrue(first.succeeded, first.failure_summary())
        self.assertEqual(tuple(_label(look) for look in LOOKS), first.looks)
        self.assertEqual([True, True, True], cell.calculator.every_look, "the loop did not hold the switch at a look")
        self.assertIs(True, first.telemetry.get("every_look"))
        self.assertIs(False, cell.orchestrator.every_look, "the switch outlived the pick")

        second = cell.service.pick(look=list(LOOKS))

        self.assertEqual((_label(LOOK_PLUS_X),), second.looks, "the next pick inherited the switch")
        self.assertNotIn("every_look", second.telemetry)

    def test_a_pick_that_raised_hands_it_back_too(self) -> None:
        cell = _ServiceCell()
        cell.calculator.raise_on = 1

        with self.assertRaises(TypeError):
            cell.service.pick(look=list(LOOKS), every_look=True)

        self.assertIs(False, cell.orchestrator.every_look)
        self.assertEqual((), tuple(cell.orchestrator.looks))

    def test_multi_view_off_looks_from_its_first_look_alone_whatever_every_look_says(self) -> None:
        cell = _ServiceCell()

        report = cell.service.pick(look=list(LOOKS), multi_view=False, every_look=True)

        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        self.assertEqual([False], cell.calculator.every_look, "the first look alone asked for every look")
        self.assertNotIn("every_look", report.telemetry)

    def test_a_fixed_camera_stops_at_the_first_look_that_finds_something_as_it_always_did(self) -> None:
        cell = _ServiceCell(fixed=True)

        report = cell.service.pick(look=list(LOOKS), every_look=True)

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual((_label(LOOK_PLUS_X),), report.looks)
        self.assertEqual([False], cell.calculator.every_look)


class ATaskHandsItToEachOfItsPicksTests(unittest.TestCase):
    def test_a_task_asked_for_every_look_hands_it_to_each_pick(self) -> None:
        from src.robot.execution.task import TaskOptions

        ran = run(["part", "part", "empty", "empty"], scope="until_empty", options=TaskOptions(every_look=True),
                  wrist=True)

        self.assertGreaterEqual(len(ran.service.calls), 2)
        for call in ran.service.calls:
            self.assertIs(True, call.get("every_look"), call)
            self.assertNotIn("multi_view", call)

    def test_the_defaults_hand_the_pick_no_every_look(self) -> None:
        ran = run(["part"])
        self.assertNotIn("every_look", ran.service.calls[0])

    def test_the_check_look_is_its_first_look_alone_where_every_look_was_asked(self) -> None:
        from src.robot.execution.task import TaskOptions

        looks = ("home", JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0))
        ran = run([Pick("part", looks=("home",), targets_first_look=1), Pick("empty", looks=("home",))],
                  scope="until_empty", options=TaskOptions(every_look=True), wrist=True, looks=looks)

        first, check = ran.service.calls
        self.assertIs(True, first.get("every_look"))
        self.assertIs(False, check["multi_view"])
        self.assertNotIn("every_look", check, "the check look drove every look")

    def test_every_look_with_multi_view_off_is_refused(self) -> None:
        from src.robot.execution.task import TaskOptions

        with self.assertRaises(ValueError):
            TaskOptions(multi_view=False, every_look=True)
        self.assertIs(False, TaskOptions().every_look)


class TheConsolesChoiceReachesTheLibraryTests(unittest.TestCase):
    _PLAN = {"object": "red cube", "place": {"kind": "pose", "pose": "drop_left", "pose_joints_deg": [0.0] * 6}}

    def test_a_plan_hands_every_look_and_the_carry_to_the_library(self) -> None:
        from api.task_run import library_plan

        plan = {**self._PLAN, "options": {"every_look": True, "carry": "over_the_rim"}}
        library, _ = library_plan(plan)

        self.assertIs(True, library.options.every_look)
        self.assertEqual("over_the_rim", library.options.carry)

    def test_a_plan_kept_without_them_looks_and_carries_as_the_cell_does(self) -> None:
        from api.task_run import library_plan

        library, _ = library_plan(self._PLAN)

        self.assertIs(False, library.options.every_look)
        self.assertIsNone(library.options.carry)

    def test_the_library_refuses_every_look_with_multi_view_off(self) -> None:
        from api.task_run import library_plan

        with self.assertRaises(ValueError):
            library_plan({**self._PLAN, "options": {"multi_view": False, "every_look": True}})

    def test_the_options_take_the_choice_and_refuse_a_carry_they_do_not_know(self) -> None:
        from api.schemas import TaskOptionsIn, TaskOptionsOut

        self.assertEqual((False, None), (TaskOptionsIn().every_look, TaskOptionsIn().carry))
        self.assertEqual((False, None), (TaskOptionsOut().every_look, TaskOptionsOut().carry))
        taken = TaskOptionsIn.model_validate({"every_look": True, "carry": "via_the_look"})
        self.assertEqual((True, "via_the_look"), (taken.every_look, taken.carry))
        with self.assertRaises(pydantic.ValidationError):
            TaskOptionsIn.model_validate({"carry": "sideways"})


if __name__ == "__main__":
    unittest.main()
