"""A task sets a part down at a taught pose: the pose says where the part's BOTTOM is let go (build plan, item 7).

The person teaches a place pose with empty jaws, the fingertips where the part's bottom should be released. Each part
hangs below the tool by its own amount, so the tool goes to the taught pose raised by the part's hang: the grasp's Z
less the part's declared support (an upper bound, so every error goes toward more air), or the declared
``payload.length_mm`` where the pick reports no grasp pose. The raised drop is screened (``nearest_configuration``,
``screen_configuration``) before the line in, and an ERROR there puts the part back where it was grasped, returns, and
asks (``pose_refused``). The place itself is ``Robot.place``: a planned move to the standoff, a line in, the release, a
line out. A line out refused AFTER the release still counts the part placed, and the return is still tried.

Before any motion a task screens every taught pose it will use (``task.pose_screened``): its place, raised by the
worst hang the cell declares, and a taught return. An ERROR there ends the task ``pose_refused`` with nothing moved. An
arm that screens nothing says so on the event, unscreened, and the moves judge each pose when they run. A Restart's
first motion is the planned move to the return pose, after its screen.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import MotionStatus
from src.robot.safety.planning.band import PoseVerdict
from tests._task_fakes import (
    GRASP_Z_MM,
    PARK_JOINTS,
    PART_XY,
    PLACE_TCP,
    PROTECTIVE,
    MeasuringHand,
    Pick,
    RecordingHooks,
    TaskArm,
    do0_changes,
    motions,
    run,
)


def _declaring(log: list[Any], length_mm: float | None, **keywords: Any) -> TaskArm:
    """An arm whose tree declares the carried part's hang past the fingertips and the bench at 0."""
    arm = TaskArm(log, **keywords)
    arm.config = SimpleNamespace(  # type: ignore[attr-defined]
        safety=SimpleNamespace(planning_world=SimpleNamespace(payload=SimpleNamespace(length_mm=length_mm),
                                                              perceived=SimpleNamespace(margin_mm=15.0))),
        grasping=SimpleNamespace(support=SimpleNamespace(height_mm=0.0, container=SimpleNamespace(
            floor_height_mm=None))))
    return arm


def _key(joints: Any) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _raised(by_mm: float) -> Pose:
    """The taught place pose raised in BASE Z by ``by_mm``, turned as taught."""
    return Pose(position_mm=PLACE_TCP.position_mm + (0.0, 0.0, by_mm), quaternion_xyzw=PLACE_TCP.quaternion_xyzw,
                frame=PLACE_TCP.frame)


def _line_in(ran: Any) -> Any:
    return [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]


class TheHangTests(unittest.TestCase):
    def test_the_drop_stands_the_grasps_hang_above_the_taught_pose(self) -> None:
        ran = run(["part"])

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), _line_in(ran)[1])
        drop = ran.hooks.of("task.drop_planned")[0]
        self.assertEqual("pose", drop["kind"])
        self.assertAlmostEqual(GRASP_Z_MM, drop["hang_mm"])
        self.assertIsNone(drop["rim_mm"])
        self.assertIsNone(drop["air_mm"])

    def test_a_part_off_a_raised_support_hangs_from_the_declared_one(self) -> None:
        log: list[Any] = []
        arm = _declaring(log, None)
        arm.config.grasping.support.height_mm = -10.0  # type: ignore[attr-defined]
        ran = run(["part"], arm=arm)

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM + 10.0), _line_in(ran)[1])

    def test_a_declared_container_floor_and_table_bound_the_parts_bottom_by_the_lower_of_the_two(self) -> None:
        """A part stood on the container floor or on the table: the lower of the two is the height no part stood
        below, so the hang errs toward more air whichever it stood on."""
        for floor_mm, lowest_mm in ((15.0, 0.0), (-25.0, -25.0)):
            with self.subTest(floor_mm=floor_mm):
                log: list[Any] = []
                arm = _declaring(log, None)
                arm.config.grasping.support.container.floor_height_mm = floor_mm  # type: ignore[attr-defined]
                ran = run(["part"], arm=arm)

                np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM - lowest_mm),
                                           _line_in(ran)[1])

    def test_without_a_grasp_pose_the_declared_length_stands_in(self) -> None:
        log: list[Any] = []
        ran = run([Pick("part", grasp=False)], arm=_declaring(log, 60.0))

        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, 60.0), _line_in(ran)[1])

    def test_with_neither_nothing_is_set_down_blind_and_the_part_cannot_go_back_either(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run([Pick("part", grasp=False)])

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(ran.log))
        self.assertNotIn("task.place_started", ran.names())


class ARefusedDropTests(unittest.TestCase):
    def test_an_error_on_the_raised_drop_puts_the_part_back_returns_and_asks(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        probe = TaskArm([])
        joints = probe.nearest_configuration(_raised(GRASP_Z_MM))
        arm = TaskArm(log, screens={_key(joints): PoseVerdict.GUARD_REFUSED})
        ran = run(["part"], arm=arm)

        self.assertIs(TaskStop.POSE_REFUSED, ran.report.stop, ran.report.sentence)
        self.assertFalse(ran.report.holding)
        self.assertEqual(0, ran.report.parts_placed)
        self.assertEqual(2, do0_changes(log), "the part was not let go where it was grasped")
        back = [m for m in motions(ran.after("task.put_back")) if m[0] == "move"]
        self.assertEqual([], back, "the put back's motions came after the event that says it is done")
        put_back = [m for m in motions(ran.after("task.drop_planned")) if m[0] == "move"]
        np.testing.assert_allclose((*PART_XY, GRASP_Z_MM), put_back[1][1])
        self.assertEqual(["task.drop_planned", "task.put_back", "task.return_started", "task.returned",
                          "task.part_finished"], ran.names()[-5:])
        self.assertEqual({"outcome": "executed"}, ran.hooks.of("task.put_back")[0])
        self.assertIs(False, ran.hooks.of("task.part_finished")[0]["placed"])
        self.assertEqual(("home",), motions(log)[-1])

    def test_a_put_back_that_does_not_release_stays_where_it_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        probe = TaskArm([])
        joints = probe.nearest_configuration(_raised(GRASP_Z_MM))
        hooks = RecordingHooks()
        refusing: list[bool] = []

        def refuse_after_the_drop(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if hooks.events and hooks.events[-1][0] == "task.drop_planned":
                refusing.append(True)
            return MotionStatus.WORKSPACE_REJECTED if refusing else None

        arm = TaskArm(log, screens={_key(joints): PoseVerdict.GUARD_REFUSED}, refuse=refuse_after_the_drop)
        ran = run(["part"], arm=arm, hooks=hooks)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertNotIn("task.return_started", ran.names())
        self.assertEqual("motion_refused", ran.hooks.of("task.put_back")[0]["outcome"])

    def test_a_line_out_refused_after_the_release_counts_the_part_placed_and_still_returns(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        hooks = RecordingHooks()
        placing: list[int] = []

        def refuse_the_line_out(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if hooks.events and hooks.events[-1][0] == "task.place_started":
                placing.append(index)
            return MotionStatus.SELF_COLLISION_REJECTED if len(placing) == 3 and index == placing[-1] else None

        arm = TaskArm(log, refuse=refuse_the_line_out)
        ran = run(["part"], arm=arm, hooks=hooks)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertFalse(ran.report.holding)
        self.assertEqual({"outcome": "motion_refused", "no_sensor": True, "line_out_refused": True},
                         ran.hooks.of("task.placed")[0])
        self.assertEqual(("home",), motions(log)[-1])

    def test_a_line_out_a_stopped_controller_refused_after_the_release_sends_no_return(self) -> None:
        """The part is placed, then a protective stop refuses the line out: the return is tried only on a controller
        that can move, and a stopped one is asked first and gets nothing. Red before: the move home was sent to it."""
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        hooks = RecordingHooks()
        placing: list[int] = []

        def stop_on_the_line_out(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if any(name == "task.place_started" for name, _ in hooks.events):
                placing.append(index)
                if len(placing) == 3:
                    arm._statuses = [PROTECTIVE]  # noqa: SLF001 (a protective stop met the line out)
                    return MotionStatus.CONTROLLER_REJECTED
            return None

        arm = TaskArm(log, refuse=stop_on_the_line_out)
        ran = run(["part"], arm=arm, hooks=hooks)

        self.assertIs(TaskStop.CONTROLLER_STOPPED, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertFalse(ran.report.holding)
        self.assertIs(True, ran.hooks.of("task.placed")[0]["line_out_refused"])
        self.assertEqual([], motions(ran.after("task.placed")), "a motion was sent to a stopped controller")
        self.assertNotIn("task.return_started", ran.names())

    def test_a_put_back_whose_release_needs_a_person_stays_where_it_stands(self) -> None:
        """A put back whose gripper raised, or whose release the gripper does not confirm: the part may still be in the
        jaws, so the arm stays where it stands and the hand needs a person."""
        from src.robot.execution.task import TaskStop

        for name, hand, outcome in (("raised", MeasuringHand(raise_on_open=True), "gripper_fault"),
                                    ("still held", MeasuringHand(sticks=True), "release_not_confirmed")):
            with self.subTest(hand=name):
                log: list[Any] = []
                joints = TaskArm([]).nearest_configuration(_raised(GRASP_Z_MM))
                arm = TaskArm(log, screens={_key(joints): PoseVerdict.GUARD_REFUSED})
                ran = run(["part"], arm=arm, jaws=hand)

                self.assertIs(TaskStop.HAND_NEEDS_PERSON, ran.report.stop, ran.report.sentence)
                self.assertTrue(ran.report.holding)
                self.assertEqual(outcome, ran.hooks.of("task.put_back")[0]["outcome"])
                self.assertEqual([], motions(ran.after("task.put_back")))
                self.assertNotIn("task.return_started", ran.names())

    def test_the_place_released_rule(self) -> None:
        from src.robot.execution.handling import HandlingOutcome, HandlingReport, HandlingVerb
        from src.robot.execution.task import place_released

        pose = PLACE_TCP
        for outcome, poses, released in (
                (HandlingOutcome.EXECUTED, 3, True),
                (HandlingOutcome.MOTION_REFUSED, 3, True),
                (HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, 3, True),
                (HandlingOutcome.MOTION_REFUSED, 2, False),
                (HandlingOutcome.MOTION_REFUSED, 1, False),
                (HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, 2, False),
                (HandlingOutcome.REFUSED, 0, False),
                (HandlingOutcome.RELEASE_NOT_CONFIRMED, 2, False),
                (HandlingOutcome.GRIPPER_FAULT, 2, False)):
            with self.subTest(outcome=outcome, poses=poses):
                report = HandlingReport(verb=HandlingVerb.PLACE, outcome=outcome, poses=(pose,) * poses)
                self.assertIs(released, place_released(report))


class TheScreenBeforeAnyMotionTests(unittest.TestCase):
    def test_every_taught_pose_is_screened_before_the_first_motion(self) -> None:
        ran = run(["part"], return_to="park")

        screened = ran.hooks.of("task.pose_screened")
        self.assertEqual([("drop_left", "place"), ("park", "return")], [(s["pose"], s["role"]) for s in screened])
        self.assertTrue(all(s["verdict"] == "clear" for s in screened))
        first_motion = next(index for index, entry in enumerate(ran.log) if entry in motions(ran.log))
        marks = [index for index, entry in enumerate(ran.log) if entry == ("event", "task.pose_screened")]
        self.assertLess(max(marks), first_motion)

    def test_a_place_pose_screened_in_error_ends_the_task_before_anything_moves(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        probe = TaskArm([])
        raised = probe.nearest_configuration(PLACE_TCP)
        arm = TaskArm(log, screens={_key(raised): PoseVerdict.PLANNER_REFUSED})
        ran = run(["part"], arm=arm)

        self.assertIs(TaskStop.POSE_REFUSED, ran.report.stop)
        self.assertEqual([], motions(log))
        self.assertEqual([], ran.service.calls)
        self.assertEqual("planner_refused", ran.hooks.of("task.pose_screened")[0]["verdict"])
        self.assertIsNotNone(ran.hooks.of("task.pose_screened")[0]["nearby_deg"])

    def test_a_return_pose_screened_in_error_ends_the_task_before_anything_moves(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, screens={_key(PARK_JOINTS): PoseVerdict.GUARD_REFUSED})
        ran = run(["part"], arm=arm, return_to="park")

        self.assertIs(TaskStop.POSE_REFUSED, ran.report.stop)
        self.assertEqual([], motions(log))

    def test_the_place_is_screened_raised_by_the_worst_hang_the_cell_declares(self) -> None:
        log: list[Any] = []
        arm = _declaring(log, 75.0)
        run(["part"], arm=arm)

        self.assertAlmostEqual(float(PLACE_TCP.position_mm[2]) + 75.0, float(arm.nearest_asked[0].position_mm[2]))

    def test_an_arm_that_screens_nothing_says_so_and_the_task_runs(self) -> None:
        from src.robot.execution.task import TaskStop

        class _Plain(TaskArm):
            screen_configuration = None  # type: ignore[assignment]
            nearest_configuration = None  # type: ignore[assignment]

        log: list[Any] = []
        ran = run(["part"], arm=_Plain(log))

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        screened = ran.hooks.of("task.pose_screened")[0]
        self.assertEqual("unscreened", screened["verdict"])
        self.assertIn("screens", screened["detail"])


class ARestartTests(unittest.TestCase):
    def test_a_restarts_first_motion_is_the_planned_move_to_the_return_pose(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part"], first_motion="return")

        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        self.assertEqual(("home",), motions(ran.log)[0], "the first motion was not the move home")
        self.assertEqual(["task.pose_screened", "task.return_started", "task.returned", "task.part_started"],
                         ran.names()[:4])

    def test_a_restart_whose_move_home_is_refused_ends_where_it_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, refuse=lambda kind, index, target: MotionStatus.SELF_COLLISION_REJECTED
                      if kind == "home" else None)
        ran = run(["part"], arm=arm, first_motion="return")

        self.assertIs(TaskStop.RETURN_FAILED, ran.report.stop)
        self.assertEqual([], ran.service.calls)
        self.assertEqual([], motions(log))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
