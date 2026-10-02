"""An "until empty" task that sets its parts down at a taught pose does not pick them back up (the owner's Q4 A).

Until empty means until nothing matching is left to pick, and the parts a task has set down still match. So a task
with a pose place and the scope ``until_empty`` keeps a circle of 150 mm about its drop's XY out of its picks, for every
label, from its setup until it ends (``TaskPlan.pose_keep_out_mm``; the drop is where the taught pose stands, read off
the arm's forward kinematics). A look that sees only parts in that circle is an empty look (``only_excluded``), not a
failed pick, so the task ends ``nothing_left`` at home after two of them, never ``failed_in_a_row`` where the arm
stands. A ``once`` task keeps nothing out: it places one part. The circle is forgotten when the task ends, whatever
ended it, so the next task picks from there again.

The pick loop runs for real in the last tests: a fixed camera over the bench sees one part standing in the drop's
circle, and the loop's own gate keeps it out.

The circle is laid only where the arm's forward kinematics are a kinematic model (``has_native_fk``): the console's
rehearsal arm (``DummyRobotArm``) answers ``fk`` with where it stands, so on it no circle says where the drop is, and
none is laid (the rehearsal scene shows the same part every look, which the part limit ends).
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from tests._task_fakes import PLACE_TCP, RecordingHooks, TaskArm, motions, run


class TheDropsCircleTests(unittest.TestCase):
    def test_an_until_empty_pose_task_keeps_its_drop_out_of_every_pick(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part", "part", "excluded", "excluded"], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.parts_placed)
        self.assertEqual(4, len(ran.service.zones_seen))
        for regions in ran.service.zones_seen:
            self.assertEqual(1, len(regions), "the drop's circle was not kept out of every pick")
            (circle,) = regions
            self.assertEqual(150.0, circle.radius_mm)
            np.testing.assert_allclose(PLACE_TCP.position_mm[:2], circle.centre_xy_mm)
            self.assertIsNone(circle.label)
        self.assertEqual((), ran.service.campaign.zones.regions(), "the circle outlived the task")
        self.assertEqual([True, True], [event["only_excluded"] for event in ran.hooks.of("task.nothing_found")])
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_once_pose_task_keeps_nothing_out(self) -> None:
        ran = run(["part"])
        self.assertEqual([()], ran.service.zones_seen)

    def test_the_circle_is_forgotten_whatever_ends_the_task(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part", "fault"], scope="until_empty")

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop)
        self.assertEqual((), ran.service.campaign.zones.regions())

    def test_the_circle_is_the_plans_to_size(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan

        plan = TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"), scope="until_empty", pose_keep_out_mm=90.0)
        ran = run(["excluded", "excluded"], plan=plan)

        self.assertEqual(90.0, ran.service.zones_seen[0][0].radius_mm)


class ThePickLoopKeepsTheDropOutTests(unittest.TestCase):
    def test_a_part_standing_in_the_drops_circle_is_an_empty_look_and_the_task_ends_nothing_left(self) -> None:
        """Red before: the pick loop asked the zones only for a label a next_target zone was made for; the drop's
        circle kept nothing out, and the task picked its own parts back up from the drop."""
        from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
        from src.robot.execution.task import PlaceAt, TaskPlan, TaskStop, run_task
        from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
        from tests._task_fakes import POSES
        from tests._wrist_views import Box
        from tests.test_a_task_keeps_its_bin_out_of_its_picks import _BenchCamera, _looking_down, _Policy, _TopGrasp

        x, y = float(PLACE_TCP.position_mm[0]), float(PLACE_TCP.position_mm[1])
        placed = Box((x - 20.0, y - 20.0, 0.0), (x + 20.0, y + 20.0, 40.0), "part")
        camera = _looking_down(x, y)
        log: list[Any] = []
        arm = TaskArm(log)
        policy = _Policy(arm)
        resolver = StaticCameraToBaseResolver(transform=Transform.from_matrix(camera, from_frame=Frame.CAMERA,
                                                                             to_frame=Frame.BASE))
        service = AutonomousGraspService.from_components(
            arm=arm, calculator=_TopGrasp(), perception=_BenchCamera((placed,), camera=camera),  # type: ignore[arg-type]
            mode=GraspMode.EASY, policy=policy, frame_resolver=resolver, max_attempts=1)  # type: ignore[arg-type]

        report = run_task(service, TaskPlan(object="", place=PlaceAt(pose="drop_left"), scope="until_empty"),
                          hooks=RecordingHooks(), poses=POSES)

        self.assertIs(TaskStop.NOTHING_LEFT, report.stop, report.sentence)
        self.assertEqual([], policy.executed, "the task picked its own part back from the drop")
        self.assertEqual(2, report.picks)
        self.assertEqual(("home",), motions(log)[-1])


class TheRehearsalCellTests(unittest.TestCase):
    """The console's rehearsal cell, ``console_dummy``, end to end through the library: its dummy arm, its scene of one
    part that never empties, the real pick service."""

    def _run(self, scope: str) -> tuple[Any, RecordingHooks]:
        from src.config import load_tree
        from src.robot.core import JointPositions
        from src.robot.execution.autonomous_grasp import build_rehearsal_cell
        from src.robot.execution.lifecycle import connect_cell, disconnect_cell
        from src.robot.execution.task import PlaceAt, TaskPlan, run_task

        service = build_rehearsal_cell(load_tree("console_dummy").robot)
        orchestrator = service.runtime.orchestrator
        connect_cell(orchestrator.arm, orchestrator.gripper)
        hooks = RecordingHooks()
        try:
            report = run_task(service, TaskPlan(object="", place=PlaceAt(pose="drop"), scope=scope, max_parts=3),
                              hooks=hooks, poses={"drop": JointPositions.deg(-60.0, -95.0, -120.0, -55.0, 90.0, 0.0)})
        finally:
            disconnect_cell(orchestrator.arm, orchestrator.gripper, service)
        return report, hooks

    def test_until_empty_on_the_rehearsal_places_its_parts_until_the_limit(self) -> None:
        """Red before: the dummy's FK answered where the arm stood, home, right over the rehearsal's part, so the drop's
        circle kept that part out of every pick and the task ended ``nothing_left`` with nothing placed."""
        from src.robot.execution.task import TaskStop

        report, hooks = self._run("until_empty")

        self.assertIs(TaskStop.PART_LIMIT, report.stop, report.sentence)
        self.assertEqual(3, report.parts_placed)
        self.assertEqual([], hooks.of("task.nothing_found"))

    def test_once_on_the_rehearsal_places_one_part(self) -> None:
        from src.robot.execution.task import TaskStop

        report, _hooks = self._run("once")

        self.assertIs(TaskStop.FINISHED, report.stop, report.sentence)
        self.assertEqual(1, report.parts_placed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
