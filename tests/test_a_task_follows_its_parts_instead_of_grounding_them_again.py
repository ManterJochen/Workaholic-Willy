"""A task follows its parts instead of grounding them again (``robot.grasping.follow_parts``, the owner, 2026-10-09).

"Qwen on the first pick and on every trigger, SAM2 on the known parts between." The owner's scene on the real chain:
three grey cubes and two red ones, every grey cube onto the drop until none is left, from four wrist looks. With
following on, the first pick grounds every part; the next pick's first look finds the parts the last one kept with SAM2
on their boxes and no detector; and the check look after the last part is always asked of the detector. A push, a
blocker, a recovery, a try that sent motion and failed, a gripped part that was not kept, ``refresh_every_picks`` and a
Restart ground again. DO0 still changes twice per part.

Real: the task, the pick service with its pick loop and grasp policy, the camera source with its kept scene, label
mapping and colour check, over a ``TwoStageBackend``; the owner's toggle on tool DO0. Stand-ins as in
``tests/test_a_task_picks_the_grey_parts_and_ends_without_a_long_search``: the arm, the ray-cast wrist camera, a
grounder that boxes every part it is shown, a segmenter that cuts each box's part, a calculator that grasps a part
straight down at its middle.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingFollowPartsConfig, GraspingSupportConfig
from src.geometry import Frame, Transform
from src.robot.grippers.jaw_io import JawIOGripper
from tests._task_fakes import POSES, RecordingHooks, TaskArm, do0_changes
from tests.test_a_task_picks_the_grey_parts_and_ends_without_a_long_search import (
    LOOKS,
    _MOUNT,
    _Calculator,
    _Grounder,
    _GrippingIO,
    _Mat,
    _owner_s_scene,
    _Part,
    _Segmenter,
)


class _CountingSegmenter(_Segmenter):
    """Cuts each box's part, and writes down every box it was handed with its label."""

    def __init__(self, mat: _Mat, log: list[Any]) -> None:
        super().__init__(mat)
        self.log = log

    def segment_detection(self, bgr: Any, det: Any) -> Any:
        self.log.append(("cut", det.label))
        return super().segment_detection(bgr, det)


@dataclass
class _Ran:
    report: Any
    log: list[Any]
    mat: _Mat
    hooks: RecordingHooks
    events: list[Any]
    calls: list[dict[str, Any]]
    service: Any

    def routes(self) -> list[Any]:
        """The route of every frame perceived, in order: ``followed``, ``grounded`` (a follow that failed) or ``None``."""
        return [event.route for event in self.events if str(event.stage) == "perceived"]

    def groundings(self) -> int:
        return sum(1 for entry in self.log if isinstance(entry, tuple) and entry[:1] == ("ground",))

    def picks_grounded(self) -> list[int]:
        """The task's picks (1 on) whose first look the detector grounded."""
        return [pick for _, pick, report in self.hooks.picks if not report.telemetry.get("followed")]


def _service(arm: TaskArm, mat: _Mat, log: list[Any], *, fails: "Callable[[], bool] | None" = None) -> Any:
    """The owner's cell on the real chain, its calculator finding no grasp while ``fails`` says so."""
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
    from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
    from src.robot.perception import RealSenseVisionPerceptionSource

    io = _GrippingIO(log, arm, mat)
    jaws = JawIOGripper(io, actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=lambda _question: "open", sleep=lambda _s: None)
    jaws.connect()
    io.closed = False
    del log[:]

    class _Calculating(_Calculator):
        def compute_result(self, seg: Any, depth: Any, *args: Any, **keywords: Any) -> GraspResult:
            if fails is not None and fails():
                return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,))
            return super().compute_result(seg, depth, *args, **keywords)

    source = RealSenseVisionPerceptionSource(streamer=mat, detector=_Grounder(mat, log),
                                             segmenter=_CountingSegmenter(mat, log), prompt="object", warmup_grabs=0)
    policy = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99,  # type: ignore[arg-type]
                                  require_steady_before_motion=True)
    resolver = EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(_MOUNT, from_frame=Frame.CAMERA,
                                                                          to_frame=Frame.TOOL))
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=_Calculating(mat), perception=source, mode=GraspMode.EASY,  # type: ignore[arg-type]
        gripper=jaws, policy=policy, frame_resolver=resolver, max_attempts=1)
    # The bench the parts stand on, as the cell declares its mat: what a part stands over.
    service.runtime.orchestrator.support_config = GraspingSupportConfig(height_mm=0.0)
    service.configured_looks = LOOKS
    return service


def _arm(log: list[Any], follow: "GraspingFollowPartsConfig | None") -> TaskArm:
    arm = TaskArm(log)
    if follow is not None:
        arm.config = SimpleNamespace(grasping=SimpleNamespace(follow_parts=follow), workspace_limits=None)
    return arm


def _task(parts: list[_Part], *, follow: "GraspingFollowPartsConfig | None" = GraspingFollowPartsConfig(enabled=True),
          scope: str = "until_empty", fails: "Callable[[int], bool] | None" = None,
          after_pick: "Callable[[int, Any, Any], Any] | None" = None, **options: Any) -> _Ran:
    """Every grey cube onto the drop, on the real chain; ``fails(pick)`` makes the calculator find no grasp in that pick,
    and ``after_pick(pick, report, service)`` may change what a pick reports to the task."""
    from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, run_task

    log: list[Any] = []
    arm = _arm(log, follow)
    mat = _Mat(arm, parts)
    calls: list[dict[str, Any]] = []
    service = _service(arm, mat, log, fails=(lambda: fails(len(calls))) if fails is not None else None)
    events: list[Any] = []
    service.attach_progress_listener(events.append)
    picked = service.pick

    def pick(**keywords: Any) -> Any:
        calls.append(dict(keywords))
        report = picked(**keywords)
        return after_pick(len(calls), report, service) if after_pick is not None else report

    service.pick = pick  # type: ignore[method-assign]
    hooks = RecordingHooks(on_event=lambda name, _data: log.append(("event", name)))
    report = run_task(service, TaskPlan(object="grey cube", place=PlaceAt(pose="drop_left"), scope=scope,  # type: ignore[arg-type]
                                        options=TaskOptions(**options)), hooks=hooks, poses=POSES)
    return _Ran(report=report, log=log, mat=mat, hooks=hooks, events=events, calls=calls, service=service)


def _grey_cubes(count: int) -> list[_Part]:
    """``count`` grey cubes in the camera's view from home, 100 mm and more apart."""
    spots = [(-200.0, -560.0), (-50.0, -440.0), (100.0, -600.0), (-200.0, -440.0), (100.0, -440.0)]
    return [_Part(f"grey {index + 1}", "grey", spot) for index, spot in enumerate(spots[:count])]


class ATaskGroundsOnceAndFollowsItsPartsTests(unittest.TestCase):
    def test_the_owner_s_scene_grounds_the_first_pick_and_the_check_and_follows_the_rest(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task(_owner_s_scene())

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.parts_placed)
        self.assertEqual(["grey", "grey", "grey"], [part.colour for part in ran.mat.gripped])
        self.assertEqual([None, "followed", "followed", None], ran.routes())
        self.assertEqual(2, ran.groundings(), "the first pick and the check look: Qwen asked twice, not four times")
        self.assertEqual([1, 4], ran.picks_grounded())
        self.assertEqual(6, do0_changes(ran.log), "DO0 changes twice per part: three closes, three releases")
        # The red cubes stayed where they stood, never gripped.
        self.assertEqual(["red 1", "red 2"], [part.name for part in ran.mat.parts])

    def test_a_followed_pick_cuts_the_kept_parts_boxes_under_their_labels(self) -> None:
        ran = _task(_owner_s_scene())

        cut = [entry[1] for entry in ran.log if isinstance(entry, tuple) and entry[:1] == ("cut",)]
        # Grounded: five boxes under the prompt's words, then two and one kept boxes, then the check's two red ones.
        self.assertEqual(["grey cube"] * 3, [label for label in cut if label == "grey cube"])

    def test_every_pick_is_handed_the_follow_and_the_check_look_nothing_to_follow(self) -> None:
        ran = _task(_owner_s_scene())

        self.assertEqual([True] * 4, [call.get("follow_looks") for call in ran.calls])
        self.assertEqual([False, True, True, False], ["follow" in call for call in ran.calls])

    def test_the_reports_say_whether_the_first_look_followed(self) -> None:
        ran = _task(_owner_s_scene())

        telemetry = [report.telemetry for _, _, report in ran.hooks.picks]
        self.assertEqual([None, True, True, None], [entry.get("followed") for entry in telemetry])
        self.assertIn("nothing new in depth", telemetry[1]["follow_why"])
        self.assertEqual(["home"], telemetry[1]["looks_followed"])


class WithFollowingOffNothingChangesTests(unittest.TestCase):
    def _events(self, ran: _Ran) -> list[Any]:
        return [(str(event.stage), event.segmentation_count, event.route, event.route_reason) for event in ran.events]

    def test_following_off_hands_no_pick_anything_and_grounds_every_pick(self) -> None:
        off = _task(_owner_s_scene(), follow=GraspingFollowPartsConfig(enabled=False))
        today = _task(_owner_s_scene(), follow=None)

        self.assertEqual([{"look": LOOKS}] * 4, [{key: call[key] for key in ("look",)} for call in off.calls])
        self.assertTrue(all("follow" not in call and "follow_looks" not in call for call in off.calls))
        self.assertEqual(4, off.groundings())
        self.assertEqual(self._events(today), self._events(off))
        timeless = [[(name, {key: value for key, value in data.items() if key != "duration_s"})
                     for name, data in ran.hooks.events] for ran in (today, off)]
        self.assertEqual(timeless[0], timeless[1])
        self.assertEqual(do0_changes(today.log), do0_changes(off.log))


class ATriggerGroundsTheNextPickTests(unittest.TestCase):
    def _with(self, change: Callable[[Any, Any], Any], at: int = 2) -> _Ran:
        """The owner's scene, pick ``at``'s report changed by ``change(report, service)`` before the task reads it."""
        def after(pick: int, report: Any, service: Any) -> Any:
            return change(report, service) if pick == at else report

        return _task(_owner_s_scene(), after_pick=after)

    def test_a_push_grounds_the_next_pick(self) -> None:
        ran = self._with(lambda report, _: replace(report, telemetry={**report.telemetry, "pushes": [{"code": "pushed"}]}))

        self.assertEqual([1, 3, 4], ran.picks_grounded())

    def test_a_blocker_cleared_grounds_the_next_pick(self) -> None:
        def blocker(report: Any, _: Any) -> Any:
            attempts = (SimpleNamespace(push=None, blocker="set_aside", tries=()),)
            return replace(report, pick_report=replace(report.pick_report, attempts=attempts))

        self.assertEqual([1, 3, 4], self._with(blocker).picks_grounded())

    def test_a_recovery_grounds_the_next_pick(self) -> None:
        ran = self._with(lambda report, _: replace(report, recovery_actions=({"action": "next_target"},)))

        self.assertEqual([1, 3, 4], ran.picks_grounded())

    def test_a_try_that_sent_motion_and_failed_grounds_the_next_pick(self) -> None:
        def sent_and_failed(report: Any, _: Any) -> Any:
            tries = ({"rank": 0, "outcome": "execution_failed", "sent": True},
                     {"rank": 1, "outcome": "executed", "sent": True})
            attempts = (SimpleNamespace(push=None, blocker=None, tries=tries),)
            return replace(report, pick_report=replace(report.pick_report, attempts=attempts))

        self.assertEqual([1, 3, 4], self._with(sent_and_failed).picks_grounded())

    def test_a_gripped_part_that_was_not_kept_grounds_the_next_pick(self) -> None:
        def elsewhere(report: Any, service: Any) -> Any:
            loop = service.runtime.orchestrator
            judged = loop.looked_around.judged
            moved = np.asarray(judged.target_cloud_base_mm) + np.array([60.0, 0.0, 0.0])
            loop.looked_around = replace(loop.looked_around, judged=replace(judged, target_cloud_base_mm=moved))
            return report

        self.assertEqual([1, 3, 4], self._with(elsewhere).picks_grounded())

    def test_refresh_every_picks_2_grounds_at_picks_1_3_and_5(self) -> None:
        ran = _task(_grey_cubes(5), follow=GraspingFollowPartsConfig(enabled=True, refresh_every_picks=2))

        self.assertEqual(5, ran.report.parts_placed)
        self.assertEqual([1, 3, 5, 6], ran.picks_grounded(), "and the check look after the last part")
        self.assertEqual(10, do0_changes(ran.log))

    def test_without_the_check_look_the_pick_after_the_last_part_grounds_too(self) -> None:
        ran = _task(_grey_cubes(3), check_look=False)

        self.assertEqual(3, ran.report.parts_placed)
        self.assertEqual(1, ran.picks_grounded()[0])
        self.assertEqual(4, ran.picks_grounded()[1], "the pick after the last part kept asks the detector")


class AFailedPickThatSentNothingKeepsTheSceneTests(unittest.TestCase):
    def test_the_pick_after_it_follows_every_part(self) -> None:
        # Pick 2 finds no grasp at its one look (multi-view off): nothing sent, the scene as it was.
        ran = _task(_grey_cubes(3), fails=lambda pick: pick == 2, multi_view=False)

        self.assertEqual(3, ran.report.parts_placed)
        outcomes = [(pick, report.succeeded, report.telemetry.get("followed")) for _, pick, report in ran.hooks.picks]
        self.assertEqual((2, False, True), outcomes[1])
        self.assertEqual((3, True, True), outcomes[2], "the pick after the failed one follows")
        self.assertEqual(6, do0_changes(ran.log), "the failed pick closed nothing")


class ARestartStartsWithNothingKeptTests(unittest.TestCase):
    def test_a_restart_grounds_its_first_pick(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan, run_task

        log: list[Any] = []
        arm = _arm(log, GraspingFollowPartsConfig(enabled=True))
        mat = _Mat(arm, _grey_cubes(3))
        service = _service(arm, mat, log)
        plan = TaskPlan(object="grey cube", place=PlaceAt(pose="drop_left"), scope="once")
        first = run_task(service, plan, hooks=RecordingHooks(), poses=POSES)
        hooks = RecordingHooks()

        again = run_task(service, replace(plan, first_motion="return"), hooks=hooks, poses=POSES)

        self.assertEqual((1, 1), (first.parts_placed, again.parts_placed))
        [(_, _, report)] = hooks.picks
        self.assertIsNone(report.telemetry.get("followed"), "the Restart's pick was handed nothing to follow")


class WhyAPickGroundsAgainTests(unittest.TestCase):
    """``task._why_grounded_again``: the owner's triggers, read off a pick's report."""

    def _report(self, *, succeeded: bool = True, tries: tuple[dict[str, Any], ...] = (), **attempt: Any) -> Any:
        row = SimpleNamespace(**{"push": None, "blocker": None, "tries": tries, **attempt})
        return SimpleNamespace(pick_report=SimpleNamespace(attempts=(row,), stands_at=""), telemetry={},
                               recovery_actions=(), succeeded=succeeded)

    def test_a_clean_pick_grounds_nothing(self) -> None:
        from src.robot.execution.task import _why_grounded_again

        self.assertEqual("", _why_grounded_again(self._report(tries=({"sent": True},))))
        self.assertEqual("", _why_grounded_again(self._report(succeeded=False, tries=({"sent": False},))))

    def test_each_trigger_is_said(self) -> None:
        from src.robot.execution.task import _why_grounded_again

        self.assertIn("pushed", _why_grounded_again(self._report(push="pushed")))
        self.assertIn("blocker", _why_grounded_again(self._report(blocker="set_aside")))
        self.assertIn("sent motion", _why_grounded_again(self._report(succeeded=False, tries=({"sent": True},))))
        report = self._report()
        report.pick_report.stands_at = "standoff of grasp 2"
        self.assertIn("standoff of grasp 2", _why_grounded_again(report))
        report = self._report()
        report.recovery_actions = ({"action": "rescan"},)
        self.assertIn("recovery", _why_grounded_again(report))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
