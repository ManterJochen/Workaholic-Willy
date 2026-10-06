"""A task whose detector fails says so: it ends ``detector_failed``, never "nothing left" or "target not found".

Two halves built apart meet here. The task (library L1) reads the cell's failure count before and after each look
(``service.detector_failures()``) and calls a look that found nothing while the count rose a failed detector: a problem
stop, where the arm stands. The count is the perception backend's (library L12): ``TwoStageBackend`` still turns a
model's exception into "no objects", so a pick never dies with a traceback, and now counts it. Before both landed
together, a detector that raised on every look read as an empty table: an "until empty" task ended ``nothing_left`` and
drove home, and a camera place ended ``target_not_found``.

The chain is real from the task down: the pick service with its pick loop and grasp policy, its camera source
(``RealSenseVisionPerceptionSource``) over a ``TwoStageBackend``, the locator the task builds from the service
(``locators_for_service``), and the owner's toggle on tool DO0. Only the camera's frames, the arm and the detector's
model are stand-ins. Beside each case, the same cell with a detector that honestly finds nothing ends as an empty cell
does: the count, not the empty answer, tells the two apart.
"""

from __future__ import annotations

import unittest
from typing import Any

from tests._task_fakes import POSES, RecordingHooks, TaskArm, do0_changes, motions, toggle


class _Detector:
    """The detector's model: raises on every look, or honestly finds nothing."""

    def __init__(self, *, raises: bool) -> None:
        self.raises = raises
        self.asked: list[str] = []

    def detect_all(self, bgr: Any, prompt: str) -> list[Any]:
        self.asked.append(prompt)
        if self.raises:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return []


def _cell(detector: _Detector) -> tuple[Any, list[Any]]:
    """The pick service on a fixed camera whose source perceives through a real two-stage backend."""
    from src.models.perception_backend import TwoStageBackend
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.perception import RealSenseVisionPerceptionSource
    from tests.test_a_stopped_controller_moves_no_jaws import _Calculator
    from tests.test_the_service_lends_its_detector_to_a_locator import _Owner, _Segmenter

    log: list[Any] = []
    arm = TaskArm(log)
    jaws = toggle(log, arm=arm)
    source = RealSenseVisionPerceptionSource(streamer=_Owner("overhead").handle(),
                                             backend=TwoStageBackend(detector, _Segmenter()), prompt="object",
                                             warmup_grabs=0)
    policy = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99,  # type: ignore[arg-type]
                                  require_steady_before_motion=True)
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=_Calculator(), perception=source, mode=GraspMode.EASY,  # type: ignore[arg-type]
        gripper=jaws, policy=policy, max_attempts=1)
    return service, log


def _run(detector: _Detector, place: Any, scope: str) -> tuple[Any, list[Any], Any]:
    from src.robot.execution.task import TaskPlan, run_task

    service, log = _cell(detector)
    report = run_task(service, TaskPlan(object="cube", place=place, scope=scope), hooks=RecordingHooks(), poses=POSES)
    return report, log, service


class AnEmptyPickTests(unittest.TestCase):
    def test_a_detector_that_raises_ends_detector_failed_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskStop

        detector = _Detector(raises=True)
        report, log, service = _run(detector, PlaceAt(pose="drop_left"), "until_empty")

        self.assertIs(TaskStop.DETECTOR_FAILED, report.stop, report.sentence)
        self.assertEqual("problem", str(report.stop.stop_class))
        self.assertIn("detector failed", report.sentence)
        # Every cube in a box of its own (``task.EACH_SEPARATE``, 2026-10-06), asked once.
        self.assertEqual(["each separate cube"], detector.asked, "one failed look ends the task; it does not look again")
        self.assertEqual(1, service.detector_failures())
        self.assertEqual([], motions(log), "the arm moved: a problem stop leaves it where it stands")
        self.assertEqual(0, do0_changes(log))
        self.assertFalse(report.at_return)

    def test_a_detector_that_finds_nothing_ends_nothing_left_at_home(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskStop

        detector = _Detector(raises=False)
        report, log, service = _run(detector, PlaceAt(pose="drop_left"), "until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, report.stop, report.sentence)
        self.assertEqual(2, len(detector.asked), "two empty looks in a row end an until-empty task")
        self.assertEqual(0, service.detector_failures())
        self.assertEqual(("home",), motions(log)[-1])
        self.assertEqual(0, do0_changes(log))


class ASurveyTests(unittest.TestCase):
    def test_a_detector_that_raises_while_the_task_looks_for_its_bin_ends_detector_failed(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskStop

        detector = _Detector(raises=True)
        report, log, service = _run(detector, PlaceAt(camera="blue bin"), "once")

        self.assertIs(TaskStop.DETECTOR_FAILED, report.stop, report.sentence)
        self.assertIn("blue bin", report.sentence)
        self.assertIn("blue bin", detector.asked, "the survey asked the service's own detector")
        self.assertNotIn("each separate cube", detector.asked, "a pick ran before the bin was found")
        self.assertEqual([], motions(log))
        self.assertEqual(0, do0_changes(log))

    def test_a_detector_that_finds_no_bin_ends_target_not_found(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskStop

        detector = _Detector(raises=False)
        report, log, service = _run(detector, PlaceAt(camera="blue bin"), "once")

        self.assertIs(TaskStop.TARGET_NOT_FOUND, report.stop, report.sentence)
        self.assertNotIn("each separate cube", detector.asked, "a pick ran with no bin to put the part into")
        self.assertEqual(0, service.detector_failures())
        self.assertEqual(0, do0_changes(log))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
