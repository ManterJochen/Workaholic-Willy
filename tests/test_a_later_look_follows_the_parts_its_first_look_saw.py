"""A later look of a pick finds the parts its first look saw by their projected boxes, and ranks the kept part first.

Map2's F and D (``follow_looks``, a task's following of its parts, the owner, 2026-10-09). The pick's first look
grounds; a look after it that the first look's grasp sends it on to segments the parts that look saw, each its surface
projected at the later look's stamped pose and padded, with SAM2 alone, every mask held to its part's footprint and
15 mm (the extent guard), all or nothing: a mask that reaches past its part grounds the look as before. And a later look
ranks the part the pick keeps first, the others only where it has no grasp. Off, every look grounds and ranks every part.

Real: the pick service and its pick loop, the camera source and the kept scene; stand-ins as in
``tests/test_a_task_picks_the_grey_parts_and_ends_without_a_long_search``, with the arm's later looks standing the tool
60 mm beside home and 40 mm lower, so the camera moved and the parts did not.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingSupportConfig
from src.geometry import Frame, Pose, Transform
from src.robot.grippers.jaw_io import JawIOGripper
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from tests._task_fakes import TaskArm, _key
from tests.test_a_task_picks_the_grey_parts_and_ends_without_a_long_search import (
    LOOKS,
    _MOUNT,
    _Calculator,
    _Grounder,
    _GrippingIO,
    _Mat,
    _Part,
    _Segmenter,
)

#: Where the later looks stand the tool: beside home and lower, the camera still over the parts.
_LATER = {_key(look): Pose.tool_down(60.0, -480.0, 360.0, label=f"look {index}")
          for index, look in enumerate(LOOKS[1:], start=1)}


def _cubes() -> list[_Part]:
    return [_Part("grey 1", "grey", (-120.0, -540.0)), _Part("grey 2", "grey", (-20.0, -440.0)),
            _Part("grey 3", "grey", (90.0, -560.0))]


class _Seeing(_Calculator):
    """Grasps each part as the scene's calculator does, the grasp uncertain on the first frame the camera took (its
    first look): the pick goes on to its next look and keeps the part. Writes down which part it computed per frame."""

    def __init__(self, mat: _Mat) -> None:
        super().__init__(mat)
        self.frames = 0
        self.by_frame: list[tuple[int, str]] = []

    def compute_result(self, seg: Any, depth: Any, *args: Any, **keywords: Any) -> GraspResult:
        result = super().compute_result(seg, depth, *args, **keywords)
        part = self.mat.part_of(seg.mask)
        self.by_frame.append((self.mat.grabs, "" if part is None else part.name))
        if self.mat.grabs == 1 and result.is_success:
            return GraspResult(candidates=result.candidates, top_score=result.top_score,
                               reasons=(GraspFailureReason.RESCAN_RECOMMENDED,))
        return result


class _CountingMat(_Mat):
    def __init__(self, arm: TaskArm, parts: list[_Part]) -> None:
        super().__init__(arm, parts)
        self.grabs = 0

    def grab(self) -> Any:
        self.grabs += 1
        return super().grab()


class _LeakingSegmenter(_Segmenter):
    """Cuts each box's part; a box of a kept label (a projected one) is cut 40 px wider, onto the bench."""

    def segment_detection(self, bgr: Any, det: Any) -> Any:
        import cv2

        seg = super().segment_detection(bgr, det)
        if det.label != "grey cube":
            return seg
        wide = cv2.dilate(np.asarray(seg.mask).astype(np.uint8), np.ones((41, 41), np.uint8))
        return type(seg)(mask=wide, label=seg.label)


def _pick(*, follow_looks: bool, segmenter: type = _Segmenter) -> "tuple[Any, list[Any], list[Any], _Seeing]":
    from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
    from src.robot.perception import RealSenseVisionPerceptionSource

    log: list[Any] = []
    arm = TaskArm(log, fk_table=_LATER)
    mat = _CountingMat(arm, _cubes())
    io = _GrippingIO(log, arm, mat)
    jaws = JawIOGripper(io, actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=lambda _question: "open", sleep=lambda _s: None)
    jaws.connect()
    io.closed = False
    del log[:]
    calculator = _Seeing(mat)
    source = RealSenseVisionPerceptionSource(streamer=mat, detector=_Grounder(mat, log), segmenter=segmenter(mat),
                                             prompt="each separate grey cube", object_labels=("grey cube",),
                                             warmup_grabs=0)
    policy = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99,  # type: ignore[arg-type]
                                  require_steady_before_motion=True)
    resolver = EyeInHandFrameResolver(t_cam_to_tool=Transform.from_matrix(_MOUNT, from_frame=Frame.CAMERA,
                                                                          to_frame=Frame.TOOL))
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=calculator, perception=source, mode=GraspMode.EASY,  # type: ignore[arg-type]
        gripper=jaws, policy=policy, frame_resolver=resolver, max_attempts=1)
    service.runtime.orchestrator.support_config = GraspingSupportConfig(height_mm=0.0)
    service.runtime.orchestrator.target_label = "grey cube"
    events: list[Any] = []
    service.attach_progress_listener(events.append)
    report = service.pick(look=LOOKS, follow_looks=follow_looks)
    return report, log, [event for event in events if str(event.stage) == "perceived"], calculator


def _groundings(log: list[Any]) -> int:
    return sum(1 for entry in log if isinstance(entry, tuple) and entry[:1] == ("ground",))


class ALaterLookFollowsTheFirstLook_sPartsTests(unittest.TestCase):
    def test_a_later_look_segments_the_kept_part_without_the_detector(self) -> None:
        report, log, perceived, _ = _pick(follow_looks=True)

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual(2, len(report.looks), "home, then the first later look, whose grasp is safe")
        self.assertEqual(2, len(perceived))
        self.assertEqual(1, _groundings(log), "the detector grounds the first look alone")
        self.assertEqual([None, "followed"], [event.route for event in perceived])
        self.assertIn("projected boxes", perceived[1].route_reason)
        self.assertEqual([report.looks[1]], report.telemetry["looks_followed"])
        self.assertNotIn("followed", report.telemetry, "the pick was handed no parts to follow at its first look")

    def test_a_projected_mask_reaching_past_the_part_grounds_the_look(self) -> None:
        report, log, perceived, _ = _pick(follow_looks=True, segmenter=_LeakingSegmenter)

        self.assertEqual(2, _groundings(log), "the later look is grounded as before")
        self.assertEqual("grounded", perceived[1].route)
        self.assertIn("past the part's footprint", perceived[1].route_reason)
        self.assertNotIn("looks_followed", report.telemetry)

    def test_with_following_off_every_look_is_grounded_as_before(self) -> None:
        report, log, perceived, _ = _pick(follow_looks=False)

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual(2, _groundings(log))
        self.assertEqual([None, None], [event.route for event in perceived])
        self.assertNotIn("looks_followed", report.telemetry)
        self.assertNotIn("followed", report.telemetry)


class ALaterLookRanksTheKeptPartFirstTests(unittest.TestCase):
    def test_later_looks_rank_only_the_kept_part_while_it_has_a_grasp(self) -> None:
        _, _, _, calculator = _pick(follow_looks=True)

        first = [name for frame, name in calculator.by_frame if frame == 1]
        later = [name for frame, name in calculator.by_frame if frame == 2]
        self.assertEqual({"grey 1", "grey 2", "grey 3"}, set(first), "the first look ranks every part")
        self.assertEqual(["grey 1"], later, "the later look ranks the part it keeps, and no other")

    def test_with_following_off_a_later_look_ranks_every_part(self) -> None:
        _, _, _, calculator = _pick(follow_looks=False)

        later = [name for frame, name in calculator.by_frame if frame == 2]
        self.assertEqual({"grey 1", "grey 2", "grey 3"}, set(later))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
