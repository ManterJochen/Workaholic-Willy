"""A sort's label gate passes every rule's kind, and the pick names the kind it went for and the parts no rule claims.

The owner's sort (2026-10-09): "Grüne Teile in die gelbe Kiste, rote in die blaue". The pick loop takes any of the
sort's kinds (``BinPickingOrchestrator.target_labels``), each compared exactly, wherever it asks whether a part may be
a target; what no rule clearly claims (the detector's ``ambiguous``, a word no rule names) is no target and stays a
neighbour, and "stays where it lies and is named at the end". So every report says:

* ``target_label``: the label of the segmentation the pick went for, the one its executed grasp belongs to, else the
  one of the last grasp it chose (a failed pick names it too), ``""`` where it chose none;
* ``unclaimed_labels``: the labels the gate turned away at the pick's first look, one per segmentation, in the camera's
  order; a wide flat surface (the mat the detector boxed) and a part standing in a region the task keeps out (its bin,
  its drop) are left out, and neither a rescan nor a later look of a wrist pick adds any.

One kind's gate and no gate at all read as they always did: the existing tests pin them
(``tests/test_pick_loop_target_label_gate.py``), and the controls here say it once more beside the sort.

The pick loop runs for real on a fixed camera over the bench, ray cast (``tests/_wrist_views.render``), and on the
looking arm's wrist camera for its looks, with a calculator that grasps a part at its top and records what it was
asked to grasp, and a policy that records what it was asked to execute.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from src.robot.core import JointPositions
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.loop.progress import PickProgress, PickStage
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion, ExclusionZones
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame
from tests._wrist_views import (
    CUBE_CENTRE,
    CUBE_HIGH,
    CUBE_LOW,
    K,
    Box,
    LookingArm,
    WristCamera,
    camera_looking_at,
    camera_to_tool,
    render,
    tool_for,
)

KINDS = ("green part", "red part")


def _looking_down(x_mm: float, y_mm: float, height_mm: float = 700.0) -> np.ndarray:
    """CAMERA to BASE of a camera at ``height_mm`` over (x, y) looking straight down; its image x along base +x."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = (x_mm, y_mm, height_mm)
    return matrix


CAMERA = _looking_down(0.0, -700.0)

#: The mat the parts lie on, 600 x 300 mm and 2 mm thick, which the detector boxed and named by no rule.
MAT = Box((-300.0, -850.0, 0.0), (300.0, -550.0, 2.0), "ambiguous")
#: Five 40 mm parts in a row on it: one of each kind, two no rule clearly claims, and one of a word no rule names.
GREEN = Box((-180.0, -720.0, 0.0), (-140.0, -680.0, 40.0), "green part")
FIRST_DOUBT = Box((-100.0, -720.0, 0.0), (-60.0, -680.0, 40.0), "ambiguous")
RED = Box((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0), "red part")
ORANGE = Box((60.0, -720.0, 0.0), (100.0, -680.0, 40.0), "orange part")
SECOND_DOUBT = Box((140.0, -720.0, 0.0), (180.0, -680.0, 40.0), "ambiguous")
BENCH = (MAT, GREEN, FIRST_DOUBT, RED, ORANGE, SECOND_DOUBT)


def _centre(box: Box) -> tuple[float, float, float]:
    return tuple((lo + hi) / 2.0 for lo, hi in zip(box.low, box.high))  # type: ignore[return-value]


@dataclass(eq=False)
class _Segment:
    mask: np.ndarray
    label: str
    box: Box
    score: float = 0.9


@dataclass(eq=False)
class _BenchCamera:
    """A fixed camera over the bench: every frame ray cast, one segmentation per box it sees, labelled as the box is;
    ``renamed`` renames a label in one frame, by the frame's number (0 for the first)."""

    boxes: tuple[Box, ...] = BENCH
    renamed: dict[int, dict[str, str]] = field(default_factory=dict)
    frames: int = 0

    def acquire(self) -> PerceptionFrame:
        number, self.frames = self.frames, self.frames + 1
        depth, hit = render(CAMERA, self.boxes)
        names = self.renamed.get(number, {})
        segments = tuple(_Segment(mask=(hit == index), label=names.get(box.label, box.label), box=box)
                         for index, box in enumerate(self.boxes) if np.any(hit == index))
        return PerceptionFrame(depth_map=depth, intrinsics=K.copy(), segmentations=segments,
                               rgb=np.zeros((*depth.shape, 3), dtype=np.uint8), timestamp=float(number))


class _TopGrasp:
    """Grasps the part a segmentation shows at its top, straight down, and keeps the label of every part it was asked
    about, in order; the red part ranks higher, so a pick that may take either kind takes the red one. ``refuse``
    answers no candidate for every part, ``rescan_first`` a rescan for every part of the first frame."""

    render_debug_images = False

    def __init__(self, camera: _BenchCamera, *, refuse: bool = False, rescan_first: bool = False) -> None:
        self.camera = camera
        self.refuse = refuse
        self.rescan_first = rescan_first
        self.asked: list[str] = []

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        self.asked.append(seg.label)
        if self.refuse:
            return GraspResult(reasons=(GraspFailureReason.NO_VALID_GRASP,))
        if self.rescan_first and self.camera.frames == 1:
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,
                                        GraspFailureReason.RESCAN_RECOMMENDED))
        x, y, _ = _centre(seg.box)
        score = 0.9 if seg.label == "red part" else 0.6
        grasp = GraspPoint(position=np.array([x, y, 20.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=score, frame=GraspFrame.BASE,
                           label=seg.label)
        return GraspResult(candidates=(grasp,), top_score=score)


class _Policy:
    """Executes nothing and answers ``outcome``: what the pick asked for is what the test reads."""

    def __init__(self, outcome: PolicyOutcome = PolicyOutcome.EXECUTED) -> None:
        self.outcome = outcome
        self.executed: list[GraspPoint] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=self.outcome)


@dataclass(eq=False)
class _Bench:
    """The pick loop over the bench camera, its calculator and policy, and every event it emitted."""

    orchestrator: BinPickingOrchestrator
    camera: _BenchCamera
    calculator: _TopGrasp
    policy: _Policy
    events: list[PickProgress]

    def run(self) -> Any:
        return self.orchestrator.run()

    def stage(self, stage: PickStage) -> list[PickProgress]:
        return [event for event in self.events if event.stage == stage]


def _bench(*, boxes: tuple[Box, ...] = BENCH, target_labels: tuple[str, ...] = KINDS, target_label: str | None = None,
           outcome: PolicyOutcome = PolicyOutcome.EXECUTED, refuse: bool = False, rescan_first: bool = False,
           renamed: dict[int, dict[str, str]] | None = None, zones: ExclusionZones | None = None,
           max_attempts: int = 1) -> _Bench:
    arm = DummyRobotArm()
    arm.connect()
    camera = _BenchCamera(boxes=boxes, renamed=renamed or {})
    calculator = _TopGrasp(camera, refuse=refuse, rescan_first=rescan_first)
    policy = _Policy(outcome)
    events: list[PickProgress] = []
    resolver = StaticCameraToBaseResolver(transform=Transform.from_matrix(CAMERA, from_frame=Frame.CAMERA,
                                                                         to_frame=Frame.BASE))
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=calculator, perception=camera, policy=policy,  # type: ignore[arg-type]
        frame_resolver=resolver, max_attempts=max_attempts, target_label=target_label, target_labels=target_labels,
        exclusion_zones=zones, on_progress=events.append,
    )
    return _Bench(orchestrator, camera, calculator, policy, events)


def _region_over(*boxes: Box) -> ExclusionZones:
    """A task's zones keeping out a square of 80 mm about each box, every label, as a bin's footprint is kept out."""
    zones = ExclusionZones()
    zones.start_pick()
    for box in boxes:
        x, y, _ = _centre(box)
        zones.keep_out_region(ExclusionRegion.rectangle((x, y), (80.0, 80.0), reason=f"the bin over {box.label}"))
    return zones


class TheGateTakesEveryKindTests(unittest.TestCase):
    def test_both_kinds_are_targets_and_nothing_else_is(self) -> None:
        bench = _bench()
        report = bench.run()
        self.assertEqual(["green part", "red part"], bench.calculator.asked,
                         "the mat, the parts no rule claims and the orange part were never asked to be grasped")
        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(["red part"], [grasp.label for grasp in bench.policy.executed])

    def test_an_executed_pick_names_the_kind_it_took(self) -> None:
        self.assertEqual("red part", _bench().run().target_label)

    def test_a_failed_pick_names_the_kind_its_grasp_was_on(self) -> None:
        """The jaws closed on nothing: the grasp the pick chose is the red part's, whatever came of it."""
        report = _bench(outcome=PolicyOutcome.OBJECT_NOT_DETECTED).run()
        self.assertIs(PickOutcome.OBJECT_NOT_DETECTED, report.outcome)
        self.assertEqual("red part", report.target_label)

    def test_a_pick_that_chose_no_grasp_names_no_kind(self) -> None:
        bench = _bench(refuse=True)
        report = bench.run()
        self.assertEqual([], bench.policy.executed)
        self.assertEqual("", report.target_label)
        self.assertEqual(("ambiguous", "orange part", "ambiguous"), report.unclaimed_labels,
                         "what no rule claims is named whatever came of the kinds")

    def test_the_targets_each_look_saw_are_both_kinds(self) -> None:
        bench = _bench()
        bench.run()
        self.assertEqual(2, bench.orchestrator._targets_in_frame)

    def test_the_parts_held_out_of_the_supports_are_the_kinds(self) -> None:
        """A surface most of whose pixels lie in the box of a part the pick may go for is that part's own top, no
        support: a sort's kinds are held out as one kind's label is, and a part no rule claims is not."""
        bench = _bench()
        frame = bench.camera.acquire()
        depth = np.asarray(frame.depth_map, dtype=np.float64)
        held = bench.orchestrator._parts_held_out_of_the_supports(frame, depth, K, CAMERA, 15.0)
        self.assertEqual(["part_1", "part_3"], [box.name for box in held], "the green and the red part, by index")
        every = _bench(target_labels=()).orchestrator._parts_held_out_of_the_supports(frame, depth, K, CAMERA, 15.0)
        self.assertEqual(len(BENCH), len(every), "with no label every segmentation is held out, as before")


class WhatNoRuleClaimsTests(unittest.TestCase):
    def test_each_part_no_rule_claims_is_named_once_and_the_mat_is_no_part(self) -> None:
        report = _bench().run()
        self.assertEqual(("ambiguous", "orange part", "ambiguous"), report.unclaimed_labels,
                         "two doubted parts are two entries, in the camera's order, and the mat is left out")

    def test_a_part_standing_where_the_task_keeps_out_is_not_unclaimed(self) -> None:
        """A part in a bin of the task, or the bin itself, is no part left lying for want of a rule."""
        report = _bench(zones=_region_over(SECOND_DOUBT)).run()
        self.assertEqual(("ambiguous", "orange part"), report.unclaimed_labels)

    def test_a_rescan_adds_nothing(self) -> None:
        """Only the pick's first look counts: the frame rescanned after it named the orange part "purple part"."""
        bench = _bench(rescan_first=True, renamed={1: {"orange part": "purple part"}}, max_attempts=2)
        report = bench.run()
        self.assertEqual(["rescan", "executed"], [attempt.action for attempt in report.attempts])
        self.assertEqual(("ambiguous", "orange part", "ambiguous"), report.unclaimed_labels)

    def test_a_frame_with_no_kind_names_both_kinds_and_what_it_saw(self) -> None:
        bench = _bench(boxes=(FIRST_DOUBT, ORANGE))
        report = bench.run()
        self.assertEqual([], bench.calculator.asked, "nothing past the gate, nothing computed")
        self.assertIn(GraspFailureReason.TARGET_LABEL_NOT_FOUND, report.attempts[-1].reasons)
        [missed] = bench.stage(PickStage.NO_CANDIDATE)
        self.assertEqual({"target_label": None, "target_labels": list(KINDS),
                          "labels_seen": ["ambiguous", "orange part"]}, dict(missed.extra))
        self.assertEqual(("ambiguous", "orange part"), report.unclaimed_labels)

    def test_parts_the_task_keeps_out_are_said_kind_by_kind(self) -> None:
        bench = _bench(boxes=(GREEN, FIRST_DOUBT, RED), zones=_region_over(GREEN, RED))
        report = bench.run()
        self.assertEqual([], bench.policy.executed)
        excluded = report.attempts[-1].excluded or ""
        self.assertIn("Every 'green part' the camera sees (1)", excluded)
        self.assertIn("Every 'red part' the camera sees (1)", excluded)
        self.assertTrue(report.attempts[-1].excluded_by_regions)
        self.assertEqual(("ambiguous",), report.unclaimed_labels)

    def test_a_region_that_names_a_kind_keeps_it_out_of_a_blocker_too(self) -> None:
        """A blocker standing in a region is never taken away: a sort asks the zones with each of its kinds, as one
        kind's pick asks with its label; a task's own regions name none and keep every kind out."""
        zones = ExclusionZones()
        zones.start_pick()
        zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 50.0, label="red part"))
        zones.keep_out_region(ExclusionRegion.circle((500.0, 0.0), 50.0))
        sort = _bench(zones=zones).orchestrator
        one_kind = _bench(zones=zones, target_labels=(), target_label="green part").orchestrator
        no_label = _bench(zones=zones, target_labels=()).orchestrator
        self.assertTrue(sort._kept_out_by_a_region(zones, (0.0, 0.0)))
        self.assertFalse(one_kind._kept_out_by_a_region(zones, (0.0, 0.0)))
        self.assertFalse(no_label._kept_out_by_a_region(zones, (0.0, 0.0)))
        for orchestrator in (sort, one_kind, no_label):
            self.assertTrue(orchestrator._kept_out_by_a_region(zones, (500.0, 0.0)))


class OneKindAndNoLabelAsBeforeTests(unittest.TestCase):
    def test_one_kind_takes_its_label_alone_and_names_nothing_unclaimed(self) -> None:
        bench = _bench(target_labels=(), target_label="green part")
        report = bench.run()
        self.assertEqual(["green part"], bench.calculator.asked)
        self.assertEqual("green part", report.target_label)
        self.assertEqual((), report.unclaimed_labels)

    def test_one_kind_not_found_says_what_it_always_said(self) -> None:
        bench = _bench(target_labels=(), target_label="blue part", boxes=(GREEN, RED))
        bench.run()
        [missed] = bench.stage(PickStage.NO_CANDIDATE)
        self.assertEqual({"target_label": "blue part", "labels_seen": ["green part", "red part"]}, dict(missed.extra))

    def test_no_label_takes_every_part_but_the_mat(self) -> None:
        bench = _bench(target_labels=())
        report = bench.run()
        self.assertEqual(["green part", "ambiguous", "red part", "orange part", "ambiguous"], bench.calculator.asked)
        self.assertEqual("red part", report.target_label)
        self.assertEqual((), report.unclaimed_labels)


# ---------------------------------------------------------------------------------------------------------------------
# A wrist pick: its looks
# ---------------------------------------------------------------------------------------------------------------------

#: Two looks at the green part, from its +x side and from its -x side, 45 degrees up.
LOOK_FIRST = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_SECOND = JointPositions.deg(180.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The part every look is about, of the green rule, and a doubted part 20 mm beside it.
WRIST_GREEN = Box(CUBE_LOW, CUBE_HIGH, "green part")
BESIDE = Box((40.0, -720.0, 0.0), (80.0, -680.0, 40.0), "ambiguous")


class _LookGrasp:
    """Grasps the green part on the frames named, at its middle, straight down; everything else gets no candidate with
    a rescan reason, as the real calculator reports it."""

    render_debug_images = False

    def __init__(self, camera: WristCamera, grasps_on: tuple[int, ...]) -> None:
        self.camera = camera
        self.grasps_on = grasps_on

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        number = len(self.camera.taken) - 1
        if seg.label == "green part" and number in self.grasps_on:
            grasp = GraspPoint(position=np.array(CUBE_CENTRE), approach=np.array([0.0, 0.0, -1.0]),
                               axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                               label=seg.label)
            return GraspResult(candidates=(grasp,), top_score=0.9)
        return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED, GraspFailureReason.RESCAN_RECOMMENDED))


def _wrist(*, grasps_on: tuple[int, ...] = (1,)) -> tuple[BinPickingOrchestrator, _Policy]:
    """A wrist pick handed two looks; the second look's frame names the doubted part "orange part"."""
    arm = LookingArm({LOOK_FIRST: tool_for(camera_looking_at(bearing_deg=0.0)),
                      LOOK_SECOND: tool_for(camera_looking_at(bearing_deg=180.0))})
    camera = WristCamera(arm, boxes=(WRIST_GREEN, BESIDE), labels={1: {"ambiguous": "orange part"}})
    policy = _Policy()
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=_LookGrasp(camera, grasps_on), perception=camera, policy=policy,  # type: ignore[arg-type]
        frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()), max_attempts=1,
        primary_camera_id="wrist", looks=(LOOK_FIRST, LOOK_SECOND), target_labels=KINDS,
    )
    return orchestrator, policy


class AWristPickCountsItsFirstLookTests(unittest.TestCase):
    def test_a_later_look_adds_nothing(self) -> None:
        orchestrator, policy = _wrist()
        report = orchestrator.run()
        looked = orchestrator.looked_around
        assert looked is not None
        self.assertEqual(2, len(looked.visited), "the first look found no grasp, so the second was driven")
        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(["green part"], [grasp.label for grasp in policy.executed])
        self.assertEqual("green part", report.target_label)
        self.assertEqual(("ambiguous",), report.unclaimed_labels, "the second look's 'orange part' is not counted")

    def test_looks_handed_on_to_the_pick_keep_what_their_first_look_said(self) -> None:
        """The decision path: the looks are driven by look_around(), handed on with go_on_with(), and run() goes on
        with them rather than looking again."""
        orchestrator, _ = _wrist()
        looked = orchestrator.look_around()
        orchestrator.go_on_with(looked)
        report = orchestrator.run()
        self.assertEqual(("ambiguous",), report.unclaimed_labels)
        self.assertEqual("green part", report.target_label)

    def test_the_parts_a_later_look_finds_again_are_the_kinds_alone(self) -> None:
        """What the later looks of a pick that follows its parts find by their projected boxes (follow_looks): one
        kind's pick keeps its label's parts, a sort its kinds'; a pick with no label keeps every part."""
        orchestrator, _ = _wrist(grasps_on=(0,))
        orchestrator.look_around()
        try:
            kept = orchestrator._kept_by_the_first_look()
            assert kept is not None
            self.assertEqual(["green part"], [part.label for part in kept.parts])
            orchestrator.target_labels = ()
            every = orchestrator._kept_by_the_first_look()
            assert every is not None
            self.assertEqual(["green part", "ambiguous"], [part.label for part in every.parts])
        finally:
            orchestrator.close_looks()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
