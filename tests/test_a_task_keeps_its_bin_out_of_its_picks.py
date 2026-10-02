"""A task keeps out of its picks the area it places into, for as long as the task runs (the owner's Q4, 2026-09-30).

A task that places into a bin the camera found must never pick the bin, or a part it already put in it; a task that
places at a taught pose "until empty" must not pick back the parts it set down there. So a task lays a keep-out region
on the campaign's exclusion zones (``ExclusionZones.keep_out_region``): the bin's footprint for both scopes, a circle of
150 mm about a pose place's drop for "until empty". Unlike the zones ``next_target`` makes, a region:

* applies to every label where it names none, the empty label of an unlabelled detector included;
* lasts until the task forgets it (``forget_regions``), whatever number of picks starts in between;
* is a turned rectangle or a circle about BASE Z, a column over the table.

The pick loop asks the zones whether anything applies to a segmentation's label (``applies``) before it reads where
the segmentation stands, so a region keeps out an unlabelled part too: before this the gate returned at once for a
label no zone was made for, and an empty label was never asked about at all. A frame that sees only kept-out parts ends
its attempt with the zones' sentence (``PickAttempt.excluded``), which the service reports as ``only_excluded``. Only
where every one of them stands in a region of the task (``PickAttempt.excluded_by_regions``, the service's
``only_kept_out``) does a task count it as an empty look: a part a zone skips after a failed pick still stands there to
be picked, and a look that saw it is a failed pick.

The pick loop runs for real here on a fixed camera over the bench, ray cast (``tests/_wrist_views.render``), with a
calculator that grasps any part at its top and a policy that records what it was asked to execute.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, AutonomousGraspService, GraspMode
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion, ExclusionZones
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import PerceptionFrame
from tests._wrist_views import K, Box, render


def _looking_down(x_mm: float, y_mm: float, height_mm: float = 700.0) -> np.ndarray:
    """CAMERA to BASE of a camera at ``height_mm`` over (x, y) looking straight down; its image x along base +x."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
    matrix[:3, 3] = (x_mm, y_mm, height_mm)
    return matrix


#: Two 40 mm parts on the bench, 160 mm apart, and the camera between them.
LEFT = Box((-100.0, -720.0, 0.0), (-60.0, -680.0, 40.0), "part")
RIGHT = Box((60.0, -720.0, 0.0), (100.0, -680.0, 40.0), "part")
CAMERA = _looking_down(0.0, -700.0)


def _centre(box: Box) -> tuple[float, float, float]:
    return tuple((lo + hi) / 2.0 for lo, hi in zip(box.low, box.high))  # type: ignore[return-value]


@dataclass(eq=False)
class _Segment:
    mask: np.ndarray
    label: str
    box: Box
    score: float = 0.9


class _BenchCamera:
    """A fixed camera over the bench: every frame ray cast, one segmentation per part it sees, labelled as asked."""

    def __init__(self, boxes: tuple[Box, ...], *, label: str | None = None, camera: np.ndarray = CAMERA) -> None:
        self.boxes = boxes
        self.label = label
        self.camera = camera

    def acquire(self) -> PerceptionFrame:
        depth, hit = render(self.camera, self.boxes)
        segments = tuple(
            _Segment(mask=(hit == index), label=box.label if self.label is None else self.label, box=box)
            for index, box in enumerate(self.boxes) if np.any(hit == index))
        return PerceptionFrame(depth_map=depth, intrinsics=K.copy(), segmentations=segments,
                               rgb=np.zeros((*depth.shape, 3), dtype=np.uint8), timestamp=1.0)


class _TopGrasp:
    """Grasps the part a segmentation shows at its top, straight down, closing along base x; the right part ranks
    higher, so a pick that may take either takes the right one."""

    render_debug_images = False

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        x, y, _ = _centre(seg.box)
        score = 0.9 if seg.box is RIGHT else 0.6
        grasp = GraspPoint(position=np.array([x, y, 20.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=score, frame=GraspFrame.BASE,
                           label=seg.label)
        return GraspResult(candidates=(grasp,), top_score=score)


class _Policy:
    """Executes nothing and says it executed: what the pick asked for is what the test reads."""

    def __init__(self, arm: Any) -> None:
        self.arm = arm
        self.gripper = None
        self.executed: list[GraspPoint] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _service(boxes: tuple[Box, ...] = (LEFT, RIGHT), *, label: str | None = None) -> tuple[Any, _Policy]:
    from src.robot.drivers.dummy.arm import DummyRobotArm

    arm = DummyRobotArm()
    arm.connect()
    policy = _Policy(arm)
    resolver = StaticCameraToBaseResolver(transform=Transform.from_matrix(CAMERA, from_frame=Frame.CAMERA,
                                                                         to_frame=Frame.BASE))
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=_TopGrasp(), perception=_BenchCamera(boxes, label=label),  # type: ignore[arg-type]
        mode=GraspMode.EASY, policy=policy, frame_resolver=resolver, max_attempts=1,  # type: ignore[arg-type]
    )
    return service, policy


# ---------------------------------------------------------------------------------------------------------------------
# A region of its own
# ---------------------------------------------------------------------------------------------------------------------


class ARegionTests(unittest.TestCase):
    def test_a_turned_rectangle_holds_what_stands_inside_it_and_nothing_beyond(self) -> None:
        region = ExclusionRegion.rectangle((400.0, -300.0), (300.0, 200.0), yaw_rad=math.radians(30.0))
        along = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
        across = np.array([-along[1], along[0]])
        centre = np.array([400.0, -300.0])
        for inside in (centre, centre + along * 145.0, centre + across * 95.0, centre + along * 140 + across * 90):
            with self.subTest(inside=tuple(inside)):
                self.assertTrue(region.contains((*inside, 50.0)))
        for outside in (centre + along * 155.0, centre + across * 105.0, centre + np.array([150.0, 100.0]) * 1.2):
            with self.subTest(outside=tuple(outside)):
                self.assertFalse(region.contains((*outside, 50.0)))
        # A column over the table: height does not matter.
        self.assertTrue(region.contains((400.0, -300.0, 900.0)))

    def test_a_circle_holds_what_stands_within_its_radius(self) -> None:
        region = ExclusionRegion.circle((300.0, -400.0), 150.0)
        self.assertTrue(region.contains((300.0 + 149.0, -400.0, 0.0)))
        self.assertFalse(region.contains((300.0 + 151.0, -400.0, 0.0)))

    def test_a_region_naming_no_label_applies_to_every_label_the_empty_one_included(self) -> None:
        region = ExclusionRegion.circle((0.0, 0.0), 50.0)
        for label in ("green cube", "Bolt ", ""):
            with self.subTest(label=label):
                self.assertTrue(region.applies(label))
        named = ExclusionRegion.circle((0.0, 0.0), 50.0, label="Bolt ")
        self.assertTrue(named.applies("bolt"))
        self.assertFalse(named.applies("nut"))
        self.assertFalse(named.applies(""))

    def test_a_region_that_holds_nothing_or_names_no_shape_is_refused(self) -> None:
        for build in (lambda: ExclusionRegion.circle((0.0, 0.0), 0.0),
                      lambda: ExclusionRegion.circle((0.0, math.nan), 50.0),
                      lambda: ExclusionRegion.rectangle((0.0, 0.0), (0.0, 10.0)),
                      lambda: ExclusionRegion((0.0, 0.0)),
                      lambda: ExclusionRegion((0.0, 0.0), radius_mm=5.0, size_mm=(5.0, 5.0))):
            with self.subTest(), self.assertRaises(ValueError):
                build()

    def test_it_says_what_it_keeps_out_and_why(self) -> None:
        region = ExclusionRegion.rectangle((400.0, -300.0), (300.0, 200.0), reason="the blue bin it places into")
        said = region.render()
        self.assertEqual(said, str(region))
        self.assertIn("the blue bin it places into", said)
        self.assertIn("300", said)
        as_data = region.to_dict()
        self.assertEqual("rectangle", as_data["shape"])
        self.assertIsNone(as_data["label"])
        self.assertEqual([400.0, -300.0], as_data["centre_xy_mm"])


class TheZonesKeepRegionsTests(unittest.TestCase):
    def test_a_region_lasts_every_pick_until_it_is_forgotten_and_the_zones_go_on_as_before(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 100.0, reason="the drop"))
        zones.exclude(label="bolt", centre_mm=(500.0, 0.0, 0.0))
        for _ in range(10):
            zones.start_pick()
        self.assertTrue(zones.excludes(label="bolt", centre_mm=(10.0, 0.0, 0.0)), "a region expired with the picks")
        self.assertFalse(zones.excludes(label="bolt", centre_mm=(500.0, 0.0, 0.0)), "a zone outlived its picks")
        self.assertEqual(1, zones.forget_regions())
        self.assertFalse(zones.excludes(label="bolt", centre_mm=(10.0, 0.0, 0.0)))
        self.assertEqual((), zones.regions())

    def test_applies_says_whether_anything_would_keep_a_label_out(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        self.assertFalse(zones.applies("bolt"))
        self.assertFalse(zones.applies(""))
        zones.exclude(label="bolt", centre_mm=(0.0, 0.0, 0.0))
        self.assertTrue(zones.applies("Bolt"))
        self.assertFalse(zones.applies("nut"))
        self.assertFalse(zones.applies(""), "a next_target zone names its label; the empty label is none")
        zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 10.0))
        self.assertTrue(zones.applies("nut"))
        self.assertTrue(zones.applies(""))

    def test_remaining_and_only_excluded_read_regions_for_any_label(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zones.keep_out_region(ExclusionRegion.rectangle((400.0, -300.0), (300.0, 200.0)))
        seen = [(400.0, -300.0, 30.0), (450.0, -280.0, 30.0), (100.0, -650.0, 20.0)]
        for label in ("part", ""):
            with self.subTest(label=label):
                self.assertTrue(zones.excludes(label=label, centre_mm=seen[0]))
                self.assertEqual((2,), zones.remaining(label=label, centres_mm=seen))
                self.assertFalse(zones.only_excluded_remain(label=label, centres_mm=seen))
                self.assertTrue(zones.only_excluded_remain(label=label, centres_mm=seen[:2]))

    def test_a_region_says_it_keeps_a_part_out_and_a_zone_does_not(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zones.exclude(label="bolt", centre_mm=(500.0, 0.0, 0.0))
        zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 100.0))
        zones.keep_out_region(ExclusionRegion.circle((-500.0, 0.0), 100.0, label="nut"))

        self.assertTrue(zones.kept_out_by_a_region(label="bolt", centre_mm=(10.0, 0.0, 0.0)))
        self.assertTrue(zones.kept_out_by_a_region(label="", centre_mm=(10.0, 0.0, 0.0)))
        self.assertTrue(zones.excludes(label="bolt", centre_mm=(500.0, 0.0, 0.0)))
        self.assertFalse(zones.kept_out_by_a_region(label="bolt", centre_mm=(500.0, 0.0, 0.0)), "a zone is no region")
        self.assertFalse(zones.kept_out_by_a_region(label="bolt", centre_mm=(-500.0, 0.0, 0.0)))
        self.assertTrue(zones.kept_out_by_a_region(label="Nut", centre_mm=(-500.0, 0.0, 0.0)))

    def test_the_sentence_says_the_area_the_task_keeps_out_and_keeps_the_zones_words_without_one(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zones.exclude(label="bolt", centre_mm=(0.0, 0.0, 0.0))
        before = zones.only_excluded_sentence(label="bolt", count=2)
        self.assertIn("failed in one of the last 3 picks", before)
        zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 50.0, reason="the blue bin it places into"))
        for label, what in (("bolt", "'bolt'"), ("", "part")):
            with self.subTest(label=label):
                said = zones.only_excluded_sentence(label=label, count=2)
                self.assertIn(what, said)
                self.assertIn("2", said)
                self.assertIn("the blue bin it places into", said)
                self.assertTrue(said.endswith("."))


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop's gate
# ---------------------------------------------------------------------------------------------------------------------


class ThePickLoopKeepsRegionsOutTests(unittest.TestCase):
    def _keep_out(self, service: Any, box: Box) -> None:
        x, y, _ = _centre(box)
        service.start_campaign()
        service.campaign.zones.keep_out_region(ExclusionRegion.rectangle((x, y), (80.0, 80.0), reason="the bin"))

    def test_a_part_inside_the_region_is_skipped_and_the_one_beside_it_picked(self) -> None:
        service, policy = _service()
        self._keep_out(service, RIGHT)

        report = service.pick()

        self.assertTrue(report.succeeded, report.failure_summary())
        self.assertEqual(1, len(policy.executed))
        np.testing.assert_allclose(_centre(LEFT)[:2], policy.executed[0].position[:2])

    def test_a_frame_that_sees_only_kept_out_parts_ends_with_the_zones_sentence_and_picks_nothing(self) -> None:
        from src.robot.execution.autonomous_grasp.service import found_nothing, only_excluded, only_kept_out

        service, policy = _service((RIGHT,))
        self._keep_out(service, RIGHT)

        report = service.pick()

        self.assertEqual([], policy.executed, "a part the task keeps out was picked")
        self.assertIs(AutonomousGraspOutcome.NO_VALID_GRASP, report.outcome)
        self.assertTrue(only_excluded(report))
        self.assertTrue(only_kept_out(report))
        self.assertFalse(found_nothing(report), "a look that saw a kept-out part did see something")
        self.assertIn("the bin", report.failure_summary())

    def test_a_part_a_zone_skips_after_a_failed_pick_is_excluded_but_no_part_the_task_keeps_out(self) -> None:
        """What ``next_target`` lays over a part a pick failed on: the pick stops with the zones' sentence, and the part
        still stands there to be picked, so it is no empty look. Red before: the service's word for an empty look read
        any exclusion, and a task ended ``nothing_left`` with the part on the bench."""
        from src.robot.execution.autonomous_grasp.service import only_excluded, only_kept_out

        service, policy = _service((RIGHT,))
        service.start_campaign()
        service.campaign.zones.exclude(label="part", centre_mm=_centre(RIGHT))

        report = service.pick()

        self.assertEqual([], policy.executed)
        self.assertTrue(only_excluded(report))
        self.assertFalse(only_kept_out(report))
        self.assertIs(False, report.pick_report.attempts[-1].excluded_by_regions)

    def test_only_a_look_whose_every_excluded_part_stands_in_a_region_is_kept_out_by_the_task(self) -> None:
        from src.robot.execution.autonomous_grasp.service import only_excluded, only_kept_out

        for seen, zoned, regioned, kept_out in (((LEFT, RIGHT), (), (LEFT, RIGHT), True),
                                                ((RIGHT,), (RIGHT,), (RIGHT,), True),
                                                ((LEFT, RIGHT), (LEFT,), (RIGHT,), False)):
            with self.subTest(zoned=[box.low for box in zoned], regioned=[box.low for box in regioned]):
                service, policy = _service(seen)
                service.start_campaign()
                for box in zoned:
                    service.campaign.zones.exclude(label="part", centre_mm=_centre(box))
                for box in regioned:
                    x, y, _ = _centre(box)
                    service.campaign.zones.keep_out_region(ExclusionRegion.rectangle((x, y), (80.0, 80.0)))

                report = service.pick()

                self.assertEqual([], policy.executed)
                self.assertTrue(only_excluded(report))
                self.assertIs(kept_out, only_kept_out(report))

    def test_an_unlabelled_part_is_kept_out_too(self) -> None:
        """Red before: the gate returned for a label no zone was made for, so the rehearsal cell's unlabelled parts
        were never kept out, and an until-empty task picked its own parts back from the drop."""
        from src.robot.execution.autonomous_grasp.service import only_excluded

        service, policy = _service((RIGHT,), label="")
        self._keep_out(service, RIGHT)

        report = service.pick()

        self.assertEqual([], policy.executed)
        self.assertTrue(only_excluded(report))

    def test_with_no_region_the_pick_is_the_one_it_was(self) -> None:
        from src.robot.execution.autonomous_grasp.service import only_excluded

        service, policy = _service()
        service.start_campaign()

        report = service.pick()

        self.assertTrue(report.succeeded)
        np.testing.assert_allclose(_centre(RIGHT)[:2], policy.executed[0].position[:2])
        self.assertFalse(only_excluded(report))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
