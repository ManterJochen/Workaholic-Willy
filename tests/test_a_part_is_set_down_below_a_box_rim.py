"""A part goes into a box below its rim, not dropped over it (the owner, 2026-10-08 night).

"Bei so kleinen Kisten ... 1-5 cm unter dem Kistenrand": with ``robot.place.release_in_a_box: below_the_rim`` the jaws
open ``below_the_rim_mm`` (30 mm unless the cell says 10 to 50) under the rim, never lower than 10 mm over what the camera
reads inside the box: its floor, or the parts already in it, read again on the frame of every check before a drop
(``recheck(..., inside=True)``), one measure with the rim band, so no extra frame and no detector. Where that leaves less
than 10 mm under the rim, the box reads full and the part goes over the rim.

Where the opening is too narrow for the open hand and the part, the drop is today's, over the rim: the part as it hangs
(its cloud from the pick, turned only about the vertical) and the open hand wherever its fingertips go below the rim
must stand ``opening_margin_mm`` inside the opening on every side, checked before anything moves (the part is not
modelled in the guard); and where the guard refuses the line into the box before anything was sent, the place lets the
part go over the rim instead, from the same standoff. A grasp off vertical, a box whose inside the camera did not read,
a drop below the rim the screen refuses go over the rim too. Off, as shipped, nothing changes.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.safety.planning.band import PoseVerdict
from tests._place_fakes import cube_cloud, hande_tree
from tests._task_fakes import (
    BIN_CENTRE,
    BIN_RIM_MM,
    GRASP_Z_MM,
    PART_XY,
    BinScene,
    MeasuringLocator,
    Pick,
    TaskArm,
    bin_object,
    bin_points,
    do0_changes,
    motions,
    run,
    toggle,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")
HAND_E = None  # set in setUpModule: the repository's Hand-E, opened


def setUpModule() -> None:
    global HAND_E
    from src.robot.execution.place_target import OpenHand

    HAND_E = OpenHand.of(hande_tree())


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _kept(points: "np.ndarray | None" = None) -> Any:
    from src.robot.execution.place_target import KeptTarget

    kept = KeptTarget.of(bin_object(points), camera="wrist", look=None, seen_at=12.5)
    assert kept is not None
    return kept


def _empty_inside(kept: Any, top_mm: float = 15.0) -> Any:
    from src.robot.execution.place_target import InsideRead

    return InsideRead("box", np.zeros((0, 3)), top_mm=top_mm, said="scripted")


def _plan(kept: Any = None, *, inside: Any = "empty", cloud: "np.ndarray | None" = None, grasp: "Pose | None" = None,
          arm: "TaskArm | None" = None, below_mm: "float | None" = 30.0, hand: Any = "hande", **keywords: Any) -> Any:
    from src.robot.execution.place_target import drop_plan

    kept = kept if kept is not None else _kept()
    return drop_plan(arm if arm is not None else TaskArm([]), kept, grasp if grasp is not None else Pose.tool_down(
        PART_XY[0], PART_XY[1], GRASP_Z_MM), part_bottom_mm=0.0, air_mm=20.0, standoff_mm=80.0,
        part_cloud_mm=cube_cloud() if cloud is None else cloud, below_rim_mm=below_mm,
        inside=_empty_inside(kept) if isinstance(inside, str) else inside, opening_margin_mm=10.0,
        hand=HAND_E if hand == "hande" else hand, **keywords)


def _along_x(length_mm: float, width_mm: float = 30.0) -> np.ndarray:
    xs = np.linspace(-length_mm / 2.0, length_mm / 2.0, 60)
    ys = np.linspace(-width_mm / 2.0, width_mm / 2.0, 8)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([PART_XY[0] + gx.ravel(), PART_XY[1] + gy.ravel(), np.full(gx.size, 40.0)])


class TheDepthRuleTests(unittest.TestCase):
    def test_a_part_goes_30_mm_under_the_rim_of_a_box_whose_inside_reads_empty(self) -> None:
        plan = _plan()

        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual("below_the_rim", plan.release)
        self.assertAlmostEqual(BIN_RIM_MM - 30.0 + GRASP_Z_MM, float(plan.pose.position_mm[2]), delta=0.5)
        self.assertAlmostEqual(30.0, plan.below_rim_mm, delta=0.5)
        self.assertAlmostEqual(-30.0, plan.air_mm, delta=0.5)
        assert plan.instead is not None
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM, float(plan.instead.position_mm[2]), delta=0.5)
        np.testing.assert_allclose(plan.instead.position_mm[:2], plan.pose.position_mm[:2])
        np.testing.assert_allclose(plan.instead.quaternion_xyzw, plan.pose.quaternion_xyzw)

    def test_the_depth_is_the_cells_choice_from_10_to_50_mm(self) -> None:
        for depth in (10.0, 50.0):
            with self.subTest(depth=depth):
                plan = _plan(below_mm=depth)
                self.assertAlmostEqual(BIN_RIM_MM - depth + GRASP_Z_MM, float(plan.pose.position_mm[2]), delta=0.5)

    def test_the_part_keeps_10_mm_over_what_the_camera_reads_inside(self) -> None:
        """Parts already in the box read up to 100 mm: the part's bottom stops at 110 mm, 10 mm under the rim."""
        kept = _kept()
        plan = _plan(kept, inside=_empty_inside(kept, top_mm=100.0))

        self.assertEqual("below_the_rim", plan.release)
        self.assertAlmostEqual(10.0, plan.below_rim_mm, delta=0.5)
        self.assertAlmostEqual(110.0 + GRASP_Z_MM, float(plan.pose.position_mm[2]), delta=0.5)

    def test_a_box_that_reads_full_to_within_10_mm_of_its_rim_is_dropped_into_over_its_rim(self) -> None:
        kept = _kept()
        plan = _plan(kept, inside=_empty_inside(kept, top_mm=101.0))

        self.assertEqual("over_the_rim", plan.release)
        self.assertIsNone(plan.instead)
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM, float(plan.pose.position_mm[2]), delta=0.5)
        self.assertIn("full", plan.over_the_rim_why)

    def test_a_box_whose_inside_the_camera_did_not_read_is_dropped_into_over_its_rim(self) -> None:
        plan = _plan(inside=None)

        self.assertEqual("over_the_rim", plan.release)
        self.assertIn("did not read", plan.over_the_rim_why)

    def test_off_the_drop_is_todays_and_says_what_it_always_said(self) -> None:
        plan = _plan(below_mm=None)

        self.assertEqual("over_the_rim", plan.release)
        self.assertIsNone(plan.instead)
        self.assertEqual("", plan.over_the_rim_why)
        self.assertEqual({"kind", "pose_mm", "rim_mm", "hang_mm", "air_mm", "verdict"}, set(plan.to_event()))

    def test_a_drop_below_the_rim_says_how_far_below(self) -> None:
        said = _plan().to_event()

        self.assertEqual("below_the_rim", said["release"])
        self.assertAlmostEqual(30.0, said["below_rim_mm"], delta=0.5)
        self.assertLess(said["air_mm"], 0.0)

    def test_a_flat_top_takes_no_part_below_it(self) -> None:
        plan = _plan(_kept(bin_points(inside=False)))

        self.assertEqual("over_the_rim", plan.release)
        self.assertIsNone(plan.instead)


class TheNarrowBoxTests(unittest.TestCase):
    """Where the opening is too narrow for the open hand and the part, the drop is today's, over the rim."""

    def test_a_part_that_fits_the_opening_but_not_its_margin_goes_over_the_rim(self) -> None:
        """285 mm along a 284 mm opening: it fits as before (``_fit``), and not 10 mm clear of the walls."""
        plan = _plan(cloud=_along_x(285.0))

        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual("over_the_rim", plan.release)
        self.assertIn("the part as it hangs", plan.over_the_rim_why)

    def test_the_open_hand_reaching_below_the_rim_of_a_narrow_box_goes_over_the_rim(self) -> None:
        """A box 60 mm wide is 44 mm inside: the Hand-E's fingers, 29 mm wide, do not stand 10 mm clear of its walls
        once the fingertips go below the rim."""
        narrow = _kept(bin_points(BIN_CENTRE, (300.0, 60.0)))
        plan = _plan(narrow, cloud=cube_cloud(side_mm=20.0))

        self.assertEqual("over_the_rim", plan.release)
        self.assertIn("the open hand", plan.over_the_rim_why)

    def test_a_hand_that_stays_over_the_rim_lets_a_narrow_box_take_the_part_below_it(self) -> None:
        """Gripped 70 mm over its bottom, the part goes 30 mm under the rim and the fingertips stay 29.5 mm over it."""
        narrow = _kept(bin_points(BIN_CENTRE, (300.0, 60.0)))
        grasp = Pose.tool_down(PART_XY[0], PART_XY[1], 70.0)
        plan = _plan(narrow, cloud=cube_cloud(side_mm=20.0), grasp=grasp)

        self.assertEqual("below_the_rim", plan.release, plan.over_the_rim_why)
        self.assertAlmostEqual(BIN_RIM_MM - 30.0 + 70.0, float(plan.pose.position_mm[2]), delta=0.5)

    def test_without_the_parts_cloud_or_the_hands_size_nothing_goes_below_the_rim(self) -> None:
        from src.robot.execution.place_target import drop_plan

        kept = _kept()
        no_cloud = drop_plan(TaskArm([]), kept, Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM), part_bottom_mm=0.0,
                             air_mm=20.0, below_rim_mm=30.0, inside=_empty_inside(kept), hand=HAND_E)
        self.assertEqual("over_the_rim", no_cloud.release)
        self.assertIn("cloud", no_cloud.over_the_rim_why)
        no_hand = _plan(kept, hand=None)
        self.assertEqual("over_the_rim", no_hand.release)
        self.assertIn("hand", no_hand.over_the_rim_why)

    def test_a_grasp_off_vertical_lets_the_part_go_over_the_rim(self) -> None:
        down = Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM)
        turn = math.radians(20.0)
        tilt = np.array([[math.cos(turn), 0.0, math.sin(turn)], [0.0, 1.0, 0.0],
                         [-math.sin(turn), 0.0, math.cos(turn)]])
        matrix = np.asarray(down.to_matrix(), dtype=np.float64).copy()
        matrix[:3, :3] = tilt @ matrix[:3, :3]
        plan = _plan(grasp=Pose.from_matrix(matrix, frame=Frame.BASE))

        self.assertEqual("over_the_rim", plan.release)
        self.assertIn("off vertical", plan.over_the_rim_why)

    def test_a_drop_below_the_rim_the_guard_refuses_on_the_screen_goes_over_the_rim(self) -> None:
        arm = TaskArm([])
        below = Pose.tool_down(BIN_CENTRE[0], BIN_CENTRE[1], BIN_RIM_MM - 30.0 + GRASP_Z_MM)
        refused = _key(arm.nearest_configuration(below))
        plan = _plan(arm=TaskArm([], screens={refused: PoseVerdict.GUARD_REFUSED}))

        self.assertTrue(plan.ok)
        self.assertEqual("over_the_rim", plan.release)
        self.assertIn("no move goes to the drop", plan.over_the_rim_why)


class TheInsideIsReadOnTheChecksFrameTests(unittest.TestCase):
    """What a check reads inside the box: on the frame its rim was read in, one measure, no detector."""

    def _check(self, scene: BinScene, *, then: Any = None, inside: bool = True) -> Any:
        """The bin as the survey kept it from L1 (its rim's colour read), ``then`` done to the bench, and the check."""
        from dataclasses import replace

        from src.robot.execution.place_target import KeptTarget, _rim_colour, located_image, recheck

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1})
        arm._tcp = AT_L1  # noqa: SLF001 (the arm stands at the bin's look)
        locator = MeasuringLocator(scene, arm=arm, log=log)
        located = locator.locate("yellow bin")
        kept = KeptTarget.of(located.objects[0], camera="wrist", look=L1, seen_at=1.0, phrase="yellow bin")
        assert kept is not None
        kept = replace(kept, colour_lab=_rim_colour(located, kept, located_image(locator, "yellow bin")))
        if then is not None:
            then(scene)
        return recheck([locator], kept, inside=inside), locator

    def test_an_empty_box_reads_its_floor_in_the_one_measure_its_rim_is_read_with(self) -> None:
        check, locator = self._check(BinScene())

        self.assertEqual("depth", check.by)
        assert check.inside is not None
        self.assertEqual("box", check.inside.kind)
        self.assertGreaterEqual(check.inside.top_mm, 5.0, check.inside.said)
        self.assertLessEqual(check.inside.top_mm, 20.0, check.inside.said)
        self.assertEqual(1, len(locator.measured), "the inside cost a frame of its own")
        self.assertEqual(["yellow bin"], locator.asked)

    def test_a_part_already_in_the_box_raises_what_its_inside_reads(self) -> None:
        """A 40 mm cube on the 5 mm floor: its top at 45 mm, read a step or two over it, never under it."""
        check, _locator = self._check(BinScene(), then=lambda scene: scene.in_front.append(
            ((BIN_CENTRE[0], BIN_CENTRE[1], 25.0), (20.0, 20.0, 20.0))))

        self.assertEqual("depth", check.by)
        assert check.inside is not None
        self.assertGreaterEqual(check.inside.top_mm, 45.0, check.inside.said)
        self.assertLessEqual(check.inside.top_mm, 60.0, check.inside.said)

    def test_a_part_the_detectors_mask_of_the_box_left_out_is_read_where_the_box_was_seen_before(self) -> None:
        """The box moved 40 mm with a part in it: the detector's new mask of the box leaves the part out, a hole where
        it stands, and the places the box showed when it was found, carried along with it, still read the part."""
        def move_with_a_part_in_it(scene: BinScene) -> None:
            scene.move(40.0)
            scene.in_front.append(((BIN_CENTRE[0] + 40.0, BIN_CENTRE[1], 25.0), (20.0, 20.0, 20.0)))

        check, _locator = self._check(BinScene(), then=move_with_a_part_in_it)

        self.assertEqual("detector", check.by)
        assert check.inside is not None
        self.assertGreaterEqual(check.inside.top_mm, 45.0, check.inside.said)

    def test_a_box_the_detector_followed_is_read_on_the_frame_the_detector_located_in(self) -> None:
        check, locator = self._check(BinScene(), then=lambda scene: scene.move(40.0))

        self.assertEqual("detector", check.by)
        assert check.inside is not None
        self.assertLessEqual(check.inside.top_mm, 20.0, check.inside.said)
        self.assertEqual(["yellow bin", "yellow bin"], locator.asked)

    def test_a_check_not_asked_to_read_inside_reads_what_it_always_read(self) -> None:
        check, locator = self._check(BinScene(), inside=False)

        self.assertIsNone(check.inside)
        self.assertEqual(1, len(locator.measured))


def _slanted_read(*, extra: "tuple[Any, ...]" = (), stray: "np.ndarray | None" = None) -> Any:
    """What a camera 45 degrees down beside the bin of ``tests/_seen_scenes.py`` reads inside it (240 x 180 mm, its
    rim at 110 mm, its floor at 3 mm), the bin kept from an empty view (``stray`` points added to what it kept), read
    again with ``extra`` solids in it."""
    from src.robot.execution.place_target import KeptTarget, _gap_mm, _inside_of, _probes_of
    from src.robot.perception.locator import _measured
    from tests._seen_scenes import K, looking_at, open_bin, render, seen_points

    camera = looking_at((450.0 - 380.0, -250.0, 110.0 + 380.0), (450.0, -250.0, 40.0))
    bin_solids = list(open_bin((450.0, -250.0), (240.0, 180.0), 110.0))
    seen = seen_points(camera, render(camera, bin_solids), above_mm=1.0)
    kept = KeptTarget.of(bin_object(seen if stray is None else np.vstack([seen, stray])), camera="wrist", look=None,
                         seen_at=0.0)
    assert kept is not None
    probes = _probes_of(kept)
    assert probes is not None
    read = _measured(probes.points, camera="wrist", captured_at_s=0.0, depth_mm=render(camera, bin_solids + list(extra)),
                     intrinsics=K, camera_to_base=camera, rgb=None)
    return _inside_of(probes, _gap_mm(read))


class AnInsideSeenAtASlantTests(unittest.TestCase):
    """The owner's camera looks down at a slant: the near wall hides the floor behind it."""

    def test_its_floor_and_a_part_on_it_read_where_they_stand(self) -> None:
        from tests._seen_scenes import Solid

        empty = _slanted_read()
        with_a_cube = _slanted_read(extra=(Solid((450.0, -250.0, 23.0), (20.0, 20.0, 20.0)),))

        self.assertLessEqual(empty.top_mm, 15.0, empty.said)
        self.assertGreaterEqual(with_a_cube.top_mm, 43.0, with_a_cube.said)
        self.assertLessEqual(with_a_cube.top_mm, 58.0, with_a_cube.said)

    def test_a_few_stray_readings_behind_the_near_wall_do_not_read_the_box_full(self) -> None:
        """Pixels that flew off the near wall's top land on the floor behind it, where the camera sees nothing: read
        again, the wall stands in front of them all the way up, and the box would read full to the shadow's top."""
        stray = np.array([[350.0, -250.0 + dy, 3.0] for dy in (-60.0, -20.0, 20.0, 60.0)])

        inside = _slanted_read(stray=stray)

        self.assertLessEqual(inside.top_mm, 15.0, inside.said)


def _below_task(picks: Any = ("part",), *, scene: "BinScene | None" = None, refuse: Any = None,
                release: str = "below_the_rim", **keywords: Any) -> Any:
    from src.robot.execution.task import PlaceAt

    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=refuse)
    arm.config = hande_tree(release_in_a_box=release)  # type: ignore[attr-defined]
    scene = scene if scene is not None else BinScene()
    locator = MeasuringLocator(scene, arm=arm, log=log)
    picks = tuple(Pick(p, cloud=cube_cloud()) if isinstance(p, str) else p for p in picks)
    ran = run(picks, place=PlaceAt(camera="yellow bin", air_mm=20.0), arm=arm, wrist=True, looks=(L1, L2),
              locators=[locator], **keywords)
    ran.locator, ran.scene = locator, scene  # type: ignore[attr-defined]
    return ran


def _place_moves(ran: Any) -> list[Any]:
    return [m for m in motions(ran.after("task.place_started")) if m[0] == "move"]


class ATaskSetsItsPartDownBelowTheRimTests(unittest.TestCase):
    def test_the_part_goes_30_mm_under_the_rim_of_the_box_and_the_arm_returns(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _below_task()

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        standoff, line_in, line_out = _place_moves(ran)[:3]
        np.testing.assert_allclose(BIN_CENTRE, line_in[1][:2], atol=10.0)
        self.assertAlmostEqual(BIN_RIM_MM - 30.0 + GRASP_Z_MM, line_in[1][2], delta=0.5)
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM + 80.0, standoff[1][2], delta=0.5,
                               msg="the part did not come to the box as it always did")
        self.assertEqual(standoff[1], line_out[1])
        drop = ran.hooks.of("task.drop_planned")[0]
        self.assertEqual(("below_the_rim", 30.0), (drop["release"], round(drop["below_rim_mm"])))
        self.assertIs(True, ran.hooks.of("task.placed")[0]["below_the_rim"])
        self.assertEqual(2, do0_changes(ran.log))
        self.assertEqual(1, len(ran.locator.measured), "the inside cost a frame of its own")
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_box_a_tall_part_fills_takes_the_next_part_over_its_rim(self) -> None:
        scene = BinScene()
        ran = _below_task((Pick("part", cloud=cube_cloud(), then=lambda: scene.in_front.append(
            ((BIN_CENTRE[0] + 60.0, BIN_CENTRE[1], 55.0), (20.0, 20.0, 50.0)))),), scene=scene)

        line_in = _place_moves(ran)[1]
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM, line_in[1][2], delta=0.5)
        self.assertNotIn("release", ran.hooks.of("task.drop_planned")[0])
        self.assertIn("full", ran.hooks.said[ran.names().index("task.drop_planned")])

    def test_as_shipped_the_part_is_let_go_over_the_rim_and_the_inside_is_not_read(self) -> None:
        ran = _below_task(release="over_the_rim")

        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM, _place_moves(ran)[1][1][2], delta=0.5)
        self.assertNotIn("below_the_rim", ran.hooks.of("task.placed")[0])
        self.assertEqual([ran.report.kept_target.rim_band_mm.shape[0]], ran.locator.measured,
                         "the check read more than the rim band")

    def test_a_line_into_the_box_the_guard_refuses_lets_the_part_go_over_the_rim_from_the_same_standoff(self) -> None:
        from src.robot.execution.task import TaskStop

        def refuse_the_line_in(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "move" and abs(float(target.position_mm[2]) - (BIN_RIM_MM - 30.0 + GRASP_Z_MM)) < 0.5:
                return MotionStatus.SELF_COLLISION_REJECTED
            return None

        ran = _below_task(refuse=refuse_the_line_in)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        moves = _place_moves(ran)
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM + 80.0, moves[0][1][2], delta=0.5)
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM, moves[1][1][2], delta=0.5)
        self.assertTrue(moves[1][2], "the drop over the rim was not a line")
        self.assertIn(("refused", "move", "self_collision_rejected"), ran.after("task.place_started"))
        self.assertIs(False, ran.hooks.of("task.placed")[0]["below_the_rim"])
        self.assertIn("over the rim", ran.hooks.said[ran.names().index("task.placed")])
        self.assertEqual(2, do0_changes(ran.log))

    def test_a_line_into_the_box_that_failed_once_sent_ends_where_the_arm_stands_with_the_part(self) -> None:
        from src.robot.execution.task import TaskStop

        def fail_the_line_in(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "move" and abs(float(target.position_mm[2]) - (BIN_RIM_MM - 30.0 + GRASP_Z_MM)) < 0.5:
                return MotionStatus.CONTROLLER_REJECTED
            return None

        ran = _below_task(refuse=fail_the_line_in)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(ran.log))
        self.assertNotIn("task.return_started", ran.names())


class ThePlaceVerbTests(unittest.TestCase):
    """``handling.place(..., instead=)``: the line into the box, and the drop over its rim where that is refused."""

    def _place(self, refuse: Any = None, *, instead: bool = True) -> Any:
        from src.robot.execution import handling
        from src.robot.execution.robot import Robot

        log: list[Any] = []
        arm = TaskArm(log, refuse=refuse)
        jaws = toggle(log, closed=True, arm=arm)
        robot = Robot.from_parts(arm=arm, gripper=jaws, lock_key=None)
        below = Pose.tool_down(400.0, -300.0, 110.0, label="set down in the bin")
        over = Pose.tool_down(400.0, -300.0, 160.0, label="drop over the bin")
        keywords = {"instead": over} if instead else {}
        return handling.place(robot, below, standoff_mm=80.0, **keywords), log, below, over

    def test_a_line_in_refused_before_anything_was_sent_goes_to_the_drop_over_the_rim(self) -> None:
        from src.robot.execution.handling import HandlingOutcome
        from src.robot.execution.task import place_released

        report, log, below, over = self._place(lambda kind, index, target: (
            MotionStatus.SELF_COLLISION_REJECTED if index == 1 else None))

        self.assertIs(HandlingOutcome.EXECUTED, report.outcome, report.message)
        self.assertEqual(3, len(report.poses))
        self.assertIs(over, report.poses[1])
        self.assertEqual([MotionStatus.EXECUTED] * 3, list(report.statuses))
        self.assertTrue(place_released(report))
        self.assertEqual(1, do0_changes(log))

    def test_a_line_in_that_failed_once_sent_ends_the_place_as_before(self) -> None:
        from src.robot.execution.handling import HandlingOutcome
        from src.robot.execution.task import place_released

        report, log, _below, _over = self._place(lambda kind, index, target: (
            MotionStatus.CONTROLLER_REJECTED if index == 1 else None))

        self.assertIs(HandlingOutcome.MOTION_REFUSED, report.outcome)
        self.assertEqual(2, len(report.poses))
        self.assertFalse(place_released(report))
        self.assertEqual(0, do0_changes(log))

    def test_a_drop_over_the_rim_refused_too_ends_the_place_before_its_release(self) -> None:
        from src.robot.execution.handling import HandlingOutcome
        from src.robot.execution.task import place_released

        report, log, _below, over = self._place(lambda kind, index, target: (
            MotionStatus.SELF_COLLISION_REJECTED if index in (1, 2) else None))

        self.assertIs(HandlingOutcome.MOTION_REFUSED, report.outcome)
        self.assertEqual(2, len(report.poses))
        self.assertIs(over, report.poses[1])
        self.assertFalse(place_released(report))
        self.assertEqual(0, do0_changes(log))

    def test_a_place_with_no_drop_over_the_rim_ends_at_a_refused_line_in_as_before(self) -> None:
        from src.robot.execution.handling import HandlingOutcome

        report, log, below, _over = self._place(lambda kind, index, target: (
            MotionStatus.SELF_COLLISION_REJECTED if index == 1 else None), instead=False)

        self.assertIs(HandlingOutcome.MOTION_REFUSED, report.outcome)
        self.assertIs(below, report.poses[1])
        self.assertEqual(0, do0_changes(log))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
