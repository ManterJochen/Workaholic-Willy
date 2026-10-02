"""A task puts its parts into the bin its camera found (OD 19, with the owner's Q13 A departures, Q3, Q4 and Q5).

The operator says "in die blaue Kiste", and the task finds the bin with the pick service's own detector (no second
model copy), before its first pick:

1. **Survey** (Q13 (a)). From each of its looks in turn, the camera locates the bin's phrase, then the part's; the first
   look whose bin has at least 20 points of surface is the look it is kept from (L_T); of several, the most confident
   (Q5). None: the arm returns and the task asks (``target_not_found``) with no pick and no change of DO0. A drop over
   the bin that no configuration reaches ends ``target_unreachable`` before any pick. A detector that failed is never
   "no bin" (``detector_failed``).
2. **Keep-out** (Q4). The bin's footprint is kept out of every pick of the task, once and until empty, so no pick takes
   the bin or a part already in it; a look that sees only those is an empty look.
3. **Before every drop** (Q13 (b)). Carrying the part, the arm goes to L_T, holds every frame its wrist camera takes in
   the planner world (``holding_views``), and the camera looks at the bin again: it is followed only within
   min(100 mm, half its footprint's diagonal) of where the survey found it, a footprint within 20 % and a rim within
   20 mm of the survey's (``recheck``), so a bin that creeps between the parts is not followed further and further.
4. **The drop** (Q3, item 6). Over the middle of the rim, at rim + the part's hang + the rim air (20 mm unless the
   operator chose 10 to 50), turned along the cell's natural closing axis for a grasp within 5 degrees of vertical;
   ``Robot.place`` without a keep-out, which would take the bin's walls out of the world the line in is judged against.
   The return runs inside the hold, which ends after it.
5. **Lost** (Q13 (c)). A bin not seen, moved too far or swapped, a drop no configuration reaches, a part that does not
   fit the opening: the part goes back where it was gripped (``service.put_back``), the arm returns, and the task asks.
   A put back that does not release stays where it stands; one whose line out a stopped controller refused after the
   release sends no return to it.

A fixed camera finds the bin once where the arm stands and checks it at the start of each part instead, with no look
to carry the part to and no frames to hold. "Stop after this part" is read before the survey and before each of its
looks, so a stop pressed before the first pick moves the arm no further than back to its return pose.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.errors import CameraWorldUnavailable, PerceptionFrameMoved
from src.robot.execution.looks import look_label
from tests._task_fakes import (
    BIN_CENTRE,
    BIN_RIM_MM,
    BIN_SIZE,
    GRASP_Z_MM,
    PART_XY,
    PROTECTIVE,
    Pick,
    RecordingHooks,
    ScriptedLocator,
    TaskArm,
    bin_object,
    bin_points,
    do0_changes,
    motions,
    run,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


class _Bin:
    """Where the bin stands, which a test moves or swaps between two locates; seen from L1 only."""

    def __init__(self) -> None:
        self.objects: list[Any] = [bin_object()]

    def seen(self, tcp: Pose) -> list[Any]:
        return list(self.objects) if tcp is AT_L1 else []

    def move(self, dx: float, dy: float = 0.0, *, size: "tuple[float, float]" = BIN_SIZE) -> None:
        self.objects = [bin_object(bin_points((BIN_CENTRE[0] + dx, BIN_CENTRE[1] + dy), size))]

    def take_away(self) -> None:
        self.objects = []


def _parts(_tcp: Pose) -> list[Any]:
    return [bin_object(bin_points(PART_XY, (40.0, 40.0), 40.0, inside=False), label="red cube", score=0.7)]


def _camera_task(picks: Any = ("part",), *, bin: "_Bin | None" = None, arm: "TaskArm | None" = None,
                 scope: str = "once", air_mm: "float | None" = None, wrist: bool = True,
                 locator: "ScriptedLocator | None" = None, **keywords: Any) -> Any:
    from src.robot.execution.task import PlaceAt

    log: list[Any] = arm.log if arm is not None else []
    arm = arm if arm is not None else TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
    bin = bin if bin is not None else _Bin()
    sees: dict[str, Any] = {"blue bin": bin.seen if wrist else (lambda _tcp: list(bin.objects)), "red cube": _parts}
    locator = locator if locator is not None else ScriptedLocator(sees, arm=arm, log=log, wrist=wrist,
                                                                  rig_id="wrist" if wrist else "overhead")
    ran = run(picks, place=PlaceAt(camera="blue bin", air_mm=air_mm), scope=scope, arm=arm, wrist=wrist,
              looks=(L1, L2), locators=[locator], **keywords)
    ran.locator = locator  # type: ignore[attr-defined]
    return ran


def _place_moves(ran: Any) -> list[Any]:
    return [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][:3]


def _said(hooks: RecordingHooks, name: str) -> bool:
    return any(event == name for event, _ in hooks.events)


def _camera_unavailable() -> CameraWorldUnavailable:
    return CameraWorldUnavailable(camera="wrist", verdict="stale", attempts=3,
                                  reason="every frame was older than the cell allows")


def _part_along_x(length_mm: float, width_mm: float = 40.0) -> np.ndarray:
    """The cloud of a part ``length_mm`` long along base x where it stood when it was gripped."""
    xs = np.linspace(-length_mm / 2.0, length_mm / 2.0, 40)
    ys = np.linspace(-width_mm / 2.0, width_mm / 2.0, 10)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([PART_XY[0] + gx.ravel(), PART_XY[1] + gy.ravel(), np.full(gx.size, 40.0)])


class TheBinFoundTests(unittest.TestCase):
    def test_the_part_goes_into_the_bin_just_above_its_rim_and_the_arm_returns(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _camera_task()

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(["task.survey_started", "task.target_found", "task.part_started", "task.carry_started",
                          "task.target_checked", "task.drop_planned", "task.place_started", "task.placed",
                          "task.return_started", "task.returned", "task.part_finished"], ran.names())
        self.assertEqual(2, do0_changes(ran.log))
        line_in = _place_moves(ran)[1]
        np.testing.assert_allclose(BIN_CENTRE, line_in[1][:2], atol=10.0)
        self.assertAlmostEqual(BIN_RIM_MM + GRASP_Z_MM + 20.0, line_in[1][2], delta=0.5)
        self.assertEqual({"place": "target:blue bin"}, ran.hooks.of("task.place_started")[0])
        drop = ran.hooks.of("task.drop_planned")[0]
        self.assertEqual(("camera", 20.0), (drop["kind"], drop["air_mm"]))
        self.assertAlmostEqual(BIN_RIM_MM, drop["rim_mm"], delta=0.5)

    def test_the_survey_finds_the_bin_first_and_keeps_its_look(self) -> None:
        ran = _camera_task()

        survey = ran.hooks.of("task.survey_started")[0]
        self.assertEqual({"phrase": "blue bin", "looks": [look_label(L1), look_label(L2)]}, survey)
        found = ran.hooks.of("task.target_found")[0]
        self.assertEqual(look_label(L1), found["look"])
        self.assertEqual(1, found["parts_seen"])
        from api.schemas import TargetOut

        TargetOut.model_validate({key: value for key, value in found["target"].items()
                                  if key in TargetOut.model_fields})
        self.assertEqual(("joints", _key(L1)), motions(ran.log)[0], "the survey did not go to its first look")
        self.assertEqual(["blue bin", "red cube", "blue bin"], ran.locator.asked)

    def test_the_drop_is_checked_from_the_bins_look_with_every_frame_held_and_no_keep_out(self) -> None:
        ran = _camera_task()

        carried = ran.after("task.carry_started")
        self.assertEqual(("joints", _key(L1)), motions(carried)[0], "the part was not carried to the bin's look")
        hold = carried.index(("hold",))
        self.assertLess(carried.index(("joints", _key(L1))), hold)
        self.assertLess(hold, carried.index(("locate", "blue bin")))
        self.assertNotIn(("offer",), ran.after("task.place_started"), "the place held the bin out of the world")
        forget = carried.index(("forget",))
        self.assertGreater(forget, carried.index(("home",)), "the hold ended before the return")
        self.assertEqual({"to_look": look_label(L1)}, ran.hooks.of("task.carry_started")[0])
        checked = ran.hooks.of("task.target_checked")[0]
        self.assertIs(True, checked["followed"])
        self.assertAlmostEqual(0.0, checked["moved_mm"], delta=1.0)

    def test_the_rim_air_is_the_operators_choice(self) -> None:
        ran = _camera_task(air_mm=45.0)

        self.assertAlmostEqual(BIN_RIM_MM + GRASP_Z_MM + 45.0, _place_moves(ran)[1][1][2], delta=0.5)

    def test_a_vertical_grasp_is_dropped_along_the_cells_natural_axis(self) -> None:
        from src.geometry.closing_axis import closing_axis_of

        ran = _camera_task(natural_axis=closing_axis_of("-y"))

        line_in = [pose for pose in ran.arm.moved if math.isclose(float(pose.position_mm[2]),
                                                                  BIN_RIM_MM + GRASP_Z_MM + 20.0, abs_tol=0.5)][0]
        tool_x = np.asarray(line_in.to_matrix())[:3, 0]
        self.assertAlmostEqual(270.0, math.degrees(math.atan2(tool_x[1], tool_x[0])) % 360.0, delta=0.01)

    def test_the_most_confident_of_several_bins_is_kept(self) -> None:
        bin = _Bin()
        bin.objects = [bin_object(bin_points((BIN_CENTRE[0] + 400.0, BIN_CENTRE[1])), score=0.6), bin_object(
            score=0.95)]
        ran = _camera_task(bin=bin)

        self.assertEqual(0.95, ran.hooks.of("task.target_found")[0]["target"]["score"])
        np.testing.assert_allclose(BIN_CENTRE, _place_moves(ran)[1][1][:2], atol=10.0)


class TheBinKeptOutTests(unittest.TestCase):
    def test_the_bins_footprint_is_kept_out_of_every_pick_for_both_scopes(self) -> None:
        for scope, picks in (("once", ("part",)), ("until_empty", ("part", "empty", "empty"))):
            with self.subTest(scope=scope):
                ran = _camera_task(picks, scope=scope)

                self.assertTrue(ran.service.zones_seen)
                for regions in ran.service.zones_seen:
                    self.assertEqual(1, len(regions))
                    (region,) = regions
                    self.assertIsNone(region.label)
                    self.assertTrue(region.contains((*BIN_CENTRE, 60.0)))
                    self.assertIn("blue bin", region.reason)
                self.assertEqual((), ran.service.campaign.zones.regions())

    def test_until_empty_into_a_bin_ends_nothing_left_when_only_its_parts_remain_and_returns(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _camera_task(("part", "part", "excluded", "excluded"), scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.parts_placed)
        self.assertEqual(("home",), motions(ran.log)[-1])
        self.assertEqual([True, True], [event["only_excluded"] for event in ran.hooks.of("task.nothing_found")])
        self.assertEqual(4, do0_changes(ran.log))


class NoBinTests(unittest.TestCase):
    def test_a_bin_no_look_sees_ends_target_not_found_before_any_pick(self) -> None:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        bin.take_away()
        ran = _camera_task(bin=bin)

        self.assertIs(TaskStop.TARGET_NOT_FOUND, ran.report.stop)
        self.assertEqual([], ran.service.calls, "a part was picked for a bin nobody found")
        self.assertEqual(0, do0_changes(ran.log))
        self.assertEqual({"phrase": "blue bin", "looks_tried": [look_label(L1), look_label(L2)]},
                         ran.hooks.of("task.target_missing")[0])
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_detector_that_failed_where_the_bin_should_be_is_not_no_bin(self) -> None:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        bin.take_away()
        failures: list[Any] = []
        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})

        def failing(tcp: Pose) -> list[Any]:
            failures.append(True)
            return []

        locator = ScriptedLocator({"blue bin": failing}, arm=arm, log=log)
        ran = _camera_task(bin=bin, arm=arm, locator=locator, failure_log=failures)

        self.assertIs(TaskStop.DETECTOR_FAILED, ran.report.stop)
        self.assertNotIn("task.return_started", ran.names())
        self.assertEqual([], ran.service.calls)

    def test_a_drop_over_the_bin_no_configuration_reaches_ends_before_any_pick(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, unreachable_above_mm=100.0)
        ran = _camera_task(arm=arm)

        self.assertIs(TaskStop.TARGET_UNREACHABLE, ran.report.stop)
        self.assertEqual([], ran.service.calls)
        self.assertEqual(("home",), motions(log)[-1])


class ABinLostAtTheDropTests(unittest.TestCase):
    def _lost(self, change: Any, why: str) -> Any:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        ran = _camera_task((Pick("part", then=lambda: change(bin)),), bin=bin)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual([{"look": look_label(L1), "why": why}], ran.hooks.of("task.target_lost"))
        self.assertEqual(["task.target_lost", "task.put_back", "task.return_started", "task.returned",
                          "task.part_finished"], ran.names()[-5:])
        self.assertEqual(2, do0_changes(ran.log), "the part was not let go where it was gripped")
        put_back = [m for m in motions(ran.after("task.target_lost")) if m[0] == "move"]
        np.testing.assert_allclose((*PART_XY, GRASP_Z_MM), put_back[1][1])
        self.assertFalse(ran.report.holding)
        self.assertEqual(0, ran.report.parts_placed)
        self.assertEqual(("home",), motions(ran.log)[-1])
        self.assertIs(False, ran.hooks.of("task.part_finished")[0]["placed"])
        return ran

    def test_a_bin_moved_150_mm_is_lost_the_part_goes_back_and_the_arm_returns(self) -> None:
        ran = self._lost(lambda bin: bin.move(150.0), "moved_too_far")
        self.assertAlmostEqual(150.0, ran.hooks.of("task.target_checked")[0]["moved_mm"], delta=3.0)

    def test_a_bin_swapped_for_another_is_lost(self) -> None:
        self._lost(lambda bin: bin.move(0.0, size=(200.0, 120.0)), "footprint_changed")

    def test_a_bin_taken_away_is_lost(self) -> None:
        self._lost(lambda bin: bin.take_away(), "not_seen")

    def test_a_bin_moved_a_little_is_followed(self) -> None:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        ran = _camera_task((Pick("part", then=lambda: bin.move(40.0)),), bin=bin)

        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        np.testing.assert_allclose((BIN_CENTRE[0] + 40.0, BIN_CENTRE[1]), _place_moves(ran)[1][1][:2], atol=10.0)

    def test_a_bin_swapped_for_one_of_its_footprint_and_another_height_is_lost(self) -> None:
        """Red before: only the shift and the footprint were bounded, so a bin 60 mm lower in its place was followed and
        the part was dropped at the new rim."""
        self._lost(lambda bin: setattr(bin, "objects", [bin_object(bin_points(BIN_CENTRE, rim_mm=BIN_RIM_MM - 60.0))]),
                   "footprint_changed")

    def test_a_bin_that_creeps_is_followed_only_within_the_bound_of_where_the_survey_found_it(self) -> None:
        """Red before: each check measured from the last bin it followed, so a bin moved 60 mm before every drop was
        followed 60 mm at a time without end. Measured from where the survey found it, the second drop stands 120 mm
        away, past the 100 mm bound: the part goes back and the task asks."""
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        moved: list[int] = []

        def creep() -> None:
            moved.append(1)
            bin.move(60.0 * len(moved))

        ran = _camera_task((Pick("part", then=creep), Pick("part", then=creep), "part"), bin=bin, scope="until_empty")

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        checks = ran.hooks.of("task.target_checked")
        self.assertEqual([True, False], [check["followed"] for check in checks])
        self.assertAlmostEqual(120.0, checks[1]["moved_mm"], delta=3.0)
        self.assertEqual("moved_too_far", ran.hooks.of("task.target_lost")[0]["why"])
        self.assertEqual(4, do0_changes(ran.log))
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_put_back_whose_line_out_a_stopped_controller_refused_sends_no_return(self) -> None:
        """The part is let go where it was gripped, then a protective stop refuses the line out: the controller is
        asked before the return, and a stopped one gets no move home. Red before: the return was sent to it."""
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        hooks = RecordingHooks()
        log: list[Any] = []
        back: list[int] = []

        def stop_on_the_line_out(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if _said(hooks, "task.target_lost") and kind == "move":
                back.append(index)
                if len(back) == 3:
                    arm._statuses = [PROTECTIVE]  # noqa: SLF001 (a protective stop met the line out)
                    return MotionStatus.CONTROLLER_REJECTED
            return None

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=stop_on_the_line_out)
        ran = _camera_task((Pick("part", then=bin.take_away),), bin=bin, arm=arm, hooks=hooks)

        self.assertIs(TaskStop.CONTROLLER_STOPPED, ran.report.stop, ran.report.sentence)
        self.assertFalse(ran.report.holding)
        self.assertEqual(2, do0_changes(log), "the part was not let go where it was gripped")
        self.assertEqual([], motions(ran.after("task.put_back")), "a motion was sent to a stopped controller")
        self.assertNotIn("task.return_started", ran.names())

    def test_a_put_back_whose_line_out_a_camera_could_not_vouch_for_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        hooks = RecordingHooks()
        log: list[Any] = []
        back: list[int] = []

        def blind_on_the_line_out(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if _said(hooks, "task.target_lost") and kind == "move":
                back.append(index)
                if len(back) == 3:
                    raise _camera_unavailable()
            return None

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=blind_on_the_line_out)
        ran = _camera_task((Pick("part", then=bin.take_away),), bin=bin, arm=arm, hooks=hooks)

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertFalse(ran.report.holding)
        self.assertEqual(2, do0_changes(log))
        self.assertNotIn("task.return_started", ran.names())

    def test_a_carry_to_the_bins_look_refused_before_anything_was_sent_puts_the_part_back_and_asks(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        log: list[Any] = []
        refused: list[int] = []

        def refuse_the_carry(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if _said(hooks, "task.carry_started") and not refused:
                refused.append(index)
                return MotionStatus.WORKSPACE_REJECTED
            return None

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=refuse_the_carry)
        ran = _camera_task(arm=arm, hooks=hooks)

        self.assertIs(TaskStop.TARGET_UNREACHABLE, ran.report.stop, ran.report.sentence)
        self.assertFalse(ran.report.holding)
        self.assertEqual(2, do0_changes(log))
        self.assertEqual(["task.put_back", "task.return_started", "task.returned", "task.part_finished"],
                         ran.names()[-4:])
        self.assertEqual(("home",), motions(log)[-1])

    def test_a_carry_that_failed_once_it_may_have_moved_stops_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        log: list[Any] = []
        refused: list[int] = []

        def fail_the_carry(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if _said(hooks, "task.carry_started") and not refused:
                refused.append(index)
                return MotionStatus.CONTROLLER_REJECTED
            return None

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=fail_the_carry)
        ran = _camera_task(arm=arm, hooks=hooks)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertEqual([], motions(ran.after("task.carry_started")))
        self.assertNotIn("task.put_back", ran.names())

    def test_a_put_back_that_does_not_release_stays_where_it_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        hooks = RecordingHooks()
        log: list[Any] = []

        def refuse_after_the_loss(kind: str, index: int, target: Any) -> "MotionStatus | None":
            lost = any(name == "task.target_lost" for name, _ in hooks.events)
            return MotionStatus.WORKSPACE_REJECTED if lost else None

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=refuse_after_the_loss)
        ran = _camera_task((Pick("part", then=lambda: bin.take_away()),), bin=bin, arm=arm, hooks=hooks)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertNotIn("task.return_started", ran.names())

    def test_a_drop_no_configuration_reaches_puts_the_part_back_and_asks(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, unreachable_above_mm=150.0)
        ran = _camera_task(arm=arm)

        self.assertIs(TaskStop.TARGET_UNREACHABLE, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, do0_changes(log))
        self.assertEqual(("home",), motions(log)[-1])

    def test_a_part_that_does_not_fit_the_opening_goes_back(self) -> None:
        from src.robot.execution.task import TaskStop

        long = np.column_stack([np.linspace(PART_XY[0] - 160.0, PART_XY[0] + 160.0, 40),
                                np.full(40, PART_XY[1]), np.full(40, 40.0)])
        ran = _camera_task((Pick("part", cloud=long),))

        self.assertIs(TaskStop.PART_DOES_NOT_FIT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, do0_changes(ran.log))

    def test_a_world_that_holds_no_frame_puts_the_part_back(self) -> None:
        """The bin cannot be checked against held frames, so it is not seen: ``why`` stays within the console's three
        words (``not_seen``), and the sentence says the frames. Red before: ``views_not_held``, a word no frontend
        catalog knows."""
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, holds=False)
        ran = _camera_task(arm=arm)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop)
        self.assertEqual("not_seen", ran.hooks.of("task.target_lost")[0]["why"])
        self.assertIn("frames", ran.hooks.said[ran.names().index("task.target_lost")])
        self.assertEqual(2, do0_changes(log))

    def test_a_camera_that_could_not_vouch_at_the_check_stops_the_task_where_it_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
        bin = _Bin()
        locator = ScriptedLocator({"blue bin": bin.seen, "red cube": _parts}, arm=arm, log=log, raises_on={
            3: PerceptionFrameMoved(camera="wrist", attempts=4, moved_mm=3.0, turned_deg=0.2, tolerance_mm=1.0,
                                    tolerance_deg=0.5)})
        ran = _camera_task(arm=arm, locator=locator, bin=bin)

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertEqual([], motions(log[log.index(("locate", "blue bin"), log.index(("hold",))):]))
        self.assertIn(("forget",), log, "the hold outlived the task")


class AFixedCameraTests(unittest.TestCase):
    def test_a_fixed_camera_finds_the_bin_where_it_stands_and_carries_to_no_look(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _camera_task(wrist=False)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertNotIn("task.carry_started", ran.names())
        self.assertNotIn(("hold",), ran.log)
        self.assertEqual(["blue bin", "red cube", "blue bin"], ran.locator.asked)
        self.assertLess(ran.log.index(("event", "task.target_checked")), ran.log.index(("event", "task.part_started")),
                        "a fixed camera checks its bin before the pick, while the arm is out of its view")
        self.assertIsNone(ran.hooks.of("task.target_found")[0]["look"])

    def test_a_bin_a_fixed_camera_lost_before_a_pick_returns_and_asks_with_nothing_picked(self) -> None:
        from src.robot.execution.task import TaskStop

        bin = _Bin()
        ran = _camera_task(("part", "part"), bin=bin, scope="until_empty", wrist=False,
                           hooks=RecordingHooks(on_event=lambda name, _d: bin.move(200.0)
                                                if name == "task.part_finished" else None))

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop)
        self.assertEqual(1, ran.report.picks)
        self.assertEqual(2, do0_changes(ran.log))


class ARestartIntoABinTests(unittest.TestCase):
    def test_a_restart_goes_home_first_then_finds_its_bin_again(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _camera_task(first_motion="return")

        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        self.assertEqual(("home",), motions(ran.log)[0])
        self.assertEqual(["task.return_started", "task.returned", "task.survey_started"], ran.names()[:3])


class StopAfterThisPartBeforeThePickTests(unittest.TestCase):
    """"Stop after this part" before the first pick: there is no part in hand to finish, so the task ends at once, the
    arm back at its return pose where the task moved it. Red before: the survey was not asked, and drove to every look
    first."""

    def test_a_stop_before_the_survey_moves_nothing(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _camera_task(hooks=RecordingHooks(stop=True))

        self.assertIs(TaskStop.STOPPED_AFTER_PART, ran.report.stop, ran.report.sentence)
        self.assertEqual([], motions(ran.log))
        self.assertEqual([], ran.service.calls)
        self.assertNotIn("task.survey_started", ran.names())

    def test_a_stop_during_the_survey_ends_it_before_its_next_look_and_returns(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        log: list[Any] = []

        def press_on_the_way(kind: str, index: int) -> None:
            if index == 0:
                hooks.stop = True

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, on_motion=press_on_the_way)
        locator = ScriptedLocator({"blue bin": lambda tcp: [bin_object()] if tcp is AT_L2 else []}, arm=arm, log=log)
        ran = _camera_task(arm=arm, hooks=hooks, locator=locator)

        self.assertIs(TaskStop.STOPPED_AFTER_PART, ran.report.stop, ran.report.sentence)
        self.assertEqual([("joints", _key(L1)), ("home",)], motions(log))
        self.assertEqual([], ran.service.calls)
        self.assertTrue(ran.report.at_return)

    def test_a_restart_stopped_on_its_way_home_surveys_nothing(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        hooks.on_event = lambda name, _data: setattr(hooks, "stop", True) if name == "task.return_started" else None
        ran = _camera_task(first_motion="return", hooks=hooks)

        self.assertIs(TaskStop.STOPPED_AFTER_PART, ran.report.stop, ran.report.sentence)
        self.assertEqual([("home",)], motions(ran.log))
        self.assertNotIn("task.survey_started", ran.names())
        self.assertTrue(ran.report.at_return)


class TheSurveyAndTheCarryEndWhereTheArmStandsTests(unittest.TestCase):
    """What ends a camera place where the arm stands before its drop: a halt between two looks of the survey, a camera
    that could not vouch on a look or on the carry, a look the arm may have started for, a detector that failed at the
    check. Each sends nothing more, the jaws included."""

    def _arm(self, log: list[Any], **keywords: Any) -> TaskArm:
        return TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, **keywords)

    def _seen_from_l2(self, arm: TaskArm, log: list[Any]) -> ScriptedLocator:
        return ScriptedLocator({"blue bin": lambda tcp: [bin_object()] if tcp is AT_L2 else []}, arm=arm, log=log)

    def test_a_halt_between_two_looks_of_the_survey_ends_it_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        log: list[Any] = []
        arm = self._arm(log, on_motion=lambda kind, index: setattr(hooks, "halt", True) if index == 0 else None)
        ran = _camera_task(arm=arm, hooks=hooks, locator=self._seen_from_l2(arm, log))

        self.assertIs(TaskStop.HALTED, ran.report.stop, ran.report.sentence)
        self.assertEqual([("joints", _key(L1))], motions(log))
        self.assertEqual([], ran.service.calls)

    def test_a_camera_that_could_not_vouch_on_a_survey_look_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        def blind(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if index == 0:
                raise _camera_unavailable()
            return None

        log: list[Any] = []
        ran = _camera_task(arm=self._arm(log, refuse=blind))

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertEqual([], motions(log))
        self.assertEqual([], ran.service.calls)
        self.assertNotIn("task.return_started", ran.names())

    def test_a_survey_look_the_arm_may_have_started_for_ends_the_survey_with_nothing_else_sent(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = self._arm(log, refuse=lambda kind, index, target: MotionStatus.CONTROLLER_REJECTED if index == 1 else None)
        ran = _camera_task(arm=arm, locator=self._seen_from_l2(arm, log))

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertEqual([("joints", _key(L1))], motions(log))
        self.assertIn(look_label(L2), ran.report.sentence)
        self.assertNotIn("task.return_started", ran.names())

    def test_a_detector_that_failed_at_the_check_is_not_a_bin_gone(self) -> None:
        from src.robot.execution.task import TaskStop

        failures: list[bool] = []
        log: list[Any] = []
        arm = self._arm(log)
        asked: list[bool] = []

        def bin_then_a_failure(tcp: Pose) -> list[Any]:
            asked.append(True)
            if len(asked) == 1:
                return [bin_object()]
            failures.append(True)
            return []

        locator = ScriptedLocator({"blue bin": bin_then_a_failure}, arm=arm, log=log)
        ran = _camera_task(arm=arm, locator=locator, failure_log=failures)

        self.assertIs(TaskStop.DETECTOR_FAILED, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertNotIn("task.drop_planned", ran.names())
        self.assertNotIn("task.target_lost", ran.names())

    def test_a_camera_that_could_not_vouch_on_the_carry_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        blinded: list[int] = []

        def blind_on_the_carry(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if _said(hooks, "task.carry_started") and not blinded:
                blinded.append(index)
                raise _camera_unavailable()
            return None

        log: list[Any] = []
        ran = _camera_task(arm=self._arm(log, refuse=blind_on_the_carry), hooks=hooks)

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertEqual([], motions(ran.after("task.carry_started")))


class APickWithNoGraspPoseTests(unittest.TestCase):
    """A pick that reports no grasp pose: the part's hang is the declared ``payload.length_mm``, the drop straight down
    over the rim; with neither, nothing is set down blind, and with no grasp pose it cannot go back either."""

    def _declaring(self, log: list[Any], length_mm: "float | None") -> TaskArm:
        from types import SimpleNamespace

        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
        arm.config = SimpleNamespace(  # type: ignore[attr-defined]
            safety=SimpleNamespace(planning_world=SimpleNamespace(payload=SimpleNamespace(length_mm=length_mm),
                                                                  perceived=SimpleNamespace(margin_mm=15.0))),
            grasping=SimpleNamespace(support=SimpleNamespace(height_mm=0.0, container=SimpleNamespace(
                floor_height_mm=None))))
        return arm

    def test_the_declared_length_stands_in_for_the_hang(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        ran = _camera_task((Pick("part", grasp=False),), arm=self._declaring(log, 60.0))

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertAlmostEqual(BIN_RIM_MM + 60.0 + 20.0, _place_moves(ran)[1][1][2], delta=0.5)

    def test_with_neither_nothing_is_set_down_blind(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        ran = _camera_task((Pick("part", grasp=False),), arm=self._declaring(log, None))

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertNotIn("task.place_started", ran.names())
        self.assertIn("length_mm", ran.report.sentence)


class MultiViewOffTests(unittest.TestCase):
    def test_with_multi_view_off_the_survey_looks_from_the_first_look_only(self) -> None:
        """Q11 A for the survey too: the first configured look and no other. A bin only the second look sees is not
        found."""
        from src.robot.execution.task import TaskOptions, TaskStop

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
        locator = ScriptedLocator({"blue bin": lambda tcp: [bin_object()] if tcp is AT_L2 else []}, arm=arm, log=log)
        ran = _camera_task(arm=arm, locator=locator, options=TaskOptions(multi_view=False))

        self.assertIs(TaskStop.TARGET_NOT_FOUND, ran.report.stop, ran.report.sentence)
        self.assertEqual({"phrase": "blue bin", "looks": [look_label(L1)]}, ran.hooks.of("task.survey_started")[0])
        self.assertEqual([("joints", _key(L1)), ("home",)], motions(log))
        self.assertEqual([look_label(L1)], ran.hooks.of("task.target_missing")[0]["looks_tried"])


_RAY_CAST: dict[str, np.ndarray] = {}


def _ray_cast_bin() -> np.ndarray:
    """What the wrist camera at L1 sees of the bin of ``tests/_seen_scenes.py``, ray cast 45 degrees down beside it:
    240 x 180 mm outside, its walls 5 mm thick, so 230 x 170 mm inside, its rim at 110 mm."""
    if "bin" not in _RAY_CAST:
        from tests._seen_scenes import looking_at, open_bin, render, seen_points

        camera = looking_at((450.0 - 380.0, -250.0, 110.0 + 380.0), (450.0, -250.0, 40.0))
        _RAY_CAST["bin"] = seen_points(camera, render(camera, open_bin((450.0, -250.0), (240.0, 180.0), 110.0)))
    return _RAY_CAST["bin"]


class ARayCastBinTests(unittest.TestCase):
    """The bin a camera really sees: its near wall's top, its far wall's inner face and top, some of its floor."""

    def _task(self, length_mm: float) -> Any:
        bin = _Bin()
        bin.objects = [bin_object(_ray_cast_bin())]
        return _camera_task((Pick("part", cloud=_part_along_x(length_mm)),), bin=bin)

    def test_a_part_wider_than_the_bins_inside_but_not_its_outside_goes_back_and_the_arm_returns(self) -> None:
        """Red before: the opening was never found on a bin seen at a slant, so the part was judged against the bin's
        top, 239 mm, and a 235 mm part went onto the rim of a bin 230 mm wide inside."""
        from src.robot.execution.task import TaskStop

        ran = self._task(235.0)

        self.assertIs(TaskStop.PART_DOES_NOT_FIT, ran.report.stop, ran.report.sentence)
        self.assertIn("opening", ran.report.sentence)
        self.assertFalse(ran.report.holding)
        self.assertEqual(2, do0_changes(ran.log))
        self.assertEqual(["task.put_back", "task.return_started", "task.returned", "task.part_finished"],
                         ran.names()[-4:])
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_part_that_fits_the_inside_is_dropped_over_its_middle(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = self._task(200.0)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        line_in = _place_moves(ran)[1]
        np.testing.assert_allclose((450.0, -250.0), line_in[1][:2], atol=10.0)
        self.assertAlmostEqual(110.0 + GRASP_Z_MM + 20.0, line_in[1][2], delta=1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
