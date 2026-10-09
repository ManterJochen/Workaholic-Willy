"""A part is carried straight over the rim to the bin, where the cell chose so (the owner, 2026-10-08 night: a switch).

``robot.place.carry: over_the_rim`` (or the task's ``carry``) skips the stop at the bin's look (the motion map's C8):
where one of the pick's looks was the bin's, the bin is checked on that look's frame, its rim's depth and colour as a
check reads them first, with no detector and no new frame; the arm's world holds the pick's frames again for the place,
so the line into the bin is judged against the walls that look saw; and the part goes up, then straight over to the
drop's standoff, each a line the guard judges sample by sample, at one height: its bottom the rim air and
``rim_floor_margin_mm`` over the rim and over everything the pick's looks saw on the way, never lower than the
standoff. Anything that cannot be so carries the part via the look, as before: a world that cannot hold the pick's
frames again, a bin unsure on that frame, a way the looks did not see, a line refused before anything was sent. A line
that failed once sent ends the task where the arm stands with the part, as a carry to the look does. As shipped, every
part goes via the look.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.execution.looks import look_label
from tests._place_fakes import AgainWorld, cube_cloud, hande_tree, pick_view, run_placing, what_the_looks_saw
from tests._task_fakes import (
    BIN_CENTRE,
    BIN_RIM_MM,
    GRASP_Z_MM,
    PART_XY,
    BinScene,
    MeasuringLocator,
    RecordingHooks,
    TaskArm,
    do0_changes,
    motions,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: L1 looks straight down at the bin, L2 at the parts: between them the pick's looks see the way.
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")
AIR_MM = 20.0
#: The drop over the rim and its standoff: the rim, the part's 20 mm hang, the air; 80 mm up.
DROP_Z = BIN_RIM_MM + GRASP_Z_MM + AIR_MM
STANDOFF_Z = DROP_Z + 80.0


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _carry(scene: "BinScene | None" = None, *, carry: str = "over_the_rim", again: "bool | None" = True,
           refuse: Any = None, looks: "tuple[Any, ...]" = (L1, L2), options: Any = None,
           hooks: "RecordingHooks | None" = None) -> Any:
    """A task into the yellow bin its wrist camera found from L1, the cell's carry ``carry``, the pick's looks seeing
    the bench as it stands when the pick looks; ``again`` None is a world that cannot hold a pick's frames again."""
    from src.robot.execution.task import PlaceAt

    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, refuse=refuse)
    if again is not None:
        arm.live_planner_world = AgainWorld(log, again=again)
    arm.config = hande_tree(carry=carry)  # type: ignore[attr-defined]
    scene = scene if scene is not None else BinScene()
    locator = MeasuringLocator(scene, arm=arm, log=log)
    tcps = {_key(L1): AT_L1, _key(L2): AT_L2}

    def looked(_pick: int) -> Any:
        views = [pick_view(scene, tcps[_key(look)], look_label(look)) for look in looks]
        return what_the_looks_saw(cloud=cube_cloud(), views=views)

    ran = run_placing(("part",), arm=arm, place=PlaceAt(camera="yellow bin", air_mm=AIR_MM), wrist=True,
                      looks=(L1, L2), locators=[locator], looked_around=looked, options=options, hooks=hooks)
    ran.locator, ran.scene = locator, scene  # type: ignore[attr-defined]
    return ran


def _carried(ran: Any) -> list[Any]:
    """The arm's motions from the carry on."""
    return motions(ran.after("task.carry_started"))


def _via_the_look(test: unittest.TestCase, ran: Any) -> None:
    from src.robot.execution.task import TaskStop

    test.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
    carries = ran.hooks.of("task.carry_started")
    test.assertEqual({"to_look": look_label(L1)}, carries[-1])
    after = ran.log[max(index for index, entry in enumerate(ran.log) if entry == ("event", "task.carry_started")):]
    test.assertEqual(("joints", _key(L1)), motions(after)[0], "the part was not carried to the bin's look")
    test.assertEqual(2, do0_changes(ran.log))


class ACarryOverTheRimTests(unittest.TestCase):
    def test_the_part_goes_up_and_straight_over_to_the_bin_and_is_placed_with_no_stop_at_its_look(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _carry()

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        carried = _carried(ran)
        self.assertNotIn(("joints", _key(L1)), carried, "the part stopped at the bin's look")
        up, over = carried[0], carried[1]
        self.assertEqual(("move", (*PART_XY, STANDOFF_Z), True), (up[0], (up[1][0], up[1][1], up[1][2]), up[2]))
        self.assertTrue(over[2], "the carry over the rim was not a line")
        np.testing.assert_allclose(BIN_CENTRE, over[1][:2], atol=10.0)
        self.assertAlmostEqual(STANDOFF_Z, over[1][2], delta=0.5)
        line_in = [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]
        self.assertAlmostEqual(DROP_Z, line_in[1][2], delta=0.5)
        self.assertEqual(2, do0_changes(ran.log))
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_the_bin_is_checked_on_the_picks_look_with_no_frame_and_no_detector(self) -> None:
        ran = _carry()

        self.assertEqual([], ran.locator.measured, "the bin's rim was read on a new frame")
        self.assertEqual(["yellow bin"], ran.locator.asked, "the detector was asked again")
        checked = ran.hooks.of("task.target_checked")
        self.assertEqual([(True, "depth")], [(c["followed"], c["by"]) for c in checked])
        self.assertLess(ran.names().index("task.target_checked"), ran.names().index("task.carry_started"))
        self.assertEqual({"to_look": None, "over_the_rim": True}, ran.hooks.of("task.carry_started")[0])

    def test_the_world_holds_the_picks_frames_again_from_before_the_carry_to_after_the_return(self) -> None:
        ran = _carry()

        carried = ran.after("task.target_checked")
        held = ran.log.index(("hold_again",))
        self.assertLess(held, ran.log.index(("event", "task.carry_started")))
        self.assertGreater(len(ran.log) - 1 - ran.log[::-1].index(("forget",)), ran.log.index(("home",), held))
        self.assertNotIn(("hold",), carried)

    def test_something_tall_on_the_way_lifts_the_carry_over_it(self) -> None:
        """A post 200 mm tall between the part and the bin: the part's bottom clears it by the air and the margin."""
        scene = BinScene(in_front=[((275.0, -475.0, 100.0), (15.0, 15.0, 100.0))])
        ran = _carry(scene)

        over = _carried(ran)[1]
        self.assertAlmostEqual(200.0 + AIR_MM + 15.0 + GRASP_Z_MM, over[1][2], delta=1.0)


class TheCarryGoesViaTheLookTests(unittest.TestCase):
    """Wherever the carry over the rim cannot be so, the part goes via the bin's look, as before."""

    def test_as_shipped_the_part_goes_via_the_look(self) -> None:
        _via_the_look(self, _carry(carry="via_the_look"))

    def test_a_world_that_cannot_hold_the_picks_frames_again_sends_the_part_via_the_look(self) -> None:
        for name, again in (("no such world", None), ("none held", False)):
            with self.subTest(name):
                _via_the_look(self, _carry(again=again))

    def test_a_bin_moved_since_it_was_found_is_checked_at_its_look(self) -> None:
        scene = BinScene()
        hooks = RecordingHooks(on_event=lambda name, _d: scene.move(40.0) if name == "task.part_started" else None)
        ran = _carry(scene, hooks=hooks)

        _via_the_look(self, ran)
        self.assertEqual([(True, "detector")], [(c["followed"], c["by"]) for c in ran.hooks.of("task.target_checked")])

    def test_a_bin_taken_away_since_it_was_found_is_lost_at_its_look_and_the_part_goes_back(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        hooks = RecordingHooks(on_event=lambda name, _d: scene.take_away() if name == "task.part_started" else None)
        ran = _carry(scene, hooks=hooks)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual({"to_look": look_label(L1)}, ran.hooks.of("task.carry_started")[-1])
        self.assertEqual(["task.target_lost", "task.put_back", "task.return_started", "task.returned",
                          "task.part_finished"], ran.names()[-5:])
        self.assertEqual(2, do0_changes(ran.log))
        self.assertNotIn(("hold_again",), ran.log)

    def test_a_way_the_picks_looks_did_not_see_sends_the_part_via_the_look(self) -> None:
        _via_the_look(self, _carry(looks=(L1,)))

    def test_a_line_refused_before_anything_was_sent_sends_the_part_via_the_look_from_where_it_stands(self) -> None:
        def refuse_the_carry(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "move" and getattr(target, "label", "") == "over the yellow bin":
                return MotionStatus.SELF_COLLISION_REJECTED
            return None

        ran = _carry(refuse=refuse_the_carry)

        _via_the_look(self, ran)
        first = ran.after("task.carry_started")
        self.assertEqual(("move", (*PART_XY, STANDOFF_Z), True), motions(first)[0], "the part did not go up first")
        self.assertIn(("refused", "move", "self_collision_rejected"), first)
        self.assertEqual(("joints", _key(L1)), motions(first)[1], "the look was not carried to from where the arm stood")

    def test_a_carry_that_failed_once_sent_stops_where_the_arm_stands_with_the_part(self) -> None:
        from src.robot.execution.task import TaskStop

        def fail_the_carry(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "move" and getattr(target, "label", "") == "over the yellow bin":
                return MotionStatus.CONTROLLER_REJECTED
            return None

        ran = _carry(refuse=fail_the_carry)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(ran.log))
        self.assertNotIn("task.place_started", ran.names())
        self.assertEqual(("forget",), [e for e in ran.log if e in (("forget",), ("hold_again",))][-1])


class TheTasksOwnWordTests(unittest.TestCase):
    def test_the_tasks_carry_overrides_the_cells(self) -> None:
        from src.robot.execution.task import TaskOptions

        over = _carry(carry="via_the_look", options=TaskOptions(carry="over_the_rim"))
        self.assertNotIn(("joints", _key(L1)), _carried(over))
        _via_the_look(self, _carry(carry="over_the_rim", options=TaskOptions(carry="via_the_look")))

    def test_a_carry_no_task_knows_is_refused(self) -> None:
        from src.robot.execution.task import TaskOptions

        with self.assertRaises(ValueError):
            TaskOptions(carry="sideways")  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
