"""Before its first pick a task looks for its bin from its looks, and keeps the first look that sees it (Q13 (a)).

A caption of two phrases lets the label mapping turn "blue bin" into "blue cube" (``realsense_source.py``), so a pick's
own looks ground only the part. The bin is found first, by a survey (``survey``): the arm goes to each of the task's
looks in turn through its own judged verb (``move_to_look``), and at each the camera locates the bin's phrase, then
the part's, which only counts the parts it sees for the chat. The first look whose bin has at least 20 points of surface
is the look the bin is kept from (L_T), and the survey ends there; of several bins seen there the most confident is
kept (Q5). A look the planner refused before anything was sent is skipped and said; a look motion that may have moved
the arm, or a controller that stopped, ends the survey where the arm stands, with nothing else commanded. Before every
motion the task is asked whether it may still move (halt, disconnect). A fixed camera locates once, where it stands,
and moves nothing.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import JointPositions, MotionStatus
from src.robot.execution.looks import look_label
from tests._task_fakes import (
    BIN_CENTRE,
    TaskArm,
    ScriptedLocator,
    bin_object,
    bin_points,
    motions,
)
from src.geometry import Pose

L1 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(150.0, -650.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(400.0, -300.0, 450.0, label="L2")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _arm(log: list[Any], **keywords: Any) -> TaskArm:
    return TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2}, **keywords)


def _bin_seen_from_l2(tcp: Pose) -> list[Any]:
    """The bin stands where only the second look sees it."""
    return [bin_object()] if tcp is AT_L2 else []


def _parts(tcp: Pose) -> list[Any]:
    part = bin_object(bin_points((150.0, -650.0), (40.0, 40.0), 40.0, inside=False), label="red cube", score=0.7)
    inside = bin_object(bin_points(BIN_CENTRE, (40.0, 40.0), 40.0, inside=False), label="red cube", score=0.7)
    return [part, inside]


class AWristSurveyTests(unittest.TestCase):
    def test_the_first_look_that_sees_the_bin_keeps_it_and_the_survey_ends_there(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = _arm(log)
        locator = ScriptedLocator({"blue bin": _bin_seen_from_l2, "red cube": _parts}, arm=arm, log=log)

        found = survey(arm, [locator], "blue bin", object_phrase="red cube", looks=(L1, L2, L1))

        self.assertEqual([("joints", _key(L1)), ("joints", _key(L2))], motions(log), "the survey went on past the bin")
        assert found.kept is not None
        self.assertEqual(look_label(L2), found.kept.look_label)
        self.assertIs(L2, found.kept.look)
        self.assertEqual((look_label(L1), look_label(L2)), found.looks_tried)
        self.assertEqual(["blue bin", "red cube", "blue bin", "red cube"], locator.asked,
                         "the bin's phrase first, then the part's, at each look")
        self.assertEqual(1, found.parts_seen, "a part standing in the bin was counted as one to pick")

    def test_of_several_bins_the_most_confident_is_kept(self) -> None:
        from src.robot.execution.place_target import survey

        arm = _arm([])
        sure = bin_object(bin_points((450.0, -200.0)), score=0.9)
        locator = ScriptedLocator({"blue bin": [bin_object(score=0.6), sure, bin_object(score=None)]}, arm=arm)

        found = survey(arm, [locator], "blue bin", object_phrase="", looks=(L1,))

        assert found.kept is not None
        self.assertEqual(0.9, found.kept.score)
        self.assertIsNone(found.parts_seen, "an empty object phrase counts nothing")

    def test_a_look_refused_before_anything_was_sent_is_skipped_and_said(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = _arm(log, refuse=lambda kind, index, target: MotionStatus.WORKSPACE_REJECTED if index == 0 else None)
        locator = ScriptedLocator({"blue bin": _bin_seen_from_l2}, arm=arm)

        found = survey(arm, [locator], "blue bin", object_phrase="", looks=(L1, L2))

        assert found.kept is not None
        self.assertEqual((look_label(L2),), found.looks_tried)
        self.assertEqual(look_label(L1), found.refused[0][0])
        self.assertIsNone(found.stopped)

    def test_a_look_motion_that_may_have_moved_the_arm_ends_the_survey_there(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = _arm(log, refuse=lambda kind, index, target: MotionStatus.CONTROLLER_REJECTED if index == 1 else None)
        locator = ScriptedLocator({"blue bin": _bin_seen_from_l2}, arm=arm)

        found = survey(arm, [locator], "blue bin", object_phrase="", looks=(L1, L2))

        self.assertIsNone(found.kept)
        assert found.stopped is not None
        self.assertEqual(look_label(L2), found.stopped_look)
        self.assertEqual(1, len(motions(log)))

    def test_no_bin_from_any_look_keeps_nothing_and_names_every_look_tried(self) -> None:
        from src.robot.execution.place_target import survey

        arm = _arm([])
        found = survey(arm, [ScriptedLocator({}, arm=arm)], "blue bin", object_phrase="", looks=(L1, L2))

        self.assertIsNone(found.kept)
        self.assertEqual((look_label(L1), look_label(L2)), found.looks_tried)

    def test_a_task_that_may_not_move_any_more_is_not_moved_by_its_survey(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = _arm(log)
        asked: list[bool] = []

        def may_move() -> str:
            asked.append(True)
            return "" if len(asked) == 1 else "halt now was pressed"

        found = survey(arm, [ScriptedLocator({}, arm=arm)], "blue bin", object_phrase="", looks=(L1, L2),
                       may_move=may_move)

        self.assertEqual([("joints", _key(L1))], motions(log))
        self.assertEqual("halt now was pressed", found.halted)
        self.assertIsNone(found.kept)


class AFixedSurveyTests(unittest.TestCase):
    def test_a_fixed_camera_locates_once_where_it_stands_and_moves_nothing(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = _arm(log)
        locator = ScriptedLocator({"blue bin": [bin_object()]}, wrist=False, rig_id="overhead")

        found = survey(arm, [locator], "blue bin", object_phrase="", looks=(L1, L2))

        self.assertEqual([], motions(log))
        assert found.kept is not None
        self.assertIsNone(found.kept.look)
        self.assertIsNone(found.kept.look_label)
        self.assertEqual("overhead", found.kept.camera)
        self.assertEqual(["blue bin"], locator.asked)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
