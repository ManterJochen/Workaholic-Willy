"""A campaign puts each lifted part back where it was grasped, and says where the object was and where the tool closed.

The owner, 2026-09-24: a campaign of picks on one part, run many times, with the program printing what was computed.
So a pick that succeeded is followed by a place at the pose the tool closed at (a planned move to the standoff, a line
in, the release, a line out), and each attempt carries the object's centre in BASE and that grasp pose. A part that
does not go back stops the campaign: the next pick would start with it in the hand.

The service is the real one on the recording desk arm, hand and camera of
``test_a_pick_looks_from_the_joints_its_program_declares``, so the sequence is read off the arm's own log.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.execution.handling import HandlingOutcome, HandlingReport, HandlingVerb
from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
from tests.test_a_pick_looks_from_the_joints_its_program_declares import GRASP_MM, cell

#: The policy's own standoff, which a put back reuses: the pick's default of 80 mm, straight up from a top-down grasp.
_STANDOFF = (GRASP_MM[0], GRASP_MM[1], GRASP_MM[2] + 80.0)
_LIFT = (GRASP_MM[0], GRASP_MM[1], GRASP_MM[2] + 100.0)


class APartGoesBackWhereItWasGraspedTests(unittest.TestCase):
    def test_after_the_lift_the_part_is_placed_at_the_pose_the_tool_closed_at(self) -> None:
        events: list[Any] = []
        run = PickRun.from_service(cell(events, [True], wrist=False), runs=1, recording=Recording.off(),
                                   put_back=True).execute()

        self.assertEqual(1, run.succeeded, run.render())
        attempt = run.attempts[0]
        self.assertIsNotNone(attempt.put_back)
        self.assertTrue(attempt.put_back.ok, attempt.put_back.render())
        # The pick: pre-open, standoff, line to the grasp, close, lift. Then the put back.
        lifted = events.index(("move", _LIFT, True))
        self.assertEqual([("move", _STANDOFF, False), ("move", GRASP_MM, True), ("hand", 50.0),
                          ("move", _STANDOFF, True)], events[lifted + 1:])
        self.assertIn("put back: executed", attempt.render())

    def test_one_part_serves_the_whole_campaign(self) -> None:
        events: list[Any] = []
        run = PickRun.from_service(cell(events, [True], wrist=False), runs=3, recording=Recording.off(),
                                   put_back=True).execute()

        self.assertEqual(3, run.succeeded)
        self.assertEqual(0, run.exit_code)
        self.assertTrue(all(a.put_back is not None and a.put_back.ok for a in run.attempts))
        self.assertEqual(6, events.count(("hand", 50.0)), "each pick pre-opens once and each put back opens once")

    def test_nothing_is_put_back_after_a_pick_that_did_not_lift_a_part(self) -> None:
        events: list[Any] = []
        run = PickRun.from_service(cell(events, [False], wrist=False), runs=2, recording=Recording.off(),
                                   put_back=True).execute()

        self.assertEqual(0, run.succeeded)
        self.assertEqual([None, None], [a.put_back for a in run.attempts])
        self.assertFalse(any(isinstance(e, tuple) and e[0] in ("move", "hand") for e in events))

    def test_a_campaign_without_put_back_places_nothing(self) -> None:
        events: list[Any] = []
        run = PickRun.from_service(cell(events, [True], wrist=False), runs=1, recording=Recording.off()).execute()

        self.assertIsNone(run.attempts[0].put_back)
        self.assertEqual(("move", _LIFT, True), events[-1])


class _NotPutBack:
    """A service whose picks succeed and whose put back is refused."""

    def __init__(self) -> None:
        self.picks = 0

    def pick(self) -> Any:
        self.picks += 1
        return type("R", (), {"outcome": "succeeded", "fault": None})()

    def put_back(self, report: Any) -> HandlingReport:
        return HandlingReport(verb=HandlingVerb.PLACE, outcome=HandlingOutcome.MOTION_REFUSED,
                              message="the line into the grasp meets the camera world")


class APartNotPutBackStopsTheCampaignTests(unittest.TestCase):
    def test_the_next_pick_does_not_start_with_the_part_in_the_hand(self) -> None:
        service = _NotPutBack()
        run = PickRun.from_service(service, runs=4, recording=Recording.off(), put_back=True).execute()

        self.assertEqual(1, service.picks)
        self.assertEqual([PickOutcome.SUCCEEDED] + [PickOutcome.CANCELLED] * 3, [a.outcome for a in run.attempts])
        self.assertIn("was not put back (motion_refused: the line into the grasp meets the camera world)", run.error)
        self.assertEqual(1, run.exit_code)
        self.assertIn("REFUSED", run.render())

    def test_a_put_back_that_raised_stops_the_campaign_with_its_report(self) -> None:
        """Not raised out of the campaign: its report, the attempts so far, and the cell's teardown still come back."""
        class _Raises(_NotPutBack):
            def put_back(self, report: Any) -> HandlingReport:
                raise ConnectionError("the controller hung up during the line out")

        service = _Raises()
        run = PickRun.from_service(service, runs=3, recording=Recording.off(), put_back=True).execute()

        self.assertEqual(1, service.picks)
        self.assertIsNone(run.attempts[0].put_back)
        self.assertIn("was not put back (raised ConnectionError: the controller hung up", run.error)
        self.assertEqual(2, run.cancelled)
        self.assertEqual(1, run.exit_code)

    def test_the_service_refuses_to_put_back_what_no_pick_lifted(self) -> None:
        events: list[Any] = []
        service = cell(events, [False], wrist=False)
        placed = service.put_back(service.pick())

        self.assertIs(HandlingOutcome.REFUSED, placed.outcome)
        self.assertIn("did not succeed", placed.message)
        self.assertFalse(any(isinstance(e, tuple) and e[0] in ("move", "hand") for e in events))


class AnAttemptSaysWhatWasComputedTests(unittest.TestCase):
    def test_the_object_centre_and_the_grasp_pose_reach_the_attempt(self) -> None:
        """The part sits 500 mm below the fixed camera, whose frame puts it at base (100, -500, 100)."""
        events: list[Any] = []
        seen: list[Any] = []
        run = PickRun.from_service(cell(events, [True], wrist=False), runs=1, recording=Recording.off(),
                                   on_attempt=seen.append).execute()

        attempt = seen[0]
        self.assertIsNotNone(attempt.object_mm)
        np.testing.assert_allclose(attempt.object_mm, (100.0, -500.0, 100.0), atol=1.0)
        self.assertIsInstance(attempt.grasp_pose, Pose)
        np.testing.assert_allclose(attempt.grasp_pose.position_mm, GRASP_MM)
        # Straight down, closing along base -Y: the tool's +Z is base -Z and its +X base -Y.
        rotation = attempt.grasp_pose.to_matrix()[:3, :3]
        np.testing.assert_allclose(rotation[:, 2], (0.0, 0.0, -1.0), atol=1e-9)
        np.testing.assert_allclose(rotation[:, 0], (0.0, -1.0, 0.0), atol=1e-9)
        self.assertIn("object seen at (100.0, -500.0, 100.0) mm BASE", attempt.render())
        self.assertIn("grasp closed at (-130.0, -600.0, 80.0) mm BASE", attempt.render())
        report = run.last
        self.assertEqual(attempt.object_mm, report.object_centre_mm)
        self.assertEqual(attempt.grasp_pose, report.grasp_pose)
        self.assertEqual(list(GRASP_MM), report.to_dict()["grasp_pose"]["position_mm"])
        self.assertEqual(3, len(run.to_dict()["attempts"][0]["object_mm"]))

    def test_a_pick_that_found_nothing_says_neither(self) -> None:
        run = PickRun.from_service(cell([], [False], wrist=False), runs=1, recording=Recording.off()).execute()

        self.assertIsNone(run.attempts[0].object_mm)
        self.assertIsNone(run.attempts[0].grasp_pose)
        self.assertIsNone(run.last.object_centre_mm)
        self.assertIsNone(run.last.grasp_pose)

    def test_a_double_that_answers_every_attribute_leaves_the_attempt_as_it_was(self) -> None:
        from unittest.mock import MagicMock

        class _Service:
            def pick(self) -> Any:
                report = MagicMock()
                report.outcome = "succeeded"
                report.fault = None
                return report

        attempt = PickRun.from_service(_Service(), runs=1, recording=Recording.off()).execute().attempts[0]
        self.assertEqual((None, None, ()), (attempt.object_mm, attempt.grasp_pose, attempt.looks))


if __name__ == "__main__":
    unittest.main()
