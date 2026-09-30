"""A push and the pick after it leave the toggle alone: the real hand and the real execution policy (review of B2).

The owner's Hand-E is ``jaw_io`` single_toggle on tool DO0: every change of DO0 moves the jaws once, nothing reads them
back, and the program keeps the count. A push runs with the jaws standing open and never switches them (the owner,
2026-09-29); the pick of the pushed part then closes them once, with the one change the real ``GraspExecutionPolicy``
sends, and asks nobody. The end-to-end file (``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py``) drives a policy
double that executes nothing, so its re-pick never meets the hand. Here the re-pick runs on the real policy and the real
``JawIOGripper``, on a recording tool I/O whose writes share one log with the arm's motions, so the order of the push's
legs and of the one close can be read off it.

Pinned:

* through ``PickRun`` and through the console's run body: from the connect on, exactly one write to DO0, the close,
  after the push's last leg and after the re-pick's approach; the count at one command and still vouched for; nobody
  asked; the console run stops before its second pick on the part the jaws hold, asking nobody;
* the same with DO0 standing HIGH with the jaws open at the connect, as the owner leaves it after opening them at the
  pendant: the one write is the change from HIGH to LOW, the close. With the policy double, a push and a push that
  stopped write nothing to that DO0 at all, not even the level it stands at;
* a count that says closed never pushes and never asks, with ``next_target`` allowed as the owner's push config allows
  it: the pick goes on to next_target, which asks nobody either; in ``PickRun`` the next pick's start asks once; the
  console run ends before its next pick with the hand's sentence, asking nobody;
* a count nobody can vouch for (DO0 switched at the pendant) never pushes and never asks either, and ends the pick as a
  gripper fault (the owner, 2026-09-30): no next_target, no look driven again; ``PickRun`` and the console run stop on
  it, asking nobody.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from src.robot.core import MotionStatus
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.execution.pick_run import PickOutcome
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from tests._wrist_views import joints_key
from tests.test_a_boxed_in_part_is_pushed_inside_the_pick import LOOKS, PUSH_LABELS, _PushCell, _PushingArm, _refused
from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _changes


class _SharedLogArm(_PushingArm):
    """The pushing arm, writing every typed move (``("move", label)``) and joint move (``("joints", key)``) onto the log
    the tool I/O writes DO0 onto (``shared``), and at rest whenever the policy waits for it."""

    shared: list[Any] | None = None

    def move(self, pose: Any, **keywords: Any) -> Any:
        if self.shared is not None:
            self.shared.append(("move", str(pose.label)))
        return super().move(pose, **keywords)

    def move_to_joints(self, joints: Any, **keywords: Any) -> Any:
        if self.shared is not None:
            self.shared.append(("joints", joints_key(joints)))
        return super().move_to_joints(joints, **keywords)

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        return True


def _real_cell(test: unittest.TestCase, **keywords: Any) -> _PushCell:
    """The owner's dense_clutter cell with the real execution policy on the toggle, one log for the arm and DO0."""
    cell = _PushCell(test, arm=_SharedLogArm, **keywords)
    cell.arm.shared = cell.events
    cell.policy = GraspExecutionPolicy(arm=cell.arm, gripper=cell.jaws, pre_open_width_mm=49.99,  # type: ignore[assignment]
                                       require_steady_before_motion=True)
    cell.orchestrator.policy = cell.policy
    return cell


def _writes(cell: _PushCell) -> list[Any]:
    """Every write to DO0 since the connect, changed or not."""
    return [event for event in cell.events[cell.connected_writes:]
            if isinstance(event, tuple) and event[:2] == ("DO", 0)]


class ThePickAfterAPushClosesTheJawsOnceTests(unittest.TestCase):
    def _assert_one_close_after_the_push_and_the_approach(self, cell: _PushCell, *, closes_to: bool) -> None:
        tail = cell.events[cell.connected_writes:]
        (write,) = _writes(cell)
        self.assertEqual(("DO", 0, closes_to), tuple(write), "the one write is not the close")
        self.assertEqual([tail.index(write)], _changes(tail), "DO0 changed other than by the close")
        moves = [event[1] for event in tail if isinstance(event, tuple) and event[0] == "move"]
        self.assertEqual(PUSH_LABELS, [label for label in moves if label.startswith("push_")])
        up = tail.index(("move", "push_up"))
        approach = max(index for index, event in enumerate(tail)
                       if isinstance(event, tuple) and event[0] == "move" and str(event[1]).startswith("approach"))
        self.assertLess(up, approach, "the re-pick's approach did not follow the push")
        self.assertLess(approach, tail.index(write), "DO0 was written before the re-pick's approach")
        self.assertEqual(1, cell.jaws.commands_sent)
        self.assertTrue(cell.jaws.jaws_closed)
        self.assertFalse(cell.jaws.edge_unknown)
        self.assertEqual("", cell.jaws.why_jaws_unknown())
        self.assertEqual([], cell.nobody.asked, "a person was asked")

    def test_a_pick_run_pushes_the_part_and_closes_the_jaws_once_on_it(self) -> None:
        for high in (False, True):
            with self.subTest(do0_high_at_the_connect=high):
                cell = _real_cell(self, do0_high=high)

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign()

                self.assertIs(AutonomousGraspOutcome.SUCCEEDED, run.last.outcome, run.render())
                self._assert_one_close_after_the_push_and_the_approach(cell, closes_to=not high)

    def test_a_console_run_pushes_closes_once_and_stops_on_the_part_it_holds_asking_nobody(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = _real_cell(self)
        cell.service.configured_looks = LOOKS

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run, _hub = _drive(cell.service, picks=2)

        self.assertEqual((1, 1), (run.attempted, run.succeeded))
        self.assertIs(RunState.FAILED, run.state)
        self.assertIn("the jaws still hold the last part", run.error)
        self._assert_one_close_after_the_push_and_the_approach(cell, closes_to=True)


class APushWritesNothingToDo0StandingHighTests(unittest.TestCase):
    """DO0 left HIGH with the jaws open at the connect: a write of either level there is one nobody may send, and a
    stray LOW would move the jaws. The push and a push that stopped write nothing, not even the level DO0 stands at."""

    def test_a_push_and_a_push_that_stopped_write_nothing_to_do0(self) -> None:
        for name, answers in (("pushed", {}),
                              ("stopped", {"push_back": _refused(MotionStatus.WORKSPACE_REJECTED, "leaves the box")})):
            with self.subTest(push=name):
                cell = _PushCell(self, do0_high=True, answers=answers)

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign(runs=2)

                self.assertEqual(name == "stopped", run.last.needs_person, run.render())
                self.assertTrue(cell.jaws._io.do[0], "DO0 no longer stands HIGH")  # noqa: SLF001
                cell.assert_the_jaws_untouched(self)


class AJawCountNobodyVouchesForNeverPushesTests(unittest.TestCase):
    """A count that says closed (the one change a close sends, made as the last look is ranked): no push and no
    question; the pick goes on to next_target, which asks nobody either (the owner's rule: fall through to rescan), and
    the next pick's start is where the hand is asked about. A count nobody can vouch for (DO0 switched at the pendant
    then): no push and no question either, and the pick ends there as a gripper fault, as the toggle's own refusal ends
    one (the owner, 2026-09-30): nothing more moves, and ``PickRun`` and the console run stop on it."""

    @staticmethod
    def _spoiled_at_the_last_look(test: unittest.TestCase, kind: str) -> _PushCell:
        """The cell, whose count is spoiled as the last look is ranked: ``cell.spoiled_at`` is where the log stood
        then, the close's own write included, and nothing may write DO0 after it."""
        holder: dict[str, Any] = {}

        def spoil(number: int) -> None:
            if number != 2 or holder.get("done"):
                return
            holder["done"] = True
            cell = holder["cell"]
            if kind == "closed":
                cell.jaws.set_closed(True)
            else:
                cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # noqa: SLF001 (the pendant)
            cell.spoiled_at = len(cell.events)

        cell = _PushCell(test, on_frame=spoil)
        holder["cell"] = cell
        return cell

    @staticmethod
    def _written_since_the_spoil(cell: _PushCell) -> list[Any]:
        return [event for event in cell.events[cell.spoiled_at:]  # type: ignore[attr-defined]
                if isinstance(event, tuple) and event[:2] == ("DO", 0)]

    @staticmethod
    def _recorded(cell: _PushCell) -> list[Any]:
        """Every report the service's pick returns, in order."""
        reports: list[Any] = []
        pick = cell.service.pick

        def recorded(**keywords: Any) -> Any:
            reports.append(pick(**keywords))
            return reports[-1]

        cell.service.pick = recorded  # type: ignore[method-assign]
        return reports

    def test_a_pick_run_pushes_nothing_asks_nobody_in_the_pick_and_asks_once_at_the_next_pick(self) -> None:
        cell = self._spoiled_at_the_last_look(self, "closed")
        person = Person("open")
        cell.jaws._ask = person  # noqa: SLF001
        asked_after: list[int] = []
        reports = self._recorded(cell)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign(runs=2, on_attempt=lambda _attempt: asked_after.append(len(person.asked)))

        self.assertEqual(2, len(run.attempts), run.render())
        self.assertEqual([], cell.arm.pushes, "a push moved on jaws the count says closed")
        self.assertEqual("refused_jaws_closed", reports[0].telemetry["pushes"][0]["code"])
        self.assertEqual(["next_target"], reports[0].telemetry["recovery_trail_actions"])
        self.assertEqual([0, 1], asked_after, "asked in the first pick, or not once at the second's start")
        self.assertIn("CLOSED", person.asked[0])
        self.assertEqual([], self._written_since_the_spoil(cell), "DO0 was written")
        self.assertEqual(_keys_of(LOOKS) * 3, [key for kind_, key in cell.arm.motions if kind_ == "joints"],
                         "the looks were not what next_target and the next pick drive")

    def test_a_pick_run_on_a_count_nobody_vouches_for_stops_at_the_gripper_fault_asking_nobody(self) -> None:
        cell = self._spoiled_at_the_last_look(self, "unknown")
        person = Person()
        cell.jaws._ask = person  # noqa: SLF001 (any question fails the test)
        reports = self._recorded(cell)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign(runs=2)

        self.assertEqual(1, len(reports), "a second pick started on a hand nobody vouches for")
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [attempt.outcome for attempt in run.attempts],
                         run.render())
        self.assertIn("the gripper needs a person, so the campaign stops", run.attempts[0].detail)
        report = reports[0]
        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertIn("switched", report.gripper_fault)
        self.assertEqual([], cell.arm.pushes, "a push moved on jaws nobody vouched open")
        self.assertEqual("refused_jaws_unknown", report.telemetry["pushes"][0]["code"])
        self.assertEqual([], report.telemetry["recovery_trail_actions"], "next_target drove the looks again")
        self.assertEqual([], person.asked, "a person was asked")
        self.assertEqual([], self._written_since_the_spoil(cell), "DO0 was written")
        self.assertEqual(_keys_of(LOOKS), [key for kind_, key in cell.arm.motions if kind_ == "joints"],
                         "the arm moved after the push found the count unknown")

    def test_a_console_run_pushes_nothing_and_stops_before_its_next_pick_asking_nobody(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        # A count that says closed: the looks of the pick and of its next_target. One nobody vouches for: the pick's.
        for kind, said, looks in (("closed", "the jaws still hold the last part", 2),
                                  ("unknown", "the gripper needs a person, so the run stops", 1)):
            with self.subTest(count=kind):
                cell = self._spoiled_at_the_last_look(self, kind)
                cell.service.configured_looks = LOOKS

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run, _hub = _drive(cell.service, picks=2)

                self.assertEqual([], cell.arm.pushes, "a push moved on jaws nobody vouched open")
                self.assertIs(RunState.FAILED, run.state)
                self.assertEqual(1, run.attempted, "the second pick started on a hand nobody vouches for")
                self.assertIn(said, run.error)
                self.assertEqual([], cell.nobody.asked, "a person was asked")
                self.assertEqual([], self._written_since_the_spoil(cell), "DO0 was written")
                self.assertEqual(_keys_of(LOOKS) * looks, [key for kind_, key in cell.arm.motions if kind_ == "joints"],
                                 "the looks were not what the pick drives")


def _keys_of(looks: Any) -> list[tuple[float, ...]]:
    return [joints_key(look) for look in looks]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
