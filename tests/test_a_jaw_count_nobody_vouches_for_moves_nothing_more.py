"""A hand nobody can vouch for moves nothing more: the owner's decisions of 2026-09-30 after final part 1.

The owner's cell in miniature (``tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py``): the UR10 judging every
line, the Hand-E as ``jaw_io`` single_toggle on tool DO0 (the real :class:`JawIOGripper` on a recording tool I/O,
connected once with a person saying "open", and nobody asked after that), the tilted wrist D415 and three declared looks.
DO0 switched at the pendant is the tool I/O's level flipped behind the program's back, as a hand on the pendant does.

Pinned:

* before ``next_target`` or a rescan drives the looks again (a wrist pick handed looks), the toggle's count is read
  (``why_toggle_count_unknown``, nothing asked or sent): one nobody can vouch for ends the pick as a gripper fault,
  with no look driven again and nothing moved, and ``PickRun`` and the console run stop on it; the report and its
  record say the re-pick never ran (``re_pick_refused``). A push refused before it read the jaws (no neighbour within
  25 mm) therefore never hands the looks a hand nobody vouches for. A re-pick that drives no look, a fixed camera's or
  that of a wrist pick handed no look, reads nothing more: its own check before the arm moves asks where the jaws
  stand, as at every pick start;
* during a push, before each contact leg and once the up leg ended, the count is read again: one nobody can vouch for
  (DO0 switched while the push drove) stops the push where the arm stands, with no further motion, not even back to
  the look; the report needs a person, ``PickRun`` and the console run stop, and the service starts no further pick
  until a person decides;
* a gripper that measures its width (the Robotiq socket driver) found not connected before a push, or whose width
  read fails there (a socket that stopped answering), ends the pick as a gripper fault, as a toggle nobody vouches for
  does, and the campaign stops; a connected one whose jaws read closed is a plain refusal the pick falls through on,
  as before.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from src.config.schema.robot.robot_schema import GripperConfig
from src.geometry import Frame, Transform
from src.robot.core import RobotError
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, GraspMode
from src.robot.execution.autonomous_grasp.record_logging import to_attempt_record
from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
from src.robot.grasping.loop.pick_loop import PickOutcome as LoopOutcome
from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
from src.robot.grippers.robotiq import GripperController
from src.robot.grippers.robotiq_socket import RobotiqSocketError
from tests._wrist_views import CUBE, Box, mount
from tests.test_a_boxed_in_part_is_pushed_inside_the_pick import (
    LOOKS,
    PUSH_LABELS,
    _PushCell,
    _PushingArm,
)
from tests.test_a_push_and_its_re_pick_leave_the_toggle_alone import _real_cell
from tests.test_a_toggle_asks_where_its_jaws_stand import Person
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_PLUS_Y, _keys

#: A post 30 mm off the part: its collisions are not a neighbour's doing, so no push is planned, and the push never
#: reaches its read of the jaws.
FAR_POST = Box((-80.0, -722.0, 0.0), (-50.0, -714.0, 30.0), "post")


def _spoil(cell: _PushCell, kind: str) -> None:
    """Nobody can vouch for the count from here on: ``switched`` flips DO0 as the pendant would, ``unreadable`` makes
    every read of its level fail. ``cell.spoiled_at`` is where the log stood then: nothing may write DO0 after it."""
    io = cell.jaws._io  # noqa: SLF001 (the tool I/O the hand switches)
    if kind == "switched":
        io.do[0] = not io.do.get(0, False)
    else:
        def unreadable(pin: int, **_keywords: Any) -> bool:
            raise RobotError(f"the controller did not answer the read of tool output {pin}")

        io.get_digital_output = unreadable
    cell.spoiled_at = len(cell.events)  # type: ignore[attr-defined]


def _written_since_the_spoil(cell: _PushCell) -> list[Any]:
    return [event for event in cell.events[cell.spoiled_at:]  # type: ignore[attr-defined]
            if isinstance(event, tuple) and event[:2] == ("DO", 0)]


def _said(kind: str) -> str:
    return "somebody switched it" if kind == "switched" else "could not be read"


def _recorded(cell: _PushCell) -> list[Any]:
    """Every report the service's pick returns, in order."""
    reports: list[Any] = []
    pick = cell.service.pick

    def recorded(**keywords: Any) -> Any:
        reports.append(pick(**keywords))
        return reports[-1]

    cell.service.pick = recorded  # type: ignore[method-assign]
    return reports


def _spoiled_at_the_last_look(test: unittest.TestCase, kind: str, **keywords: Any) -> _PushCell:
    """The cell, whose count nobody can vouch for from the moment the last look is ranked."""
    holder: dict[str, Any] = {}

    def spoil(number: int) -> None:
        if number == 2 and not holder.get("done"):
            holder["done"] = True
            _spoil(holder["cell"], kind)

    cell = _PushCell(test, on_frame=spoil, **keywords)
    holder["cell"] = cell
    return cell


class BeforeNextTargetOrARescanPicksAgainTests(unittest.TestCase):
    """The push was refused before it read the jaws (no neighbour within 25 mm of the part), the count went unknown at
    the last look, and the pick falls through to next_target, which would drive the looks again: its read of the count
    ends the pick there."""

    def test_next_target_ends_as_a_gripper_fault_before_it_drives_a_look(self) -> None:
        for kind in ("switched", "unreadable"):
            with self.subTest(count=kind):
                cell = _spoiled_at_the_last_look(self, kind, boxes=(CUBE, FAR_POST))
                reports = _recorded(cell)

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign(runs=2)

                self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "next_target drove a look on a hand nobody vouches for")
                self.assertEqual(3, len(cell.camera.taken))
                self.assertEqual([], cell.arm.pushes)
                self.assertEqual(1, len(reports), "a second pick started on a hand nobody vouches for")
                self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts],
                                 run.render())
                self.assertIn("the gripper needs a person, so the campaign stops", run.attempts[0].detail)
                report = reports[0]
                self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
                self.assertIn("next_target", report.gripper_fault)
                self.assertIn("nobody can say where the jaws stand", report.gripper_fault.lower())
                self.assertIn(_said(kind), report.gripper_fault)
                self.assertFalse(report.needs_person)
                self.assertEqual(["next_target"], report.telemetry["recovery_trail_actions"])
                self.assertEqual("no_blocking_neighbour", report.telemetry["pushes"][0]["code"])
                self.assertIn("gripper    needs a person:", report.render())
                self.assertEqual([], cell.nobody.asked, "a person was asked")
                self.assertEqual([], _written_since_the_spoil(cell), "DO0 was written")
                self.assertEqual(0, cell.jaws.commands_sent)
                self.assertEqual([], cell.policy.executed)

    def test_a_console_run_stops_on_it_asking_nobody(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = _spoiled_at_the_last_look(self, "switched", boxes=(CUBE, FAR_POST))
        cell.service.configured_looks = LOOKS

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run, _hub = _drive(cell.service, picks=2)

        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(1, run.attempted, "the run picked again on a hand nobody vouches for")
        self.assertIn("the gripper needs a person, so the run stops", run.error)
        self.assertIn("somebody switched it", run.error)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves())
        self.assertEqual([], cell.nobody.asked, "a person was asked")
        self.assertEqual([], _written_since_the_spoil(cell), "DO0 was written")

    def test_its_record_says_the_re_pick_never_ran(self) -> None:
        """The trail's last step names the next_target whose pick the hand refused: the report, and the record logged
        from it, say that pick never ran (``re_pick_refused``), beside ``stage`` ``before_the_re_pick``."""
        cell = _spoiled_at_the_last_look(self, "switched", boxes=(CUBE, FAR_POST))

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            report = cell.service.pick(look=list(LOOKS))

        self.assertEqual(("before_the_re_pick", "gripper_fault"),
                         (report.telemetry["stage"], report.telemetry.get("re_pick_refused")))
        self.assertEqual(["next_target"], [row["action"] for row in report.recovery_actions])
        record = to_attempt_record(report, attempt_id="the-re-pick-refused")
        self.assertEqual(("before_the_re_pick", "gripper_fault"),
                         (record.extra.get("stage"), record.extra.get("re_pick_refused")))

    def test_a_count_that_stands_lets_next_target_drive_the_looks_again(self) -> None:
        """The control: the count stands, and next_target runs the looks again for the part, as before."""
        cell = _PushCell(self, boxes=(CUBE, FAR_POST))

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            report = cell.service.pick(look=list(LOOKS))

        self.assertEqual(_keys(*LOOKS, *LOOKS), cell.joint_moves())
        self.assertEqual("", report.gripper_fault)
        self.assertEqual(["next_target"], report.telemetry["recovery_trail_actions"])
        cell.assert_the_jaws_untouched(self)


class ARePickThatDrivesNoLookReadsNothingMoreTests(unittest.TestCase):
    """The owner's words: the count is read before ``next_target`` or a rescan drives any look. A re-pick that drives
    none reads nothing more: a fixed camera's rescan, and the rescan of a wrist pick handed no look, perceive where the
    arm stands, and the pick's own check before the arm moves asks where the jaws stand, as at every pick start. The
    real execution policy on the toggle, DO0 switched at the pendant as the first frame is ranked, the part free on the
    second, and a person at the terminal who looks and says open."""

    def _cell(self) -> _PushCell:
        holder: dict[str, Any] = {}

        def spoil(number: int) -> None:
            if number == 0 and not holder.get("done"):
                holder["done"] = True
                _spoil(holder["cell"], "switched")

        cell = _real_cell(self, mode=GraspMode.AUTO, allowed=("rescan",), max_attempts=1, boxes=(CUBE,),
                          on_frame=spoil, freed_on=(1,))
        holder["cell"] = cell
        cell.person = Person("open")  # type: ignore[attr-defined]
        cell.jaws._ask = cell.person  # type: ignore[attr-defined]  # noqa: SLF001 (the person at the terminal)
        return cell

    def _assert_asked_at_the_approach_and_picked(self, cell: _PushCell, report: Any) -> None:
        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
        self.assertEqual("", report.gripper_fault)
        self.assertNotEqual("before_the_re_pick", report.telemetry.get("stage"))
        self.assertNotIn("re_pick_refused", report.telemetry)
        self.assertEqual(["rescan"], report.telemetry["recovery_trail_actions"])
        self.assertEqual(2, len(cell.camera.taken), "the rescan did not perceive again")
        (question,) = cell.person.asked  # type: ignore[attr-defined]
        self.assertIn("somebody switched it", question)
        self.assertEqual(1, cell.jaws.commands_sent, "the jaws did not close once on the part")

    def test_a_fixed_camera_s_rescan_asks_at_the_approach(self) -> None:
        cell = self._cell()
        fixed = Transform.from_matrix(cell.arm.get_tcp_pose().to_matrix() @ mount(), from_frame=Frame.CAMERA,
                                      to_frame=Frame.BASE)
        cell.orchestrator.frame_resolver = StaticCameraToBaseResolver(transform=fixed)
        self.assertFalse(cell.service.perceives_from_the_wrist)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = PickRun.from_service(cell.service, runs=1, recording=Recording.off()).execute()

        self.assertEqual([PickOutcome.SUCCEEDED], [a.outcome for a in run.attempts], run.render())
        self._assert_asked_at_the_approach_and_picked(cell, run.last)

    def test_the_rescan_of_a_wrist_pick_handed_no_look_asks_at_the_approach(self) -> None:
        cell = self._cell()
        cell.arm.move_to_joints(LOOK_PLUS_Y)  # where the arm stands; the pick is handed no look
        self.assertTrue(cell.service.perceives_from_the_wrist)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            report = cell.service.pick()

        self._assert_asked_at_the_approach_and_picked(cell, report)


class _SwitchedWhileItPushes(_PushingArm):
    """The pushing arm, DO0 switched at the pendant as the push motion ``switch_on`` names arrives."""

    switch_on = ""
    cell: Any = None

    def move(self, pose: Any, **keywords: Any) -> Any:
        result = super().move(pose, **keywords)
        if str(pose.label) == self.switch_on and self.cell is not None:
            _spoil(self.cell, "switched")
        return result


class BeforeEachContactLegOfAPushTests(unittest.TestCase):
    """DO0 switched at the pendant while the push drives: before its next contact leg the count is read, and the push
    stops where the arm stands. Nothing more moves, not even back to the look, and a person decides."""

    def _switched_during(self, motion: str) -> _PushCell:
        cell = _PushCell(self, arm=_SwitchedWhileItPushes)
        cell.arm.switch_on = motion  # type: ignore[attr-defined]
        cell.arm.cell = cell  # type: ignore[attr-defined]
        return cell

    def test_the_push_stops_before_its_next_contact_leg_where_the_arm_stands(self) -> None:
        for index, motion in enumerate(PUSH_LABELS[:-1]):
            with self.subTest(switched_during=motion):
                cell = self._switched_during(motion)

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign(runs=2)

                self.assertEqual(PUSH_LABELS[:index + 1], cell.arm.push_motions(), "a leg ran on a hand nobody vouches for")
                self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the arm was sent back to the look after the stop")
                self.assertEqual(3, len(cell.camera.taken), "a further pick looked again")
                self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts],
                                 run.render())
                self.assertIn("a recovery stopped where the arm stands", run.attempts[0].detail)
                report = run.last
                self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, report.outcome)
                self.assertTrue(report.needs_person)
                self.assertFalse(report.controller_stopped)
                self.assertIn("Nobody can say where the jaws stand", report.telemetry["push_stopped"])
                self.assertIn("somebody switched it", report.telemetry["push_stopped"])
                (push,) = report.telemetry["pushes"]
                self.assertEqual(("unsafe_recovery_refused", True, PUSH_LABELS[index + 1].removeprefix("push_")),
                                 (push["code"], push["stopped"], push["leg"]))
                self.assertEqual(1, len(cell.service.campaign.budgets.records), "a push that moved was not counted")
                self.assertEqual([], cell.nobody.asked, "a person was asked")
                self.assertEqual([], _written_since_the_spoil(cell), "DO0 was written")
                self.assertEqual(0, cell.jaws.commands_sent)
                self.assertEqual([], cell.policy.executed)

    def test_a_switch_on_the_up_leg_stops_the_push_at_the_lift_point(self) -> None:
        """DO0 switched while the up leg runs: once it ended the count is read, as the controller is, and the push
        stops at the lift point. The arm is not sent back to the look, nothing looks again, and a person decides."""
        cell = self._switched_during("push_up")

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign(runs=2)

        self.assertEqual(PUSH_LABELS, cell.arm.push_motions())
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "the arm was sent back to the look after the stop")
        self.assertEqual(3, len(cell.camera.taken), "the part was looked at again on a hand nobody vouches for")
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
        self.assertIn("a recovery stopped where the arm stands", run.attempts[0].detail)
        report = run.last
        self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, report.outcome)
        self.assertTrue(report.needs_person)
        self.assertFalse(report.controller_stopped)
        self.assertIn("After the up leg: Nobody can say where the jaws stand", report.telemetry["push_stopped"])
        self.assertIn("somebody switched it", report.telemetry["push_stopped"])
        (push,) = report.telemetry["pushes"]
        self.assertEqual(("unsafe_recovery_refused", True, "up"), (push["code"], push["stopped"], push["leg"]))
        self.assertEqual(1, len(cell.service.campaign.budgets.records), "a push that moved was not counted")
        self.assertEqual([], cell.nobody.asked, "a person was asked")
        self.assertEqual([], _written_since_the_spoil(cell), "DO0 was written")
        self.assertEqual(0, cell.jaws.commands_sent)
        self.assertEqual([], cell.policy.executed)

    def test_the_service_starts_no_further_pick_until_a_person_decides(self) -> None:
        cell = self._switched_during("push_push")
        first = cell.service.pick(look=list(LOOKS))
        self.assertTrue(first.needs_person, first.render())
        motions = list(cell.arm.motions)

        second = cell.service.pick(look=list(LOOKS))

        self.assertIs(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED, second.outcome)
        self.assertIsNone(second.pick_report)
        self.assertEqual(motions, cell.arm.motions, "the arm left where the push stopped it")
        self.assertIn("Nobody can say where the jaws stand", cell.service.stopped_where_the_arm_stands)

    def test_a_console_run_ends_on_it(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = self._switched_during("push_down")
        cell.service.configured_looks = LOOKS

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run, hub = _drive(cell.service, picks=2)

        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(1, run.attempted, "the run picked again after the push stopped")
        self.assertIn("a recovery stopped where the arm stands", run.error)
        self.assertEqual(PUSH_LABELS[:2], cell.arm.push_motions())
        (said,) = [event for event in hub.since(run.id, 0)[0]
                   if event.type == "pick.attempt_finished" and event.data.get("action") == "push"]
        self.assertIn("Nobody can say where the jaws stand", said.human)
        self.assertEqual([], cell.nobody.asked)


class _MeasuresItsWidth:
    """A parallel jaw that measures its width (as the Robotiq socket driver does), ``connected`` or not, its jaws
    reading ``width_mm`` open."""

    min_width_mm = 0.0
    max_width_mm = 50.0

    def __init__(self, *, connected: bool, width_mm: float = 50.0) -> None:
        self.is_connected = connected
        self.width_mm = width_mm

    def width_is_measured(self) -> bool:
        return True

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def activate(self) -> None:
        return None

    def set_width_mm(self, width_mm: float, *, speed: float | None = None, force: float | None = None) -> None:
        raise AssertionError("the jaws were commanded")

    def get_width_mm(self) -> float:
        if not self.is_connected:
            raise AssertionError("a gripper that is not connected was read")
        return self.width_mm


class _RobotiqSocket:
    """The URCap socket client as the Robotiq :class:`GripperController` drives it, the jaws fully open. ``dead`` makes
    every read time out, as a socket that stopped answering does, while the controller still counts itself connected
    (only the program's own disconnect ends that)."""

    def __init__(self) -> None:
        self.dead = False
        self.moved: list[Any] = []

    def connect(self, ip: str, port: int) -> None:
        return None

    def activate_if_needed(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_current_position(self) -> int:
        if self.dead:
            raise RobotiqSocketError("GET POS timed out after 2.0 s: the URCap socket went away")
        return 0

    def move(self, *arguments: Any) -> None:
        self.moved.append(arguments)
        raise AssertionError("the jaws were commanded")


class AWidthGripperFoundNotConnectedBeforeAPushTests(unittest.TestCase):
    def test_it_ends_the_pick_as_a_gripper_fault_and_the_campaign_stops(self) -> None:
        cell = _PushCell(self)
        cell.orchestrator.gripper = _MeasuresItsWidth(connected=False)

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign(runs=2)

        self.assertEqual([], cell.arm.pushes)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "next_target drove the looks again on a hand not connected")
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
        self.assertIn("the gripper needs a person, so the campaign stops", run.attempts[0].detail)
        report = run.last
        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertIs(LoopOutcome.GRIPPER_FAULT, report.pick_report.outcome)
        self.assertIn("not connected", report.gripper_fault)
        last = report.pick_report.attempts[-1]
        self.assertEqual(("push", "refused_jaws_unknown", ()), (last.action, last.push, last.reasons))
        self.assertEqual([], report.telemetry["recovery_trail_actions"], "a recovery ran after the fault")

    def test_a_robotiq_whose_socket_stopped_answering_ends_the_pick_the_same_way(self) -> None:
        """The Robotiq socket driver counts itself connected until the program disconnects it: a socket that stops
        answering makes its width read fail instead. That hand, too, nobody can read before the push."""
        socket = _RobotiqSocket()

        def went_away(number: int) -> None:
            if number == 2:
                socket.dead = True

        cell = _PushCell(self, on_frame=went_away)
        robotiq = GripperController(GripperConfig(min_width_mm=0.0, max_width_mm=50.0), ip="127.0.0.1",
                                    driver_factory=lambda: socket)
        robotiq.connect()
        cell.orchestrator.gripper = robotiq

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign(runs=2)

        self.assertTrue(robotiq.is_connected)
        self.assertEqual([], cell.arm.pushes)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves(), "next_target drove the looks again on a hand nobody can read")
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
        self.assertIn("the gripper needs a person, so the campaign stops", run.attempts[0].detail)
        report = run.last
        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome)
        self.assertIs(LoopOutcome.GRIPPER_FAULT, report.pick_report.outcome)
        self.assertIn("could not be read", report.gripper_fault)
        self.assertIn("GET POS timed out", report.gripper_fault)
        last = report.pick_report.attempts[-1]
        self.assertEqual(("push", "refused_jaws_unknown", ()), (last.action, last.push, last.reasons))
        self.assertEqual([], report.telemetry["recovery_trail_actions"], "a recovery ran after the fault")
        self.assertEqual([], socket.moved)

    def test_a_console_run_stops_on_it(self) -> None:
        from api.runs import RunState
        from tests.test_a_console_run_stops_and_looks_as_a_campaign_does import _drive

        cell = _PushCell(self)
        cell.orchestrator.gripper = _MeasuresItsWidth(connected=False)
        cell.service.configured_looks = LOOKS

        run, _hub = _drive(cell.service, picks=2)

        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(1, run.attempted)
        self.assertIn("the gripper needs a person, so the run stops", run.error)
        self.assertIn("not connected", run.error)
        self.assertEqual(_keys(*LOOKS), cell.joint_moves())

    def test_a_connected_one_whose_jaws_read_closed_is_a_refusal_the_pick_falls_through_on(self) -> None:
        cell = _PushCell(self)
        cell.orchestrator.gripper = _MeasuresItsWidth(connected=True, width_mm=10.0)

        report = cell.service.pick(look=list(LOOKS))

        self.assertEqual([], cell.arm.pushes)
        self.assertEqual("refused_jaws_closed", report.telemetry["pushes"][0]["code"])
        self.assertEqual("", report.gripper_fault)
        self.assertEqual(["next_target"], report.telemetry["recovery_trail_actions"])
        self.assertEqual(_keys(*LOOKS, *LOOKS), cell.joint_moves())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
