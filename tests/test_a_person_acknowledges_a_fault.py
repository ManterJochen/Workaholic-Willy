"""A person at the cell acknowledges a fault, through the one door there is to it (P0i i4, owner 2026-09-26).

``python -m src.robot.faults acknowledge`` and its twin ``FaultAcknowledgement``: the robot's ability row
first, then the cell lock and the arm alone, watched and never connected
(``tests/test_a_stopped_controller_is_watched_not_driven.py``), the fault the controller describes, and nothing sent
unless the fault can be cleared from software and the person confirmed it. The console shows the command and clears nothing; no other code
calls ``acknowledge_fault`` at all (``tests/test_a_fault_has_one_door.py``).
"""

from __future__ import annotations

import contextlib
import io
import unittest
from typing import Any
from unittest import mock

from src.robot.core import ArmFault, ArmFaultClass, RobotMode, RobotStatus, SafetyMode
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.robot import Robot
from src.robot.faults import AcknowledgementOutcome, FaultAcknowledgement, acknowledge_command


def _status(*, protective: bool = False, emergency: bool = False) -> RobotStatus:
    stopped = protective or emergency
    return RobotStatus(robot_mode=RobotMode.RUNNING,
                       safety_mode=SafetyMode.PROTECTIVE_STOP if protective else (
                           SafetyMode.ROBOT_EMERGENCY_STOP if emergency else SafetyMode.NORMAL),
                       protective_stopped=protective, emergency_stopped=emergency,
                       message="stopped by the test" if stopped else "", program_running=True)


class _Controller(DummyRobotArm):
    """A desk arm that reads a controller state, takes an acknowledgement and can be watched, and records what it was
    sent. The door watches it and never connects it (P0i): ``connect`` raises, as a stopped controller refuses one."""

    def __init__(self, status: RobotStatus, *, takes: bool = True) -> None:
        super().__init__()
        self.status = status
        self.takes = takes
        self.acknowledged: list[ArmFault] = []
        self.watching = False

    @property
    def is_watching(self) -> bool:
        return self.watching

    def watch(self) -> None:
        self.watching = True

    def connect(self) -> None:
        raise AssertionError("the door connected the arm, and a stopped controller refuses a connection")

    def disconnect(self) -> None:
        super().disconnect()
        self.watching = False

    def get_robot_status(self) -> RobotStatus:
        return self.status

    def acknowledge_fault(self, fault: ArmFault) -> bool:
        self.acknowledged.append(fault)
        return self.takes


def _run(arm: Any, *, confirm: bool = True) -> tuple[Any, list[str]]:
    asked: list[str] = []

    def ask(sentence: str) -> bool:
        asked.append(sentence)
        return confirm

    robot = Robot.from_parts(arm=arm, gripper=None)
    return FaultAcknowledgement(robot=robot, confirm=ask).run(), asked


class APersonAcknowledgesAFaultTests(unittest.TestCase):
    def test_a_protective_stop_is_acknowledged_after_the_person_confirms_it(self) -> None:
        arm = _Controller(_status(protective=True))
        report, asked = _run(arm)
        self.assertIs(report.outcome, AcknowledgementOutcome.ACKNOWLEDGED)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual([fault.kind for fault in arm.acknowledged], [ArmFaultClass.PROTECTIVE_STOP])
        self.assertEqual(asked, [report.fault], "the person was asked with the fault's own sentence")
        self.assertIn("protective_stop", report.fault)
        self.assertFalse(arm.is_watching, "the watch is closed on the way out")

    def test_nothing_is_sent_that_the_person_did_not_confirm(self) -> None:
        arm = _Controller(_status(protective=True))
        report, _asked = _run(arm, confirm=False)
        self.assertIs(report.outcome, AcknowledgementOutcome.DECLINED)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(arm.acknowledged, [])
        self.assertIn("nothing was acknowledged", report.what_to_do)

    def test_an_emergency_stop_is_named_with_what_a_person_does_and_nothing_is_sent(self) -> None:
        arm = _Controller(_status(emergency=True))
        report, asked = _run(arm)
        self.assertIs(report.outcome, AcknowledgementOutcome.NOT_ACKNOWLEDGEABLE)
        self.assertEqual((arm.acknowledged, asked), ([], []))
        self.assertIn("release the emergency button", report.what_to_do)

    def test_an_arm_with_no_fault_is_left_alone(self) -> None:
        arm = _Controller(_status())
        report, asked = _run(arm)
        self.assertIs(report.outcome, AcknowledgementOutcome.NO_FAULT)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual((arm.acknowledged, asked), ([], []))

    def test_a_controller_that_does_not_take_it_is_said_so(self) -> None:
        arm = _Controller(_status(protective=True), takes=False)
        report, _asked = _run(arm)
        self.assertIs(report.outcome, AcknowledgementOutcome.NOT_TAKEN)
        self.assertEqual(report.exit_code, 1)
        self.assertIn("clear the stop where the arm is visible", report.what_to_do)

    def test_an_arm_that_reads_no_controller_state_is_refused_with_its_ability_row(self) -> None:
        robot = Robot.from_parts(arm=DummyRobotArm(), gripper=None)
        row = robot.abilities().row("acknowledge fault")
        report = FaultAcknowledgement(robot=robot, confirm=lambda _s: True).run()
        self.assertIs(report.outcome, AcknowledgementOutcome.REFUSED)
        self.assertEqual(report.detail, row.reason)

    def test_the_report_keeps_the_contract(self) -> None:
        import json

        report, _asked = _run(_Controller(_status(protective=True)))
        rendered = report.render()
        self.assertEqual(rendered, rendered.encode("ascii").decode("ascii"))
        self.assertFalse(rendered.endswith("\n"))
        self.assertEqual(json.loads(json.dumps(report.to_dict()))["outcome"], "acknowledged")


class TheCommandLineTests(unittest.TestCase):
    def _main(self, *argv: str, arm: Any = None) -> tuple[int, str]:
        from src.robot.faults import __main__ as cli

        said = io.StringIO()
        with contextlib.ExitStack() as stack:
            if arm is not None:
                stack.enter_context(mock.patch.object(
                    FaultAcknowledgement, "from_tree",
                    classmethod(lambda cls, _loaded, *, confirm: cls(Robot.from_parts(arm=arm, gripper=None), confirm))))
            stack.enter_context(contextlib.redirect_stdout(said))
            code = cli.main(["acknowledge", *argv])
        return code, said.getvalue()

    def test_a_desk_arm_is_refused_with_its_ability_row(self) -> None:
        code, said = self._main("--profile", "console_dummy")
        self.assertEqual(code, 1)
        self.assertIn("acknowledge fault  REFUSED", said)
        self.assertIn("reads no controller state and acknowledges no fault", said)

    def test_yes_is_the_person_at_the_cell_saying_yes(self) -> None:
        arm = _Controller(_status(protective=True))
        code, said = self._main("--profile", "console_dummy", "--yes", arm=arm)
        self.assertEqual(code, 0, said)
        self.assertEqual(len(arm.acknowledged), 1)

    def test_without_a_terminal_and_without_yes_nobody_is_asked_and_nothing_is_sent(self) -> None:
        arm = _Controller(_status(protective=True))
        with mock.patch("sys.stdin", io.StringIO("yes\n")):
            code, said = self._main("--profile", "console_dummy", arm=arm)
        self.assertEqual(code, 1)
        self.assertEqual(arm.acknowledged, [])
        self.assertIn("pass --yes as the person at the cell", said)

    def test_the_command_the_console_shows_names_the_chain(self) -> None:
        self.assertEqual(acknowledge_command("ursim"), "python -m src.robot.faults acknowledge --profile ursim")
        self.assertEqual(acknowledge_command(None), "python -m src.robot.faults acknowledge")


class TheDoorStandsOnItsOwnRowTests(unittest.TestCase):
    """Found by review, 2026-09-26: the door asked every row, and built the arm with every wrist camera body."""

    def test_a_cell_whose_wrist_camera_is_not_placed_yet_can_still_be_acknowledged(self) -> None:
        from types import SimpleNamespace

        from src.config import ConfigTree
        from src.config.tree import LoadedTree
        from tests._wrist_body import camera_section, wrist_cell, wrist_rig

        loaded = LoadedTree(tree=ConfigTree.from_directory(profile=None),
                            config=SimpleNamespace(robot=wrist_cell(),
                                                   camera=camera_section(wrist_rig(calibrated=False))))
        door = FaultAcknowledgement.from_tree(loaded, confirm=lambda _sentence: False)
        self.assertFalse(door.robot.arm.is_connected, "built, never connected")

    def test_the_door_asks_its_own_row_and_no_other(self) -> None:
        from dataclasses import replace

        from src.robot.execution import abilities
        from src.robot.execution.abilities import Ability

        def unreadable(_parts: Any) -> Any:
            raise ValueError("a wrist body that does not resolve for the pick")

        others_fail = tuple(declaration if declaration.ability is Ability.ACKNOWLEDGE_FAULT
                            else replace(declaration, ask=unreadable) for declaration in abilities.DECLARATIONS)
        arm = _Controller(_status(protective=True))
        with mock.patch.object(abilities, "DECLARATIONS", others_fail):
            report, _asked = _run(arm)
        self.assertIs(report.outcome, AcknowledgementOutcome.ACKNOWLEDGED)


if __name__ == "__main__":
    unittest.main()
