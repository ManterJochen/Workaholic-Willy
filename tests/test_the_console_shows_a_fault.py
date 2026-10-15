"""The console shows a cell's fault and the command a person at the cell clears it with, and clears nothing (P0i i4).

Owner, 2026-09-26: ``acknowledge_fault`` has one door, a CLI verb a person runs at the cell, and the console shows the
fault and that command. The live status reads the controller's state where it is asked to; the fault that state
describes, what a person does about it, and, where software can clear it at all, the command with the console's own
profile chain, ride on the same reading. An emergency button gets no command: no software releases it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from src.robot.core import RobotMode, RobotStatus, SafetyMode
from src.robot.drivers.dummy.arm import DummyRobotArm


def _status(*, protective: bool = False, emergency: bool = False) -> RobotStatus:
    return RobotStatus(robot_mode=RobotMode.RUNNING,
                       safety_mode=SafetyMode.PROTECTIVE_STOP if protective else (
                           SafetyMode.ROBOT_EMERGENCY_STOP if emergency else SafetyMode.NORMAL),
                       protective_stopped=protective, emergency_stopped=emergency, program_running=True)


class _Stopped(DummyRobotArm):
    def __init__(self, status: RobotStatus) -> None:
        super().__init__()
        self.status = status
        self.acknowledged: list[Any] = []
        self.connect()

    def get_robot_status(self) -> RobotStatus:
        return self.status

    def acknowledge_fault(self, fault: Any) -> bool:  # pragma: no cover - the console must never reach it
        self.acknowledged.append(fault)
        return True


def _status_out(arm: Any, *, profile: str | None = "ursim") -> Any:
    from api.routers.cell import get_status

    console = SimpleNamespace(session=SimpleNamespace(arm=arm, state="connected"), profile=profile)
    return get_status(console, include_controller_state=True)  # type: ignore[arg-type]


class TheConsoleShowsAFaultTests(unittest.TestCase):
    def test_a_protective_stop_is_shown_with_the_command_that_clears_it_and_is_not_cleared(self) -> None:
        arm = _Stopped(_status(protective=True))
        out = _status_out(arm)
        self.assertIn("protective_stop", out.fault)
        self.assertIn("clear the stop where the arm is visible", out.fault_what_to_do)
        self.assertEqual(out.fault_clear_with, "python -m src.robot.faults acknowledge --profile ursim")
        self.assertEqual(arm.acknowledged, [], "the console showed the fault and cleared it")

    def test_the_command_is_shown_with_the_lock_the_console_holds(self) -> None:
        """Found by review, 2026-09-26: the console holds the cell lock while connected, and the command takes it."""
        out = _status_out(_Stopped(_status(protective=True)))
        self.assertIn("Disconnect this console first: it holds the cell lock the command takes.", out.fault_what_to_do)
        emergency = _status_out(_Stopped(_status(emergency=True)))
        self.assertNotIn("Disconnect this console", emergency.fault_what_to_do, "no command, so no lock to give up")

    def test_an_emergency_stop_gets_no_command(self) -> None:
        out = _status_out(_Stopped(_status(emergency=True)))
        self.assertIn("emergency_stop", out.fault)
        self.assertIsNone(out.fault_clear_with)
        self.assertIn("release the emergency button", out.fault_what_to_do)

    def test_an_arm_that_can_move_shows_no_fault(self) -> None:
        out = _status_out(_Stopped(_status()))
        self.assertEqual((out.fault, out.fault_what_to_do, out.fault_clear_with), (None, None, None))


if __name__ == "__main__":
    unittest.main()
