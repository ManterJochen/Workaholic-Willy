"""The pendant clears a protective stop, never the console (the owner's OD 14).

Build plan 1.3.5: "the cell is clear" ends the arm's halt latch and the service's latch of a recovery that needs a
person, and it never clears a protective stop: a controller that says it is stopped refuses it (``409
controller_stopped``: clear it at the pendant first, where the arm is visible). Neither does any other route of the way
back: the brake, Restart and Home. ``tests/test_api_cell.py`` pins the telemetry panel's half of this; this file pins
the rest, on an arm that records every attempt to clear a stop and fails the test on it.

Honesty bucket (2): the real console and routes, on the scripted cell's arm double.
"""

from __future__ import annotations

import unittest
from typing import Any

from tests._console_task_fakes import ConsoleArm, ConsoleCase, Scripted, ScriptedCell

#: Every way a UR driver or its SDK clears a protective stop.
_CLEARS = ("recover_from_protective_stop", "unlock_protective_stop", "unlockProtectiveStop")


class _RecordingConnection:
    """What a UR arm's connection would be asked, if anything asked it to clear a stop."""

    def __init__(self, asked: list[str]) -> None:
        self.asked = asked

    def unlock_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        self.asked.append("unlock_protective_stop")
        raise AssertionError("the console cleared a protective stop")

    def unlockProtectiveStop(self) -> bool:  # noqa: N802 (the SDK's own name)  # pragma: no cover
        self.asked.append("unlockProtectiveStop")
        raise AssertionError("the console cleared a protective stop")


class _RecordingArm(ConsoleArm):
    def __init__(self, log: list[Any], **keywords: Any) -> None:
        super().__init__(log, **keywords)
        self.asked: list[str] = []
        self._conn = _RecordingConnection(self.asked)

    def unlock_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        self.asked.append("unlock_protective_stop")
        raise AssertionError("the console cleared a protective stop")

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
        self.asked.append("recover_from_protective_stop")
        raise AssertionError("the console cleared a protective stop")


class TheWayBackNeverClearsAStopTests(ConsoleCase):
    def test_brake_acknowledge_home_and_restart(self) -> None:
        from tests._console_task_fakes import _toggle
        from tests._task_fakes import PROTECTIVE, RUNNING

        log: list[Any] = []
        arm = _RecordingArm(log)
        cell = ScriptedCell(log=log, arm=arm, jaws=_toggle(log, arm), service=None)  # type: ignore[arg-type]
        from tests._console_task_fakes import ConsoleService

        cell.service = ConsoleService(arm, cell.jaws, [Scripted("failed")] * 3 + [Scripted("part")], clock=cell.clock)
        cell.install(self.cell)

        stopped = self.task(object="green cube")
        self.assertEqual("failed_in_a_row", self.finished(stopped["id"])["stop_code"])
        self.assertEqual(200, self.client.post("/v1/cell/brake").status_code)
        arm.status = PROTECTIVE
        refused = self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}), 409,
                               "controller_stopped")
        self.assertIn("pendant", refused["message"])
        self.assertIsNotNone(arm.halt_state(), "a refused acknowledge cleared the latch")
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "halted")
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "halted")
        # The person clears the stop at the pendant; then the console's own way back.
        arm.status = RUNNING
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.cell.stamp_jaws_open(opened=False)
        restart = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restart.status_code, restart.text)
        self.assertEqual("finished", self.finished(restart.json()["id"])["stop_code"])
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.finished(home.json()["id"])
        self.assertEqual([], arm.asked, "a route asked the arm to clear a protective stop")
        self.assertEqual([], [call for call in arm.calls if call in _CLEARS])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
