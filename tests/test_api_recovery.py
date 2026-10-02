"""The recovery record: a moving run that stopped on a problem leaves the arm where it stood, and the console a record.

Build plan 1.3.5 and item 4: the record lives on the console, not on the run or the cell, so it survives Disconnect,
a rebuild, a Connect and a page reload; while it stands and is not cleared every moving route refuses
``cell_not_cleared``; once cleared a new task or pick refuses ``restart_required`` and only Restart and Home remain;
arriving at the return pose (a Restart's return, a Home run's arrival) ends it. What this file pins:

* a task, a pick run and a Home run that end on a problem each write the record, announced on the cell's stream, and
  ``GET /v1/cell`` carries it for a page opened afterwards;
* the record survives Disconnect, Build and Connect (scenario 6);
* "the cell is clear" stamps it cleared and keeps it; Home ends it and says so;
* no record for an end that is no problem, and none for a stop of a run that moved nothing.

Honesty bucket (2): the real console, on console_dummy and on the scripted cell.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from tests._console_task_fakes import ConsoleCase, Scripted, events_of, wait_for_event


class AProblemStopLeavesTheRecordTests(ConsoleCase):
    def test_a_task_that_stops_on_a_problem_writes_the_record_and_the_cell_says_it(self) -> None:
        self.scripted([Scripted("failed")] * 3)
        run = self.task(object="green cube")
        body = self.finished(run["id"])
        self.assertEqual(("failed", "failed_in_a_row", "problem"), (body["state"], body["stop_code"], body["stop_class"]))
        self.assertIn("in a row failed", body["error"])
        record = self.client.get("/v1/cell").json()["recovery"]
        self.assertEqual((run["id"], "task", "failed_in_a_row", None),
                         (record["run_id"], record["kind"], record["stop_code"], record["cleared_at"]))
        announced = wait_for_event(self.cell, "cell", "cell.recovery").data
        self.assertEqual(run["id"], announced["run_id"])
        error = wait_for_event(self.cell, run["id"], "run_error").data
        self.assertEqual("failed_in_a_row", error["stop_code"])

    def test_a_pick_run_that_stops_on_a_problem_writes_it_too(self) -> None:
        """A console pick run never sets its part down: on a toggle the second pick would start on jaws the program
        believes closed, and the run ends there, a hand that needs a person."""
        self.scripted([Scripted("part"), Scripted("part")])
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 2})
        self.assertEqual(202, pick.status_code, pick.text)
        body = self.finished(pick.json()["id"])
        self.assertEqual(("failed", "pick", "hand_needs_person", "problem", True),
                         (body["state"], body["kind"], body["stop_code"], body["stop_class"], body["holding"]))
        self.assertIsNotNone(self.cell.recovery, body)
        self.assertEqual(pick.json()["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]

    def test_the_record_survives_disconnect_build_and_connect(self) -> None:
        """Scenario 6: the stopped run's record stays through the whole teardown and bring-up of a cell."""
        self.build_dummy()
        stopped: dict[str, Any] = {}

        def hold(_console: Any, run: Any) -> str:
            stopped["id"] = run.id
            run.error = "the place was refused before the release"
            return "part_still_held"

        from api.codes import RunKind

        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, hold)
        self.finished(run.id)
        self.assertEqual(run.id, self.cell.recovery.run_id)  # type: ignore[union-attr]
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        self.assertEqual(200, self.client.post("/v1/cell/build", params={"rehearse": True}).status_code)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)
        record = self.client.get("/v1/cell").json()["recovery"]
        self.assertEqual((run.id, "part_still_held"), (record["run_id"], record["stop_code"]))
        self.refused(self.client.post("/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None}}),
                     409, "cell_not_cleared")
        self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "cell_not_cleared")
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "cell_not_cleared")

    def test_the_cell_is_clear_keeps_the_record_and_home_ends_it(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "return_failed")
        self.finished(run.id)
        cleared = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
        self.assertEqual(200, cleared.status_code, cleared.text)
        record = cleared.json()["recovery"]
        self.assertGreater(record["cleared_at"], record["at"])
        acknowledged = wait_for_event(self.cell, "cell", "cell.acknowledged").data
        self.assertIn("recovery", acknowledged["cleared"])
        self.refused(self.client.post("/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None}}),
                     409, "restart_required")
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.assertEqual("finished", self.finished(home.json()["id"])["stop_code"])
        self.assertIsNone(self.client.get("/v1/cell").json()["recovery"])
        ended = wait_for_event(self.cell, "cell", "cell.recovery_ended").data
        self.assertEqual((run.id, "home", home.json()["id"]), (ended["run_id"], ended["by"], ended["ended_by"]))
        run2 = self.task(object="")
        self.assertEqual("finished", self.finished(run2["id"])["stop_code"])

    def test_no_record_where_nothing_stopped_mid_motion(self) -> None:
        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube")
        self.assertEqual("finished", self.finished(run["id"])["stop_code"])
        planner = self.client.post("/v1/cell/planner")
        self.assertEqual(202, planner.status_code, planner.text)
        self.finished(planner.json()["id"])
        self.assertIsNone(self.cell.recovery)
        self.assertNotIn("cell.recovery", [event.type for event in events_of(self.cell, "cell")])
        self.assertEqual(2, cell.do0_changes())

    def test_a_record_names_the_newest_problem_stop(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        first = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "cell_fault")
        self.finished(first.id)
        self.cell.mark_cell_clear()
        second = self.cell.registry.start_kind(self.cell, RunKind.HOME, lambda _c, _r: "halted",
                                               inputs={"to": "home"})
        self.finished(second.id)
        record = self.cell.recovery
        assert record is not None
        self.assertEqual((second.id, "home", None), (record.run_id, str(record.kind), record.cleared_at))


class TheGatesAnswerAcrossThreadsTests(ConsoleCase):
    def test_the_record_stands_before_the_run_lets_go_of_the_lock(self) -> None:
        """Nothing new can start between a problem stop and the record that gates it."""
        self.build_dummy()
        from api.codes import RunKind

        seen: list[Any] = []
        record_problem_stop = self.cell.record_problem_stop

        def recorder(run: Any) -> Any:
            seen.append((self.cell.active_run_id, self.client.post(
                "/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None}}).json()["code"]))
            return record_problem_stop(run)

        self.cell.record_problem_stop = recorder  # type: ignore[method-assign]
        release = threading.Event()
        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: release.wait(5) and "halted")
        release.set()
        self.finished(run.id)
        self.assertEqual([(run.id, "run_active")], seen)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
