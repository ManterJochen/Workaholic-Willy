"""``POST /v1/cell/home``: the manual Home button, a planned move to Home or a taught pose, started only by a click.

Build plan 1.3.6: the run moves the arm with ``robot.home()`` or ``robot.move_joints(taught)``, through the arm's own
judged verbs, after the hands-off countdown where one is due. Its arrival ends the console's recovery record, so it is
one of the two ways back after a stop. It refuses during a run, on a halted or stopped controller, while a person is
needed or a jaws question waits, while a part may still be held, and before "the cell is clear". What this file pins:

* Home and a taught pose, with ``home.started`` and ``home.arrived``, end ``finished``;
* every refusal of 1.3.6 answers its code and moves nothing; Home is allowed while a Restart is required;
* while a recovery record stands, Home reads the hand against it as Restart does: a toggle's jaws answered open since the
  stop (``jaws_not_confirmed`` until then), and a hand that measures nothing waits for "Backen leer"
  (``part_still_held``), the empty jaws of owner decision Q2 = A;
* a move the arm refuses ends ``return_failed`` where the arm stands, the record written; a halt pressed before the
  move ends it ``halted`` with nothing sent, and a part that came into the hand during the countdown
  ``part_still_held``, nothing sent.

Honesty bucket (2): the real console and routes, on console_dummy and on the scripted cell.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, event_types, wait_for_event


class HomeMovesTests(ConsoleCase):
    def test_home_on_console_dummy(self) -> None:
        self.build_dummy()
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.assertEqual(("home", "running"), (home.json()["kind"], home.json()["state"]))
        body = self.finished(home.json()["id"])
        self.assertEqual(("finished", "finished", "done"), (body["state"], body["stop_code"], body["stop_class"]))
        self.assertEqual(["run_started", "home.started", "home.arrived", "run_finished"],
                         event_types(self.cell, body["id"]))
        self.assertEqual("home", wait_for_event(self.cell, body["id"], "run_started").data["to"])

    def test_a_taught_pose(self) -> None:
        from tests._task_fakes import PARK_JOINTS, _key

        cell = self.scripted()
        home = self.client.post("/v1/cell/home", json={"to": "park"})
        self.assertEqual(202, home.status_code, home.text)
        body = self.finished(home.json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual([("joints", _key(PARK_JOINTS))], cell.motions())
        self.assertEqual(0, cell.do0_changes())

    def test_a_move_the_arm_refuses_ends_return_failed_where_it_stands(self) -> None:
        from src.robot.core import MotionStatus

        cell = self.scripted()
        cell.arm.refuse = lambda kind, _index, _target: MotionStatus.WORKSPACE_REJECTED
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        body = self.finished(home.json()["id"])
        self.assertEqual(("failed", "return_failed", "problem"), (body["state"], body["stop_code"], body["stop_class"]))
        refused = wait_for_event(self.cell, body["id"], "home.refused").data
        self.assertEqual("workspace_rejected", refused["status"])
        self.assertEqual(body["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]
        self.assertEqual([], cell.motions())

    def test_home_is_the_way_back_while_a_restart_is_required(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "return_failed")
        self.finished(run.id)
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "cell_not_cleared")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.finished(home.json()["id"])
        self.assertIsNone(self.cell.recovery)


class HomeRefusesTests(ConsoleCase):
    def test_every_refusal_moves_nothing(self) -> None:
        from tests._task_fakes import PROTECTIVE

        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "not_connected")
        cell = self.scripted()
        self.refused(self.client.post("/v1/cell/home", json={"to": "nowhere"}), 422, "unknown_pose")
        cell.arm.halt("the operator pressed halt now")
        cell.arm.status = PROTECTIVE
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "halted")
        cell.arm.clear_halt()
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "controller_stopped")
        from tests._task_fakes import RUNNING

        cell.arm.status = RUNNING
        cell.service._needs_person = "a push stopped where the arm stands"  # noqa: SLF001
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "needs_person")
        cell.service._needs_person = ""  # noqa: SLF001
        with patch("api.jaws.pending", return_value=object()):
            self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "jaws_question_pending")
        cell.jaws.set_closed(True)
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "part_still_held")
        cell.jaws.set_closed(False)
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined]
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "jaws_not_confirmed")
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined]
        cell.arm.plans_paths = False  # type: ignore[misc]
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "route_refused")
        self.assertEqual([], cell.motions())
        self.assertEqual([], self.client.get("/v1/runs").json())

    def test_a_run_holds_the_lock(self) -> None:
        import threading

        release = threading.Event()
        self.addCleanup(release.set)
        self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "run_active")
        release.set()
        self.finished(run["id"])

    def test_a_part_that_came_into_the_hand_during_the_countdown_stops_the_move(self) -> None:
        """Read again inside the run, after the countdown: the gate before it is minutes old by then."""
        cell = self.scripted()
        self.cell.registry.countdown_step_s = 0.3
        self.cell.stamp_teach_ended()
        original = self.cell.hub.publish

        def publish(stream: str, event_type: str, /, **keywords: object) -> object:
            event = original(stream, event_type, **keywords)
            if event_type == "run_countdown" and event.data.get("seconds_left") == 1:  # type: ignore[attr-defined]
                cell.arm.attach_payload(40.0)  # the planner models a part in the hand again
            return event

        self.cell.hub.publish = publish  # type: ignore[method-assign]
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        body = self.finished(home.json()["id"])
        self.assertEqual(("failed", "part_still_held"), (body["state"], body["stop_code"]), body)
        self.assertIn("planner still models a part", body["error"])
        self.assertEqual([], cell.motions())
        self.assertNotIn("home.started", event_types(self.cell, body["id"]))

    def test_a_halt_before_the_move_ends_it_halted_with_nothing_sent(self) -> None:
        cell = self.scripted()
        self.cell.registry.countdown_step_s = 0.3
        self.cell.stamp_teach_ended()
        original = self.cell.hub.publish

        def publish(stream: str, event_type: str, /, **keywords: object) -> object:
            event = original(stream, event_type, **keywords)
            if event_type == "run_countdown" and event.data.get("seconds_left") == 1:  # type: ignore[attr-defined]
                cell.arm.halt("a halt latched by another program")
            return event

        self.cell.hub.publish = publish  # type: ignore[method-assign]
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        body = self.finished(home.json()["id"])
        self.assertEqual(("failed", "halted"), (body["state"], body["stop_code"]), body)
        self.assertEqual([], cell.motions())
        self.assertIn("halted", body["error"])


class HomeReadsTheHandAgainstTheStopTests(ConsoleCase):
    """While a recovery record stands, Home is a way back like Restart: only with empty jaws (Q2 = A)."""

    def _stopped(self, driver: object) -> str:
        from api.codes import RunKind

        stopped = self.cell.registry.start_kind(self.cell, RunKind.TASK, driver)  # type: ignore[arg-type]
        self.finished(stopped.id)
        self.assertEqual(stopped.id, self.cell.recovery.run_id)  # type: ignore[union-attr]
        cleared = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
        self.assertEqual(200, cleared.status_code, cleared.text)
        return stopped.id

    def test_a_toggle_s_home_waits_for_the_jaws_answered_open_since_the_stop(self) -> None:
        cell = self.scripted()
        self._stopped(lambda _c, _r: "cell_fault")
        refused = self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "jaws_not_confirmed")
        self.assertIn("since the stop", refused["message"])
        self.assertEqual([], cell.motions())
        self.cell.stamp_jaws_open(opened=False)  # the jaws question answered open in the browser
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.assertEqual("finished", self.finished(home.json()["id"])["stop_code"])
        self.assertEqual([("home",)], cell.motions())
        self.assertIsNone(self.cell.recovery)

    def test_a_hand_that_measures_nothing_waits_for_backen_leer(self) -> None:
        self.build_dummy()

        def held(_console: object, run: object) -> str:
            run.holding = True  # type: ignore[attr-defined]
            return "part_still_held"

        self._stopped(held)
        refused = self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "part_still_held")
        self.assertIn("Backen leer", refused["message"])
        emptied = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True})
        self.assertEqual(200, emptied.status_code, emptied.text)
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.assertEqual("finished", self.finished(home.json()["id"])["stop_code"])
        self.assertIsNone(self.cell.recovery)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
