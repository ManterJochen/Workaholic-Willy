"""Disconnect and the server's shutdown take the cell down in one order (build plan item 18).

1. cancel a waiting jaws question (its connect is refused, never "open"), before anything takes the session lock a
   connect holds while the question waits;
2. end and join a teach;
3. join a planner start (it moves nothing; the sidecar's ready timeout bounds it);
4. abandon the run;
5. latch the arm of an abandoned moving run, and brake a motion in flight where the arm brakes one;
6. disconnect.

What this file pins: the order on a cell whose arm brakes and whose task is inside a motion; that a motion in flight
leaves the arm latched on an arm that lets it run to its end too (brake_on_halt off, as shipped), so the next Connect
waits for "the cell is clear"; that with nothing in flight on such an arm the latch holds only while the cell comes
down, so the cell connects again as it always did (the control of ``tests/test_api_cell.py``); that a halt a person
pressed before the Disconnect is kept; that a Disconnect returns promptly while a connect waits on a jaws question with
the session lock held; that a planner start in progress is joined and ends as itself; and that the server's shutdown
runs the same teardown before it releases the camera.

Honesty bucket (2): the real console, routes, registry and teardown, on the scripted cell and console_dummy.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, wait_for_event


class TheOrderTests(ConsoleCase):
    def test_question_teach_planner_run_brake_then_disconnect(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))],
                             arm_keywords={"brakes": True})
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        order: list[str] = []
        registry = self.cell.registry
        abandon, disconnect, halt = registry.abandon, self.cell.session.disconnect, cell.arm.halt

        def recorded(name: str, call: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                order.append(name)
                return call(*args, **kwargs)

            return wrapper

        def release_after_the_brake(*args: Any, **kwargs: Any) -> Any:
            answered = halt(*args, **kwargs)
            release.set()  # the motion in flight was braked: the run's thread goes on and ends
            return answered

        with patch("api.jaws.cancel_pending", side_effect=lambda *_a, **_k: order.append("question") or False), \
             patch("api.teach.end_and_join", side_effect=lambda *_a, **_k: order.append("teach") or True), \
             patch.object(registry, "abandon", side_effect=recorded("run", abandon)), \
             patch.object(cell.arm, "halt", side_effect=recorded("brake", release_after_the_brake)), \
             patch.object(self.cell.session, "disconnect", side_effect=recorded("disconnect", disconnect)):
            down = self.client.post("/v1/cell/disconnect")
        self.assertEqual(200, down.status_code, down.text)
        self.assertEqual(["question", "teach", "run", "brake", "disconnect"], order)
        body = self.finished(run["id"])
        self.assertEqual("disconnected", body["stop_code"], body)
        self.assertIsNotNone(cell.arm.halt_state(), "the brake did not latch the arm")

    def test_with_nothing_in_flight_on_an_arm_that_does_not_brake_the_latch_holds_only_while_the_cell_comes_down(
            self) -> None:
        """Nothing was in flight to brake or to run to its end: the latch refuses the abandoned run's next motion while
        the cell comes down, and is given back after the disconnect, so the next Connect connects as it always did."""
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        release.set()
        self.assertEqual("disconnected", self.finished(run["id"])["stop_code"])
        calls = cell.arm.calls
        self.assertLess(calls.index("halt"), calls.index("disconnect"), "the arm came down unlatched")
        self.assertLess(calls.index("disconnect"), calls.index("clear_halt"))
        self.assertIsNone(cell.arm.halt_state())
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)

    def test_a_move_in_flight_on_an_arm_that_does_not_brake_leaves_it_latched(self) -> None:
        """Build plan item 18, step 5: a motion in flight latches the arm, whether the arm brakes it or lets it run to
        its end (brake_on_halt off, as shipped). The move in flight ends, nothing after it is sent, and the next Connect
        is refused ``halted`` until a person says the cell is clear."""
        inside, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part")])

        def held_inside_the_approach(kind: str, _index: int) -> None:
            if kind == "move" and not inside.is_set():
                inside.set()
                release.wait(timeout=20)

        cell.arm.on_motion = held_inside_the_approach
        run = self.task(object="green cube")
        self.assertTrue(inside.wait(timeout=20), "the task never moved")
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        latch = cell.arm.halt_state()
        self.assertIsNotNone(latch, "a Disconnect with a move in flight left the arm unlatched")
        assert latch is not None
        self.assertTrue(latch.in_motion)
        release.set()
        self.assertEqual("disconnected", self.finished(run["id"])["stop_code"])
        self.assertEqual(1, [motion[0] for motion in cell.motions()].count("move"),
                         "a motion was sent after the move in flight")
        self.assertNotIn("clear_halt", cell.arm.calls)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.refused(self.client.post("/v1/cell/connect", json={"token": token}), 409, "halted")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)

    def test_a_halt_pressed_while_the_cell_comes_down_is_kept(self) -> None:
        """A person's "halt now" while the Disconnect is under way lands on the latch the teardown set (the arm keeps the
        first record): the teardown gives no latch back that a person pressed halt on."""
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        disconnect = self.cell.session.disconnect
        pressed: list[int] = []

        def halt_pressed_then_down() -> None:
            pressed.append(self.client.post("/v1/cell/brake").status_code)  # still connected: the arm is coming down
            disconnect()

        with patch.object(self.cell.session, "disconnect", side_effect=halt_pressed_then_down):
            self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        self.assertEqual([200], pressed)
        release.set()
        self.finished(run["id"])
        self.assertIsNotNone(cell.arm.halt_state(), "the Disconnect gave back a latch a person pressed halt now on")
        self.assertNotIn("clear_halt", cell.arm.calls)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.refused(self.client.post("/v1/cell/connect", json={"token": token}), 409, "halted")

    def test_a_halt_a_person_pressed_before_the_disconnect_is_kept(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        self.assertEqual(200, self.client.post("/v1/cell/brake").status_code)
        pressed = cell.arm.halt_state()
        self.assertIsNotNone(pressed)
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        release.set()
        self.finished(run["id"])
        self.assertIs(pressed, cell.arm.halt_state(), "the Disconnect gave back a halt a person pressed")
        self.assertNotIn("clear_halt", cell.arm.calls)

    def test_a_disconnect_with_nothing_running_leaves_no_latch_and_the_cell_connects_again(self) -> None:
        self.build_dummy()
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        self.assertIsNone(self.cell.session.arm.halt_state())
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)


class AQuestionWaitingTests(ConsoleCase):
    def test_disconnect_cancels_the_question_before_it_takes_the_session_lock(self) -> None:
        self.build_dummy()
        asked, cancelled = threading.Event(), threading.Event()
        self.addCleanup(cancelled.set)

        def connect_waiting_on_the_question() -> None:
            with self.cell.session._lock:  # noqa: SLF001 (what a connect holds while the question waits)
                asked.set()
                cancelled.wait(timeout=20)  # the browser seam's wait, ended by cancel_pending

        connecting = threading.Thread(target=connect_waiting_on_the_question, daemon=True)
        connecting.start()
        self.assertTrue(asked.wait(timeout=10))

        def cancel_pending(_console: Any, _why: str) -> bool:
            cancelled.set()
            return True

        started = time.monotonic()
        with patch("api.jaws.cancel_pending", side_effect=cancel_pending):
            down = self.client.post("/v1/cell/disconnect")
        self.assertEqual(200, down.status_code, down.text)
        self.assertLess(time.monotonic() - started, 5.0, "the disconnect waited for the question to time out")
        connecting.join(timeout=10)


class APlannerStartTests(ConsoleCase):
    def test_a_disconnect_joins_a_planner_start_and_lets_it_end_as_itself(self) -> None:
        blocks = threading.Event()
        self.addCleanup(blocks.set)
        cell = self.scripted(arm_keywords={"planner": "off", "start_blocks": blocks})
        started = self.client.post("/v1/cell/planner")
        self.assertEqual(202, started.status_code, started.text)
        wait_for_event(self.cell, started.json()["id"], "planner.starting")
        threading.Timer(0.3, blocks.set).start()
        down = self.client.post("/v1/cell/disconnect")
        self.assertEqual(200, down.status_code, down.text)
        body = self.client.get(f"/v1/runs/{started.json()['id']}").json()
        self.assertEqual("planner_ready", body["stop_code"], "the planner start was abandoned instead of joined")
        self.assertEqual("ready", cell.arm.planner_state)
        self.assertEqual("disconnect", cell.arm.calls[-1])


class TheShutdownTests(ConsoleCase):
    def test_the_server_s_shutdown_takes_the_cell_down_before_it_releases_the_camera(self) -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app

        self.build_dummy()
        order: list[str] = []
        release = self.cell.session.release
        with patch("api.cell.take_down", side_effect=lambda *_a, **_k: order.append("take_down")), \
             patch.object(self.cell.session, "release", side_effect=lambda: (order.append("release"), release())):
            with TestClient(create_app()):
                pass
        self.assertEqual(["take_down", "release"], order)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
