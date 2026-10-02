""""Halt now" is ``POST /v1/cell/brake``: the run stops commanding and the arm latches; it is not an emergency stop.

Build plan item 1 and 1.3.5, the halt-now design A: the route tells the run to halt and latches the arm through the
library's own halt (``SupportsHalt``) BEFORE it tells the run, so from then on every motion and output switch of the arm
is refused before anything is sent; with ``robot.ur.brake_on_halt`` off, as shipped (Q1), the move in flight runs to its
end and nothing after it is sent. A halt reads ``halted`` everywhere, never ``controller_stopped``, and only "the cell is
clear" (``POST /v1/cell/acknowledge``) ends the latch. What this file pins:

* the brake answers while a connect holds the session lock: it takes none;
* with no run it latches the arm and says so on the cell's stream; with a run it also halts the run, which ends
  ``halted`` with its own sentence where the arm stood, the record written, and nothing commanded after the move in
  flight, on console_dummy and on an arm that reports its controller alike;
* a halted cell refuses every moving route ``halted``, Connect included, until acknowledged; the ready bar's robot light
  says ``halted``;
* an arm with no latch answers 200 ``latched: false``, and its run stops before its next motion;
* what the latch says of the move in flight reaches ``GET /v1/cell`` (``halted.brake``), and ``braking`` is said only
  where the arm brakes and its brake is still running;
* "the cell is clear" (``POST /v1/cell/acknowledge``) refuses where 1.3.5 says: nothing built, a run still active (an
  abandoned one whose thread lives included: the latch stays set, since a verb it is still inside would send its next
  move), and "Backen leer" (``jaws_empty``) on a toggle whose count is not open or on a hand that measures a part; taken,
  "Backen leer" tells the planner the hand is empty.

Honesty bucket (2): the real console, routes and ``run_task``, on console_dummy and on the scripted cell.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from tests._console_task_fakes import ConsoleCase, ConsoleService, Scripted, event_types, events_of, wait_for_event


class TheBrakeTakesNoLockTests(ConsoleCase):
    def test_the_brake_answers_while_a_connect_holds_the_session_lock(self) -> None:
        self.build_dummy()
        held, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def connect_waiting_on_a_question() -> None:
            with self.cell.session._lock:  # noqa: SLF001 (what a connect holds while the jaws question waits)
                held.set()
                release.wait(timeout=20)

        holder = threading.Thread(target=connect_waiting_on_a_question, daemon=True)
        holder.start()
        self.assertTrue(held.wait(timeout=10))
        started = time.monotonic()
        braked = self.client.post("/v1/cell/brake")
        self.assertLess(time.monotonic() - started, 5.0, "the brake waited for the session lock")
        self.assertEqual(200, braked.status_code, braked.text)
        self.assertTrue(braked.json()["latched"])
        release.set()
        holder.join(timeout=10)

    def test_a_cell_that_is_not_connected_has_nothing_to_halt(self) -> None:
        self.refused(self.client.post("/v1/cell/brake"), 409, "not_connected")
        self.build_dummy(connect=False)
        self.refused(self.client.post("/v1/cell/brake"), 409, "not_connected")


class AHaltWithNoRunTests(ConsoleCase):
    def test_the_arm_latches_and_every_moving_route_says_halted_until_the_cell_is_clear(self) -> None:
        self.build_dummy()
        braked = self.client.post("/v1/cell/brake")
        self.assertEqual(200, braked.status_code, braked.text)
        self.assertEqual({"run_halted": False, "latched": True, "braking": False, "in_motion": False, "run_id": None},
                         {key: braked.json()[key] for key in ("run_halted", "latched", "braking", "in_motion",
                                                              "run_id")})
        self.assertIn("not an emergency stop", braked.json()["message"].lower())
        said = wait_for_event(self.cell, "cell", "cell.halted").data
        self.assertEqual(False, said["in_motion"])
        self.assertTrue(said["reason"])
        halted = self.client.get("/v1/cell").json()["halted"]
        self.assertIsNotNone(halted)
        self.assertEqual("none", halted["brake"])
        for method, path, body in (("POST", "/v1/task", {"object": "", "place": {"kind": "pose", "pose": None}}),
                                   ("POST", "/v1/pick", {"prompt": "", "picks": 1}),
                                   ("POST", "/v1/cell/home", {"to": "home"})):
            with self.subTest(route=path):
                self.refused(self.client.request(method, path, json=body), 409, "halted")
        robot = next(light for light in self.client.get("/v1/cell/readiness").json()["lights"] if light["id"] == "robot")
        self.assertEqual(("blocked", "halted"), (robot["state"], robot["code"]))
        cleared = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
        self.assertEqual(200, cleared.status_code, cleared.text)
        self.assertIsNone(cleared.json()["halted"])
        self.assertIn("halted", wait_for_event(self.cell, "cell", "cell.acknowledged").data["cleared"])
        run = self.task(object="")
        self.assertEqual("finished", self.finished(run["id"])["stop_code"])

    def test_connect_is_refused_while_the_built_arm_is_latched(self) -> None:
        self.build_dummy()
        self.assertEqual(200, self.client.post("/v1/cell/brake").status_code)
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.refused(self.client.post("/v1/cell/connect", json={"token": token}), 409, "halted")
        self.assertEqual("built", self.client.get("/v1/cell").json()["state"])
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)


class AHaltDuringATaskTests(ConsoleCase):
    def test_a_halt_pressed_during_the_grasp_ends_the_task_halted_where_the_arm_stands(self) -> None:
        """Scenario 5 on an arm that reports its controller: halted again, never controller_stopped."""
        pressed: dict[str, Any] = {}
        cell = self.scripted([Scripted("part"), Scripted("part")])

        def brake_during_the_line_in(kind: str, index: int) -> None:
            if kind == "move" and index == 1 and not pressed:
                pressed["answer"] = self.client.post("/v1/cell/brake").json()

        cell.arm.on_motion = brake_during_the_line_in
        run = self.task(object="green cube")
        body = self.finished(run["id"])
        self.assertEqual(("failed", "halted", "problem", True, False),
                         (body["state"], body["stop_code"], body["stop_class"], body["halt_requested"], body["holding"]),
                         body)
        self.assertIn("the console halted the arm", body["error"])
        self.assertEqual({"run_halted": True, "latched": True, "braking": False, "in_motion": True},
                         {key: pressed["answer"][key] for key in ("run_halted", "latched", "braking", "in_motion")})
        self.assertEqual(run["id"], pressed["answer"]["run_id"])
        # The move in flight ran to its end (brakes off); nothing after it went out, the jaws included.
        self.assertEqual("move", cell.motions()[-1][0])
        self.assertEqual(2, len(cell.motions()), "a motion was sent after the halt")
        self.assertEqual(0, cell.do0_changes(), "the jaws were switched on a halted arm")
        asked = wait_for_event(self.cell, run["id"], "run_halt_requested").data
        self.assertEqual((True, False), (asked["in_motion"], asked["braking"]))
        types = event_types(self.cell, run["id"])
        self.assertLess(types.index("run_halt_requested"), types.index("run_error"))
        self.assertEqual("halted", wait_for_event(self.cell, run["id"], "run_error").data["stop_code"])
        self.assertEqual(run["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]
        self.assertEqual("ran_out", self.client.get("/v1/cell").json()["halted"]["brake"])
        robot = next(light for light in self.client.get("/v1/cell/readiness").json()["lights"] if light["id"] == "robot")
        self.assertEqual("halted", robot["code"], robot)
        self.refused(self.client.post("/v1/task", json={"object": "green cube", "place": {"kind": "pose", "pose": None}}),
                     409, "halted")
        cleared = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
        self.assertEqual(200, cleared.status_code, cleared.text)
        self.assertEqual(["clear_halt"], [call for call in cell.arm.calls if call == "clear_halt"])
        self.cell.stamp_jaws_open(opened=False)
        restart = self.client.post("/v1/task/restart", json={"run_id": run["id"]})
        self.assertEqual(202, restart.status_code, restart.text)
        self.assertEqual("finished", self.finished(restart.json()["id"])["stop_code"])
        self.assertEqual(2, cell.do0_changes())

    def test_a_halt_on_console_dummy_ends_the_task_halted_never_controller_stopped(self) -> None:
        self.build_dummy()
        original = self.cell.hub.publish
        pressed: list[int] = []

        def publish(stream: str, event_type: str, /, **keywords: Any) -> Any:
            event = original(stream, event_type, **keywords)
            if event_type == "pick.executing" and not pressed:
                pressed.append(self.client.post("/v1/cell/brake").status_code)
            return event

        self.cell.hub.publish = publish  # type: ignore[method-assign]
        run = self.task(object="")
        body = self.finished(run["id"])
        self.assertEqual([200], pressed)
        self.assertEqual(("failed", "halted"), (body["state"], body["stop_code"]), body)
        self.assertNotIn("protective or emergency stop", body["error"])
        self.assertEqual(0, body["parts_placed"])

    def test_a_halted_pick_run_ends_halted_with_its_own_sentence(self) -> None:
        self.build_dummy()
        original = self.cell.hub.publish

        def publish(stream: str, event_type: str, /, **keywords: Any) -> Any:
            event = original(stream, event_type, **keywords)
            if event_type == "pick.executing" and stream != "cell" and not self.cell.registry.get(stream).halt_requested:
                self.client.post("/v1/cell/brake")
            return event

        self.cell.hub.publish = publish  # type: ignore[method-assign]
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 3})
        self.assertEqual(202, pick.status_code, pick.text)
        body = self.finished(pick.json()["id"])
        self.assertEqual(("failed", "halted", "pick"), (body["state"], body["stop_code"], body["kind"]), body)
        self.assertNotIn("protective or emergency stop", body["error"])
        self.assertIn("halt", body["error"])
        self.assertLess(body["attempted"], 3, "the run picked on after the halt")
        self.assertEqual(pick.json()["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]


class AnArmWithNoLatchTests(ConsoleCase):
    def _latchless(self, picks: list[Scripted]) -> Any:
        from tests._console_task_fakes import AnswersOpen, _toggle
        from tests._task_fakes import TaskArm

        class LatchlessArm(TaskArm):
            """An arm with no halt latch (a sim or KUKA arm): it stops only before its next motion."""

            def connect(self) -> None:
                self.is_connected = True

            def disconnect(self) -> None:
                self.is_connected = False

            def halt_state(self) -> Any:
                return None

        log: list[Any] = []
        arm = LatchlessArm(log)
        jaws = _toggle(log, arm)  # type: ignore[arg-type]
        jaws.answer_questions_with(AnswersOpen())  # the question seam a build hands the hand (api.jaws.install)
        service = ConsoleService(arm, jaws, picks)  # type: ignore[arg-type]
        self.cell.session.adopt(service)
        preview = self.cell.session.preview(self.cell.robot(), self.cell.fingerprint())
        self.cell.session.connect(preview.token, self.cell.fingerprint(), None)
        return arm, log

    def test_the_brake_answers_unlatched_and_the_run_stops_before_its_next_motion(self) -> None:
        pressed: dict[str, Any] = {}

        def brake_at_the_grasp() -> None:
            pressed["answer"] = self.client.post("/v1/cell/brake").json()

        arm, _log = self._latchless([Scripted("part", on_executing=brake_at_the_grasp), Scripted("part")])
        run = self.task(object="green cube", scope="until_empty")
        body = self.finished(run["id"])
        self.assertEqual((True, False, False), (pressed["answer"]["run_halted"], pressed["answer"]["latched"],
                                                pressed["answer"]["braking"]))
        self.assertIn("ends first", pressed["answer"]["message"])
        self.assertEqual("halted", body["stop_code"], body)
        self.assertEqual(1, len([e for e in events_of(self.cell, run["id"]) if e.type == "pick.pick_started"]),
                         "a second pick started after the halt")


class TheCellIsClearRefusesTests(ConsoleCase):
    """``POST /v1/cell/acknowledge`` (build plan 1.3.5): refused ``not_built``, ``run_active``, ``controller_stopped``
    (pinned in ``tests/test_api_never_clears_a_protective_stop.py``) and ``jaws_not_open``; a refusal clears nothing."""

    def test_nothing_built_has_nothing_to_say_clear_of(self) -> None:
        self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}), 409, "not_built")

    def test_a_run_still_active_keeps_the_latch_an_abandoned_one_included(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        self.assertEqual(200, self.client.post("/v1/cell/brake").status_code)
        refused = self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}), 409,
                               "run_active")
        self.assertEqual(run["id"], refused["detail"]["run_id"])
        self.assertIsNotNone(cell.arm.halt_state(), "the latch was cleared while the run's thread may be inside a verb")
        # Disconnected, the run is abandoned, and its thread still lives: the same.
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}), 409, "run_active")
        self.assertIsNotNone(cell.arm.halt_state())
        self.assertNotIn("clear_halt", cell.arm.calls)
        release.set()
        self.finished(run["id"])
        cleared = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
        self.assertEqual(200, cleared.status_code, cleared.text)
        self.assertIsNone(cell.arm.halt_state())

    def test_backen_leer_on_a_toggle_whose_count_is_not_open(self) -> None:
        cell = self.scripted()
        cell.jaws.set_closed(True)
        refused = self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True}),
                               409, "jaws_not_open")
        self.assertEqual("closed", refused["detail"]["jaws"])
        cell.jaws.set_closed(False)
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined] # switched at the pendant
        refused = self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True}),
                               409, "jaws_not_open")
        self.assertEqual("unknown", refused["detail"]["jaws"])
        self.assertNotIn("detach_payload", cell.arm.calls)
        self.assertNotIn("cell.acknowledged", event_types(self.cell, "cell"))

    def test_backen_leer_on_a_hand_that_measures_a_part(self) -> None:
        from tests._task_fakes import MeasuringHand

        hand = MeasuringHand(holding=True)
        cell = self.scripted(hand=hand)
        refused = self.refused(self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True}),
                               409, "jaws_not_open")
        self.assertIn("measures a part", refused["message"])
        self.assertNotIn("detach_payload", cell.arm.calls)
        hand.holding = False
        emptied = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True})
        self.assertEqual(200, emptied.status_code, emptied.text)

    def test_backen_leer_tells_the_planner_the_hand_is_empty(self) -> None:
        cell = self.scripted()
        cell.arm.attach_payload(40.0)
        self.assertEqual("planner_and_filter", self.client.get("/v1/cell").json()["payload_model"])
        emptied = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True})
        self.assertEqual(200, emptied.status_code, emptied.text)
        self.assertEqual("none", emptied.json()["payload_model"])
        self.assertIn("detach_payload", cell.arm.calls)
        self.assertTrue(wait_for_event(self.cell, "cell", "cell.acknowledged").data["jaws_emptied"])


class TheHaltRecordSaysWhatBecameOfTheMoveTests(unittest.TestCase):
    def test_the_brake_word_is_passed_on_only_as_the_library_says_it(self) -> None:
        from api.routers.cell import _halted
        from src.robot.core.arm_capabilities import HaltState

        class Latched:
            def __init__(self, state: Any) -> None:
                self.state = state

            def halt_state(self) -> Any:
                return self.state

        for brake in ("none", "pending", "braked", "unconfirmed", "ran_out"):
            with self.subTest(brake=brake):
                state = HaltState(reason="halt now", requested_at=1.0, in_motion=brake != "none", brake=brake)
                said = _halted(Latched(state))
                assert said is not None
                self.assertEqual(brake, said.brake)
        import types

        odd = types.SimpleNamespace(reason="halt now", requested_at=1.0, in_motion=True, braked=False, brake_s=None,
                                    brake="still_going")
        said = _halted(Latched(odd))
        assert said is not None
        self.assertIsNone(said.brake, "a word the library does not say was passed on")

    def test_braking_is_said_only_while_a_brake_is_running(self) -> None:
        from api.routers.cell import _brake_answer
        from src.robot.core.arm_capabilities import HaltState

        class BrakingArm:
            def __init__(self, brakes: bool) -> None:
                self.brakes = brakes

            def halt(self, reason: str) -> HaltState:
                return HaltState(reason=reason, requested_at=1.0, in_motion=True)

            def clear_halt(self) -> None:
                return None

            def halt_state(self) -> HaltState | None:
                return None

            def brakes_in_motion(self) -> bool:
                return self.brakes

        latched, in_motion, braking = _brake_answer(BrakingArm(True), "halt now")
        self.assertEqual((True, True, True), (latched, in_motion, braking))
        latched, in_motion, braking = _brake_answer(BrakingArm(False), "halt now")
        self.assertEqual((True, True, False), (latched, in_motion, braking), "an arm that lets the move run braked")
        self.assertEqual((False, False, False), _brake_answer(object(), "halt now"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
