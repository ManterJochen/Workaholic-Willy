"""``POST /v1/pick`` keeps its route and gains the gates every moving route has; its run says how it stopped.

Build plan item 17 and 1.2: the Pick screen goes, the route stays (programs and ``tests/test_api_pick.py`` use it), and
before anything starts it refuses ``halted``, ``cell_not_cleared``, ``restart_required``, ``needs_person`` and
``jaws_question_pending``. A pick run started anyway on a service a person is needed for ends ``recovery_needs_person``
with nothing commanded and the latch kept: no run clears it silently any more (``start_campaign`` would). ``POST
/v1/pick/stop`` refuses a run that is not a pick run, whose stop means something else. What this file pins:

* each new refusal, nothing moved, no run started, the service's latch kept (scenario 14); a run holding the lock is
  said first, as 1.3.2 orders it;
* no moving run starts while a recovery record stands, whatever a route's gates read before: the registry reads the
  record under its own lock, where a run that stopped on a problem in between has already written it;
* the run body reads the service's latch before its campaign, and keeps it;
* a pick run carries its stop code beside the pinned sentences: finished, cancelled, hand_needs_person, halted; a pick
  a halt cut short ends ``halted`` whatever else its report says;
* ``POST /v1/pick/stop`` on a task run answers ``409 not_a_pick``, and still stops a pick run between attempts.

Honesty bucket (2): the real console, routes and registry, on console_dummy and on the scripted cell.
"""

from __future__ import annotations

import threading
import types
import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, await_run, event_types, wait_for_event


class ThePickRouteRefusesTests(ConsoleCase):
    def test_a_person_needed_refuses_and_keeps_the_latch(self) -> None:
        """Scenario 14."""
        self.build_dummy()
        service = self.cell.session.service
        service._needs_person = "a push stopped where the arm stands"  # noqa: SLF001
        refused = self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "needs_person")
        self.assertIn("a push stopped where the arm stands", refused["message"])
        self.assertEqual("a push stopped where the arm stands", service.stopped_where_the_arm_stands)
        self.assertEqual([], self.client.get("/v1/runs").json())

    def test_a_halted_arm(self) -> None:
        self.build_dummy()
        self.cell.session.arm.halt("the operator pressed halt now")
        self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "halted")

    def test_the_recovery_record(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "cell_fault")
        self.finished(run.id)
        self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "cell_not_cleared")
        self.cell.mark_cell_clear()
        self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "restart_required")

    def test_a_jaws_question_waiting(self) -> None:
        self.build_dummy()
        with patch("api.jaws.pending", return_value=object()):
            self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "jaws_question_pending")

    def test_a_run_holding_the_lock_is_said_before_anything_else(self) -> None:
        """1.3.2's order: ``run_active`` before ``halted`` and before the recovery record."""
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        cell.arm.halt("a halt latched by another program")
        refused = self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "run_active")
        self.assertEqual(run["id"], refused["detail"]["run_id"])
        release.set()
        self.finished(run["id"])


class NoMovingRunStartsWhileAStopStandsTests(ConsoleCase):
    """The recovery record gates every moving run in the registry too, under the run lock: a route reads its gates
    before it starts the run, and a run that ended on a problem in between wrote its record before it let go of the
    lock, so the registry's own read cannot miss it."""

    def test_a_pick_whose_gates_passed_is_refused_where_a_stop_landed_before_it_started(self) -> None:
        import api.routers.pick as pick_router
        from api.codes import RunKind

        cell = self.scripted([Scripted("part")])
        real = pick_router._refuse_unfit_push
        landed: list[str] = []

        def a_run_stops_on_a_problem_meanwhile(console: Any, push_mm: Any) -> None:
            stopped = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "return_failed")
            await_run(self.cell, stopped.id)
            landed.append(stopped.id)
            real(console, push_mm)

        with patch.object(pick_router, "_refuse_unfit_push", a_run_stops_on_a_problem_meanwhile):
            answered = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        self.refused(answered, 409, "cell_not_cleared")
        self.assertEqual(landed, [self.cell.recovery.run_id])  # type: ignore[union-attr]
        self.assertEqual([], cell.motions(), "a pick run moved the arm from where a problem stop left it")
        self.assertEqual(landed, [run["id"] for run in self.client.get("/v1/runs").json()])

    def test_the_registry_admits_no_moving_run_while_a_stop_stands(self) -> None:
        from api.codes import RunKind
        from api.runs import RunRefused

        self.build_dummy()
        stopped = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "cell_fault")
        self.finished(stopped.id)
        nothing = "finished"
        moving: dict[str, Any] = {
            "pick": lambda: self.cell.registry.start(self.cell, prompt="", picks=1),
            "task": lambda: self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: nothing),
            "home": lambda: self.cell.registry.start_kind(self.cell, RunKind.HOME, lambda _c, _r: nothing,
                                                          inputs={"to": "home"}),
            "restart": lambda: self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: nothing,
                                                             restart_of=stopped.id),
        }
        for name, start in moving.items():
            with self.subTest(before_the_cell_is_clear=name), self.assertRaises(RunRefused) as refused:
                start()
            self.assertEqual("cell_not_cleared", refused.exception.code, name)
        # A run that moves nothing is no motion after the stop: the planner starts.
        planner = self.cell.registry.start_kind(self.cell, RunKind.PLANNER, lambda _c, _r: "planner_ready")
        self.finished(planner.id)
        self.cell.mark_cell_clear()
        for name in ("pick", "task"):
            with self.subTest(restart_required=name), self.assertRaises(RunRefused) as refused:
                moving[name]()
            self.assertEqual("restart_required", refused.exception.code, name)
        with self.assertRaises(RunRefused) as refused:
            self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: nothing, restart_of="run-another")
        self.assertEqual("not_restartable", refused.exception.code)
        # The two ways back remain: Home, and the Restart of the record's own run.
        for name in ("home", "restart"):
            with self.subTest(way_back=name):
                self.finished(moving[name]().id)
        self.assertEqual(1 + 1 + 2, len(self.client.get("/v1/runs").json()))


class AJawsCheckThatBeganAsTheRunStartedTests(ConsoleCase):
    """A route reads the jaws question among its gates, and a check of the jaws marks itself pending before it reads
    the run lock. In the window between the two, a moving run whose gates read no question can start after the check
    began: the run reads the question again on its own thread, once it holds the cell, and ends before its first motion
    (``cancelled``, nothing moved), so a run and a check never go on together. The run
    types why (``RunOut.refusal``: ``jaws_question_pending``, what the route would have answered), so the cockpit says
    "answer the jaws question, then start again" in its own words, never by reading the English error."""

    GAVE_WAY = {"code": "jaws_question_pending", "status": 409}

    def _a_check_begins(self, start: Any) -> Any:
        """Start a run the way ``start`` does, as a check of the jaws begins right after the route's gates."""
        seam = self.cell.jaws
        flows: list[Any] = []

        def as_a_check_begins(*args: Any, **kwargs: Any) -> Any:
            flows.append(seam.open_flow("check"))          # what POST /v1/cell/jaws/check does first
            return start(*args, **kwargs)

        self.addCleanup(lambda: [seam.close_flow(flow, opened=False) for flow in flows])
        return as_a_check_begins

    def test_a_task_ends_before_its_first_motion(self) -> None:
        cell = self.scripted([Scripted("part")])
        registry = self.cell.registry
        with patch.object(registry, "start_kind", side_effect=self._a_check_begins(registry.start_kind)):
            run = self.task(object="green cube")
        body = self.finished(run["id"])
        self.assertEqual("cancelled", body["stop_code"], body)
        self.assertIn("jaws question began as the run started", body["error"])
        self.assertIn("nothing moved", body["error"])
        self.assertEqual(self.GAVE_WAY, body["refusal"], body)
        self.assertEqual([], cell.motions(), "a task moved while a check of the jaws was under way")
        self.assertEqual(0, cell.do0_changes())
        self.assertNotIn("pick.pick_started", event_types(self.cell, run["id"]))

    def test_a_pick_run_ends_before_its_first_pick(self) -> None:
        cell = self.scripted([Scripted("part")])
        registry = self.cell.registry
        with patch.object(registry, "start", side_effect=self._a_check_begins(registry.start)):
            answered = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        self.assertEqual(202, answered.status_code, answered.text)
        body = self.finished(answered.json()["id"])
        self.assertEqual(("cancelled", "cancelled"), (body["stop_code"], body["state"]), body)
        self.assertIn("jaws question began as the run started", body["error"])
        self.assertEqual(self.GAVE_WAY, body["refusal"], body)
        self.assertEqual([], cell.motions(), "a pick moved while a check of the jaws was under way")

    def test_home_ends_before_its_first_motion_and_no_run_that_moves_nothing_waits_for_it(self) -> None:
        from api.codes import RunKind

        cell = self.scripted()
        registry = self.cell.registry
        start = self._a_check_begins(registry.start_kind)
        home = start(self.cell, RunKind.HOME, lambda _c, _r: "finished", inputs={"to": "home"})
        ended = self.finished(home.id)
        self.assertEqual(("cancelled", self.GAVE_WAY), (ended["stop_code"], ended["refusal"]), ended)
        self.assertEqual([], cell.motions())
        # A run that moves nothing is no motion: the planner starts while the check is under way, and refuses nothing.
        planner = registry.start_kind(self.cell, RunKind.PLANNER, lambda _c, _r: "planner_ready")
        ready = self.finished(planner.id)
        self.assertEqual(("planner_ready", None), (ready["stop_code"], ready["refusal"]), ready)


class ThePickRunReadsTheLatchFirstTests(unittest.TestCase):
    def test_a_run_started_on_a_latched_service_commands_nothing_and_keeps_the_latch(self) -> None:
        from api.events import EventHub
        from api.runs import RunRegistry

        calls: list[str] = []

        class Latched:
            stopped_where_the_arm_stands = "a push stopped where the arm stands"

            def attach_progress_listener(self, _listener: Any) -> None:
                calls.append("attach_progress_listener")

            def set_cancel_check(self, _check: Any) -> None:
                calls.append("set_cancel_check")

            def start_campaign(self, **_keywords: Any) -> None:  # pragma: no cover - the failure this pins
                calls.append("start_campaign")
                self.stopped_where_the_arm_stands = ""

            def pick(self, **_keywords: Any) -> Any:  # pragma: no cover - the failure this pins
                calls.append("pick")

        service = Latched()
        hub = EventHub()
        console = types.SimpleNamespace(active_run_id=None, session=types.SimpleNamespace(service=service))
        run = RunRegistry(hub).start(console, prompt="", picks=2)
        deadline = threading.Event()
        for _ in range(2000):
            if any(event.type == "run_finished" for event in hub.since(run.id, 0)[0]):
                break
            deadline.wait(0.005)
        self.assertEqual(("failed", "recovery_needs_person", "problem"),
                         (str(run.state), str(run.stop_code), str(run.stop_class)))
        self.assertNotIn("start_campaign", calls, "the run cleared the latch before it read it")
        self.assertNotIn("pick", calls)
        self.assertEqual("a push stopped where the arm stands", service.stopped_where_the_arm_stands)
        self.assertIn("a push stopped where the arm stands", run.error)
        self.assertEqual(0, run.attempted)


class APickRunSaysHowItStoppedTests(ConsoleCase):
    def test_finished_and_cancelled(self) -> None:
        self.build_dummy()
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        self.assertEqual(("finished", "finished", "done"),
                         tuple(self.finished(pick.json()["id"])[key] for key in ("state", "stop_code", "stop_class")))
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 50})
        self.assertEqual(200, self.client.post("/v1/pick/stop").status_code)
        body = self.finished(pick.json()["id"])
        self.assertEqual(("cancelled", "cancelled", "operator"), (body["state"], body["stop_code"], body["stop_class"]))
        self.assertIsNone(self.cell.recovery)

    def test_a_hand_that_needs_a_person_keeps_its_pinned_sentence(self) -> None:
        self.scripted([Scripted("part"), Scripted("part")])
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 2})
        body = self.finished(pick.json()["id"])
        self.assertEqual("hand_needs_person", body["stop_code"])
        self.assertIn("the gripper needs a person, so the run stops", body["error"])
        error = wait_for_event(self.cell, body["id"], "run_error").data
        self.assertEqual("hand_needs_person", error["stop_code"])
        self.assertEqual("pick", wait_for_event(self.cell, body["id"], "run_started").data["kind"])


class APickAHaltCutShortEndsHaltedTests(ConsoleCase):
    """Build plan 1.2: a pick run ends ``halted`` when the arm's halt latch is set as its pick ends, with its own
    sentence, whatever else the pick reports: a hand that refused on the latch, or a push of a recovery the latch ended,
    is the halt's doing, and the stop card's checklist is the halt's, never a gripper's or a recovery's."""

    def _braked_inside_the_pick(self, report: Any) -> dict[str, Any]:
        cell = self.scripted([Scripted("part")])

        def pick(**_keywords: Any) -> Any:
            braked = self.client.post("/v1/cell/brake")
            assert braked.status_code == 200, braked.text
            return report

        cell.service.pick = pick  # type: ignore[method-assign]
        answered = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        self.assertEqual(202, answered.status_code, answered.text)
        body = self.finished(answered.json()["id"])
        self.assertTrue(body["halt_requested"])
        self.assertIsNotNone(cell.arm.halt_state())
        return body

    def test_a_hand_that_refused_on_the_latch(self) -> None:
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
        from src.robot.grasping.loop.pick_loop import PickOutcome
        from tests._task_fakes import _report

        body = self._braked_inside_the_pick(_report(AutonomousGraspOutcome.EXECUTION_FAILED, telemetry={
            "low_level_outcome": str(PickOutcome.GRIPPER_FAULT),
            "gripper_fault": "ArmHalted: the arm is halted (halt now): output 0 was not switched"}))
        self.assertEqual(("failed", "halted", "problem"), (body["state"], body["stop_code"], body["stop_class"]), body)
        self.assertIn("the console halted the arm", body["error"])
        self.assertEqual("halted", wait_for_event(self.cell, body["id"], "run_error").data["stop_code"])
        self.assertEqual((body["id"], "halted"), (self.cell.recovery.run_id, self.cell.recovery.stop_code))  # type: ignore[union-attr]

    def test_a_recovery_the_latch_stopped_where_the_arm_stands(self) -> None:
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
        from tests._task_fakes import _report

        body = self._braked_inside_the_pick(_report(AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED,
                                                    telemetry={"push_stopped": "the push stopped: the arm is halted"}))
        self.assertEqual(("failed", "halted"), (body["state"], body["stop_code"]), body)
        self.assertIn("the console halted the arm", body["error"])


class ThePickStopIsAPickRunsTests(ConsoleCase):
    def test_a_task_run_is_not_stopped_by_the_pick_stop(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        self.refused(self.client.post("/v1/pick/stop"), 409, "not_a_pick")
        self.refused(self.client.post("/v1/pick/stop", params={"run_id": run["id"]}), 409, "not_a_pick")
        release.set()
        body = self.finished(run["id"])
        self.assertEqual("finished", body["stop_code"])
        self.assertFalse(body["stop_requested"])
        self.assertNotIn("run_stop_requested", event_types(self.cell, run["id"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
