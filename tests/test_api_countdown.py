"""Hands off: after a person's hands were at the arm, the next motion the robot makes by itself counts down first.

Build plan item 11 and the owner's rule ("before the robot drives by itself: hands off, Enter + 3 s countdown"):
after a teach session, after "open now" (a person held the part at the jaws), or after "Backen leer" (a person emptied
the hand: owner decision Q2 = A), the next moving run of any kind (a task, a Restart, Home, a pick) counts down three
seconds on the server before its first motion (``run_countdown``), and a stop or a halt during it ends the run
``cancelled`` with nothing moved. What this file pins:

* three ``run_countdown`` events, 3, 2, 1, with why, and nothing sent to the arm before the last of them has passed;
* a stop or a halt during it: ``cancelled``, nothing moved, no recovery record, and the countdown still due; a stop is
  read within a tenth of a second, and a Home run is stopped by the task's stop, which latches nothing;
* the run's plan says it counts down, and a motion since the stamp means no countdown;
* only a run that moved the arm takes the countdown with it: one that counted down and then moved nothing (its screen
  refused a pose, the planner refused its move home before anything was sent) leaves it due for the next;
* "Backen leer" counts down as "open now" does, and on a toggle hand it answers no jaws question;
* the shipped timing is three steps of one second, read every 100 ms;
* a pick run counts down the same, and a pick run on a console that was never stamped emits no countdown at all.

Honesty bucket (2): the real console, registry and drivers; the countdown's one-second step is shortened to keep the
suite fast, its count and its order are the real ones.
"""

from __future__ import annotations

import time
import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, event_types, events_of, wait_for_event


class AHomeAfterATeachCountsDownTests(ConsoleCase):
    def test_three_seconds_counted_down_before_the_first_motion(self) -> None:
        cell = self.scripted()
        self.cell.stamp_teach_ended()
        motions_at: list[int] = []
        original = self.cell.hub.publish

        def publish(stream: str, event_type: str, /, **keywords: Any) -> Any:
            if event_type == "run_countdown":
                motions_at.append(len(cell.motions()))
            return original(stream, event_type, **keywords)

        self.cell.hub.publish = publish  # type: ignore[method-assign]
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        body = self.finished(home.json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        countdown = [event.data for event in events_of(self.cell, body["id"]) if event.type == "run_countdown"]
        self.assertEqual([3, 2, 1], [data["seconds_left"] for data in countdown])
        self.assertEqual({"teach"}, {data["because"] for data in countdown})
        self.assertEqual([0, 0, 0], motions_at, "the arm moved while the countdown ran")
        types = event_types(self.cell, body["id"])
        self.assertEqual(["run_started", "run_countdown", "run_countdown", "run_countdown", "home.started"], types[:5])
        self.assertFalse(self.cell.countdown_due(), "the motion did not end the hands-off countdown")
        self.assertEqual([("home",)], cell.motions())

    def test_a_stop_during_the_countdown_cancels_with_nothing_moved(self) -> None:
        cell = self.scripted()
        self.cell.registry.countdown_step_s = 0.5
        self.cell.stamp_jaws_open(opened=True)
        run = self.task(object="green cube")
        self.assertTrue(run["plan"]["countdown"])
        wait_for_event(self.cell, run["id"], "run_countdown")
        stopped = self.client.post("/v1/task/stop", params={"run_id": run["id"]})
        self.assertEqual(200, stopped.status_code, stopped.text)
        body = self.finished(run["id"])
        self.assertEqual(("cancelled", "cancelled", "operator", "countdown"),
                         (body["state"], body["stop_code"], body["stop_class"], body["step"]), body)
        self.assertEqual([], cell.motions(), "the arm moved although the countdown was stopped")
        self.assertEqual(0, cell.do0_changes())
        self.assertIsNone(self.cell.recovery)
        self.assertTrue(self.cell.countdown_due(), "a run that never moved took the countdown with it")
        self.assertEqual("jaws_opened", wait_for_event(self.cell, run["id"], "run_countdown").data["because"])
        self.assertNotIn("task.part_started", event_types(self.cell, run["id"]))

    def test_a_halt_during_the_countdown_cancels_with_nothing_moved(self) -> None:
        cell = self.scripted()
        self.cell.registry.countdown_step_s = 0.5
        self.cell.stamp_teach_ended()
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        wait_for_event(self.cell, home.json()["id"], "run_countdown")
        braked = self.client.post("/v1/cell/brake")
        self.assertEqual(200, braked.status_code, braked.text)
        body = self.finished(home.json()["id"])
        self.assertEqual(("cancelled", "cancelled"), (body["state"], body["stop_code"]), body)
        self.assertTrue(body["halt_requested"])
        self.assertEqual([], cell.motions())
        self.assertIsNone(self.cell.recovery)

    def test_no_countdown_once_the_arm_moved_since_the_stamp(self) -> None:
        cell = self.scripted([Scripted("part"), Scripted("part")])
        self.cell.stamp_teach_ended()
        self.cell.stamp_motion_started()
        run = self.task(object="green cube")
        self.assertFalse(run["plan"]["countdown"])
        self.finished(run["id"])
        self.assertNotIn("run_countdown", event_types(self.cell, run["id"]))
        self.assertEqual(2, cell.do0_changes())

    def test_the_shipped_countdown_is_three_seconds_read_every_tenth(self) -> None:
        from api.events import EventHub
        from api.runs import RunRegistry

        registry = RunRegistry(EventHub())
        self.assertEqual((3, 1.0, 0.1), (registry.countdown_steps, registry.countdown_step_s,
                                         registry.countdown_poll_s))

    def test_a_stop_is_read_within_a_tenth_of_a_second_not_at_the_end_of_a_step(self) -> None:
        self.scripted([Scripted("part")])
        self.cell.registry.countdown_step_s = 5.0
        self.cell.stamp_teach_ended()
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "run_countdown")
        asked = time.monotonic()
        self.assertEqual(200, self.client.post("/v1/task/stop", params={"run_id": run["id"]}).status_code)
        body = self.finished(run["id"], timeout=30.0)
        self.assertLess(time.monotonic() - asked, 2.0, "the stop was read only once a countdown step had ended")
        self.assertEqual(("cancelled", "cancelled"), (body["state"], body["stop_code"]), body)


class ACountdownStaysDueUntilTheArmMovedTests(ConsoleCase):
    """Only a run that moved the arm takes the countdown with it: a run that counted down and then moved nothing leaves
    it due, so the next run's first motion still comes after one."""

    def test_a_task_whose_screen_refused_its_pose_leaves_it_due_and_the_next_task_counts_down(self) -> None:
        cell = self.scripted([Scripted("part")], arm_keywords={"unreachable_above_mm": 100.0})
        self.cell.stamp_jaws_open(opened=True)
        refused = self.finished(self.task(object="green cube")["id"])
        self.assertEqual("pose_refused", refused["stop_code"], refused)
        self.assertEqual(3, event_types(self.cell, refused["id"]).count("run_countdown"))
        self.assertEqual([], cell.motions(), "the refused task moved the arm")
        self.assertTrue(self.cell.countdown_due(), "a task that moved nothing took the hands-off countdown with it")
        cell.arm.unreachable_above_mm = None
        second = self.task(object="green cube")
        self.assertTrue(second["plan"]["countdown"])
        body = self.finished(second["id"])
        self.assertEqual("finished", body["stop_code"], body)
        types = event_types(self.cell, second["id"])
        self.assertEqual(["run_started", "run_countdown", "run_countdown", "run_countdown"], types[:4])
        self.assertLess(types.index("run_countdown"), types.index("task.part_started"))
        self.assertFalse(self.cell.countdown_due(), "the arm moved, and the countdown is still due")

    def test_a_home_whose_move_the_planner_refused_before_sending_leaves_it_due(self) -> None:
        from src.robot.core import MotionStatus

        cell = self.scripted()
        cell.arm.refuse = lambda _kind, _index, _target: MotionStatus.WORKSPACE_REJECTED
        self.cell.stamp_teach_ended()
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        body = self.finished(home.json()["id"])
        self.assertEqual("return_failed", body["stop_code"], body)
        self.assertEqual(3, event_types(self.cell, body["id"]).count("run_countdown"))
        self.assertEqual([], cell.motions())
        self.assertTrue(self.cell.countdown_due(), "a move refused before it was sent took the countdown with it")

    def test_a_restart_whose_return_was_refused_before_sending_leaves_it_due(self) -> None:
        from src.robot.core import MotionStatus

        cell = self.scripted([Scripted("failed")] * 3)
        stopped = self.finished(self.task(object="green cube")["id"])
        self.assertEqual("failed_in_a_row", stopped["stop_code"], stopped)
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.cell.stamp_jaws_open(opened=True)
        moved = len(cell.motions())
        cell.arm.refuse = lambda kind, _index, _target: MotionStatus.WORKSPACE_REJECTED if kind == "home" else None
        restarted = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restarted.status_code, restarted.text)
        body = self.finished(restarted.json()["id"])
        self.assertEqual("return_failed", body["stop_code"], body)
        self.assertEqual(moved, len(cell.motions()), "the refused return moved the arm")
        self.assertTrue(self.cell.countdown_due())


class BackenLeerCountsDownTests(ConsoleCase):
    """Owner decision Q2 = A: a person who emptied the hand ("Backen leer") had their hands at the jaws, so the next
    motion counts down 3 s, as after "open now". On a toggle hand it answers no jaws question: that stays the toggle's
    way back."""

    def _stopped_holding(self) -> dict[str, Any]:
        """console_dummy, halted right after its pick: the stopped run believes a part is in the jaws, and the dummy
        hand measures nothing, so only the person's "Backen leer" ends that belief."""
        self.build_dummy()
        from api import task_run

        original = task_run.ConsoleTaskHooks.pick_done

        def pick_done(hooks: Any, part: int, pick: int, report: Any) -> None:
            original(hooks, part, pick, report)
            self.client.post("/v1/cell/brake")

        with patch.object(task_run.ConsoleTaskHooks, "pick_done", pick_done):
            stopped = self.finished(self.task(object="")["id"])
        self.assertEqual(("halted", True), (stopped["stop_code"], stopped["holding"]), stopped)
        cleared = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
        self.assertEqual(200, cleared.status_code, cleared.text)
        self.assertFalse(cleared.json()["countdown_due"], "the arm moved since the last stamp")
        return stopped

    def test_a_restart_after_backen_leer_counts_down_before_its_first_motion(self) -> None:
        stopped = self._stopped_holding()
        emptied = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True})
        self.assertEqual(200, emptied.status_code, emptied.text)
        self.assertTrue(emptied.json()["countdown_due"], "a person just emptied the hand, and nothing counts down")
        answered = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, answered.status_code, answered.text)
        self.assertTrue(answered.json()["plan"]["countdown"])
        body = self.finished(answered.json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        types = event_types(self.cell, body["id"])
        self.assertEqual(["run_started", "run_countdown", "run_countdown", "run_countdown"], types[:4])
        self.assertLess(types.index("run_countdown"), types.index("task.return_started"))
        because = {event.data["because"] for event in events_of(self.cell, body["id"]) if event.type == "run_countdown"}
        self.assertEqual({"jaws_opened"}, because, "the hands at the jaws, in the contract's word for it")
        self.assertFalse(self.cell.countdown_due())

    def test_home_after_backen_leer_counts_down_before_its_first_motion(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        def held(_console: Any, run: Any) -> str:
            run.holding = True
            run.error = "the place was refused before the release"
            return "part_still_held"

        stopped = self.cell.registry.start_kind(self.cell, RunKind.TASK, held)
        self.finished(stopped.id)
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "part_still_held")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge",
                                               json={"cell_clear": True, "jaws_empty": True}).status_code)
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        body = self.finished(home.json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual(["run_started", "run_countdown", "run_countdown", "run_countdown", "home.started"],
                         event_types(self.cell, body["id"])[:5])
        self.assertIsNone(self.cell.recovery)

    def test_on_a_toggle_backen_leer_counts_down_and_answers_no_jaws_question(self) -> None:
        cell = self.scripted([Scripted("failed")] * 3 + [Scripted("part")])
        stopped = self.finished(self.task(object="green cube")["id"])
        self.assertEqual("failed_in_a_row", stopped["stop_code"], stopped)
        emptied = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True})
        self.assertEqual(200, emptied.status_code, emptied.text)
        self.assertTrue(emptied.json()["countdown_due"])
        self.assertIsNone(emptied.json()["jaws_confirmed_at"], "'Backen leer' was taken for an answer to the jaws")
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "jaws_not_confirmed")
        self.cell.stamp_jaws_open(opened=False)  # the jaws question answered open: no change of DO0
        restarted = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restarted.status_code, restarted.text)
        body = self.finished(restarted.json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual(3, event_types(self.cell, body["id"]).count("run_countdown"))
        self.assertEqual(2, cell.do0_changes(), "one close at the part and one open at the drop, nothing more")


class AHomeRunIsStoppedBeforeItsMoveTests(ConsoleCase):
    """Build plan item 11 and 1.3.6: a stop during a Home run's countdown ends it ``cancelled`` with nothing moved. The
    task's stop (``POST /v1/task/stop``) stops a Home run before its one move is sent; it latches nothing, so no "the
    cell is clear" is owed, as the brake would owe one. The pick's stop stays a pick run's."""

    def test_a_stop_during_a_home_countdown_cancels_it_with_nothing_moved_and_nothing_latched(self) -> None:
        cell = self.scripted()
        self.cell.registry.countdown_step_s = 0.5
        self.cell.stamp_teach_ended()
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        run_id = home.json()["id"]
        wait_for_event(self.cell, run_id, "run_countdown")
        self.refused(self.client.post("/v1/pick/stop"), 409, "not_a_pick")
        stopped = self.client.post("/v1/task/stop")
        self.assertEqual(200, stopped.status_code, stopped.text)
        self.assertTrue(stopped.json()["stop_requested"])
        body = self.finished(run_id)
        self.assertEqual(("cancelled", "cancelled", "operator", "countdown"),
                         (body["state"], body["stop_code"], body["stop_class"], body["step"]), body)
        self.assertEqual([], cell.motions(), "the arm moved although its countdown was stopped")
        self.assertNotIn("home.started", event_types(self.cell, run_id))
        self.assertIsNone(cell.arm.halt_state(), "stopping a countdown latched the arm")
        self.assertIsNone(self.cell.recovery)
        self.assertTrue(self.cell.countdown_due(), "a run that never moved took the countdown with it")
        (asked,) = [event for event in events_of(self.cell, run_id) if event.type == "run_stop_requested"]
        self.assertEqual("before_motion", asked.data["scope"], "a Home run's stop acts before its one move is sent")

    def test_a_home_run_stopped_before_its_move_sends_nothing(self) -> None:
        """The stop read once more right before the move: one that lands after the countdown ended still sends nothing."""
        from api.codes import RunKind, StopCode
        from api.runs import Run
        from api.task_run import drive_home

        cell = self.scripted()
        run = Run(id="run-0000000001", prompt="", requested_picks=0, kind=RunKind.HOME, inputs={"to": "home"},
                  stop_requested=True)
        self.assertEqual(StopCode.CANCELLED, drive_home(self.cell, run))
        self.assertEqual([], cell.motions())
        self.assertNotIn("home.started", event_types(self.cell, run.id))


class APickRunCountsDownToo(ConsoleCase):
    def test_a_pick_after_open_now_counts_down_before_its_first_pick(self) -> None:
        cell = self.scripted([Scripted("part")])
        self.cell.stamp_jaws_open(opened=True)
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        self.assertEqual(202, pick.status_code, pick.text)
        body = self.finished(pick.json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        types = event_types(self.cell, body["id"])
        self.assertEqual(["run_started", "run_countdown", "run_countdown", "run_countdown"], types[:4])
        self.assertEqual(1, cell.do0_changes(), "one close, at the part")

    def test_a_pick_stopped_during_its_countdown_starts_no_pick(self) -> None:
        cell = self.scripted([Scripted("part")])
        self.cell.registry.countdown_step_s = 0.5
        self.cell.stamp_teach_ended()
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        wait_for_event(self.cell, pick.json()["id"], "run_countdown")
        self.assertEqual(200, self.client.post("/v1/pick/stop").status_code)
        body = self.finished(pick.json()["id"])
        self.assertEqual(("cancelled", "cancelled", 0), (body["state"], body["stop_code"], body["attempted"]))
        self.assertEqual([], cell.motions())

    def test_a_bare_console_s_pick_run_emits_no_countdown(self) -> None:
        """``tests/test_api_pick.py`` pins a zero-pick run on a bare console to run_started, run_finished."""
        import types as pytypes

        from api.events import EventHub
        from api.runs import RunRegistry

        hub = EventHub()
        service = pytypes.SimpleNamespace(attach_progress_listener=lambda _l: None, set_cancel_check=lambda _c: None)
        bare = pytypes.SimpleNamespace(active_run_id=None, session=pytypes.SimpleNamespace(service=service))
        run = RunRegistry(hub).start(bare, prompt="", picks=0)
        wait_for_event(pytypes.SimpleNamespace(hub=hub), run.id, "run_finished")
        self.assertEqual(["run_started", "run_finished"], [event.type for event in hub.since(run.id, 0)[0]])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
