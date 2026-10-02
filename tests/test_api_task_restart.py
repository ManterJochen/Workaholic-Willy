"""``POST /v1/task/restart``: after a problem stop, the way back starts with the planned move to the return pose.

Build plan 1.3.5 and the owner's OD 9: a problem stop leaves the arm where it stood and the console's recovery record.
Nothing new starts while it stands. A person first confirms "the cell is clear"; then Restart runs the stopped task's
plan again as a NEW run whose first motion is the planned move to its return pose, and arriving there ends the record.
What this file pins:

* Restart is refused until "the cell is clear" (``cell_not_cleared``), for any run but the record's
  (``not_restartable``), for a run that is neither a task nor a Home run, and for a run the console does not know;
* the Restart is a new task run of the stopped plan, ``restart_of`` the stopped run, ``first_motion`` ``return``: its
  poses are screened, then its first motion is the move home, whose arrival ends the record (``cell.recovery_ended``);
* a stopped Home run restarts as a Home run to the same pose;
* a toggle hand is asked where its jaws stand since the stop before a Restart moves (``jaws_not_confirmed``), and a
  hand that may still hold the part refuses it (``part_still_held``) until the person's word that it is empty;
* a Restart keeps every refusal of a new task but ``restart_required`` (1.3.5), read against the cell as it is now,
  which may have changed since the stop (a rebuild, a perception stack that grounds otherwise, a camera gone): a camera
  target it can no longer find, a phrase it would ground wrong, an empty object on a cell that grounds a phrase.

Honesty bucket (2): the real console and ``run_task``, on the scripted cell and on console_dummy.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, event_types, events_of, wait_for_event


def clear(case: ConsoleCase) -> None:
    """A person confirms the cell is clear, and answers in the browser that the toggle's jaws stand open (what
    ``api.jaws`` stamps once the jaws question ends open)."""
    acknowledged = case.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
    case.assertEqual(200, acknowledged.status_code, acknowledged.text)
    case.cell.stamp_jaws_open(opened=False)


class ARestartComesAfterTheCellIsClearTests(ConsoleCase):
    def _stopped(self, picks: list[Scripted] | None = None) -> tuple[Any, dict[str, Any]]:
        cell = self.scripted(picks or [Scripted("failed"), Scripted("failed"), Scripted("failed"), Scripted("part")])
        run = self.task(object="green cube")
        body = self.finished(run["id"])
        self.assertEqual("failed_in_a_row", body["stop_code"], body)
        return cell, body

    def test_a_restart_is_refused_until_a_person_said_the_cell_is_clear(self) -> None:
        cell, stopped = self._stopped()
        motions = list(cell.motions())
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "cell_not_cleared")
        self.assertEqual(motions, cell.motions(), "a refused restart moved the arm")

    def test_only_the_record_s_run_restarts(self) -> None:
        _cell, stopped = self._stopped()
        clear(self)
        self.refused(self.client.post("/v1/task/restart", json={"run_id": "run-nobody"}), 404, "no_such_run")
        # A planner run, which the console does know, is no stopped motion to come back from.
        planner = self.client.post("/v1/cell/planner")
        self.assertEqual(202, planner.status_code, planner.text)
        self.finished(planner.json()["id"])
        self.refused(self.client.post("/v1/task/restart", json={"run_id": planner.json()["id"]}), 409,
                     "not_restartable")
        self.assertEqual(stopped["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]

    def test_a_restart_of_an_older_problem_run_is_refused(self) -> None:
        cell, first = self._stopped([Scripted("failed")] * 6)
        clear(self)
        restarted = self.client.post("/v1/task/restart", json={"run_id": first["id"]})
        self.assertEqual(202, restarted.status_code, restarted.text)
        second = self.finished(restarted.json()["id"])
        self.assertEqual("failed_in_a_row", second["stop_code"], second)
        clear(self)
        self.refused(self.client.post("/v1/task/restart", json={"run_id": first["id"]}), 409, "not_restartable")
        self.assertEqual(second["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]
        self.assertGreater(len(cell.motions()), 0)

    def test_the_restart_s_first_motion_is_the_move_home_and_its_arrival_ends_the_record(self) -> None:
        cell, stopped = self._stopped()
        clear(self)
        before = len(cell.log)
        answered = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, answered.status_code, answered.text)
        restart = answered.json()
        self.assertEqual(("task", stopped["id"], "return"),
                         (restart["kind"], restart["restart_of"], restart["plan"]["first_motion"]))
        body = self.finished(restart["id"])
        self.assertEqual(("finished", "finished", 1), (body["state"], body["stop_code"], body["parts_placed"]), body)
        first_motion = next(entry for entry in cell.log[before:]
                            if isinstance(entry, tuple) and entry and entry[0] in ("move", "joints", "home"))
        self.assertEqual(("home",), first_motion, "the restart's first motion was not the planned move home")
        types = [kind for kind in event_types(self.cell, restart["id"]) if not kind.startswith("pick.")]
        self.assertEqual(["run_started", "task.pose_screened", "task.return_started", "task.returned"], types[:4])
        self.assertIsNone(self.cell.recovery, "arriving home did not end the record")
        ended = wait_for_event(self.cell, "cell", "cell.recovery_ended")
        self.assertEqual({"run_id": stopped["id"], "by": "restart", "ended_by": restart["id"]}, ended.data)
        # The plan is the stopped run's.
        for key in ("object", "place", "return_to", "scope"):
            self.assertEqual(stopped["plan"][key], body["plan"][key], key)

    def test_a_new_task_after_a_problem_stop_waits_for_the_restart(self) -> None:
        _cell, stopped = self._stopped()
        clear(self)
        self.refused(self.client.post("/v1/task", json={"object": "green cube", "place": {"kind": "pose", "pose": None}}),
                     409, "restart_required")
        self.refused(self.client.post("/v1/pick", json={"prompt": "", "picks": 1}), 409, "restart_required")
        restart = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restart.status_code, restart.text)
        self.finished(restart.json()["id"])
        self.assertIsNone(self.cell.recovery)


class AStoppedHomeRunRestartsAsAHomeRunTests(ConsoleCase):
    def test_a_home_run_whose_move_was_refused_restarts_to_the_same_pose(self) -> None:
        from src.robot.core import MotionStatus

        cell = self.scripted()
        cell.arm.refuse = lambda kind, _index, _target: MotionStatus.WORKSPACE_REJECTED if kind == "joints" else None
        home = self.client.post("/v1/cell/home", json={"to": "park"})
        self.assertEqual(202, home.status_code, home.text)
        stopped = self.finished(home.json()["id"])
        self.assertEqual(("failed", "return_failed", "home"), (stopped["state"], stopped["stop_code"], stopped["kind"]))
        cell.arm.refuse = None
        clear(self)
        restarted = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restarted.status_code, restarted.text)
        again = self.finished(restarted.json()["id"])
        self.assertEqual(("home", "finished", stopped["id"]), (again["kind"], again["stop_code"], again["restart_of"]))
        arrived = wait_for_event(self.cell, again["id"], "home.arrived").data
        self.assertEqual("park", arrived["to"])
        self.assertIsNone(self.cell.recovery)


class TheHandBeforeARestartTests(ConsoleCase):
    def test_a_toggle_restarts_only_once_its_jaws_were_answered_open_since_the_stop(self) -> None:
        cell = self.scripted([Scripted("failed")] * 3 + [Scripted("part")])
        run = self.task(object="green cube")
        stopped = self.finished(run["id"])
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "jaws_not_confirmed")
        # The jaws question answered open in the browser (api.jaws stamps it so).
        self.cell.stamp_jaws_open(opened=False)
        answered = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, answered.status_code, answered.text)
        self.finished(answered.json()["id"])
        self.assertEqual(2, cell.do0_changes())

    def test_a_part_still_held_refuses_the_restart_until_the_person_says_the_jaws_are_empty(self) -> None:
        """Scenario 7 on console_dummy: halt while carrying; the run's record says a part is held, the dummy hand
        cannot measure it, so only the person's "Backen leer" lets the Restart go."""
        self.build_dummy()

        def halt_after_the_pick(name: str) -> None:
            if name == "pick_result":
                self.client.post("/v1/cell/brake")

        from api import task_run

        original = task_run.ConsoleTaskHooks.pick_done

        def pick_done(hooks: Any, part: int, pick: int, report: Any) -> None:
            original(hooks, part, pick, report)
            halt_after_the_pick("pick_result")

        from unittest.mock import patch

        with patch.object(task_run.ConsoleTaskHooks, "pick_done", pick_done):
            run = self.task(object="")
            stopped = self.finished(run["id"])
        self.assertEqual(("halted", True), (stopped["stop_code"], stopped["holding"]), stopped)
        # The arm's latch is read before the record, as every moving route reads it (1.3.2).
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "halted")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "part_still_held")
        emptied = self.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True})
        self.assertEqual(200, emptied.status_code, emptied.text)
        acknowledged = [event.data for event in events_of(self.cell, "cell") if event.type == "cell.acknowledged"]
        self.assertTrue(acknowledged[-1]["jaws_emptied"])
        restarted = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restarted.status_code, restarted.text)
        self.assertEqual("finished", self.finished(restarted.json()["id"])["stop_code"])


def _stack(backend: str, weights_present: bool | None) -> Any:
    """What ``GET /v1/diagnostics`` says the cell grounds with: the route guard's one reading of the perception stack."""
    from api.schemas import PerceptionStackOut

    return PerceptionStackOut(
        pipeline_configured=True, kind="zero_shot", backend=backend, segmenter="sam2", router_enabled=backend == "vlm",
        vlm_model_id="Qwen/Qwen3-VL-4B-Instruct" if backend == "vlm" else None, vlm_weights_present=weights_present,
        vlm_on_unavailable="refuse", detail="scripted")


class ARestartKeepsTheRefusalsOfANewTaskTests(ConsoleCase):
    """Build plan 1.3.5: a Restart refuses every refusal of 1.3.2 but ``restart_required``, read against the cell as it
    is now: the record outlives a rebuild, and what the cell grounds or sees may have changed since the stop."""

    def _wrist_cell(self, phrase: str) -> Any:
        from tests._console_task_fakes import LOOK_1, LOOK_2, SightedLocator
        from tests._task_fakes import BIN_CENTRE, bin_object, bin_points

        cell = self.scripted([Scripted("failed", looks=("look_1",))] * 3, wrist=True, looks=(LOOK_1, LOOK_2))
        locator = SightedLocator({phrase: lambda _tcp: (bin_object(bin_points(BIN_CENTRE)),)}, arm=cell.arm,
                                 wrist=True)
        for target in ("src.robot.execution.place_target.locators_for_service",
                       "src.robot.execution.task.locators_for_service"):
            patcher = patch(target, return_value=[locator])
            patcher.start()
            self.addCleanup(patcher.stop)
        return cell

    def _stopped(self, **body: Any) -> dict[str, Any]:
        stopped = self.finished(self.task(**body)["id"])
        self.assertEqual("failed_in_a_row", stopped["stop_code"], stopped)
        clear(self)
        return stopped

    def test_a_camera_target_the_cell_can_no_longer_find(self) -> None:
        cell = self._wrist_cell("blue bin")
        stopped = self._stopped(object="red cube", place={"kind": "camera", "phrase": "blue bin"})
        cell.arm.live_planner_world = None
        moved = list(cell.motions())
        refused = self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409,
                               "camera_target_unavailable")
        self.assertIn("live planner world", refused["message"])
        self.assertEqual(moved, cell.motions())
        self.assertEqual(stopped["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]

    def test_an_object_phrase_the_cell_would_now_ground_wrong(self) -> None:
        self.scripted([Scripted("failed")] * 3 + [Scripted("part")])
        with patch("api.routers.diagnostics._perception", return_value=_stack("vlm", True)):
            stopped = self._stopped(object="den grünen Würfel")
        with patch("api.routers.diagnostics._perception", return_value=_stack("grounded_sam", None)):
            self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 422,
                         "prompt_not_routable")
        with patch("api.routers.diagnostics._perception", return_value=_stack("vlm", True)):
            restarted = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restarted.status_code, restarted.text)
        self.assertEqual("finished", self.finished(restarted.json()["id"])["stop_code"])

    def test_a_target_phrase_the_cell_would_now_ground_wrong(self) -> None:
        cell = self._wrist_cell("die blaue Kiste")
        with patch("api.routers.diagnostics._perception", return_value=_stack("vlm", True)):
            stopped = self._stopped(object="red cube", place={"kind": "camera", "phrase": "die blaue Kiste"})
        moved = list(cell.motions())
        with patch("api.routers.diagnostics._perception", return_value=_stack("grounded_sam", None)):
            self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 422,
                         "target_not_routable")
        self.assertEqual(moved, cell.motions())

    def test_an_empty_object_the_rehearsal_took_on_a_cell_that_grounds_a_phrase(self) -> None:
        """Q7 A+: the rehearsal picks what it shows; rebuilt as a cell whose detector grounds a phrase, the same plan
        would pick anything its camera sees, bin walls included, without the operator's word."""
        self.build_dummy()
        from api import task_run

        original = task_run.ConsoleTaskHooks.pick_done

        def pick_done(hooks: Any, part: int, pick: int, report: Any) -> None:
            original(hooks, part, pick, report)
            self.client.post("/v1/cell/brake")

        with patch.object(task_run.ConsoleTaskHooks, "pick_done", pick_done):
            stopped = self.finished(self.task(object="")["id"])
        self.assertEqual("halted", stopped["stop_code"], stopped)
        self.assertEqual(200, self.client.post("/v1/cell/disconnect").status_code)
        cell = self.scripted([Scripted("part")])  # the record stays: the console outlives the cell it stopped on
        clear(self)
        self.assertEqual(stopped["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]
        refused = self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 422,
                               "object_required")
        self.assertIn("pick_anything", refused["message"])
        self.assertEqual([], cell.motions())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
