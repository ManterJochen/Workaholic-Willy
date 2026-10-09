"""``POST /v1/task``: a task picks a part, places it, returns, and looks again; once, or until nothing is left.

The console's task is the library's ``run_task`` driven from a run thread (build plan 1.3): the route refuses what the
cell or the request is not fit for before anything moves, in the order of 1.3.2, answers 202 with the run at once, and
everything after that arrives on the run's event stream in the order of 1.3.3. What this file pins:

* on console_dummy, a task with a pose place ends ``finished`` after one part, and "until empty" places two parts of a
  scene that then empties and ends ``nothing_left``, each in the documented event order;
* on the scripted cell, whose arm knows where a pose puts the tool, the drop stands the part's hang above the taught
  pose, and "until empty" into a pose whose drop area holds the only parts left ends ``nothing_left`` and returns,
  never ``failed_in_a_row`` (scenario 3);
* "stop after this part" lets the part in hand be placed and the arm return, and ends ``stopped_after_part``;
* every refusal of 1.3.2 answers its code and status, moves nothing and starts no run, and the cell's state is refused
  before the request's; a refusal the library itself meets inside the run moves nothing and writes no record, and a
  controller stop it meets on a latched arm is the halt;
* a camera place surveys first, keeps the bin it found (its overlay served), and places into it; a bin that moved too
  far is lost, the part is put back, the arm returns and the task asks;
* the resolved plan is echoed in ``run_started`` and in the run, every event is plain JSON, and the history keeps the
  run's kind, stop code and parts placed.

Honesty bucket (2): the real console, routes, registry and ``run_task``; the rehearsal cell's real pick loop on the
dummy arm, or the scripted cell of ``tests/_console_task_fakes.py`` on the task tests' arm double and the real toggle hand.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import (
    ConsoleCase,
    EmptyingScene,
    POSES_DEG,
    Scripted,
    SightedLocator,
    event_types,
    events_of,
    wait_for_event,
)


def _without_pick_stages(types: list[str]) -> list[str]:
    return [kind for kind in types if not kind.startswith("pick.")]


class ATaskOnTheRehearsalCellTests(ConsoleCase):
    """console_dummy end to end: the rehearsal scene, the dummy arm, the real pick loop."""

    def test_once_with_the_default_place_picks_places_returns_and_ends_finished(self) -> None:
        from api.schemas import RunOut, TaskPlanOut

        self.build_dummy()
        run = self.task(object="", scope="once")
        self.assertEqual(("task", "running", ""), (run["kind"], run["state"], run["stop_code"]))
        body = self.finished(run["id"])
        self.assertEqual(("finished", "finished", "done", 1, 1, 1),
                         (body["state"], body["stop_code"], body["stop_class"], body["parts_placed"],
                          body["attempted"], body["succeeded"]), body)
        self.assertFalse(body["holding"])
        self.assertEqual("return", body["step"])
        self.assertIsNone(body["refusal"], "a task the library ran carries no refusal")

        types = event_types(self.cell, run["id"])
        self.assertEqual(
            ["run_started", "task.pose_screened", "task.part_started", "pick_result", "task.drop_planned",
             "task.place_started", "task.placed", "task.return_started", "task.returned", "task.part_finished",
             "run_finished"],
            _without_pick_stages(types))
        # The pick's own stages sit between the part's start and its result.
        first, result = types.index("task.part_started"), types.index("pick_result")
        stages = types[first + 1:result]
        self.assertEqual("pick.pick_started", stages[0])
        self.assertIn("pick.executing", stages)
        self.assertEqual("pick.pick_finished", stages[-1])

        events = events_of(self.cell, run["id"])
        plan = TaskPlanOut.model_validate(events[0].data["plan"])
        self.assertEqual(("pose", "drop_left", "Ablage links", "look", "once"),
                         (plan.place.kind, plan.place.pose, plan.place.pose_label, plan.first_motion, plan.scope))
        self.assertEqual(list(POSES_DEG["drop_left"][1]), plan.place.pose_joints_deg)
        self.assertIsNone(plan.return_joints_deg)
        self.assertEqual(plan.model_dump(), RunOut.model_validate(body).plan.model_dump())  # type: ignore[union-attr]
        # Every event is plain JSON, the backend's sentence in `human` and never in `data`.
        json.dumps([event.to_dict() for event in events])
        for event in events:
            self.assertTrue(event.human, event.type)
            self.assertNotIn("said", event.data, event.type)
            self.assertNotIn("image_png", event.data, event.type)

    def test_the_dummy_s_pose_drop_is_planned_and_said(self) -> None:
        """Where the drop stands is pinned on the scripted cell (``test_the_drop_stands_the_hang_above_the_taught_pose``):
        the dummy's fk answers where the arm stands, not where a pose puts the tool."""
        self.build_dummy()
        run = self.task(object="")
        self.finished(run["id"])
        drop = wait_for_event(self.cell, run["id"], "task.drop_planned").data
        self.assertEqual("pose", drop["kind"])
        self.assertGreater(drop["hang_mm"], 0.0, "the part was let go at the taught pose itself, not over it")

    def test_the_pick_result_says_the_part_the_pick_and_what_the_look_came_to(self) -> None:
        self.build_dummy()
        run = self.task(object="")
        self.finished(run["id"])
        result = wait_for_event(self.cell, run["id"], "pick_result").data
        for key, value in (("part", 1), ("pick", 1), ("succeeded", True), ("found_nothing", False),
                           ("only_excluded", False), ("detector_failed", False), ("controller_stopped", False),
                           ("needs_person", False), ("gripper_fault", "")):
            self.assertEqual(value, result.get(key), key)
        self.assertEqual(6.0, result["hand_eye_warn_mm"])
        executing = wait_for_event(self.cell, run["id"], "pick.executing").data
        self.assertEqual(f"/v1/runs/{run['id']}/overlays/1", executing["overlay"])
        self.assertEqual(executing["overlay"], result["overlay"])

    def test_until_empty_places_two_parts_of_an_emptying_scene_then_ends_nothing_left(self) -> None:
        self.build_dummy()
        EmptyingScene.install(self.cell.session.service, after=2)
        run = self.task(object="", scope="until_empty")
        body = self.finished(run["id"])
        self.assertEqual(("finished", "nothing_left", 2, 4, 2),
                         (body["state"], body["stop_code"], body["parts_placed"], body["attempted"], body["succeeded"]),
                         body)
        events = events_of(self.cell, run["id"])
        empty = [event.data for event in events if event.type == "task.nothing_found"]
        self.assertEqual([1, 2], [data["empty_in_a_row"] for data in empty])
        self.assertEqual([None, None, None, None],
                         [event.data["of"] for event in events if event.type == "task.part_started"])
        self.assertEqual(2, sum(1 for event in events if event.type == "task.placed"))
        self.assertEqual("task.returned", _without_pick_stages([e.type for e in events])[-2])

    def test_the_history_keeps_the_kind_the_stop_code_and_the_parts_placed(self) -> None:
        self.build_dummy()
        run = self.task(object="")
        self.finished(run["id"])
        lines = self.client.get("/v1/history/runs.csv").text.strip().splitlines()
        header = lines[0].split(",")
        for column in ("kind", "stop_code", "parts_placed", "succeeded"):
            self.assertIn(column, header)
        row = dict(zip(header, lines[1].split(",")))
        self.assertEqual(("task", "finished", "1"), (row["kind"], row["stop_code"], row["parts_placed"]))

    def test_the_console_says_its_version(self) -> None:
        self.assertEqual("0.2.0", self.client.get("/v1/health").json()["version"])


class AScriptedTaskTests(ConsoleCase):
    """The owner's cell in miniature: the task tests' arm double, the real toggle hand on tool DO0, the scripted service."""

    def test_the_drop_stands_the_hang_above_the_taught_pose(self) -> None:
        """Build plan item 7: the taught pose says where the part's bottom is let go, so the tool goes to it raised by
        the part's hang (the scripted arm's fk puts ``drop_left`` at 300, -400, 150 mm)."""
        from tests._task_fakes import PLACE_TCP

        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube")
        self.assertEqual("finished", self.finished(run["id"])["stop_code"])
        drop = wait_for_event(self.cell, run["id"], "task.drop_planned").data
        self.assertEqual("pose", drop["kind"])
        self.assertGreater(drop["hang_mm"], 0.0)
        taught = [float(v) for v in PLACE_TCP.position_mm]
        expected = [taught[0], taught[1], taught[2] + float(drop["hang_mm"])]
        for axis, (said, wanted) in enumerate(zip(drop["pose_mm"], expected)):
            self.assertAlmostEqual(wanted, said, delta=0.1, msg=f"axis {axis}: {drop['pose_mm']} vs {expected}")
        released_at = tuple(round(v, 1) for v in expected)
        self.assertIn(("move", released_at, True), cell.motions(), "the line in did not end at the planned drop")

    def test_two_changes_of_do0_per_part_and_the_arm_back_home(self) -> None:
        cell = self.scripted([Scripted("part"), Scripted("part"), Scripted("empty"), Scripted("empty")])
        run = self.task(object="green cube", scope="until_empty")
        body = self.finished(run["id"])
        self.assertEqual(("nothing_left", 2), (body["stop_code"], body["parts_placed"]), body)
        self.assertEqual(4, cell.do0_changes(), "a part closes and opens the toggle once each, nothing more")
        self.assertEqual(("home",), cell.motions()[-1])
        placed = wait_for_event(self.cell, run["id"], "task.placed").data
        self.assertTrue(placed["no_sensor"], "a toggle's release is never measured")

    def test_stop_after_this_part_places_it_returns_and_ends_stopped_after_part(self) -> None:
        def press_stop() -> None:
            answered = self.client.post("/v1/task/stop")
            self.assertEqual(200, answered.status_code, answered.text)
            self.assertTrue(answered.json()["stop_requested"])

        cell = self.scripted([Scripted("part", on_executing=press_stop), Scripted("part")])
        run = self.task(object="green cube", scope="until_empty")
        body = self.finished(run["id"])
        self.assertEqual(("cancelled", "stopped_after_part", "operator", 1),
                         (body["state"], body["stop_code"], body["stop_class"], body["parts_placed"]), body)
        self.assertTrue(body["stop_requested"])
        self.assertEqual(2, cell.do0_changes(), "the part in hand was not placed, or a second one was picked")
        self.assertEqual(("home",), cell.motions()[-1])
        asked = wait_for_event(self.cell, run["id"], "run_stop_requested")
        self.assertEqual("after_part", asked.data["scope"])
        types = _without_pick_stages(event_types(self.cell, run["id"]))
        self.assertLess(types.index("run_stop_requested"), types.index("task.placed"))
        self.assertIsNone(self.cell.recovery, "an operator's stop wrote a recovery record")

    def test_a_stop_asks_a_task_and_nothing_else(self) -> None:
        self.refused(self.client.post("/v1/task/stop"), 404, "no_such_run")
        self.refused(self.client.post("/v1/task/stop", params={"run_id": "run-nobody"}), 404, "no_such_run")
        release = threading.Event()
        self.addCleanup(release.set)
        self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        pick = self.client.post("/v1/pick", json={"prompt": "", "picks": 1})
        self.assertEqual(202, pick.status_code, pick.text)
        wait_for_event(self.cell, pick.json()["id"], "pick.executing")
        self.refused(self.client.post("/v1/task/stop"), 409, "not_a_task")
        release.set()
        self.finished(pick.json()["id"])

    def test_a_return_to_a_taught_pose_goes_there(self) -> None:
        from tests._task_fakes import PARK_JOINTS, _key

        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube", return_to="park")
        body = self.finished(run["id"])
        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual(("joints", _key(PARK_JOINTS)), cell.motions()[-1])
        plan = body["plan"]
        self.assertEqual(("park", "Parkposition"), (plan["return_to"], plan["return_label"]))
        self.assertEqual(list(POSES_DEG["park"][1]), plan["return_joints_deg"])

    def test_until_empty_into_a_pose_ends_nothing_left_where_the_drop_area_holds_the_only_parts(self) -> None:
        """Scenario 3: the task keeps a circle of 150 mm about its drop out of its picks (Q4), so the parts it placed
        there are no parts left to pick: two looks that see only them end ``nothing_left`` and the return, never
        ``failed_in_a_row``. The scripted pick skips a part only where a region of the task holds it."""
        from tests._task_fakes import PLACE_TCP

        at_the_drop = (float(PLACE_TCP.position_mm[0]) + 20.0, float(PLACE_TCP.position_mm[1]) + 20.0)
        cell = self.scripted([Scripted("part"), Scripted("kept_out", xy=at_the_drop),
                              Scripted("kept_out", xy=at_the_drop)])
        run = self.task(object="green cube", scope="until_empty")
        body = self.finished(run["id"])
        self.assertEqual(("finished", "nothing_left", "done", 1),
                         (body["state"], body["stop_code"], body["stop_class"], body["parts_placed"]), body)
        events = events_of(self.cell, run["id"])
        empty = [(event.data["empty_in_a_row"], event.data["only_excluded"]) for event in events
                 if event.type == "task.nothing_found"]
        self.assertEqual([(1, True), (2, True)], empty, "the looks at the task's own parts were not empty looks")
        results = [event.data["only_excluded"] for event in events if event.type == "pick_result"]
        self.assertEqual([False, True, True], results)
        self.assertEqual("task.returned", _without_pick_stages([event.type for event in events])[-2])
        self.assertEqual(("home",), cell.motions()[-1])
        self.assertEqual(2, cell.do0_changes())
        self.assertIsNone(self.cell.recovery, "an empty drop area wrote a recovery record")

    def test_the_plan_resolves_the_options_and_keeps_the_command_for_the_record(self) -> None:
        self.scripted([Scripted("part")])
        command = {"text": "Leg den grünen Würfel ab", "source": "spoken", "language": "de", "parsed": True,
                   "edited": ["scope"]}
        run = self.task(object="green cube", object_said="den grünen Würfel", command=command,
                        options={"push_mm": 40, "closing_axis": "+y", "multi_view": False})
        plan = run["plan"]
        self.assertEqual((40.0, "y", False, None), (plan["options"]["push_mm"], plan["options"]["closing_axis"],
                                                    plan["options"]["multi_view"], plan["options"]["rim_air_mm"]))
        self.assertEqual(command, plan["command"])
        self.assertEqual("den grünen Würfel", plan["object_said"])
        self.assertFalse(plan["countdown"])
        self.finished(run["id"])


class ATaskIsRefusedBeforeAnythingMovesTests(ConsoleCase):
    """Build plan 1.3.2: every refusal answers its code, moves nothing and starts no run; the cell before the request."""

    def assertNothingStarted(self, cell: Any = None) -> None:  # noqa: N802 (unittest's own style)
        self.assertEqual([], self.client.get("/v1/runs").json(), "a refused task started a run")
        if cell is not None:
            self.assertEqual([], cell.motions(), "a refused task moved the arm")
            self.assertEqual(0, cell.do0_changes(), "a refused task switched the jaws")

    def test_a_cell_that_is_not_connected(self) -> None:
        self.refused(self.client.post("/v1/task", json={"place": {"kind": "pose", "pose": None}}), 409,
                     "not_connected")
        self.build_dummy(connect=False)
        self.refused(self.client.post("/v1/task", json={"place": {"kind": "pose", "pose": None}}), 409,
                     "not_connected")
        self.assertNothingStarted()

    def test_a_run_holds_the_lock(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        first = self.task(object="green cube")
        wait_for_event(self.cell, first["id"], "pick.executing")
        refused = self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                                  "place": {"kind": "pose", "pose": None}}),
                               409, "run_active")
        self.assertEqual(first["id"], refused["detail"]["run_id"])
        release.set()
        self.finished(first["id"])

    def test_a_halted_arm_is_halted_before_its_controller_is_read(self) -> None:
        from tests._task_fakes import PROTECTIVE

        cell = self.scripted()
        cell.arm.halt("the operator pressed halt now")
        cell.arm.status = PROTECTIVE
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "pose", "pose": "nowhere"}}), 409, "halted")
        self.assertNothingStarted(cell)

    def test_a_stopped_controller(self) -> None:
        from tests._task_fakes import PROTECTIVE

        cell = self.scripted()
        cell.arm.status = PROTECTIVE
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "pose", "pose": None}}),
                     409, "controller_stopped")
        self.assertNothingStarted(cell)

    def test_the_recovery_record_gates_until_cleared_then_asks_for_a_restart(self) -> None:
        cell = self.scripted([Scripted("failed"), Scripted("failed"), Scripted("failed")])
        stopped = self.task(object="green cube")
        self.assertEqual("failed_in_a_row", self.finished(stopped["id"])["stop_code"])
        body = {"object": "green cube", "place": {"kind": "pose", "pose": None}}
        self.refused(self.client.post("/v1/task", json=body), 409, "cell_not_cleared")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.refused(self.client.post("/v1/task", json=body), 409, "restart_required")
        self.assertEqual(1, len(self.client.get("/v1/runs").json()))
        self.assertEqual(0, cell.do0_changes())

    def test_a_person_needed_after_a_recovery(self) -> None:
        cell = self.scripted()
        cell.service._needs_person = "a push stopped where the arm stands"  # noqa: SLF001
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "pose", "pose": None}}), 409, "needs_person")
        self.assertEqual("a push stopped where the arm stands", cell.service.stopped_where_the_arm_stands,
                         "the refusal cleared the latch")
        self.assertNothingStarted(cell)

    def test_a_jaws_question_waiting(self) -> None:
        cell = self.scripted()
        with patch("api.jaws.pending", return_value=object()):
            self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                            "place": {"kind": "pose", "pose": None}}),
                         409, "jaws_question_pending")
        self.assertNothingStarted(cell)

    def test_a_part_still_held_by_the_count_or_by_the_planner(self) -> None:
        cell = self.scripted()
        cell.jaws.set_closed(True)
        del cell.log[:]
        body = {"object": "green cube", "place": {"kind": "pose", "pose": None}}
        self.refused(self.client.post("/v1/task", json=body), 409, "part_still_held")
        cell.jaws.set_closed(False)
        cell.arm.attach_payload(40.0)
        del cell.log[:]
        self.refused(self.client.post("/v1/task", json=body), 409, "part_still_held")
        self.assertNothingStarted(cell)

    def test_jaws_nobody_can_vouch_for(self) -> None:
        cell = self.scripted()
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined] # at the pendant
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "pose", "pose": None}}),
                     409, "jaws_not_confirmed")
        self.assertNothingStarted(cell)

    def test_an_arm_whose_place_and_return_no_planner_judges(self) -> None:
        cell = self.scripted()
        cell.arm.plans_paths = False  # type: ignore[misc]
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "pose", "pose": None}}),
                     409, "route_refused")
        self.assertNothingStarted(cell)

    def test_a_carried_part_nobody_models(self) -> None:
        cell = self.scripted()
        cell.arm.declined = "safety.planning_world.payload.length_mm is not declared"
        refused = self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                                  "place": {"kind": "pose", "pose": None}}),
                               409, "carried_part_not_modelled")
        self.assertIn("length_mm", refused["message"])
        # A camera place alike.
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "camera", "phrase": "blue bin"}}),
                     409, "carried_part_not_modelled")
        self.assertNothingStarted(cell)

    def test_a_camera_place_on_the_rehearsal_cell(self) -> None:
        self.build_dummy()
        self.refused(self.client.post("/v1/task", json={"object": "", "place": {"kind": "camera", "phrase": "blue bin"}}),
                     409, "camera_target_unavailable")
        self.assertNothingStarted()

    def test_a_wrist_camera_place_on_an_arm_with_no_world_to_hold_the_bin_s_frames(self) -> None:
        """The library's pre-check, at the route: a wrist place holds the frames of the bin's look while the part goes in."""
        cell = self.scripted([Scripted("part")], wrist=True)
        cell.arm.live_planner_world = None
        locator = SightedLocator({"blue bin": lambda _tcp: ()}, arm=cell.arm, wrist=True)
        with patch("src.robot.execution.place_target.locators_for_service", return_value=[locator]):
            refused = self.refused(self.client.post("/v1/task", json={
                "object": "green cube", "place": {"kind": "camera", "phrase": "blue bin"}}), 409,
                "camera_target_unavailable")
        self.assertIn("live planner world", refused["message"])
        self.assertNothingStarted(cell)

    def test_both_faces_on_a_cell_whose_motion_turns_every_grasp_off_the_judged_faces(self) -> None:
        """The library's pre-check, at the route: ``both_faces`` and ``align_closing_to_base_x`` together are refused."""
        cell = self.scripted([Scripted("part")])
        cell.service.runtime.orchestrator.policy.align_closing_to_base_x = True
        refused = self.refused(self.client.post("/v1/task", json={
            "object": "green cube", "place": {"kind": "pose", "pose": None}, "options": {"both_faces": True}}), 422,
            "bad_request")
        self.assertEqual("options.both_faces", refused["detail"]["field"])
        self.assertNothingStarted(cell)

    def test_an_empty_object_on_a_cell_that_grounds_a_phrase(self) -> None:
        cell = self.scripted([Scripted("part")])
        self.refused(self.client.post("/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None}}),
                     422, "object_required")
        self.assertNothingStarted(cell)
        run = self.task(object="", options={"pick_anything": True})
        self.assertEqual("finished", self.finished(run["id"])["stop_code"])

    def test_an_empty_object_on_the_rehearsal_cell_needs_no_word(self) -> None:
        self.build_dummy()
        run = self.task(object="")
        self.finished(run["id"])

    def test_phrases_the_detector_would_ground_wrong(self) -> None:
        cell = self.scripted()
        self.refused(self.client.post("/v1/task", json={"object": "den grünen Würfel",
                                                        "place": {"kind": "pose", "pose": None}}),
                     422, "prompt_not_routable")
        with patch("src.robot.execution.place_target.locators_for_service", return_value=[object()]):
            self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                            "place": {"kind": "camera", "phrase": "die blaue Kiste"}}),
                         422, "target_not_routable")
        self.assertNothingStarted(cell)

    def test_a_push_distance_the_cell_refuses(self) -> None:
        cell = self.scripted()
        refused = self.refused(self.client.post("/v1/task", json={"object": "green cube", "options": {"push_mm": 80},
                                                                  "place": {"kind": "pose", "pose": None}}),
                               422, "push_distance_refused")
        self.assertEqual(80, refused["detail"]["push_mm"])
        self.assertNothingStarted(cell)

    def test_a_pose_nobody_taught(self) -> None:
        cell = self.scripted()
        refused = self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                                  "place": {"kind": "pose", "pose": "nowhere"}}),
                               422, "unknown_pose")
        self.assertEqual("place", refused["detail"]["which"])
        refused = self.refused(self.client.post("/v1/task", json={"object": "green cube", "return_to": "nowhere",
                                                                  "place": {"kind": "pose", "pose": None}}),
                               422, "unknown_pose")
        self.assertEqual("return", refused["detail"]["which"])
        self.assertNothingStarted(cell)

    def test_an_axis_that_names_no_direction(self) -> None:
        cell = self.scripted()
        self.refused(self.client.post("/v1/task", json={"object": "green cube", "options": {"closing_axis": "diagonal"},
                                                        "place": {"kind": "pose", "pose": None}}),
                     422, "closing_axis_refused")
        self.assertNothingStarted(cell)

    def test_a_request_that_is_no_task(self) -> None:
        self.scripted()
        self.refused(self.client.post("/v1/task", json={"object": "green cube"}), 422, "bad_request")
        self.refused(self.client.post("/v1/task", json={"place": {"kind": "camera", "phrase": "blue bin"},
                                                        "options": {"rim_air_mm": 60}}), 422, "bad_request")
        self.assertNothingStarted()


class NoDefaultPlaceTests(ConsoleCase):
    default_place = False

    def test_a_pose_place_that_names_no_pose_and_a_cell_with_no_default(self) -> None:
        cell = self.scripted()
        self.refused(self.client.post("/v1/task", json={"object": "green cube",
                                                        "place": {"kind": "pose", "pose": None}}),
                     422, "no_place_declared")
        self.assertEqual([], cell.motions())

    def test_an_unknown_pose_is_said_before_the_missing_default_place(self) -> None:
        """1.3.2 lists ``unknown_pose`` before ``no_place_declared``: a return nobody taught is said first."""
        self.scripted()
        refused = self.refused(self.client.post("/v1/task", json={
            "object": "green cube", "return_to": "nowhere", "place": {"kind": "pose", "pose": None}}), 422,
            "unknown_pose")
        self.assertEqual("return", refused["detail"]["which"])


class WhatTheLibraryMeetsInsideTheRunTests(ConsoleCase):
    """``run_task`` refuses before it moves what the route refuses already (``TaskRefused``), as a backstop for a cell
    that changed between the request and the run; and a controller stop it meets on a latched arm is the halt's."""

    def test_a_refusal_inside_the_run_moves_nothing_and_writes_no_record(self) -> None:
        from src.robot.execution.task import TaskRefused

        cell = self.scripted([Scripted("part")])

        def refuses(*_args: Any, **_keywords: Any) -> Any:
            raise TaskRefused("push_distance_refused", "a push of 45 mm is refused: this cell's ceiling is 40 mm")

        with patch("src.robot.execution.task.run_task", side_effect=refuses):
            run = self.task(object="green cube")
            body = self.finished(run["id"])
        self.assertEqual(("cancelled", "cancelled"), (body["state"], body["stop_code"]), body)
        self.assertIn("push_distance_refused, 422", body["error"])
        self.assertIn("ceiling is 40 mm", body["error"])
        # Typed beside the sentence (RunOut.refusal), so the browser says it in its own words.
        self.assertEqual({"code": "push_distance_refused", "status": 422}, body["refusal"])
        finished = [event for event in events_of(self.cell, run["id"]) if event.type == "run_finished"]
        self.assertEqual(body["refusal"], finished[-1].data["refusal"])
        self.assertIsNone(self.cell.recovery, "a refusal that moved nothing wrote a recovery record")
        self.assertEqual([], cell.motions())
        self.assertNotIn("run_error", event_types(self.cell, run["id"]))

    def test_a_controller_stop_the_library_met_on_a_latched_arm_is_the_halt(self) -> None:
        from src.robot.execution.task import TaskReport, TaskStop

        cell = self.scripted([Scripted("part")])

        def stopped(*_args: Any, **_keywords: Any) -> Any:
            cell.arm.halt("a halt latched by another program")
            return TaskReport(stop=TaskStop.CONTROLLER_STOPPED, sentence="the controller cannot move: the return "
                                                                         "was refused")

        with patch("src.robot.execution.task.run_task", side_effect=stopped):
            run = self.task(object="green cube")
            body = self.finished(run["id"])
        self.assertEqual(("failed", "halted", "problem"), (body["state"], body["stop_code"], body["stop_class"]), body)
        self.assertIn("the arm is halted (a halt latched by another program)", body["error"])
        self.assertEqual("halted", self.cell.recovery.stop_code)  # type: ignore[union-attr]


class ACameraPlaceTests(ConsoleCase):
    """A camera place on a scripted wrist cell: the survey, the kept bin and its overlay, the check before the drop."""

    def _bin(self, moved_mm: float = 0.0, size: "tuple[float, float] | None" = None) -> Any:
        from tests._task_fakes import BIN_CENTRE, BIN_SIZE, bin_object, bin_points

        def sees(_tcp: Any) -> Any:
            return (bin_object(bin_points((BIN_CENTRE[0] + moved_mm, BIN_CENTRE[1]), size or BIN_SIZE)),)

        return sees

    def _wrist(self, picks: list[Scripted], *, moved_after_survey_mm: float = 0.0,
               size_after_survey: "tuple[float, float] | None" = None) -> Any:
        from tests._console_task_fakes import LOOK_1, LOOK_2

        cell = self.scripted(picks, wrist=True, looks=(LOOK_1, LOOK_2))
        moved: dict[str, Any] = {"mm": 0.0, "size": None}
        locator = SightedLocator({"blue bin": lambda tcp: self._bin(moved["mm"], moved["size"])(tcp)}, arm=cell.arm,
                                 wrist=True)

        def survey_done(*_args: Any) -> None:
            moved["mm"] = moved_after_survey_mm
            moved["size"] = size_after_survey

        patcher = patch("src.robot.execution.place_target.locators_for_service", return_value=[locator])
        patcher.start()
        self.addCleanup(patcher.stop)
        task_patcher = patch("src.robot.execution.task.locators_for_service", return_value=[locator])
        task_patcher.start()
        self.addCleanup(task_patcher.stop)
        return cell, survey_done

    def test_a_bin_found_before_the_first_pick_takes_the_part(self) -> None:
        cell, _moved = self._wrist([Scripted("part", looks=("look_1", "look_2"))])
        run = self.task(object="red cube", place={"kind": "camera", "phrase": "blue bin", "said": "in die blaue Kiste"})
        body = self.finished(run["id"])
        self.assertEqual(("finished", 1), (body["stop_code"], body["parts_placed"]), body)
        types = _without_pick_stages(event_types(self.cell, run["id"]))
        self.assertLess(types.index("task.survey_started"), types.index("task.target_found"))
        self.assertLess(types.index("task.target_found"), types.index("task.part_started"))
        self.assertIn("task.target_checked", types)
        found = wait_for_event(self.cell, run["id"], "task.target_found").data
        self.assertEqual("blue bin", found["target"]["label"])
        url = found["target"]["overlay"]
        self.assertEqual(f"/v1/runs/{run['id']}/target/overlay", url)
        image = self.client.get(url)
        self.assertEqual(200, image.status_code)
        self.assertEqual("image/png", image.headers["content-type"])
        self.assertTrue(image.content.startswith(b"\x89PNG"))
        self.assertEqual(2, cell.do0_changes())
        self.assertEqual("camera", body["plan"]["place"]["kind"])
        self.assertEqual(20.0, body["plan"]["options"]["rim_air_mm"])

    def test_a_bin_that_moved_too_far_is_lost_the_part_goes_back_and_the_task_asks(self) -> None:
        """A bin that moved too far is looked for again (the owner, 2026-10-09): here one of another size stands there,
        so none of the bin's size is found anywhere, and the part goes back."""
        cell, survey_done = self._wrist([Scripted("part", looks=("look_1",), then=lambda: survey_done())],
                                        moved_after_survey_mm=180.0, size_after_survey=(200.0, 120.0))
        run = self.task(object="red cube", place={"kind": "camera", "phrase": "blue bin"})
        body = self.finished(run["id"])
        self.assertEqual(("finished", "target_lost", "ask", 0, False),
                         (body["state"], body["stop_code"], body["stop_class"], body["parts_placed"], body["holding"]),
                         body)
        lost = wait_for_event(self.cell, run["id"], "task.target_lost").data
        self.assertEqual("moved_too_far", lost["why"])
        checked = wait_for_event(self.cell, run["id"], "task.target_checked").data
        self.assertFalse(checked["followed"])
        self.assertAlmostEqual(180.0, checked["moved_mm"], delta=1.0)
        types = _without_pick_stages(event_types(self.cell, run["id"]))
        self.assertLess(types.index("task.put_back"), types.index("task.returned"))
        self.assertEqual(2, cell.do0_changes(), "the part was not put back with the one change that opens the jaws")
        self.assertIsNone(self.cell.recovery, "a question wrote a recovery record")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
