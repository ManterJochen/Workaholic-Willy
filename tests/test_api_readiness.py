"""The ready bar (``GET /v1/cell/readiness``), the cell's facts (``GET /v1/cell/facts``) and the planner run.

Build plan 1.5, 1.10, 1.3.6 and item 19: six lights (the robot, the cameras, the planner, the gripper, the carried
part and the commands) and the blockers; ``ready`` only where every light that blocks is ok and nothing blocks. It
always answers and moves nothing, and it reads the controller from the receive stream alone (``quick_robot_status``),
never through the dashboard. What this file pins:

* nothing built: not ready, every light says why; console_dummy connected: ready, the rehearsal's camera and planner
  labelled as such, the dummy carrying nothing real;
* a halt reads ``halted``, a stopped controller ``controller_stopped``, a controller that cannot be read
  ``controller_unreadable``; the dashboard is never asked;
* a toggle hand's light follows the program's count; a carried part nobody models blocks;
* each blocker of 1.5, and the commands light never blocks;
* the facts say the route, the brake, the carried part and the rehearsal, the looks, the hand's natural axis, what a
  push may do (its ceiling the service's own, the hard cap where the cell declares no fixture) and the detector with the
  precision it runs at;
* ``POST /v1/cell/planner`` starts cuRobo as a run that moves nothing, and Connect starts it by itself on a cuRobo arm
  whose planner is off, and on no other.

Honesty bucket (2): the real console, on console_dummy and on the scripted cell.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, ScriptedCell, wait_for_event


def lights(answer: dict[str, Any]) -> dict[str, tuple[str, str, bool]]:
    return {light["id"]: (light["state"], light["code"], light["blocks"]) for light in answer["lights"]}


class TheReadyBarTests(ConsoleCase):
    def test_nothing_built_is_not_ready_and_says_why(self) -> None:
        answer = self.client.get("/v1/cell/readiness").json()
        self.assertFalse(answer["ready"])
        seen = lights(answer)
        self.assertEqual({"robot", "cameras", "planner", "gripper", "carried_part", "commands"}, set(seen))
        self.assertEqual(("blocked", "not_built"), seen["robot"][:2])
        self.assertFalse(seen["commands"][2], "the commands light blocks Start")

    def test_console_dummy_connected_is_ready(self) -> None:
        self.build_dummy()
        answer = self.client.get("/v1/cell/readiness").json()
        seen = lights(answer)
        self.assertEqual(("ok", "connected"), seen["robot"][:2])
        self.assertEqual(("ok", "rehearsal"), seen["cameras"][:2])
        self.assertEqual(("ok", "unplanned"), seen["planner"][:2])
        self.assertEqual(("ok", "connected"), seen["gripper"][:2])
        self.assertEqual(("ok", "not_applicable"), seen["carried_part"][:2])
        self.assertEqual([], answer["blockers"])
        self.assertTrue(answer["ready"], answer)

    def test_built_but_not_connected(self) -> None:
        self.build_dummy(connect=False)
        seen = lights(self.client.get("/v1/cell/readiness").json())
        self.assertEqual(("blocked", "not_connected"), seen["robot"][:2])

    def test_the_controller_is_read_from_the_receive_stream_alone(self) -> None:
        from src.robot.core.errors import RobotConnectionError
        from tests._task_fakes import PROTECTIVE

        cell = self.scripted()
        seen = lights(self.client.get("/v1/cell/readiness").json())
        self.assertEqual(("ok", "connected"), seen["robot"][:2])
        cell.arm.status = PROTECTIVE
        seen = lights(self.client.get("/v1/cell/readiness").json())
        self.assertEqual(("blocked", "controller_stopped"), seen["robot"][:2])
        cell.arm.status_raises = RobotConnectionError("the receive stream dropped")
        seen = lights(self.client.get("/v1/cell/readiness").json())
        self.assertEqual(("blocked", "controller_unreadable"), seen["robot"][:2])
        cell.arm.status_raises = None
        cell.arm.halt("the operator pressed halt now")
        seen = lights(self.client.get("/v1/cell/readiness").json())
        self.assertEqual(("blocked", "halted"), seen["robot"][:2])
        self.assertNotIn("get_robot_status", cell.arm.calls, "the ready bar paid a dashboard round trip")
        self.assertIn("quick_robot_status", cell.arm.calls)

    def test_a_toggle_s_light_follows_the_program_s_count(self) -> None:
        cell = self.scripted()
        self.assertEqual(("ok", "open_confirmed"), lights(self.client.get("/v1/cell/readiness").json())["gripper"][:2])
        cell.jaws.set_closed(True)
        self.assertEqual(("blocked", "jaws_closed"), lights(self.client.get("/v1/cell/readiness").json())["gripper"][:2])
        cell.jaws.set_closed(False)
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined]
        self.assertEqual(("blocked", "jaws_unknown"), lights(self.client.get("/v1/cell/readiness").json())["gripper"][:2])
        with patch("api.jaws.pending", return_value=object()):
            answer = self.client.get("/v1/cell/readiness").json()
        self.assertEqual(("blocked", "question_pending"), lights(answer)["gripper"][:2])
        self.assertIn("jaws_question_pending", [blocker["code"] for blocker in answer["blockers"]])

    def test_a_carried_part_nobody_models_blocks_and_says_why(self) -> None:
        cell = self.scripted()
        seen = lights(self.client.get("/v1/cell/readiness").json())
        self.assertEqual(("ok", "modelled"), seen["carried_part"][:2])
        cell.arm.declined = "safety.planning_world.payload.length_mm is not declared"
        answer = self.client.get("/v1/cell/readiness").json()
        light = next(light for light in answer["lights"] if light["id"] == "carried_part")
        self.assertEqual(("blocked", "not_modelled"), (light["state"], light["code"]))
        self.assertIn("length_mm", light["message"])
        self.assertFalse(answer["ready"])

    def test_the_planner_light(self) -> None:
        cell = self.scripted(arm_keywords={"planner": "off"})
        self.assertEqual(("blocked", "off"), lights(self.client.get("/v1/cell/readiness").json())["planner"][:2])
        cell.arm._planner = "starting"  # noqa: SLF001
        self.assertEqual(("wait", "starting"), lights(self.client.get("/v1/cell/readiness").json())["planner"][:2])
        cell.arm._planner = "ready"  # noqa: SLF001
        self.assertEqual(("ok", "ready"), lights(self.client.get("/v1/cell/readiness").json())["planner"][:2])
        cell.arm.plans_paths = False  # type: ignore[misc]
        self.assertEqual(("blocked", "route_refused"),
                         lights(self.client.get("/v1/cell/readiness").json())["planner"][:2])

    def test_each_blocker(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        cell = self.scripted([Scripted("part", on_executing=lambda: release.wait(timeout=20))])
        run = self.task(object="green cube")
        wait_for_event(self.cell, run["id"], "pick.executing")
        codes = [blocker["code"] for blocker in self.client.get("/v1/cell/readiness").json()["blockers"]]
        self.assertIn("run_active", codes)
        release.set()
        self.finished(run["id"])
        cell.service._needs_person = "a push stopped where the arm stands"  # noqa: SLF001
        cell.jaws.set_closed(True)
        from api.codes import RunKind

        stopped = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "cell_fault")
        self.finished(stopped.id)
        answer = self.client.get("/v1/cell/readiness").json()
        codes = [blocker["code"] for blocker in answer["blockers"]]
        for code in ("cell_not_cleared", "needs_person", "part_still_held"):
            self.assertIn(code, codes)
        self.assertFalse(answer["ready"])
        self.cell.mark_cell_clear()
        codes = [blocker["code"] for blocker in self.client.get("/v1/cell/readiness").json()["blockers"]]
        self.assertIn("restart_required", codes)
        self.assertNotIn("cell_not_cleared", codes)


class TheCellFactsTests(ConsoleCase):
    def test_console_dummy_s_facts(self) -> None:
        self.build_dummy()
        facts = self.client.get("/v1/cell/facts").json()
        self.assertTrue(facts["rehearsal"])
        self.assertEqual("unplanned", facts["route"]["route"])
        self.assertEqual({"latches": True, "brakes_in_motion": False}, facts["brake"])
        self.assertFalse(facts["payload"]["modelled"])
        self.assertFalse(facts["wrist_camera"])
        self.assertEqual(6.0, facts["hand_eye_warn_mm"])

    def test_a_scripted_cell_s_facts(self) -> None:
        cell = self.scripted(wrist=True, looks=())
        facts = self.client.get("/v1/cell/facts").json()
        self.assertEqual("planned", facts["route"]["route"])
        self.assertTrue(facts["payload"]["modelled"])
        self.assertIsNone(facts["payload"]["declined_reason"])
        self.assertFalse(facts["rehearsal"])
        self.assertTrue(facts["wrist_camera"])
        cell.arm.declined = "no hand modelled"
        facts = self.client.get("/v1/cell/facts").json()
        self.assertEqual((False, "no hand modelled"), (facts["payload"]["modelled"], facts["payload"]["declined_reason"]))

    def test_facts_of_nothing_built(self) -> None:
        facts = self.client.get("/v1/cell/facts")
        self.assertEqual(200, facts.status_code, facts.text)
        self.assertEqual("refused", facts.json()["route"]["route"])

    def test_the_facts_say_the_looks_the_axis_the_push_and_the_detector(self) -> None:
        """Build plan 1.10, on the scripted wrist cell over the shipped tree (GroundingDINO, its dtype left unset)."""
        from src.geometry.closing_axis import closing_axis_of
        from src.robot.execution.looks import look_label
        from src.robot.grasping.recovery.push_planner import PUSH_DISTANCE_CAP_MM
        from tests._console_task_fakes import LOOK_1, LOOK_2

        self.scripted(wrist=True, looks=(LOOK_1, LOOK_2), natural_axis=closing_axis_of("-y"))
        facts = self.client.get("/v1/cell/facts").json()
        self.assertEqual([look_label(LOOK_1), look_label(LOOK_2)], facts["looks"])
        self.assertEqual("-y", facts["natural_closing_axis"])
        push = facts["push"]
        # The scripted service pushes nothing (no push cell), and its distance answers as the service's does.
        self.assertEqual((False, 30.0, PUSH_DISTANCE_CAP_MM), (push["can_push"], push["default_mm"], push["ceiling_mm"]))
        self.assertTrue(push["why_not"])
        detector = facts["detector"]
        self.assertEqual(("grounded_sam", False), (detector["backend"], detector["router_enabled"]))
        self.assertEqual("fp32 weights, fp16 autocast", detector["precision"])

    def test_the_detector_s_precision_follows_its_configured_dtype(self) -> None:
        from types import SimpleNamespace

        from api.readiness import detector_precision

        def models(dtype: str | None) -> Any:
            optim = SimpleNamespace(torch_dtype=dtype)
            return SimpleNamespace(objectdetector=SimpleNamespace(optim=optim), rtdetr=SimpleNamespace(optim=optim))

        self.assertEqual("fp32 weights, fp16 autocast", detector_precision(models(None), "grounded_sam", False))
        self.assertEqual("float16", detector_precision(models("float16"), "grounded_sam", False))
        self.assertEqual("auto", detector_precision(models("auto"), "rtdetr", False))
        self.assertEqual("auto (the checkpoint's own)", detector_precision(models(None), "vlm", False))
        self.assertEqual("VLM auto (the checkpoint's own); GroundingDINO fp32 weights, fp16 autocast",
                         detector_precision(models(None), "vlm", True))
        self.assertIsNone(detector_precision(None, "grounded_sam", False))


class ThePlannerRunTests(ConsoleCase):
    def test_the_planner_run_starts_cuRobo_and_moves_nothing(self) -> None:
        # Connected through the session itself, not the route: nothing started the planner yet.
        cell = self.scripted(arm_keywords={"planner": "off"})
        self.assertEqual([], self.client.get("/v1/runs").json())
        started = self.client.post("/v1/cell/planner")
        self.assertEqual(202, started.status_code, started.text)
        body = self.finished(started.json()["id"])
        self.assertEqual(("planner", "planner_ready", "finished"), (body["kind"], body["stop_code"], body["state"]))
        self.assertEqual("ready", cell.arm.planner_state)
        self.assertEqual([], cell.motions())
        ready = wait_for_event(self.cell, body["id"], "planner.ready").data
        self.assertIn("loaded", ready)
        self.assertEqual("ready", [event.data["state"] for event in self.cell.hub.since("cell", 0)[0]
                                   if event.type == "cell.planner"][-1])

    def test_a_planner_that_fails_to_start_ends_planner_failed(self) -> None:
        self.scripted(arm_keywords={"planner": "off", "start_fails": "the sidecar did not answer"}, connect=False)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": token}).status_code)
        # Connect started it by itself, and it failed.
        runs = self.client.get("/v1/runs").json()
        self.assertEqual(1, len(runs))
        body = self.finished(runs[0]["id"])
        self.assertEqual(("planner_failed", "planner", "failed"), (body["stop_code"], body["stop_class"], body["state"]))
        self.assertIn("did not answer", wait_for_event(self.cell, body["id"], "planner.failed").data["refusal"])
        self.assertIsNone(self.cell.recovery, "a planner start that moved nothing wrote a recovery record")
        self.assertEqual(("blocked", "failed"), lights(self.client.get("/v1/cell/readiness").json())["planner"][:2])

    def test_refusals(self) -> None:
        self.refused(self.client.post("/v1/cell/planner"), 409, "not_built")
        self.build_dummy()
        self.refused(self.client.post("/v1/cell/planner"), 409, "planner_not_used")

    def test_connect_starts_the_planner_on_a_cuRobo_arm_whose_planner_is_off(self) -> None:
        """Scenario 16: a planner run is active when Connect answers; on the dummy there is none."""
        blocks = threading.Event()
        self.addCleanup(blocks.set)
        cell = ScriptedCell.build(arm_keywords={"planner": "off", "start_blocks": blocks})
        cell.install(self.cell, connect=False)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        up = self.client.post("/v1/cell/connect", json={"token": token})
        self.assertEqual(200, up.status_code, up.text)
        active = up.json()["active_run_id"]
        self.assertIsNotNone(active, "Connect answered before it started the planner")
        self.assertEqual("planner", self.client.get(f"/v1/runs/{active}").json()["kind"])
        self.refused(self.client.post("/v1/task", json={"object": "green cube", "place": {"kind": "pose", "pose": None}}),
                     409, "run_active")
        blocks.set()
        self.assertEqual("planner_ready", self.finished(active)["stop_code"])

    def test_connect_on_the_dummy_starts_nothing(self) -> None:
        self.build_dummy()
        self.assertIsNone(self.client.get("/v1/cell").json()["active_run_id"])
        self.assertEqual([], self.client.get("/v1/runs").json())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
