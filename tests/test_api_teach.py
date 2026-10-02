"""Teaching one pose from the browser (build plan 1.8, OD 15, Q10, Q14).

A person guides the freed arm by hand to where a pose should be, Save captures it once the arm stands still, the arm
holds, the pose is screened by the exact guard and the planner at once, and only a pose both clear, or one in the
planner's band, is written into the cell's own layer. One pose per session, on its own run (kind ``teach``), driven by
the library's ``teach_one``. What is held here, over HTTP, on a scripted hand-guided arm sampled in real time:

* a teach is refused before anything is freed, in the plan's order: the cell's state (not connected, a run, a halt, a
  stopped controller, a stop not cleared, a jaws question), a part in the hand or a toggle count nobody can vouch for,
  the library's own refusals (no hand guiding, a planner not ready, an arm that screens nothing, a name or label that is
  none, a name taken, a chain with no layer of the cell's own), and a payload that is not the one the person confirmed
  (a payload that is no number included);
* a jaws check and a teach never go on together: a teach that comes in while a check reads the latch and the controller
  is refused, and a check that marks itself after the route's own read is read again by the run, before anything is
  freed;
* Save captures once the arm stands still, the pose is screened and written, and the run ends ``taught``; a refused
  pose is never written (``teach_refused``, with a pose nearby), and neither is an unscreened one (``teach_not_saved``);
  the place pose and replace reach the pose door; a Save sent before the free arm was first sampled is not lost;
* every poll is the heartbeat: without one for 3 s the session holds the arm only once it stands still, never while it
  moves in a person's hands, and ends ``heartbeat_lost``; the 5-minute limit is announced 30 s before and holds the same
  way (``teach_time_limit``); polls keep a session free; the owner's numbers (3 s, 5 minutes, 30 s) are pinned;
* Cancel (also the page's beacon on close), a halt (the run's flag or the arm's own latch), a stop request and a
  Disconnect hold at once and save nothing; a Disconnect holds the arm before it takes it down;
* polls, Save and Cancel carry the session's token; a wrong one is refused, and Save is refused where the arm is not free;
* the next moving run counts down: a person's hands were on the arm.

Honesty bucket (2): the arm is a scripted double with the UR's teach surface; the hand guiding, the session, the screen
order, the pose door and the run are real.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

from src.robot.core import RobotError
from src.robot.core.freedrive import ControllerPayload
from tests.test_a_pose_is_taught_by_hand import LOOK_1_DEG, _sample
from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _pulses, _toggle
from tests.test_api_contract import _JawAsking
from tests.test_api_jaws import STOPPED, _InBackground, wait_for
from tests.test_api_poses import GUARD_REFUSED, UNSCREENED, LayerCell, TeachArm

#: What the person saw before the arm was freed: the controller's own payload, as ``TeachArm`` reports it.
SEEN = {"mass_kg": 1.9, "cog_mm": [0.0, 12.0, 61.0]}


def _moving() -> list[Any]:
    """A person who keeps moving the arm for about 8 s (50 Hz samples), then lets it stand: a hold that waits for
    stillness does not come within a test's few seconds, so only a hold at once does. A fresh stand each call: the arm
    takes its samples off it."""
    return [_sample(LOOK_1_DEG, speed=0.3)] * 400 + [_sample(LOOK_1_DEG)]


def _teach(**fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "drop_right", "label": "Ablage rechts", "role": "place", "payload_seen": SEEN}
    body.update(fields)
    return body


class _TeachCell(LayerCell):
    """The layer cell with a teach-capable arm connected."""

    def setUp(self) -> None:
        super().setUp()
        self.arm = self.connect_arm(TeachArm())

    def start(self, **fields: Any) -> tuple[str, str]:
        started = self.client.post("/v1/teach", json=_teach(**fields))
        self.assertEqual(202, started.status_code, started.text)
        body = started.json()
        self.assertEqual(("teach", "running"), (body["run"]["kind"], body["run"]["state"]))
        return body["run"]["id"], body["token"]

    def poll(self, run_id: str, token: str) -> dict[str, Any]:
        answered = self.client.get(f"/v1/teach/{run_id}", params={"token": token})
        self.assertEqual(200, answered.status_code, answered.text)
        return dict(answered.json())

    def until_state(self, run_id: str, token: str, *states: str, timeout: float = 15.0) -> dict[str, Any]:
        return wait_for(lambda: (lambda s: s if s["state"] in states else None)(self.poll(run_id, token)),
                        timeout=timeout, what=f"a teach state in {states}")

    def finished(self, run_id: str, timeout: float = 20.0) -> dict[str, Any]:
        def done() -> dict[str, Any] | None:
            run = self.client.get(f"/v1/runs/{run_id}").json()
            return run if run["state"] != "running" else None

        return wait_for(done, timeout=timeout, what="the teach run's end")

    def events(self, run_id: str) -> list[Any]:
        return self.cell.hub.since(run_id, 0)[0]

    def types(self, run_id: str) -> list[str]:
        return [e.type for e in self.events(run_id)]

    def held(self) -> list[str]:
        return [line for line in self.arm.log if line.startswith("hold at")]

    def ended_while_polled(self, run_id: str, token: str, *, timeout: float = 3.0) -> dict[str, Any]:
        """The run once it ended, polled as the browser polls it (so the heartbeat never lapses meanwhile)."""

        def ended() -> dict[str, Any] | None:
            self.client.get(f"/v1/teach/{run_id}", params={"token": token})
            run = self.client.get(f"/v1/runs/{run_id}").json()
            return run if run["state"] != "running" else None

        return wait_for(ended, timeout=timeout, what="the teach run's end, at once")


# ---------------------------------------------------------------------------------------------------------------------
# Refused before anything is freed
# ---------------------------------------------------------------------------------------------------------------------


class ATeachIsRefusedBeforeAnythingIsFreedTests(LayerCell):
    def refused(self, body: dict[str, Any]) -> tuple[int, str]:
        answered = self.client.post("/v1/teach", json=body)
        return answered.status_code, str(answered.json().get("code"))

    def test_the_cell_s_state_is_asked_first_in_the_plan_s_order(self) -> None:
        from api.cell import RecoveryRecord
        from api.codes import RunKind, StopCode

        self.assertEqual((409, "not_connected"), self.refused(_teach()))
        arm = self.connect_arm(TeachArm())
        arm.status = STOPPED
        arm.halt("the operator pressed halt now")
        self.cell.active_run_id = "run-busy"
        self.assertEqual((409, "run_active"), self.refused(_teach()))
        self.cell.active_run_id = None
        self.assertEqual((409, "halted"), self.refused(_teach()), "a halt read as a controller stop")
        arm.clear_halt()
        self.assertEqual((409, "controller_stopped"), self.refused(_teach()))
        arm.status = None
        self.assertEqual((409, "controller_stopped"), self.refused(_teach()), "an unreadable controller was trusted")
        from tests.test_api_jaws import RUNNING

        arm.status = RUNNING
        self.cell.record_problem_stop(type("_Run", (), {"id": "run-stopped", "kind": RunKind.TASK,
                                                        "stop_code": StopCode.HALTED, "finished_at": time.time(),
                                                        "holding": False})())
        self.assertIsInstance(self.cell.recovery, RecoveryRecord)
        self.assertEqual((409, "cell_not_cleared"), self.refused(_teach()))
        self.cell.mark_cell_clear()
        # Once cleared, a teach is allowed again (only a new task or pick waits for Restart or Home): the next refusal
        # is the request's own.
        self.assertEqual((422, "invalid_name"), self.refused(_teach(name="home")), "a cleared stop still refused")
        self.assertEqual([], [line for line in arm.log if line in ("free", "screen")])

    def test_a_waiting_jaws_question_refuses_a_teach(self) -> None:
        self.connect_arm(TeachArm())
        asking = threading.Thread(target=lambda: self._ask_quietly(), daemon=True)
        asking.start()
        wait_for(lambda: self.cell.jaws.pending(), what="the jaws question")
        try:
            self.assertEqual((409, "jaws_question_pending"), self.refused(_teach()))
        finally:
            self.cell.jaws.cancel("the test ends the question")
            asking.join(5.0)

    def _ask_quietly(self) -> None:
        try:
            self.cell.jaws(_JawAsking(at_connect=False))
        except EOFError:
            pass

    def test_a_part_in_the_hand_refuses_a_teach(self) -> None:
        class _ClosedToggle:
            toggles_without_sensor = True
            jaws_closed = True
            edge_unknown = False
            is_connected = True

            def jaws_open_for_a_pick(self) -> str:
                return ""

        self.connect_arm(TeachArm(), gripper=_ClosedToggle())
        self.assertEqual((409, "part_in_hand"), self.refused(_teach()))

    def test_a_toggle_whose_count_nobody_can_vouch_for_frees_nothing(self) -> None:
        """A close whose write raised may still have reached the controller: the count stays open and nobody can vouch
        for it. A part may be in the jaws, uncompensated by the payload the person confirmed, so nothing is freed until a
        person answered where the jaws stand (``jaws_not_confirmed``, the jaws question's own code)."""
        events: list[Any] = []
        hand = _toggle(events, Person("open"))
        arm = self.connect_arm(TeachArm(), gripper=hand)
        hand.connect()                                 # a person said open (the test's own string seam)
        hand._io.refuse_all = True                     # type: ignore[attr-defined] # the pick's close write raises
        with self.assertRaises(RobotError):
            hand.set_closed(True)
        hand._io.refuse_all = False                    # type: ignore[attr-defined]
        self.assertEqual((False, True), (hand.jaws_closed, hand.edge_unknown))

        answered = self.client.post("/v1/teach", json=_teach())

        self.assertEqual((409, "jaws_not_confirmed"), (answered.status_code, answered.json()["code"]), answered.text)
        self.assertIn("jaws", answered.json()["message"])
        self.assertEqual([], [line for line in arm.log if line == "free"], "the arm was freed on a count nobody keeps")
        self.assertFalse(self.client.get("/v1/poses").json()["teachable"])

    def test_a_payload_that_is_no_number_frees_nothing(self) -> None:
        """JSON's ``NaN`` and ``Infinity`` parse as floats, and a NaN compares false against every tolerance: a payload
        nobody confirmed must never read as the controller's. The request's own bound refuses one as malformed (422
        ``bad_request``, ``PayloadSeenIn``), before the route compares anything."""
        arm = self.connect_arm(TeachArm())
        for seen in ('{"mass_kg": NaN, "cog_mm": [NaN, NaN, NaN]}', '{"mass_kg": 1.9, "cog_mm": [0.0, NaN, 61.0]}',
                     '{"mass_kg": Infinity, "cog_mm": [0.0, 12.0, 61.0]}'):
            body = ('{"name": "drop_right", "label": "Ablage rechts", "role": "place", "payload_seen": ' + seen + "}")
            answered = self.client.post("/v1/teach", content=body, headers={"content-type": "application/json"})
            self.assertEqual((422, "bad_request"), (answered.status_code, answered.json()["code"]),
                             f"{seen}: {answered.text}")
        self.assertEqual([], [line for line in arm.log if line == "free"])
        # And the route's own comparison takes no NaN for a payload anybody saw, for a caller past the schema.
        from types import SimpleNamespace

        from api.teach import _payload_changed

        self.assertIn("no number", _payload_changed(arm, SimpleNamespace(mass_kg=float("nan"), cog_mm=[0.0, 12.0, 61.0])))
        # The controller's own answer is no number: nothing anybody saw can be it.
        broken = self.connect_arm(TeachArm(payload=ControllerPayload(float("nan"), (0.0, 12.0, 61.0))))
        self.assertEqual((409, "payload_changed"), self.refused(_teach()))
        self.assertEqual([], [line for line in broken.log if line == "free"])

    def test_the_library_s_own_refusals_with_their_codes(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        dummy = DummyRobotArm()
        dummy.connect()
        self.addCleanup(dummy.disconnect)
        self.connect_arm(dummy)
        self.assertEqual((409, "no_hand_guiding"), self.refused(_teach()))
        arm = self.connect_arm(TeachArm())
        arm.planner_state = "starting"
        self.assertEqual((409, "planner_not_ready"), self.refused(_teach()))
        arm.planner_state = "ready"
        self.connect_arm(type("_Blind", (TeachArm,), {"screen_configuration": None})())
        self.assertEqual((409, "screen_unavailable"), self.refused(_teach()))
        self.connect_arm(TeachArm())
        self.assertEqual((422, "invalid_name"), self.refused(_teach(name="home")))
        self.assertEqual((422, "invalid_name"), self.refused(_teach(name="2nd pose")))
        self.assertEqual((422, "invalid_label"), self.refused(_teach(label="")))
        self.assertEqual((422, "invalid_label"), self.refused(_teach(label="Ablage links")), "a label two poses share")
        self.assertEqual((409, "name_taken"), self.refused(_teach(name="drop_left")))

    def test_a_payload_that_is_not_the_one_the_person_saw_refuses_a_teach(self) -> None:
        self.connect_arm(TeachArm())
        self.assertEqual((409, "payload_changed"), self.refused(_teach(payload_seen={"mass_kg": 2.4,
                                                                                    "cog_mm": [0.0, 12.0, 61.0]})))
        self.assertEqual((409, "payload_changed"), self.refused(_teach(payload_seen={"mass_kg": 1.9,
                                                                                    "cog_mm": [0.0, 30.0, 61.0]})))
        self.assertEqual((409, "payload_changed"), self.refused(_teach(payload_seen=None)),
                         "a payload nobody confirmed freed the arm")
        self.connect_arm(TeachArm(payload=None))
        started = self.client.post("/v1/teach", json=_teach(payload_seen=None))
        self.assertEqual(202, started.status_code, "a payload the controller does not report was confirmed unreadable")
        run_id, token = started.json()["run"]["id"], started.json()["token"]
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        wait_for(lambda: self.client.get(f"/v1/runs/{run_id}").json()["state"] != "running", what="the run's end")

    def test_nothing_was_freed_screened_or_written_by_a_refusal(self) -> None:
        arm = self.connect_arm(TeachArm())
        before = self.layer_text()
        self.refused(_teach(name="drop_left"))
        self.refused(_teach(label=""))
        self.assertEqual(before, self.layer_text())
        self.assertEqual([], arm.log)
        self.assertEqual([], self.client.get("/v1/runs").json(), "a refused teach started a run")


class AJawsCheckAndATeachNeverGoOnTogetherTests(LayerCell):
    """Found 2026-10-01: a teach that came in while ``POST /v1/cell/jaws/check`` read the controller freed the arm, and
    the check then asked about the jaws and sent "open now" while a person guided the arm."""

    def setUp(self) -> None:
        from api import jaws as browser_jaws

        super().setUp()
        self.events: list[Any] = []
        self.hand = _toggle(self.events, Person("open"))
        self.arm = self.connect_arm(TeachArm(), gripper=self.hand)
        self.hand.connect()                            # a person said open; the count stands OPEN, so a teach may run
        self.events.clear()
        browser_jaws.install(self.cell)

    def test_a_teach_that_comes_in_while_a_check_reads_the_controller_is_refused(self) -> None:
        from api import jaws as browser_jaws

        in_read, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        real = browser_jaws.controller_stop
        reads: list[str] = []

        def slow_controller(arm: Any) -> str:
            reads.append("read")
            if len(reads) == 1:                        # the check's own read, a dashboard round trip on a cell
                in_read.set()
                release.wait(10.0)
            return real(arm)

        with patch("api.jaws.controller_stop", side_effect=slow_controller):
            checking = _InBackground(self.app, "POST", "/v1/cell/jaws/check")
            self.assertTrue(in_read.wait(10.0), "the check never read the controller")

            started = self.client.post("/v1/teach", json=_teach())

            self.assertEqual((409, "jaws_question_pending"), (started.status_code, started.json()["code"]),
                             started.text)
            release.set()
            asked = wait_for(lambda: self.client.get("/v1/cell/jaws").json().get("question"), what="the question")
            self.client.post("/v1/cell/jaws/answer", json={"question_id": asked["question_id"], "choice": "open"})
            self.assertEqual(200, checking.result().status_code)
        self.assertEqual([], [line for line in self.arm.log if line == "free"], "the arm was freed during a check")
        self.assertEqual(0, _pulses(self.events))

    def test_a_check_that_marks_itself_after_the_route_s_read_is_read_again_before_anything_is_freed(self) -> None:
        """The route reads the jaws question before it starts the run; a check that marks itself in between is read
        again by the run itself, before the arm is freed: the teach ends saying why, and nothing was freed."""
        from api import teach as browser_teach

        real = browser_teach._payload_changed
        flows: list[Any] = []

        def payload_then_a_check(arm: Any, seen: Any) -> str:
            flows.append(self.cell.jaws.open_flow("check"))   # a check that marked itself in the window
            return real(arm, seen)

        with patch("api.teach._payload_changed", side_effect=payload_then_a_check):
            started = self.client.post("/v1/teach", json=_teach())
        self.assertEqual(202, started.status_code, started.text)
        run_id = started.json()["run"]["id"]
        try:
            run = wait_for(lambda: (lambda r: r if r["state"] != "running" else None)(
                self.client.get(f"/v1/runs/{run_id}").json()), timeout=5.0, what="the teach run's end")
        finally:
            self.cell.jaws.close_flow(flows[0], opened=False)
        self.assertEqual(("failed", "teach_not_saved"), (run["state"], run["stop_code"]))
        self.assertIn("jaws question", run["error"])
        self.assertEqual([], [line for line in self.arm.log if line == "free"], "the arm was freed during a check")
        self.assertIsNone(self.cell.teach_ended_at, "a teach that freed nothing counted as hands on the arm")


class AChainWithNoLayerOfItsOwnRefusesATeachTests(LayerCell):
    PROFILE = None
    LAYER = None

    def test_no_layer(self) -> None:
        self.connect_arm(TeachArm())
        answered = self.client.post("/v1/teach", json=_teach())
        self.assertEqual((422, "no_layer"), (answered.status_code, answered.json()["code"]), answered.text)
        self.assertIn("robot.yaml", answered.json()["message"])


class ThePayloadIsReadTests(LayerCell):
    def test_the_controller_s_payload_is_shown_before_the_arm_is_freed(self) -> None:
        self.assertEqual((409, "not_connected"), (lambda r: (r.status_code, r.json()["code"]))(
            self.client.get("/v1/teach/payload")))
        self.connect_arm(TeachArm())
        body = self.client.get("/v1/teach/payload").json()
        self.assertEqual((1.9, [0.0, 12.0, 61.0], True), (body["mass_kg"], body["cog_mm"], body["readable"]))
        self.assertTrue(body["source"])
        self.connect_arm(TeachArm(payload=None))
        body = self.client.get("/v1/teach/payload").json()
        self.assertEqual((None, None, False), (body["mass_kg"], body["cog_mm"], body["readable"]))


# ---------------------------------------------------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------------------------------------------------


class SaveCapturesScreensAndWritesTests(_TeachCell):
    def test_a_clear_pose_is_captured_once_still_screened_written_and_the_run_ends_taught(self) -> None:
        run_id, token = self.start()
        free = self.until_state(run_id, token, "free")
        self.assertIsNotNone(free["time_left_s"])
        self.assertEqual([round(v, 4) for v in LOOK_1_DEG], [round(v, 4) for v in free["joints_deg"]])
        self.assertTrue(any("drop_right" in line for line in free["lines"]))

        saved = self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        self.assertEqual(200, saved.status_code, saved.text)

        run = self.finished(run_id)
        self.assertEqual(("finished", "taught", "done"), (run["state"], run["stop_code"], run["stop_class"]))
        state = self.poll(run_id, token)
        self.assertEqual(("saved", "clear"), (state["state"], state["verdict"]))
        self.assertEqual([[round(v, 4) for v in LOOK_1_DEG]], self.arm.screened)
        pose = next(p for p in self.client.get("/v1/poses").json()["poses"] if p["name"] == "drop_right")
        self.assertEqual(("Ablage rechts", "clear", "taught"), (pose["label"], pose["screen"], pose["source"]))
        types = self.types(run_id)
        for earlier, later in (("teach.free", "teach.holding"), ("teach.holding", "teach.screening"),
                               ("teach.screening", "teach.saved"), ("teach.saved", "run_finished")):
            self.assertLess(types.index(earlier), types.index(later), f"{earlier} came after {later}")
        (said,) = [e for e in self.events(run_id) if e.type == "teach.saved"]
        self.assertEqual(("drop_right", "Ablage rechts", "clear"), (said.data["name"], said.data["label"],
                                                                    said.data["verdict"]))
        started = next(e for e in self.events(run_id) if e.type == "run_started")
        self.assertEqual(("teach", "drop_right", "Ablage rechts"), (started.data["kind"], started.data["name"],
                                                                     started.data["label"]))
        self.assertEqual("teach", self.cell.countdown_because(), "a person's hands were on the arm: no countdown")
        self.assertEqual(1, self.arm.sessions, "one pose per session")

    def test_a_refused_pose_is_never_written_and_the_run_asks_with_a_pose_nearby(self) -> None:
        self.arm.screen = GUARD_REFUSED
        before = self.layer_text()
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        run = self.finished(run_id)
        self.assertEqual(("teach_refused", "ask"), (run["stop_code"], run["stop_class"]))
        self.assertEqual(before, self.layer_text())
        state = self.poll(run_id, token)
        self.assertEqual(("refused", "guard_refused"), (state["state"], state["verdict"]))
        self.assertEqual(6, len(state["nearby_deg"]))
        (refused,) = [e for e in self.events(run_id) if e.type == "teach.refused"]
        self.assertEqual("guard_refused", refused.data["verdict"])
        self.assertEqual(state["nearby_deg"], refused.data["nearby_deg"])

    def test_a_boundary_is_said_and_the_arm_is_never_held_for_it(self) -> None:
        """Below the workspace box the session says so (``teach.outside``) and the poll shows it; back inside it says
        that too. The arm is never held for a boundary: an arm that locks while a person pushes it catches a hand."""
        below = _sample(LOOK_1_DEG, xyz=(450.0, -120.0, 20.0))
        self.arm.stands = [[below] * 60 + [_sample(LOOK_1_DEG)]]
        run_id, token = self.start()
        outside = self.until_state(run_id, token, "free")
        wait_for(lambda: "teach.inside" in self.types(run_id), what="the arm back inside the box")
        (said,) = [e for e in self.events(run_id) if e.type == "teach.outside"]
        self.assertIn("z_min", said.data["sentence"])
        self.assertTrue(outside["outside"] or "teach.inside" in self.types(run_id))
        self.assertFalse(self.poll(run_id, token)["outside"])
        self.assertEqual([], self.held(), "the arm was held for a boundary")
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.finished(run_id)

    def test_the_place_pose_and_replace_reach_the_pose_door(self) -> None:
        """``make_default_place`` makes the taught pose the default place; ``replace`` writes over a pose of that name
        instead of refusing it (``name_taken``): the owner's step at the cell, a default place taught over a bin."""
        run_id, token = self.start(make_default_place=True)
        self.until_state(run_id, token, "free")
        self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        self.assertEqual("taught", self.finished(run_id)["stop_code"])
        self.assertEqual("drop_right", self.client.get("/v1/poses").json()["default_place"])

        run_id, token = self.start(name="drop_left", label="Ablage links neu", replace=True)
        self.until_state(run_id, token, "free")
        self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        self.assertEqual("taught", self.finished(run_id)["stop_code"])
        poses = {p["name"]: p for p in self.client.get("/v1/poses").json()["poses"]}
        self.assertEqual(("Ablage links neu", "taught"), (poses["drop_left"]["label"], poses["drop_left"]["source"]))
        self.assertEqual("drop_right", self.client.get("/v1/poses").json()["default_place"])

    def test_a_save_sent_before_the_free_arm_was_first_sampled_still_saves(self) -> None:
        """``HandGuide.wait`` forgets a console line sent before it waits, as at the terminal: a Save handed to it before
        the first sample would be lost, and the arm held only once the heartbeat lapsed. The session keeps it until the
        free arm is sampled."""
        from api import teach as browser_teach

        freed, sent = threading.Event(), threading.Event()
        self.addCleanup(sent.set)
        real = browser_teach._TeachSession.on_step

        def on_step(session: Any, step: str) -> None:
            real(session, step)
            if step == "free":                         # free, and the guide's wait has not sampled anything yet
                freed.set()
                sent.wait(5.0)

        with patch.object(browser_teach._TeachSession, "on_step", on_step):
            run_id, token = self.start()
            self.assertTrue(freed.wait(10.0), "the arm was never freed")
            saved = self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
            self.assertEqual(200, saved.status_code, saved.text)
            sent.set()
        run = self.finished(run_id)
        self.assertEqual("taught", run["stop_code"], "a Save sent before the first sample was lost")
        self.assertEqual([[round(v, 4) for v in LOOK_1_DEG]], self.arm.screened)

    def test_an_unscreened_pose_is_never_written(self) -> None:
        self.arm.screen = UNSCREENED
        before = self.layer_text()
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        run = self.finished(run_id)
        self.assertEqual(("failed", "teach_not_saved", "teach"), (run["state"], run["stop_code"], run["stop_class"]))
        self.assertEqual(before, self.layer_text())
        self.assertEqual("not_saved", self.poll(run_id, token)["state"])
        self.assertIn("teach.not_saved", self.types(run_id))
        self.assertIsNone(self.cell.recovery, "a teach wrote a recovery record")


class TheHeartbeatAndTheLimitHoldOnlyOnceStillTests(_TeachCell):
    def test_the_owner_s_numbers_hold(self) -> None:
        """Q14: a heartbeat lapse of 3 s, a 5-minute limit, announced 30 s before. Every other test of the heartbeat and
        the limit patches them, so they are pinned here, and on a session nothing patched."""
        from api import teach as browser_teach

        self.assertEqual((3.0, 300.0, 30.0), (browser_teach.TEACH_HEARTBEAT_S, browser_teach.TEACH_TIME_LIMIT_S,
                                              browser_teach.TEACH_WARNING_S))
        run_id, token = self.start()
        free = self.until_state(run_id, token, "free")
        self.assertGreater(free["time_left_s"], 290.0)
        self.assertLessEqual(free["time_left_s"], 300.0)
        (freed,) = [e for e in self.events(run_id) if e.type == "teach.free"]
        self.assertEqual(300.0, freed.data["time_limit_s"])
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.finished(run_id)
        self.assertIsNone(self.poll(run_id, token)["time_left_s"], "a session that ended still counts its free time")

    def test_polls_keep_the_session_free(self) -> None:
        with patch("api.teach.TEACH_HEARTBEAT_S", 0.4):
            run_id, token = self.start()
            self.until_state(run_id, token, "free")
            deadline = time.monotonic() + 1.2
            while time.monotonic() < deadline:
                self.assertEqual("free", self.poll(run_id, token)["state"])
                time.sleep(0.1)
            self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
            self.assertEqual("cancelled", self.finished(run_id)["stop_code"])

    def test_a_lapsed_heartbeat_holds_the_arm_only_once_it_stands_still(self) -> None:
        # The person moves the arm for about 1.5 s after the browser fell silent, then lets it stand.
        self.arm.stands = [[_sample(LOOK_1_DEG, speed=0.3)] * 75 + [_sample(LOOK_1_DEG)]]
        with patch("api.teach.TEACH_HEARTBEAT_S", 0.3):
            run_id, token = self.start()
            run = self.finished(run_id)
        self.assertEqual(("failed", "heartbeat_lost", "teach"), (run["state"], run["stop_code"], run["stop_class"]))
        (lapse,) = [e for e in self.events(run_id) if e.type == "teach.holding_when_still"]
        self.assertEqual("heartbeat", lapse.data["because"])
        (held,) = self.held()
        held_at = float(held.split(" at ")[1])
        still_since = self.arm.still_since[0]
        self.assertGreaterEqual(held_at - still_since, 0.45, "the arm was held while it still moved")
        self.assertEqual([], self.arm.screened, "a pose was captured without a Save")
        self.assertEqual("ended", self.poll(run_id, token)["state"])

    def test_the_time_limit_is_announced_and_holds_once_still(self) -> None:
        with patch("api.teach.TEACH_HEARTBEAT_S", 30.0), patch("api.teach.TEACH_TIME_LIMIT_S", 1.2), \
                patch("api.teach.TEACH_WARNING_S", 0.6):
            run_id, token = self.start()
            run = self.finished(run_id)
        self.assertEqual(("teach_time_limit", "teach"), (run["stop_code"], run["stop_class"]))
        types = self.types(run_id)
        self.assertIn("teach.time_warning", types)
        self.assertLess(types.index("teach.time_warning"), types.index("teach.holding_when_still"))
        (warned,) = [e for e in self.events(run_id) if e.type == "teach.time_warning"]
        self.assertAlmostEqual(0.6, warned.data["seconds_left"], delta=0.3)
        (lapse,) = [e for e in self.events(run_id) if e.type == "teach.holding_when_still"]
        self.assertEqual("time_limit", lapse.data["because"])


class TheWaysOutHoldAtOnceTests(_TeachCell):
    def test_cancel_holds_at_once_and_saves_nothing(self) -> None:
        before = self.layer_text()
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        cancelled = self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.assertEqual(200, cancelled.status_code, cancelled.text)
        run = self.finished(run_id)
        self.assertEqual(("cancelled", "cancelled", "operator"), (run["state"], run["stop_code"], run["stop_class"]))
        self.assertEqual(1, len(self.held()))
        self.assertEqual(before, self.layer_text())
        self.assertEqual([], self.arm.screened)
        again = self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.assertEqual(200, again.status_code, "the page's beacon after the end was refused")

    def test_a_cancel_while_save_waits_for_stillness_holds_at_once(self) -> None:
        self.arm.stands = [[_sample(LOOK_1_DEG, speed=0.3)] * 400 + [_sample(LOOK_1_DEG)]]
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        time.sleep(0.3)
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        run = self.finished(run_id, timeout=5.0)
        self.assertEqual("cancelled", run["stop_code"])
        self.assertEqual([], self.arm.screened)

    def test_a_cancel_while_the_pose_is_screened_writes_nothing(self) -> None:
        """Save was pressed and the arm holds while the pose is screened (about a second with the planner); a Cancel
        that comes in then saves nothing either."""
        screening, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        screen = self.arm.screen

        def slow_screen(joints: Any, *, ask_planner: bool = True) -> Any:
            screening.set()
            release.wait(10.0)
            return screen

        self.arm.screen_configuration = slow_screen  # type: ignore[method-assign]
        before = self.layer_text()
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        self.assertTrue(screening.wait(10.0), "the pose was never screened")
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        release.set()
        run = self.finished(run_id)
        self.assertEqual("cancelled", run["stop_code"])
        self.assertEqual(before, self.layer_text(), "a cancelled pose was written")

    def test_a_halt_during_a_teach_holds_at_once(self) -> None:
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        live = self.cell.registry.get(run_id)
        assert live is not None
        live.halt_requested = True              # what "halt now" sets on the active run
        run = self.finished(run_id)
        self.assertEqual("cancelled", run["stop_code"])
        self.assertEqual(1, len(self.held()))

    def test_a_halt_latched_on_the_arm_holds_a_teach_at_once(self) -> None:
        """Whatever latched the arm, the teach holds at once: the run's own flag is not the only way a halt comes."""
        self.arm.stands = [_moving()]
        run_id, token = self.start()
        self.until_state(run_id, token, "free")

        self.arm.halt("the operator pressed halt now")

        run = self.ended_while_polled(run_id, token)
        self.assertEqual("cancelled", run["stop_code"])
        self.assertEqual(1, len(self.held()))
        self.assertIn("halted", run["error"])

    def test_an_abandoned_run_holds_its_teach_at_once(self) -> None:
        """A Disconnect abandons the active run first: its teach holds at once, while the person still moves the arm."""
        self.arm.stands = [_moving()]
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        live = self.cell.registry.get(run_id)
        assert live is not None

        live.abandoned = "the cell was disconnected from the console"

        run = self.ended_while_polled(run_id, token)
        self.assertEqual(("failed", "disconnected"), (run["state"], run["stop_code"]))
        self.assertEqual(1, len(self.held()))

    def test_a_stop_request_holds_a_teach_at_once(self) -> None:
        self.arm.stands = [_moving()]
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        live = self.cell.registry.get(run_id)
        assert live is not None

        live.stop_requested = True

        run = self.ended_while_polled(run_id, token)
        self.assertEqual("cancelled", run["stop_code"])
        self.assertEqual(1, len(self.held()))

    def test_a_disconnect_during_a_teach_holds_the_arm_before_it_takes_it_down(self) -> None:
        """Q14: a Disconnect holds at once, and the arm is held before it comes down: an arm taken down in teach mode is
        stopped by its watchdog in a person's hands, not held by the console."""
        real = self.arm.disconnect

        def disconnect() -> None:
            self.arm.log.append("disconnect")
            real()

        self.arm.disconnect = disconnect  # type: ignore[method-assign]
        self.arm.stands = [_moving()]
        run_id, token = self.start()
        self.until_state(run_id, token, "free")

        down = self.client.post("/v1/cell/disconnect")

        self.assertEqual(200, down.status_code, down.text)
        order = [line.split(" at ")[0] for line in self.arm.log if line.startswith(("hold", "disconnect"))]
        self.assertEqual(["hold", "disconnect"], order[:2], "the arm came down before the teach held it")
        run = self.finished(run_id)
        self.assertEqual("disconnected", run["stop_code"])
        self.assertEqual(1, len(self.held()))

    def test_a_teach_hold_that_raises_never_keeps_the_cell_up(self) -> None:
        """The session asks the teach to hold before it takes the arm down; a hold that raises is logged, and the cell
        comes down all the same."""

        class _Raises:
            def hold_at_once(self, why: str) -> bool:
                raise RuntimeError("the teach would not say")

        self.cell.session.teaches = _Raises()  # type: ignore[assignment]

        self.cell.session.disconnect()

        self.assertFalse(self.cell.session.connected)
        self.assertFalse(self.arm.is_connected)

    def test_end_and_join_holds_an_open_teach_and_waits_for_its_thread(self) -> None:
        from api import teach as browser_teach

        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        self.assertTrue(browser_teach.end_and_join(self.cell, timeout_s=10.0))
        run = self.client.get(f"/v1/runs/{run_id}").json()
        self.assertEqual("cancelled", run["stop_code"])
        self.assertIsNone(self.cell.active_run_id, "the teach kept the run lock")
        self.assertEqual(1, len(self.held()))


class TheSessionIsTheTokenHoldersTests(_TeachCell):
    def test_a_wrong_token_no_such_run_and_a_save_where_the_arm_is_not_free(self) -> None:
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        wrong = self.client.get(f"/v1/teach/{run_id}", params={"token": "not-the-token"})
        self.assertEqual((403, "wrong_token"), (wrong.status_code, wrong.json()["code"]), wrong.text)
        for verb in ("capture", "cancel"):
            refused = self.client.post(f"/v1/teach/{run_id}/{verb}", params={"token": "not-the-token"})
            self.assertEqual((403, "wrong_token"), (refused.status_code, refused.json()["code"]))
        nobody = self.client.get("/v1/teach/run-nobody", params={"token": token})
        self.assertEqual((404, "no_such_run"), (nobody.status_code, nobody.json()["code"]))
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.finished(run_id)
        late = self.client.post(f"/v1/teach/{run_id}/capture", params={"token": token})
        self.assertEqual((409, "not_free"), (late.status_code, late.json()["code"]), late.text)

    def test_a_second_teach_is_refused_while_one_runs(self) -> None:
        run_id, token = self.start()
        self.until_state(run_id, token, "free")
        second = self.client.post("/v1/teach", json=_teach(name="park_2", label="Parken 2"))
        self.assertEqual((409, "run_active"), (second.status_code, second.json()["code"]), second.text)
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.finished(run_id)

    def test_the_payload_confirmed_is_never_asked_again(self) -> None:
        self.arm.payload = ControllerPayload(1.9, (0.0, 12.0, 61.0))
        run_id, token = self.start()
        state = self.until_state(run_id, token, "free")
        self.assertFalse(any("Is this payload right" in line for line in state["lines"]))
        self.client.post(f"/v1/teach/{run_id}/cancel", params={"token": token})
        self.finished(run_id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
