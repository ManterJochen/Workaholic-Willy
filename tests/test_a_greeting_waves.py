"""A greeting in the chat waves (the owner, 2026-10-06: "Sofort winken, aber man kann es in der App-Config einstellen,
dass man es vorher bestätigen muss. Es ist aber jetzt erstmal so, dass er direkt winkt").

What this file pins:

* the reader: a sentence the model read as no command that opens with a greeting or a farewell, or asks to wave, is a
  greeting; a sentence that is nothing but one is one whatever the model made of it, bar a stop; a command with a
  greeting in front stays the model's task, and a negation greets nobody;
* the app config: ``runtime.greeting.wave`` is ``direct`` as shipped and in the schema, and the card says how the
  console answers (``CommandOut.greeting``: ``direct``, ``confirm``, ``off``; ``None`` for any other sentence); the
  console's literal is the config's;
* the gesture: wrist 2 out 15 degrees to each side, twice, and back, every other joint as it stood, each swing a
  straight joint line the arm judges; a refused swing or a stop ends it where the arm stands, and nothing is sent to
  make up for it;
* the route: ``POST /v1/cell/wave`` is a moving run of kind ``wave`` with its events, refused as a new task is (bar a
  carried part), stopped before its next swing by ``POST /v1/task/stop``; a swing refused before the first one ran ends
  it ``cancelled`` with nothing moved and no stop record, after one ran ``return_failed`` with the record written.

Honesty bucket (2): the real reader over a scripted model, the real console and routes on console_dummy and on the
scripted cell, whose arm drives a judged joint line here.
"""

from __future__ import annotations

import typing
import unittest
from typing import Any

from src.config.schema.runtime import GreetingConfig, GreetingWave, RuntimeConfig
from src.models.vlm.command import greets, understand
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.execution import gestures
from tests._console_task_fakes import ConsoleCase, event_types, wait_for_event
from tests._task_fakes import HOME_JOINTS, TaskArm, _key
from tests.test_api_commands import SENTENCE, _CommandCell, _continuation
from tests.test_vlm_command import POSES, _answer, _Canned

#: Sentences, the model's reading of them (``None``: no usable answer), and whether they greet.
SENTENCES: tuple[tuple[str, str | None, bool], ...] = (
    ("Hallo Willy!", "none", True),
    ("Hallo Willy, wie geht's?", "none", True),
    ("Grüß Gott, Willy", "none", True),
    ("Guten Morgen Willy", "none", True),
    ("Tschüss Willy", "none", True),
    ("Willy, wink mal!", "none", True),
    ("Kannst du winken?", "none", True),
    ("Moin moin", "none", True),
    # nothing but a greeting: whatever the model made of it
    ("Hallo Willy!", "task", True),
    ("Hallo zusammen", None, True),
    ("Willy, wink mal!", "task", True),
    # a command with a greeting in front stays the model's task
    ("Hallo Willy, nimm den roten Würfel", "task", False),
    ("Hallo Willy, räum die Kiste aus", "task", False),
    ("Hallo Willy, wie geht's?", "task", False),
    # a stop, a negation, a greeting only mentioned, or no greeting at all
    ("Hallo, stopp!", "stop", False),
    ("Bitte nicht winken", "none", False),
    ("Don't wave", "none", False),
    ("Was heißt hallo auf Englisch?", "none", False),
    ("wie spät ist es?", "none", False),
    ("Willy", "none", False),
)


class TheReaderKnowsAGreetingTests(unittest.TestCase):
    def test_each_sentence(self) -> None:
        for sentence, intent, want in SENTENCES:
            with self.subTest(sentence=sentence, intent=intent):
                self.assertEqual(want, greets(sentence, intent=intent))  # type: ignore[arg-type]

    def test_the_reading_carries_it_and_the_card_says_it(self) -> None:
        reading = understand("Hallo Willy, wie geht's?", ask=_Canned(_answer(intent="none")), poses=POSES)
        self.assertTrue(reading.greeting)
        self.assertIn("a greeting", reading.render())

    def test_a_task_is_no_greeting(self) -> None:
        reading = understand("Hallo Willy, nimm den roten Würfel",
                             ask=_Canned(_answer(object="red cube", object_said="den roten Würfel")), poses=POSES)
        self.assertEqual(("task", False), (reading.intent, reading.greeting))

    def test_nothing_but_a_greeting_greets_when_the_answer_was_unusable(self) -> None:
        reading = understand("Hallo Willy!", ask=_Canned("kein JSON", "wieder kein JSON"), poses=POSES)
        self.assertEqual((False, True), (reading.understood, reading.greeting))


class TheAppConfigTests(unittest.TestCase):
    def test_it_waves_at_once_as_shipped(self) -> None:
        self.assertEqual("direct", GreetingConfig().wave)
        self.assertEqual("direct", RuntimeConfig().greeting.wave)

    def test_a_bare_yaml_off_means_off(self) -> None:
        """YAML reads ``wave: off`` as false, and ``wave: on`` as true."""
        self.assertEqual("off", GreetingConfig.model_validate({"wave": False}).wave)
        self.assertEqual("direct", GreetingConfig.model_validate({"wave": True}).wave)

    def test_the_shipped_tree_says_direct(self) -> None:
        from src.config.loader import load_config

        self.assertEqual("direct", load_config().runtime.greeting.wave)

    def test_the_consoles_answer_is_the_configs(self) -> None:
        from api.schemas import GreetingAnswer

        self.assertEqual(typing.get_args(GreetingWave), typing.get_args(GreetingAnswer))


class _LineArm(TaskArm):
    """The scripted arm, driving a judged joint line as the UR does (``DrivesJointLines``)."""

    def move_to_joints_on_the_line(self, joints: JointPositions) -> MotionResult:
        return self._motion("joints", joints, MotionCommand.MOVE_JOINTS, joints=joints)


def _swung(joints: JointPositions) -> float:
    """How far wrist 2 stands from where it stands at Home, degrees."""
    return round(joints.degrees()[gestures.WAVE_JOINT] - HOME_JOINTS.degrees()[gestures.WAVE_JOINT], 6)


class TheGestureTests(unittest.TestCase):
    def test_wrist_2_swings_out_to_each_side_twice_and_back(self) -> None:
        targets = gestures.wave_targets(HOME_JOINTS)
        self.assertEqual([15.0, -15.0, 15.0, -15.0, 0.0], [_swung(t) for t in targets])
        for target in targets:
            others = [v for i, v in enumerate(target.degrees()) if i != gestures.WAVE_JOINT]
            self.assertEqual([round(v, 6) for i, v in enumerate(HOME_JOINTS.degrees()) if i != gestures.WAVE_JOINT],
                             [round(v, 6) for v in others])

    def test_every_swing_is_a_joint_line_and_it_ends_where_it_started(self) -> None:
        log: list[Any] = []
        waved = gestures.wave(_LineArm(log))
        self.assertTrue(waved.ok, waved.render())
        moves = [entry for entry in log if entry and entry[0] == "joints"]
        self.assertEqual(5, len(moves))
        self.assertEqual(("joints", _key(HOME_JOINTS)), moves[-1])

    def test_a_refused_swing_ends_it_where_the_arm_stands(self) -> None:
        log: list[Any] = []
        arm = _LineArm(log, refuse=lambda _kind, index, _target: MotionStatus.WORKSPACE_REJECTED if index == 1 else None)
        waved = gestures.wave(arm)
        self.assertEqual(("refused", True, False), (waved.ended, waved.moved, waved.ok))
        self.assertEqual(1, sum(1 for entry in log if entry and entry[0] == "joints"))
        self.assertEqual(15.0, _swung(arm.get_joint_positions()))
        self.assertIn("refused after 1 of 5", waved.render())

    def test_a_stop_ends_it_before_the_next_swing(self) -> None:
        log: list[Any] = []
        arm = _LineArm(log)
        waved = gestures.wave(arm, should_stop=lambda: sum(1 for e in log if e and e[0] == "joints") >= 2)
        self.assertEqual(("stopped", 2), (waved.ended, len(waved.reports)))

    def test_an_arm_that_plans_around_a_line_is_refused_before_the_first_swing(self) -> None:
        log: list[Any] = []
        waved = gestures.wave(TaskArm(log))
        self.assertEqual(("refused", False), (waved.ended, waved.moved))
        self.assertIs(MotionStatus.UNSUPPORTED, waved.refusal.status)  # type: ignore[union-attr]
        self.assertEqual([], [entry for entry in log if entry and entry[0] == "joints"])


class TheWaveRouteTests(ConsoleCase):
    def _line_arm(self, cell: Any) -> list[JointPositions]:
        """The scripted cell's arm drives a judged joint line; the lines it was asked for."""
        lines: list[JointPositions] = []

        def on_the_line(joints: JointPositions) -> MotionResult:
            lines.append(joints)
            return cell.arm._motion("joints", joints, MotionCommand.MOVE_JOINTS, joints=joints)  # noqa: SLF001

        cell.arm.move_to_joints_on_the_line = on_the_line
        return lines

    def test_a_wave_on_console_dummy(self) -> None:
        self.build_dummy()
        wave = self.client.post("/v1/cell/wave")
        self.assertEqual(202, wave.status_code, wave.text)
        self.assertEqual(("wave", "running"), (wave.json()["kind"], wave.json()["state"]))
        body = self.finished(wave.json()["id"])
        self.assertEqual(("finished", "finished", "done"), (body["state"], body["stop_code"], body["stop_class"]))
        self.assertEqual(["run_started", "wave.started", "wave.done", "run_finished"],
                         event_types(self.cell, body["id"]))

    def test_on_a_cell_every_swing_is_a_judged_line_and_the_wrist_comes_back(self) -> None:
        cell = self.scripted()
        lines = self._line_arm(cell)
        body = self.finished(self.client.post("/v1/cell/wave").json()["id"])
        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual([15.0, -15.0, 15.0, -15.0, 0.0], [_swung(j) for j in lines])
        self.assertEqual(("joints", _key(HOME_JOINTS)), cell.motions()[-1])
        self.assertEqual(0, cell.do0_changes())

    def test_a_swing_refused_before_any_ran_ends_cancelled_with_nothing_moved(self) -> None:
        cell = self.scripted()
        self._line_arm(cell)
        cell.arm.refuse = lambda _kind, _index, _target: MotionStatus.WORKSPACE_REJECTED
        body = self.finished(self.client.post("/v1/cell/wave").json()["id"])
        self.assertEqual(("cancelled", "cancelled"), (body["state"], body["stop_code"]), body)
        self.assertFalse(wait_for_event(self.cell, body["id"], "wave.refused").data["moved"])
        self.assertIsNone(self.cell.recovery)
        self.assertEqual([], cell.motions())

    def test_a_swing_refused_after_one_ran_ends_return_failed_where_the_arm_stands(self) -> None:
        cell = self.scripted()
        self._line_arm(cell)
        cell.arm.refuse = lambda _kind, index, _target: MotionStatus.WORKSPACE_REJECTED if index >= 1 else None
        body = self.finished(self.client.post("/v1/cell/wave").json()["id"])
        self.assertEqual(("failed", "return_failed", "problem"), (body["state"], body["stop_code"], body["stop_class"]))
        self.assertTrue(wait_for_event(self.cell, body["id"], "wave.refused").data["moved"])
        self.assertEqual(body["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]
        self.assertEqual(1, len(cell.motions()))

    def test_the_stop_ends_it_before_the_next_swing(self) -> None:
        cell = self.scripted()
        lines = self._line_arm(cell)
        registry = self.cell.registry

        def stop_after_the_first(kind: str, index: int) -> None:
            if index == 1:
                active = registry.active()
                assert active is not None
                registry.stop_home(active.id)

        cell.arm.on_motion = stop_after_the_first
        body = self.finished(self.client.post("/v1/cell/wave").json()["id"])
        self.assertEqual(("cancelled", "cancelled", "operator"),
                         (body["state"], body["stop_code"], body["stop_class"]), body)
        self.assertEqual(2, len(lines), "the swing under way runs to its end, and no next one is sent")
        self.assertIsNone(self.cell.recovery)

    def test_the_task_stop_route_stops_a_wave(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        run = self.cell.registry.start_kind(self.cell, RunKind.WAVE, lambda _c, r: "cancelled" if r.stop_requested
                                            else "finished")
        stopped = self.client.post("/v1/task/stop", params={"run_id": run.id})
        self.assertEqual(200, stopped.status_code, stopped.text)
        self.finished(run.id)

    def test_every_refusal_moves_nothing(self) -> None:
        from unittest.mock import patch

        from tests._task_fakes import PROTECTIVE, RUNNING

        self.refused(self.client.post("/v1/cell/wave"), 409, "not_connected")
        cell = self.scripted()
        self._line_arm(cell)
        cell.arm.halt("the operator pressed halt now")
        cell.arm.status = PROTECTIVE
        self.refused(self.client.post("/v1/cell/wave"), 409, "halted")
        cell.arm.clear_halt()
        self.refused(self.client.post("/v1/cell/wave"), 409, "controller_stopped")
        cell.arm.status = RUNNING
        cell.service._needs_person = "a push stopped where the arm stands"  # noqa: SLF001
        self.refused(self.client.post("/v1/cell/wave"), 409, "needs_person")
        cell.service._needs_person = ""  # noqa: SLF001
        with patch("api.jaws.pending", return_value=object()):
            self.refused(self.client.post("/v1/cell/wave"), 409, "jaws_question_pending")
        cell.jaws.set_closed(True)
        self.refused(self.client.post("/v1/cell/wave"), 409, "part_still_held")
        cell.jaws.set_closed(False)
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined]
        self.refused(self.client.post("/v1/cell/wave"), 409, "jaws_not_confirmed")
        cell.jaws._io.do[0] = not cell.jaws._io.do.get(0, False)  # type: ignore[attr-defined]
        cell.arm.plans_paths = False  # type: ignore[misc]
        self.refused(self.client.post("/v1/cell/wave"), 409, "route_refused")
        self.assertEqual([], cell.motions())
        self.assertEqual([], self.client.get("/v1/runs").json())

    def test_a_wave_is_no_way_back_after_a_stop(self) -> None:
        self.build_dummy()
        from api.codes import RunKind

        run = self.cell.registry.start_kind(self.cell, RunKind.TASK, lambda _c, _r: "return_failed")
        self.finished(run.id)
        self.refused(self.client.post("/v1/cell/wave"), 409, "cell_not_cleared")
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        self.refused(self.client.post("/v1/cell/wave"), 409, "restart_required")


class TheCardSaysHowTheConsoleAnswersTests(_CommandCell):
    """``POST /v1/commands/parse`` on the command cell of ``tests/test_api_commands.py``, its model scripted."""

    def _say(self, wave: str) -> None:
        (self.tmp / "app" / "runtime.yaml").write_text(f"runtime:\n  greeting:\n    wave: {wave}\n", encoding="utf-8")

    def test_a_greeting_says_direct_as_shipped(self) -> None:
        self.scripted(_continuation(intent="none"))
        card = self.parse("Hallo Willy!").json()
        self.assertEqual(("none", "direct"), (card["intent"], card["greeting"]))
        self.assertEqual([], self.client.get("/v1/runs").json(), "reading a greeting started a run")

    def test_the_app_config_decides(self) -> None:
        for wave in ("confirm", "off"):
            with self.subTest(wave=wave):
                self._say(wave)
                self.scripted(_continuation(intent="none"))
                self.assertEqual(wave, self.parse("Hallo Willy!").json()["greeting"])

    def test_any_other_sentence_says_none(self) -> None:
        self.assertIsNone(self.parse(SENTENCE).json()["greeting"])


if __name__ == "__main__":
    unittest.main()
