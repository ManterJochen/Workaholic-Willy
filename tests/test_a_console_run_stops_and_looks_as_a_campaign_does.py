"""A run started from the console stops where a campaign stops, and looks from where a campaign looks.

Review of 2026-09-24. ``PickRun`` (``src/robot/execution/pick_run.py``) ends a campaign on three things a pick reports
rather than raises: a fault of the cell, a controller that cannot move (``controller_stopped``), and a hand that needs a
person (``gripper_fault``: a gripper that raised, or a toggle that would not start a pick on jaws it believes closed with
nobody at a terminal). The console's run loop (``api/runs.py``, ``RunRegistry._drive``) stopped on the fault alone, so a
console run went on picking after the other two: the next pick perceived again, and after a gripper that raised at the
close it approached the next part with jaws in a state nobody could name.

And a campaign on a wrist camera looks from the arm's home before each pick (``PickRun._looks_for``); the console called
``pick()`` with no look, so a wrist camera perceived from wherever the last pick left the arm.

Each test calls the run body, ``RunRegistry._drive``, on this thread: it is what the run's own thread runs, and calling it
here makes the order of what it did readable without waiting on a thread. The hand is the real ``JawIOGripper`` on a
recording tool I/O, and the service the real ``AutonomousGraspService`` on the fakes the campaign tests use.
"""

from __future__ import annotations

import io
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from api.events import EventHub
from api.runs import Run, RunRegistry, RunState


def _registry_and_console(service: Any) -> tuple[RunRegistry, Any, EventHub]:
    hub = EventHub()
    console = types.SimpleNamespace(active_run_id=None, session=types.SimpleNamespace(service=service))
    return RunRegistry(hub), console, hub


def _drive(service: Any, *, picks: int) -> tuple[Run, EventHub]:
    """One console run of ``picks`` picks on ``service``, driven to its end on this thread.

    The console's run lock is set as ``RunRegistry.start`` sets it, and every run here must hand it back, a failed one
    as a finished one does: a lock kept by a run that ended leaves the console read-only.
    """
    registry, console, hub = _registry_and_console(service)
    run = Run(id="run-console", prompt="", requested_picks=picks)
    console.active_run_id = run.id
    registry._drive(console, run)
    assert console.active_run_id is None, "the run ended and kept the console's run lock"
    return run, hub


def _types(hub: EventHub, run: Run) -> list[str]:
    return [e.type for e in hub.since(run.id, 0)[0]]


class _Report:
    """A pick that succeeded, saying nothing a run stops on."""

    outcome = "succeeded"
    succeeded = True
    fault = None
    controller_stopped = False
    gripper_fault = ""

    def failure_summary(self) -> str:
        return ""


class _Service:
    """A service that records what each pick was handed; ``perceives_from_the_wrist`` as the cell says it."""

    def __init__(self, *, wrist: Any) -> None:
        self.perceives_from_the_wrist = wrist
        self.calls: list[dict[str, Any]] = []

    def pick(self, **kwargs: Any) -> _Report:
        self.calls.append(dict(kwargs))
        return _Report()

    def attach_progress_listener(self, _listener: Any) -> None:
        return None

    def set_cancel_check(self, _check: Any) -> None:
        return None

    def set_prompt(self, _prompt: Any) -> None:
        return None


# ---------------------------------------------------------------------------------------------------
# Where a run stops
# ---------------------------------------------------------------------------------------------------


class ARunStopsOnAHandThatNeedsAPersonTests(unittest.TestCase):
    def test_a_gripper_that_raised_at_the_close_ends_the_run_failed_after_one_pick(self) -> None:
        """Red before: three picks, the run FINISHED, and nothing said the hand needed a person."""
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _service
        from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _toggle

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined] # the controller refuses from here
        service = _Counted(_service(_Arm(events), jaws))

        run, hub = _drive(service, picks=3)

        self.assertEqual(1, service.picks, "the run picked again after a gripper that raised")
        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(1, run.attempted, "the pick that ran is not counted")
        self.assertEqual(["execution_failed"], run.outcomes)
        self.assertIn("the gripper needs a person", run.error)
        self.assertIn("the gripper raised: RobotError", run.error)
        self.assertIsNotNone(run.finished_at)
        kinds = _types(hub, run)
        self.assertIn("run_error", kinds)
        self.assertEqual("run_finished", kinds[-1])
        self.assertLess(kinds.index("pick_result"), kinds.index("run_error"))

    def test_a_toggle_believed_closed_with_nobody_to_ask_ends_the_run_before_anything(self) -> None:
        """Since the review of 2026-09-24 (C1) the run ends before it calls ``pick()`` at all: it no longer hands the
        question to the pick, which would have asked at the server's terminal from the run's own thread."""
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _service
        from tests.test_a_toggle_asks_where_its_jaws_stand import _NO_TERMINAL, Person, _toggle

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws.set_closed(True)                               # the last pick of an earlier run, never placed
        jaws._ask = None
        events.clear()
        service = _Counted(_service(_Arm(events), jaws))

        with patch("sys.stdin", _NO_TERMINAL):
            run, hub = _drive(service, picks=3)

        self.assertEqual(0, service.picks, "a pick was started on jaws the program believes closed")
        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(0, run.attempted)
        for part in ("the gripper needs a person", _STILL_HELD, "terminal"):
            self.assertIn(part, run.error)
        self.assertEqual([], events, "the arm moved, the camera was asked or the jaws were pulsed")
        self.assertIn("run_error", _types(hub, run))
        self.assertTrue(jaws.jaws_closed, "the program's count moved although nothing was pulsed")


#: What a console run says when it will not start a pick on a toggle whose jaws the program believes closed.
_STILL_HELD = "the jaws still hold the last part (a toggle hand with no sensor): release or place it, then start a new run"


class _Terminal(io.StringIO):
    """The server's stdin where the console was started from a terminal: a question would block on ``input()``."""

    def isatty(self) -> bool:
        return True


class ARunNeverAsksAtTheServersTerminalTests(unittest.TestCase):
    """C1 of the review of 2026-09-24: a console run of N picks never releases the part it lifted, so on a toggle hand
    every pick after the first success started on jaws the program believed CLOSED, and ``pick()`` asked where they
    stand on the SERVER's terminal from the run's thread. The browser showed nothing, Stop could not interrupt
    ``input()``, and a later 'p' pulsed the part out wherever the arm stood. The run now ends FAILED before such a pick
    and never blocks on the terminal.
    """

    def test_the_second_pick_on_a_toggle_ends_the_run_and_asks_nobody(self) -> None:
        """Red before: ``input()`` asked twice from the run's thread, and three picks each pulsed the jaws."""
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _pulses, _service, _toggle

        events: list[Any] = []
        jaws = _toggle(events)
        jaws._ask = None                                    # the console's hand: the question goes to stdin
        service = _Counted(_service(_Arm(events), jaws))

        with patch("sys.stdin", _Terminal("")), patch("builtins.input", return_value="") as typed:
            run, hub = _drive(service, picks=3)

        typed.assert_not_called()
        self.assertEqual(1, service.picks, "a pick started on jaws the program believes closed on the last part")
        self.assertEqual(1, _pulses(events), "exactly one pulse, at the part; none after it")
        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(1, run.attempted)
        self.assertEqual(1, run.succeeded, "the pick that ran is not counted as it ended")
        self.assertIn("the gripper needs a person", run.error)
        self.assertIn(_STILL_HELD, run.error)
        self.assertIn("terminal", run.error)
        self.assertTrue(jaws.jaws_closed)
        kinds = _types(hub, run)
        self.assertLess(kinds.index("pick_result"), kinds.index("run_error"))
        self.assertEqual("run_finished", kinds[-1])
        said = [e for e in hub.since(run.id, 0)[0] if e.type == "run_error"]
        self.assertEqual(1, len(said))
        self.assertIn(_STILL_HELD, said[0].human)

    def test_a_pulse_whose_edge_is_unknown_ends_the_run_before_the_first_pick(self) -> None:
        """Red before: the pick asked at the terminal from the run's thread; here the run ends and asks nobody."""
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _service
        from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _toggle

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined]
        with self.assertRaises(Exception):
            jaws.set_closed(True)                           # the high write raised: nobody can say where they stand
        jaws._io.refuse_high = False                        # type: ignore[attr-defined]
        jaws._ask = None
        events.clear()
        service = _Counted(_service(_Arm(events), jaws))

        with patch("sys.stdin", _Terminal("")), patch("builtins.input", return_value="") as typed:
            run, _hub = _drive(service, picks=2)

        typed.assert_not_called()
        self.assertEqual(0, service.picks)
        self.assertIs(RunState.FAILED, run.state)
        for part in ("the gripper needs a person", "nobody can say where they stand", "terminal"):
            self.assertIn(part, run.error)
        self.assertEqual([], events, "the arm moved, the camera was asked or the jaws were pulsed")

    def test_the_first_pick_on_open_jaws_runs_and_asks_nobody(self) -> None:
        """⭐ THE CONTROL: jaws the person said stand open at connect; the one pick runs, one pulse at the part."""
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _pulses, _service, _toggle

        events: list[Any] = []
        jaws = _toggle(events)
        jaws._ask = None
        service = _Counted(_service(_Arm(events), jaws))

        with patch("sys.stdin", _Terminal("")), patch("builtins.input", return_value="") as typed:
            run, hub = _drive(service, picks=1)

        typed.assert_not_called()
        self.assertEqual(1, service.picks)
        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual("", run.error)
        self.assertEqual(1, _pulses(events))
        self.assertNotIn("run_error", _types(hub, run))

    def test_a_hand_that_is_not_a_toggle_is_not_asked_about(self) -> None:
        """A service whose hand does not say it toggles with no sensor (a double answering every attribute included)
        runs every pick: the gate reads ``toggles_without_sensor is True``, as every pick path does."""
        service = _Service(wrist=False)
        service.runtime = types.SimpleNamespace(orchestrator=types.SimpleNamespace(gripper=MagicMock()))  # type: ignore[attr-defined]
        run, _hub = _drive(service, picks=3)

        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual(3, len(service.calls))


class ARunStopsOnAStoppedControllerTests(unittest.TestCase):
    def test_a_protective_stop_during_the_lift_ends_the_run_failed_with_the_part_still_held(self) -> None:
        """Red before: five picks; each later one perceived again for an arm a person had to walk up to."""
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _pulses, _service, _toggle

        events: list[Any] = []
        jaws = _toggle(events)
        # Four approach waypoints, then the lift: the lift is motion 4, and it meets the protective stop.
        service = _Counted(_service(_Arm(events, stop_at_move=4), jaws))

        run, hub = _drive(service, picks=5)

        self.assertEqual(1, service.picks)
        self.assertIs(RunState.FAILED, run.state)
        self.assertEqual(["cancelled"], run.outcomes)
        self.assertIn("the controller cannot move", run.error)
        self.assertIn("a person clears the stop", run.error)
        self.assertEqual(1, _pulses(events), "the jaws were pulsed after the stop: the held part dropped")
        self.assertIn("run_error", _types(hub, run))


class ARunGoesOnWhereNothingSaysStopTests(unittest.TestCase):
    def test_a_run_on_a_running_controller_runs_every_pick(self) -> None:
        """⭐ THE CONTROL: the same cell, nothing refused, three picks and FINISHED.

        Each part is set down before the next pick, as a person or a place does: a console run never releases what it
        lifted, and on a toggle a pick on jaws the program believes closed ends the run (C1 above) instead of asking.
        """
        from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _Counted, _service, _toggle

        events: list[Any] = []
        jaws = _toggle(events)

        class _SetDown(_Counted):
            def pick(self) -> Any:
                report = super().pick()
                jaws.set_closed(False)                      # the part set down: one pulse at the release
                return report

        service = _SetDown(_service(_Arm(events), jaws))

        run, hub = _drive(service, picks=3)

        self.assertEqual(3, service.picks)
        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual("", run.error)
        self.assertNotIn("run_error", _types(hub, run))

    def test_a_double_that_answers_every_attribute_does_not_stop_a_run(self) -> None:
        """``is True`` and a string, as ``PickRun`` reads them: a mock's answer to every attribute is neither."""

        class _Mocked(_Service):
            def pick(self, **kwargs: Any) -> Any:
                self.calls.append(dict(kwargs))
                report = MagicMock()
                report.outcome = "succeeded"
                report.succeeded = True
                report.fault = None
                return report

        service = _Mocked(wrist=False)
        run, _hub = _drive(service, picks=2)

        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual(2, len(service.calls))


# ---------------------------------------------------------------------------------------------------
# Where a run looks from
# ---------------------------------------------------------------------------------------------------


class ARunLooksFromWhereACampaignLooksTests(unittest.TestCase):
    def test_a_wrist_camera_looks_from_home_before_every_pick(self) -> None:
        """Red before: ``pick()`` with no look, so the camera perceived from wherever the last pick left the arm."""
        from src.robot.execution.looks import HOME

        service = _Service(wrist=True)
        run, _hub = _drive(service, picks=3)

        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual([{"look": HOME}] * 3, service.calls)

    def test_a_fixed_camera_is_handed_no_look(self) -> None:
        """⭐ THE CONTROL: a fixed camera perceives from where the arm stands, as it always did."""
        service = _Service(wrist=False)
        run, _hub = _drive(service, picks=2)

        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual([{}, {}], service.calls)

    def test_a_service_that_does_not_say_or_answers_every_attribute_is_handed_no_look(self) -> None:
        for wrist in (MagicMock(), "yes", 1):
            with self.subTest(wrist=wrist):
                service = _Service(wrist=wrist)
                run, _hub = _drive(service, picks=1)
                self.assertEqual([{}], service.calls)
        bare = _Service(wrist=False)
        del bare.perceives_from_the_wrist
        run, _hub = _drive(bare, picks=1)
        self.assertIs(RunState.FINISHED, run.state, run.error)
        self.assertEqual([{}], bare.calls)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
