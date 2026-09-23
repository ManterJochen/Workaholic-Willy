"""A single toggle counts its own pulses, and the count outlives the program (owner's cell, 2026-09-23).

Measured on the owner's cell and reproduced on a CB3 URSim: a toggle with no open switch flips its jaws on every
pulse, the driver took them to stand open at every connect, and after any program that ended with them closed every
command of the next one was inverted. The pick's pre-open closed the jaws before the part, the close at the part
opened them, and a release closed them. The count is now written down after every pulse and the next program starts
from it; a program that died between "on the way" and "there" leaves the next one refusing until a person declares
where the jaws stand.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.robot.core import RobotError
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.execution import handling
from src.robot.execution.robot import Robot
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grippers import jaw_toggle_state
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_grasp_execution_policy import _FakeArm, _grasp
from tests.test_jaw_io_gripper import CLOSE_PIN, FakeIO
from tests.test_robot_pick_and_place import _Log, _pose, _RecordingArm

_BENCH = "unit double: a bench with no cameras"


def _toggle(io: FakeIO, record: Path | None, **kw: float) -> JawIOGripper:
    widths = {"min_width_mm": 5.0, "max_width_mm": 49.99, "closed_below_mm": 5.0, **kw}
    return JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=0.0, close_settle_s=0.0,
                        state_file=record, sleep=lambda _s: None, **widths)


def _device_closed(io: FakeIO, *, started_closed: bool = False) -> bool:
    """The jaws as the device sees them: one flip per rising edge on the toggle's pin."""
    closed, level = started_closed, False
    for pin, high, _port in io.writes:
        if pin != CLOSE_PIN:
            continue
        if high and not level:
            closed = not closed
        level = high
    return closed


def _rises(io: FakeIO) -> int:
    level, rises = False, 0
    for pin, high, _port in io.writes:
        if pin == CLOSE_PIN:
            rises += int(high and not level)
            level = high
    return rises


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.record = Path(folder.name) / "jaw_toggle_test.json"


class TheCountIsWrittenTests(_Tmp):
    def test_every_pulse_leaves_where_the_jaws_stand(self) -> None:
        g = _toggle(FakeIO(), self.record)
        g.connect()
        g.set_closed(True)
        self.assertEqual({"closed": True, "in_flight": False}, {k: v for k, v in json.loads(
            self.record.read_text(encoding="utf-8")).items() if k in ("closed", "in_flight")})
        g.set_closed(False)
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertEqual((False, False), (loaded.closed, loaded.in_flight))

    def test_no_record_is_today_s_start_open(self) -> None:
        io = FakeIO()
        g = _toggle(io, self.record)
        g.connect()
        self.assertFalse(g.jaws_closed)
        self.assertFalse(self.record.exists(), "a connect with nothing to go on writes no claim")
        self.assertEqual(0, _rises(io))


class TheNextProgramStartsFromItTests(_Tmp):
    def test_jaws_left_closed_are_opened_by_one_pulse_in_the_next_program(self) -> None:
        io = FakeIO()                                   # one controller, two programs
        first = _toggle(io, self.record)
        first.connect()
        first.set_closed(True)
        first.disconnect()                              # a pick with no place: the jaws stay closed
        second = _toggle(io, self.record)
        with self.assertLogs(second.logger, "WARNING") as said:
            second.connect()
        self.assertTrue(second.jaws_closed)
        self.assertIn("CLOSED", "\n".join(said.output))
        second.set_closed(False)
        self.assertFalse(_device_closed(io), "the next program's open opens them")
        self.assertEqual(2, _rises(io))

    def test_the_owners_pick_after_a_program_that_ended_closed(self) -> None:
        """⛔ The owner's symptom: the pre-open closed the jaws before the part, and the grasp opened them."""
        io = FakeIO()
        log = _Log()
        arm = _RecordingArm(log)
        arm.connect()
        first = _toggle(io, self.record)
        first.connect()
        picked = Robot.from_parts(arm=arm, gripper=first, lock_key=None).pick(_pose(), 40.0, decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.EXECUTED, picked.outcome, picked.render())
        first.disconnect()                              # the program ends holding: no place ran
        self.assertTrue(_device_closed(io))
        second = _toggle(io, self.record)
        second.connect()
        before = _rises(io)
        again = Robot.from_parts(arm=arm, gripper=second, lock_key=None).pick(_pose(), 40.0, decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.EXECUTED, again.outcome, again.render())
        self.assertEqual(2, _rises(io) - before, "one flip to open before the part, one to close at it")
        self.assertTrue(_device_closed(io), "closed at the part, as the grasp meant")

    def test_a_program_that_died_mid_pulse_leaves_the_next_refusing_until_declared(self) -> None:
        class _HighFails(FakeIO):
            def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
                if value:
                    raise RuntimeError("the link dropped on the high write")
                super().set_digital_output(pin, value, port=port)

        dying = _toggle(_HighFails(), self.record)
        dying.connect()
        with self.assertRaises(RuntimeError):
            dying.set_closed(True)
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None and loaded.in_flight
        io = FakeIO()
        after = _toggle(io, self.record)
        after.connect()
        with self.assertRaises(RobotError) as refused:
            after.set_closed(True)
        self.assertIn("declare", str(refused.exception))
        self.assertEqual(0, _rises(io))
        after.declare_jaws(closed=False)
        after.set_closed(True)
        self.assertEqual(1, _rises(io))

    def test_a_record_that_does_not_parse_is_a_pulse_in_flight(self) -> None:
        self.record.write_text("{ not json", encoding="utf-8")
        g = _toggle(FakeIO(), self.record)
        g.connect()
        with self.assertRaises(RobotError):
            g.set_closed(True)

    def test_declaring_is_refused_where_the_driver_does_not_count(self) -> None:
        solenoid = JawIOGripper(FakeIO(), actuation="single_solenoid", close_output_pin=CLOSE_PIN,
                                sleep=lambda _s: None)
        with self.assertRaises(RobotError):
            solenoid.declare_jaws(closed=False)


class AnOpenWaitsTheStrokeTests(unittest.TestCase):
    """⛔ The owner's Hand-E takes up to about 2 s to open, and a place backed out the moment the valve was told."""

    def _gripper(self, actuation: str, slept: list[float]) -> JawIOGripper:
        g = JawIOGripper(FakeIO(), actuation=actuation, close_output_pin=CLOSE_PIN, pulse_s=0.0, close_settle_s=2.0,
                         sleep=slept.append, min_width_mm=5.0, max_width_mm=49.99)
        g.connect()
        return g

    def test_an_open_after_a_close_waits_the_travel_time(self) -> None:
        for actuation in ("single_toggle", "single_solenoid"):
            with self.subTest(actuation=actuation):
                slept: list[float] = []
                g = self._gripper(actuation, slept)
                g.set_closed(True)
                slept.clear()
                g.set_closed(False)
                self.assertIn(2.0, slept, "the open waited the stroke")

    def test_an_open_on_open_jaws_waits_nothing(self) -> None:
        slept: list[float] = []
        g = self._gripper("single_toggle", slept)
        g.set_closed(False)
        self.assertNotIn(2.0, slept)


class TheDeskAsksForTheStrokeTests(unittest.TestCase):
    """⛔ Review 4: an open with no open switch waits close_settle_s, still the 0.3 s schema default unless measured, so a
    release returned 0.5 s after the edge on a Hand-E that takes about 2 s, and nothing at the desk asked for the time."""

    def _row(self, **jaw_io: object) -> object:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.real_cell.preflight import run_config_preflight

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "ik"},
            "gripper": {"vendor": "jaw_io", "jaw_io": {"close_output_pin": 0, **jaw_io}},
            "safety": {"self_collision": {"backend": "capsule"}},
        })
        rows = {check.name: check for check in run_config_preflight(cfg, curobo_available=True,
                                                                    collision_engine="coal").checks}
        return rows.get("jaw travel time")

    def test_the_owners_toggle_at_the_default_settle_is_warned(self) -> None:
        row = self._row(actuation="single_toggle")
        assert row is not None
        self.assertEqual("warn", str(getattr(row, "status")))
        self.assertIn("both ways", getattr(row, "detail"))
        self.assertIn("close_settle_s", getattr(row, "fix"))
        self.assertIn("2 s", getattr(row, "fix"))

    def test_any_jaw_io_with_no_feedback_at_the_default_settle_is_warned(self) -> None:
        for actuation in ("single_solenoid", "double_solenoid"):
            with self.subTest(actuation=actuation):
                extra = {"open_output_pin": 1} if actuation == "double_solenoid" else {}
                row = self._row(actuation=actuation, **extra)
                assert row is not None
                self.assertEqual("warn", str(getattr(row, "status")))

    def test_a_measured_settle_reads_ok(self) -> None:
        """⭐ THE CONTROL: the owner's layer sets 2.0 s."""
        row = self._row(actuation="single_toggle", close_settle_s=2.0)
        assert row is not None
        self.assertEqual("ok", str(getattr(row, "status")))
        self.assertIn("2.0 s", getattr(row, "detail"))

    def test_a_solenoid_with_both_switches_is_not_warned(self) -> None:
        row = self._row(actuation="single_solenoid", closed_confirm_input_pin=1, open_confirm_input_pin=2)
        self.assertNotEqual("warn", str(getattr(row, "status", "")))


class ThePickLoopSaysWhatItMeansTests(_Tmp):
    def test_the_loop_closes_at_the_part_whatever_closed_below_mm_says(self) -> None:
        """⛔ URSim, the owner's hand: with closed_below_mm 5 the loop's close at 39 mm was an open, and it reported
        object_not_detected on jaws it never closed."""
        io = FakeIO()
        g = _toggle(io, self.record, closed_below_mm=5.0)
        g.connect()
        report = GraspExecutionPolicy(arm=_FakeArm(), gripper=g, pre_open_width_mm=49.99).execute(_grasp())
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome)
        self.assertEqual(1, _rises(io), "no pulse at the pre-open, one at the part")
        self.assertTrue(_device_closed(io))


class TheBenchKeepsTheCountHonestTests(_Tmp):
    def _toggle_record(self) -> tuple[Path, str, int]:
        return self.record, "tool", 0

    def test_a_person_declares_where_the_jaws_stand(self) -> None:
        from src.robot.drivers.ur.__main__ import _declare_jaws

        jaw_toggle_state.record_pulse_started(self.record, toward_closed=True, by="test")
        code = _declare_jaws(SimpleNamespace(jaws_stand="open", yes=True), self._toggle_record())
        self.assertEqual(0, code)
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertEqual((False, False), (loaded.closed, loaded.in_flight))

    def test_a_bench_pulse_on_the_toggle_pin_demands_a_declaration(self) -> None:
        from src.robot.drivers.ur.__main__ import _uncount_a_toggle_pulse

        jaw_toggle_state.record_stands(self.record, closed=False, by="test")
        _uncount_a_toggle_pulse(SimpleNamespace(pulse=0, set=None, measure=None), DigitalIOPort.TOOL,
                                self._toggle_record())
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertTrue(loaded.in_flight)

    def test_a_bench_pulse_on_another_pin_leaves_the_count(self) -> None:
        from src.robot.drivers.ur.__main__ import _uncount_a_toggle_pulse

        jaw_toggle_state.record_stands(self.record, closed=True, by="test")
        _uncount_a_toggle_pulse(SimpleNamespace(pulse=4, set=None, measure=None), DigitalIOPort.TOOL,
                                self._toggle_record())
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertEqual((True, False), (loaded.closed, loaded.in_flight))


class _BenchArm(FakeIO):
    """What ``create_arm`` hands the bench: the controller's digital I/O, and a connect."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


class TheBenchMarksTheRecordBeforeTheEdgeTests(unittest.TestCase):
    """⛔ Review 4: ``--measure`` on the toggle's pin, and a ``--pulse``/``--set`` that was interrupted or raised, flipped
    the jaws and left the record saying where they no longer stood; the next program ran inverted and nothing refused."""

    def setUp(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.robot_parts import jaw_toggle_record_path

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        state = patch.dict(os.environ, {"WILLY_JAW_STATE_DIR": folder.name})
        state.start()
        self.addCleanup(state.stop)
        self.cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"ip": "192.168.1.100"},
            "gripper": {"vendor": "jaw_io", "jaw_io": {"actuation": "single_toggle", "close_output_pin": CLOSE_PIN,
                                                       "io_port": "tool"}},
        })
        self.record = jaw_toggle_record_path(self.cfg)
        jaw_toggle_state.record_stands(self.record, closed=False, by="the last program ended open")
        self.arm = _BenchArm()

    def _bench(self, *argv: str) -> int:
        from src.robot.drivers.ur import __main__ as bench_cli

        with patch.object(bench_cli, "_load_robot_config", lambda *_a: self.cfg), \
                patch("src.robot.drivers.create_arm", lambda *_a, **_k: self.arm):
            return bench_cli.main(list(argv))

    def _the_next_program_refuses(self) -> None:
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertTrue(loaded.in_flight, loaded.describe())
        io = FakeIO()
        after = _toggle(io, self.record)
        after.connect()
        with self.assertRaises(RobotError):
            after.set_closed(False)
        self.assertEqual(0, _rises(io))

    def test_a_measure_on_the_toggle_pin_demands_a_declaration(self) -> None:
        self._bench("--measure", f"{CLOSE_PIN}=1", "--watch", "0", "--for", "0.01", "--yes")
        self.assertTrue(_device_closed(self.arm), "the measure closed the jaws")
        self._the_next_program_refuses()

    def test_a_pulse_interrupted_mid_stroke_demands_a_declaration(self) -> None:
        from src.robot.drivers.ur import io_bench

        def ctrl_c(_s: float) -> None:
            raise KeyboardInterrupt

        with patch.dict(io_bench.pulse_output.__kwdefaults__, {"sleep": ctrl_c}):
            self.assertEqual(3, self._bench("--pulse", str(CLOSE_PIN), "--for", "3", "--yes"))
        self.assertTrue(_device_closed(self.arm))
        self._the_next_program_refuses()

    def test_a_set_that_raises_after_the_edge_demands_a_declaration(self) -> None:
        def link_dropped(*_a: object, **_k: object) -> bool:
            raise RuntimeError("the RTDE link dropped on the read-back")

        self.arm.get_digital_output = link_dropped  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            self._bench("--set", f"{CLOSE_PIN}=1", "--yes")
        self.assertTrue(_device_closed(self.arm))
        self._the_next_program_refuses()

    def test_a_refused_confirmation_leaves_the_count(self) -> None:
        from src.robot.drivers.ur import __main__ as bench_cli

        with patch.object(bench_cli, "_confirm", lambda *_a: False):
            self.assertEqual(1, self._bench("--pulse", str(CLOSE_PIN)))
        self.assertEqual([], self.arm.writes, "nothing was energised")
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertEqual((False, False), (loaded.closed, loaded.in_flight))

    def test_a_measure_on_another_pin_leaves_the_count(self) -> None:
        self._bench("--measure", f"{CLOSE_PIN + 1}=1", "--watch", "0", "--for", "0.01", "--yes")
        loaded = jaw_toggle_state.load(self.record)
        assert loaded is not None
        self.assertEqual((False, False), (loaded.closed, loaded.in_flight))


class AnExampleThatNeverPlacesOpensBeforeItLooksTests(unittest.TestCase):
    """⛔ Review 4: example 11 picks and never places. With the record, its next run carried the part to the viewing
    pose unmodelled, and the pick's pre-open dropped it there. It opens where the last run left the arm, as 13 does."""

    def test_example_11_releases_before_it_moves_to_look(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "real_robot" / "11_locate_and_pick.py"
        calls = sorted((node.lineno, node.func.attr) for node in ast.walk(ast.parse(example.read_text(encoding="utf-8")))
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr in ("release", "move"))
        self.assertEqual(["release", "move"], [name for _line, name in calls])


class TheRecordIsKeyedByTheControllerTests(unittest.TestCase):
    def test_one_file_per_controller_bank_and_pin_under_the_state_dir(self) -> None:
        from src.robot.execution.robot_parts import jaw_toggle_record_path

        cfg = SimpleNamespace(vendor="ur", ur=SimpleNamespace(ip="10.0.0.5"), kuka=SimpleNamespace(controller_ip=""),
                              gripper=SimpleNamespace(jaw_io=SimpleNamespace(io_port="tool", close_output_pin=0)))
        path = jaw_toggle_record_path(cfg)  # type: ignore[arg-type]
        self.assertEqual(jaw_toggle_state.state_dir(), path.parent,
                         "the record lives where the driver reads it (tests/conftest.py isolates it under pytest)")
        self.assertEqual("jaw_toggle_ur-10.0.0.5_tool_0.json", path.name)


if __name__ == "__main__":
    unittest.main()
