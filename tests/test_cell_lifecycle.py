"""A cell comes up and goes down one way, because two callers used to do it two ways.

⛔⛔ **THE DEFECT, MEASURED ON 2026-09-04 BEFORE THIS MODULE EXISTED.** Two implementations of the
same transaction, in two packages, and their teardown halves disagreed:

    api/lifecycle.py:380-406   gripper.disconnect(), then arm.disconnect(), then the lock
    real_cell/__main__.py:185  arm.disconnect()
    real_cell/__main__.py:214  arm.disconnect()

The CLI never disconnected the gripper at all, on either path, having connected it twelve lines
earlier under a banner reading "ACTIVATION MOVES THE FINGERS, keep hands clear". The console's own
comment says why the order matters: a vacuum cup's `disconnect` is what RELEASES ITS OUTPUT, and
doing that while the arm is still up is what makes the release reach the I/O. On the CLI path the
release never happened. The CLI also never closed a camera.

⚠ **AND THE DISAGREEMENT WAS ALREADY WRITTEN DOWN.** `real_cell/__main__.py:182-183` read, four
lines above the arm-only teardown: *"A one-shot CLI got away with that; the console cannot, and the
two must agree."* Somebody saw it, wrote the sentence, and fixed one side. **A comment is not a
mechanism**, which is why the last class in this file is a structural guard rather than a note.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from src.contracts import Rendered, Structured
from src.robot.execution.lifecycle import (
    ConnectedCell,
    ConnectStage,
    StepOutcome,
    TeardownReport,
    connect_cell,
    disconnect_cell,
    release_perception,
)
from src.robot.grippers.null import GripperSubstitution, SubstitutionReason

_REPO = Path(__file__).resolve().parent.parent


class _Recorder:
    """An arm or gripper that writes down what was called on it, in order."""

    def __init__(self, log: list[str], name: str, *, refuse: str | None = None) -> None:
        self.log, self.name, self.refuse = log, name, refuse

    def connect(self) -> None:
        if self.refuse == "connect":
            raise RuntimeError(f"{self.name} refuses to connect")
        self.log.append(f"{self.name}.connect")

    def disconnect(self) -> None:
        if self.refuse == "disconnect":
            raise OSError(f"{self.name} will not release")
        self.log.append(f"{self.name}.disconnect")


def _service(arm: object, gripper: object, perception: object = None) -> object:
    orchestrator = type("O", (), {"arm": arm, "gripper": gripper, "perception": perception})()
    return type("S", (), {"runtime": type("R", (), {"orchestrator": orchestrator})()})()


class TheOrderTests(unittest.TestCase):

    def test_up_is_arm_then_gripper(self) -> None:
        """⛔ A CONTRACT, NOT A STYLE. `VacuumGripper.connect()` drives digital I/O the moment it
        runs, and `from_robot_config` deliberately never connects the arm, so the reverse order
        commands the vacuum line with no arm to command it through."""
        log: list[str] = []
        connect_cell(_Recorder(log, "arm"), _Recorder(log, "gripper"))
        self.assertEqual(log, ["arm.connect", "gripper.connect"])

    def test_down_is_the_exact_reverse(self) -> None:
        """⛔ THE HALF THE CLI DID NOT DO. A vacuum cup's disconnect releases its output, and doing
        that while the arm is still up is what makes the release actually reach the I/O."""
        log: list[str] = []
        disconnect_cell(_Recorder(log, "arm"), _Recorder(log, "gripper"))
        self.assertEqual(log, ["gripper.disconnect", "arm.disconnect"])

    def test_a_gripper_that_refuses_rolls_the_arm_back(self) -> None:
        """⛔ THERE IS NO HALF-CONNECTED STATE WORTH RENDERING, and leaving one means a UR
        controller's single control script is held by a process that has already reported failure."""
        log: list[str] = []
        with self.assertRaises(RuntimeError):
            connect_cell(_Recorder(log, "arm"), _Recorder(log, "gripper", refuse="connect"))
        self.assertEqual(log, ["arm.connect", "arm.disconnect"])

    def test_a_cell_with_no_gripper_still_connects(self) -> None:
        log: list[str] = []
        connect_cell(_Recorder(log, "arm"), None)
        self.assertEqual(log, ["arm.connect"])


class TeardownReportsTests(unittest.TestCase):
    """⛔ THE POINT IS THAT A FAILURE HAS SOMEWHERE TO GO. Teardown must not raise, or it masks the
    result being reported when it runs; but "must not raise" became `except Exception: pass` in the
    CLI, and a gripper that would not release left no trace anywhere."""

    def test_a_refusing_gripper_is_reported_not_swallowed(self) -> None:
        log: list[str] = []
        report = disconnect_cell(_Recorder(log, "arm"), _Recorder(log, "g", refuse="disconnect"))
        self.assertIs(report.gripper, StepOutcome.FAILED)
        self.assertIs(report.arm, StepOutcome.RELEASED, "the arm still comes down")
        self.assertFalse(report.clean)
        self.assertIn("OSError", dict(report.detail)["gripper"])

    def test_teardown_never_raises(self) -> None:
        for refuse_arm, refuse_gripper in ((True, False), (False, True), (True, True)):
            with self.subTest(arm=refuse_arm, gripper=refuse_gripper):
                log: list[str] = []
                report = disconnect_cell(
                    _Recorder(log, "arm", refuse="disconnect" if refuse_arm else None),
                    _Recorder(log, "g", refuse="disconnect" if refuse_gripper else None),
                )
                self.assertIsInstance(report, TeardownReport)

    def test_absent_is_not_failed(self) -> None:
        """A cell with no gripper is normal, not broken. Collapsing the two would make every
        gripperless rehearsal look like a teardown fault."""
        report = disconnect_cell(_Recorder([], "arm"), None)
        self.assertIs(report.gripper, StepOutcome.ABSENT)
        self.assertTrue(report.clean)

    def test_the_warning_line_appears_only_when_something_failed(self) -> None:
        clean = disconnect_cell(_Recorder([], "arm"), None)
        dirty = disconnect_cell(_Recorder([], "arm"), _Recorder([], "g", refuse="disconnect"))
        self.assertNotIn("still asserted", clean.render())
        self.assertIn("still asserted", dirty.render())


class PerceptionTests(unittest.TestCase):
    """⛔ EVERY CAMERA, NOT THE PRIMARY ONE. A fused cell opens one device per camera, and on real
    hardware a second `pipeline.start()` on a streaming device FAILS."""

    def test_a_source_owning_no_device_is_absent_not_an_error(self) -> None:
        self.assertIs(release_perception(None), StepOutcome.ABSENT)
        self.assertIs(release_perception(_service(None, None)), StepOutcome.ABSENT)

    def test_a_closeable_source_is_closed(self) -> None:
        closed: list[str] = []
        holder = type("P", (), {"close": lambda self: closed.append("closed")})()
        self.assertIs(release_perception(_service(None, None, holder)), StepOutcome.RELEASED)
        self.assertEqual(closed, ["closed"])

    def test_a_camera_that_will_not_close_is_reported(self) -> None:
        def _boom(self: object) -> None:
            raise OSError("device busy")

        holder = type("P", (), {"close": _boom})()
        self.assertIs(release_perception(_service(None, None, holder)), StepOutcome.FAILED)


class TheContextManagerTests(unittest.TestCase):
    """⭐ THE VALUE IS THE EXIT, NOT THE ENTRY. Any caller remembers to connect; the paths that get
    missed are after a refusal, after an exception and after a keyboard interrupt."""

    def test_teardown_runs_when_the_block_raises(self) -> None:
        log: list[str] = []
        cell = ConnectedCell(_service(_Recorder(log, "arm"), _Recorder(log, "gripper")))
        with self.assertRaises(ValueError), cell:
            raise ValueError("the pick blew up")
        self.assertEqual(log, ["arm.connect", "gripper.connect",
                               "gripper.disconnect", "arm.disconnect"])
        self.assertIsNotNone(cell.teardown)

    def test_teardown_runs_on_a_keyboard_interrupt(self) -> None:
        """A run of this length is stopped by hand, so this is the likeliest exit of all."""
        log: list[str] = []
        cell = ConnectedCell(_service(_Recorder(log, "arm"), _Recorder(log, "gripper")))
        with self.assertRaises(KeyboardInterrupt), cell:
            raise KeyboardInterrupt
        self.assertEqual(log[-2:], ["gripper.disconnect", "arm.disconnect"])

    def test_a_refused_connect_gives_the_lock_back(self) -> None:
        """⛔ OTHERWISE A REFUSAL HOLDS THE CELL FOREVER. The lock is cross-process, so a leaked one
        is not cleaned up by the failing process exiting a function."""
        released: list[str] = []
        lock = type("L", (), {"acquire": lambda self: None,
                              "release": lambda self: released.append("released")})()
        cell = ConnectedCell(_service(_Recorder([], "arm"), _Recorder([], "g", refuse="connect")),
                             lock=lock)
        with self.assertRaises(RuntimeError):
            cell.__enter__()
        self.assertEqual(released, ["released"])

    def test_the_announce_hook_fires_before_the_gripper_moves(self) -> None:
        """⛔ THE ONLY REASON THE HOOK EXISTS. Robotiq activation is a calibration sweep of the full
        finger travel, so the warning has to reach a person at the bench BEFORE the fingers move."""
        # ONE log for both the hook and the hardware calls, so the interleaving is the assertion
        # rather than two lists that each look right on their own.
        log: list[str] = []
        connect_cell(_Recorder(log, "arm"), _Recorder(log, "gripper"),
                     announce=lambda stage: log.append(f"<{stage.value}>"))
        self.assertEqual(log, [
            "arm.connect",
            f"<{ConnectStage.ARM_CONNECTED.value}>",
            f"<{ConnectStage.GRIPPER_MOVING.value}>",      # the warning reaches the bench...
            "gripper.connect",                             # ...and only THEN do the fingers move
            f"<{ConnectStage.GRIPPER_CONNECTED.value}>",
        ])


class ASubstitutedGripperIsNotACellTests(unittest.TestCase):
    """⛔⛔ **A CELL WITH NO END-EFFECTOR CAME UP AND REPORTED SUCCESS.** MEASURED on this tree,
    before this guard existed: `python -m src.robot.execution.real_cell --rehearse --runs 3` on the
    shipped `robot.yaml` printed `gripper  NullGripper` at BUILD and `RESULT: 3/3 succeeded` at the
    end, exit 0. `gripper.vendor: robotiq` cannot be reached from a non-UR arm, so the build
    substitutes a working `NullGripper`, every commanded width is accepted, `get_width_mm()` answers
    the configured 85.0 mm maximum forever, and nothing downstream disagrees.

    ⚠ **THE VERIFIER REPAIR DOES NOT REACH THE DEFAULT PICK.** `WidthDeltaGripperVerifier` refuses a
    substituted gripper by name, but `grasping.verification.enabled` is `false` in the shipped tree,
    so on the default open-loop attempt NOBODY READS A WIDTH. The fact is knowable before any motion
    at all, so it is answered before any motion at all.

    ⭐ **AND IT IS ANSWERED HERE SO THERE IS ONE ANSWER.** `api/lifecycle.py` already refuses this
    connect (`ConnectRefused.NO_REAL_GRIPPER`) and reaches the hardware through this function, so
    the console's typed refusal is now a rendering of this rule rather than a second copy of it.
    """

    @staticmethod
    def _substituted(log: "list[str]") -> object:
        """What the shipped `robot.yaml` builds on a dummy arm, with the real widths and reason."""
        gripper = _Recorder(log, "gripper")
        gripper.substitution = GripperSubstitution(
            reason=SubstitutionReason.ROBOTIQ_NEEDS_UR,
            requested="robotiq",
            detail=("gripper.vendor='robotiq' but the arm in hand reports vendor 'dummy'. A "
                    "Robotiq lives on the UR controller's tool I/O and cannot be reached from here."),
            fix="On a real cell, set robot.vendor: ur.",
        )
        return gripper

    def test_a_substituted_gripper_refuses_the_connect(self) -> None:
        """⛔ AND NOTHING IS COMMANDED, not even the arm. The arm connect is the first motion of the
        run and this fact was decidable before it."""
        log: list[str] = []
        with self.assertRaises(RuntimeError):
            connect_cell(_Recorder(log, "arm"), self._substituted(log))
        self.assertEqual(log, [], "a cell with no end-effector must command nothing at all")

    def test_the_refusal_says_what_was_asked_for_and_what_to_do(self) -> None:
        """⚠ THE OPERATOR IS THE READER. "no gripper" sends them to the wiring; the substitution
        record already knows it is a config line."""
        with self.assertRaises(RuntimeError) as caught:
            connect_cell(_Recorder([], "arm"), self._substituted([]))
        message = str(caught.exception)
        self.assertIn("robotiq", message, "the refusal must name what was asked for")
        self.assertIn("robot.vendor: ur", message, "the refusal must carry the fix")

    def test_a_cell_that_declares_no_end_effector_still_connects(self) -> None:
        """⛔ THE GUARD IS KEYED ON THE SUBSTITUTION, NOT ON THE ABSENCE OF JAWS. `gripper.vendor:
        none` is the same jawless object with `substitution=None` and is a legitimate cell: a
        calibration rig, a camera-only bring-up. Refusing it would be a different rule."""
        log: list[str] = []
        gripper = _Recorder(log, "gripper")
        gripper.substitution = None
        connect_cell(_Recorder(log, "arm"), gripper)
        self.assertEqual(log, ["arm.connect", "gripper.connect"])

    def test_the_context_manager_refuses_and_gives_the_lock_back(self) -> None:
        """⛔ OTHERWISE THE REFUSAL HOLDS THE CELL. The lock is cross-process, so a leaked one
        outlives the function that leaked it."""
        released: list[str] = []
        lock = type("L", (), {"acquire": lambda self: None,
                              "release": lambda self: released.append("released")})()
        log: list[str] = []
        cell = ConnectedCell(_service(_Recorder(log, "arm"), self._substituted(log)), lock=lock)
        with self.assertRaises(RuntimeError):
            cell.__enter__()
        self.assertEqual(released, ["released"])
        self.assertEqual(log, [])
        self.assertIsNone(cell.teardown, "nothing came up, so nothing came down")


class TheReportContractTests(unittest.TestCase):

    def test_it_is_rendered_and_structured(self) -> None:
        report = disconnect_cell(_Recorder([], "arm"), None)
        self.assertIsInstance(report, Rendered)
        self.assertIsInstance(report, Structured)

    def test_render_obeys_the_contract(self) -> None:
        for report in (disconnect_cell(_Recorder([], "arm"), None),
                       disconnect_cell(_Recorder([], "a"), _Recorder([], "g", refuse="disconnect"))):
            with self.subTest(report.clean):
                text = report.render()
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))

    def test_to_dict_is_json_safe(self) -> None:
        payload = disconnect_cell(_Recorder([], "a"),
                                  _Recorder([], "g", refuse="disconnect")).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)


class NeitherCallerWritesTheOrderTests(unittest.TestCase):
    """⛔⛔ THE STRUCTURAL GUARD, AND THE REASON IT IS ONE. The disagreement this module fixes was
    already documented in a comment saying "the two must agree", four lines above the code that did
    not. A comment cannot fail. This can.

    ⚠ READ FROM THE AST, so the prose above the code explaining what USED to be here does not count
    as the thing itself. A substring search would pass on the defect and fail on the repair.
    """

    CALLERS = (
        "api/lifecycle.py",
        "src/robot/execution/real_cell/__main__.py",
    )

    @staticmethod
    def _direct_calls(path: Path) -> list[str]:
        """`x.connect()` / `x.disconnect()` on an `arm` or `gripper` name."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in ("connect", "disconnect"):
                continue
            target = node.func.value
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
            if name in ("arm", "gripper"):
                found.append(f"{name}.{node.func.attr}()")
        return found

    def test_no_caller_connects_or_disconnects_a_handle_itself(self) -> None:
        offenders = {rel: found for rel in self.CALLERS
                     if (found := self._direct_calls(_REPO / rel))}
        self.assertEqual(offenders, {},
                         "a caller is writing its own connect/disconnect order again; that is how "
                         "the CLI came to leave a gripper energised")

    def test_the_guard_can_see_the_shape_it_forbids(self) -> None:
        """⭐ THE SELF-FAILING CONTROL. The module that legitimately OWNS the order is the positive
        fixture: a scan that found nothing there would be looking in the wrong place."""
        owner = self._direct_calls(_REPO / "src/robot/execution/lifecycle.py")
        self.assertIn("arm.disconnect()", owner)
        self.assertIn("arm.connect()", owner)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
