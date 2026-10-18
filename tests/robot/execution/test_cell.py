"""The sequence was the part nobody could import.

⛔⛔ **EVERY PIECE OF BRINGING A CELL UP WAS ALREADY PUBLIC.** `load_robot_config`,
`run_config_preflight`, `build_real_cell`, `SafetyAttestation.of`, `cell_lock_key`, `ConnectedCell`
and `service.pick()` are all importable and were before this class existed. What was NOT public is
the ORDER, and the order is the part that matters:

    preflight before build      a blocking config item is decidable at a desk and costs a trip later
    attestation before motion   an arm that gates nothing must say so while there is time to stop
    lock before connect         a UR controller accepts ONE control script
    arm before gripper          a vacuum cup's connect drives digital I/O immediately

A caller who wanted a pick from Python had to learn all four, and the only place that knowledge
existed was a CLI's `main()`, in the order its `print` statements happened to fall.

⭐ **THE CLI'S OUTPUT IS BYTE-IDENTICAL ACROSS THE REWIRE**, verified by diffing
`--rehearse --runs 3` before and against after. That is the check that matters for "keep every CLI":
not that the tests pass, but that the bytes an operator reads did not move.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.config import load_robot_config
from src.config.tree import load_tree
from src.contracts import Rendered
from src.robot.execution.cell import Cell, CellNotBuilt
from src.robot.execution.lifecycle import ConnectedCell, NoRealGripper
from src.robot.safety import SafetyPosture


def _config():
    """The repo's own base tree, with no profile, so a shell variable cannot change the answer."""
    return load_robot_config(profile=None)


def _declares_no_end_effector():
    """The same tree with `gripper.vendor: none`, which is what a cell that CAN be rehearsed says.

    ⛔⛔ **AND THE SHIPPED TREE IS NOT ONE, WHICH IS THE MEASUREMENT THIS FILE USED TO MISS.** The
    base tree says `gripper.vendor: robotiq`, a rehearsal moves `robot.vendor` to `dummy`, and a
    Robotiq lives on the UR controller's tool I/O, so the rehearsal's own swap makes the gripper
    unbuildable and the build substitutes a `NullGripper`. Until 2026-09-09 that cell connected and
    picked, and `--rehearse --runs 3` reported `3/3 succeeded` with nothing on the flange.
    `connect_cell` refuses it now, so the sequence tests below need a cell that declares it has no
    end-effector on purpose: `substitution` is `None`, the operator said so, and it connects.
    """
    base = _config()
    return base.model_copy(update={"gripper": base.gripper.model_copy(update={"vendor": "none"})})


class TheOrderIsEnforcedTests(unittest.TestCase):
    """⛔ A TYPED REFUSAL, NOT AN AttributeError THREE FRAMES DOWN. This repository has spent the day
    finding places where a comment described a rule the code did not check."""

    def test_connecting_before_building_is_refused_by_name(self) -> None:
        with self.assertRaises(CellNotBuilt) as caught:
            Cell.rehearsal(_config()).connected()
        self.assertIn("build()", str(caught.exception), "the refusal must name the missing call")

    def test_reaching_for_the_arm_before_building_is_refused(self) -> None:
        cell = Cell.rehearsal(_config())
        for attribute in ("service", "arm", "gripper"):
            with self.subTest(attribute), self.assertRaises(CellNotBuilt):
                getattr(cell, attribute)

    def test_preflight_needs_no_build_and_no_hardware(self) -> None:
        """⭐ THE REASON IT IS THE FIRST STEP. A blocking item found here costs nothing; the same
        item found at a bench costs a trip."""
        report = Cell.rehearsal(_config()).preflight()
        self.assertIsInstance(report, Rendered)
        self.assertTrue(report.render().strip())


class TheStepsTests(unittest.TestCase):

    def test_build_is_idempotent(self) -> None:
        """⚠ A CALLER WHO ASKS TWICE MEANS "make sure it is built", and a second build would open a
        second camera on a device that has one."""
        cell = Cell.rehearsal(_config())
        self.assertIs(cell.build(), cell.build())
        self.assertIs(cell.service, cell.build())

    def test_a_rehearsal_overrides_only_the_vendor(self) -> None:
        """⛔ THE OPERATOR'S OWN CONFIG WITH ONE FIELD CHANGED. A rehearsal against a different tree
        would prove things about a cell nobody owns; against this one it exercises their profile
        chain, their gripper branch and their grasping block."""
        base = _config()
        cell = Cell.rehearsal(base)
        self.assertEqual(cell.vendor, "dummy")
        self.assertNotEqual(str(base.vendor), "dummy", "the caller's config is not mutated")
        self.assertEqual(cell.robot_config.grasping, base.grasping, "only the vendor moved")

    def test_a_rehearsal_reports_that_it_gates_nothing(self) -> None:
        """⛔ A REHEARSAL PROVES WIRING, NEVER SAFETY. The dummy arm carries no preflight and the
        attestation says so, which is the whole reason that class exists."""
        cell = Cell.rehearsal(_config())
        cell.build()
        attestation = cell.safety()
        self.assertIs(attestation.posture, SafetyPosture.UNGATED)
        self.assertFalse(attestation.enforced)

    def test_safety_is_read_off_the_built_arm_not_the_config(self) -> None:
        """⛔ A CONFIG CAN REQUEST A PREFLIGHT THE DRIVER DOES NOT CARRY. An example in this
        repository once printed "cleared safety" while driving an arm that had none."""
        cell = Cell.rehearsal(_config())
        cell.build()
        self.assertEqual(cell.safety().arm, type(cell.arm).__name__)

    def test_connected_yields_a_session_that_tears_down(self) -> None:
        cell = Cell.rehearsal(_declares_no_end_effector())
        cell.build()
        with cell.connected() as live:
            self.assertIsInstance(live, ConnectedCell)
            self.assertIs(live.service, cell.service)
        assert live.teardown is not None
        self.assertTrue(live.teardown.clean)

    def test_a_rehearsal_takes_no_cross_process_lock(self) -> None:
        """⚠ A SIMULATED OR DUMMY CELL OWNS NO CONTROLLER, which is why two rehearsals can run at
        once while two real cells cannot."""
        cell = Cell.rehearsal(_config())
        cell.build()
        self.assertIsNone(cell.connected().lock)

    def test_a_rehearsal_of_the_shipped_tree_cannot_connect(self) -> None:
        """⛔⛔ **THE GAP THIS FILE DOCUMENTED AS A FEATURE.** `test_a_pick_runs_end_to_end_from
        _python` ran this exact cell and asserted `succeeded`, and the CLI above it printed
        `3/3 succeeded`, on a build whose own log line says `Built a NullGripper (no real
        end-effector)`. The rehearsal manufactures that substitution itself, by moving the arm
        vendor under an unchanged `gripper.vendor: robotiq`; the refusal cannot tell a manufactured
        one from an operator's, and it must not, because the cell in front of it has no jaws either
        way."""
        cell = Cell.rehearsal(_config())
        cell.build()
        with self.assertRaises(NoRealGripper) as caught:
            cell.connected().__enter__()
        self.assertIn("close on nothing", str(caught.exception))


class TheWholeSequenceTests(unittest.TestCase):
    """⭐ THE FOUR STEPS, IN THE ORDER, AS A CUSTOMER WOULD WRITE THEM. If this reads badly the class
    is wrong, whatever the unit tests say."""

    def test_a_pick_runs_end_to_end_from_python(self) -> None:
        cell = Cell.rehearsal(_declares_no_end_effector())

        self.assertTrue(cell.preflight().render())          # 1. no hardware
        cell.build()                                        # 2. drivers + grasp stack
        self.assertFalse(cell.safety().enforced)            # 3. what it will refuse
        with cell.connected() as live:                      # 4. lock, arm, then gripper
            report = live.service.pick()

        self.assertEqual(str(report.outcome), "succeeded")
        self.assertEqual(report.layers_that_ran(), (),
                         "the default pick is open-loop and the report says so")
        assert live.teardown is not None
        self.assertTrue(live.teardown.clean)


class TheGraspModeIsChosenAtTheBuildTests(unittest.TestCase):
    """⚠ THE TWO DENSE MODES HAD NO DOOR THROUGH `Cell`, AND THEY ARE WHAT A BIN NEEDS.

    `AutonomousGraspService.from_robot_config` takes `mode=`, and `build_real_cell` forwards
    anything it is given, but `Cell.build()` passed neither, so every cell built through the
    operator-facing noun was `auto`. `pick(mode=...)` is not the same lever: it narrows the
    behaviour of ONE attempt and never the sampler the service was built with, so asking a service
    built in `auto` for `dense_clutter` comes back `MODE_NOT_AVAILABLE` -- correctly, and with no
    way to get what was asked for. Reaching `dense_clutter` meant bypassing `Cell` and calling
    `build_real_cell` by hand, which also gives up the preflight, the lock and the teardown.
    """

    def _mode_reaching(self, **kwargs: object) -> object:
        """What `Cell.build()` hands the rehearsal builder, without building anything."""
        cell = Cell.rehearsal(load_tree("console_dummy").robot, **kwargs)  # type: ignore[arg-type]
        with mock.patch("src.robot.execution.autonomous_grasp.build_rehearsal_cell") as build:
            cell.build()
        return build.call_args.kwargs.get("mode", "NOT FORWARDED")

    def test_a_chosen_mode_reaches_the_builder(self) -> None:
        self.assertEqual("dense_clutter", self._mode_reaching(mode="dense_clutter"))

    def test_no_mode_forwards_nothing_so_the_callee_keeps_its_own_default(self) -> None:
        """A default copied into a wrapper is a second declaration of it, agreeing until it does not."""
        self.assertEqual("NOT FORWARDED", self._mode_reaching())

    def test_every_factory_carries_it(self) -> None:
        robot = load_tree("console_dummy").robot
        for cell in (Cell.rehearsal(robot, mode="easy"),
                     Cell.from_robot_config(robot, mode="easy"),
                     Cell.from_tree(load_tree("console_dummy"), mode="easy")):
            with self.subTest(cell):
                self.assertEqual("easy", cell.mode)

    def test_the_service_is_really_built_in_it(self) -> None:
        """The end of the chain: a cell built in one mode runs that mode, and reports it."""
        from willy import GraspMode

        for mode in (GraspMode.EASY, GraspMode.DENSE_CLUTTER):
            with self.subTest(mode=mode):
                cell = Cell.rehearsal(load_tree("console_dummy").robot, mode=mode)
                service = cell.build()
                with cell.connected():
                    report = service.pick()
                self.assertIs(mode, report.mode)
                self.assertEqual("succeeded", report.outcome.value, report.render())

    def test_a_mode_the_service_was_not_built_for_still_refuses_by_name(self) -> None:
        """The build-time lever does not weaken the per-attempt one: no silent re-sampling."""
        cell = Cell.rehearsal(load_tree("console_dummy").robot, mode="easy")
        service = cell.build()
        with cell.connected():
            report = service.pick(mode="dense_clutter")
        self.assertEqual("mode_not_available", report.outcome.value)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
