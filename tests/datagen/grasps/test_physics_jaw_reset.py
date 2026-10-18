"""The jaw reset that stopped one exploded trial from poisoning every trial behind it.

MEASURED 2026-08-24. Replaying the same eight trials in the same order, before and after:

    before   2 scored, 5 refused `the jaw joint blew up`, the joint value DECAYING across them
             (9.96e8 -> 9.23e8 rad) as the position drive pulled at an articulation that had left
             physics and never once got back into range.
    after    6 scored, 1 refused -- and the trial sitting directly behind the surviving explosion
             (bin_000153/4) went from refusing to HOLDING.

Over 144 trials that is 27.8 % -> 91.7 % of draws that produce a measurement at all, and it is what
turned the physics into something that agrees with the analytic reference: `valid` holds 91.7 % of the
time and `rejected:too_wide` 9.1 %.

The 2026-08-13 physics sample ran with the defect: 26 of its 36 trials were poisoned. Its report was
honest -- refusals are counted apart from trials and excluded from every rate -- so it published ten
trials, not thirty-six. The JSONL rows are the trap, because a refused row still carries `held=false`.

No Isaac here. `PhysicsCell.__init__` touches no vendor SDK, so a fake articulation is enough to hold
the contract: a blown-up joint is REFUSED before the approach rather than discovered after the close,
and the reset writes the captured rest DOFs back by teleport rather than by command.
"""

from __future__ import annotations

import unittest

import numpy as np

from datagen.grasps.physics_isaac import PhysicsCell


class FakeArticulation:
    """Records what was written to it, and reports whatever it was last given."""

    def __init__(self, positions: list[float]) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities: np.ndarray | None = None
        self.actions: list[np.ndarray] = []
        self.dof_names = [f"j{i}" for i in range(len(positions))]

    def get_joint_positions(self) -> np.ndarray:
        return self.positions

    def set_joint_positions(self, values: np.ndarray) -> None:
        self.positions = np.asarray(values, dtype=np.float64).copy()

    def set_joint_velocities(self, values: np.ndarray) -> None:
        self.velocities = np.asarray(values, dtype=np.float64).copy()


class JawSanityTests(unittest.TestCase):
    def _cell(self, positions: list[float]) -> tuple[PhysicsCell, FakeArticulation]:
        cell = PhysicsCell()
        gripper = FakeArticulation(positions)
        cell._gripper = gripper
        cell._joint_index = 0
        return cell, gripper

    def test_a_joint_in_range_is_sane(self) -> None:
        cell, _ = self._cell([0.44, 0.0])
        self.assertEqual(cell.jaw_fault(), "")

    def test_the_blown_up_value_from_the_poisoned_run_is_refused(self) -> None:
        # The literal value the 2026-08-13 rows carried.
        cell, _ = self._cell([9.59e12, 0.0])
        self.assertIn("blew up", cell.jaw_fault())

    def test_a_non_finite_joint_is_refused(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                cell, _ = self._cell([float(value), 0.0])
                self.assertIn("blew up", cell.jaw_fault())

    def test_no_gripper_is_not_an_insanity(self) -> None:
        # Before a session is built there is nothing to judge, and reporting a refusal there would
        # refuse every trial of a cell that is merely not started yet.
        self.assertEqual(PhysicsCell().jaw_fault(), "")


class JawResetTests(unittest.TestCase):
    def test_the_reset_teleports_the_captured_rest_dofs_back(self) -> None:
        cell = PhysicsCell()
        gripper = FakeArticulation([9.96e8, 0.0, 0.0])
        cell._gripper = gripper
        cell._joint_index = 0
        cell._rest_dofs = np.array([0.44, 0.1, -0.1])
        cell._drive_joint = lambda radians, *, steps: gripper.actions.append(  # type: ignore[method-assign]
            np.array([radians, steps], dtype=np.float64))

        cell._reset_jaw()

        np.testing.assert_allclose(gripper.positions, [0.44, 0.1, -0.1])
        self.assertEqual(cell.jaw_fault(), "")

    def test_the_reset_zeroes_velocity_before_it_moves_anything(self) -> None:
        # A joint teleported into place while still carrying 1e9 rad/s is back out of range on the
        # next step, which would make the reset look like it worked and change nothing.
        cell = PhysicsCell()
        gripper = FakeArticulation([9.96e8, 0.0])
        cell._gripper = gripper
        cell._rest_dofs = np.array([0.44, 0.0])
        cell._drive_joint = lambda radians, *, steps: None  # type: ignore[method-assign]

        cell._reset_jaw()

        assert gripper.velocities is not None
        np.testing.assert_allclose(gripper.velocities, [0.0, 0.0])

    def test_the_reset_re_commands_the_drive(self) -> None:
        # Without this the next `apply_action` re-asserts the pre-explosion target the moment physics
        # steps, and the teleport is undone before anything can observe it.
        cell = PhysicsCell()
        gripper = FakeArticulation([9.96e8, 0.0])
        cell._gripper = gripper
        cell._rest_dofs = np.array([0.44, 0.0])
        commanded: list[float] = []
        cell._drive_joint = lambda radians, *, steps: commanded.append(radians)  # type: ignore[method-assign]

        cell._reset_jaw()

        self.assertEqual(commanded, [0.44])

    def test_a_cell_with_no_captured_rest_state_does_nothing(self) -> None:
        cell = PhysicsCell()
        gripper = FakeArticulation([9.96e8, 0.0])
        cell._gripper = gripper
        cell._reset_jaw()          # _rest_dofs is None -- never initialised
        np.testing.assert_allclose(gripper.positions, [9.96e8, 0.0])


class RestoreCallsTheResetTests(unittest.TestCase):
    def test_restore_resets_the_jaw_before_it_puts_objects_back(self) -> None:
        """The ordering is the fix. Teleporting an object into a broken gripper is not a restore."""
        cell = PhysicsCell()
        order: list[str] = []
        cell._park = lambda: order.append("park")                      # type: ignore[method-assign]
        cell._reset_jaw = lambda: order.append("reset_jaw")            # type: ignore[method-assign]
        cell._set_table_z = lambda z: order.append("table")            # type: ignore[method-assign]
        cell._objects = None                                            # restore returns after the table

        cell.restore(object())

        self.assertEqual(order, ["park", "reset_jaw", "table"])


if __name__ == "__main__":
    unittest.main()
