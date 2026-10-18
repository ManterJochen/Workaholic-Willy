"""Native physics shake-labeller — the Isaac-free held decision + the injected-callable orchestration.

No Isaac. A stateful fake 'world' drives ``run_shake_test``: a held object keeps a constant pose in the
gripper frame; a dropped object drifts out of it after the kicks.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.willy_sim.shake_label import (
    ShakeConfig,
    evaluate_held,
    object_in_gripper_frame,
    run_shake_test,
)


class GripperFrameTests(unittest.TestCase):
    def test_identity_frame_is_translation(self) -> None:
        rel = object_in_gripper_frame(
            np.array([10.0, 0.0, 0.0]), np.eye(3), np.array([13.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(rel, [3.0, 0.0, 0.0])

    def test_rotated_frame(self) -> None:
        # gripper rotated 90deg about z: world +x maps to gripper -y
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rel = object_in_gripper_frame(np.zeros(3), rot, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rel, [0.0, -1.0, 0.0], atol=1e-9)


class EvaluateHeldTests(unittest.TestCase):
    def test_held_when_within_threshold(self) -> None:
        seq = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0])]
        held, max_disp = evaluate_held(seq, threshold_mm=25.0)
        self.assertTrue(held)
        self.assertAlmostEqual(max_disp, 3.0)

    def test_dropped_when_exceeds(self) -> None:
        seq = [np.zeros(3), np.array([5.0, 0.0, 0.0]), np.array([40.0, 0.0, 0.0])]
        held, max_disp = evaluate_held(seq, threshold_mm=25.0)
        self.assertFalse(held)
        self.assertAlmostEqual(max_disp, 40.0)

    def test_empty_sequence_is_not_held(self) -> None:
        held, _ = evaluate_held([], threshold_mm=25.0)
        self.assertFalse(held)


class _FakeWorld:
    """Gripper fixed at origin (identity). The object sits at ``offset`` in the gripper frame; when
    ``drop_after`` kicks have landed it drifts away (a dropped grasp)."""

    def __init__(self, *, drop: bool) -> None:
        self._drop = drop
        self._kicks = 0
        self._drift = 0.0

    def step(self, n: int) -> None:
        if self._drop and self._kicks >= 3:
            self._drift += 30.0  # object slides out once shaken

    def gripper_pose(self):
        return np.zeros(3), np.eye(3)

    def object_position(self):
        return np.array([0.0, 0.0, 5.0 + self._drift])

    def kick(self, vel: np.ndarray) -> None:
        self._kicks += 1

    def set_gravity_scale(self, s: float) -> None:
        pass


class RunShakeTestTests(unittest.TestCase):
    def _run(self, drop: bool):
        w = _FakeWorld(drop=drop)
        return run_shake_test(
            step=w.step,
            gripper_pose=w.gripper_pose,
            object_position=w.object_position,
            kick=w.kick,
            set_gravity_scale=w.set_gravity_scale,
            config=ShakeConfig(held_displacement_threshold_mm=25.0),
        )

    def test_held_object_is_labelled_held(self) -> None:
        result = self._run(drop=False)
        self.assertTrue(result.held)
        self.assertEqual(result.failed_phase, "")
        self.assertLess(result.max_displacement_mm, 25.0)

    def test_dropped_object_is_labelled_dropped(self) -> None:
        result = self._run(drop=True)
        self.assertFalse(result.held)
        self.assertTrue(result.failed_phase.startswith("kick_") or result.failed_phase in ("overload", "final_still"))
        self.assertGreater(result.max_displacement_mm, 25.0)


if __name__ == "__main__":
    unittest.main()
