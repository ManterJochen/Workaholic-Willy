"""A Robotiq hand opens only as wide as a grasp needs, and the guard holds its fingers there (lever G, 2026-10-13).

The owner's "Option 1 aber nur wenn Robotiq genutzt wird": the jaws open to the span and 7 mm beside each jaw
(``support_footprint.NARROW_ROOM_MM``; 2 mm until the owner's Robotiq came down onto the parts, 2026-10-14), not to
the whole 49.99 mm stroke, and only on a Robotiq cell. Pinned on the owner's cell (``tests/_cell_2026_10_01``, the Hand-E on the UR10, its exact guard):

* only a Robotiq cell plans so (``calculator_factory.opens_as_needed``);
* the guard told a width holds both fingers moved in by half of what the stroke is wider, and fully open again when told
  ``None``; a width over the stroke stands them as baked;
* a box where the fully open finger stands refuses the hand fully open and not the hand opened to 20 mm; a box where the
  20 mm finger stands refuses the hand opened to 20 mm and not the hand fully open: the guard judges the fingers where
  the hand was told to put them, both ways;
* the space between the jaws the camera world leaves out at a goal is as wide as the jaws are open
  (``self_envelope.goal_keep_out``): what stands beside the narrow fingers stays in the world;
* the execution opens to the grasp's planned width, tells the guard the wider of that and what the hand reads back, and
  tells it ``None`` again when the pick's motions end, whatever they did (``execution_policy``).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from tests._cell_2026_10_01 import owner_cell

_STROKE_MM = 49.99
#: The grasp the tests stand the hand at: straight down over the owner's mat, closing along BASE x.
_GOAL = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, -650.0], [0.0, 0.0, -1.0, 120.0], [0.0, 0.0, 0.0, 1.0]])
_NEAR = np.radians([-90.0, -100.0, -110.0, -60.0, 90.0, 0.0])


def _box(centre: Any, half: float = 2.0) -> Any:
    from src.robot.safety._capsule import AxisAlignedBox

    return AxisAlignedBox(center_mm=np.asarray(centre, dtype=np.float64), half_extents_mm=np.full(3, half),
                          name="probe")


class OnlyARobotiqCellPlansSo(unittest.TestCase):
    def test_the_vendor_decides(self) -> None:
        from src.robot.grasping.calculator_factory import opens_as_needed

        self.assertTrue(opens_as_needed(SimpleNamespace(gripper=SimpleNamespace(vendor="robotiq"))))
        self.assertTrue(opens_as_needed(SimpleNamespace(gripper=SimpleNamespace(vendor=" Robotiq "))))
        for vendor in ("jaw_io", "onrobot", "dummy", "none", ""):
            self.assertFalse(opens_as_needed(SimpleNamespace(gripper=SimpleNamespace(vendor=vendor))))


class TheGuardHoldsTheFingersWhereTheyStand(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cell = owner_cell()
        if not cls.cell.has_the_engine():
            raise unittest.SkipTest("the exact mesh engine is not on this machine")
        cls.joints = cls.cell.ik_tcp(_GOAL, _NEAR)
        cls.tcp = cls.cell.tcp(cls.joints)

    def tearDown(self) -> None:
        self.cell.preflight.set_finger_opening(None)

    def test_the_fingers_move_in_by_half_of_what_the_stroke_is_wider(self) -> None:
        backend: Any = self.cell.guard._exact_mesh_backend("ur10")  # noqa: SLF001
        before = {name: np.array(backend._sph_c[name]) for name in ("lfinger", "rfinger")}  # noqa: SLF001
        self.cell.preflight.set_finger_opening(20.0)
        moved = [float(np.linalg.norm(backend._sph_c[name] - before[name])) for name in before]  # noqa: SLF001
        for distance in moved:
            self.assertAlmostEqual(distance, (_STROKE_MM - 20.0) / 2.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(
            (backend._sph_c["lfinger"] - backend._sph_c["rfinger"]) - (before["lfinger"] - before["rfinger"]))),  # noqa: SLF001
            _STROKE_MM - 20.0, places=6)
        self.cell.preflight.set_finger_opening(None)
        for name in before:
            np.testing.assert_allclose(backend._sph_c[name], before[name])  # noqa: SLF001
        self.cell.preflight.set_finger_opening(80.0)
        for name in before:
            np.testing.assert_allclose(backend._sph_c[name], before[name])  # noqa: SLF001

    def test_a_box_where_the_open_finger_stands_and_one_where_the_narrow_finger_stands(self) -> None:
        tcp = self.tcp
        closing = tcp[:3, 0]
        # The open finger's middle, 25 mm and half its 10.8 mm off the TCP along the closing axis, and the 20 mm finger's.
        wide = tcp[:3, 3] - (_STROKE_MM / 2.0 + 5.4) * closing
        narrow = tcp[:3, 3] - (20.0 / 2.0 + 5.4) * closing
        self.cell.preflight.set_finger_opening(None)
        self.assertTrue(self.cell.refused([_box(wide)], self.joints))
        self.assertFalse(self.cell.refused([_box(narrow)], self.joints))
        self.cell.preflight.set_finger_opening(20.0)
        self.assertFalse(self.cell.refused([_box(wide)], self.joints))
        self.assertTrue(self.cell.refused([_box(narrow)], self.joints))

    def test_the_space_between_the_jaws_is_as_wide_as_they_are_open(self) -> None:
        self.cell.preflight.set_finger_opening(None)
        full = self.cell.goal_keep_out(self.tcp).region
        self.cell.preflight.set_finger_opening(20.0)
        told = self.cell.goal_keep_out(self.tcp).region
        self.assertAlmostEqual(float(full.half_extents_mm[0]), _STROKE_MM / 2.0, places=6)
        self.assertAlmostEqual(float(told.half_extents_mm[0]), 10.0, places=6)
        beside = self.tcp[:3, 3] - 15.0 * self.tcp[:3, 0]
        self.assertTrue(bool(full.contains(beside[None])[0]))
        self.assertFalse(bool(told.contains(beside[None])[0]))


class TheExecutionTellsTheGuard(unittest.TestCase):
    """The policy against a hand and an arm that only record what they are told."""

    class _Hand:
        def __init__(self, reads: float) -> None:
            self.max_width_mm, self.min_width_mm = _STROKE_MM, 0.0
            self.reads = reads
            self.widths: list[float] = []

        def set_width_mm(self, width_mm: float, **_: Any) -> None:
            self.widths.append(float(width_mm))

        def get_width_mm(self) -> float:
            return self.reads

    def test_open_to_the_plan_tell_the_wider_reading_then_none(self) -> None:
        from src.robot.core.gripper import FINGER_OPENING_SLACK_MM, guard_holds_the_fingers_at
        from src.robot.grasping.motion import execution_policy as policy

        told: list[Any] = []
        arm = SimpleNamespace(set_finger_opening=lambda width: told.append(width))
        hand = self._Hand(reads=24.3)
        grasp = SimpleNamespace(metadata={"score_components": {"support_footprint": {"open_width_mm": 24.0}}})
        self.assertEqual(policy._planned_opening(grasp), 24.0)  # noqa: SLF001
        self.assertTrue(guard_holds_the_fingers_at(arm, hand, 24.0))
        self.assertAlmostEqual(told[-1], 24.3 + FINGER_OPENING_SLACK_MM)
        with policy._fingers_as_planned(arm, grasp):  # noqa: SLF001
            self.assertAlmostEqual(told[-1], 24.0 + FINGER_OPENING_SLACK_MM)
        self.assertIsNone(told[-1])
        unplanned = SimpleNamespace(metadata={"score_components": {"support_footprint": {"stage": "prism"}}})
        self.assertIsNone(policy._planned_opening(unplanned))  # noqa: SLF001
        count = len(told)
        with policy._fingers_as_planned(arm, unplanned):  # noqa: SLF001
            pass
        self.assertEqual(len(told), count)


if __name__ == "__main__":
    unittest.main()
