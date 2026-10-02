"""The exact guard holds the bench, and the small things on it (fix plan Track S, test 5; design finding 1, attack N3).

The guard's declared list is ``self_collision.fixtures``, and the owner's is empty: the bench slab went to the planner
alone, so over the bare table the guard held nothing, and only the workspace guard's TCP floor kept a hand off it. A side
grasp's palm hangs lower than its TCP, and side grasps are on. Now the camera world holds the declared bench as an
upright solid up to its band and the 2 mm allowance, which the guard judges like every box the camera saw. Small cubes
on the bare table, which the noise rule dropped (the hand went down to the table, -1.0 mm), stand on that solid and are
held (F1).

"The mat is gone tomorrow": the owner's camera at LOOK[0] (P1) sees a level bare table at the declared height.
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from typing import Any

import numpy as np

from src.robot.safety.planning.live_world import _guard_boxes
from tests import _cell_2026_10_01 as cell

_R3_DEG = (-87.9, -94.2, -129.2, -46.5, 89.7, -288.9)
_ARM = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")


@lru_cache(maxsize=1)
def _hand_parts() -> Any:
    from src.robot.safety._fcl_self_collision import composed_parts
    from src.robot.safety.planning.hand import planner_hand

    hand: Any = planner_hand(cell.owner_cell().config)
    parts = composed_parts("ur10", None, hand.guard_variant, hand.coupling_mm, placement=hand.placement,
                           coupling_boxes=hand.coupling_boxes)
    return {name: part for name, part in parts.items() if name not in _ARM}


def _lowest(joints: np.ndarray) -> "tuple[float, str]":
    """The lowest point of the hand at ``joints``, BASE z, and the part it lies on."""
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm

    frames = ur_link_transforms_mm("ur10", joints)
    assert frames is not None
    low, name = math.inf, ""
    for part, (vertices, _, frame) in _hand_parts().items():
        z = float((vertices @ np.asarray(frames[frame])[:3, :3].T + np.asarray(frames[frame])[:3, 3])[:, 2].min())
        if z < low:
            low, name = z, part
    return low, name


def _hand_over_the_bench(height_mm: float, tilt_deg: float, at: "tuple[float, float]" = (-130.0, -690.0)) -> np.ndarray:
    """Joints that hold the hand R3 held, turned ``tilt_deg`` about its own closing axis, its lowest point
    ``height_mm`` over the bench at ``at``."""
    owner = cell.owner_cell()
    q = np.radians(_R3_DEG)
    start = owner.tcp(q)
    a = math.radians(tilt_deg)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    goal = start.copy()
    goal[:3, :3] = start[:3, :3] @ turn
    goal[:3, 3] = (at[0], at[1], 150.0)
    for _ in range(4):
        q = owner.ik_tcp(goal, q)
        low, _ = _lowest(q)
        goal[2, 3] -= low - height_mm
    q = owner.ik_tcp(goal, q)
    assert abs(_lowest(q)[0] - height_mm) < 0.05
    return q


def _bare_bench_boxes(depth: "np.ndarray | None" = None) -> Any:
    owner = cell.owner_cell()
    look = cell.look("P1")
    snapshot = owner.world(look, cell.level_table(look) if depth is None else depth).world_for(
        self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)), near_point_mm=(-130.0, -690.0, 100.0))
    assert snapshot.perceived is not None, snapshot.reason
    return snapshot.perceived


class TheGuardHoldsTheBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        if not cell.owner_cell().has_the_engine():
            self.skipTest("no exact mesh engine or UR10 bundle on this box")

    def test_a_side_tilted_hand_3_mm_over_the_bare_bench_is_refused(self) -> None:
        owner = cell.owner_cell()
        world = _bare_bench_boxes()
        boxes = _guard_boxes(world)
        low = _hand_over_the_bench(3.0, 45.0)
        refusal = owner.refused(boxes, low)
        # What holds the bench: its own solid, or the bare table read as a support where the noise lifts its reading.
        self.assertRegex(refusal, r"fixture:seen_s\d\d_(bench|support\d)")
        self.assertIn("< 5.000 mm", refusal)
        # 15 mm over the plane keeps more than 5 mm over what holds the bench (its band 5, the allowance 2, and the
        # reading's excess where the table is read as a support).
        self.assertEqual(owner.refused(boxes, _hand_over_the_bench(15.0, 45.0)), "")

    def test_the_bench_is_held_up_to_its_band_and_the_allowance(self) -> None:
        bench = [box for box in _bare_bench_boxes().boxes if box.kind == "bench"]
        self.assertEqual(len(bench), 1)
        self.assertAlmostEqual(bench[0].center_mm[2] + bench[0].dims_mm[2] / 2.0, 7.0, places=6)
        self.assertAlmostEqual(bench[0].center_mm[2] - bench[0].dims_mm[2] / 2.0, -50.0, places=6)

    def test_small_cubes_on_the_bare_bench_are_held(self) -> None:
        look = cell.look("P1")
        table = cell.level_table(look)
        for size in (8.0, 10.0, 12.0, 15.0):
            for x, y in ((-130.0, -690.0), (60.0, -600.0)):
                with self.subTest(size_mm=size, at=(x, y)):
                    depth = cell.with_solids(look, boxes=[((x, y, size / 2.0), (size, size, size), None)],
                                             depth_mm=table, seed=int(abs(size * 3 + x)))
                    world = _bare_bench_boxes(depth)
                    top = np.array([x, y, size - 0.5])
                    held = [box.name for box in world.boxes if box.kind == "seen" and _inside(box, top)]
                    self.assertTrue(held, f"a {size:g} mm cube on the bare bench left the world")


def _inside(box: Any, point: np.ndarray) -> bool:
    local = (np.asarray(point) - np.asarray(box.center_mm)) @ box.rotation_matrix
    return bool(np.all(np.abs(local) <= np.asarray(box.dims_mm) / 2.0))


if __name__ == "__main__":
    unittest.main()
