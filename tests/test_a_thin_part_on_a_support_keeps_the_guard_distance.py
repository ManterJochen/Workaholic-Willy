"""A thin part the mat's band swallows keeps the guard's 5 mm, while the camera reads up to 4 mm low (fix plan Track S,
"why 5 mm is safe", test 7).

A part thinner than the band over the mat's reading leaves the obstacles with the mat, and the solid the guard holds
stands over it. The guard keeps its 5 mm from the solid, the solid stands the 2 mm allowance over what it took out, so
the part keeps the 5 mm plus whatever the solid stands over it, less how low the camera reads. The worst the analysis
found on the owner's mat: a 15 x 15 x 6 mm square at (40, -790) under R3's hand coming straight down, 5.2 mm with the
reading 4 mm low; without the allowance 3.2 mm, so the allowance carries it. The naive world, the band erased and nothing
held, lets the hand within 2.2 mm of such a part at a true reading.

The look is P1, recorded (LOOK[0]); the part drawn into it on the mat's own reading; the hand the owner's Hand-E on the
23 mm plate, R3's orientation (robot.log:297), its left finger's lowest point over the part, lowered 1 mm at a time and
judged by the exact guard at 5 mm against the world the cell builds. What a step admits is measured against the true
part, placed as high as the camera would read it low.
"""

from __future__ import annotations

import unittest
from functools import lru_cache
from typing import Any

import numpy as np

from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety.planning.live_world import _guard_boxes
from tests import _cell_2026_10_01 as cell

_R3_DEG = (-87.9, -94.2, -129.2, -46.5, 89.7, -288.9)


@lru_cache(maxsize=1)
def _finger() -> "tuple[np.ndarray, float]":
    """How the left finger's lowest point stands from the TCP at R3's orientation: in BASE XY, and under it."""
    from src.robot.safety._fcl_self_collision import composed_parts
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm
    from src.robot.safety.planning.hand import planner_hand

    owner = cell.owner_cell()
    hand: Any = planner_hand(owner.config)
    parts = composed_parts("ur10", None, hand.guard_variant, hand.coupling_mm, placement=hand.placement,
                           coupling_boxes=hand.coupling_boxes)
    joints = np.radians(_R3_DEG)
    frames = ur_link_transforms_mm("ur10", joints)
    assert frames is not None
    vertices, _, frame = parts["lfinger"]
    placed = vertices @ np.asarray(frames[frame])[:3, :3].T + np.asarray(frames[frame])[:3, 3]
    low = placed[placed[:, 2] <= placed[:, 2].min() + 2.0].mean(axis=0)
    tcp = owner.tcp(joints)
    return low[:2] - tcp[:2, 3], float(tcp[2, 3] - low[2])


def _descent(x: float, y: float, top_mm: float, bottom_mm: float) -> "list[tuple[float, np.ndarray]]":
    """R3's hand, its left finger's lowest point over (x, y), from ``top_mm`` down to ``bottom_mm`` 1 mm at a time."""
    owner = cell.owner_cell()
    offset, below = _finger()
    q = np.radians(_R3_DEG)
    start = owner.tcp(q)
    poses = []
    for step in range(int(round(top_mm - bottom_mm)) + 1):
        tip = top_mm - step
        goal = start.copy()
        goal[:3, 3] = (x - offset[0], y - offset[1], tip + below)
        q = owner.ik_tcp(goal, q)
        poses.append((tip, q))
    return poses


def _deepest(boxes: Any, poses: "list[tuple[float, np.ndarray]]") -> np.ndarray:
    """The deepest pose the guard admits before its first refusal."""
    owner = cell.owner_cell()
    last = None
    for _, q in poses:
        if owner.refused(boxes, q):
            break
        last = q
    assert last is not None, "the guard refused the hand already at the top"
    return last


def _world_boxes(depth: np.ndarray, near: np.ndarray, *, allowance: float = 2.0, held: bool = True) -> Any:
    from src.config.schema.robot import RobotConfig

    owner = cell.owner_cell()
    look = cell.look("P1")
    config = RobotConfig.model_validate(cell.owner_robot(perceived={"support_allowance_mm": allowance}))
    snapshot = owner.world(look, depth, config=config).world_for(
        self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)), near_point_mm=near)
    assert snapshot.perceived is not None, snapshot.reason
    boxes = _guard_boxes(snapshot.perceived)
    if not held:
        # The naive world: what the band erased is in no solid the guard holds.
        boxes = tuple(box for box in boxes if not box.name.startswith("seen_s"))
    return boxes


def _keeps(x: float, y: float, size: "tuple[float, float, float]", lows: "tuple[float, ...]", **world: Any) -> list[float]:
    """The hand's least distance to the true part at the deepest admitted step, for each reading ``lows`` mm low."""
    owner = cell.owner_cell()
    look = cell.look("P1")
    mat = cell.local_reading(look, x, y, 0.0, 25.0)
    poses = _descent(x, y, mat + 60.0, mat - 2.0)
    depth = cell.with_solids(look, boxes=[((x, y, mat + size[2] / 2.0), size, None)], seed=int(size[2] * 7 + size[0]))
    deepest = _deepest(_world_boxes(depth, owner.tcp(poses[len(poses) // 2][1])[:3, 3], **world), poses)
    half = np.asarray(size, dtype=np.float64) / 2.0
    return [owner.distance_mm(deepest, [AxisAlignedBox(center_mm=np.array([x, y, mat + half[2] + low]),
                                                       half_extents_mm=half, name="true")]) for low in lows]


class AThinPartOnTheMatKeepsTheGuardsDistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        if not cell.owner_cell().has_the_engine():
            self.skipTest("no exact mesh engine or UR10 bundle on this box")

    def test_the_worst_square_keeps_5_mm_with_the_reading_4_mm_low(self) -> None:
        true, two_low, four_low = _keeps(40.0, -790.0, (15.0, 15.0, 6.0), (0.0, 2.0, 4.0))
        self.assertGreaterEqual(true, 9.0)
        self.assertGreaterEqual(two_low, 7.0)
        self.assertGreaterEqual(four_low, 5.0)

    def test_without_the_allowance_the_same_square_comes_within_5_mm(self) -> None:
        true, four_low = _keeps(40.0, -790.0, (15.0, 15.0, 6.0), (0.0, 4.0), allowance=0.0)
        self.assertGreaterEqual(true, 5.0)
        self.assertLess(four_low, 5.0, "the allowance carries nothing: the square keeps 5 mm without it")

    def test_plates_2_to_4_mm_keep_5_mm_at_a_true_reading(self) -> None:
        for thickness in (2.0, 3.0, 4.0):
            for x, y in ((40.0, -790.0), (-280.0, -770.0)):
                with self.subTest(thickness_mm=thickness, at=(x, y)):
                    (true,) = _keeps(x, y, (15.0, 15.0, thickness), (0.0,))
                    self.assertGreaterEqual(true, 5.0)

    def test_the_naive_world_lets_the_hand_within_5_mm(self) -> None:
        """The band erased and nothing held: the hazard the solids close, so this test can see it."""
        (true,) = _keeps(40.0, -790.0, (15.0, 15.0, 4.0), (0.0,), held=False)
        self.assertLess(true, 5.0)


if __name__ == "__main__":
    unittest.main()
