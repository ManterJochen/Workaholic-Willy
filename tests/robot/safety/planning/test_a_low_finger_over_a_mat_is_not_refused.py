"""A low finger over the mat is no longer refused by the mat (fix plan RC1, Track S, test 1).

R8 on 2026-10-01 (robot.log:503): an orange cylinder on the owner's 55 mm foam mat, example 13's line in from an 80 mm
standoff, refused 4-6 mm before the grasp with the fingers 3.4-4.5 mm from a box. The box was the mat: the camera world
held it as strips of obstacles from the bench up to the mat's reading plus 15 mm, while the fingertips stood about 18 mm
over the mat itself. Now the world finds the mat as the surface the parts stand on and holds it as solid up to its
reading, its excess, the band and the 2 mm allowance, so the line in is admitted, every sample of it judged by the exact
guard at the owner's 5 mm.

The scene is the owner's: each recorded look (P1, P5; both at LOOK[0]) with a 40 mm cylinder drawn in where R8's
grasp stood, from the mat's local reading up to 4 mm over the grasp's TCP, held out as the pick holds its target, the
goal's jaw region at the grasp. The world is the one the cell builds from the owner's profile
(``_live_planner_world``), the arm and the guard the owner's UR10 and Hand-E on the 23 mm plate with its tool frame.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.safety.planning.live_world import _guard_boxes
from tests import _cell_2026_10_01 as cell

#: R8's refused sample (robot.log:503): its joints, its index s in a line of N samples; the line ran 80 mm.
_R8_DEG = (-99.3, -95.7, -127.4, -46.9, 89.7, -281.4)
_SAMPLE, _SAMPLES, _LINE_MM = 43, 46, 80.0
#: How far the fingertips hang under the TCP on this hand (robot.log:297: TCP z 95.4, lowest finger z 83.1).
_TIP_BELOW_TCP_MM = 12.3


def _cylinder_points(x: float, y: float, z0: float, z1: float, radius: float = 20.0) -> np.ndarray:
    points = []
    for ring in np.arange(0.0, radius + 0.1, 2.0):
        count = max(1, int(2.0 * math.pi * ring / 2.0))
        for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
            points.append((x + ring * math.cos(angle), y + ring * math.sin(angle), z1))
    for z in np.arange(z0 + 1.0, z1, 2.0):
        for angle in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle), z))
    return np.asarray(points)


class R8sLineInOverTheMatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = cell.owner_cell()
        if not self.owner.has_the_engine():
            self.skipTest("no exact mesh engine or UR10 bundle on this box")

    def test_the_line_in_is_admitted_on_both_recorded_looks(self) -> None:
        owner = self.owner
        q_ref = np.radians(_R8_DEG)
        tcp = owner.tcp(q_ref)
        grasp = tcp.copy()
        grasp[:3, 3] = tcp[:3, 3] + tcp[:3, 2] * (_SAMPLES - 1 - _SAMPLE) * _LINE_MM / (_SAMPLES - 1)
        gx, gy, gz = (float(v) for v in grasp[:3, 3])
        poses = owner.line(q_ref, grasp, _LINE_MM)
        envelope = owner.envelope(np.radians(cell.LOOK_0_DEG))
        flange = (grasp @ np.linalg.inv(owner.tool))[:3, 3]
        for name in cell.LOOKS:
            look = cell.look(name)
            mat = cell.local_reading(look, gx, gy)
            with self.subTest(look=name):
                self.assertTrue(math.isfinite(mat))
                # A low grasp: the TCP stands less than 50 mm over the mat, its fingertips some 25 mm.
                self.assertLess(gz - mat, 50.0)
                self.assertGreater(gz - _TIP_BELOW_TCP_MM - mat, 5.0)
                depth = cell.with_solids(look, cylinders=[(gx, gy, 20.0, mat, gz + 4.0)], seed=8)
                world = owner.world(look, depth)
                world.offer_segmentation(camera="EIH_Cam", target_points_base_mm=_cylinder_points(gx, gy, mat, gz + 4.0),
                                         target_label="part", hold=True)
                snapshot = world.world_for(self_envelope=envelope, near_point_mm=flange,
                                           goal_keep_out=owner.goal_keep_out(grasp))
                self.assertIsNotNone(snapshot.perceived, snapshot.reason)
                assert snapshot.perceived is not None
                boxes = _guard_boxes(snapshot.perceived)
                refusals = [(index, said) for index, said in ((i, owner.refused(boxes, q)) for i, q in enumerate(poses))
                            if said]
                self.assertEqual(refusals, [], f"{name}: the line in is refused: {refusals[:1]}")


if __name__ == "__main__":
    unittest.main()
