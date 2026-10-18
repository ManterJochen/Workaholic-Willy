"""The other arm of the cell reaches this arm's planner and its cameras' self filter, not only its guard.

Owner, 2026-10-15: a second arm is optional. Where there is one, the planner plans round it (its parts as declared
boxes, so the planner keeps them), the cameras take its body out of what they see as they take this arm's (or it would
stand in the world twice, once as a seen box), and the guard judges its hulls. ``PlannedArm.hold_partners`` hands all
three over at once; a cell of one arm holds none and changes nothing.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.drivers.planned.curobo_motion import CuroboPlanner
from src.robot.safety.partner_arm import PartnerWorld, own_from_other, partner_cuboids, partner_world
from src.robot.safety.planning.perceived import LinkCapsule, SelfBody, SelfEnvelope
from src.robot.safety.planning.world import planner_cuboid
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_planner import FakePlanner
from tests._fake_transport import fake_transport
from tests.robot.safety.test_another_arm_is_an_obstacle import _Q, _arm

#: Where the other arm stands in this arm's base: 1.2 m along x, turned half a turn.
_OTHER = np.array([[-1.0, 0.0, 0.0, 1200.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _quaternion_matrix(wxyz: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _envelope() -> SelfEnvelope:
    """A small body: two frames, a capsule on the second from its origin 300 mm along its x."""
    second = np.eye(4)
    second[:3, 3] = (0.0, 0.0, 400.0)
    return SelfEnvelope(frames_mm=(np.eye(4), second),
                        capsules=(LinkCapsule(frame=1, start_mm=(0.0, 0.0, 0.0), end_mm=(300.0, 0.0, 0.0),
                                              radius_mm=50.0),))


class TheOtherArmAsThePlannerTakesItTests(unittest.TestCase):
    def setUp(self) -> None:
        werker, werker_guard = _arm()
        hulls = werker_guard.body_hulls(werker, _Q)
        self.world = partner_world([("werker", hulls, _envelope(), own_from_other(np.eye(4), _OTHER))])

    def test_each_part_is_a_declared_box_along_its_hull_that_holds_every_vertex(self) -> None:
        names = [box["name"] for box in self.world.cuboids]
        self.assertIn("arm_werker_forearm", names)
        self.assertFalse(any(name.startswith("seen_") for name in names), "declared, so the planner keeps them")
        for body, box in zip(self.world.bodies, self.world.cuboids):
            with self.subTest(part=body.name):
                centre = np.asarray(box["pose"][:3]) * 1000.0
                turn = _quaternion_matrix(box["pose"][3:])
                half = np.asarray(box["dims_m"]) * 500.0
                local = (np.asarray(body.hull_mm) - centre) @ turn
                self.assertTrue(np.all(np.abs(local) <= half + 1e-6))
        self.assertEqual(self.world.arms, ("werker",))

    def test_the_planner_registers_the_cells_boxes_and_the_other_arm_and_again_when_it_moves(self) -> None:
        planner = FakePlanner()
        glue = CuroboPlanner(fake_transport(joints=list(_Q)), joint_names=planner.joint_names, controller_yaw_deg=0,
                             client_factory=lambda: planner,
                             world_cuboids=[planner_cuboid("bench", (0.0, 0.0, -50.0), (2000.0, 2000.0, 100.0))])
        glue.set_partner_cuboids(self.world.cuboids)
        glue._client_or_start()
        sent = planner.calls_of("set_world")
        self.assertEqual([box["name"] for box in sent[-1].args[0]][0], "bench")
        self.assertEqual(len(sent[-1].args[0]), 1 + len(self.world.cuboids))
        glue._client_or_start()
        self.assertEqual(len(planner.calls_of("set_world")), 1, "an unchanged world is not sent again")
        moved = partner_cuboids(self.world.bodies[:3])
        glue.set_partner_cuboids(moved)
        glue._client_or_start()
        self.assertEqual(len(planner.calls_of("set_world")[-1].args[0]), 1 + 3, "the other arm moved: sent again")


class TheCamerasTakeTheOtherArmOutTests(unittest.TestCase):
    def test_a_point_on_the_other_arm_is_body_and_the_reach_stays_this_arms(self) -> None:
        own = _envelope()
        other = _envelope().moved(_OTHER)
        both = SelfEnvelope(frames_mm=own.frames_mm, capsules=own.capsules, others=(other,))
        frames, capsules = both.chains()
        self.assertEqual(len(frames), 4)
        self.assertEqual([capsule.frame for capsule in capsules], [1, 3])
        body = SelfBody.from_frames(frames, capsules)
        # The middle of the other arm's capsule: 150 mm along its x, which the half turn sends along -x from 1200 mm.
        on_the_other = np.array([[1200.0 - 150.0, 0.0, 400.0]])
        self.assertTrue(bool(body.on_itself(on_the_other, within_mm=1.0)[0]))
        self.assertFalse(bool(SelfBody.from_frames(own.frames_mm, own.capsules).on_itself(on_the_other,
                                                                                          within_mm=1.0)[0]))
        self.assertEqual(both.reach(), own.reach(), "the reach is this arm's own")


class AnArmHoldsTheOtherArmsTests(unittest.TestCase):
    def test_hold_partners_reaches_the_guard_the_planner_and_the_self_filter(self) -> None:
        werker, werker_guard = _arm()
        world = partner_world([("werker", werker_guard.body_hulls(werker, _Q), None, _OTHER)])
        planner = FakePlanner()
        arm = planned_arm(kit_tree("curobo", safety={"self_collision": {"backend": "fcl"}}),
                          curobo_client_factory=lambda: planner)
        arm.reserve_partner_slots(len(world.cuboids))
        arm.hold_partners(world)
        guard = next(g for g in arm.safety_preflight.guards if g.name == "self_collision")
        self.assertEqual(guard.partner_bodies, world.bodies)
        self.assertIs(arm.partners, world)
        reserved = arm._reservation()
        self.assertEqual(reserved.cuboid_slots,
                         planned_arm(kit_tree("curobo"))._reservation().cuboid_slots + len(world.cuboids))
        arm.hold_partners(None)
        self.assertEqual(guard.partner_bodies, ())
        self.assertEqual(arm.partners, PartnerWorld())


if __name__ == "__main__":
    unittest.main()
