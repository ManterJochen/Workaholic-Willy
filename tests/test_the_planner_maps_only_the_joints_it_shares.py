"""The planner maps a joint order only between the same joints, and refuses anything else (P0e E12).

The sidecar reports its joints by name, and the planner remaps between that order and the controller's (the arm
description's). Where the two named different joints, the remap fell back to the identity and planned on: a planner
started on another arm's descriptor would have been handed this arm's joint vector position by position, and a path
planned for other links would have been sent. Found on the P0e map. It refuses now, before anything is planned or
checked; a reordering of the same joints still maps, which is what the remap is for.
"""

from __future__ import annotations

import unittest

from src.config.arms import load_arm
from src.robot.drivers.planned.curobo_motion import CuroboPlanner
from src.robot.safety.planning import CuroboUnavailableError
from tests._fake_transport import fake_transport

_NAMES = tuple(joint.name for joint in load_arm("ur5e").joints)
_Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _planner() -> CuroboPlanner:
    return CuroboPlanner(fake_transport(), joint_names=_NAMES, controller_yaw_deg=180, client_factory=object)  # type: ignore[arg-type]


class TheRemapTests(unittest.TestCase):
    def test_the_same_joints_in_another_order_are_mapped_both_ways(self) -> None:
        planner, reordered = _planner(), list(reversed(_NAMES))
        there = planner._to_client_order(list(_Q), reordered)
        self.assertEqual(list(reversed(_Q)), there)
        self.assertEqual(_Q, planner._to_ur_order(there, reordered))

    def test_other_joints_are_refused_both_ways_and_named(self) -> None:
        other = [f"joint_a{i}" for i in range(1, 7)]
        for remap in (_planner()._to_client_order, _planner()._to_ur_order):
            with self.subTest(remap.__name__), self.assertRaisesRegex(CuroboUnavailableError, r"joint_a1.*shoulder_pan"):
                remap(list(_Q), other)

    def test_a_planner_that_names_no_joints_is_refused(self) -> None:
        with self.assertRaisesRegex(CuroboUnavailableError, "names no joints"):
            _planner()._to_client_order(list(_Q), [])


if __name__ == "__main__":
    unittest.main()
