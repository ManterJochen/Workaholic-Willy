"""A configuration the sim arm solves is turned onto a full turn its own guard admits.

On the dense cell, with the policy's descent driven as a checked line, every run was refused at sample 3 of 16 with
``axis 5: -359.986 deg outside [-355.000, 355.000]``. That is the same arm pose as +0.014 deg, one full turn away. The
driver unwinds a solution onto the branch nearest the current configuration, and with the wrist standing near its own
limit that nearest branch is the one outside it, so a line the arm can physically drive was refused in ten runs of ten
until the driver asked which turn the joint-limit guard admits.

Turning a revolute joint by 2 pi is the identical pose, so this changes no grasp and no path; it changes which of the
identical numbers is commanded. A joint with no limits to read, or none whose turns land inside them, keeps what the
solver returned: the guard owns that refusal.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import JointPositions
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig
from src.robot.safety import SafetyPreflight

#: A ur5e's axes run to +/- 360 degrees, and the guard keeps 5 degrees of margin off each end.
_LIMIT_DEG = 355.0


def _arm(*, safety: bool = True) -> IsaacRobotArm:
    arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                                       robot_model="ur5e"))
    arm._connected = True
    if safety:
        # The sim profile declares the table itself, as every non-UR cell must (robot.sim.yaml).
        cell = RobotConfig.model_validate({"vendor": "sim", "ur": {"model": "ur5e"}, "safety": {"joint_limits": {
            "min_deg": [-360.0] * 6, "max_deg": [360.0] * 6}}})
        arm._preflight = SafetyPreflight.from_safety_config(cell.safety, cell.workspace_limits)
    else:
        arm._preflight = None
    return arm


def _pose() -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE, label="sample")


class ASolvedConfigurationStaysInsideTheJointLimitsTests(unittest.TestCase):
    def test_a_full_turn_outside_the_limits_is_turned_back(self) -> None:
        arm = _arm()
        seed = np.array([0.0, -1.5, 1.5, 0.0, 1.5, math.radians(-350.0)])

        turned = arm._into_joint_limits(np.array([0.0, -1.5, 1.5, 0.0, 1.5, math.radians(-359.986)]), seed)

        self.assertAlmostEqual(0.014, math.degrees(float(turned[5])), places=3)
        np.testing.assert_allclose(turned[:5], [0.0, -1.5, 1.5, 0.0, 1.5], atol=1e-12)

    def test_a_configuration_the_guard_admits_is_untouched(self) -> None:
        arm = _arm()
        inside = np.array([0.0, -1.5, 1.5, 0.0, 1.5, math.radians(-350.0)])

        np.testing.assert_allclose(arm._into_joint_limits(inside, np.zeros(6)), inside, atol=1e-12)

    def test_the_turn_nearest_the_seed_wins_among_those_the_guard_admits(self) -> None:
        """Two turns can both be inside the limits; the one the arm has to travel least far to reach is taken."""
        arm = _arm()
        far = np.array([0.0, -1.5, 1.5, 0.0, 1.5, math.radians(370.0)])

        near_zero = arm._into_joint_limits(far, np.zeros(6))

        self.assertAlmostEqual(10.0, math.degrees(float(near_zero[5])), places=6)

    def test_a_joint_with_no_turn_inside_the_limits_keeps_what_the_solver_said(self) -> None:
        arm = _arm()
        beyond = np.array([0.0, -1.5, 1.5, 0.0, 1.5, math.radians(-359.986)])
        arm._preflight = None

        np.testing.assert_allclose(arm._into_joint_limits(beyond, np.zeros(6)), beyond, atol=1e-12)

    def test_the_resolver_returns_a_configuration_the_guard_admits(self) -> None:
        """The whole path: the solver answers one full turn out, and the resolver hands back the turn back."""
        arm = _arm()
        solved = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, math.radians(-359.986)])
        arm.get_joint_positions = lambda: JointPositions(  # type: ignore[method-assign]
            [0.0, -1.5, 1.5, 0.0, 1.5, math.radians(-350.0)])
        arm.ik = lambda pose, *, seed=None: solved  # type: ignore[method-assign]
        arm.fk = lambda joints: _pose()  # type: ignore[method-assign]

        answer = arm._resolve_ik(_pose())

        self.assertAlmostEqual(0.014, math.degrees(float(answer.values[5])), places=3)
        guard = next(g for g in arm._preflight.guards if g.name == "joint_limit")  # type: ignore[union-attr]
        limits = guard.limits_for(vendor="sim", model="ur5e")
        assert limits is not None
        low, high = limits
        for axis, value in enumerate(answer.values):
            with self.subTest(axis=axis):
                self.assertLessEqual(low[axis] + guard.margin_deg, math.degrees(float(value)))
                self.assertGreaterEqual(high[axis] - guard.margin_deg, math.degrees(float(value)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
