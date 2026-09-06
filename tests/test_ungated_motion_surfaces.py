"""Every public way to command a joint move goes through the preflight.

``RobotArm.move_joint`` promises, in the Protocol's own docstring, to raise ``RobotMotionRejected``
"If a workspace / safety pre-check denies the move". None of the three drivers ran one, while
``move_to_joints`` -- its typed twin, reaching the same controller call -- always did. So the two
halves of one contract disagreed, and the half that was public, protocol-level and already called
directly by runner code was the unguarded one.

``move_home`` had the same split. The UR arm was hardened on 2026-08-09; Isaac's called ``move_joint``
straight through and KUKA's sent ``send_movej`` down the EKI link with no check of any kind. UR's
async twin ``amove_home`` skipped the hardened method entirely and warned past the workspace box.

These tests are the fence. A driver added later that forgets the gate fails the last test here.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock


from src.robot.core import JointPositions, MotionStatus, RobotMotionRejected
from src.robot.safety import SafetyPreflight
from src.robot.safety.decision import SafetyDecision, SafetyReason
from src.robot.safety.guard import SafetyContext


class _RefusingGuard:
    """A guard that refuses everything, so the test is about the WIRING, not about geometry."""

    def __init__(self, name: str = "self_collision") -> None:
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        self.calls += 1
        return SafetyDecision.reject(
            self._name, SafetyReason.SELF_COLLISION, message="refused by the test guard"
        )


class _AcceptingGuard(_RefusingGuard):
    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        self.calls += 1
        return SafetyDecision.accept(self._name, message="ok")


def _sim_arm(preflight):
    from src.robot.drivers.sim.arm import IsaacRobotArm
    from src.robot.drivers.sim.config import SimRobotConfig

    arm = IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True), safety_preflight=preflight)
    arm.connect()
    return arm


def _ur_arm(preflight):
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    # `motion_planner: ik` PINNED. The shipped default is `curobo` now, and these tests are about the
    # GATE, not the planner: left on the default they reach for a GPU sidecar and fail on a box that
    # has none, which says nothing about whether move_to consults its guards.
    arm = URRobotArm(RobotConfig.model_validate(
        {"vendor": "ur", "ur": {"motion_planner": "ik"},
         "safety": {"payload": {"enforce": False}}}
    ))
    conn = MagicMock()
    conn.is_connected = True
    conn.moveJ.return_value = True
    conn.get_joint_positions.return_value = [0.0] * 6
    arm._conn = conn
    arm._preflight = preflight
    return arm, conn


def _kuka_arm(preflight):
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.kuka.arm import KukaRobotArm

    arm = KukaRobotArm(RobotConfig.model_validate(
        {"vendor": "kuka", "safety": {"payload": {"enforce": False}}}
    ))
    eki = MagicMock()
    eki.is_connected = True
    arm._eki = eki
    arm._preflight = preflight
    return arm, eki


_Q = JointPositions([0.0, -1.5, 1.5, 0.0, 0.0, 0.0])


def _pose(x: float = 400.0, y: float = 0.0, z: float = 300.0):
    import numpy as np

    from src.geometry import Frame, Pose

    return Pose(
        position_mm=np.array([x, y, z], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        frame=Frame.BASE,
        label="target",
    )


class MoveJointIsGatedTests(unittest.TestCase):
    """The Protocol says this raises when a pre-check denies the move. Now it does."""

    def test_ur_move_joint_raises_and_commands_nothing(self) -> None:
        arm, conn = _ur_arm(SafetyPreflight([_RefusingGuard()]))
        with self.assertRaises(RobotMotionRejected):
            arm.move_joint(_Q)
        conn.moveJ.assert_not_called()

    def test_ur_move_joint_still_drives_when_accepted(self) -> None:
        arm, conn = _ur_arm(SafetyPreflight([_AcceptingGuard()]))
        arm.move_joint(_Q)
        conn.moveJ.assert_called_once()

    def test_sim_move_joint_raises_even_in_mock_mode(self) -> None:
        """The mock is what the off-box suite drives. A mock that accepts what the real driver
        rejects is a mock that hides regressions."""
        arm = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        before = arm.get_joint_positions()
        with self.assertRaises(RobotMotionRejected):
            arm.move_joint(_Q)
        self.assertEqual(list(arm.get_joint_positions().tolist()), list(before.tolist()))

    def test_kuka_move_joint_raises_and_sends_nothing(self) -> None:
        arm, eki = _kuka_arm(SafetyPreflight([_RefusingGuard()]))
        with self.assertRaises(RobotMotionRejected):
            arm.move_joint(_Q)
        eki.send_movej.assert_not_called()

    def test_the_typed_twin_gates_exactly_once(self) -> None:
        """`move_to_joints` gates, then drives the ungated primitive. Gating twice would be a
        second full guard pass per move, which with the exact-mesh backend is not free."""
        guard = _AcceptingGuard()
        arm = _sim_arm(SafetyPreflight([guard]))
        result = arm.move_to_joints(_Q)
        self.assertIs(result.status, MotionStatus.EXECUTED)
        self.assertEqual(guard.calls, 1, "the destination was evaluated more than once")


class MoveHomeIsGatedEverywhereTests(unittest.TestCase):
    def test_sim_move_home_is_refused_even_in_mock_mode(self) -> None:
        """The mock committed the home configuration outright, so the off-box suite could never see
        a refusal here. `move_home` returns bool on all three drivers, so this reports, not raises."""
        arm = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        before = arm.get_joint_positions()
        self.assertFalse(arm.move_home())
        self.assertEqual(list(arm.get_joint_positions().tolist()), list(before.tolist()))

    def test_kuka_move_home_is_refused_and_sends_nothing(self) -> None:
        arm, eki = _kuka_arm(SafetyPreflight([_RefusingGuard()]))
        self.assertFalse(arm.move_home())
        eki.send_movej.assert_not_called()

    def test_ur_amove_home_is_refused(self) -> None:
        import asyncio

        arm, conn = _ur_arm(SafetyPreflight([_RefusingGuard()]))
        arm._motion = MagicMock()
        self.assertFalse(asyncio.run(arm.amove_home()))
        arm._motion.amove_home.assert_not_called()


class EveryDriverGatesTests(unittest.TestCase):
    """The fence. A driver added later that forgets the gate fails here, not on a cell."""

    def test_no_driver_reaches_its_controller_from_move_joint_without_gating(self) -> None:
        cases = []
        arm, conn = _ur_arm(SafetyPreflight([_RefusingGuard()]))
        cases.append(("ur", arm, lambda: conn.moveJ.assert_not_called()))
        arm, eki = _kuka_arm(SafetyPreflight([_RefusingGuard()]))
        cases.append(("kuka", arm, lambda: eki.send_movej.assert_not_called()))
        sim = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        cases.append(("sim", sim, lambda: None))

        for vendor, driver, assert_quiet in cases:
            with self.subTest(vendor=vendor):
                with self.assertRaises(RobotMotionRejected):
                    driver.move_joint(_Q)
                assert_quiet()


if __name__ == "__main__":
    unittest.main()


class CartesianSurfacesAreGatedTests(unittest.TestCase):
    """`move_to` and `move_linear` ran the workspace box and nothing else.

    ``MotionController.move_to`` says so in its own comment: "WORKSPACE BOX ONLY". So on a cell where
    `move()` runs all six guards, these two surfaces commanded motion that no joint-limit, IK-quality,
    self-collision, fixture, payload or continuity guard ever saw. They are not obscure: the pick
    loop's viewpoint relocation called `move_to` directly.
    """

    def test_ur_move_to_is_refused_and_commands_nothing(self) -> None:
        arm, conn = _ur_arm(SafetyPreflight([_RefusingGuard()]))
        arm._motion = MagicMock()
        self.assertFalse(arm.move_to(_pose()))
        arm._motion.move_to.assert_not_called()

    def test_ur_move_linear_raises_and_commands_nothing(self) -> None:
        arm, conn = _ur_arm(SafetyPreflight([_RefusingGuard()]))
        arm._motion = MagicMock()
        with self.assertRaises(RobotMotionRejected):
            arm.move_linear(_pose())
        arm._motion.move_to.assert_not_called()

    def test_ur_move_gates_exactly_once(self) -> None:
        """`move()` drives the ungated primitive, so the pipeline runs once per commanded pose."""
        guard = _AcceptingGuard("workspace")
        arm, conn = _ur_arm(SafetyPreflight([guard]))
        arm.ik = lambda pose: _Q  # type: ignore[assignment, misc]
        arm._motion = MagicMock()
        arm._motion.move_to.return_value = True
        result = arm.move(_pose())
        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertEqual(guard.calls, 1, "the pose was evaluated more than once")

    def test_kuka_move_to_is_refused_and_sends_nothing(self) -> None:
        arm, eki = _kuka_arm(SafetyPreflight([_RefusingGuard()]))
        arm.ik = lambda pose: _Q  # type: ignore[assignment, misc]
        arm.get_joint_positions = lambda: _Q  # type: ignore[assignment, misc]
        self.assertFalse(arm.move_to(_pose()))
        eki.send_move_cartesian.assert_not_called()


class ThePlannedEndpointGetsTheBoxTests(unittest.TestCase):
    """`gate_joint_target` skips the workspace box, correctly: a joint-only context has no pose, so
    the guard accepts with "no Cartesian target" and un-skipping it there would be a no-op.

    At the cuRobo gate the commanded pose IS in hand, so the planned final configuration can get all
    four static guards, which is what the sim path always had. The real UR was checking three.
    """

    def _curobo_arm(self, preflight):
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(RobotConfig.model_validate({
            "vendor": "ur",
            "ur": {"motion_planner": "curobo"},
            "safety": {"payload": {"enforce": False}},
        }))
        conn = MagicMock()
        conn.is_connected = True
        conn.moveJ.return_value = True
        conn.get_joint_positions.return_value = [0.0] * 6
        arm._conn = conn
        arm._preflight = preflight
        return arm

    def test_the_box_is_applied_to_the_planned_move(self) -> None:
        guard = _AcceptingGuard("workspace")
        arm = self._curobo_arm(SafetyPreflight([guard]))
        planner = MagicMock()
        planner.plan.return_value = [[0.0] * 6, [0.1] * 6]
        arm._curobo_ur = planner
        arm.move(_pose(400.0, 0.0, 300.0))
        self.assertGreaterEqual(guard.calls, 1, "the workspace guard never saw the planned move")

    def test_a_refusal_stops_the_move_before_execution(self) -> None:
        arm = self._curobo_arm(SafetyPreflight([_RefusingGuard("workspace")]))
        planner = MagicMock()
        planner.plan.return_value = [[0.0] * 6, [0.1] * 6]
        arm._curobo_ur = planner
        arm.move(_pose(400.0, 0.0, 300.0))
        planner.execute.assert_not_called()

    def test_the_gate_sees_curobos_OWN_final_joints(self) -> None:
        """Not a fresh IK solution for the same pose: the configuration the arm will actually be in."""
        seen: dict = {}

        class _Recorder(_AcceptingGuard):
            def evaluate(self, ctx):
                seen["joints"] = list(ctx.target_joints.values) if ctx.target_joints else None
                seen["pose"] = ctx.target_pose is not None
                return super().evaluate(ctx)

        arm = self._curobo_arm(SafetyPreflight([_Recorder("workspace")]))
        planner = MagicMock()
        final = [0.2, -1.1, 1.0, -0.5, 1.4, 0.3]
        planner.plan.return_value = [[0.0] * 6, final]
        arm._curobo_ur = planner
        arm.move(_pose(400.0, 0.0, 300.0))
        self.assertEqual(seen["joints"], final)
        self.assertTrue(seen["pose"], "and a pose, or the box cannot be applied at all")
