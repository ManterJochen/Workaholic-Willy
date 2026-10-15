"""Every public way to command a motion goes through the preflight.

``RobotArm.move_joint`` promised, in the Protocol's own docstring, to raise ``RobotMotionRejected``
"If a workspace / safety pre-check denies the move". None of the three drivers ran one, while
``move_to_joints``, its typed twin reaching the same controller call, always did. So the two halves of one
contract disagreed, and the half that was public and called directly by runner code was the unguarded one.
``move_home`` had the same split, and so did the async twins. The raising and bool verbs and the async ones
left the Protocol on 2026-09-19; what they taught stays here, on the typed verbs that remain.

These tests are the fence. A planned arm answers no motion verb itself, only the hooks ``PlannedArm`` declares
(``tests/test_every_planned_arm_stays_inside_the_fence.py`` holds that over every registered vendor), so the gate is
held here once, on a ``PlannedArm`` over the fake transport (``tests/_arm_kit.planned_arm``, P0g slice 13), and what
the controller was sent is read off the fake's tape. The sim arm is not a planned arm yet and keeps its own rows.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock


from src.robot.core import JointPositions, MotionStatus, RobotMotionRejected
from src.robot.safety import SafetyPreflight
from src.robot.safety.decision import SafetyDecision, SafetyReason
from src.robot.safety.guard import SafetyContext
from tests._arm_kit import kit_tree, planned_arm
from tests._fake_transport import fake_transport
from tests._plan_end import pose_where_it_ends

#: Said where each arm is built: these doubles plan or check with cuRobo and carry no camera world, so every
#: motion they command declines it, as a cuRobo cell must since the world became mandatory.
_DECLINED = "unit double: this test exercises the planner and the guard on a cuRobo arm, and no camera world is wired to it"


from tests._arm_kit import AcceptingGuard  # noqa: E402


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


#: The accepting guard, held by the kit since P0g slice 13 (``tests/_arm_kit.AcceptingGuard``).
_AcceptingGuard = AcceptingGuard


def _sim_arm(preflight):
    from src.robot.drivers.sim.arm import IsaacRobotArm
    from src.robot.drivers.sim.config import SimRobotConfig

    arm = IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True), safety_preflight=preflight)
    arm.connect()
    return arm


def _planned(preflight: Any, route: str = "ik") -> tuple[Any, Any]:
    """A planned arm on ``route`` standing at zero over the fake, ``preflight`` its gate; the arm and its fake.

    `ik` PINNED by default. The shipped default is `curobo`, and these tests are about the GATE, not the planner: left
    on the default they reach for a GPU sidecar and fail on a box that has none, which says nothing about whether a
    verb consults its guards.
    """
    fake = fake_transport(joints=[0.0] * 6)
    arm = planned_arm(kit_tree(route, safety={"payload": {"enforce": False}}), transport=fake, preflight=preflight)
    return arm, fake


def _motions(fake: Any) -> list[Any]:
    """Every motion the fake was sent: joint moves and Cartesian moves."""
    return [call for call in fake.tape if call.verb in ("run_trajectory", "move_pose")]


def _assert_nothing_sent(test: unittest.TestCase, fake: Any) -> None:
    """No motion, on a tape that holds the steady gate every verb asks before its guards (the anchor): a fake that
    recorded nothing fails here rather than passing every fence."""
    test.assertTrue(fake.calls("wait_until_steady"), "the fake recorded no steady gate, so 'nothing' reads nothing")
    test.assertEqual([], _motions(fake))


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


class TheJointVerbIsGatedTests(unittest.TestCase):
    """A refused joint move commands nothing, on every driver, and says so in its result."""

    def test_a_planned_arm_refuses_and_commands_nothing(self) -> None:
        guard = _RefusingGuard()
        arm, fake = _planned(SafetyPreflight([guard]))
        self.assertFalse(arm.move_to_joints(_Q).ok)
        self.assertEqual(1, guard.calls, "the refusal did not come from the pipeline")
        _assert_nothing_sent(self, fake)

    def test_a_planned_arm_still_drives_when_accepted(self) -> None:
        arm, fake = _planned(SafetyPreflight([_AcceptingGuard()]))
        self.assertTrue(arm.move_to_joints(_Q).ok)
        self.assertEqual(["run_trajectory"], [call.verb for call in _motions(fake)])

    def test_sim_refuses_even_in_mock_mode(self) -> None:
        """The mock is what the off-box suite drives. A mock that accepts what the real driver
        rejects is a mock that hides regressions."""
        arm = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        before = arm.get_joint_positions()
        self.assertFalse(arm.move_to_joints(_Q).ok)
        self.assertEqual(list(arm.get_joint_positions().tolist()), list(before.tolist()))

    def test_the_joint_verb_gates_exactly_once(self) -> None:
        """`move_to_joints` gates, then drives the ungated primitive. Gating twice would be a
        second full guard pass per move, which with the exact-mesh backend is not free."""
        guard = _AcceptingGuard()
        arm = _sim_arm(SafetyPreflight([guard]))
        result = arm.move_to_joints(_Q)
        self.assertIs(result.status, MotionStatus.EXECUTED)
        self.assertEqual(guard.calls, 1, "the destination was evaluated more than once")

    def test_raise_unless_executed_raises_what_the_old_verb_raised(self) -> None:
        """Runner code that read "no exception means it moved" keeps that meaning in one call."""
        arm, fake = _planned(SafetyPreflight([_RefusingGuard()]))
        with self.assertRaises(RobotMotionRejected) as caught:
            arm.move_to_joints(_Q).raise_unless_executed()
        self.assertIsNotNone(caught.exception.result)
        _assert_nothing_sent(self, fake)


class MoveHomeIsGatedEverywhereTests(unittest.TestCase):
    def test_sim_move_home_is_refused_even_in_mock_mode(self) -> None:
        """The mock committed the home configuration outright, so the off-box suite could never see
        a refusal here. `move_home` returns bool on all three drivers, so this reports, not raises."""
        arm = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        before = arm.get_joint_positions()
        self.assertFalse(arm.move_home())
        self.assertEqual(list(arm.get_joint_positions().tolist()), list(before.tolist()))


class EveryDriverGatesTests(unittest.TestCase):
    """The fence. A driver added later that forgets the gate fails here, not on a cell."""

    def test_no_driver_reaches_its_controller_from_the_joint_verb_without_gating(self) -> None:
        cases = []
        arm, fake = _planned(SafetyPreflight([_RefusingGuard()]))
        cases.append(("planned", arm, lambda: _assert_nothing_sent(self, fake)))
        sim = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        cases.append(("sim", sim, lambda: None))

        for vendor, driver, assert_quiet in cases:
            with self.subTest(vendor=vendor):
                self.assertFalse(driver.move_to_joints(_Q).ok)
                assert_quiet()


class CartesianSurfacesAreGatedTests(unittest.TestCase):
    """A Cartesian move runs every guard, not only the workspace box.

    ``MotionController.move_to`` says of itself: "WORKSPACE BOX ONLY". So on a cell where `move()` runs
    all six guards, the bool `move_to` and `move_linear` once commanded motion that no joint-limit,
    IK-quality, self-collision, fixture, payload or continuity guard ever saw.
    """

    def test_a_planned_move_is_refused_and_commands_nothing(self) -> None:
        guard = _RefusingGuard()
        arm, fake = _planned(SafetyPreflight([guard]))
        # IK answers, so the refusal below is the GUARD's. Without it the fake controller finds no IK
        # and the surface refuses for a reason that has nothing to do with the pipeline.
        arm.ik = lambda pose: _Q
        self.assertFalse(arm.move(_pose()).ok)
        _assert_nothing_sent(self, fake)
        self.assertEqual(guard.calls, 1, "the refusal did not come from the pipeline")

    def test_a_planned_move_linear_raises_and_commands_nothing(self) -> None:
        guard = _RefusingGuard()
        arm, fake = _planned(SafetyPreflight([guard]))
        arm.ik = lambda pose: _Q
        with self.assertRaises(RobotMotionRejected):
            arm.move_linear(_pose())
        _assert_nothing_sent(self, fake)
        self.assertEqual(guard.calls, 1, "the refusal did not come from the pipeline")

    def test_a_planned_move_gates_exactly_once(self) -> None:
        """`move()` drives the ungated primitive, so the pipeline runs once per commanded pose."""
        guard = _AcceptingGuard("workspace")
        arm, fake = _planned(SafetyPreflight([guard]))
        arm.ik = lambda pose: _Q
        result = arm.move(_pose())
        self.assertIs(result.status, MotionStatus.EXECUTED, result.message)
        self.assertEqual(guard.calls, 1, "the pose was evaluated more than once")
        self.assertEqual(["move_pose"], [call.verb for call in _motions(fake)])


class ThePlannedEndpointGetsTheBoxTests(unittest.TestCase):
    """`gate_joint_target` skips the workspace box, correctly: a joint-only context has no pose, so
    the guard accepts with "no Cartesian target" and un-skipping it there would be a no-op.

    At the cuRobo gate the commanded pose IS in hand, so the planned final configuration can get all
    four static guards, which is what the sim path always had. The real UR was checking three.
    """

    def _curobo_arm(self, preflight: Any) -> Any:
        return _planned(preflight, "curobo")[0]

    def test_the_box_is_applied_to_the_planned_move(self) -> None:
        guard = _AcceptingGuard("workspace")
        arm = self._curobo_arm(SafetyPreflight([guard]))
        self.enterContext(arm.without_camera_world(_DECLINED))
        planner = MagicMock()
        planner.plan_joint.side_effect = lambda goal, **_: [[0.0] * 6, list(goal)]
        arm._curobo = planner
        arm.move(pose_where_it_ends(arm, [0.1] * 6))
        self.assertGreaterEqual(guard.calls, 1, "the workspace guard never saw the planned move")

    def test_a_refusal_stops_the_move_before_execution(self) -> None:
        arm = self._curobo_arm(SafetyPreflight([_RefusingGuard("workspace")]))
        planner = MagicMock()
        planner.plan_joint.side_effect = lambda goal, **_: [[0.0] * 6, list(goal)]
        arm._curobo = planner
        arm.move(pose_where_it_ends(arm, [0.1] * 6))
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
        self.enterContext(arm.without_camera_world(_DECLINED))
        planner = MagicMock()
        final = [0.2, -1.1, 1.0, -0.5, 1.4, 0.3]
        returned: list[list[float]] = []

        def plan_joint(goal, **_):
            # cuRobo ends a plan on its goal to within its tolerance, not bit for bit: its own last configuration.
            end = [float(v) + 5e-5 for v in goal]
            returned.append(end)
            return [[0.0] * 6, end]

        planner.plan_joint.side_effect = plan_joint
        arm._curobo = planner
        arm.move(pose_where_it_ends(arm, final))
        self.assertTrue(returned, "nothing was planned, so the gate on a plan's end was never reached")
        self.assertEqual(seen["joints"], returned[-1])
        self.assertTrue(seen["pose"], "and a pose, or the box cannot be applied at all")


class EveryCartesianSurfaceIsGatedTests(unittest.TestCase):
    """The attestation says the pipeline judges every motion an arm's verbs command, and it once did not.

    UR's async `amove_to` reached the controller past the preflight, the KUKA driver's `move_linear` sent LIN
    with no check, and the sim mock's `move_to` and `move_linear` committed the pose before any guard saw it.
    The async verbs and the KUKA driver are gone (2026-09-19); the sim mock's surfaces stay fenced here.
    """

    def test_sim_mock_move_linear_raises_and_keeps_the_tcp(self) -> None:
        arm = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        before = arm.get_tcp_pose().position_mm.tolist()
        with self.assertRaises(RobotMotionRejected):
            arm.move_linear(_pose(410.0, 5.0, 290.0))
        self.assertEqual(arm.get_tcp_pose().position_mm.tolist(), before)

    def test_sim_mock_move_is_refused_and_keeps_the_tcp(self) -> None:
        arm = _sim_arm(SafetyPreflight([_RefusingGuard()]))
        before = arm.get_tcp_pose().position_mm.tolist()
        self.assertFalse(arm.move(_pose(410.0, 5.0, 290.0)).ok)
        self.assertEqual(arm.get_tcp_pose().position_mm.tolist(), before)

    def test_the_accepted_surfaces_still_move(self) -> None:
        """The control. A gate that refused everything would pass every test above."""
        sim = _sim_arm(SafetyPreflight([_AcceptingGuard()]))
        target = _pose(410.0, 5.0, 290.0)
        sim.move_linear(target)
        self.assertEqual(sim.get_tcp_pose().position_mm.tolist(), target.position_mm.tolist())
        target = _pose(420.0, -5.0, 280.0)
        self.assertTrue(sim.move(target).ok)
        self.assertEqual(sim.get_tcp_pose().position_mm.tolist(), target.position_mm.tolist())


if __name__ == "__main__":
    unittest.main()
