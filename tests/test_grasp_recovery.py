"""Tests for Phase S5 dense-clutter recovery planning.

Coverage layers:

* :class:`FixtureEnvelope` validation + ``contains``.
* :class:`SceneRecoveryPolicy` validation (allow-list, fixture
  requirement for physical actions, NONE rejected from allow-list).
* :class:`NoRecoveryStrategy`.
* :class:`ActivePerceptionRecoveryStrategy` escalation + EASY gating.
* :class:`NextTargetRecoveryStrategy`.
* :class:`SmallNudgeStrategy` clamping + envelope refusal.
* :class:`ContainerAgitateStrategy` scaffolding + executor refusal.
* :func:`execute_recovery_motion`: completes non-motion plans, drives
  bounded nudges, aborts on non-EXECUTED motion status, refuses
  agitate motion.
* :class:`AutonomousGraspService` wiring surface: ``recovery_policy``
  and ``recovery_strategy`` flow through factories.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotCapabilities,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspBehaviorProfile,
    GraspMode,
)
from src.robot.grasping import (
    ActivePerceptionRecoveryStrategy,
    ContainerAgitateStrategy,
    FixtureEnvelope,
    NextTargetRecoveryStrategy,
    NoRecoveryStrategy,
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
    SmallNudgeStrategy,
    execute_recovery_motion,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.types.perception import PerceptionFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(
    *,
    mode: GraspMode = GraspMode.DENSE_CLUTTER,
    actions: tuple[str, ...] = ("rescan", "next_viewpoint"),
) -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=mode,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        refinement_enabled=False,
        verification_enabled=False,
        recovery_allowed_actions=actions,
    )


def _easy_profile() -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.EASY,
        sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        recovery_allowed_actions=(),
    )


def _ctx(
    *,
    profile: Optional[GraspBehaviorProfile] = None,
    policy: Optional[SceneRecoveryPolicy] = None,
    history: tuple[SceneRecoveryAction, ...] = (),
    last_frame: Optional[PerceptionFrame] = None,
    current_tcp: Optional[Pose] = None,
    last_grasp: Optional[GraspPoint] = None,
) -> SceneRecoveryContext:
    return SceneRecoveryContext(
        profile=profile or _profile(),
        policy=policy or SceneRecoveryPolicy(enabled=True),
        last_outcome=AutonomousGraspOutcome.NO_VALID_GRASP,
        last_frame=last_frame,
        current_tcp=current_tcp,
        last_grasp=last_grasp,
        history=history,
    )


def _seg(shape=(8, 8)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:4, 1:4] = 1
    return SimpleNamespace(mask=mask)


def _frame_with_n_segs(n: int) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.zeros((8, 8), dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=tuple(_seg() for _ in range(n)),
    )


def _tcp_at(xyz=(0.0, 0.0, 200.0)) -> Pose:
    return Pose(
        position_mm=np.asarray(xyz, dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label="tcp",
    )


_CENTERED_FIXTURE = FixtureEnvelope(
    center_mm=(0.0, 0.0, 200.0),
    half_extents_mm=(100.0, 100.0, 100.0),
    max_nudge_mm=10.0,
)


# ---------------------------------------------------------------------------
# FixtureEnvelope
# ---------------------------------------------------------------------------


class FixtureEnvelopeTests(unittest.TestCase):
    def test_contains_inside_and_outside(self) -> None:
        env = _CENTERED_FIXTURE
        self.assertTrue(env.contains((0.0, 0.0, 200.0)))
        self.assertTrue(env.contains((100.0, 100.0, 100.0)))  # on boundary
        self.assertFalse(env.contains((150.0, 0.0, 200.0)))
        self.assertFalse(env.contains((0.0, 0.0, 50.0)))

    def test_rejects_wrong_dimension(self) -> None:
        with self.assertRaises(ValueError):
            FixtureEnvelope(center_mm=(0.0, 0.0), half_extents_mm=(1.0, 1.0, 1.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            FixtureEnvelope(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(1.0, 1.0))  # type: ignore[arg-type]

    def test_rejects_negative_half_extents(self) -> None:
        with self.assertRaises(ValueError):
            FixtureEnvelope(
                center_mm=(0.0, 0.0, 0.0),
                half_extents_mm=(-1.0, 1.0, 1.0),
            )

    def test_rejects_negative_max_nudge(self) -> None:
        with self.assertRaises(ValueError):
            FixtureEnvelope(
                center_mm=(0.0, 0.0, 0.0),
                half_extents_mm=(1.0, 1.0, 1.0),
                max_nudge_mm=-1.0,
            )

    def test_rejects_negative_agitate_amplitude(self) -> None:
        with self.assertRaises(ValueError):
            FixtureEnvelope(
                center_mm=(0.0, 0.0, 0.0),
                half_extents_mm=(1.0, 1.0, 1.0),
                max_agitate_amplitude_mm=-1.0,
            )


# ---------------------------------------------------------------------------
# SceneRecoveryPolicy
# ---------------------------------------------------------------------------


class SceneRecoveryPolicyTests(unittest.TestCase):
    def test_default_is_disabled_with_safe_actions(self) -> None:
        policy = SceneRecoveryPolicy()
        self.assertFalse(policy.enabled)
        self.assertIn(SceneRecoveryAction.RESCAN, policy.allowed_actions)
        self.assertIn(
            SceneRecoveryAction.NEXT_VIEWPOINT, policy.allowed_actions
        )

    def test_physical_action_without_fixture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            )

    def test_physical_action_with_fixture_constructs(self) -> None:
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            fixture=_CENTERED_FIXTURE,
        )
        self.assertTrue(policy.permits(SceneRecoveryAction.NUDGE_TARGET))

    def test_rejects_duplicate_action(self) -> None:
        with self.assertRaises(ValueError):
            SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(
                    SceneRecoveryAction.RESCAN,
                    SceneRecoveryAction.RESCAN,
                ),
            )

    def test_rejects_none_in_allow_list(self) -> None:
        with self.assertRaises(ValueError):
            SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(SceneRecoveryAction.NONE,),
            )

    def test_rejects_negative_max_actions(self) -> None:
        with self.assertRaises(ValueError):
            SceneRecoveryPolicy(max_recovery_actions=-1)

    def test_permits_is_false_when_disabled(self) -> None:
        policy = SceneRecoveryPolicy(enabled=False)
        self.assertFalse(policy.permits(SceneRecoveryAction.RESCAN))


# ---------------------------------------------------------------------------
# NoRecoveryStrategy
# ---------------------------------------------------------------------------


class NoRecoveryStrategyTests(unittest.TestCase):
    def test_always_returns_none(self) -> None:
        plan = NoRecoveryStrategy().plan(_ctx())
        self.assertIs(plan.action, SceneRecoveryAction.NONE)


# ---------------------------------------------------------------------------
# ActivePerceptionRecoveryStrategy
# ---------------------------------------------------------------------------


class ActivePerceptionRecoveryStrategyTests(unittest.TestCase):
    def test_first_call_plans_rescan(self) -> None:
        plan = ActivePerceptionRecoveryStrategy().plan(_ctx())
        self.assertIs(plan.action, SceneRecoveryAction.RESCAN)

    def test_second_call_escalates_to_next_viewpoint(self) -> None:
        plan = ActivePerceptionRecoveryStrategy().plan(
            _ctx(history=(SceneRecoveryAction.RESCAN,))
        )
        self.assertIs(plan.action, SceneRecoveryAction.NEXT_VIEWPOINT)

    def test_exhausted_returns_none(self) -> None:
        plan = ActivePerceptionRecoveryStrategy().plan(
            _ctx(
                history=(
                    SceneRecoveryAction.RESCAN,
                    SceneRecoveryAction.NEXT_VIEWPOINT,
                ),
                policy=SceneRecoveryPolicy(enabled=True, max_recovery_actions=4),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)

    def test_easy_profile_blocks_recovery(self) -> None:
        plan = ActivePerceptionRecoveryStrategy().plan(
            _ctx(profile=_easy_profile())
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "active_perception_exhausted")

    def test_disabled_policy_blocks_recovery(self) -> None:
        plan = ActivePerceptionRecoveryStrategy().plan(
            _ctx(policy=SceneRecoveryPolicy(enabled=False))
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)

    def test_budget_exhausted_blocks_recovery(self) -> None:
        plan = ActivePerceptionRecoveryStrategy().plan(
            _ctx(
                history=(SceneRecoveryAction.RESCAN,),
                policy=SceneRecoveryPolicy(
                    enabled=True, max_recovery_actions=1
                ),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)


# ---------------------------------------------------------------------------
# NextTargetRecoveryStrategy
# ---------------------------------------------------------------------------


class NextTargetRecoveryStrategyTests(unittest.TestCase):
    def test_passes_when_multiple_segmentations(self) -> None:
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NEXT_TARGET,),
        )
        profile = _profile(actions=("next_target",))
        plan = NextTargetRecoveryStrategy().plan(
            _ctx(
                profile=profile,
                policy=policy,
                last_frame=_frame_with_n_segs(2),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NEXT_TARGET)

    def test_none_when_single_segmentation(self) -> None:
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NEXT_TARGET,),
        )
        profile = _profile(actions=("next_target",))
        plan = NextTargetRecoveryStrategy().plan(
            _ctx(
                profile=profile,
                policy=policy,
                last_frame=_frame_with_n_segs(1),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)

    def test_none_when_action_not_permitted_by_profile(self) -> None:
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NEXT_TARGET,),
        )
        plan = NextTargetRecoveryStrategy().plan(
            _ctx(
                profile=_profile(actions=("rescan",)),
                policy=policy,
                last_frame=_frame_with_n_segs(3),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "profile_disallows_action")


# ---------------------------------------------------------------------------
# SmallNudgeStrategy
# ---------------------------------------------------------------------------


class SmallNudgeStrategyTests(unittest.TestCase):
    def _nudge_policy(
        self, *, fixture: Optional[FixtureEnvelope] = None
    ) -> SceneRecoveryPolicy:
        return SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            fixture=fixture or _CENTERED_FIXTURE,
        )

    def _nudge_profile(self) -> GraspBehaviorProfile:
        return _profile(actions=("nudge_target",))

    def test_plans_bounded_nudge_along_axis(self) -> None:
        strategy = SmallNudgeStrategy(offset_axis=(1.0, 0.0, 0.0))
        plan = strategy.plan(
            _ctx(
                profile=self._nudge_profile(),
                policy=self._nudge_policy(),
                current_tcp=_tcp_at((0.0, 0.0, 200.0)),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NUDGE_TARGET)
        self.assertIsNotNone(plan.nudge_offset_mm)
        assert plan.nudge_offset_mm is not None
        dx, dy, dz = plan.nudge_offset_mm
        # Axis was +x, magnitude clamped to fixture.max_nudge_mm (10).
        self.assertAlmostEqual(dx, 10.0)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, 0.0)

    def test_refuses_outside_envelope(self) -> None:
        # TCP at the +x boundary; nudge would push past it.
        strategy = SmallNudgeStrategy(offset_axis=(1.0, 0.0, 0.0))
        plan = strategy.plan(
            _ctx(
                profile=self._nudge_profile(),
                policy=self._nudge_policy(),
                current_tcp=_tcp_at((95.0, 0.0, 200.0)),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "nudge_would_leave_envelope")

    def test_refuses_zero_axis(self) -> None:
        strategy = SmallNudgeStrategy(offset_axis=(0.0, 0.0, 0.0))
        plan = strategy.plan(
            _ctx(
                profile=self._nudge_profile(),
                policy=self._nudge_policy(),
                current_tcp=_tcp_at(),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "zero_offset_axis")

    def test_refuses_without_anchor(self) -> None:
        strategy = SmallNudgeStrategy()
        plan = strategy.plan(
            _ctx(
                profile=self._nudge_profile(),
                policy=self._nudge_policy(),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "no_anchor_pose")


# ---------------------------------------------------------------------------
# ContainerAgitateStrategy
# ---------------------------------------------------------------------------


class ContainerAgitateStrategyTests(unittest.TestCase):
    def _agitate_policy(
        self, *, amplitude: float = 0.0
    ) -> SceneRecoveryPolicy:
        fixture = FixtureEnvelope(
            center_mm=(0.0, 0.0, 200.0),
            half_extents_mm=(100.0, 100.0, 100.0),
            max_agitate_amplitude_mm=amplitude,
        )
        return SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.CONTAINER_AGITATE,),
            fixture=fixture,
        )

    def _agitate_profile(self) -> GraspBehaviorProfile:
        return _profile(actions=("container_agitate",))

    def test_disabled_when_amplitude_zero(self) -> None:
        plan = ContainerAgitateStrategy().plan(
            _ctx(
                profile=self._agitate_profile(),
                policy=self._agitate_policy(amplitude=0.0),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "agitate_amplitude_disabled")

    def test_plans_when_amplitude_positive(self) -> None:
        plan = ContainerAgitateStrategy().plan(
            _ctx(
                profile=self._agitate_profile(),
                policy=self._agitate_policy(amplitude=5.0),
            )
        )
        self.assertIs(plan.action, SceneRecoveryAction.CONTAINER_AGITATE)
        self.assertAlmostEqual(plan.agitate_amplitude_mm, 5.0)


# ---------------------------------------------------------------------------
# Motion executor
# ---------------------------------------------------------------------------


_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class _TypedFakeArm:
    def __init__(self, *, motion_status: MotionStatus = MotionStatus.EXECUTED) -> None:
        self._tcp = _tcp_at()
        self.move_calls: list[Pose] = []
        self._status = motion_status

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_calls.append(pose)
        if self._status is MotionStatus.EXECUTED:
            self._tcp = pose
            return MotionResult.executed(
                MotionCommand.MOVE_TO, target_pose=pose
            )
        # Force a failure status via low-level constructor.
        return MotionResult(
            status=self._status,
            command=MotionCommand.MOVE_TO,
            target_pose=pose,
            message="forced",
        )


class ExecuteRecoveryMotionTests(unittest.TestCase):
    def _nudge_plan(self) -> SceneRecoveryPlan:
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.NUDGE_TARGET,
            nudge_offset_mm=(10.0, 0.0, 0.0),
        )

    def test_none_plan_skipped(self) -> None:
        report = execute_recovery_motion(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            plan=SceneRecoveryPlan(action=SceneRecoveryAction.NONE),
            policy=SceneRecoveryPolicy(enabled=True),
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "skipped_no_action")

    def test_non_motion_action_completes_without_arm_calls(self) -> None:
        arm = _TypedFakeArm()
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=SceneRecoveryPlan(action=SceneRecoveryAction.RESCAN),
            policy=SceneRecoveryPolicy(enabled=True),
        )
        self.assertTrue(report.executed)
        self.assertEqual(report.outcome, "completed")
        self.assertEqual(arm.move_calls, [])

    def test_nudge_drives_arm_to_destination(self) -> None:
        arm = _TypedFakeArm()
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            fixture=_CENTERED_FIXTURE,
        )
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._nudge_plan(),
            policy=policy,
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertTrue(report.executed)
        self.assertEqual(report.outcome, "completed")
        self.assertEqual(len(arm.move_calls), 1)
        dest = arm.move_calls[0].position_mm
        self.assertTrue(np.allclose(dest, np.array([10.0, 0.0, 200.0])))

    def test_nudge_refuses_outside_envelope(self) -> None:
        arm = _TypedFakeArm()
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            fixture=FixtureEnvelope(
                center_mm=(0.0, 0.0, 200.0),
                half_extents_mm=(5.0, 5.0, 5.0),
                max_nudge_mm=10.0,
            ),
        )
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._nudge_plan(),
            policy=policy,
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "refused_envelope_violation")
        self.assertEqual(arm.move_calls, [])

    def test_nudge_aborts_on_non_executed_status(self) -> None:
        arm = _TypedFakeArm(motion_status=MotionStatus.WORKSPACE_REJECTED)
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            fixture=_CENTERED_FIXTURE,
        )
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._nudge_plan(),
            policy=policy,
            current_tcp=_tcp_at(),
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "aborted_motion_failed")

    def test_nudge_refuses_without_tcp(self) -> None:
        report = execute_recovery_motion(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            plan=self._nudge_plan(),
            policy=SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
                fixture=_CENTERED_FIXTURE,
            ),
            current_tcp=None,
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "refused_no_tcp")

    # --- G6: ContainerAgitateStrategy now executes an envelope-clamped, safety-gated oscillation ---
    def _agitate_plan(self, amplitude: float = 5.0) -> SceneRecoveryPlan:
        return SceneRecoveryPlan(
            action=SceneRecoveryAction.CONTAINER_AGITATE,
            agitate_amplitude_mm=amplitude,
        )

    def _agitate_policy(
        self,
        *,
        half_extents: tuple[float, float, float] = (50.0, 50.0, 50.0),
        max_agitate: float = 5.0,
    ) -> SceneRecoveryPolicy:
        return SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.CONTAINER_AGITATE,),
            fixture=FixtureEnvelope(
                center_mm=(0.0, 0.0, 200.0),
                half_extents_mm=half_extents,
                max_agitate_amplitude_mm=max_agitate,
            ),
        )

    def test_agitate_drives_clamped_waypoints(self) -> None:
        # The anti-theatre proof: arm.move_calls goes from [] (old hard refusal) to the bounded
        # there-and-back oscillation, every waypoint inside the fixture box, returning to the start TCP.
        arm = _TypedFakeArm()
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._agitate_plan(amplitude=5.0),
            policy=self._agitate_policy(),
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertTrue(report.executed)
        self.assertEqual(report.outcome, "completed")
        self.assertEqual(len(arm.move_calls), 3)
        np.testing.assert_allclose(arm.move_calls[0].position_mm, [5.0, 0.0, 200.0])
        np.testing.assert_allclose(arm.move_calls[1].position_mm, [-5.0, 0.0, 200.0])
        np.testing.assert_allclose(arm.move_calls[2].position_mm, [0.0, 0.0, 200.0])
        self.assertEqual(report.telemetry["agitate_mode"], "oscillation")  # default depth 0 = legacy air-shake

    def test_contact_redistribute_descends_and_sweeps(self) -> None:
        # C3: agitate_contact_depth_mm > 0 -> a CONTACT redistribute (descend toward the object layer + a
        # directed +X corridor sweep + retract up) that pushes a movable blocker out, NOT the air-shake.
        arm = _TypedFakeArm()
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.CONTAINER_AGITATE,),
            fixture=FixtureEnvelope(
                center_mm=(0.0, 0.0, 200.0), half_extents_mm=(50.0, 50.0, 50.0),
                max_agitate_amplitude_mm=5.0, agitate_contact_depth_mm=20.0, agitate_sweep_offset_mm=10.0,
            ),
        )
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._agitate_plan(amplitude=5.0),
            policy=policy,
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertTrue(report.executed)
        self.assertEqual(report.telemetry["agitate_mode"], "contact_redistribute")
        self.assertEqual(report.telemetry["contact_depth_mm"], 20.0)
        self.assertEqual(len(arm.move_calls), 3)
        np.testing.assert_allclose(arm.move_calls[0].position_mm, [10.0, 0.0, 180.0])  # descend + offset
        np.testing.assert_allclose(arm.move_calls[1].position_mm, [15.0, 0.0, 180.0])  # sweep +X (push)
        np.testing.assert_allclose(arm.move_calls[2].position_mm, [15.0, 0.0, 200.0])  # retract up

    def test_agitate_aborts_on_safety_veto(self) -> None:
        # A SafetyPreflight rejection (here workspace) on any waypoint aborts the agitate -> still
        # fail-closed; the safety layer outranks the recovery motion.
        arm = _TypedFakeArm(motion_status=MotionStatus.WORKSPACE_REJECTED)
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._agitate_plan(),
            policy=self._agitate_policy(),
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "aborted_motion_failed")

    def test_agitate_refuses_outside_envelope(self) -> None:
        # A waypoint leaving the box is refused IN THE EXECUTOR; the arm is never commanded past it.
        arm = _TypedFakeArm()
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._agitate_plan(amplitude=5.0),
            policy=self._agitate_policy(half_extents=(3.0, 50.0, 50.0)),
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "refused_envelope_violation")
        self.assertEqual(arm.move_calls, [])

    def test_agitate_refused_without_tcp(self) -> None:
        report = execute_recovery_motion(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            plan=self._agitate_plan(),
            policy=self._agitate_policy(),
            current_tcp=None,
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "refused_no_tcp")

    def test_agitate_refused_when_disabled(self) -> None:
        # amplitude 0 keeps the agitate OFF (byte-identical default) — no arm motion.
        arm = _TypedFakeArm()
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=self._agitate_plan(amplitude=0.0),
            policy=self._agitate_policy(max_agitate=0.0),
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
        )
        self.assertFalse(report.executed)
        self.assertEqual(report.outcome, "refused_agitate_disabled")
        self.assertEqual(arm.move_calls, [])


# ---------------------------------------------------------------------------
# AutonomousGraspService wiring surface
# ---------------------------------------------------------------------------


class _MinimalCalc:
    def compute_result(self, *_, **__) -> GraspResult:
        return GraspResult(candidates=(), reasons=(), top_score=0.0)


class _MinimalPerception:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        self.calls += 1
        return PerceptionFrame(
            depth_map=np.zeros((4, 4), dtype=np.float64),
            intrinsics=np.eye(3, dtype=np.float64),
            segmentations=(),
        )


class AutonomousGraspServiceRecoverySlotsTests(unittest.TestCase):
    def test_recovery_slots_are_threaded_through_from_components(self) -> None:
        policy = SceneRecoveryPolicy(enabled=True)
        strategy = ActivePerceptionRecoveryStrategy()
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_MinimalCalc(),  # type: ignore[arg-type]
            perception=_MinimalPerception(),
            mode=GraspMode.AUTO,
            recovery_policy=policy,
            recovery_strategy=strategy,
        )
        self.assertIs(service.recovery_policy, policy)
        self.assertIs(service.recovery_strategy, strategy)

    def test_recovery_slots_default_to_none(self) -> None:
        service = AutonomousGraspService.from_components(
            arm=_TypedFakeArm(),  # type: ignore[arg-type]
            calculator=_MinimalCalc(),  # type: ignore[arg-type]
            perception=_MinimalPerception(),
            mode=GraspMode.AUTO,
        )
        self.assertIsNone(service.recovery_policy)
        self.assertIsNone(service.recovery_strategy)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
