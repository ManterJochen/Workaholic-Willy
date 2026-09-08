"""SafetyPreflight, the ordered safety-guard pipeline.

This is the single entry point every driver consults before commanding motion. It runs
the configured guards in a deterministic order and short-circuits on the first
rejection:

1. workspace, the Cartesian box and its optional margin
2. joint_limit, the per-axis joint hard limits and margin
3. ik_quality, the IK solution quality
4. self_collision, link against link and link against fixture
5. payload, the mass, centre-of-gravity and inertia envelope
6. motion_continuity, the step size between consecutive commands

The workspace box is the one non-negotiable guard and is always wired. Every other
family is added by :meth:`SafetyPreflight.from_safety_config` only where its
``enforce`` flag is set, which leaves callers unchanged.

Failing open against failing closed
-----------------------------------
Every guard family has an ``enforce: bool`` flag in :class:`RobotSafetyConfig`. With
``enforce`` at ``False`` the guard is left out of the pipeline at construction and
never executes at all. With ``enforce`` at ``True`` and a guard returning
:attr:`SafetyReason.UNAVAILABLE`, the preflight fails closed: the motion is rejected
with the guard ``motion_status_override``, which defaults to
:attr:`MotionStatus.CONTROLLER_REJECTED`, so the operator sees a refusal rather than a
silent acceptance.

Driver boundary
---------------
A driver builds a context, evaluates it, and lets :meth:`as_motion_result` translate a
rejection into a typed :class:`MotionResult`, which is ``None`` on acceptance, so the
translation is written once:

.. code-block:: python

    ctx = self._preflight.context_for_pose(pose, current_joints=self.get_joint_positions())
    decision = self._preflight.evaluate(ctx)
    if (result := SafetyPreflight.as_motion_result(decision, command, target_pose=pose)) is not None:
        return result
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from src.robot.constants import SAFETY_PREFLIGHT_LOG_FILE, create_robot_logger
from src.robot.core import MotionCommand, MotionResult

from .continuity import MotionContinuityGuard
from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext, SafetyGuard
from .ik_quality import IKQualityGuard
from .joint_limits import JointLimitGuard
from .payload import PayloadGuard
from .self_collision import SelfCollisionGuard
from .workspace import WorkspaceGuard

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import (
        RobotSafetyConfig,
        WorkspaceLimitsConfig,
    )
    from src.geometry import Pose
    from src.robot.core import JointPositions, RobotArm
    from src.robot.safety._capsule import AxisAlignedBox

__all__ = [
    "SafetyPreflight",
    "WorkspaceSafetyGuard",
]


class WorkspaceSafetyGuard:
    """A :class:`SafetyGuard` adapter around the :class:`WorkspaceGuard` box check.

    Only the workspace-box part of that guard is consulted here. The diversity check
    stays on :class:`WorkspaceGuard`, because it is a calibration-time concern about
    pose sampling and not a per-move safety question.

    Parameters
    ----------
    guard
        The backing :class:`WorkspaceGuard`, reused as it is, so the
        :class:`MotionController` and :class:`PoseProvider` integrations keep working.
    """

    name = "workspace"

    def __init__(self, guard: WorkspaceGuard) -> None:
        self._guard = guard

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        if ctx.target_pose is None:
            # A joint-only command has no Cartesian target to bound-check. The
            # joint-limit and self-collision guards enforce the real bounds for a joint
            # move.
            return SafetyDecision.accept(self.name, message="no Cartesian target")
        pose = ctx.target_pose
        if not self._guard.is_inside_workspace(pose):
            label = pose.label or "<unlabeled>"
            x, y, z = pose.position_mm
            return SafetyDecision.reject(
                self.name,
                SafetyReason.WORKSPACE,
                message=f"pose '{label}' outside workspace box",
                detail={
                    "label": label,
                    "x_mm": f"{float(x):.3f}",
                    "y_mm": f"{float(y):.3f}",
                    "z_mm": f"{float(z):.3f}",
                },
            )
        return SafetyDecision.accept(self.name)


class SafetyPreflight:
    """Ordered :class:`SafetyGuard` pipeline.

    Construction
    ------------
    The plain constructor takes a pre-assembled sequence of guards in the order they
    run. Callers use :meth:`from_safety_config`, or :meth:`from_workspace_only` for a
    minimal pipeline, rather than hand-rolling the sequence, because those factories
    enforce the canonical guard order and the ``enforce``-flag handling.
    """

    # The canonical order of guard names. :meth:`from_safety_config` sorts whatever
    # guard set the operator configured into the execution order the module docstring
    # documents.
    _CANONICAL_ORDER: tuple[str, ...] = (
        "workspace",
        "joint_limit",
        "ik_quality",
        "self_collision",
        "payload",
        "motion_continuity",
    )

    def __init__(
        self,
        guards: Sequence[SafetyGuard],
        *,
        check_trajectories: bool = False,
        trajectory_stride: int = 1,
    ) -> None:
        # Copy to a tuple, so the pipeline is immutable after construction. A guard
        # instance stays mutable, because it owns its own cache, but the sequence
        # cannot be reordered or extended at runtime.
        self._guards: tuple[SafetyGuard, ...] = tuple(guards)
        self._logger = create_robot_logger("SafetyPreflight", SAFETY_PREFLIGHT_LOG_FILE)
        # The continuity memo: the last accepted Pose and JointPositions for this
        # preflight instance. Every accepted evaluation writes it, so the continuity
        # guard has the previous target to compare against on the next call.
        self._last_target_pose: "Pose | None" = None
        self._last_target_joints: "JointPositions | None" = None
        # Whether a planned path is judged configuration by configuration, and how
        # densely. It is held here rather than read by each driver, because the two
        # drivers that reach a planned trajectory hold different config objects and the
        # sim one carries no safety block at all.
        self._check_trajectories = bool(check_trajectories)
        self._trajectory_stride = max(1, int(trajectory_stride))
        #: The warn-once latch for a planned path executing with the trajectory check
        #: off. It is per preflight rather than per module, so a second cell in the same
        #: process is still told.
        self._warned_unchecked_path = False

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_workspace_only(cls, workspace_guard: WorkspaceGuard) -> "SafetyPreflight":
        """Construct a preflight carrying the workspace guard alone.

        It is for a driver that has not wired the rest of the guard pipeline.
        """
        return cls([WorkspaceSafetyGuard(workspace_guard)])

    @classmethod
    def from_safety_config(
        cls,
        safety_cfg: "RobotSafetyConfig",
        workspace_cfg: "WorkspaceLimitsConfig",
        *,
        extra_guards: Iterable[SafetyGuard] = (),
    ) -> "SafetyPreflight":
        """Build the canonical preflight pipeline from configuration.

        Parameters
        ----------
        safety_cfg
            The vendor-neutral safety knobs. Each sub-block carries an ``enforce``
            flag, and a disabled block is left out of the pipeline entirely, so its
            guard never runs.
        workspace_cfg
            The Cartesian workspace box. The workspace guard always runs, because the
            box is the single non-negotiable safety surface for the operator cell.
        extra_guards
            Optional additional guards, injected after the canonical set. Each is named
            for the canonical slot it belongs in, and the factory re-sorts the full set
            into :attr:`_CANONICAL_ORDER` before constructing the preflight.
        """
        # The workspace guard is non-negotiable and is always wired.
        workspace_margin = float(safety_cfg.limits.workspace_margin_mm)
        narrowed = _shrink_workspace(workspace_cfg, workspace_margin)
        workspace_guard = WorkspaceGuard(limits=narrowed)
        guards: list[SafetyGuard] = [WorkspaceSafetyGuard(workspace_guard)]
        # Wire the per-family guards whose ``enforce`` flag is ``True``.
        # ``enforce=False`` removes the guard from the pipeline at construction. It
        # does not merely silence the log: the safety surface for that family is gone
        # until the operator flips the flag back.
        if safety_cfg.joint_limits.enforce:
            guards.append(JointLimitGuard(safety_cfg.joint_limits))
        if safety_cfg.ik_quality.enforce:
            guards.append(
                IKQualityGuard(
                    safety_cfg.ik_quality,
                    joint_limits_config=safety_cfg.joint_limits,
                )
            )
        if safety_cfg.self_collision.enforce:
            guards.append(SelfCollisionGuard(safety_cfg.self_collision))
        if safety_cfg.payload.enforce:
            guards.append(PayloadGuard(safety_cfg.payload))
        if safety_cfg.motion_continuity.enforce:
            guards.append(MotionContinuityGuard(safety_cfg.motion_continuity))
        guards.extend(extra_guards)
        guards.sort(key=lambda g: cls._guard_order_key(g.name))
        check = getattr(safety_cfg, "trajectory_check", None)
        return cls(
            guards,
            check_trajectories=bool(getattr(check, "enabled", False)),
            trajectory_stride=int(getattr(check, "stride", 1) or 1),
        )

    @staticmethod
    def _guard_order_key(name: str) -> int:
        try:
            return SafetyPreflight._CANONICAL_ORDER.index(name)
        except ValueError:
            # An unknown guard runs last, in insertion order, which leaves room for a
            # site-local guard without a contract change.
            return len(SafetyPreflight._CANONICAL_ORDER)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def guards(self) -> tuple[SafetyGuard, ...]:
        """An ordered snapshot of the configured guard pipeline."""
        return self._guards

    @property
    def guard_names(self) -> tuple[str, ...]:
        """The guard ``name`` properties, in execution order."""
        return tuple(g.name for g in self._guards)

    @property
    def omitted_guards(self) -> tuple[str, ...]:
        """The canonical guard families that are not in this pipeline, in canonical order.

        The absence is the safety fact. :attr:`guard_names` says what will run, so an
        operator reads a guard count and stops; what a count cannot show is which family
        is gone, and `from_safety_config` removes a whole family where its ``enforce``
        flag is ``False``, which does not merely silence the log.

        A cell with self-collision off and a cell with it on both report a working
        pipeline, and this is the difference between them. It is derived rather than
        declared, being the canonical set minus what is wired, so a guard that failed to
        construct for any reason appears here too.

        ``workspace`` can never appear, because it is wired unconditionally as the one
        non-negotiable surface. Seeing it here means something is wrong with this
        property rather than with the cell.
        """
        wired = set(self.guard_names)
        return tuple(name for name in self._CANONICAL_ORDER if name not in wired)

    @property
    def checks_trajectories(self) -> bool:
        """Whether :meth:`gate_trajectory` judges a path or waves it through.

        It is printable on purpose. ``guard_names`` is how an operator sees which guards
        are in the pipeline, and this is how they see whether those guards ever meet the
        middle of a planned path.
        """
        return self._check_trajectories

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def context_for_pose(
        self,
        pose: "Pose",
        *,
        command: MotionCommand = MotionCommand.MOVE_TO,
        target_joints: "JointPositions | None" = None,
        current_pose: "Pose | None" = None,
        current_joints: "JointPositions | None" = None,
        arm: "RobotArm | None" = None,
    ) -> SafetyContext:
        """Build a :class:`SafetyContext` for a Cartesian move.

        The memoised ``last_target_*`` are folded in, so the continuity guard sees the
        previous accepted target without the driver tracking it.
        """
        return SafetyContext(
            command=command,
            target_pose=pose,
            target_joints=target_joints,
            current_pose=current_pose,
            current_joints=current_joints,
            last_target_pose=self._last_target_pose,
            last_target_joints=self._last_target_joints,
            arm=arm,
        )

    def context_for_joints(
        self,
        joints: "JointPositions",
        *,
        command: MotionCommand = MotionCommand.MOVE_JOINTS,
        target_pose: "Pose | None" = None,
        current_pose: "Pose | None" = None,
        current_joints: "JointPositions | None" = None,
        arm: "RobotArm | None" = None,
    ) -> SafetyContext:
        """Build a :class:`SafetyContext` for a joint move."""
        return SafetyContext(
            command=command,
            target_pose=target_pose,
            target_joints=joints,
            current_pose=current_pose,
            current_joints=current_joints,
            last_target_pose=self._last_target_pose,
            last_target_joints=self._last_target_joints,
            arm=arm,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, ctx: SafetyContext, *, skip_guards: "frozenset[str]" = frozenset()
    ) -> SafetyDecision:
        """Run the pipeline and return the first rejection, or an acceptance.

        On acceptance it updates the memoised ``last_target_*`` from ``ctx``, so the
        continuity guard has them on the next call.

        ``skip_guards`` names guards to bypass by ``.name``, and its default is empty.
        Its only caller is the cuRobo driver path, which skips the two continuity
        guards, the ``ik_quality`` joint-jump check and the ``motion_continuity`` step
        size, on the planned final configuration. Those two stand in for the assumption
        that the blind interpolator will not teleport, which holds only while
        ``move_joint`` interpolates in joint space, and cuRobo instead owns a
        continuous, dynamically feasible, collision-checked path, so a large net joint
        delta such as a wrap between +pi and -pi that is no physical rotation at all is
        not a teleport. The static guards, workspace, joint-limit, self-collision and
        payload, still run on the final configuration, which keeps the Coal
        self-collision gate. The joint-jump guard was measured to be the real blocker
        for cuRobo picks: it rejected the continuous configuration while self-collision
        never fired, and with these two skipped cuRobo lifts reliably.
        """
        for guard in self._guards:
            if guard.name in skip_guards:
                continue
            decision = guard.evaluate(ctx)
            if decision.rejected:
                self._logger.warning(
                    "Safety preflight rejected motion: guard=%s reason=%s "
                    "message=%s detail=%s",
                    decision.guard, decision.reason.value,
                    decision.message or "<empty>", decision.detail or {},
                )
                return decision
        # Accepted, so remember the target for the next continuity check.
        if ctx.target_pose is not None:
            self._last_target_pose = ctx.target_pose
        if ctx.target_joints is not None:
            self._last_target_joints = ctx.target_joints
        return SafetyDecision.accept("preflight")

    def reset(self) -> None:
        """Clear the continuity memo.

        A driver calls this after a controller reset, an emergency stop, or any other
        event that invalidates the previous-target assumption.
        """
        self._last_target_pose = None
        self._last_target_joints = None

    # The guards that gate a Cartesian path near a target rather than a deliberate
    # point-to-point joint command: workspace, the pose-only box; ik_quality, the
    # IK-jump step and singularity; and motion_continuity, the step size. A singularity
    # or a large step bites during Cartesian or servo control near the configuration,
    # not on an interpolated, in-limits, collision-free joint command. That holds on
    # real hardware as well: a moveJ to a singular but reachable configuration is fine,
    # and the singularity bites only under subsequent Cartesian control.
    #
    # This is a code-level constant rather than operator YAML on purpose. The skip-set
    # is a fail-closed selector over safety surfaces, and an operator typo in it would
    # silence a guard on a real joint move.
    _JOINT_MOVE_SKIP_GUARDS = ("workspace", "ik_quality", "motion_continuity")

    def set_perceived_obstacles(self, boxes: "Sequence[AxisAlignedBox]") -> int:
        """Tell every guard that can hold them about obstacles a camera saw. Returns how many were told.

        The planner and this pipeline have to be looking at the same cell. A planner routing around a
        tote this pipeline cannot see gives the worst of both: a path that avoids the tote, and a
        gate that would have passed one straight through it, so nothing is actually holding the line.

        Returning the count rather than nothing is deliberate. A caller that refreshes a world and is
        told zero guards took it has learned that its cell has no guard capable of using obstacles,
        which is a real state and one worth reporting rather than assuming.

        An empty sequence clears them, which is what a cell must do the moment nobody can vouch for
        what the cameras saw.
        """
        told = 0
        for guard in self._guards:
            setter = getattr(guard, "set_perceived_fixtures", None)
            if callable(setter):
                setter(tuple(boxes))
                told += 1
        return told

    def gate_trajectory(
        self,
        waypoints: "Sequence[Sequence[float]]",
        *,
        arm: "RobotArm | None" = None,
        stride: int | None = None,
    ) -> "MotionResult | None":
        """Gate a whole planned path before any of it is commanded.

        ``None`` means every checked configuration passed. The one-shot gates in this
        class judge where a move ends, and a planner hands over the whole path, so the
        whole path can be judged: a plan that grazes a fixture in the middle and lands
        clear is exactly what an endpoint check cannot see.

        The check runs before the first waypoint moves rather than as they execute,
        because a refusal partway leaves the arm standing on a path already judged
        unsafe, and there is nowhere good to put it.

        The guard set matches :meth:`gate_joint_target`: joint limits, self-collision,
        which carries the declared fixtures so the bench is checked here too, and
        payload. The Cartesian guards do not apply to an interpolated joint
        configuration, and continuity between waypoints that are adjacent by
        construction would reject every plan.

        It returns ``None`` immediately unless ``safety.trajectory_check.enabled`` built
        this preflight with the check on, and off is the default that leaves behaviour
        unchanged.

        ``stride`` samples the path instead of checking it. A stride above 1 trades
        coverage for speed and leaves the skipped configurations unexamined. The
        endpoint is checked whatever the stride, because it is the one configuration the
        arm certainly stops in. ``None`` uses the configured stride.
        """
        from src.robot.core import JointPositions

        if not waypoints:
            return None
        if not self._check_trajectories:
            # Said here, because here is where it becomes true. Reaching this line
            # means a multi-waypoint path is about to execute and nothing will look at
            # its middle: the sim applies each waypoint straight to the articulation and
            # a real UR moveJ runs them in turn, so a plan that grazes a bench or a bin
            # wall halfway and lands clear passes every check there is. The one-shot
            # guard on the final configuration cannot see a path.
            #
            # It is a warning rather than a refusal for a measured reason. Every profile
            # this repository ships has `motion_planner: curobo` with
            # `trajectory_check.enabled: false`, `planning_world.enabled: false` and
            # `self_collision.fixtures: []`, so a config-time refusal would refuse the
            # shipped tree, which is a rule too sharp to be a guard. What is missing is
            # not a rule but the sentence: the YAML comment states this exactly, and
            # nobody reads a YAML comment with an arm in front of them.
            #
            # Once per preflight, because a pick loop plans continuously and a line per
            # motion would teach an operator to filter the channel it appears in.
            if not self._warned_unchecked_path:
                self._warned_unchecked_path = True
                self._logger.warning(
                    "a %d-waypoint planned path is executing unexamined between its endpoints: "
                    "safety.trajectory_check.enabled is false, so only the final configuration was "
                    "gated. Turn it on to judge the path (about 9.6 ms per configuration), and "
                    "declare safety.self_collision.fixtures plus safety.planning_world so the guard "
                    "and the planner both know your bench, bin and fixtures exist",
                    len(waypoints),
                )
            return None
        step = max(1, int(self._trajectory_stride if stride is None else stride))
        indices = list(range(0, len(waypoints), step))
        if indices[-1] != len(waypoints) - 1:
            indices.append(len(waypoints) - 1)

        self.reset()  # a planned path restarts the trajectory, as a joint move does
        rejected: "MotionResult | None" = None
        for position in indices:
            joints = JointPositions(tuple(float(v) for v in waypoints[position]))
            ctx = self.context_for_joints(joints, command=MotionCommand.MOVE_JOINTS, arm=arm)
            for guard in self._guards:
                if guard.name in self._JOINT_MOVE_SKIP_GUARDS:
                    continue
                decision = guard.evaluate(ctx)
                if decision.rejected:
                    self._logger.warning(
                        "Safety preflight rejected a PLANNED PATH at waypoint %d of %d: "
                        "guard=%s reason=%s message=%s",
                        position + 1, len(waypoints), decision.guard, decision.reason.value,
                        decision.message or "<empty>",
                    )
                    rejected = SafetyPreflight.as_motion_result(
                        decision, MotionCommand.MOVE_JOINTS, target_joints=joints,
                    )
                    break
            if rejected is not None:
                break
        self.reset()
        return rejected

    def gate_joint_target(
        self,
        joints: "JointPositions",
        *,
        arm: "RobotArm | None" = None,
    ) -> "MotionResult | None":
        """Gate a commanded joint-space move, returning a typed rejection or ``None``.

        A commanded joint move to a park, home or scan pose is a deliberate trajectory
        restart, so this runs the destination guards alone, meaning joint-limit,
        self-collision and payload, which establish that the target configuration is
        in limits, collision-free and inside the payload envelope. It skips the
        Cartesian-control guards in ``_JOINT_MOVE_SKIP_GUARDS``, the workspace box, the
        IK-jump and singularity check and motion continuity, which do not apply to a
        point-to-point joint command. The continuity reference is reset on both sides,
        so neither this move nor the next Cartesian ``move()`` is compared across it.
        """
        self.reset()  # a restart, so continuity is not compared into this joint move
        ctx = self.context_for_joints(joints, command=MotionCommand.MOVE_JOINTS, arm=arm)
        rejected: "MotionResult | None" = None
        for guard in self._guards:
            if guard.name in self._JOINT_MOVE_SKIP_GUARDS:
                continue
            decision = guard.evaluate(ctx)
            if decision.rejected:
                self._logger.warning(
                    "Safety preflight rejected joint move: guard=%s reason=%s message=%s",
                    decision.guard, decision.reason.value, decision.message or "<empty>",
                )
                rejected = SafetyPreflight.as_motion_result(
                    decision, MotionCommand.MOVE_JOINTS, target_joints=joints,
                )
                break
        self.reset()  # a restart, so the next Cartesian move is not compared across this one
        return rejected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def as_motion_result(
        decision: SafetyDecision,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
    ) -> "MotionResult | None":
        """Translate a rejection into a typed :class:`MotionResult`.

        It returns ``None`` for an accepted decision, so a driver writes::

            if (result := SafetyPreflight.as_motion_result(decision, cmd, ...)) is not None:
                return result
        """
        if decision.accepted:
            return None
        status = decision.motion_status
        # :meth:`SafetyDecision.motion_status` guarantees a non-None value for a
        # rejected decision.
        assert status is not None  # noqa: S101 (invariant guard)
        message = (
            f"[safety:{decision.guard}/{decision.reason.value}] "
            f"{decision.message}"
        ) if decision.message else (
            f"[safety:{decision.guard}/{decision.reason.value}]"
        )
        return MotionResult.failed(
            status,
            command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
        )


def _shrink_workspace(
    cfg: "WorkspaceLimitsConfig",
    margin_mm: float,
) -> "WorkspaceLimitsConfig":
    """Return a :class:`WorkspaceLimitsConfig` shrunk by ``margin_mm`` on every face.

    A margin at or below zero returns ``cfg`` unchanged. A margin that would invert any
    axis raises :class:`ValueError`.
    """
    if margin_mm <= 0.0:
        return cfg
    new = {
        "x_min": cfg.x_min + margin_mm,
        "x_max": cfg.x_max - margin_mm,
        "y_min": cfg.y_min + margin_mm,
        "y_max": cfg.y_max - margin_mm,
        "z_min": cfg.z_min + margin_mm,
        "z_max": cfg.z_max - margin_mm,
    }
    if new["x_min"] >= new["x_max"] or new["y_min"] >= new["y_max"] or new["z_min"] >= new["z_max"]:
        raise ValueError(
            f"workspace_margin_mm={margin_mm} inverts the workspace box "
            f"{(cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.z_min, cfg.z_max)}"
        )
    return cfg.__class__(**new)
