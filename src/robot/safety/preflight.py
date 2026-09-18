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
from typing import TYPE_CHECKING, Any

from src.robot.constants import SAFETY_PREFLIGHT_LOG_FILE, create_robot_logger
from src.contracts import UNSET, Maybe, chosen
from src.robot.core import MotionCommand, MotionResult, MotionStatus

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
    from src.robot.safety.path_samples import PathSamples
    from src.robot.safety.planning.hand import PlannerHand

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


def no_path_guard_refusal() -> str:
    """Why every planned or sampled path is refused where no self-collision guard is wired.

    One home for the sentence, so the real cell checklist tells an operator exactly what
    the gate would say.
    """
    return ("no self_collision guard is wired, so nothing here can judge a path: "
            "safety.self_collision.enforce is false in this cell. A planned path is "
            "refused rather than executed unexamined")


def exact_mesh_path_refusal() -> str:
    """Why every planned or sampled path is refused where the guard would answer with the proxy.

    One home for the sentence, so the real cell checklist tells an operator exactly what
    the gate would say.
    """
    return ("the self_collision guard would answer this path with the capsule proxy, "
            "which bounds the arm links and cannot see the gripper on a joint "
            "configuration at all. Judging a path needs the exact mesh engine: set "
            "safety.self_collision.backend to fcl, install python-fcl or Coal, and give "
            "the cell the baked mesh bundle for its robot model")


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

    def __init__(self, guards: Sequence[SafetyGuard]) -> None:
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
    def from_tree(cls, tree: Any, *, extra_guards: Iterable[SafetyGuard] = ()) -> "SafetyPreflight":
        """The preflight a loaded tree's robot section describes, built as the cell's driver builds it.

            gate = SafetyPreflight.from_tree(load_tree())
            refusal = gate.gate_planned_path(waypoints, arm=arm)     # None when every sample passes

        The hand the cell names resolves against the tree's own root, and a UR arm passes its model to
        the guard, as the UR driver does. A tree that did not load, or loaded with no robot block, is
        refused with its own refusal (``ConfigError``); an exact mesh guard on a tree that names no hand
        is refused naming ``robot.gripper.model``.
        """
        from src.robot.safety.planning.hand import planner_hand  # noqa: PLC0415

        robot = tree.robot
        vendor = str(getattr(robot.vendor, "value", robot.vendor))
        arm_model: Maybe[str] = str(robot.ur.model) if vendor == "ur" else UNSET
        return cls.from_safety_config(
            robot.safety, robot.workspace_limits, extra_guards=extra_guards,
            hand=planner_hand(robot, data_dir=tree.root), arm_model=arm_model,
        )

    @classmethod
    def from_safety_config(
        cls,
        safety_cfg: "RobotSafetyConfig",
        workspace_cfg: "WorkspaceLimitsConfig",
        *,
        extra_guards: Iterable[SafetyGuard] = (),
        hand: "Maybe[PlannerHand]" = UNSET,
        arm_model: Maybe[str] = UNSET,
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
        hand
            The hand the cell names, ``planner_hand(robot_config)``. The self-collision
            guard loads its mesh bundle and its coupling from it.
        arm_model
            The arm the driver knows it is, for a guard with no ``kinematics_model``: the
            UR driver passes ``robot.ur.model``. With either resolving a model, an exact
            mesh guard reads hand geometry, and an unset ``hand`` raises ``ConfigError``
            naming ``robot.gripper.model`` rather than checking whatever hand the arm
            bundle carries. A declared tool frame whose approach disagrees with the hand
            model's raises the same way.
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
            # No hand is implied. An exact mesh guard on a resolvable arm reads hand
            # geometry, and with no hand named it would check the 2F-85 every arm bundle
            # carries, which on a Hand-E cell that forgot its name passes by coincidence.
            # It is refused here, at build, where the model resolves.
            from src.robot.safety.planning.hand import (
                hand_geometry_model,
                unset_hand_refusal,
            )

            reads = hand_geometry_model(safety_cfg.self_collision, arm_model)
            if reads is not None:
                from src.config.loader import ConfigError

                if not chosen(hand):
                    what, fix = unset_hand_refusal(reads)
                    raise ConfigError(f"{what} {fix}")
                # The whole placement. A frame that is a mirror, oblique, points into the
                # wrist, or is clocked by anything other than a quarter turn places no hand
                # model: the hand carries UNSET, and a guard built on it would measure the
                # hand where its own model puts it rather than where this cell says it is,
                # which is an implied default for a safety identity. The planner refuses such
                # a cell when it starts, and an ik cell never starts one. This is the only
                # frame refusal: an approach along +Z, which a real UR flange declares, is
                # placed like one along the model's +Y.
                if hand.placement_refusal is not None:
                    raise ConfigError(hand.placement_refusal)
            guards.append(SelfCollisionGuard(safety_cfg.self_collision, hand=hand))
        if safety_cfg.payload.enforce:
            guards.append(PayloadGuard(safety_cfg.payload))
        if safety_cfg.motion_continuity.enforce:
            guards.append(MotionContinuityGuard(safety_cfg.motion_continuity))
        guards.extend(extra_guards)
        guards.sort(key=lambda g: cls._guard_order_key(g.name))
        return cls(guards)

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

    def set_wrist_bodies(self, bodies: "Sequence[object]") -> int:
        """Hand a wrist camera's bodies to every guard that holds them. Returns how many were told.

        The exact-mesh guard builds its backend once, so that guard refuses this after the first
        judged path. An empty sequence clears them.
        """
        told = 0
        for guard in self._guards:
            setter = getattr(guard, "set_wrist_bodies", None)
            if callable(setter):
                setter(tuple(bodies))
                told += 1
        return told

    def wrist_bodies(self, arm: "RobotArm | None" = None) -> "tuple[object, ...]":
        """The wrist camera bodies the self-collision guard holds, empty where none is wired or none was handed in."""
        guard = self._path_authority(arm)
        return guard.wrist_bodies if guard is not None else ()

    #: The guard whose verdict a judged path rests on. Named once, because three places
    #: ask for it.
    _PATH_AUTHORITY = "self_collision"

    def _path_authority(self, arm: "RobotArm | None") -> "SelfCollisionGuard | None":
        """The self-collision guard, if one is wired at all."""
        for guard in self._guards:
            if guard.name == self._PATH_AUTHORITY and isinstance(guard, SelfCollisionGuard):
                return guard
        return None

    def planner_hand(self, arm: "RobotArm | None" = None) -> "Maybe[PlannerHand]":
        """The hand this pipeline's self-collision guard models, from ``robot.gripper.model``.

        ``UNSET`` where no self-collision guard is wired, or where it was built without a
        hand and keeps the arm bundle's own.
        """
        guard = self._path_authority(arm)
        return guard.hand if guard is not None else UNSET

    def _say_and_refuse(
        self, message: str, command: MotionCommand
    ) -> "MotionResult | None":
        """Refuse a path, and say so where somebody will read it.

        The guards log when they refuse a sample, and these refusals happen before any
        sample is judged. Without a line here, a cell that cannot judge a path at all
        produces a pass rate of zero and a log full of successful perception, with nothing
        naming the cause. The typed result carries it too, but a runner that reports only
        a rate throws the result away.
        """
        self._logger.warning("Safety preflight refused to judge a path: %s", message)
        return SafetyPreflight.as_motion_result(
            SafetyDecision.reject(
                self._PATH_AUTHORITY,
                SafetyReason.UNAVAILABLE,
                message=message,
                motion_status_override=MotionStatus.UNSUPPORTED,
            ),
            command,
        )

    def _cannot_judge_a_path(
        self, arm: "RobotArm | None", command: MotionCommand
    ) -> "MotionResult | None":
        """Refuse before the first sample where the verdict would be a fiction, else ``None``.

        There are two ways it would be. With no self-collision guard there is nothing to
        ask, and every sample would come back accepted by a pipeline that looked at
        nothing. With the capsule proxy the answer is worse than nothing: the proxy bounds
        the arm links with capsules and cannot see the gripper on a joint-only context, so
        a folded finger passes it. A path is judged on exact meshes at every sample, and a
        gate that quietly degraded would report a check that never ran.
        """
        sentence = self._unjudged_path_sentence(arm)
        return None if sentence is None else self._say_and_refuse(sentence, command)

    def _unjudged_path_sentence(self, arm: "RobotArm | None") -> str | None:
        """The sentence a path gate refuses with before its first sample, or ``None`` where a path can be judged."""
        guard = self._path_authority(arm)
        if guard is None:
            return no_path_guard_refusal()
        if guard.exact_mesh_engine(arm) is None:
            return exact_mesh_path_refusal()
        return None

    def path_judge_refusal(self, arm: "RobotArm | None") -> str | None:
        """Why no path of ``arm`` can be judged sample by sample, or ``None`` where every sample can be.

        The two sentences a path gate refuses with before its first sample, and one where no
        joint's sweep can be bounded, naming why: the guard names no kinematics model, or the hand
        it models has no mesh bundle to read its reach past the flange from. A line on such an arm
        is judged at its start only. Asked before a motion, it moves nothing; the exact mesh engine
        it loads is the guard's own, cached.
        """
        sentence = self._unjudged_path_sentence(arm)
        if sentence is not None:
            return sentence
        if self.joint_radii_mm(arm) is None:
            hand = self.planner_hand(arm)
            guard = self._path_authority(arm)
            model = guard.model_for(arm) if guard is not None else None
            if model and chosen(hand) and self.carried_reach_mm(arm, str(model)) is None:
                return ("the hand this guard models has no mesh bundle to read its reach past the flange from, so a "
                        "line cannot be sampled at a step that covers the hand, and it is judged at its start only")
            return ("the self_collision guard names no kinematics model this repository has joint radii for, so a "
                    "line cannot be sampled at the guard's step and is judged at its start only")
        return None

    def gate_joint_path(
        self,
        samples: "PathSamples",
        *,
        arm: "RobotArm | None" = None,
        command: MotionCommand = MotionCommand.MOVE_JOINTS,
    ) -> "MotionResult | None":
        """Judge every configuration of a path before any of it is commanded.

        ``None`` means every sample passed. There is no off switch and no stride. This
        gate is reached only where a path exists and is about to execute, and a path
        nobody looks at is the hole it closes: a plan that grazes a bench halfway and
        lands clear passes every endpoint check there is.

        The samples carry the step they kept (``step_bound_mm``), so a pass means the path
        is clear to within that much unexamined travel between neighbours, and a refusal
        names the sample.

        Judged before the first configuration moves rather than as they execute, because a
        refusal partway leaves the arm standing on a path already judged unsafe, and there
        is nowhere good to put it.

        ``command`` is the verb the caller is serving, and it is what the refusal carries.
        The guards themselves are always asked about a joint configuration, because that
        is what a sample is, whether it came from a joint move or from the inverse
        kinematics of a line.
        """
        from src.robot.core import JointPositions

        configs = samples.configs
        if not configs:
            return None
        if (refusal := self._cannot_judge_a_path(arm, command)) is not None:
            return refusal

        self.reset()  # a judged path is a restart, as a commanded joint move is
        total = len(configs)
        rejected: "MotionResult | None" = None
        for index, values in enumerate(configs):
            joints = JointPositions(tuple(float(v) for v in values))
            ctx = self.context_for_joints(joints, command=MotionCommand.MOVE_JOINTS, arm=arm)
            for guard in self._guards:
                if guard.name in self._JOINT_MOVE_SKIP_GUARDS:
                    continue
                decision = guard.evaluate(ctx)
                if not decision.rejected:
                    continue
                self._logger.warning(
                    "Safety preflight rejected a PLANNED PATH at sample %d of %d "
                    "(step bound %.3f mm): guard=%s reason=%s message=%s",
                    index + 1, total, samples.step_bound_mm, decision.guard,
                    decision.reason.value, decision.message or "<empty>",
                )
                located = SafetyDecision.reject(
                    decision.guard,
                    decision.reason,
                    message=(
                        f"sample {index + 1} of {total} of a path sampled at "
                        f"{samples.step_bound_mm:.3f} mm a step: {decision.message}"
                        if decision.message
                        else f"sample {index + 1} of {total} of a path sampled at "
                             f"{samples.step_bound_mm:.3f} mm a step"
                    ),
                    detail={
                        **decision.detail,
                        "sample": str(index + 1),
                        "samples": str(total),
                        "step_bound_mm": f"{samples.step_bound_mm:.6f}",
                    },
                )
                rejected = SafetyPreflight.as_motion_result(
                    located, command, target_joints=joints,
                )
                break
            if rejected is not None:
                break
        self.reset()
        return rejected

    @property
    def path_step_mm(self) -> float | None:
        """How far a sampled path may move any point of the arm between two samples, or ``None``.

        It is the collision margin of the self-collision guard. A path sampled more
        coarsely than the margin can pass a link through a body between two samples with
        both of them accepted, so the margin is not one choice among several: it is the
        coarsest sampling the check can survive. ``None`` when no self-collision guard is
        wired, which is also when a path cannot be judged at all.
        """
        guard = self._path_authority(None)
        return None if guard is None else guard.min_distance_mm

    def gate_planned_path(
        self,
        waypoints: "Sequence[Sequence[float]]",
        *,
        arm: "RobotArm | None" = None,
        command: MotionCommand = MotionCommand.MOVE_TO,
    ) -> "MotionResult | None":
        """Sample a planner's whole path and judge every configuration.

        ``None`` means every sample passed. A planner hands over corners. Between two of
        them the arm runs the joint-space line, on a real UR by moveJ-ing to each in turn
        and in the sim by applying each to the articulation, so judging the waypoints
        alone judges the corners and not the path. The legs are sampled at the collision
        margin, which is the coarsest step the check can survive, and the reach the
        sampling needs is read off the arm rather than configured.

        Fail-closed in both new ways it can fail: a robot whose reach does not derive
        cannot be sampled by this rule, and a path too long for the checker's cap is
        refused rather than thinned.
        """
        from .path_samples import waypoint_path_samples

        if not waypoints:
            return None
        if (refusal := self._cannot_judge_a_path(arm, command)) is not None:
            return refusal
        step_mm = self.path_step_mm
        reach_mm = self.joint_radii_mm(arm)
        if reach_mm is None or step_mm is None:
            return self._say_and_refuse(
                f"no reach derives for {type(arm).__name__}, and a path is sampled by how far a "
                f"joint angle can swing a point of the arm. Declare the robot this cell drives as "
                f"safety.self_collision.kinematics_model, which is also what the guard loads its "
                f"meshes for; only the UR models in UR_DH_TABLES_M carry a reach today",
                command,
            )
        try:
            samples = waypoint_path_samples(
                waypoints, reach_mm=reach_mm, max_step_mm=step_mm,
            )
        except ValueError as exc:
            return self._say_and_refuse(str(exc), command)
        return self.gate_joint_path(samples, arm=arm, command=command)

    def self_kinematics(self, arm: "RobotArm | None") -> "tuple[str, float] | None":
        """The model and base yaw this pipeline's self-collision guard places ``arm`` with.

        ``None`` where no guard is wired or no model derives. The self filter stands the
        robot where the guard does. Asking the arm instead mirrors the sim's filter through
        its base: the Isaac cell's guard turns the DH base by 180 degrees, and origins
        placed without that put the robot's own body on the other side of the bench.
        """
        guard = self._path_authority(arm)
        model = guard.model_for(arm) if guard is not None else None
        if guard is None or not model:
            return None
        return str(model), guard.base_yaw_deg

    def joint_radii_mm(self, arm: "RobotArm | None") -> "tuple[float, ...] | None":
        """Per joint, how far ``arm`` can swing a point when that joint turns, or ``None``.

        One radius per joint rather than one for the arm, because the wrist carries a hand
        and the shoulder carries the whole arm: holding both to the shoulder's radius
        samples a wrist move far more densely than the geometry asks, and on a ur5e it is
        the difference between an ordinary six-joint move fitting under the checker's cap
        and being refused by it.

        The model comes from the self-collision guard, which resolves ``kinematics_model``
        first and the arm second. It has to be the same answer: the guard places link
        meshes for one robot and the sampler bounds the sweep of another, and a cell where
        those disagree is judged twice against two arms. It is also the only answer a sim
        cell has, because the Isaac driver reports its model as ``isaac-sim``.

        Each radius also covers what the flange carries. The DH table ends at the flange, and
        its radii alone miss the tool: a wrist turn they count as 10 mm moves a 132 mm tool's
        grasp centre 11.6 mm and its fingertips further. A point beyond the flange is no
        further from a joint's axis than the flange is, plus how far that point is from the
        flange, so each radius is the DH radius plus :meth:`carried_reach_mm`. The last joint
        turns about the tool axis, which runs through the flange, so there a carried point
        counts by how far it stands off that axis rather than by how far it is from the
        flange, and a wrist 3 turn is not sampled four times denser than the hand can move.
        ``None`` where the reach cannot be read, which refuses the path rather than judging it
        at a bound that leaves the hand out.
        """
        from ._ur_kinematics import ur_joint_radii_mm

        guard = self._path_authority(arm)
        model = guard.model_for(arm) if guard is not None else None
        if not model:
            capabilities = getattr(arm, "capabilities", None)
            model = getattr(capabilities, "model", None)
        if not model:
            return None
        radii = ur_joint_radii_mm(str(model))
        carried = self.carried_reach_mm(arm, str(model), off_the_tool_axis=False)
        beside = self.carried_reach_mm(arm, str(model), off_the_tool_axis=True)
        if radii is None or carried is None or beside is None:
            return None
        return tuple(radius + carried for radius in radii[:-1]) + (max(radii[-1], beside),)

    def carried_reach_mm(
        self, arm: "RobotArm | None", model: str, *, off_the_tool_axis: bool = False,
    ) -> "float | None":
        """How far past the flange anything this guard judges on it reaches, in millimetres, or ``None``.

        ``off_the_tool_axis`` measures from the flange's own Z axis instead of from its origin,
        which is what bounds a point's arc when the last joint turns.

        The hand's sphere map as the guard places it (grown to hold its mesh, coupling plates
        included), every wrist camera's fill, and a carried part wherever the cell declares its
        length, attached or not: a bound that holds only while nothing is attached is not a
        bound. ``0.0`` where no hand is named, so there is nothing past the flange for the guard
        to judge. ``None`` where a hand is named and cannot be placed.
        """
        import numpy as np

        from .planning.self_envelope import hand_spheres, payload_capsule

        hand = self.planner_hand(arm)
        if not chosen(hand):
            return 0.0
        spheres = hand_spheres(hand, model)
        if spheres is None:
            return None

        def reach(start: "Sequence[float]", end: "Sequence[float]", radius: float) -> float:
            if off_the_tool_axis:
                return max(float(np.linalg.norm(list(start)[:2])), float(np.linalg.norm(list(end)[:2]))) + float(radius)
            return max(float(np.linalg.norm(start)), float(np.linalg.norm(end))) + float(radius)

        furthest = max(reach(s.start_mm, s.end_mm, s.radius_mm) for s in spheres)
        for wrist in self.wrist_bodies(arm):
            for centre, radius in wrist.envelope_spheres_mm():  # type: ignore[attr-defined]
                furthest = max(furthest, reach(centre, centre, radius))
        world = getattr(getattr(getattr(arm, "config", None), "safety", None), "planning_world", None)
        part = getattr(world, "payload", None)
        if part is not None and bool(getattr(part, "enabled", False)) and getattr(part, "length_mm", None):
            capsule = payload_capsule(hand, spheres, length_mm=float(part.length_mm),
                                      lateral_margin_mm=float(part.lateral_margin_mm))
            furthest = max(furthest, reach(capsule.start_mm, capsule.end_mm, capsule.radius_mm))
        return furthest

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
