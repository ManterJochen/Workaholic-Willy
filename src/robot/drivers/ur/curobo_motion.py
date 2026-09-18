"""cuRobo-planned motion for a real UR arm: plan in safety.planning, execute over ur_rtde.

It is the real-hardware counterpart of the cuRobo path in the Isaac sim driver,
:meth:`src.robot.drivers.sim.arm.IsaacRobotArm._drive_curobo`. It asks the
process-isolated cuRobo planner, :class:`~src.robot.safety.planning.CuroboPlanClient`,
for a global collision-free joint trajectory to a Cartesian goal, then executes that
trajectory on the UR controller waypoint by waypoint through the ``ur_rtde`` ``moveJ``.
Every pose reaches the planner through :class:`.planner_frame.PlannerFrameClient`,
because the planner is rooted half a turn about Z from the controller's base.

It is fail-closed. Where the cuRobo environment or service is unavailable, or no
collision-free plan exists, nothing falls back to blind IK: :meth:`CuroboUrPlanner.plan`
raises or returns ``None``, and the UR arm turns that into a typed failure,
:attr:`MotionStatus.CONTROLLER_REJECTED` or :attr:`MotionStatus.TIMEOUT`, so a real cell
never moves on an unplanned path.

The scope is honest. The cuRobo round-trip against a real robot and a real cuRobo GPU
environment is bucket 3, because neither exists here. The planning and execution logic,
meaning the goal conversion, the joint-order remap, the waypoint execution and the
fail-closed branches, is exercised through an injected planner and connection. Planner
collision-awareness is not a certified functional-safety stop, and a real cell still
needs the vendor safety-rated stop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src.contracts import UNSET, Maybe
from src.geometry import Pose
from src.robot.constants import UR_CUROBO_LOG_FILE, create_robot_logger
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.safety.planning import (
    CuroboPlanClient,
    CuroboUnavailableError,
    JointCheckVerdict,
)
from src.robot.safety.planning.curobo_client import SidecarIdentity, StateRefusal
from src.robot.safety.planning.live_world import WorldRefresh, refresh_planner_world
from src.robot.safety.planning.world import merge_planner_worlds

from .planner_frame import PlannerFrameClient

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.core.keep_out import GoalKeepOut
    from src.robot.safety.planning.reservation import PlannerReservation
    from src.robot.safety.planning.live_world import LivePlannerWorld
    from src.robot.safety.planning.perceived import SelfEnvelope

    from .connection import URConnection

__all__ = ["CuroboUrPlanner", "UR_ARM_JOINT_NAMES"]

#: The UR arm joint order, base to wrist_3: the order ``ur_rtde`` reports from
#: ``getActualQ`` and expects for ``moveJ``.
UR_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class CuroboUrPlanner:
    """Plan a UR joint trajectory with cuRobo and execute it over ``ur_rtde``, fail-closed.

    Parameters
    ----------
    connection
        A connected :class:`~src.robot.drivers.ur.connection.URConnection`, supplying
        ``is_connected``, ``get_joint_positions`` and ``moveJ``.
    client_factory
        Builds the :class:`CuroboPlanClient`. It is the injection seam for the planner.
    vel / acc
        The default joint speed and acceleration for each executed ``moveJ`` waypoint.
    world_cuboids
        The static obstacles of the cell, from
        :func:`~src.robot.safety.planning.world.build_planner_cuboids`. They are
        registered once, the first time the planner is reached. The default of none
        leaves the planner with the world it booted with, which is its own robot model
        plus a generic table and nothing of this cell.
    require_registration
        With a world declared and this true, the default, a registration the planner
        does not confirm in full raises :class:`CuroboUnavailableError` instead of
        planning. The client returns 0 and logs that planning continues against the
        previous world, and an obstacle the planner never received is one it will route
        straight through.
    world_meshes
        The declared meshes of the cell, from
        :func:`~src.robot.safety.planning.world.build_planner_meshes`, registered with the
        boxes, so a tote declared as a mesh reaches the planner without a live world.
    reservation
        What the sidecar allocates when it starts, handed to the client before ``start``
        whichever factory built it.
    """

    def __init__(
        self,
        connection: URConnection,
        *,
        client_factory: Callable[[], CuroboPlanClient] = CuroboPlanClient,
        vel: float | None = None,
        acc: float | None = None,
        world_cuboids: Sequence[dict[str, object]] | None = None,
        world_meshes: Sequence[dict[str, object]] | None = None,
        require_registration: bool = True,
        live_world: "LivePlannerWorld | None" = None,
        self_envelope: "Callable[[], SelfEnvelope | None] | None" = None,
        on_perceived_obstacles: "Callable[[Sequence[Any]], object] | None" = None,
        reservation: "PlannerReservation | None" = None,
        descriptor_check: "Callable[[SidecarIdentity], str | None] | None" = None,
    ) -> None:
        self._conn = connection
        self._client_factory = client_factory
        self._vel = vel
        self._acc = acc
        self._client: CuroboPlanClient | None = None
        self._world_cuboids = [dict(c) for c in (world_cuboids or ())]
        self._world_meshes = [dict(m) for m in (world_meshes or ())]
        self._require_registration = bool(require_registration)
        self._world_registered = False
        #: The cell as the cameras see it, asked immediately before every plan. `None` is the
        #: unchanged path: the declared world is registered once and never revisited.
        self._live_world = live_world
        #: This arm's own body now, so the camera's view of the robot can be taken back out.
        self._self_envelope = self_envelope
        #: The last refresh, for a report and for an operator asking why a move was refused.
        self._last_refresh: "WorldRefresh | None" = None
        #: Where the perceived obstacles go besides the planner. The path guard lives on the arm
        #: rather than here, and the two have to be looking at the same cell. Whatever it returns is
        #: ignored: the preflight answers how many guards took the boxes, which is a number for a
        #: report rather than a decision to make here.
        self._on_perceived_obstacles = on_perceived_obstacles
        #: Collision-sphere slots the sidecar reserves for a carried payload. At 0 the
        #: sidecar robot config is untouched and `attach_payload` refuses, which is the
        #: unchanged path.
        self._attach_spheres = 0
        #: What the sidecar allocates when it starts, told to the client before `start`. None leaves
        #: the client as its factory built it.
        self._reservation = reservation
        #: Asked of the descriptor the sidecar reports once it is ready: a sentence refuses the
        #: planner. ``None`` asks nothing, which is what a caller that injects its own client gets.
        self._descriptor_check = descriptor_check
        self.logger = create_robot_logger("CuroboUrPlanner", UR_CUROBO_LOG_FILE)

    def enable_payload(self, sphere_slots: int) -> None:
        """Reserve collision spheres for a carried part. It is called before the planner starts."""
        self._attach_spheres = max(0, int(sphere_slots))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _client_or_start(self) -> CuroboPlanClient:
        """Spawn and JIT-warm the cuRobo server lazily, raising where the environment is missing.

        The declared world is registered here rather than at construction, because
        construction happens on a cell that may never plan and starting the sidecar
        costs a JIT warm-up. It is sent once and then kept: the sidecar holds its world
        between plans, so re-sending an unchanged world on every move would pay that cost
        for nothing. A cell with a live world refreshes it separately, before each plan,
        which is a different thing from this and deliberately not free.
        """
        # Registration is retried until it succeeds rather than attempted once when the
        # client is built. Hanging it off the construction meant a refused registration was
        # never tried again: the second move found a built client, skipped registration
        # entirely, and planned against the very world the first refusal was about.
        # `_register_world` returns immediately once the planner has confirmed the world in
        # full, so the retry costs nothing on the normal path.
        if self._client is not None:
            self._register_world(self._client)
            return self._client

        if self._client is None:
            # The planner's base is not the controller's. Every pose this glue hands the planner
            # is in the controller's DH base, and the planner is rooted at UR's URDF base_link,
            # half a turn away, so the client is seen through the turn from here on: goals,
            # boxes, meshes and the live scene go in turned, and a pose the planner reports comes
            # back turned. Joints are the same numbers in both.
            client = cast("CuroboPlanClient", PlannerFrameClient(self._client_factory()))
            if self._reservation is not None:
                # For the same reason as the spheres below, and first: the slots have to be in the
                # sidecar environment before it spawns, and an injected factory never sees the config.
                reserve = getattr(client, "reserve_world", None)
                if callable(reserve):
                    reserve(self._reservation)
            if self._attach_spheres > 0:
                # The sphere budget has to be in the sidecar environment before it
                # spawns, because it decides which robot config gets built. It is set
                # here rather than in the factory, so a caller that injects its own
                # factory still gets the behaviour.
                setter = getattr(client, "reserve_attach_spheres", None)
                if callable(setter):
                    setter(self._attach_spheres)
            client.start()
            refusal = (
                self._descriptor_check(SidecarIdentity.from_client(client))
                if self._descriptor_check is not None else None
            )
            if refusal is not None:
                # Closed rather than kept: the next move would find a started client and plan with it.
                client.close()
                self.logger.error("cuRobo planner refused: %s", refusal)
                raise CuroboUnavailableError(refusal)
            self._client = client
            self._register_world(client)
        return self._client

    def _register_world(self, client: CuroboPlanClient) -> None:
        """Hand the cell's obstacles to the planner, or refuse to plan at all.

        The latch is set only after the planner confirmed the world in full. Set before the
        refusal is raised, it refuses the first move and lets every later one find the client
        already built, skip this entirely, and plan against the partial world the refusal was
        about. A guard that fires once and then waves everything through afterwards is worse
        than either answer.
        """
        if self._world_registered or not (self._world_cuboids or self._world_meshes):
            return
        expected = len(self._world_cuboids) + len(self._world_meshes)
        # Meshes are named only when there are some: a client from before meshes existed takes one
        # argument, and a cell that declares none should not need a newer one.
        count = (
            client.set_world(list(self._world_cuboids), [dict(m) for m in self._world_meshes])
            if self._world_meshes else client.set_world(list(self._world_cuboids))
        )
        if count == expected:
            self._world_registered = True
            self.logger.info("registered %d cell obstacle(s) with the planner", count)
            return
        message = (
            f"the planner confirmed {count} of {expected} declared cell obstacle(s). "
            "It is now planning against a world that is missing some of your cell, and an obstacle it "
            "never received is one it will route straight through."
        )
        if self._require_registration:
            raise CuroboUnavailableError(
                message + " Refusing to plan. Set safety.planning_world.require_registration false to "
                "plan anyway, deliberately."
            )
        self.logger.error("%s Planning anyway: require_registration is off.", message)

    def set_live_world(self, world: "LivePlannerWorld | None") -> None:
        """Point this planner at a different live world, or at none.

        `None` restores the behaviour of registering the declared world once and planning
        against it for the life of the cell, which is what a planner built without one does.
        """
        self._live_world = world

    def _refresh_world(
        self, *, near_point_mm: "Sequence[float] | None" = None, goal_keep_out: "Maybe[GoalKeepOut]" = UNSET,
    ) -> None:
        """Hand the planner the cell as the cameras see it, or refuse to plan.

        Called immediately before every plan rather than once at startup. A world
        registered once is a photograph, and everything that arrived in the cell afterwards
        is invisible to the planner while looking exactly like empty space in every log
        there is.

        Without a live world configured this does nothing at all, and the declared world
        registered at startup is what the planner keeps.

        Raises
        ------
        CuroboUnavailableError
            Where the world cannot be vouched for: no camera answered, the frame is older
            than the cell allows, or the planner confirmed fewer boxes than it was sent.
            Refusing is the safety layer's rule everywhere else, and the alternative here is
            driving against a world that had time to go out of date.
        """
        if self._live_world is None:
            return
        envelope = self._self_envelope() if self._self_envelope is not None else None
        try:
            refresh = refresh_planner_world(
                source=self._live_world,
                client=self._client_or_start(),
                self_envelope=envelope,
                near_point_mm=near_point_mm,
                require_registration=self._require_registration,
                goal_keep_out=goal_keep_out,
            )
        except CameraWorldUnavailable as exc:
            # Raised on, not refused: a camera that stayed silent, blind or stale through its attempts
            # is a fault of the cell. The guard is emptied as on any refused refresh, and the fault is
            # written down once, here, where the camera and the attempts are known.
            if self._on_perceived_obstacles is not None:
                self._on_perceived_obstacles(())
            self.logger.error("%s", exc)
            raise
        self._last_refresh = refresh
        # Including the empty list on a refusal: a guard left holding boxes from a refused
        # refresh is checking a cell that no longer exists.
        if self._on_perceived_obstacles is not None:
            self._on_perceived_obstacles(refresh.guard_boxes)
        if not refresh.ok:
            # Logged here, once: the caller turns the raise into CONTROLLER_REJECTED, and this line
            # is the only place a person reads why the move was refused.
            self.logger.error("%s", refresh.render())
            raise CuroboUnavailableError(
                f"the planner world could not be refreshed, so this move is refused. "
                f"{refresh.render()}"
            )
        self.logger.info("%s", refresh.render())

    def refresh_world(
        self, *, near_point_mm: "Sequence[float] | None" = None, goal_keep_out: "Maybe[GoalKeepOut]" = UNSET,
    ) -> None:
        """Refresh the planner world now, for a caller that judges its own path before asking the planner.

        The arm's path guard learns the perceived obstacles from this refresh, so the arm calls it
        before that guard runs and then asks :meth:`check_joint_path` with ``refresh=False``: one
        camera reading per motion, and both authorities judging the same cell. Raises what
        :meth:`_refresh_world` raises.
        """
        self._refresh_world(near_point_mm=near_point_mm, goal_keep_out=goal_keep_out)

    @property
    def last_world_refresh(self) -> "WorldRefresh | None":
        """What the most recent refresh did, for a report and for an operator asking why."""
        return self._last_refresh

    def attach_payload(
        self, joints: "Sequence[float]", dims_mm: "Sequence[float]", centre_mm: "Sequence[float]"
    ) -> bool:
        """Tell the planner the gripper is carrying a box, so later plans route the box around too.

        ``dims_mm`` are the sides of the box along the tool0 axes and ``centre_mm`` is its
        centre in tool0, as ``self_envelope.carried_part_box`` places them, along the hand's placed
        approach. A distance along tool0 +Z alone would not follow a hand that approaches along
        another axis, as the Isaac cell's hand does. It
        returns ``False`` where the planner could not attach, including a sidecar
        started with no sphere budget: a cell that cannot model its payload carries on
        and says so rather than stopping mid-pick.
        """
        dims_m = [float(d) / 1000.0 for d in dims_mm]
        pose = [*(float(c) / 1000.0 for c in centre_mm), 1.0, 0.0, 0.0, 0.0]
        try:
            return self._client_or_start().attach_payload(list(joints), dims_m, pose)
        except CuroboUnavailableError as exc:
            # The client logs that it is planning as if the gripper were empty when the
            # sidecar answers with a refusal. Where the sidecar is gone the exception
            # flies past that line, so without this the one case in which nothing is
            # modelled would be the one case nobody is told about.
            self.logger.error("payload NOT attached (%s); the planner is routing as if the gripper were "
                         "empty", exc)
            return False

    def detach_payload(self) -> bool:
        """Take the carried box off the planner's model."""
        if self._client is None:
            return True
        try:
            return self._client.detach_payload()
        except CuroboUnavailableError as exc:
            self.logger.error("payload NOT detached (%s); the planner will keep routing around a part "
                         "the gripper no longer holds", exc)
            return False

    def set_world(self, cuboids: list[dict]) -> int:
        """Register scene obstacles into the cuRobo collision world, or 0 where the planner is away.

        The declared world goes underneath. Registration replaces the planner's world
        rather than extending it, so without the merge a caller sending its own boxes
        deletes the bench, the bin and every fixture this cell declared, with nothing
        logged and nothing to notice it by. The simulator driver merges the same way, and
        this is the side whose arm can hurt someone.
        """
        merged = merge_planner_worlds(self._world_cuboids, cuboids)
        try:
            count = self._client_or_start().set_world(merged)
            if count == len(merged):
                self._world_registered = True
            return count
        except CuroboUnavailableError as exc:
            # The client line about planning continuing against the previous world sits
            # after the call, so an unavailable sidecar skips it and the scene obstacles
            # would vanish quietly.
            self.logger.error("collision world NOT set, %d cuboid(s) dropped (%s); the planner is "
                         "routing against whatever world it last had", len(merged), exc)
            return 0

    def start(self) -> SidecarIdentity:
        """Start the planning server the way the first planned move does, and say what it loaded.

        Every refusal that move meets on the way is raised here as it would be there, a
        ``CuroboUnavailableError`` naming it: the client factory's (margin, hand, retract row), the
        descriptor and evidence check, and the world registration. A started server is kept, so a move
        after this plans on it.
        """
        return SidecarIdentity.from_client(self._client_or_start())

    def close(self) -> None:
        """Shut down the planning server (idempotent)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Planning + execution
    # ------------------------------------------------------------------

    def plan(self, pose: Pose, *, goal_keep_out: "Maybe[GoalKeepOut]" = UNSET) -> list[list[float]] | None:
        """Plan from tool0 to ``pose`` in BASE with cuRobo, returning the trajectory in UR order.

        It returns ``None`` where no plan exists. ``goal_keep_out`` is the space between the jaws at
        the goal, which the refresh before the plan leaves out.

        Raises
        ------
        CuroboUnavailableError
            If the cuRobo environment or service cannot be brought up, and the caller
            then fails closed.
        """
        goal_pos_m = (np.asarray(pose.position_mm, dtype=np.float64) / 1000.0).tolist()
        qx, qy, qz, qw = (float(v) for v in pose.quaternion_xyzw)
        goal_quat_wxyz = [qw, qx, qy, qz]

        current_ur = [float(v) for v in self._conn.get_joint_positions()]
        client = self._client_or_start()
        # The goal decides which obstacles matter when the slot budget bites, so it goes in
        # as plain numbers rather than as whatever array type the pose happens to carry.
        self._refresh_world(near_point_mm=[float(v) for v in pose.position_mm], goal_keep_out=goal_keep_out)
        start = self._to_client_order(current_ur, client.joint_names)
        traj = client.plan(start, goal_pos_m, goal_quat_wxyz)
        if not traj:
            # The caller turns this into one fail-safe sentence, so the pair of links the planner
            # could not get past is recoverable from here alone. Nothing on the wire changes: the
            # verdict is still no plan.
            refusal = self.last_refusal
            if refusal is not None:
                self.logger.warning("cuRobo refused this move. %s", refusal.render())
            return None
        return [self._to_ur_order(list(wp), client.joint_names) for wp in traj]

    @property
    def last_refusal(self) -> "StateRefusal | None":
        """Why the sidecar refused what it was last asked, or ``None``.

        A client that does not carry the typed reason has none.
        """
        return getattr(self._client, "last_refusal", None)

    def check_joint_path(
        self, samples_ur: "Sequence[Sequence[float]]", *, refresh: bool = True
    ) -> JointCheckVerdict:
        """Ask cuRobo whether every configuration of a joint path is admissible, in UR joint order.

        The same planner, the same world and the same attached payload the Cartesian path
        gets, so a joint move is judged against the cell the planner actually holds rather
        than against a second model that agreed with it when somebody last looked. The world
        is refreshed first for the reason :meth:`_refresh_world` gives: a world registered
        once is a photograph.

        Judged in one request. Returning after the first refused sample would cost a round
        trip per configuration, and the sidecar already answers with the index it stopped at.

        ``refresh=False`` skips the refresh, for a caller that ran :meth:`refresh_world` itself
        before its own path guard judged the same samples. The arm does, so that guard is not
        judging against the obstacles of the motion before.

        Raises
        ------
        CuroboUnavailableError
            If the cuRobo env cannot be brought up or the world cannot be vouched for. Fail
            closed: the caller gets no verdict rather than an invented one.
        """
        configs = [[float(v) for v in sample] for sample in samples_ur]
        if not configs:
            return JointCheckVerdict(
                valid=True, first_invalid=None, checked=0,
                reason="an empty path has nothing to refuse",
            )
        client = self._client_or_start()
        if refresh:
            # The goal decides which obstacles matter when the slot budget bites, and the goal of
            # a joint path is its last configuration. Its flange position is not known here
            # without FK, so this refresh runs without a near point and without a goal region. The
            # arm refreshes near the goal flange itself, with the region at the goal's TCP, and asks
            # with `refresh=False`.
            self._refresh_world()
        ordered = [self._to_client_order(list(c), client.joint_names) for c in configs]
        return client.check_joints(ordered)

    def execute(
        self,
        traj_ur: list[list[float]],
        pose: Pose,
        *,
        vel: float | None = None,
        acc: float | None = None,
    ) -> MotionResult:
        """Execute a UR-order joint trajectory waypoint-by-waypoint via ``moveJ``."""
        v = vel if vel is not None else self._vel
        a = acc if acc is not None else self._acc
        try:
            for waypoint in traj_ur:
                ok = self._conn.moveJ(list(waypoint), vel=v, acc=a)
                if not ok:
                    return MotionResult.failed(
                        MotionStatus.CONTROLLER_REJECTED,
                        MotionCommand.MOVE_TO,
                        target_pose=pose,
                        message="UR moveJ rejected a cuRobo trajectory waypoint.",
                    )
        except (RuntimeError, OSError) as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"UR moveJ raised executing the cuRobo trajectory: {exc}",
                exception=exc,
            )
        self.logger.info("cuRobo trajectory executed on UR: %d waypoints", len(traj_ur))
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose, message="curobo")

    # ------------------------------------------------------------------
    # Joint-order remap between the planner and UR
    # ------------------------------------------------------------------

    @staticmethod
    def _to_client_order(ur_joints: list[float], client_names: list[str]) -> list[float]:
        """Reorder UR-order joints into the planner joint order, or identity where the names are unknown."""
        if not client_names or set(client_names) != set(UR_ARM_JOINT_NAMES):
            return [float(v) for v in ur_joints]
        idx = {name: i for i, name in enumerate(UR_ARM_JOINT_NAMES)}
        return [float(ur_joints[idx[name]]) for name in client_names]

    @staticmethod
    def _to_ur_order(client_joints: list[float], client_names: list[str]) -> list[float]:
        """Reorder planner-order joints back into UR joint order, or identity where the names are unknown."""
        if not client_names or set(client_names) != set(UR_ARM_JOINT_NAMES):
            return [float(v) for v in client_joints]
        pos = {name: i for i, name in enumerate(client_names)}
        return [float(client_joints[pos[name]]) for name in UR_ARM_JOINT_NAMES]
