"""cuRobo-planned motion for a real UR arm: plan in safety.planning, execute over ur_rtde.

It is the real-hardware counterpart of the cuRobo path in the Isaac sim driver,
:meth:`src.robot.drivers.sim.arm.IsaacRobotArm._drive_curobo`. It asks the
process-isolated cuRobo planner, :class:`~src.robot.safety.planning.CuroboPlanClient`,
for a global collision-free joint trajectory to a Cartesian goal, then executes that
trajectory on the UR controller waypoint by waypoint through the ``ur_rtde`` ``moveJ``.

It is fail-closed. Where the cuRobo environment or service is unavailable, or no
collision-free plan exists, it does not fall back to blind IK: it returns a typed
failure, :attr:`MotionStatus.CONTROLLER_REJECTED` or :attr:`MotionStatus.TIMEOUT`, so a
real cell never moves on an unplanned path.

The scope is honest. The cuRobo round-trip against a real robot and a real cuRobo GPU
environment is bucket 3, because neither exists here. The planning and execution logic,
meaning the goal conversion, the joint-order remap, the waypoint execution and the
fail-closed branches, is exercised through an injected planner and connection. Planner
collision-awareness is not a certified functional-safety stop, and a real cell still
needs the vendor safety-rated stop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.geometry import Frame, Pose
from src.robot.constants import UR_CUROBO_LOG_FILE, create_robot_logger
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.safety.planning import CuroboPlanClient, CuroboUnavailableError

if TYPE_CHECKING:  # pragma: no cover (typing only)
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
    """

    def __init__(
        self,
        connection: URConnection,
        *,
        client_factory: Callable[[], CuroboPlanClient] = CuroboPlanClient,
        vel: float | None = None,
        acc: float | None = None,
        world_cuboids: Sequence[dict[str, object]] | None = None,
        require_registration: bool = True,
    ) -> None:
        self._conn = connection
        self._client_factory = client_factory
        self._vel = vel
        self._acc = acc
        self._client: CuroboPlanClient | None = None
        self._world_cuboids = [dict(c) for c in (world_cuboids or ())]
        self._require_registration = bool(require_registration)
        self._world_registered = False
        #: Collision-sphere slots the sidecar reserves for a carried payload. At 0 the
        #: sidecar robot config is untouched and `attach_payload` refuses, which is the
        #: unchanged path.
        self._attach_spheres = 0
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
        costs a JIT warm-up. It is registered once: the sidecar keeps its world between
        plans, so re-sending it on every move would pay that cost for nothing.
        """
        if self._client is None:
            client = self._client_factory()
            if self._attach_spheres > 0:
                # The sphere budget has to be in the sidecar environment before it
                # spawns, because it decides which robot config gets built. It is set
                # here rather than in the factory, so a caller that injects its own
                # factory still gets the behaviour.
                setter = getattr(client, "reserve_attach_spheres", None)
                if callable(setter):
                    setter(self._attach_spheres)
            client.start()
            self._client = client
            self._register_world(client)
        return self._client

    def _register_world(self, client: CuroboPlanClient) -> None:
        """Hand the cell's obstacles to the planner, or refuse to plan at all."""
        if self._world_registered or not self._world_cuboids:
            return
        count = client.set_world(list(self._world_cuboids))
        self._world_registered = True
        if count == len(self._world_cuboids):
            self.logger.info("registered %d cell obstacle(s) with the planner", count)
            return
        message = (
            f"the planner confirmed {count} of {len(self._world_cuboids)} declared cell obstacle(s). "
            "It is now planning against a world that is missing some of your cell, and an obstacle it "
            "never received is one it will route straight through."
        )
        if self._require_registration:
            raise CuroboUnavailableError(
                message + " Refusing to plan. Set safety.planning_world.require_registration false to "
                "plan anyway, deliberately."
            )
        self.logger.error("%s Planning anyway: require_registration is off.", message)

    def attach_payload(
        self, joints: "Sequence[float]", dims_mm: "Sequence[float]", offset_mm: float
    ) -> bool:
        """Tell the planner the gripper is carrying a box, so later plans route the box around too.

        ``offset_mm`` is how far beyond the flange the centre of the part sits. It
        returns ``False`` where the planner could not attach, including a sidecar
        started with no sphere budget: a cell that cannot model its payload carries on
        and says so rather than stopping mid-pick.
        """
        dims_m = [float(d) / 1000.0 for d in dims_mm]
        pose = [0.0, 0.0, float(offset_mm) / 1000.0, 1.0, 0.0, 0.0, 0.0]
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
        """Register scene obstacles into the cuRobo collision world, or 0 where the planner is away."""
        try:
            return self._client_or_start().set_world(cuboids)
        except CuroboUnavailableError as exc:
            # The client line about planning continuing against the previous world sits
            # after the call, so an unavailable sidecar skips it and the scene obstacles
            # would vanish quietly.
            self.logger.error("collision world NOT set, %d cuboid(s) dropped (%s); the planner is "
                         "routing against whatever world it last had", len(cuboids), exc)
            return 0

    def close(self) -> None:
        """Shut down the planning server (idempotent)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Planning + execution
    # ------------------------------------------------------------------

    def plan(self, pose: Pose) -> list[list[float]] | None:
        """Plan from tool0 to ``pose`` in BASE with cuRobo, returning the trajectory in UR order.

        It returns ``None`` where no plan exists.

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
        start = self._to_client_order(current_ur, client.joint_names)
        traj = client.plan(start, goal_pos_m, goal_quat_wxyz)
        if not traj:
            return None
        return [self._to_ur_order(list(wp), client.joint_names) for wp in traj]

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

    def move(self, pose: Pose, *, vel: float | None = None, acc: float | None = None) -> MotionResult:
        """Plan from tool0 to ``pose`` in BASE with cuRobo and execute it, failing closed."""
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"CuroboUrPlanner requires Frame.BASE; got {pose.frame!r}",
            )
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="CuroboUrPlanner requires an open UR connection.",
            )
        try:
            traj_ur = self.plan(pose)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}",
                exception=exc,
            )
        if not traj_ur:
            return MotionResult.failed(
                MotionStatus.TIMEOUT,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="cuRobo found no collision-free plan; failing safe (no blind motion).",
            )
        return self.execute(traj_ur, pose, vel=vel, acc=acc)

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
