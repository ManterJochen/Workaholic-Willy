"""The real UR's planner sees the cell and the goal where the controller has them (Step 8f).

The planner is rooted at UR's URDF ``base_link``, the DH base turned half a turn about Z; the controller reports in the
DH base. Measured on the box on 2026-09-18 (``scripts/curobo/probe_payload_attach.py``, ``frame``): the planner's tool0
is the DH flange turned 180 degrees to 0.0001 mm, and 1019 mm from it without the turn. The UR driver handed the
planner its goals and the cell's geometry in the controller's frame with no turn, so on a controller it would have
planned to the goal mirrored through the base axis and routed around a mirrored cell, and nothing checked where a plan
ended before the arm drove it. The Isaac cell never met it: its world, its goals and its planner all sit in base_link.

These pin the turn at the one place the UR glue talks to the planner, every channel through it, and the check that
refuses a plan whose last configuration is not on its goal before the first waypoint moves. The UR driver no longer
hands the planner a pose at all: it chooses the goal configuration and asks for a joint plan, whose numbers are the
same in either base, so a goal can no longer be mirrored on the way in; the turn still stands for the world, the
live scene and a pose anything else hands the turned client.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.geometry import Frame, Pose
from src.robot.core import MotionStatus
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.drivers.ur.planner_frame import PlannerFrameClient, planner_pose
from src.robot.safety.planning import JointCheckVerdict


class _RecordingClient:
    """A planner client that records every pose it is handed, in whatever frame it arrives."""

    def __init__(self, traj: "list[list[float]] | None" = None) -> None:
        self.joint_names = list(UR_ARM_JOINT_NAMES)
        self.dt = 0.0
        self._traj = traj if traj is not None else [[0.0] * 6]
        self.goal: "tuple[list[float], list[float]] | None" = None
        self.joint_goal: "list[float] | None" = None
        self.cuboids: list[dict] = []
        self.meshes: list[dict] = []
        self.voxels: "dict[str, Any] | None" = None
        self.scene: "tuple[list[dict], list[dict], dict | None] | None" = None

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def plan(self, start: Any, pos_m: Any, quat_wxyz: Any) -> list[list[float]]:
        self.goal = (list(pos_m), list(quat_wxyz))
        return self._traj

    def plan_joint(self, start: Any, goal: Any) -> list[list[float]]:
        self.joint_goal = list(goal)
        return self._traj

    def set_world(self, cuboids: Any, meshes: Any = None) -> int:
        self.cuboids = [dict(c) for c in cuboids]
        self.meshes = [dict(m) for m in meshes or ()]
        return len(self.cuboids) + len(self.meshes)

    def set_voxels(self, path: Any, **kwargs: Any) -> int:
        self.voxels = {"path": path, **kwargs}
        return 1

    def set_scene(self, cuboids: Any, meshes: Any, voxels: Any) -> Any:
        self.scene = ([dict(c) for c in cuboids], [dict(m) for m in meshes or ()], voxels)
        return None

    def fk(self, joints: Any) -> tuple[list[float], list[float]]:
        return [0.5, 0.1, 0.3], [0.0, 0.0, 0.0, 1.0]

    def check_joints(self, configs: Any, **_: Any) -> JointCheckVerdict:
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="the fake accepts")


class _Conn:
    is_connected = True

    def __init__(self, joints: "list[float] | None" = None) -> None:
        self._joints = joints or [0.0] * 6
        self.moves: list[list[float]] = []

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def moveJ(self, joints: Any, vel: Any = None, acc: Any = None) -> bool:  # noqa: N802 - ur_rtde's name
        self.moves.append([float(v) for v in joints])
        return True


# -- the turn ------------------------------------------------------------------------------------------------------


def test_a_half_turn_mirrors_the_position_through_the_base_axis_and_turns_the_rotation() -> None:
    turned = planner_pose([0.5, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0])
    assert turned == pytest.approx([-0.5, -0.1, 0.05, 0.0, 0.0, 0.0, 1.0])


def test_the_half_turn_is_its_own_inverse() -> None:
    q = np.array([0.3, -0.2, 0.5, 0.78])
    q = q / np.linalg.norm(q)
    pose = [0.4, -0.3, 0.2, *q]
    back = planner_pose(planner_pose(pose))
    assert back[:3] == pytest.approx(pose[:3])
    # The same rotation: a quaternion and its negation are one.
    assert abs(float(np.dot(back[3:], pose[3:]))) == pytest.approx(1.0)


def test_the_turned_rotation_is_the_half_turn_times_the_pose() -> None:
    from src.geometry.quaternion import to_rotation_matrix

    q_xyzw = np.array([0.1, 0.7, -0.2, 0.68])
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    turned = planner_pose([0.0, 0.0, 0.0, *wxyz])
    turned_xyzw = np.array([turned[4], turned[5], turned[6], turned[3]])
    half = np.diag([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(to_rotation_matrix(turned_xyzw), half @ to_rotation_matrix(q_xyzw), atol=1e-12)


# -- every channel through the UR glue -----------------------------------------------------------------------------


def _goal(x: float = 500.0, y: float = 100.0, z: float = 300.0) -> Pose:
    return Pose(position_mm=np.array([x, y, z]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                frame=Frame.BASE, label="goal")


def _ur_planner(client, **kwargs):  # noqa: ANN001, ANN003, ANN202 - a stand-in client, typed as the real one's glue
    return CuroboUrPlanner(_Conn(), client_factory=lambda: client, **kwargs)


def test_a_pose_handed_to_the_turned_client_reaches_the_planner_turned() -> None:
    client = _RecordingClient()
    PlannerFrameClient(client).plan([0.0] * 6, [0.5, 0.1, 0.3], [1.0, 0.0, 0.0, 0.0])
    assert client.goal is not None
    pos_m, quat_wxyz = client.goal
    assert pos_m == pytest.approx([-0.5, -0.1, 0.3]), "the planner was handed the goal in the controller's frame"
    assert quat_wxyz == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_a_joint_goal_reaches_the_planner_as_it_is() -> None:
    """What the UR glue sends: the configuration the arm chose, the same numbers in either base."""
    client = _RecordingClient()
    _ur_planner(client).plan_joint([0.1, -1.2, 1.3, -0.4, 0.5, 0.6])
    assert client.joint_goal == pytest.approx([0.1, -1.2, 1.3, -0.4, 0.5, 0.6])
    assert client.goal is None, "a pose went to the planner"


def test_a_declared_box_and_mesh_reach_the_planner_turned() -> None:
    client = _RecordingClient()
    box: dict[str, Any] = {"name": "bench", "dims_m": [0.2, 0.2, 0.2], "pose": [0.5, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0]}
    mesh: dict[str, Any] = {"name": "tote", "file_path": "tote.stl", "pose": [0.3, -0.4, 0.0, 1.0, 0.0, 0.0, 0.0],
                            "scale": [1.0, 1.0, 1.0]}
    planner = _ur_planner(client, world_cuboids=[box], world_meshes=[mesh])
    planner.check_joint_path([[0.0] * 6])
    assert client.cuboids[0]["pose"] == pytest.approx([-0.5, -0.1, 0.05, 0.0, 0.0, 0.0, 1.0])
    assert client.meshes[0]["pose"] == pytest.approx([-0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert box["pose"] == [0.5, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0], "the declared box itself was changed"


def test_the_live_scene_channels_reach_the_planner_turned() -> None:
    client = _RecordingClient()
    turned = PlannerFrameClient(client)
    field = {"path": "f.npy", "dims_m": [0.3] * 3, "voxel_size_m": 0.01, "pose": [0.5, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0]}
    turned.set_voxels(field["path"], dims_m=field["dims_m"], voxel_size_m=0.01, pose=field["pose"])
    assert client.voxels is not None and client.voxels["pose"] == pytest.approx([-0.5, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
    box = {"name": "seen_00", "dims_m": [0.1] * 3, "pose": [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0]}
    turned.set_scene([box], None, field)
    assert client.scene is not None
    assert client.scene[0][0]["pose"] == pytest.approx([-0.2, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0])
    assert client.scene[2] is not None
    assert client.scene[2]["pose"] == pytest.approx([-0.5, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0])


def test_what_the_planner_reports_comes_back_in_the_controllers_frame() -> None:
    pos, quat = PlannerFrameClient(_RecordingClient()).fk([0.0] * 6)
    assert pos == pytest.approx([-0.5, -0.1, 0.3])
    assert quat == pytest.approx([-1.0, 0.0, 0.0, 0.0])


def test_the_carried_part_reaches_the_planner_unturned() -> None:
    """The control for the turn: the payload box is placed in tool0, which is the same frame in either base."""
    seen: dict = {}

    class _Attaching(_RecordingClient):
        def attach_payload(self, joints: Any, dims_m: Any, pose: Any) -> bool:
            seen.update(dims=list(dims_m), pose=list(pose))
            return True

    assert PlannerFrameClient(_Attaching()).attach_payload([0.0] * 6, [0.04, 0.04, 0.12], [0.0, 0.0, 0.2, 1.0, 0, 0, 0])
    assert seen["pose"] == [0.0, 0.0, 0.2, 1.0, 0, 0, 0]
    assert seen["dims"] == [0.04, 0.04, 0.12]


def test_a_channel_the_client_lacks_stays_absent() -> None:
    """The live world asks whether a client HAS a scene call; the turn must not answer yes for a client that has none."""

    class _Bare:
        def plan(self, *a: Any) -> None:
            return None

    turned = PlannerFrameClient(_Bare())
    assert getattr(turned, "set_scene", None) is None
    assert getattr(turned, "set_voxels", None) is None


def test_joints_reach_the_planner_as_they_are() -> None:
    """The control, green before and after: a joint vector is the same numbers in either base."""
    seen: list = []

    class _Checking(_RecordingClient):
        def check_joints(self, configs: Any) -> JointCheckVerdict:
            seen.extend(list(c) for c in configs)
            return super().check_joints(configs)

    planner = _ur_planner(_Checking())
    planner.check_joint_path([[0.1, -1.2, 1.3, -0.4, 0.5, 0.6]])
    assert seen == [pytest.approx([0.1, -1.2, 1.3, -0.4, 0.5, 0.6])]


# -- the plan's end, before anything moves ---------------------------------------------------------------------------

_DOWN = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
#: The same flange pose turned half a turn about the base axis: the plan a planner in the other base would return.
_MIRRORED = [math.pi, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


class _Planner:
    """Answers every joint goal with ``traj``; screens pass and straight lines come too close, so the move plans."""

    def __init__(self, traj: "list[list[float]]") -> None:
        self.traj = traj
        self.executed = False
        self.last_refusal = None

    def plan_joint(self, goal: Any, **_: Any) -> list[list[float]]:
        return self.traj

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        count = len(list(samples))
        if clearance_mm > 0.0:
            return JointCheckVerdict(valid=False, first_invalid=0, checked=count, reason="the line grazes the tote")
        return JointCheckVerdict(valid=True, first_invalid=None, checked=count, reason="the fake accepts")

    def execute(self, traj_ur: Any, pose: Any, **_: Any) -> Any:
        from src.robot.core import MotionCommand, MotionResult

        self.executed = True
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose, message="curobo")


def _ur_arm(planner: _Planner) -> Any:
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
        "gripper": {"model": "robotiq_2f85"},
        "safety": {"payload": {"enforce": False}, "self_collision": {"planner_margin_mm": 4.0}},
    }))
    arm._conn = _Conn(_DOWN)  # type: ignore[assignment]
    arm._curobo_ur = planner  # type: ignore[assignment]
    return arm


def _flange_pose(q: "list[float]") -> Pose:
    from src.geometry.quaternion import from_rotation_matrix
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm

    frames = ur_link_transforms_mm("ur5e", np.asarray(q, dtype=np.float64))
    assert frames is not None
    flange = np.asarray(frames[-1])
    return Pose(position_mm=flange[:3, 3], quaternion_xyzw=from_rotation_matrix(flange[:3, :3]),
                frame=Frame.BASE, label="flange")


def test_a_plan_that_ends_off_its_goal_is_refused_before_anything_moves() -> None:
    """The end check, read on the controller's own kinematics, refuses a configuration a metre from the goal."""
    goal = _flange_pose([0.0, -1.4, 1.6, -1.7, -1.5708, 0.0])
    mirrored_end = [math.pi, -1.4, 1.6, -1.7, -1.5708, 0.0]
    arm = _ur_arm(_Planner([_DOWN, mirrored_end]))

    result = arm._plan_end_refusal(goal, mirrored_end, goal)

    assert result is not None and result.status is MotionStatus.CONTROLLER_REJECTED
    assert "from its goal" in (result.message or "")


def test_a_plan_that_ends_off_the_configuration_asked_for_is_never_driven() -> None:
    """Through the arm: the plan ends half a turn of the base from what it was asked, so it is not taken at all."""
    goal = _flange_pose([0.0, -1.4, 1.6, -1.7, -1.5708, 0.0])
    mirrored_end = [math.pi, -1.4, 1.6, -1.7, -1.5708, 0.0]
    planner = _Planner([_DOWN, mirrored_end])
    arm = _ur_arm(planner)
    arm._gate_planned_config = lambda pose, joints: None  # the shipped box is not what this reads

    result = arm._drive_curobo(goal)

    assert result.status is MotionStatus.TIMEOUT, result.message
    assert not planner.executed, "a plan that ends a metre from its goal was driven"


def test_a_refused_plan_end_is_never_dropped_by_an_endpoint_gate_that_passes() -> None:
    """A refusal is a falsy ``MotionResult`` (``__bool__`` is ``ok``), so ``end_check or endpoint_gate`` dropped the end
    check's refusal and drove the plan wherever the endpoint gate passed (found porting 85f082b to dev, 2026-09-25)."""
    from src.robot.core import MotionCommand, MotionResult

    q_goal = [0.0, -1.4, 1.6, -1.7, -1.5708, 0.0]
    planner = _Planner([_DOWN, q_goal])
    arm = _ur_arm(planner)
    arm._gate_planned_config = lambda pose, joints: None  # the endpoint gate passes
    arm._plan_end_refusal = lambda goal, end, pose: MotionResult.failed(  # type: ignore[method-assign]
        MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, message="the plan ends 1019.3 mm from its goal")

    result = arm._drive_curobo(_flange_pose(q_goal))

    assert result.status is MotionStatus.CONTROLLER_REJECTED, result.message
    assert "from its goal" in (result.message or "")
    assert not planner.executed, "a plan the end check refused was driven"


def test_a_plan_that_ends_on_its_goal_passes_the_end_check() -> None:
    """The control, green before and after: a plan landing on its goal is not refused by the end check."""
    q_goal = [0.1, -1.4, 1.5, -1.6, -1.5, 0.2]
    arm = _ur_arm(_Planner([_DOWN, q_goal]))
    assert arm._plan_end_refusal(_flange_pose(q_goal), q_goal, _flange_pose(q_goal)) is None
