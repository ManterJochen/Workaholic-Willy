"""A planned path is judged, and there is no longer a key that turns that off.

This file replaces `test_unchecked_path_is_announced.py`, which pinned the opposite: four shipped
defaults composed into a path nobody looked at, and the repository answered with a warning because a
config-time refusal would have refused the tree it ships. The warning was the right answer to the
question "what do we do about a default we cannot change today". The answer now is that the default
changed: a planner's path is judged configuration by configuration, on every cell, and the switch that
could stand it down is gone from the schema rather than merely set to true.

What that costs is real and is the reason the old default existed: the exact mesh check runs per sample,
and a sampled path has many more samples than the planner had waypoints. What it buys is the only thing
that ever made a planned move safe to execute, which is that somebody looked at the middle of it.

`safety.trajectory_check` is gone whole, `enabled` and `stride` together (the owner, 2026-09-12): the
sampling density comes from the collision margin and the reach, so a second knob beside it would only
be able to make the check sparser than the geometry says it has to be.
"""

from __future__ import annotations

import pathlib
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.core import MotionResult, MotionStatus
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety._fcl_self_collision import mesh_backend_status
from tests._plan_end import OPEN_WORKSPACE, pose_where_it_ends

#: Said where each arm is built: these doubles plan or check with cuRobo and carry no camera world, so every
#: motion they command declines it, as a cuRobo cell must since the world became mandatory.
_DECLINED = "unit double: this test exercises the planner and the guard on a cuRobo arm, and no camera world is wired to it"

#: The folded configuration test_planning_world.py and test_joint_path_gate.py use: a finger in the
#: forearm, which the exact mesh backend refuses with "forearm|lfinger: mesh distance 0.367 mm".
_FOLDED = [1.95, 0.38, -1.33, -0.55, 2.00, 0.79]
#: Clear, and a third of a radian from the fold on one wrist joint, so a leg to it is short.
#: Measured on this box 2026-09-12: the exact mesh backend accepts it and refuses the fold.
_NEAR_CLEAR = [1.95, 0.38, -1.33, -0.55, 1.65, 0.79]
#: A leg whose two ends are clear and whose MIDDLE is not. Found by searching, not constructed:
#: the straight joint space line between these two crosses the fold between t=0.65 and t=0.9.
_LEG_END = [2.403, 0.805, -1.5466, -0.4135, 2.023, 1.6487]
#: Far from all of the above. A whole move to here and back sweeps about 15 m of arm travel, which
#: is over the sampler's cap, and that refusal is its own test below.
_FAR = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]


def _mesh_backend_available() -> bool:
    return mesh_backend_status("ur5e") == "ok"


def _pose() -> Pose:
    return Pose(
        position_mm=np.array([400.0, 0.0, 300.0]),
        quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
        frame=Frame.BASE,
    )


class _FakePlanner:
    """Hands back a fixed trajectory and records whether anybody asked it to execute one."""

    def __init__(self, trajectory: list[list[float]]) -> None:
        self._trajectory = trajectory
        self.executed = False

    @property
    def goal_end(self) -> list[float]:
        """Where the trajectory ends: the one goal a real planner would have handed it back for (Step 8f)."""
        return list(self._trajectory[-1])

    def plan(self, pose: Pose) -> list[list[float]]:
        return self._trajectory

    def check_joint_path(self, samples: object, *, refresh: bool = True) -> object:
        """The planner's half of the path judge. This file is about the local half, so the double accepts."""
        from src.robot.safety.planning import JointCheckVerdict

        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(list(samples)), reason="the double accepts")

    def execute(
        self, traj: object, pose: Pose, *, vel: object = None, acc: object = None
    ) -> MotionResult:
        self.executed = True
        return MotionResult.from_bool(True, MotionCommandStub.MOVE_TO, target_pose=pose)


class MotionCommandStub:
    from src.robot.core import MotionCommand

    MOVE_TO = MotionCommand.MOVE_TO


def _curobo_arm(trajectory: list[list[float]]) -> tuple[URRobotArm, _FakePlanner]:
    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}},
        "gripper": {"model": "robotiq_2f85"},
        # The configurations here were chosen for what the exact meshes say about them, and they put the flange
        # outside the shipped box; the box is not what this file is about.
        "workspace_limits": OPEN_WORKSPACE,
    })
    arm = URRobotArm(config)
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    # The endpoint gate reads the arm's CURRENT joints, so the fake controller has to hold some.
    arm._conn.get_joint_positions.return_value = list(_NEAR_CLEAR)
    planner = _FakePlanner(trajectory)
    arm._curobo_ur = planner  # type: ignore[assignment]  # bypass the lazy builder
    return arm, planner


class APlannedPathIsAlwaysJudgedTests(unittest.TestCase):
    """The middle of a cuRobo plan, on a cell that declared nothing beyond the defaults."""

    def test_a_collision_in_the_middle_refuses_and_nothing_executes(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        arm, planner = _curobo_arm([_NEAR_CLEAR, _FOLDED, _NEAR_CLEAR])
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move(pose_where_it_ends(arm, planner.goal_end))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("sample", result.message or "")
        self.assertFalse(planner.executed, "a path judged unsafe was executed anyway")

    def test_the_waypoints_alone_would_not_have_caught_it(self) -> None:
        """The legs are sampled, so a collision BETWEEN two clear waypoints is caught as well.

        The two waypoints here are both clear. The straight joint-space line between them passes
        through the folded configuration, which is what the arm actually executes and what nothing
        looked at before this step.
        """
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        from src.robot.core import JointPositions

        arm, planner = _curobo_arm([_NEAR_CLEAR, _LEG_END])
        self.enterContext(arm.without_camera_world(_DECLINED))
        endpoints_only = arm.safety_preflight
        assert endpoints_only is not None
        # The control, and it is the whole test: BOTH waypoints pass the endpoint gate, so
        # judging the waypoints alone would have executed this leg.
        for waypoint in (_NEAR_CLEAR, _LEG_END):
            with self.subTest(waypoint=waypoint):
                self.assertIsNone(
                    endpoints_only.gate_joint_target(JointPositions(tuple(waypoint)), arm=arm),
                    "this waypoint is not clear, so the test below proves nothing",
                )
        result = arm.move(pose_where_it_ends(arm, planner.goal_end))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertFalse(planner.executed)

    def test_a_clear_plan_still_executes(self) -> None:
        """The control. Without it every test above would pass on a gate that refuses everything."""
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        arm, planner = _curobo_arm([_NEAR_CLEAR, _NEAR_CLEAR])
        self.enterContext(arm.without_camera_world(_DECLINED))
        arm._gate_planned_config = lambda pose, joints: None  # type: ignore[assignment, misc]
        result = arm.move(pose_where_it_ends(arm, planner.goal_end))
        self.assertTrue(result.ok, result.message)
        self.assertTrue(planner.executed)


class ALongPathIsJudgedRatherThanRefusedTests(unittest.TestCase):
    """⭐ THE PREMISE CHANGED ON 2026-09-12, measured on the M2 Isaac gate.

    This class used to pin that a move sweeping the cell twice was REFUSED for its length, because the
    sampler was capped at 1000, which is the size of one cuRobo request. The M2 gate then refused a
    real plan of 121 waypoints, about one pick in ten, for a length nothing prevented anybody from
    checking: the client splits a long path across requests, and the local exact mesh gate is a loop.
    So a long path is judged now, and what refuses it is what is IN it.

    Nothing is thinned to fit, then or now. There is still a backstop, and it is about how long a
    caller waits rather than about a request size.
    """

    def test_a_move_that_sweeps_the_cell_twice_is_judged_and_refused_for_its_content(self) -> None:
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        arm, planner = _curobo_arm([_FAR, _FOLDED, _FAR])
        self.enterContext(arm.without_camera_world(_DECLINED))
        result = arm.move(pose_where_it_ends(arm, planner.goal_end))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("sample", result.message or "")
        self.assertNotIn("refused rather than thinned", result.message or "")
        self.assertFalse(planner.executed)

    def test_the_control_the_same_two_legs_apart_are_each_judged(self) -> None:
        """Without this the refusal above could be the configurations rather than the length."""
        if not _mesh_backend_available():
            pytest.skip("no exact mesh backend on this box")
        arm, planner = _curobo_arm([_NEAR_CLEAR, _FOLDED])
        self.enterContext(arm.without_camera_world(_DECLINED))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, arm.move(pose_where_it_ends(arm, planner.goal_end)).status)

    def test_the_backstop_still_exists_and_still_refuses(self) -> None:
        """It is a bound on waiting, not a policy, and a path that reaches it still gets a sentence."""
        from src.robot.safety.path_samples import MAX_PATH_SAMPLES, joint_path_samples

        with self.assertRaises(ValueError) as caught:
            joint_path_samples(
                (0.0,) * 6, (40.0,) + (0.0,) * 5, reach_mm=1312.3, max_step_mm=1.0
            )
        self.assertIn(str(MAX_PATH_SAMPLES), str(caught.exception))


class TheOffSwitchIsGoneTests(unittest.TestCase):
    """Not set to true: gone. A tree that still writes it is refused rather than quietly ignored."""

    def test_a_tree_that_writes_enabled_is_refused(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotSafetyConfig

        with self.assertRaises(ValidationError):
            RobotSafetyConfig.model_validate({"trajectory_check": {"enabled": False}})

    def test_a_tree_that_writes_stride_is_refused(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotSafetyConfig

        with self.assertRaises(ValidationError):
            RobotSafetyConfig.model_validate({"trajectory_check": {"stride": 4}})

    def test_the_schema_no_longer_defines_the_block(self) -> None:
        import src.config.schema.robot.safety_schema as schema

        self.assertFalse(hasattr(schema, "TrajectoryCheckConfig"))

    def test_the_shipped_yaml_no_longer_promises_an_unexamined_path(self) -> None:
        """The sentence the old warning existed to say. It is not true of this tree any more."""
        text = pathlib.Path("config/robot/robot.yaml").read_text(encoding="utf-8")
        self.assertNotIn("trajectory_check", text)
        self.assertNotIn("executes unexamined", text)


class TheAttestationReadsThePlannerTests(unittest.TestCase):
    """What a cell attests about paths is now what its arm plans with, not a key anybody could set."""

    def _rendered(self, planner: str, vendor: str = "ur") -> str:
        from src.robot.safety.attestation import SafetyAttestation

        config = RobotConfig.model_validate(
            {"vendor": vendor, "ur": {"motion_planner": planner}, "gripper": {"model": "robotiq_2f85"}} if vendor == "ur"
            else {"vendor": vendor}
        )
        arm = URRobotArm(config) if vendor == "ur" else None
        if arm is None:
            self.skipTest(f"no arm class for {vendor} in this test")
        return SafetyAttestation.of(arm).render()

    def test_a_curobo_ur_attests_endpoints_and_paths(self) -> None:
        self.assertIn("endpoints and paths", self._rendered("curobo"))

    def test_an_ik_ur_attests_endpoints_only(self) -> None:
        """An interpolated move has no path to judge, and saying otherwise would be a claim too many."""
        self.assertIn("endpoints only", self._rendered("ik"))


if __name__ == "__main__":
    unittest.main()
