"""A checked line starts at the configuration the arm is in, and each sample is solved from the one before.

On the dense cell every run was refused at sample 0 of 16 with ``cuRobo refuses the start of this move: forearm_link
and wrist_2_link overlap by 9.5 mm``, while the arm stood in a configuration cuRobo itself had planned. The configuration
being refused was not the one the arm was in: the line re-solved its own start pose with inverse kinematics, and for a
pose that many branches reach, the branch chosen is not the branch the arm stands in.

The first sample of a line is therefore the arm's own joints, not a solution for its own pose, and every later sample is
solved seeded on the sample before it. Judging anything else judges a path that starts somewhere the arm is not, and the
walk that follows jumps there in one step.

With the start repaired, sample 1 still sat 3.14 rad from sample 0: the wrist flipped into the other family, and cuRobo
refused that configuration. The dense runner switches on the natural aim seed, whose sweep the resolver tried before the
seed it was handed, and the first solution inside the tolerance wins. So a seeded solve tries its own seed first, and a
line whose samples still jump is refused before it is judged: the samples of a judged path are a path only if each
follows from the one before.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.sim.config import SimRobotConfig

_HERE = (0.1, -1.5, 1.5, 0.0, 1.5, 0.2)
#: What a re-solve of the same pose answers: the same tool pose on another branch of the arm.
_OTHER_BRANCH = (0.1, -1.4, 1.6, 0.3, 1.5, 0.2)


def _pose(x: float) -> Pose:
    return Pose(position_mm=np.array([x, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE, label="line")


class _Client:
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    dt = 0.01

    def __init__(self) -> None:
        self.checked: list[list[float]] = []

    def check_joints(self, configs: Any) -> Any:
        from src.robot.safety.planning import JointCheckVerdict

        self.checked = [[float(v) for v in config] for config in configs]
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(self.checked), reason="clear")


def _arm() -> tuple[IsaacRobotArm, _Client, list[Any]]:
    arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
    arm._connected = True
    arm._articulation = object()  # type: ignore[assignment]
    arm.get_joint_positions = lambda: JointPositions(_HERE)  # type: ignore[method-assign]
    arm.get_tcp_pose = lambda: _pose(300.0)  # type: ignore[method-assign]
    seeds: list[Any] = []

    def resolve(pose: Pose, **kwargs: Any) -> JointPositions:
        seeds.append(kwargs.get("seed"))
        return JointPositions(_OTHER_BRANCH)

    arm._resolve_ik = resolve  # type: ignore[method-assign]
    client = _Client()
    arm._get_curobo_client = lambda: client  # type: ignore[method-assign, assignment, return-value]
    arm._drive_joints = MagicMock()  # type: ignore[method-assign]
    arm._to_client_joint_order = lambda configs, client: [list(c) for c in configs]  # type: ignore[method-assign]
    guard = MagicMock()
    guard.path_step_mm = 10.0
    guard.joint_radii_mm.return_value = (500.0,) * 6
    guard.gate_joint_path.return_value = None
    guard.path_judge_refusal.return_value = None
    arm._preflight = guard
    return arm, client, seeds


class ACheckedLineStartsWhereTheArmStandsTests(unittest.TestCase):
    def test_the_first_judged_configuration_is_the_arm_s_own(self) -> None:
        arm, client, _ = _arm()

        arm._drive_checked_line(_pose(400.0))

        judged = arm._preflight.gate_joint_path.call_args.args[0].configs  # type: ignore[union-attr]
        np.testing.assert_allclose(judged[0], _HERE, atol=1e-12)
        np.testing.assert_allclose(client.checked[0], _HERE, atol=1e-12)
        self.assertGreater(len(judged), 2, "a line of one sample proves nothing")

    def test_each_sample_is_solved_from_the_one_before(self) -> None:
        arm, _, seeds = _arm()

        arm._drive_checked_line(_pose(400.0))

        self.assertTrue(seeds, "no sample was solved")
        np.testing.assert_allclose(np.asarray(seeds[0], dtype=np.float64), _HERE, atol=1e-12)
        for seed in seeds[1:]:
            np.testing.assert_allclose(np.asarray(seed, dtype=np.float64), _OTHER_BRANCH, atol=1e-12)


#: The arm's own configuration with its wrist flipped into the other family: the same tool pose, 3.14 rad away.
_FLIPPED = (0.1, -1.2, 1.7, 2.0, -1.5, 0.2 - np.pi)


class ALineStaysOnItsBranchTests(unittest.TestCase):
    def test_a_seeded_solve_tries_its_seed_first(self) -> None:
        arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
        arm._connected = True
        arm._natural_aim_seed = True
        arm._preflight = None
        tried: list[Any] = []

        def ik(pose: Pose, *, seed: Any = None) -> JointPositions:
            tried.append(np.asarray(seed.values, dtype=np.float64))
            return JointPositions(_HERE)

        arm.ik = ik  # type: ignore[method-assign]
        arm.fk = lambda joints: _pose(400.0)  # type: ignore[method-assign]
        arm.get_joint_positions = lambda: JointPositions(_OTHER_BRANCH)  # type: ignore[method-assign]

        arm._resolve_ik(_pose(400.0), seed=np.asarray(_HERE))

        np.testing.assert_allclose(tried[0], _HERE, atol=1e-12)

    def test_an_unseeded_solve_keeps_its_order(self) -> None:
        """The control: a pose move hands no seed, and the natural aim sweep still comes first."""
        arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
        arm._connected = True
        arm._natural_aim_seed = True
        arm._preflight = None
        tried: list[Any] = []

        def ik(pose: Pose, *, seed: Any = None) -> JointPositions:
            tried.append(np.asarray(seed.values, dtype=np.float64))
            return JointPositions(_HERE)

        arm.ik = ik  # type: ignore[method-assign]
        arm.fk = lambda joints: _pose(400.0)  # type: ignore[method-assign]
        arm.get_joint_positions = lambda: JointPositions(_HERE)  # type: ignore[method-assign]

        arm._resolve_ik(_pose(400.0))

        self.assertAlmostEqual(float(np.arctan2(0.0, 400.0)), float(tried[0][0]), places=12)
        self.assertAlmostEqual(-1.6, float(tried[0][1]), places=12)

    def test_a_line_that_jumps_its_branch_is_refused_before_it_is_judged(self) -> None:
        from src.robot.core import MotionStatus

        arm, client, _ = _arm()
        answers = iter([_HERE, _FLIPPED])
        arm._resolve_ik = lambda pose, **kwargs: JointPositions(next(answers, _FLIPPED))  # type: ignore[method-assign]

        result = arm._drive_checked_line(_pose(400.0))

        self.assertIs(MotionStatus.IK_FAILED, result.status, result.message)
        self.assertIn("sample 3 of", result.message or "")
        self.assertIn("rad", result.message or "")
        arm._preflight.gate_joint_path.assert_not_called()  # type: ignore[union-attr]
        self.assertEqual([], client.checked)


class TheSimLineIsJudgedFilledAndWalkedAsSampledTests(unittest.TestCase):
    """The sim walks each sample in turn, so between two of them it runs the joint line, whose sum |dq| * r can
    exceed the bound the gate is told threefold. The gate judges that line at the line's own step, and the walk
    still drives only the samples."""

    def test_every_judged_step_keeps_the_bound_and_only_the_samples_are_walked(self) -> None:
        arm, _, _ = _arm()

        result = arm._drive_checked_line(_pose(400.0))

        judged = arm._preflight.gate_joint_path.call_args.args[0]  # type: ignore[union-attr]
        walked = arm._drive_joints.call_count  # type: ignore[attr-defined]
        self.assertGreater(len(judged.configs), walked, "nothing between the samples was judged")
        for index, (a, b) in enumerate(zip(judged.configs, judged.configs[1:])):
            with self.subTest(step=index):
                self.assertLessEqual(sum(abs(y - x) * 500.0 for x, y in zip(a, b)), judged.step_bound_mm + 1e-6)
        self.assertIn(f"{walked} samples judged as {len(judged.configs)} configurations", result.message or "")

    def test_the_sim_reads_the_bound_both_drivers_share(self) -> None:
        from src.robot.safety.path_samples import LINE_MAX_JOINT_STEP_RAD

        self.assertEqual(LINE_MAX_JOINT_STEP_RAD, IsaacRobotArm._LINE_MAX_JOINT_STEP_RAD)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
