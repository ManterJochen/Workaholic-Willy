"""A line is solved on the controller's own kinematics where the cell asks, and the controller is asked at two samples
(``robot.safety.ik_quality.line_ik: local``, the owner, 2026-10-09).

A line a cuRobo UR judges before a ``moveL`` has every sample solved by the controller's ``getInverseKinematics``, seeded
on the one before, so the configurations judged are the ones the controller will pass through: 28 round trips for the
80 mm line down, 35 for the 100 mm lift, 32.5 ms each on the cell. The repository's closed form is not that solver on the
owner's arm: its controller is calibrated, and the nominal table puts the TCP 6 mm from where it does. So the driver reads
the controller's own rows (its primary interface's kinematics info, nothing sent) and its active TCP, solves every sample
on them here, nearest the seed, and asks the controller at the first and the last sample solved whether it answers the
same within 1e-6 rad. Anything else, and the line is walked again with the controller solving it, as before.

What this file pins, on the real UR driver over the test bench, its controller a stand-in that solves on a calibrated
chain by a solver of its own (scipy's least squares from the closed form, nearest the seed):

* a line judged from where the arm stands and a line judged ahead hand the gate the controller's configurations to
  1e-9 rad, with two round trips instead of 28 and 8;
* a wrong calibration, rows or a TCP that cannot be read, and a sample that cannot be vouched for here each leave the
  line to the controller, configuration for configuration as without the switch;
* a line refused on its way is refused in the same words;
* the controller solves every sample unless the cell asks, and the key loads;
* the solver itself: the nominal chain is the guards' table, a calibrated chain is solved nearest the seed to a
  nanoradian (a chain holding frames metres along parallel axes as well), and a seed as near two configurations, a pose
  out of reach and a wrist at its singularity are not vouched for.

Measured on URSim CB3 3.15.8 (UR10, the owner's TCP set on it) through ``URRobotArm._judge_linear_move``: with a
calibration, 80 lines judged ahead and 16 judged from where the arm stood gave the same verdicts and the same
configurations to 1.7e-10 rad with 217 round trips instead of 1198; without one, 60 and 12 to 9.1e-11 rad with 151
instead of 872; with rows 1 mm off what the controller computes with, all 46 lines fell back and judged as before.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionStatus, RobotKinematicsError
from src.robot.drivers.ur.connection import ControllerKinematics
from src.robot.drivers.ur.pose import URPose
from src.robot.drivers.ur.pose_adapter import pose_to_urpose
from src.robot.safety._ur_ik import ChainSolution, ur_chain_ik_nearest, ur_flange_ik, ur_pose_matrix_m
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M, URDhChain, ur_link_transforms_mm
from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import _GRASP, _STANDOFF, _cell
from tests.test_a_line_judged_ahead_is_solved_at_knots import _recorded
from tests.test_every_verb_meets_the_camera_world import _mesh_backend_available

#: What a factory calibration adds to the table, of the size a CB3's does (theta and alpha in rad, a and d in m): the
#: deltas URSim was given in a calibration.conf to measure this, which moved its flange 3.2 mm from the table's.
_DELTAS = {
    "theta_rad": (0.0012, -0.0021, 0.0035, -0.0009, 0.0017, 0.0008),
    "a_m": (0.00031, 0.00083, -0.00127, 0.00022, -0.00041, 0.0),
    "d_m": (0.00067, 0.0011, -0.0009, 0.00052, 0.00038, -0.00029),
    "alpha_rad": (0.0011, -0.0007, 0.0013, -0.0016, 0.0009, 0.0),
}


def _rows(model: str, *, a2_off_m: float = 0.0, deltas: "dict[str, tuple[float, ...]] | None" = None) -> ControllerKinematics:
    """The kinematics info a controller of ``model`` with ``deltas`` (the calibration above) reports."""
    table = UR_DH_TABLES_M[model]
    add = _DELTAS if deltas is None else deltas
    a = [row.a_m + da for row, da in zip(table, add["a_m"])]
    a[1] += a2_off_m
    return ControllerKinematics(
        theta_rad=tuple(add["theta_rad"]), a_m=tuple(a),
        d_m=tuple(row.d_m + dd for row, dd in zip(table, add["d_m"])),
        alpha_rad=tuple(row.alpha_rad + dal for row, dal in zip(table, add["alpha_rad"])),
        checksums=(0xFFFFFFFF,) * 6, calibration_status=0, port=30011,
    )


def _flange_m(rows: ControllerKinematics, q: "np.ndarray") -> "np.ndarray":
    """The rows' flange, by a product written out here rather than borrowed from the code under test."""
    T = np.eye(4)
    for j in range(6):
        ct, st = math.cos(q[j] + rows.theta_rad[j]), math.sin(q[j] + rows.theta_rad[j])
        ca, sa = math.cos(rows.alpha_rad[j]), math.sin(rows.alpha_rad[j])
        T = T @ np.array([[ct, -st * ca, st * sa, rows.a_m[j] * ct], [st, ct * ca, -ct * sa, rows.a_m[j] * st],
                          [0.0, sa, ca, rows.d_m[j]], [0.0, 0.0, 0.0, 1.0]])
    return T


def _solved_on(rows: ControllerKinematics, model: str, flange_mm: "np.ndarray", start: "np.ndarray") -> "np.ndarray":
    """What a controller on ``rows`` answers: the closed form's configuration nearest ``start``, then least squares on
    the rows' own flange to the last bit."""
    from scipy.optimize import least_squares

    solutions = ur_flange_ik(model, flange_mm, q6_if_singular=float(start[-1]))
    if not solutions:
        raise RobotKinematicsError("no solution")
    turned = [start + (np.asarray(s) - start + math.pi) % (2.0 * math.pi) - math.pi for s in solutions]
    guess = min(turned, key=lambda q: float(np.sum((q - start) ** 2)))
    goal = np.asarray(flange_mm, dtype=np.float64).copy()
    goal[:3, 3] /= 1000.0

    def residual(q: "np.ndarray") -> "np.ndarray":
        return (_flange_m(rows, q) - goal)[:3, :].ravel()

    return least_squares(residual, guess, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15).x


def _controller_on(arm: Any, rows: ControllerKinematics, solved: "list[Pose]") -> Any:
    """The controller's inverse kinematics on ``rows`` with the declared tool as its active TCP; every pose asked is
    recorded in ``solved``."""
    model = str(arm.config.ur.model)
    tool = np.asarray(arm._declared_tool_matrix(), dtype=np.float64)

    def ik(pose: Pose, *, seed: "JointPositions | None" = None) -> JointPositions:
        solved.append(pose)
        start = np.asarray(seed.tolist() if seed is not None else arm._conn.get_joint_positions(), dtype=np.float64)
        flange = np.asarray(pose.to_matrix(), dtype=np.float64) @ np.linalg.inv(tool)
        return JointPositions(tuple(float(v) for v in _solved_on(rows, model, flange, start)))

    return ik


def _standing(arm: Any, rows: ControllerKinematics, joints: "list[float]") -> None:
    """The controller reports the arm at ``joints`` and its TCP where ``rows`` put it."""
    arm._conn.get_joint_positions.return_value = [float(v) for v in joints]
    flange = _flange_m(rows, np.asarray(joints, dtype=np.float64))
    flange[:3, 3] *= 1000.0
    tcp = flange @ np.asarray(arm._declared_tool_matrix(), dtype=np.float64)
    arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(tcp, frame=Frame.BASE))


def _asking(arm: Any, line_ik: str) -> None:
    """The arm's config with ``robot.safety.ik_quality.line_ik`` at ``line_ik``."""
    ik = arm.config.safety.ik_quality.model_copy(update={"line_ik": line_ik})
    arm.config = arm.config.model_copy(update={"safety": arm.config.safety.model_copy(update={"ik_quality": ik})})


class _Line:
    """The test bench's cuRobo UR, its controller on a calibrated chain, standing over the part."""

    def __init__(self, *, reported: "ControllerKinematics | None" = None, tcp: "list[float] | None | str" = "tool",
                 start: bool = False) -> None:
        self.arm, _events, self.solved = _cell()
        model = str(self.arm.config.ur.model)
        self.rows = _rows(model)
        self.arm.ik = _controller_on(self.arm, self.rows, self.solved)
        near = [float(v) for v in self.arm.nearest_configuration(_STANDOFF).values]
        self.near = [float(v) for v in _solved_on(
            self.rows, model, np.asarray(self.arm._pose_to_flange(_STANDOFF).to_matrix()), np.asarray(near))]
        _standing(self.arm, self.rows, self.near)
        self.start = (_STANDOFF, self.near) if start else None
        self.arm._conn.controller_kinematics.return_value = self.rows if reported is None else reported
        tool = URPose.from_T(np.asarray(self.arm._declared_tool_matrix(), dtype=np.float64)).to_ur_list()
        self.arm._conn.active_tcp_offset.return_value = tool if tcp == "tool" else tcp
        self.handed = _recorded(self.arm)

    def judged(self, line_ik: str, goal: Pose = _GRASP) -> "tuple[Any, list[np.ndarray], int]":
        _asking(self.arm, line_ik)
        self.handed.clear()
        self.solved.clear()
        if self.start is None:
            refused = self.arm._judge_linear_move(goal, command=MotionCommand.MOVE_TO, commanded=False)
        else:
            refused = self.arm._judge_linear_move(goal, command=MotionCommand.MOVE_TO, commanded=False,
                                                  start=self.start)
        return refused, [np.asarray(samples.configs) for samples in self.handed], len(self.solved)


def _apart(a: "list[np.ndarray]", b: "list[np.ndarray]") -> float:
    assert len(a) == len(b) and all(x.shape == y.shape for x, y in zip(a, b)), "the gate was handed other paths"
    return max((float(np.max(np.abs(x - y))) for x, y in zip(a, b) if x.size), default=0.0)


class ALineIsSolvedOnTheControllersOwnKinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_a_line_from_where_the_arm_stands_hands_the_gate_the_controllers_configurations_for_two_round_trips(
            self) -> None:
        """⭐ Red before: the controller solved all 28 samples."""
        line = _Line()
        by_the_controller, configs, asked = line.judged("controller")
        here, local, asked_here = line.judged("local")
        self.assertIsNone(by_the_controller)
        self.assertIsNone(here)
        self.assertEqual(28, asked)
        self.assertEqual(2, asked_here)
        self.assertLess(_apart(configs, local), 1e-9)

    def test_a_line_judged_ahead_keeps_its_knots_and_asks_the_controller_twice(self) -> None:
        line = _Line(start=True)
        _, configs, asked = line.judged("controller")
        _, local, asked_here = line.judged("local")
        self.assertEqual(8, asked)
        self.assertEqual(2, asked_here)
        self.assertLess(_apart(configs, local), 1e-9)

    def test_the_first_and_the_last_sample_are_the_two_the_controller_is_asked(self) -> None:
        line = _Line()
        line.judged("controller")
        every = list(line.solved)
        line.judged("local")
        self.assertEqual([every[0].position_mm.tolist(), every[-1].position_mm.tolist()],
                         [pose.position_mm.tolist() for pose in line.solved])

    def test_a_wrong_calibration_has_the_line_solved_by_the_controller_configuration_for_configuration(self) -> None:
        model = "ur5e"
        line = _Line(reported=_rows(model, a2_off_m=0.001))
        _, configs, asked = line.judged("controller")
        with self.assertLogs("URRobotArm", level="WARNING") as said:
            _, fallen_back, asked_here = line.judged("local")
        self.assertEqual([c.tolist() for c in configs], [c.tolist() for c in fallen_back])
        self.assertEqual(1 + asked, asked_here, "the first sample's answer parts, and the controller solves the rest")
        self.assertIn("solved by the controller as before", "\n".join(said.output))
        self.assertIn("more than 1e-06 rad", "\n".join(said.output))

    def test_the_nominal_table_reported_for_a_calibrated_controller_is_a_wrong_calibration_too(self) -> None:
        line = _Line(reported=_rows("ur5e", deltas={k: (0.0,) * 6 for k in _DELTAS}))
        _, configs, asked = line.judged("controller")
        _, fallen_back, asked_here = line.judged("local")
        self.assertEqual([c.tolist() for c in configs], [c.tolist() for c in fallen_back])
        self.assertEqual(1 + asked, asked_here)

    def test_rows_or_a_tcp_that_cannot_be_read_leave_the_line_to_the_controller(self) -> None:
        for name, line in (("no rows", _Line()), ("no TCP", _Line(tcp=None))):
            with self.subTest(read=name):
                if name == "no rows":
                    line.arm._conn.controller_kinematics.return_value = None
                _, configs, asked = line.judged("controller")
                _, local, asked_here = line.judged("local")
                self.assertEqual(asked, asked_here)
                self.assertEqual([c.tolist() for c in configs], [c.tolist() for c in local])

    def test_a_sample_that_cannot_be_vouched_for_hands_the_whole_line_to_the_controller(self) -> None:
        line = _Line()
        _, configs, asked = line.judged("controller")
        calls = {"n": 0}

        def declining(*args: Any, **kwargs: Any) -> ChainSolution:
            calls["n"] += 1
            if calls["n"] == 5:
                return ChainSolution(None, "another configuration stands about as near the seed")
            return ur_chain_ik_nearest(*args, **kwargs)

        with patch("src.robot.safety._ur_ik.ur_chain_ik_nearest", declining), \
                self.assertLogs("URRobotArm", level="WARNING") as said:
            _, local, asked_here = line.judged("local")
        self.assertEqual(asked, asked_here, "no sample is checked; the controller solves every one")
        self.assertEqual([c.tolist() for c in configs], [c.tolist() for c in local])
        self.assertIn("cannot be vouched for here", "\n".join(said.output))

    def test_a_line_refused_on_its_way_is_refused_in_the_same_words(self) -> None:
        line = _Line()
        inside = line.arm._guard.is_inside_workspace
        line.arm._guard.is_inside_workspace = lambda pose: float(pose.z) > 160.0 and inside(pose)
        by_the_controller, _, asked = line.judged("controller")
        here, _, asked_here = line.judged("local")
        assert by_the_controller is not None and here is not None
        self.assertIs(MotionStatus.WORKSPACE_REJECTED, here.status)
        self.assertEqual(by_the_controller.message, here.message)
        self.assertLess(asked_here, asked)

    def test_the_controller_solves_every_sample_unless_the_cell_asks(self) -> None:
        line = _Line()
        _, _, asked = line.judged("controller")
        self.assertEqual(28, asked)
        line.arm._conn.controller_kinematics.assert_not_called()
        line.arm._conn.active_tcp_offset.assert_not_called()

    def test_the_key_loads_and_defaults_to_the_controller(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        self.assertEqual("controller", RobotConfig.model_validate({"vendor": "ur"}).safety.ik_quality.line_ik)
        asked = RobotConfig.model_validate({"vendor": "ur", "safety": {"ik_quality": {"line_ik": "local"}}})
        self.assertEqual("local", asked.safety.ik_quality.line_ik)
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"vendor": "ur", "safety": {"ik_quality": {"line_ik": "closed_form"}}})


def _random_configurations(count: int, seed: int) -> "list[np.ndarray]":
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        q = rng.uniform(-math.pi, math.pi, 6)
        q[2] = rng.uniform(0.3, 2.8) * rng.choice((-1.0, 1.0))  # away from the stretched elbow
        q[4] = rng.uniform(0.3, 2.8) * rng.choice((-1.0, 1.0))  # and from wrist 2 at 0 or pi
        out.append(q)
    return out


class TheControllersChainIsSolvedNearestTheSeedTests(unittest.TestCase):
    def test_the_nominal_chain_is_the_table_the_guards_place_the_arm_with(self) -> None:
        for model in UR_DH_TABLES_M:
            chain = URDhChain.nominal(model)
            assert chain is not None
            for q in _random_configurations(50, 7):
                with self.subTest(model=model):
                    table = np.asarray(ur_link_transforms_mm(model, q)[-1], dtype=np.float64)
                    table[:3, 3] /= 1000.0
                    self.assertLess(float(np.max(np.abs(chain.flange_m(q) - table))), 1e-15)
        self.assertIsNone(URDhChain.nominal("ur7"))

    def test_a_calibrated_chain_is_solved_nearest_the_seed_to_a_nanoradian(self) -> None:
        for model in ("ur10", "ur5e", "ur3"):
            rows = _rows(model)
            chain = URDhChain(rows.theta_rad, rows.a_m, rows.d_m, rows.alpha_rad)
            rng = np.random.default_rng(11)
            vouched = 0
            for q in _random_configurations(200, 3):
                seed = q + rng.normal(scale=0.01, size=6)
                answer = ur_chain_ik_nearest(model, chain, _flange_m(rows, q), seed)
                if answer.joints is None:
                    continue
                vouched += 1
                with self.subTest(model=model, q=q.tolist()):
                    self.assertLess(float(np.max(np.abs(np.asarray(answer.joints) - q))), 1e-9)
            self.assertGreater(vouched, 180, f"{model}: the solver vouched for too few")

    def test_a_chain_holding_frames_metres_along_parallel_axes_is_solved_as_its_flange_is(self) -> None:
        """A calibration may hold the frames of the shoulder, the elbow and wrist 1, whose axes are nearly parallel, far
        along them: a ``d`` of metres the next row cancels. Here 40 m out and back, the same flange."""
        table = UR_DH_TABLES_M["ur10"]
        far = URDhChain(theta_rad=(0.0,) * 6, a_m=tuple(row.a_m for row in table),
                        d_m=(table[0].d_m, 40.0, -40.0, table[3].d_m, table[4].d_m, table[5].d_m),
                        alpha_rad=tuple(row.alpha_rad for row in table))
        nominal = URDhChain.nominal("ur10")
        assert nominal is not None
        rng = np.random.default_rng(5)
        for q in _random_configurations(100, 9):
            with self.subTest(q=q.tolist()):
                self.assertLess(float(np.max(np.abs(far.flange_m(q) - nominal.flange_m(q)))), 1e-12)
                answer = ur_chain_ik_nearest("ur10", far, nominal.flange_m(q), q + rng.normal(scale=0.01, size=6))
                if answer.joints is not None:
                    self.assertLess(float(np.max(np.abs(np.asarray(answer.joints) - q))), 1e-9)

    def test_a_seed_as_near_two_configurations_is_not_vouched_for(self) -> None:
        chain = URDhChain.nominal("ur10")
        assert chain is not None
        q = np.array([0.4, -1.3, 1.6, -1.2, 1.1, 0.3])
        goal = chain.flange_m(q)
        goal_mm = goal.copy()
        goal_mm[:3, 3] *= 1000.0
        other = [np.asarray(s) for s in ur_flange_ik("ur10", goal_mm) if float(np.max(np.abs(np.asarray(s) - q))) > 0.1]
        elbow_down = min(other, key=lambda s: float(np.sum((s - q) ** 2)))
        answer = ur_chain_ik_nearest("ur10", chain, goal, (q + elbow_down) / 2.0)
        self.assertIsNone(answer.joints)
        self.assertIn("about as near the seed", answer.why_not)

    def test_a_pose_out_of_reach_is_not_vouched_for(self) -> None:
        chain = URDhChain.nominal("ur10")
        assert chain is not None
        goal = np.eye(4)
        goal[:3, 3] = (2.5, 0.0, 0.5)
        answer = ur_chain_ik_nearest("ur10", chain, goal, [0.0, -1.5, 1.5, 0.0, 1.5, 0.0])
        self.assertIsNone(answer.joints)
        self.assertIn("reaches this pose", answer.why_not)

    def test_a_wrist_at_its_singularity_is_not_vouched_for(self) -> None:
        rows = _rows("ur10")
        chain = URDhChain(rows.theta_rad, rows.a_m, rows.d_m, rows.alpha_rad)
        q = np.array([0.4, -1.3, 1.6, -1.2, 1e-5, 0.3])
        answer = ur_chain_ik_nearest("ur10", chain, chain.flange_m(q), q)
        self.assertIsNone(answer.joints)

    def test_a_ur_pose_is_the_matrix_ur_means_by_it(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(200):
            rotation = rng.normal(size=3)
            rotation *= rng.uniform(0.0, math.pi) / np.linalg.norm(rotation)
            pose = [float(v) for v in rng.uniform(-1.0, 1.0, 3)] + [float(v) for v in rotation]
            expected = URPose.from_ur_list(pose).to_T()
            expected[:3, 3] /= 1000.0
            self.assertLess(float(np.max(np.abs(ur_pose_matrix_m(pose) - expected))), 1e-12)
        self.assertEqual(np.eye(4).tolist(), ur_pose_matrix_m([0.0] * 6).tolist())
        with self.assertRaises(ValueError):
            ur_pose_matrix_m([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
