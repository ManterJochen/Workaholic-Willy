"""The UR's controller IK asks whether a solution exists before it asks for one (P0e E14).

Measured on URSim (UR3e, URSoftware 5.26.1, ur_rtde 1.6.5, 2026-09-22), each on a fresh connection:
``getInverseKinematics`` for a reachable pose returned a solution and the control script ran on; for a pose 2 m out
and one half a metre below the base it returned ``[]`` in 0.02 s AND stopped the RTDE control script, after which
the controller took no motion command until the arm was connected again. ``getInverseKinematicsHasSolution``
answered False for those poses and for a shoulder-singular one, the script ran on, and an IK asked right after still
solved. So one candidate with no solution stopped the cell: the URSim pick-loop probe's scenario A did exactly that,
and was read as a failed grasp until E13 let the loop see the stopped script.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.robot.drivers.ur.connection import URConnection
from tests.test_ur_io_force_status import _CTRL_API, _RECV_API

_POSE = [0.3, -0.1, 0.2, 0.0, 3.14159, 0.0]
_SEED = [0.1, -1.4, 1.6, -1.77, -1.57, 0.2]
_SOLVED = [0.2, -1.3, 1.5, -1.7, -1.5, 0.3]


def _connection(*, has_solution: bool) -> tuple[URConnection, MagicMock]:
    conn = URConnection(ip="127.0.0.1")
    ctrl = MagicMock(spec=_CTRL_API)
    ctrl.getInverseKinematicsHasSolution.return_value = has_solution
    ctrl.getInverseKinematics.return_value = list(_SOLVED) if has_solution else []
    conn._ctrl, conn._recv = ctrl, MagicMock(spec=_RECV_API)
    return conn, ctrl


class TheControllerIkAsksFirstTests(unittest.TestCase):
    def test_no_solution_is_answered_without_the_request_that_stops_the_script(self) -> None:
        for seed in (None, _SEED):
            with self.subTest(seed=seed):
                conn, ctrl = _connection(has_solution=False)
                self.assertEqual([], conn.ik(list(_POSE), q_near=seed))
                ctrl.getInverseKinematics.assert_not_called()

    def test_a_solution_is_asked_for_with_the_same_pose_and_seed(self) -> None:
        conn, ctrl = _connection(has_solution=True)
        self.assertEqual(_SOLVED, conn.ik(list(_POSE), q_near=list(_SEED)))
        ctrl.getInverseKinematicsHasSolution.assert_called_once_with(_POSE, _SEED)
        ctrl.getInverseKinematics.assert_called_once_with(_POSE, _SEED)

    def test_without_a_seed_both_questions_leave_the_seed_to_the_controller(self) -> None:
        conn, ctrl = _connection(has_solution=True)
        conn.ik(list(_POSE))
        ctrl.getInverseKinematicsHasSolution.assert_called_once_with(_POSE)
        ctrl.getInverseKinematics.assert_called_once_with(_POSE)


if __name__ == "__main__":
    unittest.main()
