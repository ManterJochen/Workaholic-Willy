"""The arms of a cell share its space: every send is granted by the cell first (owner, 2026-10-15).

Two arms move at once wherever the spaces they sweep keep the guards' distance, and one waits where they do not; a line
and a person guiding an arm hold the whole cell. The gate sits on the transport object every send of the arm goes
through, so the planner's waypoints, a joint move and hand guiding all pass it; a send the cell does not grant sends
nothing. Each arm's guard and planner hold the other where the cell says it is: standing, or the whole space it
reserved while it moves.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

import numpy as np

from src.config import ConfigError
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionSent, MotionStatus
from src.robot.core.errors import RobotMotionRejected
from src.robot.core.cell_gate import CellOccupied, gate_sends, holding, ungate_sends
from src.robot.drivers.planned.curobo_motion import CuroboPlanner
from src.robot.execution.cell_motion import CellMotion, MovingArm
from src.robot.safety._fcl_self_collision import mesh_backend_status
from tests._arm_kit import ik_arm, kit_tree, line_arm, planned_arm
from tests._fake_planner import FakePlanner
from tests._fake_transport import fake_transport

#: Where both arms stand, reaching out and down; and the upright pose a motion to it starts from.
_Q = (0.0, -1.2, 1.3, -0.4, 1.5, 0.2)
_UP = (0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0)
#: Where the kit's line arm runs a line to, tool down (``tests/test_a_checked_line_is_sampled_and_solved.py``).
_LINE_TO = Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE)


def _arm(joints: Any = _Q, *, planner: str = "curobo", backend: str = "fcl") -> Any:
    config = kit_tree(planner, safety={"self_collision": {"backend": backend}},
                      gripper={"model": "robotiq_2f85", "tool_frame": {"source": "willy", "offset_mm": (0.0, 0.0, 132.0)}})
    return planned_arm(config, transport=fake_transport(joints=list(joints)))


def _guard(arm: Any) -> Any:
    return next(guard for guard in arm.safety_preflight.guards if guard.name == "self_collision")


def _shifted(shift_mm: Any) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = shift_mm
    return matrix


class _Gate:
    """A cell that grants, or refuses, and says what it was asked."""

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.asked: list[Any] = []
        #: Read at every reservation, where a test wants to know what had happened by then.
        self.watch: Any = None
        self.watched: list[Any] = []

    def reserve(self, route: Any) -> None:
        self.asked.append(("reserve", None if route is None else [list(map(float, q)) for q in route]))
        if self.watch is not None:
            self.watched.append(self.watch())
        if self.refuse:
            raise CellOccupied("werker: it stands in the way, and that did not clear within 0 s; nothing was sent")

    def release(self) -> None:
        self.asked.append(("release", None))


class TheGateOnTheTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = fake_transport(joints=list(_UP))
        self.gate = _Gate()
        gate_sends(self.transport, self.gate)

    def test_a_joint_send_reserves_its_route_from_where_the_arm_stands(self) -> None:
        self.transport.run_trajectory([list(_Q)], vel=0.1, acc=0.1)
        self.assertEqual(self.gate.asked, [("reserve", [list(_UP), list(_Q)]), ("release", None)])

    def test_a_span_is_one_reservation_and_its_sends_pass(self) -> None:
        with holding(self.transport, lambda: [list(_UP), list(_Q), list(_UP)]):
            self.transport.run_trajectory([list(_Q)], vel=0.1, acc=0.1)
            self.transport.run_trajectory([list(_UP)], vel=0.1, acc=0.1)
        self.assertEqual([verb for verb, _ in self.gate.asked], ["reserve", "release"])
        self.assertEqual(len(self.transport.calls("run_trajectory")), 2)

    def test_a_line_and_hand_guiding_hold_the_whole_cell(self) -> None:
        self.transport.move_pose(None, linear=True, vel=None, acc=None, register=False, workspace_pose=None)
        self.assertEqual(self.gate.asked, [("reserve", None), ("release", None)])
        self.gate.asked.clear()
        self.assertTrue(self.transport.begin_hand_guiding())
        self.assertEqual(self.gate.asked, [("reserve", None)])
        self.transport.end_hand_guiding()
        self.transport.disconnect()
        self.assertEqual(self.gate.asked, [("reserve", None), ("release", None)], "released once, when it ended")

    def test_a_connection_lost_while_guided_gives_the_cell_back(self) -> None:
        self.transport.begin_hand_guiding()
        self.transport.disconnect()
        self.assertEqual(self.gate.asked[-1], ("release", None))

    def test_a_refused_send_sends_nothing(self) -> None:
        self.gate.refuse = True
        with self.assertRaises(CellOccupied):
            self.transport.run_trajectory([list(_Q)], vel=0.1, acc=0.1)
        with self.assertRaises(CellOccupied):
            self.transport.begin_hand_guiding()
        self.assertEqual(self.transport.calls("run_trajectory"), [])
        self.assertEqual(self.transport.calls("begin_hand_guiding"), [])
        self.assertNotIn(("release", None), self.gate.asked, "nothing granted, nothing to release")

    def test_the_gate_comes_off_again_and_the_transport_answers_as_before(self) -> None:
        ungate_sends(self.transport)
        self.transport.run_trajectory([list(_Q)], vel=0.1, acc=0.1)
        self.assertEqual(self.gate.asked, [])
        self.assertNotIn("run_trajectory", vars(self.transport), "the class's own verb answers again")
        gate_sends(self.transport, self.gate)
        self.transport.move_pose(None, linear=True, vel=None, acc=None, register=False, workspace_pose=None)
        self.assertEqual(self.gate.asked, [("reserve", None), ("release", None)])

    def test_a_transport_gated_twice_is_refused_and_one_no_cell_gates_holds_nothing(self) -> None:
        with self.assertRaises(RuntimeError):
            gate_sends(self.transport, _Gate())
        free = fake_transport()
        with holding(free, None):
            free.run_trajectory([list(_Q)], vel=0.1, acc=0.1)
        self.assertEqual(len(free.calls("run_trajectory")), 1)


class TheVerbsSayACellsRefusalTests(unittest.TestCase):
    def test_the_planner_reserves_its_whole_route_once_and_a_refusal_sends_nothing(self) -> None:
        planner: Any = FakePlanner()
        transport = fake_transport(joints=list(_UP))
        glue = CuroboPlanner(transport, joint_names=planner.joint_names, controller_yaw_deg=0,
                             client_factory=lambda: planner)
        gate = _Gate()
        gate_sends(transport, gate)
        route = [list(_Q), list(_UP)]
        self.assertTrue(glue.execute(route).ok)
        self.assertEqual(gate.asked, [("reserve", [list(_UP), *route]), ("release", None)])
        gate.refuse, sent = True, len(transport.calls("run_trajectory"))
        refused = glue.execute(route)
        self.assertEqual(refused.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertEqual(refused.sent, MotionSent.NOTHING)
        self.assertIn("werker", refused.message)
        self.assertEqual(len(transport.calls("run_trajectory")), sent)

    def test_a_joint_move_the_cell_refuses_is_a_result_that_sent_nothing(self) -> None:
        arm: Any = ik_arm(joints=_UP)
        arm.join_cell(_Gate(refuse=True))
        result = arm.move_to_joints(JointPositions(list(_Q)))
        self.assertFalse(result.ok)
        self.assertEqual(result.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertEqual(result.sent, MotionSent.NOTHING)
        self.assertEqual(arm._transport.calls("run_trajectory"), [])


class ALineHoldsTheWholeCellTests(unittest.TestCase):
    """The controller draws a line, so no joints reserve its space: it holds the whole cell from before its judgement
    to its end, and no other arm moves between the two."""

    def setUp(self) -> None:
        if mesh_backend_status("ur5e") != "ok":
            self.skipTest("no exact mesh backend on this box")
        self.arm: Any = line_arm()
        self.enterContext(self.arm.without_camera_world("unit double: a line on a kit arm, no camera world wired"))
        self.gate = _Gate()
        # The kit's line arm carries its planner already, so the gate goes on its transport, where join_cell puts it.
        gate_sends(self.arm._transport, self.gate)

    def test_the_cell_is_held_before_the_first_sample_is_solved_until_the_line_ran(self) -> None:
        self.gate.watch = lambda: len(self.arm.ik.calls)
        self.arm.move_linear(_LINE_TO)
        self.assertEqual(self.gate.asked, [("reserve", None), ("release", None)])
        self.assertEqual(self.gate.watched, [0], "held before the first sample was solved")
        self.assertGreater(len(self.arm.ik.calls), 1)
        self.assertEqual(len(self.arm._transport.calls("move_pose")), 1)

    def test_a_line_the_cell_refuses_is_neither_judged_nor_sent(self) -> None:
        self.gate.refuse = True
        result = self.arm.move(_LINE_TO, linear=True)
        self.assertEqual((result.status, result.sent), (MotionStatus.SELF_COLLISION_REJECTED, MotionSent.NOTHING))
        self.assertEqual(self.arm.ik.calls, [])
        self.assertEqual(self.arm._transport.calls("move_pose"), [])
        with self.assertRaises(RobotMotionRejected) as caught:
            self.arm.move_linear(_LINE_TO)
        refused: Any = caught.exception.result
        self.assertEqual(refused.sent, MotionSent.NOTHING)


class ACellOfTwoArmsTests(unittest.TestCase):
    def cell(self, werker_in_cell: np.ndarray, *, werker_at: Any = _Q, wait_s: float = 0.3) -> CellMotion:
        self.halter, self.werker = _arm(_UP), _arm(werker_at)
        cell = CellMotion([MovingArm("halter", self.halter), MovingArm("werker", self.werker, werker_in_cell)],
                          wait_s=wait_s)
        cell.install()
        return cell

    def near(self) -> np.ndarray:
        """The werker's base where its forearm, at ``_Q``, stands on the halter's hand at ``_Q``."""
        arm = _arm()
        own = _guard(arm).body_hulls(arm, _Q)
        return _shifted(own["gripper"].mean(axis=0) - own["forearm"].mean(axis=0))

    def test_two_arms_whose_spaces_do_not_meet_move_at_once(self) -> None:
        cell = self.cell(_shifted((3000.0, 0.0, 0.0)))
        cell.reserve("halter", [_UP, _Q])
        cell.reserve("werker", [_Q, _UP])
        self.assertEqual(cell.held(), {"halter": "path", "werker": "path"})
        cell.release("halter")
        cell.release("werker")
        self.assertEqual(cell.held(), {})

    def test_a_route_into_an_arm_that_stands_is_refused_naming_it(self) -> None:
        cell = self.cell(self.near())
        with self.assertRaises(CellOccupied) as caught:
            cell.reserve("halter", [_UP, _Q])
        self.assertEqual(caught.exception.status, MotionStatus.SELF_COLLISION_REJECTED)
        self.assertIn("werker stands", str(caught.exception))
        self.assertEqual(cell.held(), {})

    def test_a_route_into_the_space_another_arm_sweeps_waits_and_then_is_refused(self) -> None:
        cell = self.cell(self.near())
        cell.reserve("werker", [_Q, _Q])
        with self.assertRaises(CellOccupied) as caught:
            cell.reserve("halter", [_UP, _Q])
        self.assertEqual(caught.exception.status, MotionStatus.TIMEOUT)
        self.assertIn("werker sweeps", str(caught.exception))

    def test_an_arm_that_waits_for_the_whole_cell_goes_once_it_is_released(self) -> None:
        cell = self.cell(_shifted((3000.0, 0.0, 0.0)), wait_s=10.0)
        cell.reserve("halter", None)
        granted = threading.Event()

        def werker() -> None:
            cell.reserve("werker", [_Q, _UP])
            granted.set()

        thread = threading.Thread(target=werker)
        started = time.monotonic()
        thread.start()
        self.assertFalse(granted.wait(0.3), "the halter holds the whole cell")
        cell.release("halter")
        self.assertTrue(granted.wait(5.0))
        thread.join()
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(cell.held(), {"werker": "path"})

    def test_a_line_waits_for_a_moving_arm_and_a_moving_arm_for_a_line(self) -> None:
        cell = self.cell(_shifted((3000.0, 0.0, 0.0)))
        cell.reserve("werker", [_Q, _UP])
        with self.assertRaises(CellOccupied) as caught:
            cell.reserve("halter", None)
        self.assertIn("needs the whole cell, and werker moves", str(caught.exception))
        cell.release("werker")
        cell.reserve("halter", None)
        with self.assertRaises(CellOccupied) as caught:
            cell.reserve("werker", [_UP, _Q])
        self.assertIn("halter holds the whole cell", str(caught.exception))

    def test_each_arm_holds_the_other_standing_and_moving_and_unchanged_is_not_handed_again(self) -> None:
        cell = self.cell(_shifted((3000.0, 0.0, 0.0)))
        standing = self.halter.partners
        self.assertEqual(standing.arms, ("werker",))
        self.assertIn("werker:forearm", [body.name for body in _guard(self.halter).partner_bodies])
        cell.refresh()
        self.assertIs(self.halter.partners, standing, "nothing moved, nothing handed")
        cell.reserve("werker", [_Q, _UP])
        moving = self.halter.partners
        self.assertIsNot(moving, standing)
        forearm = {body.name: body for body in moving.bodies}["werker:forearm"]
        self.assertGreater(len(forearm.hull_mm), 8, "the forearm's whole sweep, not its place")
        cell.release("werker")
        self.assertIsNot(self.halter.partners, moving)

    def test_a_cell_that_came_down_lets_its_arms_go_and_installs_again(self) -> None:
        cell = self.cell(_shifted((3000.0, 0.0, 0.0)))
        cell.reserve("werker", [_Q, _UP])
        cell.uninstall()
        self.assertEqual(cell.held(), {})
        self.assertEqual(self.halter.partners.arms, ())
        self.assertEqual(_guard(self.halter).partner_bodies, ())
        self.halter._transport.run_trajectory([list(_Q)], vel=0.1, acc=0.1)
        self.assertEqual(cell.held(), {}, "an arm out of the cell reserves nothing")
        cell.install()
        self.assertEqual(self.halter.partners.arms, ("werker",))

    def test_a_cell_refuses_an_arm_whose_path_or_body_nobody_judges(self) -> None:
        for werker, why in ((_arm(planner="ik"), "robot.motion.planner"),
                            (_arm(backend="capsule"), "safety.self_collision.backend")):
            with self.subTest(why=why):
                cell = CellMotion([MovingArm("halter", _arm()), MovingArm("werker", werker, _shifted((3000, 0, 0)))])
                with self.assertRaises(ConfigError) as caught:
                    cell.install()
                self.assertIn(why, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
