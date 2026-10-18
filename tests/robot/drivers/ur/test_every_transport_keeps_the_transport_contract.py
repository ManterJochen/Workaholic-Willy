"""Every transport a planned arm can stand on keeps one contract: the fake and ``URTransport`` alike (P0g slice 5).

A fake that answers what the wire cannot is a fixture richer than the artefact: a test passes against it and the arm
fails against the controller. So the fake and every real transport run the same cases, each through a rig that puts
its transport in the same situation. The UR rig scripts the RTDE connection below ``URTransport`` as ur_rtde behaves:
``moveJ`` answers ``False`` for a waypoint the controller rejected, raises what the link raises, and raises
``ValueError`` for a value it refuses before sending (``ArmTransport``'s docstring, P0g R3).

The halt latch (Willy's own) lives on the UR connection, so its cases run ``URTransport`` over a real
``URConnection`` on the recording ur_rtde double (``tests/_halt_fakes.py``): a MagicMock connection would answer
anything.

One case failed on the UR until P0g slice 7 and was marked so: ``URTransport`` handed on ur_rtde's ``[]`` where the
Protocol says an IK with no solution is ``None``. ``PlannedArm`` read both as no solution (``if not joints``), so no
motion differed; slice 7 maps it, and the marker went with it.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core.arm_capabilities import ArmHalted, DigitalIOPort, MoveEnd
from src.robot.drivers.ur.transport import URTransport
from tests._fake_transport import Raises, Rejects, fake_transport

_LINE = [[0.0, -1.5, 1.5, -1.5, -1.5, 0.0], [0.1, -1.5, 1.5, -1.5, -1.5, 0.0], [0.2, -1.5, 1.5, -1.5, -1.5, 0.0]]
_POSE = Pose(position_mm=np.array([300.0, -100.0, 250.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
             frame=Frame.BASE)


class _FakeRig:
    """The fake, scripted into each situation."""

    def __init__(self) -> None:
        self.fake: Any = None

    def build(self, *, trajectory: Any = True, joints: Any = None, connected: bool = True, steady: bool = True,
              fk_mm: Any = None, ik: Any = None, model: Any = None) -> Any:
        state: dict[str, Any] = {"connected": connected, "model": model}
        if joints is not None:
            state["joints"] = joints
        self.fake = fake_transport(**state)
        self.fake.script("run_trajectory", trajectory)
        self.fake.script("wait_until_steady", steady)
        if fk_mm is not None:
            self.fake.script("controller_fk", Pose(position_mm=np.asarray(fk_mm, dtype=np.float64),
                                                   quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE))
        if ik is not None:
            self.fake.script("controller_ik", ik)
        return self.fake

    def io(self, *, write: Any = True, level: Any = None) -> Any:
        self.fake = fake_transport()
        self.fake.script("write_digital", write)
        self.fake.script("write_analog", write)
        if level is not None:
            self.fake.script("read_digital_input", level)
            self.fake.script("read_digital_output", level)
        return self.fake

    def wrote(self) -> list[tuple[Any, ...]]:
        return [(c.args[0], c.args[1], str(c.kwargs["port"])) for c in self.fake.calls("write_digital")]

    def reading(self, *, wrench: Any = None, torques: Any = None, status: Any = None, release: Any = None,
                provenance: Any = None) -> Any:
        """The fake with each reading scripted as the UR rig's controller answers it."""
        from src.robot.core import ControllerProvenance, RobotMode, RobotStatus, SafetyMode, Wrench

        self.fake = fake_transport()
        if wrench is not None:
            self.fake.script("read_tcp_wrench", wrench if isinstance(wrench, BaseException)
                             else Wrench(*[float(v) for v in wrench], frame=Frame.BASE))
        if torques is not None:
            self.fake.script("read_joint_torques", torques)
        if status is not None:
            self.fake.script("read_status", status if isinstance(status, BaseException) else RobotStatus(
                robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                emergency_stopped=False, message="PROTECTIVE_STOP", program_running=False))
        if release is not None:
            self.fake.script("release_protective_stop", release)
        if provenance is not None:
            self.fake.script("read_provenance", provenance if isinstance(provenance, BaseException)
                             else ControllerProvenance(serial=provenance[0], host=provenance[1],
                                                       is_simulator=provenance[2]))
        return self.fake

    def rejecting_at(self, k: int) -> Any:
        return Rejects(at=k)

    def raising_at(self, k: int, error: BaseException) -> Any:
        return Raises(error, at=k)

    def sent(self) -> list[list[float]]:
        return list(self.fake.sent)

    def speeds(self) -> list[tuple[Any, Any]]:
        return [(c.kwargs["vel"], c.kwargs["acc"]) for c in self.fake.calls("run_trajectory")]

    def ik_seeds(self) -> list[Any]:
        return [c.kwargs["seed"] for c in self.fake.calls("controller_ik")]

    def payloads(self) -> list[tuple[Any, Any]]:
        return [(c.args[0], c.args[1]) for c in self.fake.calls("set_payload")]

    def latched(self) -> Any:
        """A connected fake, with its latch to set."""
        self.fake = fake_transport()
        return self.fake

    def written(self) -> list[tuple[int, bool]]:
        """Every digital output the controller holds as written, by pin."""
        return [(pin, level) for (port, pin), level in self.fake.outputs.items() if port != "analog"]


class _URRig:
    """``URTransport`` over a connection that answers as ur_rtde does, and records what reached the controller."""

    def __init__(self) -> None:
        self.connection: Any = None
        self._sent: list[list[float]] = []
        self._speeds: list[tuple[Any, Any]] = []
        self._sim: Any = None

    def build(self, *, trajectory: Any = True, joints: Any = None, connected: bool = True, steady: bool = True,
              fk_mm: Any = None, ik: Any = None, model: Any = None) -> Any:
        self._sent, self._speeds, self._sim = [], [], None
        connection = MagicMock()
        connection.is_connected = connected
        connection.get_joint_positions.return_value = list(joints if joints is not None else _LINE[0])
        connection.wait_until_steady.return_value = steady
        connection.controller_model.return_value = model
        if fk_mm is not None:
            connection.fk.return_value = [float(v) / 1000.0 for v in fk_mm] + [0.0, 0.0, 0.0]
        if ik is not None:
            connection.ik.return_value = ik
        else:
            connection.ik.return_value = []  # ur_rtde's answer for a pose with no solution (connection.py)
        index = {"at": 0}

        def move_j(q: Any, vel: Any = None, acc: Any = None) -> bool:
            k = index["at"]
            index["at"] += 1
            if isinstance(trajectory, tuple) and trajectory[0] == "raise" and trajectory[1] == k:
                if not isinstance(trajectory[2], ValueError):
                    self._sent.append([float(v) for v in q])
                    self._speeds.append((vel, acc))
                raise trajectory[2]
            self._sent.append([float(v) for v in q])
            self._speeds.append((vel, acc))
            return not (isinstance(trajectory, tuple) and trajectory[0] == "reject" and trajectory[1] == k)

        connection.moveJ.side_effect = move_j
        self.connection = connection
        return URTransport(connection, MagicMock())

    def io(self, *, write: Any = True, level: Any = None) -> Any:
        connection = MagicMock()
        for setter in ("set_digital_out", "set_analog_out"):
            if isinstance(write, BaseException):
                getattr(connection, setter).side_effect = write
            else:
                getattr(connection, setter).return_value = write  # ur_rtde's setters answer a bool
        for getter in ("get_digital_in", "get_digital_out"):
            if isinstance(level, BaseException):
                getattr(connection, getter).side_effect = level
            elif level is not None:
                getattr(connection, getter).return_value = level
        self.connection = connection
        return URTransport(connection, MagicMock())

    def wrote(self) -> list[tuple[Any, ...]]:
        return [tuple(c.args) for c in self.connection.set_digital_out.call_args_list]

    def reading(self, *, wrench: Any = None, torques: Any = None, status: Any = None, release: Any = None,
                provenance: Any = None) -> Any:
        """URTransport over a connection answering each reading as ur_rtde and the dashboard do."""
        connection = MagicMock()
        for name, answer in (("get_tcp_force", wrench), ("get_joint_torques", torques),
                             ("unlock_protective_stop", release)):
            if isinstance(answer, BaseException):
                getattr(connection, name).side_effect = answer
            elif answer is not None:
                getattr(connection, name).return_value = answer
        if isinstance(status, BaseException):
            connection.get_robot_mode.side_effect = status
        elif status is not None:
            connection.get_robot_mode.return_value = 7        # RUNNING
            connection.get_safety_mode.return_value = 3       # PROTECTIVE_STOP
            connection.is_protective_stopped.return_value = True
            connection.is_emergency_stopped.return_value = False
            connection.dashboard_safety_status.return_value = "PROTECTIVE_STOP"
            connection.is_program_running.return_value = False
        if isinstance(provenance, BaseException):
            connection.controller_serial.side_effect = provenance
        elif provenance is not None:
            connection.controller_serial.return_value = provenance[0]
            connection.ip = provenance[1]
        self.connection = connection
        return URTransport(connection, MagicMock())

    def rejecting_at(self, k: int) -> Any:
        return ("reject", k)

    def raising_at(self, k: int, error: BaseException) -> Any:
        return ("raise", k, error)

    def sent(self) -> list[list[float]]:
        if self._sim is not None:
            return [call[3][0] for call in self._sim.named("moveJ")]
        return list(self._sent)

    def speeds(self) -> list[tuple[Any, Any]]:
        return list(self._speeds[:1])

    def ik_seeds(self) -> list[Any]:
        return [c.kwargs.get("q_near") for c in self.connection.ik.call_args_list]

    def payloads(self) -> list[tuple[Any, Any]]:
        return [tuple(c.args) for c in self.connection.set_payload.call_args_list]

    def latched(self) -> Any:
        """``URTransport`` over a real connection on the simulated controller, which keeps the latch and runs the
        watched move every UR sends to its end."""
        from src.robot.drivers.ur.connection import URConnection
        from tests._halt_fakes import SimulatedUr

        self._sim = SimulatedUr()
        return URTransport(self._sim.attach(URConnection(ip="127.0.0.1")), MagicMock())

    def written(self) -> list[tuple[int, bool]]:
        """Every digital output write that reached the simulated controller, by pin."""
        return [(call[3][0], call[3][1]) for call in self._sim.calls
                if call[1] == "io" and call[2].endswith("DigitalOut")]


class _TransportContract:
    """The cases; each subclass names the rig."""

    rig: Any

    def test_a_trajectory_taken_whole_answers_true_and_every_waypoint_reaches_the_controller(self) -> None:
        transport = self.rig.build()
        self.assertTrue(transport.run_trajectory(_LINE, vel=None, acc=None))
        self.assertEqual(_LINE, self.rig.sent())

    def test_a_rejected_waypoint_answers_false_and_nothing_after_it_is_sent(self) -> None:
        transport = self.rig.build(trajectory=self.rig.rejecting_at(1))
        self.assertFalse(transport.run_trajectory(_LINE, vel=None, acc=None))
        self.assertEqual(_LINE[:2], self.rig.sent())

    def test_what_the_controller_raises_propagates_and_nothing_after_it_is_sent(self) -> None:
        transport = self.rig.build(trajectory=self.rig.raising_at(1, RuntimeError("RTDE control script stopped")))
        with self.assertRaisesRegex(RuntimeError, "control script stopped"):
            transport.run_trajectory(_LINE, vel=None, acc=None)
        self.assertEqual(_LINE[:2], self.rig.sent())

    def test_a_value_error_means_that_waypoint_was_not_sent(self) -> None:
        transport = self.rig.build(trajectory=self.rig.raising_at(1, ValueError("value not within range")))
        with self.assertRaises(ValueError):
            transport.run_trajectory(_LINE, vel=None, acc=None)
        self.assertEqual(_LINE[:1], self.rig.sent())

    def test_speed_and_acceleration_pass_through_as_given(self) -> None:
        transport = self.rig.build()
        transport.run_trajectory(_LINE[:1], vel=0.25, acc=None)
        self.assertEqual([(0.25, None)], self.rig.speeds(), "limiting them is the arm's job, not the wire's")

    def test_the_joints_it_reads_are_a_list_of_floats(self) -> None:
        joints = transport_joints = self.rig.build(joints=_LINE[2]).read_joints()
        self.assertIsInstance(transport_joints, list)
        self.assertEqual(_LINE[2], [float(v) for v in joints])

    def test_whether_it_is_connected_is_a_bool(self) -> None:
        self.assertIs(True, self.rig.build(connected=True).is_connected)
        self.assertIs(False, self.rig.build(connected=False).is_connected)

    def test_not_at_rest_yet_is_false_not_a_raise(self) -> None:
        self.assertIs(False, self.rig.build(steady=False).wait_until_steady(1.0, 0.01))
        self.assertIs(True, self.rig.build(steady=True).wait_until_steady(1.0, 0.01))

    def test_forward_kinematics_is_a_base_frame_pose_carrying_the_label(self) -> None:
        pose = self.rig.build(fk_mm=[100.0, 200.0, 300.0]).controller_fk([0.0] * 6, label="home")
        self.assertIsInstance(pose, Pose)
        self.assertIs(Frame.BASE, pose.frame)
        self.assertEqual("home", pose.label)
        np.testing.assert_allclose([100.0, 200.0, 300.0], pose.position_mm, atol=1e-9)

    def test_an_ik_with_no_solution_is_none(self) -> None:
        self.assertIsNone(self.rig.build().controller_ik(_POSE, seed=[0.0] * 6))

    def test_the_ik_seed_reaches_the_controller(self) -> None:
        transport = self.rig.build(ik=[0.1] * 6)
        self.assertEqual([0.1] * 6, [float(v) for v in transport.controller_ik(_POSE, seed=_LINE[1])])
        self.assertEqual([_LINE[1]], self.rig.ik_seeds())

    def test_the_payload_reaches_the_controller_as_given(self) -> None:
        transport = self.rig.build()
        transport.set_payload(1.5, [0.0, 0.0, 60.0])
        self.assertEqual([(1.5, [0.0, 0.0, 60.0])], [(m, [float(v) for v in c]) for m, c in self.rig.payloads()])

    def test_a_digital_write_answers_whether_the_controller_took_it(self) -> None:
        from src.robot.core.arm_capabilities import DigitalIOPort

        self.assertIs(True, self.rig.io(write=True).write_digital(3, True, port=DigitalIOPort.TOOL))
        self.assertEqual([(3, True, "tool")], self.rig.wrote(), "the bank reaches the controller by its name")
        self.assertIs(False, self.rig.io(write=False).write_digital(3, True, port=DigitalIOPort.TOOL))

    def test_an_analog_write_answers_whether_the_controller_took_it(self) -> None:
        self.assertIs(True, self.rig.io(write=True).write_analog(1, 2.5, current=False))
        self.assertIs(False, self.rig.io(write=False).write_analog(1, 2.5, current=True))

    def test_what_the_io_link_raises_propagates(self) -> None:
        from src.robot.core.arm_capabilities import DigitalIOPort

        with self.assertRaisesRegex(RuntimeError, "IO interface is down"):
            self.rig.io(write=RuntimeError("IO interface is down")).write_digital(0, True, port=DigitalIOPort.STANDARD)

    def test_a_read_answers_the_level_and_raises_where_it_cannot(self) -> None:
        from src.robot.core.arm_capabilities import DigitalIOPort

        self.assertIs(True, self.rig.io(level=True).read_digital_input(2, port=DigitalIOPort.CONFIGURABLE))
        self.assertIs(True, self.rig.io(level=True).read_digital_output(2, port=DigitalIOPort.CONFIGURABLE))
        with self.assertRaises(RuntimeError):
            self.rig.io(level=RuntimeError("receive is down")).read_digital_input(2, port=DigitalIOPort.STANDARD)

    def test_the_wrench_is_six_numbers_in_the_base_frame_and_the_torques_a_tuple(self) -> None:
        from src.robot.core import Wrench

        wrench = self.rig.reading(wrench=[1.5, -2.0, 3.25, 0.1, 0.2, -0.3]).read_tcp_wrench()
        self.assertIsInstance(wrench, Wrench)
        self.assertIs(Frame.BASE, wrench.frame)
        self.assertEqual((1.5, -2.0, 3.25), (wrench.fx, wrench.fy, wrench.fz))
        self.assertEqual((0.5, -1.25), self.rig.reading(torques=[0.5, -1.25]).read_joint_torques())

    def test_a_force_reading_the_link_cannot_give_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "receive is down"):
            self.rig.reading(wrench=RuntimeError("receive is down")).read_tcp_wrench()

    def test_the_status_is_the_cores_with_its_stops_and_text(self) -> None:
        from src.robot.core import RobotMode, RobotStatus, SafetyMode

        status = self.rig.reading(status=True).read_status()
        self.assertIsInstance(status, RobotStatus)
        self.assertIs(RobotMode.RUNNING, status.robot_mode)
        self.assertIs(SafetyMode.PROTECTIVE_STOP, status.safety_mode)
        self.assertIs(True, status.protective_stopped)
        self.assertEqual("PROTECTIVE_STOP", status.message)
        self.assertIs(False, status.program_running)
        with self.assertRaisesRegex(RuntimeError, "receive is down"):
            self.rig.reading(status=RuntimeError("receive is down")).read_status()

    def test_a_stop_release_answers_whether_the_controller_took_it(self) -> None:
        self.assertIs(True, self.rig.reading(release=True).release_protective_stop())
        self.assertIs(False, self.rig.reading(release=False).release_protective_stop())

    def test_provenance_proves_a_simulator_only_where_the_evidence_does_and_raises_what_the_link_raises(self) -> None:
        from src.robot.core import ControllerProvenance

        proven = self.rig.reading(provenance=("20235599999", "127.0.0.1", True)).read_provenance()
        self.assertIsInstance(proven, ControllerProvenance)
        self.assertEqual(("20235599999", "127.0.0.1", True), (proven.serial, proven.host, proven.is_simulator))
        real = self.rig.reading(provenance=("20235512345", "192.168.1.10", None)).read_provenance()
        self.assertIsNone(real.is_simulator, "absence of proof is absence of a claim, never False")
        with self.assertRaisesRegex(RuntimeError, "dashboard closed"):
            self.rig.reading(provenance=RuntimeError("dashboard closed")).read_provenance()

    def test_the_model_is_the_controllers_string_or_none(self) -> None:
        self.assertEqual("UR5", self.rig.build(model="UR5").controller_model())
        self.assertIsNone(self.rig.build(model=None).controller_model())

    def test_a_latched_link_sends_no_trajectory_and_says_the_halt_refused_it(self) -> None:
        transport = self.rig.latched()
        state = transport.request_halt("a person pressed halt")
        self.assertEqual("a person pressed halt", state.reason)
        self.assertIs(state, transport.halt_state())
        self.assertFalse(transport.run_trajectory(_LINE, vel=None, acc=None))
        self.assertEqual([], self.rig.sent())
        self.assertIs(MoveEnd.REFUSED_HALTED, transport.last_move_end)

    def test_a_second_request_keeps_the_first_record_and_counts_once(self) -> None:
        transport = self.rig.latched()
        first = transport.request_halt("first")
        self.assertIs(first, transport.request_halt("second"))
        self.assertEqual(1, transport.halt_requests)

    def test_a_latched_link_switches_no_output_and_still_ends_a_pulse_low(self) -> None:
        """The end of a pulse a halt cut short is the one write a latched link makes, and only ever low."""
        transport = self.rig.latched()
        transport.request_halt("a person pressed halt")
        with self.assertRaises(ArmHalted):
            transport.write_digital(0, True, port=DigitalIOPort.TOOL)
        with self.assertRaises(ArmHalted):
            transport.write_analog(0, 2.5, current=False)
        self.assertTrue(transport.end_pulse(0, port=DigitalIOPort.TOOL))
        self.assertEqual([(0, False)], self.rig.written())

    def test_a_cleared_latch_sends_again(self) -> None:
        transport = self.rig.latched()
        transport.request_halt("halt")
        transport.clear_halt()
        self.assertIsNone(transport.halt_state())
        self.assertTrue(transport.run_trajectory(_LINE[:1], vel=None, acc=None))
        self.assertEqual(_LINE[:1], self.rig.sent())
        self.assertEqual(1, transport.halt_requests, "a clear does not count")


class TheFakeKeepsTheContractTests(_TransportContract, unittest.TestCase):
    rig = _FakeRig()


class TheUrTransportKeepsTheContractTests(_TransportContract, unittest.TestCase):
    rig = _URRig()


class _SendsTheRefusedWaypoint(_URRig):
    """A wire that sends the waypoint its library refused: what a transport that swallowed the ValueError would do."""

    def raising_at(self, k: int, error: BaseException) -> Any:
        return ("raise", k, RuntimeError(str(error)) if isinstance(error, ValueError) else error)


class TheContractCatchesABrokenTransportTests(unittest.TestCase):
    def test_a_transport_that_sends_a_refused_waypoint_fails_the_contract(self) -> None:
        """The sabotage control: a contract a broken transport passes holds nothing."""

        class _Broken(_TransportContract, unittest.TestCase):
            rig = _SendsTheRefusedWaypoint()

        result = unittest.TestResult()
        _Broken("test_a_value_error_means_that_waypoint_was_not_sent").run(result)
        self.assertEqual(1, len(result.failures) + len(result.errors), "the broken transport passed")


if __name__ == "__main__":
    unittest.main()
