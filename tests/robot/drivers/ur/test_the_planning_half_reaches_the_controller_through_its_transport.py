"""The UR arm's planning half reaches its controller only through the transport verbs (P0e E7), and lives apart (E8).

E8 moves the planning half into ``PlannedArm``, which knows no RTDE. So before the move, every method that is going
reads the controller through ``self._transport`` (``robot.core.transport``), and only the UR side still reads the RTDE
connection itself: the optional capabilities the plan keeps on ``URRobotArm`` until 0g (IO, force, status, provenance,
fault acknowledge). A new planning method that reached for ``self._conn`` would be a method E8 cannot move, found here
rather than at the move.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.robot.core.transport import (
    ArmTransport,
    ControllerFreedrive,
    ControllerIO,
    ControllerKinematics,
    ControllerMotion,
    HoldsAHaltLatch,
    ReadsControllerProvenance,
    ReadsForceTorque,
    ReleasesProtectiveStop,
    ReportsControllerModel,
    ReportsControllerStatus,
    TakesPayload,
    WatchesController,
)
from src.robot.drivers.ur.transport import URTransport

_ARM = Path(__file__).resolve().parents[4] / "src" / "robot" / "drivers" / "ur" / "arm.py"

#: The UR side that reads the RTDE connection itself. The list may only shrink. Since P0g it holds the two properties
#: and Willy's own UR side: the five I/O verbs left in slice 6 (``PlannedDigitalIO`` over ``ControllerIO``), the force
#: readings, the status and its quick read, the fault release and the provenance in slice 7 (one mixin each over a
#: transport verb), and the model check reads ``ReportsControllerModel`` since slice 7. Hand guiding and the payload
#: read-back left in port S4 (``PlannedFreedrive`` over ``ControllerFreedrive``).
_UR_SIDE = frozenset({
    "_conn", "_motion",  # the properties themselves
    # Willy's own UR side, each until the step that gives the capability it serves a transport verb:
    "_fk_on_a_fresh_program",  # restarts the UR control program to read the home FK again (2026-10-12)
    "_vouched_flange_chain",  # the UR's own DH rows and active TCP, read for the tool frame check at connect
    "_controller_chain",  # the same rows, the answer a line solved here asks the UR side for (line_ik: local)
})


def _of_self(node: ast.AST, attr: str) -> bool:
    """``node`` is ``self.<attr>``."""
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == "self")


def _reads_the_connection(node: ast.AST) -> bool:
    """``self._conn``, ``self._motion``, or the connection off the transport: ``self._transport.connection`` or
    ``getattr(self._transport, "connection", ...)``, the form a reader takes that must pass a fake transport with
    none."""
    if _of_self(node, "_conn") or _of_self(node, "_motion"):
        return True
    if isinstance(node, ast.Attribute) and node.attr == "connection" and _of_self(node.value, "_transport"):
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
            and len(node.args) >= 2 and _of_self(node.args[0], "_transport")
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "connection")


def _readers_of_the_connection(source: str | None = None) -> dict[str, list[int]]:
    """Every URRobotArm method that reads the RTDE connection (:func:`_reads_the_connection`), with the lines."""
    tree = ast.parse(_ARM.read_text(encoding="utf-8") if source is None else source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "URRobotArm")
    readers: dict[str, list[int]] = {}
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(item):
            if _reads_the_connection(node):
                readers.setdefault(item.name, []).append(node.lineno)
    return readers


class ThePlanningHalfUsesTheTransportTests(unittest.TestCase):
    def test_only_the_ur_side_reads_the_rtde_connection(self) -> None:
        stray = {name: lines for name, lines in _readers_of_the_connection().items() if name not in _UR_SIDE}
        self.assertEqual({}, stray, "route these through self._transport, or E8 cannot move them")

    def test_the_list_holds_only_readers_that_are_still_there(self) -> None:
        """A stale entry would admit a planning method of that name later."""
        readers = set(_readers_of_the_connection())
        self.assertEqual(set(), _UR_SIDE - readers - {"_conn", "_motion"})

    def test_the_scan_sees_a_reader(self) -> None:
        """The positive control, on a synthetic source: every real reader leaves in P0g, and a control whose target
        left would pass on a blind scan."""
        source = (
            "class URRobotArm:\n"
            "    def reads(self):\n        return self._conn.get_tcp_force()\n"
            "    def moves(self):\n        self._motion.move_to(None)\n"
            "    def other(self):\n        return other._conn\n"
            "    def asks(self):\n        return getattr(self._transport, \"connection\", None)\n"
            "    def holds(self):\n        return self._transport.connection.is_connected\n"
            "    def plans(self):\n        return getattr(self._transport, \"read_joints\", None)\n"
        )
        self.assertEqual({"reads": [3], "moves": [5], "asks": [9], "holds": [11]},
                         _readers_of_the_connection(source))


class TheUrTransportSpeaksTheVerbsTests(unittest.TestCase):
    def test_it_is_every_transport_the_planned_arm_asks_for(self) -> None:
        transport = URTransport(MagicMock(), MagicMock())
        for protocol in (ArmTransport, ControllerKinematics, ControllerMotion, TakesPayload, ReportsControllerModel,
                         HoldsAHaltLatch, ControllerIO, ReadsForceTorque, ReportsControllerStatus,
                         ReleasesProtectiveStop, ReadsControllerProvenance, ControllerFreedrive,
                         WatchesController):
            with self.subTest(protocol=protocol.__name__):
                self.assertIsInstance(transport, protocol)

    def test_the_freedrive_verbs_are_the_connections_hand_guiding_primitives(self) -> None:
        from src.robot.core import ControllerPayload

        connection = MagicMock()
        connection.set_watchdog.return_value = True
        connection.teach_mode.return_value = True
        connection.end_teach_mode.return_value = False
        connection.kick_watchdog.return_value = True
        connection.is_program_running.return_value = True
        connection.is_protective_stopped.return_value = False
        connection.is_emergency_stopped.return_value = False
        connection.get_joint_speeds.return_value = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
        connection.get_payload.return_value = (1.9, (1.0, -2.0, 60.0))
        transport = URTransport(connection, MagicMock())
        self.assertTrue(transport.arm_watchdog(2.0))
        connection.set_watchdog.assert_called_once_with(2.0)
        self.assertTrue(transport.begin_hand_guiding())
        self.assertFalse(transport.end_hand_guiding())
        self.assertTrue(transport.kick_watchdog())
        transport.restart_control_program()
        connection.restart_control_script.assert_called_once_with()
        self.assertTrue(transport.control_program_running())
        self.assertFalse(transport.stop_latched())
        connection.is_emergency_stopped.return_value = True
        self.assertTrue(transport.stop_latched())
        self.assertEqual([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], transport.read_joint_speeds())
        self.assertEqual(ControllerPayload(1.9, (1.0, -2.0, 60.0)), transport.read_payload())
        connection.get_payload.return_value = None
        self.assertIsNone(transport.read_payload())

    def test_a_trajectory_stops_at_the_first_waypoint_the_controller_rejects(self) -> None:
        connection = MagicMock()
        connection.moveJ.side_effect = [True, False, True]
        transport = URTransport(connection, MagicMock())
        self.assertFalse(transport.run_trajectory([[0.0] * 6, [0.1] * 6, [0.2] * 6], vel=0.5, acc=0.4))
        self.assertEqual(2, connection.moveJ.call_count)
        connection.moveJ.assert_called_with([0.1] * 6, vel=0.5, acc=0.4)

    def test_what_the_controller_raises_propagates(self) -> None:
        connection = MagicMock()
        connection.moveJ.side_effect = RuntimeError("RTDE control script is not running!")
        with self.assertRaisesRegex(RuntimeError, "not running"):
            URTransport(connection, MagicMock()).run_trajectory([[0.0] * 6], vel=None, acc=None)


_PLANNED = _ARM.parents[1] / "planned"

#: What URRobotArm's own body holds since E8: the UR side above, the build and the checks only a UR answers. Since P0g
#: slice 15 connect, the tool-frame check, the frame it keeps and the context manager are PlannedArm's, and the UR
#: answers three of connect's hooks: its address, its model's size class, and its series twin (a fourth, Willy's own,
#: is in the UR side above: the controller's own rows it vouches for).
#: The list may only shrink: a method added here is a method PlannedArm does not share.
_UR_OWN = _UR_SIDE | {
    "__init__", "_pose_from_controller", "_pose_to_controller", "_model_size",
    "_verify_controller_model",  # UR size classes against the dashboard's model, stays on URRobotArm
    "_address_refusal", "_refuse_if_another_model_fits_better",
    "is_inside_workspace", "_boxed", "_coerce_urpose",
    # Willy's own, each an answer the planned arm asks the UR for, or a UR capability:
    "_home_fk",  # the home's FK, read again on a fresh control program where the old one ended
    "_controller_goal_m",  # the goal a line solved here reads, through the UR's axis-angle
}


class ThePlannedArmIsApartTests(unittest.TestCase):
    def test_the_planned_package_imports_no_vendor(self) -> None:
        """No driver package, no gripper package, no UR type: a vendor joins by subclassing, never by being imported."""
        found, offenders = 0, []
        for path in sorted(_PLANNED.glob("*.py")):
            found += 1
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    module = ("." * node.level) + (node.module or "")
                    names = {alias.name for alias in node.names}
                    vendor = ((module.startswith("src.robot.drivers.") and ".planned" not in module)
                              or module.startswith("..") or module.startswith("src.robot.grippers")
                              or bool(names & {"URPose", "URConnection", "MotionController", "URTransport"}))
                    if vendor:
                        offenders.append(f"{path.name}:{node.lineno} from {module} import {sorted(names)}")
        self.assertGreaterEqual(found, 3, "the walk found no planned package")
        self.assertEqual([], offenders)

    def test_the_ur_arm_keeps_only_its_own_side(self) -> None:
        tree = ast.parse(_ARM.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "URRobotArm")
        own = {item.name for item in cls.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertEqual(set(), own - _UR_OWN, "a planning method was added to URRobotArm: add it to PlannedArm")
        self.assertEqual(set(), _UR_OWN - own, "a listed method left URRobotArm: remove it from the list")

    def test_the_ur_arm_is_a_planned_arm(self) -> None:
        from src.robot.drivers.planned.arm import PlannedArm
        from src.robot.drivers.ur.arm import URRobotArm

        self.assertTrue(issubclass(URRobotArm, PlannedArm))
        self.assertIn("move", vars(PlannedArm))
        self.assertNotIn("move", vars(URRobotArm))


if __name__ == "__main__":
    unittest.main()
