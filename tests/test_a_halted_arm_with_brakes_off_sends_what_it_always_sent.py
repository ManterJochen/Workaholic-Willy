"""With ``robot.ur.brake_on_halt`` off, as shipped, a UR arm sends its controller what it sent before the halt existed.

The owner's answer to the build plan's first question (Q1 = A, 2026-09-30): the brake ships OFF, and with it off "the UR
motion path stays exactly as today (synchronous, byte-for-byte the same commands; pinned by tests)". This file is that pin.

A table of moves runs every way a UR arm reaches its controller: the ik verbs (``move_to`` joint and linear, ``move``,
``move_joint``, ``move_to_joints``, ``move_linear``, ``move_home``), the raw transports a caller can reach
(``arm.motion``, ``arm.connection``), the cuRobo verbs with their judged routes (a planned path, a straight joint line, a
checked line) and the execution glue's waypoint legs, a move the controller refuses, one that raises, the outputs and the
status read. Each runs on a fresh arm whose four ur_rtde interfaces are recording doubles (``tests/_halt_fakes.py``). What
they recorded, every call of every interface with its arguments, in order, and what the verb answered, is compared with
``tests/data/ur_calls_before_the_halt.json``, which the same table recorded on the code before the halt was built (commit
1 plus the console contract, 2026-10-01).

The judging is a double where it is not the subject: an accepting preflight, and on cuRobo the route the judge would have
handed over. Everything below it runs as shipped: the connection, ``MotionController``, ``CuroboUrPlanner.execute``, the
arm's drives and the controller-state sentence a failed move reads.

Compared exactly: the interface, the method, their order, every integer, flag and string. Floats are compared to 1e-12,
relative: the arguments come out of the DH chain and pose conversions, where another machine's ``sin`` may round the last
bit differently, and a changed argument differs by far more than that.

``stop()`` is not in the table on purpose: it is the one verb the halt changes (it latches now, and sends nothing).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionResult
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.drivers.ur.arm import URRobotArm, _Route
from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner
from tests._halt_fakes import RecordingRtde, no_client, ur_arm_on

GOLDEN = Path(__file__).resolve().parent / "data" / "ur_calls_before_the_halt.json"

#: A grasp centre in reach of a ur5e, tool down, and joint targets near the arm, in free space.
_POSE = Pose(position_mm=np.array([450.0, -120.0, 320.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
             frame=Frame.BASE, label="pin")
_Q = JointPositions([0.2, -1.0, 1.1, -0.6, 1.45, 0.35])
_WAYPOINTS = [
    [0.0, -1.2, 1.3, -0.4, 1.5, 0.2],
    [0.05, -1.15, 1.25, -0.45, 1.49, 0.24],
    [0.12, -1.08, 1.18, -0.52, 1.47, 0.29],
    [0.2, -1.0, 1.1, -0.6, 1.45, 0.35],
]


def arm_on(rtde: RecordingRtde, planner: str = "ik") -> URRobotArm:
    """A ur5e on ``planner`` whose controller is ``rtde``, every gate accepting; on cuRobo its planner glue is real.

    The arm is built from a config that does not name ``brake_on_halt``: the table runs the shipped default.
    """
    arm: URRobotArm = ur_arm_on(rtde, planner)
    return arm


def _route(waypoints: "list[list[float]]", *, planned: bool) -> _Route:
    return _Route(waypoints=waypoints, dense=len(waypoints), how="the pin's route", planned=planned)


def outcome(answer: Any) -> Any:
    """What a verb answered, as JSON: a result's status, a bool, or nothing."""
    if isinstance(answer, MotionResult):
        return answer.status.value
    if answer is None or isinstance(answer, bool):
        return answer
    return repr(answer)


def _run(verb: Callable[[], Any]) -> Any:
    try:
        return outcome(verb())
    except Exception as exc:  # noqa: BLE001 (a verb that raises is an answer too, and is compared)
        return f"raised:{type(exc).__name__}"


# ------------------------------------------------------------------------------------------------------------------
# The table
# ------------------------------------------------------------------------------------------------------------------


def _ik(verb: Callable[[URRobotArm], Any], **overrides: Any) -> Callable[[], dict[str, Any]]:
    def case() -> dict[str, Any]:
        rtde = RecordingRtde(overrides=overrides or None)
        arm = arm_on(rtde)
        answer = _run(lambda: verb(arm))
        return {"calls": rtde.calls, "outcome": answer}

    return case


def _curobo(verb: Callable[[URRobotArm], Any], *, joint_route: "_Route | None" = None,
            pose_route: "_Route | None" = None, **overrides: Any) -> Callable[[], dict[str, Any]]:
    def case() -> dict[str, Any]:
        rtde = RecordingRtde(overrides=overrides or None)
        arm = arm_on(rtde, "curobo")
        if joint_route is not None:
            arm._judge_joint_move = (  # type: ignore[method-assign]
                lambda joints, *, command, plan_around=True: (joints, joint_route, None))
        if pose_route is not None:
            arm._route_to_the_nearest_goal = lambda *a, **k: pose_route  # type: ignore[method-assign]
        arm._judge_linear_move = lambda pose, *, command: None  # type: ignore[method-assign]
        with arm.without_camera_world("the pin judges no camera world: its routes are handed over"):
            answer = _run(lambda: verb(arm))
        return {"calls": rtde.calls, "outcome": answer}

    return case


def _glue(waypoints: "list[list[float]]", **overrides: Any) -> Callable[[], dict[str, Any]]:
    def case() -> dict[str, Any]:
        rtde = RecordingRtde(overrides=overrides or None)
        arm = arm_on(rtde)
        planner = CuroboUrPlanner(arm._conn, client_factory=no_client, vel=0.6, acc=0.9)
        answer = _run(lambda: planner.execute(waypoints, _POSE, vel=0.3, acc=0.5))
        return {"calls": rtde.calls, "outcome": answer}

    return case


def _connection() -> dict[str, Any]:
    rtde = RecordingRtde()
    arm = arm_on(rtde)
    conn = arm.connection
    answers = [
        _run(lambda: conn.moveJ(list(_Q.tolist()))),
        _run(lambda: conn.moveJ(list(_Q.tolist()), 0.2, 0.3)),
        _run(lambda: conn.moveL([0.45, -0.12, 0.45, 0.0, 3.1416, 0.0], 0.1, 0.2)),
        _run(lambda: conn.moveJ(list(_Q.tolist()), asynchronous=True)),
    ]
    return {"calls": rtde.calls, "outcome": answers}


def _outputs() -> dict[str, Any]:
    rtde = RecordingRtde()
    arm = arm_on(rtde)
    answers = [
        _run(lambda: arm.set_digital_output(0, True, port=DigitalIOPort.TOOL)),
        _run(lambda: arm.set_digital_output(0, False, port=DigitalIOPort.TOOL)),
        _run(lambda: arm.set_digital_output(3, True)),
        _run(lambda: arm.set_analog_output(0, 2.5)),
        _run(lambda: arm.get_digital_output(0, port=DigitalIOPort.TOOL)),
    ]
    return {"calls": rtde.calls, "outcome": answers}


def _status() -> dict[str, Any]:
    rtde = RecordingRtde()
    arm = arm_on(rtde)
    status = arm.get_robot_status()
    return {"calls": rtde.calls, "outcome": [status.robot_mode.value, status.safety_mode.value,
                                             status.is_operational, status.message]}


_REFUSED = RuntimeError("RTDE control script is not running!")

TABLE: dict[str, Callable[[], dict[str, Any]]] = {
    "ik_move_to": _ik(lambda arm: arm.move_to(_POSE)),
    "ik_move_to_linear": _ik(lambda arm: arm.move_to(_POSE, linear=True, vel=0.2, acc=0.3)),
    "ik_move": _ik(lambda arm: arm.move(_POSE)),
    "ik_move_line": _ik(lambda arm: arm.move(_POSE, linear=True, vel=0.25)),
    "ik_move_joint": _ik(lambda arm: arm.move_joint(_Q, velocity=0.4, acceleration=0.6)),
    "ik_move_to_joints": _ik(lambda arm: arm.move_to_joints(_Q)),
    "ik_move_linear": _ik(lambda arm: arm.move_linear(_POSE, velocity=0.1)),
    "ik_move_home": _ik(lambda arm: arm.move_home()),
    "ik_amove_to": _ik(lambda arm: asyncio.run(arm.amove_to(_POSE))),
    "motion_move_home": _ik(lambda arm: arm.motion.move_home()),
    "motion_move_to_linear": _ik(lambda arm: arm.motion.move_to(arm._pose_to_controller(_POSE), linear=True)),
    "connection_moves": _connection,
    "curobo_move_planned": _curobo(lambda arm: arm.move(_POSE, vel=0.3, acc=0.4),
                                   pose_route=_route(_WAYPOINTS, planned=True)),
    "curobo_move_to_joints_planned": _curobo(lambda arm: arm.move_to_joints(_Q, velocity=0.2),
                                             joint_route=_route(_WAYPOINTS, planned=True)),
    "curobo_move_to_joints_line": _curobo(lambda arm: arm.move_to_joints(_Q),
                                          joint_route=_route([_WAYPOINTS[0], _WAYPOINTS[-1]], planned=False)),
    "curobo_move_joint_planned": _curobo(lambda arm: arm.move_joint(_Q),
                                         joint_route=_route(_WAYPOINTS, planned=True)),
    "curobo_move_home_planned": _curobo(lambda arm: arm.move_home(),
                                        joint_route=_route(_WAYPOINTS, planned=True)),
    "curobo_checked_line": _curobo(lambda arm: arm.move(_POSE, linear=True, vel=0.1, acc=0.2)),
    "curobo_move_linear": _curobo(lambda arm: arm.move_linear(_POSE)),
    "glue_waypoints": _glue(_WAYPOINTS),
    "refused_moveJ": _ik(lambda arm: arm.move_to_joints(_Q), ctrl={"moveJ": False}),
    "refused_move_to": _ik(lambda arm: arm.move_to(_POSE), ctrl={"moveJ": False}),
    "refused_moveL_line": _curobo(lambda arm: arm.move(_POSE, linear=True), ctrl={"moveL": False}),
    "refused_move_linear": _ik(lambda arm: arm.move_linear(_POSE), ctrl={"moveL": False}),
    "refused_waypoint": _glue(_WAYPOINTS, ctrl={"moveJ": [True, True, False, True]}),
    "raised_moveJ": _ik(lambda arm: arm.move_to_joints(_Q), ctrl={"moveJ": _REFUSED}),
    "raised_waypoint": _glue(_WAYPOINTS, ctrl={"moveJ": [True, _REFUSED, True, True]}),
    "outputs": _outputs,
    "status": _status,
}


def record_table() -> dict[str, Any]:
    """Every case of :data:`TABLE`, run now: what the controller was sent and what each verb answered."""
    return {name: case() for name, case in TABLE.items()}


# ------------------------------------------------------------------------------------------------------------------
# The comparison
# ------------------------------------------------------------------------------------------------------------------


def _same(recorded: Any, golden: Any, where: str) -> "str | None":
    """Where ``recorded`` differs from ``golden``, or ``None``: floats to 1e-12 relative, everything else exactly."""
    if isinstance(golden, float) or isinstance(recorded, float):
        if isinstance(golden, bool) or isinstance(recorded, bool):
            return None if recorded is golden else f"{where}: {recorded!r} != {golden!r}"
        if not isinstance(golden, (int, float)) or not isinstance(recorded, (int, float)):
            return f"{where}: {recorded!r} != {golden!r}"
        ok = math.isclose(float(recorded), float(golden), rel_tol=1e-12, abs_tol=1e-12)
        return None if ok else f"{where}: {recorded!r} != {golden!r}"
    if isinstance(golden, list) and isinstance(recorded, list):
        if len(golden) != len(recorded):
            return f"{where}: {len(recorded)} items != {len(golden)}: {recorded!r} against {golden!r}"
        for index, (a, b) in enumerate(zip(recorded, golden)):
            said = _same(a, b, f"{where}[{index}]")
            if said is not None:
                return said
        return None
    if isinstance(golden, dict) and isinstance(recorded, dict):
        if set(golden) != set(recorded):
            return f"{where}: keys {sorted(recorded)} != {sorted(golden)}"
        for key in golden:
            said = _same(recorded[key], golden[key], f"{where}.{key}")
            if said is not None:
                return said
        return None
    return None if recorded == golden and type(recorded) is type(golden) else f"{where}: {recorded!r} != {golden!r}"


class TheBrakesOffPathIsTheOldPathTests(unittest.TestCase):
    """Brakes off: every move sends the calls it sent before the halt, with the same arguments, in the same order."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def test_the_golden_holds_every_case_and_every_kind_of_command(self) -> None:
        self.assertEqual(sorted(TABLE), sorted(self.golden), "the table and its recording name the same cases")
        sent = {call[1] for case in self.golden.values() for call in case["calls"]}
        for method in ("moveJ", "moveL", "getInverseKinematics", "getForwardKinematics", "setToolDigitalOut",
                       "getRobotMode", "safetystatus"):
            self.assertIn(method, sent, f"the recording never saw {method}, so it would pin nothing about it")

    def test_every_move_sends_what_it_sent_before_the_halt_existed(self) -> None:
        for name, case in TABLE.items():
            with self.subTest(case=name):
                recorded = json.loads(json.dumps(case()))
                said = _same(recorded, self.golden[name], name)
                self.assertIsNone(said, said)

    def test_the_moves_stay_synchronous_with_brakes_off(self) -> None:
        """The flag itself: every moveJ and moveL of the table waited for its move, except the one sent async."""
        for name, case in self.golden.items():
            for call in case["calls"]:
                if call[1] in ("moveJ", "moveL") and name != "connection_moves":
                    with self.subTest(case=name, call=call):
                        self.assertIs(False, call[2][-1], "a move was sent asynchronously with brakes off")

    def test_the_shipped_tree_leaves_the_brake_off(self) -> None:
        """No shipped profile switches it on: the owner does, in the cell's own profile, after the URSim measurements.

        Shipped means tracked by git. The owner's own layer is untracked by design (Q10: an untracked, git-ignored last
        layer of the chain, such as ``config/robot/robot.cell.yaml``), so switching the brake on there, the documented
        step, leaves this green. A checkout without git cannot tell the two apart, and skips.
        """
        repo = Path(__file__).resolve().parents[1]
        shipped = _tracked_profiles(repo)
        if shipped is None:
            self.skipTest("not a git checkout: the shipped profiles cannot be told from a cell's own layers")
        self.assertTrue(shipped, "git lists no tracked profile under config/")
        switched_on = [
            f"{path.relative_to(repo)}: {line.strip()}"
            for path in shipped
            for line in path.read_text(encoding="utf-8").splitlines()
            if re.match(r"^brake_on_halt\s*:\s*(true|yes|on)\b", line.split("#", 1)[0].strip(), re.IGNORECASE)
        ]
        self.assertEqual([], switched_on)


def _tracked_profiles(repo: Path) -> "list[Path] | None":
    """The YAML files git tracks under ``config/``, the all-keys reference aside, or ``None`` where git cannot say."""
    try:
        listed = subprocess.run(["git", "ls-files", "-z", "--", "config"], cwd=repo, capture_output=True, check=True,
                                timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = [name for name in listed.decode("utf-8").split("\0") if name.endswith(".yaml")]
    return [repo / name for name in sorted(names)
            if "all_keys" not in Path(name).parts and (repo / name).is_file()]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
