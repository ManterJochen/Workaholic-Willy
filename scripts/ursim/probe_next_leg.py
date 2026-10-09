"""Judge the next leg while the line before it runs, against a real UR controller (URSim CB3): the owner's C7b check.

The owner, 2026-10-09 (map4 motion C7b): judge the next leg on a second thread while the current leg runs, with the
watched send of ``robot.ur.brake_on_halt`` (a halt brakes the leg in flight along its judged line), "but only after a
URSim test". This is that test. The arm, its guard, its planner glue, its connection and URSim's controller are real; the
camera and the planning sidecar are the unit tests' stand-ins (``tests/test_every_verb_meets_the_camera_world.py``: a
bench a metre below an overhead camera, a sidecar that confirms what it is sent), because URSim has no camera and this
needs no GPU. The UR10 holds a Hand-E 157 mm out on its flange, as the owner's cell does (the controller's TCP is set to
it here, and back to the bare flange at the end).

* ``A`` a line runs while the joint move declared after it is judged on a second thread. Pass: the line arrived; the
  judgement ran on another thread than the one that sent and watched the line, began before the line ended, and asked
  the controller nothing; at the junction the arm stood within 0.5 mm of where the judgement said the line would end;
  the joint move then ran as judged, with no second judgement, and arrived.
* ``B`` the same line halted 0.4 s in, the judgement running. Pass: the line was braked under control (``stopL``) and
  the arm stood still; the judging thread sent nothing; while halted the move judged ahead is not run ("the arm is
  halted"); once the halt is cleared it is judged again, because the arm stands more than 0.5 mm from where it was
  judged from, and the move judged as it runs then arrives.
* ``C`` the halt's brake with and without the judging thread, five times each, the halt landing while the thread
  judges. Pass: the stop went out within 50 ms of every halt, the bar ``probe_halt.py`` M1 sets the brake. Said, not
  gated: how far the arm ran past what its 2.0 m/s^2 brake takes from the speed at the halt, with the thread and
  without; the judgement gives up and takes the GIL around its queries, and the halting thread waits for it too.

``--gate-at send`` runs every move with the steady gate at the send (``safety.dwell.gate_at``, the map's C4).

Run it in WSL, where ``ur_rtde`` loads (Smart App Control blocks ``rtde_receive`` on this box's Windows). A URSim CB3
container of its own, on the docker bridge, publishes no port and leaves the standard one to whoever uses it::

    docker run -d -t --name ursim_flow -e ROBOT_MODEL=UR10 universalrobots/ursim_cb3:latest
    # confirm its safety configuration at the pendant (xdotool in the container), power on, release the brakes
    python scripts/ursim/probe_next_leg.py --container ursim_flow --json next_leg.json

or ``--host 127.0.0.1`` for the container ``ursim.sh`` publishes. Nothing else is accepted: the probe moves the arm
without asking. Exit codes: 0 every check passed, 1 a check failed, 2 no controller reachable or an address that is not
a simulator's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Running this as a file puts only `scripts/ursim/` on sys.path; the repository root has `src` and `tests`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.config.schema.robot import RobotConfig  # noqa: E402
from src.geometry import Frame, Pose  # noqa: E402
from src.robot.core import JointPositions, MotionStatus  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES  # noqa: E402
from tests._sidecar_identity import arm_identity  # noqa: E402
from tests.test_every_verb_meets_the_camera_world import _Camera, _Client, _world  # noqa: E402

_OK, _FAILED, _BAD_REQUEST = 0, 1, 2
#: Tool down, upper arm up, forearm level: free space on every URSim model, and the start of every check here.
HOME = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
#: The joint move declared after the line: the base a third of a radian on, wrist 3 a fifth, as a carry turns.
NEXT = (0.35, -1.5708, 1.5708, -1.5708, -1.5708, 0.2)
#: A longer one for the halts: its judgement still runs when the halt lands 0.4 s into the line.
NEXT_LONG = (1.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.6)
#: The line: straight up, as a lift is, at 0.25 m/s about 0.9 s.
LINE_MM = 200.0
HALT_AFTER_S = 0.4
#: The owner's Hand-E on the flange: 157 mm out along the flange's +Z (the controller's TCP is set to it).
TCP_M = (0.0, 0.0, 0.157, 0.0, 0.0, 0.0)


class _IdentifiedClient(_Client):
    """The unit tests' sidecar, saying it was started on the UR10 descriptor with the Hand-E, as a cell's does."""

    def __init__(self) -> None:
        super().__init__(joint_names=UR_ARM_JOINT_NAMES)
        self.identity = arm_identity("ur10", hand="robotiq_hande", approach="+Z", closing="+X")


def _host(args: argparse.Namespace) -> str:
    """The controller's address: a container on the docker bridge, by name, or the loopback; never anything else."""
    if args.container:
        found = subprocess.run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", args.container],
            capture_output=True, text=True, check=False)
        address = found.stdout.strip()
        if found.returncode != 0 or not address:
            raise SystemExit(f"no running container named {args.container!r} answers docker inspect")
        image = subprocess.run(["docker", "inspect", "-f", "{{.Config.Image}}", args.container],
                               capture_output=True, text=True, check=False).stdout.strip()
        if "ursim" not in image:
            raise SystemExit(f"container {args.container!r} runs {image!r}, which is no URSim image")
        return address
    if args.host != "127.0.0.1":
        raise SystemExit("this probe moves the arm without asking: it talks to 127.0.0.1 or a URSim container alone")
    return args.host


def _arm(host: str, *, gate_at: str = "verb") -> URRobotArm:
    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "ip": host, "motion_planner": "curobo", "brake_on_halt": True},
        "motion_limits": {"max_velocity": 0.25, "max_acceleration": 0.5},
        # The UR10's reach: the probe's poses stand outside the shipped half-metre box.
        "workspace_limits": {"x_min": -1300.0, "x_max": 1300.0, "y_min": -1300.0, "y_max": 1300.0, "z_min": -100.0,
                             "z_max": 1300.0},
        "motion": {"judge_next_leg": "in_settles_and_motion"},
        "safety": {"payload": {"enforce": False}, "ik_quality": {"enforce": False},
                   "self_collision": {"planner_margin_mm": 4.0},
                   "dwell": {"gate_at": gate_at},
                   "planning_world": {"hold": {"carry": True, "drop": True}}},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [],
                    "tool_frame": {"source": "polyscope", "offset_mm": (0.0, 0.0, 157.0)}},
    })
    # The test suite's planner double, which answers as a CuroboPlanClient does for what this probe asks.
    arm = URRobotArm(config, curobo_client_factory=_IdentifiedClient)  # type: ignore[arg-type]
    arm._conn.connect()
    ctrl = arm._conn._ctrl
    assert ctrl is not None, "connected, so the control interface is up"
    ctrl.setTcp(list(TCP_M))  # the controller holds the tool, as the cell's polyscope frame says
    arm.set_live_planner_world(_world(_Camera(silent=False)))
    return arm


def _lifted(pose: Pose, mm: float) -> Pose:
    position = np.asarray(pose.position_mm, dtype=np.float64).copy()
    position[2] += float(mm)
    return Pose(position_mm=position, quaternion_xyzw=np.asarray(pose.quaternion_xyzw, dtype=np.float64),
                frame=Frame.BASE, label=f"{mm:.0f} mm up")


class _Watch:
    """What the probe sees of the arm while it runs: which thread asked the controller what, when the line's moveL was
    sent and returned, and where the judgement of the next leg ran."""

    _ASKS = ("get_joint_positions", "get_tcp_pose", "ik", "fk", "fk_current", "moveJ", "moveL", "wait_until_steady",
             "is_steady", "get_digital_out", "stop")

    def __init__(self, arm: URRobotArm) -> None:
        self.arm = arm
        self.lock = threading.Lock()
        self.asked: list[tuple[int, str, float, float]] = []
        self.judged: list[dict[str, Any]] = []
        self.ran_as_judged: list[bool] = []
        #: Where set, the next moveL is halted that many seconds after it is sent.
        self.halt_after_s: float | None = None
        #: Whether the judging thread was still running when the last halt landed, where the TCP stood then and when,
        #: and when the thread that watches the line entered its brake.
        self.judging_at_halt: bool | None = None
        self.tcp_at_halt: list[float] | None = None
        self.speed_at_halt: float | None = None
        self.halted_at: float | None = None
        self.braking_at: float | None = None
        brake = arm._conn._brake

        def braking(*args: Any, **kwargs: Any) -> Any:
            self.braking_at = time.perf_counter()
            return brake(*args, **kwargs)

        arm._conn._brake = braking  # type: ignore[method-assign]
        conn = arm._conn
        for name in self._ASKS:
            original = getattr(conn, name)
            setattr(conn, name, self._spy(name, original))
        judge = arm._judge_the_next_leg_from

        def judged_on(*args: Any, **kwargs: Any) -> str:
            row: dict[str, Any] = {"thread": threading.get_ident(), "start": time.perf_counter()}
            try:
                row["said"] = judge(*args, **kwargs)
            finally:
                row["end"] = time.perf_counter()
                with self.lock:
                    self.judged.append(row)
            return str(row["said"])

        arm._judge_the_next_leg_from = judged_on  # type: ignore[method-assign]
        run = arm._run_the_next_leg

        def ran(ahead: Any, **kwargs: Any) -> Any:
            result = run(ahead, **kwargs)
            if ahead is not None:
                self.ran_as_judged.append(result is not None)
            return result

        arm._run_the_next_leg = ran  # type: ignore[method-assign]

    def _spy(self, name: str, original: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            halt_after = self.halt_after_s if name == "moveL" else None
            if halt_after is not None:
                # The halt lands while the line runs: counted from the moment its moveL is sent.
                self.halt_after_s = None
                threading.Timer(halt_after, self._halt).start()
            try:
                return original(*args, **kwargs)
            finally:
                with self.lock:
                    self.asked.append((threading.get_ident(), name, started, time.perf_counter()))

        return call

    def _halt(self) -> None:
        self.judging_at_halt = any(t.name == "judge-the-next-leg" and t.is_alive() for t in threading.enumerate())
        recv = self.arm._conn._recv
        self.tcp_at_halt = [float(v) for v in recv.getActualTCPPose()] if recv is not None else None
        self.speed_at_halt = (float(np.linalg.norm([float(v) for v in recv.getActualTCPSpeed()][:3]))
                              if recv is not None else None)
        self.halted_at = time.perf_counter()
        self.arm.halt("the probe halts the line in flight")

    def clear(self) -> None:
        with self.lock:
            self.asked.clear()
            self.judged.clear()
        self.ran_as_judged.clear()

    def asked_from(self, thread: int) -> list[str]:
        return [name for ident, name, _start, _end in self.asked if ident == thread]

    def line_sent(self) -> "tuple[float, float] | None":
        """When the line's ``moveL`` was sent and when it returned, the watched send having seen it end."""
        sends = [(start, end) for _ident, name, start, end in self.asked if name == "moveL"]
        return sends[-1] if sends else None


def _home(arm: URRobotArm) -> None:
    result = arm.move_to_joints(JointPositions(HOME))
    if not result.ok:
        raise SystemExit(f"the arm could not be brought to the probe's start: {result.status.value}: {result.message}")


def _line_with_the_next_leg(arm: URRobotArm, watch: "_Watch", *, halt_after_s: float | None,
                            declare: bool = True, next_joints: "tuple[float, ...]" = NEXT) -> dict[str, Any]:
    """The line up from HOME, with the joint move to NEXT declared and judged during it where ``declare``, halted
    ``halt_after_s`` after its moveL is sent where that is given."""
    _home(arm)
    up = _lifted(arm.get_tcp_pose(), LINE_MM)
    started = time.perf_counter()
    with arm.expecting_next(JointPositions(next_joints)) if declare else _nothing(), \
            arm.held_world("the probe: the line and the joint move after it, judged in one world"):
        said = arm.judge_the_next_leg(up, junction="carry", now=False) if declare else "nothing declared"
        watch.halt_after_s = halt_after_s
        line = arm.move(up, linear=True)
        watch.halt_after_s = None
    ended = time.perf_counter()
    return {"said": said, "line": line, "started": started, "ended": ended, "up": up}


class _nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def check_a(arm: URRobotArm, watch: _Watch) -> dict[str, Any]:
    watch.clear()
    run = _line_with_the_next_leg(arm, watch, halt_after_s=None)
    line = run["line"]
    main = threading.get_ident()
    judged = list(watch.judged)
    sent = watch.line_sent()
    token = arm._next_leg_ahead
    moved_mm = arm._moved_since_mm(token.start) if token is not None else float("nan")
    joint = arm.move_to_joints(JointPositions(NEXT))
    row = judged[0] if judged else {}
    checks = {
        "the line arrived": line.status is MotionStatus.EXECUTED and str(arm._conn.last_move_end) == "arrived",
        "judged on a second thread": bool(row) and row["thread"] != main,
        "judged while the line ran": bool(row) and sent is not None and row["start"] < sent[1],
        "the judging thread asked the controller nothing": bool(row) and not watch.asked_from(row["thread"]),
        "kept for the junction": token is not None,
        "the arm stood within 0.5 mm of where it was judged from": moved_mm <= 0.5,
        "ran as judged": watch.ran_as_judged == [True],
        "the joint move arrived": joint.status is MotionStatus.EXECUTED,
    }
    return {"checks": checks, "said": run["said"], "line_s": round(run["ended"] - run["started"], 3),
            "judged_s": round(row["end"] - row["start"], 3) if row else None,
            "moveL_s": round(sent[1] - sent[0], 3) if sent else None,
            "judgement_began_before_moveL_sent_s": round(sent[0] - row["start"], 3) if row and sent else None,
            "judgement_ended_after_moveL_returned_s": round(row["end"] - sent[1], 3) if row and sent else None,
            "moved_at_the_junction_mm": round(moved_mm, 4), "joint_move": joint.message}


def check_b(arm: URRobotArm, watch: _Watch) -> dict[str, Any]:
    watch.clear()
    run = _line_with_the_next_leg(arm, watch, halt_after_s=HALT_AFTER_S, next_joints=NEXT_LONG)
    line = run["line"]
    state = arm.halt_state()
    judged = list(watch.judged)
    row = judged[0] if judged else {}
    token = arm._next_leg_ahead
    while_halted = arm._why_the_next_leg_is_judged_again(token) if token is not None else "nothing was kept"
    arm.clear_halt()
    after_clear = arm._why_the_next_leg_is_judged_again(token) if token is not None else "nothing was kept"
    joint = arm.move_to_joints(JointPositions(NEXT_LONG))
    checks = {
        "the judgement still ran when the halt landed": watch.judging_at_halt is True,
        "the line was braked": line.status is MotionStatus.CANCELLED and state is not None and state.brake == "braked",
        "the judging thread sent nothing": bool(row) and not [name for name in watch.asked_from(row["thread"])
                                                                if name in ("moveJ", "moveL", "stop")],
        "not run while halted": "halted" in while_halted,
        "judged again once cleared, the arm off where it was judged from": "mm from where it was judged from"
                                                                           in after_clear,
        "not run as judged": watch.ran_as_judged == [False],
        "the move judged as it runs arrived": joint.status is MotionStatus.EXECUTED,
    }
    return {"checks": checks, "line": line.message, "brake_s": None if state is None else state.brake_s,
            "while_halted": while_halted, "after_clear": after_clear, "joint_move": joint.message}


def check_c(arm: URRobotArm, watch: _Watch) -> dict[str, Any]:
    rows: dict[str, list[dict[str, float]]] = {"with the judging thread": [], "without it": []}
    overlapped: list[bool] = []
    for _ in range(5):
        for label, declare in (("with the judging thread", True), ("without it", False)):
            watch.clear()
            watch.halted_at = watch.braking_at = None
            _line_with_the_next_leg(arm, watch, halt_after_s=HALT_AFTER_S, declare=declare, next_joints=NEXT_LONG)
            state = arm.halt_state()
            if declare:
                overlapped.append(watch.judging_at_halt is True)
            recv = arm._conn._recv
            assert recv is not None, "connected, so the receive interface is up"
            tcp = [float(v) for v in recv.getActualTCPPose()]
            row = {
                "stop_after_s": (watch.braking_at - watch.halted_at) if watch.braking_at and watch.halted_at else
                float("inf"),
                "stopping_mm": (1000.0 * float(np.linalg.norm(np.subtract(tcp[:3], watch.tcp_at_halt[:3])))
                                if watch.tcp_at_halt else float("inf")),
                # What a brake at the line's own 2.0 m/s^2 takes from the speed at the halt, and what the arm ran past it.
                "speed_at_halt_m_s": watch.speed_at_halt if watch.speed_at_halt is not None else float("inf"),
                "still_seen_s": float(state.brake_s) if state is not None and state.brake_s is not None else float("inf"),
            }
            braking_mm = 1000.0 * row["speed_at_halt_m_s"] ** 2 / (2.0 * 2.0)
            row["past_the_brake_mm"] = row["stopping_mm"] - braking_mm
            rows[label].append({k: round(v, 4) for k, v in row.items()})
            arm.clear_halt()
    worst = {label: {key: max(r[key] for r in kept) for key in ("stop_after_s", "stopping_mm", "still_seen_s",
                                                                   "past_the_brake_mm")}
             for label, kept in rows.items() if kept}
    with_, without = worst.get("with the judging thread", {}), worst.get("without it", {})
    checks = {
        "the judgement still ran at every halt with the thread": bool(overlapped) and all(overlapped),
        "the stop went out within 50 ms of every halt with the thread": with_.get("stop_after_s", 1.0) <= 0.05,
    }
    # Said, not gated: how much further past the brake its own deceleration asks for the arm ran with the thread, the
    # stop going out later and the halt itself landing later while the thread judges.
    more_mm = with_.get("past_the_brake_mm", float("nan")) - without.get("past_the_brake_mm", float("nan"))
    return {"checks": checks, "runs": rows, "worst": worst, "further_past_the_brake_with_the_thread_mm": round(more_mm, 2)}


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--container", default="", help="a URSim container on the docker bridge, by name")
    parser.add_argument("--host", default="127.0.0.1", help="the loopback, for the container ursim.sh publishes")
    parser.add_argument("--only", default="A,B,C", help="which checks, comma separated")
    parser.add_argument("--json", default="", help="write the measurements here as one JSON record as well")
    parser.add_argument("--gate-at", default="verb", choices=("verb", "send"),
                        help="safety.dwell.gate_at for every move here: 'send' gates at the send (map4 C4)")
    args = parser.parse_args(argv)
    try:
        host = _host(args)
        arm = _arm(host, gate_at=args.gate_at)
    except SystemExit as exc:
        print(f"REFUSED: {exc}")
        return _BAD_REQUEST
    except Exception as exc:  # noqa: BLE001 (no controller: said, exit 2)
        print(f"NO CONTROLLER at {args.container or args.host}: {type(exc).__name__}: {exc}")
        return _BAD_REQUEST
    watch = _Watch(arm)
    results: dict[str, Any] = {"host": host, "container": args.container or None, "gate_at": args.gate_at}
    checks = {"A": check_a, "B": check_b, "C": check_c}
    try:
        for name in [part.strip().upper() for part in args.only.split(",") if part.strip()]:
            results[name] = checks[name](arm, watch)
            for label, held in results[name]["checks"].items():
                print(f"{name}  {'PASS' if held else 'FAIL'}  {label}")
    finally:
        try:
            arm.clear_halt()
            _home(arm)
        finally:
            if arm._conn._ctrl is not None:
                arm._conn._ctrl.setTcp([0.0] * 6)
            arm._conn.disconnect()
    passed = all(held for name in results if isinstance(results[name], dict) and "checks" in results[name]
                 for held in results[name]["checks"].values())
    results["passed"] = passed
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))
    return _OK if passed else _FAILED


if __name__ == "__main__":
    raise SystemExit(main())
