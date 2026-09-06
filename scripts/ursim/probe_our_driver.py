"""Drive URSim through the `URRobotArm` driver, not through ur_rtde directly.

`probe_ursim.py` proves the SDK talks to a real controller. This proves the driver does: the
payload push and the tool-frame verification inside `connect()`, the three capability protocols the
stack feature-checks on, the vendor-neutral `RobotStatus` projection, the millimetre and XYZW
frame-tagged pose API, digital I/O through the `Bench` seam, and one commanded joint move through
`SafetyPreflight`. Everything here otherwise runs against fakes.

Working at the level of `create_arm` and `URRobotArm` is what this file is for. It is a driver
probe and not a pick cell, so `src/robot/execution/cell.py` is the wrong altitude: its preflight
blocks on a camera-to-base resolver that URSim has no camera to satisfy.

The arm is built through the vendor-neutral `create_arm` and then narrowed to `URRobotArm`, because
`raise_if_stopped()` and `tcp_raw` sit on the UR driver and on no Protocol. Narrowing turns a
profile that names another vendor into a sentence rather than into a missing attribute.

Run (URSim up, in REMOTE control):  python scripts/ursim/probe_our_driver.py

Exit codes: 0 every step OK, 1 a step failed, 2 the request itself was wrong (no controller, or a
profile that is not a UR cell).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root. `scripts/examples/_common.py` does the same insert for the examples.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config.loader import active_profile, load_robot_config  # noqa: E402
from src.robot.core import (  # noqa: E402
    JointPositions,
    RobotVendor,
    SupportsDigitalIO,
    SupportsForceTorque,
    SupportsRobotStatus,
)
from src.robot.core.arm_capabilities import DigitalIOPort  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402
from src.robot.drivers.ur.bench import Bench, BenchVerdict, Read, Set  # noqa: E402

#: The chain this probe needs when the operator has not named one. It is resolved here rather than
#: written into `WILLY_PROFILE`, which would leak process-global state into whatever runs next. It
#: is not left to the loader's own default either: the base tree points `ur.ip` at a plant address,
#: and a probe that drives digital outputs must never fall back onto a real cell.
_DEFAULT_PROFILE = "ursim"

_OK, _FAILED, _BAD_REQUEST = 0, 1, 2

results: list[tuple[str, str, bool]] = []


def note(step: str, outcome: str, ok: bool = True) -> None:
    results.append((step, outcome, ok))
    print(f"  {'OK ' if ok else 'ERR'} {step:<42} {outcome}", flush=True)


def _confirm(yes: bool, what: str) -> bool:
    """Gate a physical write. In a non-interactive shell `--yes` is the only way through.

    Refusing rather than prompting where stdin is not a terminal is deliberate: a prompt that reads
    EOF and takes the default is how an unattended script drives a coil nobody authorised. It is
    the rule `python -m src.robot.drivers.ur` follows for the same writes.
    """
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"      REFUSED: {what} would drive a real output and --yes was not given.",
              flush=True)
        return False
    print(f"\n      {what}")
    print("      This drives a real output: jaws close on whatever is between them.")
    return input("      Is the cell clear? type 'yes' to proceed: ").strip().lower() == "yes"


def _probe_capabilities(arm: URRobotArm) -> None:
    """The three capability protocols the pick path feature-checks on before using them."""
    print("\n=== 2. the capabilities this stack feature-checks on ===")
    for proto in (SupportsDigitalIO, SupportsForceTorque, SupportsRobotStatus):
        note(f"isinstance(arm, {proto.__name__})", str(isinstance(arm, proto)))


def _probe_status(arm: URRobotArm) -> None:
    """The `RobotStatus` projection, and the guard that reads it."""
    print("\n=== 3. RobotStatus: the vendor-neutral projection of the controller's own state ===")
    st = arm.get_robot_status()
    note("robot_mode / safety_mode", f"{st.robot_mode.value} / {st.safety_mode.value}")
    note("protective_stopped / emergency", f"{st.protective_stopped} / {st.emergency_stopped}")
    note("is_operational / is_stopped", f"{st.is_operational} / {st.is_stopped}")
    note("controller message", repr(st.message))
    # The verdict is recorded after the call, never before it. Printing "did not raise" first
    # records a pass for a call that then raises, and the run ends in a traceback under a line
    # claiming the controller is healthy.
    try:
        arm.raise_if_stopped()
        note("raise_if_stopped()", "did not raise (controller is healthy)")
    except Exception as exc:  # noqa: BLE001
        note("raise_if_stopped()", f"{type(exc).__name__}: {str(exc)[:80]}", ok=False)


def _probe_poses(arm: URRobotArm, offset_mm_z: float) -> None:
    """Poses and joints as this API states them: millimetres, XYZW, a frame tag on every pose."""
    print("\n=== 4. pose + joints through this API (mm + XYZW, frame-tagged) ===")
    pose = arm.get_tcp_pose()
    raw = arm.tcp_raw
    note("get_tcp_pose() [TCP, tool frame applied]",
         f"frame={pose.frame.value} pos_mm={[round(float(v), 1) for v in pose.position_mm]}")
    note("controller raw pose [FLANGE, mm]", f"{[round(v * 1000.0, 1) for v in raw[:3]]}")
    note("difference equals the declared tool offset",
         f"{round(abs(raw[2] * 1000.0 - float(pose.position_mm[2])), 1)} mm against declared "
         f"{offset_mm_z} mm")
    note("get_joint_positions()", f"{[round(v, 4) for v in arm.get_joint_positions().values]}")
    w = arm.get_tcp_wrench()
    note("get_tcp_wrench()",
         f"F={[round(v, 3) for v in w.force]} T={[round(v, 3) for v in w.torque]} "
         f"frame={w.frame.value}")


def _probe_io(arm: SupportsDigitalIO, yes: bool) -> None:
    """Digital I/O through `Bench`, which is the seam carrying the write interlock.

    `io_bench.set_output` underneath energises a pin the moment it is called, so reaching past
    `Bench` to it would delete the confirmation the library publishes this class to hold.
    `Bench.from_robot_config` is the wrong door here: this profile's gripper vendor is `none` and
    declares no `io_port`, so that factory refuses rather than guessing a bank. `from_arm` with an
    explicit `port` is the door for a probe that names its own.

    The bank is STANDARD here and TOOL in `probe_grippers.py`. Both are right: this exercises the
    controller's own outputs, and that one exercises the bank both gripper drivers default to. The
    bank is part of the address, so a pin number quoted without it says nothing.
    """
    print("\n=== 5. digital I/O, the gripper path, through the Bench seam ===")
    bench = Bench.from_arm(arm, port=DigitalIOPort.STANDARD,
                           confirm=lambda what: _confirm(yes, what))
    print("\n".join("      " + ln for ln in bench.run(Read()).render().splitlines()))
    for pin in (0, 4):
        hi = bench.run(Set(pin=pin, level=True))
        time.sleep(0.2)
        lo = bench.run(Set(pin=pin, level=False))
        note(f"bench Set(standard {pin})", f"high {hi.verdict.value}, low {lo.verdict.value}",
             ok=(hi.verdict is BenchVerdict.ANSWERED and lo.verdict is BenchVerdict.ANSWERED))


def _probe_motion(arm: URRobotArm) -> None:
    """One commanded joint move, which `URRobotArm.move_joint` gates through `SafetyPreflight`."""
    print("\n=== 6. a commanded motion through SafetyPreflight ===")
    q0 = list(arm.get_joint_positions().values)
    target = list(q0)
    target[5] += 0.15  # wrist 3, about 8.6 deg: the joint with the least reach consequence
    t0 = time.time()
    # The RobotArm Protocol declares move_joint() -> None and raises on refusal, so "returned None"
    # is success and not silence. Reading a truthy return here reports a failure on a move that
    # worked.
    arm.move_joint(JointPositions(target))
    note("move_joint(+0.15 rad on wrist 3)", f"accepted (no raise) in {time.time() - t0:.1f}s")
    time.sleep(1.0)
    q1 = list(arm.get_joint_positions().values)
    note("joint actually moved", f"delta={round(q1[5] - q0[5], 4)} rad",
         ok=abs(q1[5] - q0[5]) > 0.05)
    arm.move_joint(JointPositions(q0))  # put it back


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python scripts/ursim/probe_our_driver.py",
        description="Drive URSim through the URRobotArm driver: connect, capabilities, status, "
                    "poses, digital I/O and one gated joint move. Needs URSim in remote control.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ursim' or 'ursim,ursim_ur3'")
    ap.add_argument("--yes", action="store_true",
                    help="confirm the digital-output writes in section 5")
    args = ap.parse_args(argv)

    chain = args.profile if args.profile is not None else (active_profile() or _DEFAULT_PROFILE)
    cfg = load_robot_config(profile=chain)
    print(f"=== profile {chain!r}: vendor={cfg.vendor} ip={cfg.ur.ip} model={cfg.ur.model} "
          f"tool_frame={cfg.gripper.tool_frame.source} "
          f"payload={cfg.safety.payload.mass_kg} kg ===\n")

    print("=== 1. build + connect the driver (payload push + tool-frame verification) ===")
    built = create_arm(RobotVendor.from_string(cfg.vendor), config=cfg)
    note("create_arm", type(built).__name__)
    if not isinstance(built, URRobotArm):
        note("URRobotArm", f"profile {chain!r} builds {type(built).__name__}", ok=False)
        print("\nthis probe reads UR-only seams; point --profile at a UR cell")
        return _BAD_REQUEST
    arm = built
    try:
        arm.connect()
        note("arm.connect()", "connected: payload pushed and tool frame verified")
    except Exception as exc:  # noqa: BLE001
        note("arm.connect()", f"{type(exc).__name__}: {exc}", ok=False)
        print("\nno controller reachable: is URSim up and in REMOTE control?")
        return _BAD_REQUEST

    try:
        _probe_capabilities(arm)
        _probe_status(arm)
        _probe_poses(arm, float(cfg.gripper.tool_frame.offset_mm[2]))
        if isinstance(arm, SupportsDigitalIO):
            _probe_io(arm, args.yes)
        else:
            note("digital I/O", "the arm does not advertise SupportsDigitalIO", ok=False)
        _probe_motion(arm)
    finally:
        try:
            arm.disconnect()
            note("arm.disconnect()", "clean")
        except Exception as exc:  # noqa: BLE001
            note("arm.disconnect()", f"{type(exc).__name__}: {exc}", ok=False)

    print("\n=== SUMMARY ===")
    bad = [step for step, _outcome, ok in results if not ok]
    print(f"  {len(results) - len(bad)}/{len(results)} steps OK")
    if bad:
        print(f"  NOT OK: {bad}")
    print("OUR_DRIVER_PROBE_DONE")
    return _FAILED if bad else _OK


if __name__ == "__main__":
    raise SystemExit(main())
