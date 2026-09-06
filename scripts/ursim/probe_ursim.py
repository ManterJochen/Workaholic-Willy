"""Drive URSim with ur_rtde directly, with none of this repository's code in the path.

Every UR test in this repository runs against a fake, either a `MagicMock` or a hand-rolled stub.
That proves the driver calls the right things; it cannot prove a controller accepts them. URSim
runs the real PolyScope and URControl stack and speaks real RTDE, so this script measures what the
tests only assert.

The rawness is deliberate and is the whole point of the file, so do not rewrite it onto
`URConnection`. `probe_our_driver.py` runs the same controller through `URRobotArm`; when that one
fails, the only cheap question is whether the fault is this repository's or the SDK's, and this
script is the answer. Two of its four dashboard verbs have no library door in any case: the
dashboard client `URConnection` builds inside `connect()` exposes safety status, remote control,
model, serial number and protective-stop release, and nothing that powers a controller or
releases its brakes.

It imports nothing from this repository, which is why it needs no repository root on `sys.path`
while every other probe in this directory does.

`HOST` is fixed at the loopback address, and that fixed value is the only interlock in front of a
script that drives standard outputs 0 and 4 and tool output 0 without asking anybody first.
Pointing it at a cell address would energise real coils.

`setPayload(1.5, [0.0, 0.0, 0.06])` mirrors `safety.payload.mass_kg` and `cog_mm` in
`config/robot/robot.ursim.yaml`. The pair is deliberate: a config-free probe is the design, so the
two declarations move together or the probe stops describing the profile it stands in for.

Run (URSim up, in REMOTE control):  python scripts/ursim/probe_ursim.py

Exit codes: 0 every step answered, 1 a step failed, 2 the dashboard is unreachable.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Any

HOST = "127.0.0.1"
_DASHBOARD_PORT = 29999

_OK, _FAILED, _NO_CONTROLLER = 0, 1, 2

results: list[tuple[str, str]] = []


def note(step: str, outcome: str) -> None:
    results.append((step, outcome))
    print(f"  {step:<44} {outcome}", flush=True)


def dashboard(cmds: list[str], settle: float = 0.4) -> list[str]:
    """Speak the dashboard protocol. Returns one reply per command."""
    s = socket.create_connection((HOST, _DASHBOARD_PORT), timeout=15)
    s.settimeout(15)
    s.recv(256)  # greeting
    out = []
    for c in cmds:
        s.sendall((c + "\n").encode())
        time.sleep(settle)
        try:
            out.append(s.recv(512).decode(errors="replace").strip())
        except Exception as exc:  # noqa: BLE001
            out.append(f"<no reply: {exc}>")
    s.close()
    return out


def wait_for_mode(target: str, timeout_s: float = 90.0) -> str:
    """Poll `robotmode` until it reports `target`. Returns the last mode seen."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        last = dashboard(["robotmode"])[0]
        if target in last:
            return last
        time.sleep(2)
    return last


def _step_dashboard() -> bool:
    """Is PolyScope really up: does the dashboard greet and then answer its own protocol?"""
    print("\n=== 1. PolyScope is really up (dashboard greets + answers) ===", flush=True)
    try:
        mode, safety = dashboard(["robotmode", "safetystatus"])
    except Exception as exc:  # noqa: BLE001
        note("dashboard", f"UNREACHABLE: {exc}")
        return False
    note("dashboard robotmode", mode)
    note("dashboard safetystatus", safety)
    return True


def _step_power_on() -> None:
    """Power the controller and release the brakes. RTDE control needs the mode RUNNING."""
    print("\n=== 2. Power on + brake release (RTDE control needs RUNNING) ===", flush=True)
    note("power on", dashboard(["power on"])[0])
    note("mode after power on", wait_for_mode("IDLE", 60))
    note("brake release", dashboard(["brake release"])[0])
    note("mode after brake release", wait_for_mode("RUNNING", 90))


def _step_receive() -> Any:
    """The read-only data path. Returns the receive interface, or None where it failed.

    The interface is returned rather than kept in a local, because step 4 reads outputs back
    through it. A step that borrows a name bound in an earlier try-block reports a `NameError`
    against itself for a fault that belongs to the earlier step.
    """
    print("\n=== 3. rtde_receive: the DATA path (read-only, no motion) ===", flush=True)
    try:
        import rtde_receive

        r = rtde_receive.RTDEReceiveInterface(HOST)
        q = r.getActualQ()
        note("getActualQ", f"{[round(v, 4) for v in q]}")
        note("getActualTCPPose", f"{[round(v, 4) for v in r.getActualTCPPose()]}")
        note("getRobotMode / getSafetyMode", f"{r.getRobotMode()} / {r.getSafetyMode()}")
        note("isProtectiveStopped", str(r.isProtectiveStopped()))
        note("isEmergencyStopped", str(r.isEmergencyStopped()))
        note("getActualTCPForce", f"{[round(v, 3) for v in r.getActualTCPForce()]}")
        note("getDigitalOutState(0)", str(r.getDigitalOutState(0)))
        return r
    except Exception as exc:  # noqa: BLE001
        note("rtde_receive", f"FAILED: {type(exc).__name__}: {exc}")
        return None


def _step_io(recv: Any) -> None:
    """The digital outputs a jaw or a vacuum gripper is switched on.

    The read-back is the measurement. `setStandardDigitalOut` returning without raising says the
    command was accepted, not that the pin moved, and a wrong bank looks exactly like success.
    """
    print("\n=== 4. rtde_io: the gripper path (digital outputs) ===", flush=True)
    if recv is None:
        note("rtde_io", "SKIPPED: rtde_receive failed, so there is nothing to read a pin back with")
        return
    try:
        import rtde_io

        io = rtde_io.RTDEIOInterface(HOST)
        for pin in (0, 4):
            io.setStandardDigitalOut(pin, True)
            time.sleep(0.3)
            hi = recv.getDigitalOutState(pin)
            io.setStandardDigitalOut(pin, False)
            time.sleep(0.3)
            lo = recv.getDigitalOutState(pin)
            note(f"standard out {pin}: set high/low, read back", f"{hi} / {lo}")
        io.setToolDigitalOut(0, True)
        time.sleep(0.3)
        note("tool out 0 set high", "accepted")
        io.setToolDigitalOut(0, False)
    except Exception as exc:  # noqa: BLE001
        note("rtde_io", f"FAILED: {type(exc).__name__}: {exc}")


def _step_control() -> None:
    """The control interface, which uploads URScript and takes the robot."""
    print("\n=== 5. rtde_control: uploads URScript, takes the robot ===", flush=True)
    try:
        import rtde_control

        c = rtde_control.RTDEControlInterface(HOST)
        note("RTDEControlInterface constructed", "yes")
        note("isConnected", str(c.isConnected()))
        note("getForwardKinematics()", f"{[round(v, 4) for v in c.getForwardKinematics()]}")
        ok = c.setPayload(1.5, [0.0, 0.0, 0.06])
        note("setPayload(1.5 kg, cog 60 mm in Z)", f"returned {ok}")
        note("isSteady", str(c.isSteady()))
        c.stopScript()
        note("stopScript", "sent")
    except Exception as exc:  # noqa: BLE001
        note("rtde_control", f"FAILED: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    # It takes no options. The parser is here so `--help` answers and a stray argument is refused,
    # rather than accepted and ignored, which is the shape every other probe in this directory has.
    argparse.ArgumentParser(
        prog="python scripts/ursim/probe_ursim.py",
        description="Drive URSim with ur_rtde directly: dashboard, power on, brake release, then "
                    "the receive, I/O and control interfaces. No repository code in the path.",
    ).parse_args(argv)

    if not _step_dashboard():
        print("PROBE_DONE", flush=True)
        return _NO_CONTROLLER
    _step_power_on()
    recv = _step_receive()
    _step_io(recv)
    _step_control()

    print("\n=== SUMMARY ===", flush=True)
    bad = [s for s, o in results if "FAILED" in o or "SKIPPED" in o or "UNREACHABLE" in o]
    print(f"  {len(results) - len(bad)}/{len(results)} steps produced a real answer", flush=True)
    if bad:
        print(f"  FAILED: {bad}", flush=True)
    print("PROBE_DONE", flush=True)
    return _FAILED if bad else _OK


if __name__ == "__main__":
    raise SystemExit(main())
