"""Does hand guiding on a UR hold the order its session promises, on real controller software?

`drivers/planned/freedrive.py` writes the order down, measured on the CB3 URSim with the UR's own session of 85f082b.
This measures the session over the transport (port S4) through our `URRobotArm`, four ways, each against the
controller's own answer:

  1. a verb inside a session is refused, with nothing sent: the joints stand where they stood, the arm is still free;
  2. `hold()` then a `moveJ`: the session ends with the arm held, and the next move runs and arrives;
  3. a process that stops calling `sample()`: the watchdog trips a protective stop, and `sample()` then says so;
  4. a process killed while the arm is free: the controller protective-stops within about a second.

The controller is the oracle: the dashboard's `safetystatus`, polled every 50 ms, times each stop. Every stop is
unlocked again (a UR admits the unlock 5 s after the stop), so the cell is left as it was found. Nothing here moves the
arm except step 2's small wrist turn; while the arm is free nobody pushes it.

Run (URSim up, in REMOTE control):  python scripts/ursim/probe_freedrive.py
The UR3e:                          WILLY_PROFILE=ursim,ursim_ur3 python scripts/ursim/probe_freedrive.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from typing import Any

# Honour an explicit chain (e.g. "ursim,ursim_ur3") so the same probe runs both models.
os.environ.setdefault("WILLY_PROFILE", "ursim")

from src.config.loader import load_robot_config  # noqa: E402
from src.robot.core import JointPositions, RobotVendor, SupportsFreedrive  # noqa: E402
from src.robot.core.motion_result import MotionSent  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402

HOST = "127.0.0.1"
#: How long a stop may take and still count as "within about a second": 0.6 to 1.0 s were measured (2026-09-24).
_STOP_WITHIN_S = 1.5
#: The controller admits an unlock this long after a protective stop, and not before.
_UNLOCK_AFTER_S = 5.2


class Dashboard:
    """One dashboard connection, kept open so a poll costs a round trip and not a connect."""

    def __init__(self) -> None:
        self._sock = socket.create_connection((HOST, 29999), timeout=15)
        self._sock.settimeout(15)
        self._sock.recv(256)

    def ask(self, command: str) -> str:
        self._sock.sendall((command + "\n").encode())
        return self._sock.recv(512).decode(errors="replace").strip()

    def safety(self) -> str:
        return self.ask("safetystatus").split(":", 1)[-1].strip()

    def wait_for(self, wanted: str, *, since: float, within_s: float) -> float | None:
        """Seconds from ``since`` to the first poll that reads ``wanted``, or ``None`` if ``within_s`` passed first."""
        while time.monotonic() - since < within_s:
            if self.safety() == wanted:
                return time.monotonic() - since
            time.sleep(0.05)
        return None

    def mode(self) -> str:
        return self.ask("robotmode").split(":", 1)[-1].strip()

    def running(self) -> str:
        """Power the simulated arm on and release its brakes, as ``probe_ursim.py`` does; the robot mode it reached."""
        for command, wanted, within_s in (("power on", "IDLE", 60.0), ("brake release", "RUNNING", 90.0)):
            if self.mode() == "RUNNING":
                break
            print(f"   {command:<24} -> {self.ask(command)}")
            since = time.monotonic()
            while time.monotonic() - since < within_s and self.mode() not in (wanted, "RUNNING"):
                time.sleep(0.5)
        return self.mode()

    def unlock(self, stopped_at: float) -> str:
        """Unlock a protective stop and return the safety status it leaves."""
        wait = _UNLOCK_AFTER_S - (time.monotonic() - stopped_at)
        if wait > 0:
            time.sleep(wait)
        for command in ("close safety popup", "unlock protective stop"):
            print(f"   {command:<24} -> {self.ask(command)}")
        self.wait_for("NORMAL", since=time.monotonic(), within_s=10.0)
        return self.safety()

    def close(self) -> None:
        self._sock.close()


def _connected() -> Any:
    cfg = load_robot_config()
    arm = create_arm(RobotVendor.from_string(cfg.vendor), config=cfg)
    if not isinstance(arm, SupportsFreedrive):
        raise SystemExit(f"vendor {cfg.vendor!r} builds {type(arm).__name__}, which offers no hand guiding")
    arm.connect()
    return arm


def _child() -> None:
    """Step 4's other process: free the arm, say so, then sample until killed."""
    arm = _connected()
    with arm.freedrive() as session:
        session.free()
        session.sample()
        print("FREE", flush=True)
        while True:
            session.sample()
            time.sleep(0.05)


def main() -> int:
    verdicts: dict[str, bool] = {}
    dash = Dashboard()
    print(f"=== 0. starting state: safety {dash.safety()}, {dash.ask('robotmode')}, "
          f"remote {dash.ask('is in remote control')} ===")
    print(f"   robot mode now {dash.running()}")

    arm = _connected()
    start = [float(v) for v in arm.get_joint_positions().values]

    print("\n=== 1. a verb inside a session sends nothing ===")
    with arm.freedrive() as session:
        session.free()
        session.sample()
        target = list(start)
        target[5] += 0.1
        moved = arm.move_to_joints(JointPositions(target))
        session.sample()
        after = [float(v) for v in arm.get_joint_positions().values]
        drift = max(abs(a - b) for a, b in zip(after, start))
        print(f"   move_to_joints -> ok={moved.ok} status={moved.status.value} sent={moved.sent.value}")
        print(f"   message: {moved.message[:140]}")
        print(f"   joints moved {drift:.2e} rad; still free: {session.is_free}; safety {dash.safety()}")
        verdicts["verb inside sends nothing"] = (not moved.ok and moved.sent is MotionSent.NOTHING and drift < 1e-3
                                                 and session.is_free)

        print("\n=== 2. hold, then a moveJ runs ===")
        session.hold()
        print(f"   held: still free = {session.is_free}")
    target = list(start)
    target[5] += 0.05
    t0 = time.monotonic()
    moved = arm.move_to_joints(JointPositions(target))
    after = [float(v) for v in arm.get_joint_positions().values]
    miss = max(abs(a - b) for a, b in zip(after, target))
    print(f"   move_to_joints -> ok={moved.ok} in {time.monotonic() - t0:.1f} s, off target {miss:.2e} rad")
    back = arm.move_to_joints(JointPositions(start))
    print(f"   and back -> ok={back.ok}")
    verdicts["hold then moveJ"] = bool(moved.ok) and miss < 1e-2 and bool(back.ok)

    print("\n=== 3. stop sampling: the watchdog trips ===")
    raised = ""
    took: float | None = None
    stopped_at = time.monotonic()
    try:
        with arm.freedrive() as session:
            session.free()
            for _ in range(5):
                session.sample()
                time.sleep(0.1)
            last = time.monotonic()
            took = dash.wait_for("PROTECTIVE_STOP", since=last, within_s=5.0)
            stopped_at = time.monotonic()
            print(f"   protective stop {took if took is None else round(took, 2)} s after the last sample")
            try:
                session.sample()
            except Exception as exc:  # noqa: BLE001 -- what it raises is the finding
                raised = f"{type(exc).__name__}: {exc}"
            print(f"   sample() then -> {raised[:160] or 'returned, and should not have'}")
    except Exception as exc:  # noqa: BLE001 -- leaving a stopped session is reported, and the unlock still runs
        print(f"   leaving the session: {type(exc).__name__}: {str(exc)[:140]}")
    try:
        arm.disconnect()
    except Exception as exc:  # noqa: BLE001
        print(f"   disconnect: {type(exc).__name__}: {str(exc)[:80]}")
    print(f"   unlocked: safety {dash.unlock(stopped_at)}")
    verdicts["watchdog trips"] = took is not None and took <= _STOP_WITHIN_S and bool(raised)

    print("\n=== 4. killed while free: a protective stop within about a second ===")
    child = subprocess.Popen([sys.executable, __file__, "--child"], stdout=subprocess.PIPE, text=True)
    assert child.stdout is not None
    line = ""
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline and line.strip() != "FREE":
        line = child.stdout.readline()
        if not line and child.poll() is not None:
            break
    if line.strip() != "FREE":
        child.kill()
        print(f"   the child never freed the arm (exit {child.poll()})")
        verdicts["kill while free"] = False
    else:
        time.sleep(0.5)
        print(f"   child free; safety {dash.safety()}; killing it")
        killed = time.monotonic()
        child.kill()
        child.wait(timeout=10)
        took = dash.wait_for("PROTECTIVE_STOP", since=killed, within_s=5.0)
        stopped_at = time.monotonic()
        print(f"   protective stop {took if took is None else round(took, 2)} s after the kill")
        print(f"   unlocked: safety {dash.unlock(stopped_at)}")
        verdicts["kill while free"] = took is not None and took <= _STOP_WITHIN_S

    print("\n=== verdicts ===")
    for name, passed in verdicts.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print(f"   left: safety {dash.safety()}, {dash.ask('robotmode')}")
    dash.close()
    print(f"FREEDRIVE_PROBE_DONE {sum(verdicts.values())}/{len(verdicts)}")
    return 0 if all(verdicts.values()) and len(verdicts) == 4 else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        _child()
    else:
        raise SystemExit(main())
