"""Does teaching poses by hand hold what `teach_poses` promises, on real controller software?

`src/robot/execution/teach.py` teaches joint poses in one hand-guiding session: a name while the arm holds,
the arm freed, Enter once it stands still, then held and read, until the person finishes. This runs it through
`Robot.from_config(..., gripper=None)` and `robot.connected()` against the CB3 URSim, with a scripted person at the
console (nobody pushes the simulated arm, so it stands still wherever it was left), and checks each promise against
the controller's own answer. The arm is sent to a station inside the workspace box first, because a fresh URSim
stands its TCP below the box, where Enter rightly captures nothing, and the ursim profile states no home inside it:

  1. the default name, a name typed, a skip and the finish reach the file as they should: `LOOK_1` and `PARK_VIEW`
     taught, nothing for the skip, two records in a fresh store and two lines printed to paste;
  2. every taught pose carries the joints the controller reports, and its line reads back to them in degrees;
  3. nothing moved the arm: the joints after the run are the joints before it;
  4. the session ended held and gave motion back: the dashboard says RUNNING and NORMAL, and a small `moveJ` then
     runs and arrives, and back.

Powers the arm on itself. Run (URSim up, in REMOTE control):  python scripts/ursim/probe_teach.py
The UR3e:                          WILLY_PROFILE=ursim,ursim_ur3 python scripts/ursim/probe_teach.py
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import sys
import tempfile
import time
from pathlib import Path

# Honour an explicit chain (e.g. "ursim,ursim_ur3") so the same probe runs both models.
os.environ.setdefault("WILLY_PROFILE", "ursim")

from src.config import load_tree  # noqa: E402
from src.robot.core import JointPositions  # noqa: E402
from src.robot.execution.hand_guiding import HandGuide  # noqa: E402
from src.robot.execution.robot import Robot  # noqa: E402
from src.robot.execution.teach import read_taught_poses, teach_poses  # noqa: E402

HOST = "127.0.0.1"
#: What the scripted person answers, in order: the words of a line said to them, then what they type and after how
#: long. A skip first, so the name is asked again and the file must not hold it.
_SCRIPT = (
    ("payload right", "", 0.3),
    ("Name of the next pose", "", 0.3),
    ("MOVE THE ARM BY HAND", "", 1.0),
    ("Name of the next pose", "PARK_VIEW", 0.3),
    ("MOVE THE ARM BY HAND", "s", 1.0),
    ("Name of the next pose", "PARK_VIEW", 0.3),
    ("MOVE THE ARM BY HAND", "", 1.0),
    ("Name of the next pose", "q", 0.3),
)
#: Where the arm stands while poses are taught: URSim's own FK puts the TCP of this configuration at (-133, 412, 281)
#: mm with the ursim profile's 132 mm tool frame, inside the base tree's workspace box.
_INSIDE_DEG = (-90.0, -100.0, 110.0, -100.0, -90.0, 0.0)
#: How long the scripted person waits for a line of the script before finishing.
_GIVE_UP_S = 20.0
_LINE = re.compile(r"^(\w+) = JointPositions\.deg\(([^)]*)\)")


class Person:
    """A person at the terminal who answers each line of the script once it is said to them, after a pause."""

    def __init__(self) -> None:
        self.script = list(_SCRIPT)
        self.said: list[str] = []
        self._pending: tuple[str, float] | None = None
        self._last_said = time.monotonic()

    def say(self, line: str) -> None:
        self.said.append(line)
        self._last_said = time.monotonic()
        print(f"   | {line[:150]}")
        if self.script and self.script[0][0] in line:
            _, answer, after_s = self.script.pop(0)
            self._pending = (answer, time.monotonic() + after_s)

    def bell(self) -> None:
        pass

    def poll(self) -> str | None:
        if self._pending is None and time.monotonic() - self._last_said > _GIVE_UP_S:
            # Nothing on the script was said for a while: finish, so the session holds the arm rather than a timeout
            # killing the process while it is free (which the watchdog answers with a protective stop).
            self._last_said = time.monotonic()
            print("   > 'q' (the script waited too long)")
            return "q"
        if self._pending is None or time.monotonic() < self._pending[1]:
            return None
        answer, self._pending = self._pending[0], None
        print(f"   > {answer!r}")
        return answer


def _dashboard(command: str) -> str:
    with socket.create_connection((HOST, 29999), timeout=15) as sock:
        sock.settimeout(15)
        sock.recv(256)
        sock.sendall((command + "\n").encode())
        return sock.recv(512).decode(errors="replace").strip().split(":", 1)[-1].strip()


def _running() -> str:
    """Power the simulated arm on and release its brakes, as ``probe_freedrive.py`` does; the robot mode it reached."""
    for command, wanted, within_s in (("power on", "IDLE", 60.0), ("brake release", "RUNNING", 90.0)):
        if _dashboard("robotmode") == "RUNNING":
            break
        print(f"   {command:<24} -> {_dashboard(command)}")
        since = time.monotonic()
        while time.monotonic() - since < within_s and _dashboard("robotmode") not in (wanted, "RUNNING"):
            time.sleep(0.5)
    return _dashboard("robotmode")


def _joints(robot: Robot) -> list[float]:
    return [float(v) for v in robot.arm.get_joint_positions().values]


def main() -> int:
    verdicts: dict[str, bool] = {}
    tree = load_tree()
    if not tree.ok:
        raise SystemExit(f"the tree does not load: {tree.error}")
    print(f"=== 0. starting state: safety {_dashboard('safetystatus')}, robot mode {_dashboard('robotmode')}, "
          f"remote {_dashboard('is in remote control')} ===")
    print(f"   robot mode now {_running()}")
    robot = Robot.from_config(tree.robot, gripper=None)
    store = Path(tempfile.mkdtemp(prefix="probe_teach_")) / "taught_poses.json"
    printed: list[str] = []
    person = Person()
    with robot.connected():
        inside = robot.arm.move_to_joints(JointPositions.deg(*_INSIDE_DEG))
        print(f"   to a station inside the box -> ok={inside.ok} {inside.status.value}")
        if not inside.ok:
            raise SystemExit(f"the arm did not reach the station inside the box: {inside.message}")
        before = _joints(robot)
        tcp_before = robot.arm.get_tcp_pose()
        print(f"   joints before (deg): {[round(math.degrees(v), 2) for v in before]}")

        print("\n=== 1. teach: the default name, a name typed, a skip, the finish ===")
        taught = teach_poses(robot, tree=tree, store=store, guide=HandGuide(person), out=printed.append)
        records = json.loads(store.read_text(encoding="utf-8"))
        names = [pose.name for pose in taught]
        print(f"   taught {names}; the store holds {len(records)} records; printed {len(printed)} lines")
        for line in printed:
            print(f"   {line}")
        verdicts["names, skip and finish"] = (names == ["LOOK_1", "PARK_VIEW"] and len(records) == 2
                                              and len(printed) == 2 and not person.script
                                              and any("PARK_VIEW skipped" in line for line in person.said))

        print("\n=== 2. every pose carries the controller's joints, and its line reads back to them ===")
        worst = 0.0
        for pose, line in zip(read_taught_poses(store), printed):
            match = _LINE.match(line)
            read_back = [float(v) for v in match.group(2).split(",")] if match else []
            worst = max([worst, *(abs(math.radians(d) - b) for d, b in zip(pose.joints_deg, before)),
                         *(abs(math.radians(d) - b) for d, b in zip(read_back, before))])
            print(f"   {pose.name}: {pose.taught_for()}; {pose.tcp_line()}")
        print(f"   largest difference to the controller's joints {worst:.2e} rad")
        verdicts["the controller's joints"] = worst < 1e-3 and len(read_taught_poses(store)) == 2

        print("\n=== 3. nothing moved the arm ===")
        after = _joints(robot)
        drift = max(abs(a - b) for a, b in zip(after, before))
        tcp_after = robot.arm.get_tcp_pose()
        tcp_drift = max(abs(a - b) for a, b in zip(tcp_after.position_mm, tcp_before.position_mm))
        print(f"   joints moved {drift:.2e} rad, the TCP {tcp_drift:.3f} mm")
        verdicts["nothing moved"] = drift < 1e-3 and tcp_drift < 0.5

        print("\n=== 4. held at the end, and motion given back ===")
        safety, mode = _dashboard("safetystatus"), _dashboard("robotmode")
        target = list(before)
        target[5] += 0.05
        moved = robot.arm.move_to_joints(JointPositions(target))
        miss = max(abs(a - b) for a, b in zip(_joints(robot), target))
        back = robot.arm.move_to_joints(JointPositions(before))
        print(f"   safety {safety}, robot mode {mode}; a moveJ -> ok={moved.ok}, off target {miss:.2e} rad; "
              f"back -> ok={back.ok}")
        verdicts["held, motion given back"] = (safety == "NORMAL" and mode == "RUNNING" and bool(moved.ok)
                                               and miss < 1e-2 and bool(back.ok))

    print("\n=== verdicts ===")
    for name, passed in verdicts.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print(f"   left: safety {_dashboard('safetystatus')}, robot mode {_dashboard('robotmode')}; store {store}")
    print(f"TEACH_PROBE_DONE {sum(verdicts.values())}/{len(verdicts)}")
    return 0 if all(verdicts.values()) and len(verdicts) == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
