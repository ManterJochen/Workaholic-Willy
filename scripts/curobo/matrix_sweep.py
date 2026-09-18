"""Run the matrix gate over every combination the retract table judged, one sidecar at a time.

    .venv/Scripts/python.exe scripts/curobo/matrix_sweep.py --write

The matrix is derived from ``robot/ur_retract.yaml`` rather than listed here: a pair the rule found no pose for has
nothing to start a sidecar from, and a list typed beside the table would drift from it. Each combination is measured
in its own sidecar, because a sidecar holds one robot.

A combination that does not reach the bar is a result, not a failure of this sweep. It writes its file with the
counts that show why, and the sweep carries on, so a refusal is measured rather than inferred from silence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def combinations(table_path: Path) -> "list[tuple[str, str, float, float, tuple[float, ...]]]":
    """Every (arm, hand, plate, margin, rotation) the table judged, in a stable order.

    ``rotation`` is the four numbers of the frame a row judged at another placement than the rule's carries, and empty
    for a row at the rule's own, which the gate measures with the frame undeclared.
    """
    import yaml

    table = yaml.safe_load(table_path.read_text(encoding="utf-8")) or {}
    seen = {
        (str(row["arm"]), str(row["hand"]), float(row["plate_mm"]), float(row["planner_margin_mm"]),
         tuple(float(v) for v in row.get("tool_rotation_xyzw") or ()))
        for row in (table.get("retracts") or [])
    }
    return sorted(seen)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Measure every judged combination and write its evidence file.")
    parser.add_argument("--write", action="store_true", help="write the files; without it nothing is written")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument("--table", default=str(REPO / "src/robot/safety/planning/robot/ur_retract.yaml"))
    args = parser.parse_args(argv)

    rows = combinations(Path(args.table))
    print(f"[sweep] {len(rows)} combinations from {Path(args.table).name}")
    gate = REPO / "scripts" / "curobo" / "matrix_gate.py"
    passed, below, broken = [], [], []

    for index, (arm, hand, plate, margin, rotation) in enumerate(rows, start=1):
        command = [sys.executable, str(gate), "--arm", arm, "--hand", hand,
                   "--coupling-mm", f"{plate:g}", "--planner-margin-mm", f"{margin:g}"]
        if rotation:
            command += ["--tool-rotation-xyzw", *(repr(v) for v in rotation)]
        if args.write:
            command.append("--write")
        if args.evidence_dir:
            command += ["--evidence-dir", args.evidence_dir]
        started = time.monotonic()
        done = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO))
        took = time.monotonic() - started
        line = next((row for row in done.stdout.splitlines() if row.startswith("[gate]")), "")
        label = f"{arm} {hand} c{plate:g}{' at ' + ' '.join(f'{v:g}' for v in rotation) if rotation else ''}"
        if done.returncode != 0:
            broken.append((label, (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]))
            print(f"  {index:2d}/{len(rows)} {label:34s} BROKEN after {took:5.1f}s")
            continue
        (passed if ": b1 (" in line else below).append(label)
        print(f"  {index:2d}/{len(rows)} {label:34s} {took:5.1f}s  {line[len('[gate] '):]}")

    print(f"[sweep] {len(passed)} at b1, {len(below)} below it, {len(broken)} could not be measured")
    for label in below:
        print(f"  below b1: {label}")
    for label, why in broken:
        print(f"  broken:   {label}: {why[0] if why else ''}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
