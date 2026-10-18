"""Measure where an arm of a cell stands, by touching the same marks with both arms (owner, 2026-10-15).

    python -m src.robot.execution.real_cell.register --cell clip --arm werker
    python -m src.robot.execution.real_cell.register --cell D:/zellen/clip/cells/clip.yaml --arm werker --points 5

Mark a few points both arms reach, spread out and not in a row (a cross on the table is enough). The cell's first arm
is connected alone and freed: guide its TCP onto each mark and press Enter there; it holds between marks. Then the
arm to register, onto the same marks in the same order. The fit of the two sets is that arm's base in the cell, and
every mark's disagreement is printed. Nothing moves by itself; nothing is calibrated automatically.

The result is written beside the cell file (``<cell folder>/<cell name>/<arm>.json``), in the user's own tree.

Exit codes: 0 registered and written; 1 refused (the cell, an arm, the touches); 2 usage.
"""

from __future__ import annotations

import argparse

from src.contracts import UNSET


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.robot.execution.real_cell.register",
        description="Measure where an arm of a cell stands, by touching the same marks with both arms.",
    )
    ap.add_argument("--cell", required=True, help="the cell file, or its name in <tree>/cells/")
    ap.add_argument("--arm", required=True, help="the arm to register (not the cell's first arm)")
    ap.add_argument("--points", type=int, default=4, help="how many marks to touch (default 4, at least 3)")
    ap.add_argument("--data-dir", type=str, default=None, help="the tree a bare cell name is read in")
    args = ap.parse_args(argv)

    from src.config import ConfigError  # noqa: PLC0415
    from src.config.cell import load_cell  # noqa: PLC0415
    from src.robot.core.errors import RobotError  # noqa: PLC0415
    from src.robot.execution.arm_registration import (  # noqa: PLC0415
        MIN_POINTS,
        RegistrationRefused,
        register_arm,
        registration_file,
    )
    from src.robot.execution.hand_guiding import HandGuidingRefused  # noqa: PLC0415

    if args.points < MIN_POINTS:
        ap.error(f"--points {args.points}: at least {MIN_POINTS} marks")
    cell = load_cell(args.cell, root=args.data_dir if args.data_dir else UNSET)
    print(cell, flush=True)
    if not cell.ok:
        return 1
    try:
        registration = register_arm(cell, args.arm, count=args.points)
    except (ConfigError, RegistrationRefused, HandGuidingRefused, RobotError) as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 1
    print(registration, flush=True)
    print(f"written to {registration_file(cell, registration.arm)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
