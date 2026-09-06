"""Verify the motion stack's external engines are wired on this box, for this cell.

    python -m src.robot.safety.planning --check
    python -m src.robot.safety.planning --check --profile ur3e     # read the cell's own model
    python -m src.robot.safety.planning --check --model ur3e --json
    python -m src.robot.safety.planning --doctor                   # load every engine (slower)

It prints the resolved status of both engines the motion stack builds on, the cuRobo
planner sidecar and the Coal or python-fcl exact-mesh collision engine, and exits:

* ``0`` for fully anchored: the cuRobo environment python is present, and an
  exact-mesh engine and the reference mesh bundle are importable.
* ``1`` for partially anchored: at least one engine is missing, so a degraded fallback
  is in force, meaning blind IK instead of cuRobo, the capsule proxy instead of exact
  meshes, or both.

A development box without the GPU environment reports ``1`` by design. The check is
meant to run on the target cell to confirm the anchoring.

``--check`` reads paths and ``--doctor`` reads reality. The doctor imports Coal and
runs a real distance query, spawns the interpreter of the cuRobo sidecar to see what
resolves there, including the robot descriptor ``--check`` can only guess at and
whether a second kernel backend exists, and classifies an OS application-control
refusal as its own outcome. That is exit ``2``, kept separate from ``1``, because a
blocked binary and a box with no GPU environment need opposite responses. It costs a
subprocess and a few seconds, so it stays opt-in rather than folded into ``--check``.
See :mod:`.doctor` and ``docs/code-integrity.md``.

Both readings are per-robot. The mesh bundle ships as
``{model}_collision_meshes.npz`` and the cuRobo descriptor as ``{model}.yml``, so a
present ``ur5e`` bundle says nothing about a UR3e cell. This entry point reads the
model from the config the cell will load, and ``--model`` overrides it for a box with
no config tree.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.contracts import UNSET

from .stack import MotionStack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.robot.safety.planning",
        description="Report whether the cuRobo planner + the exact-mesh collision engine are wired.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="probe both engines, print the status report, exit 0 iff fully anchored (default action)",
    )
    parser.add_argument(
        "--doctor", action="store_true",
        help="deep check: LOAD every engine and name whatever refused. Exit 2 if an OS policy blocks.",
    )
    parser.add_argument(
        "--model", default=None,
        help="robot model to probe (e.g. ur3e). Default: read from the config this box would load.",
    )
    parser.add_argument("--profile", default=None, help="profile chain to read the model from")
    parser.add_argument("--data", default=None, help="config data directory")
    parser.add_argument(
        "--json", action="store_true",
        help="emit the reading as JSON instead of prose (for the operator console and for scripts)",
    )
    args = parser.parse_args(argv)

    # `MotionStack` owns the three-step resolution, and `api/routers/diagnostics.py`
    # calls the same one. This line maps only the argparse `None`, meaning nobody typed
    # the flag, onto `UNSET`, meaning the caller did not choose, which is an argparse
    # fact and belongs where argparse is.
    stack = MotionStack.for_this_box(
        profile=UNSET if args.profile is None else args.profile,
        data_dir=UNSET if args.data is None else args.data,
        model=args.model if args.model else UNSET,
    )
    model, source = stack.model, stack.model_source

    if args.doctor:
        from .doctor import run_doctor

        report = run_doctor(model=model, robot_config=f"{model}.yml")
        if args.json:
            print(json.dumps(
                {
                    "model": model,
                    "model_source": source,
                    "healthy": report.healthy,
                    "policy_blocked": report.blocked,
                    "probes": [
                        {"name": p.name, "status": str(p.status), "detail": p.detail, "remedy": p.remedy}
                        for p in report.probes
                    ],
                    "blocked_files": list(report.blocked_files),
                },
                indent=2,
            ))
        else:
            print(report.render())
            print(f"  (model: {model}, from {source})")
        return report.exit_code

    # A different name from the doctor report above, because the two are different
    # types. Two reports in one function under one variable name is how `to_dict()`
    # gets called on the object that does not have one.
    reading = stack.probe()

    if args.json:
        print(json.dumps(reading.to_dict(), indent=2))
    else:
        print(reading.render())
    return reading.exit_code


if __name__ == "__main__":
    sys.exit(main())
