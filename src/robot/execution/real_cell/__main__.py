"""Run a config-driven pick on a real cell: the live caller of ``from_robot_config``.

    python -m src.robot.execution.real_cell --check              # config checklist, nothing else
    python -m src.robot.execution.real_cell --rehearse --runs 3  # full path, dummy arm, no camera
    python -m src.robot.execution.real_cell --dry-run            # real config, build only, no motion
    python -m src.robot.execution.real_cell --runs 10            # the real cell

``AutonomousGraspService.from_robot_config`` is the composition root this project points at. This
runner is what exercises it in combination: the vendor driver, the gripper branch, the safety
preflight, the frame resolver and record logging, in one program.

The four stages, and what each one proves:

  1. Config      load + validate the tree (with profile layers), then ``run_config_preflight``: every
                 stop-the-cell condition that is decidable at a desk, each with its concrete fix.
  2. Build       ``from_robot_config``: the vendor arm, the gripper branch, the safety preflight, the
                 CAMERA->BASE resolver, record provenance. Refuses rather than guesses.
  3. Connect     arm first, then gripper. That order is a contract, not a style: ``VacuumGripper``
                 drives digital I/O the moment it connects, and ``from_robot_config`` deliberately
                 never connects the arm (the caller owns the lifecycle). connect() also verifies the
                 controller's actual tool frame and its payload, and fails closed on either.
  4. Pick        ``service.pick()`` per run, with the typed outcome and reason printed per attempt.

Exit codes: 0 all runs succeeded (or --check/--dry-run passed); 1 preflight blocked or the build
refused; 2 the cell connected but at least one pick did not succeed; 3 an unexpected error.

Nothing below the rehearsal path has run against a physical controller. ``--rehearse`` is the same
wiring with a dummy arm and a synthetic scene.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

from src.contracts import UNSET

from ..cell import Cell
from ..cell_lock import CellBusy
from ..lifecycle import ConnectStage
from ..pick_run import PickOutcome, PickRun, Recording

_EXIT_OK, _EXIT_CONFIG, _EXIT_PICK, _EXIT_ERROR = 0, 1, 2, 3


def _load_robot_config(profile: str | None, data_dir: str | None) -> "RobotConfig":
    """Load the config tree, honouring an explicit profile chain.

    The translation line is the whole function. argparse hands back ``None`` for a flag nobody
    typed, while ``load_robot_config(profile=None)`` means "the base tree, ignore
    ``WILLY_PROFILE``". Passing one straight into the other silently disables the environment
    variable for every operator who exports it and does not also pass ``--profile``. ``UNSET`` is
    what "nobody typed it" translates to.

    Saving, setting and restoring ``os.environ["WILLY_PROFILE"]`` around the load is not the way to
    do this: process-global state for the duration of one call leaks to any other thread and
    survives an exception. `drivers/ur/__main__.py` and `real_cell/calibrate.py` translate the same
    way.
    """
    from src.config.loader import load_robot_config

    return load_robot_config(data_dir, profile=UNSET if profile is None else profile)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.robot.execution.real_cell",
        description="Config-driven pick loop on a real cell (the first live caller of from_robot_config).",
    )
    ap.add_argument("--check", action="store_true",
                    help="run the config preflight and stop: no build, no hardware, no motion")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight + build the service, then stop. Proves the wiring without moving")
    ap.add_argument("--rehearse", action="store_true",
                    help="dummy arm + synthetic scene: the whole path at a desk, no camera, no robot")
    ap.add_argument("--runs", type=int, default=1, help="how many picks to attempt (default 1)")
    ap.add_argument("--prompt", type=str, default="an object",
                    help="what the vision front-end should look for (real cell only)")
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ur3e' or 'ur3e,tiltcam'")
    ap.add_argument("--data-dir", type=str, default=None, help="override the config tree root")
    args = ap.parse_args(argv)

    # ---- 1. Config ---------------------------------------------------------------------------
    try:
        robot_cfg = _load_robot_config(args.profile, args.data_dir)
    except Exception as exc:  # noqa: BLE001 (a config fault must read as a config fault)
        print(f"[config] FAILED to load: {type(exc).__name__}: {exc}", flush=True)
        return _EXIT_CONFIG
    # The four stages below are `Cell`'s four steps, and the order is the cell's rather than this
    # file's: preflight before build, attestation before motion, and the lock before the connect.
    # The banners stay here, because narration belongs to whoever is being narrated to, and this
    # one is a bench.
    cell = (Cell.rehearsal(robot_cfg) if args.rehearse
            else Cell.from_robot_config(robot_cfg, prompt=args.prompt))
    vendor = "dummy (rehearsal)" if args.rehearse else cell.vendor
    print(f"\n=== 1. CONFIG === vendor={vendor} profile={args.profile or '<none>'}", flush=True)

    report = cell.preflight()
    print(report.render(), flush=True)
    # Two rules, and only one of them is a verdict. The first is a policy this runner applies: a
    # rehearsal continues past a blocking checklist, because it exists to exercise the path at a
    # desk where blocking items are the expected state. The second is the report's verdict, which
    # is what `--check` asks for, so it reads `report.exit_code` rather than restating the rule.
    #
    # The two read alike and are not the same rule. On this tree `--rehearse --check` exits 0 and
    # `--check` exits 1, because a rehearsal preflights the dummy config, which blocks on nothing.
    # They can only disagree for a config whose rehearsal still blocks.
    if not report.ok and not args.rehearse:
        print("\nrefusing to continue: fix the blocking items above.", flush=True)
        return _EXIT_CONFIG
    if args.check:
        return report.exit_code

    # ---- 2. Build ----------------------------------------------------------------------------
    print("\n=== 2. BUILD === from_robot_config", flush=True)
    try:
        service = cell.build()
    except Exception as exc:  # noqa: BLE001 (a build refusal is the designed outcome, not a crash)
        print(f"[build] REFUSED: {type(exc).__name__}: {exc}", flush=True)
        return _EXIT_CONFIG
    print(f"  arm      {type(cell.arm).__name__}", flush=True)
    print(f"  gripper  {type(cell.gripper).__name__}", flush=True)
    print(f"  records  {cell.record_log_path or '<not logged>'}", flush=True)
    # What this arm will actually refuse, printed before anything moves. The posture is read from
    # the arm at run time rather than derived from the driver registry, because a registry cannot
    # see an arm a caller supplied.
    #
    # Above the --dry-run exit on purpose: "what would this cell enforce" is exactly the question
    # a dry run is asked to answer.
    attestation = cell.safety()
    print(attestation.render(), flush=True)
    if not attestation.enforced and not args.rehearse:
        print("             ^ a real run with nothing gating it. --rehearse is the safe way to "
              "exercise this path.", flush=True)
    if args.dry_run:
        print("\n--dry-run: built cleanly, stopping before any motion.", flush=True)
        return _EXIT_OK

    # ---- 3. Connect --------------------------------------------------------------------------
    # The order, the rollback and the teardown live in `lifecycle`, not here, and one
    # implementation serves this runner and the operator console. A teardown that disconnects the
    # arm alone leaves a vacuum line asserted, because a vacuum cup's disconnect is what releases
    # its output, and leaves the camera open.
    print("\n=== 3. CONNECT === arm first, then gripper", flush=True)
    def _narrate(stage: ConnectStage) -> None:
        """The bench wording. `lifecycle` owns the order; this file owns how it reads."""
        if stage is ConnectStage.ARM_CONNECTED:
            print("  arm connected (tool frame + payload verified)", flush=True)
        elif stage is ConnectStage.GRIPPER_MOVING:
            # This moves: Robotiq activation is a calibration sweep of the full finger travel. It is
            # here, and not on the first commanded width, precisely so it happens in the arm's
            # starting pose rather than with the fingers around the object.
            print("  connecting gripper: activation moves the fingers, keep hands clear",
                  flush=True)
        elif stage is ConnectStage.GRIPPER_CONNECTED:
            print("  gripper connected + activated", flush=True)

    # Taking the cross-process lock is part of `connected()`. A UR controller accepts one control
    # script, so whoever asks second is told who holds it. A simulated or dummy cell owns no
    # controller and yields no key, so two rehearsals can run at once.
    session = cell.connected(announce=_narrate)
    try:
        session.__enter__()
    except CellBusy as exc:
        print(f"[connect] REFUSED: {exc}", flush=True)
        return _EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001 (report the typed refusal, do not stack-trace at a bench)
        # Connect is a transaction, and `connect_cell` has already rolled the arm back and given up
        # the lock. There is nothing to undo here.
        print(f"[connect] FAILED: {type(exc).__name__}: {exc}", flush=True)
        return _EXIT_CONFIG

    # ---- 4. Pick -----------------------------------------------------------------------------
    print(f"\n=== 4. PICK === {args.runs} run(s)", flush=True)
    # The connect already happened above, so this is `from_service`: the runner owns the session
    # and the teardown, and `PickRun` must not claim to know how a cell it did not bring up came
    # down. Its `teardown` field stays None and the runner prints its own.
    run = PickRun.from_service(
        service,
        runs=args.runs,
        # Explicit, with no default at the factory. The shipped tree has `record_log_path: null`,
        # so this runner writes no corpus, and the result block says so.
        recording=Recording.off(),
        # Printed as each attempt lands, not collected and printed at the end. Without the hook
        # every `run N:` line arrives after the log lines of every pick, which at a bench is the
        # difference between watching a campaign and reading its transcript afterwards.
        on_attempt=lambda attempt: print(attempt.render(), flush=True),
    )
    pick_report = run.execute()
    if pick_report.raised:
        # At a bench a typed message beats a stack trace. `PickRun` has stopped the campaign, and
        # the teardown below still runs.
        failed = next(a for a in pick_report.attempts if a.outcome is PickOutcome.RAISED)
        print(f"[pick] ERROR: {failed.detail}", flush=True)
    elif pick_report.last is not None:
        # Which layers actually ran, once. Every advanced grasping block ships `enabled: false`,
        # so the default pick is open-loop, and a run that printed only its outcomes would read
        # identically whether one layer fired or six. Printed once rather than per run because it is
        # a property of the service, not of an attempt.
        print("", flush=True)
        print(pick_report.last.render(), flush=True)

    # Printed, not swallowed. A gripper that will not release must leave a trace: on a real cell
    # that is the difference between an operator knowing an output is still asserted and finding
    # out with their hands.
    session.__exit__(None, None, None)
    if session.teardown is not None:
        print("\n=== 5. DOWN === gripper first, then arm, then cameras", flush=True)
        print(session.teardown.render(), flush=True)

    print("", flush=True)
    print(pick_report.summary(), flush=True)
    return pick_report.exit_code


if __name__ == "__main__":
    sys.exit(main())
