"""03: one grasp, end to end, and a measured account of which layers ran.

    python scripts/examples/cell/03_first_pick.py                 # rehearse on a dummy arm
    python scripts/examples/cell/03_first_pick.py --live          # drive the real cell
    python scripts/examples/cell/03_first_pick.py --live --prompt "a red cube"

The decision here is not a config key with two values. It is whether the pick you are about to
run is the pick you think you configured, and the answer has three parts that this file separates
on purpose: which layers the config asked for, which layers produced something, and what the
cell that executed the motion was.

Every advanced block in `robot.grasping` ships `enabled: false`, so the default attempt is
open-loop: perceive, generate and score candidates, safety preflight and IK, move and close, log.
No AUTO decision gate, no closed-loop refine, no multi-view commit, no learned success model, no
RL. The step below counts the disabled blocks off the config, and the layers line under the pick
is read off report fields that stay `None` when a layer produced nothing. Those two are computed
from different places on purpose: a config flag says what was asked for and can be inert, and the
`None` field says what ran.

The safety layer is reported and never asserted. `SafetyPreflight` is constructed inside the
vendor driver, so a run on a dummy arm reaches no guard at all, and that driver states it: "The
dummy carries no preflight, so it simply drives". A script that prints "cleared safety" while
driving an arm that has none is worse than one that says nothing about safety at all.
`cell.safety()` asks the built arm what it will refuse and this file prints that answer, so the
claim and the check are one thing.

The frame contract is the next thing a first pick meets. Perception reports grasps in the CAMERA
frame, and a driver refuses a pose that has not been resolved into BASE: the dummy answers
"DummyRobotArm.move_linear requires Frame.BASE", and a real cell answers with
`MotionStatus.INVALID_TARGET`. The resolver is built from the calibration artifact the
calibration example writes, which is why an uncalibrated cell fails here rather than in the
gripper, and why `01_robot_setup.py` reports a missing one as a blocking preflight check.

The calculator comes from the config. `robot.grasping.calculator` selects `geometric` or `deep`,
and `Cell.build()` reaches it through `build_calculator`, so nothing in this file names a
calculator class and a cell configured for `deep` cannot quietly run the analytic stack.

A rehearsal and a live run differ by one factory. `Cell.rehearsal` changes the vendor on the
operator's own config and nothing else, so the profile chain, the gripper branch and the grasping
block are the ones the reader runs; `--live` is `Cell.from_robot_config`. The four steps after
that, preflight, build, attestation and one pick, are `Cell` plus `PickRun`, which is the
composition `python -m src.robot.execution.real_cell` runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    EXIT_FAILED,
    EXIT_NOT_READY,
    EXIT_OK,
    Example,
    hardware_parser,
    not_ready,
)

EPILOG = """
exit codes
  0  the pick ran and succeeded
  1  the pick ran and did not succeed, or it raised
  2  the run never reached a pick: no `robot` block, a blocking preflight, or a cell that
     refused to build or to connect
"""

#: `PickRunReport.exit_code` in this example's terms. The report separates a refusal before any
#: pick (1) from a pick that ran and did not pass (2) and one that raised (3); the first is a cell
#: that is not ready, the other two are a failure. An unrecognised code reads as a failure,
#: because the safe reading of "not zero" is not "fine".
_EXIT_FROM_PICK: dict[int, int] = {0: EXIT_OK, 1: EXIT_NOT_READY, 2: EXIT_FAILED, 3: EXIT_FAILED}


def _optional_blocks(grasping: Any) -> tuple[list[str], list[str]]:
    """Every `grasping.*` sub-block carrying an `enabled` switch, split into on and off.

    Walked rather than listed, because a list in an example goes stale the first time a block is
    added and then reads as an inventory. What this cannot tell you is whether an enabled block
    did anything, which is the whole reason the layers line is read off the report instead.
    """
    on: list[str] = []
    off: list[str] = []
    def walk(model: Any, prefix: str) -> None:
        for name in type(model).model_fields:
            value = getattr(model, name, None)
            if not hasattr(type(value), "model_fields"):
                continue
            enabled = getattr(value, "enabled", None)
            if enabled is not None:
                (on if bool(enabled) else off).append(f"{prefix}.{name}")
            walk(value, f"{prefix}.{name}")
    walk(grasping, "grasping")
    return on, off


def main(argv: list[str] | None = None) -> int:
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument(
        "--prompt", default="an object",
        help="what the vision front end looks for. A rehearsal runs on a synthetic scene and "
             "ignores it; only --live passes it to a cell.")
    args = parser.parse_args(argv)

    # Imported after the parse, so `--help` pays for none of it and still answers in a checkout
    # whose dependencies are not installed yet.
    from src.config.loader import load_config
    from src.robot.execution.cell import Cell
    from src.robot.execution.lifecycle import ConnectStage
    from src.robot.execution.pick_run import PickRun, Recording
    from src.robot.grippers.null import NullGripper

    what = f"one grasp: '{args.prompt}'" if args.live else "one grasp on a synthetic scene"
    with Example("03 first_pick", what, live=args.live, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            robot = load_config().robot
            report("no `robot` block in this tree" if robot is None else
                   f"vendor {robot.vendor}, calculator {robot.grasping.calculator}")
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "point the loader at a tree that has one; `config/robot/robot.yaml` "
                             "in this repository is one")

        with run.step("what the config asked for") as report:
            enabled, disabled = _optional_blocks(robot.grasping)
            report(f"{len(enabled)} optional block(s) on, {len(disabled)} off; "
                   f"default_mode {robot.grasping.default_mode}")
        run.note(f"on:  {', '.join(enabled) if enabled else '(none)'}")
        run.note("Compare that with the layers line under the RESULT block: this one is read off")
        run.note("config, that one off the report, and only the second is evidence.")
        run.note("")

        # One factory apart, and everything below it is the same for a desk and for a cell.
        cell = (Cell.from_robot_config(robot, prompt=args.prompt) if args.live
                else Cell.rehearsal(robot))

        with run.step("config preflight") as report:
            checklist = cell.preflight()
            report(f"{len(checklist.blocking)} blocking of {len(checklist.checks)} check(s)")
        if not checklist.ok:
            for line in checklist.render().splitlines():
                run.note(line)
            if args.live:
                return not_ready("the config preflight blocks this cell",
                                 "fix the BLOCK lines above, then run 01_robot_setup.py again")
            # A rehearsal continues past a blocking checklist on purpose. A desk is where the
            # blocking items are expected, and stopping there would leave the path unexercised.
            run.finding("config preflight", "blocking items, waived because this is a rehearsal")

        with run.step("build the cell, config-driven") as report:
            cell.build()
            report(f"arm {type(cell.arm).__name__}, gripper {type(cell.gripper).__name__}, "
                   f"records {cell.record_log_path or '<not logged>'}")
        if isinstance(cell.gripper, NullGripper) and cell.gripper.substitution is not None:
            # Said before the pick, not only after it. A reader who learns only from the closing
            # note that the end-effector was a stand-in has already read the SUCCEEDED line as a
            # grasp. 02_which_gripper.py is the whole fork; this is the one line it owes a pick.
            run.finding("gripper", f"substituted, reason "
                                   f"{cell.gripper.substitution.reason.value}; every close below "
                                   f"reaches no hardware")

        with run.step("what this arm will refuse") as report:
            attestation = cell.safety()
            report(f"{attestation.posture.value.upper()}, {len(attestation.guards)} guard(s)")
        for line in attestation.render().splitlines():
            run.note(line)
        run.note("`SafetyPreflight` is constructed inside the vendor driver, so only a real cell")
        run.note("runs it. The block above is read off the arm that was built, not asserted here.")

        def narrate(stage: ConnectStage) -> None:
            """Bench wording for a connect. `lifecycle` owns the order, this owns how it reads."""
            if stage is ConnectStage.ARM_CONNECTED:
                run.note("arm connected, tool frame and payload verified")
            elif stage is ConnectStage.GRIPPER_MOVING:
                run.note("connecting the gripper. Activation moves it, so keep hands clear.")
            elif stage is ConnectStage.GRIPPER_CONNECTED:
                run.note("gripper connected and activated")

        run.note("")
        with run.step("one pick, connect to teardown") as report:
            # `from_cell` owns the connect: it takes the cross-process cell lock, connects the arm
            # before the gripper, and always takes the cell down again, which is the only place a
            # gripper that did not release is reported. Recording is stated rather than inherited,
            # and the shipped tree records nothing.
            pick = PickRun.from_cell(
                cell, runs=1, recording=Recording.off(),
                announce=narrate if args.live else None,
            ).execute()
            report(f"refused: {pick.error}" if pick.error
                   else f"{pick.succeeded}/{pick.attempted} attempt(s) succeeded")

        run.note("")
        for line in pick.render().splitlines():
            run.note(line)
        if pick.last is not None:
            run.note("")
            # The service's own account of the attempt, including which optional layers produced
            # something. Every advanced grasping block ships disabled, so "(none)" is the expected
            # answer on the shipped tree and is printed rather than left out.
            for line in pick.last.render().splitlines():
                run.note(line)
            ran = pick.last.layers_that_ran()
            run.note("")
            run.note(f"asked for {len(enabled)} optional block(s); {len(ran)} produced something "
                     f"on this attempt.")
            if enabled and not ran:
                # The gap this file exists to make visible. A block can be enabled in config and
                # still be unreachable in the mode the cell runs, which is why the two counts are
                # printed side by side rather than one of them being trusted.
                run.finding("layers", "blocks are enabled in config and no layer produced "
                                      "anything; check the grasp mode gate before believing a "
                                      "measurement attributed to one of them")
        # Gated on an attempt having succeeded, because that is the only claim this note
        # qualifies. Printed after a refusal it would describe an outcome the run never reached,
        # which is the shape of prose this file exists to keep out of its own output.
        if pick.succeeded and isinstance(cell.gripper, NullGripper):
            run.note("")
            run.note("This cell built a NullGripper, which accepts every command and holds")
            run.note("nothing, so the SUCCEEDED outcome above is evidence about the path rather")
            run.note("than about a grasp. The build step named it, and the builder printed why.")

        if pick.passed:
            run.note("")
            run.note("Next: 04_planner_or_ik.py, which decides how the arm gets from here to "
                     "the grasp.")
        else:
            # A campaign that ran and did not pass is a finding rather than a failed step: the
            # step did complete, and its answer is the result the reader came for.
            detail = pick.error or (pick.attempts[-1].detail if pick.attempts else "")
            run.finding("pick", detail or "the RESULT block above says how it was judged")
        if pick.error:
            # Called for the fix line it prints. The exit code comes from `_EXIT_FROM_PICK`,
            # which maps a refusal to the same value `not_ready` returns.
            not_ready(f"the cell refused before any pick: {pick.error}",
                      "the preflight above lists what is decidable at a desk; 01_robot_setup.py "
                      "checks the same cell without moving it")

        # The verdict is the report's rather than `run.exit_code`. `Example.finding` is
        # non-blocking by design, so `run.exit_code` answers 0 for a run whose pick failed, and
        # whether the pick succeeded is the question this example is asked.
        return _EXIT_FROM_PICK.get(pick.exit_code, EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
