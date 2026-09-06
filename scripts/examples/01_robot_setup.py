"""01: point the stack at a real cell, and find out what is wrong with it before anything moves.

    python scripts/examples/01_robot_setup.py                    # rehearse: check, command nothing
    python scripts/examples/01_robot_setup.py --profile ur3e
    python scripts/examples/01_robot_setup.py --live             # also connect and read the arm back

This is the first thing to run against a new cell. It answers, in order: which robot does the config
describe, would the safety layer accept it, does the controller answer, and does the arm agree with
us about its own tool frame and payload.

It drives the real builders rather than re-implementing them. `run_config_preflight` is the same
check `python -m src.robot.execution.real_cell --check` runs, and `create_arm` is the same factory
the production path uses. An example that hand rolls its own version of a check teaches a pipeline
that does not exist.

What a green run here does not mean. The config preflight reads your YAML; the connect step reads
the controller. Neither knows what is bolted to the robot. Payload, tool frame and workspace are
declared values, and this example checks that the declaration is coherent and that the controller
accepts it, never that it matches the hardware in front of you.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import Example, hardware_parser, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the cell is described coherently and (with --live) answered
  1  a check came out wrong: read the FAILED lines and the preflight above them
  2  nothing to check yet: no config, or the controller is unreachable
"""


def main(argv: list[str] | None = None) -> int:
    args = hardware_parser(__doc__ or "", epilog=EPILOG).parse_args(argv)

    import os

    from src.config.loader import load_config
    from src.robot.core import RobotVendor, SupportsRobotStatus
    from src.robot.execution.real_cell.preflight import run_config_preflight

    with Example("01 robot_setup", "describe a cell, then ask the cell",
                 live=args.live, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            try:
                config = load_config().robot
            except Exception as error:                               # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the config tree did not load ({type(error).__name__}: {error})",
                    "check `python -m src.config --print`, and WILLY_PROFILE if you set one")
            if config is None:
                # `robot` is optional in the config tree: a perception only deployment has none.
                # Without this branch the example would die with `AttributeError` on such a tree,
                # which reads as a broken example rather than as an unconfigured cell.
                return not_ready(
                    "this config tree has no `robot` block",
                    "add one (config/robot/robot.yaml is the shipped one), or select a profile "
                    "that has it: WILLY_PROFILE=ursim")
            report(f"vendor {config.vendor}, profile {os.environ.get('WILLY_PROFILE', '(default)')}")

        with run.step("what the config says this cell is") as report:
            report(f"ip {config.ur.ip}; tool frame from {config.gripper.tool_frame.source}; "
                   f"payload {config.safety.payload.mass_kg} kg")

        with run.step("safety preflight over the config") as report:
            outcome = run_config_preflight(config)
            report(f"{len(outcome.checks)} check(s), "
                   f"{'all pass' if outcome.ok else f'{len(outcome.blocking)} blocking'}")
        if not outcome.ok:
            # Reported in full, then stopped. A cell whose declared configuration is already refused
            # must not be connected to: every later step would be measuring a cell we have said we
            # cannot drive.
            #
            # The status is the report's own, not `run.exit_code`. A finding is deliberately non
            # blocking, so returning `run.exit_code` here would answer 0 for a cell that cannot be
            # driven. `PreflightReport.exit_code` answers exactly this question: 0 when nothing
            # blocks, 1 when something does.
            run.note(outcome.render())
            run.finding("config preflight",
                        f"{len(outcome.blocking)} blocking check(s); fix these before connecting")
            return outcome.exit_code

        if not args.live:
            run.note("Rehearsal stops here. `--live` adds: connect, read the tool frame and payload")
            run.note("back off the controller, and compare them against the config above.")
            return run.exit_code

        from src.robot.drivers import create_arm

        # `create_arm`, not a whole cell. `build_real_cell` opens one RGB-D camera and loads two
        # models onto the GPU, and the question here is only whether the controller answers.
        arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
        with run.step("connect") as report:
            try:
                arm.connect()
            except Exception as error:                               # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the controller did not answer ({type(error).__name__}: {error})",
                    f"is {config.ur.ip} reachable, is the robot in REMOTE control, and is the "
                    f"program running? For a simulator: scripts/ursim/ursim.sh up MODEL")
            # `connect()` does more than open a socket on the UR path: it pushes the payload and
            # verifies the tool frame against the controller. That is why this step is the one that
            # catches a config which describes a different robot than the one on the bench.
            report("connected: payload pushed and tool frame verified against the controller")

        try:
            with run.step("read the pose back, in our units") as report:
                pose = arm.get_tcp_pose()
                report(f"frame {pose.frame.value}, "
                       f"pos {[round(float(v), 1) for v in pose.position_mm]} mm")

            with run.step("the controller's own state") as report:
                # A capability, not part of `RobotArm`. Vendors differ in what they can report, so
                # the stack feature checks the Protocol instead of assuming, and an example that
                # called this blind would crash on any driver that does not carry it.
                if not isinstance(arm, SupportsRobotStatus):
                    report(f"{type(arm).__name__} does not report controller state "
                           f"(SupportsRobotStatus not implemented), skipped")
                else:
                    status = arm.get_robot_status()
                    report(f"{status.robot_mode.value} / {status.safety_mode.value}; "
                           f"protective {status.protective_stopped}; "
                           f"operational {status.is_operational}")
                    if status.is_stopped:
                        run.finding("controller state",
                                    "the cell is STOPPED; clear it before a pick")
        finally:
            arm.disconnect()
            run.note("disconnected.")

        run.note("")
        run.note("Next: 02_calibration.py. The camera does not know where the robot is yet.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
