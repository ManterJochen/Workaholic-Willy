"""01: point the stack at a real cell, and find out what is wrong with it before anything moves.

    python scripts/examples/cell/01_robot_setup.py                # rehearse: check, command nothing
    python scripts/examples/cell/01_robot_setup.py --profile ur3e
    python scripts/examples/cell/01_robot_setup.py --live         # also connect and read the arm back

The decision is `robot.vendor`: ur, kuka, sim or dummy. Two of the six names in the vendor enum,
franka and ros2, are reserved slots with no driver, so the choice is really four, and only one of
the four can be driven by any given machine without more software installed. This example answers,
in order: which robot does the config describe, can this host drive that vendor at all, would the
safety layer accept the description, does the controller answer, and does the arm agree with us
about its own tool frame and payload.

It drives the real builders rather than re-implementing them. `run_config_preflight` is the same
check `python -m src.robot.execution.real_cell --check` runs, `Host.local().readiness()` is the same
table `python -m src.robot.drivers.doctor` prints and the same rows the startup gate queries, and
`create_arm` is the same factory the production path uses. An example that hand rolls its own
version of a check teaches a pipeline that does not exist.

What a green run here does not mean. The readiness table reads this interpreter, the config
preflight reads your YAML, and the connect step reads the controller. None of the three knows what
is bolted to the robot. Payload, tool frame and workspace are declared values, and this example
checks that the declaration is coherent and that the controller accepts it, never that it matches
the hardware in front of you.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, hardware_parser, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the cell is described coherently and (with --live) answered
  1  a check came out wrong: read the FAILED lines and the preflight above them
  2  nothing to check yet: no config, no driver for this vendor, or an unreachable controller
"""


def main(argv: list[str] | None = None) -> int:
    args = hardware_parser(__doc__ or "", epilog=EPILOG).parse_args(argv)

    import os

    from src.config.loader import load_config
    from src.robot.core import RobotConnectionError, RobotVendor, SupportsRobotStatus
    from src.robot.drivers.host import Host
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

        # The cheapest question in the stack, and the one that has to be asked first: it needs no
        # config, no network and nothing powered. `ready` on a row means a driver is registered for
        # that vendor and every SDK it needs imports in this interpreter. A row that is not ready
        # for want of a driver is a reserved slot in the enum rather than something to install.
        readiness = Host.local().readiness()
        with run.step("which vendors this host can drive") as report:
            arms = [row for row in readiness.rows if row.kind == "arm"]
            ready = [row.vendor for row in arms if row.ready]
            report(f"{len(ready)} of {len(arms)} arm vendor(s): {', '.join(ready) or '(none)'}")
        for line in readiness.render().splitlines():
            run.note(line)
        run.note("A row that is not ready for want of a driver is a reserved name in the vendor")
        run.note("enum, so no install fixes it. The gripper half of the table is 02_which_gripper.")
        run.note("")

        with run.step("this host against the vendor the config names") as report:
            if readiness.ready_for(config.vendor):
                report(f"{config.vendor}: ready")
            else:
                # A finding rather than a stop. Everything below this line except the connect is
                # decidable without a driver, and a reader on a laptop should still get the
                # preflight for the cell they are describing.
                row = readiness.row_for("arm", config.vendor)
                report(f"{config.vendor}: NOT ready, {row.note if row else 'no such vendor'}")
                run.finding("host readiness",
                            f"this interpreter cannot build a {config.vendor} arm; --live will "
                            f"refuse. `python -m src.robot.drivers.doctor` prints the same table")

        with run.step("what the config says this cell is") as report:
            report(f"ip {config.ur.ip}; tool frame from {config.gripper.tool_frame.source}; "
                   f"payload {config.safety.payload.mass_kg} kg")
        limits = config.workspace_limits
        run.note(f"model {config.ur.model}, gripper {config.gripper.vendor} "
                 f"({config.gripper.min_width_mm} to {config.gripper.max_width_mm} mm), "
                 f"planner {config.ur.motion_planner}")
        run.note(f"workspace box in BASE, mm: x {limits.x_min} to {limits.x_max}, "
                 f"y {limits.y_min} to {limits.y_max}, z {limits.z_min} to {limits.z_max}")
        run.note("Those are declarations. 04_planner_or_ik.py is the planner decision, and "
                 "02_which_gripper.py is the end-effector one.")
        run.note("")

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

        # The same gate `from_robot_config` runs before it builds anything, called here for its
        # message: it names the missing SDK modules and what to install, which is a better answer
        # than the ImportError the factory would raise a moment later.
        try:
            readiness.require(config.vendor)
        except RobotConnectionError as error:
            return not_ready(f"this host cannot drive vendor {config.vendor!r}: {error}",
                             "install the vendor extra, or point the config at a vendor this "
                             "host has: WILLY_PROFILE=ursim runs a UR against a container")

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
            # The declared box is only worth something if the arm is standing inside it. An arm
            # parked outside its own workspace limits is a cell that refuses its first motion, and
            # the reason reads as a broken guard rather than as a parked robot.
            inside = all(low <= float(value) <= high for value, low, high in (
                (pose.position_mm[0], limits.x_min, limits.x_max),
                (pose.position_mm[1], limits.y_min, limits.y_max),
                (pose.position_mm[2], limits.z_min, limits.z_max)))
            if not inside:
                run.finding("workspace box",
                            "the arm is standing outside the declared box; the first commanded "
                            "move will be refused as WORKSPACE_REJECTED")

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
        run.note("Next: 02_which_gripper.py, which end-effector this cell builds and what it "
                 "silently stands in for.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
