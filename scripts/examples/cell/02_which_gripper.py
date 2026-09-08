"""02: which end-effector this cell builds, what each one needs wired, and what stands in for it.

    python scripts/examples/cell/02_which_gripper.py           # build all six, command nothing
    python scripts/examples/cell/02_which_gripper.py --live     # connect the one the config names

The decision is `robot.gripper.vendor`. Six drivers ship and none of them needs a pip package, so
what separates them is what has to be wired:

    robotiq   a socket on port 63352, opened by the URCap on the UR controller, at `robot.ur.ip`
    onrobot   Modbus TCP to a Compute Box at `gripper.onrobot.host`, its own device on the network
    vacuum    an ejector on a controller output pin, `gripper.vacuum.*`, optionally a vacuum switch
    jaw_io    a solenoid jaw on controller pins, `gripper.jaw_io.*`, optionally reed switches
    dummy     pure Python, a width in memory
    none      the explicit no-op, for a cell that has no end-effector on purpose

Only the first two speak a protocol. The next two are pins, so bringing one up is measuring which
pin does what rather than writing anything, which is why every wiring number lives in config and
`python -m src.robot.drivers.ur --read` exists to measure them without moving the arm.

The one fact a reader must not miss is the substitution. Four config combinations cannot produce
the end-effector they name, and none of them is an error: the builder logs a line, hands back a
`NullGripper` and the cell comes up. A `NullGripper` accepts every command, holds nothing, and
reports the configured opening whatever it was told to do, so every pick reports SUCCEEDED while
the jaws close on nothing. `GripperSubstitution` puts the reason on the object, and this example
reads it off there rather than off a log.

Nothing here moves anything without `--live`, and that is not caution for its own sake: a Robotiq
`connect()` activates, and activation is a calibration sweep of the full finger travel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, hardware_parser, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  every vendor built and reported what it is
  1  a vendor raised where it should have refused, or (with --live) the gripper did not answer
  2  nothing to build: no `robot` block in the config tree
"""

#: The six drivers, in the order the docstring introduces them, and where each one is reached.
_VENDORS: tuple[tuple[str, str], ...] = (
    ("robotiq", "URCap socket on the UR controller, port 63352"),
    ("onrobot", "Modbus TCP to a Compute Box on its own address"),
    ("vacuum", "an ejector on the controller's digital output"),
    ("jaw_io", "a solenoid jaw on the controller's digital I/O"),
    ("dummy", "pure Python, no transport at all"),
    ("none", "no end-effector, on purpose"),
)


def _build_gripper(robot_config: Any, vendor: str, arm: Any, arm_vendor: str) -> Any:
    """Ask the real builder for `vendor` on `arm`, and hand back what it produced.

    This is the config-driven build path, the one `Cell.build()` reaches, called one layer down
    from `Cell` for one reason: `Cell` builds the arm from config too, and the question here is
    which gripper comes out for a given arm, which needs the arm to be an argument. Perception and
    the calculator come from the rehearsal components so no camera is opened and no model is
    loaded; neither is consulted by the gripper branch.

    `arm_vendor` is set on the config as well as supplied as a handle, and the two must agree.
    The branches do not all read the same thing: the Robotiq branch reads `robot.vendor` from the
    config, because a Robotiq lives on a UR controller and that is a statement about the cell, and
    the vacuum and jaw branches ask the arm object whether it advertises `SupportsDigitalIO`. A
    config saying `ur` with a dummy handle would therefore build a real Robotiq driver against an
    arm that has no controller behind it.

    `model_copy` does not validate, so the swapped block is re-validated. That is not ceremony: it
    is where an unknown vendor is refused, and it keeps this function honest about which refusals
    belong to the schema and which belong to the builder.
    """
    from src.config.schema.robot import GripperConfig
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

    block = GripperConfig.model_validate(
        robot_config.gripper.model_copy(update={"vendor": vendor}).model_dump())
    config = robot_config.model_copy(update={"gripper": block, "vendor": arm_vendor})
    calculator, perception, resolver, _cameras, _lenses = build_rehearsal_components(config)
    service = AutonomousGraspService.from_robot_config(
        config, calculator=calculator, perception=perception, frame_resolver=resolver, arm=arm)
    return service.runtime.orchestrator.gripper


def main(argv: list[str] | None = None) -> int:
    args = hardware_parser(__doc__ or "", epilog=EPILOG).parse_args(argv)

    from src.config.loader import load_config
    from src.config.schema.robot import GripperConfig
    from src.robot.core import RobotVendor, SupportsDigitalIO
    from src.robot.drivers import create_arm
    from src.robot.grippers import available_gripper_vendors
    from src.robot.grippers.null import NullGripper

    with Example("02 which_gripper", "six drivers, one config key",
                 live=args.live, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            robot = load_config().robot
            if robot is None:
                report("no `robot` block in this tree")
            else:
                report(f"gripper.vendor {robot.gripper.vendor}, arm vendor {robot.vendor}")
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "point the loader at a tree that has one; `config/robot/robot.yaml` "
                             "in this repository is one")

        with run.step("which drivers are registered here") as report:
            # The registry, not a list in this file. A vendor missing from it is a vendor with no
            # driver, and the registry is the only thing that knows.
            report(", ".join(available_gripper_vendors()))
        run.note("None of them needs a pip package. The Robotiq speaks the port-63352 grammar")
        run.note("directly, the OnRobot client speaks Modbus with `socket` and `struct`, and the")
        run.note("two I/O drivers switch pins on the arm. So `pip install` fixes nothing here:")
        run.note("what is missing on a bring-up is a URCap, a Compute Box or a wire.")
        run.note("")

        with run.step("what this config wires for its vendor") as report:
            report(_wiring_line(robot))

        # Two arms, because the builder's answer depends on the arm as much as on the vendor.
        # Nothing is connected: constructing a driver opens no socket, and the branch below reads
        # a capability off the object rather than off the cell.
        arms: list[tuple[str, Any]] = [("dummy", create_arm(RobotVendor.DUMMY))]
        try:
            arms.insert(0, ("ur", create_arm(RobotVendor.UR, config=robot)))
        except Exception as error:                                   # noqa: BLE001  (report, not raise)
            run.finding("ur arm", f"not constructible on this host ({type(error).__name__}); the "
                                  f"table below has the dummy column only")

        run.note("")
        run.note("What the builder produces, per arm. `SupportsDigitalIO` is the capability the")
        run.note("vacuum and jaw branches test for, and only the real UR driver advertises it.")
        for arm_name, arm in arms:
            io = "digital I/O yes" if isinstance(arm, SupportsDigitalIO) else "digital I/O no"
            run.note("")
            run.note(f"  arm vendor {arm_name} ({type(arm).__name__}, {io})")
            for vendor, needs in _VENDORS:
                # Filled by the step body and read after it. A `note` printed inside a step lands
                # on the step's own open line, so anything a step wants to say in full is said
                # once the verdict is out.
                aside: list[str] = []
                with run.step(f"{arm_name} arm, gripper.vendor {vendor}") as report:
                    try:
                        gripper = _build_gripper(robot, vendor, arm, arm_name)
                    except Exception as error:                       # noqa: BLE001  (report, not raise)
                        # A build that refuses is an answer too, and the likely one is not about
                        # the gripper: `build_calculator` refuses a `deep` cell with no artifact,
                        # and `from_robot_config` refuses a tree that declares no grasping block.
                        report(f"the builder refused: {type(error).__name__}: "
                               f"{str(error).splitlines()[0][:60]}")
                        continue
                    substitution = getattr(gripper, "substitution", None)
                    if substitution is None:
                        report(f"{type(gripper).__name__}: {needs}")
                    else:
                        # The reason travels on the object. A caller that had to grep a log for
                        # this would find it after the cell had already reported ten successes.
                        report(f"SUBSTITUTED {type(gripper).__name__}, "
                               f"reason {substitution.reason.value}")
                        aside = [f"     {substitution.detail}", f"     fix: {substitution.fix}"]
                for line in aside:
                    run.note(line)

        run.note("")
        reserved_note: list[str] = []
        with run.step("a name in the enum with no driver behind it") as report:
            # `GripperVendor` carries two names this repository has no driver for. They pass the
            # schema, because they are real members, and substitute at build. A reader whose
            # gripper is one of them needs to see that it is a reserved slot rather than an
            # install away, and the readiness table in 01 says the same thing.
            reserved = _build_gripper(robot, "schunk", arms[0][1], arms[0][0])
            substitution = getattr(reserved, "substitution", None)
            report(f"schunk: {type(reserved).__name__}"
                   + (f", reason {substitution.reason.value}" if substitution else ""))
            if substitution is not None:
                reserved_note = [f"     {substitution.detail}", f"     fix: {substitution.fix}"]
        for line in reserved_note:
            run.note(line)

        run.note("")
        with run.step("what a substituted gripper then does") as report:
            # Built here rather than reused from the table above so the numbers below are this
            # cell's own configured opening. Nothing is commanded: a NullGripper reaches no
            # hardware by construction, which is exactly the problem being demonstrated.
            stand_in = NullGripper(min_width_mm=robot.gripper.min_width_mm,
                                   max_width_mm=robot.gripper.max_width_mm)
            stand_in.connect()
            stand_in.set_width_mm(robot.gripper.min_width_mm)
            report(f"commanded {robot.gripper.min_width_mm} mm, reports "
                   f"{stand_in.get_width_mm()} mm, raised nothing")
        run.note("It reports the configured opening whatever it was commanded, so a closed jaw and")
        run.note("an empty one read alike. `grasping.verification` ships disabled, so nothing looks")
        run.note("again either: the attempt is judged by the motion and reports SUCCEEDED.")
        run.note("03_first_pick.py prints that outcome on the shipped tree.")

        with run.step("an unknown vendor never reaches the builder") as report:
            # The fifth substitution reason, `unknown_vendor`, is unreachable through a loaded
            # config: the schema validates `gripper.vendor` against the enum at load. It exists for
            # a config assembled in code, where `model_copy` skips validation.
            try:
                GripperConfig(vendor="acme")
            except Exception as error:                               # noqa: BLE001  (the point)
                lines = str(error).splitlines()
                detail = next((ln.strip() for ln in lines if "unknown gripper vendor" in ln), "")
                report(f"{type(error).__name__} at config load: "
                       f"{(detail or str(error)).split('[type=')[0].strip()}")
            else:
                run.finding("schema", "GripperConfig accepted vendor='acme'; the enum check that "
                                      "should refuse it at load did not fire")
                report("accepted, which it should not have been")

        if not args.live:
            run.note("")
            run.note("Rehearsal stops here. `--live` connects the end-effector this config names,")
            run.note("which for a Robotiq is a full-travel activation sweep, and for a vacuum or a")
            run.note("jaw is the arm connect first, because those two switch the controller's I/O.")
            return run.exit_code

        # The live half: the gripper this config actually names, on the arm this config actually
        # names, connected in the documented order.
        arm = create_arm(RobotVendor.from_string(robot.vendor), config=robot)
        gripper = _build_gripper(robot, str(robot.gripper.vendor), arm, str(robot.vendor))
        substitution = getattr(gripper, "substitution", None)
        if substitution is not None:
            return not_ready(
                f"this config builds no real end-effector: {substitution.detail}",
                substitution.fix)
        if isinstance(gripper, NullGripper):
            # No substitution and a NullGripper means `vendor: none`, a legitimate cell, so this is
            # not a refusal. It still stops here: a NullGripper connects, activates and reports the
            # configured opening without touching anything, so going on would print three
            # successes that mean nothing, which is the failure mode this whole file is about.
            with run.step("connect the end-effector this config names") as report:
                report("gripper.vendor is 'none': there is nothing to connect")
            run.note("That is a cell with no end-effector, not a broken one. Set gripper.vendor to")
            run.note("what is on the flange to give this half of the example something to do.")
            return run.exit_code

        # The two I/O drivers, by type rather than by vendor string, because what decides the
        # order is what the object does on connect and not what the config called it.
        from src.robot.grippers.jaw_io import JawIOGripper
        from src.robot.grippers.vacuum import VacuumGripper

        needs_arm = isinstance(gripper, (VacuumGripper, JawIOGripper))
        try:
            if needs_arm:
                with run.step("connect the arm first") as report:
                    # Documented lifecycle, not a preference: an I/O gripper drives a pin as soon
                    # as it connects, and there is no pin to drive until the arm holds the
                    # controller connection.
                    try:
                        arm.connect()
                    except Exception as error:                       # noqa: BLE001  (report, not raise)
                        return not_ready(
                            f"the controller did not answer ({type(error).__name__}: {error})",
                            "an I/O gripper switches pins on the arm, so the arm has to be "
                            "connected first; 01_robot_setup.py checks that half alone")
                    report(f"{type(arm).__name__} connected; its digital I/O is what the gripper "
                           f"switches")
            with run.step("connect the gripper") as report:
                try:
                    gripper.connect()
                except Exception as error:                           # noqa: BLE001  (report, not raise)
                    return not_ready(
                        f"the end-effector did not answer ({type(error).__name__}: {error})",
                        _live_fix(str(robot.gripper.vendor), robot))
                report(f"{type(gripper).__name__} connected and activated")
            with run.step("read the opening back") as report:
                report(f"{gripper.get_width_mm():.1f} mm "
                       f"(configured range {gripper.min_width_mm} to {gripper.max_width_mm} mm)")
        finally:
            gripper.disconnect()
            if needs_arm:
                arm.disconnect()
            run.note("disconnected, gripper first.")

        run.note("")
        run.note("Next: 03_first_pick.py, the same gripper inside one whole grasp.")
        return run.exit_code


def _wiring_line(robot: Any) -> str:
    """The block this cell's vendor actually reads, so a reader sees their own numbers.

    Every vendor has its own block and the others stay valid and unused, which is how a cell is
    commissioned: declare the jaw and the cup, then swap one string.
    """
    vendor = str(robot.gripper.vendor)
    if vendor == "robotiq":
        return f"robotiq on {robot.ur.ip}:63352, opened by the URCap"
    if vendor == "onrobot":
        rg = robot.gripper.onrobot
        return f"onrobot on {rg.host}:{rg.port}, unit {rg.unit_id}, {rg.default_force_n} N"
    if vendor == "vacuum":
        vac = robot.gripper.vacuum
        return (f"vacuum on output pin {vac.vacuum_output_pin}, port {vac.io_port}, "
                f"switch on input {vac.vacuum_ok_input_pin}")
    if vendor == "jaw_io":
        jaw = robot.gripper.jaw_io
        return (f"jaw_io {jaw.actuation}, close pin {jaw.close_output_pin}, "
                f"open pin {jaw.open_output_pin}, port {jaw.io_port}")
    return f"{vendor}: no wiring to declare"


def _live_fix(vendor: str, robot: Any) -> str:
    """What to check when the end-effector this config names did not answer."""
    if vendor == "robotiq":
        return (f"is the Robotiq URCap installed and running on {robot.ur.ip}? It opens port "
                f"63352; no pip package does. Nothing answers there without it")
    if vendor == "onrobot":
        return (f"is the Compute Box at {robot.gripper.onrobot.host} powered and on this network, "
                f"and is unit {robot.gripper.onrobot.unit_id} the one the tool is mounted on?")
    if vendor in ("vacuum", "jaw_io"):
        return ("is the solenoid's 24 V supply on, and are the pins in `gripper."
                f"{vendor}` the ones it is wired to? `python -m src.robot.drivers.ur --read` "
                "prints the bank without moving anything")
    return "check the transport this vendor needs; the table above names it"


if __name__ == "__main__":
    raise SystemExit(main())
