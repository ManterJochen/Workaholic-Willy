"""Bench exerciser for a gripper wired to the UR controller's digital I/O.

    python -m src.robot.drivers.ur --read                      # read every pin, changes nothing
    python -m src.robot.drivers.ur --watch 0 --for 15          # trip a sensor by hand, see it
    python -m src.robot.drivers.ur --set 4=1 --yes             # drive one output
    python -m src.robot.drivers.ur --pulse 4 --for 0.2 --yes   # the double-solenoid and single-toggle shape
    python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes   # <- the number you came for
    python -m src.robot.drivers.ur --jaws open --yes           # the configured jaw_io hand, through its driver
    python -m src.robot.drivers.ur --where                     # where the arm stands, as lines to paste

Why it exists. ``VacuumGripperConfig`` and ``JawIOGripperConfig`` deliberately put every
wiring number in config, so the I/O end-effectors could be built before the hardware
was chosen. That trade pays off only where the numbers are cheap to measure, and
without a tool the question of which pin closes the jaws is answered by editing a YAML,
running a pick, watching it fail and guessing again, with a powered arm in the room.

``--measure`` is the one that matters: it drives an output, times how long the watched
input takes to answer, and prints the milliseconds that become ``close_settle_s`` and
``engage_timeout_s``. Run it a few times and configure the worst reading, because the
driver waits that budget out and a typical value turns into a dropped part on a slow
stroke.

``--where`` reads the arm's joints and TCP and prints them, the joints as a line to paste
into a program (``JointPositions.deg(...)``, a look pose for instance), and exits 0. It
reads and commands nothing else.

Nothing here moves the arm. There is no path from this CLI to ``arm.move``: it opens
the connection, which the UR driver gates on a declared tool frame and a coherent
payload, and then touches digital I/O alone.

Driving an output is still a physical action. A close pin closes real jaws on whatever
is between them, and an ejector pin starts real suction. Every write is therefore gated
behind ``--yes``, and in a non-interactive shell the gate refuses rather than prompts.
``--read``, ``--watch`` and ``--where`` are read-only and need no gate.

A single toggle's pin may be pulsed here freely: no count of its pulses outlives a
program, and every program asks at its connect where the jaws stand. ``--jaws`` moves
them through the driver, which asks that question first, and its last line says how many
pulses went out, the one a person chose at that question included: it reads the driver's
own count (``JawIOGripper.commands_sent``), never where the jaws stood before and after.

Every refusal says which config it read: the profile chain from ``--profile`` or
``WILLY_PROFILE``, or that neither was given and the base tree was loaded (it names a
Robotiq, not a jaw_io hand), and, once the config loaded, the gripper vendor it names.
No refusal prints a placeholder to fill in: it says ``--profile NAME`` in words.

Exit codes: 0 where the command ran, 1 where config or the connection refused, 2 where
the measurement timed out, meaning the output was driven and the input never answered,
and 3 for an unexpected error.

The honest scope is bucket 3 for the cell and not for the SDK. ``ur_rtde==1.6.5`` is
installed here and every symbol this repo calls was verified present, so the import
path is real. What no line of this has done is speak to a controller: URSim is the next
step and a physical arm the one after. The wiring is exactly what this exists to
measure.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from src.config.loader import ConfigError
from src.contracts import UNSET
from src.robot.constants import UR_IO_CLI_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

_EXIT_OK, _EXIT_REFUSED, _EXIT_TIMEOUT, _EXIT_ERROR = 0, 1, 2, 3

# The bench prints for the operator standing there, and this logs for whoever asks later
# why an output on a real cell was driven. It is deliberately session-level: which
# action was asked for, and every refusal on the way to it. ``io_bench`` logs the pins
# themselves, so nothing is written twice.
logger = create_robot_logger("URIOBenchCLI", UR_IO_CLI_LOG_FILE)


def _load_robot_config(profile: str | None, data_dir: str | None) -> "RobotConfig":
    """Load the config tree, honouring an explicit profile chain, as ``real_cell`` does.

    The argparse ``None`` means nobody typed ``--profile`` and becomes ``UNSET`` rather
    than ``None``, because ``load_robot_config(profile=None)`` means the base tree with
    ``WILLY_PROFILE`` ignored, which would disable that variable silently for an
    operator who exports it.
    """
    from src.config.loader import load_robot_config

    return load_robot_config(data_dir, profile=UNSET if profile is None else profile)


def _profile_chain(profile: str | None, robot_cfg: "RobotConfig | None" = None) -> str:
    """Which config this session read, as a refusal says it: the chain and where it came from, or that there was none.

    A refusal that does not say it sent an operator after the wrong fix: on the owner's PC the bench refused a jaw_io
    command because neither ``--profile`` nor ``WILLY_PROFILE`` was given, the base tree names a Robotiq, and the fix
    it printed carried a literal ``<cell>`` that the shell could not run (2026-09-24). Where the config loaded, the
    gripper it names is said too.
    """
    from src.config.loader import active_profile

    names = "" if robot_cfg is None else f", which names gripper vendor {str(robot_cfg.gripper.vendor)!r}"
    if profile is not None:
        return f"profile chain {profile!r} (from --profile){names}"
    exported = active_profile()
    if exported:
        return f"profile chain {exported!r} (from WILLY_PROFILE){names}"
    return (f"no profile: neither --profile nor WILLY_PROFILE was given, so the base tree was loaded{names}; pass "
            "--profile NAME for your cell")


def _refuse(profile: str | None, what: str, robot_cfg: "RobotConfig | None" = None) -> int:
    """Print and log one refusal with the config it was read from; the refused exit code."""
    chain = _profile_chain(profile, robot_cfg)
    print(f"REFUSED: {what}\n   config: {chain}", flush=True)
    logger.error("refused: %s (config: %s)", what, chain)
    return _EXIT_REFUSED


def _parse_pin_value(spec: str) -> tuple[int, bool]:
    """Turn ``"4=1"`` into ``(4, True)``, rejecting anything ambiguous rather than picking a reading."""
    if "=" not in spec:
        raise SystemExit(f"--set/--measure want PIN=VALUE (e.g. 4=1), got {spec!r}")
    pin_s, val_s = spec.split("=", 1)
    try:
        pin = int(pin_s)
    except ValueError:
        raise SystemExit(f"--set/--measure: {pin_s!r} is not a pin number") from None
    val = val_s.strip().lower()
    if val in ("1", "true", "high", "on"):
        return pin, True
    if val in ("0", "false", "low", "off"):
        return pin, False
    raise SystemExit(f"--set/--measure: {val_s!r} is not a level (use 1/0, high/low, on/off)")


def _confirm(args: argparse.Namespace, what: str) -> bool:
    """Gate a physical write. In a non-interactive shell ``--yes`` is the only way through.

    Refusing rather than prompting where stdin is not a terminal is deliberate: a prompt
    that reads EOF and takes the default is how an unattended script drives a coil
    nobody authorised.
    """
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print(
            f"REFUSED: {what} would drive a real output and --yes was not given (stdin is not a "
            "terminal, so there is nobody to ask). Re-run with --yes if the cell is clear.",
            flush=True,
        )
        logger.warning("refused (no --yes, stdin is not a terminal): %s", what)
        return False
    print(f"\n{what}")
    print("   This drives a real output: jaws close on whatever is between them, an ejector starts.")
    return input("   Is the cell clear? type 'yes' to proceed: ").strip().lower() == "yes"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.robot.drivers.ur",
        description="Bench exerciser for a gripper on the UR controller's digital I/O. Never moves "
                    "the arm; every write is gated behind --yes.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ur3e' or 'ur3e,tiltcam'")
    ap.add_argument("--data-dir", type=str, default=None, help="config data root")
    ap.add_argument("--port", choices=["standard", "configurable", "tool"], default="tool",
                    help="which I/O bank (a tool-mounted gripper is usually on TOOL)")
    ap.add_argument("--read", action="store_true", help="read every pin on the bank and exit")
    ap.add_argument("--watch", type=int, default=None, metavar="PIN",
                    help="poll an input and print every transition (read-only)")
    ap.add_argument("--set", type=str, default=None, metavar="PIN=VALUE",
                    help="drive one output, then read it back")
    ap.add_argument("--pulse", type=int, default=None, metavar="PIN",
                    help="drive an output high for --for seconds, then low")
    ap.add_argument("--measure", type=str, default=None, metavar="PIN=VALUE",
                    help="drive an output and time how long --watch's input takes to answer")
    ap.add_argument("--for", dest="duration", type=float, default=None,
                    help="seconds: --watch timeout (default 10), --pulse length (0.2), "
                         "--measure timeout (2)")
    ap.add_argument("--jaws", choices=["open", "closed"], default=None,
                    help="open or close the configured jaw_io hand through its driver; a single toggle asks first "
                         "where its jaws stand, and pulses only where they stand the other way")
    ap.add_argument("--where", action="store_true",
                    help="print the arm's joints as a JointPositions.deg(...) line to paste, and its TCP in mm and "
                         "degrees, then exit (read-only)")
    ap.add_argument("--yes", action="store_true",
                    help="confirm that the cell is clear; required for any write")
    args = ap.parse_args(argv)

    if args.measure is not None and args.watch is None:
        ap.error("--measure needs --watch PIN: it times how long THAT input takes to answer")

    # One action, and asking for two is refused rather than resolved silently. Five
    # booleans evaluated in a fixed order with `--read` first would make
    # `--read --set 4=1 --yes` read the bank, never write, and exit 0: an operator who
    # typed a write and a read together would get the read, with nothing saying the
    # write had been dropped. A flag accepted and ignored answers a question nobody
    # asked.
    #
    # `--watch` is not counted where `--measure` is present, because there it is an
    # argument of the measurement rather than an action of its own, which is what the
    # check two lines above enforces.
    named = [
        name for name, given in (
            ("--read", bool(args.read)),
            ("--watch", args.watch is not None and args.measure is None),
            ("--set", args.set is not None),
            ("--pulse", args.pulse is not None),
            ("--measure", args.measure is not None),
            ("--jaws", args.jaws is not None),
            ("--where", bool(args.where)),
        ) if given
    ]
    if not named:
        ap.error("nothing to do: pass --read, --watch, --set, --pulse, --measure, --jaws or --where")
    if len(named) > 1:
        ap.error(
            f"{' and '.join(named)} are separate actions and this bench performs one at a time. "
            f"Run them in sequence; a session that both reads and writes cannot say which answer "
            f"belongs to which command."
        )

    from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

    from .bench import Bench, BenchAction, Measure, Pulse, Read, Set, Watch

    port = DigitalIOPort(args.port)

    logger.info(
        "bench session: port=%s profile=%s read=%s watch=%s set=%s pulse=%s measure=%s jaws=%s where=%s",
        port.value, args.profile, args.read, args.watch, args.set, args.pulse, args.measure, args.jaws, args.where,
    )

    try:
        robot_cfg = _load_robot_config(args.profile, args.data_dir)
    # `ConfigError` is in the tuple because the config load raises it for a broken or
    # unvalidatable tree, which is the failure an operator is most likely to hit, and
    # without it that failure reaches the terminal as a traceback rather than as the
    # refusal written here. `SystemExit` stays because other refusals on this path raise
    # it.
    except (SystemExit, ConfigError) as exc:
        return _refuse(args.profile, f"the config did not load: {exc}")

    try:
        from src.robot.drivers import create_arm
        from src.robot.core import RobotVendor

        vendor = RobotVendor.from_string(robot_cfg.vendor)
        if vendor is not RobotVendor.UR:
            return _refuse(args.profile, f"robot.vendor is {vendor.value!r}. Digital I/O is a UR capability here; "
                                         "only the UR driver advertises SupportsDigitalIO.", robot_cfg)
        arm = create_arm(vendor, config=robot_cfg)
    except Exception as exc:  # noqa: BLE001 (a build failure is a refusal, not a crash)
        return _refuse(args.profile, f"could not build the arm: {type(exc).__name__}: {exc}", robot_cfg)

    if not isinstance(arm, SupportsDigitalIO):
        return _refuse(args.profile, "this arm does not advertise SupportsDigitalIO.", robot_cfg)

    try:
        # connect() is where the UR driver fails closed on an undeclared tool frame and
        # an incoherent payload. Those checks are worth passing even for an I/O-only
        # session, because a cell whose TCP is undeclared is not one to commission a
        # gripper on.
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        return _refuse(args.profile, f"the connect was refused: {type(exc).__name__}: {exc}", robot_cfg)

    try:
        if args.where:
            return _where(arm)
        # One action, one verb, and the interlock travels with it. `_confirm` is passed
        # in as the confirmation seam rather than called here, which is what lets the
        # library twin keep the gate: the `io_bench` functions energise a pin the moment
        # they are called and have no gate of their own, so an API that did not carry
        # `confirm` across would delete the only thing standing in front of a coil.
        if args.jaws is not None:
            return _drive_jaws(args, robot_cfg, arm)

        bench = Bench.from_arm(arm, port=port, confirm=lambda what: _confirm(args, what))
        action: BenchAction
        if args.read:
            action = Read()
        elif args.measure is not None:
            pin, value = _parse_pin_value(args.measure)
            action = Measure(
                pin=pin, level=value, watching=int(args.watch),
                for_s=args.duration if args.duration is not None else UNSET,
            )
        elif args.set is not None:
            pin, value = _parse_pin_value(args.set)
            action = Set(pin=pin, level=value)
        elif args.pulse is not None:
            action = Pulse(
                pin=args.pulse,
                for_s=args.duration if args.duration is not None else UNSET,
            )
        else:
            timeout = args.duration if args.duration is not None else 10.0
            print(f"watching input {args.watch} on {port.value} for {timeout:.1f} s, "
                  "trip the sensor by hand now", flush=True)
            action = Watch(pin=int(args.watch), for_s=timeout)

        reading = bench.run(action)
        for line in reading.lines:
            print(f"  {line}" if isinstance(action, (Watch,)) else line, flush=True)
        if reading.advice:
            print(f"\n-> {reading.advice}", flush=True)
        return reading.exit_code
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        return _EXIT_ERROR
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 (teardown must not mask the result)
            pass


def _where(arm: object) -> int:
    """Print where the arm stands: its joints as a ``JointPositions.deg(...)`` line to paste, and its TCP. Read-only."""
    from .bench import where_lines

    joints = arm.get_joint_positions()  # type: ignore[attr-defined]
    tcp = arm.get_tcp_pose()  # type: ignore[attr-defined]
    for line in where_lines(list(joints), tcp):
        print(line, flush=True)
    logger.info("--where: %s", " | ".join(where_lines(list(joints), tcp)))
    return _EXIT_OK


def _drive_jaws(args: argparse.Namespace, robot_cfg: "RobotConfig", arm: object) -> int:
    """Open or close the configured jaw_io hand through its own driver; a single toggle asks where its jaws stand first."""
    from src.robot.core.gripper import OpensAndCloses
    from src.robot.execution.robot_parts import build_gripper

    gripper = build_gripper(robot_cfg, arm=arm)  # type: ignore[arg-type]
    if not isinstance(gripper, OpensAndCloses):
        return _refuse(args.profile, f"--jaws drives a jaw_io hand, and this config builds {type(gripper).__name__}.",
                       robot_cfg)
    if not _confirm(args, f"{args.jaws} the jaws through the {robot_cfg.gripper.jaw_io.actuation} driver"):
        return _EXIT_REFUSED
    from src.robot.core import RobotError

    # What went out is counted by the driver across the connect and the command, not inferred from where the count
    # stood before and after the command: the connect's question may itself have pulsed the jaws open (a person said
    # closed and chose the pulse), and a summary that looked only after the connect then said "nothing was pulsed"
    # about a pulse that went out (review, 2026-09-24).
    toggle = bool(getattr(gripper, "toggles_without_sensor", False))
    start = _commands_sent(gripper)
    at_connect: "int | None" = None
    try:
        gripper.connect()  # type: ignore[attr-defined]
        try:
            at_connect = _commands_sent(gripper)
            before = bool(getattr(gripper, "jaws_closed", False))
            gripper.set_closed(args.jaws == "closed")
            after = bool(getattr(gripper, "jaws_closed", False))
        finally:
            gripper.disconnect()  # type: ignore[attr-defined]
    except RobotError as exc:
        # The driver's own refusal, a toggle nobody could ask about above all, is the answer here, not a traceback.
        # What went out before it is said too: a refusal after a pulse is a hand that moved.
        sent = _sent_before_a_refusal(start, at_connect, _commands_sent(gripper), toggle=toggle)
        return _refuse(args.profile, f"--jaws {args.jaws}: {exc}{sent}", robot_cfg)
    print(_jaws_summary(args.jaws, before, after, start, at_connect, _commands_sent(gripper), toggle=toggle),
          flush=True)
    return _EXIT_OK


def _commands_sent(gripper: object) -> "int | None":
    """The driver's own count of the commands it sent the jaws (``JawIOGripper.commands_sent``); ``None`` without one."""
    sent = getattr(gripper, "commands_sent", None)
    return sent if isinstance(sent, int) and not isinstance(sent, bool) else None


def _counted(n: int, *, toggle: bool) -> str:
    """``1 pulse``, ``2 pulses``, ``nothing``; ``command`` in place of ``pulse`` for a hand that is not a toggle."""
    if n == 0:
        return "nothing"
    noun = "pulse" if toggle else "command"
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _jaws_summary(jaws: str, before: bool, after: bool, start: "int | None", at_connect: "int | None",
                  end: "int | None", *, toggle: bool) -> str:
    """The line ``--jaws`` ends on: where the command took the jaws, and what went out at the connect and for it.

    "Nothing was pulsed" is said only where the driver's count says nothing went out at all, the connect included. A
    driver that keeps no count is not credited with sending nothing: the line then says where the jaws went and no
    more.
    """
    line = f"the driver took the jaws from {'CLOSED' if before else 'OPEN'} to {'CLOSED' if after else 'OPEN'}"
    if start is None or at_connect is None or end is None:
        return line
    connect, command = at_connect - start, end - at_connect
    if connect + command == 0:
        return f"{line} (they already stood there: nothing was {'pulsed' if toggle else 'sent'})"
    parts = [f"{_counted(connect, toggle=toggle)} at connect, to open them"] if connect else []
    parts.append(f"{_counted(command, toggle=toggle)} for --jaws {jaws}"
                 + ("" if command else " (they already stood there after the connect)"))
    return f"{line}; {_counted(connect + command, toggle=toggle)} went out: {', and '.join(parts)}"


def _sent_before_a_refusal(start: "int | None", at_connect: "int | None", end: "int | None", *, toggle: bool) -> str:
    """What a refusal adds about what went out before it: ``""`` where nothing did, or the driver keeps no count."""
    if start is None or end is None or end == start:
        return ""
    # A connect that raised never set `at_connect`: then everything counted went out at the connect.
    where = " at connect" if at_connect is None else (
        f" ({_counted(at_connect - start, toggle=toggle)} at connect, {_counted(end - at_connect, toggle=toggle)} "
        "for the command)")
    return (f". Before this refusal the driver sent {_counted(end - start, toggle=toggle)}{where}, a write that raised "
            "counted, because it may still have reached the controller: look at the jaws before anything else")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
