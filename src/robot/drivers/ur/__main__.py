"""Bench exerciser for a gripper wired to the UR controller's digital I/O.

    python -m src.robot.drivers.ur --read                      # read every pin, changes nothing
    python -m src.robot.drivers.ur --watch 0 --for 15          # trip a sensor by hand, see it
    python -m src.robot.drivers.ur --set 4=1 --yes             # drive one output
    python -m src.robot.drivers.ur --pulse 4 --for 0.2 --yes   # the double-solenoid and single-toggle shape
    python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes   # <- the number you came for
    python -m src.robot.drivers.ur --jaws open --yes           # a single-toggle hand, through its driver and count
    python -m src.robot.drivers.ur --jaws-stand open --yes     # you looked: where a toggle's jaws stand

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

Nothing here moves the arm. There is no path from this CLI to ``arm.move``: it opens
the connection, which the UR driver gates on a declared tool frame and a coherent
payload, and then touches digital I/O alone.

Driving an output is still a physical action. A close pin closes real jaws on whatever
is between them, and an ejector pin starts real suction. Every write is therefore gated
behind ``--yes``, and in a non-interactive shell the gate refuses rather than prompts.
``--read`` and ``--watch`` are read-only and need no gate.

A single toggle with no open switch keeps a count of its own pulses on disk, and a write on
its pin (``--pulse``, ``--set``, ``--measure``) moves jaws that count never sees. So once the
write is confirmed, and before the edge, the record is marked, and the next program refuses
to pulse until ``--jaws-stand`` says where a person saw the jaws stand. ``--jaws`` moves them
through the driver and its count instead.

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
from pathlib import Path
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
                    help="open or close the configured jaw_io hand through its driver, which pulses a toggle only "
                         "when its count says the jaws stand the other way, and records the pulse")
    ap.add_argument("--jaws-stand", dest="jaws_stand", choices=["open", "closed"], default=None,
                    help="a single toggle with no open switch: you looked at the jaws, and this is where they stand. "
                         "Nothing is pulsed and nothing connects; the driver's count is set to it")
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
            ("--jaws-stand", args.jaws_stand is not None),
        ) if given
    ]
    if not named:
        ap.error("nothing to do: pass --read, --watch, --set, --pulse, --measure, --jaws or --jaws-stand")
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
        "bench session: port=%s profile=%s read=%s watch=%s set=%s pulse=%s measure=%s",
        port.value, args.profile, args.read, args.watch, args.set, args.pulse, args.measure,
    )

    try:
        robot_cfg = _load_robot_config(args.profile, args.data_dir)
    # `ConfigError` is in the tuple because the config load raises it for a broken or
    # unvalidatable tree, which is the failure an operator is most likely to hit, and
    # without it that failure reaches the terminal as a traceback rather than as the
    # refusal written here. `SystemExit` stays because other refusals on this path raise
    # it.
    except (SystemExit, ConfigError) as exc:
        print(f"config refused: {exc}", flush=True)
        logger.error("config refused: %s", exc)
        return _EXIT_REFUSED

    toggle = _toggle_record(robot_cfg)
    if args.jaws_stand is not None:
        return _declare_jaws(args, toggle)

    try:
        from src.robot.drivers import create_arm
        from src.robot.core import RobotVendor

        vendor = RobotVendor.from_string(robot_cfg.vendor)
        if vendor is not RobotVendor.UR:
            print(f"REFUSED: robot.vendor is {vendor.value!r}. Digital I/O is a UR capability here; "
                  "only the UR driver advertises SupportsDigitalIO.", flush=True)
            logger.error("refused: robot.vendor is %r, not 'ur'", vendor.value)
            return _EXIT_REFUSED
        arm = create_arm(vendor, config=robot_cfg)
    except Exception as exc:  # noqa: BLE001 (a build failure is a refusal, not a crash)
        print(f"could not build the arm: {type(exc).__name__}: {exc}", flush=True)
        logger.error("could not build the arm: %s: %s", type(exc).__name__, exc)
        return _EXIT_REFUSED

    if not isinstance(arm, SupportsDigitalIO):
        print("REFUSED: this arm does not advertise SupportsDigitalIO.", flush=True)
        logger.error("refused: %s does not advertise SupportsDigitalIO", type(arm).__name__)
        return _EXIT_REFUSED

    try:
        # connect() is where the UR driver fails closed on an undeclared tool frame and
        # an incoherent payload. Those checks are worth passing even for an I/O-only
        # session, because a cell whose TCP is undeclared is not one to commission a
        # gripper on.
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"connect refused: {type(exc).__name__}: {exc}", flush=True)
        logger.error("connect refused: %s: %s", type(exc).__name__, exc)
        return _EXIT_REFUSED

    uncounted = False
    try:
        # One action, one verb, and the interlock travels with it. `_confirm` is passed
        # in as the confirmation seam rather than called here, which is what lets the
        # library twin keep the gate: the `io_bench` functions energise a pin the moment
        # they are called and have no gate of their own, so an API that did not carry
        # `confirm` across would delete the only thing standing in front of a coil.
        if args.jaws is not None:
            return _drive_jaws(args, robot_cfg, arm)

        def confirm(what: str) -> bool:
            # A write on a toggle's own pin marks its record once confirmed and before the
            # edge, so a pulse cut short by Ctrl-C, or a write that raises after the jaws
            # flipped, leaves the record demanding a declaration as a finished one does.
            nonlocal uncounted
            if not _confirm(args, what):
                return False
            uncounted = _uncount_a_toggle_pulse(args, port, toggle)
            return True

        bench = Bench.from_arm(arm, port=port, confirm=confirm)
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
        if uncounted and toggle is not None:
            print(f"\n-> this drove the toggle's own pin, which the driver does not count. Look at the jaws and say "
                  f"where they stand: python -m src.robot.drivers.ur --profile <cell> --port {toggle[1]} --jaws-stand "
                  f"open (or closed) --yes. Until then the driver refuses to pulse.", flush=True)
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 (teardown must not mask the result)
            pass


def _toggle_record(robot_cfg: "RobotConfig") -> "tuple[Path, str, int] | None":
    """The record, bank and pin of the configured hand when it is a single toggle with no open switch, else None.

    Read through ``robot_parts``: no module under ``drivers/ur`` imports the grippers package.
    """
    from src.robot.execution.robot_parts import jaw_toggle_record

    return jaw_toggle_record(robot_cfg)


def _declare_jaws(args: argparse.Namespace, toggle: "tuple[Path, str, int] | None") -> int:
    """Set a toggle's count to where a person saw the jaws stand. Nothing is pulsed and nothing connects."""
    from src.robot.execution.robot_parts import declare_jaw_toggle_record, describe_jaw_toggle_record

    if toggle is None:
        print("REFUSED: --jaws-stand is for a jaw_io single_toggle with no open switch, the one hand that keeps a "
              "count of its own pulses; this profile's gripper is not one.", flush=True)
        return _EXIT_REFUSED
    path, port, pin = toggle
    closed = args.jaws_stand == "closed"
    what = f"declare that the jaws on {port} pin {pin} stand {args.jaws_stand.upper()}, having looked at them"
    if not args.yes and not (sys.stdin.isatty() and input(f"{what}? type 'yes': ").strip().lower() == "yes"):
        print(f"REFUSED: {what} needs --yes, or 'yes' typed at a terminal.", flush=True)
        return _EXIT_REFUSED
    said = describe_jaw_toggle_record(path)
    declare_jaw_toggle_record(path, closed=closed)
    logger.warning("jaws declared %s on %s pin %s by the bench (the record said: %s)", args.jaws_stand, port, pin, said)
    print(f"recorded: the jaws stand {args.jaws_stand.upper()} ({path}); the record said {said}", flush=True)
    return _EXIT_OK


def _drive_jaws(args: argparse.Namespace, robot_cfg: "RobotConfig", arm: object) -> int:
    """Open or close the configured jaw_io hand through its own driver, so a toggle's count follows the pulse."""
    from src.robot.core.gripper import OpensAndCloses
    from src.robot.execution.robot_parts import build_gripper

    gripper = build_gripper(robot_cfg, arm=arm)  # type: ignore[arg-type]
    if not isinstance(gripper, OpensAndCloses):
        print(f"REFUSED: --jaws drives a jaw_io hand; this profile builds {type(gripper).__name__}.", flush=True)
        return _EXIT_REFUSED
    if not _confirm(args, f"{args.jaws} the jaws through the {robot_cfg.gripper.jaw_io.actuation} driver"):
        return _EXIT_REFUSED
    from src.robot.core import RobotError

    try:
        gripper.connect()  # type: ignore[attr-defined]
        try:
            before = bool(getattr(gripper, "jaws_closed", False))
            gripper.set_closed(args.jaws == "closed")
            after = bool(getattr(gripper, "jaws_closed", False))
        finally:
            gripper.disconnect()  # type: ignore[attr-defined]
    except RobotError as exc:
        # The driver's own refusal, a count nobody can vouch for above all, is the answer here, not a traceback.
        print(f"REFUSED: {exc}", flush=True)
        logger.error("--jaws %s refused: %s", args.jaws, exc)
        return _EXIT_REFUSED
    moved = "" if before != after else " (they already stood there: nothing was pulsed)"
    print(f"the driver took the jaws from {'CLOSED' if before else 'OPEN'} to {'CLOSED' if after else 'OPEN'}{moved}",
          flush=True)
    return _EXIT_OK


def _uncount_a_toggle_pulse(args: argparse.Namespace, port: object,
                            toggle: "tuple[Path, str, int] | None") -> bool:
    """A --pulse, --set or --measure on a toggle's own pin moves jaws the driver does not count: its record says so.

    Called once the write is confirmed and before the pin is driven, and it answers whether it marked the record.
    Marked after the action instead, the record missed a --measure, which drives the pin as surely as a pulse, and a
    pulse that was interrupted or raised, and the next program ran inverted with nothing refusing.

    Flipping the count instead would keep a wrong count wrong, and a bench pulse is usually given because the count
    was wrong. So the next program refuses to pulse until a person has said where the jaws stand.
    """
    if toggle is None:
        return False
    path, toggle_port, pin = toggle
    spec = args.set if args.set is not None else args.measure
    driven = args.pulse if args.pulse is not None else (_parse_pin_value(spec)[0] if spec is not None else None)
    if driven is None or int(driven) != pin or getattr(port, "value", port) != toggle_port:
        return False
    from src.robot.execution.robot_parts import mark_jaw_toggle_record_uncounted

    mark_jaw_toggle_record_uncounted(path, by="a bench write on the toggle's pin, which the driver does not count")
    logger.warning("the bench drives the toggle's own %s pin %s: its record now demands a declaration (%s)",
                   toggle_port, pin, path)
    return True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
