"""Drive the two I/O grippers against real controller software.

`probe_our_driver.py` proves the arm driver talks to a real controller, and it feature-checks
`isinstance(arm, SupportsDigitalIO)`. That is where it stops: no other probe in this directory
constructs a gripper. `JawIOGripper` (a jaw gripper over digital I/O) and `VacuumGripper` otherwise
run only against fakes, so every one of their state machines is unmeasured: the `connect()` sensor
gate, the single- against double-solenoid pulse, the read-back, and the `is_object_detected` poll.

    Run (URSim up, in REMOTE control):  python scripts/ursim/probe_grippers.py --yes

What this does not validate, and saying so is the point of the script rather than a footnote.
URSim runs the real controller software and a simulated robot: the I/O registers exist and read
back, but nothing is wired to them. So this probe validates the call path and the state machine,
that a close lands a write on the pin the config names, that the read-back is what the driver
believes, that the timeouts fire where they should. It validates none of:

  * the pin assignment, meaning which physical output the valve is on,
  * whether the reed switches are active-high or active-low,
  * cylinder travel time, and therefore whether `close_timeout_s` is right,
  * that a closed jaw is holding anything at all.

Those four need a bench with the gripper bolted on. A green run here means the software is right
about itself, never that the cell is right.

The grippers are hand-built rather than taken from `create_gripper`, because the `ursim` profile
sets `gripper.vendor: none` and the config-driven factory would return a no-op gripper: the probe
would exercise nothing and report success.

Every write is gated behind `--yes`. The gripper drivers carry no confirmation seam of their own,
unlike `Bench`, and `probe_robotiq_urcap.py` in this same directory accepts a real controller
address, so an ungated `set_width_mm(0.0)` here is one command line away from closing real jaws.

Exit codes: 0 every step OK, 1 a step failed, 2 the request itself was wrong (no controller, or an
arm with no digital I/O). They are this script's own codes and are unrelated to
`BenchReading.exit_code`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root. `scripts/examples/_common.py` does the same insert for the examples.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config.loader import active_profile, load_robot_config  # noqa: E402
from src.robot.core import RobotVendor, SupportsDigitalIO  # noqa: E402
from src.robot.core.arm_capabilities import DigitalIOPort  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402

#: The chain this probe needs when the operator has not named one. It is resolved here rather than
#: written into `WILLY_PROFILE`, which would leak process-global state into whatever runs next. It
#: is not left to the loader's own default either: the base tree points `ur.ip` at a plant address,
#: and a probe that energises solenoids must never fall back onto a real cell.
_DEFAULT_PROFILE = "ursim"

_OK, _FAILED, _BAD_REQUEST = 0, 1, 2

results: list[tuple[str, str, bool]] = []


def note(step: str, outcome: str, ok: bool = True) -> None:
    results.append((step, outcome, ok))
    print(f"  {'OK ' if ok else 'ERR'} {step:<44} {outcome}", flush=True)


def _confirm(yes: bool, what: str) -> bool:
    """Gate a physical write. In a non-interactive shell `--yes` is the only way through.

    Refusing rather than prompting where stdin is not a terminal is deliberate: a prompt that reads
    EOF and takes the default is how an unattended script drives a coil nobody authorised. It is
    the rule `python -m src.robot.drivers.ur` follows for the same writes.
    """
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"  REFUSED: {what} would energise a real output and --yes was not given.",
              flush=True)
        return False
    print(f"\n  {what}")
    print("  This energises a real output: jaws close on whatever is between them, an ejector "
          "starts.")
    return input("  Is the cell clear? type 'yes' to proceed: ").strip().lower() == "yes"


def _read(io: SupportsDigitalIO, pin: int) -> bool:
    """Read an output back on the bank both grippers default to.

    The bank is part of the address. Both grippers default to `DigitalIOPort.TOOL`, and reading pin
    0 on STANDARD would report a different pin entirely: a probe that reads the wrong bank reports
    "the write did not land" about a write that landed perfectly. `probe_our_driver.py` reads
    STANDARD for the same reason in reverse, because there it is the controller's own outputs that
    are under test rather than a gripper's.
    """
    return bool(io.get_digital_output(pin, port=DigitalIOPort.TOOL))


def _probe_jaw(arm: SupportsDigitalIO) -> None:
    """The jaw gripper, which has a test file of its own and has never talked to a controller."""
    from src.robot.grippers.jaw_io import JawIOGripper

    gripper = JawIOGripper(
        arm,
        actuation="single_solenoid",
        close_output_pin=0,
        # No sensor pins. URSim has no reed switches wired, and naming one would make `connect()`
        # gate on an input that reads a constant, which would either pass for the wrong reason or
        # fail for one that has nothing to do with the driver. The sensorless path is the one this
        # probe can honestly grade.
        part_present_input_pin=None,
        closed_confirm_input_pin=None,
        open_confirm_input_pin=None,
        close_timeout_s=1.0,
        # Shorter than the driver's own default settle, because nothing is wired here and the
        # settle only costs run time. A cell with a real cylinder uses the driver's value.
        close_settle_s=0.1,
    )
    note("JawIOGripper(...)", f"{type(gripper).__name__}, single_solenoid on output 0")

    gripper.connect()
    note("jaw.connect()", f"borrowed the arm's I/O, feedback={gripper.has_feedback}")

    # The gripper Protocol is width-based. There is no close() or open(): every gripper in this
    # repository, jaw or suction, is commanded through `set_width_mm`, and the driver decides what
    # a width means for its hardware, here a close at or below `closed_below_mm`.
    before = _read(arm, 0)
    gripper.set_width_mm(0.0)
    note("jaw.set_width_mm(0.0) [close]",
         f"pin 0 read back {_read(arm, 0)} (was {before}), state={gripper.read_jaw_state().value}")
    note("jaw.get_width_mm()", f"{gripper.get_width_mm():.1f} mm")

    note("jaw.is_object_detected()",
         f"{gripper.is_object_detected()}: without a sensor this is the driver's belief and not a "
         f"measurement, which is exactly the distinction a bench has to settle")

    gripper.set_width_mm(gripper.max_width_mm)
    note("jaw.set_width_mm(max) [open]",
         f"pin 0 read back {_read(arm, 0)}, state={gripper.read_jaw_state().value}")

    gripper.disconnect()
    note("jaw.disconnect()", "clean, the arm's connection is untouched")


def _probe_vacuum(arm: SupportsDigitalIO) -> None:
    """The suction side, same seam, same controller."""
    from src.robot.grippers.vacuum import VacuumGripper

    gripper = VacuumGripper(arm, vacuum_output_pin=1, vacuum_ok_input_pin=None)
    note("VacuumGripper(...)", f"{type(gripper).__name__}, vacuum output 1")

    gripper.connect()
    note("vacuum.connect()", "connected")

    gripper.set_width_mm(0.0)
    note("vacuum.set_width_mm(0.0) [engage]",
         f"pin 1 read back {_read(arm, 1)}, vacuum_on={gripper.vacuum_on}")
    note("vacuum.is_object_detected()", f"{gripper.is_object_detected()}, no vacuum switch wired")
    time.sleep(0.2)
    gripper.set_width_mm(gripper.max_width_mm)
    note("vacuum.set_width_mm(max) [release]",
         f"pin 1 read back {_read(arm, 1)}, vacuum_on={gripper.vacuum_on}")

    gripper.disconnect()
    note("vacuum.disconnect()", "clean")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python scripts/ursim/probe_grippers.py",
        description="Walk the JawIOGripper and VacuumGripper state machines against real "
                    "controller software. Every write is gated behind --yes.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ursim' or 'ursim,ursim_ur3'")
    ap.add_argument("--yes", action="store_true",
                    help="confirm the solenoid writes; without it the probe refuses them")
    args = ap.parse_args(argv)

    chain = args.profile if args.profile is not None else (active_profile() or _DEFAULT_PROFILE)
    config = load_robot_config(profile=chain)
    print(f"=== profile {chain!r}: vendor={config.vendor} ip={config.ur.ip} ===")
    print("=== URSim validates the call path, never the wiring. See the module docstring. ===\n")

    arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
    note("create_arm", type(arm).__name__)
    try:
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        note("arm.connect()", f"{type(exc).__name__}: {exc}", ok=False)
        print("\nno controller reachable: is URSim up and in REMOTE control?")
        return _BAD_REQUEST
    note("arm.connect()", "connected")

    if not isinstance(arm, SupportsDigitalIO):
        # Fail closed. Both grippers refuse a non-I/O source in their constructor, and reaching
        # that refusal here would report a TypeError as if the probe itself were broken.
        note("isinstance(arm, SupportsDigitalIO)", "False, neither gripper can be built", ok=False)
        arm.disconnect()
        return _BAD_REQUEST
    note("isinstance(arm, SupportsDigitalIO)", "True")

    try:
        print("\n=== 1. JawIOGripper, a jaw gripper over digital I/O ===")
        if _confirm(args.yes, "drive tool output 0, the jaw close solenoid"):
            _probe_jaw(arm)
        else:
            note("jaw probe", "refused, nothing was energised", ok=False)
        print("\n=== 2. VacuumGripper, the suction side ===")
        if _confirm(args.yes, "drive tool output 1, the vacuum solenoid"):
            _probe_vacuum(arm)
        else:
            note("vacuum probe", "refused, nothing was energised", ok=False)
    except Exception as exc:  # noqa: BLE001
        note("probe", f"{type(exc).__name__}: {exc}", ok=False)
    finally:
        try:
            arm.disconnect()
            note("arm.disconnect()", "clean")
        except Exception as exc:  # noqa: BLE001
            note("arm.disconnect()", f"{type(exc).__name__}: {exc}", ok=False)

    print("\n=== SUMMARY ===")
    bad = [step for step, _outcome, ok in results if not ok]
    print(f"  {len(results) - len(bad)}/{len(results)} steps OK")
    if bad:
        print(f"  NOT OK: {bad}")
    print("  The call path is what a green run measures. Pin assignment, reed polarity, cylinder "
          "travel time and whether a closed jaw holds anything need a bench.")
    print("GRIPPER_PROBE_DONE")
    return _FAILED if bad else _OK


if __name__ == "__main__":
    raise SystemExit(main())
