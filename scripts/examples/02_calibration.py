"""02: teach the camera where the robot is, and refuse to ship a bad answer.

    python scripts/examples/02_calibration.py --rig realsense_d435           # rehearse: validate
    python scripts/examples/02_calibration.py --rig realsense_d435 --live    # move, capture, solve
    python scripts/examples/02_calibration.py --rig realsense_d435 --mode eye_in_hand --live

Nothing downstream works without this. `01_robot_setup` reports it as a blocking check, in the words
that matter: without a `CAMERA->BASE` transform the driver refuses every motion as
`INVALID_TARGET`, which looks like a broken cell. It is not. It is a cell that has never been told
where its camera is.

What the two modes are, because picking the wrong one wastes a whole session:

  `eye_to_hand`  a fixed camera watching the cell. The artifact is `CAMERA->BASE`, and it is what
                 the pick path consumes through `grasping.fusion`.
  `eye_in_hand`  a camera on the wrist. The artifact is `CAMERA->TOOL`; the `CAMERA->BASE`
                 transform is then recomputed from it at every pose, which is why a wrist camera
                 can look from anywhere and a fixed one cannot.

The marker length is the one number nobody checks and everybody gets wrong. It is the printed
square's edge in millimetres: the black square, not the white border, and not what the PDF was
called. A marker declared 40 mm and printed at 38 mm produces a calibration that solves cleanly,
reports a plausible RMSE, and is wrong by 5 % everywhere. Measure the print with calipers.

And a low RMSE is not a good calibration. It says the poses are self consistent, not that the frame
is right, because a systematically wrong marker size fits beautifully. The verdict this example
prints is a band, and the band is about consistency. The thing that proves the frame is right is a
pick that lands where it was aimed, which is `03_pick.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EXIT_FAILED, Example, hardware_parser, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the rig is valid, and with --live the routine wrote an artifact
  1  the routine refused the sweep, or ran it and wrote nothing; its own exit code is reported
  2  nothing to calibrate yet: rig unknown, no config, or no hardware

The quality band is printed, not enforced: a marginal RMSE still exits 0. Read the band.
"""


def main(argv: list[str] | None = None) -> int:
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument("--rig", required=True,
                        help="rig_id of the camera to calibrate. It must be in camera.cameras.rigs "
                             "and it must be an RGB-D rig; the routine refuses the others by name")
    parser.add_argument("--mode", choices=("eye_to_hand", "eye_in_hand"), default="eye_to_hand",
                        help="a fixed camera (default) or one on the wrist")
    parser.add_argument("--poses", type=int, default=22,
                        help="how many generated TCP poses to visit (default 22)")
    parser.add_argument("--marker-length-mm", type=float, default=None,
                        help="printed ArUco square edge in mm. Measure it. Default: the config's")
    args = parser.parse_args(argv)

    from src.robot.execution.real_cell import calibrate

    with Example("02 calibration", f"{args.mode} for rig '{args.rig}'",
                 live=args.live, profile=args.profile) as run:
        # This example drives a command line, and it is the only one here that does. Calibration has
        # no library twin: `CalibrationRoutine.run_auto` returns a `CalibrationResult` that carries
        # neither `render()` nor `to_dict()`, so the twin would have to be a `Calibration` noun,
        # built by `from_config`, whose one verb returns a frozen report holding the rig, the mode,
        # the accepted sample count, the RMSE, the quality band and the written artifact path. Until
        # that exists, `calibrate.main` is the only door that composes rig lookup, marker source,
        # sweep, band and artifact write, and re-implementing the rig lookup here would drift from
        # the thing this example is teaching.
        #
        # The routine narrates its own stages, so every call to it is made outside `run.step`: a
        # step holds its announcement line open until the body reports, and anything printed from
        # inside the body lands on that line.
        run.note("the routine's own --check runs next, and narrates itself:")
        check_argv = ["--rig", args.rig, "--mode", args.mode, "--check"]
        if args.profile:
            check_argv += ["--profile", args.profile]
        code = calibrate.main(check_argv)
        if code != 0:
            return not_ready(
                f"the routine refused rig '{args.rig}' before touching anything (exit {code})",
                "check `camera.cameras.rigs` in your config; the refusal above names what is "
                "missing")
        with run.step("validate the rig and the config") as report:
            report(f"rig '{args.rig}' is declared and the config is coherent")

        with run.step("what this run would do") as report:
            report(f"{args.poses} generated poses; "
                   f"marker {args.marker_length_mm or '(from config)'} mm; "
                   f"artifact {'CAMERA->BASE' if args.mode == 'eye_to_hand' else 'CAMERA->TOOL'}")
        # The routine's default output directory is relative, so where the artifact lands depends on
        # where this was started from.
        run.note("the artifact lands under `calibration/real/`, relative to your working directory")

        if not args.live:
            run.note("Rehearsal stops here. The next step moves the arm through every pose.")
            run.note("")
            run.note("Before you pass --live:")
            run.note("  the marker is flat, rigidly fixed, and fully inside every view")
            run.note("  you have measured the printed square with calipers")
            run.note(f"  the cell is clear: the arm visits {args.poses} poses across its workspace")
            return run.exit_code

        live_argv = ["--rig", args.rig, "--mode", args.mode, "--poses", str(args.poses)]
        if args.marker_length_mm is not None:
            live_argv += ["--marker-length-mm", str(args.marker_length_mm)]
        if args.profile:
            live_argv += ["--profile", args.profile]
        run.note("")
        run.note("the sweep runs now, and the routine narrates every stage:")
        code = calibrate.main(live_argv)
        with run.step(f"drive {args.poses} poses and solve AX=XB") as report:
            report(f"routine exit {code}")

        if code != 0:
            # Its own exit codes carry the meaning, and flattening them to "failed" would throw away
            # the distinction the routine was careful to make: a configuration it refused, a sweep
            # that produced no artifact and an unexpected error are three different problems.
            #
            # The status returned is `EXIT_FAILED`, not `run.exit_code`. A finding is deliberately
            # non blocking, so `run.exit_code` would answer 0 for a run that wrote nothing.
            run.finding("calibration", f"exit {code} from the routine; read its RESULT block above")
            return EXIT_FAILED

        run.note("")
        run.note("Point the stack at what it wrote. Two keys, and they buy different things:")
        run.note("  robot.grasping.fusion.extrinsics_artifact_path: <the file named above>")
        run.note("      one CAMERA->BASE resolver, and the key the 'camera -> base' preflight")
        run.note("      check reads. Setting it is what turns that check in 01_robot_setup green.")
        run.note("  robot.grasping.fusion.cameras.<rig_id>: the per camera map, which the routine")
        run.note("      printed above as ready to paste. This is what the geometry fusion reads;")
        run.note("      without it fusion stands down to a single view and says so in telemetry.")
        run.note("The first buys permission to move. The second buys the fused view.")
        run.note("")
        run.note("Then run 03_pick.py: a pick that lands where it was aimed is what proves the")
        run.note("frame is right. The RMSE above only proves the poses agree with each other.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
