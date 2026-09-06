"""10: a camera bolted in the cell, and the one transform that never changes.

    python scripts/examples/calibration/10_eye_to_hand.py               # rehearse: check and build
    python scripts/examples/calibration/10_eye_to_hand.py --live        # move, capture, solve
    python scripts/examples/calibration/10_eye_to_hand.py --rig cam_left --marker-length-mm 39.7

The decision is where the camera sits. The config spells it
`grasping.fusion.cameras.<id>.mounting_mode: eye_to_hand | eye_in_hand`, and it is the mounting the
whole run-time frame chain is built on. This file takes the fixed side: a camera bolted to the cell,
watching a marker carried near the tool. `11_eye_in_hand.py` takes the wrist.

Both sides solve AX = XB with the same solver and the same `add_sample(T_base_to_tool,
T_cam_to_marker)` call. What differs is which side of that equation is unknown, and therefore what
the answer means at run time. A fixed camera answers CAMERA to BASE: one static transform, read
from a file, valid for every arm pose, returned by `StaticCameraToBaseResolver` without ever
consulting the arm. A wrist camera answers CAMERA to TOOL, which is half a transform until the live
TCP completes it, every frame.

What the fixed side costs: the camera looks from one place forever, so what it cannot see it never
sees, and the arm itself is the dominant occluder over a bin. The answer to that is a second camera,
`12_two_cameras.py`, not a better calibration.

What it buys: this is the only side a config key can carry.
`grasping.fusion.extrinsics_artifact_path` loads an eye-to-hand artifact and nothing else, and it is
the only key that satisfies the CAMERA to BASE refusal `AutonomousGraspService.from_robot_config`
raises for a real cell. Nothing is read off the arm at run time either, so a TCP that has drifted
moves the grasp but never the camera.

Calibration is the one capability here with no library twin. `CalibrationRoutine.run_auto` returns a
`CalibrationResult` that is a plain dataclass carrying neither `render()` nor `to_dict()`, so this
file drives `real_cell.calibrate` and reads the written artifact back itself. A twin would be a
`Calibration` noun built by `from_config(...)`, with one verb, returning a frozen report holding the
rig, the mounting, the accepted sample count, the RMSE, the quality band and the artifact path.

Two numbers decide whether the answer is any good, and the RMSE is not one of them. The marker
length is the printed black square's edge in millimetres: not the white border, not what the PDF was
named. A board declared 50 mm and printed at 48 mm scales every solved translation by 4 %, converges
cleanly and reports a fine RMSE. And a low RMSE only says the poses agree with each other. What
proves the frame is right is a pick that lands where it was aimed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, hardware_parser, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the rig is valid, the build answered, and with --live an artifact was written
  1  the sweep ran and wrote nothing, or the routine refused it; its own exit code is reported
  2  nothing to calibrate yet: no rig of this kind, no config, or no camera on this machine

The quality band is printed, not enforced: a marginal RMSE still exits 0. Read the band.
"""


def _primary_rgbd_rig(camera_cfg: object) -> str | None:
    """The first `source: rgbd` entry in `camera.cameras.rigs`, or None.

    The same rule `build_real_components` applies when it picks the camera a cell synthesises its
    grasp from, so the default here is the camera that matters most, taken from config rather than
    named in this file. Rig order is therefore not cosmetic: moving the first RGB-D rig moves which
    camera every later example is talking about.
    """
    rigs = getattr(getattr(camera_cfg, "cameras", None), "rigs", None) or []
    for rig in rigs:
        if getattr(rig, "source", None) == "rgbd":
            return str(rig.rig_id)
    return None


def _read_back(run: Example, artifact: Path, rig_id: str) -> None:
    """Load the written artifact with the loader a cell uses, and report what is in it.

    The routine prints its own RESULT block, but that block is the solve talking about itself. This
    reads the file back through `load_extrinsics`, which is the exact call
    `build_config_frame_resolver` makes at cell construction, so a file that will not load is found
    here rather than at the next build.

    Never prints the transform object. `repr(Transform)` contains a non-ASCII arrow, and a stock
    Windows console raises `UnicodeEncodeError` on it, which would end the run in a stack trace
    about text encoding while a calibration was being reported.
    """
    from src.calibration.serialization import load_extrinsics

    with run.step("read the artifact back, as a cell would") as report:
        ext = load_extrinsics(artifact)
        position = [round(float(v), 1) for v in ext.transform.translation_mm]
        report(f"rig '{ext.rig_id}', {ext.num_samples} sample(s), rmse {ext.rmse_mm:.3f} mm, "
               f"{ext.quality}; camera at {position} mm in BASE")
    if ext.rig_id != rig_id:
        # The artifact is keyed by rig id, and `fusion.cameras` is keyed by the same id. An artifact
        # stamped with another camera's name loads without complaint and places this camera's cloud
        # where the other one stands.
        run.finding("artifact rig id",
                    f"the file says rig '{ext.rig_id}' but this run calibrated '{rig_id}'")


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument(
        "--rig", default=None,
        help="rig_id of the camera to calibrate. Default: the first `source: rgbd` rig in "
             "camera.cameras.rigs, which is the camera the cell grasps from. The routine refuses "
             "any rig that is not an RGB-D device, by name")
    parser.add_argument("--poses", type=int, default=22,
                        help="how many generated TCP poses to visit (default 22)")
    parser.add_argument("--marker-length-mm", type=float, default=None,
                        help="printed ArUco square edge in mm. Measure it. Default: the config's")
    parser.add_argument("--out", default="calibration/real",
                        help="directory for the artifact and the sample dataset")
    args = parser.parse_args(argv)

    # Imported after the parse, so `--help` costs nothing and answers in a checkout whose
    # dependencies are not installed.
    from src.config.loader import load_config
    from src.robot.execution.real_cell import calibrate

    with Example("10 eye_to_hand", "a fixed camera, in the robot's frame",
                 live=args.live, profile=args.profile) as run:
        # Inside the block, because `Example` applies `--profile` on entry and puts it back on exit.
        # Loading above it would read a different tree than every step below reports on.
        with run.step("load the config tree") as report:
            try:
                app = load_config()
            except Exception as error:                          # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the config tree did not load ({type(error).__name__}: {error})",
                    "run `python -m src.config --print`, and check WILLY_PROFILE if you set one")
            rig_id = args.rig or _primary_rgbd_rig(app.camera)
            report(f"{len(app.camera.cameras.rigs)} rig(s) declared; "
                   f"robot vendor {getattr(app.robot, 'vendor', '(no robot block)')}")
        if rig_id is None:
            return not_ready(
                "no `source: rgbd` rig in camera.cameras.rigs, so there is no camera to calibrate",
                "add an RGB-D rig (config/camera/cam.yaml ships one), or pass --rig explicitly. A "
                "stereo pair calibrates through the routine's own stereo path, not this one")

        settings = app.camera.hand_eye.eye_to_hand
        with run.step("pose acceptance thresholds in force") as report:
            report(f"min_samples {settings.min_samples}, "
                   f"min_distance {settings.min_distance_mm} mm, "
                   f"min_angle {settings.min_angle} deg, "
                   f"marker {args.marker_length_mm or settings.marker_length_mm} mm, "
                   f"{settings.aruco_dict_name}")
        run.note("A pose is stored only if it differs from every stored pose by more than the")
        run.note("distance or the angle. Raise them to widen the spread; raise them too far")
        run.note("and the generator cannot find enough poses inside the workspace limits.")
        run.note("The two `enabled` flags in camera.hand_eye are read by no code. They record what")
        run.note("the cell intends. The mounting is chosen by which block the routine is handed,")
        run.note("which for this path is the eye_to_hand one, always.")

        # The routine narrates its own stages, so every call to it is made outside `run.step`: a
        # step holds its announcement line open until the body reports, and anything printed from
        # inside the body lands on that line.
        run.note("")
        run.note("the routine's own --check runs next, and narrates itself:")
        base_argv = ["--rig", rig_id, "--mode", "eye_to_hand", "--out", args.out]
        if args.profile:
            base_argv += ["--profile", args.profile]
        code = calibrate.main([*base_argv, "--check"])
        if code != 0:
            return not_ready(
                f"the routine refused rig '{rig_id}' before touching anything (exit {code})",
                "the refusal above names what is missing; check camera.cameras.rigs")
        with run.step("validate the rig and the config") as report:
            report(f"rig '{rig_id}' is declared, is RGB-D, and the tree is coherent")

        artifact = Path(args.out) / f"eth_{rig_id}.json"
        with run.step("what a live run would do") as report:
            report(f"{args.poses} poses, tool down; writes {artifact}")
        run.note("The path is relative, so where it lands depends on your working directory.")

        if artifact.is_file() and not args.live:
            # A rehearsal on an already calibrated cell has something real to say, and this is the
            # check that matters: not that a file exists, but that the loader accepts it. Only in a
            # rehearsal: a live run is about to overwrite this file, and reporting the old solve
            # here and the new one below would print two calibrations under one heading.
            _read_back(run, artifact, rig_id)
            # Through getattr, because a perception-only tree has no `robot` block at all and this
            # is a remark about wiring, not a reason to stop calibrating.
            fusion = getattr(getattr(app.robot, "grasping", None), "fusion", None)
            if fusion is not None and not fusion.extrinsics_artifact_path:
                run.finding(
                    "fusion.extrinsics_artifact_path",
                    "unset, so this artifact reaches nothing; a real cell is refused at build")

        if not args.live:
            # The rehearsal ends by building the real components: the vendor arm driver and this
            # camera, opened through the frame provider. It commands no motion, which is the whole
            # contract, and it is where a cell with no camera attached finds out.
            run.note("")
            run.note("the routine's own --dry-run runs next: it builds the arm driver and opens")
            run.note("the camera, and stops before any motion.")
            code = calibrate.main([*base_argv, "--dry-run"])
            if code != 0:
                return not_ready(
                    f"the build refused before any motion (exit {code}); the reason is printed "
                    "above",
                    f"is the camera plugged in and is its serial the one in camera.cameras.rigs "
                    f"for '{rig_id}'? Prove it alone first: python -m src.robot.perception "
                    f"--rig {rig_id}")
            with run.step("build the arm driver and open the camera") as report:
                report("both answered, and nothing was commanded")

            run.note("")
            run.note("Rehearsal stops here. --live moves the arm through every pose.")
            run.note("Before you pass --live:")
            run.note("  the marker is flat, rigidly fixed near the tool, and inside every view")
            run.note("  you have measured the printed square with calipers")
            run.note(f"  the cell is clear: the arm visits {args.poses} poses")
            return run.exit_code

        live_argv = [*base_argv, "--poses", str(args.poses)]
        if args.marker_length_mm is not None:
            live_argv += ["--marker-length-mm", str(args.marker_length_mm)]
        run.note("")
        run.note("the sweep runs now, and the routine narrates every stage:")
        code = calibrate.main(live_argv)
        with run.step(f"drive {args.poses} poses and solve AX=XB") as report:
            report(f"routine exit {code}")
        if code != 0:
            # Its own exit codes carry the meaning, and flattening them to "failed" throws away a
            # distinction the routine was careful to make: a configuration it refused (1), a sweep
            # that ran and produced no artifact (2) and an unexpected error (3) are three problems.
            #
            # `EXIT_FAILED`, not `run.exit_code`: a finding is deliberately non blocking, so
            # `run.exit_code` would answer 0 for a run that wrote nothing.
            run.finding("calibration", f"exit {code} from the routine; read its RESULT block above")
            return EXIT_FAILED

        _read_back(run, artifact, rig_id)
        run.note("")
        run.note("Point the stack at what it wrote. Two keys, and they buy different things:")
        run.note(f"  robot.grasping.fusion.extrinsics_artifact_path: {artifact}")
        run.note("      the single CAMERA to BASE resolver. It is what the 'camera -> base'")
        run.note("      preflight check reads, and the only key that satisfies the refusal a real")
        run.note("      cell raises at build. Set it for the primary camera and nothing else.")
        run.note("  robot.grasping.fusion.cameras.<rig_id>: the per camera map, printed above by")
        run.note("      the routine as ready to paste. It is what geometry fusion reads, and the")
        run.note("      primary camera does not belong in it. 12_two_cameras.py is that decision.")
        run.note("")
        run.note("Then run a pick. A grasp that lands where it was aimed is what proves the frame;")
        run.note("the RMSE above only proves the poses agree with each other.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
