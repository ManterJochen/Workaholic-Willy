"""11: a camera on the flange, and half a transform the arm completes every frame.

    python scripts/examples/calibration/11_eye_in_hand.py               # rehearse: check and build
    python scripts/examples/calibration/11_eye_in_hand.py --live        # move, capture, solve
    python scripts/examples/calibration/11_eye_in_hand.py --rig wrist --marker-length-mm 39.7

The decision is where the camera sits, the same one `10_eye_to_hand.py` takes from the other side.
The config spells it `grasping.fusion.cameras.<id>.mounting_mode: eye_to_hand | eye_in_hand`. This
file takes the wrist: a camera bolted to the flange, watching a marker that lies still in the
workspace.

Both sides solve AX = XB with the same solver and the same `add_sample(T_base_to_tool,
T_cam_to_marker)` call. What differs is which side of that equation is unknown, and therefore what
the answer means at run time. A fixed camera answers CAMERA to BASE and is done: one transform, read
from a file, right for every arm pose. A wrist camera answers CAMERA to TOOL, which is not yet a
usable answer. `EyeInHandFrameResolver` takes the TCP pose for the frame, reinterprets it as TOOL to
BASE and composes the two, once per frame.

What that costs. The camera pose is now only as good as the arm's pose, so a drifted TCP or a wrong
tool frame moves the camera as well as the grasp, and one bad calibration corrupts two things
instead of one. The pose also has to be the one from the moment the shutter opened: the resolver
uses `frame.tool_pose` when the perception source stamped it and asks the arm at resolve time when
it did not, and that fallback puts everything the tool travelled between capture and resolve into
the grasp. The closed-loop path moves the arm between the two by design.

What it buys. The camera can look from anywhere the arm can reach, so an occlusion is a move rather
than a permanent blind spot, and the answer is checkable with a tape measure: the solved translation
is where the camera sits relative to the tool, tens of millimetres, and you can see whether it is
plausible. Nobody can eyeball an eye-to-hand translation the same way.

The trap that decides whether this cell can be built at all. The artifact is a
`willy.calibration.cam_to_tool/1` file, and `grasping.fusion.extrinsics_artifact_path` loads
`willy.calibration.extrinsics/1` and nothing else. So the wrist artifact cannot be the primary
CAMERA to BASE resolver, and a real cell whose only camera is on the wrist is refused at build with
the message about `INVALID_TARGET`. Two ways out, and this file runs the refusal that separates
them: list the camera in `grasping.fusion.cameras` with `mounting_mode: eye_in_hand`, which builds
an `EyeInHandFrameResolver` in the per camera map and is the fusion path rather than the primary
one, or pass `frame_resolver=` in code, which is what the simulator's wrist camera runner does and
what the refusal itself recommends.

One more thing to know before tuning anything. On this path `real_cell.calibrate` reads
`camera.hand_eye.eye_to_hand` for both mountings, so the thresholds and the marker length in force
here come out of the eye-to-hand block; `camera.hand_eye.eye_in_hand` is read by the simulator's
wrist camera calibration and by nothing else. The base tree ships both blocks with the same numbers,
so the difference stays invisible until somebody edits one of them. The step below prints both.

Calibration is the one capability here with no library twin: `CalibrationRoutine.run_auto` returns a
`CalibrationResult` that is a plain dataclass with neither `render()` nor `to_dict()`, so this file
drives `real_cell.calibrate` and reads the written artifact back itself. A twin would be a
`Calibration` noun built by `from_config(...)`, with one verb, returning a frozen report holding the
rig, the mounting, the accepted sample count, the RMSE, the quality band and the artifact path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, hardware_parser, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the rig is valid, the build answered, and with --live an artifact was written
  1  the sweep ran and wrote nothing, or the routine refused it; its own exit code is reported
  2  nothing to calibrate yet: no rig of this kind, no config, or no camera on this machine

The quality band is printed, not enforced: a marginal RMSE still exits 0. Read the band, and then
read the solved translation, which is the number you can check against the mount.
"""


def _primary_rgbd_rig(camera_cfg: object) -> str | None:
    """The first `source: rgbd` entry in `camera.cameras.rigs`, or None.

    The same rule `build_real_components` applies when it picks a cell's primary camera. On a cell
    whose only camera is the wrist one that is the right default; on a cell that also has a fixed
    camera it is not, and `--rig` is how the wrist one is named.
    """
    rigs = getattr(getattr(camera_cfg, "cameras", None), "rigs", None) or []
    for rig in rigs:
        if getattr(rig, "source", None) == "rgbd":
            return str(rig.rig_id)
    return None


def _why(error: Exception) -> str:
    """One line of a refusal: the reason, without the paragraph that follows it.

    These messages are written to be read at a terminal, so they run to several sentences and often
    quote an absolute path. A step reports one line, and the first sentence after the path is the
    part that says what was wrong. Both trims degrade to the whole message rather than to nothing.
    """
    text = " ".join(str(error).split())
    if "failed to load: " in text:
        text = text.split("failed to load: ", 1)[1]
    return text.split(". ", 1)[0][:110]


def _the_key_refuses_this_artifact(run: Example, robot: Any, artifact: Path) -> None:
    """Point the single CAMERA to BASE key at the wrist artifact, and show it refused.

    A config experiment on a copy of the loaded tree: nothing is written and no YAML is touched. The
    refusal is a schema check inside `load_extrinsics`, so it fires on the first build rather than
    on the first pick, which is the difference between an operator reading an error and an operator
    watching a cell grasp at the wrong place.
    """
    from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolver

    fusion = robot.grasping.fusion.model_copy(
        update={"enabled": True, "extrinsics_artifact_path": str(artifact)})
    probe = robot.grasping.model_copy(update={"fusion": fusion})
    with run.step("the single key refuses a wrist artifact") as report:
        try:
            build_config_frame_resolver(probe)
        except Exception as error:                              # noqa: BLE001  (the refusal is it)
            # The wanted schema is named in the reason, and a reader who sees it once recognises it
            # in a build log. The absolute path in front of it is trimmed away by `_why`.
            report(f"{type(error).__name__}: {_why(error)}")
            return
        run.finding("fusion.extrinsics_artifact_path",
                    "it LOADED a wrist artifact, which the schema is supposed to refuse")


def _a_wrist_only_cell_is_refused(run: Example, robot: Any, rig_id: str, artifact: Path) -> None:
    """Build a cell whose only calibrated camera is the wrist one, and show it refused.

    Again a copy of the loaded tree, and the reason to run it rather than describe it: the refusal
    comes from `from_robot_config` itself, so it stays true when that method changes. The rehearsal
    components are the library's own desk stand-ins, the ones `build_rehearsal_cell` uses; they are
    here only because `calculator` and `perception` are required arguments, and the refusal fires
    before either is used.
    """
    from src.config.schema.robot.grasping_schema import CameraExtrinsicsConfig
    from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components
    from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

    fusion = robot.grasping.fusion.model_copy(update={
        "enabled": True,
        "extrinsics_artifact_path": None,
        "cameras": {rig_id: CameraExtrinsicsConfig(
            enabled=True, mounting_mode="eye_in_hand", extrinsics_artifact_path=str(artifact))},
    })
    probe = robot.model_copy(
        update={"grasping": robot.grasping.model_copy(update={"fusion": fusion})})
    try:
        calculator, perception, _resolver, _cameras, _lenses = build_rehearsal_components(probe)
    except Exception as error:                                  # noqa: BLE001  (report, not raise)
        # `build_calculator` is fail-closed on `grasping.calculator: deep` with no readable
        # artifact, and that refusal arrives before the one under test here. Saying so is better
        # than reporting a missing generator as a frame problem.
        run.finding("wrist-only cell",
                    f"not reached: the calculator refused first ({type(error).__name__})")
        return
    with run.step("a wrist-only real cell is refused at build") as report:
        try:
            AutonomousGraspService.from_robot_config(
                probe, calculator=calculator, perception=perception)
        except Exception as error:                              # noqa: BLE001  (the refusal is it)
            report(f"{type(error).__name__}: {_why(error)}")
            return
        run.finding("wrist-only cell",
                    "it BUILT, so this vendor is exempt from the CAMERA to BASE refusal "
                    "(sim and dummy are)")


def _read_back(run: Example, artifact: Path) -> None:
    """Load the written artifact with `load_cam_to_tool`, the call the per camera map makes.

    Reports the translation, because on this mounting that number is checkable: it is where the
    camera sits relative to the tool. Never prints the transform object itself, because
    `repr(Transform)` contains a non-ASCII arrow and a stock Windows console raises
    `UnicodeEncodeError` on it.
    """
    from src.calibration.serialization import load_cam_to_tool

    with run.step("read the artifact back, as a cell would") as report:
        transform = load_cam_to_tool(artifact)
        offset = [round(float(v), 1) for v in transform.translation_mm]
        report(f"camera at {offset} mm from the tool, frames "
               f"{transform.from_frame.value} to {transform.to_frame.value}")
    run.note("Check that offset against the mount with a ruler. It is the one number on this")
    run.note("mounting that a person can falsify without running a pick.")


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument(
        "--rig", default=None,
        help="rig_id of the wrist camera. Default: the first `source: rgbd` rig in "
             "camera.cameras.rigs. On a cell that also has a fixed camera, name the wrist one")
    parser.add_argument("--poses", type=int, default=22,
                        help="how many generated TCP poses to visit (default 22)")
    parser.add_argument("--marker-length-mm", type=float, default=None,
                        help="printed ArUco square edge in mm. Measure it. Default: the config's")
    parser.add_argument("--out", default="calibration/real",
                        help="directory for the artifact and the sample dataset")
    args = parser.parse_args(argv)

    from src.config.loader import load_config
    from src.robot.execution.real_cell import calibrate

    with Example("11 eye_in_hand", "a wrist camera, in the tool's frame",
                 live=args.live, profile=args.profile) as run:
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
                "add an RGB-D rig (config/camera/cam.yaml ships one), or pass --rig explicitly")

        in_force = app.camera.hand_eye.eye_to_hand
        declared = app.camera.hand_eye.eye_in_hand
        with run.step("pose acceptance thresholds in force") as report:
            report(f"min_samples {in_force.min_samples}, "
                   f"min_distance {in_force.min_distance_mm} mm, "
                   f"min_angle {in_force.min_angle} deg, "
                   f"marker {args.marker_length_mm or in_force.marker_length_mm} mm, "
                   f"{in_force.aruco_dict_name}")
        run.note(f"Those come from camera.hand_eye.eye_to_hand. The eye_in_hand block declares "
                 f"min_samples {declared.min_samples}, marker {declared.marker_length_mm} mm,")
        run.note("and on this path nothing reads it: the real-cell routine hands the eye-to-hand")
        run.note("block to both mountings, and only the simulator's wrist runner reads the other.")
        run.note("Tune the block above, not the one named after this file.")

        # The routine narrates its own stages, so every call to it is made outside `run.step`: a
        # step holds its announcement line open until the body reports, and anything printed from
        # inside the body lands on that line.
        run.note("")
        run.note("the routine's own --check runs next, and narrates itself:")
        base_argv = ["--rig", rig_id, "--mode", "eye_in_hand", "--out", args.out]
        if args.profile:
            base_argv += ["--profile", args.profile]
        code = calibrate.main([*base_argv, "--check"])
        if code != 0:
            return not_ready(
                f"the routine refused rig '{rig_id}' before touching anything (exit {code})",
                "the refusal above names what is missing; check camera.cameras.rigs")
        with run.step("validate the rig and the config") as report:
            report(f"rig '{rig_id}' is declared, is RGB-D, and the tree is coherent")

        artifact = Path(args.out) / f"eih_{rig_id}.json"
        with run.step("what a live run would do") as report:
            report(f"{args.poses} poses, tool down; writes {artifact}")
        run.note("Tool down is how the sweep is generated for both mountings, so the marker lies")
        run.note("flat in the workspace and the wrist looks at it from above, from many angles.")

        robot = app.robot
        if robot is None:
            # A perception-only tree has no `robot` block, and the two refusals below are about
            # what a robot cell does with the artifact. Reported rather than crashed.
            run.finding("frame wiring", "this config tree has no `robot` block, so nothing below "
                                        "can be shown")
        else:
            run.note("")
            run.note("Where this artifact can and cannot be wired. Both checks below run on a copy")
            run.note("of the loaded config; no file is written and no YAML is edited.")
            if artifact.is_file():
                _the_key_refuses_this_artifact(run, robot, artifact)
            else:
                run.note(f"  ({artifact} does not exist yet, so the schema refusal is not run;")
                run.note("   it fires the moment that path is used as the single key.)")
            _a_wrist_only_cell_is_refused(run, robot, rig_id, artifact)

        if artifact.is_file() and not args.live:
            # Only in a rehearsal. A live run is about to overwrite this file, and reporting the
            # old solve here and the new one below would print two calibrations under one heading.
            _read_back(run, artifact)

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
                    f"is the wrist camera plugged in and is its serial the one in "
                    f"camera.cameras.rigs for '{rig_id}'? Prove it alone first: "
                    f"python -m src.robot.perception --rig {rig_id}")
            with run.step("build the arm driver and open the camera") as report:
                report("both answered, and nothing was commanded")

            run.note("")
            run.note("Rehearsal stops here. --live moves the arm through every pose.")
            run.note("Before you pass --live:")
            run.note("  the marker is flat, rigidly fixed in the workspace, and stays put")
            run.note("  you have measured the printed square with calipers")
            run.note("  the camera cable has slack for every pose, and cannot snag")
            run.note(f"  the cell is clear: the arm visits {args.poses} poses across its workspace")
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
            # The routine's own codes carry the meaning: a configuration it refused (1), a sweep
            # that ran and produced no artifact (2) and an unexpected error (3) are three problems.
            # `EXIT_FAILED`, not `run.exit_code`, because a finding is deliberately non blocking and
            # `run.exit_code` would answer 0 for a run that wrote nothing.
            run.finding("calibration", f"exit {code} from the routine; read its RESULT block above")
            return EXIT_FAILED

        _read_back(run, artifact)
        run.note("")
        run.note("Wire it in one of the two places, and not in the third:")
        run.note(f"  robot.grasping.fusion.cameras.{rig_id}.mounting_mode: eye_in_hand")
        run.note(f"  robot.grasping.fusion.cameras.{rig_id}.extrinsics_artifact_path: {artifact}")
        run.note("      builds an EyeInHandFrameResolver in the per camera map, which is what")
        run.note("      geometry fusion consumes. It does not give the cell its primary resolver.")
        run.note("  frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=...), passed in code")
        run.note("      the primary route for a wrist camera, and the one the refusal recommends.")
        run.note("  robot.grasping.fusion.extrinsics_artifact_path: NOT this file. That key loads")
        run.note("      CAMERA to BASE extrinsics only, and refuses this schema by name.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
