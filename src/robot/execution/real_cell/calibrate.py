"""Calibrate one camera of a real cell against the robot: the missing half of multi-view.

    python -m src.robot.execution.real_cell.calibrate --rig overhead --check
    python -m src.robot.execution.real_cell.calibrate --rig overhead --poses 22

`grasping.fusion.geometry`, the biggest measured lever this stack has (top-1 43.50 % single-view
-> 55.93 % fused on the datagen reference, n=354), states its own precondition: ``cameras`` must be
populated, each camera individually calibrated. The consuming side is complete: each rig's declared
calibration, `build_config_frame_resolvers` turning the fused cameras into
`{camera_id -> FrameResolver}`, the pick loop reading that map, the schema-versioned `Extrinsics`
artifact keyed by `rig_id`, and `CalibrationRoutine` itself. `real_cell/preflight.py` only checks
for an artifact and refuses without one; it cannot make one. This command makes one.

It follows the simulated runner's path: the same `CalibrationRoutine`, the same `run_auto`, the same
`save_extrinsics` keyed by camera id, with three real-hardware substitutions:

* the marker comes from `RGBDArucoMarkerSource` on a live RGB-D rig instead of simulated ground
  truth,
* the arm is the configured vendor driver, built alone through
  `Robot.from_config(robot_config, gripper=None)`, instead of a simulated one,
* and the frames come from the rig's `Camera` owner, so this holds exactly one camera and gives it
  back, the same owner the pick path uses.

The flow is the library noun `src.robot.execution.hand_eye.HandEyeCalibration`, and this file is a
caller of it: it loads the tree, prints the stage banners and what each stage reports, and returns
the report's exit code.

This moves the robot. `run_auto` drives the arm to N generated poses, in bearing order round the
base. `--fixed-poses PATH` visits your own stations instead, in the order a JSON file lists them:
poses, and joint stations written as the pendant shows a configuration (`{"joints_deg": [six
values]}`, see `pose_provider`). A joint station is sent as one joint move, which the arm judges as
it judges any joint move, after the grasp centre the arm's forward kinematics puts at its joints has
passed the workspace box; a station that does not pass is reported and never moved to. `--check`
validates everything, reads and checks a stations file and counts its stations, and on a cuRobo UR cell
that declares its planner margin looks up the committed evidence the sweep's planner starts on, so a
combination nobody measured (a declared `payload.length_mm` among them) is refused at the desk rather
than after the arm connects. It touches nothing. `--dry-run` additionally runs the arm-vendor
readiness gate, builds the arm, opens the camera and prints what the arm's safety pipeline refuses,
but takes no lock and never commands a motion. Run both before the first live sweep.

`--aim-at=X,Y,Z` aims a wrist camera (`--mode eye_in_hand`) at one marker lying flat, face up, at
that point in BASE millimetres, with `--board aruco:ID:SIZE_MM[:DICT]` or the block's marker: the
sweep looks once from where the arm stands, estimates from that view where the camera sits on the
tool, and visits a cone of views round the marker from `--aim-distance-mm` (500 by default), each
tilting the tool as little as it can from tool-down with `--closing-axis`, re-aimed as the marker is
seen (`src/robot/execution/camera_aim.py`). Write it with `=`, because a value that starts with a
minus sign reads as a flag otherwise. A first look that sees no marker moves nothing and exits 3
saying what to do.

A refused move skips its pose. The sweep stops at a pose, and exits 3 naming it, where the
controller cannot be reached, refused a move it had been sent, or reports a protective or
emergency stop, rather than trying every pose after it.

The sweep connects through `Robot.connected()`. The cell lock comes first, the lock a pick run and
the operator console take for the same controller, so a sweep is refused while either holds it and
is told who does. Then the arm, and no gripper: a sweep beside a board needs no activation stroke,
and a gripper this tree cannot build must not stand in the way of calibrating a camera. On the way
out the arm comes down, the lock is given back, and then the camera. Every move of the sweep
declines the camera world, because the sweep is what produces the transform a camera world needs.

The marker length, its id and the ArUco dictionary default to the mode's `camera.hand_eye` block, and
a wrist sweep reads `camera.hand_eye.eye_in_hand`. On every shipped tree the two blocks agree and the
dictionary is `DICT_5X5_100`. That block's `target` may name a ChArUco board instead, and `--board` or
`--board-file` name the target for one run; `--check` validates whichever applies against the config
schema, so a dictionary OpenCV does not know, a length that is not above zero or an id the dictionary
does not hold is refused before anything is built.

Each pose prints one line per event as it happens (`render_sweep_event` in `hand_eye.py`): where the
arm is going, what the camera saw, and whether the pose counted or why not. The result stage lists
every pose again.

A window beside the sweep shows the camera while the arm moves and pins each judged frame with the
target drawn on it (`src/calibration/preview.py`). By default it opens only where OpenCV has a GUI,
a display is there and stdout is a terminal, so a piped or logged run prints what it printed before;
`--preview` asks for it and the build says why when it cannot open, `--no-preview` and
`WILLY_NO_PREVIEW=1` keep it shut. It is display only: closing it or pressing ESC closes the window,
and the sweep goes on. The pendant stops the robot.

The routine and the solve are exercised in simulation only. The ArUco marker source has never seen
a physical D435, and nothing here has run against a physical controller.

Exit codes: 0 success; 1 configuration refused, build refused, the cell held by another process, or
the connect refused; 2 the calibration ran but did not produce an artifact; 3 the sweep raised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.loader import ConfigError
from src.contracts import UNSET
from src.robot.execution.hand_eye import (
    DEFAULT_OUT_DIR,
    CalibrationStage,
    HandEyeCalibration,
    SweepOptions,
    print_sweep_progress,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.hand_eye import CalibrationBuild
    from src.robot.execution.lifecycle import ConnectStage

_EXIT_OK, _EXIT_CONFIG, _EXIT_NO_ARTIFACT, _EXIT_ERROR = 0, 1, 2, 3


def _load(profile: str | None, data_dir: str | None) -> Any:
    """The whole config tree, honouring an explicit profile chain.

    argparse's ``None`` means "nobody typed --profile" and must become ``UNSET``, not ``None``:
    ``load_config(profile=None)`` means "the base tree, ignore ``WILLY_PROFILE``", which would
    silently disable the variable for an operator who exports it. Saving, setting and restoring
    ``os.environ`` around the load is not the way to do this: process-global state for the
    duration of one call leaks to any other thread and survives an exception.
    """
    from src.config.loader import load_config

    return load_config(data_dir, profile=UNSET if profile is None else profile)


def build_parser() -> argparse.ArgumentParser:
    # Every sweep setting defaults to UNSET, so the noun resolves it once: the flag, else the tree, else its own
    # default. `--mode` keeps its literal default, the one the runbook relies on.
    ap = argparse.ArgumentParser(
        prog="python -m src.robot.execution.real_cell.calibrate",
        description="Calibrate one camera of a real cell against the robot. This moves the robot.")
    ap.add_argument("--rig", required=True,
                    help="rig_id of the RGB-D camera to calibrate (must be in camera.cameras.rigs). "
                         "The artifact is keyed by this id, and the rig block it prints declares "
                         "it on that rig")
    ap.add_argument("--mode", choices=("eye_to_hand", "eye_in_hand"), default="eye_to_hand",
                    help="eye_to_hand = a fixed camera; the artifact is CAMERA->BASE and this is what "
                         "multi-view fusion consumes. eye_in_hand = a wrist camera; the artifact is "
                         "CAMERA->TOOL and is composed with the live TCP each frame")
    stations = ap.add_mutually_exclusive_group()
    stations.add_argument("--poses", type=int, default=UNSET,
                          help="how many generated TCP poses to visit (default 22, as in sim), in bearing order round "
                               "the base")
    stations.add_argument("--fixed-poses", dest="fixed_poses", default=UNSET, metavar="PATH",
                          help="visit your own stations instead, in the order a JSON file lists them: poses "
                               "{x, y, z, rx, ry, rz} (mm, axis-angle rad) and joint stations {\"joints_deg\": [six "
                               "values]} or {\"joints_rad\": [...]}, from the base to the last wrist joint. --check "
                               "reads and checks the file and counts its stations. A joint station runs as one judged "
                               "joint move; a station whose grasp centre is outside the workspace box, or too close to "
                               "one before it, is reported and never moved to")
    stations.add_argument("--aim-at", dest="aim_at", default=UNSET, metavar="X,Y,Z",
                          help="eye_in_hand only: aim the CAMERA at one marker lying flat, face up, at this point in "
                               "BASE mm (write --aim-at=-130,-700,50). The sweep looks once from where the arm stands, "
                               "estimates where the camera sits on the tool, and visits views round the marker; a "
                               "first look that sees no marker moves nothing. Needs one marker as the target")
    ap.add_argument("--aim-distance-mm", dest="aim_distance_mm", type=float, default=UNSET,
                    help="with --aim-at: how far the camera stands from the marker (default 500)")
    ap.add_argument("--closing-axis", dest="closing_axis", default=UNSET,
                    help="with --aim-at: the heading each aimed station keeps, tilting only to aim, as Pose.tool_down "
                         "takes it (x, -x, y, -y, radial, tangential and their negatives; default x)")
    ap.add_argument("--marker-length-mm", type=float, default=UNSET,
                    help="printed ArUco square edge in mm. Default: camera.hand_eye's. A wrong "
                         "value scales every sample uniformly; the solve converges and is uniformly "
                         "wrong. Measure the printed board")
    ap.add_argument("--marker-id", type=int, default=UNSET,
                    help="the ArUco id to pose. Default: camera.hand_eye's marker_id, for the mode (0 as shipped)")
    ap.add_argument("--dict", dest="dict_name", default=UNSET,
                    help="ArUco dictionary; case and the DICT_ prefix are optional. Default: camera.hand_eye's, "
                         "for the mode")
    board = ap.add_mutually_exclusive_group()
    board.add_argument("--board", dest="board", default=UNSET, metavar="SPEC",
                       help="the whole target for this run, instead of camera.hand_eye's: "
                            "aruco:ID:SIZE_MM[:DICT] for one marker, or "
                            "charuco:XxY:SQUARE_MM:MARKER_MM[:DICT][:legacy] for a ChArUco board (MARKER_MM is the "
                            "marker inside a square; legacy is the layout OpenCV generated before 4.6). A dictionary "
                            "left out is camera.hand_eye's. Not with --marker-length-mm, --marker-id or --dict")
    board.add_argument("--board-file", dest="board_file", default=UNSET, metavar="PATH",
                       help="the same, read from a YAML or JSON file of one target mapping, such as "
                            "{kind: charuco, squares_x: 7, squares_y: 5, square_length_mm: 30, marker_length_mm: 22}")
    ap.add_argument("--out", default=UNSET,
                    help=f"directory for the artifact + dataset (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--check", action="store_true",
                    help="validate config and the rig, touch no hardware, exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the arm and open the camera, then stop before any motion")
    ap.add_argument("--profile", default=None, help="WILLY_PROFILE chain for this run")
    ap.add_argument("--data-dir", default=None, help="override the config tree root")
    ap.add_argument("--unmodelled-wrist-body", dest="unmodelled_wrist_body", default=UNSET, metavar="REASON",
                    help="sweep a wrist camera whose declared body cannot be placed yet (no calibration, no record, "
                         "or a stale one), without its body in the planner and the guard, and say why. Printed and "
                         "logged; refused without a reason")
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None,
                    help="a window beside the sweep: the camera while the arm moves, each judged frame with the "
                         "target drawn on it, and whether the pose counted. Default: open where OpenCV has a GUI, a "
                         "display is there and stdout is a terminal. WILLY_NO_PREVIEW=1 keeps it shut. Closing it "
                         "closes the window only; the pendant stops the robot")
    return ap


def _aim(args: argparse.Namespace) -> Any:
    """The ``MarkerAim`` the flags describe, ``UNSET`` without ``--aim-at``; ``ValueError`` names a bad one."""
    shaped = (("--aim-distance-mm", args.aim_distance_mm), ("--closing-axis", args.closing_axis))
    extras = [flag for flag, value in shaped if value is not UNSET]
    if args.aim_at is UNSET:
        if extras:
            raise ValueError(f"{' and '.join(extras)} shape an aimed sweep, and there is none without --aim-at")
        return UNSET
    from src.robot.execution.camera_aim import MarkerAim

    parts = [part.strip() for part in str(args.aim_at).split(",")]
    try:
        point = tuple(float(part) for part in parts)
    except ValueError:
        point = ()
    if len(point) != 3:
        raise ValueError(f"--aim-at is the marker's centre as X,Y,Z in base millimetres, not {args.aim_at!r}")
    shape: dict[str, Any] = {}
    if args.aim_distance_mm is not UNSET:
        shape["distance_mm"] = args.aim_distance_mm
    if args.closing_axis is not UNSET:
        shape["closing_axis"] = args.closing_axis
    return MarkerAim(marker_mm=point, **shape)  # type: ignore[arg-type]


def _narrate(stage: "ConnectStage") -> None:
    """The bench wording. `lifecycle` owns the order; this file owns how it reads."""
    from src.robot.execution.lifecycle import ConnectStage

    if stage is ConnectStage.ARM_CONNECTED:
        print("  arm connected, and no gripper: the sweep drives the arm alone", flush=True)


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- 1. Config ------------------------------------------------------------------------------
    print(CalibrationStage.CONFIG.banner(), flush=True)
    try:
        cfg = _load(args.profile, args.data_dir)
    # `ConfigError` is caught: the config load raises it for a broken or unvalidatable tree, the
    # failure an operator is most likely to hit, and it must reach the terminal as the refusal
    # written here rather than as a traceback. A refused rig is not an exception: it is the check
    # report printed below.
    except ConfigError as exc:
        print(f"[config] REFUSED: {exc}", flush=True)
        return _EXIT_CONFIG

    def _built(build: "CalibrationBuild") -> None:
        """What the arm will refuse, printed before it moves, and the banner of the sweep that follows."""
        print(build.render(), flush=True)
        if build.ok and not args.dry_run:
            print(f"\n{CalibrationStage.SWEEP.banner()}", flush=True)

    # A board file is read and validated by the check, like a spec, so `--check` refuses a bad one.
    target = args.board if args.board_file is UNSET else Path(args.board_file)
    try:
        aim = _aim(args)
    except ValueError as exc:
        print(f"[config] REFUSED: {exc}", flush=True)
        return _EXIT_CONFIG
    calibration = HandEyeCalibration.from_config(
        cfg, rig_id=args.rig, mode=args.mode, data_dir=args.data_dir,
        options=SweepOptions(poses=args.poses, fixed_poses=args.fixed_poses, marker_length_mm=args.marker_length_mm,
                             marker_id=args.marker_id, dict_name=args.dict_name, out_dir=args.out,
                             unmodelled_wrist_body=args.unmodelled_wrist_body, target=target, aim=aim,
                             preview="auto" if args.preview is None else bool(args.preview)),
        announce=_narrate, on_event=print_sweep_progress, on_built=_built,
    )
    check = calibration.check()
    print(check.render(), flush=True)
    if not check.ok:
        return _EXIT_CONFIG
    if args.check:
        print("\n--check: the config and the rig are usable. Nothing was touched.", flush=True)
        return _EXIT_OK

    # ---- 2. Build, 3. Sweep, 4. Result ------------------------------------------------------------
    print(f"\n{CalibrationStage.BUILD.banner()}", flush=True)
    report = calibration.run(dry_run=args.dry_run)
    tail = report.summary()
    if tail:
        print(tail, flush=True)
    if report.rig_block:
        # The block is YAML text ending in a newline, printed as it will be pasted.
        print(flush=True)
        print(report.rig_block, flush=True)
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
