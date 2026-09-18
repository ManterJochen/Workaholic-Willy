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

This moves the robot. `run_auto` drives the arm to N generated poses. `--check` validates everything
and touches nothing. `--dry-run` additionally runs the arm-vendor readiness gate, builds the arm,
opens the camera and prints what the arm's safety pipeline refuses, but takes no lock and never
commands a motion. Run both before the first live sweep.

The sweep connects through `Robot.connected()`. The cell lock comes first, the lock a pick run and
the operator console take for the same controller, so a sweep is refused while either holds it and
is told who does. Then the arm, and no gripper: a sweep beside a board needs no activation stroke,
and a gripper this tree cannot build must not stand in the way of calibrating a camera. On the way
out the arm comes down, the lock is given back, and then the camera. Every move of the sweep
declines the camera world, because the sweep is what produces the transform a camera world needs.

The marker length and the ArUco dictionary default to the mode's `camera.hand_eye` block, and a
wrist sweep reads `camera.hand_eye.eye_in_hand`. On every shipped tree the two blocks agree and the
dictionary is `DICT_5X5_100`.

The routine and the solve are exercised in simulation only. The ArUco marker source has never seen
a physical D435, and nothing here has run against a physical controller.

Exit codes: 0 success; 1 configuration refused, build refused, the cell held by another process, or
the connect refused; 2 the calibration ran but did not produce an artifact; 3 the sweep raised.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from src.config.loader import ConfigError
from src.contracts import UNSET
from src.robot.execution.hand_eye import (
    DEFAULT_OUT_DIR,
    CalibrationStage,
    HandEyeCalibration,
    SweepOptions,
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
    ap.add_argument("--poses", type=int, default=UNSET,
                    help="how many generated TCP poses to visit (default 22, as in sim)")
    ap.add_argument("--marker-length-mm", type=float, default=UNSET,
                    help="printed ArUco square edge in mm. Default: camera.hand_eye's. A wrong "
                         "value scales every sample uniformly; the solve converges and is uniformly "
                         "wrong. Measure the printed board")
    ap.add_argument("--marker-id", type=int, default=UNSET, help="the ArUco id to pose (default 0)")
    ap.add_argument("--dict", dest="dict_name", default=UNSET,
                    help="ArUco dictionary. Default: camera.hand_eye's, for the mode")
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
    return ap


def _narrate(stage: "ConnectStage") -> None:
    """The bench wording. `lifecycle` owns the order; this file owns how it reads."""
    from src.robot.execution.lifecycle import ConnectStage

    if stage is ConnectStage.ARM_CONNECTED:
        print("  arm connected, and no gripper: the sweep drives the arm alone", flush=True)


def _progress(event_type: str, data: dict[str, Any]) -> None:
    """One line per pose of the sweep, as the routine emits them."""
    print(f"  [{event_type}] {data.get('reason', data.get('accepted', ''))}", flush=True)


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

    calibration = HandEyeCalibration.from_config(
        cfg, rig_id=args.rig, mode=args.mode, data_dir=args.data_dir,
        options=SweepOptions(poses=args.poses, marker_length_mm=args.marker_length_mm, marker_id=args.marker_id,
                             dict_name=args.dict_name, out_dir=args.out,
                             unmodelled_wrist_body=args.unmodelled_wrist_body),
        announce=_narrate, on_event=_progress, on_built=_built,
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
