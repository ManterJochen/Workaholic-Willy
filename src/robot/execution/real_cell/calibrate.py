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

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

_EXIT_OK, _EXIT_CONFIG, _EXIT_NO_ARTIFACT, _EXIT_ERROR = 0, 1, 2, 3

#: Where an artifact lands unless `--out` says otherwise. Mirrors the sim runner's layout so a cell
#: that was brought up in sim and then on hardware keeps one place to look.
_DEFAULT_OUT = "calibration/real"


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


def _pick_rig(camera_cfg: Any, rig_id: str) -> Any:
    """The named rig, or a refusal that lists what there is.

    Refuses a non-RGB-D rig by name. A stereo pair calibrates through the routine's own
    stereo-rectified path, not through this one, and letting a webcam rig through here would
    fail much later with an error about ArUco rather than about the rig.

    It now reads `enabled`. Measured 2026-09-10: nothing on this path did. `--check` printed
    "the config and the rig are usable" for a rig the cell does not run, and the only thing
    between that sentence and 22 commanded poses in front of a camera nobody switched on was
    the operator reading `enabled: false` out of their own YAML. `CameraSystemConfig`
    deliberately does not refuse a disabled rig at load time, because a profile that is not
    ready to run is not a malformed file, and it says in that comment that the refusal belongs
    where the cell is built. This is that place.

    The "rigs is empty" refusal was deleted rather than moved. `CameraSystemConfig._validate_rigs`
    raises "at least one camera rig must be configured" during `load_config`, and `main` is the
    only caller, passing `cfg.camera` from exactly that loader. So the branch could fire for a
    hand-built object and nothing else, and its only witness was the test written to cover it.
    An empty list still refuses here, one line down, naming the rig it could not find.

    The refusals are the camera noun's (`select_rig`), so this runner and the cell say one thing
    about a rig. The sentence about the artifact is added to the disabled refusal, because that is
    the case an operator is most likely to meet here.
    """
    from src.camera.orchestration.camera import CameraRefused, select_rig

    try:
        return select_rig(camera_cfg, rig_id=rig_id)
    except CameraRefused as refused:
        tail = (" A sweep against a camera the cell will not open produces an artifact nothing consumes."
                if refused.reason == "disabled" else "")
        raise SystemExit(f"{refused}{tail}") from refused


def _snippet(camera_id: str, mode: str, path: str) -> str:
    """The rig block an operator pastes into the camera section so this camera's calibration is read.

    Writing the artifact is only half the job. Until the rig declares it, the cell has no CAMERA to
    BASE for this camera. A fixed camera's block is complete as printed. A wrist camera's shutter
    motion tolerances are a fact of the cell that is not measured here, so its block names them as
    comments to fill in, and the loader refuses the block until they are written.
    """
    tolerances = "" if mode == "eye_to_hand" else (
        "          # shutter_motion_tolerance_mm: <measure: how far the tool may travel while a frame is taken>\n"
        "          # shutter_motion_tolerance_deg: <measure: how far the tool may turn while a frame is taken>\n"
    )
    return (
        "camera:\n"
        "  cameras:\n"
        "    rigs:\n"
        f"      - rig_id: {camera_id}\n"
        "        extrinsics:\n"
        f"          mounting_mode: {mode}\n"
        f"          artifact_path: {path}\n"
        + tolerances
    )


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--poses", type=int, default=22,
                    help="how many generated TCP poses to visit (default 22, as in sim)")
    ap.add_argument("--marker-length-mm", type=float, default=None,
                    help="printed ArUco square edge in mm. Default: camera.hand_eye's. A wrong "
                         "value scales every sample uniformly; the solve converges and is uniformly "
                         "wrong. Measure the printed board")
    ap.add_argument("--marker-id", type=int, default=0, help="the ArUco id to pose (default 0)")
    ap.add_argument("--dict", dest="dict_name", default="DICT_5X5_100", help="ArUco dictionary")
    ap.add_argument("--out", default=None,
                    help=f"directory for the artifact + dataset (default {_DEFAULT_OUT})")
    ap.add_argument("--check", action="store_true",
                    help="validate config and the rig, touch no hardware, exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the arm and open the camera, then stop before any motion")
    ap.add_argument("--profile", default=None, help="WILLY_PROFILE chain for this run")
    ap.add_argument("--data-dir", default=None, help="override the config tree root")
    return ap


def main(argv: "list[str] | None" = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    args = build_parser().parse_args(argv)
    out_dir = args.out or _DEFAULT_OUT

    # ---- 1. Config ------------------------------------------------------------------------------
    print("=== 1. CONFIG ===", flush=True)
    try:
        cfg = _load(args.profile, args.data_dir)
        robot_cfg: RobotConfig = cfg.robot
        rig = _pick_rig(cfg.camera, args.rig)
    # Both exception types are caught. The config load raises `ConfigError` for a broken or
    # unvalidatable tree, the failure an operator is most likely to hit, and it must reach the
    # terminal as the refusal written here rather than as a traceback. `SystemExit` stays in
    # the tuple because other refusals on this path raise it.
    except (SystemExit, ConfigError) as exc:
        print(f"[config] REFUSED: {exc}", flush=True)
        return _EXIT_CONFIG
    settings = cfg.camera.hand_eye.eye_to_hand
    marker_mm = args.marker_length_mm or float(getattr(settings, "marker_length_mm", 50.0))
    print(f"  rig        {rig.rig_id!r} ({rig.source})", flush=True)
    print(f"  mode       {args.mode}", flush=True)
    print(f"  arm        {robot_cfg.vendor}", flush=True)
    print(f"  marker     {marker_mm:.1f} mm, id {args.marker_id}, {args.dict_name}", flush=True)
    print(f"  poses      {args.poses}", flush=True)
    print(f"  artifact   {out_dir}/{'eth' if args.mode == 'eye_to_hand' else 'eih'}_{rig.rig_id}.json",
          flush=True)
    if args.check:
        print("\n--check: the config and the rig are usable. Nothing was touched.", flush=True)
        return _EXIT_OK

    # ---- 2. Build -------------------------------------------------------------------------------
    print("\n=== 2. BUILD === the arm alone, and one camera", flush=True)
    handle = None
    try:
        import numpy as np

        from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
        from src.camera.orchestration.camera import Camera
        from src.calibration.eye_hand import MountingMode
        from src.robot.execution.calibration import CalibrationRoutine
        from src.robot.execution.robot import Robot

        # The arm through the builder a pick run uses, so the arm-vendor readiness gate runs first.
        # `gripper=None` builds no gripper: a sweep needs none, and one this tree cannot build is
        # not a reason to refuse calibrating a camera.
        robot = Robot.from_config(robot_cfg, gripper=None)
        # Through the camera noun, and it opens exactly one rig. A calibration sweep that claimed
        # every configured camera would fight the console for devices it does not need, and a rig
        # the console already holds is refused here naming the holder.
        camera = Camera.from_config(cfg.camera, rig_id=rig.rig_id)
        handle = camera.handle()
        camera.open()
        marker_source = RGBDArucoMarkerSource(
            streamer=handle, marker_length_mm=marker_mm,
            dict_name=args.dict_name, target_id=args.marker_id,
        )
        calibration = robot_cfg.calibration
        routine = CalibrationRoutine(
            arm=robot.arm, marker_source=marker_source,
            workspace_limits=robot_cfg.workspace_limits, eth_settings=settings,
            rig_id=rig.rig_id, marker_id=args.marker_id,
            settle_time_s=calibration.settle_time_s,
            calibration_mode=(MountingMode.EYE_TO_HAND if args.mode == "eye_to_hand"
                              else MountingMode.EYE_IN_HAND),
            on_event=lambda evt, data: print(
                f"  [{evt}] {data.get('reason', data.get('accepted', ''))}", flush=True),
        )
        gripper = ("none, the sweep connects the arm alone" if robot.gripper is None
                   else type(robot.gripper).__name__)
        print(f"  arm      {type(robot.arm).__name__}", flush=True)
        print(f"  gripper  {gripper}", flush=True)
        print(f"  lock     {robot.lock_key or 'none, this arm drives no controller of its own'}",
              flush=True)
        print(f"  camera   {handle!r} intrinsics={'yes' if handle.get_intrinsics() is not None else 'NO'}",
              flush=True)
    except Exception as exc:  # noqa: BLE001 (a build refusal is the designed outcome)
        print(f"[build] REFUSED: {type(exc).__name__}: {exc}", flush=True)
        if handle is not None:
            handle.release()
        return _EXIT_CONFIG
    # What this arm will refuse, read off the built arm and printed above the --dry-run exit,
    # because that is the question a dry run is asked.
    print(robot.safety().render(), flush=True)

    if args.dry_run:
        print("\n--dry-run: built cleanly and the camera answered. Stopping before any motion.",
              flush=True)
        handle.release()
        return _EXIT_OK

    # ---- 3. Sweep -------------------------------------------------------------------------------
    print("\n=== 3. SWEEP === the robot moves now, keep hands clear", flush=True)
    from pathlib import Path

    from src.robot.execution.cell_lock import CellBusy
    from src.robot.execution.lifecycle import ConnectStage

    def _narrate(stage: ConnectStage) -> None:
        """The bench wording. `lifecycle` owns the order; this file owns how it reads."""
        if stage is ConnectStage.ARM_CONNECTED:
            print("  arm connected, and no gripper: the sweep drives the arm alone", flush=True)

    # The lock, the connect and the teardown are `Robot.connected()`, the enter and the exit a pick
    # run uses, so this file writes no connect or disconnect of its own. A refused connect is
    # answered before the sweep starts; a sweep that raises is reported once the arm is down.
    failure: Exception | None = None
    try:
        with robot.connected(announce=_narrate) as live:
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                result = routine.run_auto(
                    args.poses,
                    # Tool down, so a board lying face-up on the table faces the fixed camera. The
                    # spread is the config's, floored at 30 deg: a planar ArUco viewed
                    # near-frontally has an IPPE flip ambiguity that ruins the AX=XB rotation, and a
                    # narrow spread never breaks it.
                    base_orientation=[float(np.pi), 0.0, 0.0],
                    orientation_spread_deg=max(30.0, float(calibration.orientation_spread_deg)),
                    max_attempts_per_pose=calibration.max_attempts_per_pose,
                    seed=0,
                    dataset_save_path=f"{out_dir}/{args.mode}_{rig.rig_id}_dataset.json",
                )
            except Exception as exc:  # noqa: BLE001
                # Reported below, once the arm is down.
                failure = exc
    except CellBusy as exc:
        # Another process holds this controller, and nothing was commanded.
        print(f"[connect] REFUSED: {exc}", flush=True)
        return _EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001
        # Connect is a transaction, and it has already rolled the arm back and given up the lock.
        print(f"[connect] FAILED: {type(exc).__name__}: {exc}", flush=True)
        return _EXIT_CONFIG
    finally:
        # After the block, so the arm is down and the lock given back before the camera is.
        handle.release()

    print("\n  down: the arm, then the lock, then the camera", flush=True)
    if live.teardown is not None:
        print(live.teardown.render(), flush=True)
    if failure is not None:
        print(f"[sweep] FAILED: {type(failure).__name__}: {failure}", flush=True)
        return _EXIT_ERROR

    # ---- 4. Result ------------------------------------------------------------------------------
    from src.calibration.quality import QualityBandsMm, classify_rmse

    # The config carries its own bands model; `classify_rmse` takes the calibration layer's. The
    # two hold the same three numbers, and the conversion is written out so that a future field on
    # either side fails here rather than at a band boundary nobody reads.
    _bands = calibration.quality_bands_mm
    quality = str(classify_rmse(result.rmse_mm, QualityBandsMm(
        excellent=float(_bands.excellent), good=float(_bands.good), marginal=float(_bands.marginal))))
    print("\n=== 4. RESULT ===", flush=True)
    print(f"  accepted samples  {result.num_samples}/{args.poses}", flush=True)
    print(f"  AX=XB rmse        {result.rmse_mm:.4f} mm (max {result.max_error_mm:.4f}) -> {quality}",
          flush=True)

    #
    # The guard covers both modes. An eye-in-hand run whose solver produced no `Transform` would
    # otherwise reach `save_cam_to_tool(..., None)`: nothing downstream catches that, and the
    # artifact is simply wrong.
    carrier = result.extrinsics if args.mode == "eye_to_hand" else result.transform
    if carrier is None:
        # Loud, because the failure mode is silence. With no carrier nothing is written, the cell
        # keeps whatever calibration it had, and the run otherwise looks like it finished.
        kind = "Extrinsics" if args.mode == "eye_to_hand" else "Transform"
        print(f"\n[result] no {kind} carrier; nothing was written. This camera keeps its "
              "previous calibration, if it had one.", flush=True)
        return _EXIT_NO_ARTIFACT

    import dataclasses

    from src.calibration.serialization import save_cam_to_tool, save_extrinsics

    if args.mode == "eye_to_hand":
        # rig_id is stamped to the camera id so the artifact and the rig that declares it line up.
        assert result.extrinsics is not None                    # guarded by `carrier` above
        extrinsics = dataclasses.replace(result.extrinsics, rig_id=rig.rig_id)
        written = save_extrinsics(f"{out_dir}/eth_{rig.rig_id}.json", extrinsics)
    else:
        assert result.transform is not None                     # guarded by `carrier` above
        written = save_cam_to_tool(f"{out_dir}/eih_{rig.rig_id}.json", result.transform,
                                   rig_id=rig.rig_id)
    print(f"  written           {written}", flush=True)
    print(f"  dataset           {result.dataset_path}", flush=True)
    print("\nPaste this into the camera section so the camera reaches the pick path. Until its rig\n"
          "declares it, the cell has no CAMERA->BASE for this camera:\n", flush=True)
    print(_snippet(rig.rig_id, args.mode, str(written)), flush=True)
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
