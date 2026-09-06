"""22: a stereo pair against an RGB-D rig, and which of them a pick can actually be planned from.

    python scripts/examples/perception/22_depth_source.py            # rehearse: read, open nothing
    python scripts/examples/perception/22_depth_source.py --live      # open the rig and grab
    python scripts/examples/perception/22_depth_source.py --live --rig cam_left

This one needs the camera. Without `--live` it reads `camera.cameras.rigs`, reports what each rig
would deliver, and touches no device; with `--live` it opens one rig, grabs a frame and prints what
came back. No robot is involved either way, and nothing here commands a motion.

The decision is `source` on each rig in `config/camera/cam.yaml`, and it is a discriminated union:
`source` picks the rig variant and therefore which fields that rig even accepts.

  webcam_pair, single_device   a stereo pair. Two views, no depth in the frame. Depth is computed
                               by block matching, which needs a recorded calibration: a ChArUco
                               board, enough image pairs, and the `stereoMap.xml` the solve writes.
                               The rig hands back a `StereoFrame` and no camera matrix; the
                               geometry lives in `StereoCam3D` instead of in one pinhole K.
  rgbd                         a depth camera. One `RGBDFrame` per grab, colour plus uint16
                               millimetre depth on one pixel grid, and a camera matrix the device
                               reports. Nothing to calibrate before it measures.

The pick path takes the second and only the second. `build_real_components` selects the cell's
camera as the first rig whose `source` is `rgbd`, and `RealSenseVisionPerceptionSource` needs a
frame with a depth channel and a `get_intrinsics()` that answers. A stereo rig satisfies neither,
so it calibrates, it measures 3-D points, it feeds hand detection, and it never reaches a grasp.

Two traps in that selection, both live on the shipped tree.

`enabled` is not consulted. The cell takes the first rig with `source: rgbd` whatever its
`enabled` says, so a leftover rig in the list is a camera the cell might open, and the enabled
stereo pair next to it is not the camera the cell will use.

`rgbd_backend` defaults to `opencv`, and that default returns colour with an empty depth channel on
any device whose depth is reachable only through its vendor SDK, a RealSense included. It also
reports no intrinsics at all, so the cell hands the grasp calculator nothing and the calculator
refuses with `camera_matrix must be (3, 3), got ()`, a sentence that names neither the rig nor the
backend. Set `rgbd_backend: realsense` on a RealSense. The run below reads the value per rig and
says so.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, hardware_parser, not_ready  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import numpy as np

EPILOG = """
exit codes
  0  this tree names a depth source a pick can be planned from, and with --live it answered
  1  it names one that cannot deliver depth: an RGB-D rig on the generic backend
  2  nothing to read: no camera block, no RGB-D rig at all, or the device did not answer

--live opens a camera. It commands no motion; nothing in this file can move an arm.
"""


def _depth_promise(rig: Any) -> str:
    """What one configured rig would hand a consumer, read off `source` and `rgbd_backend`."""
    source = getattr(rig, "source", "?")
    if source != "rgbd":
        return ("StereoFrame(left, right), no depth channel and no camera matrix; depth needs a "
                "recorded stereo calibration and block matching")
    backend = getattr(rig, "rgbd_backend", "?")
    if backend == "realsense":
        return ("RGBDFrame(colour, uint16 mm depth) plus the intrinsics the device reports, "
                "through pyrealsense2")
    return (f"rgbd_backend={backend!r}: colour through OpenCV, an empty depth channel on any "
            f"device whose depth lives behind its vendor SDK, and no intrinsics")


def _stereo_ready(rig: Any) -> str:
    """Whether this stereo rig's calibration artifact exists yet, and where it would be.

    The configured path is relative, so it resolves against the process working directory, which
    is the repository root for every command in these examples. A run started elsewhere reports
    the artifact as missing while it sits where it always did.
    """
    paths = getattr(rig, "calibration_paths", None)
    stereo_map = getattr(paths, "stereo_map_file", "")
    if not stereo_map:
        return "no stereo_map_file configured"
    return f"{stereo_map}: {'present' if Path(stereo_map).exists() else 'not recorded yet'}"


def _depth_health(depth: "np.ndarray") -> str:
    """Hole fraction and the range of what is left, for one depth array as the streamer gave it.

    Counted here rather than after a perception source has run, because `acquire()` overwrites the
    depth inside every mask with the grasp plane, and the same count taken on its output can only
    come out at 0 or 100 percent. A zero is a hole: a surface the sensor could not measure, left
    honest rather than filled, because filling it invents geometry a grasp is then planned onto.
    """
    import numpy as np

    array = np.asarray(depth)
    if array.size == 0:
        return "empty: the streamer returned no depth channel at all"
    holes = int((array == 0).sum())
    real = array[array > 0]
    if real.size == 0:
        return f"{array.shape}: every one of {array.size} pixels is a hole"
    return (f"{array.shape}: {holes}/{array.size} holes "
            f"({100.0 * holes / array.size:.1f}%), the rest spans "
            f"[{float(real.min()):.0f}, {float(real.max()):.0f}] mm")


def main(argv: list[str] | None = None) -> int:
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument("--rig", default=None,
                        help="rig_id to open with --live. Default: the one a cell would open, "
                             "which is the first rig whose source is rgbd")
    args = parser.parse_args(argv)

    # After the parse, so --help answers in a checkout whose dependencies are not installed.
    from src.config import load_config

    with Example("22 depth_source", "stereo against RGB-D, and what each one delivers",
                 live=args.live, profile=args.profile) as run:
        # Said immediately, because the banner above is written for the examples that drive an arm.
        # This one opens a camera and commands nothing; the rehearsal it describes is a rehearsal
        # of the device, not of a motion.
        run.note("Nothing in this file can move a robot. Here `--live` means: open the camera.")
        run.note("")

        with run.step("load the camera tree") as report:
            cameras = load_config().camera.cameras
            rigs = list(cameras.rigs)
            rgbd = [rig for rig in rigs if getattr(rig, "source", None) == "rgbd"]
            report(f"{len(rigs)} rig(s), {len(rgbd)} of them RGB-D; "
                   f"active_mode {cameras.active_mode}")

        with run.step("what each rig would hand a consumer") as report:
            report(f"{len(rigs)} rig(s) described below")
        for rig in rigs:
            run.note(f"  {rig.rig_id} ({rig.source}, enabled={rig.enabled})")
            run.note(f"    {_depth_promise(rig)}")
            if getattr(rig, "source", None) == "rgbd":
                # Printed because it is the identity of the device, and on a two-camera cell it is
                # the only identity there is. A rig with none binds to whatever the SDK offers.
                serial = getattr(rig, "serial_number", None)
                run.note(f"    serial_number: {serial!r}"
                         + ("" if serial else "  (binds to the first device the SDK offers)"))
            else:
                run.note(f"    {_stereo_ready(rig)}")
            run.note("")

        if not rgbd:
            # The builder's own sentence, because an operator who meets it in a cell should meet
            # the same words here.
            return not_ready(
                "no RGB-D rig in camera.cameras.rigs; the real cell needs one",
                "enable the realsense rig in the camera config, and prove it standalone with "
                "`python -m src.robot.perception`")

        with run.step("which rig a cell would open") as report:
            # The cell's own predicate, restated nowhere: `source == "rgbd"`, first match wins,
            # `enabled` not consulted. Selecting on anything narrower here would mean a bench run
            # opening a different camera from the cell, which is not evidence about the cell.
            primary = rgbd[0]
            report(f"{primary.rig_id}, the first rig with source: rgbd")
        if not primary.enabled:
            run.finding("the cell's camera is disabled in config",
                        f"{primary.rig_id} has enabled=false and is still the rig a cell opens")
            run.note("  `enabled` is read by validation and by the stereo capture pipeline. The")
            run.note("  RGB-D selector on the pick path is not one of its readers, so a rig left")
            run.note("  in the list is a camera the cell might open.")
        enabled_stereo = [r.rig_id for r in rigs
                          if getattr(r, "source", None) != "rgbd" and r.enabled]
        if enabled_stereo:
            run.note(f"  Enabled stereo rigs here: {', '.join(enabled_stereo)}. None of them is")
            run.note("  reachable from a pick. No stereo rig satisfies the perception source's")
            run.note("  contract, and the selector above would not choose one in any case.")

        opencv_backed = getattr(primary, "rgbd_backend", None) != "realsense"
        if opencv_backed:
            run.finding("the cell's camera is on the generic backend",
                        f"{primary.rig_id} has rgbd_backend="
                        f"{getattr(primary, 'rgbd_backend', None)!r}")
            run.note("  That backend hands back an empty depth channel on a device whose depth is")
            run.note("  reachable only through its vendor SDK, and reports no intrinsics, so the")
            run.note("  grasp calculator refuses with `camera_matrix must be (3, 3), got ()`.")

        run.note("")
        run.note("What this decides downstream, on the RGB-D side:")
        run.note("  align_depth_to_color makes a colour pixel and its depth pixel the same pixel.")
        run.note("  Every mask read for depth assumes it, and only the realsense driver acts on")
        run.note("  it. The generic one stores the value and aligns nothing.")
        run.note("  hole_filling ships off. A hole is a surface the sensor could not measure, and")
        run.note("  the grasp depth is referenced to the nearest real surface over the mask, so a")
        run.note("  filled hole would be invented geometry with a grasp planned onto it.")
        run.note("  A serial is the only stable way to tell two identical cameras apart. Without")
        run.note("  one the SDK hands over whatever it offers first, an order that survives")
        run.note("  neither a reboot nor a replug, and two swapped cameras do not fail: they")
        run.note("  return a plausible scene with the views exchanged.")

        if not args.live:
            run.note("")
            run.note("Rehearsal stops here. Nothing was opened. `--live` adds: open one rig, grab")
            run.note("a frame, and print the depth the sensor actually delivered.")
            return EXIT_FAILED if opencv_backed else run.exit_code

        rig_id = args.rig or primary.rig_id
        target = next((rig for rig in rigs if rig.rig_id == rig_id), None)
        if target is None:
            return not_ready(f"this tree has no rig named {rig_id!r}",
                             f"pick one of: {', '.join(rig.rig_id for rig in rigs)}")
        # The two rig kinds fail to open for different reasons and are fixed at different ends of
        # the bench, so the fix line is chosen from the rig rather than written once for both.
        fix = ("is the camera plugged in? `rs-enumerate-devices -s` lists the RealSense serials "
               "on this bus, and this rig binds by serial, never by device_index"
               if getattr(target, "source", None) == "rgbd" else
               "are both cameras plugged in, and are cam_left_id and cam_right_id the OpenCV "
               "indices they actually enumerate at? Unset, the rig scans and takes the last two "
               "that answered")

        from src.camera.orchestration.frame_provider import FrameProvider

        # Through the frame provider, which is the class the cell uses: it owns rig identity and
        # per-rig lifecycle, and constructing a streamer touches no device, so knowing every rig
        # costs nothing while holding one is a deliberate act.
        provider = FrameProvider(rigs)
        with run.step(f"open {rig_id}") as report:
            try:
                provider.open_rig(rig_id)
            except Exception as error:                          # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the rig did not open ({type(error).__name__}: {error})", fix)
            report("streaming")

        try:
            with run.step("grab one frame") as report:
                frame = provider.grab(rig_id)
                report(type(frame).__name__)

            with run.step("the camera matrix this rig reports") as report:
                matrix = provider.get_intrinsics(rig_id)
                report("None: this rig has no single pinhole matrix, so no perception source can "
                       "unproject its frames" if matrix is None else
                       f"fx {float(matrix[0][0]):.1f}, fy {float(matrix[1][1]):.1f}, "
                       f"cx {float(matrix[0][2]):.1f}, cy {float(matrix[1][2]):.1f}")

            with run.step("the depth this rig delivered") as report:
                depth = getattr(frame, "depth", None)
                report(_depth_health(depth) if depth is not None else
                       "no depth attribute: this is a stereo frame, two views and nothing else")
        finally:
            provider.release_rig(rig_id)
            run.note("released.")

        run.note("")
        run.note("A frame is not a calibration. This rig now streams and reports a matrix; where")
        run.note("it stands in the robot's frame is still unknown, and the calibration examples,")
        run.note("10 through 12, are what settle that.")
        return EXIT_FAILED if opencv_backed else run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
