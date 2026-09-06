"""Smoke-exercise the live-camera PerceptionSource against a real RGB-D camera. No robot, no motion.

    (S) Needs a real Intel RealSense (D435/D435i) + `pip install -r requirements.txt`
        (`requirements-cpu.txt` on a CPU box; both pin pyrealsense2) + the perception models
        (GroundingDINO + SAM2, on the GPU box). Bucket (3).

    python -m src.robot.perception --prompt "a red cube"
    python -m src.robot.perception --prompt "cube ; screwdriver ; mug"   # multi-phrase clutter
    python -m src.robot.perception --rig realsense_d435 --warmup 10

This is the smallest proof that the whole vision front-end works end to end on a physical camera
before any of it is wired to a robot: open the RGB-D rig, grab a frame, run GroundingDINO + SAM2,
and print what came back, meaning intrinsics, per-object mask pixel counts, and (the real-hardware
risk) how much of the depth inside each mask is holes. A high hole fraction means the grasp-depth
top-referencing is sampling very little real surface, which is a bench-tuning signal, not a code
bug.

The hole fraction is counted on the depth the adapter consumed, kept by `_RawDepthTap`, and not on
`frame.depth_map`: `acquire()` overwrites the depth inside every mask with one constant, so the same
count taken there is 0.0 % whenever a mask holds any real depth and 100.0 % when it holds none. The
grasp plane is printed beside it, so the operator sees both the sensor's answer and the number the
robot would drive to.

The hole fraction is measured on the streamer's output, so it is post-filter. Turning on
`camera.cameras.rigs.realsense.post_processing.hole_filling` makes a blind camera report ~0 % here,
honestly, because the filter really did fill them.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from src.config.schema.camera import RGBDDeviceRigConfig


def _find_rgbd_rig(camera_cfg: Any, rig_id: str | None) -> RGBDDeviceRigConfig:
    """Pick the RGB-D rig to open: the named one, or the single one if unambiguous.

    Selects on the same key as the cell: ``build_real_components`` in
    ``robot/execution/autonomous_grasp/cells.py`` filters on ``source == "rgbd"``. A bench run that
    opened a different camera from the one the cell opens would be no evidence about the cell.

    ``source`` is the discriminated union's own tag (``RGBDDeviceRigConfig.source``), so
    ``source == "rgbd"`` means exactly "this rig is an RGBDDeviceRigConfig", which is what the cast
    below asserts. ``RGBDDeviceRigConfig.rgbd_backend`` is the driver (default ``"opencv"``) and a
    strictly narrower filter; selecting on it hides the misconfiguration this exerciser exists to
    find, because a rig on the ``opencv`` backend returns colour with an empty depth channel and the
    depth-hole report is the instrument for exactly that.

    Neither this predicate nor the cell's honours ``enabled``, while the schema's own two-camera
    validator (``CameraSystemConfig._validate_rigs``) uses ``source == "rgbd" and r.enabled``. Three
    predicates, and the strictest is the one that reasons about safety. Closing that is an owner
    decision rather than a line of code.
    """
    rigs = list(getattr(getattr(camera_cfg, "cameras", None), "rigs", []) or [])
    rgbd = [r for r in rigs if getattr(r, "source", None) == "rgbd"]
    if rig_id is not None:
        for r in rgbd:
            if getattr(r, "rig_id", None) == rig_id:
                return cast("RGBDDeviceRigConfig", r)
        raise SystemExit(
            f"no RGB-D rig with rig_id={rig_id!r}; rgbd rigs: {[getattr(r, 'rig_id', '?') for r in rgbd]}"
        )
    if len(rgbd) == 1:
        return cast("RGBDDeviceRigConfig", rgbd[0])
    raise SystemExit(
        f"found {len(rgbd)} RGB-D rigs; pass --rig <rig_id>. "
        f"candidates: {[getattr(r, 'rig_id', '?') for r in rgbd]}"
    )


class _RawDepthTap:
    """Delegates to the rig handle and keeps a copy of the depth of the last frame it passed through.

    ``acquire()`` overwrites the depth inside every mask with one scalar
    (``depth_mm = np.where(mask, grasp_depth_mm, depth_mm)``, guarded by ``if vals.size:``), so a
    hole count taken on the returned depth can only be 0 %, when the mask held even one real depth
    pixel, or 100 %, when the mask was all holes and the overwrite was skipped. Holes inside objects
    are erased before they can be counted. This tap keeps the depth as the streamer delivered it,
    which is the array that count has to be taken on.

    A tap rather than a second grab before ``acquire()``: a separate grab reads a different frame
    from the one the masks were cut from, and reads it before the warmup grabs have let
    auto-exposure settle. Against a truth of 15 hole pixels a pre-grab reports 10 and the tap
    reports 15. The tap also costs no extra device frame, which matters on a stateful RealSense
    filter chain.

    The tapped depth is the streamer's output, so it is post-filter.
    ``camera.cameras.rigs.realsense`` ships ``hole_filling: false``; turning it on makes a blind
    camera report ~0 % holes here, honestly, because the filter really did fill them. Which filters
    are on is not printed by this CLI: the RealSense streamer logs their count when it opens the
    rig, and the chain itself is set in ``camera.cameras.rigs.realsense.post_processing``.
    """

    def __init__(self, handle: Any) -> None:
        # `object.__setattr__` and a direct `__dict__` read below: `__getattr__` fires for any miss,
        # so a wrapper that reads `self._handle` through the normal path recurses forever if the
        # attribute is not there yet.
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "last_depth_mm", None)

    def grab(self) -> Any:
        frame = self.__dict__["_handle"].grab()
        depth = getattr(frame, "depth", None)
        if depth is not None:
            object.__setattr__(
                self, "last_depth_mm", np.asarray(depth, dtype=np.float64).copy())
        return frame

    def __getattr__(self, name: str) -> Any:
        # Everything else (get_intrinsics, get_distortion, release, is_open) passes straight
        # through, because the adapter and the calibration code duck-type the full handle surface.
        return getattr(self.__dict__["_handle"], name)


def _backend_note(rig: Any) -> str:
    """Say it up front when the selected rig is not on the RealSense driver.

    Selecting on ``source == "rgbd"`` can pick a rig the generic OpenCV path drives, which returns
    an empty depth channel on a device whose depth lives behind its vendor SDK. Unannounced, that
    surfaces three frames later as a RuntimeError about intrinsics. Finding it is what this tool is
    for, so it is announced, not refused.
    """
    backend = getattr(rig, "rgbd_backend", None)
    if backend == "realsense":
        return ""
    return (
        f"  rgbd_backend={backend!r}: this rig is driven by the generic OpenCV path, which "
        f"returns an empty depth channel on a device whose depth only exists behind its vendor SDK. "
        f"Set rgbd_backend: realsense if this is a RealSense."
    )


def _print_frame_health(frame: Any, tap: Any, *, backend_note: str) -> None:
    """Depth health for one acquired frame, counted on the sensor depth, not the adapter's."""
    raw = tap.last_depth_mm
    rendered = np.asarray(frame.depth_map)
    if raw is None or raw.size == 0:
        # An RGBDFrame may legally carry an empty depth array, and a percentage of nothing is not a
        # number, so this branch reports the empty channel instead of a hole fraction.
        print("depth: empty; the streamer returned no depth channel, so no hole fraction exists."
              + (" " + backend_note.strip() if backend_note else ""))
    else:
        holes = int((raw == 0).sum())
        real = raw[raw > 0]
        lo = float(real.min()) if real.size else 0.0
        print(f"depth {raw.shape}: {holes}/{raw.size} holes ({100.0 * holes / raw.size:.1f}%), "
              f"non-hole range [{lo:.0f}, {float(raw.max()):.0f}] mm  (sensor, post-filter)")
    print(f"segmentations: {len(frame.segmentations)}")
    all_hole = 0
    for i, seg in enumerate(frame.segmentations):
        mask = np.asarray(seg.mask).astype(bool)
        label = getattr(seg, "label", "?")
        n = int(mask.sum())
        if not n:
            # An empty mask is a segmenter miss, and calling it 100 % holes would hand the operator
            # a camera fault to chase.
            print(f"  [{i}] label={label!r} mask_px=0 (empty)")
            continue
        if raw is None or raw.shape != mask.shape:
            print(f"  [{i}] label={label!r} mask_px={n} depth-holes-in-mask=? (no sensor depth)")
            continue
        holes = int((raw[mask] == 0).sum())
        if holes == n:
            all_hole += 1
        # Both numbers, side by side: what the sensor said, and the plane the robot would drive to.
        # The second is the adapter's `min(real surface) + grasp_top_penetration_mm`.
        plane = rendered[mask]
        grasp = f"{float(plane.max()):.0f}" if plane.size else "?"
        print(f"  [{i}] label={label!r} mask_px={n} "
              f"depth-holes-in-mask={holes}/{n} ({100.0 * holes / n:.1f}%) "
              f"grasp-plane={grasp} mm")
    if all_hole:
        # The case the exit code is blind to, said out loud. A D435 on a specular or dark scene
        # returns lit RGB, so grounding succeeds, and dead depth; the adapter deliberately leaves
        # those zeros in place, so the grasp depth is referenced to nothing. This CLI still exits 0.
        print(f"  {all_hole} of {len(frame.segmentations)} mask(s) have no real depth at all. "
              f"The grasp depth for those is not referenced to any surface.")
    if not frame.segmentations:
        print("  (nothing grounded; check the prompt, the lighting, and that the object is in view)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grab one RGB-D frame, detect+segment, print stats. No robot.")
    ap.add_argument("--prompt", default="an object", help="GroundingDINO phrase(s); ' ; '-separate for clutter")
    # The default is the only rig whose `source` is `rgbd`, which is what the cell selects, so the
    # help string names RGB-D rather than the RealSense driver.
    ap.add_argument("--rig", default=None, help="rig_id of the RGB-D rig (default: the only RGB-D rig)")
    ap.add_argument("--warmup", type=int, default=5, help="throwaway grabs so auto-exposure settles")
    args = ap.parse_args(argv)

    # Deferred: config is light, but the streamer pulls pyrealsense2 and the factory pulls torch.
    from src.config import load_config
    from src.camera.orchestration.frame_provider import FrameProvider
    from src.models.factory import build_object_detector, build_segmenter
    from src.robot.perception import RealSenseVisionPerceptionSource

    cfg = load_config()
    rig = _find_rgbd_rig(cfg.camera, args.rig)
    # Through the frame provider, exactly as `build_real_components` does. This exerciser exists to
    # prove the same chain the cell runs before a robot is involved: if the two acquire their frames
    # differently, a green bench run stops being evidence about the cell.
    provider = FrameProvider(list(cfg.camera.cameras.rigs))
    handle = provider.rig(rig.rig_id)
    # See `_RawDepthTap`: the adapter overwrites the depth inside every mask before returning the
    # frame, so counting holes on `frame.depth_map` yields 0.0 % or 100.0 % and nothing else. The
    # tap keeps the depth of the frame the masks were cut from, and costs no extra device frame.
    tap = _RawDepthTap(handle)
    backend_note = _backend_note(rig)
    detector = build_object_detector(cfg.models)
    segmenter = build_segmenter(cfg.models)
    source = RealSenseVisionPerceptionSource(
        streamer=tap, detector=detector, segmenter=segmenter,
        prompt=args.prompt, warmup_grabs=args.warmup,
    )

    print(f"opening rig {rig.rig_id!r} ...", flush=True)
    if backend_note:
        print(backend_note, flush=True)
    provider.open_rig(rig.rig_id)
    try:
        frame = source.acquire()
    finally:
        handle.release()

    print(f"intrinsics ({source.intrinsics_source}):\n{np.asarray(frame.intrinsics)}")
    _print_frame_health(frame, tap, backend_note=backend_note)
    print("OK: grabbed, detected, segmented. No robot touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
