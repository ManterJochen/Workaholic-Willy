"""`python -m src.models.handdetection`: check the install, or locate a hand.

Three modes, in increasing order of what they need:

    --check                 nothing but the config. Reports whether mediapipe and each `.task`
                            bundle are present, and exits non-zero if not.
    --frame IMAGE           one image file. Detects hands and gestures in image space; needs no
                            camera, no calibration and no robot.
    --rig RIG_ID            live camera. Adds depth and, given `--transform`, the base frame.

The middle mode isolates the detector: with a camera, a calibration and a robot in the loop, "it
does not work" leaves an operator no way to tell which of the four is at fault.

Exit codes: 0 success, 1 nothing detected, 2 a setup problem (missing model, missing extra, bad
config). A missing hand and a missing model file must not look alike to a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import load_config
from src.models.handdetection.constants import (
    GESTURE_MODEL_URL,
    HAND_LANDMARK_MODEL_URL,
)
from src.models.handdetection.model_files import (
    MEDIAPIPE_AVAILABLE,
    MEDIAPIPE_INSTALL_HINT,
)

_OK, _NOTHING_FOUND, _SETUP_PROBLEM = 0, 1, 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.handdetection",
        description="MediaPipe hand + gesture detection: install check, single frame, or live rig.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report install state and exit")
    mode.add_argument("--frame", metavar="IMAGE", help="detect in one image file (no camera)")
    mode.add_argument("--rig", metavar="RIG_ID", help="detect on a live rig and locate in 3-D")
    parser.add_argument(
        "--transform", metavar="NPY",
        help="4x4 CAMERA->BASE matrix (.npy) for --rig. Without it the hand is reported in the "
             "CAMERA frame only, and said so.",
    )
    parser.add_argument(
        "--intrinsics", metavar="JSON",
        help="intrinsics.json for an RGB-D rig. Without it an RGB-D rig REFUSES rather than "
             "back-projecting through invented intrinsics.",
    )
    parser.add_argument(
        "--gestures", action="store_true",
        help="use the canned gesture model (thumbs up/down) instead of landmarks alone. It also "
             "returns the palm centre, so nothing is detected twice.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


# --- --check -------------------------------------------------------------------------------------


def _check(config: Any, as_json: bool) -> int:
    """Report what is installed without constructing anything that could raise."""
    report: dict[str, Any] = {
        "mediapipe_installed": MEDIAPIPE_AVAILABLE,
        "models": {},
    }
    for name, block, url in (
        ("handdetect", config.models.handdetect, HAND_LANDMARK_MODEL_URL),
        ("gesturedetect", config.models.gesturedetect, GESTURE_MODEL_URL),
    ):
        path = Path(block.model_path)
        resolved = path if path.is_absolute() else Path.cwd() / path
        report["models"][name] = {
            "config_key": f"models.{name}.model_path",
            "path": str(resolved),
            "present": resolved.is_file(),
            "download": url,
        }

    ready = report["mediapipe_installed"] and all(
        entry["present"] for entry in report["models"].values()
    )
    report["ready"] = ready

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"mediapipe installed : {'yes' if MEDIAPIPE_AVAILABLE else 'NO'}")
        if not MEDIAPIPE_AVAILABLE:
            print(f"                      {MEDIAPIPE_INSTALL_HINT}")
        for name, entry in report["models"].items():
            print(f"models.{name}:")
            print(f"  path    : {entry['path']}")
            print(f"  present : {'yes' if entry['present'] else 'NO'}")
            if not entry["present"]:
                print(f"  download: {entry['download']}")
        print(f"\nready: {'yes' if ready else 'NO'}")
    return _OK if ready else _SETUP_PROBLEM


# --- --frame -------------------------------------------------------------------------------------


def _observe_frame(config: Any, image_path: str, use_gestures: bool, as_json: bool) -> int:
    import cv2 as cv

    from src.models.handdetection.factory import (
        build_gesture_recognizer,
        build_palm_detector,
    )

    frame = cv.imread(image_path, cv.IMREAD_COLOR)
    if frame is None:
        print(f"could not read image: {image_path}", file=sys.stderr)
        return _SETUP_PROBLEM

    if use_gestures:
        with build_gesture_recognizer(config.models.gesturedetect) as recognizer:
            observations = recognizer.observe(frame)
    else:
        with build_palm_detector(config.models.handdetect) as detector:
            observations = detector.observe(frame)

    payload = [
        {
            "hand_index": item.palm.hand_index,
            "handedness": str(item.palm.handedness),
            "palm_center_xy": list(item.palm.palm_center_xy),
            "gesture": str(item.gesture.gesture),
            "gesture_confidence": round(item.gesture.confidence, 4),
            "gesture_raw_label": item.gesture.raw_label,
        }
        for item in observations
    ]
    if as_json:
        print(json.dumps({"hands": payload}, indent=2))
    elif not observations:
        print("no hands detected")
    else:
        for item in observations:
            centre_x, centre_y = item.palm.palm_center_xy
            raw = f", raw={item.gesture.raw_label}" if item.gesture.raw_label else ""
            print(
                f"hand {item.palm.hand_index} ({item.palm.handedness}): "
                f"palm at ({centre_x:.1f}, {centre_y:.1f}) px, "
                f"gesture {item.gesture.gesture} ({item.gesture.confidence:.2f}{raw})"
            )
    return _OK if observations else _NOTHING_FOUND


# --- --rig ---------------------------------------------------------------------------------------


def _load_transform(path: Optional[str], rig_id: str) -> dict[str, np.ndarray]:
    """`{rig: 4x4}`, or an empty map when no path was given.

    On an empty map `HandFinder` skips the rig with a warning rather than reporting camera-frame
    coordinates as base-frame ones.
    """
    if path is None:
        return {}
    matrix = np.load(path)
    if matrix.shape != (4, 4):
        raise ValueError(f"--transform must hold a 4x4 matrix, got {matrix.shape}")
    return {rig_id: matrix.astype(np.float64)}


def _pick_rig(camera_cfg: Any, rig_id: str) -> Any:
    """The named rig, or a `ValueError` (exit 2) that says what is wrong with the id.

    Neither check existed, and both failures wore the costume of "no hand in the workspace".
    `HandFinder` walks the `rig_ids` it is handed and grabs from each, so an id that names
    nothing simply never matches: `--rig ovrhead` reached the end of the loop and this command
    exited 1, the code that means the workspace is clear, on a path whose whole job is to say
    whether a person is standing in it. `enabled: false` was never read at all, so a rig an
    operator switched off, and the shipped base profile switches its only RGB-D rig off, was
    opened and grabbed from.

    A `ValueError` rather than a `SystemExit` because `main` already routes the three setup
    failures to exit 2 through one handler, and raising `SystemExit` here would leave that
    handler unused.
    """
    rigs = list(camera_cfg.cameras.rigs)
    by_id = {rig.rig_id: rig for rig in rigs}
    rig = by_id.get(rig_id)
    if rig is None:
        raise ValueError(
            f"no rig {rig_id!r} in camera.cameras.rigs. Configured: {', '.join(sorted(by_id))}"
        )
    if not getattr(rig, "enabled", True):
        raise ValueError(
            f"rig {rig_id!r} is configured with `enabled: false`, so this cell does not run it. "
            f"Set camera.cameras.rigs[{rig_id!r}].enabled: true, or name a rig that is on."
        )
    return rig


def _locate_on_rig(config: Any, args: argparse.Namespace) -> int:
    from src.camera.orchestration.frame_provider import FrameProvider
    from src.camera.setup.image_taking.intrinsics import load_intrinsics
    from src.models.handdetection.factory import (
        build_gesture_recognizer,
        build_hand_finder,
    )

    rig = _pick_rig(config.camera, args.rig)

    transforms = _load_transform(args.transform, args.rig)
    if not transforms:
        print(
            "no --transform given: the rig will be SKIPPED rather than reporting camera-frame "
            "coordinates as base-frame ones.",
            file=sys.stderr,
        )

    camera_matrices: dict[str, np.ndarray] = {}
    if args.intrinsics:
        matrix, _dist = load_intrinsics(args.intrinsics)
        camera_matrices[args.rig] = matrix

    observer = (
        build_gesture_recognizer(config.models.gesturedetect) if args.gestures else None
    )
    # `FrameProvider` wants the list of rigs, which is what `.rigs` supplies. Handing it
    # `config.camera.cameras`, the `CameraSystemConfig` model, gets past the provider's own
    # `if not rigs` guard, because a Pydantic model is truthy; iterating a model then yields
    # `('primary_rig_id', 'webcam_main')` pairs, so `rig.rig_id` raises `AttributeError`, which the
    # `except (FileNotFoundError, ImportError, ValueError)` handler in `main` does not catch, and
    # the command exits 1 with a traceback that reads like "no hand was found". The other
    # `FrameProvider` call sites spell it the same way: `cells.py:139`,
    # `real_cell/calibrate.py:193` and `perception/__main__.py:73`. Nothing imported this
    # module then, which is why four correct siblings did not make the fifth correct.
    # `tests/test_hand_detection_cli.py` now does.
    #
    # One rig, not the cell. `FrameProvider.open()`, which `with` calls, claims every
    # configured streamer, so asking one camera whether a hand is in the way took three
    # cameras away from the console. Constructing the provider touches no device, because
    # every streamer `__init__` only stores config, so it may know every rig while holding
    # the one it was asked for. `real_cell/calibrate.py` has used this pair since it was
    # written, and this command had not.
    provider = FrameProvider(list(config.camera.cameras.rigs))
    provider.open_rig(rig.rig_id)
    try:
        finder = build_hand_finder(
            config.models.handdetect,
            provider=provider,
            transforms=transforms,
            observer=observer,
            camera_matrices=camera_matrices,
            rig_ids=[rig.rig_id],
        )
        located, _annotated = finder.find_hand()
    finally:
        provider.release_rig(rig.rig_id)

    if located is None:
        if not args.json:
            print("no single hand with usable depth found")
        else:
            print(json.dumps({"hand": None}))
        return _NOTHING_FOUND

    payload = {
        "rig_id": located.position.rig_id,
        "position_base_mm": [round(float(v), 2) for v in located.position.position_base],
        "position_cam_mm": [round(float(v), 2) for v in located.position.position_cam],
        "depth_mm": round(located.position.depth_mm, 2),
        "palm_center_xy": list(located.palm.palm_center_xy),
        "handedness": str(located.palm.handedness),
        "gesture": str(located.gesture.gesture),
        "gesture_confidence": round(located.gesture.confidence, 4),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        base_x, base_y, base_z = (float(v) for v in located.position.position_base)
        print(
            f"hand on rig {located.position.rig_id}: "
            f"base ({base_x:.1f}, {base_y:.1f}, {base_z:.1f}) mm, "
            f"depth {located.position.depth_mm:.1f} mm, "
            f"gesture {located.gesture.gesture} ({located.gesture.confidence:.2f})"
        )
    return _OK


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config()

    try:
        if args.check:
            return _check(config, args.json)
        if args.frame:
            return _observe_frame(config, args.frame, args.gestures, args.json)
        return _locate_on_rig(config, args)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        # The three setup failures: a missing bundle, a missing extra, a config that cannot
        # describe the cell. The exit code keeps them apart from "no hand found".
        print(str(exc), file=sys.stderr)
        return _SETUP_PROBLEM


if __name__ == "__main__":  # pragma: no cover (CLI entry)
    raise SystemExit(main())
