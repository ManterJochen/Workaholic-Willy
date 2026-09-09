"""Does every camera THIS cell names have a calibration artifact that opens and resolves?

    python scripts/checks/camera_artifacts.py

⛔ **`run_config_preflight` DOES NOT ANSWER THIS.** Its `camera -> base` check tests
`if artifact and fusion.enabled` (preflight.py:196): it reads one scalar key, never opens the file,
and never looks at the per-camera map at all. So it prints an ok beside a cell whose second camera's
artifact was never written, is half a JSON file, or is a CAMERA->TOOL transform where a
CAMERA->BASE one was declared. The pytest suite does not answer it
either, because every calibration test loads a fixture it wrote itself in the same run.

`build_config_frame_resolvers` is the only function in the tree that opens EVERY named camera's
artifact, and it is fail-closed: one unreadable artifact raises instead of returning a short map.
Today that refusal arrives at build time, which is the moment a cell is being brought up. This runs
the same function over the operator's own tree, at a desk, with nothing built and no device opened.

Every value below comes from the YAML this box loads: the camera ids, the mounting modes and the
artifact paths of `grasping.fusion.cameras`, plus the scalar
`grasping.fusion.extrinsics_artifact_path` that a real cell's build refusal names. That map is the
whole camera inventory, the primary included
(pinned by tests/test_fusion_camera_map_means_every_camera.py), so every entry in it is opened here.

`fusion.enabled` is forced true on a `model_copy` of the config, and each camera is offered on its
own. Both are deliberate. A cell that ships with fusion off still has artifacts it will need the day
it is switched on, and switching it on must not be the first time anyone opens the file; and the
library raises on the FIRST bad artifact, so asking for one camera at a time names all of them in
one run instead of one per bring-up attempt. No YAML is edited and nothing is written.

Exit codes: 0 every named camera resolved, 1 an artifact a camera names cannot be opened or
resolved, 2 this tree names no camera artifact at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.robot.execution.autonomous_grasp.builders import (  # noqa: E402
    build_config_frame_resolver,
    build_config_frame_resolvers,
)

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def _not_ready(what: str, fix: str) -> int:
    print(f"NOT READY: {what}\n  fix: {fix}")
    return EXIT_NOT_READY


def _why(error: BaseException) -> str:
    """The sentence of a refusal that says what was wrong, without the paragraph after it.

    These messages are written for a terminal: the path and the underlying error come first, then
    several sentences of remedy. Both trims degrade to the whole message rather than to nothing.
    """
    text = " ".join(str(error).split())
    if "failed to load " in text:
        text = text.split("failed to load ", 1)[1]
    sentence = text.split(". ", 1)[0]
    return sentence if len(sentence) <= 140 else sentence[:140] + " ..."


def _probe(grasping: Any, **fusion_fields: Any) -> Any:
    """A copy of the grasping config with `fusion.enabled` on and these fields replaced."""
    fusion = grasping.fusion.model_copy(update={"enabled": True, **fusion_fields})
    return grasping.model_copy(update={"fusion": fusion})


def main() -> int:
    try:
        config = load_robot_config()
    except ConfigError as error:
        return _not_ready(f"the config tree has no cell to check ({error})",
                          "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")

    fusion = config.grasping.fusion
    declared = dict(fusion.cameras or {})
    cameras = {cam: cfg for cam, cfg in declared.items() if bool(getattr(cfg, "enabled", True))}
    switched_off = sorted(set(declared) - set(cameras))
    scalar = fusion.extrinsics_artifact_path

    if not cameras and not scalar:
        off = f" ({len(switched_off)} camera(s) are enabled: false)" if switched_off else ""
        return _not_ready(
            "this tree names no camera calibration artifact: grasping.fusion.cameras is empty and "
            f"grasping.fusion.extrinsics_artifact_path is unset{off}",
            "point this at a calibrated cell (WILLY_PROFILE=sim is the shipped multi-camera one), "
            "or write the first artifact: scripts/examples/api/02_calibration/calibrate_fixed_camera.py")

    print(f"{len(cameras)} camera(s) in grasping.fusion.cameras"
          f"{', plus the primary artifact key' if scalar else ''}; "
          f"fusion.enabled is {fusion.enabled} and every one of them is opened here")
    if switched_off:
        print(f"  not opened (enabled: false): {switched_off}")

    failed: list[str] = []
    for cam_id, cam_cfg in sorted(cameras.items()):
        try:
            # ⛔ NOT `SystemExit` AND NOT `AttributeError`. Those were the shapes of an older build
            # path; a config a person has to go and fix is an `Exception` now. `CellBuildRefused`
            # (cells.py:50) is a `RuntimeError`, and so is the fail-closed refusal this builder
            # raises for an artifact it could not read, so this one clause covers both.
            alone = _probe(config.grasping, cameras={cam_id: cam_cfg})
            resolvers = build_config_frame_resolvers(alone)
        except RuntimeError as error:
            failed.append(f"{cam_id}: {_why(error)}")
            print(f"  {cam_id:20s} FAILED    {_why(error)}")
            continue
        resolver = resolvers.get(cam_id)
        if resolver is None:
            failed.append(f"{cam_id}: named and enabled, and no resolver was built for it")
            print(f"  {cam_id:20s} ABSENT    <- enabled, and not in the resolver map")
        else:
            print(f"  {cam_id:20s} {type(resolver).__name__:26s} {cam_cfg.mounting_mode}")

    # The scalar key is a second name for one camera's artifact, and it is the one
    # `from_robot_config` quotes when it refuses a real cell for having no CAMERA->BASE transform.
    # The preflight's `camera -> base` line reads this key and stops; this opens what it names.
    if scalar:
        try:
            primary = build_config_frame_resolver(_probe(config.grasping))
        except RuntimeError as error:
            failed.append(f"extrinsics_artifact_path: {_why(error)}")
            print(f"  {'(primary key)':20s} FAILED    {_why(error)}")
        else:
            if primary is None:
                failed.append(f"extrinsics_artifact_path names {scalar!r} and resolved to nothing")
                print(f"  {'(primary key)':20s} ABSENT    <- {scalar!r} resolved to nothing")
            else:
                print(f"  {'(primary key)':20s} {type(primary).__name__:26s} {scalar}")

    if failed:
        print(f"\nFAILED: {len(failed)} artifact(s) this cell names cannot be opened or resolved")
        for line in failed:
            print(f"  {line}")
        print("  calibrate the camera that failed, one run per rig: python "
              "scripts/examples/api/02_calibration/calibrate_fixed_camera.py --rig <rig> --live")
        return EXIT_FAILED
    print(f"\nOK: every named camera resolved ({len(cameras) + (1 if scalar else 0)} artifact(s) "
          "opened)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
