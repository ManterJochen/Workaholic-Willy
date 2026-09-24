"""Does every calibration this tree declares open and resolve?

    python scripts/checks/camera_artifacts.py

A camera's calibration is declared on its rig, `camera.cameras.rigs[<id>].extrinsics`, and every
reader loads it through one function, `RigCalibration.from_config`, when a cell is built.

`run_config_preflight` does not answer this. Its `camera -> base` row reads the primary rig's key
and opens nothing, so it prints an ok beside a rig whose artifact was never written, is half a JSON
file, or holds a CAMERA->TOOL transform where the rig declares a fixed camera. The test suite does
not answer it either, because every calibration test loads a fixture it wrote itself in the same run.

This opens each declared artifact through the same loader, over the operator's own camera section,
at a desk, with nothing built and no device opened. Rigs are opened one at a time, so one run names
every bad artifact instead of one per bring-up attempt. A rig switched off is opened too: a camera
that ships off still needs its artifact the day it is switched on. No YAML is edited and nothing is
written.

Exit codes: 0 every declared calibration resolved, 1 an artifact a rig declares cannot be opened or
resolved, 2 this tree declares no rig calibration at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ConfigError, load_camera_section  # noqa: E402
from src.calibration.rig_calibration import RigCalibration, RigCalibrationError  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def _not_ready(what: str, fix: str) -> int:
    print(f"NOT READY: {what}\n  fix: {fix}")
    return EXIT_NOT_READY


def _why(error: BaseException) -> str:
    """The sentence of a refusal that says what was wrong, without the paragraph after it."""
    sentence = " ".join(str(error).split()).split(". ", 1)[0]
    return sentence if len(sentence) <= 140 else sentence[:140] + " ..."


def main() -> int:
    try:
        camera = load_camera_section()
    except ConfigError as error:
        return _not_ready(f"the camera section does not load ({error})",
                          "fix the camera files the refusal names")

    declared = [rig for rig in camera.cameras.rigs if getattr(rig, "extrinsics", None) is not None]
    if not declared:
        return _not_ready(
            "no rig in camera.cameras.rigs declares `extrinsics`, so this tree has no calibration to open",
            "calibrate a camera (python -m src.robot.execution.real_cell.calibrate --rig <id> --freedrive) and "
            "paste the rig block it prints, or point this at a calibrated tree")

    print(f"{len(declared)} rig(s) declare a calibration; every artifact is opened here")
    failed: list[str] = []
    for rig in declared:
        try:
            calibration = RigCalibration.from_config(rig.rig_id, rig.extrinsics)
        except RigCalibrationError as error:
            failed.append(f"{rig.rig_id}: {_why(error)}")
            print(f"  {rig.rig_id:20s} FAILED    {_why(error)}")
            continue
        switched_off = "" if rig.enabled else "  (the rig is enabled: false)"
        print(f"  {rig.rig_id:20s} {calibration.mounting_mode:12s} {calibration.artifact_path}{switched_off}")

    if failed:
        print(f"\nFAILED: {len(failed)} artifact(s) this tree declares cannot be opened or resolved")
        for line in failed:
            print(f"  {line}")
        print("  calibrate the camera that failed, one run per rig: python -m "
              "src.robot.execution.real_cell.calibrate --rig <rig> --freedrive")
        return EXIT_FAILED
    print(f"\nOK: every declared calibration resolved ({len(declared)} artifact(s) opened)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
