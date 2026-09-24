"""The calibrate CLI prints what it printed: three transcripts, byte for byte.

A pin, green before the flow moved into ``HandEyeCalibration`` and green after it. Step 7 of
docs/runbooks/real_cell_first_pick.md reads this output at the cell, so the move must not change a line
of it: not the banners, not the order of the build lines, not the blank line after the rig block.

Everything below the arm-vendor gate is a double: the gate, the arm factory (a dummy arm), the camera noun
(a handle with a fixed repr) and, for the sweep, the routine's ``run_from_json``.

Since the owner deleted every automatic station generator (2026-09-24) a sweep names its stations: the transcripts
run ``--fixed-poses`` over a file of 22 stations, and a fourth pins ``--freedrive`` on an arm a person cannot guide,
which the build refuses before the camera opens.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.config.schema.camera import HandEyeConfig
from src.config.schema.robot import RobotConfig
from src.calibration import Extrinsics, MountingMode
from src.geometry import Frame, Transform
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.calibration import CalibrationResult, CalibrationRoutine
from src.robot.execution.cell_lock import lock_path_for
from src.robot.execution.lifecycle import StepOutcome, TeardownReport
from src.robot.execution.real_cell import calibrate
from src.robot.safety import SafetyAttestation
from tests.test_camera_boundaries import _rgbd_rig

_IP = "10.253.253.21"
_KEY = f"ur@{_IP}"
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"
_CREATE_ARM = "src.robot.drivers.create_arm"
_CAMERA = "src.camera.orchestration.camera.Camera"

_CONFIG = (
    "=== 1. CONFIG ===\n"
    "  rig        'overhead' (rgbd)\n"
    "  mode       eye_to_hand\n"
    "  arm        ur\n"
    "  marker     50.0 mm, id 0, DICT_5X5_100\n"
    "  poses      {poses}\n"
    "  artifact   {out}/eth_overhead.json\n"
)


class _Handle:
    """The camera handle the CLI prints and the marker source reads: a fixed repr, a camera matrix, a count."""

    def __init__(self) -> None:
        self.released = 0

    def __repr__(self) -> str:
        return "RigHandle('overhead')"

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def release(self) -> None:
        self.released += 1


def _tree() -> SimpleNamespace:
    return SimpleNamespace(
        robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}}),
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")]), hand_eye=HandEyeConfig()),
    )


def _solved() -> CalibrationResult:
    transform = Transform(translation_mm=np.array([400.0, -25.0, 812.5]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                          from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    extrinsics = Extrinsics(transform=transform, rmse_mm=0.8, max_error_mm=1.5, num_samples=20,
                            captured_at=dt.datetime(2026, 9, 18, tzinfo=dt.timezone.utc), rig_id="rig_0")
    return CalibrationResult(T_cam_to_base=transform.to_matrix(), rmse_mm=0.8, max_error_mm=1.5, num_samples=20,
                             dataset_path="dataset.json", extrinsics=extrinsics, transform=transform,
                             mode=MountingMode.EYE_TO_HAND)


class TheCalibrateCliPrintsWhatItPrintedTests(unittest.TestCase):

    def setUp(self) -> None:
        self.arm = DummyRobotArm()
        self.handle = _Handle()
        self.addCleanup(lambda: lock_path_for(_KEY).unlink(missing_ok=True))
        self.stations = Path(self.enterContext(tempfile.TemporaryDirectory())) / "stations.json"
        self.stations.write_text(json.dumps([
            {"x": 300.0 + 10.0 * index, "y": 0.0, "z": 350.0, "rx": 0.0, "ry": 3.14159265, "rz": 0.0,
             "label": f"pose_{index}"} for index in range(22)]), encoding="utf-8")
        self.poses = f"22 from {self.stations}"

    def _main(self, *argv: str, sweep: Any = None, stations: bool = True) -> tuple[int, str]:
        camera_cls = MagicMock(name="Camera")
        camera_cls.from_config.return_value.handle.return_value = self.handle
        sweep = sweep if sweep is not None else MagicMock(side_effect=AssertionError("the sweep ran"))
        named = ["--fixed-poses", str(self.stations)] if stations else []
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=_tree()), patch(_READY), \
                patch(_CREATE_ARM, return_value=self.arm), patch(_CAMERA, camera_cls), \
                patch.object(CalibrationRoutine, "run_from_json", sweep), redirect_stdout(printed):
            code = calibrate.main(["--rig", "overhead", *named, *argv])
        return code, printed.getvalue()

    def _build(self) -> str:
        return (
            "\n=== 2. BUILD === the arm alone, and one camera\n"
            "  arm      DummyRobotArm\n"
            "  gripper  none, the sweep connects the arm alone\n"
            f"  lock     {_KEY}\n"
            "  camera   RigHandle('overhead') intrinsics=yes\n"
            + SafetyAttestation.of(self.arm).render() + "\n"
            "  camera world  not handed any camera; this sweep declines for itself\n"
        )

    def test_check(self) -> None:
        code, printed = self._main("--check")
        self.assertEqual(code, calibrate._EXIT_OK)
        self.assertEqual(printed, _CONFIG.format(out="calibration/real", poses=self.poses)
                         + "\n--check: the config and the rig are usable. Nothing was touched.\n")

    def test_dry_run(self) -> None:
        code, printed = self._main("--dry-run")
        self.assertEqual(code, calibrate._EXIT_OK)
        self.assertEqual(printed, _CONFIG.format(out="calibration/real", poses=self.poses) + self._build()
                         + "\n--dry-run: built cleanly and the camera answered. Stopping before any motion.\n")
        self.assertEqual(self.handle.released, 1)

    def test_freedrive_on_an_arm_nobody_can_guide_is_refused_at_the_build(self) -> None:
        code, printed = self._main("--freedrive", stations=False)
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        self.assertEqual(printed, (
            _CONFIG.format(out="calibration/real", poses="15 guided by hand")
            + "\n=== 2. BUILD === the arm alone, and one camera\n"
            "[build] REFUSED: HandGuidingRefused: freedrive guides the arm to every pose by hand, and DummyRobotArm "
            "offers no hand guiding (SupportsFreedrive; the UR teach mode is one). This arm has fixed stations only: "
            "run them without adjust (--fixed-poses PATH)\n"))
        self.assertEqual(self.handle.released, 0, "the camera was opened for an arm nobody can guide")

    def test_a_sweep_that_writes_its_artifact(self) -> None:
        out = str(self.enterContext(tempfile.TemporaryDirectory()))
        code, printed = self._main("--out", out, sweep=MagicMock(return_value=_solved()))
        written = str(Path(f"{out}/eth_overhead.json"))
        teardown = TeardownReport(StepOutcome.ABSENT, StepOutcome.RELEASED, StepOutcome.ABSENT).render()
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertEqual(printed, (
            _CONFIG.format(out=out, poses=self.poses) + self._build()
            + "\n=== 3. SWEEP === the robot moves now, keep hands clear\n"
            "  arm connected, and no gripper: the sweep drives the arm alone\n"
            "\n  down: the arm, then the lock, then the camera\n"
            + teardown + "\n"
            "\n=== 4. RESULT ===\n"
            "  accepted samples  20/22\n"
            "  AX=XB rmse        0.8000 mm (max 1.5000) -> excellent\n"
            f"  written           {written}\n"
            "  dataset           dataset.json\n"
            "\nPaste this into the camera section so the camera reaches the pick path. Until its rig\n"
            "declares it, the cell has no CAMERA->BASE for this camera:\n"
            "\n"
            "camera:\n"
            "  cameras:\n"
            "    rigs:\n"
            "      - rig_id: overhead\n"
            "        extrinsics:\n"
            "          mounting_mode: eye_to_hand\n"
            f"          artifact_path: {written}\n"
            "\n"
        ))
        self.assertEqual(self.handle.released, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
