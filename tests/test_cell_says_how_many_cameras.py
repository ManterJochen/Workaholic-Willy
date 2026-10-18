"""A cell says at BOOT which cameras it fuses, and warns when it was meant to fuse and cannot.

⛔ THE GAP THIS CLOSES. A missing camera at a pick is warned of where it fails to deliver. Nothing there can answer "I
never got a second one", so the case said nowhere at all would be the one an operator is most likely to get wrong:
believing they built a multi-camera cell and having built a single-view one. Since 2026-10-16 a cell fuses by itself
wherever its camera section declares two or more enabled rigs with their calibration (the owner's cleanup), so the
boot line is the fusion plan: what it fuses, or, where two or more enabled rigs that give depth are declared and they
do not fuse, a WARNING that names what is missing. A stereo pair gives depth once its stereo calibration is written
(2026-10-17), and the shipped base profile's webcam pair has none, so it warns of nothing.

⚠ SINGLE-VIEW IS A MEASURED COST, not a neutral choice. On the datagen reference, top-1 goes 43.50 % single-view
against 55.93 % fused. That is the number in the warning, so the operator can weigh it.

⚠ ONE LINE AT CONSTRUCTION, and INFO when the cell is what it says. A warning on every pick, or on every one-camera
cell, trains an operator to stop reading warnings, which is how the one that mattered gets missed: a cell of one
camera is a legitimate cell, and a wrist camera fuses its own looks.

⚠ NEVER RUN ON HARDWARE. Bucket 3 with the rest of WS3.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

from src.config import load_config
from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import builders
from src.robot.execution.autonomous_grasp.config import GraspMode

_PLAN_LINES = ("fusing ", "no camera fusion", "this cell cannot fuse")


def _camera(cameras: dict[str, str | None]):  # noqa: ANN202
    """The shipped camera section with one RGB-D rig per camera, the first the primary, each declaring its
    calibration on its rig, `camera.cameras.rigs[<id>].extrinsics`. A mounting of None leaves that rig uncalibrated.
    """
    shipped = load_config().camera
    data = shipped.model_dump(mode="json")
    template = next(rig for rig in data["cameras"]["rigs"] if rig["source"] == "rgbd")
    for name, mode in cameras.items():
        extrinsics: dict | None = None
        if mode is not None:
            extrinsics = {"mounting_mode": mode, "artifact_path": f"{name}.json"}
            if mode == "eye_in_hand":
                extrinsics.update(shutter_motion_tolerance_mm=2.0, shutter_motion_tolerance_deg=0.5)
        # Enabled with a serial each, as a real cell of several RGB-D cameras has to be.
        data["cameras"]["rigs"].append({**template, "rig_id": name, "enabled": True,
                                        "serial_number": f"serial_{name}", "extrinsics": extrinsics})
    if cameras:
        data["cameras"]["primary_rig_id"] = next(iter(cameras))
    return type(shipped).model_validate(data)


def _lines(cameras: dict[str, str | None]) -> list[tuple[str, str]]:
    # A bare orchestrator stand-in: the overlays only set attributes on it. The artifacts are never opened: a fused
    # cell's resolvers stand in for the calibrations they would load.
    runtime = SimpleNamespace(orchestrator=SimpleNamespace())
    grasping = RobotConfig.model_validate({"vendor": "dummy", "grasping": {}}).grasping
    captured: list[tuple[str, str]] = []

    handler = logging.Handler()
    handler.emit = lambda record: captured.append((record.levelname, record.getMessage()))  # type: ignore[method-assign]
    builders.logger.addHandler(handler)
    try:
        with mock.patch.object(builders, "_resolver_for_rig", lambda _rig_id, _extrinsics: object()):
            builders.apply_orchestrator_overlays(runtime, grasping, resolved_mode=GraspMode.EASY,  # type: ignore[arg-type]
                                                 camera=_camera(cameras))
    finally:
        builders.logger.removeHandler(handler)
    return [(level, message) for level, message in captured if message.startswith(_PLAN_LINES)]


class ItWarnsWhereTheCellWasMeantToFuseTests(unittest.TestCase):

    def test_two_rgbd_rigs_of_which_one_declares_no_calibration_WARN(self) -> None:
        """⭐ The cell an operator believes is multi-camera: two RGB-D rigs, and one of them cannot be placed."""
        ((level, message),) = _lines({"overhead": "eye_to_hand", "side": None})

        self.assertEqual("WARNING", level)
        self.assertIn("rig 'side' declares no calibration", message, "the warning does not name what is missing")
        self.assertIn("43.50", message)
        self.assertIn("55.93", message)


class ItIsQuietWhenTheCellIsWhatItSaysTests(unittest.TestCase):

    def test_two_cameras_report_at_INFO(self) -> None:
        """⚠ Not silence and not a warning. A boot line that says what the cell fuses is how an operator confirms the
        config they meant; a warning here would be noise that teaches them to skip the one above."""
        ((level, message),) = _lines({"overhead": "eye_to_hand", "wrist": "eye_in_hand"})

        self.assertEqual("INFO", level)
        self.assertIn("fusing 'overhead' (primary) and 'wrist' (on the wrist)", message)

    def test_one_camera_reports_at_INFO(self) -> None:
        """A cell of one camera is a legitimate cell, the owner's among them: it is said, not warned of."""
        ((level, message),) = _lines({"overhead": "eye_to_hand"})

        self.assertEqual("INFO", level)
        self.assertIn("no camera fusion", message)
        self.assertIn("one enabled, calibrated rig, 'overhead'", message)


if __name__ == "__main__":
    unittest.main()
