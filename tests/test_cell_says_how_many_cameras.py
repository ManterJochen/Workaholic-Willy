"""A cell says at BOOT how many calibrated cameras it has, and warns when that is fewer than two.

⛔ THE GAP THIS CLOSES. `on_camera_unavailable` answers "a camera I asked for did not deliver". It
cannot answer "I never asked for a second one", because a one-camera cell is a legitimate cell and
refusing it would be wrong. So the case that was said nowhere at all is the one an operator is most
likely to get wrong: believing they built a multi-camera cell and having built a single-view one.

⚠ SINGLE-VIEW IS A MEASURED COST, not a neutral choice. On the datagen reference, top-1 goes 43.50 %
single-view against 55.93 % fused. That is the number in the warning, so the operator can weigh it.

⚠ ONE LINE AT CONSTRUCTION, and INFO when the count is fine. A warning on every pick trains an
operator to stop reading warnings, which is how the one that mattered gets missed.

⚠ NEVER RUN ON HARDWARE. Bucket 3 with the rest of WS3.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from src.config.schema.robot import RobotConfig
from src.robot.execution.autonomous_grasp import builders
from src.robot.execution.autonomous_grasp.config import GraspMode


def _grasping(cameras: dict[str, str], *, fusion_enabled: bool = False):  # noqa: ANN202
    """A REAL validated grasping config, because a namespace would not prove the keys exist."""
    return RobotConfig.model_validate({
        "vendor": "dummy",
        "grasping": {
            "fusion": {
                "enabled": fusion_enabled,
                "cameras": {
                    name: {"enabled": True, "mounting_mode": mode,
                           "extrinsics_artifact_path": f"{name}.json"}
                    for name, mode in cameras.items()
                },
            },
        },
    }).grasping


def _lines(cameras: dict[str, str], **kwargs) -> list[tuple[str, str]]:  # noqa: ANN003
    # `frame_resolver=None` on purpose: with fusion enabled the overlay builds the voxel substrate
    # only when a resolver exists, and this test is about the CAMERA COUNT line, not that substrate.
    runtime = SimpleNamespace(orchestrator=SimpleNamespace(frame_resolver=None))
    captured: list[tuple[str, str]] = []

    handler = logging.Handler()
    handler.emit = lambda record: captured.append((record.levelname, record.getMessage()))
    builders.logger.addHandler(handler)
    try:
        builders.apply_orchestrator_overlays(
            runtime, _grasping(cameras, **kwargs), resolved_mode=GraspMode.EASY)
    finally:
        builders.logger.removeHandler(handler)
    return [(level, message) for level, message in captured if "calibrated camera" in message]


class ItWarnsBelowTwoTests(unittest.TestCase):

    def test_one_camera_WARNS(self) -> None:
        """⭐ The case the per-pick policy structurally cannot see."""
        lines = _lines({"overhead": "eye_to_hand"})

        self.assertTrue(lines, "a single-camera cell said nothing about being single-view")
        level, message = lines[0]
        self.assertEqual(level, "WARNING")
        self.assertIn("SINGLE-VIEW", message)
        self.assertIn("overhead", message, "the warning does not name what it did find")

    def test_the_warning_carries_the_MEASUREMENT(self) -> None:
        """An operator deciding whether to add a camera needs the size of the prize, not an
        instruction. 43.50 against 55.93 is what a second view was worth on the reference."""
        _level, message = _lines({"overhead": "eye_to_hand"})[0]

        self.assertIn("43.50", message)
        self.assertIn("55.93", message)

    def test_no_cameras_at_all_still_warns(self) -> None:
        """The commonest shape of all: a cell whose fusion block was never filled in."""
        lines = _lines({})

        self.assertTrue(lines)
        self.assertEqual(lines[0][0], "WARNING")
        self.assertIn("0 calibrated camera", lines[0][1])


class ItIsQuietWhenTheCellIsFineTests(unittest.TestCase):

    def test_two_cameras_report_at_INFO(self) -> None:
        """⚠ Not silence and not a warning. A boot line that says what the cell has is how an
        operator confirms the config they meant; a warning here would be noise that teaches them to
        skip the one above."""
        lines = _lines({"overhead": "eye_to_hand", "wrist": "eye_in_hand"})

        self.assertTrue(lines)
        self.assertEqual(lines[0][0], "INFO")
        self.assertIn("2 calibrated camera", lines[0][1])

    def test_an_eye_in_hand_rig_COUNTS_and_is_named(self) -> None:
        """A wrist camera is a second VIEWPOINT from one device: it sees the far side of a part the
        fixed camera cannot, which is the whole reason fusing helps."""
        _level, message = _lines({"overhead": "eye_to_hand", "wrist": "eye_in_hand"})[0]

        self.assertIn("1 eye-in-hand", message)


class ItDoesNotDependOnTheFusionSWITCHTests(unittest.TestCase):
    """⛔ THE TRAP THIS AVOIDS, and it is the same one WS3.4 was about. Every other camera check in
    this file sits inside `if geometry_cfg.enabled`, so a cell with `fusion.enabled: false` gets no
    camera reporting at all. That is exactly the cell most likely to be single-view by accident."""

    def test_it_fires_with_fusion_DISABLED(self) -> None:
        lines = _lines({"overhead": "eye_to_hand"}, fusion_enabled=False)

        self.assertTrue(lines, "the boot line is gated behind the fusion switch again")

    def test_it_fires_with_fusion_ENABLED_too(self) -> None:
        lines = _lines({"overhead": "eye_to_hand"}, fusion_enabled=True)

        self.assertTrue(lines)
        self.assertEqual(lines[0][0], "WARNING")


if __name__ == "__main__":
    unittest.main()
