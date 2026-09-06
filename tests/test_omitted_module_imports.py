"""L7 D3 — import-smoke for coverage-omitted ML / driver / camera / viz modules.

The pyproject [tool.coverage.run] omit-list excludes the heavy ML / SDK / camera / GUI modules (they can't
EXECUTE headless on CI — no GPU, robot, camera, or display). But omitting them from coverage also means a
module that fails to IMPORT (a syntax error, a broken top-level import, a signature drift) is never caught.

These modules are all import-clean mock-side (heavy deps are lazy-imported inside functions), so this test
simply ``importlib.import_module``'s each STILL-uncovered omitted module to guard importability. It extends
the existing ``test_driver_imports.py`` (UR/KUKA) + ``test_models_imports.py`` (handdetection/speech) which
already cover their slice. Do NOT trim the omit-list itself — these modules genuinely cannot run on CI.
"""

from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace

import pytest

#: Coverage-omitted modules not already smoke-imported elsewhere (detector/segmenter/_inference,
#: robotiq, the GUI viz path, and the camera-setup capture modules).
_OMITTED_MODULES = (
    "src.models._inference",
    "src.models.detection.zero_shot.detector",
    "src.models.detection.closed_set.detector",
    "src.models.segmentation.realtime.segmenter",
    "src.models.segmentation.research.segmenter",
    "src.robot.grippers.robotiq",
    "src.robot.grasping.visualization.debug_draw",
    "src.robot.grasping.visualization.open3d_viewer",
    "src.camera.setup.quality",
    "src.camera.setup.devices.stereocamera",
    "src.camera.setup.devices.webcam",
    "src.camera.setup.image_taking.single",
    "src.camera.setup.image_taking.webcam",
    "src.camera.setup.image_taking.rgbd",
)


class OmittedModuleImportSmokeTests(unittest.TestCase):
    def test_omitted_modules_import_clean(self) -> None:
        for module_name in _OMITTED_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


def test_factory_selection_guards_are_torch_free() -> None:
    """The factory raises a clear error when a selected backend has no config block,
    without importing torch (the guard runs before the lazy wrapper import)."""
    from src.models.factory import build_object_detector, build_segmenter

    with pytest.raises(ValueError):
        build_object_detector(SimpleNamespace(detector="rtdetr", rtdetr=None))
    with pytest.raises(ValueError):
        build_segmenter(SimpleNamespace(segmenter_backend="oneformer", oneformer=None))


if __name__ == "__main__":
    unittest.main()
