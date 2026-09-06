"""Iteration-2 regression test: every robot-driver module must IMPORT cleanly
without its vendor SDK installed (the SDKs are lazy-imported on connect()).

This guards the class of bug where ur/arm.py + ur/motion.py imported
``pose_to_urpose`` / ``urpose_to_pose`` from the WRONG module
(``geometry.conversions`` instead of ``ur.pose_adapter``), which raised
``ImportError`` the moment the production UR driver was instantiated — invisible
to the rest of the suite because the hardware drivers are never imported
elsewhere in CI.
"""

from __future__ import annotations

import importlib
import unittest

_DRIVER_MODULES = [
    "src.robot.drivers.registry",
    "src.robot.drivers.ur.arm",
    "src.robot.drivers.ur.motion",
    "src.robot.drivers.ur.connection",
    "src.robot.drivers.ur.pose",
    "src.robot.drivers.ur.pose_adapter",
    "src.robot.drivers.kuka.arm",
    "src.robot.drivers.kuka.eki_client",
    "src.robot.drivers.kuka.pose_convert",
    "src.robot.drivers.sim.arm",
    "src.robot.drivers.dummy.arm",
]


class DriverImportSmokeTests(unittest.TestCase):
    def test_all_driver_modules_import_without_sdk(self) -> None:
        for mod in _DRIVER_MODULES:
            with self.subTest(module=mod):
                importlib.import_module(mod)


if __name__ == "__main__":
    unittest.main()
