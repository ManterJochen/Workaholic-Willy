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
import pkgutil
import unittest

#: The modules the hand-written list named until P0e, kept as the walk's positive control: a walk that stops reaching
#: the tree would otherwise pass on whatever it still finds.
_NAMED_BEFORE_THE_WALK = {
    "src.robot.drivers.registry",
    "src.robot.drivers.ur.arm",
    "src.robot.drivers.ur.motion",
    "src.robot.drivers.ur.connection",
    "src.robot.drivers.ur.pose",
    "src.robot.drivers.ur.pose_adapter",
    "src.robot.drivers.sim.arm",
    "src.robot.drivers.dummy.arm",
}


def _driver_modules() -> list[str]:
    """Every module under ``drivers``, walked rather than listed.

    The list was hand-kept, and P0e moves the planning half into ``drivers/planned``: a new module a list does not
    name is a module this test never imports. A package the walk cannot import raises instead of being skipped,
    which is what ``pkgutil`` does by default. ``ur/__main__`` is guarded by ``if __name__ == "__main__"``.
    """
    import src.robot.drivers as drivers

    def refuse(name: str) -> None:
        raise ImportError(f"the walk could not import the package {name}")

    return sorted(module.name for module in
                  pkgutil.walk_packages(drivers.__path__, drivers.__name__ + ".", onerror=refuse))


class DriverImportSmokeTests(unittest.TestCase):
    def test_all_driver_modules_import_without_sdk(self) -> None:
        for mod in _driver_modules():
            with self.subTest(module=mod):
                importlib.import_module(mod)

    def test_the_walk_reaches_every_module_the_old_list_named(self) -> None:
        walked = set(_driver_modules())
        self.assertEqual(set(), _NAMED_BEFORE_THE_WALK - walked)
        self.assertGreater(len(walked), len(_NAMED_BEFORE_THE_WALK))


if __name__ == "__main__":
    unittest.main()
