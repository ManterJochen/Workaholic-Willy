"""A UR description is found where its meshes are, and a descriptor check with nothing to check fails.

`build_ur_config.py` refuses a description whose mesh references resolve nowhere. Isaac writes every UR mesh as
`package://ur_description/meshes/...`, and the meshes a descriptor then uses sit under the cuRobo asset root, so the
check has to resolve a package reference there. Measured on the box on 2026-09-15: all four rebuilds refused with
"all 14 mesh references are missing from disk" while the 14 ur5e files were under the cuRobo root, and
`check_ur_descriptors.py` then reported "all 0 descriptors load and plan" and exited 0.

Both scripts run in the cuRobo environment, where this repository cannot be imported, so their modules are loaded by
path, and the script and this test reach one implementation.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "curobo"


def _load(name: str) -> ModuleType:
    """Load one of the cuRobo scripts the way it is really run: by path, with its own folder importable.

    These scripts run in the cuRobo environment, where this repository cannot be imported, so they import their
    siblings by bare name. Python puts a script's own directory on sys.path when it starts one; loading it by
    path here does not, and the sibling import then fails for a reason that exists only in this test.
    """
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS))
    return module


def _urdf(folder: Path, *filenames: str) -> Path:
    """A description with one collision mesh per filename, written at ``folder/robot/ur5e.urdf``."""
    links = "".join(
        f'<link name="link{index}"><collision><geometry><mesh filename="{name}"/></geometry></collision></link>'
        for index, name in enumerate(filenames)
    )
    path = folder / "robot" / "ur5e.urdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<robot name="ur5e">{links}</robot>', encoding="utf-8")
    return path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"solid mesh")


class ADescriptionFindsItsMeshesTests(unittest.TestCase):

    def setUp(self) -> None:
        self.check = _load("_description_check")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_package_reference_resolves_under_the_asset_root(self) -> None:
        urdf = _urdf(self.tmp / "isaac", "package://ur_description/meshes/ur5e/collision/base.stl")
        root = self.tmp / "curobo" / "ur_description"
        _touch(root / "meshes" / "ur5e" / "collision" / "base.stl")

        usable, why = self.check.describes_a_body(urdf, asset_roots=(root,))

        self.assertTrue(usable, why)
        self.assertIn("1 resolvable", why)

    def test_the_same_reference_with_no_root_holding_it_is_refused(self) -> None:
        """The control: a reference that resolves nowhere still refuses, which is what the check is for."""
        urdf = _urdf(self.tmp / "isaac", "package://ur_description/meshes/ur5e/collision/base.stl")

        usable, why = self.check.describes_a_body(urdf, asset_roots=(self.tmp / "empty",))

        self.assertFalse(usable)
        self.assertIn("missing from disk", why)

    def test_a_description_with_one_missing_mesh_is_refused(self) -> None:
        """Half a body is not a body. On 2026-09-15 ur16e resolved 10 of its 14 meshes and the build wrote a descriptor."""
        urdf = _urdf(self.tmp / "isaac", "package://ur_description/meshes/ur16e/collision/base.stl",
                     "package://ur_description/meshes/ur16e/collision/forearm.stl")
        root = self.tmp / "curobo" / "ur_description"
        _touch(root / "meshes" / "ur16e" / "collision" / "base.stl")

        usable, why = self.check.describes_a_body(urdf, asset_roots=(root,))

        self.assertFalse(usable)
        self.assertIn("1 of 2 mesh references are missing from disk", why)
        self.assertIn("forearm.stl", why)

    def test_a_reference_beside_the_description_still_resolves(self) -> None:
        """The importer package's copies reference their meshes relative to the description."""
        urdf = _urdf(self.tmp / "importer", "../meshes/collision/base.stl")
        _touch(self.tmp / "importer" / "meshes" / "collision" / "base.stl")

        usable, why = self.check.describes_a_body(urdf)

        self.assertTrue(usable, why)

    def test_a_description_with_no_geometry_is_refused(self) -> None:
        path = self.tmp / "kinematics.urdf"
        path.write_text('<robot name="ur10"><link name="base_link"/></robot>', encoding="utf-8")

        usable, why = self.check.describes_a_body(path)

        self.assertFalse(usable)
        self.assertIn("no geometry", why)


class ADescriptorCheckWithNothingToCheckFailsTests(unittest.TestCase):

    def test_an_empty_content_directory_exits_1(self) -> None:
        checker = _load("check_ur_descriptors")
        with tempfile.TemporaryDirectory() as content:
            (Path(content) / "configs" / "robot").mkdir(parents=True)
            with mock.patch.dict(os.environ, {"WILLY_CUROBO_CONTENT": content}):
                # A hand is required since S12: the check composes the arm descriptor with one, as a cell does.
                self.assertEqual(checker.main(["--hand", "robotiq_2f85"]), 1)


if __name__ == "__main__":
    unittest.main()
