"""A descriptor check with nothing to check fails.

Measured on the box on 2026-09-15: `check.py` reported "all 0 descriptors load and plan" and exited 0
over a content directory that held none. The other half of this file tested `_description_check.py`, which judged
whether an Isaac description's mesh references resolved; it served the Isaac source only and went with it in P0f F5.

Both scripts run in the cuRobo environment, where this repository cannot be imported, so their modules are loaded by
path, and the script and this test reach one implementation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "arms"


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


class ADescriptorCheckWithNothingToCheckFailsTests(unittest.TestCase):

    def test_an_empty_content_directory_exits_1(self) -> None:
        checker = _load("check")
        with tempfile.TemporaryDirectory() as content:
            (Path(content) / "configs" / "robot").mkdir(parents=True)
            with mock.patch.dict(os.environ, {"WILLY_CUROBO_CONTENT": content}):
                # A hand is required since S12: the check composes the arm descriptor with one, as a cell does.
                self.assertEqual(checker.main(["--hand", "robotiq_2f85"]), 1)


if __name__ == "__main__":
    unittest.main()
