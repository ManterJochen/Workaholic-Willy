"""``python -m src.config`` as a real subprocess, against a config tree that is not the repo's.

Rescued from ``test_config_editor_cli.py`` when ``backend/config/editor.py`` was deleted (2026-08-09).
Despite living in that file, this test never touched the editor: it exercises the config CLI's entry
point end to end.

**Why a subprocess and not ``main([...])``.** ``tests/test_config_explain.py`` already calls ``main``
in-process, which is cheaper and covers the argument handling. It cannot cover what this does: that
``python -m src.config`` works as a module entry point at all, with a clean environment, no inherited
``WILLY_PROFILE``, and a ``--data`` tree somewhere else on disk. Those are exactly the conditions an
operator's first command runs under, and the ones an import-time or packaging mistake breaks first. It is
also the only test anywhere of the ``--data`` and ``--print`` flags.

**It contributes zero measured coverage, and that is the point.** ``coverage`` does not instrument the
child process, so running this file alone reports *"Module src.config was never imported"*. What it
protects is therefore invisible to ``--cov-fail-under``: deleting it would cost nothing on the coverage
report and would remove the only check that the entry point runs at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "config"


class ConfigCliSubprocessTests(unittest.TestCase):
    def test_it_validates_and_prints_a_config_tree_outside_the_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            shutil.copytree(DATA_DIR, root)

            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            # Cleared deliberately: a profile leaking in from the developer's shell would silently
            # change which overlays are merged, and the test would then be asserting about a tree it
            # did not describe.
            env.pop("WILLY_PROFILE", None)

            validate = subprocess.run(
                [sys.executable, "-m", "src.config", "--data", str(root)],
                cwd=ROOT, env=env, text=True, capture_output=True, check=False,
            )
            printed = subprocess.run(
                [sys.executable, "-m", "src.config", "--data", str(root), "--print"],
                cwd=ROOT, env=env, text=True, capture_output=True, check=False,
            )

        self.assertEqual(validate.returncode, 0, validate.stderr)
        self.assertIn("OK", validate.stdout)
        self.assertEqual(printed.returncode, 0, printed.stderr)
        self.assertIn('"camera"', printed.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
