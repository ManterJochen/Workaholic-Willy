"""The cell's own config layer is kept out of git by the repository itself, so the console can teach in a checkout.

The owner's decision (Q10 = A): taught poses go into an untracked last layer of the profile chain, such as
``robot/robot.cell.yaml`` (``--profile ur10,hande,cell``), git-ignored. The pose door refuses every write into a last
layer git does not keep out (``src.config.edit.pose_layer_refusal``: tracked, or neither tracked nor ignored), because
the next ``git add`` would carry the cell's poses into every clone. So the repository's ``.gitignore`` names the cell
layers (``config/**/*.cell.yaml``); without it, teaching in the checkout is refused until each machine excludes the
file by hand. No cell layer is tracked, either: a tracked file is never ignored, whatever a pattern says.

Skipped outside a git checkout, where the pose door asks git nothing.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

from src.config.edit import pose_layer_refusal

_REPO = Path(__file__).resolve().parents[1]


def _in_a_checkout() -> bool:
    return shutil.which("git") is not None and (_REPO / ".git").exists()


@unittest.skipUnless(_in_a_checkout(), "not a git checkout with git on the PATH: the pose door asks git nothing here")
class TheCellsLayerIsKeptOutTests(unittest.TestCase):
    def test_the_owners_chain_may_take_a_taught_pose_in_the_checkout(self) -> None:
        """Red before: 'neither tracked nor ignored', so every pose write in the checkout was refused no_layer."""
        self.assertEqual("", pose_layer_refusal(_REPO / "config", ("ur10", "hande", "cell")))

    def test_git_ignores_every_sections_cell_layer(self) -> None:
        for path in ("config/robot/robot.cell.yaml", "config/camera/camera.cell.yaml", "config/models/models.cell.yaml"):
            with self.subTest(path=path):
                said = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=_REPO, capture_output=True,
                                      timeout=30, check=False)
                self.assertEqual(0, said.returncode, f"git does not ignore {path}")

    def test_no_cell_layer_is_tracked(self) -> None:
        tracked = subprocess.run(["git", "ls-files", "--", "config/*.cell.yaml", "config/**/*.cell.yaml"], cwd=_REPO,
                                 capture_output=True, encoding="utf-8", timeout=30, check=True).stdout.split()
        self.assertEqual([], tracked)

    def test_a_shared_layer_is_still_refused(self) -> None:
        """The rule still bites where it should: a chain ending in a tracked profile layer is no cell's own."""
        self.assertNotEqual("", pose_layer_refusal(_REPO / "config", ("ur10", "hande")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
