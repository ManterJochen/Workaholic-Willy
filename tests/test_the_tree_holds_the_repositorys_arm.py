"""A tree that describes the arm its cell names differently from the repository is refused at load (P0e E12).

``tree_arm_refusal`` held a deployment tree's ``arms/<model>.yaml`` to the repository's file, as the hand and camera
registries hold theirs, but the tree loader never asked it: ``tree.py`` ran the hand and camera checks only, so an
edited arm file in a deployment tree was silently ignored while the collision bundles, sphere maps and evidence stayed
the repository's. The loader asks it now, for the arm the cell selects.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config.tree import load_tree

_DATA = Path(__file__).resolve().parents[1] / "config"


class TheLoaderAsksTheArmRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("WILLY_PROFILE", None)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "data"
        shutil.copytree(_DATA, self.root)

    def test_an_identical_copy_loads(self) -> None:
        loaded = load_tree(None, root=self.root)
        self.assertEqual("", loaded.error)

    def test_a_copy_that_describes_the_selected_arm_differently_is_refused_naming_both_files(self) -> None:
        arm = self.root / "arms" / "ur5e.yaml"
        arm.write_text(arm.read_text(encoding="utf-8").replace("reach_mm: 850.0", "reach_mm: 1850.0"), encoding="utf-8")
        loaded = load_tree(None, root=self.root)
        self.assertIn("describes the arm ur5e differently", str(loaded.error))
        self.assertIn("1850.0", str(loaded.error))

    def test_an_arm_the_cell_does_not_select_is_not_compared(self) -> None:
        arm = self.root / "arms" / "ur10e.yaml"
        arm.write_text(arm.read_text(encoding="utf-8").replace("reach_mm: 1300.0", "reach_mm: 1301.0"), encoding="utf-8")
        self.assertEqual("", load_tree(None, root=self.root).error)


class AStaleCopyNamesTheFileToCopyOverTests(unittest.TestCase):
    """A tree copied before a description gained a key is refused once, naming every stale file (P0f F6-0).

    P0f F6 adds six required keys to every arm description, and a deployment tree is normally a copytree of the
    shipped one, so each of those commits leaves every older tree holding eight files the schema can no longer read.
    One unreadable file refuses the tree's whole registry, which is the rule. What the refusal said until F6-0 was
    pydantic's "Field required" for the first file alone, so the operator fixed one file, loaded again and met the
    next, eight times, and never read which file to put there.
    """

    def setUp(self) -> None:
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("WILLY_PROFILE", None)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # Resolved, because the temp folder can come back as a short 8.3 name the loader spells out in full.
        self.root = Path(folder.name).resolve() / "data"
        shutil.copytree(_DATA, self.root)

    def _age(self, model: str) -> Path:
        """The tree's copy of ``model`` as it read before the ``admitted`` key existed."""
        arm = self.root / "arms" / f"{model}.yaml"
        text = arm.read_text(encoding="utf-8")
        kept = [line for line in text.splitlines(keepends=True) if not line.lstrip().startswith("admitted:")]
        self.assertEqual(len(kept) + 1, len(text.splitlines(keepends=True)), f"{model}.yaml holds one admitted line")
        arm.write_text("".join(kept), encoding="utf-8")
        return arm

    def test_every_stale_copy_is_named_with_the_repository_file_to_copy_over_it(self) -> None:
        aged = [self._age("ur10"), self._age("ur3")]
        error = str(load_tree(None, root=self.root).error)
        for path in aged:
            with self.subTest(path=path.name):
                self.assertIn(str(path), error)
                self.assertIn(f"config/arms/{path.name}", error)
        self.assertIn("admitted: Field required", error)
        self.assertIn("copy", error)
        self.assertIn("delete", error, "the other remedy, a tree with no arms/ reading through the repository's")

    def test_the_remedy_holds_the_tree_loads_once_the_files_are_copied_over(self) -> None:
        for path in (self._age("ur10"), self._age("ur3")):
            shutil.copyfile(_DATA / "arms" / path.name, path)
        self.assertEqual("", load_tree(None, root=self.root).error)

    def test_an_unreadable_file_the_repository_does_not_hold_keeps_the_plain_refusal(self) -> None:
        """The control: no repository file stands behind it, so there is nothing to copy over it."""
        stranger = self.root / "arms" / "ur99.yaml"
        stranger.write_text("arm:\n  model: \"ur99\"\n", encoding="utf-8")
        error = str(load_tree(None, root=self.root).error)
        self.assertIn(f"{stranger} is not a valid arm description", error)
        self.assertNotIn("copy", error)


if __name__ == "__main__":
    unittest.main()
