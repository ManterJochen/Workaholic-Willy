"""A cell of several arms is one small file naming each arm and the tree it runs on (owner, 2026-10-15).

The owner's answers: one tree per arm ("gerade wenn verschiedene kommen"), the fewest keys, free names, and
``WILLY_CELL`` / ``--cell`` beside ``WILLY_PROFILE``; most users keep their trees outside the repository, so a cell file
and its arms' trees may lie anywhere. A tree with one arm needs no cell file and loads as it always did.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import ConfigError, default_data_dir
from src.config.cell import CELL_ENV_VAR, cell_file, load_cell


def _cell(folder: Path, text: str, name: str = "clip") -> Path:
    (folder / "cells").mkdir(parents=True, exist_ok=True)
    path = folder / "cells" / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class ACellNamesItsArmsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.repo = default_data_dir()

    def test_each_arm_runs_on_its_own_chain_and_the_first_stands_at_the_frame(self) -> None:
        path = _cell(self.folder, "arms:\n  halter: console_dummy\n  werker: ursim\n")
        cell = load_cell(path, root=self.repo)
        self.assertTrue(cell.ok, cell.error)
        self.assertEqual(cell.names, ("halter", "werker"))
        self.assertEqual(cell.frame_arm.name, "halter")
        self.assertEqual(cell.arm("WERKER").tree.profile, "ursim")
        self.assertEqual(cell.arm("halter").tree.robot.vendor, "dummy")
        self.assertIn("the frame stands at halter", cell.render())
        self.assertEqual([arm["name"] for arm in cell.to_dict()["arms"]], ["halter", "werker"])

    def test_an_arm_may_bring_a_tree_of_its_own_from_anywhere(self) -> None:
        path = _cell(self.folder, f"arms:\n  halter: console_dummy\n  werker: {{tree: '{self.repo.as_posix()}', "
                                  "profile: ursim}\n")
        cell = load_cell(path, root=self.repo)
        self.assertTrue(cell.ok, cell.error)
        self.assertEqual(cell.arm("werker").tree.root, self.repo)

    def test_a_bare_name_is_read_from_the_trees_cells_folder(self) -> None:
        self.assertEqual(cell_file("clip", root=self.folder), (self.folder / "cells" / "clip.yaml").resolve())
        _cell(self.folder, f"arms:\n  halter: {{tree: '{self.repo.as_posix()}', profile: console_dummy}}\n")
        self.assertTrue(load_cell("clip", root=self.folder).ok)

    def test_the_variable_names_the_cell_where_nobody_passes_one(self) -> None:
        path = _cell(self.folder, "arms:\n  halter: console_dummy\n")
        with mock.patch.dict(os.environ, {CELL_ENV_VAR: str(path)}):
            self.assertEqual(load_cell(root=self.repo).names, ("halter",))
        with mock.patch.dict(os.environ, {CELL_ENV_VAR: ""}), self.assertRaises(ConfigError):
            load_cell()

    def test_two_arms_behind_one_controller_are_refused(self) -> None:
        path = _cell(self.folder, "arms:\n  halter: ursim\n  werker: ursim,ur10\n")
        cell = load_cell(path, root=self.repo)
        self.assertFalse(cell.ok)
        self.assertIn("'halter' and 'werker' both drive ur@127.0.0.1", cell.error)

    def test_what_a_cell_file_refuses(self) -> None:
        for text, said in (
            ("arms: {}\n", "names no arm"),
            ("arms:\n  2nd: console_dummy\n", "is not a name"),
            ("arms:\n  halter: console_dummy\n  HALTER: ursim\n", "differ only in case"),
            ("arms:\n  halter: [console_dummy]\n", "takes a profile chain"),
            ("robot: {}\n", "must hold one key, 'arms:'"),
            ("arms:\n  halter: nosuchlayer\n", "the arm 'halter' (nosuchlayer)"),
        ):
            with self.subTest(said=said):
                cell = load_cell(_cell(self.folder, text), root=self.repo)
                self.assertFalse(cell.ok)
                self.assertEqual(cell.exit_code, 1)
                self.assertIn(said, cell.error)
        self.assertIn("no cell file at", load_cell(self.folder / "nowhere.yaml").error)

    def test_an_arm_the_cell_does_not_have_is_named_with_the_ones_it_has(self) -> None:
        cell = load_cell(_cell(self.folder, "arms:\n  halter: console_dummy\n"), root=self.repo)
        with self.assertRaisesRegex(ConfigError, "has no arm 'werker': its arms are halter"):
            cell.arm("werker")


if __name__ == "__main__":
    unittest.main()
