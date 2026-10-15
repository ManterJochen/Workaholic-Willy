"""`real_cell --check` prints, after the checklist, what each operator function can do on the cell (P0i i4).

Owner decision 3 (2026-09-26): the checklist rows, then the ability rows, and the exit code stays the checklist's: a UR
cell that cannot teach is not a cell that cannot pick. The rows are ``Cell.abilities()`` of the same tree, asked of the
arm it builds, never connected, printed as that report renders itself.
"""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest import mock

from src.config import ConfigTree
from src.robot.execution.cell import Cell
from src.robot.execution.real_cell.__main__ import main


def _check(*argv: str) -> tuple[int, str]:
    said = io.StringIO()
    env = {k: v for k, v in os.environ.items() if k != "WILLY_PROFILE"}
    with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(said):
        code = main(["--check", *argv])
    return code, said.getvalue()


class TheCheckPrintsTheAbilitiesTests(unittest.TestCase):
    def test_the_rows_follow_the_checklist_and_are_the_cells_own(self) -> None:
        code, said = _check("--profile", "console_dummy")
        self.assertEqual(code, 0, said)
        checklist, _, abilities = said.partition("\n=== ABILITIES ===")
        self.assertIn("0 blocking", checklist)
        expected = Cell.from_tree(ConfigTree.from_directory(profile="console_dummy").load()).abilities().render()
        self.assertIn(expected, abilities)
        self.assertIn("[ no ] teach", abilities)

    def test_a_refused_ability_changes_no_exit_code(self) -> None:
        """console_dummy refuses six functions and still checks green; the base tree blocks for its checklist alone."""
        code, said = _check("--profile", "console_dummy")
        self.assertEqual(code, 0)
        self.assertIn("6 refused", said)
        code, said = _check()
        self.assertEqual(code, 1)
        self.assertIn("=== ABILITIES ===", said)
        self.assertIn("as move", said, "an arm that does not build says so once, for every row")

    def test_the_refusal_to_continue_names_the_checklist(self) -> None:
        """A full run that stops on the checklist says which rows stopped it: the ability rows block nothing."""
        said = io.StringIO()
        env = {k: v for k, v in os.environ.items() if k != "WILLY_PROFILE"}
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(said):
            code = main(["--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("fix the BLOCK rows of the checklist above; no ability row blocks.", said.getvalue())

    def test_rows_that_cannot_be_read_leave_the_checklist_the_verdict(self) -> None:
        """Found by review, 2026-09-26: a row that raised ended --check in a traceback, whatever the checklist said."""
        with mock.patch.object(Cell, "abilities", side_effect=ValueError("a body link that does not resolve")):
            code, said = _check("--profile", "console_dummy")
        self.assertEqual(code, 0, said)
        self.assertIn("the abilities could not be read: ValueError: a body link that does not resolve", said)


if __name__ == "__main__":
    unittest.main()
