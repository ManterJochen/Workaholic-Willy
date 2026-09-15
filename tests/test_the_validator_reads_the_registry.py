"""``python -m src.config`` reads the gripper registry, so a broken hand file no longer validates green.

Before lane (i) the validator ran the profile chain and ``load_config`` and nothing else (``ConfigTree.load``),
so a registry file that could not be read, an alias two hands claimed, or a cell naming a hand no file
describes all printed ``OK``. The registry was the one part of the tree its own validator never opened.

What it reads, and what it leaves to the build (owner, Step 4 Q5: the hand refusal belongs at build, where a
model resolves): the validator lists the hands whenever the tree has a ``grippers/`` directory, and resolves the
hand a cell names with ``aliases=False``. A tree with no registry that names no hand still validates, which is
every generated ``data_example`` tree; a tree that names a hand and has no registry does not.

Each case copies the shipped tree, breaks it in one way, and runs ``main`` in process with ``WILLY_PROFILE``
cleared, as tests/test_config_encoding_refusal.py does.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src.config.__main__ import main

_DATA = Path(__file__).resolve().parents[1] / "config"
_LAYER = "handcheck"


class TheValidatorReadsTheRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._environment = mock.patch.dict(os.environ)
        self._environment.start()
        os.environ.pop("WILLY_PROFILE", None)
        self._tmp = tempfile.mkdtemp(prefix="willy_validator_registry_")
        self.root = Path(self._tmp) / "data"
        shutil.copytree(_DATA, self.root)
        self.shipped_2f85 = (self.root / "grippers" / "robotiq_2f85.yaml").read_text(encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._environment.stop()

    def _validate(self, *extra: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--data", str(self.root), *extra])
        return code, out.getvalue() + err.getvalue()

    def _name_the_hand(self, model: str) -> tuple[int, str]:
        """A one-line profile layer that names the cell's hand, validated under that layer."""
        (self.root / "robot" / f"robot.{_LAYER}.yaml").write_text(
            f"robot:\n  gripper:\n    model: {model}\n", encoding="utf-8",
        )
        return self._validate("--profile", _LAYER)

    def test_the_shipped_tree_validates_and_names_its_hands(self) -> None:
        code, printed = self._validate()
        self.assertEqual(code, 0, printed)
        self.assertIn("hands: robotiq_2f85, robotiq_hande, schunk_egu50", printed)

    def test_a_malformed_hand_file_fails_validation_and_names_the_file(self) -> None:
        (self.root / "grippers" / "robotiq_2f85.yaml").write_text("gripper: [never closed\n", encoding="utf-8")
        code, printed = self._validate()
        self.assertEqual(code, 1, printed)
        self.assertIn("robotiq_2f85.yaml", printed)

    def test_an_alias_two_hands_claim_fails_validation(self) -> None:
        (self.root / "grippers" / "other_hand.yaml").write_text(
            self.shipped_2f85.replace("model: robotiq_2f85", "model: other_hand"), encoding="utf-8",
        )
        code, printed = self._validate()
        self.assertEqual(code, 1, printed)
        self.assertIn("claimed by", printed)

    def test_a_named_hand_no_file_defines_fails_validation(self) -> None:
        code, printed = self._name_the_hand("no_such_hand")
        self.assertEqual(code, 1, printed)
        self.assertIn("no_such_hand", printed)

    def test_the_short_name_fails_validation_and_names_the_model(self) -> None:
        code, printed = self._name_the_hand("2f85")
        self.assertEqual(code, 1, printed)
        self.assertIn("robotiq_2f85", printed)

    def test_a_named_hand_the_registry_resolves_validates(self) -> None:
        """The control on the named path: a hand the registry holds is not refused."""
        code, printed = self._name_the_hand("robotiq_hande")
        self.assertEqual(code, 0, printed)

    def test_a_tree_without_a_registry_that_names_no_hand_still_validates(self) -> None:
        """The control, green before and after: every generated data_example tree has no grippers/ directory."""
        shutil.rmtree(self.root / "grippers")
        code, printed = self._validate()
        self.assertEqual(code, 0, printed)

    def test_a_tree_without_a_registry_that_names_a_hand_fails_validation(self) -> None:
        shutil.rmtree(self.root / "grippers")
        code, printed = self._name_the_hand("robotiq_2f85")
        self.assertEqual(code, 1, printed)
        self.assertIn("no gripper registry", printed)


if __name__ == "__main__":
    unittest.main()
