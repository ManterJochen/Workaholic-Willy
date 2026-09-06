"""Validate the shipped grasping presets against the Pydantic schema.

Presets are deep-merged onto ``robot.grasping`` *outside* Pydantic, so a typo'd key
would otherwise be applied silently. ``validate_all_presets()`` re-validates each
preset through ``RobotConfig`` (``extra='forbid'``); this test is the CI guard that a
committed preset typo fails the build.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from src.robot.grasping.replay import presets
from src.robot.grasping.replay.presets import (
    list_presets,
    validate_all_presets,
    validate_preset,
)


class PresetSchemaValidationTests(unittest.TestCase):
    def test_all_shipped_presets_are_schema_valid(self) -> None:
        validated = validate_all_presets()
        self.assertEqual(set(validated), set(list_presets()))
        self.assertEqual(
            set(validated), {"easy", "dense_clutter", "verification_heavy"}
        )

    def test_typo_preset_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # `defualt_mode` is a typo of `default_mode` — an unknown grasping key.
            (Path(tmp) / "typo.yaml").write_text(
                "grasping:\n  defualt_mode: easy\n", encoding="utf-8"
            )
            with mock.patch.object(presets, "_PRESETS_DIR", Path(tmp)):
                self.assertIn("typo", list_presets())
                with self.assertRaises(ValidationError):
                    validate_preset("typo")


if __name__ == "__main__":
    unittest.main()
