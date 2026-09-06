from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utility.io import atomic_write_text, dump_json, load_json
from src.utility.unit_scaling import unit_scaling


class UtilityBoundaryTests(unittest.TestCase):
    def test_unit_scaling_supported_units(self) -> None:
        self.assertEqual(unit_scaling("mm"), 1.0)
        self.assertEqual(unit_scaling(" cm "), 0.1)
        self.assertEqual(unit_scaling("M"), 0.001)

    def test_unit_scaling_rejects_invalid_unit(self) -> None:
        with self.assertRaises(ValueError):
            unit_scaling("inch")

    def test_atomic_text_write_and_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "data.json"
            atomic_write_text(path, "placeholder")
            dump_json({"b": 2, "a": 1}, path, sort_keys=True)

            self.assertEqual(load_json(path), {"a": 1, "b": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()