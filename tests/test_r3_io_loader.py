"""R3.4a — the consolidated fail-closed JSONL loader (rl/_io.load_jsonl).

Pins the fail-closed contract that promotion + train_recovery now inherit (they used to crash late on a raw
JSONDecodeError + silently append non-dict lines): a malformed line + a non-object line each raise a typed
ValueError; valid packs round-trip; both str and Path inputs work. Also asserts dataset.load_jsonl is now the
SAME object (the re-export), so the ~9 `from .dataset import load_jsonl` sites resolve unchanged.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.robot.grasping.rl._io import load_jsonl
from src.robot.grasping.rl.dataset import load_jsonl as dataset_load_jsonl


class IoLoaderTests(unittest.TestCase):
    def test_dataset_reexports_shared_loader(self) -> None:
        self.assertIs(dataset_load_jsonl, load_jsonl)

    def test_valid_pack_roundtrips_str_and_path(self) -> None:
        with TemporaryDirectory() as d:
            p = Path(d) / "ok.jsonl"
            p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")  # blank line skipped
            self.assertEqual(load_jsonl(p), [{"a": 1}, {"b": 2}])
            self.assertEqual(load_jsonl(str(p)), [{"a": 1}, {"b": 2}])  # str|Path normalisation

    def test_malformed_line_raises(self) -> None:
        with TemporaryDirectory() as d:
            p = Path(d) / "bad.jsonl"
            p.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(p)

    def test_non_object_line_raises(self) -> None:
        with TemporaryDirectory() as d:
            p = Path(d) / "arr.jsonl"
            p.write_text('{"a": 1}\n[1, 2, 3]\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(p)


if __name__ == "__main__":
    unittest.main()
