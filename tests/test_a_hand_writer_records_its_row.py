"""A hand writer records its bundle's row in bundles.json, so a customer never hand-edits the index (lane C1b, C1d).

A body written from registry numbers landed beside ``bundles.json`` and nothing recorded it, so the suite refused the
very bundle a customer had just written (finding 4 of the audit of 2026-09-17). And a fourth registry hand turned the
suite red on a test that pinned exactly three names (finding 3), whether or not that hand had a body.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"
_REGISTRY = _REPO / "config" / "grippers"


def _by_path(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ruler(path: Path) -> str:
    """The suite's own hash, not the writer's."""
    return _by_path("_provenance_ruler", "tests/test_bundle_provenance.py").arrays_sha256(path)


class _Scratch:
    """A registry holding the shipped hands plus acme_2f (the EGU-50's numbers under another name), and a data folder
    holding a copy of the committed bundles.json."""

    def __init__(self, case: unittest.TestCase, *, finger_width_mm: float | None = None) -> None:
        root = Path(case.enterContext(tempfile.TemporaryDirectory()))
        self.data_dir = root / "config"
        grippers = self.data_dir / "grippers"
        shutil.copytree(_REGISTRY, grippers)
        text = (_REGISTRY / "schunk_egu50.yaml").read_text(encoding="utf-8").replace("model: schunk_egu50", "model: acme_2f")
        if finger_width_mm is not None:
            text = text.replace("finger_width_mm: 25.0", f"finger_width_mm: {finger_width_mm}")
        (grippers / "acme_2f.yaml").write_text(text, encoding="utf-8")
        self.bundles = root / "data"
        self.bundles.mkdir()
        shutil.copy(_DATA / "bundles.json", self.bundles / "bundles.json")
        case.enterContext(mock.patch("src.config.grippers._DEFAULT_DATA_DIR", self.data_dir))

    def write(self, *extra: str) -> tuple[int, str]:
        out = io.StringIO()
        target = self.bundles / "acme_2f_hand_meshes.npz"
        argv = ["acme_2f", "--inflation-mm", "10", "--origin", "flange", "--out", str(target), *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                code = _by_path("_dims_cli_for_rows", "scripts/grippers/write_hand_from_dimensions.py").main(argv)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                out.write(str(exc.code))
        return int(code), out.getvalue()


class TheWriterRecordsItsRowTests(unittest.TestCase):
    def test_the_writer_records_the_row_it_wrote(self) -> None:
        import json

        scratch = _Scratch(self)
        code, said = scratch.write()
        self.assertEqual(code, 0, said)
        row = json.loads((scratch.bundles / "bundles.json").read_text(encoding="utf-8"))["bundles"]["acme_2f_hand_meshes.npz"]
        self.assertEqual((row["hand"], row["source"]), ("acme_2f", "registry_dimensions"))
        self.assertIn("10 mm", row["note"])
        self.assertEqual(row["arrays_sha256"], _ruler(scratch.bundles / "acme_2f_hand_meshes.npz"))

    def test_a_refused_write_changes_no_row(self) -> None:
        """⭐ THE CONTROL: a second write is refused, and the index is not touched by it."""
        scratch = _Scratch(self)
        self.assertEqual(scratch.write()[0], 0)
        before = (scratch.bundles / "bundles.json").read_bytes()
        self.assertNotEqual(scratch.write()[0], 0)
        self.assertEqual((scratch.bundles / "bundles.json").read_bytes(), before)

    def test_recording_keeps_the_committed_layout(self) -> None:
        """⭐ THE CONTROL: the writer's layout is the committed file's, so one recorded row cannot rewrite every line."""
        import json

        from src.robot.safety.planning.bundle_index import index_bytes

        raw = (_DATA / "bundles.json").read_bytes()
        self.assertEqual(index_bytes(json.loads(raw.decode("utf-8")), crlf=b"\r\n" in raw), raw)

    def test_a_source_the_index_does_not_declare_is_refused(self) -> None:
        from src.robot.safety.planning.bundle_index import BundleRowRefused, record_hand_bundle

        scratch = _Scratch(self)
        self.assertEqual(scratch.write()[0], 0)
        with self.assertRaises(BundleRowRefused) as caught:
            record_hand_bundle(scratch.bundles / "acme_2f_hand_meshes.npz", source="guessed", note="a note")
        self.assertIn("registry_dimensions", str(caught.exception))


class ACommittedDimensionsBodyRebuildsTests(unittest.TestCase):
    def test_every_committed_dimensions_bundle_rebuilds_from_its_registry_file(self) -> None:
        import json

        import numpy as np

        writer = _by_path("_dims_writer_for_rebuild", "src/robot/safety/planning/robot/hand_from_dimensions.py")
        rows = json.loads((_DATA / "bundles.json").read_text(encoding="utf-8"))["bundles"]
        for name, row in sorted(rows.items()):
            if row["source"] != "registry_dimensions":
                continue
            with self.subTest(bundle=name), tempfile.TemporaryDirectory() as folder:
                with np.load(_DATA / name, allow_pickle=True) as held:
                    inflation = float(np.asarray(held["hand__inflation_mm"]).reshape(-1)[0])
                    origin = str(np.asarray(held["gripper__origin"]).reshape(-1)[0])
                rebuilt = Path(folder) / name
                writer.write_bundle(row["hand"], rebuilt, inflation_mm=inflation, origin=origin)
                self.assertEqual(_ruler(rebuilt), row["arrays_sha256"])

    def test_a_moved_registry_number_no_longer_rebuilds_the_bundle(self) -> None:
        """⭐ THE CONTROL: the rebuild really depends on the registry, so an edited number is seen."""
        scratch = _Scratch(self)
        self.assertEqual(scratch.write()[0], 0)
        first = _ruler(scratch.bundles / "acme_2f_hand_meshes.npz")
        moved = _Scratch(self, finger_width_mm=26.0)
        self.assertEqual(moved.write()[0], 0)
        self.assertNotEqual(_ruler(moved.bundles / "acme_2f_hand_meshes.npz"), first)


class AFourthHandTests(unittest.TestCase):
    """C1d. The shipped hands stay pinned as present; every registry hand has a body."""

    @staticmethod
    def _without_a_body(registry: Path, data: Path) -> list[str]:
        from src.config.grippers import available_grippers

        return [hand for hand in available_grippers(registry.parent) if not (data / f"{hand}_hand_meshes.npz").is_file()]

    def test_a_fourth_hand_with_a_body_is_a_registry_hand_like_the_others(self) -> None:
        scratch = _Scratch(self)
        self.assertEqual(scratch.write()[0], 0)
        for path in _DATA.glob("*_hand_meshes.npz"):
            shutil.copy(path, scratch.bundles / path.name)
        self.assertEqual(self._without_a_body(scratch.data_dir / "grippers", scratch.bundles), [])

    def test_a_fourth_hand_with_no_body_is_named(self) -> None:
        """⭐ THE CONTROL: the property sees a registry hand nobody wrote a body for."""
        scratch = _Scratch(self)
        for path in _DATA.glob("*_hand_meshes.npz"):
            shutil.copy(path, scratch.bundles / path.name)
        self.assertEqual(self._without_a_body(scratch.data_dir / "grippers", scratch.bundles), ["acme_2f"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
