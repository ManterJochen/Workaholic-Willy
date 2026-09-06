"""`assets.jaw_screen_path` has to reach the bank, and a re-run of the screen has to be seen.

⛔ THE BANK IS CACHED BEHIND A KEY. The comment already at that site says why it matters: "a bank
cached under a key that does not mention a filter is served to the next config that changes it, and
the filter reads as inert -- the exact shape that cost this repo a 103-scene run when `depth_source`
was declared, decoded and never popped."

A screen is worse than a flag for this, because it is normally re-run IN PLACE under the same
filename. Keying on the path alone would serve a bank built from yesterday's verdict and nothing
would say so, so the key carries the file's content stamp.
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import trimesh

from datagen.config import DatagenConfig
from datagen.scenes.layout import _mesh_bank, load_jaw_screen, reset_mesh_bank


def _library(root: Path) -> None:
    (root / "gso").mkdir(parents=True, exist_ok=True)
    for stem, extents in (("fat", (0.09, 0.12, 0.20)), ("thin", (0.04, 0.10, 0.18))):
        trimesh.creation.box(extents=np.asarray(extents)).export(root / "gso" / f"{stem}.obj")


def _screen(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


class TheScreenPathReachesTheBankTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_mesh_bank()

    def tearDown(self) -> None:
        reset_mesh_bank()

    def test_a_row_without_an_ok_status_carries_no_verdict(self) -> None:
        """"Not measured" must not be read as "not graspable"."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.json"
            _screen(path, [{"asset_id": "gso_a", "status": "ok", "jaw": 7},
                           {"asset_id": "gso_b", "status": "unclosable"},
                           {"asset_id": "gso_c", "status": "ok", "jaw": 0}])
            screen = load_jaw_screen(str(path))
            assert screen is not None
            self.assertEqual(screen, {"gso_a": 7, "gso_c": 0})
            self.assertNotIn("gso_b", screen, "an unloadable mesh must fall back, not be refused")

    def test_no_path_means_no_screen(self) -> None:
        self.assertIsNone(load_jaw_screen(""))

    def test_a_missing_file_falls_back_rather_than_raising(self) -> None:
        """A build must not die on a stale path, but it must say so; the warning is the contract."""
        with self.assertLogs("SceneLayout", level="WARNING") as caught:
            self.assertIsNone(load_jaw_screen("nowhere/at/all.json"))
        self.assertTrue(any("jaw_screen_path" in line for line in caught.output))

    def test_the_configured_screen_changes_which_meshes_the_bank_keeps(self) -> None:
        """The differential: the flag is set, so the bank has to come out different."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _library(root)
            path = root / "screen.json"
            _screen(path, [{"asset_id": "gso_fat", "status": "ok", "jaw": 900},
                           {"asset_id": "gso_thin", "status": "ok", "jaw": 0}])
            base = DatagenConfig()
            without = base.model_copy(update={"assets": base.assets.model_copy(update={
                "max_mesh_extent_mm": 300.0, "max_jaw_span_mm": 85.0})})
            withit = base.model_copy(update={"assets": base.assets.model_copy(update={
                "max_mesh_extent_mm": 300.0, "max_jaw_span_mm": 85.0,
                "jaw_screen_path": str(path)})})

            import datagen.scenes.layout as layout
            original = layout.MeshAssetBank

            def bank_of(config):
                reset_mesh_bank()
                from datagen.assets.library import MeshLibrary
                layout.MeshAssetBank = lambda **kw: original(MeshLibrary(root), **kw)  # noqa: E731
                try:
                    return {a.asset_id for a in _mesh_bank(config).load("gso")}
                finally:
                    layout.MeshAssetBank = original

            self.assertEqual(bank_of(without), {"gso_thin"}, "the proxy no longer decides by default")
            self.assertEqual(bank_of(withit), {"gso_fat"},
                             "the configured screen never reached the bank")

    def test_re_running_the_screen_in_place_invalidates_the_cache(self) -> None:
        """The trap this key exists for: same filename, new verdict, stale bank."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _library(root)
            path = root / "screen.json"
            base = DatagenConfig()
            config = base.model_copy(update={"assets": base.assets.model_copy(update={
                "max_mesh_extent_mm": 300.0, "max_jaw_span_mm": 85.0,
                "jaw_screen_path": str(path)})})

            import datagen.scenes.layout as layout
            original = layout.MeshAssetBank
            from datagen.assets.library import MeshLibrary
            layout.MeshAssetBank = lambda **kw: original(MeshLibrary(root), **kw)  # noqa: E731
            try:
                _screen(path, [{"asset_id": "gso_fat", "status": "ok", "jaw": 900},
                               {"asset_id": "gso_thin", "status": "ok", "jaw": 0}])
                first = {a.asset_id for a in _mesh_bank(config).load("gso")}
                self.assertEqual(first, {"gso_fat"})

                time.sleep(0.01)          # a distinct mtime, since the size may not change
                _screen(path, [{"asset_id": "gso_fat", "status": "ok", "jaw": 000},
                               {"asset_id": "gso_thin", "status": "ok", "jaw": 900}])
                second = {a.asset_id for a in _mesh_bank(config).load("gso")}
                self.assertEqual(second, {"gso_thin"},
                                 "the bank was served from a cache keyed before the screen changed")
            finally:
                layout.MeshAssetBank = original
                reset_mesh_bank()


class TheDensityTravelsWithTheVerdictTests(unittest.TestCase):
    """⛔ A screen taken at `default` and one taken at `grid` are different measurements.

    MEASURED 2026-09-01, by running both by accident: over the same 28 meshes the same object
    scored 84 labels and 1,960, a factor of twenty, and the two files were indistinguishable. The
    cause was the shared `--density` flag, whose default is `default` because `label-grasps` at that
    setting reproduces every label set this repo has written. A screen inheriting it silently
    produced a much weaker verdict than the corpus would get.

    Two halves, both pinned here: a screen command defaults to the density the corpus uses, and the
    file it writes says which one that was.
    """

    def setUp(self) -> None:
        reset_mesh_bank()

    def tearDown(self) -> None:
        reset_mesh_bank()

    def test_a_screen_command_does_not_inherit_the_label_default(self) -> None:
        import argparse

        from datagen.__main__ import _screen_density  # noqa: PLC0415

        # ⛔⛔ THE MECHANISM CHANGED ON 2026-09-04 AND THE OLD ONE WAS THE DEFECT. This asserted
        # `density="default" -> "grid"`, which is what the code did: it decided "the operator did not
        # choose" by comparing the value to the FLAG'S OWN DEFAULT STRING. So typing
        # `--density default` on a screen was indistinguishable from not typing it, and the explicit
        # choice was overridden. Same shape as deciding "not given" by scanning sys.argv.
        #
        # The flag carries no default now, so `None` means nobody said and a string means somebody
        # did. What this test guards is unchanged: a screen must not INHERIT the label default.
        self.assertEqual(_screen_density(argparse.Namespace(density=None)), "grid")
        self.assertEqual(_screen_density(argparse.Namespace(density="default")), "default")
        self.assertEqual(_screen_density(argparse.Namespace(density="dense")), "dense")
        self.assertEqual(_screen_density(argparse.Namespace(density="grid")), "grid")

    def test_the_flag_itself_carries_no_default(self) -> None:
        """The mechanism, pinned. A shared flag that carries one command's default cannot tell a
        chosen value from an inherited one, and the two commands here disagree by a factor of twenty
        on the same object: 84 labels against 1,960."""
        import sys

        from datagen.__main__ import build_parser  # noqa: PLC0415

        argv = sys.argv
        try:
            sys.argv = ["datagen"]
            parsed = build_parser().parse_args(["screen-meshes"])
        finally:
            sys.argv = argv
        self.assertIsNone(parsed.density)

    def test_the_written_file_records_the_density_and_the_poses(self) -> None:
        import inspect

        from datagen.assets import prepare  # noqa: PLC0415

        source = inspect.getsource(prepare.screen_meshes)
        self.assertIn('"density": density', source)
        self.assertIn('"rest_poses"', source)

    def test_a_file_without_a_density_still_loads_but_says_so(self) -> None:
        """Refusing one would strand every screen already on disk; reading one silently is worse."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            _screen(path, [{"asset_id": "gso_a", "status": "ok", "jaw": 3}])
            with self.assertLogs("SceneLayout", level="WARNING") as caught:
                screen = load_jaw_screen(str(path))
            self.assertEqual(screen, {"gso_a": 3})
            self.assertTrue(any("density" in line for line in caught.output))

    def test_a_file_with_a_density_loads_without_complaint(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stamped.json"
            path.write_text(json.dumps({
                "density": "grid", "rest_poses": ["upright"],
                "rows": [{"asset_id": "gso_a", "status": "ok", "jaw": 3}]}), encoding="utf-8")
            self.assertEqual(load_jaw_screen(str(path)), {"gso_a": 3})
