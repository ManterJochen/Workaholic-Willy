"""The bank may be told what the labeller found, instead of guessing from the bounding box.

⛔ `min(extent_mm) <= max_jaw_span_mm` asks whether the object's HULL fits between the fingers. A jaw
closes on a LINE through the object, so a round pot 86 mm across has plenty of chords under 85 and a
60 mm sphere has none longer than 60. The proxy is wrong in both directions, measured 2026-09-01 by
putting every mesh through the real labeller alone and upright:

    gso   31 meshes the proxy REJECTED, the labeller finds grasps on 31   (100 %)
    ycb   58 meshes the proxy ACCEPTED, the labeller finds grasps on 35   (over-states by 23)

`gso_Ecoforms_Plant_Pot_GP9` carries 1,044 labels and missed the span limit by ONE millimetre;
`ycb_021_bleach_cleanser` carries 1,373 and missed the LENGTH limit by one.

The screen is written by `datagen screen-meshes`. Passing none keeps the shipped proxy, so every
bank built before this existed is byte-identical.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import trimesh

from datagen.assets.library import MeshLibrary
from datagen.assets.meshes import MeshAssetBank


def _write(directory: Path, stem: str, extent_m: tuple[float, float, float]) -> None:
    """A box of a given size, in METRES, which is how `measure_mesh` reads a file."""
    directory.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=np.asarray(extent_m, dtype=float))
    box.export(directory / f"{stem}.obj")


class TheScreenOverridesTheProxyTests(unittest.TestCase):
    #: A slab 0.09 m on its shortest axis: 90 mm, so the 85 mm proxy refuses it outright.
    FAT = (0.09, 0.12, 0.20)
    #: A slab 0.04 m on its shortest axis, which the proxy happily accepts.
    THIN = (0.04, 0.10, 0.18)

    def _bank(self, tmp: str, **kwargs) -> MeshAssetBank:
        _write(Path(tmp) / "gso", "fat", self.FAT)
        _write(Path(tmp) / "gso", "thin", self.THIN)
        return MeshAssetBank(MeshLibrary(tmp), max_extent_mm=300.0, max_jaw_span_mm=85.0, **kwargs)

    def test_without_a_screen_the_proxy_still_decides(self) -> None:
        """The default has to be byte-identical, or every existing bank silently changes."""
        with TemporaryDirectory() as tmp:
            ids = {a.asset_id for a in self._bank(tmp).load("gso")}
            self.assertEqual(ids, {"gso_thin"}, "the shipped proxy no longer refuses a 90 mm slab")

    def test_a_positive_verdict_admits_what_the_proxy_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = self._bank(tmp, jaw_screen={"gso_fat": 1044, "gso_thin": 12})
            self.assertEqual({a.asset_id for a in bank.load("gso")}, {"gso_fat", "gso_thin"})

    def test_a_zero_verdict_refuses_what_the_proxy_admitted(self) -> None:
        """The other direction, which is the one the proxy gets wrong more often."""
        with TemporaryDirectory() as tmp:
            bank = self._bank(tmp, jaw_screen={"gso_fat": 0, "gso_thin": 0})
            self.assertEqual({a.asset_id for a in bank.load("gso")}, set())

    def test_a_mesh_the_screen_does_not_mention_falls_back_rather_than_vanishing(self) -> None:
        """A stale screen must shrink the bank VISIBLY, not quietly.

        Dropping unjudged meshes would make a screen written before the last fetch look like a
        library that lost half its objects, with no line anywhere saying why.
        """
        with TemporaryDirectory() as tmp:
            bank = self._bank(tmp, jaw_screen={"gso_fat": 900})
            ids = {a.asset_id for a in bank.load("gso")}
            self.assertIn("gso_fat", ids, "the screen's own verdict was ignored")
            self.assertIn("gso_thin", ids, "an unjudged mesh was dropped instead of falling back")

    def test_the_length_limit_is_untouched_by_the_screen(self) -> None:
        """The screen replaces the JAW question only. Fitting the workspace is a different one."""
        with TemporaryDirectory() as tmp:
            _write(Path(tmp) / "gso", "long", (0.05, 0.10, 0.40))
            bank = MeshAssetBank(MeshLibrary(tmp), max_extent_mm=300.0, max_jaw_span_mm=85.0,
                                 jaw_screen={"gso_long": 5000})
            self.assertEqual({a.asset_id for a in bank.load("gso")}, set(),
                             "a 400 mm object entered a 300 mm workspace on a jaw verdict")


class TheScreenFileIsShapedAsTheBankExpectsTests(unittest.TestCase):
    def test_the_screen_writes_what_the_bank_reads(self) -> None:
        """The two halves live in different modules; this pins the fields that join them."""
        import inspect

        from datagen.assets import prepare  # noqa: PLC0415

        source = inspect.getsource(prepare)
        self.assertIn('"asset_id"', source, "the screen must key rows by asset id")
        self.assertIn('"jaw"', source, "the bank reads the jaw count from each row")
        self.assertIn('"unclosable"', source,
                      "a mesh that will not load must get a STATUS, never a count: the screen this "
                      "replaces wrote a constant 24 for 161 such meshes and it read as a measurement")
        self.assertNotIn('"jaw": 24', source, "a sentinel count is exactly the defect being replaced")
        self.assertIn('"density": density', source,
                      "a verdict without the density it was taken at cannot be checked: the same "
                      "meshes score 1,960 and 84 labels at `grid` and `default`")
