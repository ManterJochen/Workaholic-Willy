"""Any vendor USD bakes from the command line, with its control never skipped silently (customer chain lane C2d, C1c).

The USD baker knew two hands, typed into a catalogue, and every bake read Isaac's 2F-85 as its control (finding 1 of the
audit of 2026-09-17), so a customer with a USD and no Isaac assets could not bake at all. The stage READER is unchanged;
what is new is everything around it, and that is tested here with the reader standing in. The reader itself needs pxr,
which the last class exercises on a synthetic stage wherever pxr loads.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"
_HANDE_ARGS = ["--bodies", "housing,left,right", "--closing", "+Z", "--approach", "+Y", "--binormal", "-X",
               "--mount-face-mm", "-86.10", "--origin", "mounting_face"]


def _baker() -> Any:
    spec = importlib.util.spec_from_file_location("_baker_under_test", _REPO / "scripts" / "grippers" / "bake_gripper_variant.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hande_vendor_mm() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The committed Hand-E in the Isaac Hand-E asset's axes, millimetres: what the stage reader returns for that asset."""
    rotation = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    with np.load(_DATA / "robotiq_hande_hand_meshes.npz", allow_pickle=True) as data:
        out = {}
        for part in ("gripper", "lfinger", "rfinger"):
            points = np.asarray(data[f"{part}__v"], dtype=np.float64) @ rotation
            points[:, 1] += -86.10
            out[part] = (points, np.asarray(data[f"{part}__f"]))
    return out


def _run(module: Any, argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            out.write(str(exc.code))
    return int(code), out.getvalue()


class ABakeFromTheCommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = self.folder / "data"
        self.data.mkdir()
        shutil.copy(_DATA / "bundles.json", self.data / "bundles.json")
        self.baker = _baker()
        self.enterContext(mock.patch.object(self.baker, "_read_usd_parts", lambda usd, bodies, *, scale_to_mm: _hande_vendor_mm()))

    def test_a_hand_outside_the_catalogue_bakes_and_records_its_row(self) -> None:
        out = self.data / "acme_2f_hand_meshes.npz"
        code, said = _run(self.baker, ["--usd", str(self.folder / "acme.usd"), "--hand", "acme_2f", *_HANDE_ARGS,
                                       "--out", str(out), "--skip-control", "CI has no Isaac assets", "--write"])
        self.assertEqual(code, 0, said)
        with np.load(_DATA / "robotiq_hande_hand_meshes.npz", allow_pickle=True) as committed, \
                np.load(out, allow_pickle=True) as written:
            for part in ("gripper", "lfinger", "rfinger"):
                np.testing.assert_allclose(written[f"{part}__v"], committed[f"{part}__v"], rtol=0.0, atol=1e-9)
                np.testing.assert_array_equal(written[f"{part}__f"], committed[f"{part}__f"])
            self.assertEqual(str(written["gripper__origin"][0]), "mounting_face")
            recipe = json.loads(str(written["hand__recipe"][0]))
        self.assertEqual(recipe["control"], {"skipped": "CI has no Isaac assets"})
        row = json.loads((self.data / "bundles.json").read_text(encoding="utf-8"))["bundles"]["acme_2f_hand_meshes.npz"]
        self.assertEqual(row["source"], "standalone_usd")
        self.assertIn("acme.usd", row["note"])
        self.assertIn("scripts/grippers/measure_jaw_from_bundle.py", said)
        self.assertIn("scripts/curobo/fit_cover_spheres.py --hand acme_2f --write", said)
        self.assertNotIn("build_gripper_spheres", said)

    def test_the_control_is_never_skipped_silently(self) -> None:
        out = self.data / "acme_2f_hand_meshes.npz"
        empty = self.folder / "no_assets"
        empty.mkdir()
        with mock.patch.dict("os.environ", {"WILLY_ISAAC_ASSETS": str(empty / "missing")}), \
                mock.patch.object(self.baker, "_ASSET_HINTS", ()):
            code, said = _run(self.baker, ["--usd", str(self.folder / "acme.usd"), "--hand", "acme_2f", *_HANDE_ARGS,
                                           "--out", str(out), "--write"])
        self.assertNotEqual(code, 0)
        self.assertIn("WILLY_ISAAC_ASSETS", said)
        self.assertIn("--skip-control", said)
        self.assertFalse(out.exists())
        code, said = _run(self.baker, ["--usd", str(self.folder / "acme.usd"), "--hand", "acme_2f", *_HANDE_ARGS,
                                       "--out", str(out), "--skip-control", "   ", "--write"])
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_every_placing_number_is_stated(self) -> None:
        code, said = _run(self.baker, ["--usd", str(self.folder / "acme.usd"), "--hand", "acme_2f", "--bodies", "a,b,c",
                                       "--closing", "+Z", "--approach", "+Y", "--binormal", "-X", "--skip-control", "x"])
        self.assertEqual(code, 2)
        for flag in ("--mount-face-mm", "--origin"):
            self.assertIn(flag, said)

    def test_a_rebake_of_a_preset_keeps_every_array_but_its_recipe(self) -> None:
        """⭐ C1c's control: a re-bake records a new row, because its recipe is new, and every array the guard composes
        comes back as it was."""
        out = self.data / "robotiq_hande_hand_meshes.npz"
        code, said = _run(self.baker, ["robotiq_hande", "--usd", str(self.folder / "hande.usd"), "--out", str(out),
                                       "--skip-control", "the reader stands in", "--write"])
        self.assertEqual(code, 0, said)
        with np.load(_DATA / "robotiq_hande_hand_meshes.npz", allow_pickle=True) as committed, \
                np.load(out, allow_pickle=True) as written:
            self.assertEqual(sorted(set(written.files) - {"hand__recipe"}), sorted(committed.files))
            for key in committed.files:
                if key.endswith("__v"):
                    np.testing.assert_allclose(written[key], committed[key], rtol=0.0, atol=1e-9)
                else:
                    np.testing.assert_array_equal(written[key], committed[key])
        row = json.loads((self.data / "bundles.json").read_text(encoding="utf-8"))["bundles"]["robotiq_hande_hand_meshes.npz"]
        self.assertEqual(row["source"], "standalone_usd")


class APresetIsACommandLineTests(unittest.TestCase):
    def test_a_preset_is_the_same_cli_line(self) -> None:
        code, said = _run(_baker(), ["--list"])
        self.assertEqual(code, 0)
        self.assertIn("--closing +Z --approach +Y --binormal -X --mount-face-mm -86.1 --origin mounting_face", said)
        self.assertIn("--closing +Y --approach +Z --binormal +X --mount-face-mm 0 --origin flange", said)


def _pxr_loads() -> tuple[bool, str]:
    try:
        from pxr import Usd  # noqa: F401
    except Exception as error:  # noqa: BLE001 (reported, not failed)
        return False, f"{type(error).__name__}: {error}"
    return True, ""


_PXR, _PXR_WHY_NOT = _pxr_loads()


class TheStageReaderTests(unittest.TestCase):
    """The reader on a stage this test authors. ⚠ pxr is declared (requirements.txt); a box where the OPERATING
    SYSTEM refuses its DLL (Smart App Control, measured on this box on 2026-09-17) skips with the OS's own words, and
    any other reason it does not load fails."""

    def setUp(self) -> None:
        if not _PXR:
            if "blockiert" in _PXR_WHY_NOT or "blocked" in _PXR_WHY_NOT or "4551" in _PXR_WHY_NOT:
                self.skipTest(f"pxr is installed and the operating system refuses to load it: {_PXR_WHY_NOT}")
            self.fail(f"pxr will not load, and it is declared in requirements.txt: {_PXR_WHY_NOT}")

    def _stage(self, path: Path, *, metres_per_unit: float, twice: bool = False) -> None:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, metres_per_unit)
        root = UsdGeom.Xform.Define(stage, "/hand")
        stage.SetDefaultPrim(root.GetPrim())
        for name, (vertices, faces) in zip(("housing", "left", "right"), _hande_vendor_mm().values(), strict=True):
            mesh = UsdGeom.Mesh.Define(stage, f"/hand/{name}/mesh")
            mesh.CreatePointsAttr([tuple(float(v) / (metres_per_unit * 1000.0) for v in row) for row in vertices])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([int(i) for i in np.asarray(faces).reshape(-1)])
        if twice:
            UsdGeom.Xform.Define(stage, "/other/left")
        stage.GetRootLayer().Save()

    def test_an_ambiguous_prim_name_and_an_unstated_unit_are_refused(self) -> None:
        baker = _baker()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        twice = folder / "twice.usda"
        self._stage(twice, metres_per_unit=1.0, twice=True)
        with self.assertRaises(SystemExit) as caught:
            baker._read_usd_parts(twice, ("housing", "left", "right"), scale_to_mm=None)
        self.assertIn("/other/left", str(caught.exception.code))
        centimetres = folder / "centimetres.usda"
        self._stage(centimetres, metres_per_unit=0.01)
        with self.assertRaises(SystemExit) as caught:
            baker._read_usd_parts(centimetres, ("housing", "left", "right"), scale_to_mm=None)
        self.assertIn("--scale-to-mm 10", str(caught.exception.code))
        parts = baker._read_usd_parts(centimetres, ("housing", "left", "right"), scale_to_mm=10.0)
        np.testing.assert_allclose(parts["gripper"][0], _hande_vendor_mm()["gripper"][0], rtol=0.0, atol=1e-4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
