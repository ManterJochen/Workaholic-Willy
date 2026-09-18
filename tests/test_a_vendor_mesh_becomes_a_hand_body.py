"""A vendor STL or OBJ becomes the hand bundle a bake writes, with no Isaac and no USD (customer chain lane C2c).

Nothing in the tree turned a loose mesh into a guard body (finding 0 of the audit of 2026-09-17): the one mesh route
wrote a planner sphere map and no bundle, so a customer with a vendor file got a map and a guard on the capsule proxy.
``scripts/grippers/write_hand_from_mesh.py`` writes the bundle, through the one frame change every route shares.

The known answers are the committed hands themselves, exported into their vendor frames, so each test compares with
arrays that were measured long before this writer existed.
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
_HANDE_WORDS = ["--closing", "+Z", "--approach", "+Y", "--binormal", "-X"]
_F85_WORDS = ["--closing", "+Y", "--approach", "+Z", "--binormal", "+X"]


def _cli() -> Any:
    spec = importlib.util.spec_from_file_location("_mesh_writer_under_test", _REPO / "scripts" / "grippers" / "write_hand_from_mesh.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _committed(hand: str) -> dict[str, np.ndarray]:
    with np.load(_DATA / f"{hand}_hand_meshes.npz", allow_pickle=True) as data:
        return {key: np.array(data[key]) for key in data.files}


def _vendor(hand: str, closing: str, approach: str, binormal: str, mount_face_mm: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from src.robot.safety.planning.robot.hand_from_mesh import VendorAxes

    axes = VendorAxes.from_words(closing=closing, approach=approach, binormal=binormal)
    rotation = np.asarray(axes.rotation)
    committed = _committed(hand)
    out = {}
    for part in ("gripper", "lfinger", "rfinger"):
        points = np.asarray(committed[f"{part}__v"], dtype=np.float64) @ rotation
        points[:, axes.approach_index] += mount_face_mm
        out[part] = (points / 1000.0, np.asarray(committed[f"{part}__f"]))
    return out


def _obj_files(folder: Path, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Path]:
    files = {}
    for part, (vertices, faces) in parts.items():
        path = folder / f"{part}.obj"
        with path.open("w", encoding="ascii") as handle:
            for x, y, z in vertices:
                handle.write(f"v {float(x)!r} {float(y)!r} {float(z)!r}\n")
            for a, b, c in faces:
                handle.write(f"f {a + 1} {b + 1} {c + 1}\n")
        files[part] = path
    return files


def _stl_files(folder: Path, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Path]:
    import trimesh

    files = {}
    for part, (vertices, faces) in parts.items():
        path = folder / f"{part}.stl"
        path.write_bytes(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(file_type="stl"))
        files[part] = path
    return files


def _run(module: Any, argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            out.write(str(exc.code))
    return int(code), out.getvalue()


def _argv(files: dict[str, Path], words: list[str], *, hand: str = "acme_2f", scale: str = "1000",
          mount_face: str = "-86.10", origin: str = "mounting_face", out: Path | None = None, write: bool = True) -> list[str]:
    argv = [hand, "--gripper-mesh", str(files["gripper"]), "--lfinger-mesh", str(files["lfinger"]),
            "--rfinger-mesh", str(files["rfinger"]), "--scale-to-mm", scale, *words, "--mount-face-mm", mount_face,
            "--origin", origin]
    if out is not None:
        argv += ["--out", str(out)]
    if write:
        argv.append("--write")
    return argv


class TheCommittedHandsComeBackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cli = _cli()

    def test_the_hande_exported_as_obj_reproduces_its_bundle(self) -> None:
        files = _obj_files(self.folder, _vendor("robotiq_hande", "+Z", "+Y", "-X", -86.10))
        out = self.folder / "acme_2f_hand_meshes.npz"
        code, said = _run(self.cli, _argv(files, _HANDE_WORDS, out=out))
        self.assertEqual(code, 0, said)
        committed = _committed("robotiq_hande")
        with np.load(out, allow_pickle=True) as written:
            for part in ("gripper", "lfinger", "rfinger"):
                np.testing.assert_allclose(written[f"{part}__v"], committed[f"{part}__v"], rtol=0.0, atol=1e-6)
                np.testing.assert_array_equal(written[f"{part}__f"], committed[f"{part}__f"])
                self.assertEqual(int(written[f"{part}__frame"][0]), 6)
            self.assertEqual(str(written["gripper__origin"][0]), "mounting_face")
            recipe = json.loads(str(written["hand__recipe"][0]))
        self.assertEqual(recipe["writer"], "scripts/grippers/write_hand_from_mesh.py")
        self.assertIn("self_check_worst_mm", recipe)

    def test_the_2f85_exported_as_stl_reproduces_its_triangles(self) -> None:
        from scipy.spatial import cKDTree

        files = _stl_files(self.folder, _vendor("robotiq_2f85", "+Y", "+Z", "+X", 0.0))
        out = self.folder / "acme_2f_hand_meshes.npz"
        code, said = _run(self.cli, _argv(files, _F85_WORDS, mount_face="0", origin="flange", out=out))
        self.assertEqual(code, 0, said)
        committed = _committed("robotiq_2f85")
        with np.load(out, allow_pickle=True) as written:
            for part in ("gripper", "lfinger", "rfinger"):
                used = np.asarray(committed[f"{part}__v"], dtype=np.float64)[np.asarray(committed[f"{part}__f"]).reshape(-1)]
                got = np.asarray(written[f"{part}__v"], dtype=np.float64)
                self.assertLess(float(cKDTree(used).query(got)[0].max()), 1e-3, part)
                self.assertLess(float(cKDTree(got).query(used)[0].max()), 1e-3, part)
            self.assertEqual(str(written["gripper__origin"][0]), "flange")


class WhatIsRefusedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cli = _cli()
        self.files = _obj_files(self.folder, _vendor("robotiq_hande", "+Z", "+Y", "-X", -86.10))

    def test_a_right_handed_but_wrong_spelling_is_refused_by_the_finger_axis(self) -> None:
        out = self.folder / "acme_2f_hand_meshes.npz"
        code, said = _run(self.cli, _argv(self.files, ["--closing", "+Z", "--approach", "+X", "--binormal", "+Y"], out=out))
        self.assertNotEqual(code, 0)
        self.assertIn("+Y", said)
        self.assertFalse(out.exists())

    def test_metres_read_as_millimetres_is_refused(self) -> None:
        out = self.folder / "acme_2f_hand_meshes.npz"
        code, said = _run(self.cli, _argv(self.files, _HANDE_WORDS, scale="1", mount_face="-0.0861", out=out))
        self.assertNotEqual(code, 0)
        self.assertIn("--scale-to-mm", said)
        self.assertFalse(out.exists())

    def test_nothing_that_decides_geometry_has_a_default(self) -> None:
        full = _argv(self.files, _HANDE_WORDS, write=False)
        for flag in ("--scale-to-mm", "--closing", "--approach", "--binormal", "--mount-face-mm", "--origin"):
            with self.subTest(left_out=flag):
                index = full.index(flag)
                code, _ = _run(self.cli, full[:index] + full[index + 2:])
                self.assertEqual(code, 2)
        self.assertNotEqual(_run(self.cli, _argv(self.files, _HANDE_WORDS, hand="Acme-2F", write=False))[0], 0)
        self.assertEqual(_run(self.cli, _argv(self.files, _HANDE_WORDS, write=False))[0], 0)

    def test_an_existing_bundle_is_never_overwritten(self) -> None:
        out = self.folder / "acme_2f_hand_meshes.npz"
        self.assertEqual(_run(self.cli, _argv(self.files, _HANDE_WORDS, out=out))[0], 0)
        before = out.read_bytes()
        self.assertNotEqual(_run(self.cli, _argv(self.files, _HANDE_WORDS, out=out))[0], 0)
        self.assertEqual(out.read_bytes(), before)

    def test_the_writer_proves_its_reader_before_it_writes(self) -> None:
        """⭐ THE CONTROL: a reader that moves every vertex a hundredth of a millimetre fails the self check, and a failed
        one writes nothing. The reference itself is not what is broken: a moved reference is exported and read back
        unchanged, so only a broken READER is what the check can see."""
        lib = self.cli._mesh_lib()
        real = lib.HandBundle.from_mesh_files

        def drifting(**kwargs: Any) -> Any:
            read = real(**kwargs)
            moved = {part: np.asarray(vertices) + np.array([0.01, 0.0, 0.0]) for part, vertices in read.vertices.items()}
            return lib.HandBundle(vertices=moved, faces=read.faces, origin=read.origin)

        out = self.folder / "acme_2f_hand_meshes.npz"
        with mock.patch.object(lib.HandBundle, "from_mesh_files", drifting):
            code, said = _run(self.cli, _argv(self.files, _HANDE_WORDS, out=out))
        self.assertEqual(code, 1, said)
        self.assertIn("self check", said)
        self.assertFalse(out.exists())


class TheRowIsRecordedTests(unittest.TestCase):
    def test_a_write_beside_an_index_records_its_vendor_mesh_row(self) -> None:
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        files = _obj_files(folder, _vendor("robotiq_hande", "+Z", "+Y", "-X", -86.10))
        data = folder / "data"
        data.mkdir()
        shutil.copy(_DATA / "bundles.json", data / "bundles.json")
        out = data / "acme_2f_hand_meshes.npz"
        code, said = _run(_cli(), _argv(files, _HANDE_WORDS, out=out))
        self.assertEqual(code, 0, said)
        row = json.loads((data / "bundles.json").read_text(encoding="utf-8"))["bundles"]["acme_2f_hand_meshes.npz"]
        self.assertEqual(row["source"], "vendor_mesh")
        self.assertIn("gripper.obj", row["note"])
        self.assertNotIn(str(folder), row["note"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
