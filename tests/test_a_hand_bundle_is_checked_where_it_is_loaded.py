"""A hand bundle is checked where it is loaded, so a customer's body is refused by name instead of modelled wrong.

Customer chain lane C2a (finding 28 of the audit of 2026-09-17). The guard took every array of a hand bundle that is not
a record as a mesh, placed only the three parts it knows, and asked nothing about the axis the fingers lie along. Every
bundle it ever read was one this repository baked, so that held. A customer's does not have to.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"


def _hande() -> dict[str, np.ndarray]:
    with np.load(_DATA / "robotiq_hande_hand_meshes.npz", allow_pickle=True) as data:
        return {key: np.array(data[key]) for key in data.files}


def _turned_about_x(arrays: dict[str, np.ndarray], degrees: float) -> dict[str, np.ndarray]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return {key: (np.asarray(value, dtype=np.float64) @ rotation.T if key.endswith("__v") else value)
            for key, value in arrays.items()}


class _Folder:
    """A mesh folder holding the committed ur5e arm and a hand bundle written from ``arrays``."""

    def __init__(self, case: unittest.TestCase, arrays: dict[str, np.ndarray], hand: str = "robotiq_hande") -> None:
        self.path = Path(case.enterContext(tempfile.TemporaryDirectory()))
        shutil.copy(_DATA / "ur5e_collision_meshes.npz", self.path / "ur5e_collision_meshes.npz")
        np.savez(self.path / f"{hand}_hand_meshes.npz", **arrays)


class TheGuardRefusesABundleItCannotPlaceTests(unittest.TestCase):
    def _compose(self, arrays: dict[str, np.ndarray]) -> dict:
        from src.robot.safety.planning.environment import compose_collision_meshes

        return compose_collision_meshes("ur5e", "robotiq_hande", _Folder(self, arrays).path)

    def test_a_part_the_guard_does_not_know_is_refused_at_compose(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused

        arrays = _hande()
        arrays.update({"adapter__v": arrays["gripper__v"], "adapter__f": arrays["gripper__f"],
                       "adapter__frame": arrays["gripper__frame"]})
        with self.assertRaises(HandBundleRefused) as caught:
            self._compose(arrays)
        for word in ("adapter", "gripper", "lfinger", "rfinger"):
            self.assertIn(word, str(caught.exception))

    def test_the_committed_bundle_and_its_records_compose(self) -> None:
        """⭐ THE CONTROL: an origin and hand__ records are not parts, and the unmodified bundle composes whole."""
        arrays = _hande()
        arrays["hand__source"] = np.asarray(["bake"])
        composed = self._compose(arrays)
        for part in ("gripper", "lfinger", "rfinger"):
            self.assertIn(f"{part}__v", composed)

    def test_a_hand_missing_a_finger_is_refused(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused

        arrays = {key: value for key, value in _hande().items() if not key.startswith("rfinger__")}
        with self.assertRaises(HandBundleRefused) as caught:
            self._compose(arrays)
        self.assertIn("rfinger", str(caught.exception))

    def test_fingers_off_the_model_axis_are_refused_and_a_lean_inside_the_tolerance_is_not(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused

        for degrees in (90.0, 5.0):
            with self.subTest(degrees=degrees), self.assertRaises(HandBundleRefused) as caught:
                self._compose(_turned_about_x(_hande(), degrees))
            self.assertIn("+Y", str(caught.exception))
        self._compose(_turned_about_x(_hande(), 0.5))

    def test_crossed_fingers_are_refused(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused

        arrays = _hande()
        for suffix in ("v", "f", "frame"):
            arrays[f"lfinger__{suffix}"], arrays[f"rfinger__{suffix}"] = arrays[f"rfinger__{suffix}"], arrays[f"lfinger__{suffix}"]
        with self.assertRaises(HandBundleRefused) as caught:
            self._compose(arrays)
        self.assertIn("left finger", str(caught.exception))

    def test_a_malformed_array_is_refused_by_name(self) -> None:
        from src.robot.safety.planning._hand_bundle import hand_bundle_refusal

        cases = {}
        bad = _hande()
        faces = np.array(bad["lfinger__f"])
        faces[0, 0] = len(bad["lfinger__v"])
        bad["lfinger__f"] = faces
        cases["a face past the last vertex"] = (bad, "lfinger__f")
        bad = _hande()
        vertices = np.array(bad["gripper__v"])
        vertices[0, 0] = np.nan
        bad["gripper__v"] = vertices
        cases["a NaN vertex"] = (bad, "gripper__v")
        bad = _hande()
        bad["rfinger__v"] = np.asarray(bad["rfinger__v"])[:, :2]
        cases["two coordinates"] = (bad, "rfinger__v")
        bad = _hande()
        bad["gripper__frame"] = np.asarray([5], dtype=np.int32)
        cases["frame 5"] = (bad, "gripper__frame")
        for label, (arrays, part) in cases.items():
            with self.subTest(case=label):
                sentence = hand_bundle_refusal(arrays, name="scratch")
                self.assertIsNotNone(sentence)
                self.assertIn(part, str(sentence))

    def test_every_committed_hand_bundle_passes(self) -> None:
        """⭐ THE CONTROL on the check itself: it refuses what is wrong, not everything."""
        from src.robot.safety.planning._hand_bundle import hand_bundle_refusal

        for path in sorted(_DATA.glob("*_hand_meshes.npz")):
            with self.subTest(bundle=path.name), np.load(path, allow_pickle=True) as data:
                self.assertIsNone(hand_bundle_refusal({key: data[key] for key in data.files}, name=path.name))


class TheStatusAndTheDoctorNameARefusedBundleTests(unittest.TestCase):
    def _adapter(self) -> Path:
        arrays = _hande()
        arrays.update({"adapter__v": arrays["gripper__v"], "adapter__f": arrays["gripper__f"],
                       "adapter__frame": arrays["gripper__frame"]})
        return _Folder(self, arrays).path

    def test_the_status_token_comes_before_any_engine(self) -> None:
        from src.robot.safety import _fcl_self_collision as fcl

        with mock.patch.object(fcl, "import_collision_engine", side_effect=AssertionError("asked for an engine")):
            self.assertEqual(fcl.mesh_backend_status("ur5e", str(self._adapter()), "robotiq_hande"), "hand_bundle_refused")

    def test_an_absent_hand_bundle_is_still_no_hand_bundle(self) -> None:
        """⭐ THE CONTROL: the new token does not swallow the old one."""
        from src.robot.safety import _fcl_self_collision as fcl

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copy(_DATA / "ur5e_collision_meshes.npz", folder / "ur5e_collision_meshes.npz")
        self.assertEqual(fcl.mesh_backend_status("ur5e", str(folder), "robotiq_hande"), "no_hand_bundle")

    def test_the_doctor_reports_a_refused_bundle_as_broken(self) -> None:
        from src.robot.safety.planning import doctor, environment

        folder = self._adapter()
        with mock.patch.object(environment, "COLLISION_MESH_DIR", folder):
            probes = doctor._probe_gripper("ur5e", "robotiq_hande")
        (bundle,) = [probe for probe in probes if probe.name.startswith("gripper bundle")]
        self.assertEqual(str(bundle.status), "broken")
        self.assertIn("adapter", bundle.detail)


class TheCoverFitRefusesBeforeItFitsTests(unittest.TestCase):
    def test_the_cover_fit_refuses_before_it_fits(self) -> None:
        path = _REPO / "scripts" / "curobo" / "fit_cover_spheres.py"
        spec = importlib.util.spec_from_file_location("_fit_cover_for_refusal", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
        arrays = _hande()
        arrays.update({"adapter__v": arrays["gripper__v"], "adapter__f": arrays["gripper__f"],
                       "adapter__frame": arrays["gripper__frame"]})
        folder = _Folder(self, arrays).path
        with mock.patch.object(module, "DATA", folder), \
                mock.patch.object(module, "fit_bundle", side_effect=AssertionError("fitted a refused bundle")), \
                self.assertRaises(SystemExit) as caught:
            module.main(["--hand", "robotiq_hande"])
        self.assertIn("adapter", str(caught.exception.code))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
