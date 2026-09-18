"""A hand in a vendor's frame and units becomes the bundle a bake writes, by one frame change (customer chain lane C2b).

The frame change lived inline in ``bake_gripper_variant.py``, reachable only through a catalogue of two hands. A customer
with a vendor mesh had no way in at all (finding 0 of the audit of 2026-09-17), and one with a USD had to edit code
(finding 1). ``robot/hand_from_mesh.py`` owns it now, with axes stated as words so a mirror cannot be typed.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"


def _catalogue() -> dict:
    path = _REPO / "scripts" / "grippers" / "bake_gripper_variant.py"
    spec = importlib.util.spec_from_file_location("_bake_for_axes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {asset.key: asset for asset in module.CATALOGUE}


def _hande() -> dict[str, np.ndarray]:
    with np.load(_DATA / "robotiq_hande_hand_meshes.npz", allow_pickle=True) as data:
        return {key: np.array(data[key]) for key in data.files}


def _hande_in_its_asset_frame(mount_face_mm: float = -86.10) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The committed Hand-E parts carried back into the Isaac asset's frame, in metres: what a vendor would ship."""
    from src.robot.safety.planning.robot.hand_from_mesh import VendorAxes

    rotation = np.asarray(VendorAxes.from_words(closing="+Z", approach="+Y", binormal="-X").rotation)
    committed = _hande()
    parts = {}
    for part in ("gripper", "lfinger", "rfinger"):
        vendor = np.asarray(committed[f"{part}__v"], dtype=np.float64) @ rotation
        vendor[:, 1] += mount_face_mm
        parts[part] = (vendor / 1000.0, committed[f"{part}__f"])
    return parts


class AxesAreWordsTests(unittest.TestCase):
    def test_the_axis_words_are_the_catalogue_matrices(self) -> None:
        from src.robot.safety.planning.robot.hand_from_mesh import VendorAxes

        catalogue = _catalogue()
        spelled = {"robotiq_2f85": ("+Y", "+Z", "+X"), "robotiq_hande": ("+Z", "+Y", "-X")}
        for hand, (closing, approach, binormal) in spelled.items():
            with self.subTest(hand=hand):
                axes = VendorAxes.from_words(closing=closing, approach=approach, binormal=binormal)
                self.assertEqual(axes.rotation, catalogue[hand].rotation)
                self.assertEqual(axes.approach_index, catalogue[hand].approach_axis)

    def test_one_flipped_sign_is_another_matrix(self) -> None:
        """⭐ THE CONTROL: the equality above can fail."""
        from src.robot.safety.planning.robot.hand_from_mesh import VendorAxes

        flipped = VendorAxes.from_words(closing="-Z", approach="+Y", binormal="+X")
        self.assertNotEqual(flipped.rotation, _catalogue()["robotiq_hande"].rotation)

    def test_a_mirror_and_a_repeated_axis_are_refused(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused
        from src.robot.safety.planning.robot.hand_from_mesh import VendorAxes

        with self.assertRaises(HandBundleRefused) as caught:
            VendorAxes.from_words(closing="+Y", approach="+Z", binormal="-X")
        self.assertIn("mirror", str(caught.exception))
        with self.assertRaises(HandBundleRefused) as caught:
            VendorAxes.from_words(closing="+Y", approach="+Y", binormal="+X")
        self.assertIn("twice", str(caught.exception))
        with self.assertRaises(HandBundleRefused):
            VendorAxes.from_words(closing="+W", approach="+Y", binormal="+X")


class NoPlacingNumberHasADefaultTests(unittest.TestCase):
    def test_no_safety_number_has_a_default(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused
        from src.robot.safety.planning.robot.hand_from_mesh import HandBundle, VendorAxes

        axes = VendorAxes.from_words(closing="+Z", approach="+Y", binormal="-X")
        stated = {"parts": _hande_in_its_asset_frame(), "axes": axes, "scale_to_mm": 1000.0,
                  "mount_face_mm": -86.10, "origin": "mounting_face"}
        HandBundle.from_parts(**stated)
        for left_out in ("scale_to_mm", "mount_face_mm", "origin"):
            with self.subTest(left_out=left_out), self.assertRaises(TypeError):
                HandBundle.from_parts(**{key: value for key, value in stated.items() if key != left_out})
        with self.assertRaises(HandBundleRefused):
            HandBundle.from_parts(**{**stated, "origin": "wrist"})
        with self.assertRaises(HandBundleRefused):
            HandBundle.from_parts(**{**stated, "scale_to_mm": 0.0})


class TheCommittedHandRoundTripsTests(unittest.TestCase):
    def _arrays(self, mount_face_mm: float) -> dict[str, np.ndarray]:
        from src.robot.safety.planning.robot.hand_from_mesh import HandBundle, VendorAxes

        bundle = HandBundle.from_parts(
            parts=_hande_in_its_asset_frame(), axes=VendorAxes.from_words(closing="+Z", approach="+Y", binormal="-X"),
            scale_to_mm=1000.0, mount_face_mm=mount_face_mm, origin="mounting_face",
        )
        return bundle.arrays()

    def test_the_committed_hande_round_trips_through_its_asset_frame(self) -> None:
        committed = _hande()
        arrays = self._arrays(-86.10)
        for part in ("gripper", "lfinger", "rfinger"):
            with self.subTest(part=part):
                np.testing.assert_allclose(arrays[f"{part}__v"], committed[f"{part}__v"], rtol=0.0, atol=1e-9)
                np.testing.assert_array_equal(arrays[f"{part}__f"], committed[f"{part}__f"])
                self.assertEqual(int(arrays[f"{part}__frame"][0]), 6)
        self.assertEqual(str(arrays["gripper__origin"][0]), "mounting_face")

    def test_a_mount_face_off_by_a_hundredth_is_seen(self) -> None:
        """⭐ THE CONTROL: the tolerance above can see a mount face error far smaller than any bench measurement."""
        committed = _hande()
        arrays = self._arrays(-86.09)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(arrays["gripper__v"], committed["gripper__v"], rtol=0.0, atol=1e-9)


class AWrittenBundleIsOneTheGuardComposesTests(unittest.TestCase):
    def test_a_written_bundle_is_composed_and_never_overwritten(self) -> None:
        from src.robot.safety.planning._hand_bundle import HandBundleRefused
        from src.robot.safety.planning.environment import compose_collision_meshes
        from src.robot.safety.planning.robot.hand_from_mesh import HandBundle, VendorAxes

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copy(_DATA / "ur5e_collision_meshes.npz", folder / "ur5e_collision_meshes.npz")
        bundle = HandBundle.from_parts(
            parts=_hande_in_its_asset_frame(), axes=VendorAxes.from_words(closing="+Z", approach="+Y", binormal="-X"),
            scale_to_mm=1000.0, mount_face_mm=-86.10, origin="mounting_face",
        )
        target = folder / "acme_2f_hand_meshes.npz"
        report = bundle.write(target, records={"hand__recipe": "scratch"})
        for word in ("acme_2f_hand_meshes.npz", "mounting_face", "lfinger"):
            self.assertIn(word, report.render())
        composed = compose_collision_meshes("ur5e", "acme_2f", folder)
        self.assertIn("rfinger__v", composed)
        before = target.read_bytes()
        with self.assertRaises(HandBundleRefused):
            bundle.write(target)
        self.assertEqual(target.read_bytes(), before)

    def test_a_write_validates_the_axes_it_produced(self) -> None:
        """⭐ THE CONTROL: a vendor frame stated wrongly, fingers along the binormal, is refused before anything is written."""
        from src.robot.safety.planning._hand_bundle import HandBundleRefused
        from src.robot.safety.planning.robot.hand_from_mesh import HandBundle, VendorAxes

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        wrong = HandBundle.from_parts(
            parts=_hande_in_its_asset_frame(), axes=VendorAxes.from_words(closing="+Z", approach="+X", binormal="+Y"),
            scale_to_mm=1000.0, mount_face_mm=-86.10, origin="mounting_face",
        )
        with self.assertRaises(HandBundleRefused):
            wrong.write(folder / "wrong_hand_meshes.npz")
        self.assertFalse((folder / "wrong_hand_meshes.npz").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
