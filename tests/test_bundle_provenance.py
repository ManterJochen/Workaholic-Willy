"""Every committed collision bundle says where its geometry came from, and the record matches the bytes (B5, S5).

A ``.npz`` of vertices is the exact mesh guard's whole authority, and nothing in the file says which robot asset it was
read from, which reader produced it, or whether it is the vendor's collision geometry or a hull of its visual. Those
differences decide what the guard watches: measured on ur10e, its own collision meshes are 1.259x the volume of its
visual, and a convex hull of the visual comes out at 1.357x. A bundle whose source nobody recorded cannot be rebuilt,
compared, or retired when a better source appears.

``bundles.json`` beside the files is that record. Two things make it a record rather than a note: every entry's
``arrays_sha256`` is recomputed from the file here, so a bundle silently rewritten stops matching its own row, and
every source string has to be one this repository can name a script for.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "data"
_INDEX = _DATA / "bundles.json"

#: Every way a bundle in this repository is produced, arms and hands. A source outside this set is a bundle nobody can
#: rebuild. The last three are the ways a CUSTOMER hand gets a body (customer chain lane C1a): from its registry numbers,
#: from one standalone hand USD, or from a vendor mesh.
_SOURCES = {
    "isaac_usd": "the composed Isaac articulation, read by scripts/isaac/bake_ur_collision_meshes.py",
    "isaac_importer_visual_hull": "convex hulls of the visual meshes in Isaac's URDF importer assets",
    "ur_description_stl": "Universal Robots' own collision STL files, through the rendered description",
    "registry_dimensions": "an envelope from a hand's registry numbers, written by scripts/grippers/write_hand_from_dimensions.py",
    "standalone_usd": "one standalone hand USD, baked by scripts/grippers/bake_gripper_variant.py",
    "vendor_mesh": "a vendor STL or OBJ, written by scripts/grippers/write_hand_from_mesh.py",
}

#: A record key a hand bundle may carry, as a note names one.
_RECORD_KEY = __import__("re").compile(r"\bhand__[a-z_]+")


def _keys_a_note_names_that_a_bundle_lacks(note: str, bundle: Path) -> list[str]:
    with np.load(bundle, allow_pickle=True) as data:
        carried = set(data.files)
    return sorted({key for key in _RECORD_KEY.findall(note) if key not in carried})


def arrays_sha256(path: Path) -> str:
    """One hash over a bundle's arrays, in name order: the bytes the guard actually loads.

    Not a hash of the file. A ``.npz`` is a zip, and a zip carries timestamps and a compression level, so the same
    arrays written twice give two different files and neither is wrong.
    """
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=True) as bundle:
        for name in sorted(bundle.files):
            array = np.asarray(bundle[name])
            digest.update(name.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class TheBundlesSayWhereTheyCameFromTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(_INDEX.is_file(), f"{_INDEX} does not exist, so no bundle says where it came from")
        self.index = json.loads(_INDEX.read_text(encoding="utf-8"))

    def test_every_bundle_has_an_entry_and_every_entry_has_a_bundle(self) -> None:
        files = sorted(path.name for path in _DATA.glob("*.npz"))
        self.assertTrue(files, "no bundle was found, so this test would pass by having nothing to check")
        self.assertEqual(sorted(self.index["bundles"]), files)

    def test_each_entry_matches_the_bytes_it_describes(self) -> None:
        for name, entry in sorted(self.index["bundles"].items()):
            with self.subTest(bundle=name):
                self.assertEqual(entry["arrays_sha256"], arrays_sha256(_DATA / name))

    def test_the_hash_can_fail(self) -> None:
        """⭐ THE CONTROL. A hash that cannot notice a changed vertex is a comment.

        The perturbation is measured against the array's own precision rather than assumed: at 400 mm one ulp is
        about 3e-5 mm in float32 and 6e-14 mm in float64, so 10 micrometres is above either and far below anything a
        guard would act on.
        """
        import tempfile

        original = _DATA / "ur5e_collision_meshes.npz"
        with np.load(original, allow_pickle=True) as bundle:
            arrays = {name: np.array(bundle[name]) for name in bundle.files}
        self.assertGreater(0.01, 10.0 * float(np.spacing(np.float32(400.0))))
        arrays["forearm__v"][0, 0] += 0.01
        with tempfile.TemporaryDirectory() as folder:
            moved = Path(folder) / original.name
            np.savez_compressed(moved, **arrays)
            self.assertNotEqual(arrays_sha256(original), arrays_sha256(moved))

    def test_the_hash_is_over_the_arrays_and_not_the_file(self) -> None:
        """⭐ THE CONTROL. A zip carries a timestamp, so a file hash would change on every rewrite of the same arms."""
        import tempfile

        original = _DATA / "ur3e_collision_meshes.npz"
        with np.load(original, allow_pickle=True) as bundle:
            arrays = {name: np.array(bundle[name]) for name in bundle.files}
        with tempfile.TemporaryDirectory() as folder:
            rewritten = Path(folder) / original.name
            np.savez_compressed(rewritten, **arrays)
            self.assertEqual(arrays_sha256(original), arrays_sha256(rewritten))

    def test_every_source_is_one_this_repository_can_name_a_script_for(self) -> None:
        for name, entry in sorted(self.index["bundles"].items()):
            with self.subTest(bundle=name):
                self.assertIn(entry["source"], _SOURCES)
                self.assertTrue(entry["note"].strip(), "a source with no note is a word, not a provenance")

    def test_no_note_names_a_record_key_its_bundle_does_not_carry(self) -> None:
        """A note that cites a retired key describes a bundle that is gone (UM lane S22 retired hand__admitted_arms)."""
        for name, entry in sorted(self.index["bundles"].items()):
            with self.subTest(bundle=name):
                self.assertEqual(_keys_a_note_names_that_a_bundle_lacks(entry["note"], _DATA / name), [])

    def test_the_note_check_sees_a_key_a_bundle_lacks(self) -> None:
        """⭐ THE CONTROL: the scan reports a retired key and accepts one the bundle carries."""
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "scratch_hand_meshes.npz"
            np.savez(bundle, hand__source=np.asarray(["dimensions"]), gripper__v=np.zeros((3, 3)))
            self.assertEqual(_keys_a_note_names_that_a_bundle_lacks("records hand__admitted_arms", bundle),
                             ["hand__admitted_arms"])
            self.assertEqual(_keys_a_note_names_that_a_bundle_lacks("records hand__source", bundle), [])

    def test_each_hand_writer_has_a_source_this_file_and_the_index_accept(self) -> None:
        for source in ("registry_dimensions", "standalone_usd", "vendor_mesh"):
            with self.subTest(source=source):
                self.assertIn(source, _SOURCES)
                self.assertIn(source, self.index["sources"])

    def test_the_index_and_this_file_name_one_set_of_sources(self) -> None:
        """⭐ THE CONTROL: a token added on one side only is a source one reader accepts and the other refuses."""
        self.assertEqual(sorted(self.index["sources"]), sorted(_SOURCES))

    def test_an_arm_bundle_says_which_arm_and_a_hand_bundle_says_which_hand(self) -> None:
        for name, entry in sorted(self.index["bundles"].items()):
            with self.subTest(bundle=name):
                if name.endswith("_hand_meshes.npz"):
                    self.assertEqual(entry["hand"], name.split("_hand_meshes")[0])
                    self.assertNotIn("model", entry)
                else:
                    self.assertEqual(entry["model"], name.split("_collision_meshes")[0])
                    self.assertNotIn("hand", entry)

    def test_the_index_names_the_rule_it_was_written_under(self) -> None:
        """A provenance file that does not say what it is for reads as a cache."""
        self.assertIn("schema", self.index)
        self.assertTrue(self.index["what"].strip())


if __name__ == "__main__":
    unittest.main()
