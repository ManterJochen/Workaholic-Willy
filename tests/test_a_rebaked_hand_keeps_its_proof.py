"""Baking a hand again keeps its evidence exactly when it reproduces the geometry, and loses it the moment it does not.

``scripts/grippers/bake_gripper_variant.py --write`` writes ``{hand}_hand_meshes.npz``. Until UM lane S22 the bundle
carried ``hand__admitted_arms``, and this file held that a byte for byte re-bake kept the list while one changed vertex
emptied it: writing an empty list over the committed Hand-E would have dropped six cells to the capsule proxy with no
geometry changed at all.

The list is retired and the property is not. Admission is a committed evidence file per combination, and what binds a
file to the hand is ``guard_sha256``, a hash over the arrays the exact mesh guard composes. So the same two halves hold
here against that hash: identical arrays keep every file valid, one moved vertex invalidates every file for that hand.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.robot.safety._fcl_self_collision import composed_parts
from src.robot.safety.planning.evidence import guard_sha256

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "src" / "robot" / "safety" / "data"
_HANDE = "robotiq_hande_hand_meshes.npz"


def _folder_with(hand_arrays: "dict[str, np.ndarray]") -> tempfile.TemporaryDirectory:
    """A folder holding the ur5e arm and a Hand-E bundle made of ``hand_arrays``, as a re-bake would leave it."""
    folder = tempfile.TemporaryDirectory()
    root = Path(folder.name)
    (root / "ur5e_collision_meshes.npz").write_bytes((_DATA / "ur5e_collision_meshes.npz").read_bytes())
    np.savez_compressed(root / _HANDE, **hand_arrays)
    return folder


def _committed() -> "dict[str, np.ndarray]":
    with np.load(_DATA / _HANDE) as bundle:
        return {key: np.array(bundle[key]) for key in bundle.files}


def _hash(folder: str) -> str:
    return guard_sha256(composed_parts("ur5e", folder, "robotiq_hande", 20.0))


class ARebakedHandKeepsItsProofTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.committed_hash = guard_sha256(composed_parts("ur5e", None, "robotiq_hande", 20.0))

    def test_the_same_arrays_keep_every_evidence_file_valid(self) -> None:
        with _folder_with(_committed()) as folder:
            self.assertEqual(_hash(folder), self.committed_hash)

    def test_one_changed_vertex_invalidates_them(self) -> None:
        """⭐ THE CONTROL. Keeping the proof must not mean keeping it whatever was written."""
        arrays = _committed()
        arrays["lfinger__v"] = arrays["lfinger__v"].copy()
        arrays["lfinger__v"][0, 0] += 0.01
        with _folder_with(arrays) as folder:
            self.assertNotEqual(_hash(folder), self.committed_hash)

    def test_a_changed_origin_invalidates_them(self) -> None:
        """The origin is the one fact a reader cannot recover from the numbers: it decides whether the plate is added,
        so a flipped origin moves every hand vertex the guard is handed."""
        arrays = _committed()
        origin = str(arrays["gripper__origin"].reshape(-1)[0])
        arrays["gripper__origin"] = np.array(["flange" if origin != "flange" else "mounting_face"])
        with _folder_with(arrays) as folder:
            self.assertNotEqual(_hash(folder), self.committed_hash)

    def test_a_retired_admission_list_in_a_rebake_changes_nothing(self) -> None:
        """A bundle that still carries the old list is the same hand: record keys are not geometry."""
        arrays = _committed()
        arrays["hand__admitted_arms"] = np.array(["ur5e"])
        with _folder_with(arrays) as folder:
            self.assertEqual(_hash(folder), self.committed_hash)


if __name__ == "__main__":
    unittest.main()
