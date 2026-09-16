"""A hand is its own guard bundle, and the arm plus the hand composed at load is exactly what the per arm files held.

Until UM lane S05 a hand's guard bundle was an arm plus a hand: ``bake_gripper_variant.py`` copied every arm array of
one arm and replaced the three hand arrays, so the Hand-E existed six times and the EGU-50 once, implicitly on a ur5e.
B3 keeps one bundle per hand (``{hand}_hand_meshes.npz``) and composes the arm at load. Measured before S05
(logs/b3/hand_array_agreement.log): the 2F-85 arrays are byte identical in all six arm bundles, the Hand-E arrays in all
six variants, and every variant's arm arrays are byte copies of its arm's own bundle.

S08 deleted the per arm files. What they held is recorded key by key in ``tests/data/b3_legacy_hand_bundles.json``, a
sha256 over each array's dtype, shape and bytes, taken from the files just before they went.

⚠ Those records hold TWO halves, and only one of them is still a live claim. The HAND arrays are pinned to the
record, which is what S05 and S08 are about: a hand composed onto an arm has to be the same hand the per arm
file carried. The ARM arrays in the record describe the bundles Isaac produced, and B5 re-baked every one of
them from Universal Robots' own collision STLs on 2026-09-16, so they are history rather than a target: four
arms' geometry moved where Isaac's URDFs disagreed with UR's own description, and the rest kept their geometry
with a different vertex order. The arm half is therefore held against the arm's OWN committed bundle, which is
what composition promises, and the record's arm hashes are checked for shape rather than for bytes.

Each hand bundle also records the arms it was proven equal on (``hand__admitted_arms``). The EGU-50 was baked on a ur5e
only, and on every other arm the guard answers ``variant_model_mismatch``; composition must not quietly turn that into
exact meshes before evidence admits the pairing (UM10).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from src.robot.safety.planning import environment

_DATA = environment.COLLISION_MESH_DIR
_RECORDS = pathlib.Path(__file__).resolve().parent / "data" / "b3_legacy_hand_bundles.json"
_ARMS = ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e")


def _digest(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


class TheComposedGuardIsWhatTheFilesHeldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = json.loads(_RECORDS.read_text(encoding="utf-8"))

    #: The arrays the ARM bundle owns. Everything else in a composed set came from the hand, including the hand's
    #: own records (``gripper__origin``, ``hand__admitted_arms``), so the split is stated from the arm's side.
    ARM_KEYS = frozenset(
        f"{link}__{suffix}"
        for link in ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
        for suffix in ("v", "f", "frame")
    )

    def assertHolds(self, composed: dict[str, np.ndarray], record: str) -> None:
        """The hand arrays are the record's, byte for byte; the arm arrays are the arm's own bundle."""
        expected = self.records[record]["arrays"]
        self.assertEqual(sorted(composed), sorted(expected), record)
        arm = record.split("/")[0]
        with np.load(environment.collision_mesh_bundle(arm), allow_pickle=True) as own:
            arm_arrays = {name: np.array(own[name]) for name in own.files}
        for key, digest in expected.items():
            with self.subTest(record=record, key=key):
                if key not in self.ARM_KEYS:
                    self.assertEqual(_digest(np.asarray(composed[key])), digest)
                else:
                    self.assertIn(key, arm_arrays)
                    self.assertEqual(_digest(np.asarray(composed[key])), _digest(arm_arrays[key]))

    def test_the_records_cover_every_file_that_carried_a_hand(self) -> None:
        """The control against a record that holds nothing: thirteen files, each with arm and hand arrays."""
        self.assertEqual(len(self.records), 13)
        for record in self.records.values():
            with self.subTest(file=record["file"]):
                self.assertIn("forearm__v", record["arrays"])
                self.assertIn("gripper__v", record["arrays"])

    def test_the_hand_e_on_every_arm_is_its_variant_file(self) -> None:
        for arm in _ARMS:
            self.assertHolds(environment.compose_collision_meshes(arm, "robotiq_hande"), f"{arm}/robotiq_hande")

    def test_the_2f85_on_every_arm_is_the_arm_bundle(self) -> None:
        for arm in _ARMS:
            self.assertHolds(environment.compose_collision_meshes(arm, "robotiq_2f85"), f"{arm}/robotiq_2f85")

    def test_the_egu50_on_ur5e_is_the_flat_file(self) -> None:
        self.assertHolds(environment.compose_collision_meshes("ur5e", "schunk_egu50"), "ur5e/schunk_egu50")

    def test_a_vertex_moved_by_a_micrometre_is_seen(self) -> None:
        """The control: a comparison that cannot see one moved vertex proves nothing about the others."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("ur3e_collision_meshes.npz", "robotiq_hande_hand_meshes.npz"):
                shutil.copyfile(_DATA / name, pathlib.Path(tmp) / name)
            hand = dict(np.load(pathlib.Path(tmp) / "robotiq_hande_hand_meshes.npz"))
            moved = hand["gripper__v"].copy()
            moved[0, 0] += 1e-3
            hand["gripper__v"] = moved
            np.savez_compressed(pathlib.Path(tmp) / "robotiq_hande_hand_meshes.npz", **hand)
            composed = environment.compose_collision_meshes("ur3e", "robotiq_hande", mesh_dir=tmp)
        self.assertNotEqual(_digest(composed["gripper__v"]),
                            self.records["ur3e/robotiq_hande"]["arrays"]["gripper__v"])

    def test_another_arm_is_seen(self) -> None:
        composed = environment.compose_collision_meshes("ur3", "robotiq_hande")
        self.assertNotEqual(_digest(composed["forearm__v"]),
                            self.records["ur3e/robotiq_hande"]["arrays"]["forearm__v"])

    def test_each_hand_names_the_arms_it_was_proven_on(self) -> None:
        expected = {"robotiq_2f85": set(_ARMS), "robotiq_hande": set(_ARMS), "schunk_egu50": {"ur5e"}}
        for hand, arms in expected.items():
            with self.subTest(hand=hand), np.load(environment.hand_mesh_bundle(hand)) as data:
                self.assertEqual({str(a) for a in np.asarray(data["hand__admitted_arms"]).reshape(-1)}, arms)

    def test_no_record_key_reaches_the_guard(self) -> None:
        """The guard takes every key not ending in __origin as a mesh name, so a record key must never pass through."""
        composed = environment.compose_collision_meshes("ur5e", "schunk_egu50")
        self.assertEqual([key for key in composed if key.startswith("hand__")], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
