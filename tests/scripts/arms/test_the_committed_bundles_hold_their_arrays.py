"""Every committed collision bundle holds the parts, frames and arrays its index says (P0f F0, the proof before any move).

``bundles.json`` records one ``arrays_sha256`` per bundle, and ``tests/scripts/grippers/test_bundle_provenance.py`` recomputes it. What
that leaves unsaid is WHICH arrays: a bundle that lost a part or gained one still hashes to a value, only another
one, and the failure would not say which key moved. P0f rebuilds these files through a new baker, so this states the
shape of each bundle outright: six arm links at frames 1 to 6 and three hand parts at frame 6 in every arm bundle,
the mesh keys of every hand bundle, and the 2F-85 riding in every arm bundle as the very arrays of its own hand
bundle. For ur16e that last check is the only direct one: it has no evidence and no guard golden
(``tests/data/b2b3_hand_geometry_golden.json`` holds six arms).

Measured 2026-09-23: 10 of 10 bundles equal their row with ``allow_pickle=False`` as with the library's loader, 27
keys in each arm bundle, and 63 of 63 embedded 2F-85 arrays equal the hand bundle's in dtype, shape and bytes.
"""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from src.robot.safety.planning.bundle_index import arrays_sha256 as library_arrays_sha256
from tests import _evidence_rulers as ruler

_ARM_LINKS = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
_MESH_KINDS = ("v", "f", "frame")
_HAND_KEYS = {f"{part}__{kind}" for part in ruler.HAND_PARTS for kind in _MESH_KINDS}
_ARM_KEYS = _HAND_KEYS | {f"{link}__{kind}" for link in _ARM_LINKS for kind in _MESH_KINDS}
#: The bundles this repository ships. A customer's cell writes its own hand bundles beside them (the customer-hand trial
#: adds two), so what is checked is that none of these is missing, not that nothing else is there.
_SHIPPED = ({f"{arm}_collision_meshes.npz" for arm in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e")}
            | {f"{hand}_hand_meshes.npz" for hand in ("robotiq_2f85", "robotiq_hande", "schunk_egu50")})


def _missing_from_what_ships(names: "list[str]") -> list[str]:
    return sorted(_SHIPPED - set(names))


def _arrays(name: str) -> dict[str, np.ndarray]:
    """A bundle's arrays, never unpickled: a bundle that needs pickling holds an object, not a mesh."""
    with np.load(ruler.DATA / name, allow_pickle=False) as bundle:
        return {key: np.array(bundle[key]) for key in bundle.files}


def _arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """The index's recipe, restated: name, dtype, shape, C-order bytes, in name order, no separators."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _digest(array: np.ndarray) -> str:
    """One array as dtype, shape and bytes, so a failure names the array that moved."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class EveryBundleIsItsRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = json.loads((ruler.DATA / "bundles.json").read_text(encoding="utf-8"))["bundles"]

    def test_every_shipped_bundle_is_there_and_every_bundle_has_its_row(self) -> None:
        """A deleted or renamed bundle fails here; a hand a customer added does not."""
        names = sorted(self.index)
        self.assertEqual(sorted(path.name for path in ruler.DATA.glob("*.npz")), names)
        self.assertEqual([], _missing_from_what_ships(names))

    def test_the_check_sees_a_missing_bundle_and_admits_an_added_hand(self) -> None:
        """THE CONTROL. The exact count this replaced (seven arms, three hands) failed the customer-hand trial of
        2026-09-23 on the two hands the trial adds, 3 != 5, while every bundle it held was right."""
        shipped = sorted(_SHIPPED)
        self.assertEqual(["ur5_collision_meshes.npz"],
                         _missing_from_what_ships([name for name in shipped if name != "ur5_collision_meshes.npz"]))
        self.assertEqual([], _missing_from_what_ships([*shipped, "acme_dims_hand_meshes.npz"]))

    def test_every_bundle_hashes_to_its_row_without_unpickling(self) -> None:
        for name, row in sorted(self.index.items()):
            with self.subTest(bundle=name):
                self.assertEqual(row["arrays_sha256"], _arrays_sha256(_arrays(name)))
                self.assertEqual(row["arrays_sha256"], library_arrays_sha256(ruler.DATA / name))


class EveryBundleHoldsItsPartsTests(unittest.TestCase):
    def test_each_arm_bundle_holds_six_links_at_frames_one_to_six_and_one_hand_at_six(self) -> None:
        for path in sorted(ruler.DATA.glob("*_collision_meshes.npz")):
            arrays = _arrays(path.name)
            with self.subTest(bundle=path.name):
                self.assertEqual(sorted(_ARM_KEYS), sorted(arrays))
                self.assertEqual([1, 2, 3, 4, 5, 6], [int(arrays[f"{link}__frame"][0]) for link in _ARM_LINKS])
                self.assertEqual([6, 6, 6], [int(arrays[f"{part}__frame"][0]) for part in ruler.HAND_PARTS])

    def test_each_hand_bundle_holds_the_three_parts_and_only_named_records_beside_them(self) -> None:
        for path in sorted(ruler.DATA.glob("*_hand_meshes.npz")):
            arrays = _arrays(path.name)
            with self.subTest(bundle=path.name):
                self.assertLessEqual(_HAND_KEYS, set(arrays))
                extra = set(arrays) - _HAND_KEYS
                self.assertEqual(set(), {key for key in extra if key != "gripper__origin" and not key.startswith("hand__")})
                self.assertEqual([6, 6, 6], [int(arrays[f"{part}__frame"][0]) for part in ruler.HAND_PARTS])

    def test_the_2f85_in_every_arm_bundle_is_the_2f85_hand_bundle(self) -> None:
        """The baker copies these nine arrays from the committed ur5e bundle (``scripts/arms/bake.py:320-327``),
        so nothing but this says they are the hand's: dtype and shape included, float32 and int32 as the 2F-85
        ships, since the guard hashes both."""
        hand = {key: _digest(array) for key, array in _arrays("robotiq_2f85_hand_meshes.npz").items()}
        self.assertEqual(sorted(_HAND_KEYS), sorted(hand))
        for path in sorted(ruler.DATA.glob("*_collision_meshes.npz")):
            arrays = _arrays(path.name)
            for key in sorted(hand):
                with self.subTest(bundle=path.name, key=key):
                    self.assertEqual(hand[key], _digest(arrays[key]))


if __name__ == "__main__":
    unittest.main()
