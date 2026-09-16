"""A hand bundle records the arms it was proven on, and baking the hand again must not erase that by accident.

``scripts/grippers/bake_gripper_variant.py --write`` writes ``{hand}_hand_meshes.npz``. A freshly baked hand is proven
on no arm, so the guard refuses to compose it until evidence admits the pairing. But the committed Hand-E records six
arms, and a re-bake that reproduces its arrays byte for byte has changed nothing those arms were proven on: writing an
empty record there would drop six cells to the capsule proxy with no geometry changed at all. So the record stays
exactly when every array is the committed one, and goes the moment any array differs.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_HANDE = _ROOT / "src" / "robot" / "safety" / "data" / "robotiq_hande_hand_meshes.npz"
_ADMITTED = "hand__admitted_arms"


def _bake_module():
    spec = importlib.util.spec_from_file_location(
        "_bake_gripper_variant_under_test", _ROOT / "scripts" / "grippers" / "bake_gripper_variant.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The script declares dataclasses under `from __future__ import annotations`, which look their module up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _committed_payload() -> dict[str, np.ndarray]:
    with np.load(_HANDE) as bundle:
        return {key: np.array(bundle[key]) for key in bundle.files if key != _ADMITTED}


class ARebakedHandKeepsItsProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.still_admitted = staticmethod(_bake_module()._still_admitted)
        with np.load(_HANDE) as bundle:
            cls.recorded = bundle[_ADMITTED].tolist()

    def test_the_committed_hand_records_arms_at_all(self) -> None:
        """Without this the other tests could pass by comparing two empty records."""
        self.assertGreaterEqual(len(self.recorded), 2)

    def test_the_same_arrays_keep_the_arms(self) -> None:
        self.assertEqual(self.still_admitted(_HANDE, _committed_payload()).tolist(), self.recorded)

    def test_one_changed_vertex_drops_every_arm(self) -> None:
        """⭐ THE CONTROL. Keeping the record must not mean keeping it whatever was written."""
        payload = _committed_payload()
        payload["lfinger__v"] = payload["lfinger__v"].copy()
        payload["lfinger__v"][0, 0] += 0.01
        self.assertEqual(self.still_admitted(_HANDE, payload).tolist(), [])

    def test_a_changed_origin_drops_every_arm(self) -> None:
        """The origin is the one fact a reader cannot recover from the numbers, so it counts as an array too."""
        payload = _committed_payload()
        origin = str(payload["gripper__origin"][0])
        payload["gripper__origin"] = np.array(["flange" if origin != "flange" else "mounting_face"])
        self.assertEqual(self.still_admitted(_HANDE, payload).tolist(), [])

    def test_a_hand_with_no_committed_bundle_is_proven_on_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.still_admitted(Path(tmp) / "new_hand_hand_meshes.npz", _committed_payload()).tolist(), [])


if __name__ == "__main__":
    unittest.main()
