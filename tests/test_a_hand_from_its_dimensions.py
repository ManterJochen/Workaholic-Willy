"""A gripper nobody baked becomes collision geometry from the numbers its registry file carries (B7).

The arc goal is every UR arm planning with every gripper that has a mesh OR config dimensions. The first half landed
with the cover fit; this is the second. A customer describes their hand in `config/grippers/<name>.yaml` and never
bakes anything, and the exact mesh guard and the planner still model it.

⚠ **AN ENVELOPE IS NOT THE HAND, AND THE DIFFERENCE IS MEASURED.** All three shipped hands carry both a baked
bundle and a full set of dimensions, so the envelope can be held against the body it stands for. Measured at
each hand's own declared grasp centre: the Hand-E needs 5.73 mm of inflation to be enclosed and the EGU-50
9.50 mm, while the 2F-85 is 81.49 mm out. The first two had their palm measured and the third did not, which
the registry already records. So the writer refuses a hand whose palm nobody measured, names the field, and
the rest carry an inflation stated by the caller.

⛔ **AND IT REFUSES TO OVERWRITE A BAKED BUNDLE.** A hand that was scanned is described better by the scan than by
five numbers, and a writer that quietly replaced one would swap the better model for the worse with nothing said.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLES = _ROOT / "src" / "robot" / "safety" / "data"


def _module():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "robot" / "hand_from_dimensions.py"
    spec = importlib.util.spec_from_file_location("_hand_from_dimensions_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jaw(name: str):
    from src.config.grippers import load_gripper

    return load_gripper(name).jaw


class TheEnvelopeEnclosesTheHandItStandsFor(unittest.TestCase):
    """⭐ THE MEASUREMENT THAT MAKES THIS HONEST, and it is only possible while both halves exist."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _module()

    def _outside_mm(self, name: str, inflation_mm: float) -> float:
        boxes = self.module.boxes_for_hand(_jaw(name), inflation_mm=inflation_mm)
        with np.load(_BUNDLES / f"{name}_hand_meshes.npz", allow_pickle=True) as held:
            points = np.concatenate([np.asarray(held[f"{part}__v"], dtype=np.float64)
                                     for part in ("gripper", "lfinger", "rfinger")])
        worst = np.full(len(points), np.inf)
        for box in boxes:
            gap = np.abs(points - np.asarray(box.centre_mm)) - np.asarray(box.half_extents_mm)
            worst = np.minimum(worst, np.linalg.norm(np.maximum(gap, 0.0), axis=1))
        return float(worst.max())

    def test_a_measured_hand_is_enclosed_once_it_carries_its_inflation(self) -> None:
        """The smallest inflation that encloses each hand, measured: 5.73 mm and 9.50 mm.

        Held loosely here, because a refit of either bundle moves it by a fraction of a millimetre and this
        file is not the place that would have to be re-blessed for that. What is held tightly is that ten
        millimetres is enough and that nothing like it is needed for a placement error."""
        for name, needs in (("robotiq_hande", 5.73), ("schunk_egu50", 9.50)):
            with self.subTest(hand=name):
                bare = self._outside_mm(name, 0.0)
                self.assertAlmostEqual(bare, needs, delta=0.5,
                                       msg=f"{name} needs {bare:.2f} mm of inflation and the record says {needs:.2f}")
                self.assertAlmostEqual(self._outside_mm(name, 10.0), 0.0, places=6,
                                       msg=f"{name} is still outside its own envelope at a 10 mm inflation")

    def test_the_inflation_is_what_it_takes_and_not_less(self) -> None:
        """⭐ THE CONTROL. Without it the test above would pass on an inflation of a metre."""
        self.assertGreater(self._outside_mm("schunk_egu50", 0.0), 1.0,
                           "a bare envelope already encloses this hand, so the inflation is measuring nothing")


class AHandNobodyMeasuredIsRefused(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _module()

    def test_an_estimated_palm_is_refused_by_the_field_that_says_so(self) -> None:
        """The schema already knows: `palm_measured` is false on the 2F-85, and its envelope is 81.49 mm out
        against 5.73 and 9.50 for the two that were measured."""
        said = self.module.refusal_for(_jaw("robotiq_2f85"))

        self.assertIsNotNone(said)
        assert said is not None
        self.assertIn("palm_measured", said)

    def test_a_measured_hand_is_not_refused(self) -> None:
        """⭐ THE OTHER HALF. A refusal that fires for every hand refuses nothing in particular."""
        for name in ("robotiq_hande", "schunk_egu50"):
            with self.subTest(hand=name):
                self.assertIsNone(self.module.refusal_for(_jaw(name)))


class TheWriterWillNotReplaceAScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _module()

    def test_writing_over_a_baked_bundle_is_refused_by_name(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as into:
            target = Path(into) / "schunk_egu50_hand_meshes.npz"
            target.write_bytes(b"a baked bundle")
            with self.assertRaises(self.module.DimensionsRefused) as raised:
                self.module.write_bundle("schunk_egu50", target, inflation_mm=5.0, origin="flange")

        self.assertIn("schunk_egu50_hand_meshes.npz", str(raised.exception))

    def test_an_inflation_nobody_stated_is_refused(self) -> None:
        """A safety number with no declared value is the one thing this lane does not invent."""
        import tempfile

        with tempfile.TemporaryDirectory() as into:
            with self.assertRaises(TypeError):
                self.module.write_bundle("schunk_egu50", Path(into) / "x.npz", origin="flange")  # type: ignore[call-arg]

    def test_what_it_writes_is_a_bundle_the_rest_of_the_stack_can_read(self) -> None:
        """The same arrays a baked bundle carries, plus a provenance that says these came from numbers."""
        import tempfile

        with tempfile.TemporaryDirectory() as into:
            target = Path(into) / "schunk_egu50_hand_meshes.npz"
            self.module.write_bundle("schunk_egu50", target, inflation_mm=5.0, origin="flange")

            with np.load(target, allow_pickle=True) as held:
                names = sorted(held.files)
                self.assertIn("gripper__v", names)
                self.assertIn("lfinger__f", names)
                self.assertIn("rfinger__frame", names)
                self.assertIn("hand__source", names)
                self.assertIn("hand__inflation_mm", names)
                self.assertEqual(str(np.asarray(held["hand__source"]).reshape(-1)[0]), "dimensions")
                self.assertAlmostEqual(float(np.asarray(held["hand__inflation_mm"]).reshape(-1)[0]), 5.0)
                self.assertEqual(np.asarray(held["gripper__v"]).shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
