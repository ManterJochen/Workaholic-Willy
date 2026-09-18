"""A refused cover fit says whether its reach was tried at all (customer chain lane C4i).

The fitter skips a reach below twice a body's sample spacing, and rightly, and ``fit_bundle`` then refused with the
sentence of a real miss: "a reach of r mm does not cover this body within N spheres". ``--curve`` already told the two
apart. A customer reading the first sentence raises the cap, which cannot help a reach that was never tried.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "src" / "robot" / "safety" / "data"


def _fitter():
    name = "_fit_cover_spheres_for_why"
    if name in sys.modules:
        return sys.modules[name]
    path = _REPO / "scripts" / "curobo" / "fit_cover_spheres.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ARefusedFitSaysWhyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fitter = _fitter()
        cls.fitter = fitter
        cls.bundle = _DATA / "schunk_egu50_hand_meshes.npz"
        cls.body = fitter.Body(fitter.load_meshes(cls.bundle, ["lfinger"]), samples=fitter.SAMPLES, seed=0)
        cls.spacing_mm = cls.body.spacing_m * 1000.0

    def test_a_reach_below_twice_the_sample_spacing_is_reported_as_not_tried(self) -> None:
        reach = round(self.spacing_mm, 3)
        with self.assertRaises(SystemExit) as caught:
            self.fitter.fit_bundle(self.bundle, {"lfinger": self.fitter.Ask(6, reach, 128, "test")}, reach_mm=reach,
                                   target="--hand schunk_egu50")
        said = str(caught.exception.code)
        self.assertIn("not tried", said)
        self.assertIn(f"{self.spacing_mm:.2f}", said)
        self.assertIn("fit_cover_spheres.py --hand schunk_egu50 --bodies lfinger --curve", said)
        self.assertNotIn("does not cover this body", said)

    def test_the_curve_already_called_that_rung_not_tried(self) -> None:
        """⭐ THE CONTROL: the premise, before and after: the ladder walk reports the same rung as skipped."""
        reach = round(self.spacing_mm, 3)
        self.assertEqual(list(self.body.curve(cap=128, ladder_mm=(reach,))), [(reach, None, True)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
