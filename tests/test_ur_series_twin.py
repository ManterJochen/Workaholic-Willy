"""A UR3 and a UR3e are the two robots this stack can confuse without any check noticing.

⛔ **THE MEASUREMENT THAT MADE THIS NECESSARY.** Two checks run at connect and neither separates a
CB-series arm from its e-series namesake:

* ``_verify_controller_model`` compares SIZE CLASSES on purpose, because URSim reports ``UR3`` for a
  UR3e (measured 2026-08-19). Comparing strings would refuse a correctly configured cell.
* ``_verify_tool_frame`` compares a distance against ``verify_tolerance_mm``, default 10 mm.

Over 20000 random joint vectors on 2026-09-09, the flange separation between the twins::

    ur3  vs ur3e    min  8.50   median 21.26   max 28.90 mm     under 10 mm in 12.2 % of poses
    ur5  vs ur5e    min 59.17   median 79.34   max 95.26 mm     never
    ur10 vs ur10e   min 28.43   median 59.78   max 80.31 mm     never

So the one pair that hides inside the tolerance is the UR3 pair, which is the size this cell is
built on. It only became reachable when the CB-series models were added to ``UR_MODEL_KEYS``: before
that a cell could not name ``ur3``, so the confusion had nowhere to live.

⭐ **THE FIX IS A COMPARISON, NOT A TIGHTER THRESHOLD.** The flange to TCP transform is one physical
thing; deriving it through the wrong DH table moves it. Whichever of the two models puts it closer to
what the cell expects is the arm on the other end. That reading cannot be swallowed by a tolerance,
and it needs no vendor field that does not exist.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.drivers.ur.tool_frame import compare_tool_frames, derive_active_tool_frame
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M, ur_link_transforms_mm, ur_series_twin

#: The driver margin, mirrored so this file states what it is testing against.
_TWIN_MARGIN_MM = 1.0

#: The default tool-frame tolerance every shipped profile carries.
_VERIFY_TOLERANCE_MM = 10.0

_TWINS = (("ur3", "ur3e"), ("ur5", "ur5e"), ("ur10", "ur10e"))


def _controller_fk(model: str, joints: np.ndarray) -> np.ndarray:
    """What a controller of ``model`` would report as base -> TCP, for a cell running a BARE flange."""
    links = ur_link_transforms_mm(model, joints)
    assert links is not None, model
    return np.asarray(links[-1], dtype=np.float64)


class TheTwinRelationTests(unittest.TestCase):
    def test_every_size_with_two_series_has_a_twin_both_ways(self) -> None:
        for a, b in _TWINS:
            with self.subTest(pair=(a, b)):
                self.assertEqual(ur_series_twin(a), b)
                self.assertEqual(ur_series_twin(b), a)

    def test_a_model_whose_other_series_is_not_bundled_has_none(self) -> None:
        """⭐ THE CONTROL. Without it, a twin function returning a plausible string for everything
        would pass the test above. ur16e has no bundled UR16 row, so it must return None rather than
        a name nothing can look up."""
        self.assertNotIn("ur16", UR_DH_TABLES_M, "this control needs ur16 to stay unbundled")
        self.assertIsNone(ur_series_twin("ur16e"))
        self.assertIsNone(ur_series_twin("definitely-not-a-robot"))
        self.assertIsNone(ur_series_twin(None))

    def test_the_twins_really_are_different_robots(self) -> None:
        """A twin pair that shared a DH table would make every discrimination below meaningless, and
        would pass it. ur3 and ur3e differ by only 18.70 mm in their worst link, which is exactly why
        this is worth asserting rather than assuming."""
        for a, b in _TWINS:
            with self.subTest(pair=(a, b)):
                self.assertNotEqual(UR_DH_TABLES_M[a], UR_DH_TABLES_M[b])


class TheTwinsHideInsideTheToleranceTests(unittest.TestCase):
    """The measurement this whole file rests on, re-run rather than quoted.

    ⚠ A quoted number is a sentence with no edge to the code: if a DH row were corrected tomorrow,
    the docstring above would keep saying 8.50 mm and nothing would notice.
    """

    @staticmethod
    def _separations(a: str, b: str, n: int = 4000) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.array([
            float(np.linalg.norm(_controller_fk(a, q)[:3, 3] - _controller_fk(b, q)[:3, 3]))
            for q in rng.uniform(-math.pi, math.pi, (n, 6))
        ])

    def test_the_ur3_pair_fits_inside_the_tool_frame_tolerance(self) -> None:
        """THE HOLE. If this ever stops being true the twin check is no longer load-bearing and the
        comment explaining it has gone stale, which is worth being told about."""
        gaps = self._separations("ur3", "ur3e")
        self.assertLess(gaps.min(), _VERIFY_TOLERANCE_MM,
                        "a ur3/ur3e swap no longer hides inside the tolerance")

    def test_the_larger_pairs_do_not(self) -> None:
        """THE CONTROL, and the reason the hole is specific rather than general."""
        for a, b in (("ur5", "ur5e"), ("ur10", "ur10e")):
            with self.subTest(pair=(a, b)):
                self.assertGreater(self._separations(a, b).min(), _VERIFY_TOLERANCE_MM)


class TheDerivationTellsTheTwinsApartTests(unittest.TestCase):
    """The check itself, at the level of the two functions the driver composes.

    The driver method needs a live RTDE connection; this exercises the arithmetic it performs, on a
    controller whose FK is generated from a KNOWN model, so the right answer is not in question.
    """

    #: A cell whose controller carries no tool: the flange IS the TCP, so the derivation should
    #: return identity for the correct model.
    _EXPECTED = np.eye(4, dtype=np.float64)

    @staticmethod
    def _joints() -> np.ndarray:
        """One ordinary working pose. Not zeros: at q=0 every UR is folded out along one axis and the
        twins are at their most separated, which would make this test easier than reality."""
        return np.array([0.3, -1.2, 1.1, -1.4, -1.5, 0.2], dtype=np.float64)

    def _distances(self, truth: str, claimed: str) -> tuple[float, float]:
        """(claimed model distance, twin distance) for a controller that really is ``truth``."""
        q = self._joints()
        tcp = _controller_fk(truth, q)
        own = derive_active_tool_frame(claimed, q, tcp)
        twin = derive_active_tool_frame(ur_series_twin(claimed), q, tcp)
        assert own is not None and twin is not None
        return (compare_tool_frames(own, self._EXPECTED)[0],
                compare_tool_frames(twin, self._EXPECTED)[0])

    def test_a_correct_cell_explains_itself_best(self) -> None:
        """No false refusal: the model a cell names must beat its twin by more than the margin, in
        the right direction. A check that refuses a good cell at connect is not a safe failure."""
        for a, b in _TWINS:
            for model in (a, b):
                with self.subTest(model=model):
                    own, twin = self._distances(truth=model, claimed=model)
                    self.assertLess(own, 1e-6, "the correct DH table must reproduce the controller")
                    self.assertGreater(twin, own + _TWIN_MARGIN_MM,
                                       f"{model} and its twin are indistinguishable in this pose")

    def test_a_swapped_cell_is_explained_better_by_its_twin(self) -> None:
        """⭐ THE ONE THAT MATTERS. A ur3 config on a ur3e controller, which passes the size check and
        can pass the tolerance check, has to lose this comparison."""
        for a, b in _TWINS:
            for truth, claimed in ((a, b), (b, a)):
                with self.subTest(truth=truth, claimed=claimed):
                    own, twin = self._distances(truth=truth, claimed=claimed)
                    self.assertLess(twin, own - _TWIN_MARGIN_MM,
                                    f"a {truth} controller read as a {claimed} is not caught")

    def test_the_ur3_swap_would_survive_the_tolerance_alone(self) -> None:
        """⚠ THE PROOF THAT THE RELATIVE CHECK IS NOT REDUNDANT. Find a pose where a ur3/ur3e swap
        stays under the 10 mm tolerance, and show the comparison still refuses it. Without this, the
        twin check could be deleted tomorrow as duplicated work."""
        rng = np.random.default_rng(0)
        for q in rng.uniform(-math.pi, math.pi, (4000, 6)):
            tcp = _controller_fk("ur3e", q)
            own = derive_active_tool_frame("ur3", q, tcp)
            twin = derive_active_tool_frame("ur3e", q, tcp)
            assert own is not None and twin is not None
            d_own = compare_tool_frames(own, self._EXPECTED)[0]
            if d_own >= _VERIFY_TOLERANCE_MM:
                continue  # the tolerance would have caught this one on its own
            d_twin = compare_tool_frames(twin, self._EXPECTED)[0]
            self.assertLess(d_twin, d_own - _TWIN_MARGIN_MM,
                            f"a swap the tolerance lets through ({d_own:.2f} mm) is not caught")
            return
        self.fail("no pose under the tolerance was found, so this test proved nothing; the "
                  "measurement in the module docstring says 12.2 % of poses qualify")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
