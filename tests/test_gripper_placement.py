"""Where a gripper's spheres end up on the flange, and that the coupling is really applied.

A refusal proves a path is not taken in silence. It does not prove the number does what it says: the
plate could be added to the wrong axis, in the wrong units, or not at all, and every one of those
produces a descriptor that loads and plans. So this asserts the movement, not only the refusal.

THE FAMILY OF TEST THAT WOULD NOT CATCH IT. Adding a coupling to the wrong axis leaves every
sphere radius, every count and every pairwise distance unchanged, exactly as mirroring a symmetric
gripper leaves every extent unchanged. Assertions about sizes are blind to both. The assertions here
are about POSITION along a named axis, which is the only kind that sees it.

The module under test lives beside ``scripts/curobo/build_ur_config.py`` rather than in the backend
package, because that script runs under the cuRobo sidecar interpreter where ``src.robot...``
is un-importable. It is loaded by path here so the script and this test reach ONE implementation: the
sphere fit was written twice once already, one copy calling itself a mirror of the other with nothing
checking, and a cell planned against the wrong hand for it.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"
_HELPER = _ROOT / "scripts" / "curobo" / "_gripper_placement.py"

_spec = importlib.util.spec_from_file_location("_gripper_placement", _HELPER)
assert _spec is not None and _spec.loader is not None
placement = importlib.util.module_from_spec(_spec)
sys.modules["_gripper_placement"] = placement
_spec.loader.exec_module(placement)

FLANGE = placement.FLANGE
MOUNTING_FACE = placement.MOUNTING_FACE
PlacementError = placement.PlacementError
place = placement.place_tool0_spheres


def _spheres(name: str) -> list[dict]:
    path = _MAPS / f"{name}_gripper_spheres.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["collision_spheres"]["tool0"]


def _origin(name: str) -> str:
    path = _MAPS / f"{name}_gripper_spheres.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["_provenance"]["origin"]


class TheCouplingIsRefusedRatherThanAssumedTests(unittest.TestCase):
    def test_a_hand_measured_from_its_mounting_face_needs_the_plate(self) -> None:
        """Assuming zero puts every sphere one plate too close to the flange. That is optimistic in
        the one direction a planner must not be, and the file looks entirely reasonable."""
        with self.assertRaises(PlacementError) as caught:
            place(_spheres("robotiq_hande"), origin=MOUNTING_FACE, coupling_mm=None)
        message = str(caught.exception)
        self.assertIn("--coupling-mm", message)
        self.assertIn("tool_frame.offset_mm", message, "the operator has to know it is one measurement")

    def test_a_hand_already_at_the_flange_needs_nothing(self) -> None:
        placed = place(_spheres("ur5e"), origin=FLANGE, coupling_mm=None)
        self.assertEqual(placed, _spheres("ur5e"))

    def test_a_coupling_on_a_flange_map_is_a_contradiction_not_a_refinement(self) -> None:
        """⭐ The other direction, and it is the one an operator reaches for when a plan looks short.
        The 2F-85's map already includes wherever the arm put it, so adding a plate moves the hand
        away from where it was measured. Refused rather than added."""
        with self.assertRaises(PlacementError):
            place(_spheres("ur5e"), origin=FLANGE, coupling_mm=12.0)

    def test_an_unknown_origin_is_refused(self) -> None:
        with self.assertRaises(PlacementError):
            place(_spheres("ur5e"), origin="somewhere", coupling_mm=None)

    def test_a_negative_plate_would_put_the_hand_inside_the_wrist(self) -> None:
        with self.assertRaises(PlacementError):
            place(_spheres("robotiq_hande"), origin=MOUNTING_FACE, coupling_mm=-5.0)


class TheCouplingActuallyMovesTheHandTests(unittest.TestCase):
    """The half a refusal cannot cover."""

    def test_every_sphere_moves_by_the_plate_along_the_approach(self) -> None:
        original = _spheres("robotiq_hande")
        placed = place(original, origin=MOUNTING_FACE, coupling_mm=12.0)
        self.assertEqual(len(placed), len(original))
        for before, after in zip(original, placed, strict=True):
            self.assertAlmostEqual(after["center"][1] - float(before["center"][1]), 0.012, places=9)

    def test_nothing_else_moves(self) -> None:
        """⭐ THE DISCRIMINATING HALF. A shift applied to every axis, or to the wrong one, passes any
        assertion about how far the hand moved. This is what says it moved along the approach."""
        original = _spheres("robotiq_hande")
        placed = place(original, origin=MOUNTING_FACE, coupling_mm=12.0)
        for before, after in zip(original, placed, strict=True):
            self.assertAlmostEqual(after["center"][0], float(before["center"][0]), places=9)
            self.assertAlmostEqual(after["center"][2], float(before["center"][2]), places=9)
            self.assertAlmostEqual(after["radius"], float(before["radius"]), places=9)

    def test_millimetres_in_and_metres_out(self) -> None:
        """The map is metres and a plate is measured in millimetres. A factor of a thousand the wrong
        way is a plate 12 metres thick, which fails loudly, or 12 microns, which does not."""
        original = _spheres("robotiq_hande")
        placed = place(original, origin=MOUNTING_FACE, coupling_mm=1000.0)
        self.assertAlmostEqual(
            placed[0]["center"][1] - float(original[0]["center"][1]), 1.0, places=9
        )

    def test_zero_is_a_legitimate_answer_and_is_not_the_same_as_not_saying(self) -> None:
        """A hand bolted straight to the flange really does need nothing added. Saying so out loud is
        what separates it from having forgotten."""
        original = _spheres("robotiq_hande")
        placed = place(original, origin=MOUNTING_FACE, coupling_mm=0.0)
        self.assertEqual(
            [s["center"] for s in placed], [[float(v) for v in s["center"]] for s in original]
        )
        with self.assertRaises(PlacementError):
            place(original, origin=MOUNTING_FACE, coupling_mm=None)

    def test_the_input_is_not_mutated(self) -> None:
        """The caller writes the map back out; a shift applied in place would double on a second run."""
        original = _spheres("robotiq_hande")
        first = [list(s["center"]) for s in original]
        place(original, origin=MOUNTING_FACE, coupling_mm=12.0)
        self.assertEqual([list(s["center"]) for s in original], first)


class TheShippedMapsSayWhereTheyStartTests(unittest.TestCase):
    def test_the_hand_e_starts_at_its_own_mounting_face(self) -> None:
        """MEASURED: the Isaac Hand-E asset holds no coupling part, and its housing is 99.20 mm, the
        published body length."""
        self.assertEqual(_origin("robotiq_hande"), MOUNTING_FACE)

    def test_the_robotiq_2f85_starts_at_the_flange(self) -> None:
        """The pair that makes the field mean something. MEASURED: reading the standalone 2F-85 in
        its own root frame reproduces the committed bundle to 0.00 mm on all six corners of the palm,
        which is what says that asset root IS the flange."""
        self.assertEqual(_origin("ur5e"), FLANGE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
