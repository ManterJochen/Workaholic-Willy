"""Describing the SCENE of a cell, not just its cameras.

⭐ **WHAT THIS COVERS AND WHY IT WAS MISSING.** The brief was "camera settings AND scene-specific
things, so it can be adapted to your cell". The cameras were built and the scenes were silently
treated as covered by that work. They were not: nine numbers describing physical furniture and
placement difficulty lived as module constants in `scenes/layout.py`, where the only way to change
the bin was to edit the package.

⛔ **THE TABLE WAS WRITTEN DOWN FOUR TIMES**, once per file that needed it, and one of those in
metres. Comments held them in step: `noengine.py` pointed at `isaac.py:125`, `verify_robot.py`
pointed at the same. A comment is not a mechanism, and two engines that disagree on the table render
one scene onto two different ones.

⛔ **AND ONE CONSTANT WAS ALREADY DEAD WHILE CALLING ITSELF THE DEFAULT.**
`_PILE_FOOTPRINT_FRACTION = 0.6` was read by nothing; the live value is
`families.pile_footprint_fraction`, default 0.75. They have disagreed since the commit that made the
fraction configurable left the constant behind.
"""

from __future__ import annotations

import unittest
import warnings

import pydantic

from datagen.config import BinExceedsWorkspace, DatagenConfig
from datagen.scenes.layout import layout_scene

_BIN_ONLY = {"sparse_weight": 0.0, "packed_weight": 0.0, "pile_weight": 0.0, "bin_weight": 1.0}


def _bin_scene(**overrides: object):
    config = DatagenConfig(seed=7, families=dict(_BIN_ONLY), **overrides)   # type: ignore[arg-type]
    return layout_scene(config, 0)


def _x_span(scene) -> float:
    xs = [o.position_mm[0] for o in scene.objects]
    return max(xs) - min(xs)


class TheDescribedBinIsTheBuiltBinTests(unittest.TestCase):
    """⚠ THE WALLS ARE RENDERED AND COLLIDED FROM ONE DESCRIPTION, so a number that moved the render
    without moving the collision fixture would be worse than no wall at all. These assert the
    geometry, not the config, because the config agreeing with itself proves nothing.
    """

    @staticmethod
    def _x_wall(scene):
        return next(w for w in scene.bin_walls if w.name == "bin_x_pos")

    def test_a_wider_bin_moves_its_wall(self) -> None:
        default = self._x_wall(_bin_scene()).center_mm[0]
        wide = self._x_wall(_bin_scene(
            workspace={"half_extents_mm": (300.0, 250.0),
                       "bin": {"inner_mm": (600.0, 400.0)}})).center_mm[0]
        self.assertAlmostEqual(604.0, default, places=1)
        self.assertAlmostEqual(754.0, wide, places=1)

    def test_a_taller_wall_is_taller_in_the_geometry(self) -> None:
        wall = self._x_wall(_bin_scene(workspace={"bin": {"wall_height_mm": 220.0}}))
        self.assertAlmostEqual(110.0, wall.half_extents_mm[2], places=1)
        self.assertAlmostEqual(110.0, wall.center_mm[2], places=1,
                               msg="a wall standing on the table is centred at half its height")

    def test_the_inset_was_a_bare_literal_and_now_has_a_name(self) -> None:
        """⛔ IT WAS `- 30.0` IN THE PLACEMENT CALL, with no name and no reason written down. It is
        a jaw clearance, so a cell with a wider gripper needs it wider, and nobody tuning one could
        have found it.
        """
        self.assertLess(_x_span(_bin_scene(workspace={"bin": {"object_inset_mm": 100.0}})),
                        _x_span(_bin_scene(workspace={"bin": {"object_inset_mm": 0.0}})))


class TheBinMustFitWhereTheArmReachesTests(unittest.TestCase):
    """⛔ THIS BECAME POSSIBLE THE DAY THE BIN BECAME CONFIGURABLE. The bin family places inside the
    BIN, not inside the workspace, so a 600 x 400 KLT against the default 300 x 300 workspace centres
    a dataset 270 mm outside the patch the arm was declared to reach. `WorkspaceConfig` already says
    what that costs: it renders fine and then fails every reach, which reads as a grasping problem
    and is a geometry one.
    """

    def test_a_DECLARED_bin_wider_than_the_workspace_is_refused(self) -> None:
        with self.assertRaises(pydantic.ValidationError) as caught:
            DatagenConfig(workspace={"bin": {"inner_mm": (600.0, 400.0)}})
        message = str(caught.exception)
        self.assertIn("270", message, "the refusal names how far out it would have placed")
        self.assertIn("half_extents_mm", message, "and which key to change")

    def test_an_INHERITED_bin_is_reported_and_accepted(self) -> None:
        """⛔ THE UR3e, AND IT IS A REAL PRE-EXISTING OVERHANG. That arm reaches a 100 x 100 mm
        patch; the DEFAULT bin places out to 120 x 70, so every UR3e bin scene ever rendered put
        objects 20 mm outside the declared reach. Refusing it would refuse the very configuration
        `test_datagen_robots.py` asserts is accepted, and which this schema's own sibling refusal
        names as the workspace that DOES fit. The caller inherited the bin; they did not ask for it.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = DatagenConfig(
                workspace={"center_mm": (300.0, 0.0), "half_extents_mm": (100.0, 100.0)},
                render={"arm": {"mode": "posed", "robot_model": "ur3e"}})
        warned = [w for w in caught if issubclass(w.category, BinExceedsWorkspace)]
        self.assertEqual(1, len(warned), "the overhang was silent")
        self.assertIn("(120, 70)", str(warned[0].message))
        self.assertEqual((100.0, 100.0), config.workspace.half_extents_mm)

    def test_the_report_has_its_OWN_category(self) -> None:
        """So a caller can silence exactly this. A bare UserWarning would force them to silence
        every warning this package might ever raise."""
        self.assertTrue(issubclass(BinExceedsWorkspace, UserWarning))
        self.assertIsNot(BinExceedsWorkspace, UserWarning)

    def test_declaring_both_together_is_accepted(self) -> None:
        """The control. Without it the rule above passes for one that refuses every bin."""
        config = DatagenConfig(workspace={"half_extents_mm": (300.0, 250.0),
                                          "bin": {"inner_mm": (600.0, 400.0)}})
        self.assertEqual((600.0, 400.0), config.workspace.bin.inner_mm)

    def test_the_DEFAULT_cell_passes(self) -> None:
        """⚠ A RULE THAT FIRES ON THE DEFAULT IS A BUG, NOT A GUARD. An earlier validator on this
        model was written too sharply and six existing tests said so.
        """
        self.assertEqual((300.0, 200.0), DatagenConfig().workspace.bin.inner_mm)


class ThePlacementKnobsReachThePlacementTests(unittest.TestCase):

    def test_a_wider_sparse_margin_spreads_the_objects(self) -> None:
        def closest(margin: float) -> float:
            config = DatagenConfig(
                seed=3, scenes=1,
                families={"sparse_weight": 1.0, "packed_weight": 0.0, "pile_weight": 0.0,
                          "bin_weight": 0.0, "sparse_objects": (6, 7),   # ⚠ HALF-OPEN: (6, 6) is empty
                          "placement": {"sparse_margin_mm": margin}})
            objects = layout_scene(config, 0).objects
            return min(((a.position_mm[0] - b.position_mm[0]) ** 2
                        + (a.position_mm[1] - b.position_mm[1]) ** 2) ** 0.5
                       for i, a in enumerate(objects) for b in objects[i + 1:])
        self.assertGreater(closest(80.0), closest(2.0))

    def test_a_higher_pile_clearance_lifts_the_drop(self) -> None:
        config = DatagenConfig(
            seed=5, families={"sparse_weight": 0.0, "packed_weight": 0.0, "pile_weight": 1.0,
                              "bin_weight": 0.0, "placement": {"pile_base_clearance_mm": 50.0}})
        self.assertAlmostEqual(50.0, layout_scene(config, 0).drop_height_mm, places=1)

    def test_one_try_is_legal_and_zero_is_not(self) -> None:
        """A budget of zero would hand back the sentinel position for every object, silently."""
        DatagenConfig(families={"placement": {"max_place_tries": 1, "pile_drop_tries": 1}})
        with self.assertRaises(pydantic.ValidationError):
            DatagenConfig(families={"placement": {"max_place_tries": 0}})


class TheTableIsDeclaredOnceTests(unittest.TestCase):

    def test_no_module_still_declares_its_own(self) -> None:
        """⛔ FOUR COPIES, ONE OF THEM IN METRES, KEPT IN STEP BY COMMENTS. This is the mechanism
        those comments were standing in for.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "datagen"
        offenders = [f"{path.relative_to(root)}:{number}"
                     for path in root.rglob("*.py")
                     for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                     if line.startswith(("_TABLE_SIZE_MM =", "_TABLE_SIZE_M ="))]
        self.assertEqual([], offenders)

    def test_the_size_reaches_the_engine_free_geometry(self) -> None:
        from datagen.render.noengine import environment_geometry

        config = DatagenConfig(workspace={"table_size_mm": 800.0})
        pieces = environment_geometry(layout_scene(config, 0), config)
        widest = max(float(v[:, 0].max() - v[:, 0].min()) for v, _ in pieces)
        self.assertAlmostEqual(800.0, widest, places=1)


if __name__ == "__main__":
    unittest.main()
