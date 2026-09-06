"""Label density is a parameter now, and the default must not have moved a single label.

⛔ WHY THE DEFAULT IS THE FIRST TEST. Five module constants became fields on a dataclass. If any
default shifted, every label set this repo has ever written would be silently unreproducible, and the
corpus that discovered the 0.00 % result would no longer be the corpus that was measured. The
relabelling check for that lives in the measurement log; what is pinned here is the thing a future
edit would break: the dataclass defaults ARE the constants.

⚠ AND ONE BELIEF IN THIS FILE IS A CORRECTION. `vertical_approach` was built on the idea that the
top-down grasp was missing from the approach ring. It is not: MEASURED over 2,000 random closing axes,
the ring already carries the exact vertical for 1,852 of them, because `u` is `axis x z_hat` and so
`v = axis x u` IS the most-downward direction. The switch is kept for the ~7 % where the ring is out
of phase, and the test below pins that it is a small effect rather than the large one first claimed.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from datagen.grasps import labels as L


class DensityDefaultsTest(unittest.TestCase):
    """The parameterisation must be a no-op until somebody asks for something else."""

    def test_the_defaults_are_the_constants_they_replaced(self) -> None:
        self.assertEqual(L.DEFAULT_DENSITY.approach_azimuths, L._APPROACH_AZIMUTHS)
        self.assertEqual(L.DEFAULT_DENSITY.anchor_fractions, L._ANCHOR_FRACTIONS)
        self.assertEqual(L.DEFAULT_DENSITY.mesh_closing_axes, L._MESH_CLOSING_AXES)
        self.assertEqual(L.DEFAULT_DENSITY.radial_axes, L._RADIAL_AXES)
        self.assertEqual(L.DEFAULT_DENSITY.mesh_suction_contacts, L._MESH_SUCTION_CONTACTS)
        self.assertFalse(L.DEFAULT_DENSITY.vertical_approach)

    def test_every_sampler_defaults_to_the_shipped_density(self) -> None:
        """A sampler that took the density but defaulted to something else would move the corpus
        without any caller asking. Checked by calling each one with nothing but its geometry."""
        axis = np.array([1.0, 0.0, 0.0])
        self.assertEqual(len(list(L._approaches(axis))), L._APPROACH_AZIMUTHS)

    def test_the_presets_are_registered_under_the_names_the_cli_offers(self) -> None:
        """READ FROM THE CLI, not from a copy of it.

        This used to assert a hardcoded `{"default", "dense"}`, so adding a third preset failed it
        for the one reason the test does not care about, and the two lists could still have drifted
        apart in the direction it does care about: a preset that exists and no command line can ask
        for. Parsing the real argument recovers the property the name promises.
        """
        import inspect
        import re

        from datagen import __main__ as cli

        match = re.search(r'"--density".*?choices=\(([^)]*)\)',
                          inspect.getsource(cli), re.DOTALL)
        assert match is not None, "the --density argument no longer declares its choices"
        offered = set(re.findall(r'"([a-z_]+)"', match.group(1)))
        self.assertEqual(set(L.DENSITIES), offered)
        self.assertIs(L.DENSITIES["default"], L.DEFAULT_DENSITY)
        self.assertIs(L.DENSITIES["dense"], L.DENSE_DENSITY)

    def test_the_density_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            L.DEFAULT_DENSITY.approach_azimuths = 99  # type: ignore[misc]

    def test_every_field_reaches_the_stamp(self) -> None:
        """The report carries the density so a label COUNT can be read. A field that never reaches it
        makes two corpora look comparable when they are not."""
        stamp = L.DENSE_DENSITY.as_row()
        for field in dataclasses.fields(L.LabelDensity):
            self.assertIn(field.name, stamp)


class ApproachRingTest(unittest.TestCase):
    """What the ring actually contains, which is not what the switch was built believing."""

    def test_a_horizontal_closing_axis_already_carries_the_exact_vertical(self) -> None:
        """⚠ THE CORRECTION. At 12 azimuths the tilts off vertical are 0, 30, 30, 60 ... -- the
        top-down grasp is sample number three, exactly, and it always was."""
        for azimuths in (12, 24):
            density = dataclasses.replace(L.DEFAULT_DENSITY, approach_azimuths=azimuths)
            ring = np.array(list(L._approaches(np.array([1.0, 0.0, 0.0]), density)))
            tilt = np.degrees(np.arccos(np.clip(-ring[:, 2], -1.0, 1.0)))
            self.assertAlmostEqual(float(tilt.min()), 0.0, places=6,
                                   msg=f"{azimuths} azimuths lost the vertical")

    def test_the_guarantee_is_a_small_effect_and_is_pinned_as_one(self) -> None:
        """It fires for the axes whose ring is out of phase with the vertical, and for no others.

        Pinned as a RANGE rather than a claim of usefulness: the honest summary is that this adds a
        direction for roughly one axis in fourteen, and a future edit that made it fire for all of
        them would be changing the label set far more than the docstring admits.
        """
        with_guarantee = dataclasses.replace(L.DENSE_DENSITY, vertical_approach=True)
        without = dataclasses.replace(L.DENSE_DENSITY, vertical_approach=False)
        rng = np.random.default_rng(0)
        fired = 0
        for _ in range(2000):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            if len(list(L._approaches(axis, with_guarantee))) > len(list(L._approaches(axis, without))):
                fired += 1
        self.assertGreater(fired, 50, "the guarantee never fires, so it is inert")
        self.assertLess(fired, 400, "the guarantee fires far more than measured; the ring changed")

    def test_an_upright_closing_axis_is_skipped_rather_than_faked(self) -> None:
        """There IS no near-vertical approach perpendicular to an upright axis: the whole ring is
        horizontal. Inventing one would put a grasp in the set that the geometry does not admit."""
        density = dataclasses.replace(L.DENSE_DENSITY, vertical_approach=True)
        upright = np.array([0.0, 0.0, 1.0])
        self.assertEqual(len(list(L._approaches(upright, density))), density.approach_azimuths)

    def test_every_approach_is_a_unit_vector_perpendicular_to_the_axis(self) -> None:
        """Including the guaranteed one, which is constructed differently from the rest."""
        density = dataclasses.replace(L.DENSE_DENSITY, vertical_approach=True)
        rng = np.random.default_rng(3)
        for _ in range(200):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            for approach in L._approaches(axis, density):
                self.assertAlmostEqual(float(np.linalg.norm(approach)), 1.0, places=9)
                self.assertAlmostEqual(float(approach @ axis), 0.0, places=9)


class DenseSamplingTest(unittest.TestCase):
    """What `dense` changes, stated as the ratios a corpus build will actually see."""

    def test_dense_is_finer_in_every_dimension_it_claims(self) -> None:
        self.assertGreater(L.DENSE_DENSITY.approach_azimuths, L.DEFAULT_DENSITY.approach_azimuths)
        self.assertGreater(len(L.DENSE_DENSITY.anchor_fractions),
                           len(L.DEFAULT_DENSITY.anchor_fractions))
        self.assertGreater(L.DENSE_DENSITY.mesh_closing_axes, L.DEFAULT_DENSITY.mesh_closing_axes)
        self.assertGreater(L.DENSE_DENSITY.radial_axes, L.DEFAULT_DENSITY.radial_axes)

    def test_the_anchor_fractions_stay_inside_the_object(self) -> None:
        """They are fractions of a half-extent, so anything past 1.0 anchors outside the solid and
        every label from it is refused as ANCHOR_OUTSIDE -- work spent to produce nothing."""
        for fraction in L.DENSE_DENSITY.anchor_fractions:
            self.assertLess(abs(fraction), 1.0)

    def test_the_anchors_are_symmetric_about_the_centre(self) -> None:
        """An asymmetric spread biases every elongated object's labels toward one end."""
        fractions = sorted(round(f, 9) for f in L.DENSE_DENSITY.anchor_fractions)
        self.assertEqual(sorted(round(-f, 9) for f in fractions), fractions)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TheLabellerStreamsAndReportsProgressTests(unittest.TestCase):
    """⛔ It used to hoard every row in memory and log once, at the very end.

    That cost two things at the same time, and both bit on 2026-09-02 during the v5 corpus run.

    MEMORY: sixteen shards labelling 1,250 scenes each took the machine from 34 GB free to 12,
    because each held its whole output. At 477,416 jaw labels per shard that is the design, not a
    leak.

    OBSERVABILITY: a shard three hours in looked exactly like one that had died in its first minute.
    The only way to read progress was `py-spy dump --locals` against a live interpreter, reaching
    inside `label_dataset` for its own `scenes` counter. A fine tool and an absurd requirement.

    Both are the same fix: write each row as it is produced. This pins that the fix is still there,
    because a later refactor that reintroduces a list would look tidier and be neither.
    """

    def test_rows_are_written_as_they_are_produced(self) -> None:
        import inspect

        source = inspect.getsource(L.label_dataset)
        self.assertIn("handle.write(", source, "the labeller no longer streams its rows")
        self.assertNotIn("rows.append(", source,
                         "a list of every row is exactly the memory profile this replaced")

    def test_progress_is_logged_before_the_end(self) -> None:
        import inspect

        source = inspect.getsource(L.label_dataset)
        self.assertIn("scenes % 100", source,
                      "a run with no progress line is indistinguishable from a dead one")

    def test_the_report_still_counts_what_it_always_counted(self) -> None:
        """The counts moved from `len(rows)` to a running tally; the report must not have changed."""
        import inspect

        source = inspect.getsource(L.label_dataset)
        for field in ('"labels": counts["rows"]', '"jaw": counts["jaw"]',
                      '"suction": counts["suction"]'):
            self.assertIn(field, source)
