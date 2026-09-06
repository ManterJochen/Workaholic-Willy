"""Labelling a dataset for a gripper that is NOT the 2F-85.

⚠ THE DANGEROUS HALF IS THE OUTPUT FILE, not the geometry. `label_dataset` writes `grasps.jsonl`
unconditionally, and every layer beneath it already accepted a `JawModel`. The obvious repair was to
let this one accept a model too, and that would have put a second gripper's partial run on top of the
corpus labels: **8,061,895 of them**, with nothing in the call to say so.

So the function takes a NAME out of a registry rather than a model, which makes the output path a
function of the input and puts the corpus out of reach structurally instead of behind a check. These
tests pin that, because a later refactor that "simplifies" it back to a model argument would look
tidier and would be the defect.

The second thing pinned is why the variation set exists at all: the generator is conditioned on a
gripper description, and with one gripper that description never changes, so the seam carries no
gradient. These jaws are not grippers we own and no result from them is gripper generalisation.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datagen.grasps.verdict import PROCEDURAL_JAWS, JawModel


class RegistryTests(unittest.TestCase):

    def test_every_jaw_differs_from_the_shipped_one(self) -> None:
        """A registry entry equal to the 2F-85 would add scenes and no variation, which is the exact
        failure the whole set exists to avoid."""
        base = JawModel()
        for name, jaw in PROCEDURAL_JAWS.items():
            with self.subTest(name):
                self.assertNotEqual(jaw, base)

    def test_the_set_varies_the_pad_independently_of_the_aperture(self) -> None:
        """⭐ NOT A COSMETIC CHOICE. Two of the fourteen gripper numbers describe the contact patch,
        and `too_wide` is 47.6 % of every label rejection we produce. If every jaw differed only in
        aperture, nothing could tell the pad channels apart from the width ones."""
        base = JawModel()
        slim = PROCEDURAL_JAWS["slim_pad"]
        self.assertEqual(slim.aperture_mm, base.aperture_mm)
        self.assertLess(slim.pad_ahead_mm + slim.pad_behind_mm,
                        0.6 * (base.pad_ahead_mm + base.pad_behind_mm))

    def test_the_apertures_bracket_the_shipped_one(self) -> None:
        apertures = {jaw.aperture_mm for jaw in PROCEDURAL_JAWS.values()}
        self.assertTrue(any(a < 85.0 for a in apertures))
        self.assertTrue(any(a > 85.0 for a in apertures))

    def test_a_scaled_jaw_keeps_its_proportions(self) -> None:
        """Scaled from the 2F-85's own MEASURED shapes, not invented. A jaw with a 140 mm opening and
        a 2F-85 fingertip would be a gripper that exists nowhere and would teach the conditioning a
        relationship no real gripper has."""
        base, wide = JawModel(), PROCEDURAL_JAWS["wide_140"]
        ratio = wide.aperture_mm / base.aperture_mm
        for attribute in ("finger_ahead_mm", "finger_behind_mm", "finger_thickness_mm",
                          "finger_width_mm"):
            self.assertAlmostEqual(getattr(wide, attribute) / getattr(base, attribute),
                                   ratio, delta=0.02, msg=attribute)

    def test_friction_is_not_scaled(self) -> None:
        """A bigger jaw is not a grippier one: friction is a material property and scaling it would
        make the antipodal verdict move with size for no physical reason."""
        for name, jaw in PROCEDURAL_JAWS.items():
            with self.subTest(name):
                self.assertEqual(jaw.friction_coefficient, JawModel().friction_coefficient)


class OutputPathTests(unittest.TestCase):
    """⛔ The corpus file must be unreachable from a non-default gripper."""

    def test_an_unknown_jaw_is_refused_before_anything_is_written(self) -> None:
        from datagen.grasps.labels import label_dataset

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "set"
            (root / "scenes").mkdir(parents=True)
            with self.assertRaises(ValueError) as raised:
                label_dataset(root, jaw="a_gripper_we_do_not_have")
            self.assertIn("unknown jaw", str(raised.exception))
            self.assertFalse((root / "grasps.jsonl").exists(),
                             "a refused run must not have truncated anything")

    def test_the_output_name_is_a_function_of_the_jaw(self) -> None:
        """Read off the source rather than run: labelling a real dataset needs a rendered scene tree,
        and what has to be guaranteed is the NAMING rule, which is a property of the code."""
        source = Path("datagen/grasps/labels.py").read_text(encoding="utf-8")
        self.assertIn('stem = "grasps" if jaw is None else f"grasps_jaw_{jaw}"', source)
        self.assertIn('report_name = ("grasp_label_report.json" if jaw is None', source)
        self.assertNotIn('(root / "grasps.jsonl").open("w"', source,
                         "the corpus file is written unconditionally again")

    def test_label_dataset_takes_a_name_and_not_a_model(self) -> None:
        """The structural guard itself. A `model=` parameter here would restore the ability to write
        another gripper's labels into `grasps.jsonl`, which is the defect this design removes."""
        import inspect

        from datagen.grasps.labels import label_dataset

        parameters = inspect.signature(label_dataset).parameters
        self.assertIn("jaw", parameters)
        self.assertNotIn("model", parameters)
        self.assertIs(parameters["jaw"].default, None)


class ReportTests(unittest.TestCase):

    def test_the_report_stamps_which_gripper_produced_it(self) -> None:
        """Same reason the density is stamped: a label count from a 140 mm jaw and one from the 2F-85
        are not comparable, and without the stamp the difference reads as a fact about the objects."""
        source = Path("datagen/grasps/labels.py").read_text(encoding="utf-8")
        self.assertIn('"jaw_model": jaw or "2f85"', source)

    def test_the_stamp_does_not_collide_with_the_jaw_label_count(self) -> None:
        """`report["jaw"]` was already the number of jaw labels. A stamp under the same key would
        have silently replaced a count with a string in every consumer of this report."""
        source = Path("datagen/grasps/labels.py").read_text(encoding="utf-8")
        self.assertIn('"jaw": counts["jaw"]', source)


class CommandTests(unittest.TestCase):

    def test_the_flag_reaches_the_handler(self) -> None:
        from unittest import mock

        from datagen import __main__ as cli

        with mock.patch.object(cli, "_cmd_label_grasps", return_value=0) as handler:
            cli.main(["label-grasps", "--name", "v5_s0", "--jaw", "wide_140"])
        self.assertEqual(handler.call_args.kwargs["jaw"], "wide_140")

    def test_the_default_is_the_shipped_gripper(self) -> None:
        from unittest import mock

        from datagen import __main__ as cli

        with mock.patch.object(cli, "_cmd_label_grasps", return_value=0) as handler:
            cli.main(["label-grasps", "--name", "v5_s0"])
        self.assertIsNone(handler.call_args.kwargs["jaw"])


class CloudIdentityTests(unittest.TestCase):
    """⛔ The half that fails SILENTLY, and it is not the labelling.

    Two grippers' clouds carry the same scene filenames. `_refuse_foreign_overwrite` compares the
    stamped dataset name, and before the label file entered that name both grippers stamped the same
    string, so extracting the second over the first would have been accepted and 421 of 1,981 scenes
    have already been lost to that shape once.
    """

    def test_a_non_default_label_file_changes_the_stamped_identity(self) -> None:
        source = Path("datagen/corpus/clouds.py").read_text(encoding="utf-8")
        self.assertIn(
            'identity = root.name if labels == "grasps.jsonl" else f"{root.name}::{Path(labels).stem}"',
            source)

    def test_the_guard_compares_the_identity_and_not_the_bare_dataset_name(self) -> None:
        source = Path("datagen/corpus/clouds.py").read_text(encoding="utf-8")
        self.assertIn("_refuse_foreign_overwrite(target, identity)", source)
        self.assertNotIn("_refuse_foreign_overwrite(target, root.name)", source)

    def test_the_stamp_and_the_guard_read_the_same_value(self) -> None:
        """Two spellings of one identity is how a guard stops guarding without anyone noticing."""
        source = Path("datagen/corpus/clouds.py").read_text(encoding="utf-8")
        self.assertIn('"source_dataset": np.asarray([identity], dtype="<U64"),', source)


if __name__ == "__main__":
    unittest.main()
