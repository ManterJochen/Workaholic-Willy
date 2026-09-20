"""`PublicCorpus`: the door that makes the public-corpus import reachable from a program.

⚠ WHY THIS NOUN EXISTS. The reader in `foreign/grasp_anything.py` shipped complete and had exactly
one caller: `python -m src.robot.grasping.deep import-foreign`. A user writing their own script had
no way in, and `willy` exported nothing for it, so the capability that exists for the user with no
simulator and no cell was reachable only from a shell. The CLI handler owned the glue -- validating
the gripper, printing the licence, turning a provenance dict into something a program can branch on
-- and a second caller would have written all of it again.

Everything here runs OFFLINE. The fetch itself is the network, exercised by hand and by
`examples/offline/training/04_train_on_a_public_corpus.py`; what is pinned here is that nothing
reaches the network before it has to, that a bad gripper is refused before a byte is downloaded, and
that a provenance block becomes the report a caller reads.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.robot.grasping.deep.foreign.service import (
    DEFAULT_SOURCE, ImportReport, PublicCorpus, public_sources)

#: A provenance block in the shape `import_scenes` returns, so the report can be pinned with no fetch.
_PROVENANCE = {
    "source": "grasp_anything_6d", "licence": "MIT", "gripper": "wide_140",
    "url": "https://huggingface.co/datasets/airvlab/Grasp-Anything-6D",
    "written": 24, "skipped_present": 2, "no_usable_grasp": 1, "failed": 0,
    "training_units": 46, "contract": {"checked": 3, "failures": []},
}


class WhatItSaysBeforeItFetchesTests(unittest.TestCase):
    """`describe()` is the call a program makes first, and it must cost nothing."""

    def test_describing_a_corpus_opens_no_connection(self) -> None:
        import socket

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("describe() reached the network")

        with TemporaryDirectory() as tmp:
            corpus = PublicCorpus.from_source(out_dir=Path(tmp) / "public")
            original = socket.socket.connect
            socket.socket.connect = refuse                     # type: ignore[method-assign]
            try:
                text = corpus.describe()
            finally:
                socket.socket.connect = original               # type: ignore[method-assign]
        self.assertIn("grasp_anything_6d", text)
        self.assertIn("MIT", text, "the licence travels with the data and must be said up front")
        self.assertIn("0 scene(s) already imported", text)

    def test_it_is_ascii_because_it_reaches_a_cp1252_console(self) -> None:
        with TemporaryDirectory() as tmp:
            PublicCorpus.from_source(out_dir=tmp).describe().encode("ascii")

    def test_the_listing_names_every_source_and_its_licence(self) -> None:
        rows = public_sources()
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["licence"], row["key"])
            self.assertTrue(row["url"].startswith("https://"), row["key"])
        self.assertIn(DEFAULT_SOURCE, {row["key"] for row in rows})

    def test_an_already_imported_corpus_counts_what_is_there(self) -> None:
        """A fetch resumes, so the number a caller sees before one is the number already on disk."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "public"
            out.mkdir()
            for name in ("a", "b", "c"):
                (out / f"{name}.npz").write_bytes(b"")
            (out / "notes.txt").write_text("not a scene", encoding="utf-8")
            self.assertEqual(3, PublicCorpus.from_source(out_dir=out).scenes_present())


class WhatItRefusesBeforeItCostsAnythingTests(unittest.TestCase):

    def test_an_unknown_source_lists_what_there_was(self) -> None:
        with self.assertRaises(ValueError) as caught:
            PublicCorpus.from_source("a_corpus_from_the_future", out_dir="x")
        self.assertIn(DEFAULT_SOURCE, str(caught.exception))

    def test_an_unknown_gripper_is_refused_at_CONSTRUCTION_not_after_the_download(self) -> None:
        """A 229 GB source is the wrong place to learn that a hand name was a typo."""
        with self.assertRaises(ValueError) as caught:
            PublicCorpus.from_source(out_dir="x", gripper="no_such_hand")
        self.assertIn("no_such_hand", str(caught.exception))

    def test_the_default_gripper_is_the_readers_own(self) -> None:
        from src.robot.grasping.deep.foreign.grasp_anything import DEFAULT_GRIPPER

        self.assertEqual(DEFAULT_GRIPPER, PublicCorpus.from_source(out_dir="x").gripper)


class WhatTheReportSaysTests(unittest.TestCase):

    def _report(self, **changed: object) -> ImportReport:
        corpus = PublicCorpus.from_source(out_dir="x")
        return corpus._report({**_PROVENANCE, **changed})              # noqa: SLF001

    def test_the_counts_and_the_licence_survive_the_provenance_block(self) -> None:
        report = self._report()
        self.assertEqual(26, report.scenes, "written plus already present")
        self.assertEqual(46, report.training_units)
        self.assertEqual("MIT", report.licence)
        self.assertTrue(report.ok)

    def test_a_run_that_wrote_nothing_and_found_nothing_is_not_ok(self) -> None:
        self.assertFalse(self._report(written=0, skipped_present=0).ok)

    def test_a_run_that_wrote_nothing_because_it_was_ALREADY_there_is_ok(self) -> None:
        """A resumed fetch that had nothing left to do is a success, not a failure."""
        self.assertTrue(self._report(written=0, skipped_present=24).ok)

    def test_a_contract_failure_is_not_ok_even_with_scenes_written(self) -> None:
        """Files on disk that the loader cannot read are worse than no files: training starts."""
        report = self._report(contract={"checked": 3, "failures": ["scene_a: points_m in mm"]})
        self.assertFalse(report.ok)
        self.assertIn("points_m in mm", report.render())

    def test_it_prints_as_its_report_and_stays_ascii(self) -> None:
        report = self._report()
        self.assertEqual(report.render(), str(report))
        report.render().encode("ascii")

    def test_as_dict_is_json_safe(self) -> None:
        json.dumps(self._report().as_dict())


class ReadingTheLastImportBackTests(unittest.TestCase):

    def test_a_directory_with_no_import_reports_none(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertIsNone(PublicCorpus.from_source(out_dir=tmp).report())

    def test_an_unreadable_stamp_reports_none_rather_than_raising(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "provenance.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(PublicCorpus.from_source(out_dir=tmp).report())

    def test_a_written_stamp_comes_back_as_the_report(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "provenance.json").write_text(json.dumps(_PROVENANCE), encoding="utf-8")
            report = PublicCorpus.from_source(out_dir=tmp).report()
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(26, report.scenes)


class TheDoorIsTheSameOneTheCommandUsesTests(unittest.TestCase):

    def test_the_command_refuses_a_bad_gripper_through_this_noun(self) -> None:
        import contextlib
        import io

        from src.robot.grasping.deep.__main__ import main

        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["import-foreign", "--out", "x", "--gripper", "no_such_hand"])
        self.assertEqual(2, code, "a bad argument exits 2, and reaches no network")
        self.assertIn("no_such_hand", err.getvalue())

    def test_willy_exports_it(self) -> None:
        import willy

        self.assertIs(PublicCorpus, willy.PublicCorpus)


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
