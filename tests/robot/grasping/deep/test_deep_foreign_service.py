"""`PublicCorpus`: the door that makes the public-corpus import reachable from a program.

⚠ WHY THIS NOUN EXISTS. The reader in `foreign/grasp_anything.py` shipped complete and had exactly
one caller: `python -m src.robot.grasping.deep import-foreign`. A user writing their own
script had no way in, so the capability that exists for the user with no simulator and no cell was
reachable only from a shell. The CLI handler owned the glue (validating the gripper, printing the
licence, turning a provenance dict into something a program can branch on), and a second caller
would have written all of it again.

Everything here runs OFFLINE. The fetch itself is the network and is exercised by hand; what is
pinned here is that nothing reaches the network before it has to, that a bad gripper is refused
before a byte is downloaded, and that a provenance block becomes the report a caller reads. Where a
test goes through `fetch()`, the reader's `import_scenes` is patched and every socket connect is
refused, so a regression that reached for the archive fails here instead of downloading.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import unittest
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.robot.grasping.deep.foreign import grasp_anything
from src.robot.grasping.deep.foreign.service import (
    DEFAULT_SOURCE, ImportReport, PublicCorpus, public_sources)

#: A provenance block in the shape `import_scenes` returns, so the report can be pinned with no fetch.
_PROVENANCE = {
    "source": "grasp_anything_6d", "licence": "MIT", "gripper": "wide_140",
    "url": "https://huggingface.co/datasets/airvlab/Grasp-Anything-6D",
    "written": 24, "skipped_present": 2, "no_usable_grasp": 1, "failed": 0,
    "training_units": 46, "contract": {"checked": 3, "failures": []},
}


@contextlib.contextmanager
def _no_network() -> Iterator[None]:
    """Every socket connect raises for the duration, so reaching the archive is a failure here."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("this test reached the network")

    with mock.patch.object(socket.socket, "connect", refuse):
        yield


class WhatItSaysBeforeItFetchesTests(unittest.TestCase):
    """`describe()` is the call a program makes first, and it must cost nothing."""

    def test_describing_a_corpus_opens_no_connection(self) -> None:
        with TemporaryDirectory() as tmp:
            corpus = PublicCorpus.from_source(out_dir=Path(tmp) / "public")
            with _no_network():
                text = corpus.describe()
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
            PublicCorpus.from_source(key="a_corpus_from_the_future", out_dir="x")
        self.assertIn(DEFAULT_SOURCE, str(caught.exception))

    def test_an_unknown_gripper_is_refused_at_CONSTRUCTION_not_after_the_download(self) -> None:
        """A 229 GB source is the wrong place to learn that a hand name was a typo."""
        with self.assertRaises(ValueError) as caught:
            PublicCorpus.from_source(out_dir="x", gripper="no_such_hand")
        self.assertIn("no_such_hand", str(caught.exception))

    def test_the_refusal_names_the_hands_in_the_order_the_command_always_printed(self) -> None:
        """The command printed this refusal before the noun existed, sorted. Moving the check into
        the noun must not reorder what an operator reads."""
        from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY

        with self.assertRaises(ValueError) as caught:
            PublicCorpus.from_source(out_dir="x", gripper="no_such_hand")
        self.assertTrue(str(caught.exception).endswith(", ".join(sorted(JAW_GEOMETRY))))

    def test_the_default_gripper_is_the_readers_own(self) -> None:
        self.assertEqual(grasp_anything.DEFAULT_GRIPPER,
                         PublicCorpus.from_source(out_dir="x").gripper)

    def test_the_factory_takes_keywords_only(self) -> None:
        """The calling convention: a positional corpus key reads as an out_dir at a glance."""
        with self.assertRaises(TypeError):
            PublicCorpus.from_source(DEFAULT_SOURCE, out_dir="x")  # type: ignore[misc]


class WhatTheFetchForwardsTests(unittest.TestCase):
    """The reader owns its defaults. `fetch()` forwards what the caller chose and nothing else, so
    the two cannot drift apart the way a copied default drifts."""

    def _fetch(self, **chosen: object) -> tuple[ImportReport, dict[str, object]]:
        with TemporaryDirectory() as tmp:
            corpus = PublicCorpus.from_source(out_dir=tmp)
            with (_no_network(),
                  mock.patch.object(grasp_anything, "import_scenes",
                                    return_value=dict(_PROVENANCE)) as fetched):
                report = corpus.fetch(**chosen)                        # type: ignore[arg-type]
        return report, dict(fetched.call_args.kwargs)

    def test_nothing_chosen_forwards_only_the_hand(self) -> None:
        report, forwarded = self._fetch()
        self.assertEqual({"gripper": grasp_anything.DEFAULT_GRIPPER}, forwarded)
        self.assertEqual(26, report.scenes)

    def test_a_chosen_value_reaches_the_reader_as_given(self) -> None:
        _, forwarded = self._fetch(limit=5, jobs=2, validate=0, centre_offset_m=None)
        self.assertEqual(5, forwarded["limit"])
        self.assertEqual(2, forwarded["jobs"])
        self.assertEqual(0, forwarded["validate"], "a chosen 0 is a value, not absence")
        self.assertIn("centre_offset_m", forwarded, "a chosen None is a value, not absence")
        self.assertIsNone(forwarded["centre_offset_m"])


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

    def test_the_provenance_it_carries_cannot_be_edited_through_the_report(self) -> None:
        report = self._report()
        with self.assertRaises(TypeError):
            report.provenance["written"] = 0                           # type: ignore[index]


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
        from src.robot.grasping.deep.__main__ import main

        err = io.StringIO()
        with (_no_network(), contextlib.redirect_stderr(err),
              contextlib.redirect_stdout(io.StringIO())):
            code = main(["import-foreign", "--out", "x", "--gripper", "no_such_hand"])
        self.assertEqual(2, code, "a bad argument exits 2, and reaches no network")
        self.assertIn("no_such_hand", err.getvalue())

    def test_the_command_says_what_the_noun_describes_before_it_fetches(self) -> None:
        """The command prints `describe()`, so the shell and a program read the same header.

        The reader's `import_scenes` is patched for both the old and the new handler shape, and every
        connect is refused, so this cannot download whichever way the handler is written.
        """
        from src.robot.grasping.deep.__main__ import main

        out = io.StringIO()
        with (TemporaryDirectory() as tmp, _no_network(),
              mock.patch.object(grasp_anything, "import_scenes", return_value=dict(_PROVENANCE)),
              contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO())):
            code = main(["import-foreign", "--out", tmp, "--limit", "3"])
            expected = PublicCorpus.from_source(out_dir=tmp, gripper="wide_140").describe()
        self.assertEqual(0, code)
        self.assertIn(expected, out.getvalue())

    def test_willy_exports_it(self) -> None:
        import willy

        self.assertIs(PublicCorpus, willy.PublicCorpus)


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
