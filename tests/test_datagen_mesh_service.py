"""Preparing a customer's own meshes, from Python, with the prefix trap named and tested.

⭐ **WHY THIS PATH.** Everything a customer does to their OWN parts before a scene is rendered lived
in argparse handlers. `decompose` had no library twin at all, so the one step that makes a MuJoCo
build affordable was reachable only from a shell, and `normalise-meshes`, `screen-meshes` and
`why-no-jaw` each had a twin that took one collection or a list of already-chosen meshes, with
everything that DECIDES the list living in `__main__`.

⛔⛔ **THE PREFIX TRAP, WHICH THIS REPOSITORY HAS PAID FOR FOUR TIMES.** Asset ids sort by source and
then by name, so `sorted(...)[:N]` is not a sample of a library, it is one collection. `why-no-jaw`
used to append until it had `limit` entries and break out of both loops. The same shape produced a
multimodality claim standing on eight files that were all `packed`, a held-out set whose ceiling was
0.580 against the population's 0.385, and `deep propose`'s "40 unseen scenes" that were 40 `packed`
ones and carried the first referee success rate this repository ever computed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from datagen.assets.service import (
    DecompositionReport,
    FetchReport,
    SourceFetch,
    available_sources,
    JawDiagnosis,
    MeshPreparation,
    ScreenReport,
    even_sample,
    library_sources,
)


class TheDrawIsEvenNeverAPrefixTests(unittest.TestCase):

    #: A library shaped like the real one: sorted, so each source is a contiguous block.
    LIBRARY = ([("gso_%03d" % i, f"g{i}.obj") for i in range(700)]
               + [("ycb_%03d" % i, f"y{i}.obj") for i in range(50)]
               + [("asos_%03d" % i, f"a{i}.obj") for i in range(50)]
               + [("objaverse_%03d" % i, f"o{i}.obj") for i in range(200)])

    @staticmethod
    def _families(picked) -> Counter:
        return Counter(asset_id.rsplit("_", 1)[0] for asset_id, _ in picked)

    def test_a_prefix_of_twenty_is_ONE_collection(self) -> None:
        """The defect, stated as a measurement rather than as a worry."""
        self.assertEqual({"gso": 20}, dict(self._families(self.LIBRARY[:20])))

    def test_an_even_draw_of_twenty_reaches_every_collection(self) -> None:
        families = self._families(even_sample(self.LIBRARY, 20))
        self.assertGreater(len(families), 1, "the draw is still one collection")
        for source in ("gso", "ycb", "asos", "objaverse"):
            self.assertIn(source, families, f"{source} is invisible to this draw")

    def test_the_draw_is_proportional_rather_than_equal(self) -> None:
        """⚠ EVEN OVER THE LIST, NOT EQUAL PER SOURCE, and the difference matters. A draw that took
        five from each would over-represent a 50-mesh collection fifteenfold against a 700-mesh one
        and answer a different question than "what does this library look like"."""
        families = self._families(even_sample(self.LIBRARY, 100))
        self.assertGreater(families["gso"], families["ycb"] * 5)

    def test_asking_for_more_than_exists_returns_everything_once(self) -> None:
        self.assertEqual(len(self.LIBRARY), len(even_sample(self.LIBRARY, 99_999)))
        self.assertEqual(len(set(even_sample(self.LIBRARY, 5))), 5, "the draw repeats an item")

    def test_a_zero_or_negative_limit_means_no_limit(self) -> None:
        self.assertEqual(len(self.LIBRARY), len(even_sample(self.LIBRARY, 0)))

    def test_an_empty_library_does_not_raise(self) -> None:
        self.assertEqual([], even_sample([], 10))


class TheSourceRuleIsWrittenOnceTests(unittest.TestCase):
    """⚠ THREE COMMANDS SPELLED IT OUT SEPARATELY. A source the fetcher can download but
    `SUPPORTED_SOURCES` does not carry is invisible to all of them."""

    def test_naming_none_means_every_supported_source(self) -> None:
        from datagen.assets.library import SUPPORTED_SOURCES

        self.assertEqual([s for s in SUPPORTED_SOURCES if s != "custom"], library_sources())

    def test_custom_is_excluded_from_the_default_sweep(self) -> None:
        """⚠ ON PURPOSE. A customer's own meshes are the point of the package and are the one
        collection nobody can re-fetch, so a bulk operation never touches them unless it is named."""
        self.assertNotIn("custom", library_sources())

    def test_but_it_is_reachable_when_asked_for(self) -> None:
        self.assertEqual(["custom"], library_sources(["custom"]))


class TheScreenWritesBOTHFilesTests(unittest.TestCase):
    """⛔ THE SECOND FILE WAS HANDLER-ONLY. `screen_meshes` writes the screen; the
    `<stem>_asset_ids.json` that every downstream config actually points at was derived and written
    in `__main__`, so a Python caller got the screen and not the thing built from it."""

    ROWS = [{"asset_id": "gso_a", "status": "ok", "jaw": 12},
            {"asset_id": "gso_b", "status": "ok", "jaw": 0},
            {"asset_id": "ycb_c", "status": "unreadable", "jaw": 0}]

    def test_the_asset_id_list_lands_beside_the_screen(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "screen_v9.json"
            with mock.patch("datagen.assets.prepare.screen_meshes", return_value=self.ROWS), \
                    mock.patch("datagen.assets.prepare.asset_ids_from_screen",
                               return_value=["gso_a"]):
                report = MeshPreparation.from_sources(["gso"]).screen(out)
            self.assertEqual(Path(name) / "screen_v9_asset_ids.json", report.ids_path)
            assert report.ids_path is not None
            self.assertEqual(["gso_a"], json.loads(report.ids_path.read_text(encoding="utf-8")))

    def test_it_is_written_with_indent_1_as_the_cli_always_did(self) -> None:
        """⚠ THIS FILE IS COMMITTED. Reformatting it would put a whole-file diff in front of the
        next reader for nothing."""
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "s.json"
            with mock.patch("datagen.assets.prepare.screen_meshes", return_value=self.ROWS), \
                    mock.patch("datagen.assets.prepare.asset_ids_from_screen",
                               return_value=["a", "b"]):
                report = MeshPreparation.from_sources(["gso"]).screen(out)
            assert report.ids_path is not None
            self.assertEqual('[\n "a",\n "b"\n]', report.ids_path.read_text(encoding="utf-8"))

    def test_the_count_of_graspable_meshes_reads_the_rows(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with mock.patch("datagen.assets.prepare.screen_meshes", return_value=self.ROWS), \
                    mock.patch("datagen.assets.prepare.asset_ids_from_screen", return_value=[]):
                report = MeshPreparation.from_sources(["gso"]).screen(Path(name) / "s.json")
        self.assertEqual((3, 1), (report.rows, report.graspable))

    def test_write_ids_False_skips_the_second_file(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with mock.patch("datagen.assets.prepare.screen_meshes", return_value=self.ROWS):
                report = MeshPreparation.from_sources(["gso"]).screen(
                    Path(name) / "s.json", write_ids=False)
        self.assertIsNone(report.ids_path)


class TheDiagnosisAsksAboutTheRightMeshesTests(unittest.TestCase):

    ENTRIES = [(f"gso_{i}", f"g{i}.obj") for i in range(10)]

    def _prepared(self) -> MeshPreparation:
        preparation = MeshPreparation.from_sources(["gso"])
        preparation.entries = lambda: list(self.ENTRIES)          # type: ignore[method-assign]
        return preparation

    def test_from_screen_narrows_to_the_meshes_that_earned_nothing(self) -> None:
        """⭐ THE SHARPER QUESTION. Without it the probe samples EVERYTHING, and meshes that earn no
        jaw label are a minority, so most of the budget goes on objects that already work."""
        asked: list = []

        def capture(meshes, **kwargs):                 # type: ignore[no-untyped-def]
            asked.extend(meshes)
            return {"probed": len(meshes), "unreadable": 0, "with_no_label_in_any_pose": 0,
                    "rescued_by_a_different_pose": 0, "reasons": {}}

        with tempfile.TemporaryDirectory() as name:
            screen = Path(name) / "s.json"
            screen.write_text(json.dumps([
                {"asset_id": "gso_3", "status": "ok", "jaw": 0},
                {"asset_id": "gso_7", "status": "ok", "jaw": 0},
                {"asset_id": "gso_1", "status": "ok", "jaw": 44}]), encoding="utf-8")
            with mock.patch("datagen.assets.diagnose.why_no_jaw", side_effect=capture):
                self._prepared().why_no_jaw(from_screen=screen)
        self.assertEqual({"gso_3", "gso_7"}, {asset_id for asset_id, _ in asked})

    def test_a_screen_with_no_zero_rows_refuses_rather_than_probing_everything(self) -> None:
        """Falling back to the whole library would answer a question nobody asked, expensively."""
        with tempfile.TemporaryDirectory() as name:
            screen = Path(name) / "s.json"
            screen.write_text(json.dumps([{"asset_id": "gso_1", "status": "ok", "jaw": 4}]),
                              encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                self._prepared().why_no_jaw(from_screen=screen)
        self.assertIn("nothing to ask about", str(caught.exception))

    def test_an_empty_library_refuses_with_the_fetch_command(self) -> None:
        preparation = MeshPreparation.from_sources(["gso"])
        preparation.entries = lambda: []                          # type: ignore[method-assign]
        with self.assertRaises(ValueError) as caught:
            preparation.why_no_jaw()
        self.assertIn("datagen.assets.fetch", str(caught.exception))

    def test_the_summary_drops_the_per_candidate_table(self) -> None:
        """⚠ `verdicts` IS THE BULK OF THE OBJECT. A summary that carries it is not a summary."""
        diagnosis = JawDiagnosis(raw={"probed": 2, "verdicts": [1] * 10_000, "reasons": {}},
                                 probed_meshes=2)
        self.assertNotIn("verdicts", diagnosis.as_dict())
        self.assertIn("probed", diagnosis.as_dict())


class TheDownloadIsPartOfTheSameServiceTests(unittest.TestCase):
    """⛔ IT WAS NOT REACHABLE FROM PYTHON IN ANY USEFUL FORM. `datagen.assets.fetch.fetch` exists,
    but it returns a PROCESS EXIT CODE and prints its counts, so a caller who wanted to know how many
    meshes arrived had to scrape stdout. An exit code is what a shell needs; a report is what a
    program needs, and the same run can produce both. It is step zero of everything else in this
    class, so it belongs on the same noun.
    """

    @staticmethod
    def _counts(**over: int) -> dict[str, int]:
        base = {"fetched": 12, "skipped_present": 3, "skipped_license": 1, "failed": 0}
        base.update(over)
        return base

    @staticmethod
    def _with_fetcher(key: str, fetcher):                     # type: ignore[no-untyped-def]
        """⚠ `Source` IS A FROZEN DATACLASS, so `patch.object` on its `fetch` field raises. The entry
        is replaced instead, which is also closer to what a caller sees."""
        import dataclasses

        from datagen.assets import fetch as fetch_module

        replaced = dict(fetch_module.SOURCES)
        replaced[key] = dataclasses.replace(replaced[key], fetch=fetcher)
        return mock.patch.dict(fetch_module.SOURCES, replaced, clear=True)

    def _fetched(self, counts: dict[str, int], key: str = "gso") -> FetchReport:
        with self._with_fetcher(key, lambda dest, limit, report: counts),                 tempfile.TemporaryDirectory() as d:
            return MeshPreparation.from_sources([key]).fetch(library=d)

    def test_the_counts_arrive_as_a_report_rather_than_as_printed_text(self) -> None:
        report = self._fetched(self._counts())
        self.assertEqual(1, len(report.sources))
        self.assertEqual((12, 3, 1, 0), (report.sources[0].fetched,
                                         report.sources[0].already_present,
                                         report.sources[0].skipped_licence,
                                         report.sources[0].failed))
        self.assertTrue(report.ok)

    def test_a_missing_scan_is_NOT_counted_as_a_failure(self) -> None:
        """⚠ THE VERDICT MUST AGREE WITH THE SENTENCE. 20 of YCB's 103 objects were captured only on
        the Berkeley rig; counting them as errors makes a healthy run look broken and buries the ones
        that really did fail."""
        report = self._fetched(self._counts(skipped_absent=20), key="ycb")
        self.assertEqual(20, report.sources[0].skipped_absent)
        self.assertTrue(report.ok, "a source with no scan for some objects failed the run")

    def test_a_real_failure_DOES_fail_the_run(self) -> None:
        """The control. Without it the test above passes for a report that is always ok."""
        self.assertFalse(self._fetched(self._counts(failed=4)).ok)

    def test_a_refused_source_is_recorded_rather_than_raised(self) -> None:
        """One collection that cannot be reached must not lose the counts of the others."""
        def refuse(dest, limit, report):                   # type: ignore[no-untyped-def]
            raise RuntimeError("no network")

        with self._with_fetcher("gso", refuse), tempfile.TemporaryDirectory() as d:
            report = MeshPreparation.from_sources(["gso"]).fetch(library=d)
        self.assertIn("no network", report.sources[0].refused)
        self.assertFalse(report.ok)

    def test_the_attribution_obligation_is_in_the_rendered_text(self) -> None:
        """⚠ CC-BY MAKES ATTRIBUTION OBLIGATORY, and this repository's licence audit has already had
        two holes: `cc-by-sa` and `cc-by-nd` both PASSED a `startswith("cc-by")` test."""
        text = self._fetched(self._counts()).render()
        text.encode("ascii")
        self.assertIn("Attribution is obligatory", text)

    def test_the_source_table_says_which_licences_were_VERIFIED(self) -> None:
        """`licence_verified=False` means the collection publishes terms for itself and states
        nothing per object, so a dataset shipped on that basis inherits a claim, not a check."""
        rows = {row["key"]: row for row in available_sources()}
        self.assertTrue(rows["gso"]["licence_verified"])
        self.assertFalse(rows["ycb"]["licence_verified"],
                         "ycb states no per-object licence; saying it does would be a false claim")

    def test_as_dict_survives_json(self) -> None:
        json.dumps(FetchReport(library=Path("a"), sources=(SourceFetch(
            key="gso", fetched=1, already_present=0, skipped_licence=0, skipped_absent=0,
            failed=0, licence="CC-BY-4.0", licence_verified=True),)).as_dict())


class TheLibraryPathIsDeclaredOnceTests(unittest.TestCase):
    """⛔ TWO DECLARATIONS OF `assets/meshes`, held in step by a comment claiming the copy kept
    `--list` working in a dependency-free clone. It did not: `datagen/assets/__init__.py` pulls numpy,
    so importing `fetch` at all already required it. The copy was protecting a property the package
    had already lost."""

    def test_every_reader_gets_the_same_object(self) -> None:
        from datagen.assets import fetch as fetch_module
        from datagen.assets.library import MESH_LIBRARY_DIR as from_library
        from datagen.constants import MESH_LIBRARY_DIR as declared

        self.assertIs(declared, from_library)
        self.assertIs(declared, fetch_module._DEFAULT_LIBRARY)

    def test_the_declaration_costs_nothing_to_import(self) -> None:
        """It sits in `constants.py` precisely so a module that cannot afford `library.py` can read
        it. That property is the reason for the file, so it is asserted rather than assumed."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; from datagen.constants import MESH_LIBRARY_DIR; "
             "print('numpy' in sys.modules)"],
            capture_output=True, text=True, check=True)
        self.assertEqual("False", result.stdout.strip())


class TheReportsReadAsTextTests(unittest.TestCase):
    """⚠ A Windows console under cp1252 cannot print anything but ASCII, and a non-ASCII character
    in a printed report line has broken this repository's CLI before."""

    def test_every_report_renders_ascii(self) -> None:
        DecompositionReport(assets=463, computed=400, already_cached=63, parts=3_918,
                            cache_dir=Path("c")).render().encode("ascii")
        ScreenReport(screen_path=Path("s.json"), ids_path=Path("i.json"), rows=10,
                     graspable=4).render().encode("ascii")
        JawDiagnosis(raw={"probed": 1, "unreadable": 0, "with_no_label_in_any_pose": 1,
                          "rescued_by_a_different_pose": 0,
                          "reasons": {"too_wide": {"count": 12, "share": 0.693}}},
                     probed_meshes=1).render().encode("ascii")

    def test_an_empty_decomposition_does_not_divide_by_zero(self) -> None:
        self.assertEqual(0.0, DecompositionReport(assets=0, computed=0, already_cached=0, parts=0,
                                                  cache_dir=Path("c")).parts_per_asset)

    def test_a_missing_decomposer_refuses_WITH_THE_REASON(self) -> None:
        """⚠ IT USED TO ASSERT THE INSTALL LINE, and that was the defect one level up: a fixed
        sentence is right for one of the two faults and a false instruction for the other. What the
        refusal must carry is whatever `decomposition_refusal()` established, verbatim."""
        from datagen.config import DatagenConfig

        with mock.patch("datagen.render.convex_decomposition.decomposition_refusal",
                        return_value="CoACD IS installed, and its native library could not be "
                                     "LOADED ([WinError 4551])"),                 self.assertRaises(RuntimeError) as caught:
            MeshPreparation.from_sources().decompose(DatagenConfig())
        message = str(caught.exception)
        self.assertIn("4551", message)
        self.assertNotIn("pip install", message,
                         "a blocked library must not be reported as a missing one")


if __name__ == "__main__":
    unittest.main()


class DocumentedConstantsAreLoadBearingTests(unittest.TestCase):
    """⚠ A CONSTANT NOTHING READS IS EITHER DEAD OR IT IS DOCUMENTATION, and the two look identical
    to a sweep. `_FINGER_CLEARANCE_MM` was reported as orphaned because no code reads it; what reads
    it is a COMMENT, explaining why the mug handle's reach band is what it is. Deleting it would have
    deleted the explanation. Asserting the relationship makes it a mechanism instead."""

    def test_the_mug_handle_band_straddles_the_finger_clearance(self) -> None:
        """⛔ THE STRADDLE IS THE POINT. A dataset where every handle is graspable would teach a model
        that handles always are, which is false and would cost it on the first real mug. The first
        version of these families used an 18-30 mm reach and produced ZERO handle grasps, with
        `FINGER_COLLISION` as the sole rejection."""
        import inspect
        import re

        from datagen.assets import composite

        source = inspect.getsource(composite._mug)
        low, high = (float(v) for v in
                     re.search(r"reach = float\(rng\.uniform\(([\d.]+), ([\d.]+)\)\)",
                               source).groups())
        self.assertLess(low, composite._FINGER_CLEARANCE_MM,
                        "no handle in this band is pressed close enough to refuse the jaw")
        self.assertGreater(high, composite._FINGER_CLEARANCE_MM,
                           "no handle in this band stands clear enough to admit the jaw")


class AbsentAndUnloadableAreTwoFaultsTests(unittest.TestCase):
    """⛔⛔ ONE SENTENCE FOR BOTH IS WORSE THAN NO SENTENCE.

    Widening an import guard from `ImportError` to `(ImportError, OSError)` is half a repair, and
    this package shipped the half. It then told an operator "CoACD is not installed here: `pip
    install coacd`" for a library that IS installed and that the machine refused to LOAD. Sending
    someone to reinstall a package they already have costs them the one thing the raw error gave
    them: the name of the policy that blocked it.

    ⚠ MEASURED ON THIS BOX THE SAME DAY, on a sibling library. Windows Smart App Control began
    refusing `mujoco.dll` days after it was installed, and the import raised `OSError:
    [WinError 4551]`. The distinction is not hypothetical here, and a peer session found the false
    instruction before it reached anyone standing next to a live arm.
    """

    @staticmethod
    def _refusal_when(raising: BaseException | None) -> "str | None":
        """`decomposition_refusal()` with the import forced to a chosen outcome."""
        import builtins

        from datagen.render import convex_decomposition

        real = builtins.__import__

        def fake(name, *args, **kwargs):                       # type: ignore[no-untyped-def]
            if name == "coacd" and raising is not None:
                raise raising
            return real(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", fake):
            return convex_decomposition.decomposition_refusal()

    def test_absent_says_install_it(self) -> None:
        reason = self._refusal_when(ImportError("No module named 'coacd'"))
        assert reason is not None
        self.assertIn("pip install coacd", reason)

    def test_unloadable_says_the_OPPOSITE_of_install_it(self) -> None:
        """⛔ THE HALF THAT WAS MISSING. `IS installed` and `reinstalling will not help` are the two
        facts the operator needs, and the old message asserted the negation of both."""
        reason = self._refusal_when(OSError("[WinError 4551] a policy blocked this file"))
        assert reason is not None
        self.assertIn("IS installed", reason)
        self.assertIn("reinstalling will not help", reason)
        self.assertNotIn("pip install coacd", reason,
                         "a blocked library must not be reported as a missing one")

    def test_unloadable_carries_the_ORIGINAL_error(self) -> None:
        """The policy name is what the raw OSError gave, and losing it is what made the old message
        worse than no message."""
        reason = self._refusal_when(OSError("[WinError 4551] a policy blocked this file"))
        assert reason is not None
        self.assertIn("4551", reason)

    def test_available_when_nothing_raises(self) -> None:
        """The control. Without it every test above passes for a probe that always refuses."""
        self.assertIsNone(self._refusal_when(None))

    def test_the_boolean_probe_still_answers_and_never_raises(self) -> None:
        """`decomposition_available()` is asked once per scene, so it must stay a yes/no."""
        from datagen.render.convex_decomposition import decomposition_available

        self.assertIs(True, decomposition_available())
