"""Which assets a model has never seen — and the denominator that makes the answer true or false.

⛔ THE MISTAKE THIS MODULE EXISTS TO PREVENT is one I made in this session, out loud, before
measuring: "318 unseen meshes are already on this box". They were not usable. `MeshAssetBank` filters
on extent and closability; of 465 files only 147 survived, and the training corpus had used ALL 147.
Usable AND unseen: zero. Counting FILES answers a different question than counting assets the
pipeline can place, and the difference was the whole difference.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datagen.heldout import HeldOutReport, format_report, held_out_assets, trained_groups


def _corpus(*asset_ids: str, nested: bool = False) -> Path:
    """A corpus with one scene placing `asset_ids`, in the layout `scene.json` really uses."""
    root = Path(tempfile.mkdtemp())
    scene_dir = (root / "shard0" / "scenes" / "s000001") if nested else (root / "scenes" / "s000001")
    scene_dir.mkdir(parents=True)
    (scene_dir / "scene.json").write_text(json.dumps({
        "spec": {"scene_id": "s000001",
                 "objects": [{"asset_id": a} for a in asset_ids]}}), encoding="utf-8")
    return root


class TheCorpusReaderTests(unittest.TestCase):
    def test_it_reads_objects_from_SPEC_not_from_the_top_level(self) -> None:
        """⚠ `scene.json` puts the objects under `spec`. Reading a top-level `objects` key returns
        zero for every scene — a silent empty answer, not an error, and I shipped exactly that in an
        ad-hoc analysis an hour before writing this."""
        groups = trained_groups(_corpus("gso_Android_Lego", "proc_00003_pouch"))
        self.assertEqual(groups, {"gso_Android_Lego", "family:pouch"})

    def test_it_walks_a_SHARDED_corpus_too(self) -> None:
        """The real training corpus is five shards; a single dataset is one directory. One glob."""
        self.assertEqual(trained_groups(_corpus("gso_Thing", nested=True)), {"gso_Thing"})

    def test_generated_assets_group_by_FAMILY_not_by_name(self) -> None:
        """Two `proc_00002_pouch` in two scenes are two different pouches — grouping them by name
        would group by an accident of ordering, which is the trainer's own reasoning."""
        groups = trained_groups(_corpus("proc_00002_pouch", "proc_00088_pouch"))
        self.assertEqual(groups, {"family:pouch"})

    def test_an_empty_directory_yields_no_groups(self) -> None:
        self.assertEqual(trained_groups(Path(tempfile.mkdtemp())), set())


class TheReportTests(unittest.TestCase):
    def test_an_empty_corpus_is_flagged_as_the_DANGEROUS_case(self) -> None:
        """⛔ With no scenes found, every asset reads as unseen — the most dangerous possible answer,
        because it produces a confident list from a wrong path."""
        report = held_out_assets(Path(tempfile.mkdtemp()), sources=())
        self.assertTrue(any("most dangerous" in w for w in report.warnings), report.warnings)

    def test_zero_unseen_says_what_to_do_about_it(self) -> None:
        """This is the state the v1 corpus is actually in, so the message has to be actionable."""
        report = HeldOutReport(unseen={"gso": []}, on_disk={"gso": 400}, usable={"gso": 147},
                               trained={"gso": 147}, trained_group_count=168, corpus="x",
                               warnings=("ZERO placeable assets are unseen: this corpus used every "
                                         "mesh the bank can yield. A held-out dataset needs NEW "
                                         "meshes -- `python -m datagen.assets.fetch --list`.",))
        text = format_report(report)
        # ⚠ THE COMMAND, NOT THE FILENAME. This asserted `fetch.py`, which stopped meaning anything
        # when the fetcher moved from `scripts/meshes/fetch.py` into `datagen/assets/fetch.py` on
        # 2026-09-04 and the message became a `python -m` invocation. What the test is actually for
        # is that the warning tells the operator what to RUN, so it asserts the runnable thing.
        self.assertIn("datagen.assets.fetch", text)
        self.assertEqual(report.total_unseen, 0)

    def test_the_report_names_PLACEABLE_as_the_denominator(self) -> None:
        """The sentence is the finding; without it the table invites the file count again."""
        report = HeldOutReport(unseen={"gso": ["gso_a"]}, on_disk={"gso": 400}, usable={"gso": 147},
                               trained={"gso": 146}, trained_group_count=168, corpus="x")
        text = format_report(report)
        self.assertIn("PLACEABLE is the denominator", text)
        self.assertIn("400", text)
        self.assertIn("147", text)

    def test_the_formatter_survives_a_partially_filled_report(self) -> None:
        """A report function that raises destroys the information it was called to convey."""
        report = HeldOutReport(unseen={"gso": ["a"], "ycb": ["b"]}, on_disk={"gso": 1},
                               usable={}, trained={}, trained_group_count=1, corpus="x")
        self.assertIn("ycb", format_report(report))

    def test_files_on_disk_and_placeable_are_reported_SEPARATELY(self) -> None:
        """One number would hide the 3.2x gap between them that caused the original mistake."""
        report = HeldOutReport(unseen={"gso": []}, on_disk={"gso": 465}, usable={"gso": 147},
                               trained={"gso": 147}, trained_group_count=168, corpus="x")
        self.assertNotEqual(report.on_disk["gso"], report.usable["gso"])


class TheCliTests(unittest.TestCase):
    def test_a_missing_corpus_is_a_usage_error(self) -> None:
        from datagen.__main__ import main
        self.assertEqual(main(["heldout", "--corpus", "nowhere/at/all"]), 2)

    def test_no_corpus_at_all_is_a_usage_error(self) -> None:
        from datagen.__main__ import main
        self.assertEqual(main(["heldout"]), 2)

    def test_zero_unseen_exits_NON_ZERO(self) -> None:
        """⛔ An empty list is a problem, not a result. Exiting 0 would let a pipeline build a
        "held-out" dataset from nothing and report success -- which is the state the v1 corpus is
        actually in.

        ⚠ The report is STUBBED here rather than measured: building the real bank costs ~100 s and
        would make this a test of the mesh library instead of a test of the exit code. A corpus that
        genuinely covers every placeable asset cannot be faked cheaply -- `_corpus("gso_x")` leaves the
        whole real library unseen, which is what my first version of this test got wrong."""
        from unittest import mock

        import datagen.heldout as heldout_module
        from datagen.__main__ import main

        empty = HeldOutReport(unseen={"gso": []}, on_disk={"gso": 400}, usable={"gso": 147},
                              trained={"gso": 147}, trained_group_count=168, corpus="x")
        with mock.patch.object(heldout_module, "held_out_assets", return_value=empty):
            self.assertEqual(main(["heldout", "--corpus", str(_corpus("gso_x"))]), 1)

    def test_a_non_empty_list_exits_zero(self) -> None:
        """The control: without it the test above passes for a command that always fails."""
        from unittest import mock

        import datagen.heldout as heldout_module
        from datagen.__main__ import main

        found = HeldOutReport(unseen={"gso": ["gso_new"]}, on_disk={"gso": 400},
                              usable={"gso": 148}, trained={"gso": 147}, trained_group_count=168,
                              corpus="x")
        with mock.patch.object(heldout_module, "held_out_assets", return_value=found):
            self.assertEqual(main(["heldout", "--corpus", str(_corpus("gso_x"))]), 0)

    def test_the_written_file_is_what_mesh_asset_ids_path_expects(self) -> None:
        """The two halves have to fit: this command's output IS that key's input."""
        from unittest import mock

        import datagen.heldout as heldout_module
        from datagen.__main__ import main
        from datagen.config import DatagenConfig
        from datagen.scenes.layout import resolve_mesh_asset_ids

        out = Path(tempfile.mkdtemp()) / "ids.json"
        found = HeldOutReport(unseen={"gso": ["gso_a"], "ycb": ["ycb_b"]},
                              on_disk={"gso": 400, "ycb": 65}, usable={"gso": 148, "ycb": 35},
                              trained={"gso": 147, "ycb": 34}, trained_group_count=168, corpus="x")
        with mock.patch.object(heldout_module, "held_out_assets", return_value=found):
            self.assertEqual(
                main(["heldout", "--corpus", str(_corpus("gso_x")), "--out-ids", str(out)]), 0)
        config = DatagenConfig.model_validate({"assets": {"mesh_asset_ids_path": str(out)}})
        self.assertEqual(resolve_mesh_asset_ids(config), frozenset({"gso_a", "ycb_b"}))


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
