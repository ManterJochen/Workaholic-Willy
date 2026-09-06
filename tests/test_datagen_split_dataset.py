"""`python -m datagen split-dataset` — dividing one rendered dataset so the cloud step can use every core.

⚠ WHY THIS EXISTS. `build_cloud_corpus` is single-process per dataset and takes 36 minutes per 1,250
scenes, and the flag that looks like it divides the work does not: `scenes` is an even STRIDE, so two
processes given half each walk the SAME half. The split has to happen one level up, by building
datasets whose scene lists are disjoint.

⚠⚠ AND A MIS-SPLIT IS SILENT. Nothing downstream notices scenes that were dropped or extracted twice;
it shows up, if at all, as a training run on the wrong data hours later. So the two properties that
matter are asserted at the source and pinned here: the parts are DISJOINT and they are COMPLETE.

The third thing pinned is the cleanup refusal. Views are removed by name pattern (`<name>_p*`), and
`_p0` is a plausible suffix for a hand-made dataset, so the marker file is the only thing standing
between a cleanup and a deleted render.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen import __main__ as cli
from datagen.corpus.split import MARKER, clean_split, split_dataset


def _dataset(root: Path, name: str, scenes: int, *, unfinished: int = 0) -> Path:
    """A dataset shaped like a rendered one: `scenes/<id>/scene.json` plus the sidecars."""
    dataset = root / name
    (dataset / "scenes").mkdir(parents=True)
    for index in range(scenes):
        scene = dataset / "scenes" / f"packed_{index:06d}"
        scene.mkdir()
        (scene / "scene.json").write_text("{}", encoding="utf-8")
        (scene / "depth.png").write_bytes(b"depth")
    for index in range(unfinished):
        # A directory with no manifest: `datagen build` writes the directory first and the manifest
        # last, so this is what an interrupted render leaves behind.
        (dataset / "scenes" / f"bin_{index:06d}").mkdir()
    (dataset / "grasps.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset / "provenance.json").write_text("{}", encoding="utf-8")
    return dataset


class SplitTests(unittest.TestCase):

    def test_parts_are_disjoint_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 10)
            report = split_dataset(dataset, 3)

            self.assertEqual(report["scenes"], 10)
            seen: list[str] = []
            for view in report["views"]:
                names = sorted(p.name for p in (Path(view["path"]) / "scenes").iterdir())
                self.assertEqual(len(names), view["scenes"])
                seen.extend(names)
            self.assertEqual(len(seen), 10, "a scene was handed to two parts, or to none")
            self.assertEqual(sorted(seen),
                             sorted(p.name for p in (dataset / "scenes").iterdir()))

    def test_an_unfinished_scene_is_not_handed_to_a_part(self) -> None:
        """A directory with no `scene.json` is a half-rendered scene the extractor would refuse."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 6, unfinished=2)
            report = split_dataset(dataset, 2)
            self.assertEqual(report["scenes"], 6)
            self.assertEqual(sum(v["scenes"] for v in report["views"]), 6)

    def test_scenes_are_linked_not_copied(self) -> None:
        """The point of the link: a part view must not double the corpus on disk.

        Checked by WRITING through the source after the split. A copy would not see it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 4)
            report = split_dataset(dataset, 2)
            (dataset / "scenes" / "packed_000000" / "late.txt").write_text("x", encoding="utf-8")

            through = [Path(v["path"]) / "scenes" / "packed_000000" / "late.txt"
                       for v in report["views"]]
            self.assertTrue(any(p.is_file() for p in through),
                            "no view sees a file written into the source afterwards")

    def test_sidecars_reach_every_part(self) -> None:
        """The grasp table is joined by `(file, row_index)`, so a part needs the WHOLE of it."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 4)
            for view in split_dataset(dataset, 2)["views"]:
                self.assertTrue((Path(view["path"]) / "grasps.jsonl").is_file())
                self.assertTrue((Path(view["path"]) / "provenance.json").is_file())

    def test_a_second_split_replaces_the_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 8)
            split_dataset(dataset, 2)
            report = split_dataset(dataset, 4)
            self.assertEqual(len(report["views"]), 4)
            self.assertEqual(sum(v["scenes"] for v in report["views"]), 8)

    def test_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 3)
            with self.assertRaises(ValueError):
                split_dataset(dataset, 1)
            with self.assertRaises(ValueError):
                split_dataset(dataset, 4)          # more parts than finished scenes
            with self.assertRaises(FileNotFoundError):
                split_dataset(Path(tmp) / "absent", 2)


class CleanupTests(unittest.TestCase):

    def test_clean_removes_views_and_keeps_the_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), "v_s0", 6)
            split_dataset(dataset, 3)
            removed = clean_split(dataset)

            self.assertEqual(len(removed), 3)
            self.assertEqual(len(list((dataset / "scenes").iterdir())), 6)
            self.assertTrue((dataset / "scenes" / "packed_000000" / "depth.png").is_file(),
                            "the cleanup followed a link and deleted real scene data")

    def test_clean_refuses_a_directory_without_the_marker(self) -> None:
        """`_p0` is a plausible name for a real dataset, so the NAME may not be what decides."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, "v_s0", 4)
            impostor = _dataset(root, "v_s0_p9", 2)

            self.assertEqual(clean_split(dataset), [])
            self.assertTrue((impostor / "scenes" / "packed_000000" / "scene.json").is_file())

    def test_clean_refuses_a_view_of_another_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, "v_s0", 4)
            other = _dataset(root, "v_s1", 4)
            split_dataset(other, 2)
            # Rename one of v_s1's views so it LOOKS like one of v_s0's.
            (root / "v_s1_p0").rename(root / "v_s0_p7")

            self.assertEqual(clean_split(dataset), [])
            self.assertTrue((root / "v_s0_p7" / MARKER).is_file())
            self.assertEqual(
                json.loads((root / "v_s0_p7" / MARKER).read_text(encoding="utf-8"))["source"],
                str(other))


class RegistrationTests(unittest.TestCase):

    def test_the_command_routes_to_its_own_handler(self) -> None:
        """Registration AND routing. A command in the choices list that no branch reads is inert."""
        with mock.patch.object(cli, "_cmd_split_dataset", return_value=0) as handler:
            self.assertEqual(cli.main(["split-dataset", "--name", "v_s0", "--parts", "3"]), 0)
        self.assertEqual(handler.call_count, 1)
        args = handler.call_args.args[0]
        self.assertEqual((args.name, args.parts, args.clean), ("v_s0", 3, False))

    def test_a_missing_dataset_refuses_with_a_usage_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(command="split-dataset", name="absent", out=tmp,
                                      parts=4, clean=False)
            self.assertEqual(cli._cmd_split_dataset(args, None), 2)


if __name__ == "__main__":
    unittest.main()
