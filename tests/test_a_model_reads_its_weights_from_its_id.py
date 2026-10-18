"""A model is named by its id alone and read from one weights folder, set once (the owner, 2026-10-17).

``src/models/weights.py`` is the one resolver every model asks: RT-DETR, SAM2, the VLM, Whisper, Silero and the two
MediaPipe bundles. A Hugging Face id is read from ``<root>/<kind>/<org>--<name>``, the folder
``scripts/model_weights/fetch.py`` writes, else from the Hugging Face cache where an earlier download put it, and
nothing is downloaded at run time unless a caller asks for it (``local=False``). A path names a folder of its own and is
read as it is. A checkpoint that is nowhere on the machine is refused naming the key, the folder looked in and the
fetch that fills it. Before this, every block carried its own ``model_path`` and ``local``, and the paths were written
eleven times over.

Pure Python: no weights, no torch, and the Hugging Face cache of the machine running the tests is held out wherever the
answer would depend on it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.models import weights
from src.models.weights import (
    BUNDLES,
    CATALOGUE,
    PACKAGED_FILES,
    fetch_hint,
    fixed_file,
    locate_weights,
    missing_sentence,
    model_folder,
    names_a_folder,
    resolve_weights,
    weights_note,
    weights_present,
)


class _NoCache(unittest.TestCase):
    """A weights folder of the test's own, and no Hugging Face cache unless a test says it holds the id."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        cache = mock.patch.object(weights, "_in_the_cache", return_value=False)
        self.cache = cache.start()
        self.addCleanup(cache.stop)


class AnIdOrAFolderTests(unittest.TestCase):

    def test_a_path_names_a_folder_and_an_org_name_never_does(self) -> None:
        for path in ("D:/models/chess", "/opt/models/chess", "~/chess", "./chess", "../chess", ".\\chess",
                     "models\\chess", ".", ".."):
            with self.subTest(path=path):
                self.assertTrue(names_a_folder(path))
        self.assertTrue(names_a_folder(Path("chess")), "a PathLike always is one")
        for model_id in ("PekingU/rtdetr_r50vd", "facebook/sam2-hiera-large", "gpt2"):
            with self.subTest(model_id=model_id):
                self.assertFalse(names_a_folder(model_id))

    def test_an_id_is_read_from_its_kinds_folder_under_the_root(self) -> None:
        self.assertEqual(Path("R/detection/PekingU--rtdetr_r50vd"),
                         model_folder("PekingU/rtdetr_r50vd", kind="detection", root="R"))
        self.assertEqual(Path("R/vlm/Qwen--Qwen3-VL-4B-Instruct"),
                         model_folder("Qwen/Qwen3-VL-4B-Instruct", kind="vlm", root="R"))
        self.assertEqual(Path("D:/models/chess"), model_folder("D:/models/chess", kind="detection", root="R"))

    def test_the_default_root_is_the_repositorys(self) -> None:
        from src.config.loader import load_weights_root
        from src.utility.paths import project_root, weights_root

        self.assertEqual(project_root() / "assets" / "models" / "hf", weights_root())
        self.assertEqual(weights_root() / "segmentation" / "facebook--sam2-hiera-large",
                         model_folder("facebook/sam2-hiera-large", kind="segmentation"))
        self.assertEqual(weights_root().as_posix(), load_weights_root(profile=None))

    def test_a_name_that_is_neither_is_refused(self) -> None:
        for name in ("not an id", "a/b/c", "org/", ""):
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                model_folder(name, kind="detection", root="R")


class WhereTheWeightsAreReadTests(_NoCache):

    def test_the_fetched_folder_is_read_first_and_on_this_machine(self) -> None:
        folder = self.root / "segmentation" / "facebook--sam2-hiera-large"
        folder.mkdir(parents=True)
        found = locate_weights("facebook/sam2-hiera-large", kind="segmentation", root=self.root)
        self.assertEqual((folder.as_posix(), True, True, True),
                         (found.source, found.local, found.present, found.from_the_folder))
        self.assertEqual(folder.as_posix(), found.describe())
        self.cache.assert_not_called()

    def test_else_the_hugging_face_cache_where_an_earlier_download_put_it(self) -> None:
        self.cache.return_value = True
        found = locate_weights("facebook/sam2-hiera-large", kind="segmentation", root=self.root)
        self.assertEqual(("facebook/sam2-hiera-large", True, True, False),
                         (found.source, found.local, found.present, found.from_the_folder))
        self.assertEqual("facebook/sam2-hiera-large (from the Hugging Face cache)", found.describe())

    def test_else_nowhere_and_nothing_is_downloaded_unless_a_caller_asks(self) -> None:
        here = locate_weights("facebook/sam2-hiera-large", kind="segmentation", root=self.root)
        self.assertEqual((False, True), (here.present, here.local))
        self.assertEqual((self.root / "segmentation" / "facebook--sam2-hiera-large").as_posix(), here.source)
        asked = locate_weights("facebook/sam2-hiera-large", kind="segmentation", root=self.root, local=False)
        self.assertEqual(("facebook/sam2-hiera-large", False, False), (asked.source, asked.local, asked.present))
        self.assertEqual("facebook/sam2-hiera-large (from the Hugging Face Hub)", asked.describe())

    def test_a_folder_is_read_as_it_is_and_never_from_the_cache(self) -> None:
        folder = self.root / "chess"
        found = locate_weights(folder.as_posix(), kind="detection", root="R")
        self.assertEqual((folder.as_posix(), False), (found.source, found.present))
        folder.mkdir()
        self.assertTrue(locate_weights(folder.as_posix(), kind="detection", root="R").present)
        self.cache.assert_not_called()


class WhatIsRefusedTests(_NoCache):

    def test_weights_that_are_nowhere_are_refused_naming_the_key_the_folder_and_the_fetch(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            resolve_weights("PekingU/rtdetr_r50vd", kind="detection", root=self.root, key="models.rtdetr.model_id")
        message = str(caught.exception)
        for part in ("models.rtdetr.model_id is 'PekingU/rtdetr_r50vd'",
                     f"no folder at {(self.root / 'detection' / 'PekingU--rtdetr_r50vd').as_posix()}",
                     "nothing in the Hugging Face cache", "Nothing is downloaded at run time",
                     "`python scripts/model_weights/fetch.py rtdetr` writes them"):
            self.assertIn(part, message)

    def test_a_caller_that_asks_for_a_download_is_handed_the_id(self) -> None:
        found = resolve_weights("PekingU/rtdetr_r50vd", kind="detection", root=self.root, local=False)
        self.assertEqual(("PekingU/rtdetr_r50vd", False), (found.source, found.local))

    def test_a_folder_that_is_not_there_says_what_writes_one(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            resolve_weights((self.root / "chess").as_posix(), kind="detection", key="models.rtdetr.model_id")
        message = str(caught.exception)
        self.assertNotIn("Hugging Face cache", message, "a folder is never looked for in the cache")
        self.assertIn("save_pretrained", message)
        self.assertIn("DetectorTraining out_dir", message)

    def test_an_id_the_catalogue_lacks_says_how_a_checkpoint_comes_onto_the_machine(self) -> None:
        found = locate_weights("acme/no-such-model", kind="vlm", root=self.root)
        message = missing_sentence(found)
        self.assertIn("'acme/no-such-model' is named", message)
        self.assertIn("fetch.py --list", message)
        self.assertIn((self.root / "vlm" / "acme--no-such-model").as_posix(), message)

    def test_presence_is_asked_without_a_refusal_or_a_load(self) -> None:
        self.assertFalse(weights_present("PekingU/rtdetr_r50vd", kind="detection", root=self.root))
        self.assertFalse(weights_present("not an id", kind="detection", root=self.root), "a bad name is not here")
        (self.root / "detection" / "PekingU--rtdetr_r50vd").mkdir(parents=True)
        self.assertTrue(weights_present("PekingU/rtdetr_r50vd", kind="detection", root=self.root))


class TheFixedFilesTests(unittest.TestCase):

    def test_silero_and_the_bundles_have_their_places_under_the_root(self) -> None:
        self.assertEqual(Path("R/speech/silero-vad-6.2.1/silero_vad.jit"), fixed_file("silero-vad", root="R"))
        self.assertEqual(Path("R/mediapipe/hand_landmarker.task"), fixed_file("mediapipe-hand", root="R"))
        self.assertEqual(Path("R/mediapipe/gesture_recognizer.task"), fixed_file("mediapipe-gesture", root="R"))

    def test_a_key_that_is_no_fixed_file_is_refused_naming_the_ones_there_are(self) -> None:
        with self.assertRaises(KeyError) as caught:
            fixed_file("sam2", root="R")
        for key in ("silero-vad", "mediapipe-hand", "mediapipe-gesture"):
            self.assertIn(key, str(caught.exception))


class TheCatalogueTests(unittest.TestCase):

    def test_every_key_and_every_id_is_named_once(self) -> None:
        keys = [entry.key for entry in (*CATALOGUE, *PACKAGED_FILES, *BUNDLES)]
        self.assertEqual(len(keys), len(set(keys)))
        ids = [entry.repo_id for entry in CATALOGUE]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_entry_lands_in_a_kind_the_resolver_reads(self) -> None:
        for entry in CATALOGUE:
            with self.subTest(key=entry.key):
                self.assertIn(entry.category, ("detection", "segmentation", "speech", "vlm"))
                self.assertEqual(entry.local_dir(Path("R")),
                                 model_folder(entry.repo_id, kind=entry.category, root="R"))  # type: ignore[arg-type]

    def test_every_model_a_shipped_tree_names_is_in_it(self) -> None:
        from src.config import load_config

        models = load_config(profile=None).models
        named = {models.segmenter.model_id, models.stt.model_id, models.pipeline.zero_shot.vlm.model_id}
        if models.rtdetr is not None:
            named.add(models.rtdetr.model_id)
        self.assertLessEqual(named, {entry.repo_id for entry in CATALOGUE})

    def test_the_fetch_hint_names_the_key_that_fetches_an_id(self) -> None:
        for entry in CATALOGUE:
            with self.subTest(key=entry.key):
                self.assertEqual(f"`python scripts/model_weights/fetch.py {entry.key}` writes them",
                                 fetch_hint(entry.repo_id))

    def test_the_fetch_script_fetches_from_this_catalogue(self) -> None:
        from scripts.model_weights import fetch

        self.assertIs(CATALOGUE, fetch.CATALOGUE)
        self.assertIs(PACKAGED_FILES, fetch.PACKAGED_FILES)


class HowAFolderWasReadTests(unittest.TestCase):
    """A refusal under a relative weights folder says how the folder was read, as a relative model path did."""

    def test_a_path_under_a_folder_a_tree_resolved_carries_the_folders_note(self) -> None:
        import shutil

        from src.config.loader import load_weights_root

        repo_config = Path(__file__).resolve().parents[1] / "config"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tree = Path(tmp) / "cell"
            shutil.copytree(repo_config, tree)
            line = 'weights_root: "${WILLY_PROJECT_ROOT}/assets/models/hf"'
            file = tree / "models" / "weights.yaml"
            text = file.read_text(encoding="utf-8").replace("\r\n", "\n")
            self.assertEqual(1, text.count(line), "the shipped weights.yaml stopped naming the folder")
            file.write_text(text.replace(line, "weights_root: weights"), encoding="utf-8")
            root = load_weights_root(tree, profile=None)
            self.assertEqual((tree / "weights").resolve(), Path(root).resolve())
            note = weights_note(f"{root}/vlm/Qwen--Qwen3-VL-4B-Instruct")
        self.assertIn(f"The weights folder is {root}.", note)
        self.assertIn("The config gives it as weights, and a relative path in a config is read against the config "
                      "folder", note)

    def test_a_path_no_tree_read_says_nothing(self) -> None:
        self.assertEqual("", weights_note(""))
        self.assertEqual("", weights_note(Path(tempfile.gettempdir()) / "never-read-by-a-tree" / "x.bin"))


if __name__ == "__main__":
    unittest.main()
