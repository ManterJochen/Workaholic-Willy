"""A missing local model directory says it is a missing local model directory.

MEASURED 2026-08-19, building a real (non-rehearsal) cell through the operator console on a fresh
checkout, where the shipped detector's directory did not exist:

    OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
    'backend/src/models/detection/model'. Use `repo_type` argument if needed.

Nothing in that sentence is the problem. ``from_pretrained`` does not treat a missing directory as a
missing directory -- it falls through to interpreting the string as a Hub repo id and rejects it as
one, so the reader is sent hunting for a Hub misconfiguration that does not exist. On a September
bring-up that is an afternoon.

Since 2026-10-17 every model is read from its ``model_id`` under ``models.weights_root``, through the one
resolver (``src/models/weights.py``), so the check is the resolver's: weights that are not on this machine
are refused naming the key, the folder looked in and the fetch, before anything loads. Here on RT-DETR, the
closed-set detector; the VLM, SAM2 and Whisper refuse through the same function. A folder that IS there is
untouched, and a caller that asks for a download (``local=False``) is handed the id.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from src.config.schema.models.models_schema import ObjectDetectorConfig

_ID = "PekingU/rtdetr_r50vd"


def _detector(root: str, *, model_id: str = _ID, local: bool = True) -> object:
    from src.models.detection.closed_set.detector import RtDetrObjectDetector

    return RtDetrObjectDetector(ObjectDetectorConfig(model_id=model_id, threshold=0.4), device="cpu",
                                weights_root=root, local=local)


class MissingLocalPathTests(unittest.TestCase):
    def setUp(self) -> None:
        # The Hugging Face cache of the machine running the tests is no part of this question.
        cache = patch("src.models.weights._in_the_cache", return_value=False)
        cache.start()
        self.addCleanup(cache.stop)

    def test_it_names_the_key_the_folder_and_the_fetch(self) -> None:
        with TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError) as caught:
            _detector(tmp)
        message = str(caught.exception)
        self.assertIn(f"models.rtdetr.model_id is '{_ID}'", message)
        # The folder it looked in: "it is not there" is only actionable if you know where it looked.
        self.assertIn(f"{Path(tmp).as_posix()}/detection/PekingU--rtdetr_r50vd", message)
        self.assertIn("python scripts/model_weights/fetch.py rtdetr", message)
        self.assertIn("Nothing is downloaded at run time", message)

    def test_it_is_raised_before_anything_is_loaded(self) -> None:
        # The whole point: fail with a readable message instead of inside the transformers stack.
        from src.models.detection.closed_set import detector as module

        with TemporaryDirectory() as tmp, patch.object(module, "AutoImageProcessor") as processor:
            with self.assertRaises(FileNotFoundError):
                _detector(tmp)
            processor.from_pretrained.assert_not_called()

    def test_a_download_asked_for_is_handed_the_id(self) -> None:
        # `local=False` (ObjectDetector.from_weights, a training run's base model) lets transformers fetch it.
        from src.models.detection.closed_set import detector as module

        with TemporaryDirectory() as tmp, patch.object(module, "AutoImageProcessor") as processor, \
             patch.object(module, "AutoModelForObjectDetection") as model, \
             patch.object(module, "finalize_model", side_effect=lambda m, *a, **k: m):
            model.from_pretrained.return_value = MagicMock(config=MagicMock(id2label={0: "person"}))
            _detector(tmp, local=False)  # must not raise
        self.assertEqual(_ID, processor.from_pretrained.call_args.args[0])
        self.assertNotIn("local_files_only", processor.from_pretrained.call_args.kwargs)

    def test_a_folder_that_exists_is_read_as_it_is(self) -> None:
        from src.models.detection.closed_set import detector as module

        with TemporaryDirectory() as tmp, patch.object(module, "AutoImageProcessor") as processor, \
             patch.object(module, "AutoModelForObjectDetection") as model, \
             patch.object(module, "finalize_model", side_effect=lambda m, *a, **k: m):
            model.from_pretrained.return_value = MagicMock(config=MagicMock(id2label={0: "person"}))
            folder = Path(tmp) / "detection" / "PekingU--rtdetr_r50vd"
            folder.mkdir(parents=True)
            _detector(tmp)  # must not raise
        self.assertEqual(folder.as_posix(), processor.from_pretrained.call_args.args[0])
        self.assertTrue(processor.from_pretrained.call_args.kwargs["local_files_only"])

    def test_an_id_that_is_no_id_is_refused(self) -> None:
        with TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError) as caught:
            _detector(tmp, model_id="not an id")
        self.assertIn("neither a folder nor a Hugging Face id", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
