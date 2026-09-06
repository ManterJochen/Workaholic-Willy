"""A missing local model directory says it is a missing local model directory.

MEASURED 2026-08-19, building a real (non-rehearsal) cell through the operator console on a fresh
checkout, where the shipped ``models.objectdetector.model_path`` does not exist:

    OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
    'backend/src/models/detection/model'. Use `repo_type` argument if needed.

Nothing in that sentence is the problem. ``from_pretrained`` does not treat a missing directory as a
missing directory -- it falls through to interpreting the string as a Hub repo id and rejects it as
one, so the reader is sent hunting for a Hub misconfiguration that does not exist. On a September
bring-up that is an afternoon.

The check is deliberately narrow: only ``local=True``, only a path that is not a directory. Remote
mode is untouched, and a path that IS there is untouched, so nothing that works today changes.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


def _config(*, local: bool, model_path: str, model_id: str = "") -> MagicMock:
    cfg = MagicMock()
    cfg.local = local
    cfg.model_path = model_path
    cfg.model_id = model_id
    cfg.threshold = 0.4
    cfg.optim = MagicMock(
        torch_dtype="auto", attn_implementation="eager", channels_last=True,
        compile=False, compile_mode="reduce-overhead",
    )
    return cfg


class MissingLocalPathTests(unittest.TestCase):
    def test_it_names_the_directory_and_both_ways_out(self) -> None:
        from src.models.detection.zero_shot.detector import (
            GroundingDinoObjectDetector,
        )

        with self.assertRaises(FileNotFoundError) as caught:
            GroundingDinoObjectDetector(_config(local=True, model_path="no/such/directory"))
        message = str(caught.exception)
        self.assertIn("no/such/directory", message)
        # The resolved absolute path: "it is not there" is only actionable if you know where it looked.
        self.assertIn(str(Path("no/such/directory").resolve()), message)
        # Both fixes, because either is legitimate and the reader should not have to pick blind.
        self.assertIn("local: false", message)
        self.assertIn("model_id", message)

    def test_it_is_raised_before_anything_is_loaded(self) -> None:
        # The whole point: fail with a readable message instead of inside the transformers stack.
        from src.models.detection.zero_shot import detector as module

        with patch.object(module, "AutoProcessor") as processor:
            with self.assertRaises(FileNotFoundError):
                module.GroundingDinoObjectDetector(_config(local=True, model_path="nope"))
            processor.from_pretrained.assert_not_called()

    def test_remote_mode_is_untouched(self) -> None:
        # `local=False` never looks at the filesystem; a nonexistent model_path is irrelevant there.
        from src.models.detection.zero_shot import detector as module

        with patch.object(module, "AutoProcessor"), patch.object(
            module, "AutoModelForZeroShotObjectDetection"
        ):
            module.GroundingDinoObjectDetector(
                _config(local=False, model_path="no/such/directory", model_id="org/model")
            )  # must not raise

    def test_a_directory_that_exists_is_untouched(self) -> None:
        from src.models.detection.zero_shot import detector as module

        with TemporaryDirectory() as tmp, patch.object(module, "AutoProcessor"), patch.object(
            module, "AutoModelForZeroShotObjectDetection"
        ):
            module.GroundingDinoObjectDetector(_config(local=True, model_path=tmp))  # must not raise

    def test_an_empty_source_still_reports_the_older_error(self) -> None:
        from src.models.detection.zero_shot.detector import (
            GroundingDinoObjectDetector,
        )

        with self.assertRaises(ValueError):
            GroundingDinoObjectDetector(_config(local=True, model_path=""))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
