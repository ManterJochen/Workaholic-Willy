"""Every model path the shipped config names must be one the fetch script can produce.

Measured on 2026-09-08, before this existed, three of them could not be. The base profile set
``local: true`` on `objectdetector`, `segmenter` and `stt`, pointing at
``src/models/{detection,segmentation,speech}/model``, and none of those directories exists in the
repository or is created by anything. The detector's ``model_id`` was the empty string as well, so it
could load neither locally nor from the hub. A cell built from the shipped tree could not assemble a
perception stack at all.

The check asks the fetch script, it does not keep a copy of the answer. A list of expected paths
written here would be a second declaration of the layout, maintained by whoever remembers this file
exists. `Weights.local_dir` computes where a model lands, so the config is compared against the thing
that puts it there.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.model_weights.fetch import CATALOGUE  # noqa: E402
from src.utility.paths import weights_root  # noqa: E402


def _blocks() -> list[tuple[str, str | None, str | None, bool | None]]:
    """Every config block that names a model, as (name, model_id, model_path, local)."""
    from src.config.loader import load_config

    models = load_config().models
    out = []
    for name in ("objectdetector", "segmenter", "stt", "rtdetr", "oneformer"):
        block = getattr(models, name, None)
        if block is not None:
            out.append((name, getattr(block, "model_id", None),
                        getattr(block, "model_path", None), getattr(block, "local", None)))
    vlm = models.pipeline.zero_shot.vlm if models.pipeline is not None else None
    if vlm is not None:
        out.append(("vlm", vlm.model_id, vlm.model_path, vlm.local))
    return out


class EveryNamedModelIsInTheCatalogueTests(unittest.TestCase):
    def test_every_model_id_the_config_names_can_be_fetched(self) -> None:
        """A `local: true` path that no command produces is worse than `local: false`, which at
        least downloads. This is the half that makes `local: true` honest."""
        catalogue = {w.repo_id for w in CATALOGUE}
        missing = sorted(
            f"{name}: {model_id}" for name, model_id, _path, _local in _blocks()
            if model_id and model_id not in catalogue
        )
        self.assertEqual(
            missing, [],
            "the config names a model the fetch script cannot download, so its local path can "
            "never come into existence. Add it to CATALOGUE in scripts/model_weights/fetch.py.",
        )

    def test_no_block_names_a_model_by_nothing(self) -> None:
        """`objectdetector` shipped with `model_id: ""` and a path to a directory that does not
        exist, which is neither a local model nor a remote one."""
        for name, model_id, path, _local in _blocks():
            with self.subTest(block=name):
                self.assertTrue(model_id or path, f"{name} names neither a model id nor a path")


class EveryLocalPathIsWhereTheFetchPutsItTests(unittest.TestCase):
    def test_a_local_block_points_at_the_directory_the_fetch_script_writes(self) -> None:
        root = weights_root()
        by_id = {w.repo_id: w for w in CATALOGUE}
        checked = 0
        for name, model_id, path, local in _blocks():
            if not local or not model_id or model_id not in by_id:
                continue
            with self.subTest(block=name):
                expected = by_id[model_id].local_dir(root).relative_to(_REPO).as_posix()
                self.assertEqual(
                    Path(path).as_posix(), expected,
                    f"{name} is local: true at a path the fetch script does not produce",
                )
                checked += 1
        self.assertGreaterEqual(checked, 3, "a test over no local block passes loudest")

    def test_every_local_path_is_inside_the_weights_root(self) -> None:
        """The whole point of the location: deleting the repository takes the weights with it. A
        path outside it is a download that survives, which is what this move was for."""
        root = weights_root().relative_to(_REPO).as_posix()
        for name, _model_id, path, local in _blocks():
            if not local or not path:
                continue
            with self.subTest(block=name):
                self.assertTrue(
                    Path(path).as_posix().startswith(root),
                    f"{name} is local: true outside {root}, so a repo delete leaves it behind",
                )


class TheIgnoreListCannotEmptyAModelTests(unittest.TestCase):
    def test_the_one_repository_with_no_safetensors_has_no_ignore_list(self) -> None:
        """Measured against the hub: `shi-labs/oneformer_coco_swin_large` is 1.83 GB whole and
        0.00 GB of safetensors, so a blanket `*.bin` filter fetches it empty. The filter is per
        model for that reason, and this is the entry that proves it cannot be global."""
        oneformer = next(w for w in CATALOGUE if "oneformer" in w.repo_id)
        self.assertEqual(oneformer.ignore, ())

    def test_the_filtered_ones_still_keep_a_weight_format(self) -> None:
        """A control on the entry above: every other model is filtered, and none of the filters
        removes safetensors."""
        filtered = [w for w in CATALOGUE if w.ignore]
        self.assertGreaterEqual(len(filtered), 5)
        for w in filtered:
            with self.subTest(model=w.key):
                self.assertNotIn("*.safetensors", w.ignore)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
