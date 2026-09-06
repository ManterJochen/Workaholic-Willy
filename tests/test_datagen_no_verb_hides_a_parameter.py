"""No verb may quietly withhold a parameter its callee accepts.

⛔⛔ **THE QUIETER HALF OF THE WRAPPER PROBLEM.** A wrapper that declares a competing default becomes
loud the day the two drift. A wrapper that simply OMITS a parameter is never loud at all: nothing
raises, the caller silently gets the callee's default, and the artifact is not the one they asked
for. There is no error to read and no number that looks wrong.

Measured on 2026-09-04, the day the services were written:

    DatasetBuild.clouds        forwarded 4 of 9   -- mask_suffix, voxel_mm, with_environment,
                                                     environment_voxel_mm, environment_margin_mm
    GraspEvaluation.evaluate   forwarded 6 of 14  -- model, files, scene_step, with_jaw,
                                                     with_suction, suction_every, progress_every,
                                                     support_cfg

⭐ **NEITHER SET IS COSMETIC.** `--masks pred` is the corpus a real camera would produce and had a CLI
route with no code route; `voxel_mm` decides how much geometry survives at all; `with_jaw` and
`with_suction` decide which modality is graded.

⚠ **A PARAMETER THE NOUN ALREADY OWNS IS NOT HIDDEN.** `PhysicsSampling` carries `engine` and
`headless` as fields and forwards them, which is the point of a noun: it is asked once and remembered.
This test reads the dataclass fields and counts those as reachable, so it does not force every verb to
restate what its object already knows.
"""

from __future__ import annotations

import dataclasses
import inspect
import unittest
from typing import Any, Callable


def _reachable(noun: type, verb: Callable[..., Any]) -> set[str]:
    """What a caller can influence: the verb's own parameters plus the noun's fields."""
    fields = {f.name for f in dataclasses.fields(noun)} if dataclasses.is_dataclass(noun) else set()
    return set(inspect.signature(verb).parameters) | fields


def _accepted(callee: Callable[..., Any]) -> set[str]:
    return {name for name, p in inspect.signature(callee).parameters.items()
            if p.kind is not inspect.Parameter.VAR_KEYWORD}


#: ⚠ REACHED UNDER ANOTHER NAME, ON PURPOSE, AND LISTED RATHER THAN EXEMPTED. A noun should offer
#: the domain word, not the implementation word: `RankerCorpus` takes `mask_source` ("gt" / "pred")
#: and derives the suffix the walker actually wants. Writing the mapping here keeps that a decision
#: somebody made rather than a hole in the guard, and the test below asserts each substitute really
#: exists, so a rename cannot quietly turn one of these into a genuine omission.
_RENAMED: dict[str, dict[str, str]] = {
    "RankerCorpus.build": {"mask_suffix": "mask_source"},
}

#: Arguments the noun supplies from its own identity, so a verb never takes them.
_OWNED = {"root", "config", "out_dir", "self", "report", "dataset_dir", "table", "spec",
          "trials", "out_name", "configs", "random_state", "pattern", "name", "out_root",
          "physics_path", "corpus", "density"}


class NoVerbHidesAParameterTests(unittest.TestCase):

    def _check(self, label: str, noun: type, verb_name: str, callee: Callable[..., Any]) -> None:
        reachable = _reachable(noun, getattr(noun, verb_name))
        for hidden, substitute in _RENAMED.get(label, {}).items():
            self.assertIn(substitute, reachable,
                          f"{label} claims to reach {hidden} as {substitute}, and does not")
            reachable.add(hidden)
        missing = _accepted(callee) - reachable - _OWNED
        self.assertEqual(set(), missing,
                         f"{label} cannot reach {sorted(missing)}, which {callee.__name__} accepts; "
                         f"a caller gets the callee's default and no warning")

    def test_the_dataset_build_verbs(self) -> None:
        from datagen.api import DatasetBuild
        from datagen.build import build_dataset
        from datagen.corpus.clouds import build_cloud_corpus
        from datagen.grasps.labels import label_dataset

        self._check("DatasetBuild.render", DatasetBuild, "render", build_dataset)
        self._check("DatasetBuild.label", DatasetBuild, "label", label_dataset)
        self._check("DatasetBuild.clouds", DatasetBuild, "clouds", build_cloud_corpus)

    def test_the_evaluation_verbs(self) -> None:
        from datagen.eval.ladder import evaluate_dataset, run_gate
        from datagen.eval.service import GraspEvaluation

        self._check("GraspEvaluation.evaluate", GraspEvaluation, "evaluate", evaluate_dataset)
        self._check("GraspEvaluation.gate", GraspEvaluation, "gate", run_gate)

    def test_the_physics_verbs(self) -> None:
        from datagen.grasps.physics import run_physics_sample
        from datagen.grasps.service import PhysicsSampling

        self._check("PhysicsSampling.sample", PhysicsSampling, "sample", run_physics_sample)

    def test_the_corpus_verbs(self) -> None:
        from datagen.corpus.build import build_ranker_corpus
        from datagen.corpus.service import RankerCorpus

        self._check("RankerCorpus.build", RankerCorpus, "build", build_ranker_corpus)

    def test_the_rl_verbs(self) -> None:
        from datagen.rl.collect import collect_trials
        from datagen.rl.occupancy import measure_occupancy
        from datagen.rl.service import RecordCollection

        self._check("RecordCollection.collect", RecordCollection, "collect", collect_trials)
        self._check("RecordCollection.measure_occupancy", RecordCollection, "measure_occupancy",
                    measure_occupancy)

    def test_the_check_covers_more_than_one_verb(self) -> None:
        """The control on the checks above: an `_OWNED` set that swallowed everything would make all
        of them pass while checking nothing."""
        from datagen.corpus.clouds import build_cloud_corpus

        self.assertGreater(len(_accepted(build_cloud_corpus) - _OWNED), 5,
                           "_OWNED has grown until there is nothing left to check")

    def test_a_verb_that_DROPS_a_parameter_is_caught(self) -> None:
        """⚠ THE GUARD'S OWN CONTROL. Restoring the real defect proved this fires; keeping a
        synthetic one means it stays proved without a broken package on disk."""
        @dataclasses.dataclass
        class Noun:
            owned_field: int = 0

            def verb(self, *, forwarded: int = 0) -> None:
                """Takes one of the callee's two."""

        def callee(*, forwarded: int = 0, hidden: int = 0) -> None:
            """Accepts two."""

        with self.assertRaises(AssertionError) as caught:
            self._check("Noun.verb", Noun, "verb", callee)
        self.assertIn("hidden", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
