"""No module in this repo reads another product's corpus — and the guard that keeps it that way.

⚠ WHAT WAS REMOVED, AND WHY IT IS WORTH A GUARD. `backend/src/robot/grasping/training_data/` held a
reader, a full-scorer and an ingest adapter for a corpus produced by a SEPARATE product, plus a second
`GraspAttemptRecord` synthesiser. Measured before deleting:

* **Nothing in production called any of it.** Only two test files imported the package.
* **The record synthesiser was the loser of a duplicate pair.** `datagen/rl/collect.py` builds
  `GraspAttemptRecord`s directly on the live path and stamps MORE provenance than the deleted one
  did — `reward_model`, a full bucket-① `reward_interpretation`, and `dataset_origin`, which the RL
  trainers already read.
* **The readers read a format this repo does not produce.** Re-pointing them at our data would have
  built a SECOND reader beside `datagen/corpus/sources.py`, which already knows our three formats.
* **`datagen check-dataset` defaulted `--source` to the foreign format** — our own tool reaching for
  another product's data unless told otherwise. It defaults to `grasp_labels` now.
* **`willy_sim/gso_assets.py` defaulted to an absolute path into the other project's checkout.** The
  same 400 meshes have been imported here since, with their sha256 index and their licences; a spot
  check of 12 found them byte-identical.

⛔ WHAT WAS DELIBERATELY **KEPT**: every comment that records what that project's post-mortem TAUGHT
us. `deep/net.py`'s "classify, never regress" rationale is the single most load-bearing decision in
the learned generator and it is justified by exactly such a lesson; `grasping_schema.py` explains why
the learned ranker ships in shadow by citing a ranker fitted on that corpus that beat the calculator
by 17.7 pp and still lost to random. Deleting prose that says WHAT WE LEARNED would destroy the reason
for decisions we would otherwise re-litigate. An artefact is an import, an identifier, a file name or
a path. A lesson is neither.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: An ARTEFACT: an import of it, an identifier named after it, or a path into its checkout. Prose that
#: merely cites what it taught is not matched, and must not be.
_ARTEFACT = re.compile(
    r"^\s*(?:from|import)\s+\S*aletheia"      # importing it
    r"|aletheia_\w+"                          # a module or symbol named after it
    r"|AletheiaSample|ALETHEIA_\w+"           # its value type / its constants
    r"|[dD]:[\\/]dev[\\/]aletheia"            # a path into its checkout
    r"|bootstrap_npz",                        # the corpus-source name that only it had
    re.IGNORECASE,
)

#: Files allowed to contain the pattern: the guard itself, and nothing else.
_ALLOWED = {Path("tests/test_no_foreign_corpus.py")}


def _tracked() -> list[Path]:
    listing = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    return [Path(name) for name in listing
            if Path(name).suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".json"}]


class NoForeignCorpusArtefactTests(unittest.TestCase):
    def test_nothing_imports_reads_or_points_at_the_other_product(self) -> None:
        offenders: list[str] = []
        for relative in _tracked():
            if relative in _ALLOWED or relative.parts[0] == ".ai-memory":
                continue
            try:
                text = (_ROOT / relative).read_text(encoding="utf-8", errors="replace")
            except OSError:                                  # pragma: no cover - unreadable file
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if _ARTEFACT.search(line):
                    offenders.append(f"{relative.as_posix()}:{number}: {line.strip()[:110]}")
        self.assertEqual(offenders, [], "\n".join(
            ["these reach into another product's corpus or name it as an artefact:", *offenders]))

    def test_the_guard_can_actually_FAIL(self) -> None:
        """A guard nobody has seen fail is a guard nobody can trust."""
        for sample in ("from backend.aletheia_reader import x",
                       "path = 'D:/dev/aletheia/data/assets/gso'",
                       "ALETHEIA_READER_LOG_FILE = 'x.log'",
                       '"--source", default="bootstrap_npz",'):
            self.assertTrue(_ARTEFACT.search(sample), sample)

    def test_a_LESSON_is_not_an_artefact(self) -> None:
        """⛔ THE HALF THAT MUST NOT BE DELETED. `deep/net.py` justifies "classify, never regress" --
        the most load-bearing decision in the learned generator -- by citing that project's
        post-mortem. A guard that also matched prose would quietly delete the reason for decisions we
        would then re-litigate from scratch."""
        for sample in ("Aletheia's post-mortem on 1.2M labels named the orientation representation",
                       "the FIRST ranker -- fitted on aletheia's shaken corpus -- beat the calculator",
                       "# aletheia's ``quality`` is 0.000000 over all 218,529 rows"):
            self.assertIsNone(_ARTEFACT.search(sample), sample)


class EverySourceIsOursTests(unittest.TestCase):
    def test_the_corpus_registry_names_only_our_own_formats(self) -> None:
        from datagen.corpus.sources import SOURCES

        self.assertEqual(set(SOURCES), {"grasp_labels", "grasp_eval", "ranker_npz"})
        for name, description in SOURCES.items():
            self.assertNotIn("aletheia", description.lower(), name)

    def test_load_corpus_refuses_the_removed_source_BY_NAME(self) -> None:
        """An operator with an old command line must be told what happened, not handed a traceback
        about a missing file."""
        from datagen.corpus.sources import load_corpus

        with self.assertRaises((ValueError, FileNotFoundError)) as caught:
            load_corpus(_ROOT / "pyproject.toml", "bootstrap_npz")
        self.assertIn("bootstrap_npz", str(caught.exception))

    def test_the_gso_default_points_INSIDE_this_repo(self) -> None:
        from src.willy_sim.gso_assets import _DEFAULT_GSO_DIR

        self.assertFalse(Path(_DEFAULT_GSO_DIR).is_absolute(),
                         "the GSO default is an absolute path again")
        self.assertTrue((_ROOT / _DEFAULT_GSO_DIR).is_dir(),
                        f"{_DEFAULT_GSO_DIR} does not exist in this repo")


class TheDeletedPackageIsGoneTests(unittest.TestCase):
    def test_it_is_not_importable_and_not_re_exported(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            __import__("src.robot.grasping.training_data")

    def test_the_LIVE_record_synthesiser_is_the_one_that_survived(self) -> None:
        """It stamps more provenance than the deleted one did, and the trainers read it."""
        text = (_ROOT / "datagen/rl/collect.py").read_text(encoding="utf-8")
        for stamp in ("reward_model", "reward_interpretation", "dataset_origin"):
            self.assertIn(stamp, text, stamp)


if __name__ == "__main__":
    unittest.main()
