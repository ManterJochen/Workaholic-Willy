"""`python -m datagen build-cloud-corpus` — the step the WS2 runbook opens with.

⚠ WHY THIS FILE EXISTS. `build_cloud_corpus` had NO CLI at all: it was reachable only by importing it
from a Python session. So the dev corpus the whole generator was designed against was extracted by
hand, its provenance was "someone ran a function once", and the runbook's first step cited a
subcommand — `build-corpus` — that did not exist. Nothing caught it, because nothing in the suite
dispatches a datagen subcommand.

⚠⚠ AND WRITING THESE FOUND A SECOND ONE. `--corpus-out` is a SHARED flag that carried
`build-ranker-corpus`'s default: `logs/dl/corpus/ranker_v1.npz`, a FILE. The new command writes a
DIRECTORY of per-scene clouds, so without an explicit value it tried to `mkdir` over the ranker's
corpus — and the "you must pass --corpus-out" guard was dead, because argparse never handed it a
`None`. A guard behind a default is not a guard.

What these pin is the seam, not the extraction: the command is registered, it routes to its OWN
handler, and both ways of getting the destination wrong refuse with a usage code.
"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest import mock

from datagen import __main__ as cli


def _record(seen: list):
    """Stand in for `write_corpus`: remember the path and return it, because the caller `.stat()`s it."""
    def written(path, _table):
        seen.append(Path(path))
        stub = Path(cli.__file__)          # any real file; only its size is printed
        return stub
    return written


def _args(**kwargs) -> argparse.Namespace:
    """The Namespace `main()` hands `_dispatch`, with this command's fields."""
    # ⚠ EVERY FIELD THE PARSER WOULD SET, including the ones added later. A partial Namespace here
    # is not a smaller test, it is a different caller: the real one always comes from argparse, which
    # fills every default. `kinds` was added on 2026-09-03 and this dict not updating is what the two
    # failures below were.
    base = {"command": "build-cloud-corpus", "name": "v1", "out": None, "corpus_out": None,
            "scenes": None, "masks": None, "physics": None, "labels": "grasps.jsonl",
            "kinds": "jaw"}
    return argparse.Namespace(**{**base, **kwargs})


class RegistrationTests(unittest.TestCase):
    def test_the_subcommand_is_in_the_parsers_CHOICES(self) -> None:
        """The runbook named `build-corpus` for hours and nothing could say otherwise. Parsed out of
        the real parser rather than grepped, so a command registered nowhere fails here."""
        with self.assertRaises(SystemExit):
            cli.main(["definitely-not-a-command"])          # argparse rejects unknown names
        with mock.patch.object(cli, "_cmd_build_cloud_corpus", return_value=0) as handler:
            self.assertEqual(cli.main(["build-cloud-corpus", "--corpus-out", "x"]), 0)
        handler.assert_called_once()

    def test_it_does_not_collide_with_the_RANKER_corpus_command(self) -> None:
        """Two commands walk the same dataset. One writes a point-cloud directory for the generator,
        the other a flat feature table for the ranker. Routing to the wrong one produces a file that
        loads fine and means something else."""
        seen: list[str] = []
        with mock.patch.object(cli, "_cmd_build_cloud_corpus",
                               side_effect=lambda a, c: seen.append("cloud") or 0), \
             mock.patch.object(cli, "_cmd_build_ranker_corpus",
                               side_effect=lambda a, c: seen.append("ranker") or 0):
            cli._dispatch(_args(), mock.MagicMock())
            cli._dispatch(_args(command="build-ranker-corpus"), mock.MagicMock())
        self.assertEqual(seen, ["cloud", "ranker"])


def _config(**overrides):
    """A DatagenConfig for the handlers, which now take one -- see the root test below."""
    from datagen.config import DatagenConfig

    return DatagenConfig(**overrides)


class DestinationTests(unittest.TestCase):
    """Both ways of getting `--corpus-out` wrong, and the shared default that caused one of them."""

    def test_the_shared_flag_carries_NO_command_specific_default(self) -> None:
        """It used to default to the ranker's output FILE. A per-command default on a shared flag is
        wrong for every command that is not the one it was written for — and it silently disabled
        this command's own required-argument check."""
        with mock.patch.object(cli, "_cmd_build_cloud_corpus", side_effect=lambda a, c: a) as handler:
            cli.main(["build-cloud-corpus"])
        self.assertIsNone(handler.call_args.args[0].corpus_out)

    def test_the_RANKER_keeps_its_old_default_in_its_own_handler(self) -> None:
        """Moving a default is a behaviour change until it is shown not to be. Checked by watching
        the path `write_corpus` actually receives -- a source grep here would pass while the handler
        wrote somewhere else entirely.

        ⚠ THE PATCH TARGET MOVED WITH THE IMPLEMENTATION, THE ASSERTION DID NOT. The handler now
        calls `corpus.service.RankerCorpus`, which imports from `datagen.corpus.build` rather than
        through the package re-export, so patching `datagen.corpus.write_corpus` stopped
        intercepting. The claim under test is unchanged and is the one that matters: the default
        destination is still `logs/dl/corpus/ranker_v1.npz`.
        """
        import numpy as np

        table = {"valid": np.array([True, False]),
                 "object_key": np.array(["a", "b"], dtype="<U4")}
        wrote: list[Path] = []
        with mock.patch("datagen.corpus.build.build_ranker_corpus", return_value=table),              mock.patch("datagen.corpus.build.write_corpus",
                        side_effect=_record(wrote)):
            cli._cmd_build_ranker_corpus(
                _args(command="build-ranker-corpus", name="whatever"), _config())
        self.assertEqual(wrote, [Path("logs/dl/corpus/ranker_v1.npz")])

    def test_the_ranker_still_HONOURS_an_explicit_destination(self) -> None:
        """The fallback must not shadow the flag it replaced."""
        import numpy as np

        table = {"valid": np.array([True]), "object_key": np.array(["a"], dtype="<U4")}
        wrote: list[Path] = []
        with mock.patch("datagen.corpus.build.build_ranker_corpus", return_value=table),              mock.patch("datagen.corpus.build.write_corpus",
                        side_effect=_record(wrote)):
            cli._cmd_build_ranker_corpus(
                _args(command="build-ranker-corpus", name="whatever", corpus_out="elsewhere.npz"),
                _config())
        self.assertEqual(wrote, [Path("elsewhere.npz")])

    def test_no_destination_REFUSES_with_a_usage_code(self) -> None:
        self.assertEqual(cli._cmd_build_cloud_corpus(_args(), _config()), cli._EXIT_USAGE)

    def test_a_FILE_destination_REFUSES_rather_than_mkdir_ing_over_it(self) -> None:
        """The exact accident the shared default made automatic: this command writes a directory."""
        code = cli._cmd_build_cloud_corpus(
            _args(corpus_out="logs/dl/corpus/ranker_v1.npz", out="logs/dl/mix_probe"), _config())
        self.assertEqual(code, cli._EXIT_USAGE)

    def test_a_missing_dataset_REFUSES_rather_than_raising(self) -> None:
        """A typo in `--name` is a usage error, not a traceback out of the extractor."""
        code = cli._cmd_build_cloud_corpus(
            _args(name="no-such-dataset", out="logs/dl/mix_probe", corpus_out="unused-dir"),
            _config())
        self.assertEqual(code, cli._EXIT_USAGE)

    def test_the_dataset_root_comes_from_the_CONFIG_like_every_sibling(self) -> None:
        """⛔ IT USED TO BE HARD-CODED to `logs/p5/datasets` whatever the config said, while `build`
        and `label-grasps` both read `config.output.root`. So `--config x.json --name v3` wrote the
        dataset in one place and looked for it in another. It failed loudly only because nothing of
        that name sat in the old directory: with a same-named dataset there, this would have built a
        corpus from the WRONG scenes and stamped it with the right name.

        Checked on the path the refusal REPORTS, not on a source grep -- a grep would pass while the
        handler read somewhere else entirely.
        """
        import io
        from contextlib import redirect_stderr

        captured = io.StringIO()
        with redirect_stderr(captured):
            code = cli._cmd_build_cloud_corpus(
                _args(name="no-such-dataset", corpus_out="unused-dir"),
                _config(output={"root": "some/configured/place"}))
        self.assertEqual(code, cli._EXIT_USAGE)
        self.assertIn("some", captured.getvalue().replace("\\", "/"),
                      "the refusal names a root the config never mentioned")


if __name__ == "__main__":
    unittest.main()
