"""Every `python -m` CLI must render its help on a cp1252 console.

⛔ ONE GLYPH KILLS A WHOLE COMMAND. A `⚠` in a `help=` string makes argparse's own `print_help`
raise `UnicodeEncodeError` on Windows, so `--help` exits non-zero and prints a traceback instead of
the help. It happened twice: once in `deep train --help`, and once in `python -m datagen --help`,
where the second was still broken because the guard written for the first only covered `deep`.

The guard therefore enumerates the CLIs rather than naming one, and renders the TOP-LEVEL help plus
every subcommand's, because argparse only reaches a subparser's strings when that subparser is asked.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import unittest
from typing import Any

#: Every module in this repo with a `python -m <module>` entry point that parses arguments.
CLIS = (
    "datagen",
    "src.config",
    "src.robot.grasping.deep",
    "src.robot.grasping.replay",
    "src.robot.grasping.rl",
)


def _render(module: str, argv: list[str]) -> str:
    import importlib
    import sys

    cli = importlib.import_module(f"{module}.__main__")
    buf = io.StringIO()
    old = sys.argv
    sys.argv = [module.split(".")[-1], *argv]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            with contextlib.suppress(SystemExit):
                cli.main()
    finally:
        sys.argv = old
    return buf.getvalue()


def _subcommands(text: str) -> list[str]:
    """The choice list argparse prints as `{a,b,c}`, which is where SUBPARSERS are named.

    ⛔⛔ **THIS FOUND THE FIRST BRACE GROUP, WHICH IS NOT ALWAYS THE SUBCOMMANDS, AND FOR ONE OF THE
    TWO CLIs IT NEVER WAS.** `datagen` is a flat parser: its command is a positional with `choices`,
    and its usage line opens with `--engine`'s `{isaac,mujoco,none}`. So this guard has been running
    `datagen isaac --help`, `datagen mujoco --help` and `datagen none --help` -- three commands that
    do not exist -- and has never checked a single one of the 28 real ones.

    A guard that tests three non-existent commands passes for exactly the same reason it would pass
    if every real one were broken, which is the inert-switch shape this repository fences everywhere
    else. Found on 2026-09-03 by an audit that counted the subcommands independently.

    ⚠ The repair reads the PARSER rather than the rendered text. Text was the wrong source: the same
    brackets mean subcommands in one CLI and a flag's choices in the other, and no amount of care
    with string offsets fixes that.
    """
    return _from_usage(text)


def _every_cli_reports_more_than_the_usage_line() -> None:
    """Documentation for the reader: the two paths disagree, and that disagreement was the bug."""


def _from_usage(text: str) -> list[str]:
    """The old text-scraping path, kept so the two CLIs can be compared against each other."""
    start = text.find("{")
    if start < 0:
        return []
    end = text.find("}", start)
    return [c.strip() for c in text[start + 1:end].split(",") if c.strip()] if end > start else []


def _commands_of(module: Any) -> list[str]:
    """Every subcommand the module's own parser accepts, read from the parser itself.

    Handles both shapes this repository uses: `add_subparsers` (the `deep` CLI) and a positional with
    `choices` (the `datagen` CLI). Reading the parser is what makes the guard cover what actually
    exists rather than what a usage line happens to render first.
    """
    builder = getattr(module, "build_parser", None)
    if builder is None:
        # ⚠ NOT AN ERROR AND NOT A SILENT SKIP. Three of the five CLIs build their parser inside
        # `main`, so there is nothing to ask. Returning an empty list means "no subcommands were
        # checked here", which the coverage test below then asserts is true only of the CLIs that
        # HAVE no subcommands. The old text-scraping path is what produced three fake ones, so it is
        # not the fallback.
        return []
    names: list[str] = []
    for action in builder()._actions:                          # noqa: SLF001 - argparse has no API
        if isinstance(action, argparse._SubParsersAction):     # noqa: SLF001
            names.extend(action.choices)
        elif not action.option_strings and action.choices:
            names.extend(action.choices)
    return names


class HelpMustSurviveACp1252ConsoleTests(unittest.TestCase):
    def _assert_ascii(self, where: str, text: str) -> None:
        self.assertTrue(text.strip(), f"{where} rendered nothing")
        bad = sorted({c for c in text if ord(c) > 127})
        self.assertFalse(
            bad,
            f"{where} contains {[(c, f'U+{ord(c):04X}') for c in bad]}. "
            "argparse writes help through the console encoding, so on cp1252 this raises "
            "UnicodeEncodeError and the command dies instead of printing its help.")
        # The same call the user actually makes, through the real stdout encoding.
        text.encode("cp1252")

    def test_every_cli_help_is_ascii(self) -> None:
        for module in CLIS:
            with self.subTest(cli=module):
                top = _render(module, ["--help"])
                self._assert_ascii(f"{module} --help", top)
                for sub in _commands_of(importlib.import_module(f"{module}.__main__")):
                    with self.subTest(cli=module, sub=sub):
                        self._assert_ascii(f"{module} {sub} --help",
                                           _render(module, [sub, "--help"]))

    def test_the_two_CLIs_WITH_subcommands_are_actually_covered(self) -> None:
        """⛔ THE ASSERTION THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT. The guard used to scrape the
        first `{a,b,c}` group out of a usage line, which for `datagen` is `--engine`'s choices, so it
        ran `datagen isaac --help` and two other commands that do not exist and never touched one of
        the 28 real ones. A guard that tests three non-existent commands passes for the same reason
        it would pass if every real one were broken.

        Counting them is what makes the coverage a fact rather than an assumption."""
        counts = {module: len(_commands_of(importlib.import_module(f"{module}.__main__")))
                  for module in CLIS}
        self.assertGreaterEqual(counts.get("datagen", 0), 20,
                                f"datagen's subcommands are not being checked: {counts}")
        self.assertGreaterEqual(counts.get("src.robot.grasping.deep", 0), 10,
                                f"the deep CLI's subcommands are not being checked: {counts}")

    def test_the_usage_scrape_and_the_parser_DISAGREE_which_was_the_bug(self) -> None:
        """Kept as evidence. If these two ever agree for `datagen`, the reason the guard was rewritten
        has gone away and somebody should know."""
        scraped = _from_usage(_render("datagen", ["--help"]))
        real = _commands_of(importlib.import_module("datagen.__main__"))
        self.assertNotEqual(sorted(scraped), sorted(real))
        self.assertEqual(sorted(scraped), ["isaac", "mujoco", "none"],
                         "the usage line's first brace group is no longer --engine's choices")

    def test_the_guard_would_actually_fail(self) -> None:
        """A guard nobody has seen fail is a guard nobody knows works."""
        parser = argparse.ArgumentParser(prog="x")
        parser.add_argument("--y", help="a \u26a0 warning")
        with self.assertRaises(UnicodeEncodeError):
            parser.format_help().encode("cp1252")
