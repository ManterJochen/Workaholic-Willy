"""Every runner's ``--help`` must actually render.

argparse runs help strings through %-formatting when it expands them, so a single literal ``%`` in a
help string does not fail at import, at parse, or in any test that calls the runner -- it fails ONLY
when a human types ``--help``. MEASURED 2026-07-24: ``run_dense_demo_endgame --help`` raised
``ValueError: unsupported format character ')'`` from a help text reading "recovery is ~12%", so that
runner's flags were undiscoverable and nothing in the suite noticed.

A source scan rather than 28 subprocess ``--help`` invocations: it is instant, it needs no Isaac, and
it names the offending line directly.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: A run of '%' characters. argparse consumes '%%' as a literal percent; any ODD-length run leaves a
#: dangling '%' that starts a format spec and blows up on the next character.
_PERCENT_RUN = re.compile(r"%+")


def _cli_source_files() -> list[Path]:
    """Every module that builds an argparse parser (runners, package CLIs, examples)."""
    out: list[Path] = []
    for base in ("src", "examples"):
        root = _ROOT / base
        if root.exists():
            out.extend(p for p in root.rglob("*.py") if "argparse" in p.read_text(encoding="utf-8", errors="ignore"))
    return sorted(out)


def _bad_help_strings(path: Path) -> list[tuple[int, str]]:
    """(lineno, text) for every ``help=`` / ``description=`` literal argparse cannot expand."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a syntax error is another test's problem
        return []
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in ("help", "description", "epilog"):
                continue
            if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                continue  # f-strings / concatenations are not literal help text
            text = kw.value.value
            if any(len(m.group(0)) % 2 for m in _PERCENT_RUN.finditer(text)):
                bad.append((kw.value.lineno, text))
    return bad


class RunnerHelpStringTests(unittest.TestCase):
    def test_no_argparse_help_string_has_an_unescaped_percent(self) -> None:
        offenders = [
            f"{path.relative_to(_ROOT).as_posix()}:{line} -> {text!r}"
            for path in _cli_source_files()
            for line, text in _bad_help_strings(path)
        ]
        self.assertEqual(
            offenders, [],
            "argparse %-expands help text, so these would raise ValueError on --help only. "
            "Double the percent sign ('12%%' renders as '12%'):\n  " + "\n  ".join(offenders),
        )

    def test_the_guard_actually_catches_the_bug_it_was_written_for(self) -> None:
        """A guard that cannot fail is not a guard. Pin the exact shape of the defect that was found."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.py"
            probe.write_text(
                'import argparse\n'
                'ap = argparse.ArgumentParser()\n'
                'ap.add_argument("--a", help="recovery is ~12%")\n'   # odd run -> caught
                'ap.add_argument("--b", help="recovery is ~12%%")\n',  # even run -> fine
                encoding="utf-8",
            )
            found = _bad_help_strings(probe)
        self.assertEqual([t for _, t in found], ["recovery is ~12%"])


if __name__ == "__main__":
    unittest.main()
