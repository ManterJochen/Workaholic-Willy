"""What a customer sees while a six-hour training run works, and after every epoch of it.

⛔ THE GAP. A `train-set` run's only feedback was one log line per EPOCH, about ten minutes apart, and
the learning curve was drawn once at the end by a separate command. Between epochs the terminal was
silent, so a working run and a hung one looked identical, and the only way to tell was to open a log
file and read a timestamp.

⚠ TWO CONSTRAINTS THIS PROJECT HAS ALREADY PAID FOR, and both are load-bearing here.

`ascii=True` is not a style choice. tqdm's default bar is Unicode block characters, this terminal is
cp1252, and printing one raises `UnicodeEncodeError`. That exact failure has broken `deep report`
once. A crash inside a progress bar would end a six-hour run for a decoration.

And the bar goes to `stderr`, only when `stderr` is a terminal. The training logger also writes to
`logs/deep_generator.log`; a bar emits a carriage return per update, so one written into that file
turns it into thousands of unreadable lines. Every long arm here is started with `nohup`, which is
exactly the case that must stay silent.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from src.robot.grasping.deep.train.progress import epoch_bar, is_interactive, postfix


class TheBarStaysOutOfTheLogTests(unittest.TestCase):

    def test_it_is_SILENT_when_nobody_is_watching(self) -> None:
        """⭐ THE PROPERTY THAT MAKES IT SAFE. Under `nohup`, a redirect, CI or a container without a
        TTY, the bar must emit nothing at all."""
        with epoch_bar(total=10, epoch=1, epochs=3, enabled=False) as bar:
            for _ in range(10):
                bar.update(1)
                bar.set_postfix_str("anything")

    def test_is_interactive_is_False_under_a_pipe(self) -> None:
        """pytest captures stderr, so this IS the non-interactive case and answering True here would
        mean the check is not reading what it claims to."""
        self.assertFalse(is_interactive())

    def test_a_zero_length_epoch_does_not_draw(self) -> None:
        """An empty order is a real state (a fold whose split came out empty) and a bar over zero
        batches would divide by it."""
        with epoch_bar(total=0, epoch=1, epochs=1, enabled=True) as bar:
            bar.update(1)

    def test_the_bar_survives_tqdm_being_absent(self) -> None:
        """⚠ A training run must not die because its progress bar could not import. tqdm is pinned,
        so this is about the failure being HANDLED rather than about it being likely."""
        import builtins

        real = builtins.__import__

        def refuse(name, *args, **kwargs):  # noqa: ANN001, ANN202
            if name == "tqdm":
                raise ImportError("pretend it is not installed")
            return real(name, *args, **kwargs)

        builtins.__import__ = refuse
        try:
            with epoch_bar(total=5, epoch=1, epochs=1, enabled=True) as bar:
                bar.update(1)
        finally:
            builtins.__import__ = real


class TheBarIsASCIITests(unittest.TestCase):

    def test_the_call_passes_ascii_True(self) -> None:
        """⛔ Read off the source, because the alternative is a crash on a customer's terminal that
        this suite would never see: pytest captures stderr, so the bar never actually draws here."""
        source = Path(
            "src/robot/grasping/deep/train/progress.py").read_text(encoding="utf-8")

        self.assertIn("ascii=True", source)

    def test_the_postfix_is_ASCII_and_short(self) -> None:
        """It is written to the same terminal. A postfix carrying twelve metrics wraps the line on a
        narrow terminal and is read by nobody."""
        text = postfix({"total": 1.204, "approach_error_deg": 124.0, "axis_error_deg": 60.0,
                        "offset_error_mm": 31.9, "coverage": 0.4}, batches=4)

        self.assertTrue(text.isascii(), text)
        self.assertLess(len(text), 60, f"the postfix is too long for a terminal: {text!r}")
        self.assertIn("total", text)
        self.assertNotIn("coverage", text, "the postfix grew past the two numbers a person watches")

    def test_the_postfix_AVERAGES_over_batches(self) -> None:
        """⚠ THE DENOMINATOR. `totals` accumulates a sum across batches; showing it raw would climb
        forever and read as a loss that never improves. This repository has a memory about that."""
        text = postfix({"total": 8.0}, batches=4)

        self.assertIn("2.000", text)

    def test_an_empty_epoch_yields_an_empty_postfix(self) -> None:
        self.assertEqual(postfix({"total": 1.0}, batches=0), "")


class TheCurveGrowsDuringTheRunTests(unittest.TestCase):
    """⛔ The chart existed and only afterwards. `deep report` drew it once a run had finished, so for
    the six hours that matter a customer had a JSON file and a log line every ten minutes. The data
    was on disk after every epoch; nothing redrew it."""

    def test_the_loop_refreshes_the_curve_after_writing_epochs(self) -> None:
        import ast

        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "_refresh_curve"]

        self.assertTrue(calls, "nothing redraws the curve during a run")

    def test_the_reader_can_no_longer_be_pointed_at_the_wrong_thing(self) -> None:
        """⛔ MY OWN DEFECT, caught by running it rather than reading it, and then removed at the root.

        `build_report` used to take `source="run" | "log"`. Passing anything but `"run"` fell through
        to the LOG parser, which opened the directory as a file and raised PermissionError, and the
        `except` around the curve swallowed it, so the picture was silently never written. This test
        first pinned `source="run"` at the call site.

        The parameter is gone on 2026-09-04, and with it the defect: the log reader was removed
        because it could not see a live run at all, so there is one reader and nothing to select.
        The assertion therefore moved from the CALL to the SIGNATURE, which is the only place that
        can make the mistake impossible rather than merely absent today.

        ⚠ BY AST, NOT BY A SLICE BETWEEN TWO FUNCTION NAMES. The earlier version cut the source from
        `def _refresh_curve(` to `def _refit_on_everything(`, a boundary named after a NEIGHBOUR, and
        this file has already lost one such boundary when the neighbour was deleted.
        """
        import inspect

        from src.robot.grasping.deep.eval.run_report import build_report

        parameters = inspect.signature(build_report).parameters
        self.assertNotIn("source", parameters,
                         "build_report can be pointed at a second reader again")
        self.assertEqual(["name", "path"], list(parameters))

    def test_the_curve_is_built_from_the_run_DIRECTORY(self) -> None:
        """The other half: the signature cannot be wrong, so check what is actually handed to it."""
        module = ast.parse(
            Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8"))
        refresh = next(n for n in ast.walk(module)
                       if isinstance(n, ast.FunctionDef) and n.name == "_refresh_curve")
        calls = [n for n in ast.walk(refresh)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_report"]
        self.assertEqual(1, len(calls), "build_report is not called exactly once")
        self.assertEqual([], calls[0].keywords, "the call regrew a keyword argument")
        self.assertEqual("directory", ast.unparse(calls[0].args[1]),
                         "the curve is built from something other than the run directory")

    def test_it_never_raises(self) -> None:
        """⚠ THE WHOLE CONTRACT. This runs inside the training loop, and a picture must not end a
        six-hour run: a missing backend, a file locked by a viewer on Windows, a report too short to
        plot."""
        from src.robot.grasping.deep.train.trainer import _refresh_curve

        _refresh_curve(Path("this/directory/does/not/exist"))

    def test_one_epoch_is_not_a_curve(self) -> None:
        """Plotting a single point produces a picture that says nothing and looks like a result."""
        source = Path("src/robot/grasping/deep/train/trainer.py").read_text(encoding="utf-8")
        body = source[source.index("def _refresh_curve("):source.index("def _refit_on_everything(")]

        self.assertIn("len(report.epochs) < 2", body)


class TheDependencyIsDeclaredTests(unittest.TestCase):

    def test_tqdm_is_PINNED_rather_than_transitive(self) -> None:
        """⚠ It was importable and only transitively, which is worse than absent: the customer
        display would have depended on whatever else happened to pull it in."""
        text = Path("requirements.txt").read_text(encoding="utf-8")

        self.assertIn("tqdm==", text)


if __name__ == "__main__":
    unittest.main()
