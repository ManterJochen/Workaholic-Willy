"""`--want-graspable N`: budget a corpus in the unit a customer actually has.

⭐ WHY THE UNIT IS NOT "MESHES". A library is not 800,000 usable objects. MEASURED across the v5 bank,
36.25 % of screened meshes earn a jaw label, and on objaverse it is 6.1 %, so "screen 1,000 meshes" is
a number nobody can turn into a plan. "Give me 300 jaw-graspable objects, and tell me how many you had
to look at" is one they can.

⛔ AND A STOPPED SCREEN IS STAMPED. The file it writes looks exactly like a complete one, and every
consumer treats a mesh with no row as unmeasured, so an unstamped partial screen silently becomes the
claim "this library has few graspable objects" when the truth is that most of it was never looked at.

⚠ NO MESH IS LOADED HERE. The screen's worker is replaced and so is its process pool, because what is
under test is the STOPPING RULE and the stamp, not the labeller. A test that also loaded meshes would
be slower and would fail for reasons that have nothing to do with either.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from datagen.assets import prepare


class _FakePool:
    """Stands in for the process pool. `map` keeps SUBMISSION order, as the real one does."""

    answers: dict[str, dict] = {}
    cancelled: bool = False

    def __init__(self, *_a, **_k) -> None:
        pass

    def map(self, _fn, items):
        return (dict(_FakePool.answers[Path(item[1]).name]) for item in items)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        _FakePool.cancelled = cancel_futures


def _screen(verdicts: list[int], want: int | None, out: Path) -> list[dict]:
    """Run the real `screen_meshes` over a temp library of empty files with canned verdicts."""
    with TemporaryDirectory() as library:
        directory = Path(library) / "test"
        directory.mkdir(parents=True)
        _FakePool.answers = {}
        for index, jaw in enumerate(verdicts):
            (directory / f"m{index:03d}.obj").write_text("", encoding="utf-8")
            _FakePool.answers[f"m{index:03d}.obj"] = {
                "asset_id": f"a{index}", "source": "test", "status": "ok",
                "jaw": jaw, "suction": 0, "box_screen_jaw_ok": True}
        sources = {**prepare.SUPPORTED_SOURCES, "test": "obj"}
        with mock.patch.object(prepare, "SUPPORTED_SOURCES", sources), \
             mock.patch.object(prepare.futures, "ProcessPoolExecutor", _FakePool):
            return prepare.screen_meshes(["test"], library=Path(library), out=out,
                                         density="grid", want_graspable=want,
                                         report=lambda _line: None)


class StopTests(unittest.TestCase):

    def test_it_stops_once_it_has_enough(self) -> None:
        """The whole point: four graspable meshes are wanted and the rest are never screened."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen.json"
            rows = _screen([1, 0, 1, 1, 0, 1, 1, 1, 1, 1], want=4, out=out)
        self.assertEqual(sum(1 for r in rows if r["jaw"] > 0), 4)
        self.assertLess(len(rows), 10, "it screened everything anyway")

    def test_the_queued_work_is_CANCELLED_and_not_merely_ignored(self) -> None:
        """⚠ Without `cancel_futures` the stop is only a stop for the reader: the pool drains every
        task already queued, which over a large library is most of the work the caller asked to skip.
        The loop would still return early and every other test here would still pass."""
        with TemporaryDirectory() as tmp:
            _screen([1] * 20, want=2, out=Path(tmp) / "screen.json")
        self.assertTrue(_FakePool.cancelled)

    def test_the_stopped_file_is_STAMPED_partial(self) -> None:
        """⛔ Without this a screen that covered a fifth of a library reads as a measurement of it."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen.json"
            _screen([1] * 10, want=3, out=out)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(payload["partial"])
        self.assertEqual(payload["screened"], 3)
        self.assertEqual(payload["of"], 10)
        self.assertEqual(payload["want_graspable"], 3)

    def test_a_COMPLETE_screen_is_not_stamped_partial(self) -> None:
        """The control. A stamp that is always on says nothing."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen.json"
            _screen([1, 0, 1], want=None, out=out)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertFalse(payload["partial"])
        self.assertEqual(payload["screened"], payload["of"])

    def test_a_target_it_never_reaches_screens_everything(self) -> None:
        """Asking for more graspable objects than a library holds is not an error, it is a library
        that ran out, and the honest answer is the whole screen plus a count."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen.json"
            rows = _screen([1, 0, 0, 1], want=99, out=out)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 4)
        self.assertFalse(payload["partial"])

    def test_the_prefix_is_REPRODUCIBLE(self) -> None:
        """⚠ Reproducibility, not tidiness. `map` yields in submission order, so the same library and
        the same target screen the same prefix on every machine. A completion-ordered stream would
        hand a different subset to every run and the corpus would stop being a thing you can
        rebuild."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen.json"
            first = _screen([0, 1, 0, 1, 1, 1], want=2, out=out)
            second = _screen([0, 1, 0, 1, 1, 1], want=2, out=out)
        self.assertEqual([r["asset_id"] for r in first], [r["asset_id"] for r in second])
        self.assertEqual([r["asset_id"] for r in first], ["a0", "a1", "a2", "a3"])


class ReaderTests(unittest.TestCase):

    def test_the_layout_WARNS_when_it_reads_a_partial_screen(self) -> None:
        """⛔ The reader treats a mesh with no row as unmeasured, which is right, and that is exactly
        why a partial screen has to announce itself: otherwise the bank silently shrinks and the run
        looks like a library with few graspable objects."""
        source = Path("datagen/scenes/layout.py").read_text(encoding="utf-8")
        self.assertIn('payload.get("partial")', source)
        self.assertIn("PARTIAL", source)


class FlagTests(unittest.TestCase):

    def test_the_flag_is_reachable(self) -> None:
        from datagen.__main__ import build_parser

        args = build_parser().parse_args(["prepare-assets", "--want-graspable", "250"])
        self.assertEqual(args.want_graspable, 250)

    def test_both_commands_pass_it_through(self) -> None:
        """A flag that parses and never reaches the function is the shape of inert switch this
        repository built a wiring guard for."""
        source = Path("datagen/__main__.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("want_graspable=args.want_graspable"), 2)
        self.assertEqual(source.count("want_graspable=want_graspable"), 2)


if __name__ == "__main__":
    unittest.main()
