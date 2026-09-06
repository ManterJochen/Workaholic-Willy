"""`deep propose --scenes N` draws across families instead of taking a prefix.

⛔⛔ **A PREFIX IS NOT A SAMPLE, AND THIS REPOSITORY HAS NOW PAID FOR IT THREE TIMES.** A scene id is
`<family>_<index>`, so `sorted()` groups by FAMILY and any prefix is one family.

1. A multimodality claim stood on eight files that were all `packed`, and was retracted.
2. A held-out set that was a prefix of a group-ordered array turned out to be a handful of asset
   groups, with its own ceiling at 0.580 against the population's 0.385.
3. `deep propose` itself: its "40 unseen scenes" were 40 `packed` scenes, and the FIRST referee
   success rate this project ever computed was measured on them.

The lesson was written into `datagen/grasps/labels.py::_scene_order` the same evening the third one
was created, in a different file. So this guard checks the PROPERTY, at every cut, rather than an
example of it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.robot.grasping.deep.corpus.discovery import stratified_scenes as _stratified


def _corpus(counts: dict[str, int]) -> list[Path]:
    """A sorted list of scene paths, exactly as `scene_files` returns one."""
    return sorted(Path("clouds") / f"{family}_{i:06d}.npz"
                  for family, n in counts.items() for i in range(n))


_EVEN = _corpus({"bin": 30, "packed": 30, "sparse": 30})
_UNEVEN = _corpus({"bin": 4, "packed": 60, "sparse": 200})


class DrawTests(unittest.TestCase):

    def test_EVERY_cut_holds_every_family(self) -> None:
        """⭐ THE PROPERTY, not an example. Wherever `--scenes` happens to stop, the draw is a
        cross-section. Checking one N would pass on an order that is only balanced at the front."""
        for count in range(3, 91, 3):
            with self.subTest(count):
                drawn = _stratified(_EVEN, count, seed=0)
                self.assertEqual(len(drawn), count)
                families = {p.stem.rsplit("_", 1)[0] for p in drawn}
                self.assertEqual(families, {"bin", "packed", "sparse"})

    def test_a_PLAIN_prefix_would_have_failed_that(self) -> None:
        """The control. If `sorted()` also passed the test above, the draw would be decoration."""
        families = {p.stem.rsplit("_", 1)[0] for p in _EVEN[:40]}
        self.assertEqual(families, {"bin", "packed"},
                         "sorted() no longer groups by family; retire this guard")

    def test_it_is_NOT_a_prefix_within_a_family_either(self) -> None:
        """⚠ Round robin alone takes the first scenes of every family, which is a prefix one level
        down. The scenes inside a family are shuffled under the seed."""
        drawn = _stratified(_EVEN, 30, seed=0)
        packed = [p.stem for p in drawn if p.stem.startswith("packed")]
        self.assertNotEqual(packed, sorted(packed)[:len(packed)],
                            f"the packed scenes are still the first ones: {packed}")

    def test_the_draw_is_REPRODUCIBLE_under_a_seed(self) -> None:
        """A referee number has to be repeatable on the same scenes, or two runs are two experiments."""
        self.assertEqual(_stratified(_EVEN, 21, seed=7), _stratified(_EVEN, 21, seed=7))
        self.assertNotEqual(_stratified(_EVEN, 21, seed=7), _stratified(_EVEN, 21, seed=8))

    def test_a_SHORT_family_runs_out_without_dropping_anything(self) -> None:
        """A real corpus is not balanced: v5 has far more `sparse` scenes than `bin` ones. The short
        family runs out and the others continue, which is the honest answer."""
        drawn = _stratified(_UNEVEN, 40, seed=0)
        self.assertEqual(len(drawn), 40)
        self.assertEqual(len(drawn), len(set(drawn)))
        counts = {f: sum(1 for p in drawn if p.stem.startswith(f))
                  for f in ("bin", "packed", "sparse")}
        self.assertEqual(counts["bin"], 4, f"the short family was over- or under-drawn: {counts}")

    def test_asking_for_MORE_than_the_corpus_holds_returns_the_corpus(self) -> None:
        drawn = _stratified(_UNEVEN, 10_000, seed=0)
        self.assertEqual(sorted(drawn), sorted(_UNEVEN))

    def test_a_family_filter_restricts_and_says_so_when_it_matches_nothing(self) -> None:
        drawn = _stratified(_UNEVEN, 20, seed=0, families="bin,packed")
        self.assertEqual({p.stem.rsplit("_", 1)[0] for p in drawn}, {"bin", "packed"})
        with self.assertRaises(FileNotFoundError) as caught:
            _stratified(_UNEVEN, 5, seed=0, families="pile")
        self.assertIn("pile", str(caught.exception))
        self.assertIn("bin", str(caught.exception), "the message does not say what IS available")


class WiringTests(unittest.TestCase):

    def test_the_flags_exist(self) -> None:
        from src.robot.grasping.deep.__main__ import build_parser

        args = build_parser().parse_args(
            ["propose", "--artifact", "w.pt", "--clouds", "c", "--out", "o.jsonl",
             "--scenes", "40", "--families", "bin,sparse"])
        self.assertEqual(args.scenes, 40)
        self.assertEqual(args.families, "bin,sparse")

    def test_the_command_no_longer_slices_a_prefix(self) -> None:
        """⛔ The exact line that produced the contaminated denominator.

        ⚠ BY AST, ON THE CALL, not by searching the file for a substring. The earlier version looked
        for the text `_stratified(files, args.scenes`, which stopped meaning anything the moment the
        function was promoted out of `__main__` and renamed. Reading the call from the parse tree
        asks the question that actually matters: does the propose command draw a stratified sample,
        whatever that function is currently called.
        """
        import ast as _ast

        source = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertNotIn("files = files[:args.scenes]", source)

        command = next(node for node in _ast.walk(_ast.parse(source))
                       if isinstance(node, _ast.FunctionDef) and node.name == "_cmd_propose")
        called = {node.func.id for node in _ast.walk(command)
                  if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)}
        self.assertIn("stratified_scenes", called,
                      "the propose command stopped drawing a stratified sample")

    def test_the_run_PRINTS_which_families_it_drew(self) -> None:
        """A denominator nobody can see is a denominator nobody checks."""
        source = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertIn("famil(y/ies)", source)


if __name__ == "__main__":
    unittest.main()
