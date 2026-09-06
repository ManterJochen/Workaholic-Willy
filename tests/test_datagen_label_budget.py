"""`--label-budget N`: stop labelling at N labels, without handing back one family's scenes.

⛔⛔ THE TRAP THIS FILE EXISTS FOR. Scene directories are named `<family>_<number>`, so `sorted()`
groups them by FAMILY. A budget that simply stopped early would return every `bin` scene and no
`sparse` one, and the corpus would carry a bias nobody put there on purpose. This repository has
already paid for the same shape once: a held-out set that was a PREFIX of a group-ordered array turned
out to be a handful of asset groups, with its own ceiling at 0.580 against the population's 0.385.

So a budgeted run walks the families round robin. Without a budget the order is `sorted()` exactly,
byte for byte, because a default that changed would silently invalidate every corpus already built.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from datagen.grasps.labels import _scene_order


def _dirs(names: list[str]) -> list[Path]:
    return [Path("scenes") / name for name in names]


_CORPUS = _dirs(sorted(
    [f"bin_{i:06d}" for i in range(4)]
    + [f"packed_{i:06d}" for i in range(4)]
    + [f"sparse_{i:06d}" for i in range(4)]))


class OrderTests(unittest.TestCase):

    def test_WITHOUT_a_budget_the_order_is_untouched(self) -> None:
        """⛔ The default has to be byte-identical. A corpus built before this flag existed and one
        built after must be the same file, or every number taken on the old one is unmoored."""
        self.assertEqual(_scene_order(_CORPUS, balanced=False), _CORPUS)

    def test_a_budgeted_run_walks_the_families_ROUND_ROBIN(self) -> None:
        order = _scene_order(_CORPUS, balanced=True)
        self.assertEqual([p.name for p in order[:3]],
                         ["bin_000000", "packed_000000", "sparse_000000"])

    def test_ANY_prefix_of_a_budgeted_order_holds_every_family(self) -> None:
        """⭐ THE PROPERTY, not an example of it. Wherever the budget happens to stop, the slice is a
        cross-section. Checking one prefix would pass on an order that is only balanced at the front."""
        order = _scene_order(_CORPUS, balanced=True)
        for cut in range(3, len(order) + 1, 3):
            with self.subTest(cut):
                families = {p.name.rsplit("_", 1)[0] for p in order[:cut]}
                self.assertEqual(families, {"bin", "packed", "sparse"})

    def test_a_PLAIN_prefix_would_have_failed_that(self) -> None:
        """The control. If `sorted()` order also passed the test above, the test would be measuring
        nothing and the round robin would be decoration."""
        families = {p.name.rsplit("_", 1)[0] for p in _CORPUS[:3]}
        self.assertEqual(families, {"bin"}, "sorted() no longer groups by family; retire this guard")

    def test_nothing_is_lost_or_repeated(self) -> None:
        order = _scene_order(_CORPUS, balanced=True)
        self.assertEqual(sorted(p.name for p in order), sorted(p.name for p in _CORPUS))
        self.assertEqual(len(order), len(set(order)))

    def test_uneven_families_still_come_out_whole(self) -> None:
        """A real corpus is not balanced: v5 has far more `sparse` scenes than `bin` ones. The short
        families run out and the long one continues, which is the honest answer, and the point is
        that nothing is dropped when they do."""
        uneven = _dirs(["bin_000000"] + [f"sparse_{i:06d}" for i in range(5)])
        order = _scene_order(uneven, balanced=True)
        self.assertEqual(len(order), 6)
        self.assertEqual(order[0].name, "bin_000000")
        self.assertEqual(sorted(p.name for p in order), sorted(p.name for p in uneven))

    def test_an_empty_corpus_does_not_raise(self) -> None:
        self.assertEqual(_scene_order([], balanced=True), [])


class StampTests(unittest.TestCase):

    def test_the_budget_stops_BETWEEN_scenes(self) -> None:
        """⚠ A scene's labels are written together or not at all. Half an object's grasps is a corpus
        row that looks complete and is not, and every count downstream is per object."""
        source = Path("datagen/grasps/labels.py").read_text(encoding="utf-8")
        self.assertIn("CHECKED BETWEEN SCENES, NEVER INSIDE ONE", source)

    def test_the_report_carries_the_budget_and_says_it_is_partial(self) -> None:
        """A budgeted corpus and a complete one are different statements about a dataset and the
        files look identical, which is the same reason `density` and `jaw_model` are in there."""
        source = Path("datagen/grasps/labels.py").read_text(encoding="utf-8")
        for key in ('"label_budget"', '"scenes_available"', '"partial"'):
            self.assertIn(key, source)

    def test_the_flag_reaches_the_function(self) -> None:
        from datagen.__main__ import build_parser

        args = build_parser().parse_args(["label-grasps", "--label-budget", "50000"])
        self.assertEqual(args.label_budget, 50000)
        source = Path("datagen/__main__.py").read_text(encoding="utf-8")
        self.assertIn("label_budget=args.label_budget", source)
        self.assertIn("label_budget=label_budget", source)


if __name__ == "__main__":
    unittest.main()
