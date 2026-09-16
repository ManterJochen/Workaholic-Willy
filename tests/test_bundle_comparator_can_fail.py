"""The comparator that decides whether two bundles are the same arm can fail, and on what (B5, S6).

Two readers produce a collision bundle for one robot: Isaac's composed articulation, and UR's own collision STL files
through the rendered description. Whether those two agree is the question that says the frame chain is right, and it
is answered by a comparator. A comparator that cannot fail turns that whole gate into a formality, so this file is
about the comparator itself, on the bundles this repository already ships.

Vertex ORDER differs between readers, so the comparison is order free: centroid, extent, and the distance from every
vertex of one mesh to the nearest point of the other, in both directions. What it must catch:

* a link moved by 1 mm, which is a frame error small enough to look like nothing and large enough to matter;
* a link MIRRORED across one of its own DH planes, which keeps every distance, every extent and every centroid a
  naive check would compare, and would leave the guard watching a left handed arm.

What it must NOT call a difference: the same arrays in another order.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "src" / "robot" / "safety" / "data"

#: The arms Isaac baked. ur10 is left out on purpose: its bundle is hulls of a visual mesh until B5 S7.
_ISAAC_ARMS = ("ur3", "ur3e", "ur5", "ur5e", "ur10e")


def _compare():
    path = _ROOT / "scripts" / "isaac" / "_bundle_compare.py"
    spec = importlib.util.spec_from_file_location("_bundle_compare_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arrays(model: str) -> dict:
    with np.load(_DATA / f"{model}_collision_meshes.npz", allow_pickle=True) as bundle:
        return {name: np.array(bundle[name]) for name in bundle.files}


class TheComparatorIsOrderFreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compare = _compare()

    def test_a_bundle_against_itself_is_zero(self) -> None:
        for model in _ISAAC_ARMS:
            with self.subTest(model=model):
                difference = self.compare.compare(_arrays(model), _arrays(model))
                self.assertEqual(len(difference.links), len(self.compare.LINKS))
                for link in difference.links:
                    self.assertEqual((link.centroid_mm, link.p99_mm, link.max_mm), (0.0, 0.0, 0.0))
                self.assertTrue(difference.agrees(self.compare.CEILINGS))

    def test_the_same_vertices_in_another_order_are_the_same_arm(self) -> None:
        """⭐ Two readers list a mesh's vertices in their own order. A comparator that called that a difference would
        report every honest re-bake as a frame error."""
        rng = np.random.default_rng(20260916)
        shuffled = _arrays("ur5e")
        for key in (f"{link}__v" for link in self.compare.LINKS):
            vertices = shuffled[key]
            shuffled[key] = vertices[rng.permutation(len(vertices))]

        difference = self.compare.compare(_arrays("ur5e"), shuffled)
        self.assertTrue(difference.agrees(self.compare.CEILINGS), difference.render())
        self.assertEqual(max(link.max_mm for link in difference.links), 0.0)

    def test_one_link_moved_by_a_millimetre_is_caught_on_every_arm(self) -> None:
        """⭐ THE CONTROL that sets the ceiling. 1 mm is a frame error nobody would notice by looking."""
        self.assertLessEqual(self.compare.CEILINGS.centroid_mm, 1.0,
                             "a centroid ceiling above 1 mm would let the case below pass")
        for model in _ISAAC_ARMS:
            for link in self.compare.LINKS:
                with self.subTest(model=model, link=link):
                    moved = _arrays(model)
                    moved[f"{link}__v"] = moved[f"{link}__v"] + np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    difference = self.compare.compare(_arrays(model), moved)
                    self.assertFalse(difference.agrees(self.compare.CEILINGS))
                    row = next(row for row in difference.links if row.link == link)
                    self.assertGreater(row.centroid_mm, self.compare.CEILINGS.centroid_mm)

    def test_a_mirrored_link_is_caught_on_the_axes_that_are_not_its_own_symmetry(self) -> None:
        """⭐ THE CONTROL this whole comparator exists for. A reflection keeps every distance, every extent and every
        centroid of a symmetric part, so an extent check calls a left handed arm identical.

        An axis is skipped only where the link is its OWN mirror across that plane, measured here rather than
        assumed, and every link must be caught by at least one axis.
        """
        for model in _ISAAC_ARMS:
            for link in self.compare.LINKS:
                caught = []
                for axis in range(3):
                    sign = np.ones(3, dtype=np.float32)
                    sign[axis] = -1.0
                    mirrored = _arrays(model)
                    mirrored[f"{link}__v"] = mirrored[f"{link}__v"] * sign

                    difference = self.compare.compare(_arrays(model), mirrored)
                    row = next(row for row in difference.links if row.link == link)
                    if row.max_mm <= self.compare.CEILINGS.max_mm:
                        continue  # this link IS its own mirror across this plane
                    caught.append(axis)
                    self.assertFalse(difference.agrees(self.compare.CEILINGS))
                with self.subTest(model=model, link=link):
                    self.assertTrue(caught, f"{model}/{link} survives a mirror on all three axes")

    def test_the_report_names_the_link_and_the_numbers(self) -> None:
        moved = _arrays("ur5e")
        moved["wrist_2__v"] = moved["wrist_2__v"] + np.array([0.0, 2.0, 0.0], dtype=np.float32)
        difference = self.compare.compare(_arrays("ur5e"), moved)

        text = difference.render()
        self.assertIn("wrist_2", text)
        self.assertIn("2.0", text)
        self.assertEqual(difference.to_dict()["worst"]["link"], "wrist_2")

    def test_a_bundle_missing_a_link_is_refused_rather_than_skipped(self) -> None:
        """A comparison that quietly drops what it cannot find reports agreement about the part it never read."""
        short = _arrays("ur5e")
        del short["forearm__v"]
        with self.assertRaises(KeyError):
            self.compare.compare(_arrays("ur5e"), short)


if __name__ == "__main__":
    unittest.main()
