"""The sidecar lists the sphere pairs its self collision checks exactly as cuRobo lists them, in one array operation.

cuRobo builds the list in ``SelfCollisionKinematicsCfg.create_from_sphere_pair_distances``, a Python loop over every
sphere pair, once per robot load, and the sidecar loads the robot three times: 4.8 s of a 24 s start on the desk at 841
spheres (2026-10-07). ``collision_pair_rows`` builds the same rows in the same order from the same matrix, which the
sidecar hands cuRobo in the loop's place. Held here against the loop itself, transcribed from the installed cuRobo
(``curobo/_src/robot/types/self_collision_params.py``, v0.8.0-42-g8e734f3), on the matrices cuRobo builds: a link
against itself and an ignored pair of links at ``-inf``, a sphere that checks against nothing, a NaN, the dtypes.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _pairs():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "_curobo_pairs.py"
    spec = importlib.util.spec_from_file_location("_curobo_pairs_listed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _curobos_loop(coll_cpu: torch.Tensor) -> torch.Tensor:
    """cuRobo's own loop, as it stands in ``create_from_sphere_pair_distances``."""
    num_spheres = coll_cpu.shape[0]
    collision_pairs = torch.zeros((num_spheres * num_spheres, 2), dtype=torch.int16)
    collision_pairs_idx = 0
    for i in range(num_spheres):
        if torch.max(coll_cpu[i]) == -torch.inf:
            continue
        for j in range(i + 1, num_spheres):
            if coll_cpu[i, j] != -torch.inf:
                ix = collision_pairs_idx
                collision_pairs[ix, 0] = i
                collision_pairs[ix, 1] = j
                collision_pairs_idx += 1
    return collision_pairs[:collision_pairs_idx, :].contiguous().clone()


def _matrix(links: "list[int]", ignored: "set[tuple[int, int]]", seed: int) -> torch.Tensor:
    """The matrix cuRobo builds: radii sums between spheres of two links, ``-inf`` inside a link and on an ignored pair,
    and the smaller of the two directions, as ``compute_sphere_pair_distance_with_link_pair_ignores`` leaves it."""
    rng = np.random.default_rng(seed)
    owners = np.repeat(np.arange(len(links)), links)
    radii = rng.uniform(0.005, 0.08, owners.size)
    matrix = np.full((owners.size, owners.size), -np.inf, dtype=np.float32)
    for a in range(owners.size):
        for b in range(owners.size):
            pair = (min(owners[a], owners[b]), max(owners[a], owners[b]))
            if owners[a] != owners[b] and pair not in ignored:
                matrix[a, b] = radii[a] + radii[b] + rng.uniform(0.0, 0.004)
    tensor = torch.from_numpy(matrix)
    return torch.minimum(tensor, tensor.transpose(0, 1))


class TheRowsAreCurobosTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = _pairs()

    def _same(self, matrix: torch.Tensor) -> None:
        theirs = _curobos_loop(matrix)
        ours = self.pairs.collision_pair_rows(matrix.numpy())
        self.assertEqual(np.int16, ours.dtype)
        self.assertTrue(ours.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(theirs.numpy(), ours)

    def test_an_arm_and_a_hand_with_ignored_neighbours(self) -> None:
        # Seven links of an arm, a hand, a camera: neighbours ignored as a descriptor ignores them.
        links = [3, 9, 11, 7, 5, 4, 6, 8, 2]
        ignored = {(i, i + 1) for i in range(len(links) - 1)} | {(5, 7), (6, 8), (4, 6)}
        for seed in range(4):
            with self.subTest(seed=seed):
                self._same(_matrix(links, ignored, seed))

    def test_a_sphere_that_checks_against_nothing_lists_no_pair(self) -> None:
        # Link 1 is ignored against every other link, so each of its rows is -inf throughout.
        links = [4, 3, 5]
        self._same(_matrix(links, {(0, 1), (1, 2)}, 7))
        self._same(_matrix([2, 2], {(0, 1)}, 8))

    def test_a_nan_is_listed_as_cuRobo_lists_it(self) -> None:
        matrix = _matrix([3, 4, 2], {(0, 2)}, 9)
        matrix[1, 5] = float("nan")
        matrix[0, :] = -torch.inf
        matrix[0, 6] = float("nan")
        self._same(matrix)

    def test_too_few_spheres_list_nothing(self) -> None:
        for count in (0, 1):
            with self.subTest(count=count):
                matrix = torch.full((count, count), -torch.inf)
                self._same(matrix)
        self._same(torch.tensor([[-torch.inf, 0.1], [0.1, -torch.inf]]))

    def test_every_dtype_cuRobo_builds_in(self) -> None:
        base = _matrix([3, 5, 4], {(0, 1)}, 11)
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self._same(base.to(dtype))

    def test_a_matrix_that_is_not_square_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.pairs.collision_pair_rows(np.zeros((3, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.pairs.collision_pair_rows(np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
