"""A bare bench read lower than it is declared is said, and nothing is raised (fix plan Track S, test 10; the owner,
2026-10-02).

The support solids follow the camera's reading. Where the camera reads the bench lower than it is declared, it may read
a thin part as low, and the part keeps less than the guard's distance plus the allowance. That cannot be told from a
bench that sags, so the world says where and how much, at most once a minute, and changes nothing: the solids are not
raised, a recalibration is a person's decision. On the owner's looks the bare table beside the mat reads 5-7 mm low.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from src.robot.safety.planning.support_surfaces import BENCH_LOW_LEAST_CELLS
from tests import _cell_2026_10_01 as cell

_LOGGER = "src.robot.safety.planning.live_world"


def _build(world, owner):  # noqa: ANN001, ANN202
    snapshot = world.world_for(self_envelope=owner.envelope(np.radians(cell.LOOK_0_DEG)),
                               near_point_mm=(-130.0, -690.0, 100.0))
    assert snapshot.perceived is not None, snapshot.reason
    return snapshot.perceived


class ABenchReadLowIsSaidTests(unittest.TestCase):
    def test_4_mm_low_is_said_once_naming_where_and_1_mm_is_not(self) -> None:
        owner = cell.owner_cell()
        look = cell.look("P1")
        low_world = owner.world(look, cell.level_table(look, -4.0))
        with self.assertLogs(_LOGGER, logging.WARNING) as said:
            low = _build(low_world, owner)
        self.assertEqual(len(said.records), 1)
        message = said.records[0].getMessage()
        self.assertIn("reads lower than declared", message)
        self.assertRegex(message, r"x -?\d+ to -?\d+ mm, y -?\d+ to -?\d+ mm read -[34]\.\d mm")
        self.assertIn("Nothing is raised", message)
        reading = low.supports.bench_reading
        self.assertGreaterEqual(reading.low_cells, BENCH_LOW_LEAST_CELLS)
        self.assertAlmostEqual(reading.median_mm, -4.0, delta=0.5)
        self.assertIn("bare bench read -4", low.supports.render())

        fine_world = owner.world(look, cell.level_table(look, -1.0))
        logger = logging.getLogger(_LOGGER)
        with self.assertNoLogs(logger, logging.WARNING):
            fine = _build(fine_world, owner)
        self.assertEqual(fine.supports.bench_reading.low_cells, 0)

        # Nothing is raised either way: the bench's solid stands at its band and the allowance.
        for world in (low, fine):
            bench = [box for box in world.boxes if box.kind == "bench"]
            self.assertEqual(len(bench), 1)
            self.assertAlmostEqual(bench[0].center_mm[2] + bench[0].dims_mm[2] / 2.0, 7.0, places=6)
            self.assertEqual([box.name for box in world.boxes if box.kind == "support"], [])

    def test_it_is_said_at_most_once_a_minute(self) -> None:
        owner = cell.owner_cell()
        look = cell.look("P1")
        world = owner.world(look, cell.level_table(look, -4.0))
        logger = logging.getLogger(_LOGGER)
        with self.assertLogs(logger, logging.WARNING):
            _build(world, owner)
        with self.assertNoLogs(logger, logging.WARNING):
            _build(world, owner)
        world._bench_warned_at -= 61.0  # noqa: SLF001 (a minute passes)
        with self.assertLogs(logger, logging.WARNING) as said:
            _build(world, owner)
        self.assertEqual(len(said.records), 1)


if __name__ == "__main__":
    unittest.main()
