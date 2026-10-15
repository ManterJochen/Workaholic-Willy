"""A sphere map names the cover fit that wrote it, under the fit's path since P0f F4 or the one before (P0f F10).

The planner reads a hand's map only where its provenance says the cover fit wrote it (``hand.cover_fit_refusal``): a grid
fit left every shipped hand a hole. The fit moved from ``scripts/curobo/fit_cover_spheres.py`` to ``scripts/arms/fit.py``
in F4 and writes its new path since F10, while a map a customer fitted before still carries the old one. Both are the
same fit and both read; any other writer is refused by name.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.robot.safety.planning.hand import COVER_FIT_WRITER, cover_fit_refusal

_MAPS = Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "planning" / "robot"


def _provenance(writer: str) -> dict:
    """The committed 2F-85 map's provenance under another writer, and nothing else changed."""
    document = yaml.safe_load((_MAPS / "robotiq_2f85_gripper_spheres.yml").read_text(encoding="utf-8"))
    return {**document["_provenance"], "generated_by": writer}


class TheFitReadsUnderEitherOfItsPathsTests(unittest.TestCase):
    def test_the_fit_writes_its_path_under_scripts_arms(self) -> None:
        self.assertEqual("scripts/arms/fit.py", COVER_FIT_WRITER)

    def test_a_map_the_moved_fit_wrote_reads(self) -> None:
        self.assertIsNone(cover_fit_refusal(_provenance("scripts/arms/fit.py"), "robotiq_2f85"))

    def test_a_map_the_fit_wrote_before_the_move_still_reads(self) -> None:
        self.assertIsNone(cover_fit_refusal(_provenance("scripts/curobo/fit_cover_spheres.py"), "robotiq_2f85"))

    def test_another_writer_is_refused_by_name(self) -> None:
        """THE CONTROL: the alias admits the fit's old path, not every writer."""
        refused = cover_fit_refusal(_provenance("scripts/curobo/build_gripper_spheres.py"), "robotiq_2f85")
        assert refused is not None
        self.assertIn("scripts/curobo/build_gripper_spheres.py", refused)
        self.assertIn("scripts/arms/fit.py", refused)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
