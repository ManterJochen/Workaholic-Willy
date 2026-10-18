"""A config file that writes one key twice in one mapping is refused, naming both lines (found 2026-10-16).

YAML keeps the last of two equal keys and drops the first without a word: a tree with two ``motion:`` blocks under
``robot`` loaded its first block's ``planner`` as the default. Pinned here:

* two blocks of one name under one parent are refused, the refusal naming the key, the first line and the second;
* the same name under two different parents, and a key a merge (``<<:``) brings and the mapping overrides, load.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config.loader import ConfigError, _load_yaml


def _yaml(text: str) -> Path:
    folder = Path(tempfile.mkdtemp())
    path = folder / "robot.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class AKeyWrittenTwiceIsRefusedTests(unittest.TestCase):
    def test_two_blocks_of_one_name_under_one_parent_are_refused_naming_both_lines(self) -> None:
        path = _yaml("""\
            robot:
              motion:
                planner: ik
              vendor: ur
              motion:
                planner: curobo
            """)
        with self.assertRaises(ConfigError) as caught:
            _load_yaml(path)
        said = str(caught.exception)
        self.assertIn("found the key 'motion' twice in one mapping, first on line 2", said)
        self.assertIn("line 5", said)

    def test_one_name_under_two_parents_and_a_merge_that_is_overridden_load(self) -> None:
        path = _yaml("""\
            robot:
              motion:
                planner: ik
              gripper:
                motion: none
              base: &base
                x: 1
              place:
                <<: *base
                x: 2
            """)
        loaded = _load_yaml(path)
        self.assertEqual("ik", loaded["robot"]["motion"]["planner"])
        self.assertEqual(2, loaded["robot"]["place"]["x"])


if __name__ == "__main__":
    unittest.main()
