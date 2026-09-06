"""No shipped config YAML may declare the same key twice in one mapping.

PyYAML silently keeps the LAST occurrence. There is no warning, no error, and nothing downstream can
tell the difference between "this block was never written" and "this block was written and thrown
away" -- so an entire config section can vanish and the only symptom is behaviour that ignores what
the file plainly says.

MEASURED 2026-08-09, on this repo, while converging the sim tool frame: `robot.sim.yaml` grew a second
`gripper:` key. A freshly added `tool_frame` block under the first one was discarded on load, and the
config CLI reported `source: undeclared` while the YAML in front of me said `willy`. Reproduced
standalone:

    yaml.safe_load("robot:\\n  gripper:\\n    max_width_mm: 85.0\\n  gripper:\\n    min_width_mm: 5.0\\n")
    -> {'robot': {'gripper': {'min_width_mm': 5.0}}}          # max_width_mm is simply gone

That was a cosmetic loss because it was caught in minutes. The same accident on `safety:` or
`workspace_limits:` would ship a cell whose guards are configured by whichever duplicate happened to be
last in the file.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DATA = _ROOT / "config"


class _DuplicateDetectingLoader(yaml.SafeLoader):
    """A SafeLoader that refuses, rather than silently overwrites, a repeated mapping key."""


def _no_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None,
                f"duplicate key {key!r} (first seen at line {seen[key] + 1})",
                key_node.start_mark,
            )
        seen[key] = key_node.start_mark.line
    return yaml.SafeLoader.construct_mapping(loader, node, deep)  # type: ignore[arg-type]


_DuplicateDetectingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def _yaml_files() -> list[Path]:
    return sorted(p for p in _CONFIG_DATA.rglob("*.yaml") if p.is_file())


class NoDuplicateConfigKeysTests(unittest.TestCase):
    def test_the_guard_catches_the_defect_it_was_written_for(self) -> None:
        """A guard that cannot fail is not a guard. This is the exact shape that bit us."""
        doc = "robot:\n  gripper:\n    max_width_mm: 85.0\n  gripper:\n    min_width_mm: 5.0\n"
        self.assertEqual(  # what PyYAML does by default: the first block is gone, no error
            yaml.safe_load(doc)["robot"]["gripper"], {"min_width_mm": 5.0}
        )
        with self.assertRaises(yaml.constructor.ConstructorError) as ctx:
            yaml.load(doc, Loader=_DuplicateDetectingLoader)
        self.assertIn("duplicate key", str(ctx.exception))
        self.assertIn("gripper", str(ctx.exception))

    def test_every_shipped_config_yaml_is_free_of_duplicate_keys(self) -> None:
        files = _yaml_files()
        self.assertTrue(files, "no config YAML found -- the glob is wrong, not the tree")
        offenders: list[str] = []
        for path in files:
            try:
                yaml.load(path.read_text(encoding="utf-8"), Loader=_DuplicateDetectingLoader)
            except yaml.constructor.ConstructorError as exc:
                offenders.append(f"{path.relative_to(_ROOT).as_posix()}: {exc.problem} {exc.problem_mark}")
        self.assertEqual(
            offenders, [],
            "PyYAML keeps the LAST occurrence silently, so everything under the earlier one is "
            "discarded with no error:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
