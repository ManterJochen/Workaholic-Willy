"""The arm descriptions as a build script reads them: raw YAML, no schema in between (P0f F6, decision D9).

From F7 the build scripts read ``config/arms/<model>.yaml`` by path with PyYAML, some of them in the
cuRobo environment, where the config layer cannot be imported. So what a value IS in the raw file matters, not what
pydantic would coerce it to: a commit YAML read as a number, an integer written where a float was meant, or a ``-0.0``
would each reach a script unconverted and move the bytes it writes. Each F6 key adds its rows to ``_KEYS``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.config.arms import available_arms

_ARMS = Path(__file__).resolve().parents[2] / "config" / "arms"

#: A key's path under ``arm:`` and the one type its raw value must be. ``*`` is every joint row.
_KEYS: dict[tuple[str, ...], type] = {
    ("base", "link"): str,
    ("base", "chain_yaw_deg"): int,
    ("base", "controller_yaw_deg"): int,
    ("joints", "*", "link"): str,
    ("joints", "*", "part"): str,
    ("joints", "*", "planner_limit_deg"): list,
    ("joints", "*", "planner_limit_deg", "*"): float,
    ("planner", "position_limit_clip_rad"): float,
    ("upstream", "adapter"): str,
    ("upstream", "repo"): str,
    ("upstream", "commit"): str,
    ("upstream", "licence"): str,
    ("upstream", "manifest_rows_sha256"): str,
}


def _raw(model: str) -> dict:
    return yaml.safe_load((_ARMS / f"{model}.yaml").read_text(encoding="utf-8"))["arm"]


def _at(data: object, path: tuple[str, ...]) -> list[tuple[str, object]]:
    """Every value at ``path``, each with where it was found; ``*`` fans out over a list."""
    if not path:
        return [("", data)]
    head, rest = path[0], path[1:]
    if head == "*":
        assert isinstance(data, list)
        return [(f"{index}.{where}".rstrip("."), value) for index, item in enumerate(data) for where, value in _at(item, rest)]
    assert isinstance(data, dict)
    return [(f"{head}.{where}".rstrip("."), value) for where, value in _at(data[head], rest)]


class EveryKeyIsTheTypeAScriptExpectsTests(unittest.TestCase):
    def test_every_arm_file_holds_every_key_as_its_type(self) -> None:
        wrong = [f"{model}: {where} is {type(value).__name__}, not {kind.__name__}"
                 for model in available_arms() for path, kind in _KEYS.items()
                 for where, value in _at(_raw(model), path) if type(value) is not kind]
        self.assertEqual([], wrong)
        self.assertEqual(6, len(_at(_raw("ur5e"), ("joints", "*", "link"))), "the fan-out reaches every row")

    def test_an_unquoted_commit_of_digits_would_be_read_as_a_number(self) -> None:
        """THE CONTROL: the check sees what YAML does to an unquoted value."""
        self.assertIs(int, type(yaml.safe_load("commit: 1234567")["commit"]))


if __name__ == "__main__":
    unittest.main()
