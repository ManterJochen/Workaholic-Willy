"""Every key the schema accepts is written down in ``config/all_keys``.

That tree is the only place an operator can find out a cell can be told something. The shipped
config under `config/` carries the values a cell runs with and deliberately omits keys whose default
is fine, so a key that exists in the schema and appears in neither is a feature nobody can discover,
which for a customer is the same as not having it.

MEASURED on this repo while adding the live planner world: ten key names were declared by the schema
and written down nowhere. Eight were the sim camera block, including `hfov_deg`, whose unset default
keeps Isaac's own narrower lens and frames less of the table than the real sensor does. A cell can be
mis-framed against its own hardware by a key the reference never mentions.

Names rather than dotted paths: the reference writes a map's contents as a commented example entry
under the empty map, and a top-level section as a directory, so a path comparison would report keys
that are plainly there. What must never happen is a key nobody can find at all.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import BaseModel

from src.config.schema.app import AppConfig

_ROOT = Path(__file__).resolve().parents[1]
_ALL_KEYS = _ROOT / "config" / "all_keys"


def _nested_models(annotation: object) -> list[type[BaseModel]]:
    """Every schema model reachable from a field annotation.

    Through unions, lists and maps alike: a key inside `list[PlannerMeshConfig]` is as real as one
    inside a plain block, and the reference has to describe both.
    """
    found: list[type[BaseModel]] = []
    stack: list[object] = [annotation]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            found.append(current)
            continue
        stack.extend(getattr(current, "__args__", ()) or ())
    return list(dict.fromkeys(found))


def _declared_paths(model: type[BaseModel], prefix: str = "") -> dict[str, str]:
    """Every key the schema declares, as name -> one dotted path that reaches it."""
    out: dict[str, str] = {}
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        out.setdefault(name, path)
        for sub in _nested_models(field.annotation):
            for sub_name, sub_path in _declared_paths(sub, path).items():
                out.setdefault(sub_name, sub_path)
    return out


def _written_names(text: str) -> set[str]:
    """Every key name in one reference file, whether it is live YAML or a commented example."""
    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip().lstrip("-").strip()
        if ":" not in line:
            continue
        head = line.split(":", 1)[0].strip()
        if head and head.replace("_", "").isalnum():
            names.add(head)
    return names


def _reference_names() -> set[str]:
    """Every name the reference offers, from its files and from its directory layout.

    The top-level sections are directories rather than keys: `camera:` is the folder holding
    `cam.yaml`, `stereomatcher.yaml` and `hand_eye.yaml`. A reader finds the section either way.
    """
    names: set[str] = set()
    for path in sorted(_ALL_KEYS.rglob("*.yaml")):
        names |= _written_names(path.read_text(encoding="utf-8"))
    names |= {p.name for p in _ALL_KEYS.iterdir() if p.is_dir()}
    return names


class AllKeysReferenceTests(unittest.TestCase):
    def test_the_reference_tree_is_where_this_test_thinks_it_is(self) -> None:
        # Without this, a moved or renamed tree makes the test below pass by reading nothing.
        self.assertTrue(_ALL_KEYS.is_dir(), f"no reference tree at {_ALL_KEYS}")
        self.assertGreater(len(list(_ALL_KEYS.rglob("*.yaml"))), 5)

    def test_the_check_catches_the_defect_it_was_written_for(self) -> None:
        # A schema key the reference does not mention has to come out as missing, or the assertion
        # below is a test that cannot fail.
        declared = _declared_paths(AppConfig)
        self.assertIn("hfov_deg", declared)
        self.assertNotIn("a_key_no_reference_would_carry", _reference_names())

    def test_every_key_the_schema_accepts_is_written_down(self) -> None:
        declared = _declared_paths(AppConfig)
        present = _reference_names()
        missing = sorted(f"{name} ({declared[name]})" for name in declared if name not in present)
        self.assertEqual(
            [],
            missing,
            "these keys are configurable and appear nowhere in config/all_keys:\n  "
            + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
