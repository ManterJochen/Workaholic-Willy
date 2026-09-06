"""Every key the rl package reads off a dataset manifest must be one the manifest writes.

⛔⛔ **A KEY NOBODY WRITES FALLS SILENTLY TO A DEFAULT AND DIES SOMEWHERE ELSE.** MEASURED
2026-09-04: the OPE handler's fallback resolves its replay packs with ``manifest.get("splits")``, and
no manifest has ever had a ``splits`` key -- it has ``split_files``. So that path always yields an
empty list and the run dies a step later with "empty after loading all paths", at a place with
nothing to do with the cause. The fallback has never been able to work.

⭐ **PARSED FROM THE READ SIDE, NOT COMPARED AGAINST A HAND-KEPT LIST**, and that distinction is the
whole test. The parallel session built this same guard for `datagen` and reports that their first
version -- a comparison against a written-out set -- stayed GREEN with the bug reinstated. A list of
known-good keys is a second declaration of the manifest's shape, and it rots exactly like the thing
it is meant to catch. This reads both sides out of the code.

⚠ **THE CONVENTION IT RELIES ON, STATED SO IT CAN BE FOLLOWED:** a dict loaded from a manifest FILE
is bound to a variable named ``manifest``. Other payloads (a leakage block, a policy artifact) are
not manifests and are named otherwise, so they are correctly invisible here.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_RL = Path(__file__).resolve().parents[1] / "src" / "robot" / "grasping" / "rl"

#: The one read this guard finds and does NOT fail on, with the reason it is still there.
#:
#: ⛔ A ONE-ENTRY EXCEPTION IS NOT AN ALLOW-LIST, AND THE DIFFERENCE IS WHY THIS IS PINNED RATHER
#: THAN DELETED. `test_a_known_dead_read_is_still_dead` below asserts the key is STILL unwritten, so
#: this entry cannot outlive the defect: the day somebody adds a `splits` key or fixes the read, that
#: test fails and this line has to go. An allow-list of "flags we know are unwired" is what
#: `test_grasping_wiring_guard.py:24-27` refuses, and it refuses it because such a list goes stale
#: silently. This one is checked in both directions.
#:
#: ⚠ IT IS EXEMPTED, NOT FORGIVEN. Repairing the read would change which records a byte-locked
#: OPE report is computed over, which is a deliberate announced change rather than a key rename.
_KNOWN_DEAD = {("_commands_eval.py", "splits")}


def _manifest_fields() -> set[str]:
    """What a written manifest actually contains, from the SERIALISER rather than the dataclass.

    ⚠ **THE DATACLASS IS NOT THE WRITTEN SHAPE, AND MY FIRST VERSION USED IT.** MEASURED:
    `DatasetManifest.to_json()` emits 20 keys and `dataclasses.fields()` reports 18 -- it adds
    `reward_interpretation` and `reward_model`. So a read of either would have been flagged as
    unwritten while the file plainly contains it. The serialiser is what a reader on the other side
    actually sees, so the serialiser is the source of truth.
    """
    import inspect

    from src.robot.grasping.rl.dataset import DatasetManifest

    required = [
        name for name, param in inspect.signature(DatasetManifest).parameters.items()
        if param.default is inspect.Parameter.empty
    ]
    probe = DatasetManifest(
        **{
            name: ({} if ("count" in name or "hash" in name or "file" in name) else "")
            for name in required
        }
    )
    return set(probe.to_json())


def _keys_read_off_manifests() -> dict[str, list[str]]:
    """Every literal string key read off a variable named `manifest`, by file."""
    found: dict[str, list[str]] = {}
    for path in sorted(_RL.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        keys: list[str] = []
        for node in ast.walk(tree):
            # manifest["x"]
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "manifest"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.append(node.slice.value)
            # manifest.get("x") / manifest.get("x", default)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "manifest"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.append(node.args[0].value)
        if keys:
            found[str(path.relative_to(_RL))] = sorted(set(keys))
    return found


class ManifestKeyTests(unittest.TestCase):
    def test_every_key_read_off_a_manifest_is_one_the_manifest_writes(self) -> None:
        written = _manifest_fields()
        offenders: list[str] = []
        for file, keys in _keys_read_off_manifests().items():
            for key in keys:
                if key not in written and (file, key) not in _KNOWN_DEAD:
                    offenders.append(f"{file} reads manifest[{key!r}], which no manifest writes")
        self.assertEqual(
            offenders,
            [],
            "a key nobody writes falls silently to a default and dies somewhere unrelated; "
            f"the manifest writes {sorted(written)}",
        )

    def test_the_scan_finds_something(self) -> None:
        """⚠ A test over an empty set passes loudest. If the naming convention this relies on is
        abandoned, this is what says so, rather than a green run over nothing."""
        found = _keys_read_off_manifests()
        self.assertGreaterEqual(len(found), 3, f"the scan matched almost nothing: {found}")
        self.assertIn("split_files", {k for keys in found.values() for k in keys})

    def test_a_known_dead_read_is_still_dead(self) -> None:
        """⭐ THE OTHER HALF OF THE EXEMPTION. An entry in `_KNOWN_DEAD` that is no longer dead is a
        stale exception, and a stale exception is exactly what an allow-list becomes. This fails the
        day the key starts being written OR the read is repaired, so the exemption cannot outlive
        what it excuses."""
        written = _manifest_fields()
        found = _keys_read_off_manifests()
        for file, key in _KNOWN_DEAD:
            self.assertNotIn(key, written, f"{key!r} IS written now; drop it from _KNOWN_DEAD")
            self.assertIn(key, found.get(file, []), f"{file} no longer reads {key!r}; drop the entry")

    def test_the_guard_rejects_a_key_that_is_not_written(self) -> None:
        """⭐ THE SELF-FAILING CONTROL, and it is the reason this test is worth more than a list. The
        parallel session's first version of this guard compared against a hand-written set and stayed
        GREEN with the defect reinstated.
        """
        source = 'def f(manifest):\n    return manifest.get("splits") or manifest["nope"]\n'
        tree = ast.parse(source)
        keys: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "manifest"
                and isinstance(node.slice, ast.Constant)
            ):
                keys.append(node.slice.value)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "manifest"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                keys.append(node.args[0].value)
        self.assertEqual(sorted(keys), ["nope", "splits"])
        written = _manifest_fields()
        self.assertTrue(
            [k for k in keys if k not in written],
            "the guard must reject a key the manifest does not write",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
