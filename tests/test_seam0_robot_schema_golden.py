"""R7.1 Seam-0 — pin the robot config schema + the loaded config trees before the robot_schema.py split.

Regenerate the goldens (ONLY when the current behavior is the intended baseline):

    WILLY_SEAM0_UPDATE_GOLDEN=1 python -m pytest tests/test_seam0_robot_schema_golden.py

R7.1 splits robot_schema.py (2337 lines, 50 StrictModel classes) into per-concern leaf modules behind a
re-exporting robot_schema.py shim (RobotConfig stays the aggregate; every import path unchanged). The split
must be byte-identical at three surfaces:

(A) ``RobotConfig.model_json_schema()`` — every class name (the 49 ``$defs``), field type, default, numeric/
    string constraint, and ``additionalProperties:false`` (``extra='forbid'``) across the whole tree. This is
    the strongest single proof. NOTE: ``model_json_schema()`` output is pydantic-version-sensitive — captured
    on **pydantic==2.13.4**; a routine pydantic bump reds this independent of the split (regen deliberately then).
(B) the loaded PRODUCTION config tree (``backend/config/data``).
(C) the loaded SIM config = the same ``backend/config/data`` tree under the ``sim`` profile (the ``*.sim.yaml``
    overlays; the old separate ``data_sim`` tree was consolidated into it). Reached through the same loader.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from src.config import load_config, reload_config
from src.config.loader import set_active_profile
from src.config.schema.robot import RobotConfig

_DIR = Path(__file__).resolve().parent / "data" / "seam0_robot_schema"
_SCHEMA = _DIR / "robot_config_schema_golden.json"
_DATA = _DIR / "loaded_config_data_golden.json"
_DATA_SIM = _DIR / "loaded_config_data_sim_golden.json"
# The sim config is now the production tree under the `sim` profile (no separate data_sim/ tree).
_SIM_DATA_DIR = Path(__file__).resolve().parent.parent / "config" / "data"


def _schema_blob() -> str:
    return json.dumps(RobotConfig.model_json_schema(), indent=2, sort_keys=True) + "\n"


def _data_blob() -> str:
    reload_config()  # loader is lru_cached by (root, profile) — clear before each load
    set_active_profile(None)
    return load_config(None).model_dump_json(indent=2) + "\n"


def _data_sim_blob() -> str:
    reload_config()
    set_active_profile("sim")
    try:
        return load_config(str(_SIM_DATA_DIR)).model_dump_json(indent=2) + "\n"
    finally:
        set_active_profile(None)


def _flat(obj, prefix: str = "") -> dict[str, object]:
    """Every leaf of a JSON blob as ``dotted.key -> value``, for a key-level drift report."""
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for key, value in obj.items():
            out.update(_flat(value, f"{prefix}.{key}"))
        return out
    if isinstance(obj, list):
        out = {}
        for i, value in enumerate(obj):
            out.update(_flat(value, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def _drift_report(path: Path, old_text: str, new_text: str, *, limit: int = 12) -> str:
    """Say WHICH keys moved, and how to accept it -- not just that 29,000 characters differ.

    These goldens are byte-compared, so the most routine config edit (changing one IP) fails them. The
    default unittest output for that is `Diff is 29243 characters long. Set self.maxDiff to None`, which
    names no key and offers no next step -- so an intended one-value change reads exactly like a
    regression. Splitting the drift into ADDED / REMOVED / CHANGED keys is what makes the difference
    between "I meant that" and "I broke something" visible in one line, which is the whole job of a
    golden.
    """
    try:
        old, new = _flat(json.loads(old_text)), _flat(json.loads(new_text))
    except Exception:  # noqa: BLE001 - a malformed golden still deserves the plain failure
        return f"{path.name} drifted vs the Seam-0 golden (and could not be parsed for a key diff)"
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    lines = [
        f"{path.name} drifted vs the Seam-0 golden: "
        f"+{len(added)} added, -{len(removed)} removed, ~{len(changed)} changed.",
    ]
    for label, keys in (("CHANGED", changed), ("ADDED", added), ("REMOVED", removed)):
        for key in keys[:limit]:
            detail = f"{old.get(key)!r} -> {new.get(key)!r}" if label == "CHANGED" else repr(
                new.get(key) if label == "ADDED" else old.get(key))
            lines.append(f"  {label:8s} {key.lstrip('.')} = {detail}")
        if len(keys) > limit:
            lines.append(f"  {label:8s} ... and {len(keys) - limit} more")
    lines.append(
        "If every line above is intended, accept it with "
        "WILLY_SEAM0_UPDATE_GOLDEN=1 python -m pytest tests/test_seam0_robot_schema_golden.py"
    )
    return "\n".join(lines)


class RobotSchemaSeam0Tests(unittest.TestCase):
    def test_schema_and_trees_byte_identical(self) -> None:
        blobs = {
            _SCHEMA: _schema_blob(),
            _DATA: _data_blob(),
            _DATA_SIM: _data_sim_blob(),
        }
        if os.environ.get("WILLY_SEAM0_UPDATE_GOLDEN"):
            _DIR.mkdir(parents=True, exist_ok=True)
            for path, blob in blobs.items():
                path.write_text(blob, encoding="utf-8")
            self.skipTest("seam0 goldens regenerated (WILLY_SEAM0_UPDATE_GOLDEN)")
        for path, blob in blobs.items():
            with self.subTest(golden=path.name):
                self.assertTrue(
                    path.exists(),
                    f"missing golden {path}; regenerate with WILLY_SEAM0_UPDATE_GOLDEN=1",
                )
                self.assertEqual(
                    blob, path.read_text(encoding="utf-8"),
                    _drift_report(path, path.read_text(encoding="utf-8"), blob),
                )


if __name__ == "__main__":
    unittest.main()
