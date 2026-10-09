"""The promotion record a passed proof would leave beside a generator artifact, written for tests.

Nothing in the library writes one yet: ``deep judge`` and ``deep promote`` come after 2026-10-09. So a test that
builds a deep calculator for a cell writes the record itself, with every field name spelled out here rather than taken
from the reader, because they are the on-disk contract ``src/robot/grasping/deep/promotion.py`` reads and next week's
writer has to meet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def promote(artifact: str | Path, *, verdict: str = "pass", phase: str = "active", sha256: str | None = None,
            **fields: Any) -> Path:
    """Write ``<stem>.promotion.json`` beside ``artifact``: a passed proof at ``active`` unless told otherwise.

    ``sha256`` defaults to the artifact's own bytes; ``fields`` replace or add any other key, ``kind`` included.
    """
    path = Path(artifact)
    record: dict[str, Any] = {
        "kind": "set_grasp_generator_promotion",
        "artifact_sha256": sha256 if sha256 is not None else hashlib.sha256(path.read_bytes()).hexdigest(),
        "verdict": verdict,
        "phase": phase,
        "protocol": "test_proof",
        "protocol_version": 1,
        "evidence": {"written_by": "a test, not a proof"},
        "promoted_by": "tests",
        "promoted_at": "2026-10-09T12:00:00Z",
    }
    record.update(fields)
    target = path.with_suffix(".promotion.json")
    target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return target
