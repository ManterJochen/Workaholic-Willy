"""R10.1 — GraspAttemptRecord schema-evolution regression guard.

``GraspAttemptRecord.to_dict()`` is hand-written key-by-key, NOT derived from ``dataclasses.fields()``.
``tests/test_u0_telemetry_contract.py`` pins the literal key *list* (so a rename / removal of an existing
serialized field is caught), but nothing links the dataclass fields to the serializer — so a NEW field
added to the dataclass that someone forgets to add to ``to_dict()`` would silently never serialize, and
no test would fail.

This closes that gap: it asserts the serializer key-set is exactly the dataclass field-set plus the one
synthetic ``schema_version`` key. Adding a field to the record without wiring it into ``to_dict()`` (or
vice-versa) now fails here. NB: we deliberately keep BOTH this fields()-linkage test AND test_u0's literal
list — the literal pins names+order for the JSONL contract; this pins completeness against the dataclass.
"""

from __future__ import annotations

import dataclasses
import unittest

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord

# The single serializer key that is NOT a dataclass field (a ClassVar stamped on every record).
_SYNTHETIC_KEYS = {"schema_version"}


def _sample_record() -> GraspAttemptRecord:
    return GraspAttemptRecord.new(
        attempt_id="schema-evo-1",
        mode="dense_clutter",
        final_outcome="succeeded",
        timestamp=1.0,
    )


class RecordSchemaEvolutionTests(unittest.TestCase):
    def test_every_dataclass_field_is_serialized(self) -> None:
        field_names = {f.name for f in dataclasses.fields(GraspAttemptRecord)}
        keys = set(_sample_record().to_dict())
        missing = field_names - keys
        self.assertEqual(
            missing,
            set(),
            f"GraspAttemptRecord field(s) {sorted(missing)} are NOT emitted by to_dict() — "
            f"a new field was added to the dataclass without wiring it into the serializer.",
        )

    def test_no_stray_serializer_keys(self) -> None:
        field_names = {f.name for f in dataclasses.fields(GraspAttemptRecord)}
        keys = set(_sample_record().to_dict())
        stray = keys - field_names - _SYNTHETIC_KEYS
        self.assertEqual(
            stray,
            set(),
            f"to_dict() emits key(s) {sorted(stray)} that are neither a dataclass field nor the "
            f"synthetic schema_version — the serializer drifted from the record.",
        )

    def test_serializer_keyset_is_exactly_fields_plus_synthetic(self) -> None:
        field_names = {f.name for f in dataclasses.fields(GraspAttemptRecord)}
        keys = set(_sample_record().to_dict())
        self.assertEqual(keys, field_names | _SYNTHETIC_KEYS)


if __name__ == "__main__":
    unittest.main()
