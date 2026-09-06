"""Determinism guard for the canonical replay packs + their MANIFEST.

The canonical packs are the byte-stable, deterministically-generated authority that
all KPI / soak / baseline / SLO gating replays against. Previously nothing verified
that the committed bytes still reproduce from the generator — it rested on developer
discipline (regenerate via `python -m backend.src.robot.grasping.replay
--regenerate-canonical` + manual commit). These tests close that gap so CI fails the
moment a generator change, a dependency bump, or a hand-edit drifts the packs.
"""

from __future__ import annotations

import json
import unittest

from src.robot.grasping.replay.canonical_datasets import (
    CANONICAL_PACKS,
    build_manifest,
    render_pack_jsonl,
    repo_root_from_module,
    sha256_hex,
)

_MANIFEST_REL = "tests/data/replay/MANIFEST.json"


class CanonicalPackDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = repo_root_from_module()

    def test_each_pack_regenerates_to_committed_bytes(self) -> None:
        for pack in CANONICAL_PACKS:
            with self.subTest(pack=pack.name):
                on_disk = (self.root / pack.relative_path).read_text(encoding="utf-8")
                self.assertEqual(
                    render_pack_jsonl(pack),
                    on_disk,
                    f"{pack.name}: render_pack_jsonl no longer reproduces the committed "
                    f"bytes. Regenerate with `--regenerate-canonical` and re-commit, or "
                    f"investigate a determinism regression.",
                )

    def test_manifest_matches_on_disk_packs(self) -> None:
        committed = json.loads(
            (self.root / _MANIFEST_REL).read_text(encoding="utf-8")
        )
        self.assertEqual(
            committed,
            build_manifest(self.root),
            f"{_MANIFEST_REL} is out of sync with the canonical packs on disk.",
        )

    def test_manifest_sha256_matches_pack_bytes(self) -> None:
        committed = json.loads(
            (self.root / _MANIFEST_REL).read_text(encoding="utf-8")
        )
        by_name = {entry["name"]: entry for entry in committed["packs"]}
        for pack in CANONICAL_PACKS:
            with self.subTest(pack=pack.name):
                raw = (self.root / pack.relative_path).read_bytes()
                self.assertEqual(by_name[pack.name]["sha256"], sha256_hex(raw))


if __name__ == "__main__":
    unittest.main()
