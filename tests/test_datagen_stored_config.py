"""Reading the config a dataset RECORDS, as against validating one a person wrote.

⛔⛔ **THE DEFECT THIS GUARDS, AND IT WAS INVISIBLE UNTIL SOMETHING MOVED.** Every dataset's
`provenance.json` embeds the whole `DatagenConfig` it was built with, and three places rebuild it to
re-derive the asset manifest and check the assets have not drifted: `eval/ladder.py`,
`grasps/labels.py` and `prompts/build.py`. All three did `DatagenConfig(**stamp["config"])` against
the CURRENT schema, and `StrictModel` sets `extra="forbid"`.

So every schema change retroactively invalidated every dataset ever built, and nothing said so until
a schema change happened.

MEASURED on 2026-09-04: removing two keys that were read by nothing (`output.jsonl`, which described
behaviour it did not control, and `output.coco`, which promised a COCO exporter datagen has never
had) broke eight tests, because `logs/p5/datasets/v1_proof/provenance.json` was written in August
and carries both.

⭐ The keys were the trigger. The defect is that a record of the past was parsed by the rules of the
present, and it would have fired on the next schema change either way.
"""

from __future__ import annotations

import unittest
import warnings

from datagen.config import DatagenConfig, StoredConfigDrift


def _stamp(**output: object) -> dict:
    """A stored config shaped like the real one: nested, and mostly defaults."""
    return {"output": {"root": "data/somewhere", **output}}


class ARetiredKeyIsToleratedAndReportedTests(unittest.TestCase):

    def test_a_key_this_build_no_longer_has_is_dropped(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoredConfigDrift)
            config = DatagenConfig.from_stamp(_stamp(coco=True, jsonl=True))
        self.assertEqual("data/somewhere", config.output.root)

    def test_the_drop_is_REPORTED_rather_than_silent(self) -> None:
        """⚠ `extra="ignore"` WOULD HAVE BEEN THE WRONG FIX. It makes a typo in a live config and a
        key retired three months ago look identical, and the first of those is a defect the strict
        model exists to catch. What is dropped is named, so a reader can tell the two apart."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DatagenConfig.from_stamp(_stamp(coco=True, jsonl=True))
        drift = [w for w in caught if issubclass(w.category, StoredConfigDrift)]
        self.assertEqual(1, len(drift), "the drop was silent")
        message = str(drift[0].message)
        self.assertIn("output.coco", message)
        self.assertIn("output.jsonl", message)

    def test_it_has_its_OWN_warning_category(self) -> None:
        """So a caller can silence exactly this and nothing else. A bare UserWarning would force
        them to silence every warning the package might ever raise."""
        self.assertTrue(issubclass(StoredConfigDrift, UserWarning))
        self.assertIsNot(StoredConfigDrift, UserWarning)

    def test_a_clean_stamp_warns_about_nothing(self) -> None:
        """The control. Without it the test above passes for an implementation that always warns."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DatagenConfig.from_stamp(_stamp())
        self.assertEqual([], [w for w in caught if issubclass(w.category, StoredConfigDrift)])


class RealIncompatibilitiesStillRefuseTests(unittest.TestCase):
    """⛔ TOLERANT IS NOT PERMISSIVE. A retired key is a difference between the record and this
    build. A key whose TYPE changed is a difference the rebuild cannot paper over, and reading it as
    if it were fine would hand the manifest check a config that means something else."""

    def test_a_changed_type_is_refused(self) -> None:
        with self.assertRaises(Exception) as caught:
            DatagenConfig.from_stamp({"output": {"root": 17}})
        self.assertIn("root", str(caught.exception))

    def test_an_unknown_key_at_the_TOP_level_is_dropped_too(self) -> None:
        """The prune is recursive, so a retired top-level BLOCK is tolerated the same way a retired
        leaf is. Both are things this build no longer has."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DatagenConfig.from_stamp({"output": {"root": "x"}, "retired_block": {"a": 1}})
        self.assertIn("retired_block", str(caught[0].message))


class TheStrictPathIsUnCHANGEDTests(unittest.TestCase):
    """⚠ THE TOLERANCE IS FOR STORED CONFIGS ONLY. A config a person wrote is still validated
    strictly, because there an unknown key is a typo and refusing it is the whole point."""

    def test_a_live_config_still_forbids_an_unknown_key(self) -> None:
        with self.assertRaises(Exception):
            DatagenConfig(**{"output": {"root": "x", "coco": True}})

    def test_from_stamp_and_the_constructor_agree_on_a_clean_input(self) -> None:
        payload = _stamp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StoredConfigDrift)
            lenient = DatagenConfig.from_stamp(payload)
        strict = DatagenConfig(**payload)
        self.assertEqual(strict.model_dump(), lenient.model_dump())


class TheShippedDatasetStillRebuildsTests(unittest.TestCase):
    """The regression in its original form: the August dataset, read by today's schema."""

    def test_the_v1_proof_provenance_rebuilds(self) -> None:
        import json
        from pathlib import Path

        stamp_path = (Path(__file__).resolve().parents[1]
                      / "logs/p5/datasets/v1_proof/provenance.json")
        if not stamp_path.is_file():
            self.skipTest("the v1_proof dataset is not on this machine")
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))["config"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = DatagenConfig.from_stamp(stored)
        self.assertTrue(config.output.root)
        drift = [str(w.message) for w in caught if issubclass(w.category, StoredConfigDrift)]
        self.assertEqual(1, len(drift), "the August stamp should report its two retired keys")


if __name__ == "__main__":
    unittest.main()
