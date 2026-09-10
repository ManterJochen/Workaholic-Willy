"""What the byte-identity family is allowed to skip, and what it may never skip.

MEASURED 2026-09-10, and it is why this file exists: ``tests/conftest.py`` held an allow-list of
27 node ids that skipped unless ``WILLY_DETERMINISM_NATIVE`` was set, and nothing in the repository
ever set it. ``.github/workflows/ci.yml`` does not set it; no runbook, script or Makefile sets it.
So 27 byte-identity claims skipped in every automated run since the list was written, and the only
number the suite ever printed about them was "skipped".

Opening the gate by hand on this box turned 27 skips into 19 failures. Three causes, and only one
of them is the float drift the allow-list was written for:

* 42 lines of 1240 across the four canonical packs differ in the last ULP of a ``random.gauss``
  draw (``_enrich_latency_pack`` goes through ``math.log`` / ``math.cos``, which are libm, which is
  per-platform). That is real and it is unfixable without re-blessing the packs.
* Five of the failures were CRLF: ``core.autocrlf=true`` checks the packs out with CRLF while the
  committed blob is LF, so every guard that hashes FILE BYTES saw a different file than the one in
  the manifest. ``sha256(worktree bytes)`` and ``sha256(worktree bytes with CRLF folded to LF)``
  were computed side by side against the manifest and the LF form matched all four packs exactly.
  Repairing it exposed a sixth, which had been passing only because two stale goldens agreed with
  each other about the CRLF reading.
* The rest were goldens their own generators can no longer produce: five prose fields in
  ``tests/data/replay/MANIFEST.json`` hand-edited out of step with the generator, both replay
  manifests still naming a ``backend.src`` import path this tree does not use, provenance hashes
  taken over the CRLF reading, a ``config_hash`` from an older recovery action space, and five
  artifacts spelling a connector their sources had stopped writing. Only sha256 fields were ever
  compared, and prose is not hashed.

The tests below are the guards that would have caught each of those from a normal ``pytest tests``
run, which is the property the allow-list destroyed.
"""

from __future__ import annotations

import unittest

from tests._determinism import (
    LF_LOCKED_GOLDEN_GLOBS,
    PLATFORM_FLOAT_LOCKED_NODEIDS,
    _drift_between_lines,
    classify_canonical_pack_drift,
    goldens_rewritten_during_session,
    iter_lf_locked_goldens,
    repo_root,
)


class LineEndingLockTests(unittest.TestCase):
    """The goldens whose FILE BYTES are hashed must be LF in the working tree.

    A byte-identity guard reads bytes. ``core.autocrlf=true`` rewrites those bytes on checkout
    unless the path carries ``-text``, so without the attribute the guard compares a Windows
    checkout against a Linux hash and fails for a reason that has nothing to do with determinism.
    """

    def test_hashed_goldens_have_no_crlf_in_the_worktree(self) -> None:
        offenders = [
            str(path.relative_to(repo_root())).replace("\\", "/")
            for path in iter_lf_locked_goldens()
            if b"\r\n" in path.read_bytes()
        ]
        self.assertEqual(
            offenders,
            [],
            msg=(
                "these goldens are hashed byte-for-byte and are CRLF in the working tree; "
                "the committed blob is LF, so every sha256 guard over them fails. Fix the "
                "checkout (the paths need `-text` in .gitattributes), not the hashes."
            ),
        )

    def test_gitattributes_pins_the_hashed_goldens(self) -> None:
        """The worktree can be repaired by hand; only the attribute keeps it repaired."""

        attributes = (repo_root() / ".gitattributes").read_text(encoding="utf-8")
        declared = {
            line.split()[0]
            for line in attributes.splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "-text" in line
        }
        missing = [g for g in LF_LOCKED_GOLDEN_GLOBS if g not in declared]
        self.assertEqual(
            missing,
            [],
            msg=(
                "a path whose bytes are hashed is not pinned with `-text` in .gitattributes; "
                "it will come back as CRLF on the next Windows checkout."
            ),
        )


class SkipListHonestyTests(unittest.TestCase):
    """Whatever stays platform-locked must be small.

    The other direction, that every entry still names a real file, class and test, is owned by
    ``TheDeterminismLockIsNotInertTests`` in ``tests/test_promotion_gate_cannot_be_talked_down.py``.
    That class is the one that caught the five renamed entries; it stays where it is rather than
    being copied here, because two guards over one list is how one of them rots unnoticed.
    """

    def test_the_lock_is_narrow(self) -> None:
        """Three entries, each measured unable to hold off-origin. Growth here needs evidence."""

        self.assertLessEqual(
            len(PLATFORM_FLOAT_LOCKED_NODEIDS),
            4,
            msg=(
                "the platform lock is growing again. Every entry costs a claim nobody tests on "
                "the boxes that run the suite; measure the failure before adding one."
            ),
        )


class DriftClassifierTests(unittest.TestCase):
    """The skip decision is measured per run, not assumed from the hostname."""

    def test_classifier_reports_a_verdict_for_this_box(self) -> None:
        verdict = classify_canonical_pack_drift()
        self.assertIn(verdict.kind, {"identical", "float_only", "structural"})

    def test_structural_drift_is_never_classified_as_float_only(self) -> None:
        """A changed non-float value must not buy a skip."""

        self.assertEqual(
            _drift_between_lines(
                '{"a":1.0,"outcome":"succeeded"}',
                '{"a":1.0,"outcome":"failed"}',
            ),
            "structural",
        )

    def test_last_ulp_float_drift_is_classified_as_float_only(self) -> None:
        self.assertEqual(
            _drift_between_lines(
                '{"a":27.840233759192085}',
                '{"a":27.84023375919208}',
            ),
            "float_only",
        )

    def test_a_wide_float_gap_is_structural(self) -> None:
        """"Float" is not a licence. Only re-formatting of the same quantity is tolerated."""

        self.assertEqual(
            _drift_between_lines('{"a":27.84}', '{"a":31.02}'),
            "structural",
        )

    def test_reordered_keys_are_structural_not_float(self) -> None:
        """Same values, different bytes: serialization moved, and no platform is to blame."""

        self.assertEqual(
            _drift_between_lines('{"a":1.0,"b":2.0}', '{"b":2.0,"a":1.0}'),
            "structural",
        )

    def test_a_dropped_field_is_structural(self) -> None:
        self.assertEqual(
            _drift_between_lines('{"a":1.0,"b":2.0}', '{"a":1.0}'),
            "structural",
        )


class TreeCleanlinessTests(unittest.TestCase):
    """No test in this family may leave a tracked golden rewritten.

    MEASURED: running the old allow-list with the gate open left five tracked files modified
    (``tests/data/replay/MANIFEST.json`` plus the four packs) and, worse, the tests that ran after
    it compared against the rewritten files instead of the committed ones. Four failures in that
    run were the rewrite, not the code.
    """

    def test_no_replay_golden_was_rewritten_by_this_session(self) -> None:
        """Compared against a start-of-session fingerprint, not against git.

        Against git this would also fail on a deliberate re-bless someone is in the middle of, and
        a guard that cries at honest work gets deleted. Against the session snapshot it says one
        thing only: a test wrote here.
        """

        self.assertEqual(
            goldens_rewritten_during_session(),
            [],
            msg=(
                "a test rewrote a committed replay golden mid-run. Every test that ran after it "
                "compared against the rewrite instead of the committed bytes. Regeneration "
                "belongs in a temporary directory."
            ),
        )

    def test_the_cleanliness_check_can_actually_fire(self) -> None:
        """A negative control. Without it, "no golden was rewritten" and "nothing is watched" read alike.

        The snapshot is poisoned with a digest the file cannot have, and the check must name that
        file. Nothing on disk is touched.
        """

        from tests import _determinism

        watched = sorted(_determinism._SESSION_GOLDEN_DIGESTS)
        self.assertTrue(watched, "the session snapshot is empty; nothing is being watched")
        victim = watched[0]
        original = _determinism._SESSION_GOLDEN_DIGESTS[victim]
        _determinism._SESSION_GOLDEN_DIGESTS[victim] = "0" * 64
        try:
            self.assertIn(victim, goldens_rewritten_during_session())
        finally:
            _determinism._SESSION_GOLDEN_DIGESTS[victim] = original
        self.assertEqual(goldens_rewritten_during_session(), [])


class TheStandDownAnnouncesItselfTests(unittest.TestCase):
    """A skip whose reason nobody prints is the old defect in a smaller costume.

    MEASURED 2026-09-10 in this tree: ``pytest`` over the former 27 prints "25 passed, 2 skipped",
    and before ``-rs`` not one word about why. The verdict that decides it (42 differing lines of
    1240, every one a re-spelled ``random.gauss`` draw) reaches the terminal only with ``-rs``,
    and no CI step passed it. One line per skipped test is what a stand-down costs to state its
    own measurement.
    """

    def test_pytest_prints_skip_reasons_by_default(self) -> None:
        import tomllib

        addopts = tomllib.loads(
            (repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        )["tool"]["pytest"]["ini_options"]["addopts"]
        self.assertIn(
            "-rs",
            addopts.split(),
            msg=(
                "pyproject addopts does not print skip reasons, so a measured stand-down and a "
                "claim nobody ever wrote a test for look the same in the summary: a number."
            ),
        )


class TheRetiredLockLeavesNoStaleProseTests(unittest.TestCase):
    """A sentence about the gate is a claim about how this suite runs, and it needs an edge.

    MEASURED 2026-09-10, once the 27-node allow-list was retired: four test modules still told the
    reader that a byte-identity test "skips off the canonical origin". Two of them named the
    deleted ``conftest`` constant (``test_promotion_gate_cannot_be_talked_down``,
    ``test_u12_docs_and_soak_gate``) and two named the environment variable from outside the files
    that implement it (``test_r3_linucb_precise_float``, ``test_r3_shadow_seam_golden``). All four
    of those tests were running, and passing, while the sentences said otherwise. Prose has no
    edge to a change made underneath it; these two assertions are the edge for the two spellings
    that can be matched exactly.
    """

    #: Spelled in two pieces on purpose: written out, this module would match its own scan.
    _RETIRED_SYMBOL = "_DETERMINISM_NATIVE" + "_NODEIDS"

    #: The only modules that may name the override: the one that implements the stand-down, the one
    #: that measures the drift, and this one, which guards both. Elsewhere the name describes a
    #: mechanism the file does not own, and every file that tried described it wrongly.
    _MAY_NAME_THE_OVERRIDE = frozenset({
        "tests/conftest.py",
        "tests/_determinism.py",
        "tests/test_determinism_lock.py",
    })

    def _sources(self) -> list[tuple[str, str]]:
        root = repo_root()
        return [
            (str(path.relative_to(root)).replace("\\", "/"), path.read_text(encoding="utf-8"))
            for path in sorted((root / "tests").rglob("*.py"))
        ]

    def test_the_deleted_constant_is_named_nowhere(self) -> None:
        offenders = sorted(
            rel for rel, text in self._sources() if self._RETIRED_SYMBOL in text
        )
        self.assertEqual(
            offenders,
            [],
            msg=(
                f"a test module still points its reader at {self._RETIRED_SYMBOL}, which does not "
                "exist. What is left of the lock lives in tests/_determinism.py."
            ),
        )

    def test_only_the_gate_itself_names_the_override(self) -> None:
        offenders = sorted(
            rel
            for rel, text in self._sources()
            if "WILLY_DETERMINISM_NATIVE" in text and rel not in self._MAY_NAME_THE_OVERRIDE
        )
        self.assertEqual(
            offenders,
            [],
            msg=(
                "a test module describes the determinism override from outside the files that "
                "implement it. Every such sentence measured on 2026-09-10 said the byte-identity "
                "tests skip off one origin platform; they skipped on all of them. A new mention is "
                "fine, but add the file to _MAY_NAME_THE_OVERRIDE deliberately."
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
