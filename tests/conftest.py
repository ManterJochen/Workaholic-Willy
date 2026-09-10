"""Top-level pytest configuration for the Workaholic-Willy suite.

⛔ **THIS FILE USED TO SKIP 27 TESTS IN EVERY RUN THAT HAS EVER HAPPENED.** It held an allow-list
of node ids that skipped unless ``WILLY_DETERMINISM_NATIVE`` was set, and it described that
variable as belonging to "the canonical determinism CI job". MEASURED 2026-09-10: no such job
exists. ``.github/workflows/ci.yml`` never sets the variable, no script or runbook sets it, and
nothing in the tree ever has. Twenty-seven byte-identity claims were therefore never once checked.
All 27 did at least resolve here: file, class and method were confirmed one by one on 2026-09-10,
so this tree did not also carry the stale-entry hole that a 2026-09-05 pass had already closed.
Resolving and being checked are different things, and only the first was ever true of them.

Opening the gate by hand turned 27 skips into 19 failures, in three groups:

* CRLF. ``core.autocrlf=true`` checked ``tests/data/replay/**`` out with carriage returns while
  the committed blob is LF, so every guard hashing FILE BYTES compared a different file than the
  one the manifest describes. sha256 of the working-tree bytes and sha256 of the same bytes with
  CRLF folded to LF were computed against the manifest: the LF form matched all four packs, the
  CRLF form matched none. Pinning the paths ``-text`` cleared 5 of the 19 and none of them was
  float drift. It also EXPOSED one more, which had been passing only because two stale goldens
  agreed with each other about a CRLF reading; that is the shape of the whole problem.
* Stale goldens. Nine committed artifacts could not be produced by their own generators. Six
  under ``docs/baselines/`` carried provenance hashes taken from a CRLF reading of the packs, so
  they disagreed with ``tests/data/replay/MANIFEST.json`` about the sha256 of the same four
  files; one also recorded a ``config_hash`` from an older recovery action space; two
  ``coverage_warning`` strings in the OPE report had lost a phase prefix their generator no
  longer emits; five artifacts still spelled a connector the sources had stopped writing; the
  replay manifest had five prose fields its generator could not produce; and both replay
  manifests still named the ``backend.src`` import path this tree does not use. All regenerated
  with this tree's own commands.
* Real float drift: 42 lines of 1240 across the canonical packs differ in the last ULP of a
  ``random.gauss`` draw. libm, per-platform, unfixable without re-blessing the packs.

Only the third group survives, and only for assertions that compare bytes rather than values.
Those are named in :data:`tests._determinism.PLATFORM_FLOAT_LOCKED_NODEIDS` and they are skipped
only when a regeneration on THIS box is measured to drift in float formatting alone. A box that
reproduces the packs runs them; a box where anything structural moved runs them and fails.
MEASURED here afterwards: 25 of the 27 run and pass, 2 stand down.
``WILLY_DETERMINISM_NATIVE=1`` forces them to run regardless, which is what to set when
re-blessing.

(The integration ``isaac`` marker is gated separately in ``tests/integration/conftest.py``.)
"""

from __future__ import annotations

import os

import pytest

from tests._determinism import (
    DRIFT_PROBES,
    PLATFORM_FLOAT_LOCKED_NODEIDS,
    snapshot_lf_locked_goldens,
)


def pytest_configure(config: pytest.Config) -> None:
    """Fingerprint the committed replay goldens before a single test has run.

    ``TreeCleanlinessTests`` compares against this, so it detects a test rewriting a golden DURING
    the run and stays quiet about an edit someone made before it.
    """

    del config
    snapshot_lf_locked_goldens()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Stand a locked byte-comparison test down, and only for a drift measured in this run.

    Three node ids are locked; how many actually stand down is decided here, per run, per probe.
    """

    if os.environ.get("WILLY_DETERMINISM_NATIVE"):
        return
    locked = [
        (item, PLATFORM_FLOAT_LOCKED_NODEIDS[nodeid])
        for item in items
        if (nodeid := item.nodeid.replace("\\", "/")) in PLATFORM_FLOAT_LOCKED_NODEIDS
    ]
    if not locked:
        return
    # Probes are measured lazily and cached, and only for the tests actually collected: rendering
    # the packs or retraining the ranker costs seconds and every other run should not pay it.
    for item, probe_name in locked:
        verdict = DRIFT_PROBES[probe_name]()
        if verdict.kind != "float_only":
            # identical -> the test can hold here, so run it. structural -> something real moved
            # and it MUST run, so it can say what.
            continue
        item.add_marker(
            pytest.mark.skip(
                reason=(
                    f"byte-identity stood down: {verdict.render()}. The values are right and the "
                    "decimal spelling is not, which no assertion on bytes can survive. Set "
                    "WILLY_DETERMINISM_NATIVE=1 to run it anyway (do that when re-blessing). "
                    "See tests/_determinism.py."
                )
            )
        )
