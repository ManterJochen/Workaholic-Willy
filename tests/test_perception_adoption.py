"""Who actually builds perception through the factory — asserted, so the gap cannot be forgotten.

MEASURED 2026-08-10, and it is the reason this file exists: ``build_perception()`` had **zero callers**.
Every runner constructed GroundingDINO and SAM2 itself, which meant the VLM route and the prompt router
were unreachable from any of them -- a cell could set ``pipeline.zero_shot.backend: vlm`` and nothing
would happen.

That is not fixed by one commit, and it must not be fixed by a blanket sweep either: ``run_dense_pick``
deliberately grounds with a BIGGER GroundingDINO than the config default (env-overridable, isolated
from M2 on purpose, and measured), so a blanket ``build_perception(cfg.models)`` would silently revert
a measured decision. Each runner converts on its own, on-box, with its own measurement.

So this file is a **ledger, not a gate**. It records which construction sites are converted and which
are not, and fails when reality drifts from the ledger in either direction -- a new hand-rolled site
appearing, or a converted one silently reverting. Moving a runner from ``PENDING`` to ``ADOPTED`` is
part of the commit that converts it, next to the measurement that justifies it.

2026-10-17 (the owner's cleanup): GroundingDINO, OneFormer and the router left, and with them the dense
path's bigger checkpoint and every reason a runner had to build its own models. Every runner and every
production site builds through the factory now, so ``PENDING`` is empty; the runners converted that day
were not measured on-box, which the sim README says beside each gate.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNERS = REPO / "src" / "willy_sim"

#: ⭑⭑ **THE LEDGER LOOKED ONLY AT SIM RUNNERS, AND THE SITE THAT MATTERED WAS NOT ONE.**
#: `build_real_components` -- the path a PHYSICAL cell takes, and the path the operator console
#: reaches through `api/cell.py` -- built its detector and segmenter by hand, exactly like the
#: runners this file was written to count, and sat outside its glob. So the ledger read "2 adopted,
#: 4 pending" while the one construction site with a customer on the other end was invisible.
#:
#: ⚠ NAMED PATHS RATHER THAN A WIDER GLOB, DELIBERATELY. A blanket sweep of `backend/` would
#: match `factory.py` itself, which constructs those four classes because that is its job. What this
#: ledger measures is sites that build a stack FOR A CELL, and that is a judgement, so it is written
#: down as one. The list fails in both directions like everything else here: a path that stops
#: existing fails `test_every_named_path_exists`.
PRODUCTION_SITES: dict[str, str] = {
    # 2026-09-05: converged onto `PerceptionSpec.from_config(app_cfg.models).build()`. What changed
    # is that `models.pipeline` reaches hardware at all.
    "src/robot/execution/autonomous_grasp/cells.py": "ADOPTED",
    # 2026-10-17: the exerciser stood on the component doors (`build_object_detector`,
    # `build_segmenter`) to prove a camera and two models apart from a cell. Those doors left with the
    # legacy keys they read, so it proves the very stack the cell builds, before a cell exists.
    "src/robot/perception/__main__.py": "ADOPTED",
}

#: Runners that build their perception stack through the factory -- so config, and therefore the VLM
#: route and the router, actually reach them.
ADOPTED: frozenset[str] = frozenset({
    # 2026-08-10, MEASURED: M2 10/10 (lifts 98.6-99.6 mm) after switching to build_perception(). That
    # number is the acceptance criterion, not a formality -- the old path called `detect` (the single
    # best match) and a backend runs `detect_all`. Those disagree whenever a scene holds more than one
    # candidate; see test_isaac_source_selection.py for the rule that reconciles them, and note that
    # M2 could not have caught it (one object).
    "run_m2_pick.py",
    # NB `run_suction_pick.py` was here (2026-08-11, measured: suction pick OK, seal 1.000, object rose
    # 100.0/100 mm) and is now in NEITHER list. Its factory call lived in the analytical-vs-learned
    # cross-check, which was deleted with the learned suction backend for licensing on the same day.
    # Every remaining path in that runner uses GroundTruthPerceptionSource, so it builds no perception
    # stack at all -- nothing to adopt, and nothing hand-rolled to owe. Recorded rather than silently
    # dropped, because "it used to be adopted" is exactly the kind of fact a ledger exists to keep.
    # 2026-08-11, MEASURED (P5): born on the factory rather than converted to it -- it exists to
    # compare routes, so it could not have been written any other way. VLM route 10/10 on both "the
    # red cube" and "der rote Wuerfel"; the phrase route 0/10 on both, and on the English one it
    # lifted the BLUE cube 10/10 while reporting outcome=succeeded.
    "run_attribute_pick.py",
    # 2026-10-17, NOT MEASURED ON-BOX: converted when GroundingDINO left. The dense path's bigger
    # checkpoint (`WILLY_YCB_DETECTOR`) left with it, and the fused runner's clutter pass takes the
    # backend's own detector and segmenter.
    "run_dense_pick.py",
    "run_fused_pick.py",
})

#: Runners that still construct a detector/segmenter directly. Each needs its own on-box run before it
#: moves up. Listed by name so the size of the debt is a number, not a feeling.
#: NB this measures runners that CONSTRUCT models, which is the debt that blocks config from reaching
#: them. A runner that merely hand-rolls the detect->segment chain on models handed to it (e.g.
#: run_pile_baseline) is a smaller, separate debt: it cannot ignore config, only duplicate the loop.
#: Empty since 2026-10-17. `run_bin_clearing_demo.py` is in neither list: its language mode asks the process's
#: one VLM for boxes (`shared_vlm().grounder_for`) and cuts no mask, so it builds no perception stack at all.
PENDING: frozenset[str] = frozenset()

#: The classes a site would build itself, the VLM's grounder among them: one built by hand is a second copy of
#: its weights beside the holder's.
_DIRECT_CONSTRUCTION = re.compile(r"\b(Sam2Segmenter|RtDetrObjectDetector|Qwen3VLGrounder)\s*\(")


def _runners_constructing_models_directly() -> set[str]:
    found = set()
    for path in sorted(RUNNERS.glob("run_*.py")):
        if _DIRECT_CONSTRUCTION.search(path.read_text(encoding="utf-8", errors="replace")):
            found.add(path.name)
    return found


class AdoptionLedgerTests(unittest.TestCase):
    def test_the_ledger_matches_reality(self) -> None:
        """Fails in BOTH directions: a new hand-rolled site, or a silent revert of a converted one."""
        actual = _runners_constructing_models_directly()
        self.assertEqual(
            actual, set(PENDING),
            "the perception-adoption ledger is stale.\n"
            f"  still hand-rolling: {sorted(actual)}\n"
            f"  ledger says PENDING: {sorted(PENDING)}\n"
            "If you converted a runner, move it from PENDING to ADOPTED in this file, in the same "
            "commit, next to the on-box measurement that justifies it. If you added a new runner that "
            "builds models directly, either use build_perception() or add it to PENDING with a reason.",
        )

    def test_a_runner_is_never_in_both_lists(self) -> None:
        self.assertEqual(ADOPTED & PENDING, set())

    def test_adopted_runners_really_do_call_the_factory(self) -> None:
        for name in sorted(ADOPTED):
            with self.subTest(runner=name):
                source = (RUNNERS / name).read_text(encoding="utf-8", errors="replace")
                self.assertIn(
                    "build_perception", source,
                    f"{name} is listed as ADOPTED but does not call build_perception()",
                )

    def test_the_production_sites_match_what_they_claim(self) -> None:
        """⭑ THE HALF THE GLOB COULD NOT SEE. An ADOPTED production site must not construct a
        model class by hand, and must go through PerceptionSpec."""
        for rel, state in sorted(PRODUCTION_SITES.items()):
            with self.subTest(site=rel):
                source = (REPO / rel).read_text(encoding="utf-8", errors="replace")
                self.assertEqual("ADOPTED", state, "every production site builds through the factory")
                self.assertFalse(
                    _DIRECT_CONSTRUCTION.search(source),
                    f"{rel} is listed as ADOPTED but still constructs a model class directly",
                )
                self.assertIn(
                    "PerceptionSpec", source,
                    f"{rel} is listed as ADOPTED but does not go through PerceptionSpec",
                )

    def test_every_named_path_exists(self) -> None:
        """A path that has been renamed would silently measure nothing, and pass forever."""
        for rel in sorted(PRODUCTION_SITES):
            with self.subTest(site=rel):
                self.assertTrue((REPO / rel).is_file(), f"{rel} is in the ledger and not on disk")

    def test_the_factory_is_reachable_at_all(self) -> None:
        """A guard against the ledger quietly measuring nothing, which would pass forever."""
        from src.models.factory import build_perception

        self.assertTrue(callable(build_perception))
        self.assertTrue(_DIRECT_CONSTRUCTION.search("Sam2Segmenter("), "the detector regex is broken")


if __name__ == "__main__":
    unittest.main()
