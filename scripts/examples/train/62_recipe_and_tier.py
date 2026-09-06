"""62: `--recipe` and `--tier`, the two knobs, and exactly what each one resolves to.

    python scripts/examples/train/62_recipe_and_tier.py
    python scripts/examples/train/62_recipe_and_tier.py --scenes 8000
    python scripts/examples/train/62_recipe_and_tier.py --recipe v1 --tier smoke

The decision: how long to train, and under whose settings. It lives in two arguments,
`GeneratorTraining.from_recipe(recipe=..., tier=...)`, and nowhere in any YAML file. Nothing is
trained here. Every combination is resolved and priced, and the run that would follow is described.

A recipe is a named bundle of settings, frozen once it ships, written into the artifact so a model
can say which bundle produced it. A tier is a compute budget applied on top. The tier wins where
they overlap, and it has to: `smoke` sets `refit=False` precisely so a run that only proves the
chain closes does not pay for a second training pass.

The combination worth knowing before you pick one, because the obvious reading is wrong:
`--tier full` on its own does not turn the refit on. The refit is a recipe setting, so a full tier
with no recipe runs one pass at thirty-six epochs, and `--recipe v1 --tier full` runs two. That is
the difference between an afternoon and an evening, and neither flag says so by name.

Both names fail closed. An unknown recipe or tier refuses and lists what exists rather than falling
back to the defaults, because a customer who typed `v2` before it exists would otherwise get `v1`'s
behaviour under `v2`'s name, which is the failure a version string is there to prevent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  every combination resolved and was priced
  1  a combination that should resolve did not
  2  the recipe or tier you named does not exist; the message lists the ones that do
"""

#: The combinations printed side by side. Naming them here rather than looping over the two
#: vocabularies is deliberate: the pairs are what a reader has to choose between, and the empty
#: pair is one of them. A product of the two dicts would also print pairs nobody would run.
_COMBINATIONS: Final[tuple[tuple[str | None, str | None], ...]] = (
    (None, None),
    (None, "smoke"),
    (None, "full"),
    ("v1", None),
    ("v1", "smoke"),
    ("v1", "full"),
)

#: Corpus size the training line is priced over when the caller names none. The bill scales with it,
#: so a number has to be assumed; the calculator's own warning says the assumption is linear.
_PRICED_SCENES: Final[int] = 2000


def _passes(plan: Any) -> int:
    """Full passes over the corpus this plan pays for.

    `run_folds` is how many folds are actually fitted, and the refit is one more pass over every
    unit afterwards. The training bill is per pass, so this is the multiplier that separates two
    schedules with the same epoch count.
    """
    return int(plan.run_folds) + (1 if plan.refit else 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=_PRICED_SCENES,
                        help="corpus size to price the training line over. Default "
                             f"{_PRICED_SCENES}")
    parser.add_argument("--recipe", default=None,
                        help="resolve this one name as well, and refuse if it does not exist")
    parser.add_argument("--tier", default=None, help="the same, for a tier")
    args = parser.parse_args(argv)

    from src.robot.grasping.deep.train.recipes import RECIPES, TIERS, describe, settings

    with Example("62 recipe and tier", "what the two knobs resolve to", hardware=False) as run:
        with run.step("the vocabulary that ships") as report:
            report(f"{len(RECIPES)} recipe(s): {', '.join(sorted(RECIPES))}; "
                   f"{len(TIERS)} tier(s): {', '.join(sorted(TIERS))}")
        run.note("")
        run.note("what a tier is for, in its own words:")
        for name in sorted(TIERS):
            why = str(TIERS[name].get("why", ""))
            run.note(f"  {name:6s} {why}")
        run.note("")

        # The caller's own names, resolved before anything else. `settings` refuses an unknown name,
        # and refusing here rather than six steps later is the whole point of a fail-closed lookup.
        if args.recipe or args.tier:
            try:
                settings(args.recipe, args.tier)
            except ValueError as error:
                return not_ready(str(error),
                                 "pass one of the names that message lists, or leave both off to "
                                 "resolve the shipped defaults")

        with run.step("resolve every combination, without torch") as report:
            # `recipes.settings` is a dict merge over two literal dicts and imports nothing heavy.
            # `build_plan` below needs the real defaults and the enum vocabularies, which live
            # behind torch, so the cheap answer is available long before the expensive one.
            resolved = {pair: settings(*pair) for pair in _COMBINATIONS}
            report(f"{len(resolved)} combination(s) merged from two literal dicts")
        run.note("")
        run.note("  the merged bundle, which is what a run is handed:")
        for pair in _COMBINATIONS:
            label = describe(*pair) if any(pair) else "(no recipe, no tier): (nothing set)"
            run.note(f"    {label}")
        run.note("")
        run.note("  the tier wins where the two overlap. `v1` asks for the refit and `smoke` turns")
        run.note("  it off, so `recipe v1 tier smoke` shows refit=False above. A recipe a tier")
        run.note("  could not narrow would make the fast tier slower than the run it precedes.")
        run.note("")

        # Everything below costs torch, because a plan is built from `SetTrainingPlan`'s real
        # defaults and validated against the enum vocabularies, and both live behind it.
        from datagen.cost import estimate
        from src.robot.grasping.deep.train.plan import PlanOverrides, build_plan

        with run.step("build the plan each one produces") as report:
            plans = {pair: build_plan(recipe=pair[0], tier=pair[1])[0] for pair in _COMBINATIONS}
            report(f"{len(plans)} plan(s) assembled from SetTrainingPlan's defaults")
        run.note("")
        run.note(f"  {'recipe':8s} {'tier':6s} {'epochs':>7s} {'units':>7s} {'passes':>7s} "
                 f"{'train h':>8s}   what the schedule is")
        for pair, plan in plans.items():
            # Priced through the same calculator a datagen bill uses, with this plan's own fold
            # count and refit flag rather than the calculator's defaults. Its `refit=True` default
            # is the trap: a bill taken without passing these would price two passes for a schedule
            # that runs one, or one for a schedule that runs two.
            bill = estimate(args.scenes, engine="none", label=False, corpus=False,
                            epochs=plan.epochs, folds=plan.run_folds, refit=plan.refit)
            hours = next(stage.hours for stage in bill.stages if stage.name == "train")
            units = "all" if plan.train_units is None else str(plan.train_units)
            run.note(f"  {pair[0] or 'none':8s} {pair[1] or 'none':6s} {plan.epochs:7d} "
                     f"{units:>7s} "
                     f"{_passes(plan):7d} {hours:8.2f}   "
                     f"{'refit on, two passes' if plan.refit else 'one pass'}")
        run.note("")
        run.note(f"  the hours are over {args.scenes} scenes and scale with that count. The")
        run.note("  calculator states the assumption itself: the training line is linear in corpus")
        run.note("  size, which is an assumption rather than a measurement.")
        run.note("")
        run.note("WARN `--tier full` alone does not turn the refit on: thirty-six epochs and")
        run.note("  one pass. The refit is a recipe setting, so two passes need `--recipe v1`.")
        run.note("")

        with run.step("what the tier does to the run itself") as report:
            smoke = plans[("v1", "smoke")]
            full = plans[("v1", "full")]
            report(f"smoke {smoke.epochs} epoch(s) over {smoke.train_units} unit(s) against "
                   f"full {full.epochs} over {full.train_units or 'all'}")
        run.note("")
        run.note("  smoke narrows three settings at once, not one: epochs, the unit cap and the")
        run.note("  refit. It is a shorter run over a smaller slice with the second pass off,")
        run.note("  so it proves the chain closes on your corpus and your box and says nothing")
        run.note("  whatever about grasp quality. The tier is recorded in the plan, the run")
        run.note("  report and the model card, and `deep inspect` prints a loud line for it, so")
        run.note("  a two-epoch artifact is never indistinguishable from one that took hours.")
        run.note("")
        run.note("  thirty-six is a floor and not a ceiling. If your own curve is still climbing")
        run.note("  there, pass more epochs; 63_read_the_report.py prints the verdict that decides")
        run.note("  it, on your data rather than on anybody else's.")
        run.note("")

        with run.step("an explicit choice outranks both") as report:
            # A recipe fills only what the caller left UNSET, and `applied["overridden"]` is the
            # record of what it therefore declined to set. Deciding that from `sys.argv` would be
            # wrong in a hosted process, where the arguments belong to the host and `--epochs` never
            # appears; `PlanOverrides` states it as data, so the rule holds in both worlds.
            plan, applied = build_plan(recipe="v1", tier="full",
                                       overrides=PlanOverrides(epochs=7, refit=False))
            report(f"{plan.epochs} epoch(s), refit {'on' if plan.refit else 'off'}, "
                   f"overridden {applied['overridden']}")
        run.note("")
        run.note("  `refit=False` is passed as a value, not left out. Turning a recipe's refit off")
        run.note("  is a choice, and it has to stay distinguishable from saying nothing at all,")
        run.note("  which is what the UNSET sentinel in `PlanOverrides` is for. Comparing against")
        run.note("  the default instead would make typing the default indistinguishable from")
        run.note("  silence, and the recipe would then overwrite a value the caller chose.")
        run.note("")

        with run.step("an unknown name refuses, and says what exists") as report:
            try:
                settings("v2")
            except ValueError as error:
                report(str(error).split(";")[0])
            else:
                run.finding("an unknown name refuses", "settings('v2') returned instead of raising")
        run.note("")
        run.note("  the same refusal reaches a caller through `GeneratorTraining.from_recipe`, at")
        run.note("  construction, before the corpus walk and the probes. A misspelled name costs a")
        run.note("  second rather than the first hour of a run.")
        run.note("")
        run.note("what to pick: `--tier smoke` first, always, on a corpus you have not trained on")
        run.note("  before. It is minutes, and it fails on a broken corpus instead of failing")
        run.note("  after a night.")
        run.note("  Then `--recipe v1 --tier full` for the model you deploy: the recipe is what")
        run.note("  makes the shipped weights see every part you own rather than one fold's share.")
        run.note("")
        run.note("the settings a recipe deliberately does not carry are the unsettled ones.")
        run.note("  `slots`, `slot_mixing`, `axis_mode` and the crop radius reach a run through")
        run.note("  `PlanOverrides` from code, because pinning an unmeasured setting inside a")
        run.note("  version number hands a customer a coin flip wearing a version number.")
        run.note("")
        run.note("Next: 60_your_own_corpus.py to produce a corpus, then 63_read_the_report.py.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
