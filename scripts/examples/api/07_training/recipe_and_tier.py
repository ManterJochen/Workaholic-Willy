"""What `recipe` and `tier` resolve to, and what each combination costs. Nothing is trained here.

A recipe is a frozen bundle of settings written into the artifact; a tier is a compute budget laid
on top, and the tier wins where the two overlap. An explicit override outranks both of them, and an
unknown name refuses rather than serving one bundle under another's name.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 07_training, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.robot.grasping.deep.train.plan import PlanOverrides, build_plan  # noqa: E402
from src.robot.grasping.deep.train.recipes import (  # noqa: E402
    RECIPES,
    TIERS,
    describe,
    settings,
)
from datagen.cost import estimate  # noqa: E402

# 1. The vocabulary that ships, and what each tier says it is for, in its own words.
print(f"recipes: {', '.join(sorted(RECIPES))}   tiers: {', '.join(sorted(TIERS))}")
for name in sorted(TIERS):
    print(f"  {name:6s} {TIERS[name]['why']}")

# 2. What each combination merges to, and the training hours it buys over 2,000 scenes. Priced with
#    the plan's OWN folds and refit: `estimate` defaults to refit=True and would otherwise bill two
#    passes for a schedule that runs one.
for pair in ((None, None), (None, "full"), ("v1", "smoke"), ("v1", "full")):
    plan, _applied = build_plan(recipe=pair[0], tier=pair[1])
    bill = estimate(2000, engine="none", label=False, corpus=False,
                    epochs=plan.epochs, folds=plan.run_folds, refit=plan.refit)
    hours = next(stage.hours for stage in bill.stages if stage.name == "train")
    print(f"  {describe(*pair) if any(pair) else '(no recipe, no tier)':52s} {hours:6.2f} h")

# 3. `tier="full"` alone does NOT turn the refit on: the refit is a recipe setting and a tier only
#    ever supplies a budget, so the two flags are not interchangeable.
print(f"tier full alone: {settings(None, 'full')}   with the recipe: {settings('v1', 'full')}")

# 4. An explicit value outranks a bundle. `refit=False` is a choice and stays distinguishable from
#    silence, which is what the UNSET sentinel inside `PlanOverrides` is for.
plan, applied = build_plan(recipe="v1", tier="full", overrides=PlanOverrides(epochs=7, refit=False))
print(f"{plan.epochs} epoch(s), refit {plan.refit}, the caller had already chosen "
      f"{applied['overridden']}")

# 5. An unknown name refuses and lists what exists, rather than serving v1 under v2's name.
try:
    settings("v2")
except ValueError as refusal:
    print(f"refused: {refusal}")
