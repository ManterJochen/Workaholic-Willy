"""Resolve a training recipe and a compute tier into the plan a run would use, without training anything.

A recipe is a frozen bundle of settings that is written into the model, a tier is a compute budget laid over it,
and an explicit value outranks both; an unknown name refuses rather than serving another bundle under it.
"""

import tempfile
from pathlib import Path

from willy import DatasetBuild, GeneratorTraining, PlanOverrides

with tempfile.TemporaryDirectory() as work:
    # A training is built on the corpus it will train on, so a small one is extracted first.
    corpus = Path(work) / "clouds"
    dataset = DatasetBuild.from_file(name="parts", scenes=8, seed=0, engine="none", out_root=work)
    print(dataset.run(corpus))

    # The tier wins over the recipe where the two overlap. The full tier alone leaves the refit off, because
    # the refit is a setting of the recipe and a tier only sets a budget.
    for recipe, tier in ((None, None), (None, "full"), ("v1", "smoke"), ("v1", "full")):
        plan = GeneratorTraining.from_recipe(corpus=corpus, recipe=recipe, tier=tier).plan
        print(f"recipe {recipe or '-':3s} tier {tier or '-':6s} {plan.epochs:3d} epochs, {plan.run_folds} "
              f"of {plan.folds} folds, refit {plan.refit}, train units {plan.train_units}")

    # An explicit value outranks both: this run trains longer than the tier and keeps the refit off.
    training = GeneratorTraining.from_recipe(corpus=corpus, recipe="v1", tier="full",
                                             overrides=PlanOverrides(epochs=48, refit=False))
    print(training.describe())
    print("chosen by the caller over the recipe and tier:", training.recipe_notes["overridden"])
