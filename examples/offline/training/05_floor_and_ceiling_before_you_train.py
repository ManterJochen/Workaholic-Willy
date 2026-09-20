"""What a number measured on your corpus could possibly mean, before a six-hour run earns one.

A hit rate on its own is unreadable. Both ends of its scale are properties of the corpus and the
plan, not of a model, so both can be measured with no weights, on a CPU, in minutes: what a perfect
head would score here, and what four heads that learned nothing score. A trained number outside that
range is a bug in the measurement; a trained number that sits on the floor is a model that learned
nothing, and where the floor and the ceiling nearly meet there was nothing to learn.
"""

import tempfile
from pathlib import Path

from willy import DatasetBuild, GeneratorTraining

with tempfile.TemporaryDirectory() as work:
    corpus = Path(work) / "clouds"
    built = DatasetBuild.from_file(name="probed", scenes=8, seed=0, engine="none", out_root=work)
    print(built.run(corpus))

    # The plan matters as much as the corpus: K slots, the hit thresholds and the target all move
    # the ceiling, so the probe is asked of the run you would actually start, not of the data alone.
    run = GeneratorTraining.from_recipe(corpus=corpus, recipe="v1", tier="smoke")
    probe = run.probe(units=32)
    print(probe)

    # `top_down` is the arm to beat, not `random`: every slot straight down at the seed is the grasp
    # a cell with no model at all would try, and a learned generator that does not clear it has
    # bought nothing. Reading against `random` instead flatters every model ever trained.
    floor = probe.floor["top_down"]["top1_hit"]
    ceiling = probe.ceiling["top1_hit"]
    print(f"\ntop-1 hit: {floor:.3f} with no model, {ceiling:.3f} for a perfect one, "
          f"so {ceiling - floor:.3f} is the whole of what training can win here")

    # The ceiling is below 1.0 on coverage by construction: K slots cannot answer a seed that
    # admits more than K grasps, and that cap belongs to the head, not to the model's ability.
    print(f"coverage ceiling {probe.ceiling['coverage']:.3f} at {run.plan.model.head.slots} slots: "
          "what is left is the seeds holding more grasps than there are slots to put them in")

    # And whether the approach direction is learnable at all: how many degrees a single constant
    # direction already explains, against what per-object and per-seed knowledge would add.
    signal = probe.headroom["unaugmented"]
    print(f"approach: {signal['global']:.1f} deg from one constant direction, "
          f"{signal['per_object']:.1f} knowing the object, {signal['per_seed']:.1f} knowing the seed")
