"""Train a grasp generator on the parts your cell actually handles: meshes in, a model card out.

`engine="none"` seats objects analytically and rasterises in numpy, so the data stages need neither
a GPU nor a simulator; the smoke tier is minutes on one, and `tier="full"` is the model you deploy.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 07_training, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.robot.grasping.deep.eval import run_report  # noqa: E402
from src.robot.grasping.deep.train.api import GeneratorTraining  # noqa: E402
from src.robot.grasping.deep.train.plan import PlanOverrides  # noqa: E402
from datagen.api import DatasetBuild  # noqa: E402
from datagen.assets.library import import_from_directory  # noqa: E402
from datagen.cost import estimate, format_estimate  # noqa: E402

MESHES = Path("assets/meshes/your_parts")  # your own parts, which no public dataset contains
CORPUS = "logs/examples/training/my_parts_clouds"
MODEL = "logs/examples/training/my_parts_model"

# 1. Price the chain first, and read BOTH scene numbers: a request is not a yield.
plan = estimate(8, engine="none", epochs=2, folds=1, refit=False)
print(format_estimate(plan))

# 2. Copy your meshes in. `license` has no default: this repo's CI audits that string.
try:
    imported = import_from_directory("custom", MESHES, license="own")
    assets = {"procedural_weight": 0.0, "custom_weight": 1.0, "refuse_procedural_fallback": True}
    print(f"{len(imported)} mesh(es) imported as `custom` from {MESHES}")
except FileNotFoundError as missing:
    assets = {"procedural_weight": 1.0}
    print(f"{missing} -- generated objects instead, proving the chain and not your cell")

# 3. Build from those parts ONLY: zeroing the weight leaves the fallback, so refuse that too.
build = DatasetBuild.from_file(None, name="my_parts", scenes=plan.requested_scenes, engine="none",
                               out_root="logs/examples/training", overrides={"assets": assets})
print(build.describe())

# 4. Render, label and extract the corpus, stopping at the first stage that produces nothing.
try:
    print(build.run(CORPUS).render())
except OSError as unwritable:
    # Render, label and extract all write under logs/examples/training, and so does the fit
    # below, so there is nothing left to show once this refuses.
    print(f"cannot write under logs/examples/training ({unwritable}); every stage from here "
          f"on writes there")
    raise SystemExit

# 5. Fit the generator. The smoke tier is minutes and says nothing about grasp quality.
training = GeneratorTraining.from_recipe(corpus=CORPUS, recipe="v1", tier="smoke",
                                         overrides=PlanOverrides(run_folds=1), out_dir=MODEL)
print(training.describe())
result = training.train()
training.write_report(result)
print(result.render())

# 6. The lift is the number: the held-out hit rate minus its own measured floor.
print(run_report.format_report(run_report.build_report("my_parts", MODEL)))
