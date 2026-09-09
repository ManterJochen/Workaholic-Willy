"""Extracting the corpus a generator trains on, and what a second grasp kind changes.

`clouds()` fuses every rendered view of a scene into one point cloud in base millimetres, keeps the
environment in it because a generator that never sees a bin wall proposes grasps into one, and writes
the labels beside the geometry as one .npz per scene. `kinds` is the one place a corpus decides which
grasps exist; `--kinds` on the labelling step decides nothing, because the labeller writes both.
"""

import json
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 06_datagen, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from datagen.api import DatasetBuild  # noqa: E402
from datagen.config import DatagenConfig  # noqa: E402

dataset = Path("logs/examples/example_corpus")
if not (dataset / "index.jsonl").is_file():
    print(f"no dataset at {dataset} -- choose_an_engine.py builds it in about ten seconds")
    raise SystemExit
build = DatasetBuild.from_config(DatagenConfig(scenes=1, seed=0), name=dataset.name,
                                 out_root=dataset.parent)
if not (dataset / "grasp_label_report.json").is_file():
    build.label(density="default")   # label_and_shake.py is where this step is the subject

# 1. The decision, counted on your own labels rather than on somebody else's ratio. Every object in
#    the third column and not the second is one a jaw-only corpus records as carrying nothing.
report = json.loads((dataset / "grasp_label_report.json").read_text(encoding="utf-8"))
for family, bucket in sorted(report["by_family"].items()):
    print(family, bucket["objects"], bucket["objects_with_jaw"], bucket["objects_with_suction"])

# 2. Extract with both kinds. `physics` folds the screen's verdicts in as `grasp_held`; a missing
#    file leaves every grasp stored unmeasured, which is not the same fact as a failure.
both = build.clouds("logs/examples/clouds_both", kinds=("jaw", "suction"),
                    physics=str(dataset / "grasp_physics.jsonl"))
print(both.ok, both.summary["scenes_written"], both.summary["grasps"],
      both.summary["grasps_with_physics"], both.summary["points"])

# 3. The jaw-only corpus: the default, and what every corpus in this repository was built as.
#    Note that `grasps` counts the jaw half only, so the summary alone cannot tell the two apart.
jaw = build.clouds("logs/examples/clouds_jaw", kinds=("jaw",))
print(jaw.ok, jaw.summary["scenes_written"], jaw.summary["grasps"])

# 4. Where the difference actually shows. The suction_* arrays exist in BOTH corpora and are empty
#    in the jaw-only one, so what `kinds` changes is the rows and never the keys.
for corpus in ("logs/examples/clouds_both", "logs/examples/clouds_jaw"):
    scene = sorted(Path(corpus).glob("*.npz"))[0]
    with np.load(scene, allow_pickle=False) as handle:
        print(scene.name, handle["points_mm"].shape, handle["grasp_position_mm"].shape,
              handle["suction_position_mm"].shape)
