"""Train a grasp generator on a corpus built from your own parts, at the smoke tier: about a minute on a GPU,
and proof that the chain closes on your data and your machine rather than a model to deploy.

Your parts are the `custom` meshes in the library under the working directory; with none there, generated shapes
stand in. Everything is written to a temporary directory; your run names its `out_dir`, where the weights stay.
"""

import shutil
import tempfile
from pathlib import Path

from willy import DatasetBuild, GeneratorTraining, MeshPreparation

# The smoke tier trains for about a minute on a GPU and for many times longer on a CPU alone.
if shutil.which("nvidia-smi") is None:
    print("no NVIDIA GPU driver on this machine, and training the generator wants one.")
    raise SystemExit

parts = MeshPreparation.from_sources(["custom"]).entries()
print(f"{len(parts)} of your parts in the mesh library" if parts
      else "none of your parts in the mesh library; generated shapes stand in for them")
# Your parts alone, with the fallback to generated shapes refused, as datagen/05_bring_your_own_parts.py
# explains.
assets = {"procedural_weight": 0.0, "custom_weight": 1.0, "refuse_procedural_fallback": True} if parts else {}

with tempfile.TemporaryDirectory() as work:
    corpus = Path(work) / "clouds"
    # Render, label and extract, stopping at the first stage that produces nothing. The engine none
    # refuses the pile family, so eleven requested scenes give about eight.
    dataset = DatasetBuild.from_file(name="my_parts", scenes=11, seed=0, engine="none", out_root=work,
                                     overrides={"assets": assets})
    built = dataset.run(corpus)
    print(built)
    if not built.succeeded:
        raise SystemExit(built.failure_summary())

    training = GeneratorTraining.from_recipe(corpus=corpus, recipe="v1", tier="smoke",
                                             out_dir=Path(work) / "model")
    print(training.describe())
    result = training.train()
    # The held-out numbers of the last epoch and where the weights went; a cell loads those weights
    # through robot.grasping.deep_generator.artifact_path.
    print(result)
    print("report written to", training.write_report(result))
