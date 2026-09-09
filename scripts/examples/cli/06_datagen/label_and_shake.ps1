# Grasp labels, then the physics screen that grades them, from the command line.
#
# The Python twin of this file is scripts/examples/api/06_datagen/label_and_shake.py and it calls the
# same `label` verb and the same `PhysicsSampling`. Both steps address a dataset by --name and --out,
# so either can be re-run on its own: labelling is minutes where the render that made it was a night.

# 1. Label every scene. Closed-form geometry, CPU only, writes grasps.jsonl into the dataset.
#    `--density grid` is an order of magnitude more labels on the same objects, and much slower.
python -m datagen label-grasps --name example_corpus --out logs/examples --density default

# 2. The screen. A separate opt-in referee over jaw rows: needs `pip install mujoco`, or Isaac.
python -m datagen physics-sample --name example_corpus --out logs/examples `
    --physics-engine mujoco --per-class 4

# 3. Which meshes a jaw cannot grasp at all, and why. Geometry only, no dataset involved.
python -m datagen why-no-jaw --collection thingi10k --limit 3

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
