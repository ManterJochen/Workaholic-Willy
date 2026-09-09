# Extracting the point-cloud corpus a generator trains on, from the command line.
#
# The Python twin of this file is scripts/examples/api/06_datagen/extract_a_corpus.py and it calls
# the same `clouds` verb. `--kinds` is the one place a corpus decides which grasps exist: the same
# word on the labelling step decides nothing, because the labeller always writes both.

# 1. Both kinds, with the physics verdicts folded in as `grasp_held`. Without --physics every grasp
#    is stored unmeasured, and unmeasured is not the same fact as a measured failure.
python -m datagen build-cloud-corpus --name example_corpus --out logs/examples `
    --corpus-out logs/examples/clouds_both --kinds both `
    --physics logs/examples/example_corpus/grasp_physics.jsonl

# 2. The jaw-only default, for contrast. The suction_* arrays exist in this corpus too and are
#    empty, so what --kinds changes is the rows and never the keys.
python -m datagen build-cloud-corpus --name example_corpus --out logs/examples `
    --corpus-out logs/examples/clouds_jaw

# 3. Is what came out trainable at all: is the ground truth where a network can see it?
python -m src.robot.grasping.deep sanity --clouds logs/examples/clouds_both --scenes 8

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
