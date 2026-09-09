# Import a public 6-DoF grasp corpus, from the command line.
#
# The Python twin of this file is scripts/examples/api/07_training/import_a_public_corpus.py, and it
# calls the same `import_scenes`: the command resolves flags into arguments and then calls the Python
# door. Both need the network, and both write the same .npz scene files the local generator writes,
# so everything downstream reads an imported corpus with no flag and no branch.

# 1. What the tool will do, including which jaw the labels are cut for and why the flags matter.
python -m src.robot.grasping.deep import-foreign --help

# 2. Import four scenes. The full archive index is read once and cached under --out, about 460 MB;
#    every scene after that is a few hundred kilobytes by range request.
python -m src.robot.grasping.deep import-foreign --out logs/examples/training/foreign `
    --limit 4 --gripper wide_140 --jobs 8

# 3. Train on it exactly as on a local corpus. The trainer cannot tell the two producers apart.
#    python -m src.robot.grasping.deep train-set --tier smoke --recipe v1 --run-folds 1 `
#        --clouds logs/examples/training/foreign --out logs/examples/training/foreign_model

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
