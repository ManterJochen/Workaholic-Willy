# A watchable clip of a sim pick, from the command line.
#
# The Python twin of this file is scripts/examples/api/08_sim/record_a_pick.py and it calls the same
# record_demo. None of these score anything: they film the run that happened, including a failed one.
# For a rate over the same scenes, the gates in scripts/examples/cli/08_sim/sim_pick_rate.ps1.
#
# The first two lines need Isaac's bundled interpreter: python.bat, not this repository's venv.
# --out is not optional in practice, because the module default is an absolute path to another tree.

# 1. One eye-in-hand pick, filmed. --fps is playback speed; the simulator runs faster than an eye.
python -m src.willy_sim.run_eih_demo --fps 18 --out logs/demo/eih_pick_demo.mp4

# 2. A bin emptied object by object, with the tool choice drawn on each object as it is picked.
python -m src.willy_sim.run_bin_clearing_demo --out logs/demo/bin_clearing_demo.mp4

# 3. Stitching finished clips into one file. This one needs no Isaac: segments are separate boots,
#    because the simulator app is a singleton, so a multi-part demo is concatenated afterwards.
#    The title card before each segment is the sorting demo's own fixed text, whatever you feed it,
#    and a clip path that does not exist dies inside the encoder rather than being named.
python -m src.willy_sim.run_sorting_demo --concat logs/demo/eih_pick_demo.mp4,logs/demo/bin_clearing_demo.mp4 --out logs/demo/all.mp4

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
