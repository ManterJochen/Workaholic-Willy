#!/usr/bin/env bash
# Which backend renders your scenes, and what it costs, from the command line.
#
# The Python twin of this file is scripts/examples/api/06_datagen/choose_an_engine.py and it prices
# and builds through the same `estimate` and `DatasetBuild`. `--engine` is an installability fork
# and not a fidelity preference: `none` needs nothing, `mujoco` is a pip wheel, `isaac` is a GPU install.
set -euo pipefail

# 1. Price the request. `request 11 to get 8` is the yield: `none` refuses the whole pile family.
python -m datagen cost --engine none --scenes 8

# 2. The what-if. One pip wheel buys the family back, and the yield goes to 1.0.
python -m datagen cost --engine mujoco --scenes 8

# 3. Build with the engine that needs nothing, asking for the requested count and not the wanted one.
#    The renderer is stamped into the dataset's provenance.json; nothing downstream substitutes it.
python -m datagen build --engine none --scenes 11 --name example_corpus --out logs/examples
