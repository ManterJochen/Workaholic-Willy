#!/usr/bin/env bash
# Which perception stack this config builds, from the command line.
#
# The Python twin of this file is scripts/examples/api/03_perception/resolve_perception_stack.py,
# and it reads the same keys through PerceptionSpec. `explain` answers where a value was set and
# why, which is the half of this decision worth having before fetching gigabytes of weights.
set -euo pipefail

# 1. The key that picks the fork: an open-vocabulary grounder or a closed-set one.
python -m src.config explain models.pipeline.kind

# 2. The two legacy keys a tree without that block falls back to. They are also the only two the
#    bench exerciser reads, so a tree where the two halves disagree grounds twice, differently.
python -m src.config explain models.detector
python -m src.config explain models.segmenter_backend

# 3. Ground one prompt with the legacy pair, on a cell whose primary rig delivers depth. This one
#    opens the camera and loads the weights.
#    python -m src.robot.perception --prompt "a red cube"
