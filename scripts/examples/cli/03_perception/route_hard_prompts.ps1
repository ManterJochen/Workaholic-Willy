# Where a prompt a phrase grounder cannot represent should go, from the command line.
#
# Routing itself has no shell surface: there is no src/models/routing/__main__.py and no
# `python -m` anywhere that routes a prompt. Its one non-Python door is the operator console's
# GET /v1/diagnostics/route?prompt=... The config half is what a shell can answer, and the Python
# twin scripts/examples/api/03_perception/route_hard_prompts.py routes the panel itself.

# 1. Which model grounds. `grounded_sam` is the phrase grounder; `vlm` is the only route here that
#    reaches negation, comparatives, relative clauses, quantifiers and a prompt that is not English.
python -m src.config explain models.pipeline.zero_shot.backend

# 2. Whether the per-prompt router is on at all. It reads the text, loads no model, and is free.
python -m src.config explain models.pipeline.router.enabled

# 3. What a cell does when that checkpoint will not load: `refuse` stops, `degrade` grounds with the
#    phrase detector instead and is wrong on exactly the prompts the route existed for.
python -m src.config explain models.pipeline.zero_shot.vlm.on_unavailable

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
