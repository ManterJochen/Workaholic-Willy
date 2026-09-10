"""The prompts a phrase grounder answers confidently and wrongly, and where they should go.

`route()` is deterministic, reads the prompt text and nothing else, and loads no model, so every
verdict below is free. A prompt the grounder cannot represent does not fail: it comes back as a
high-scoring box for the wrong object, which is why `models.pipeline.zero_shot.backend: vlm`
exists and why `on_unavailable: refuse` is its default rather than a quiet fallback.
"""

import sys
from dataclasses import replace
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 03_perception, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_config  # noqa: E402
from src.config.schema.models.models_schema import (  # noqa: E402
    PipelineConfig, PromptRouterConfig, ZeroShotPipelineConfig)
from src.models.perception_spec import PerceptionSpec  # noqa: E402
from src.models.routing import Route, route  # noqa: E402
from src.models.vlm import VlmUnavailableError  # noqa: E402

PANEL = ("a red cube", "not the red cube", "the largest cube",
         "the cube to the left of the tray", "every cube", "greif den kaputten Wuerfel")

# 1. Route every prompt, with no model and no image. `reason` names the linguistic property the
#    phrase grounder cannot represent, and the first matching rule wins.
for prompt in PANEL:
    print(f"{prompt!r:38} {route(prompt).describe()}")

# 2. The route a prompt wants and the route this cell HAS are two different facts.
try:
    models = load_config().models
except ConfigError as broken:
    # A tree that does not parse is the state an operator is in five seconds after a bad edit,
    # and the four files under scripts/checks/ answer it with a sentence and exit 2. An example
    # that raises instead teaches a reader nothing about the thing it was written to show.
    print(f"this config tree does not load: {broken}")
    raise SystemExit
spec = PerceptionSpec.from_config(models)
here = spec.resolve()
print(here.render())

# 3. On a cell with no VLM route, every hard prompt is ground by the one model that will not fail
#    on it. What comes back is a normal-looking pick on the wrong object.
hard = [p for p in PANEL if route(p).route is Route.VLM]
if here.detector not in ("vlm", "routed"):
    print(f"{len(hard)} of {len(PANEL)} prompts here have nowhere better to go")

# 4. What switching that one key resolves to. The `vlm` sub-block is carried over rather than
#    defaulted: a fresh VlmConfig() names a different checkpoint, and a fresh router says on.
zs = models.pipeline.zero_shot if models.pipeline is not None else ZeroShotPipelineConfig()
routed = replace(spec, pipeline=PipelineConfig(
    kind="zero_shot", router=PromptRouterConfig(enabled=True),
    zero_shot=ZeroShotPipelineConfig(backend="vlm", segmenter=zs.segmenter, vlm=zs.vlm))).resolve()
print(routed.render())

# 5. And the sentence such a cell prints when that checkpoint will not load, read off the error
#    class rather than paraphrased here. `vlm_model_id` is None on a stack with no VLM route.
if routed.vlm_model_id:
    print(VlmUnavailableError(routed.vlm_model_id))
