"""Which perception stack this config builds, decided before a single weight loads.

`models.pipeline.kind` puts either an open-vocabulary grounder (GroundingDINO: the prompt is text
that reaches a text encoder) or a closed-set one (RT-DETR: no text encoder, the prompt survives
only as a class-name filter) into the detector slot. `PerceptionSpec.resolve()` reports what
`build()` would construct, quoting the builder's own refusal, without touching a checkpoint.
"""

import sys
from dataclasses import replace
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 03_perception, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.models.perception_spec import PerceptionSpec  # noqa: E402

# 1. The config tree, projected onto the seven fields that decide a perception stack.
try:
    models = load_config().models
except ConfigError as broken:
    # A tree that does not parse is the state an operator is in five seconds after a bad edit,
    # and the four files under scripts/checks/ answer it with a sentence and exit 2. An example
    # that raises instead teaches a reader nothing about the thing it was written to show.
    print(f"this config tree does not load: {broken}")
    raise SystemExit
spec = PerceptionSpec.from_config(models)

# 2. What this tree would build, and why. Nothing is loaded: resolve() reads the config and quotes
#    factory.py's own REFUSE_* constants rather than restating them in words the builder never uses.
here = spec.resolve()
print(here.render())

# 3. The closed-set side through the plain-Python door, then the fork nobody writes down: a tree
#    older than `models.pipeline` falls back to the legacy keys, which is all that `python -m
#    src.robot.perception` reads. Both say "legacy keys": `source` has no word for Python.
print(PerceptionSpec.closed_set(
    rtdetr=models.rtdetr, segmenter=models.segmenter).resolve().render())
print(replace(spec, pipeline=None).resolve().render())

# 4. One drawn frame: a grey field with a red square, 80 px, centred at (320, 240). Not a camera,
#    so either side of the fork can be asked the same question with no rig attached.
image = np.full((480, 640, 3), 200, dtype=np.uint8)
image[200:280, 280:360] = (40, 40, 220)          # BGR, so this is red

# 5. Load the weights and ground the prompt: the only step that reaches a GPU, and the only one
#    with a prerequisite. `local: true` downloads nothing, by design, so the files must be there.
missing = [b.model_path for b in (models.objectdetector, models.segmenter)
           if b is not None and b.local and not Path(b.model_path).is_dir()]
if not here.buildable:
    print(here.refusal)
elif missing:
    print(f"not built: `local: true` names {missing}, and no such directory exists here")
else:
    # `perceive` swallows a detector refusal and returns (), so that a model error reaches the
    # pick loop as no_valid_grasp; out here an empty bin and a refused prompt look the same.
    for obj in spec.build().perceive(image, "a red cube"):
        print(obj.detection.label, round(obj.detection.score, 3),
              obj.segmentation.mask_area_px, obj.segmentation.centroid_xy)
