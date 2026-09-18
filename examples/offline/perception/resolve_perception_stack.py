"""Which perception stack a config tree builds, decided before a weight loads, then built and asked.

`models.pipeline.kind` puts an open-vocabulary grounder in the detector slot, which reads the prompt
as text, or a closed-set detector, for which the prompt only filters its trained class names.
"""

from pathlib import Path

import numpy as np

from willy import PerceptionSpec, load_tree

# The base tree: the models block every cell inherits. Perception reads no robot block.
tree = load_tree(None)
spec = PerceptionSpec.from_config(tree.app_config.models)
print(spec.resolve())

# The closed-set stack, given in memory. Resolving quotes the builder's own refusal, if it has one,
# without touching a checkpoint.
closed = tree.with_values({"models.pipeline.kind": "closed_set"})
print(PerceptionSpec.from_config(closed.app_config.models).resolve())

# Building loads the weights, and `local: true` downloads nothing, so they must be on this machine.
models = tree.app_config.models
missing = [block.model_path for block in (models.objectdetector, models.segmenter)
           if block is not None and block.local and not Path(block.model_path).is_dir()]
if missing:
    print(f"not built: no weights at {', '.join(missing)} on this machine")
    raise SystemExit

# A drawn frame, not a photograph: a grey field with a red square 80 px across in the middle.
image = np.full((480, 640, 3), 200, dtype=np.uint8)
image[200:280, 280:360] = (40, 40, 220)  # BGR, so this is red

# The grounder may find nothing in a drawing. Nothing comes back as an empty tuple, and so does a
# model error, which a pick loop then reports as nothing found rather than as an exception.
found = spec.build().perceive(image, "a red square")
print(f"{len(found)} object(s) for 'a red square'")
for obj in found:
    print(obj.detection.label, round(obj.detection.score, 3), obj.segmentation.mask_area_px)
