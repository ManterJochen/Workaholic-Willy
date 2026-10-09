"""Find what a sentence names with the open-vocabulary detector a cell builds, and cut each object's mask.

    python examples/offline/perception/detect_with_a_prompt.py <image> "<what to find>" [<backend>]

The detector is the one the base tree's models.pipeline builds for a cell, GroundingDINO as shipped. A third argument
names another: grounded_sam (GroundingDINO), vlm (the vision-language model, which reads harder sentences and gives no
score) or router (the two behind the cell's prompt router). Several kinds at once are one call, "red cube | blue bin",
each box under its own description. GroundingDINO and SAM2 are read from this machine; fetch them with
`python scripts/model_weights/fetch.py dino-tiny sam2`. Beside the image lands a drawing of what was found.
"""

import sys
from pathlib import Path

from willy import ObjectDetector, load_tree

if len(sys.argv) < 3:
    print('give an image and what to find: python examples/offline/perception/detect_with_a_prompt.py <image> '
          '"a red cube" [grounded_sam|vlm|router]')
    raise SystemExit
image = Path(sys.argv[1])
if not image.is_file():
    print(f"there is no image at {image.resolve()}")
    raise SystemExit
backend = sys.argv[3] if len(sys.argv) > 3 else None

# The base tree's blocks; `local: true` downloads nothing, so the weights must be on this machine. The VLM's own block
# says where its weights come from, and it loads at its first answer.
models = load_tree(None).app_config.models
blocks = [models.segmenter] + ([] if backend == "vlm" else [models.objectdetector])
missing = [block.model_path for block in blocks
           if block is not None and block.local and not Path(block.model_path).is_dir()]
if missing:
    print(f"not built: no weights at {', '.join(missing)} on this machine; fetch them with "
          "`python scripts/model_weights/fetch.py dino-tiny sam2`")
    raise SystemExit
detector = ObjectDetector.from_config(models, backend=backend)
print(detector)  # which model answers, where its weights come from, and where the masks come from

# The prompt is free text; every box found for it comes back under it. segment=True cuts all the boxes in one pass.
found = detector.detect(image, prompt=sys.argv[2], segment=True)
print(found)  # which backend answered (a router names the route it took), each object, and how many
for obj in found[:3]:
    pixels = 0 if obj.mask is None else int(obj.mask.sum())
    print(f"{obj.label}: {obj.score:.2f}, box {obj.box}, {pixels} px of mask")
print(f"drawn to {found.write_drawing(image, image.with_name(f'{image.stem}_prompt.png'))}")
