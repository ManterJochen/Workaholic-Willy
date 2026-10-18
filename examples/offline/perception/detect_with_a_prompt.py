"""Find what a sentence names with the open-vocabulary detector a cell builds, and cut each object's mask.

    python examples/offline/perception/detect_with_a_prompt.py <image> "<what to find>"

The detector is the vision-language model the base tree's models.pipeline names, Qwen3-VL: it reads German and
English, colours and where a part lies, and gives no score. Several kinds at once are one call, "red cube | blue bin",
each box under its own description. The VLM and SAM2 are read from this machine, nothing is downloaded; fetch them
with `python scripts/model_weights/fetch.py vlm-4b sam2`. Beside the image lands a drawing of what was found.
"""

import sys
from pathlib import Path

from willy import ObjectDetector, load_tree

if len(sys.argv) < 3:
    print('give an image and what to find: python examples/offline/perception/detect_with_a_prompt.py <image> '
          '"a red cube"')
    raise SystemExit
image = Path(sys.argv[1])
if not image.is_file():
    print(f"there is no image at {image.resolve()}")
    raise SystemExit

# The base tree's models, each read from models.weights_root. Building loads nothing: the VLM loads at its first
# answer and SAM2 at the first mask, and a model not on this machine is refused then, naming its key and the fetch.
detector = ObjectDetector.from_config(load_tree(None), backend="vlm")
print(detector)  # which model answers, where its weights come from, and where the masks come from

# The prompt is free text; every box found for it comes back under it. segment=True cuts all the boxes in one pass.
try:
    found = detector.detect(image, prompt=sys.argv[2], segment=True)
except FileNotFoundError as missing:
    print(f"not run: {missing}")
    raise SystemExit
print(found)  # each object, and how many
for obj in found[:3]:
    pixels = 0 if obj.mask is None else int(obj.mask.sum())
    print(f"{obj.label}: box {obj.box}, {pixels} px of mask")
print(f"drawn to {found.write_drawing(image, image.with_name(f'{image.stem}_prompt.png'))}")
