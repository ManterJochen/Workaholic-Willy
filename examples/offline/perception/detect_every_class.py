"""Every class a closed-set detector knows, found in one call, and each object's mask from the same call.

    python examples/offline/perception/detect_every_class.py <image> [<model folder or Hugging Face id>]

The model is a DetectorTraining out_dir (the run's best epoch; <out_dir>/last holds its last), any folder holding a
transformers RT-DETR, or a Hugging Face id. Left out, it is the one the base tree names, models.rtdetr: COCO's 80
classes, which `python scripts/model_weights/fetch.py rtdetr` writes. The masks are SAM2's (`fetch.py sam2`). Beside the
image land a drawing of what was found and the same as JSON, every mask run-length encoded.
"""

import json
import sys
from pathlib import Path

from willy import ObjectDetector, load_tree

if len(sys.argv) < 2:
    print("give an image: python examples/offline/perception/detect_every_class.py <image> [<model folder or id>]")
    raise SystemExit
image = Path(sys.argv[1])
if not image.is_file():
    print(f"there is no image at {image.resolve()}")
    raise SystemExit

if len(sys.argv) > 2:
    # A folder or an id. SAM2 is facebook/sam2-hiera-large unless segmenter= names another, and it loads only once
    # the first mask is asked for; local=True would keep both on this machine.
    detector = ObjectDetector.from_weights(sys.argv[2])
else:
    # The base tree's two blocks, the closed-set detector and the segmenter; `local: true` downloads nothing. The tree's
    # pipeline grounds free text as shipped, so the closed set is named.
    models = load_tree(None).app_config.models
    missing = [block.model_path for block in (models.rtdetr, models.segmenter)
               if block is not None and block.local and not Path(block.model_path).is_dir()]
    if missing:
        print(f"not built: no weights at {', '.join(missing)} on this machine; fetch them with "
              "`python scripts/model_weights/fetch.py rtdetr sam2`, or name a model after the image")
        raise SystemExit
    detector = ObjectDetector.from_config(models, backend="closed_set")
print(detector)  # where the weights come from, and every class the model knows, in id order

# Every class at once, the highest score first; segment=True cuts all the boxes in one SAM2 pass.
found = detector.detect(image, segment=True)
print(found)
for obj in found[:3]:  # what a program reads off each object, the surest first
    pixels = 0 if obj.mask is None else int(obj.mask.sum())
    print(f"{obj.label}: {obj.score:.2f}, box {obj.box}, centre {obj.centre}, {pixels} px of mask")

drawing = found.write_drawing(image, image.with_name(f"{image.stem}_found.png"))
data = image.with_name(f"{image.stem}_found.json")
data.write_text(json.dumps(found.to_dict()), encoding="utf-8")
print(f"drawn to {drawing}, written to {data}")
