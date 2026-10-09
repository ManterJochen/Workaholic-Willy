# Detection: boxes for a prompt, or every class at once (`src/models/detection`)

Two detector families behind one result type. **GroundingDINO** (`zero_shot/`) reads a prompt as text and grounds it
to boxes; **RT-DETR** (`closed_set/`) knows a fixed list of classes, the ones it was trained on, and reads no text at
all. Both answer with the frozen `Detection` of [`types.py`](types.py), so the cell's perception takes either.
**`ObjectDetector`** ([`object_detector.py`](object_detector.py)) is every detection model a cell can run, as a program of
your own uses it (the owner, 2026-10-09: "alle unter einem Dach"): the closed-set RT-DETR, every class it knows in one
call; GroundingDINO; the VLM ([`../vlm/`](../vlm/README.md)); and the router between those two. One
`detect(image, prompt= or classes=, segment=)`, one result type, and each object's mask from SAM2 in the same call when
you ask for it. **`DetectorTraining`** ([`closed_set/training/`](closed_set/training/)) trains the closed-set model on
your own classes.

## `ObjectDetector`: every class at once, with its masks

The owner, 2026-10-09: a chess program takes the closed-set detector, always wants every piece at once, and wants the
masks switched on and off by one flag. Train once on your labelled pieces, then build the detector once and keep it:

```python
from willy import DetectorTraining, ObjectDetector

# Once: your pieces, labelled in COCO or YOLO as CVAT, Label Studio or Roboflow export them.
run = DetectorTraining.from_dataset(dataset="data/chess_pieces", out_dir="models/chess")
print(run.train())                                    # the best epoch in models/chess, the last in models/chess/last

# Then, frame after frame: every piece the model knows, each with its mask.
detector = ObjectDetector.from_weights("models/chess")  # loads RT-DETR; SAM2 waits for the first mask
print(detector.classes)                                 # ("white_pawn", ..., "black_king"): its id2label, in id order
found = detector.detect(frame_bgr, segment=True)        # segment=False: boxes only, and SAM2 is never asked
print(found)                                            # a line per piece, and how many of each class
print(found.counts)                                     # {"white_pawn": 8, ..., "white_queen": 0, ...}
for piece in found:                                     # the highest score first
    print(piece.label, piece.score, piece.box, piece.centre, piece.mask.sum(), piece.mask_score)
kings = found.of("white_king")                          # one class; a name it never looked for raises
found.write_drawing(frame_bgr, "board.png")             # boxes, classes, scores and the masks tinted
json_ready = found.to_dict()                            # every mask as COCO run-length encoding
```

[`examples/offline/perception/detect_every_class.py`](../../../examples/offline/perception/detect_every_class.py)
runs the same on any image you name.

| Name | What it is |
| --- | --- |
| `ObjectDetector.from_weights(path_or_id, *, threshold=0.5, device=None, segmenter=None, local=None)` | A `DetectorTraining` out_dir (its best epoch; `out_dir/last` loads the last the same way), any folder `save_pretrained` wrote a transformers RT-DETR and its image processor to, or a Hugging Face id. `segmenter` names SAM2 as a folder or an id, `facebook/sam2-hiera-large` when left out. An id is read from the folder `scripts/model_weights/fetch.py` writes when that is there, else from the Hugging Face cache, else downloaded; `local=True` reads this machine alone. `device` (`"cpu"`, `"cuda"`, `"mps"`) runs both models there; left out, `WILLY_DEVICE` or the first of CUDA, MPS and the CPU |
| `ObjectDetector.from_config(tree, *, backend=None)` | What the tree's `models.pipeline` builds for a cell (`kind: closed_set` is RT-DETR; `zero_shot` is GroundingDINO, the VLM, or both behind the router), or the backend named, from its block: `closed_set` from `models.rtdetr` (its threshold too), `grounded_sam` from `models.objectdetector`, `vlm` from `models.pipeline.zero_shot.vlm`, `router` from the last two. SAM2 is `models.segmenter`. Takes a loaded tree, its `AppConfig` or its models block |
| `detector.backend`, `detector.open_vocabulary` | Which model answers (`closed_set`, `grounded_sam`, `vlm`, `router`), and whether it reads text |
| `detector.classes` | Every class name a closed-set model knows, in id order: exactly its `id2label`; `()` for an open-vocabulary one |
| `detector.detect(image_bgr, *, prompt=None, classes=None, threshold=None, segment=False)` | The objects found, as `Detections`. `image_bgr` is an OpenCV BGR image or an image file's path. A closed-set detector answers with every class, `classes` narrowing the call to those names, exactly but for case and blanks; an open-vocabulary one takes exactly one of `prompt` and `classes`. `threshold` stands in for the detector's own for this call |
| `Detections` | A frozen sequence of `DetectedObject`, the highest score first, with the question it answered: `classes` looked for, `threshold` (`None` where the model gives no score), `image_hw`, `segmented`, `backend` (the route a router took) and `prompt` (the text the model read). `of(label)`, `labels`, `counts`, `render()` (what `print` shows), `draw(image)`, `write_drawing(image, path)`, `to_dict(masks=True)`, `from_dict(data)` |
| `DetectedObject` | `label`, `score`, `box` (`x0, y0, x1, y1` in the image's pixels), `centre` (the box's), `mask` (the image's height by width, `True` on the object, read-only; `None` without `segment=True`), `mask_score` (SAM2's predicted IoU of the mask) |

**What one call costs.** One RT-DETR pass over the image. `segment=True` adds one SAM2 pass: the image is encoded once
and every box is decoded in the same call, where the cell's perception asks SAM2 once per box. An image with no box asks
SAM2 nothing. Building loads RT-DETR, and the first `segment=True` with a box to cut loads SAM2 (0.9 GB), which is then
kept, so build one detector and reuse it. Calls on one detector take turns under a lock, so threads may share it; two
detectors hold two copies of both models. How long a call takes is not measured yet.

## `ObjectDetector` with a prompt: GroundingDINO, the VLM and the router

The same door reads text. `from_config` builds what a cell's `models.pipeline` builds, GroundingDINO as shipped, or the
backend you name; a call names exactly one of a prompt or classes:

```python
from willy import ObjectDetector, load_tree

detector = ObjectDetector.from_config(load_tree())                 # the cell's own; backend="vlm" names another
found = detector.detect(frame_bgr, prompt="the red cube", segment=True)   # every box for the prompt, with its mask
found = detector.detect(frame_bgr, classes=["red cube", "blue bin"])      # several kinds in one call
print(found)                                                       # which backend answered, and each object
```

| `backend` | The model | Reads | Score and `threshold` | Built from |
| --- | --- | --- | --- | --- |
| `closed_set` | RT-DETR | no text: `classes=` narrows its trained classes, a prompt is refused | its score; `threshold` | `models.rtdetr`, or `from_weights` |
| `grounded_sam` | GroundingDINO | a prompt or classes | its score; `threshold` | `models.objectdetector` |
| `vlm` | the VLM (Qwen3-VL) | a prompt or classes, the harder sentences too | none: a VLM writes boxes, so `Detections.threshold` is `None` and a `threshold` is refused | `models.pipeline.zero_shot.vlm`, the process's one copy, which a cell's console shares |
| `router` | GroundingDINO, and the VLM for the prompts the cell's router sends there (another language, a negation, a comparison) | a prompt or classes | GroundingDINO's where it answers; `Detections.backend` names the route | both blocks; the VLM loads at the first prompt routed to it |

Every box found for a prompt comes back under the prompt: GroundingDINO grounds a sub-phrase ("cube" of "the red
cube") and the VLM writes its own words, and both are the prompt's here, as a cell takes them for its task's object.
Classes are found as the cell finds its sorting rules and bins, one class list in one call (`"red cube | blue bin"`),
each box under the description the model gave it; a box it gave none, or two, is neither and is left out, as a cell
keeps it out of its targets. [`examples/offline/perception/detect_with_a_prompt.py`](../../../examples/offline/perception/detect_with_a_prompt.py)
runs it on any image you name, its third argument the backend.

**The masks are the cell's masks.** Of the three masks SAM2 proposes for a box the first is kept, and cleaned as
`Sam2Segmenter` cleans its one (opened, closed, its largest piece kept), so a box cut in the batch comes out as the
cell's segmenter cuts it alone; `tests/test_one_sam2_pass_cuts_what_one_pass_per_box_cuts.py` holds the two paths
together through the real transformers SAM2 at toy size. A box where SAM2 cut nothing has an all-`False` mask.

**`to_dict()`** is plain data `json.dumps` takes as it is. Each mask is COCO's uncompressed run-length encoding,
`{"size": [height, width], "counts": [...]}`, read column by column with the background counted first, so
`pycocotools` reads it and `Detections.from_dict` (or `decode_mask`) turns it back into the mask; `to_dict(masks=False)`
leaves the masks out, and `mask_encoding` says which (`"coco_rle"` or `None`).

### What it refuses

| Refusal | When |
| --- | --- |
| `FileNotFoundError` | A folder that holds no detector, naming what it lacks (`config.json`, the weights, `preprocessor_config.json`); an out_dir whose run kept only its last epoch, naming `last/`; a path that is not there, naming where it looked; a `local` block of `from_config` whose folder is missing, naming the key and the fetch (the detector's when it is built, SAM2's at the first mask); an image file that is not there |
| `ValueError` | Another kind of checkpoint (a SAM2 folder given as the detector, or a detector as SAM2); a class name the model does not know, listing every class it knows, so a misspelt piece never reads as an empty board; `classes=[]`; a threshold outside 0 to 1; `segment=True` on a detector built without SAM2; an image that is not height x width x 3 `uint8`; a tree with no `models.rtdetr` block; `of()` a class the call did not look for |
| `TypeError` | `from_weights(None)`, which is what a report's `model_dir` is when its run wrote no model; `from_config` of something that is no config |
| `ConfigError` | `from_config` of a tree that did not load |

### Traps

- **One object, one class.** RT-DETR scores every class of each of its queries on its own, so one piece can clear
  the threshold under two classes, with the same box to the bit. It comes back once, under its best class, and
  `classes=` picks from those: a knight the model scores 0.8 as a knight and 0.55 as a bishop is no bishop. Two
  queries that land on one piece are two objects; nothing merges boxes that differ.
- **`classes=` is exact on the closed set.** `"pawn"` names no class of a chess model; the cell's prompt filter reads a substring of a
  class name, this does not. Case and blanks aside, `" White_Pawn "` is `white_pawn`.
- **A box is clipped to the image.** A box reaching past the edge ends at it, and one wholly outside is dropped.
- **SAM2's memory grows with the boxes.** Its decoder answers every box of the image in the one call, and a threshold
  near 0 hands it as many boxes as RT-DETR has queries (300 for the COCO checkpoint).
- **`mask_score` is SAM2's own guess** of the mask's IoU, not a measurement.

### Licences

Both model families are Apache-2.0, code and weights, read rather than assumed:

| What | Licence | Read in |
| --- | --- | --- |
| RT-DETR and SAM2 as transformers implements them (transformers 5.5.4) | Apache-2.0 | its package metadata, and the headers of `modeling_rt_detr.py` and `modeling_sam2.py` |
| `PekingU/rtdetr_r50vd`, the COCO checkpoint, and the base every `DetectorTraining` run fine-tunes from | Apache-2.0 | [NOTICE](../../../NOTICE), from its model card |
| `facebook/sam2-hiera-large`, the SAM2 that cuts the masks | Apache-2.0 | [NOTICE](../../../NOTICE), from its model card |

A model you train starts from the Apache-2.0 RT-DETR weights and learns from your dataset, so your dataset's own terms
apply to it as well.

## The rest of the package

| Path | Holds |
| --- | --- |
| [`object_detector.py`](object_detector.py) | `ObjectDetector`, `Detections`, `DetectedObject`, `encode_mask`, `decode_mask`; imports no torch until a detector is built |
| [`types.py`](types.py) | `Detection`, the frozen box, label and score every backend emits |
| [`closed_set/detector.py`](closed_set/detector.py) | `RtDetrObjectDetector`: the trained classes (`classes`), a prompt read only as a class-name filter, `detect_all(image, prompt, threshold=)` |
| [`closed_set/training/`](closed_set/training/) | `DetectorTraining`: COCO or YOLO in, the best epoch and the last out ([`src/models/README.md`](../README.md), training) |
| [`zero_shot/detector.py`](zero_shot/detector.py) | `GroundingDinoObjectDetector`: a prompt read as text, `detect_all(image, prompt, threshold=)` |

## Status

| Capability | Evidence |
| --- | --- |
| `ObjectDetector`: every class, the filter, the refusals, one SAM2 call for every box | unit tests on a CPU with stand-ins, the real RT-DETR and SAM2 classes at toy size, and the loading of a `DetectorTraining`-shaped folder |
| `ObjectDetector` with a prompt or classes: GroundingDINO, the VLM, the router, and `from_config` building each from its block | unit tests with stand-ins for the three models (the router is the cell's own `RuleBasedRouter`) |
| SAM2-large on a GPU cutting a batch as it cuts each box alone | `tests/test_an_object_detector_on_real_weights_inference.py`, written and not run yet |
| GroundingDINO, the VLM or the router on real weights through this door | not run yet; the cell runs the same models through its own perception every pick |
| A chess board, or any real camera frame | never touched |

Tests: `tests/test_an_object_detector_finds_every_class_at_once.py`,
`tests/test_an_object_detector_finds_what_a_prompt_names_on_every_backend.py`,
`tests/test_an_object_detector_cuts_every_box_in_one_sam2_pass.py`,
`tests/test_one_sam2_pass_cuts_what_one_pass_per_box_cuts.py`,
`tests/test_a_detector_folder_loads_its_best_or_last_epoch_and_anything_else_is_refused.py`.
