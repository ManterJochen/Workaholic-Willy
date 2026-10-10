# Perception models: detection, masks, speech (`src/models`)

The neural networks the rest of the stack talks to: a detector that grounds a prompt to boxes, a
segmenter that cuts a mask for each box, a vision-language model for the prompts a detector gets wrong,
and speech to text. Each sits behind a typed wrapper that returns frozen results, and none loads torch
until it builds. A cell reaches the stack through `Locator`; call `PerceptionSpec` yourself to see or
build the stack a config describes, before a weight loads.

```python
from willy import load_tree
from src.models.perception_spec import PerceptionSpec

spec = PerceptionSpec.from_config(load_tree().app_config.models)   # the cell WILLY_PROFILE names
print(spec.resolve())                  # which stack, decided by which half of the config, any refusal
objects = spec.build().perceive(image_bgr, "a green cube")        # loads the weights on this machine
for obj in objects:
    print(obj.detection.label, obj.detection.score, obj.segmentation.mask_area_px)
```

Images go in as OpenCV BGR arrays; each wrapper swaps to RGB itself, so do not swap before calling.
That is the cell's own path, the one every pick takes; a located pick at a cell runs it in
[15_speak_pick_and_hand_handover.py](../../examples/real_robot/15_speak_pick_and_hand_handover.py). A program of your
own takes `ObjectDetector` ([detection/](detection/README.md)), every model here behind one
`detect(image, prompt= or classes=, segment=)`, as
[detect_with_a_prompt.py](../../examples/offline/perception/detect_with_a_prompt.py) and
[detect_every_class.py](../../examples/offline/perception/detect_every_class.py) do.
`python -m src.robot.perception --rig <rig id>` runs detection and segmentation on a live camera
([docs/cli.md](../../docs/cli.md)).

## The stacks a config can build

`models.pipeline` in [`config/models/object.yaml`](../../config/models/object.yaml) chooses one:

| `pipeline.kind` | grounding | mask source |
| --- | --- | --- |
| `zero_shot` | `grounded_sam`: GroundingDINO reads the prompt as text (the shipped stack) | `sam2` or `oneformer` |
| `zero_shot` | `vlm`: Qwen3-VL grounds the prompt; a router can send only hard prompts to it | `sam2` or `oneformer` |
| `closed_set` | RT-DETR's trained classes; a prompt only filters class names | `sam2` or `oneformer` |

Either mask source works under either grounding model, because OneFormer takes a box exactly as SAM2
does; which segments better is unmeasured. [`routing/`](routing/README.md) decides which prompts
reach the VLM, and [`vlm/`](vlm/README.md) is the VLM route itself.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `PerceptionSpec` | `from_config(models)`, `zero_shot(...)`, `closed_set(...)` | `build()` | a `PerceptionBackend` |
| `PerceptionSpec` | the same | `resolve()` | a `PerceptionResolution`: what `build()` would construct, and why |
| `PerceptionBackend` | `spec.build()`, `build_perception(models)` | `perceive(image_bgr, prompt)` | a tuple of `PerceivedObject` |
| `PerceivedObject` | the backend | read | a `Detection` (box, score, label) and its `SegmentationResult` (mask) |
| `ObjectDetector` | `from_weights(path_or_id)`, `from_config(tree)` | `detect(image_bgr, classes=, threshold=, segment=)` | `Detections`: every class of a closed-set detector in one call, each object with its SAM2 mask where asked ([`detection/`](detection/README.md)) |
| `SpeechEngine` and friends | see [`speech/`](speech/README.md) | `propose(samples, samplerate=...)` | a `Proposal` of text a person confirms |

`build_perception` is the one builder; `PerceptionSpec.build()` calls it and `resolve()` predicts it
from the same refusal constants. `build_object_detector` and `build_segmenter` stay for assembling the
two stages by hand, with a checkpoint the config does not name.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` naming the missing block | a stack whose model block is absent, such as `closed_set` with no `models.rtdetr` | add the block; `resolve()` quotes the sentence first |
| a `ConfigError` at load | `router.enabled: true` without `zero_shot.backend: vlm`, or under `closed_set` | switch the VLM on, or the router off |
| `FileNotFoundError` from GroundingDINO | `local: true` and no directory at `model_path` | `python scripts/model_weights/fetch.py dino-tiny`, or `local: false` |
| `VlmUnavailableError` | the VLM cannot load and `on_unavailable` is `refuse` | see [`vlm/`](vlm/README.md) |
| an empty tuple from `perceive` | the detector raised, or the segmenter raised on every detection; the traceback is in the model log | read `logs/`; a pick reports nothing found |

**A model error is counted, not hidden.** `TwoStageBackend` turns a model that raised into "no objects", so one
bad frame never kills a pick, and counts it: `failures` only grows and `last_failure` names the model and the
exception (`failures_of(backend)` and `last_failure_of(backend)` read them on any backend, the routed and the
guarded ones pass them through). A caller that must tell "the scene is empty" from "the model could not look"
reads the count before and after: the console's task ends `detector_failed` on it, never "nothing left". A VLM that
cannot load raises inside the detector, so it is counted the same way, before `on_unavailable` sees it.

**One colour word, where the pixels cannot tell.** The camera source judges a part's colour on its pixels and asks
the backend only when they leave it unsure (`src/robot/perception/colour_check.py`): `name_colour(image_bgr, box)`
answers one colour word for the object in the box, or `""`. `TwoStageBackend` asks its detector where it can answer
(Qwen3-VL's `name_colour`; GroundingDINO answers nothing), and counts a question that raises in `failures` as a
perceive that raised; `GuardedVlmBackend` asks the VLM while it is not degraded; `RoutedPerceptionBackend` asks its VLM
route, building it as a prompt that chooses it would. No answer refuses the part.

**Boxes cut with no detector.** A task that follows its parts (`robot.grasping.follow_parts`,
`src/robot/perception/kept_scene.py`) has SAM2 cut the boxes of the parts it kept, and asks no detector:
`segment_boxes(image_bgr, boxes)` takes `((x0, y0, x1, y1), label)` pairs and answers `SegmentedBoxes`, the object cut
for each box in their order and a sentence for each box that could not be. `TwoStageBackend` cuts each as a detection of
score 1.0 under its label, and never counts a box it could not cut in `failures`: the camera source grounds the frame
instead, and that perceive counts its own. `GuardedVlmBackend` forwards to the VLM route's segmenter without loading
the VLM, and cuts nothing once degraded; `RoutedPerceptionBackend` cuts with the route that grounded last, and builds
no route for it, so a cell that has grounded nothing yet cuts nothing.

The shipped `model_path` directories sit under `assets/models/hf/` and are empty in a fresh clone:
`python scripts/model_weights/fetch.py --list` names what to fetch. GroundingDINO, Whisper and Silero
check for their files and name the fix; SAM2 does not, and fails inside `from_pretrained` instead.

## Traps

**Which half of the config decides depends on the builder.** While `models.pipeline` is set it decides
for everything built through `build_perception`, which is how a cell is assembled. `models.detector`
and `models.segmenter_backend` decide for `build_object_detector` and `build_segmenter`, which is what
`python -m src.robot.perception` uses. On the shipped values both resolve to the same pair, and
`resolve()` prints which half decided.

**`torch_dtype` matters for small objects.** Half precision costs recall on small objects, and
`build_load_kwargs` warns whenever it resolves to fp16. Naming `"float32"` is not the same as leaving
the key unset, because a named dtype also goes to `torch.autocast`; the loader's `"__null__"` value
resets the key to unset. `channels_last` and `compile` apply on CUDA only and fall back to eager on
failure, with a warning.

**Debug images are off by default.** `debug_images=True` on a detector and `save_debug=True` on a
segmenter write PNGs under `logs/debug/<name>` (or `WILLY_DEBUG_DIR`), capped at 200 per folder. Keep
them off in production: the writes add latency.

**A GPU is recommended.** With neither CUDA nor MPS the device falls back to the CPU with a warning, and
everything runs slowly. MediaPipe runs on the CPU in any case.

## Status

| Capability | Evidence |
| --- | --- |
| GroundingDINO and SAM2 on a prompted pick | measured in simulation (Isaac, real-vision picks) |
| The VLM route | measured in simulation ([`vlm/`](vlm/README.md) has the numbers) |
| RT-DETR and OneFormer in a pick | never touched hardware |
| RT-DETR training on your own classes | run on the development RTX 5080: the shapes dataset reaches mAP 1.0 in 3 epochs, the smoke tier runs in the suite; never on a dataset of real parts |
| Perception on a real camera frame | run on a physical cell: GroundingDINO, SAM2 and the VLM route on a wrist D415; no measurement is kept here |

## Training RT-DETR on your own classes

`DetectorTraining` (from `willy`) trains the closed-set detector on **your own fixed classes** from a COCO or
YOLO folder and keeps the **best epoch** by validation mAP. [`detection/closed_set/training/`](detection/closed_set/training/)
holds it; the command line is its shell face:

```bash
python -m src.models.detection.closed_set.train inspect --data-dir data/detect/v1
python -m src.models.detection.closed_set.train train --data-dir data/detect/v1 --output-dir assets/models/rtdetr/v1
python -m src.models.detection.closed_set.train eval --model-dir assets/models/rtdetr/v1 --data-dir data/detect/v1
```

- **Datasets:** COCO as CVAT, Label Studio and Roboflow export it, or YOLO (`data.yaml`, or `classes.txt` with
  `labels/`). Without its own validation split a dataset gives **15 %** of its images to one, chosen by a hash of
  each path, so the same images stay in validation. `inspect` names every box and image it left out, and why.
- **The run** is RT-DETR's own recipe: AdamW with the backbone at a tenth, warm-up then cosine, bf16 or fp16,
  gradients clipped at 0.1, an average of the weights (EMA), colour, zoom-out, IoU-crop, flip and multi-scale
  until the last tenth of the epochs, COCO mAP@0.5:0.95 after every epoch, and a stop after 15 epochs without a
  better one. `--tier smoke` proves the chain in a minute; `--tier full` (50 epochs at most) is the model to deploy.
  `--classes "white pawn" "black pawn"` trains some of the dataset's classes only, the objects of the others left
  in the images as background; `inspect --classes` shows what that leaves.
- **What `--output-dir` holds:** the best epoch (wire it in with `models.detector: "rtdetr"`,
  `models.rtdetr.model_path: "${WILLY_PROJECT_ROOT}/assets/models/rtdetr/v1"` and `models.rtdetr.local: true`; a
  relative path in the tree is read against the config folder, [`src/config/paths.py`](../config/paths.py)),
  `last/`, `checkpoint_last.pt` for `--resume`, `results.csv`, `curves.png` (drawn again after every epoch),
  `manifest.json`, `report.json` and `report.html`, the whole run on one page.
- **Speed, measured** on the development RTX 5080 with every augmentation on, on 400 chess-piece photos of up to
  12 megapixels (2026-10-10): about **20 training images per second**. An image is read at most 1333 px on its long
  side at 640, its boxes scaled with it, and four loader processes read beside the step.
- Exit codes: `0` ok, `2` bad arguments or no dataset, `3` training failed. `inspect` runs without the training
  stack. Loader processes start on Windows as anywhere, without running the training script again, so a script of
  your own needs no `if __name__ == "__main__":` guard.

## Files

| Path | Holds |
| --- | --- |
| [`perception_spec.py`](perception_spec.py) | `PerceptionSpec` and `PerceptionResolution`; imports no torch |
| [`factory.py`](factory.py) | `build_perception`, `build_object_detector`, `build_segmenter`, the refusal sentences |
| [`perception_backend.py`](perception_backend.py) | the `PerceptionBackend` seam and `TwoStageBackend`, detector then segmenter |
| [`routed_backend.py`](routed_backend.py) | `RoutedPerceptionBackend`: two backends chosen per prompt, one shared segmenter |
| [`detection/`](detection/README.md) | `Detection`, the GroundingDINO and RT-DETR wrappers, `ObjectDetector` (every class in one call, each object with its SAM2 mask), RT-DETR training on your own classes (`DetectorTraining`) and its command |
| [`segmentation/`](segmentation/) | `SegmentationResult` (a `uint8` mask of 0 and 1), the SAM2 and OneFormer wrappers |
| [`routing/`](routing/README.md), [`vlm/`](vlm/README.md) | which route a prompt takes, and the VLM that answers the hard ones |
| [`speech/`](speech/README.md) | speech to a prompt: Whisper, Silero, push to talk, a person's confirmation |
| [`handdetection/`](handdetection/README.md) | palm centre and thumbs up or down; nothing on the grasp path builds it |
| [`_inference.py`](_inference.py) | the shared torch load and optimise helpers |

## Details

- Guide: [models](../../docs/guide/02-models.md), install, weights, `models.pipeline` and the dtype trap
- [`src/robot/perception/`](../robot/perception/README.md), the real-camera `Locator` built on this stack
- [`src/camera/`](../camera/README.md) supplies the images; [`src/calibration/`](../calibration/README.md) the CAMERA to BASE transform
- Tests: `tests/test_perception_spec.py`, `tests/test_routed_backend.py`, `tests/test_models_imports.py`
