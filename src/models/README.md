# Perception models: detection, masks, speech (`src/models`)

The neural networks the rest of the stack talks to: a detector that grounds a prompt to boxes, a
segmenter that cuts a mask for each box, a vision-language model for the prompts a detector gets wrong,
and speech to text. Each sits behind a typed wrapper that returns frozen results, and none loads torch
until it builds. A cell reaches the stack through `Locator`; call `PerceptionSpec` yourself to see or
build the stack a config describes, before a weight loads.

```python
from willy import PerceptionSpec, load_tree

spec = PerceptionSpec.from_config(load_tree().app_config.models)   # the cell WILLY_PROFILE names
print(spec.resolve())                  # which stack, decided by which half of the config, any refusal
objects = spec.build().perceive(image_bgr, "a green cube")        # loads the weights on this machine
for obj in objects:
    print(obj.detection.label, obj.detection.score, obj.segmentation.mask_area_px)
```

Images go in as OpenCV BGR arrays; each wrapper swaps to RGB itself, so do not swap before calling.
The same calls run in [resolve_perception_stack.py](../../examples/offline/perception/resolve_perception_stack.py),
and a located pick at a cell in [09_locate_and_pick.py](../../examples/real_robot/09_locate_and_pick.py).
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
| an empty tuple from `perceive` | the detector raised; the traceback is in the model log | read `logs/`; a pick reports nothing found |

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
| Perception on a real camera frame | never touched hardware |

## Fine-tuning RT-DETR on your own classes

```bash
python -m src.models.detection.closed_set.train inspect --data-dir data/detect/v1 --split train
python -m src.models.detection.closed_set.train train --data-dir data/detect/v1 --output-dir assets/models/rtdetr/v1 --epochs 20
```

A dataset is `<data-dir>/{train,val}/`, each with `images/` and a COCO `annotations.json`; `val` is
optional. The head is re-initialised from the dataset's `categories`, and the export is a
`save_pretrained` checkpoint plus `manifest.json`. Wire it in with `models.detector: "rtdetr"`,
`models.rtdetr.model_path: "assets/models/rtdetr/v1"` and `models.rtdetr.local: true`. Exit codes: `0`
ok, `2` bad arguments or missing data, `3` training failed. `inspect` runs without the training stack.

## Files

| Path | Holds |
| --- | --- |
| [`perception_spec.py`](perception_spec.py) | `PerceptionSpec` and `PerceptionResolution`; imports no torch |
| [`factory.py`](factory.py) | `build_perception`, `build_object_detector`, `build_segmenter`, the refusal sentences |
| [`perception_backend.py`](perception_backend.py) | the `PerceptionBackend` seam and `TwoStageBackend`, detector then segmenter |
| [`routed_backend.py`](routed_backend.py) | `RoutedPerceptionBackend`: two backends chosen per prompt, one shared segmenter |
| [`detection/`](detection/) | `Detection`, the GroundingDINO and RT-DETR wrappers, the RT-DETR training command |
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
