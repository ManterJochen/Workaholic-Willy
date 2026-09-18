# The vision-language route for hard prompts (`src/models/vlm`)

A vision-language model (Qwen3-VL) used as a detector: it reads a prompt a phrase grounder cannot
represent, such as a negation, a comparison or German, and answers with boxes. Boxes are what the
two-stage backend already hands to the segmenter, so this replaces the detector stage and nothing
downstream changes. A cell switches it on in config; you rarely construct it yourself.

```python
from willy import PerceptionSpec, load_tree

tree = load_tree().with_values({"models.pipeline.zero_shot.backend": "vlm"})   # or set it in object.yaml
backend = PerceptionSpec.from_config(tree.app_config.models).build()           # Qwen3-VL grounds, SAM2 cuts
objects = backend.perceive(image_bgr, "the broken part")
```

Built by hand, it is the same two stages:

```python
from src.models.perception_backend import TwoStageBackend
from src.models.vlm import Qwen3VLGrounder

backend = TwoStageBackend(detector=Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct"), segmenter=segmenter)
```

[`routing/`](../routing/README.md) decides which prompts reach this route when both are configured, and
[route_hard_prompts.py](../../../examples/offline/perception/route_hard_prompts.py) shows the switch.

## Why the route exists

A phrase grounder does not fail on a prompt it cannot represent. It returns a confident box on the wrong
object, and nothing downstream (not the gate, not the record, not the operator) can tell that from a
right answer. Measured in simulation with
[`run_attribute_pick`](../../willy_sim/run_attribute_pick.py): four objects of one size, a red and a blue
cube and a red and a blue cylinder, so only the conjunction names one.

| Prompt | Route | Intended object lifted | Wrong object lifted |
| --- | --- | --- | --- |
| `"the red cube"` | phrase grounder | 0 of 10 | 10 of 10, each reported a success |
| `"the red cube"` | VLM | 10 of 10 | 0 |
| `"der rote Wuerfel"` | phrase grounder | 0 of 10 | 0: no target, no motion |
| `"der rote Wuerfel"` | VLM | 10 of 10 | 0 |

On one rendered frame of that scene, eight prompts each naming one object in English or German, the VLM
grounded 5 of 8 and the phrase grounder 2 of 8. Every VLM miss put a cylinder prompt on the cube of the
same colour: the colour bound, the shape did not, on a frame where the circles and squares are plainly
distinct ([the frame](../../../docs/assets/attribute_scene.png)). The phrase grounder did get
`"the red cylinder"` right where the VLM did not, so neither route is strictly better.

## What it refuses

`models.pipeline.zero_shot.vlm.on_unavailable` decides what happens when the model cannot load:

| Mode | Behaviour |
| --- | --- |
| `refuse`, the default | raises `VlmUnavailableError` with the cause: weights missing, a dependency absent, or no VRAM |
| `degrade` | falls back to the phrase grounder and warns on every use, so a degraded run never looks normal |

"No objects found" and "I could not look" are different answers, and a pick must not read the second as
the first. Refuse is the default because the prompts that reach this route are the ones the phrase
grounder gets confidently wrong: a quiet fallback means grasping something the operator did not ask for.
Only those three causes degrade; any other exception is a bug and surfaces. `degrade` without a
`models.objectdetector` block is refused when the stack is built, since there is nothing to fall back to.

## The parsing is the hard part

A VLM returns text that is supposed to contain JSON: it can arrive as prose, a markdown fence or an
apology, be malformed, or be well formed and describe a nonsensical box. Every box
[`parsing.py`](parsing.py) accepts becomes a grasp pose, so it drops rather than repairs wherever a
repair would be a guess.

| Case | Behaviour | Why |
| --- | --- | --- |
| inverted corners (`x1 < x0`) | swapped | exactly one possible intent |
| coordinates outside the frame | clamped, then re-checked | a box wholly off the frame collapses and drops |
| a value in the wrong space, such as `0.1` under `absolute` | dropped, not rescaled | a rescale is a guess that sends the gripper to a corner |
| prose only, a malformed entry, a degenerate box | dropped | one bad entry never discards the good ones |

**The coordinate space is declared, never inferred.** `CoordinateSpace.GRID_1000`, the default, is the
0 to 1000 grid Qwen3-VL emits; `CoordinateSpace.ABSOLUTE` is pixels of the image as sent. A wrong space
is silent: on a 1280x720 frame, grid values are in range and sanely sized, pass every check, and send
the gripper to the wrong place. So there is no rule such as "a value past the image width means a grid".

`VLM_NOMINAL_SCORE` is `1.0` and is not a confidence, because a grounding VLM emits none: it means the
model asserted the box, and it keeps a detector's box threshold from discarding the route's output. The
model's own order is kept as its only ranking. The prompt tells the model: "If no object matches,
return an empty array []. Do not guess."

## Traps

**The checkpoint shares the card.** The model sits beside the segmenter and, in simulation, the renderer.
The 4B checkpoint in bf16 held 8.93 GB of VRAM on an RTX 5080; the 8B checkpoint's bf16 weights,
17.5 GB, do not fit that card, so the two were not compared there, and the FP8 variants were not measured.

**The schema default and the shipped value differ.** The schema names the 4B FP8 variant;
[`config/models/object.yaml`](../../../config/models/object.yaml) ships the 4B bf16 checkpoint, which
the coordinate space in `qwen.py` is written for. FP8 needs `compressed-tensors`, which `requirements.txt`
does not install.

**`preload: false` is the default**, so the model loads on the first prompt that reaches this route, a
one-off pause mid-session. `preload: true` loads it when the cell is built, except under
`router.enabled`, where the route stays lazy.

## Status

| Capability | Evidence |
| --- | --- |
| The route picking the intended object, and the grounding table above | measured in simulation |
| The coordinate space, against real weights on a synthetic scene (IoU about 0.87) | measured in simulation (`tests/test_vlm_inference.py`) |
| The route on a real cell's camera | never touched hardware |

The parser, the coordinate space and the unavailability contract are also tested without a GPU, and no
module here imports torch or transformers at import time, so a cell that never sends a hard prompt never
pays for the model. `tests/test_vlm_inference.py` needs CUDA and the weights, and skips without them.

## Files

| File | Holds |
| --- | --- |
| [`qwen.py`](qwen.py) | `Qwen3VLGrounder`: `detect_all(bgr, prompt)`, the shape GroundingDINO has; loads on first call unless `preload` |
| [`parsing.py`](parsing.py) | `parse_grounding_response`, `CoordinateSpace`, `VLM_NOMINAL_SCORE`; runs without a GPU |
| [`availability.py`](availability.py) | `GuardedVlmBackend` and `VlmUnavailableError`; the fallback is built only when needed |

## Details

- [`../routing/`](../routing/README.md) decides which route a prompt needs, before any weights load
- [`../README.md`](../README.md) is the perception layer and its factory
- Weights: `python scripts/model_weights/fetch.py vlm-4b`
- Tests: `tests/test_vlm_grounding.py`, `tests/test_vlm_inference.py`
