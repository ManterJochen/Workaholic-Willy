# VLM grounding route

A vision-language model used as a detector, so the existing two-stage backend composes it with the
segmenter unchanged.

```python
from src.models.perception_backend import TwoStageBackend
from src.models.vlm import Qwen3VLGrounder

backend = TwoStageBackend(detector=Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct"),
                          segmenter=sam2)
objects = backend.perceive(image_bgr, "the broken part")
```

That is the whole design decision. A grounding VLM produces boxes, and boxes are what
`TwoStageBackend` already hands to the segmenter. So this is not a second kind of pipeline; it is a
drop-in replacement for the detector stage, and nothing downstream changes.

## Honesty, first

**Nothing in this package has run against real weights.** The grounding quality, the VRAM cost and
the choice between the 4B and 8B checkpoints are unmeasured. The config default therefore has to
follow an on-box measurement rather than the other way round, and any load time or memory figure
you find attributed to this route is an estimate, not a measurement.

What is settled is the shape: the parser, the coordinate space and the unavailability contract are
exercised without a GPU, and no module here imports torch or transformers at import time, so the
package loads on CI and on a box with no GPU. A cell that never sends a complex prompt never pays
for the model.

## Why the route exists

The phrase grounder does not fail on a prompt it cannot represent. It returns a confident,
high-scoring box for the wrong object, and nothing downstream (not the gate, not the record, not
the operator) can tell that apart from a correct answer. Negation, comparatives, relative clauses,
quantifiers and non-English wording are exactly that class of prompt.

So the route exists to answer those prompts, and [`routing/`](../routing/README.md) decides which
prompts reach it, from the prompt alone.

## What is actually hard here

Not the model call. The parsing.

A VLM does not return a tensor, it returns text that is supposed to contain JSON, and every part of
that phrase carries weight: it is text, so it can arrive as prose, a markdown fence or an apology;
it is only supposed to be JSON, so it can be malformed; and the JSON can be well formed while
describing a nonsensical box.

[`parsing.py`](parsing.py) is therefore written from one angle: every box it accepts becomes a
grasp pose. So it drops rather than repairs wherever repair would mean guessing.

| Case | Behaviour | Why |
| --- | --- | --- |
| inverted corners (`x1 < x0`) | repaired by swapping | exactly one possible intent, no ambiguity |
| out-of-frame coordinates | clamped, then re-checked | a box entirely off-frame collapses and drops |
| a value in the wrong space, such as `0.1` under `absolute` | dropped, not rescaled | rescaling is a guess, and a wrong guess sends the gripper to a corner of the scene; dropping yields an honest empty result |
| prose-only answer, malformed entry, degenerate box | dropped | one bad entry never discards the scene's good objects |

**The coordinate space is declared, never inferred.** `CoordinateSpace.GRID_1000`, the constructor
default, means the model emits a 0 to 1000 normalised grid, which is Qwen3-VL's convention;
`CoordinateSpace.ABSOLUTE` means pixels of the image as submitted. Getting it wrong is silent: on a
1280x720 frame, grid values are in range, ordered and sane-sized, so they pass every validation and
the gripper simply goes to the wrong place. A rule like "if a value exceeds the image width, assume
a grid" would misread every small object in a large frame, which is why there is no such rule.

`VLM_NOMINAL_SCORE` is `1.0` and is not a confidence: a grounding VLM emits none. It means "the
model asserted this". It is 1.0 because downstream filters compare against a detector's box
threshold, and dropping an asserted box on a threshold meant for GroundingDINO's logits would
discard the route's entire output. The model's output order is preserved as its only ranking.

The prompt template's most safety-relevant line is `"If no object matches, return an empty array
[]. Do not guess."` Instruct-tuned models are agreeable by default and will invent a plausible box
rather than return nothing.

## When the model is not there

`models.pipeline.zero_shot.vlm.on_unavailable` selects the behaviour, implemented in
[`availability.py`](availability.py):

| Mode | Behaviour |
| --- | --- |
| `refuse` (the default) | raises `VlmUnavailableError`, carrying the underlying cause so an operator sees whether the weights are missing, a dependency is absent, or the GPU is out of VRAM. "No objects found" and "I could not look" are different answers, and the pick loop must not treat the second as the first. |
| `degrade` | falls back to the phrase grounder, and warns on every use rather than once, so a run that has fallen back never looks normal again. |

Refuse is the default because the prompts that reach this route are exactly the ones the phrase
grounder gets confidently wrong. A quiet fallback does not mean slightly worse perception; it means
grasping something the operator did not ask for.

Only the three shapes of a missing or unloadable model degrade (no dependency, no weights on disk,
no VRAM). Any other exception is a real bug and is left to surface. `degrade` without a
`models.objectdetector` block is refused at build time rather than mid-pick, since there would be
nothing to fall back to.

## Contents

| File | Role |
| --- | --- |
| [`qwen.py`](qwen.py) | `Qwen3VLGrounder`: `detect_all(bgr, prompt) -> [Detection]`, the shape GroundingDINO has. Weights load inside the first call unless `preload` is set. |
| [`parsing.py`](parsing.py) | `parse_grounding_response`, `extract_json_payload`, `CoordinateSpace`, `VLM_NOMINAL_SCORE`. The half of this route that runs without a GPU. |
| [`availability.py`](availability.py) | `GuardedVlmBackend` and `VlmUnavailableError`: the `on_unavailable` contract. The fallback arrives as a factory, so a healthy cell never builds it. |

## Traps

**The checkpoint has to fit beside the segmenter.** VRAM is the constraint on this block: the model
shares a card with the mask model and, in simulation, with the renderer.

**The schema default and the shipped value differ.** The schema names the FP8 variant;
[`config/models/object.yaml`](../../../config/models/object.yaml) sets the bf16 checkpoint, which
is the one the coordinate space in `qwen.py` is written for.

**`preload: false` is the default**, so the model loads on the first prompt that reaches this
route, a one-off pause mid-session. `preload: true` loads it at cell build instead, except under
`router.enabled`, where the route stays lazy by construction.

## See also

- [`src/models/routing/`](../routing/README.md), which decides which route a prompt needs, before
  any weights load
- [`src/models/`](../README.md), the perception model layer and its factory
- [`config/models/object.yaml`](../../../config/models/object.yaml), the `models.pipeline` block
  that selects this route
