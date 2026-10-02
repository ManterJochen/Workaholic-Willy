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

**The same copy reads the operator's commands.** A process holds one Qwen3-VL (`shared_vlm()`): the build takes
its grounder from it, and the console's command reader asks it a text-only question, so a cell whose detector is
the VLM reads commands with the copy it detects with:

```python
from src.models.vlm import shared_vlm, understand

vlm = models.pipeline.zero_shot.vlm                 # the models section the cell was built with
ask = shared_vlm().asker_for(vlm, may_load=True)    # the model's text answer to one question
reading = understand("Leg den grünen Würfel in die blaue Kiste", ask=ask, poses={"Ablage links": "ablage_links"})
print(reading)                                      # object 'green cube', place 'blue bin', scope once
```

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

## Reading a command

[`command.py`](command.py) reads an operator's sentence, German or English, into a task card, and moves
nothing: `understand(text, ask=, poses=)` calls `ask` and nothing else, and no robot, camera or console module is
imported. The owner reads commands only through the VLM: no word list reads the sentence.

- **The question.** The whole sentence goes to the model with one fixed instruction (`COMMAND_INSTRUCTION`,
  1,286 tokens) and the taught poses as spoken label to name pairs, so a spoken "Ablage links" comes back as its
  name. `Qwen3VLGrounder.answer_text(system, user)` asks it with no image, the answer prefilled with `{`, at
  most 160 new tokens, greedy.
- **Hard checks, one retry.** The answer must be one JSON object with exactly the instruction's keys. A missing or
  extra key, a scope other than `once` and `until_empty`, a place and a pose that contradict each other, or a
  phrase for the detector that is not English or carries a quantifier is asked again once, saying what was wrong;
  a German command's phrase made of the command's own words alone was copied, not translated. Word lists check
  the answer, never the sentence: `GERMAN_WORDS` (German part, colour and material words the router reads as
  English), `SHARED_WORDS` (words English shares, such as "Box", which are no copy) and `ANY_PART_WORDS` (a word
  for any part names none, so the card asks what to pick). A second bad answer is "not understood", and the card
  is filled in by hand.
- **Read narrowly.** A place whose own words are a taught pose's label is that pose, with no second question. A
  place said as Home is no place: Home is where the arm returns, and the note says so. A counted command ("nimm
  drei Würfel") is read as `once` with the note that a count is not supported.
- **Soft notes**, in `NOTE_ORDER`: `object_not_in_sentence`, `place_not_in_sentence`, `pose_unknown`,
  `count_not_supported`, `retried`. The card marks them "bitte prüfen".
- **`read_command`** is the console's door: the availability rule (`reader_availability`) first, then the
  reader through the shared copy. Its refusals are the console's codes: `vlm_not_loaded`, `vlm_unavailable` (also
  `VlmAnswerFailedError`, a loaded model that failed while it answered, VRAM mostly) and `vlm_model_missing`.

**One copy per process, kept per weights** ([`holder.py`](holder.py)): `grounder_for(vlm)` is the build's door, and
only a build switches the weights, unloading the copy it replaces. A command and `load_for` ("Laden") never load
beside a copy of other weights: they refuse with "rebuild the cell" (`VlmCopyConflictError`). A cell that detects
with the VLM loads the copy at its first command; any other cell refuses a command until a person loads it, and
the console's build of such a cell calls `release_unless_requested(vlm)`, which unloads a copy a command or another
cell loaded and keeps one a person loaded (`vlm_detects(models)` says which kind a cell is). A routed build lets
a copy of other weights go itself (`release_other_than`). `unload()` hands the memory back: 0.01 GB stays
allocated. `forget()` drops what the holder knows and unloads nothing: a test's teardown.

## Traps

**The checkpoint shares the card.** The model sits beside the segmenter and, in simulation, the renderer.
The 4B checkpoint in bf16 held 8.93 GB of VRAM on an RTX 5080; the 8B checkpoint's bf16 weights,
17.5 GB, do not fit that card, so the two were not compared there, and the FP8 variants were not measured.
Measured on that card against the real 4B weights: a load of 6.3 to 7.3 s, a peak of 9.8 to 9.9 GB while a
command is answered, about 2.2 s per command on a free card and 5.5 to 19 s while another GPU job runs, answers
of 53 to 73 tokens. The cell PC's card is unmeasured, beside SAM2, Whisper, the cuRobo sidecar and the desktop.

**The `on_unavailable` contract does not fire through the composed stack.** `TwoStageBackend` turns a detector
that raises, a VLM that cannot load included, into "no objects" before `GuardedVlmBackend` sees it, and counts it
(`failures`, `last_failure`): a pick reads nothing found, and the console's task ends `detector_failed` on the
count instead of "nothing left". Whether the guard should see the load error first is open.

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
| The route on a real cell's camera | run on a physical cell: a wrist D415 on a UR10 (CB3); no measurement is kept here |
| The command reader against the real 4B weights: German part nouns translated, a spoken label read back as the pose's name, the load and the VRAM | measured on the development box's RTX 5080 (`tests/test_vlm_command_inference.py`); not on the cell PC |

The parser, the coordinate space and the unavailability contract are also tested without a GPU, and no
module here imports torch or transformers at import time, so a cell that never sends a hard prompt never
pays for the model. `tests/test_vlm_inference.py` needs CUDA and the weights, and skips without them.

## Files

| File | Holds |
| --- | --- |
| [`qwen.py`](qwen.py) | `Qwen3VLGrounder`: `detect_all(bgr, prompt)`, the shape GroundingDINO has; `answer_text`, a text-only answer; a load guarded by a lock, one inference at a time, `unload()`; loads on first call unless `preload` |
| [`holder.py`](holder.py) | `VlmHolder` and `shared_vlm()`: the process's one copy, shared by detection and the command reader; `grounder_for`, `status_for`, `load_for`, `asker_for`, `release_unless_requested`, `forget` |
| [`command.py`](command.py) | the command reader: `understand`, `read_command`, `COMMAND_INSTRUCTION`, the answer model, the checks, the retry and the notes |
| [`parsing.py`](parsing.py) | `parse_grounding_response`, `CoordinateSpace`, `VLM_NOMINAL_SCORE`; runs without a GPU |
| [`availability.py`](availability.py) | `GuardedVlmBackend` and `VlmUnavailableError`; the fallback is built only when needed; `reader_availability`, whether a command can be read now, and the reader's refusals |

## Details

- [`../routing/`](../routing/README.md) decides which route a prompt needs, before any weights load
- [`../README.md`](../README.md) is the perception layer and its factory
- Weights: `python scripts/model_weights/fetch.py vlm-4b`
- The console's commands: [`api/README.md`](../../../api/README.md), Commands
- Tests: `tests/test_vlm_grounding.py`, `tests/test_vlm_inference.py`, `tests/test_vlm_command.py`,
  `tests/test_vlm_holder.py`, `tests/test_vlm_command_inference.py` (real weights, skips without them),
  `tests/test_a_failed_detector_is_not_nothing_found.py`, `tests/test_a_clean_holder_loads_nothing_at_build.py`
