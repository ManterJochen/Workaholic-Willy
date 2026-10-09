# The vision-language route for hard prompts (`src/models/vlm`)

A vision-language model (Qwen3-VL) used as a detector: it reads a prompt a phrase grounder cannot
represent, such as a negation, a comparison or German, and answers with boxes. Boxes are what the
two-stage backend already hands to the segmenter, so this replaces the detector stage and nothing
downstream changes. A cell switches it on in config; you rarely construct it yourself.

```python
from willy import ObjectDetector, load_tree

tree = load_tree().with_values({"models.pipeline.zero_shot.backend": "vlm"})   # or set it in object.yaml
detector = ObjectDetector.from_config(tree)                                     # Qwen3-VL grounds, SAM2 cuts
found = detector.detect(image_bgr, prompt="the broken part", segment=True)     # no score: a VLM writes boxes
```

Built by hand, it is the same two stages:

```python
from src.models.perception_backend import TwoStageBackend
from src.models.vlm import Qwen3VLGrounder

backend = TwoStageBackend(detector=Qwen3VLGrounder(model_id="Qwen/Qwen3-VL-4B-Instruct"), segmenter=segmenter)
```

[`routing/`](../routing/README.md) decides which prompts reach this route when both are configured, and
[detect_with_a_prompt.py](../../../examples/offline/perception/detect_with_a_prompt.py) takes `vlm` or `router` as
its third argument.

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
| a class list's object boxed under two descriptions, or a box named by none | relabelled `ambiguous` | no rule takes a part by guess |

**The coordinate space is declared, never inferred.** `CoordinateSpace.GRID_1000`, the default, is the
0 to 1000 grid Qwen3-VL emits; `CoordinateSpace.ABSOLUTE` is pixels of the image as sent. A wrong space
is silent: on a 1280x720 frame, grid values are in range and sanely sized, pass every check, and send
the gripper to the wrong place. So there is no rule such as "a value past the image width means a grid".

`VLM_NOMINAL_SCORE` is `1.0` and is not a confidence, because a grounding VLM emits none: it means the
model asserted the box, and it keeps a detector's box threshold from discarding the route's output. The
model's own order is kept as its only ranking. The prompt tells the model: "If no object matches,
return an empty array []. Do not guess."

**Every box is logged with what the model called it** (`VLM grounded 3 box(es) for 'each separate grey cube' ...:
'grey cube' at [x0, y0, x1, y1]; ...`). Until 2026-10-08 only the count was, and what the model called the parts of
another colour it boxed at the presentation could not be read afterwards.

**One colour word.** `name_colour(bgr, box)` is what the camera source asks where a part's pixels leave its colour
unsure (`src/robot/perception/colour_check.py`): the box grown by 15 %, scaled to a long side of 224 px (about 50
image tokens), and "What colour is the object in the middle of this picture? Answer with one colour word.", greedy, at
most 3 new tokens, under the same generate lock. A colour word and not a yes or a no, which an instruct model gives too
easily. Its time on the cell's 8B is unmeasured (about 0.3 to 1.5 s estimated).

**Following asks no VLM.** A task that follows its parts (`robot.grasping.follow_parts`) has the boxes of the parts the
VLM grounded cut by SAM2 alone: `GuardedVlmBackend.segment_boxes` forwards to the VLM route's segmenter and never loads
the VLM, and a degraded guard cuts nothing, so the frame is grounded by the fallback, warned about as ever. The VLM stays
the grounding of record: the first pick of a task, every trigger and every end check ask it.

**One call for several kinds** (the sorting package, the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die
blaue"). A class list, `class_list_prompt(["green part", "red part"])` or `"green part | red part"`, grounds every rule's
kind of part, or every bin a task looks for, in one call: it is asked with `GROUNDING_CLASS_LIST_INSTRUCTION`, each box
under the one description it matches best, copied exactly, and `classes_of(prompt)` reads the descriptions back for
whoever tells the boxes apart by label (the camera source's object labels, the locator's `Located.split`). A prompt of
one description is asked with `GROUNDING_INSTRUCTION`, byte for byte as before. The answer fails closed
(`parsing.claimed_once`): boxes that overlap by an IoU of 0.7 or more (`AMBIGUOUS_IOU`), one with the next as a chain,
are one object, and where that object carries two descriptions every one of its boxes is `ambiguous`
(`AMBIGUOUS_LABEL`), as is a box the model named by none. `ambiguous` is no description's: no rule takes such a part, and
it stays an obstacle. A description that holds the bar or the word `ambiguous` is refused by `class_list_prompt`.
GroundingDINO gets a class list as a caption of one phrase each (`"green part . red part"`) and its boxes pass the same
rule, and the router routes each description on its own ([routing/](../routing/README.md)). How well the cell's 8B, or
the 4B, keeps two colours apart in one call is not measured: `tests/test_vlm_class_list_inference.py` is written and has
not run.

## Reading a command

[`command.py`](command.py) reads an operator's sentence, German or English, into a task card, and moves
nothing: `understand(text, ask=, poses=)` calls `ask` and nothing else, and no robot, camera or console module is
imported. The VLM reads every sentence but the everyday ones, which a closed grammar answers as the model was
measured to answer them ([`known.py`](known.py), below); no other word list reads a sentence.

- **The question.** The whole sentence goes to the model with one fixed instruction (`COMMAND_INSTRUCTION`,
  1,533 tokens) and the taught poses as spoken label to name pairs, so a spoken "Ablage links" comes back as its
  name. `Qwen3VLGrounder.answer_text(system, user)` asks it with no image, the answer prefilled with `{`, at
  most 160 new tokens, greedy.
- **The answer writes only what was said** (2026-10-08). On the cell every answer token costs about 140 ms and the
  instruction almost nothing, so the answer is one compact JSON object with `intent` and only the keys the sentence
  fills: "Hallo Willy" is `{"intent":"none"}`, 6 tokens where every key written was 53. A key left out is "not
  said", never "all"; a full answer in the earlier format reads the same. Two keys narrow what is picked: `which`,
  the one part a sentence singles out ("den grauen Würfel, der oben auf dem anderen liegt" -> "the gray cube on top
  of the other one"), which the task's detector is asked alone, and `from`, where the parts lie ("von der schwarzen
  Matte" -> "on the black mat"). A size stays in the object ("large gray cube"); a superlative is a which.
- **Hard checks, one retry.** The answer must be one JSON object with `intent` and no key the instruction does not
  name. A missing intent or an extra key, a scope other than `once` and `until_empty`, a place and a pose that
  contradict each other, a which longer than 120 characters or without the object's noun, an object that holds
  where the parts lie, or a phrase for the detector (object, from, which, place) that is not English or carries a
  quantifier is asked again once, saying what was wrong;
  a German command's phrase made of the command's own words alone was copied, not translated. Word lists check
  the answer, never the sentence: `GERMAN_WORDS` (German part, colour and material words the router reads as
  English), `SHARED_WORDS` (words English shares, such as "Box", which are no copy) and `ANY_PART_WORDS` (a word
  for any part names none, so the card asks what to pick). A second bad answer is "not understood", and the card
  is filled in by hand.
- **Read narrowly.** A place whose own words are a taught pose's label is that pose, with no second question. A
  place said as Home is no place: Home is where the arm returns, and the note says so. A counted command ("nimm
  drei Würfel") is read as `once` with the note that a count is not supported, and a command that singles out one
  part as `once` too.
- **Soft notes**, in `NOTE_ORDER`: `object_not_in_sentence`, `place_not_in_sentence`, `pose_unknown`,
  `count_not_supported`, `retried`. The card marks them "bitte prüfen". A which and a from are found in the
  sentence through the operator's words for the part, which hold them; where those words are not there, the note
  is `object_not_in_sentence`.
- **`read_command`** is the console's door: with `known=True` (the console's `runtime.commands.known_sentences`,
  on) the known sentences first, before the availability rule, so a known sentence is read before "Laden" and on a
  box without the weights; then the availability rule (`reader_availability`), then the reader through the shared
  copy, answering a question it was asked before from memory. A sentence over `MAX_SENTENCE_CHARS` (1000) is
  refused before anything is asked. Its refusals are the console's codes: `vlm_not_loaded`, `vlm_unavailable` (also
  `VlmAnswerFailedError`, a loaded model that failed while it answered, VRAM mostly) and `vlm_model_missing`.
- **`startable`.** A reading says whether the console may start its task on the person's Enter, with no card to
  look at first (`CommandReading.startable`, the owner, 2026-10-08): an understood task with no note (no retry,
  every word found, every pose taught, no count), its part named and found in the sentence, and its place, where one
  is named, found there too. Every other reading opens the card, which errs narrow as ever: a part named by no word,
  or a place nobody can check, waits for a person. A task starts at the console's `POST /v1/task` either way.

**Known sentences** ([`known.py`](known.py), the owner's "speed first", 2026-10-08). On the cell every answer token
costs about 140 ms, so the morning's "Alle grauen Würfel in die Gelbe Kiste." took 9.8 s to read and every "Hallo
Willy" 7.2 s. `read_known(text, poses=)` answers a closed grammar of such sentences with the answer the model gives
for them, in the words it was measured to choose, and the reader's own code makes the card of that answer, with the
same checks and notes: a verb or none, "alle"/"all" or a definite article, one of five colours (rot, blau, gelb,
grün and grau with their endings, read as red, blue, yellow, green and gray, as the model wrote them), a measured
part word (Würfel, Schraube, Mutter, Becher, Zylinder), and a place said as where the part goes (a verb that puts,
"und leg ihn" or "and put it", "into", "onto", "ins" or a German accusative article) that is a taught pose's label
or name or a box (Kiste, Box, Kasten, bin). A sentence that is nothing but a greeting is a greeting. Everything else
goes to the model: a count ("einen Würfel" and "a cube" included), a negation, Home as the place, where the parts
lie ("in der Kiste", "auf der schwarzen Matte"), a plural without "alle", "bitte", two languages, another word. Its
reading's `model_id` is `known-sentence` (`CommandReading.known`), with no attempt and nothing loaded. Of the 4B's
measured answers in `tests/test_vlm_command.py` the table takes 7, and each reads as the model read it.

**Remembered answers** (the owner's R2, 2026-10-08). The model decodes greedily, so the same weights asked the same
question give the same answer, and a question the loaded copy answered before is answered from memory
(`_Remembering`): the key is the weights, the instruction's sha256 and the user message, which holds the poses and
the sentence, so a pose taught or renamed since, or a new instruction, is a new question. Only the raw answer is
kept, and it is checked as a new one; at most `REMEMBERED_ANSWERS` (256) per copy, the least recently asked first to
go. Memory answers only while the copy that gave the answer is loaded: a VLM cell whose copy is not loaded loads it
at its command, as before. A reading wholly from memory says `remembered`; `forget_remembered_answers()` drops them.

**One copy per process, kept per weights** ([`holder.py`](holder.py)): `grounder_for(vlm)` is the build's door, and
only a build switches the weights, unloading the copy it replaces. A command and `load_for` ("Laden") never load
beside a copy of other weights: they refuse with "rebuild the cell" (`VlmCopyConflictError`). A cell that detects
with the VLM loads the copy at its first command; any other cell refuses a command until a person loads it, and
the console's build of such a cell calls `release_unless_requested(vlm)`, which unloads a copy a command or another
cell loaded and keeps one a person loaded (`vlm_detects(models)` says which kind a cell is). A routed build lets
a copy of other weights go itself (`release_other_than`). `unload()` hands the memory back: 0.01 GB stays
allocated. `forget()` drops what the holder knows and unloads nothing: a test's teardown.

## What an answer costs

Every token of an answer is one pass of the model, and the passes are the time: about 46 ms a pass with the 4B on the
dev box's RTX 5080 (2026-10-09, its CPU busy with other work; 32 ms on a quiet box) and about 140 ms with the cell's
8B, against about 0.25 s to read the image and the instruction. Four keys of the `vlm` block decide how an answer is
written. A build hands them to the process's one copy (`Qwen3VLGrounder.configure`), and so do a command and "Laden"
(`holder._answer_as`), so a cell whose detector is not the VLM, or a routed cell before its first VLM prompt, reads its
commands as its block says too. The owner's cell sets both lookups to 10 (2026-10-08) and keeps the stop at its
default; its decode graphs are delivered `off` and go on (`auto`) once they pass their check on the cell's 8B:

| Key | Default | What it does | Measured with the 4B, 90 calls on the cell's frames |
| --- | --- | --- | --- |
| `stop_at_answer_end` | `true` | a grounding answer ends where its JSON array closes, a command's reading where its object closes | the same boxes and cards, one pass sooner |
| `prompt_lookup_tokens` | `0` | proposes a grounding answer's next tokens from the answer so far and checks them in one pass | 3025 passes instead of 6773; 42 of 90 answers byte-identical, a coordinate's last digit moves |
| `text_prompt_lookup_tokens` | `0` | the same for a command's reading | 67 sentences of the tests and the cell's logs: 1458 passes instead of 2539, 3 readings otherwise (below) |
| `decode_graphs` | `auto` | each pass as CUDA graphs, the attention over the answer as before | every answer byte-identical; 19.7 ms a pass instead of 46.3 |

**The text lookup and today's instruction.** Measured on ten commands with the earlier instruction, every reading was
byte-identical; with the compact one of 2026-10-08, 3 of 67 came out otherwise: "Alle grauen Würfel von der schwarzen
Matte in die gelbe Kiste" said `"object_said":"Alle grauen Würfel von der schwarzen Matte"` where the plain answer
said `"graue Würfel von der schwarzen Matte"` (not in the sentence, so the card asks), "Alle Würfel und Zylinder, in
die gelbe Kiste." likewise with "Alle", and "Leg den Becher in die Kiste neben Ablage links" placed the cup in `"bin
next to Ablage links"` where the plain answer said `"bin"`. So it stays off by default, and a cell that turns it on
reads those sentences differently from the plain model.

**How the lookup proposes** (`_AnswerLookup`, through transformers' assisted decoding, which keeps the proposals the
model itself would write and one token of its own). transformers' own prompt lookup takes the first match from the
left, which in a grounding call lies in the image's span, and an image placeholder proposed there failed every image
call. This one searches after the image only, takes the most recent of the longest matches of the answer's tail
(16 tokens at most) and treats every digit alike, so a box's tail finds the previous box's whatever its numbers and
mostly the coordinates' digits cost a pass each; it never proposes an image placeholder or an end token. A lookup
that fails is logged, switched off for the loaded copy and the question asked again without it; running out of
memory skips it for that question only.

**A pass waits on the CPU, not on the card.** A pass over one new token is about 2400 kernel launches, and the card
runs them faster than the CPU hands them over: the dev box's card was busy 45 % of a plain pass, and the cell's 8B
writes a token in about 140 ms where reading its weights takes about 20. `decode_graphs` hands each stretch between
two layers' attentions to the card as one CUDA graph (37 a pass for the 36 layers), captured once per pass shape: one
token, and each chunk the answer lookup checks. The attention over the answer so far stays as the plain pass runs it,
through the model's own cache and attention function, because its kernels are shaped by the answer's length and this
torch has no flash attention on Windows. A static KV cache would fix the shapes, but it attends over the whole cache
behind a mask, which the math attention sums in another order: 4 in 10 of one layer's outputs moved in their last bit,
which is why the earlier static-cache trial matched only 41 of the 90 answers. Graphs per length would hold about 1 MB
of the card for each layer and length.

So the graphs launch the plain pass's kernels, in its order, on its data, and the answers are byte-identical: all 90
calls with and without the answer lookup, and the command readings. The first pass of each shape is also compared
with the plain pass bit for bit (logits and cache) before graphs answer, and logged (`decode graphs for 1 token(s) a
pass ... bit for bit`). `auto` decodes with graphs on a CUDA card of compute capability 8.0 or newer where that check
matched; `on` on any CUDA card, a check that differed only warned; `off` as before. A capture or a pass that fails
cuts the cache back to before the pass, runs it plainly and turns graphs off until the copy is loaded again; the
decision and every check are logged in `vlm_grounder.log`.

| On the dev box (4B) | ms a pass | ms a token | card busy | CPU a pass |
| --- | --- | --- | --- | --- |
| plain | 46.3 | 45.7 | 45 % | 42.5 ms |
| graphs | 19.7 | 19.4 | 83 % | 10.9 ms |
| plain, lookup 10 | 50.6 | 22.3 | 48 % | 42.2 ms |
| graphs, lookup 10 | 22.9 | 10.1 | 80 % | 8.1 ms |

A command's reading gains less, 32.5 ms a pass instead of 56.3 (24.4 ms a token with the text lookup): its instruction
is about 1,500 tokens long, so the attention over it, which the graphs leave out, is a larger share of each pass.

On the cell the CPU does this work three to four times slower (140 ms a plain pass against 32 to 46 here), so a pass
through the graphs should cost about 33 to 53 ms there (11 to 12 ms of the CPU here), where the 8B's weights take the
card about 27 ms: the 108 passes of three cubes in about 4 to 6 s instead of 15, about 1.5 to 2 s with the lookup.
Estimated from the CPU's share, not measured on the cell; its first answer logs the check and both pass times.

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
| The decode graphs: every answer byte-identical, 19.7 ms a pass instead of 46.3 | measured with the 4B on the development box's RTX 5080, 90 calls on the cell's frames (`tests/test_vlm_decode_graphs_inference.py`); the cell's 8B and card not measured |
| The known sentences: each of the 7 measured sentences the table takes reads as the 4B read it, its near misses go to the model, and nothing is loaded | pinned by tests against the 4B's recorded answers (`tests/test_vlm_command.py`); the 8B's words for the same sentences not measured |
| Remembered answers: the same question to the same loaded copy is answered once | pinned by tests on a stand-in model (`tests/test_vlm_command.py`, `tests/test_api_commands.py`) |
| One call for several kinds: the class-list instruction, one description byte for byte as before, `ambiguous` for an object two descriptions claim | pinned by tests on a stand-in model through the real grounder and parser (`tests/test_vlm_class_list.py`, `tests/test_a_sorts_pick_prompt_passes_only_its_kinds.py`); against real weights not measured (`tests/test_vlm_class_list_inference.py` has not run) |

The parser, the coordinate space and the unavailability contract are also tested without a GPU, and no
module here imports torch or transformers at import time, so a cell that never sends a hard prompt never
pays for the model. `tests/test_vlm_inference.py` needs CUDA and the weights, and skips without them.

## Files

| File | Holds |
| --- | --- |
| [`qwen.py`](qwen.py) | `Qwen3VLGrounder`: `detect_all(bgr, prompt)`, the shape GroundingDINO has; `answer_text`, a text-only answer; `name_colour(bgr, box)`, one colour word; a load guarded by a lock, one inference at a time, `unload()`; loads on first call unless `preload`; how answers are written (`configure`): the stop at the answer's end, the answer lookup, and `_DecodeGraphs`, each decode pass as CUDA graphs; one call for several kinds: `class_list_prompt`, `classes_of`, `grounding_instruction` |
| [`holder.py`](holder.py) | `VlmHolder` and `shared_vlm()`: the process's one copy, shared by detection and the command reader; `grounder_for`, `status_for`, `load_for`, `asker_for`, `release_unless_requested`, `forget` |
| [`command.py`](command.py) | the command reader: `understand`, `read_command`, `COMMAND_INSTRUCTION`, the answer model, the checks, the retry and the notes; `startable`; the remembered answers (`_Remembering`, `forget_remembered_answers`) |
| [`known.py`](known.py) | the known sentences: `read_known`, `known_answer`, the closed grammar and its measured words (`KNOWN_PARTS`, `KNOWN_COLOURS`, `KNOWN_PLACES`); pure, nothing loaded |
| [`parsing.py`](parsing.py) | `parse_grounding_response`, `CoordinateSpace`, `VLM_NOMINAL_SCORE`; a class list's rule, `claimed_once` and `AMBIGUOUS_LABEL`; runs without a GPU |
| [`availability.py`](availability.py) | `GuardedVlmBackend` and `VlmUnavailableError`; the fallback is built only when needed; `reader_availability`, whether a command can be read now, and the reader's refusals |

## Details

- [`../routing/`](../routing/README.md) decides which route a prompt needs, before any weights load
- [`../README.md`](../README.md) is the perception layer and its factory
- Weights: `python scripts/model_weights/fetch.py vlm-4b`
- The console's commands: [`api/README.md`](../../../api/README.md), Commands
- Tests: `tests/test_vlm_grounding.py`, `tests/test_vlm_inference.py`, `tests/test_vlm_command.py`,
  `tests/test_vlm_holder.py`, `tests/test_vlm_command_inference.py` (real weights, skips without them),
  `tests/test_a_failed_detector_is_not_nothing_found.py`, `tests/test_a_clean_holder_loads_nothing_at_build.py`;
  what an answer costs: `tests/test_vlm_answer_speed.py`, `tests/test_vlm_decode_graphs.py` (no card) and their
  `_inference` twins (real weights), and `tests/test_a_command_reads_with_the_vlm_blocks_answer_settings.py`
