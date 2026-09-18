# The perception models

You have a validating config tree ([01-configuration.md](01-configuration.md)). This guide gets the
vision half of the stack to load its weights and hand back a box and a mask for a prompt. It covers
[`src/models/`](../../src/models/README.md) and [`config/models/`](../../config/models/). Sections 1
to 7 need no GPU and no simulator; section 8 is the first simulator boot.

```python
from willy import PerceptionSpec, load_tree

spec = PerceptionSpec.from_config(load_tree().app_config.models)
print(spec.resolve())   # what build() would construct, and which config half decided it; no weight loads
```

`spec.build()` then loads the weights and returns an object with `perceive(image_bgr, prompt)`.
[`examples/offline/perception/resolve_perception_stack.py`](../../examples/offline/perception/resolve_perception_stack.py)
runs both halves on a drawn frame.

---

## 1. What is here, and what the pick loop consumes

The model families are listed in [`src/models/README.md`](../../src/models/README.md): GroundingDINO
zero-shot detection, RT-DETR closed-set detection, SAM2 and OneFormer segmentation, the Qwen3-VL
grounder, MediaPipe hand and gesture, and Whisper speech to text. This guide is about which of them a
given config builds.

Two seams matter.

**`PerceptionBackend`** is prompt in, grounded objects out: `perceive(image_bgr, prompt)` returns
`PerceivedObject`s, each carrying a `Detection` and a `SegmentationResult`. It lives in
[`perception_backend.py`](../../src/models/perception_backend.py) and imports no torch.
`PerceptionSpec.build()` constructs one through `build_perception` in
[`factory.py`](../../src/models/factory.py).

**`PerceptionSource`** is one step further out: `acquire() -> PerceptionFrame`, with a depth map in
millimetres, a 3x3 camera matrix and a tuple of segmentations
([`src/robot/grasping/types/`](../../src/robot/grasping/types/README.md)). The pick loop consumes this,
never a model. A source owns a camera and usually a backend. The simulator ships two sources and a
depth-noise decorator; the rehearsal uses a synthetic scene; and a real cell uses the live-camera
adapter in [`src/robot/perception/`](../../src/robot/perception/README.md), whose streamer, detector and
segmenter are injected, so it imports with neither `pyrealsense2` nor torch.
`python -m src.robot.perception --prompt "a red cube"` runs that adapter against a real RGB-D camera
with no robot. It has only been driven by a fake streamer: never touched hardware.

**There is no depth model here.** Depth comes from the simulator's rendered annotator, from stereo
block matching in [`src/calibration/`](../../src/calibration/README.md), or from an RGB-D stream. See
[03-calibration.md](03-calibration.md).

Which builder a caller uses decides which half of the config it reads (sections 4 and 5). A real cell
builds through `PerceptionSpec.from_config(app_cfg.models).build()`, so it honours `models.pipeline`.
`python -m src.robot.perception` builds through `build_object_detector` and `build_segmenter`, so it
honours only `models.detector` and `models.segmenter_backend`. That is on purpose: the exerciser proves
a camera and two models work before a cell exists.

---

## 2. Install

Two files hold the whole dependency set. There are no optional extras.

| File | For |
|---|---|
| [`requirements.txt`](../../requirements.txt) | the supported path: torch and torchvision from the CUDA 12.8 wheel index |
| [`requirements-cpu.txt`](../../requirements-cpu.txt) | a host that cannot take the CUDA wheels |

They differ in three lines: the index URL and the two torch pins. Prefer the CUDA file even on a
machine with no card. Those wheels install and import fine, and torch reports
`cuda.is_available() == False`, which is why CI installs that file.

Nothing creates the virtual environment for you. After the requirements, install the repository
itself with `pip install -e . --no-deps`: it puts `src`, `api`, `datagen` and `willy` on the
environment's path and installs no dependency. Every bare `python`, `pip` and `pytest` below assumes
the environment is active.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # POSIX: source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The `+cu128` local version in the pin matters. `--extra-index-url` adds an index rather than replacing
one, so a plain `torch==2.7.1` is also satisfied by PyPI's CPU build, and the resolver may pick either.
Naming the local version rules the CPU wheel out. The CUDA 12.8 wheels are also what a Blackwell card
needs: they carry `sm_120` kernels, and the older `cu121` wheels do not.

A simulator install ships its own Python, whose pins you do not control. That is fine, and it is why
the weights live in one directory both environments read (section 3).

---

## 3. The weights, and what happens when they are missing

**A fresh checkout has no weights, and one script fetches every one of them.** Every model block ships
`local: True` at a path the fetch script writes. `models.objectdetector.model_path` is
`assets/models/hf/detection/IDEA-Research--grounding-dino-tiny`, and `models.segmenter`, `models.stt`,
`models.rtdetr`, `models.oneformer` and the VLM name their own directories under the same root.

```bash
python scripts/model_weights/fetch.py --list
python scripts/model_weights/fetch.py dino-tiny sam2      # the pair a real-vision pick needs
python scripts/model_weights/fetch.py --mediapipe         # the hand and gesture .task bundles
```

[`scripts/model_weights/fetch.py`](../../scripts/model_weights/fetch.py) takes the keys `dino-tiny`,
`dino-base`, `rtdetr`, `sam2`, `oneformer`, `whisper-turbo`, `silero-vad`, `vlm-2b`, `vlm-4b`, `vlm-8b`
and `vlm-4b-fp8`. `--list` prints each with its approximate size, its pin and what it is for. Exit
codes: `0` everything asked for is present, `1` at least one fetch failed, `2` an unknown key.

**Two models check their directory before they load.** GroundingDINO raises `FileNotFoundError`
naming the key, the configured path, the resolved path, and both ways out:

```
FileNotFoundError: models.objectdetector.local is true and model_path is
'assets/models/hf/detection/IDEA-Research--grounding-dino-tiny', but there is no such
directory (resolved: ...).
Nothing is downloaded in local mode: that is the point of the flag.
```

The Whisper wrapper refuses the same way, naming `models.stt.model_path` and the fetch that fixes it.
**SAM2, OneFormer, RT-DETR and the VLM do not check.** They pass `local_files_only=True` to
`from_pretrained`, which reads the missing path as a Hub repository id and fails as one. So a missing
segmenter surfaces as a library error about repository ids, not as a named config key. Read the
traceback for which model it is.

**What the script does for you:**

- **Everything lands under `assets/models/hf/` inside the repository**, by what the model is for.
  Deleting the checkout deletes the weights with it. The directory is gitignored, so the layout travels
  with a clone and the gigabytes do not. The default location, `~/.cache/huggingface`, outlives every
  checkout and is shared silently between them (12 GB on the development workstation).
- **It drops duplicate formats.** Several repositories publish the same weights as `.safetensors` and
  as `.bin`, and `transformers` reads only what the index names. A per-model ignore list drops the
  duplicate: `openai/whisper-large-v3-turbo` is 1.62 GB filtered, and `IDEA-Research/grounding-dino-tiny`
  is 1.38 GB whole against 0.69 filtered. It cannot be a blanket rule:
  `shi-labs/oneformer_coco_swin_large` publishes no safetensors at all, so a global `*.bin` filter would
  fetch it empty.
- **`silero-vad` is not a Hub repository.** It is one TorchScript file inside a PyPI wheel. The script
  downloads the pinned wheel, checks the wheel's sha256, reads the one member out of it and checks that
  member's sha256. Nothing is installed. The speech entries also carry a Hub commit, so a second fetch
  cannot replace the bytes under a config path without a trace.
- **It fences the Hub cache** with `fence_model_downloads()` from
  [`src/utility/paths.py`](../../src/utility/paths.py), before `huggingface_hub` is imported, because
  the Hub reads its cache locations once at import.

**Fetch with the environment that reaches the network, load with the one that owns the GPU.** Every
simulator vision runner sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before transformers
imports, because a detector that calls the Hub at boot can take the whole cell down with it. The
runners use `setdefault`, so exporting `HF_HUB_OFFLINE=0` overrides them; fetch first instead. Two
interpreters on one machine can also disagree about TLS: behind a proxy that re-signs TLS, the
simulator's interpreter can fail certificate verification where the project environment succeeds.
`truststore` fixes that, is pinned in both requirements files, and the script reports which trust
store it used.

**The `sim` profile changes the detector and segmenter numerics, not their weights.** Its model
overlays inherit the base paths. They load fp32 weights with fp16 autocast (section 6), because the
production `torch_dtype: auto` misses a small overhead cube that the unset path finds, and they lower
the detector threshold. For the rest of this guide, set it:

```powershell
$env:WILLY_PROFILE = "sim"
```

**Unset it before guide 3 or 4** (`Remove-Item Env:\WILLY_PROFILE`, or `unset WILLY_PROFILE` on POSIX).
It is sticky for the whole shell and would change every base-profile transcript in the later guides.
The alternative that is not sticky is `--profile sim`, passed after the subcommand.

---

## 4. `models.pipeline`: the whole stack in one block

This block decides what perception is on a cell. It lives in
[`config/models/object.yaml`](../../config/models/object.yaml), and `build_perception` is the one
function that builds from it. In outline:

```yaml
pipeline:
  kind: zero_shot            # zero_shot = free-text prompt; closed_set = fixed classes
  zero_shot:
    backend: grounded_sam    # grounded_sam = GroundingDINO grounds; vlm = Qwen3-VL grounds
    segmenter: sam2          # sam2 | oneformer
    vlm:
      model_id: "Qwen/Qwen3-VL-4B-Instruct"
      model_path: "assets/models/hf/vlm/Qwen--Qwen3-VL-4B-Instruct"
      local: true
      preload: false         # false = load on the first prompt that reaches the VLM; true = at cell build
      on_unavailable: refuse # refuse = reject the pick; degrade = fall back to the phrase grounder
  router:
    enabled: false
```

Read the live values rather than this outline: `python -m src.config explain models.pipeline.kind`,
and the same for `models.pipeline.zero_shot.backend` and `models.pipeline.router.enabled`.

**`kind`** picks the detector family. `closed_set` needs a `models.rtdetr` block, or the builder
refuses.

**`zero_shot.backend`** picks what grounds the phrase. `grounded_sam` is the default: GroundingDINO
grounds the phrase and the segmenter cuts the mask. `vlm` puts Qwen3-VL in the detector slot instead,
with the same segmenter, and nothing downstream changes. The route exists because **the phrase
grounder does not fail on a prompt it cannot represent**. It returns a confident box for the wrong
object, and nothing downstream (not the gate, not the record, not the operator) can tell that from a
correct answer. Negation, comparatives, relative clauses, quantifiers and non-English wording are that
class of prompt. For the same reason there is no cheap-first cascade: a cascade needs the cheap stage
to fail loudly. **The VLM package has never run against real weights here**, so its grounding quality,
its VRAM cost and the choice between the 4B and 8B checkpoints are yours to measure
([`src/models/vlm/README.md`](../../src/models/vlm/README.md)).

**`router.enabled`** routes each prompt from its text alone, before any weights load: plain English
noun phrases go to the phrase grounder, everything else to the VLM. It is deterministic: the first
matching rule wins, with no model and no image. The reason travels with the pick, so an operator seeing
a slow pick can find out which word chose the expensive route
([`src/models/routing/README.md`](../../src/models/routing/README.md)).

Two combinations are refused at load, each naming the legal one:

- `kind: closed_set` with `router.enabled: true` written out: a closed-set detector answers only from
  its class list and has no free-text route to send anything to.
- `router.enabled: true` with any `zero_shot.backend` other than `vlm`: routing needs somewhere better
  to send a hard prompt.

Both are errors only when you wrote the value. Left unwritten, `router.enabled` becomes `false`, so a
bare `pipeline: {}` is legal.

**`vlm.on_unavailable`** is the one to think about. `refuse`, the default, rejects the pick with a
typed error carrying the cause, so an operator sees whether the weights are missing, a dependency is
absent or the GPU is out of memory. `degrade` falls back to the phrase grounder and warns on every use,
so a run that fell back never looks normal. Degrading produces exactly the failure the route exists to
prevent, and only three causes degrade at all: no dependency, no weights, no VRAM. Any other exception
is a bug and surfaces. `GET /v1/diagnostics/route?prompt=...` previews a prompt with no GPU and no
image: the route, the reason, and whether it could run here ([`api/README.md`](../../api/README.md)).

Nothing ties the segmenter to a backend. Both mask sources take a box the same way, so either works
with either grounding model, and which one segments better is unmeasured here. It is a knob, not a
recommendation. To compare the two routes in simulation, `run_attribute_pick` takes
`--route simple|vlm|auto`.

---

## 5. The leaf blocks

`ModelsConfig` requires `objectdetector`, `segmenter` and `stt`. `handdetect` and `gesturedetect` have
schema defaults; `rtdetr`, `oneformer` and `pipeline` default to `None`. List what exists with
`python -m src.config where "models."` and read one key with `explain`
([01 section 4](01-configuration.md)).

**`models.objectdetector`** and **`models.segmenter`** share the fields `model_path`, `model_id`,
`local` and `optim`, and the detector adds `threshold`. `local` selects which of the first two is the
source. The YAML comments describe each field. Three behaviours they do not:

- `detect()` returns the best box only and raises when nothing clears the threshold; `detect_all()`
  returns an empty list.
- GroundingDINO boxes come back normalised and are scaled to pixels inside the wrapper; RT-DETR's
  boxes are already pixels.
- SAM2 raises on a mask that is empty after post-processing.

**The checkpoint is a per-cell choice.** `grounding-dino-tiny`, the shipped one, grounds sparse scenes
on which `grounding-dino-base` returns nothing at all, and a detector that grounds nothing is a cell
that cannot pick. `grounding-dino-base` is the better one on dense clutter. On a scene with no
detection, try the other checkpoint before lowering `threshold`. To switch, fetch `dino-base` and
point `model_path` at `assets/models/hf/detection/IDEA-Research--grounding-dino-base`: with `local:
true`, `model_id` is not read.

**`models.detector` and `models.segmenter_backend`** assemble a stack by hand, for a checkpoint the
pipeline block does not name. They are read: `build_object_detector` and `build_segmenter` use them,
that is what `python -m src.robot.perception` calls, and `build_perception` falls back to them
whenever `models.pipeline` is absent. What they lack is a cross-check, which is what `pipeline` adds:
no validator relates the two keys, so every detector and segmenter pair builds, including pairs where
the prompt means something different to each half.

**`models.stt`** is Whisper. The operator console reads the section alone (`load_speech_section`) and
keeps one engine for the process, behind `POST /v1/voice/transcribe`, `/v1/voice/talk` and
`/v1/voice/listen`. A recording becomes a text proposal that a person reads and confirms before it
becomes a prompt. The text stays in the language it was spoken in: `task` accepts only `transcribe`,
and `language: auto` lets Whisper detect German or English per recording. The base block is
`local: True`, so a console on a fresh checkout is refused on the first transcription, by name and
with the fetch that fixes it. The microphone keys (`samplerate`, `blocksize`, `channels`, `dtype`) open
the cell PC's microphone for push to talk (`PushToTalkSource.from_config`) and for
`Listener.from_config`, which no console route or cell verb opens. A tree that writes `chunk_duration`
is refused as an unknown key. From a program:
[`examples/real_robot/12_speak_a_command.py`](../../examples/real_robot/12_speak_a_command.py); the
package is [`src/models/speech/README.md`](../../src/models/speech/README.md).

**`handdetect` and `gesturedetect`** are standalone MediaPipe and are not on the grasp path. Nothing
builds them automatically, so their blocks carry no `enabled` flag: a switch would have no reader. The
`.task` bundles come from `fetch.py --mediapipe`, and `python -m src.models.handdetection --check`
says whether this host can run them
([`src/models/handdetection/README.md`](../../src/models/handdetection/README.md)).

---

## 6. `optim`, and the `torch_dtype` trap

`InferenceOptimization` has five fields, each defaulting to the safe or off value: `torch_dtype`,
`attn_implementation`, `channels_last`, `compile`, `compile_mode`. Only `torch_dtype` changes numbers.
`channels_last` is a memory format; `compile` applies only on CUDA and falls back to eager with a
warning on any failure. `python -m src.config where torch_dtype` lists every block that has one.

The gate is one condition in [`src/models/_inference.py`](../../src/models/_inference.py): when
`torch_dtype` is set, the resolved dtype goes to `from_pretrained` **and** becomes the autocast dtype.
On CUDA that gives three different behaviours:

| YAML value | Weights | Autocast compute |
|---|---|---|
| unset (`"__null__"` in the `sim` overlay) | fp32, read from the checkpoint | the CUDA autocast default, fp16 |
| `auto` (the base value) | fp16 | fp16 |
| `"float32"` | fp32 | `autocast(dtype=torch.float32)`, so autocast is effectively off |

**This is the paragraph to remember.** "The detector must run fp32" is true about the weights and
incomplete as an instruction. Writing `torch_dtype: "float32"`, the obvious way to say it, also turns
off the fp16 autocast. The simulator overlays choose fp32 weights with fp16-autocast compute, and the
only way to express that is to leave `torch_dtype` unset. A plain YAML `null` in an overlay keeps the
base's `auto` (01 section 3), so the reset sentinel `"__null__"` is the only way back. Before a vision
run, confirm that the chain ends at `"__null__"`, not `auto`:

```bash
python -m src.config explain models.objectdetector.optim.torch_dtype --profile sim
```

fp16 costs recall on small objects, and `build_load_kwargs` logs a warning whenever it resolves to
fp16. Small means a few tens of pixels across: a 30 mm part under an overhead camera a metre above it.

Two more traps. This project's `"auto"` is not Hugging Face's: here it means fp16 on CUDA and fp32
elsewhere, while Hugging Face's `dtype="auto"` reads the checkpoint, so "unset gives fp32 weights" is
a property of these checkpoints rather than a guarantee of the code. And the trap is CUDA-only: half
dtypes become fp32 on CPU and MPS, and autocast does nothing off CUDA, so a CPU cannot reproduce the
loss. A deprecation warning that `torch_dtype` is renamed `dtype` is harmless; do not rename the
config key. `WILLY_DEVICE` forces the device whatever the config says: `auto`, `cuda`, `cpu` or `mps`.

---

## 7. Prove the stack before you boot the simulator

A simulator boot takes a long time before it can tell you anything. Settle the perception half first,
in two steps.

**Step one needs no weights.** `PerceptionSpec.resolve()` reports what `build()` would construct, and
why, from the same refusals the builder uses:

```bash
python -c "from willy import PerceptionSpec, load_tree; print(PerceptionSpec.from_config(load_tree().app_config.models).resolve())"
```

On the shipped tree it prints the stack, which half of the config decided it, and whether the prompt
router is on:

```
perception stack: zero_shot / groundingdino + sam2
  decided by      : models.pipeline
  prompt router   : off
```

If it says `models.detector` instead of `models.pipeline`, your `pipeline` block is absent and the
legacy keys decide. If it names a refusal, fix that before fetching gigabytes.

**Step two runs the stack on a drawn image.** It needs no simulator, no camera and no image file, and
it builds through the same path a cell does, so it exercises the stack your config selects. Save it
wherever you like:

```python
# smoke_models.py
import numpy as np

from willy import PerceptionSpec, load_tree

spec = PerceptionSpec.from_config(load_tree().app_config.models)
print(spec.resolve())

image = np.full((480, 640, 3), 200, dtype=np.uint8)   # a grey table
image[200:280, 280:360] = (40, 40, 220)              # BGR, so a red square 80 px across
for found in spec.build().perceive(image, "a red cube"):
    box = [round(v, 1) for v in found.detection.box]
    print(found.detection.label, round(found.detection.score, 3), box,
          found.segmentation.mask_area_px, found.segmentation.centroid_xy)
```

```powershell
$env:WILLY_PROFILE = "sim"
$env:HF_HUB_OFFLINE = "1"          # fail fast on missing weights instead of downloading mid-test
python smoke_models.py
```

The square covers `(280, 200)` to `(360, 280)`: 6400 px, centred on `(320.0, 240.0)`. A box near
those corners, a mask area near 6400 px and a centroid on that centre mean the front end works. The
wrappers log their load line; `dtype=None` in it confirms the regime from section 6 on the loaded
object, not only in the YAML. The box, the score and the pixel count are the stable part; timings
vary.

Expect the label to come back as a fragment of the prompt, `red` rather than `a red cube`, because
GroundingDINO grounds sub-phrases. The simulator's vision source maps each label back to the nearest
object name for that reason. Nothing found comes back as an empty tuple, and so does a model error.

CI never runs model inference: CI has no GPU and no weights, and the torch wrappers are left out of
coverage. It runs import checks, the builder guards, and the pipeline, routing and spec logic, all of
which are pure. Tests that need real weights on a real GPU are marked and skipped without them.

---

## 8. The simulator pick

Ground truth before vision. The known-pose runner loads no models, so a failure there is the cell, the
arm, the gripper or the planner, not perception. The simulator refuses to boot without the two motion
engines; a standard `scripts/ext_deps/install.ps1` install needs no environment variables
([04-robot-and-safety.md](04-robot-and-safety.md), [`docs/isaac-ready.md`](../isaac-ready.md)).

```powershell
$env:WILLY_PROFILE = "sim"
python -m src.robot.safety.planning --doctor   # 0 = both load, 1 = degraded, 2 = blocked by OS policy
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_m1_pick --runs 10 > m1.log 2>&1"
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_m2_pick --runs 10 --prompt ""a red cube"" > m2.log 2>&1"
Select-String "GATE:" m1.log, m2.log
```

`--check` reads paths; `--doctor` loads the engines. It runs a real distance query, starts the
planner's own interpreter to see what resolves there, and reports an operating-system
application-control block as exit `2`, not `1`, because a blocked binary and a box with no GPU
environment need opposite fixes.

Run the simulator through `cmd /c`: `cmd` writes UTF-8 logs, while PowerShell redirection writes
UTF-16 and mangles them. Use no compound command, and one simulator process at a time. Both runners
print one line per pick carrying `succeeded`, `lift_mm` and `passed`, then a gate line with how many
of the N passed. A run passes only when `pick()` reports success and, measured separately, the
object's world Z rose by at least `robot.sim.gate.lift_threshold_mm`, which the tree sets to `50.0`.
The campaign passes at `int(pass_fraction * runs)`, with `pass_fraction` set to `0.8`.

If the known-pose run fails, go to [04-robot-and-safety.md](04-robot-and-safety.md). If it passes and
the real-vision run finds nothing, it is perception: check the `torch_dtype` winner (section 6), the
weights (section 3), and the near clip below.

Two preconditions the runners handle, and that any source you write must reproduce:

- **Park the arm out of the camera's view before acquiring.** An overhead camera sees an arm that is
  over the workspace.
- **Set the near clip.** A simulator camera defaults to a 1.0 m near plane. Anything closer renders a
  black RGB image while depth still reports geometry, which looks exactly like a broken detector. The
  wrist camera and the oblique cameras carry an explicit `near_clip_m`. The overhead camera does not,
  on purpose: on that geometry a clean instance mask and an object-bearing rendered depth cannot both
  hold, so the ground-truth runners keep the default and the real-vision path sets its own clip in
  code. Read the clipping range back off the camera before you believe a value you wrote.

What happens after the mask is [04-robot-and-safety.md](04-robot-and-safety.md) and
[05-pick-loop.md](05-pick-loop.md). The default pick is open-loop: the decision gate, the closed-loop
refine, verify and recover path, fusion with its commit gate, the rerank stage, the learned success
model and the reinforcement-learning layer are all built and default to `enabled: false`, and the
simulator runners turn them on per flag in runner code.

**Training your own closed-set detector.**
[`src/models/detection/closed_set/train.py`](../../src/models/detection/closed_set/train.py) fine-tunes
RT-DETR from COCO annotations and writes a provenance manifest beside the checkpoint. Its
classification head is built from the dataset's categories, not COCO's, so any classes work. The
checkpoint drops into the inference path through `models.detector: "rtdetr"` and a
`models.rtdetr.model_path`. No dataset ships here, and no model has been trained in this repository.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: models.objectdetector.local is true ...` | `local: True` and the directory is absent | fetch it (section 3), or set `local: false` with a Hub id |
| `SpeechModelMissing: models.stt.model_path is ...` | the Whisper directory is absent | `python scripts/model_weights/fetch.py whisper-turbo` |
| A `from_pretrained` error about repository ids, naming no key | SAM2, OneFormer, RT-DETR or the VLM, which do not check | fetch that model (section 3) |
| A Hub-offline error on the first simulator run | the runners set `HF_HUB_OFFLINE=1` on purpose | fetch outside the simulator; do not export `HF_HUB_OFFLINE=0` |
| `CERTIFICATE_VERIFY_FAILED` while fetching | a proxy re-signs TLS with a CA that `certifi` does not know | fetch from the project environment, with `truststore` installed |
| `ValueError: No object detected with description: ...` | nothing cleared `threshold`; `detect()` raises by design | rephrase, try the other checkpoint, lower the threshold, or use `detect_all()` |
| Small-object recall collapses after a "make it fp32" edit | `torch_dtype: "float32"` also turns off fp16 autocast | use `"__null__"`, not `"float32"` and not `null` (section 6) |
| The label is a fragment of the prompt | GroundingDINO grounds sub-phrases | map it back to your object name, as the simulator's source does |
| The wrong object is lifted and reported as a success | the phrase grounder fails confidently on complex prompts | the VLM route (section 4) |
| An edit to `models.pipeline` changed nothing | that caller builds through the legacy keys | check which builder it uses (section 1), and `PerceptionSpec.resolve()` |
| Black RGB from a simulator camera while depth looks fine | the 1.0 m default near plane | `near_clip_m` in the simulator camera config (section 8) |

---

## 10. What is settled and what is not

**Pure logic, pinned by the test suite on a machine with no GPU:** the builder guards and their
refusal messages, the prompt router's rules, `PerceptionSpec.resolve()` agreeing with what `build()`
constructs, and the VLM response parser with its coordinate-space contract.

| Capability | Evidence |
|---|---|
| GroundingDINO and SAM2 on rendered images | measured in simulation: the real-vision pick in section 8 |
| RT-DETR, OneFormer, the MediaPipe detectors, the RT-DETR training script | never touched hardware: unit tests only; SAM2 against OneFormer is not compared |
| Whisper and the Silero voice detector | never touched hardware: fakes, a random Whisper, and the real weights for load time, latency and memory only |
| Every model against a physical camera, and the live RGB-D adapter | never touched hardware: the adapter has seen only a fake streamer |
| A learned depth model | does not exist |

No recording of a spoken command exists here to be right or wrong about, so speech accuracy is
unmeasured ([`src/models/speech/README.md`](../../src/models/speech/README.md)).

Before you start guide 3 or 4: `Remove-Item Env:\WILLY_PROFILE` (POSIX: `unset WILLY_PROFILE`).

## See also

- [01-configuration.md](01-configuration.md): profiles and the `"__null__"` sentinel
- [03-calibration.md](03-calibration.md): the camera-to-base transform
- [04-robot-and-safety.md](04-robot-and-safety.md), what the mask feeds, and
  [05-pick-loop.md](05-pick-loop.md), the loop and the stages that are off by default
- [`src/models/README.md`](../../src/models/README.md): the package README
- [`routing/`](../../src/models/routing/README.md) and [`vlm/`](../../src/models/vlm/README.md): the
  two-route perception decision
- [`src/robot/perception/README.md`](../../src/robot/perception/README.md): the live-camera adapter
- [`docs/isaac-ready.md`](../isaac-ready.md): the workstation setup for section 8
