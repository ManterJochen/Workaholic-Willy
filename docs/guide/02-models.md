# The perception models

You have a validating config tree ([01-configuration.md](01-configuration.md)) and you want the
vision half of the stack to load weights and hand back a box and a mask. This chapter covers
[`src/models/`](../../src/models/README.md) and [`config/models/`](../../config/models/). Sections 1
to 7 need no GPU and no simulator; section 8 is the first simulator boot. Work from the repository
root, because the repository is not pip-installed and `import src` only resolves from there.

---

## 1. What is here, and what the pick loop consumes

The family list lives in [`src/models/README.md`](../../src/models/README.md): GroundingDINO
zero-shot detection, RT-DETR closed-set detection, SAM2 and OneFormer segmentation, the Qwen3-VL
grounder, MediaPipe hand and gesture, Whisper speech to text. Read it there. This chapter is about
which of them a given config builds.

Two seams matter.

**`PerceptionBackend`** is prompt in, grounded objects out: `perceive(image_bgr, prompt)` returns
`PerceivedObject`s, each carrying a `Detection` and a `SegmentationResult`. It lives in
[`perception_backend.py`](../../src/models/perception_backend.py), imports no torch, and
[`factory.py`](../../src/models/factory.py) builds one from config in `build_perception`.

**`PerceptionSource`** is one step further out: `acquire() -> PerceptionFrame`, with a depth map in
millimetres, a 3x3 camera matrix and a tuple of segmentations
([`src/robot/grasping/types/`](../../src/robot/grasping/types/README.md)). The pick loop consumes
this, never a model. A source owns a camera and usually a backend. Two simulator sources ship, plus
a depth-noise decorator, the synthetic rehearsal scene in `src/robot/execution/autonomous_grasp/cells.py`, and the
live-camera adapter in [`src/robot/perception/`](../../src/robot/perception/README.md), whose
streamer, detector and segmenter are injected so it imports with neither `pyrealsense2` nor torch.
`python -m src.robot.perception --prompt "a red cube"` exercises that one against a real RGB-D camera
with no robot involved; it has only ever been driven by a fake streamer here.

**There is no depth model here.** Depth comes from the simulator's rendered annotator, from stereo
block matching in [`src/calibration/`](../../src/calibration/README.md), or from an RGB-D stream. See
[03-calibration.md](03-calibration.md).

Which builder a caller uses decides which half of the config is read, and the two halves are
described in section 4 and section 5. The physical-cell path,
`build_real_components` in `src/robot/execution/autonomous_grasp/cells.py`, goes through
`PerceptionSpec.from_config(app_cfg.models).build()`, so it honours `models.pipeline`.
`python -m src.robot.perception` goes through `build_object_detector` plus `build_segmenter`, so it
honours only `models.detector` and `models.segmenter_backend`. That is deliberate: the exerciser
exists to prove a camera and two models work before a cell exists.

---

## 2. Install

Two files, and they are the whole dependency set. There are no optional extras, because a package
installed only "if you need it" is one whose absence is discovered on the day it is needed.

| File | For |
|---|---|
| [`requirements.txt`](../../requirements.txt) | the supported path: torch and torchvision from the CUDA 12.8 wheel index |
| [`requirements-cpu.txt`](../../requirements-cpu.txt) | the escape hatch, for a host that cannot take the CUDA wheels |

They differ in three lines, the index URL and the two torch pins. Prefer the CUDA file even on a
machine with no card: those wheels install and import fine, and torch simply reports
`cuda.is_available() == False`, which is why CI installs that file rather than the CPU one.

Nothing creates the virtual environment for you, and the repository is not pip-installable, so there
is no editable-install step either. Every bare `python`, `pip` and `pytest` below assumes the
environment is active.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # POSIX: source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The `+cu128` local version in the pin is load-bearing. `--extra-index-url` adds an index rather than
replacing one, so a plain `torch==2.7.1` is equally satisfiable from PyPI, where it is the CPU build,
and the resolver is free to pick either. Naming the local version makes the CPU wheel a non-match.
The CUDA 12.8 wheels are also what a Blackwell card needs: they carry `sm_120` kernels and the older
`cu121` wheels do not.

A simulator install ships its own bundled Python, whose pins you do not control, and it will not be
exactly these. That is fine, and it is why weights live in a shared cache rather than in either
environment (section 3).

---

## 3. The weights, and what happens when they are missing

**A fresh checkout has no weights.** The base profile sets `local: True` against checkpoint
directories that are not in this repository and never were: `models.objectdetector.model_path` points
at `src/models/detection/model`, and `models.segmenter` and `models.stt` do the same for their own
packages. Either put the files there, or set `local: false` and name a Hub id in `model_id`.

**Only the detector checks.** GroundingDINO raises `FileNotFoundError` before it loads anything,
naming the key, the configured path, the resolved absolute path, and both ways out:

```
FileNotFoundError: models.objectdetector.local is true and model_path is
'src/models/detection/model', but there is no such directory (resolved: ...).
Nothing is downloaded in local mode: that is the point of the flag.
```

The guard exists because, left to `from_pretrained`, the path is taken for a Hub repository id and
rejected as one, with a message about repository-id form that sends an operator hunting a token
problem that does not exist. **The SAM2 segmenter and the Whisper wrapper do not check.** They pass
`local_files_only=True` straight through and fail inside `from_pretrained`, so a missing segmenter or
a missing speech model surfaces as a library error rather than as a named config key. Read the
traceback for which of the three you are looking at.

**The fetch script.** [`scripts/model_weights/fetch.py`](../../scripts/model_weights/fetch.py)
carries a catalogue with sizes and notes. It writes into the standard Hugging Face cache, and nothing
lands in the repository.

```bash
python scripts/model_weights/fetch.py --list
python scripts/model_weights/fetch.py dino-tiny sam2      # the pair a real-vision pick needs
```

Its keys are `dino-tiny`, `dino-base`, `sam2`, `vlm-2b`, `vlm-4b`, `vlm-8b` and `vlm-4b-fp8`, and
`--list` prints each with its approximate download size and what it is for. Exit codes: `0`
everything asked for is in the cache, `1` at least one fetch failed, `2` an unknown key was named.

**Fetch with the environment that can reach the network, load with the one that owns the GPU.** The
cache is shared, and the split matters for two reasons the script's own docstring gives. Every
simulator vision runner sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before transformers
imports, because a detector that pings the hub at boot can take the whole cell down with it; the
runners use `setdefault`, so exporting `HF_HUB_OFFLINE=0` overrides them, and the fix is to fetch
first rather than to override. And two interpreters on one machine can disagree about TLS, where the
one that loads a model fails certificate verification and the project environment succeeds. That
split has a fix, `pip install truststore`, which is pinned in both requirements files; the script
imports it inside a `try` and reports which trust store it used.

Weights land in `~/.cache/huggingface/hub` unless `HF_HOME` or `HF_HUB_CACHE` moves them. One
in-repository cache is deliberate and separate: the paraphraser under `datagen/prompts/` points
`HF_HOME` at `assets/models/hf` so its downloads travel with the clone.

**The overlay that makes the shipped tree loadable is the `sim` profile.** Its model overlays set
`local: false` with a Hub id, so a simulated run fetches on first use instead of reading a directory
that is not there:

```powershell
$env:WILLY_PROFILE = "sim"
```

Everything from here to section 8 assumes that. **Unset it before guide 3 or 4**
(`Remove-Item Env:\WILLY_PROFILE`, POSIX `unset WILLY_PROFILE`): it is sticky for the whole shell and
will silently re-tint every base-profile transcript in the later guides. The non-sticky alternative
is `--profile sim`, passed after the subcommand.

---

## 4. `models.pipeline`: the whole stack in one block

This is the block that decides what perception is on a cell. It lives in
[`config/models/object.yaml`](../../config/models/object.yaml), and `build_perception` is the one
function that builds from it.

```yaml
pipeline:
  kind: zero_shot          # zero_shot = free-text prompt | closed_set = fixed classes
  zero_shot:
    backend: grounded_sam  # grounded_sam = GroundingDINO grounds | vlm = Qwen3-VL grounds
    segmenter: sam2        # sam2 | oneformer
    vlm:
      model_id: "Qwen/Qwen3-VL-4B-Instruct"
      preload: false       # false = load on the first prompt that reaches the VLM | true = at cell build
      on_unavailable: refuse   # refuse | degrade
  router:
    enabled: false
```

Read the live values rather than trusting that snippet, with
`python -m src.config explain models.pipeline.kind` and the same for
`models.pipeline.zero_shot.backend` and `models.pipeline.router.enabled`.

**`kind`** picks the detector family, and `closed_set` needs a `models.rtdetr` block or the builder
refuses.

**`zero_shot.backend`** is the interesting one. `grounded_sam` is the default: GroundingDINO grounds
the phrase, the segmenter cuts the mask. `vlm` puts Qwen3-VL in the detector slot instead, so the
same two-stage chain composes it with the same segmenter and nothing downstream changes. The route
exists because of a specific failure: **the phrase grounder does not fail on a prompt it cannot
represent.** It returns a confident, high-scoring box for the wrong object, and nothing downstream,
not the gate, not the record, not the operator, can tell that apart from a correct answer. Negation,
comparatives, relative clauses, quantifiers and non-English wording are exactly that class of prompt.
That is also why there is no cheap-first cascade: a cascade needs the cheap stage to fail loudly, and
there is no signal to fall back on. **Nothing in the VLM package has run against real weights here**,
so its grounding quality, its VRAM cost and the choice between the 4B and 8B checkpoints are yours to
measure ([`src/models/vlm/README.md`](../../src/models/vlm/README.md)).

**`router.enabled`** decides each prompt from its text alone, before any weights load: plain English
noun phrases to the phrase grounder, everything else to the VLM. It is deterministic, first matching
rule wins, no model and no image, and the reason travels with the pick so an operator seeing a slow
one can find out which word chose the expensive route. Rules and reasons:
[`src/models/routing/README.md`](../../src/models/routing/README.md).

Two combinations are refused at load, each naming the legal one: `kind: closed_set` with an explicit
`router.enabled: true`, because a closed-set detector answers only from its class list and has no
free-text route to send anything to; and `router.enabled: true` with any `zero_shot.backend` other
than `vlm`, because routing needs somewhere better to send a hard prompt. Both are errors only when
you wrote the value. Left unwritten, `router.enabled` is corrected to `false` rather than refused, so
a bare `pipeline: {}` is legal.

**`vlm.on_unavailable`** is the one to think about. `refuse` rejects the pick with a typed error
carrying the underlying cause, so an operator sees whether the weights are missing, a dependency is
absent, or the GPU is out of VRAM. `degrade` falls back to the phrase grounder and warns on every use
rather than once, so a run that has fallen back never looks normal again. `refuse` is the default,
because degrading produces exactly the failure the route exists to prevent, and only three shapes of
a missing model degrade at all: no dependency, no weights, no VRAM. Any other exception is a real bug
and is left to surface. `GET /v1/diagnostics/route?prompt=...` previews a prompt with no GPU and no
image: the route, the reason, and whether it could run here
([`api/README.md`](../../api/README.md)).

Nothing pins the segmenter to a backend. Both mask sources implement the same box-prompted contract,
so either works with either grounding model, and which one segments better is unmeasured here. It is
a knob, not a recommendation. To compare the two routes yourself, `run_attribute_pick.py` takes
`--route simple|vlm|auto`.

---

## 5. The leaf blocks

`ModelsConfig` requires `objectdetector`, `segmenter` and `stt`. `handdetect` and `gesturedetect`
have schema defaults; `rtdetr`, `oneformer` and `pipeline` default to `None`. List what exists with
`python -m src.config where "models."` and read one key with `explain`. Both commands, and the rule
that shared flags work on either side of the subcommand, belong to
[01 section 4](01-configuration.md).

**`models.objectdetector`** and **`models.segmenter`** share the shape `model_path`, `model_id`,
`local`, `optim`, with the detector adding `threshold`; `local` selects which of the first two is the
source. The YAML carries the field comments, so read them rather than a table here. Three behaviours
it will not tell you: `detect()` returns the argmax only and raises when nothing clears the
threshold, while `detect_all()` returns an empty list; GroundingDINO boxes come back normalised and
are scaled to pixels inside the wrapper, whereas RT-DETR passes `target_sizes` and its boxes are
already pixels; and SAM2 raises on an empty mask after post-processing.

The checkpoint is a per-cell choice rather than a fixed best. The `sim` overlay names the tiny
GroundingDINO checkpoint, because the base checkpoint returns nothing at all on sparse scenes and a
detector that grounds nothing is a cell that cannot pick; the base checkpoint is the better one on
dense clutter, which is why `run_dense_pick` overrides the field for its own scenes through
`WILLY_YCB_DETECTOR`. On a scene that yields no detection, try the other checkpoint before lowering
`threshold`. Note that a config-only audit will mis-report which weights that runner ran.

**`models.detector` and `models.segmenter_backend`** are the do-it-yourself path, kept because
assembling a stack by hand is legitimate: a caller may need a checkpoint the pipeline block does not
name. They are not inert. `build_object_detector` and `build_segmenter` read them, that is what
`python -m src.robot.perception` calls, and `build_perception` falls back to them whenever
`models.pipeline` is absent, byte-identically. What they lack is a cross-check, which is what
`pipeline` adds: the two keys have no validator between them, so every detector and segmenter
combination builds, including ones where the prompt means something different to each half.

**`models.stt`** is Whisper, and it has a caller: the operator console passes `cfg.models.stt` into
`WhisperSpeechToText` to turn a recording into a prompt, which a human reads before pressing the
button. Its base block is `local: True` against an absent directory, and it does not check, so a
base-profile console fails inside `from_pretrained` on the first transcription.

**`handdetect` and `gesturedetect`** are standalone MediaPipe and are not on the grasp path. Nothing
auto-builds them, which is why their config blocks carry no `enabled` flag: a switch would have no
reader. The `.task` bundles are an operator download whose URLs are in the comments of
`config/models/hand.yaml`, and `python -m src.models.handdetection --check` says whether this host
can run them ([`src/models/handdetection/README.md`](../../src/models/handdetection/README.md)).

---

## 6. `optim`, and the `torch_dtype` trap

`InferenceOptimization` has five fields, each defaulting to the safe or off value: `torch_dtype`,
`attn_implementation`, `channels_last`, `compile`, `compile_mode`. Only `torch_dtype` changes
numbers. `channels_last` is a memory format and numerically inert; `compile` is applied only on CUDA
and falls back to eager with a warning on any failure.
`python -m src.config where torch_dtype` lists every block that has one.

The gate is a single condition in [`src/models/_inference.py`](../../src/models/_inference.py): when
`torch_dtype` is not `None`, the resolved dtype goes to `from_pretrained` **and** is reused as the
autocast dtype. On CUDA that gives three genuinely different behaviours.

| YAML value | Weights | Autocast compute |
|---|---|---|
| unset (`"__null__"` in the `sim` overlay) | fp32, read from the checkpoint | the CUDA autocast default, fp16 |
| `auto` (the base value) | fp16 | fp16 |
| `"float32"` | fp32 | `autocast(dtype=torch.float32)`, so autocast effectively off |

**This is the paragraph to remember.** "The detector must run fp32" is true about the weights and
dangerously incomplete as an instruction. Writing `torch_dtype: "float32"`, the obvious way to say
it, also kills the fp16 autocast. The configuration the simulator overlays choose is fp32 weights
with fp16-autocast compute, and the only way to express that is to leave `torch_dtype` unset. A plain
YAML `null` in an overlay is treated as "no value" by the deep merge and would keep the base's
`auto`, so the loader's reset sentinel `"__null__"` is the only way back. Confirm before a vision run
that the chain ends at `"__null__"`, not `auto`:

```bash
python -m src.config explain models.objectdetector.optim.torch_dtype --profile sim
```

Why it matters: fp16 costs recall on small objects, and `build_load_kwargs` warns whenever it
resolves to fp16 for exactly that reason. Small here means a few tens of pixels across, which is what
a 30 mm part under an overhead camera a metre above it comes to.

Two sub-traps. This project's `"auto"` is not Hugging Face's: here it means fp16 on CUDA and fp32
elsewhere, while Hugging Face's `dtype="auto"` means read the checkpoint, so "unset gives fp32
weights" is a property of these particular checkpoints rather than a guarantee of the code. And the
trap is CUDA-only, because half dtypes downgrade to fp32 on CPU and MPS and the autocast context is a
no-op off CUDA, so you cannot reproduce the regression on a CPU. A deprecation warning about
`torch_dtype` being renamed `dtype` is harmless; do not rename the config key. `WILLY_DEVICE` forces
the device independently of any config: `auto`, `cuda`, `cpu` or `mps`.

---

## 7. Prove the stack before you boot the simulator

A simulator boot costs a lot of seconds before it can tell you anything. Settle the perception half
first, in two steps.

**Step one needs no weights at all.** `PerceptionSpec.resolve()` reports what `build()` would
construct, and why, from the same refusal constants the builder uses:

```bash
python -c "from src.config import load_config; from src.models.perception_spec import PerceptionSpec; print(PerceptionSpec.from_config(load_config().models).resolve().render())"
```

On the shipped tree that prints the stack, which half of the config decided it, and whether the
prompt router is on:

```
perception stack: zero_shot / groundingdino + sam2
  decided by      : models.pipeline
  prompt router   : off
```

If that line says `models.detector` instead of `models.pipeline`, your `pipeline` block is absent and
the legacy keys are deciding. If it names a refusal, fix that before fetching gigabytes.

**Step two runs the stack on a synthetic image.** This needs no simulator, no camera and no image
file, and it goes through the factory, so it exercises the stack your config selects rather than a
hand-wired pair. The repository does not ship this file; save it wherever you like.

```python
# smoke_models.py
import numpy as np, cv2
from src.config import load_config
from src.models.factory import build_perception

img = np.full((480, 640, 3), 200, np.uint8)                     # grey table
cv2.rectangle(img, (280, 200), (360, 280), (40, 40, 220), -1)   # BGR red square, 80x80 px
for o in build_perception(load_config().models).perceive(img, "a red cube"):
    print(o.detection.label, round(o.detection.score, 3), [round(v, 1) for v in o.detection.box],
          int(o.segmentation.mask.sum()), tuple(round(float(v), 1) for v in o.segmentation.centroid_xy))
```

```powershell
$env:WILLY_PROFILE = "sim"
$env:HF_HUB_OFFLINE = "1"          # fail fast on a missing cache instead of pulling gigabytes mid-test
$env:PYTHONPATH = (Get-Location).Path
python smoke_models.py
```

The drawn square is 6400 px at `(280,200)-(360,280)` and its centre is `(320.0, 240.0)`, so a box
near those corners, a mask area near 6400 px and a centroid on that centre mean the front end works.
The wrappers log the load line themselves; `dtype=None` in it confirms the regime from section 6 at
the object level rather than only in the YAML. Timings vary between runs, and the box, the score and
the pixel count are the stable part.

One thing to expect: the label comes back as a fragment of the prompt, `red` rather than
`a red cube`, because GroundingDINO grounds sub-phrases. That is why the simulator's vision source
normalises each detection's label to the nearest canonical spawn name before the orchestrator matches
on exact equality.

CI never executes a line of model inference: the torch wrappers sit in the coverage omit list,
because CI has neither a GPU nor weights. What runs there is import smoke, the builder guards, and
the pipeline, routing and spec logic, all of which are pure. Anything that needs real weights on a
real GPU is marked and skipped without them.

---

## 8. The simulator pick

Ground truth before vision. The known-pose runner loads no models at all, so a failure there is the
cell, the arm, the gripper or the planner, and not perception. First anchor the two motion sidecars,
because the simulator refuses to boot without them; a standard `scripts/ext_deps/install.ps1` install
needs no environment variables at all, and the full treatment is in
[04-robot-and-safety.md](04-robot-and-safety.md).

```powershell
$env:WILLY_PROFILE = "sim"
python -m src.robot.safety.planning --doctor   # 0 = both load, 1 = degraded, 2 = policy-blocked
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_m1_pick --runs 10 > m1.log 2>&1"
cmd /c "<isaac-sim>\python.bat -m src.willy_sim.run_m2_pick --runs 10 --prompt ""a red cube"" > m2.log 2>&1"
Select-String "GATE:" m1.log, m2.log
```

`--check` reads paths and `--doctor` reads reality: the doctor imports the collision engine, runs a
real distance query, spawns the planner sidecar's interpreter to see what resolves there, and
classifies an operating-system application-control refusal as its own outcome, which is exit `2` and
deliberately not exit `1`, because a blocked binary and a box with no GPU environment need opposite
responses.

The `cmd /c` wrapper is the documented pattern: `cmd` writes UTF-8 logs and PowerShell redirection
writes UTF-16 and mangles them. Never a compound command, and one simulator process at a time. Both
runners print one line per pick carrying `succeeded`, `lift_mm` and `passed`, then a gate line
carrying how many of the N passed. A run counts as passing only when `pick()` reports success and,
independently, the object's measured world-Z rose by at least `robot.sim.gate.lift_threshold_mm`,
which the tree sets to `50.0`; the campaign passes at `int(pass_fraction * runs)`, with
`pass_fraction` set to `0.8`.

If the known-pose run fails, go to [04-robot-and-safety.md](04-robot-and-safety.md). If it passes and
the real-vision run finds nothing, it is perception: re-check the `torch_dtype` winner (section 6),
the weight cache (section 3), and the near-clip row below. **No physical camera has ever fed these
wrappers.**

Two preconditions the runners handle for you, and that any source you write must reproduce.

- **Park the arm out of the camera view before acquiring**, because an overhead camera sees an arm
  that is over the workspace.
- **Set the near clip.** A simulator camera defaults to a 1.0 m near plane, and anything closer
  renders a black RGB image while depth still reports geometry, which misdiagnoses beautifully as a
  broken detector. The wrist camera and the obliques therefore carry an explicit `near_clip_m`. The
  overhead camera deliberately does not: on that geometry a clean instance mask and an
  object-bearing rendered depth cannot both hold, so the ground-truth runners keep the default and
  the real-vision path sets its own close clip in code. Authoring a value is not proof the camera
  took it; read the clipping range back off the camera before believing it.

What happens after the mask is [04-robot-and-safety.md](04-robot-and-safety.md) and
[05-pick-loop.md](05-pick-loop.md). Remember that the default pick is open-loop: the decision gate,
the closed-loop refine, verify and recover path, fusion with its commit gate, the rerank stage, the
learned success model and the reinforcement-learning layer are all built and all default to
`enabled: false`, and the simulator runners turn them on per flag in runner code.

**Training your own closed-set detector.**
[`src/models/detection/closed_set/train.py`](../../src/models/detection/closed_set/train.py) fine-tunes RT-DETR
from COCO annotations and writes a provenance manifest beside the checkpoint. Its classification head
is re-initialised from the dataset's categories rather than fixed to COCO's, so arbitrary classes
work, and the exported checkpoint drops straight into the inference path via `models.detector:
"rtdetr"` plus a `models.rtdetr.model_path`. No dataset ships here and no model has ever been trained
in this repository.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: models.objectdetector.local is true ...` | base profile, `local: True`, directory absent | put the weights there, or set `local: false` with a Hub id, or use the `sim` profile (section 3) |
| A `from_pretrained` failure naming no config key | the segmenter or the speech model, neither of which checks the directory | the same fix, applied to `models.segmenter` or `models.stt` |
| A hub-offline error on the first simulator run | the runners set `HF_HUB_OFFLINE=1` on purpose | pre-fetch outside the simulator; do not export `HF_HUB_OFFLINE=0` |
| `CERTIFICATE_VERIFY_FAILED` while fetching | a proxy re-signs TLS with a CA in the OS store that `certifi` has never heard of | fetch from the project environment, and `pip install truststore` |
| `ValueError: No object detected with description: ...` | nothing cleared `threshold`; `detect()` raises by design | lower the threshold, rephrase, try the other checkpoint, or use `detect_all()` |
| Recall on small objects collapses after a "make it fp32" edit | `torch_dtype: "float32"` also disables the fp16 autocast | use `"__null__"`, not `"float32"` and not `null` (section 6) |
| The label is a fragment of the prompt | GroundingDINO grounds sub-phrases | map back to your canonical name, as the simulator vision source does |
| The right object is named and the wrong one is lifted, reported as success | the phrase grounder fails confidently on complex prompts | this is the VLM route's reason to exist (section 4) |
| A config edit to `models.pipeline` changed nothing | that call site builds through the legacy keys, or names the class directly | check which builder it uses (section 1), and `PerceptionSpec.resolve().render()` |
| Black RGB from a simulator camera while depth looks fine | the 1.0 m default near plane | `near_clip_m` in the simulator camera config (section 8) |

---

## 10. What is settled and what is not

**Exercised without a GPU, and therefore reliable:** the builder guards and their refusal messages,
the prompt router's rules, `PerceptionSpec.resolve()` agreeing with what `build()` constructs, and
the VLM response parser and its coordinate-space contract.

**Analytical or unit-tested only:** `RtDetrObjectDetector`, `OneFormerSegmenter`, the RT-DETR
training CLI, `WhisperSpeechToText`, the MediaPipe detectors, and the SAM2 against OneFormer
comparison, which has not been made here.

**Not validated against real hardware:** every model against a physical camera, the `local: True`
production path, and the live RGB-D adapter, which has only ever seen a fake streamer.

**Not present at all:** a learned depth model. Do not claim one.

Before you start guide 3 or 4: `Remove-Item Env:\WILLY_PROFILE` (POSIX `unset WILLY_PROFILE`).

## See also

- [01-configuration.md](01-configuration.md), profiles and the `"__null__"` sentinel
- [03-calibration.md](03-calibration.md), intrinsics, the camera-to-base transform and the near clip
- [04-robot-and-safety.md](04-robot-and-safety.md), what the mask feeds, and
  [05-pick-loop.md](05-pick-loop.md), the orchestrator and the default-off stages
- [`src/models/README.md`](../../src/models/README.md), the package README
- [`routing/`](../../src/models/routing/README.md) and [`vlm/`](../../src/models/vlm/README.md), the
  two-route perception decision
- [`src/robot/perception/README.md`](../../src/robot/perception/README.md), the live-camera adapter
- [`docs/isaac-ready.md`](../isaac-ready.md), box setup for section 8
