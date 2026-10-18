# The perception models

You have a validating config tree ([01-configuration.md](01-configuration.md)). This guide gets the
vision half of the stack to load its weights and hand back a box and a mask for a prompt. It covers
[`src/models/`](../../src/models/README.md) and [`config/models/`](../../config/models/). Sections 1
to 7 need no simulator; section 8 is the first simulator boot.

```python
from willy import ObjectDetector, load_tree

detector = ObjectDetector.from_config(load_tree())   # what the cell's models.pipeline builds; loads nothing yet
print(detector)                                      # which model answers, from which weights, and the masks
found = detector.detect(image_bgr, prompt="a red cube", segment=True)   # every box for it, each with its SAM2 mask
print(found)
```

`ObjectDetector` is every model here behind one door: the vision-language model, Qwen3-VL, as shipped, and the
closed-set RT-DETR, which answers with every class it knows (section 8), where the tree names it (or `backend=`
does). `classes=["red cube", "blue bin"]` finds several kinds in one call.
[`examples/offline/perception/detect_with_a_prompt.py`](../../examples/offline/perception/detect_with_a_prompt.py)
runs it on any image you name. Section 7 proves the stack a cell builds, before a weight loads.

---

## 1. What is here, and what the pick loop consumes

The model families are listed in [`src/models/README.md`](../../src/models/README.md): the Qwen3-VL
grounder, RT-DETR closed-set detection, SAM2 segmentation, MediaPipe hand and gesture, and Whisper
speech to text. GroundingDINO, OneFormer and the prompt router between GroundingDINO and the VLM left
on 2026-10-17 (the owner's cleanup): the owner's cell detected with the VLM alone. This guide is about
which of them a given config builds.

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
with no robot. The adapter has been measured on a real camera setup (the
[perception status](../../src/robot/perception/README.md#status)).

**There is no depth model here.** Depth comes from the simulator's rendered annotator, from an RGB-D
stream, or from a calibrated stereo pair matched by OpenCV's semi-global matcher in
[`src/calibration/`](../../src/calibration/README.md), which the pick path takes as an RGB-D camera
([`src/camera/`](../../src/camera/README.md#a-stereo-pair-as-the-cells-camera)). None of them is a
learned model. See [03-calibration.md](03-calibration.md).

**Every caller builds through the one builder.** A real cell builds through
`PerceptionSpec.from_config(app_cfg.models).build()`, and so does `python -m src.robot.perception`, so the
exerciser proves the camera and the very stack the cell will run, before a cell exists.

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

**A fresh checkout has no weights, and one script fetches every one of them.** A model is named by its
`model_id` alone, and every model reads its weights from one folder, `models.weights_root`
([`config/models/weights.yaml`](../../config/models/weights.yaml)), set once:
`${WILLY_PROJECT_ROOT}/assets/models/hf` as shipped, inside the repository. A Hugging Face id is read
from `<weights_root>/<kind>/<org>--<name>`, the folder the fetch writes, else from the Hugging Face
cache where an earlier download put it, and nothing downloads at run time. A folder of your own, the
`out_dir` of a `DetectorTraining` run, is named by its path in `model_id` and read as it is.

```bash
python scripts/model_weights/fetch.py --list
python scripts/model_weights/fetch.py vlm-4b sam2        # the pair a real-vision pick needs
python scripts/model_weights/fetch.py --mediapipe         # the hand and gesture .task bundles
```

[`scripts/model_weights/fetch.py`](../../scripts/model_weights/fetch.py) takes the keys `rtdetr`, `sam2`,
`whisper-turbo`, `silero-vad`, `vlm-2b`, `vlm-4b`, `vlm-8b` (the owner's cell's) and `vlm-4b-fp8`, and
writes into the tree's `models.weights_root` (`--root` names another folder). `--list` prints each with
its approximate size, its pin and what it is for. Exit codes: `0` everything asked for is present, `1`
at least one fetch failed, `2` an unknown key.

**Every model refuses a missing checkpoint by name**, through the one resolver
([`src/models/weights.py`](../../src/models/weights.py)): the key, the folder it looked in, and the fetch
that fills it.

```
FileNotFoundError: models.segmenter.model_id is 'facebook/sam2-hiera-large', and its weights are not on
this machine: no folder at <repo>/assets/models/hf/segmentation/facebook--sam2-hiera-large and nothing
in the Hugging Face cache. Nothing is downloaded at run time; `python scripts/model_weights/fetch.py
sam2` writes them.
```

RT-DETR and SAM2 refuse when the cell is built, the VLM at its first prompt (it loads then), Whisper at
the first transcription (`SpeechModelMissing`), and the voice detector and the MediaPipe bundles when
they are opened.

**What the script does for you:**

- **Everything lands under one folder, by what the model is for**, inside the repository as shipped.
  Deleting the checkout deletes the weights with it. The directory is gitignored, so the layout travels
  with a clone and the gigabytes do not. The default location, `~/.cache/huggingface`, outlives every
  checkout and is shared silently between them (12 GB on the development workstation).
- **It drops duplicate formats.** Several repositories publish the same weights as `.safetensors` and
  as `.bin`, and `transformers` reads only what the index names. A per-model ignore list drops the
  duplicate: `openai/whisper-large-v3-turbo` is 1.62 GB filtered.
- **`silero-vad` is not a Hub repository.** It is one TorchScript file inside a PyPI wheel. The script
  downloads the pinned wheel, checks the wheel's sha256, reads the one member out of it and checks that
  member's sha256. Nothing is installed. The speech entries also carry a Hub commit, so a second fetch
  cannot replace the bytes under a model's folder without a trace.
- **It fences the Hub cache** with `fence_model_downloads()` from
  [`src/utility/paths.py`](../../src/utility/paths.py), before `huggingface_hub` is imported, because
  the Hub reads its cache locations once at import.

**Fetch with the environment that reaches the network, load with the one that owns the GPU.** Every
simulator vision runner sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before transformers
imports, because a model that calls the Hub at boot can take the whole cell down with it. The
runners use `setdefault`, so exporting `HF_HUB_OFFLINE=0` overrides them; fetch first instead. Two
interpreters on one machine can also disagree about TLS: behind a proxy that re-signs TLS, the
simulator's interpreter can fail certificate verification where the project environment succeeds.
`truststore` fixes that, is pinned in both requirements files, and the script reports which trust
store it used.

**The `sim` profile runs the cell's models.** Its model overlays left on 2026-10-17: a simulated cell
grounds with the same VLM and cuts with the same SAM2 as the owner's cell, loaded the same way
(section 6). For the simulator in section 8, set it:

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
  kind: zero_shot            # zero_shot = the VLM grounds free text; closed_set = RT-DETR's classes
  zero_shot:
    vlm:
      model_id: "Qwen/Qwen3-VL-4B-Instruct"
```

Read the live values rather than this outline: `python -m src.config explain models.pipeline.kind`,
and the same for `models.pipeline.zero_shot.vlm.model_id`.

**`kind`** picks the detector. `zero_shot`, the default, puts Qwen3-VL in the detector slot: it grounds
any prompt, German or English, a colour, a negation, where a part lies, and SAM2 cuts each box it
finds. `closed_set` answers with RT-DETR's trained classes and needs a `models.rtdetr` block, or the
builder refuses. **There is no other grounder and no fallback.** A cell whose VLM cannot load (no
weights, no dependency, no VRAM) answers its looks with nothing and a counted detector failure, and a
task ends `detector_failed`; it never grounds with something else. GroundingDINO, the phrase grounder
that shipped before, did not fail on a prompt it could not represent: it returned a confident box for
the wrong object, which nothing downstream could tell from a correct answer, and that is why it went
with its router. **The VLM has run on a physical cell**, on a wrist D415, and nothing measured there
is kept here: its grounding quality, its VRAM cost and the choice between the 4B and 8B checkpoints are
yours to measure ([`src/models/vlm/README.md`](../../src/models/vlm/README.md)).

**One VLM copy reads commands and detects.** The operator console reads an operator's sentence, German or
English, with the same Qwen3-VL, as a text-only question (`src/models/vlm/command.py`), and a process holds
**one copy** for both (`shared_vlm()`, [`vlm/`](../../src/models/vlm/README.md)): the factory takes its grounder
from that holder, so a cell whose detector is the VLM reads commands with the copy it detects with. The 4B
checkpoint holds **8.93 GB** of VRAM once loaded and peaks near **9.9 GB** while it answers (the development box's
RTX 5080), beside SAM2, Whisper, the cuRobo sidecar and the desktop, so a second copy is not an option. The rule
for loading: a cell that detects with the VLM loads it at its first prompt or command, never at its build; any
other cell refuses a command until a person presses "Laden" (`POST /v1/commands/warmup`), and building such a cell
lets the copy go unless a person loaded it. **Switching the checkpoint takes a rebuild**: a command or "Laden"
beside a copy of other weights is refused. A load takes about 6.3 to 7.3 s and a command about 2.2 s on a free
card, 5.5 to 19 s while another GPU job runs. The cell PC's card is unmeasured.

**Not every sentence needs the model, and an answer is cheaper than it was** (2026-10-08 and 2026-10-09). The
cell's everyday sentences ("Alle grauen Würfel in die Gelbe Kiste.", "Hallo Willy") are read by a closed grammar
as the model was measured to read them, with nothing loaded (`runtime.commands.known_sentences` in
`config/app/runtime.yaml`, on), and a question the loaded copy answered before is answered from memory. How fast
the model writes is fixed at the owner's cell's values since 2026-10-17
([`src/models/vlm/qwen.py`](../../src/models/vlm/qwen.py)): an answer ends where its JSON closes, and its decode
passes run as CUDA graphs where the card allows (`auto`), both leaving every answer byte for byte as it was; the
answer lookups propose ten tokens a pass, which saves about half the passes and may move a coordinate's last
digit or a word of a reading ([`vlm/`](../../src/models/vlm/README.md), What an answer costs;
[05](05-pick-loop.md) 5.5).

SAM2 cuts every mask, whichever detector found the box
([`config/models/segmenting.yaml`](../../config/models/segmenting.yaml)). To measure the VLM's grounding
of attributes in simulation, `run_attribute_pick --mode grounding` scores eight prompts, half of them
German, against the scene's ground truth.

---

## 5. The leaf blocks

`ModelsConfig` requires `segmenter` and `stt`; `weights_root` and `pipeline` have schema defaults, and
`rtdetr` defaults to `None`. List what exists with `python -m src.config where "models."` and read one
key with `explain` ([01 section 4](01-configuration.md)).

**A block holds what a cell chooses, and nothing else** (the owner, 2026-10-17): every model's
`model_id`, RT-DETR's `threshold` and Whisper's `language`. How each model loads and answers is fixed in
code at the values the owner's cell ran (section 6), and a key that chose anything else (`model_path`,
`local`, `optim`, the VLM's answer settings, the microphone's numbers) is refused at load with the
sentence that says what to write instead ([`src/config/schema/_removed.py`](../../src/config/schema/_removed.py)).

**`models.rtdetr`** is the closed-set detector: its `model_id` (a Hugging Face id, or a
`DetectorTraining` out_dir) and `threshold`, the score a box must clear. **`models.segmenter`** is
SAM2's `model_id`. Two behaviours their comments do not say:

- RT-DETR's boxes come back in pixels; the VLM's come back on its 0 to 1000 grid and are scaled to
  pixels inside its wrapper ([`vlm/`](../../src/models/vlm/README.md)).
- SAM2 raises on a mask that is empty after post-processing, and the perception seam drops that box.

**`models.stt`** is Whisper. The operator console reads the section alone (`load_speech_section`) and
keeps one engine for the process, behind `POST /v1/voice/transcribe`, `/v1/voice/talk` and
`/v1/voice/listen`. A recording becomes a text proposal that a person reads and confirms before it
becomes a prompt. The text stays in the language it was spoken in: Whisper transcribes and never
translates, and `language: auto` lets it detect German or English per recording. A console on a fresh
checkout is refused on the first transcription, by name and with the fetch that fixes it. The
microphone opens at 16 kHz, one channel of int16 in blocks of 8000 frames, fixed at the cell PC's values
in [`src/models/speech/`](../../src/models/speech/README.md), for push to talk
(`PushToTalkSource.from_config`) and for `Listener.from_config`, which no console route or cell verb
opens. From a program:
[`examples/real_robot/14_speak_a_command.py`](../../examples/real_robot/14_speak_a_command.py).

**The hand and gesture detectors** are standalone MediaPipe and are not on the grasp path. Their blocks
left on 2026-10-17: the numbers they run at are the owner's cell's, fixed in
[`src/models/handdetection/constants.py`](../../src/models/handdetection/constants.py), and the `.task`
bundles are read from `<models.weights_root>/mediapipe`, where `fetch.py --mediapipe` writes them.
`python -m src.models.handdetection --check` says whether this host can run them
([`src/models/handdetection/README.md`](../../src/models/handdetection/README.md)).

---

## 6. How each model loads, and the `torch_dtype` trap

`optim` left every model block on 2026-10-17 (the owner's cleanup): how a model loads is fixed per model
in [`src/models/_inference.py`](../../src/models/_inference.py) at what the owner's cell ran, and a tree
that writes `optim` is refused at load. Only the dtype changes numbers; channels last is a memory
format, and nothing is compiled.

| Model | `torch_dtype` | Weights | Autocast compute | Attention | Channels last |
|---|---|---|---|---|---|
| SAM2 | unset | fp32, read from the checkpoint | the CUDA autocast default, fp16 | `sdpa` | on |
| RT-DETR | `auto` | fp16 | fp16 | `eager` | on |
| Whisper | `auto` | fp16 | fp16 | `sdpa` | off |

The VLM loads in its checkpoint's own dtype (Hugging Face's `dtype="auto"`: bf16, or FP8 for an `-FP8`
id) and is not in this table.

The gate is one condition in `_inference.py`: when `torch_dtype` is set, the resolved dtype goes to
`from_pretrained` **and** becomes the autocast dtype. **This is the paragraph to remember.** "The
segmenter must run fp32" is true about the weights and incomplete as an instruction: a dtype of
`float32` would also turn off the fp16 autocast, so SAM2's fp32 weights under fp16 compute exist only
with the dtype unset, which is what its settings say. fp16 weights cost recall on small objects, and
`build_load_kwargs` logs a warning whenever it resolves to fp16. Small means a few tens of pixels across:
a 30 mm part under an overhead camera a metre above it.

Two more traps. This project's `"auto"` is not Hugging Face's: here it means fp16 on CUDA and fp32
elsewhere, while Hugging Face's `dtype="auto"` reads the checkpoint, as the VLM's load does. And the
trap is CUDA-only: half dtypes become fp32 on CPU and MPS, and autocast does nothing off CUDA, so a CPU
cannot reproduce the loss. `WILLY_DEVICE` forces the device of every model but the VLM, which loads
onto CUDA: `auto`, `cuda`, `cpu` or `mps`.

---

## 7. Prove the stack before you boot the simulator

A simulator boot takes a long time before it can tell you anything. Settle the perception half first,
in two steps.

**Step one needs no weights.** `PerceptionSpec.resolve()` reports what `build()` would construct, from
the same refusals the builder uses:

```bash
python -c "from src.models.perception_spec import PerceptionSpec; from willy import load_tree; print(PerceptionSpec.from_config(load_tree().app_config.models).resolve())"
```

On the shipped tree it prints the stack and the VLM's checkpoint:

```
perception stack: zero_shot / vlm + sam2
  vlm             : Qwen/Qwen3-VL-4B-Instruct
```

If it names a refusal (`REFUSED`), fix that before fetching gigabytes.

**Step two runs the stack on a drawn image.** It needs the VLM's and SAM2's weights and a GPU, but no
simulator, no camera and no image file, and it builds through the same path a cell does, so it
exercises the stack your config selects. Save it wherever you like:

```python
# smoke_models.py
import numpy as np

from src.models.perception_spec import PerceptionSpec   # the cell's own builder, the path every pick takes
from willy import load_tree

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
$env:HF_HUB_OFFLINE = "1"          # fail fast on missing weights instead of downloading mid-test
python smoke_models.py
```

The square covers `(280, 200)` to `(360, 280)`: 6400 px, centred on `(320.0, 240.0)`. A box near
those corners, a mask area near 6400 px and a centroid on that centre mean the front end works. The
wrappers log their load line; SAM2's `dtype=None` in it confirms the regime from section 6 on the
loaded object. The box and the pixel count are the stable part; timings vary. The VLM writes no score
of its own, so every box it finds carries the same one.

Nothing found comes back as an empty tuple, and so does a model error, which the seam counts: a cell
reads that count as a detector failure, never as an empty scene.

CI never runs model inference: CI has no GPU and no weights, and the torch wrappers are left out of
coverage. It runs import checks, the builder guards, the weights resolver and the spec logic, all of
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
the real-vision run finds nothing, it is perception: check the weights (section 3), the VLM's load line
in the perception log, and the near clip below.

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
[05-pick-loop.md](05-pick-loop.md). The default pick is open-loop: the recover path and the fusion of two
or more calibrated RGB-D cameras run on every cell with no switch, the decision gate, the rerank stage, the
learned success model and the reinforcement-learning layer left on 2026-10-16 with their code, and the
simulator runners set what they run per flag in runner code.

**Training your own closed-set detector.** `DetectorTraining` trains RT-DETR on **your own fixed classes** from a
COCO or YOLO folder, as CVAT, Label Studio or Roboflow export it, and keeps the **best epoch** by validation mAP.
It reads like `GeneratorTraining`:

```python
from willy import DetectorTraining

run = DetectorTraining.from_dataset(dataset="D:/data/my_parts", out_dir="assets/models/rtdetr/my_parts",
                                    epochs=80)
print(run.probe())        # boxes per class and split, and every box or image left out
report = run.train()      # the best epoch in out_dir, the last in out_dir/last
print(report)
run.write_report(report)  # report.json and report.html beside the model
```

`epochs=`, `batch=` and `image_size=` are arguments of `from_dataset`, and `classes=["white pawn", "black pawn"]`
trains some of the dataset's classes only, the objects of the others left in the images as background;
`DetectorPlanOverrides` holds every other setting, the learning rate among them. While it trains, a terminal shows a
bar over each epoch's batches and a line per epoch with the best mAP so far and the time left, and `curves.png` in
`out_dir` (the losses, the mAPs, the learning rate, AP per class) is drawn again after every epoch. `report.html` is
the whole run on one page.

The run is RT-DETR's own recipe: the backbone at a tenth of the learning rate, mixed precision, gradients clipped at
0.1, an average of the weights, the strong augmentations and multi-scale batches until the last tenth of the epochs,
COCO mAP@0.5:0.95 after every epoch and a stop after 15 epochs without a better one. A dataset without a validation
split gives 15 % of its images to one. An image larger than training needs is read shrunk (at most 1333 px on its long
side at 640), its boxes scaled with it, so a phone photo costs no more than it must and never trips PIL's
decompression-bomb check; loader processes read beside the training step, on Windows too, without running your
script again, so it needs no `if __name__ == "__main__":` guard. The model in `out_dir` drops into the inference path
as `models.rtdetr.model_id: <out_dir>` with `models.pipeline.kind: closed_set`. Measured on the development RTX 5080 on 400
chess-piece photos of up to 12 megapixels (2026-10-10): about **20 training images per second** with every
augmentation on, multi-scale up to 800 px. `tier="smoke"` proves the chain in a minute,
[`examples/offline/training/06_train_a_detector_on_your_images.py`](../../examples/offline/training/06_train_a_detector_on_your_images.py)
shows the whole run, and [`src/models/README.md`](../../src/models/README.md) has the command line.

**Detecting with it, every class at once** (2026-10-09). A program of your own, such as a chess program that wants
every piece on the board, uses the trained model through `ObjectDetector`. One call answers with every class the model
knows, and `segment=True` gives each object its SAM2 mask from the same call, every box cut in one SAM2 pass:

```python
from willy import ObjectDetector

detector = ObjectDetector.from_weights("assets/models/rtdetr/my_parts")   # an out_dir; .../last is the last epoch
print(detector.classes)                          # its id2label, in id order
found = detector.detect(image_bgr, segment=True) # Detections, the highest score first
print(found)                                     # a line per object, and how many of each class
```

`classes=["cube"]` narrows a call to exact class names, case and blanks aside, and a name the model does not know is
refused with every name it does know; `found.to_dict()` writes the masks as COCO run-length encoding.
`ObjectDetector.from_config(tree, backend="closed_set")` builds the same from a cell's `models.rtdetr` and
`models.segmenter`; without `backend=` it builds what the tree's `models.pipeline` names, the VLM as shipped (the top
of this guide). The details, the refusals and the licences (Apache-2.0 for both model families) are
in
[`src/models/detection/`](../../src/models/detection/README.md), and
[`examples/offline/perception/detect_every_class.py`](../../examples/offline/perception/detect_every_class.py) runs it
on any image you name.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: models.<block>.model_id is '...', and its weights are not on this machine` | the folder under `models.weights_root` and the Hugging Face cache are both empty for it | the fetch the message names (section 3), or `models.weights_root` where the weights already live |
| `SpeechModelMissing: models.stt.model_id ...` | the Whisper folder is absent | `python scripts/model_weights/fetch.py whisper-turbo` |
| `... removed on purpose ...` at load, naming a models key | a tree still writes a key that left on 2026-10-17 (`model_path`, `local`, `optim`, `objectdetector`, the router) | the sentence says what to write instead; delete the key |
| A Hub-offline error on the first simulator run | the runners set `HF_HUB_OFFLINE=1` on purpose | fetch outside the simulator; do not export `HF_HUB_OFFLINE=0` |
| `CERTIFICATE_VERIFY_FAILED` while fetching | a proxy re-signs TLS with a CA that `certifi` does not know | fetch from the project environment, with `truststore` installed |
| A task ends `detector_failed` | the VLM did not load or failed while answering: no weights, no dependency, or no VRAM | the perception log names the cause; there is no fallback detector |
| Small-object recall collapses on SAM2 | its weights were loaded fp16 by an edit to `_inference.py` | leave SAM2's dtype unset (section 6) |
| Black RGB from a simulator camera while depth looks fine | the 1.0 m default near plane | `near_clip_m` in the simulator camera config (section 8) |

---

## 10. What is settled and what is not

**Pure logic, pinned by the test suite on a machine with no GPU:** the builder guards and their
refusal messages, the weights resolver and its refusals, `PerceptionSpec.resolve()` agreeing with what
`build()` constructs, and the VLM response parser with its coordinate-space contract.

| Capability | Evidence |
|---|---|
| SAM2 on rendered images | measured in simulation with GroundingDINO's boxes, before 2026-10-17; the VLM's boxes there are unmeasured |
| RT-DETR, the MediaPipe detectors, the RT-DETR training script | never touched hardware: unit tests only |
| `ObjectDetector`: every class in one call, SAM2's masks for every box in one pass | unit tests on a CPU, with the real RT-DETR and SAM2 classes at toy size; the real-weights file `tests/test_an_object_detector_on_real_weights_inference.py` is written and not run |
| Whisper and the Silero voice detector | never touched hardware: fakes, a random Whisper, and the real weights for load time, latency and memory only |
| The VLM and SAM2 on a physical camera | run on a physical cell: a wrist D415 on a UR10 (CB3); no measurement is kept here |
| The VLM reading commands: load time, VRAM, latency, German and English sentences | measured on the development box's RTX 5080 against the real 4B weights (`tests/test_vlm_command_inference.py`); not on the cell PC |
| The live RGB-D adapter | measured on a real camera setup ([perception status](../../src/robot/perception/README.md#status)) |
| A learned depth model | does not exist |

No recording of a spoken command exists here to be right or wrong about, so speech accuracy is
unmeasured ([`src/models/speech/README.md`](../../src/models/speech/README.md)).

Before you start guide 3 or 4: `Remove-Item Env:\WILLY_PROFILE` (POSIX: `unset WILLY_PROFILE`).

## See also

- [01-configuration.md](01-configuration.md): profiles and the merge rules
- [03-calibration.md](03-calibration.md): the camera-to-base transform
- [04-robot-and-safety.md](04-robot-and-safety.md), what the mask feeds, and
  [05-pick-loop.md](05-pick-loop.md), the loop and the stages that are off by default
- [`src/models/README.md`](../../src/models/README.md): the package README
- [`vlm/`](../../src/models/vlm/README.md): the VLM, the one grounder, and the command reader, with
  the two readings it takes of a phrase ([`routing/`](../../src/models/routing/README.md))
- [`src/robot/perception/README.md`](../../src/robot/perception/README.md): the live-camera adapter
- [`docs/isaac-ready.md`](../isaac-ready.md): the workstation setup for section 8
