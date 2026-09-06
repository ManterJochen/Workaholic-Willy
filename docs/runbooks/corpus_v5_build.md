# Building the v5 corpus

Copy and paste, in order. Each step says what it costs and what breaks if it is skipped.

This is the build procedure for the asset selection that ships in `datagen/assets/screens/`, run at
corpus scale. If you are training on your own parts rather than on this selection, start from
[`train_your_own_generator.md`](train_your_own_generator.md), which is the same pipeline with your
meshes at the front; this runbook adds the sharding, memory and resume rules that only a long build
needs.

The two long steps are the render and the labelling, and they are separate commands on purpose: a
different label density costs no render time, so a question about density is settled on a finished
corpus rather than guessed at before one exists.

## What ships, and what you can read off it

| | |
|---|---|
| asset list | `datagen/assets/screens/train_asset_ids_v5.json`, 962 ids across `gso`, `ycb`, `thingi10k` and `asos` |
| graspability verdict | `datagen/assets/screens/screen_v5.json`, at `grid` density over the three rest poses, 1,178 meshes screened |
| build configuration | `datagen/assets/screens/v5_config.json` |

Of the 1,014 meshes the screen could close into a solid, 552 earn at least one jaw label and 880 a
suction one; the remaining 164 are reported `unclosable` rather than passed. Those counts are in the
shipped file, so recount them rather than trusting this table. The meshes themselves are not in the
repository: `python -m datagen.assets --check` reports what is present per source.

## Step 0. The supermarket meshes, if they are still raw

Skip this if `assets/meshes/asos` is already normalised.

```bash
python -m datagen normalise-meshes --collection asos
python -m datagen screen-meshes --collection asos --out datagen/assets/screens/screen_asos.json
```

Then merge that verdict into `screen_v5.json` and re-derive the id list, or drop `asos` from the
list and move on.

Two rules about normalising a collection. Meshes are read as metres, and `datagen/assets/prepare.py`
carries a table of the scale factor per source with the measurement that justifies each one; `asos`
is authored in decimetres. A source absent from that table keeps its authored scale, which is the
honest default, and `--scale` is how you override it. Decimation is quadric only: vertex clustering
is much faster and it shatters the surface, and a mesh in pieces cannot be closed into a solid, so
it can be neither labelled nor collided. No cleanup step recovers that, because it is structural.

## Step 1. Warm the convex decomposition

```bash
python -m datagen decompose --config datagen/assets/screens/v5_config.json --jobs 10
```

Skipping this does not fail. The engine decomposes on demand and caches, so a build works without
it, but it then pays the decomposition inside its own render loop, one mesh at a time, and nearly
all of that time is the decomposition itself. Across several processes the same work is minutes and
the build finds every entry warm.

## Step 2. Render

```bash
python -m datagen build --config datagen/assets/screens/v5_config.json \
    --name v5_s0 --out logs/dl --scenes 10000 --seed 0
```

Shard it by running several with different `--name` and `--seed`.

**One shard per physical core is the ceiling.** Adding shards past that returns the same total
throughput while each shard takes proportionally longer, so the extra processes only slow each other
down. Hyperthreading buys nothing here. Memory is not what limits the render: a shard holds a few
hundred megabytes.

Do not read the rate during the first few minutes. Each shard loads its own mesh bank first, and a
throughput measured across that start-up is a measurement of the start-up.

Treat `python -m datagen cost` as a lower bound rather than a prediction. Its per-scene figure comes
from a small build against a much smaller asset bank, and cost scales with how many convex parts
each object enters the physics as. It is not wrong; it is answering about a different corpus.

`datagen build` resumes: re-running the identical command skips scenes already written, so an
interrupted run costs only what it had not written.

Measure progress by the `ok` count, never by the exit code. `build` returns 0 when it gives up.

## Step 3. Label

```bash
python -m datagen label-grasps --name v5_s0 --out logs/dl --density grid
```

CPU only, no GPU, and shard it exactly like the render: one process per dataset. Single-process over
a whole corpus is an order of magnitude longer than the sharded run, which is the number that
matters when somebody reaches for one command out of habit.

`python -m datagen cost` under-reports this step badly, because its figure is for `default` density.
`grid` places five times the anchors, and per object the labeller tries axes by anchors by
approaches, each one a line test against a decimated mesh.

**One process per physical core is the memory ceiling too, not only the core ceiling.** Each
labeller holds its own mesh bank plus per-object raycasting scenes, and the pressure grows as the
run proceeds, so a launch that reads plenty of free memory is not evidence that it fits. Past the
ceiling, processes are killed inside numpy or inside the raycaster.

The per-process rate falls as concurrency rises while total throughput still climbs, so more
processes buy less than proportionally more work. Do not read the rate before every process has
loaded its bank.

**Decide the density against the corpus, never against isolated meshes.** The share of candidates
rejected as below the table is far higher in a scene than on the same meshes screened alone, because
in a scene the object sits among others on a surface.

## Step 4. The corpus the trainer reads

```bash
python -m datagen build-cloud-corpus --name v5_s0 --out logs/dl --corpus-out logs/dl/clouds/v5/s0
```

One per shard, each into its own subdirectory. The trainer walks them with `rglob`, and scene ids
count within one dataset, so two shards hold different scenes under identical names and a shared
directory silently overwrites the collisions. Pointing several shards at one `--corpus-out` is
refused, correctly, and that guard is the reason the overwrite cannot happen quietly.

It writes one `.npz` per scene inside its loop, so memory stays flat, but it does not skip a scene
it already wrote: killing a builder costs its whole run. Check what a process has produced before
stopping it, rather than what you assume about it.

Each process loads the full mesh bank first, and the shards each do it separately.

### Labelling for a different gripper

```bash
python -m datagen label-grasps --name v5_s0 --out logs/dl --density grid --jaw wide_140
python -m datagen build-cloud-corpus --name v5_s0 --out logs/dl \
    --labels grasps_jaw_wide_140.jsonl --corpus-out logs/dl/clouds/v5_jaw_wide_140/s0
```

This writes `grasps_jaw_<name>.jsonl` and never the default corpus file. The jaw is taken as a name
out of `PROCEDURAL_JAWS` in `datagen/grasps/verdict.py`, which holds `narrow_55`, `wide_140` and
`slim_pad`, rather than as a model object, so the output path is a function of the input and a
non-default gripper cannot reach `grasps.jsonl` at all.

A non-default `--labels` changes the stamped dataset identity, so the overwrite guard can tell two
grippers' clouds apart in one directory and every cloud records which jaw its labels came from.
Without that stamp a jaw-varied corpus is indistinguishable from the default gripper's at training
time, and a wrong gripper vector is worse than a constant one: it teaches a relationship that is not
there.

### If one shard is left running alone, split it

The step is single-process per dataset, and `--scenes` cannot divide it: that flag is an even
stride, so two calls with it walk the same subset rather than complementary ones. Hand each process
its own dataset instead.

```bash
python -m datagen split-dataset --name v5_s2 --out logs/dl --parts 4   # prints the follow-up commands
# ... run those, then:
python -m datagen split-dataset --name v5_s2 --out logs/dl --clean
```

Scene directories are junctioned rather than copied, and the split is checked disjoint and complete
before anything runs. Give every part its own `--corpus-out`. And note that splitting throws away
what the running builder had already produced, since the extraction step does not resume, so
splitting a shard that is nearly finished costs more than it saves.

## Step 5. Train

The training half is [`train_your_own_generator.md`](train_your_own_generator.md) from Step 4
onward, run against `logs/dl/clouds/v5` instead of your own corpus:

```bash
python -m src.robot.grasping.deep train-set --recipe v1 --tier full \
    --clouds logs/dl/clouds/v5 --out logs/dl/models/v5_corpus --artifact-gripper 2f85
```

Keep `--epochs` where the tier puts it even when you intend to stop early, because
`CosineAnnealingLR` takes its `T_max` from it: asking for fewer epochs changes the learning-rate
trajectory, which breaks the pairing against any run you are comparing to.

## Two levers that are measured but not applied

Both rescue objects that currently earn no label in any rest pose, and neither is applied, because
the screen that built the asset list was taken at the shipped setting. Applying either means
re-screening first, or the corpus labels objects the screen never let in.

| lever | what it does | sweep |
|---|---|---|
| `mesh_closing_axes` | more sampled closing directions per mesh, on top of the three principal axes, recovers a small share of the objects that earn nothing, at a proportional cost in labelling time | `datagen/eval/sweep_closing_axes.py` |
| `anchor_fractions` | more anchors along the free plane does the same, and does not saturate over the range swept | `datagen/eval/sweep_anchor_and_poses.py` |

The largest one is a build rather than a switch. With `--kinds both` the suction labels do reach the
`.npz`, in their own `suction_*` arrays, and they train the stage that decides where to grasp. What
does not exist is a suction head in the network, so no suction pose is ever learned from them. Using
them fully means building that head.

## Do not run anything else on this machine while a long step runs

A training run holding many gigabytes is killed by a test run, by a couple of type-checker
invocations, or by any agent that starts a mesh script. Edits and file writes are fine; execution is
not.

## See also

- [`train_your_own_generator.md`](train_your_own_generator.md), the same pipeline on your own parts.
- [`datagen/`](../../datagen/README.md), the scene generator, its three render engines and its asset
  sources.
- [`src/robot/grasping/deep/`](../../src/robot/grasping/deep/README.md), what the corpus is for.
