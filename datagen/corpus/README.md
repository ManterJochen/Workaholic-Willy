# Training corpora, and whether one is worth training on (`datagen/corpus`)

This package turns a labelled dataset into what a model trains on, the per-scene point clouds a grasp generator
reads and the flat table a grasp ranker reads, and it judges a corpus before a training run rather than after.
The judgement opens no cell, loads no model and trains nothing.

```python
from datagen.corpus.service import CorpusCheck

report = CorpusCheck.from_corpus("data/datagen/first/grasps.jsonl", source="grasp_labels").assess()
print(report.render())                   # every finding, blocking or a warning, and each column's numbers
print("usable" if report.usable else "refused")
```

On a label file alone the verdict is `REFUSED`: `grasps.jsonl` carries no `held` outcome for the default spec
to predict, and the gate says so instead of letting a trainer fit a constant. The table a ranker trains on is
the one `build-ranker-corpus` writes from the `eval-grasps` output, with `--physics` joining `held` from a
physics run. The same checks from a shell:

```bash
python -m datagen check-dataset --corpus data/datagen/first/grasps.jsonl --source grasp_labels
python -m datagen check-dataset --corpus <table>.npz --source ranker_npz --spec held_jaw_v1
python -m datagen check-dataset --corpus <dataset>/grasps.jsonl --source grasp_labels \
    --serve <dataset>/grasp_eval.jsonl --serve-source grasp_eval \
    --spec willy_jaw_with_contact_angle --serve-spec willy_jaw_with_contact_angle
```

`check-dataset` exits 0 when the corpus is usable, 1 when it is refused and 2 when it cannot be read. The third
command compares a training corpus with the population it will be served on. `--source` and `--serve-source`
take `grasp_labels`, `grasp_eval` or `ranker_npz`, all written by `datagen`; there is no reader for a corpus
from another product. The clouds come from `DatasetBuild.clouds`, shown in
[04_extract_a_corpus.py](../../examples/offline/datagen/04_extract_a_corpus.py).

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `CorpusCheck` | `from_corpus(path, source=..., spec=..., serving=...)` | `assess()` | `CorpusCheckReport`: `usable`, `render()`, `as_dict()` |
| `RankerCorpus` | `from_dataset(root, mask_source="gt")`, then `with_physics(file)` | `build(out)` | `RankerCorpusReport`, and the `.npz` table |
| `RankerFit` | `from_corpus(path, spec=...)`, or `from_report(report)` | `fit()` | `RankerFitReport`; `beats_baseline` is the verdict |
| `DatasetBuild` | `from_file(...)`, see [datagen](../README.md) | `clouds(out_dir, kinds=...)` | a `StageResult`; one `.npz` per scene on disk |

The first three import from `datagen.corpus.service` and are the library side of `check-dataset`,
`build-ranker-corpus` and `train-ranker`.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| the spec's label is not in the corpus | a label or eval file, which carries no `valid` or `held` column | check the ranker table: it carries `valid`, and `held` with `--physics` |
| a zero-variance feature | a declared feature holds one value on every row | fix the writer, or drop it from the spec |
| train/serve skew | a feature constant in training varies at serving | the model would send every served row down one branch; fix the data |
| group leakage | one object on both sides of a `split`, with zero tolerance | split by the spec's group column |
| no usable positives | no positive row, or positives in fewer than two groups | more physics trials, or a different label |
| two corpora in different frames | the two specs name different frames | convert one, or compare corpora in the same frame |
| a fit that does not beat `width_mm` alone | it loses on AUROC or on precision at 250 | `train-ranker` exits 1 and says so; nothing is promoted |
| a physics join that resolves no row | the physics file belongs to other label rows | join the physics run of this dataset; the key is (file, line) |

## Why each check exists

**Zero variance.** A column with one value on every row looks exactly like a feature, and a study over it
reports "no signal" for a reason that has nothing to do with grasping. A trainer reading a key nobody wrote
lands on a constant the same way.

**Train/serve skew.** Only two corpora can show it. The same field can hold one value in the analytic labels
and a wide range over the calculator's own candidates: same name, two populations.
`willy_jaw_with_contact_angle` declares such a field on purpose, so this check stays honest.

**Group leakage.** A `split` column with one value is no held-out set, and a row-level split puts the same
object on both sides, which inflates every number reported from it.

**Class balance.** A trainer needs the positive rate before it starts.

**The frame.** Object, grasp and BASE frame vectors sit under identical column names. A `FeatureSpec` carries
its frame, and the gate refuses to compare two corpora whose frames differ.

The gate checks the features a spec declares, never columns inferred from the file. The analytic labeller's
worst contact deviation is zero by construction, which is correct and must not fail a dataset, so a constant
column that is not a feature comes back as a note. The specs are declared in
[src/robot/grasping/deep/ranker/features.py](../../src/robot/grasping/deep/ranker/features.py), which the
trainer reads too.

| Spec | Frame | Label | For |
|---|---|---|---|
| `bootstrap_jaw_v1` | grasp | `held` | the default: the first scorer, over a physics-labelled corpus |
| `bootstrap_object_frame_v0` | object | `held` | the object-frame alternative, kept so the frame check has a real mismatch to refuse |
| `valid_jaw_v1` | grasp | `valid` | stage one: does a candidate close; fitted on the analytic verdict, so self-referential |
| `held_jaw_v1` | grasp | `held` | stage two: does a closed grasp survive the shake; the only spec promotion is measured on |
| `willy_jaw_with_contact_angle` | base | `held` | declares a field unfit to be a feature, to keep the skew check honest |

`valid` trains and the shake alone promotes, because nothing measured on `valid` separates a model that learnt
the world from one that learnt the verdict. The shake is stratified, so a pooled positive rate describes the
draw rather than the corpus: slice by `physics_source`, or re-weight by the stratum sizes written beside it.

## What a cloud file holds

One `.npz` per scene: every rendered view fused into one cloud in BASE millimetres, with the table, the bin
walls and the posed arm left in, because a generator that never sees a wall proposes grasps into it. Each point
carries how many views saw it, and normals point towards the cameras that saw them. `kinds` decides which grasp
labels a corpus carries; the suction arrays exist in a jaw-only corpus and are empty. Every loader converts
units at the edge, so everything past [sources.py](sources.py) is millimetres.

## Status

| Capability | Evidence |
|---|---|
| The gate's column statistics | measured in simulation: `tests/test_corpus_gate.py` holds them equal to the RL readiness check's |
| Clouds from a rendered dataset | measured in simulation: CI runs `examples/offline/datagen/04_extract_a_corpus.py` |
| A model trained on these corpora at a physical cell | never touched hardware |

## Files

| File | Holds |
|---|---|
| [gate.py](gate.py) | the checks and the verdict: blocking findings stop a run, warnings are things to know |
| [sources.py](sources.py) | each file format read into canonical columns, units converted once |
| [columns.py](columns.py) | one column's rows, distinct values, non-zero count, spread, range, non-finite entries |
| [build.py](build.py) | the flat feature table the ranker trains on, and the physics join |
| [clouds.py](clouds.py) | the per-scene point clouds the generator trains on |
| [tables.py](tables.py) | another hand's grasp table beside clouds that already exist, for `build-grasp-tables` |
| [split.py](split.py) | one rendered dataset split into disjoint views, for `split-dataset` |
| [service.py](service.py) | `CorpusCheck`, `RankerCorpus` and `RankerFit` |

## Details

- [datagen](../README.md), the scenes; [grasps/](../grasps/README.md), the labels and the physics join's source.
- [src/robot/grasping/deep/](../../src/robot/grasping/deep/README.md), the generator these clouds train, and
  [train_your_own_generator.md](../../docs/runbooks/train_your_own_generator.md), the procedure.
- Tests: `tests/test_corpus_gate.py`, `tests/test_corpus_physics_join.py`, `tests/test_datagen_ranker_service.py`,
  `tests/test_datagen_grasp_tables.py`.
