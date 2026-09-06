# `datagen/corpus`: is this corpus worth training on?

Four checks, run before a training run rather than after it, plus the builders that turn a rendered
dataset into the two things a model trains on. The gate opens no cell, loads no model and trains
nothing.

```bash
# one corpus on its own
python -m datagen check-dataset --corpus <dataset>/grasps.jsonl --source grasp_labels

# a training corpus against the population it will actually be served on
python -m datagen check-dataset \
    --corpus <dataset>/grasps.jsonl          --source grasp_labels \
    --serve  <dataset>/grasp_eval_sfe.jsonl  --serve-source grasp_eval \
    --spec willy_jaw_with_contact_angle --serve-spec willy_jaw_with_contact_angle
```

Exit `0` usable, `1` refused, `2` unreadable. The three formats `--source` and `--serve-source` accept
are `grasp_labels`, `grasp_eval` and `ranker_npz`, all of them written by this package.

## The four checks, and why each one exists

Not one of these is hypothetical, and in every case the data looked fine.

**Zero variance.** A column holding one value on every row looks exactly like a feature. Training on it
is harmless to the fit and fatal to the diagnosis, because a correlation study over it reports "no
signal" for a reason that has nothing to do with grasping. The mirror image is a trainer reading a key
nobody wrote, which lands on a constant the same way.

**Train/serve skew.** The more dangerous half, and the one only two corpora can reveal. A feature that
is constant in training and varies at serving is not merely wasted: a tree learns one branch and sends
every serving row down it, and a network applies an untrained weight to a live input. The same field
can hold one distinct value in the analytic reference and span a wide range over the calculator's own
candidates. Same name, two populations.

**Group leakage.** A `split` column with one value on every row is no held-out set at all, and a
row-level split over a corpus with many rows per object puts the same object on both sides, which
inflates every number reported from it. The tolerance is zero: a single shared object is the leak, and
a tolerance here would only decide how much of one to accept.

**Class balance.** A trainer has to be told the positive rate before it starts. A corpus with no
positive row, or with a positive in fewer than two groups, is unusable rather than difficult.

## A fifth check the numbers alone can never give: the frame

Object-frame, grasp-frame and BASE-frame vectors sit under identical column names, and the quantities
are not the same. A `FeatureSpec` carries its frame, and the gate refuses to compare two corpora that
disagree rather than producing a confident comparison of different things.

That refusal is not theoretical caution. Frame and sign errors are the defect class this repository has
paid for most often, and each time the numbers looked plausible: a closing axis zeroed in the wrong
frame, a table check reading a camera depth as a height, a base frame rotated by half a turn, a surface
normal pointing into the table because a principal-axis eigenvector has no sign.

## Why a declared feature list, and not inference from the file

A gate that reads columns off a file can only say that a column is constant. It cannot say that a
feature is constant, and that is the whole distinction. The analytic labeller enumerates exact
antipodal closing axes, so its worst contact deviation really is zero by construction. That is correct,
and it must not fail a dataset.

So [`src/robot/grasping/deep/ranker/features.py`](../../src/robot/grasping/deep/ranker/features.py)
declares what the model eats, the trainer reads the same file, and the gate checks exactly that. A
constant column that is not a feature comes back as a note naming it, because the thing that must never
happen is nobody knowing.

| spec | frame | label | for |
|---|---|---|---|
| `bootstrap_jaw_v1` | grasp | `held` | the first scorer, over the physics-labelled bootstrap corpus |
| `bootstrap_object_frame_v0` | object | `held` | the object-frame alternative, kept on the record and so the frame check has a genuine mismatch to refuse |
| `valid_jaw_v1` | grasp | `valid` | stage one: does a candidate close. Fitted on the analytic verdict, which makes any evaluation against that same verdict self-referential |
| `held_jaw_v1` | grasp | `held` | stage two: does a closed grasp survive a shake. Same features, physics label, and the only one promotion is measured on |
| `willy_jaw_with_contact_angle` | base | `held` | deliberately declares a field that is not fit to be a feature, so the skew check stays honest against the case that motivated it |

The two-stage split is the point of the pair in the middle. Nothing measured on `valid` can distinguish
a model that learnt the world from one that learnt the analytic verdict; physics can. So `valid` trains
and the shake alone promotes.

A pooled positive rate over a shake corpus is an artefact of the draw rather than a property of the
corpus, because the shake is stratified and the strata sit far apart. Slice by `physics_source`, or
re-weight by the stratum sizes the shake writes beside its results.

## What is here

| file | what it is |
|---|---|
| [`gate.py`](gate.py) | the four checks and the verdict. Blocking findings stop a run; warnings are things to know. |
| [`sources.py`](sources.py) | file format to canonical columns, and where units and format quirks are paid for, once. |
| [`columns.py`](columns.py) | what one column contains: rows, distinct values, non-zero count, standard deviation, range, non-finite entries. |
| [`build.py`](build.py) | the flat feature table the learned ranker trains on: a fixed set of numbers per candidate. |
| [`clouds.py`](clouds.py) | the per-scene point clouds the learned generator trains on: fused, in BASE millimetres, with the environment and the arm in them. |
| [`split.py`](split.py) | splits one rendered dataset into disjoint linked views, because the cloud extractor is single-process per dataset and its `scenes` argument is a stride rather than a slice. |
| [`service.py`](service.py) | `RankerCorpus`, `RankerFit` and `CorpusCheck`: the library twins of `build-ranker-corpus`, `train-ranker` and `check-dataset`. |

## Two conversions in `sources.py` that are easy to get wrong

**Units are converted at the edge**, the way a vendor unit stays inside a driver: everything past this
module is in millimetres. A corpus that reached a comparison still in metres would report a
thousandfold distribution shift and read like a real finding.

**`np.load` on an `.npz` returns a lazy file object**, and every key access re-decompresses that whole
column. Indexing it inside a per-row comprehension turns a sub-second read into hours. Every loader
here hoists its columns in one pass instead.

There is no reader for a foreign corpus. Every source reads a format this package produces, so no
foreign unit or frame convention can enter through this module.

## On the duplication with the RL readiness check

The RL layer computes the same statistics over its own attempt-record logs and reaches the same verdict
on a constant column. This package re-implements rather than imports it, for two reasons: a datagen
package should not depend on the RL layer to borrow a dozen lines of arithmetic, and a pure-Python loop
over a large corpus takes minutes where numpy takes milliseconds.

Two implementations of the same arithmetic are only safe while something checks that they agree, so
both are run over the same rows and asserted to produce identical verdicts and identical numbers. The
duplication is a checked invariant, not a hope.
