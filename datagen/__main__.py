"""`python -m datagen.assets`: fetch the real meshes, or check whether this machine has them.

    --check                    what is present, per source. Exit 0 only when every source is ready.
    --fetch --from DIR         import meshes from a local directory into the gitignored library.
    --attribution              print the attribution text the licences oblige.

The meshes are not in the repository, for the same reason the MediaPipe bundles are not: they are
hundreds of megabytes of third-party binaries, and an operator may hold a different revision from
the one measured against. `--check` is what turns "it should be there" into an exit code.

Exit codes: 0 ready, 1 something is missing, 2 the request itself was wrong (bad source, absent
origin). A missing mesh library and a mistyped flag must not look alike to a script.
"""


from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict

from src.utility.log_cfg import create_logger

from datagen.assets.manifest import ManifestAuditError
from pydantic import ValidationError

from datagen.config import DatagenConfig
from datagen.constants import CLI_LOG_FILE, DATAGEN_LOG_DIR
from datagen.scenes.layout import layout_scene, plan_families

logger = create_logger("datagen.cli", CLI_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

_EXIT_OK = 0
_EXIT_PROBLEM = 1
_EXIT_USAGE = 2


def _config(args: argparse.Namespace) -> DatagenConfig:
    """The config, from a JSON file when given, with CLI overrides applied on top.

    The merge itself lives in `datagen.api.merged_settings`, so this door and the code-level one
    cannot drift. `--engine` names a key nested under `render`, so a caller who writes
    `{"engine": "mujoco"}` at the top level gets a config that keeps the engine it already had.
    """
    from datagen.api import merged_settings  # noqa: PLC0415

    base, _ = merged_settings(args.config, scenes=args.scenes, seed=args.seed,
                              domain=args.domain, engine=getattr(args, "engine", None))
    return DatagenConfig(**base)

def _first_problem(exc: Exception) -> str:
    """The one line an operator needs out of a pydantic/OS error, not the whole report."""
    if isinstance(exc, ValidationError):
        problems = exc.errors()
        if problems:
            first = problems[0]
            field = ".".join(str(part) for part in first.get("loc", ())) or "?"
            return f"{field}: {first.get('msg', 'invalid')} (got {first.get('input')!r})"
    return f"{type(exc).__name__}: {exc}"


def _cmd_plan(config: DatagenConfig) -> int:
    families = plan_families(config)
    counts = Counter(family.value for family in families)
    print(f"dataset: {config.scenes} scene(s), seed {config.seed}, domain {config.domain}")
    print(f"views:   {', '.join(config.camera_rig.views)}  "
          f"-> {config.scenes * len(config.camera_rig.views)} render(s)")
    print("families:")
    for name in sorted(counts):
        print(f"  {name:8} {counts[name]:5}")
    objects = sum(len(layout_scene(config, i).objects) for i in range(min(config.scenes, 25)))
    sampled = min(config.scenes, 25)
    print(f"objects: ~{objects / sampled:.1f} per scene (measured over the first {sampled})")
    return _EXIT_OK


def _cmd_describe(config: DatagenConfig, index: int) -> int:
    if not 0 <= index < config.scenes:
        print(f"scene index {index} outside 0..{config.scenes - 1}", file=sys.stderr)
        return _EXIT_USAGE
    scene = layout_scene(config, index)
    print(json.dumps(asdict(scene), indent=2, default=str))
    return _EXIT_OK


def _cmd_audit(config: DatagenConfig) -> int:
    """Build the manifest this config would draw from, and put it through the licence gate."""
    # The manifest the build would actually draw from. Synthesising a stand-in set of procedural
    # assets and auditing that instead makes the licence pre-flight answer about assets the build
    # will never place, so a customer whose config draws meshes learns nothing about the meshes they
    # are about to spend GPU hours on. A check that answers wrongly is worse than a missing one.
    from datagen.build import build_manifest  # noqa: PLC0415

    manifest = build_manifest(config)
    try:
        manifest.audit()
    except ManifestAuditError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_PROBLEM
    print(f"manifest OK: {len(manifest)} asset(s), every one renderable and shippable")
    print(f"  sources:  {sorted({r.source for r in manifest})}")
    print(f"  licences: {sorted({r.license for r in manifest})}")
    return _EXIT_OK


def _cmd_verify(config: DatagenConfig, *, name: str, out_root: str | None) -> int:
    """Check what a build actually wrote. No GPU: this is the one gate that can run anywhere."""
    from pathlib import Path

    from datagen.verify import verify_dataset

    root = Path(out_root) if out_root is not None else Path(config.output.root)
    report = verify_dataset(root / name)
    print(report.summary())
    return _EXIT_OK if report.ok and report.scenes else _EXIT_PROBLEM


def _cmd_preview(config: DatagenConfig, *, name: str, out_root: str | None) -> int:
    """Human-viewable copies of a written dataset. No GPU: it only re-colours what is on disk."""
    from pathlib import Path

    from datagen.preview import preview_dataset

    root = Path(out_root) if out_root is not None else Path(config.output.root)
    report = preview_dataset(root / name)
    print(report.summary())
    return _EXIT_OK if report.written else _EXIT_PROBLEM


def _cmd_prompts(config: DatagenConfig, *, name: str, out_root: str | None) -> int:
    """Referring expressions over a written dataset. No GPU: it reads scene.json and nothing else."""
    from pathlib import Path

    from datagen.prompts.build import build_prompts

    root = Path(out_root) if out_root is not None else Path(config.output.root)
    report = build_prompts(root / name)
    print(report.summary())
    return _EXIT_OK if report.ok else _EXIT_PROBLEM


def _cmd_build_cloud_corpus(args, config: DatagenConfig) -> int:  # noqa: ANN001 (argparse Namespace, as siblings take)
    """Extract the point-cloud corpus the learned generator trains on. No GPU, no Isaac, no model.

    One `.npz` per scene: the fused multi-view cloud in BASE, per-point instance / normal /
    observedness, the environment and arm channels, and the scene's grasp table joined by
    `(file, row_index)`.

    Not the same thing as `build-ranker-corpus`, which walks the same dataset and writes a flat
    feature table for the learned ranker. Clouds feed the generator; the table feeds the ranker.
    """
    from pathlib import Path

    from datagen.corpus.clouds import build_cloud_corpus

    # The config's output root, not a hard-coded one. Every sibling command reads
    # `config.output.root`, so a hard-coded root builds the dataset in one place and then looks for
    # it in another: with a same-named dataset sitting in that other directory, the result is a
    # corpus quietly built from the wrong scenes and stamped with the right name. `--out` still
    # overrides, which is what it is for.
    root = Path(args.out) if args.out is not None else Path(config.output.root)
    dataset = root / args.name
    if args.corpus_out is None:
        print("build-cloud-corpus needs --corpus-out DIR (one .npz per scene is written there)",
              file=sys.stderr)
        return _EXIT_USAGE
    destination = Path(args.corpus_out)
    if destination.suffix:
        # A second, independent guard. This command writes a directory; the ranker's writes a file,
        # and the two share the flag. A name check catches a file path here whatever the default
        # happens to be.
        print(f"build-cloud-corpus writes a DIRECTORY of per-scene clouds; {args.corpus_out!r} names "
              f"a file. Did you mean build-ranker-corpus, which writes one table?", file=sys.stderr)
        return _EXIT_USAGE
    try:
        from datagen.grasps.masks import PRED_SUFFIX

        suffix = "instances" if (args.masks or "gt") == "gt" else PRED_SUFFIX
        report = build_cloud_corpus(dataset, destination, scenes=args.scenes,
                                    mask_suffix=suffix, physics=args.physics,
                                    kinds=(("jaw", "suction") if args.kinds == "both"
                                           else (args.kinds,)),
                                    labels=args.labels)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("build-cloud-corpus refused: %s: %s", type(exc).__name__, exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    written = int(report.get("scenes_written", 0))
    print(f"{written}/{report.get('scenes_in', '?')} scene(s) -> {args.corpus_out}")
    # The keys `build_cloud_corpus` actually returns, checked against it rather than guessed: a
    # summary that prints nothing because the names were invented is worse than no summary.
    print(f"  points: {report.get('points', 0)} "
          f"({report.get('environment_points', 0)} environment, {report.get('arm_points', 0)} arm)")
    print(f"  grasps: {report.get('grasps', 0)} "
          f"({report.get('grasps_with_physics', 0)} with a physics verdict), "
          f"{report.get('contacts', 0)} contact(s)")
    if report.get("n_scenes_with_no_cloud"):
        # Named, not counted away: a scene that rendered but produced no cloud is the blank-RGB
        # class of defect, where a scene passes every check as ok and still carries nothing usable.
        print(f"  {report['n_scenes_with_no_cloud']} scene(s) produced NO cloud: "
              f"{', '.join(map(str, report.get('scenes_with_no_cloud', [])))}")
    if not written:
        # An empty extraction is a failure, not a result. Without this line the symptom shows up
        # hours later in a trainer rather than here, where the cause is.
        print("no scene produced a cloud; check the mask suffix and the dataset root",
              file=sys.stderr)
        return _EXIT_PROBLEM
    return 0


def _cmd_split_dataset(args, config: DatagenConfig) -> int:  # noqa: ANN001 (argparse Namespace, as siblings take)
    """Split one rendered dataset into disjoint views so the cloud extraction can use every core.

    `build-cloud-corpus` is single-process, and its `--scenes` flag cannot divide the work: it is an
    even stride, so two calls with it walk the same subset. This hands each process its own dataset
    instead.

    It prints the follow-up commands rather than leaving them to be composed, because the one
    thing that must not be got wrong is that every part needs its own `--corpus-out`. See
    `datagen/corpus/split.py` for why the shared-directory version fails, and what it costs.
    """
    from pathlib import Path

    from datagen.corpus.split import clean_split, split_dataset

    root = Path(args.out) if args.out is not None else Path(config.output.root)
    dataset = root / args.name
    try:
        if args.clean:
            removed = clean_split(dataset)
            print(f"{len(removed)} view(s) removed" if removed else "no split view to remove")
            for path in removed:
                print(f"  {path}")
            return _EXIT_OK
        report = split_dataset(dataset, args.parts)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("split-dataset refused: %s: %s", type(exc).__name__, exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    print(f"{report['scenes']} scene(s) -> {report['parts']} disjoint view(s), "
          f"linked not copied")
    for view in report["views"]:
        print(f"  {view['path']}  {view['scenes']} scene(s)")
    print("\nExtract them, one corpus directory EACH (a shared one is refused, and rightly):")
    for view in report["views"]:
        name = Path(view["path"]).name
        print(f"  python -m datagen build-cloud-corpus --name {name} "
              f"--out {root.as_posix()} --corpus-out logs/dl/clouds/{name}")
    print(f"\nThen: python -m datagen split-dataset --name {args.name} "
          f"--out {root.as_posix()} --clean")
    return _EXIT_OK


def _cmd_build_ranker_corpus(args, config: DatagenConfig) -> int:  # noqa: ANN001 (argparse Namespace, as siblings take)
    """Walk a rendered dataset and write a ranker training corpus. No GPU, no cell, no model.

    The work is in `datagen.corpus.service.RankerCorpus` and this is the argparse door to it. The
    walk, the physics join and the write live there together, so the whole path stays reachable from
    Python and not only from a shell.
    """
    from datagen.corpus.service import RankerCorpus

    corpus = RankerCorpus.from_config(
        config, name=args.name, out_root=args.out, scenes=args.scenes,
        mask_source="gt" if (args.masks or "gt") == "gt" else "pred",
        physics_path=args.physics)
    try:
        report = corpus.build(args.corpus_out)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("build-ranker-corpus refused: %s: %s", type(exc).__name__, exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE if isinstance(exc, FileNotFoundError) else _EXIT_PROBLEM
    print(report.render())
    return _EXIT_OK


def _cmd_train_ranker(args) -> int:  # noqa: ANN001 (argparse Namespace, as every sibling takes)
    """Fit the learned grasp ranker on a corpus and write its artifact. Offline: no cell, no policy.

    The work is in `datagen.corpus.service.RankerFit`. The CLI lives here and not in
    `grasping/deep/`, because the library may never import `datagen`: a generator that becomes a
    runtime dependency of a robot is a generator nobody can delete or decline to install. The fit
    itself has no such problem, because `deep.ranker.training.fit_ranker` takes a table, never a
    path.

    Exit 0 when the fit beats its width-only baseline on both AUROC and P@250, 1 when it does not,
    2 when the corpus cannot be read. The verdict is a comparison because an absolute is not one: an
    AUROC that sounds like a result says nothing until the straw man is scored on the same folds.

    Every flag here shares one namespace across every command, so a new one can collide on the flag,
    which argparse refuses outright, or silently on its `dest`, which hands one command's value to
    another. Check `dest`, not just the flag, against `build_parser` before adding one.
    """
    from datagen.corpus.service import RankerFit

    fit = RankerFit.from_corpus(
        args.corpus, source=args.source, spec=args.spec, model_dir=args.model_out,
        folds=args.folds, trees=args.trees, learning_rate=args.learning_rate,
        tree_depth=args.tree_depth)
    try:
        report = fit.fit()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("train-ranker refused: %s: %s", type(exc).__name__, exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    print(report.render())
    if not report.beats_baseline:
        logger.warning("%s does not beat its width-only baseline", report.spec)
    return _EXIT_OK if report.beats_baseline else _EXIT_PROBLEM


def _cmd_check_dataset(args) -> int:  # noqa: ANN001 (argparse Namespace, as every sibling takes)
    """Judge a corpus before anyone trains on it. Exit 0 = usable, 1 = refused, 2 = unreadable.

    Opens no cell, loads no model, trains nothing. Runs on a laptop, and is meant to run before the
    training code it protects exists, which is why it lives here and not beside a model.

    The work is in `corpus.service.CorpusCheck`, including the derived-spec branch, which is
    load-bearing: a derived spec's features are computed from the raw columns, so a gate that judged
    the raw ones would be judging what the model does not read.
    """
    from datagen.corpus.service import CorpusCheck

    try:
        report = CorpusCheck.from_corpus(
            args.corpus, source=args.source, spec=args.spec, serving=args.serve,
            serving_source=args.serve_source, serving_spec=args.serve_spec).assess()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(report.render(verbose=not args.quiet))
    return 0 if report.usable else 1


def _cmd_label_grasps(config: DatagenConfig, *, name: str, out_root: str | None,
                      density: str = "default", jaw: str | None = None,
                      label_budget: int | None = None) -> int:
    """Grasp labels over a written dataset. No GPU: closed-form geometry from the settled poses."""
    from pathlib import Path

    from datagen.grasps.labels import DENSITIES, label_dataset

    root = Path(out_root) if out_root is not None else Path(config.output.root)
    try:
        report = label_dataset(root / name, density=DENSITIES[density], jaw=jaw,
                               label_budget=label_budget)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return _EXIT_USAGE
    print(f"{report['labels']} label(s) over {report['objects']} object(s) in {report['scenes']} "
          f"scene(s): {report['jaw']} jaw, {report['suction']} suction "
          f"[gripper {report['jaw_model']}]")
    for family, bucket in sorted(report["by_family"].items()):
        objects = max(1, bucket["objects"])
        print(f"  {family:<8} {bucket['objects']:>4} objects, jaw-graspable "
              f"{bucket['objects_with_jaw'] / objects * 100:5.1f}%, suctionable "
              f"{bucket['objects_with_suction'] / objects * 100:5.1f}%")
    if report.get("partial"):
        print(f"  PARTIAL: stopped at the {report['label_budget']} label budget after "
              f"{report['scenes']} of {report['scenes_available']} scene(s). Families were walked "
              f"round robin, so this is a slice of the dataset and not its first families")
    for problem in report["problems"]:
        print(f"  PROBLEM: {problem}")
    return _EXIT_OK if report["labels"] and not report["problems"] else _EXIT_PROBLEM


def _cmd_predict_masks(config: DatagenConfig, *, name: str, out_root: str | None) -> int:
    """GPU: GroundingDINO + SAM2 over every rendered view, written as predicted instance maps.

    The second mask source. `eval-grasps --masks pred` then re-grades the calculator on those
    instead of the renderer's ground truth.
    """
    from pathlib import Path

    from datagen.grasps.masks import predict_masks

    root = (Path(out_root) if out_root is not None else Path(config.output.root)) / name
    report = predict_masks(root)
    print(f"{report['found']}/{report['objects']} objects found "
          f"({report['recall'] * 100:.1f}%) over {report['views']} views; "
          f"median IoU {report['iou'].get('median', 0):.3f}")
    print(f"  {report['detections_per_view']} detections/view, "
          f"{report['unmatched_detections_per_view']} of them matching no object")
    for family, bucket in sorted(report["by_family"].items()):
        print(f"  {family:<8} recall {bucket['recall'] * 100:5.1f}%  "
              f"median IoU {bucket['median_iou']:.3f}  ({bucket['objects']} objects)")
    for band, bucket in sorted(report["by_visibility"].items()):
        if bucket["objects"]:
            print(f"  visibility {band}  recall {bucket['recall'] * 100:5.1f}%  "
                  f"median IoU {bucket['median_iou']:.3f}  ({bucket['objects']} objects)")
    return _EXIT_OK if report["objects"] else _EXIT_PROBLEM


def _cmd_side_approach(args: argparse.Namespace, config: DatagenConfig) -> int:
    """Coverage by the object's most top-down admissible grasp.

    The number the ladder's `top1` cannot give. One rate cannot distinguish "found a side grasp
    for a short object" from "found a top grasp for a tall one", so a generator could solve the whole
    off-vertical population and the headline would barely move. Reads the rows a previous
    `eval-grasps` wrote: no re-run, no GPU.

    All of the file i/o is in `eval.service.GraspEvaluation.approach_tilt`, so locating
    `grasps.jsonl`, globbing `grasp_eval*.jsonl` and parsing both is done in one place rather than
    by every caller.
    """
    from datagen.eval.service import GraspEvaluation

    files = [args.rungs] if args.rungs and args.rungs.endswith(".jsonl") else None
    try:
        report = GraspEvaluation.from_config(
            config, name=args.name, out_root=args.out).approach_tilt(files=files)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_USAGE
    print(report.render())
    return _EXIT_OK


def _cmd_heldout(args: argparse.Namespace, config: DatagenConfig) -> int:
    """Which placeable assets a trained corpus never covered.

    The list a held-out dataset is drawn from. Feed the output to `assets.mesh_asset_ids_path` and
    the generator is measured on objects it has never seen, which the eval ladder's own reference
    dataset cannot do: its fold groups are the training corpus's.
    """
    from pathlib import Path

    from datagen.heldout import format_report, held_out_assets

    corpus = Path(args.corpus or "")
    if not args.corpus or not corpus.is_dir():
        print(f"heldout needs --corpus DIR (a written dataset or a sharded corpus); "
              f"{corpus!s} is not a directory", file=sys.stderr)
        return _EXIT_USAGE

    report = held_out_assets(corpus, max_extent_mm=config.assets.max_mesh_extent_mm)
    print(format_report(report))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report.unseen, indent=1), encoding="utf-8")
        print(f"\n  written: {args.out}"
              f"\n  use it:  assets.mesh_asset_ids_path = {args.out!r}")

    # An empty list is a problem, not a result, and it exits non-zero so a pipeline cannot build a
    # "held-out" dataset from nothing and report success. A corpus that used every mesh the bank can
    # yield has no held-out set at all.
    return _EXIT_OK if report.total_unseen else _EXIT_PROBLEM


def _cmd_cost(args: argparse.Namespace, config: DatagenConfig) -> int:
    """What a dataset will cost, before anybody starts it.

    The first question an operator asks. Every coefficient is measured rather than assumed, and
    carries its evidence into the output: see `datagen/cost.py` for what each was measured over.
    """
    from datagen.cost import ENGINE_COSTS, estimate, format_estimate

    # The config's own scene count. `cost` exists to price a corpus before somebody spends the night
    # on it, so reading a scene count the config did not ask for prints a confident total for a
    # different corpus, with nothing anywhere saying so. `--scenes` still wins, because a what-if is
    # the other half of what this command is for.
    scenes = int(args.scenes) if args.scenes else int(config.scenes)
    # The engine comes from the config like everything else, so `--config x.json` answers for the
    # dataset that config would actually build. `--engine` overrides it for a what-if.
    engine = args.engine or config.render.engine
    # How many distinct scanned meshes this config would decompose. Counted from the id restriction
    # when there is one, which is a JSON read, rather than by loading the bank, which is two minutes
    # of mesh parsing and would make the cheap command the slow one. Zero when nothing restricts the
    # draw, because then the honest answer is "as many as the bank holds" and this tool does not
    # guess at a number it would then print as a cost.
    meshes = 0
    if engine == "mujoco":
        from datagen.scenes.layout import resolve_mesh_asset_ids  # noqa: PLC0415

        restricted = resolve_mesh_asset_ids(config)
        meshes = len(restricted) if restricted else 0
    try:
        result = estimate(scenes, engine=engine, jobs=max(1, int(args.jobs)),
                          epochs=int(args.epochs), folds=max(1, int(args.train_folds)),
                          refit=not args.no_refit, dense=(args.density == "dense"),
                          meshes=meshes)
    except ValueError as exc:
        print(f"cost: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    print(format_estimate(result))
    if not args.engine:
        others = [name for name in sorted(ENGINE_COSTS) if name != engine]
        print(f"\n  compare: --engine {' | --engine '.join(others)}")
    return _EXIT_OK


def _cmd_camera_probe(args: argparse.Namespace) -> int:
    """Drive the real camera adapter over a written dataset. No camera, no robot, no GPU.

    A bench for the d435 that needs no d435: `RealSenseVisionPerceptionSource` is the adapter every
    real-hardware pick goes through, and it has only ever met a fake streamer. A datagen dataset
    ships sensor-shaped depth and masks a real detector produced, so the adapter can be driven over
    hundreds of realistic frames. It validates everything above the sensor; the sensor itself still
    needs real hardware.
    """
    from pathlib import Path

    # The camera model, intrinsics and look-at convention live in `render/camera.py`. This bench,
    # which drives the real RealSense adapter over rendered scenes, is `eval/camera_replay.py`:
    # two top-level things named camera in one package is the worst navigation trap this package
    # can have, so no module sits under `datagen/camera/` beside them.
    from datagen.eval.camera_replay import probe_dataset, render_report

    config = _config(args)
    root = (Path(args.out) if args.out is not None else Path(config.output.root)) / args.name
    if not (root / "scenes").is_dir():
        print(f"no scenes under {root}; build or point --out/--name at a written dataset")
        return _EXIT_PROBLEM
    masks = args.masks or "pred"     # the realistic pair; `--masks gt` is the control
    probes = probe_dataset(root, limit=args.limit, depth_source=args.depth, mask_source=masks)
    if not probes:
        print(f"{root} has no rendered view to probe")
        return _EXIT_PROBLEM
    print(f"dataset {root}   depth={args.depth}   masks={masks}")
    print()
    print(render_report(probes))
    # An adapter that raised on any view is the finding, and it must reach the exit code.
    return _EXIT_PROBLEM if any(p.error for p in probes) else _EXIT_OK


def _cmd_gate(config: DatagenConfig, *, name: str, out_root: str | None, write: bool) -> int:
    """The regression gate: two rungs on a fixed subset, against a committed baseline.

    Minutes, not an hour. Exit 1 on a drop, so it can sit in front of a commit that touches the grasp
    path and say whether the change made grasping better or worse on data the calculator has never
    seen.
    """
    from pathlib import Path

    from datagen.eval.ladder import GATE_SCENE_STEP, run_gate

    root = (Path(out_root) if out_root is not None else Path(config.output.root)) / name
    result = run_gate(root, write=write)
    if write:
        print(f"baseline written to {result['wrote']}")
        for rung, rates in sorted(result["rates"].items()):
            print(f"  {rung:<12} TOP-1 {rates['top1'] * 100:5.2f}%  "
                  f"precision {rates['precision'] * 100:5.2f}%  "
                  f"coverage {rates['coverage'] * 100:5.2f}%  "
                  f"(incl. unseen {rates['coverage_incl_unseen'] * 100:5.2f}%, "
                  f"n={rates['objects_with_labels']})")
        print("  commit it; a gate whose reference is regenerated on failure is not a gate")
        return _EXIT_OK
    print(f"every {GATE_SCENE_STEP}th scene, against {result['baseline']} "
          f"(tolerance {result['tolerance_pp']} pp):")
    for rung, delta in sorted(result["deltas"].items()):
        if delta.get("missing"):
            print(f"  {rung:<12} MISSING from this run")
            continue
        marks = {k: ("" if abs(v) <= result["tolerance_pp"] else ("  <-- DROP" if v < 0 else "  <-- up"))
                 for k, v in delta.items() if k != "n"}
        # top1 first, because it is the only one of the four a pick actually experiences: the robot
        # takes candidates[0]. A gate that compares a number without printing it hides the decisive
        # one, which is the habit this gate exists to break.
        print(f"  {rung:<12} TOP-1 {delta['top1']:+6.2f} pp{marks['top1']:<11}"
              f"precision {delta['precision']:+6.2f} pp{marks['precision']:<11}"
              f"coverage {delta['coverage']:+6.2f} pp{marks['coverage']:<11}"
              f"incl.unseen {delta['coverage_incl_unseen']:+6.2f} pp  (n={delta['n']})")
    print("PASS" if result["ok"] else "FAIL; a rate dropped further than the tolerance")
    return _EXIT_OK if result["ok"] else _EXIT_PROBLEM


def _cmd_decompose(config: DatagenConfig, *, jobs: int, scenes: int | None) -> int:
    """Fill the convex-decomposition cache for everything this config will place, in parallel.

    Run this before a mujoco build, not instead of it. The engine decomposes on demand and caches,
    so a build works without this; it just pays seconds per mesh inside its own render loop, one at
    a time, and nearly all of that time is CoACD. Across several processes the same work is minutes,
    and the build then finds every entry warm.

    The work is in `assets.service.MeshPreparation.decompose`, so the manifest walk, the process
    pool and the cached-versus-computed split are reachable from Python and not only from a shell.
    """
    from datagen.assets.service import MeshPreparation

    try:
        report = MeshPreparation.from_sources().decompose(config, jobs=jobs, scenes=scenes)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{report.assets} scanned asset(s) decomposed over {jobs} process(es)")
    print(report.render())
    return _EXIT_OK


def _resolve_jobs(jobs: int) -> int:
    """``0`` means cpu_count minus 2, to leave the box usable; anything else verbatim, floored at 1."""
    import os

    if jobs != 0:
        return max(1, int(jobs))
    return max(1, (os.cpu_count() or 2) - 2)


def _cmd_eval_grasps(config: DatagenConfig, *, name: str, out_root: str | None,
                     summary_only: bool, masks: str = "gt", jobs: int = 1,
                     mask_completion: str = "none", rungs: str | None = None,
                     limit: int | None = None) -> int:
    """Grade GraspCalculator against those labels. CPU only, but minutes-to-hours: it re-runs the
    generator on every object in every view, in every configuration.

    The work is in `eval.service.GraspEvaluation`, including the output-file policy, which is the
    load-bearing part: `grasp_eval.jsonl` is appended across runs, so an exploratory run over two
    scenes writes rows from a throwaway model into the file every result is quoted from, plus a torn
    line when it is interrupted, and a later `--summary-only` reports that model as a result. The
    naming rule is `eval.service.evaluation_output_name`.

    ``--summary-only`` does not run anything: it prints the last report. It returns in seconds and
    can carry rungs that no longer exist, which reads exactly like a fast run.

    ``--jobs`` spreads the scenes over processes. A scene reads only its own files, so the ladder
    parallelises almost perfectly, and on a many-core box that is the whole difference.
    """
    from datagen.eval.service import GraspEvaluation

    evaluation = GraspEvaluation.from_config(config, name=name, out_root=out_root)
    try:
        report = (evaluation.last_report() if summary_only
                  # `limit` goes to `evaluate_dataset`, which takes it. Stopping it here instead
                  # leaves `--limit 3` accepted, warning-free, and evaluated over the whole
                  # dataset, so a run launched as a smoke test returns the full figure.
                  else evaluation.evaluate(mask_source="pred" if masks != "gt" else "gt",
                                           mask_completion=mask_completion, rungs=rungs,
                                           jobs=jobs, limit=limit))
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return _EXIT_USAGE
    print(report.render())
    return _EXIT_OK if report.rows else _EXIT_PROBLEM


def _cmd_physics_sample(config: DatagenConfig, *, name: str, out_root: str | None,
                        headless: bool, per_class: int, physics_engine: str = "isaac",
                        proposals: str | None = None) -> int:
    """On-box: does a grasp the geometry calls valid actually hold? Samples both the accepted and
    the rejected, because only the second half can show the reference discriminates.

    The work is in `grasps.service.PhysicsSampling`, including the output-file rule: a proposal
    verdict and a label verdict answer different questions, and a shared file would let one be quoted
    as the other. The naming rule is `physics_output_name`.
    """
    from datagen.grasps.service import PhysicsSampling

    sampling = PhysicsSampling.from_config(
        config, name=name, out_root=out_root, engine=physics_engine,  # type: ignore[arg-type]
        headless=headless)
    if proposals:
        print(f"  judging the proposals in {proposals}")
    report = sampling.sample(per_class=per_class, proposals=proposals)
    if proposals:
        print(f"\n  REFEREE SUCCESS RATE: {report.held}/{report.trials} = "
              f"{report.hold_rate * 100:.1f}%")
        print("  the plan's PRIMARY metric, on the model's own proposals")
    print(report.render())
    return _EXIT_OK if report.trials else _EXIT_PROBLEM


def _cmd_physics_compare(config: DatagenConfig, *, name: str, out_root: str | None,
                         headless: bool, per_class: int, configs: str,
                         physics_engine: str = "isaac") -> int:
    """On-box: which of two generators actually holds more, on the same objects.

    The promotion gate. `paired_trials` is the only instrument here that can compare two generators
    without object difficulty sitting inside the comparison: `sample_trials` gives each arm its own
    strata, so the two land on different objects, and in a corpus where some objects fail for reasons
    that have nothing to do with which generator proposed the grasp, that variance is not small.

    An arm that proposed nothing is a refusal, not a tie. Objects where any arm proposed nothing
    are dropped from the draw and counted, because "arm B held 60 % of what it proposed" says nothing
    without "and it proposed on 40 % fewer objects". Both halves are printed.
    """
    from datagen.grasps.service import PhysicsSampling

    sampling = PhysicsSampling.from_config(
        config, name=name, out_root=out_root, engine=physics_engine,  # type: ignore[arg-type]
        headless=headless)
    try:
        report = sampling.compare([c for c in configs.split(",")], per_class=per_class)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_USAGE if "at least two" in str(exc) else _EXIT_PROBLEM
    print(report.render())
    return _EXIT_OK if report.trials else _EXIT_PROBLEM


#: The rest poses `why-no-jaw` tries, named for the report. Read from the module that defines them so
#: the printed count cannot drift from the measurement.
REST_LABELS = [name for name, _ in __import__(
    "datagen.assets.diagnose", fromlist=["REST_ROTATIONS"]).REST_ROTATIONS]


def _cmd_why_no_jaw(config: DatagenConfig, *, collections_: list[str] | None, limit: int,
                    from_screen: str | None = None,
                    density: str, out: str | None) -> int:
    """Why do these meshes earn no grasp? The question a user with their own parts actually asks.

    The selection is in `assets.service.MeshPreparation.why_no_jaw`, which is where it can be tested.

    The answer refutes the obvious guess. Over the candidates tried on meshes that score zero, the
    commonest refusal by a wide margin is `too_wide`, and `below_table` is rare: thin flat objects
    look as though the fingers must meet the table underneath them, and they do not. What refuses
    them is the width along the line tried.

    And the draw is even, never a prefix. Appending entries until `limit` of them are collected and
    breaking out of both loops returns twenty meshes from the first source and none from any other:
    asset ids sort by source and then by name, so a prefix of that list is one collection wearing
    the name of all of them. `assets.service.even_sample` is the draw.
    """
    from pathlib import Path

    from datagen.assets.service import MeshPreparation

    preparation = MeshPreparation.from_sources(collections_)
    try:
        diagnosis = preparation.why_no_jaw(
            limit=limit, density=density, from_screen=from_screen,
            report=lambda line: print(line, flush=True))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_PROBLEM
    print(f"  probed {diagnosis.probed_meshes} mesh(es) at density {density}, "
          f"in {len(REST_LABELS)} rest pose(s)")
    print(diagnosis.render())
    if out:
        Path(out).write_text(json.dumps(diagnosis.as_dict(), indent=2, sort_keys=True),
                             encoding="utf-8")
        print(f"  wrote {out}")
    return _EXIT_OK


def _cmd_verify_robot(config: DatagenConfig, *, headless: bool, require_curobo: bool) -> int:
    """On-box: does the arm land where it is claimed to? Asks USD, not this package's arithmetic."""
    from datagen.verify_robot import verify_robot

    # verify_robot prints its own summary before Isaac tears the process down; see the note there.
    report = verify_robot(
        config.render.arm.robot_model, headless=headless, require_curobo=require_curobo,
    )
    return _EXIT_OK if report.ok else _EXIT_PROBLEM


#: Where a screen lands when the caller names no path. Under the package rather than under `logs/`
#: because a config points at it: the verdict and the asset list are inputs to a build, not output
#: of one, and `logs/` is gitignored, so a screen written there is gone on the next clone.
_DEFAULT_SCREEN = "datagen/assets/screens/screen.json"


def _cmd_build(
    config: DatagenConfig, *, name: str, out_root: str | None, headless: bool, preview: bool,
) -> int:
    """The on-box path. Imported here so the laptop commands never touch anything Isaac-shaped."""
    from datagen.build import build_dataset

    summary = build_dataset(
        config, name=name, out_root=out_root, headless=headless, preview=preview,
    )
    rendered = summary["by_status"].get("ok", 0)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return _EXIT_OK if rendered else _EXIT_PROBLEM


def _library_sources() -> list[str]:
    from datagen.assets.library import SUPPORTED_SOURCES

    return [s for s in SUPPORTED_SOURCES if s != "custom"]


def _screen_density(args: argparse.Namespace) -> str:
    """`--density` for a screen defaults to what the corpus is labelled with, not to `default`.

    The shared `--density` flag defaults to `default`, because `label-grasps` at that setting
    reproduces every label set this repo has written. A screen taken there and one taken at `grid`
    are different measurements of the same meshes, an order of magnitude apart in label count, so
    inheriting the flag's default silently produces a much weaker verdict than the one the corpus
    would get, and the two files look identical.
    """
    return str(args.density) if args.density is not None else "grid"


def _cmd_normalise_meshes(*, collections: list[str], jobs: int, faces: int,
                          scale: float | None) -> int:
    """Scale a fetched collection into metres and decimate it into the face budget.

    The loop over collections is in `assets.service.MeshPreparation`. `normalise_meshes` handles one
    source; deciding which ones is one rule in one place, for the three commands that need it.
    """
    from datagen.assets.service import MeshPreparation

    results = MeshPreparation.from_sources(collections).normalise(
        faces=faces, scale=scale, jobs=jobs)
    total = sum(r["meshes"] for r in results.values())
    changed = sum(r["changed"] for r in results.values())
    print(f"{changed} of {total} mesh(es) changed")
    return _EXIT_OK


def _cmd_screen_meshes(*, collections: list[str], out: str, density: str, jobs: int,
                       want_graspable: int | None = None) -> int:
    """Ask the labeller which meshes are graspable, in every rest pose, and write the verdict.

    Both files belong to the service. `screen_meshes` writes the screen; the
    `<stem>_asset_ids.json` that every downstream config actually points at is derived from it, so a
    Python caller gets the screen and the thing built from it together.
    """
    from datagen.assets.service import MeshPreparation

    report = MeshPreparation.from_sources(collections).screen(
        out, density=density, jobs=jobs, want_graspable=want_graspable)
    if not report.rows:
        return _EXIT_PROBLEM
    print(f"\nscreen    -> {report.screen_path}\nasset ids -> {report.ids_path}")
    print("Point assets.jaw_screen_path and assets.mesh_asset_ids_path at those two, then run "
          "`decompose` BEFORE `build`.")
    return _EXIT_OK


def _cmd_prepare_assets(*, collections: list[str], out: str, density: str, jobs: int,
                        faces: int, want_graspable: int | None = None) -> int:
    """normalise, then screen, in that order, and say what still has to happen."""
    from pathlib import Path as _Path

    from datagen.assets.prepare import prepare_assets

    destination = _Path(out)
    summary = prepare_assets(
        collections or None, screen_out=destination,
        ids_out=destination.with_name(destination.stem + "_asset_ids.json"),
        density=density, faces=faces, jobs=jobs, want_graspable=want_graspable)
    return _EXIT_OK if summary["assets"] else _EXIT_PROBLEM


# One declaration, two doors. These belong to `corpus.service`, which is where the library caller
# reads them; the parser imports rather than restating, so the Python twin and the identical CLI
# call cannot disagree about their own defaults.
from src.contracts.options import UNSET  # noqa: E402
from datagen.corpus.service import DEFAULT_SOURCE as _DEFAULT_SOURCE  # noqa: E402
from datagen.corpus.service import DEFAULT_SPEC as _DEFAULT_SPEC  # noqa: E402


def _fit_default(name: str) -> str:
    """`(default: 400)` read off `fit_ranker` itself, so --help cannot quote a stale number.

    Imported lazily and failing soft. `--help` must work on a box with no sklearn, and the fit lives
    in `src` behind an optional dependency. A parser that cannot be built is worse than a help
    string that omits one number.
    """
    try:
        import inspect

        from src.robot.grasping.deep.ranker.training import fit_ranker
        return f"(default: {inspect.signature(fit_ranker).parameters[name].default})"
    except Exception:                                          # noqa: BLE001 (help text only)
        return "(default: the trainer's)"


def build_parser() -> argparse.ArgumentParser:
    """The parser, built once so something other than `main` can read it.

    Extracted because the subcommands cannot be recovered from the rendered usage line: the first
    `{a,b,c}` group there is `--engine`'s choices, not the command list. A guard reading that line
    checks names no command has, and passes for the same reason it would pass if every real command
    were broken.

    Text was the wrong source: the same brackets mean subcommands in the sibling CLI and a flag's
    choices here, and no care with string offsets fixes that. A parser can be asked.
    """
    parser = argparse.ArgumentParser(prog="python -m datagen", description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=["plan", "describe", "audit", "verify", "verify-robot", "preview",
                 "prompts", "label-grasps", "predict-masks", "eval-grasps", "grasp-gate",
                 "check-dataset", "build-cloud-corpus", "build-ranker-corpus", "split-dataset",
                 "train-ranker", "physics-sample", "physics-compare", "why-no-jaw",
                 "camera-probe", "cost", "heldout", "side-approach",
                 "decompose", "normalise-meshes", "screen-meshes", "prepare-assets",
                 "init-config", "build"])
    parser.add_argument(
        "--recipe", default=None, metavar="NAME",
        help="init-config: which named, versioned corpus recipe to write. A recipe is a bundle of "
             "the settings a MEASUREMENT or a stated product decision justifies, and it is frozen "
             "once it ships, so a better bundle becomes v2 beside v1. The training half of the same "
             "recipe is `deep train-set --recipe NAME`.")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="eval-grasps: worker processes over scenes (0 = auto, cpu_count - 2). Default 1 "
             "(sequential). Output is byte-identical either way; results are consumed in "
             "submission order, not completion order.")
    parser.add_argument(
        "--engine", type=str, default=None, choices=sorted(("isaac", "mujoco", "none")),
        help="cost: price a DIFFERENT engine than the config names, for a what-if",
    )
    parser.add_argument(
        "--proposals", type=str, default=None, metavar="JSONL",
        help="physics-sample: judge a trained model's OWN grasps instead of the corpus labels, "
             "written by `deep propose`. This is what the architecture plan's PRIMARY metric is "
             "made of, the referee success rate: does a grasp the MODEL proposes actually hold. "
             "Every physics run before this judged grasps drawn from the labels, which answers a "
             "different question. The verdicts land in their own file and keep source=generator, "
             "because two numbers answering different questions must not share a name",
    )
    parser.add_argument(
        "--physics-engine", type=str, default="isaac", choices=("isaac", "mujoco"),
        help="physics-sample / physics-compare: which simulator referees the shake. Isaac is Windows "
             "or Linux, NVIDIA only and about 47 GB, and it was the ONE step in this pipeline that "
             "forced a workstation: the v5 corpus with 8 million labels was RENDERED on mujoco and "
             "says so in its provenance. Both engines pass the same four harness controls. That "
             "makes each a working harness and does NOT make their verdicts interchangeable, so the "
             "engine is written into every row",
    )
    parser.add_argument(
        "--configs", type=str, default="sfe_fused,fused", metavar="A,B",
        help="physics-compare: the generator arms to shake against each other, comma separated. "
             "Every drawn object contributes one trial to EVERY arm, which is what makes the "
             "comparison paired: sample_trials gives each arm its own strata and they land on "
             "different objects, measured on v1_proof as 104 against 153, so object difficulty sits "
             "inside an unpaired comparison as variance. An object where any arm proposed nothing is "
             "dropped and counted, because a hold rate says nothing without how often the arm "
             "refused",
    )
    parser.add_argument(
        "--kinds", type=str, default="jaw", choices=("jaw", "suction", "both"),
        help="build-cloud-corpus: which grasp kinds reach the corpus. `jaw` is what every corpus "
             "this repository has built and stays the default, so an existing one is reproduced "
             "byte for byte. MEASURED over every label report on disk, on v5 across 174,438 "
             "objects: 36.2 %% of objects earn a jaw label and 87.5 %% earn a suction one, and the "
             "suction half is only 9.3 %% of the label count. So it is 2.4x the object coverage for "
             "a tenth of the labels, and the plan's `68.5 %% of training units are empty` is "
             "measured on jaw labels alone. NOT a fix for the pose heads, which cannot learn a jaw "
             "pose from a suction label; it changes the WHERE stage, which today learns to call a "
             "suction-only object empty",
    )
    parser.add_argument(
        # No default on the flag, and that is the point. With a default value here, deciding
        # whether the operator chose means comparing the value to it, and then typing
        # `--density default` on a screen is indistinguishable from not typing it at all, so the
        # explicit choice is overridden with `grid`. That is the same defect as deciding "not
        # given" by scanning sys.argv, wearing a different costume.
        #
        # And the consequence is measured, in `_screen_density`'s own words: the same object earns
        # an order of magnitude more labels at `grid` than at `default`. The two reports look
        # identical.
        #
        # `None` means nobody said. Each command supplies its own default, which is what the two
        # differ on: `label-grasps` wants `default` because it reproduces every label set this repo
        # has written, and a screen wants `grid` because a screen taken at `default` grades meshes
        # far more weakly than the corpus ever will.
        "--density", type=str, default=None, choices=("default", "dense", "grid"),
        help="label-grasps: how finely each object is sampled. `default` reproduces every label set "
             "this repo has written. `dense` is ~5.5x the labels AND guarantees the top-down approach "
             "per closing axis, which uniform azimuth sampling reaches only by coincidence; the "
             "shipped corpus sits a median 61 degrees off vertical. The choice is stamped into "
             "grasp_label_report.json, because a label COUNT means nothing without it.",
    )
    parser.add_argument(
        "--out-ids", dest="out", type=str, default=None, metavar="JSON",
        help="heldout: write the unseen asset ids here, ready for assets.mesh_asset_ids_path",
    )
    parser.add_argument(
        "--collection", action="append", default=None, metavar="NAME",
        help="normalise-meshes / screen-meshes / prepare-assets: restrict to one mesh collection "
             "(repeatable). Default is every one the library supports.",
    )
    parser.add_argument(
        "--from-screen", dest="from_screen", type=str, default=None, metavar="JSON",
        help="why-no-jaw: ask only about the meshes a SCREEN REPORT says earn no jaw label at all, "
             "instead of a sample of every mesh. The report is what `screen-meshes` writes. This is "
             "the sharper question by a wide margin: on the reference screen the meshes that earn "
             "nothing are a small minority, so a sample of everything spends most of its budget on "
             "objects that already work and reports their reasons.",
    )
    parser.add_argument(
        "--label-budget", type=int, default=None, metavar="N",
        dest="label_budget",
        help="label-grasps: stop once N labels have been written. The companion to "
             "--want-graspable at the other end of the pipeline, for when the constraint is disk "
             "or training time rather than objects. Families are walked ROUND ROBIN under a budget, "
             "because scene directories sort by family and a plain prefix would hand you every bin "
             "scene and no sparse one; the report is stamped partial either way",
    )
    parser.add_argument(
        "--want-graspable", type=int, default=None, metavar="N",
        dest="want_graspable",
        help="screen-meshes / prepare-assets: stop once N meshes have turned out to be "
             "JAW-GRASPABLE, and report how many had to be looked at to find them. This is the "
             "unit a corpus is actually budgeted in: a library is not 800,000 usable objects, and "
             "the yield differs by collection. MEASURED: 36.25 %% of screened meshes earn a jaw "
             "label across the v5 bank, 6.1 %% on objaverse. The screen it writes is STAMPED "
             "partial, because a stopped screen and a complete one are different statements about "
             "a library and the files are otherwise identical",
    )
    parser.add_argument(
        "--faces", type=int, default=20000, metavar="N",
        help="normalise-meshes / prepare-assets: decimate anything above this triangle count. "
             "Default 20000, which is the band the convex decomposition was measured at (5 to 30 s "
             "per mesh). A scanned collection arrives at 3 to 10 MILLION and is out of reach there.",
    )
    parser.add_argument(
        "--scale", type=float, default=None,
        help="normalise-meshes: override the recorded factor for a collection. Meshes are read as "
             "METRES, so one authored in anything else lands wrong by that factor in every axis and "
             "by its cube in mass, while every check downstream still passes.",
    )
    parser.add_argument(
        "--train-folds", type=int, default=1, metavar="N",
        help="cost: how many cross-validation folds the training run will make. Default 1, which is "
             "`TrainingPlan.run_folds`. NOT `--folds`, which belongs to train-ranker and defaults "
             "to 5; two different numbers with one obvious name is how a budget goes 5x wrong",
    )
    parser.add_argument(
        "--no-refit", action="store_true",
        help="cost: the trainer refits on the whole corpus after the folds by default; this drops "
             "that pass from the estimate",
    )
    parser.add_argument(
        "--epochs", type=int, default=0, metavar="N",
        help="cost: also price N training epochs over the corpus (0 = data only)",
    )
    parser.add_argument(
        "--rungs", type=str, default=None, metavar="A,B",
        help="eval-grasps: measure only these rungs instead of the whole ladder. The full ladder is "
             "hours; a comparison is usually a PAIR, e.g. --rungs sfe_fused,deep. An unknown name "
             "refuses and lists what there was. `deep` needs WILLY_DEEP_ARTIFACT set to a trained "
             "generator .pt (NOT the .ckpt.pt; a checkpoint is refused by kind, on purpose)")
    parser.add_argument("--config", type=str, default=None, help="JSON file of DatagenConfig fields")
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--domain", type=str, default=None, choices=["cell", "tabletop"])
    parser.add_argument("--index", type=int, default=0, help="describe: which scene")
    parser.add_argument("--name", type=str, default="v1", help="build: dataset name (a directory)")
    parser.add_argument("--out", type=str, default=None,
                        help="build: dataset root (default: output.root, i.e. data/datagen)")
    parser.add_argument(
        "--labels", type=str, default="grasps.jsonl",
        help="build-cloud-corpus: which grasp file inside the dataset to join. Not the default when "
             "the scenes were labelled for another gripper (grasps_jaw_<name>.jsonl). A non-default "
             "file changes the stamped dataset identity, so the overwrite guard can tell two "
             "grippers' clouds apart in one directory.",
    )
    parser.add_argument(
        "--jaw", type=str, default=None,
        help="label-grasps: label for a DIFFERENT gripper than the 2F-85, by name (see "
             "PROCEDURAL_JAWS). Writes grasps_jaw_<name>.jsonl, never the corpus file. Exists so the "
             "generator's gripper input VARIES: with one gripper it is constant, carries no "
             "gradient, and the conditioning seam cannot be shown to do anything.",
    )
    parser.add_argument(
        "--parts", type=int, default=4,
        help="split-dataset: how many disjoint views to build. One per process you intend to run "
             "build-cloud-corpus in; on this box sixteen shards is the measured ceiling ACROSS all "
             "of them, not per shard.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="split-dataset: remove the part views instead of building them. Removes only the "
             "links, never the scenes they point at, and refuses any directory without the marker.",
    )
    parser.add_argument("--gui", action="store_true", help="build: open the Isaac window")
    parser.add_argument(
        "--preview", action="store_true",
        help="build: also write the viewable strips as each scene lands (see the preview command)",
    )
    parser.add_argument(
        "--mask-completion", type=str, default="none",
        choices=["none", "axis_aligned_box", "oriented_box"],
        help="eval-grasps: apply a mask-completion policy BEFORE the calculator sees a mask; the "
             "transform both perception sources apply and this ladder never has. `none` is what every "
             "number in this repo was measured with; `axis_aligned_box` is what a real cell does "
             "today. Writes its own file, so two policies cannot overwrite each other.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="camera-probe / eval-grasps: stop after this many scenes (default: the whole dataset). "
             "WARNING: scenes are walked in sorted order and scene ids sort by FAMILY, so a limit smaller "
             "than one family returns only that family; the run warns when it does.",
    )
    parser.add_argument(
        "--depth", type=str, default="noisy", choices=["noisy", "clean"],
        help="camera-probe: which depth to feed the adapter. `noisy` is the sensor-shaped one and "
             "the only honest default; `clean` is Isaac's exact depth, which no camera produces.",
    )
    parser.add_argument(
        "--per-class", type=int, default=40,
        help="physics-sample: trials per source class (labels, accepted, each rejection reason)",
    )
    parser.add_argument(
        "--write-baseline", action="store_true",
        help="grasp-gate: replace the committed baseline with this run instead of checking against it",
    )
    parser.add_argument(
        "--masks", type=str, default=None, choices=["gt", "pred"],
        help="eval-grasps / camera-probe: which segmentation to feed; the renderer's ground truth "
             "or GroundingDINO+SAM2's (run predict-masks first). Reported separately, never merged. "
             "Defaults differ by command, deliberately: eval-grasps defaults to `gt` so the "
             "calculator is isolated from the perception ahead of it; camera-probe defaults to "
             "`pred`, the pair a real cell actually gets. For camera-probe `gt` is then the CONTROL; "
             "a rule that fires as often on complete masks is not detecting incompleteness.",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="eval-grasps: re-derive every rate and curve from the existing grasp_eval.jsonl "
             "instead of re-running the generator",
    )
    parser.add_argument(
        "--corpus",
        help="check-dataset: the corpus to judge (a .npz or .jsonl, per --source)",
        # Shared with `check-dataset`, deliberately: two flags meaning "the corpus" would be two
        # things to keep straight for one idea. For `heldout` it is the directory whose scenes
        # define what the model has seen: a written dataset or a sharded corpus, both are walked.
    )
    parser.add_argument(
        "--source", default=_DEFAULT_SOURCE,
        help="check-dataset: which format --corpus is. NAMED rather than sniffed: a corpus whose "
             "format was guessed had its units and its frame guessed with it",
    )
    parser.add_argument(
        "--spec", default=_DEFAULT_SPEC,
        help="check-dataset: the feature spec the corpus is judged against "
             "(src/robot/grasping/deep/ranker/features.py)",
    )
    parser.add_argument(
        "--serve",
        help="check-dataset: a second corpus standing in for what the model will actually see. "
             "Enables the train/serve check; the one that catches a feature which is constant in "
             "training and alive at inference",
    )
    parser.add_argument(
        "--serve-source", default="grasp_labels",
        help="check-dataset: which format --serve is",
    )
    parser.add_argument(
        "--serve-spec",
        help="check-dataset: the serving corpus's spec, when it differs. A frame mismatch is REFUSED "
             "rather than compared; the columns line up by name and hold different quantities",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="check-dataset: the verdict and its findings, without the per-column table",
    )
    parser.add_argument(
        "--out-model", dest="model_out", default="assets/models/grasp_ranker/v1",
        help="train-ranker: artifact directory. The trees are gitignored; the card beside them is not",
    )
    parser.add_argument(
        # No default. A default naming the ranker's own output file is actively wrong for
        # `build-cloud-corpus`, which writes a directory of per-scene clouds and would try to mkdir
        # over that file. The fallback lives in the ranker's own handler, where it belongs and where
        # it cannot reach anyone else.
        "--corpus-out", default=None,
        help="build-ranker-corpus: the feature table FILE (default logs/dl/corpus/ranker_v1.npz). "
             "build-cloud-corpus: the DIRECTORY the per-scene clouds are written to (required)",
    )
    parser.add_argument(
        "--physics", default=None,
        help="build-ranker-corpus: a grasp_physics*.jsonl to join `held` from. Without it the corpus "
             "carries only `valid`, our own verdict; with it, stage two's target",
    )
    # These four default to `UNSET`, and that is not a style choice. Declaring the numbers here as
    # well as in `fit_ranker` puts them in exactly the state in which a drift is invisible: both
    # sides are valid values of the right type, so nothing raises when they part, and the wrong
    # number looks as plausible as the right one. The trainer owns them; a flag left alone forwards
    # nothing. Help text reads the trainer's real signature so it cannot drift either.
    parser.add_argument("--folds", type=int, default=UNSET,
                        help=f"train-ranker: GroupKFold splits, grouped by object; never by row "
                             f"{_fit_default('folds')}")
    parser.add_argument("--trees", type=int, default=UNSET, dest="trees",
                        help=f"train-ranker: boosting stages {_fit_default('n_estimators')}")
    parser.add_argument("--learning-rate", type=float, default=UNSET,
                        help=f"train-ranker: boosting learning rate {_fit_default('learning_rate')}")
    # `--tree-depth`, and its dest is `tree_depth` rather than `depth`. `camera-probe` already owns
    # the flag `--depth` for its depth source, which argparse refuses outright; `dest="depth"`
    # collides silently on the same attribute instead, and hands the trainer a depth-source string
    # as a tree depth, leaving sklearn to catch it. The loud refusal is the useful one.
    parser.add_argument("--tree-depth", type=int, default=UNSET,
                        help=f"train-ranker: max tree depth, 400 trees x depth 3 is what was measured "
                             f"{_fit_default('max_depth')}")
    parser.add_argument(
        "--require-curobo", action="store_true",
        help="verify-robot: fail if the cuRobo world-collision check cannot run (it reports "
             "UNAVAILABLE and does not fail the robot by default)",
    )
    return parser


def _cmd_init_config(*, recipe: str | None, out: str | None) -> int:
    """Write a starter config for a named recipe, and print the commands that follow it.

    A command and not a doc snippet, because a document and a program drift apart: `prepare-assets`
    prints the instruction to point `assets.jaw_screen_path` at what it wrote, and a document that
    omits that step wins over the program that prints it. A config that arrives with the key already
    present cannot be forgotten.

    Minimal on purpose. It writes the recipe's overrides and the two paths only the customer
    knows, not a dump of every default. A file repeating two hundred defaults hides the handful of
    lines that were chosen.
    """
    import json as _json
    from pathlib import Path

    from datagen.recipes import CORPUS_RECIPES, config_for, steps

    name = recipe or "v1"
    if name not in CORPUS_RECIPES:
        print(f"unknown corpus recipe {name!r}; known: {', '.join(sorted(CORPUS_RECIPES))}",
              file=sys.stderr)
        return _EXIT_USAGE
    destination = Path(out or "datagen_config.json")
    if destination.exists():
        # Refuses rather than overwrites. This file is where a customer's edits live, and a
        # command that silently replaced them would destroy the one thing here they authored.
        print(f"{destination} already exists; move it aside or pass --out elsewhere. This command "
              f"will not overwrite a config you have edited.", file=sys.stderr)
        return _EXIT_USAGE
    body = config_for(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {destination} for corpus recipe {name}")
    print("")
    print("WHAT IS IN IT, and why it is not the default:")
    print("  families.flat_orientation = random   objects are dumped, not stood up. MEASURED: a")
    print("      tipped object is 3.0x as likely to admit a jaw grasp (17.6 % against 5.9 %), and")
    print("      on rebuilt scenes that is six times the jaw labels per scene, net +57 % graspable")
    print("      objects per scene. The default is `upright` so existing corpora rebuild unchanged.")
    print("  assets.jaw_screen_path               the screen step 1 produces. Without it the bank")
    print("      is filtered by a box proxy on the mesh HULL, and a jaw closes on a LINE through the")
    print("      object: MEASURED, the proxy called 514 meshes graspable where the labeller finds")
    print("      331, and REFUSED 50 that do have grasps.")
    print("")
    print("EDIT THE TWO PATHS, then run these in order:")
    for index, (command, why) in enumerate(steps(name), start=1):
        print(f"  {index}. {command}")
        for line in _wrap(why, 86):
            print(f"       {line}")
    print("")
    print("Then train with the matching half of the same recipe:")
    print(f"  python -m src.robot.grasping.deep train-set --recipe {name} --tier smoke ...")
    print(f"  python -m src.robot.grasping.deep train-set --recipe {name} --tier full  ...")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    """Plain wrapping, so the reasons stay readable in a terminal without a dependency."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Before `_config`, deliberately: this command exists to create a config, so requiring one to
    # run it would be a loop a new customer cannot break out of.
    if args.command == "init-config":
        return _cmd_init_config(recipe=args.recipe, out=args.out)

    try:
        config = _config(args)
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        # A mistyped flag and a crash must not look alike. Raw pydantic output reaches the operator
        # as a stack trace with a documentation link in it, on every subcommand, and a stack trace
        # reads as "the tool is broken" when the tool is fine and the request was wrong.
        print(f"config: {_first_problem(exc)}", file=sys.stderr)
        return _EXIT_USAGE
    # The run header. Every command below writes into its own file (see datagen/constants.py), and
    # nothing in those files says which invocation produced them: which dataset, which fidelity,
    # which seed. One line here makes all of them attributable.
    logger.info("%s: name=%s out=%s scenes=%d seed=%d domain=%s masks=%s jobs=%s config=%s",
                args.command, args.name, args.out or config.output.root, config.scenes,
                config.seed, config.domain, args.masks or "default", args.jobs,
                args.config or "defaults")
    started = time.perf_counter()
    code = _dispatch(args, config)
    # The exit code is the whole verdict of a CLI and it otherwise reaches only the shell that ran it:
    # an overnight `grasp-gate` that returned 1 leaves per-module files that look exactly like a pass.
    logger.info("%s finished in %.1f s: exit %d", args.command, time.perf_counter() - started, code)
    return code


def _dispatch(args: argparse.Namespace, config: DatagenConfig) -> int:
    """The command table. Split out of :func:`main` so the run header and the exit code can be logged
    around it without the outcome having to be repeated at every return statement."""
    if args.command == "plan":
        return _cmd_plan(config)
    if args.command == "describe":
        return _cmd_describe(config, args.index)
    if args.command == "audit":
        return _cmd_audit(config)
    if args.command == "verify":
        return _cmd_verify(config, name=args.name, out_root=args.out)
    if args.command == "preview":
        return _cmd_preview(config, name=args.name, out_root=args.out)
    if args.command == "prompts":
        return _cmd_prompts(config, name=args.name, out_root=args.out)
    if args.command == "build-cloud-corpus":
        return _cmd_build_cloud_corpus(args, config)
    if args.command == "build-ranker-corpus":
        return _cmd_build_ranker_corpus(args, config)
    if args.command == "split-dataset":
        return _cmd_split_dataset(args, config)
    if args.command == "train-ranker":
        return _cmd_train_ranker(args)
    if args.command == "check-dataset":
        return _cmd_check_dataset(args)
    if args.command == "normalise-meshes":
        return _cmd_normalise_meshes(collections=args.collection or [], jobs=args.jobs or 4,
                                     faces=args.faces, scale=args.scale)
    if args.command == "screen-meshes":
        return _cmd_screen_meshes(collections=args.collection or [],
                                  out=args.out or _DEFAULT_SCREEN,
                                  density=_screen_density(args), jobs=args.jobs or 10,
                                  want_graspable=args.want_graspable)
    if args.command == "prepare-assets":
        return _cmd_prepare_assets(collections=args.collection or [],
                                   out=args.out or _DEFAULT_SCREEN,
                                   density=_screen_density(args), jobs=args.jobs or 6,
                                   faces=args.faces, want_graspable=args.want_graspable)
    if args.command == "decompose":
        return _cmd_decompose(config, jobs=_resolve_jobs(args.jobs), scenes=args.scenes)
    if args.command == "label-grasps":
        if args.kinds and args.kinds != "jaw":
            # A silently ignored flag. `label_dataset` takes no `kinds` and always writes both, so
            # `--kinds` decides nothing here; the step where it decides something is
            # `build-cloud-corpus`, whose default is `jaw`.
            #
            # A warning rather than a refusal, deliberately. The documented command passes it here,
            # so refusing would break that command for everyone who copied it, and the flag is
            # harmless: both kinds get labelled either way. What it costs is belief.
            print(f"note: --kinds {args.kinds} does nothing on label-grasps, which always labels "
                  f"BOTH jaw and suction. It decides something on build-cloud-corpus, whose default "
                  f"is `jaw` and would DROP the suction labels this step is about to write.",
                  file=sys.stderr)
        # `default` is this command's own default: it reproduces every label set this repo has
        # written. A screen picks `grid` instead, and the flag itself carries neither.
        return _cmd_label_grasps(config, name=args.name, out_root=args.out,
                                 density=args.density or "default", jaw=args.jaw,
                                 label_budget=args.label_budget)
    if args.command == "predict-masks":
        return _cmd_predict_masks(config, name=args.name, out_root=args.out)
    if args.command == "grasp-gate":
        return _cmd_gate(config, name=args.name, out_root=args.out, write=args.write_baseline)
    if args.command == "eval-grasps":
        return _cmd_eval_grasps(config, name=args.name, out_root=args.out,
                                summary_only=args.summary_only, masks=args.masks or "gt",
                                mask_completion=args.mask_completion,
                                rungs=args.rungs, limit=args.limit,
                                jobs=_resolve_jobs(args.jobs))
    if args.command == "camera-probe":
        return _cmd_camera_probe(args)
    if args.command == "cost":
        return _cmd_cost(args, config)
    if args.command == "heldout":
        return _cmd_heldout(args, config)
    if args.command == "side-approach":
        return _cmd_side_approach(args, config)
    if args.command == "why-no-jaw":
        return _cmd_why_no_jaw(config, collections_=args.collection, limit=args.limit,
                               from_screen=args.from_screen,
                               density=args.density or "default", out=args.out)
    if args.command == "physics-compare":
        return _cmd_physics_compare(config, name=args.name, out_root=args.out,
                                    headless=not args.gui, per_class=args.per_class,
                                    configs=args.configs, physics_engine=args.physics_engine)
    if args.command == "physics-sample":
        return _cmd_physics_sample(config, name=args.name, out_root=args.out,
                                   physics_engine=args.physics_engine, proposals=args.proposals,
                                   headless=not args.gui, per_class=args.per_class)
    if args.command == "verify-robot":
        return _cmd_verify_robot(
            config, headless=not args.gui, require_curobo=args.require_curobo,
        )
    return _cmd_build(config, name=args.name, out_root=args.out, headless=not args.gui,
                      preview=args.preview)


if __name__ == "__main__":
    raise SystemExit(main())
