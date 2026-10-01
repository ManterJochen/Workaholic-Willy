"""RT-DETR fine-tuning from the command line: the shell face of ``DetectorTraining`` (``training/api.py``).

A dataset is a COCO or YOLO folder (``training/datasets.py`` lists the layouts the labelling tools export). Training
keeps the best epoch by validation mAP in ``--output-dir``, which ``RtDetrObjectDetector`` loads with
``model_path=<out>, local=True``; the class vocabulary comes from the dataset, so the classes need not be COCO's.

    python -m src.models.detection.closed_set.train inspect --data-dir data/detect/v1
    python -m src.models.detection.closed_set.train train   --data-dir data/detect/v1 --output-dir assets/models/rtdetr/v1
    python -m src.models.detection.closed_set.train eval    --model-dir assets/models/rtdetr/v1 --data-dir data/detect/v1

``--tier smoke`` proves the chain in minutes; ``--tier full`` (the default) is the model to deploy. A flag you pass
wins over the tier. Exit codes: ``0`` ok, ``2`` bad arguments or no dataset, ``3`` training failed. ``inspect`` runs
without the training stack, and so does a ``train`` that has no dataset to train on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.contracts.options import UNSET, chosen
from src.models.constants import MODELS_LOG_DIR, RTDETR_TRAIN_LOG_FILE
from src.models.detection.closed_set.training.plan import DEFAULT_SEED, DetectorPlanOverrides
from src.models.detection.closed_set.training.recipes import DEFAULT_BASE_MODEL, RECIPES, TIERS
from src.utility.log_cfg import create_logger

_LOG = create_logger("RtDetrTrain", log_file=RTDETR_TRAIN_LOG_FILE, log_dir=MODELS_LOG_DIR)

DEFAULT_OUTPUT_DIR = "assets/models/rtdetr/v1"
DEFAULT_DATA_DIR = "data/detect/v1"


# --------------------------------------------------------------------------------------------------
# The small COCO reader older callers import (``training/datasets.py`` is the one a run uses)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Fine-tuning settings for :func:`train_rtdetr`; ``DetectorPlan`` is the full description of a run."""

    base_model_id: str = DEFAULT_BASE_MODEL
    epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 8
    image_size: int = 640
    seed: int = DEFAULT_SEED
    num_workers: int | None = None

    def to_dict(self) -> dict:
        from dataclasses import asdict  # noqa: PLC0415

        return asdict(self)


@dataclass(frozen=True, slots=True)
class CocoImageRecord:
    """One image + its annotations, in the shape the HF image processor's ``annotations`` arg wants."""

    image_id: int
    file_name: str
    coco_annotations: list[dict]


@dataclass(frozen=True, slots=True)
class CocoIndex:
    """A parsed COCO ``annotations.json``: category map + per-image records."""

    categories: dict[int, str]
    records: list[CocoImageRecord]
    source_path: str
    num_annotations: int = field(default=0)

    def image_records(self) -> list[CocoImageRecord]:
        return self.records


def load_coco_index(annotations_path: str | Path) -> CocoIndex:
    """Parse a COCO detection ``annotations.json``; ``FileNotFoundError`` when missing, ``ValueError`` when malformed."""
    p = Path(annotations_path)
    if not p.is_file():
        raise FileNotFoundError(f"COCO annotations.json not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in doc:
            raise ValueError(f"COCO file {p} missing required key {key!r}")
    categories = {int(c["id"]): str(c["name"]) for c in doc["categories"]}
    if not categories:
        raise ValueError(f"COCO file {p} declares no categories")
    categories = dict(sorted(categories.items()))
    by_image: dict[int, list[dict]] = {int(im["id"]): [] for im in doc["images"]}
    for ann in doc["annotations"]:
        iid = int(ann["image_id"])
        if iid not in by_image:  # an annotation for an image the file doesn't declare -> skip, don't crash
            continue
        bbox = [float(v) for v in ann["bbox"]]
        by_image[iid].append({
            "bbox": bbox,
            "category_id": int(ann["category_id"]),
            "area": float(ann.get("area", bbox[2] * bbox[3])),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "id": int(ann.get("id", 0)),
        })
    records = [
        CocoImageRecord(image_id=int(im["id"]), file_name=str(im["file_name"]),
                        coco_annotations=by_image.get(int(im["id"]), []))
        for im in doc["images"]
    ]
    return CocoIndex(categories=categories, records=records, source_path=str(p),
                     num_annotations=len(doc["annotations"]))


def build_id2label(index: CocoIndex) -> dict[int, str]:
    """The model ``id2label``: the dataset's categories over a contiguous ``0..N-1`` label space, by sorted id."""
    return {i: name for i, (_cid, name) in enumerate(index.categories.items())}


def _category_remap(index: CocoIndex) -> dict[int, int]:
    """COCO ``category_id`` -> contiguous model label id (0..N-1), by sorted category id."""
    return {cid: i for i, (cid, _name) in enumerate(index.categories.items())}


# --------------------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------------------
def train_rtdetr(*, data_dir: str, output_dir: str, config: TrainConfig) -> dict:
    """Train at the ``full`` tier with ``config``'s settings and return the written manifest.

    The function the first command line exposed, kept for its callers; it runs through ``DetectorTraining``.
    """
    from src.models.detection.closed_set.training.api import DetectorTraining  # noqa: PLC0415

    overrides = DetectorPlanOverrides(base_model=config.base_model_id, epochs=config.epochs,
                                      learning_rate=config.learning_rate, batch=config.batch_size,
                                      image_size=config.image_size, seed=config.seed, workers=config.num_workers)
    run = DetectorTraining.from_dataset(dataset=data_dir, overrides=overrides, out_dir=output_dir)
    report = run.train()
    run.write_report(report)
    return json.loads(Path(report.raw["manifest"]).read_text(encoding="utf-8"))


def evaluate_rtdetr(*, model_dir: str, data_dir: str, batch_size: int = 8, format: str = "auto",
                    val_fraction: float = 0.15) -> dict:
    """mAP of a trained checkpoint on the dataset's validation images, split exactly as training split them."""
    from src.models.detection.closed_set.training.datasets import load_dataset  # noqa: PLC0415
    from src.models.detection.closed_set.training.trainer import evaluate_checkpoint  # noqa: PLC0415

    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"no model folder at {model_dir}")
    dataset = load_dataset(data_dir, format=format, val_fraction=val_fraction)
    scores = evaluate_checkpoint(model_dir, dataset, batch=batch_size)
    result = scores.as_dict()
    result["split"] = "val" if dataset.val else "train (no validation images: not a held-out score)"
    _LOG.info("eval: mAP %.4f (50: %.4f) over %d image(s) of %s", scores.map, scores.map50, scores.images, data_dir)
    return result


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _overrides(args: argparse.Namespace) -> DetectorPlanOverrides:
    values = {
        "base_model": args.base_model, "epochs": args.epochs, "batch": args.batch_size, "learning_rate": args.lr,
        "image_size": args.image_size, "seed": args.seed, "patience": args.patience, "workers": args.workers,
        "val_fraction": args.val_fraction, "augment": UNSET if args.augment is None else args.augment,
        "multiscale": UNSET if args.multiscale is None else args.multiscale,
    }
    return DetectorPlanOverrides(**{key: value for key, value in values.items() if chosen(value)})


def _guarded_workers(run: Any) -> int | None:
    """Loader workers for the command line on Windows, where the library chooses none (this module is guarded)."""
    if sys.platform != "win32" or run.plan.workers is not None:
        return run.plan.workers
    import torch  # noqa: PLC0415

    big = len(run.dataset().train) >= 200
    return max(0, min(4, (os.cpu_count() or 2) // 2)) if big and torch.cuda.is_available() else 0


def _cmd_train(args: argparse.Namespace) -> int:
    from src.models.detection.closed_set.training.api import DetectorTraining  # noqa: PLC0415

    try:
        run = DetectorTraining.from_dataset(dataset=args.data_dir, recipe=args.recipe, tier=args.tier,
                                            overrides=_overrides(args), out_dir=args.output_dir, format=args.format,
                                            device=args.device, resume=args.resume)
        run.dataset()
    except FileNotFoundError as exc:
        _LOG.error("train refused: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        _LOG.error("train refused: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    import dataclasses  # noqa: PLC0415

    run.plan = dataclasses.replace(run.plan, workers=_guarded_workers(run))
    print(run.describe())
    print(run.probe())
    try:
        report = run.train()
    except Exception as exc:  # noqa: BLE001 (a training failure is exit 3 with the reason on stderr)
        _LOG.exception("training failed: %s", exc)
        print(f"training failed: {exc}", file=sys.stderr)
        return 3
    print(report)
    print(f"  report     {run.write_report(report)}")
    return 0 if report.succeeded else 3


def _cmd_eval(args: argparse.Namespace) -> int:
    try:
        result = evaluate_rtdetr(model_dir=args.model_dir, data_dir=args.data_dir, batch_size=args.batch_size,
                                 format=args.format, val_fraction=args.val_fraction)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.error("eval refused: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """What a dataset holds and what was left out of it; no training stack needed."""
    from src.models.detection.closed_set.training.api import DatasetProbe  # noqa: PLC0415
    from src.models.detection.closed_set.training.datasets import load_dataset  # noqa: PLC0415

    try:
        probe = DatasetProbe.of(load_dataset(args.data_dir, format=args.format, val_fraction=args.val_fraction,
                                             require_boxes=False))
    except (FileNotFoundError, ValueError) as exc:
        _LOG.error("inspect refused: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(probe.as_dict(), sort_keys=True, indent=2) if args.json else probe.render())
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.detection.closed_set.train",
        description="Train, evaluate or inspect the closed-set RT-DETR detector on a COCO or YOLO dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def dataset_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="a COCO or YOLO dataset folder")
        p.add_argument("--format", default="auto", choices=("auto", "coco", "yolo"))
        p.add_argument("--val-fraction", type=float, default=UNSET,
                       help="the validation share cut from a dataset without its own validation split (0.15)")

    tr = sub.add_parser("train", help="Train; keep the best epoch, the last one, results.csv, manifest and report.")
    dataset_args(tr)
    tr.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="where the model and its files go")
    tr.add_argument("--recipe", default="v1", choices=sorted(RECIPES))
    tr.add_argument("--tier", default="full", choices=sorted(TIERS))
    tr.add_argument("--base-model", default=UNSET, help=f"a Hub id or a local checkpoint folder ({DEFAULT_BASE_MODEL})")
    tr.add_argument("--epochs", type=int, default=UNSET)
    tr.add_argument("--batch-size", type=int, default=UNSET)
    tr.add_argument("--lr", type=float, default=UNSET)
    tr.add_argument("--image-size", type=int, default=UNSET)
    tr.add_argument("--seed", type=int, default=UNSET)
    tr.add_argument("--patience", type=int, default=UNSET, help="epochs without a better val mAP before stopping; 0 never")
    tr.add_argument("--workers", type=int, default=UNSET, help="data-loader processes")
    tr.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    tr.add_argument("--no-multiscale", dest="multiscale", action="store_false", default=None)
    tr.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda when there is one)")
    tr.add_argument("--resume", action="store_true", help="continue the run in --output-dir from its last epoch")
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="mAP of a trained model on the dataset's validation images.")
    dataset_args(ev)
    ev.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.set_defaults(func=_cmd_eval)

    ins = sub.add_parser("inspect", help="What a dataset holds, per class and split (no training stack).")
    dataset_args(ins)
    ins.add_argument("--split", default="train", help=argparse.SUPPRESS)  # kept for old invocations; both splits print
    ins.add_argument("--json", action="store_true", help="print JSON instead of the table")
    ins.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if hasattr(args, "val_fraction") and not chosen(args.val_fraction):
        args.val_fraction = 0.15
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
