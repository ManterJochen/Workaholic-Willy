"""The detector training loop: RT-DETR fine-tuned the way its authors and the large toolkits train it.

A run, in order:

- AdamW at the plan's learning rate, the backbone at ``backbone_lr_scale`` of it, weight decay off norms and biases;
- linear warm-up (three epochs, 100 to 2,000 optimizer steps, at most a fifth of the run), then cosine decay to 1 %
  (Ultralytics and the transformers fine-tuning guide; RT-DETR's own single step assumes 72 COCO epochs);
- mixed precision: bf16 where the GPU has it, else fp16 with a gradient scaler, fp32 on a CPU;
- gradients clipped at ``clip_grad_norm``;
- an exponential moving average of the weights: every evaluation and every exported model is the average;
- multi-scale batches and the strong augmentations until the last tenth of the epochs;
- after every epoch, COCO mAP@0.5:0.95 on the validation split. A better epoch is exported at once, the last epoch is
  kept for resuming, and the run stops after ``patience`` epochs without a better mAP.

Every image is read at most ``loading.working_side`` px on its long side, its boxes scaled with it. On a GPU loader
processes read them beside the step, on Windows too, where they start without running the training script again
(``loading.WorkerBatches``); where there are none, threads do (``loading.ThreadedBatches``). A person watching the terminal sees a bar over each epoch's batches and over the
validation, and a line per epoch with the best so far and the time left; ``curves.png`` is drawn again after every
epoch.

What lands in ``out_dir``:

    config.json, model.safetensors, preprocessor_config.json   the best epoch: what RtDetrObjectDetector and a cell load
    last/                                                       the last epoch, the same three files
    checkpoint_last.pt                                          everything ``resume`` needs
    results.csv                                                 one row per epoch
    curves.png                                                  the losses, the mAPs, the learning rate, AP per class
    manifest.json                                               dataset fingerprint, classes, plan, metrics, versions

The dataset class and the collate function live at module level, so data-loader workers can be spawned on Windows.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.constants import MODELS_LOG_DIR, RTDETR_TRAIN_LOG_FILE
from src.models.detection.closed_set.training.augment import augment_sample
from src.models.detection.closed_set.training.datasets import DetectionDataset, LabelledImage
from src.models.detection.closed_set.training.loading import (
    ThreadedBatches,
    WorkerBatches,
    decode_threads,
    read_image,
    working_side,
)
from src.models.detection.closed_set.training.metrics import DetectionScores, evaluate_detections
from src.models.detection.closed_set.training.plan import DetectorPlan
from src.utility.log_cfg import create_logger
from src.utility.progress import progress_bar

__all__ = ["MANIFEST_SCHEMA", "ModelEMA", "evaluate_checkpoint", "train_detector"]

_LOG = create_logger("RtDetrTrain", log_file=RTDETR_TRAIN_LOG_FILE, log_dir=MODELS_LOG_DIR)

MANIFEST_SCHEMA = "willy.rtdetr.train_manifest/2"
CHECKPOINT = "checkpoint_last.pt"
RESULTS = "results.csv"
CURVES = "curves.png"
#: Detections kept per image for the mAP, and the score below which a query is not a detection at all.
_EVAL_MAX_DETS = 100
_EVAL_MIN_SCORE = 0.001
#: Non-finite losses in a row after which a run is called diverged rather than unlucky.
_MAX_BAD_STEPS = 10

ModelFactory = Callable[[DetectorPlan, Mapping[int, str]], tuple[Any, Any]]
#: What a training or validation loop iterates: batches, ``len()`` of them per epoch.
Batches = DataLoader | ThreadedBatches | WorkerBatches


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
class DetectionSamples(Dataset):
    """Images and boxes turned into what RT-DETR trains on: pixel values and normalised ``cxcywh`` labels.

    Each image is read at most ``max_side`` px on its long side (``loading.read_image``), its boxes scaled with it.
    ``decode`` reads and ``prepare`` augments and encodes, so the reading can run in threads apart from everything that
    draws a random number (``loading.ThreadedBatches``); ``__getitem__`` is the two, one after the other.
    """

    def __init__(self, images: Sequence[LabelledImage], processor: Any, *, augment: bool,
                 max_side: int = working_side(640)) -> None:
        self.images = tuple(images)
        self.processor = processor
        self.augment = augment
        self.strong = augment
        self.max_side = int(max_side)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.prepare(index, self.decode(index))

    def decode(self, index: int) -> tuple[Any, tuple[int, int]]:
        """The image of sample ``index``, read shrunk, and the size of its file."""
        return read_image(self.images[index].path, max_side=self.max_side)

    def prepare(self, index: int, decoded: tuple[Any, tuple[int, int]]) -> dict[str, Any]:
        """Sample ``index`` from its decoded image: the boxes scaled to it, augmented, encoded."""
        image, (file_width, file_height) = decoded
        entry = self.images[index]
        sx, sy = image.width / float(file_width), image.height / float(file_height)
        if sx == 1.0 and sy == 1.0:
            boxes = [list(box.xyxy) for box in entry.boxes]
        else:
            boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in (box.xyxy for box in entry.boxes)]
        labels = [box.label for box in entry.boxes]
        if self.augment:
            image, boxes, labels = augment_sample(image, boxes, labels, strong=self.strong)
        annotations = [{"bbox": [x1, y1, x2 - x1, y2 - y1], "category_id": int(label), "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0} for (x1, y1, x2, y2), label in zip(boxes, labels)]
        encoded = self.processor(images=image, annotations={"image_id": index, "annotations": annotations},
                                 return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"][0], "labels": encoded["labels"][0], "index": index}


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": [item["labels"] for item in batch],
            "index": [int(item["index"]) for item in batch]}


def _seed_worker(_worker: int) -> None:
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _loader(samples: DetectionSamples, plan: DetectorPlan, *, shuffle: bool, workers: int, seed: int,
            device: torch.device | None = None) -> Batches:
    """A loader whose workers live as long as it does: one per phase (strong, then plain) and one for validation.

    Workers copy the dataset when they start, so the strong/plain switch takes a new loader; keeping each one alive
    otherwise saves the seconds a spawned worker spends importing torch, every epoch. They start with ``__main__``
    hidden (``WorkerBatches``), so a training script without a ``__main__`` guard is not run again in each of them.
    On a GPU without worker processes the images are read in threads (``ThreadedBatches``): the same batches in the
    same order. On a CPU the training step has no time to give, and the loader reads in the step's own thread.
    """
    if _threaded(workers, device):
        return ThreadedBatches(samples, batch=plan.batch, shuffle=shuffle, seed=seed, threads=decode_threads(),
                               collate=collate, pin=device is not None and device.type == "cuda")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(samples, batch_size=plan.batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                        worker_init_fn=_seed_worker, generator=generator, pin_memory=torch.cuda.is_available(),
                        persistent_workers=workers > 0)
    return WorkerBatches(loader) if workers > 0 else loader


def _threaded(workers: int, device: torch.device | None) -> bool:
    """Whether the images are read in threads: on a GPU, where no loader process reads them."""
    return workers == 0 and device is not None and device.type in ("cuda", "mps")


def _workers(plan: DetectorPlan, images: int, device: torch.device) -> int:
    """Loader processes: up to four on a GPU, on Windows as anywhere, since they start without running the training
    script again (``WorkerBatches``). Four kept the owner's RTX 5080 busy (28.9 images/s against 20.7 read by threads,
    2026-10-10); none on a CPU, whose training step has no time to give, or for under 200 images, too few to pay for
    their start."""
    if plan.workers is not None:
        return plan.workers
    if device.type != "cuda" or images < 200:
        return 0
    return max(0, min(4, (os.cpu_count() or 2) // 2))


# --------------------------------------------------------------------------------------------------
# Model, optimizer, schedule, average
# --------------------------------------------------------------------------------------------------
def resolve_base_model(base_model: str) -> tuple[str, bool]:
    """Where the base weights come from: a local folder when one exists, the Hub cache or a download otherwise.

    The default Hub id resolves to the repository's fetched copy (``scripts/model_weights/fetch.py rtdetr``) when it
    is there, so a cell trains offline on the same weights its detector config names.
    """
    if Path(base_model).is_dir():
        return base_model, True
    from src.config.paths import repository_root  # noqa: PLC0415

    fetched = repository_root() / "assets" / "models" / "hf" / "detection" / base_model.replace("/", "--")
    if (fetched / "config.json").is_file():
        return str(fetched), True
    return base_model, False


def pretrained_factory(plan: DetectorPlan, id2label: Mapping[int, str]) -> tuple[Any, Any]:
    """The base RT-DETR with its classification head rebuilt for these classes, and its image processor."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: PLC0415

    source, local = resolve_base_model(plan.base_model)
    kwargs = {"local_files_only": True} if local else {}
    try:
        with _QuietHub():
            processor = AutoImageProcessor.from_pretrained(
                source, do_resize=True, size={"height": plan.image_size, "width": plan.image_size}, **kwargs)
            # The classes come as id2label alone: num_labels beside it is checked against the base's 80 COCO classes
            # before id2label replaces them, and warns.
            model = AutoModelForObjectDetection.from_pretrained(
                source, id2label=dict(id2label), label2id={v: k for k, v in id2label.items()},
                ignore_mismatched_sizes=True, **kwargs)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"the base model {plan.base_model!r} could not be loaded ({exc}). Fetch it once with "
            f"`python scripts/model_weights/fetch.py rtdetr`, or pass base_model=<a local checkpoint folder>") from exc
    _LOG.info("base model %s: its COCO class head replaced by a new one for the %d class(es) of the dataset, the rest "
              "of the network keeps its trained weights", plan.base_model, len(id2label))
    return model, processor


class _QuietHub:
    """Transformers and the Hub at errors only while the base model loads: their load report lists the class head
    rebuilt for the dataset's classes as a weight MISMATCH, which is the rebuild meant, and an unauthenticated download
    warns on every run. A download's own progress bar stays."""

    def __enter__(self) -> "_QuietHub":
        from huggingface_hub.utils import logging as hub_logging  # noqa: PLC0415
        from transformers.utils import logging as hf_logging  # noqa: PLC0415

        self._levels = [(hf_logging, hf_logging.get_verbosity()), (hub_logging, hub_logging.get_verbosity())]
        for module, _level in self._levels:
            module.set_verbosity_error()
        return self

    def __exit__(self, *_exc: Any) -> None:
        for module, level in self._levels:
            module.set_verbosity(level)


def param_groups(model: torch.nn.Module, plan: DetectorPlan) -> list[dict[str, Any]]:
    """AdamW groups: the backbone at ``backbone_lr_scale``, and no weight decay on norms, biases and 1-D parameters."""
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = ".backbone." in f".{name}."
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        groups.setdefault((backbone, no_decay), []).append(parameter)
    out = []
    for (backbone, no_decay), params in sorted(groups.items()):
        out.append({"params": params,
                    "lr": plan.learning_rate * (plan.backbone_lr_scale if backbone else 1.0),
                    "weight_decay": 0.0 if no_decay else plan.weight_decay,
                    "name": f"{'backbone' if backbone else 'head'}{'/no_decay' if no_decay else ''}"})
    return out


def warmup_steps(steps_per_epoch: int, total_steps: int) -> int:
    return max(1, min(min(2000, max(100, 3 * steps_per_epoch)), max(1, total_steps // 5)))


def lr_factor(step: int, warmup: int, total: int, final: float = 0.01) -> float:
    """Linear warm-up to 1, then cosine decay to ``final``; ``step`` counts optimizer steps from 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ModelEMA:
    """An exponential moving average of a model's weights with a warm-up, as RT-DETR and YOLO keep it.

    The decay ramps as ``decay * (1 - exp(-updates / warmup))``: early on the average follows the model closely, later
    it moves slowly. Only floating-point tensors are averaged; integer buffers are copied.
    """

    def __init__(self, model: torch.nn.Module, *, decay: float, warmup: int, updates: int = 0) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = updates

    def factor(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / self.warmup))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.factor()
        current = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(current[key].detach().to(value.dtype), alpha=1.0 - d)
            else:
                value.copy_(current[key])


def _amp_dtype(plan: DetectorPlan, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or plan.amp == "off":
        return None
    if plan.amp in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(labels: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in label.items()} for label in labels]


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(model: torch.nn.Module, processor: Any, loader: Batches,
              images: Sequence[LabelledImage], classes: Sequence[str], device: torch.device, dtype: torch.dtype | None,
              *, desc: str = "validation") -> tuple[DetectionScores, float]:
    """mAP and the mean loss on the validation images, with the model as it stands (in practice the average). Each
    image's detections come back in its own pixels, whatever size it was read at."""
    model.eval()
    predictions: dict[int, list[tuple[int, float, tuple[float, float, float, float]]]] = {}
    losses: list[float] = []
    with progress_bar(len(loader), desc=desc, unit="batch") as bar:
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                outputs = model(pixel_values=pixels, labels=_to_device(batch["labels"], device))
            if outputs.loss is not None and torch.isfinite(outputs.loss):
                losses.append(float(outputs.loss))
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()
            sizes = [(images[i].height, images[i].width) for i in batch["index"]]
            results = processor.post_process_object_detection(outputs, threshold=_EVAL_MIN_SCORE, target_sizes=sizes)
            for i, result in zip(batch["index"], results):
                order = torch.argsort(result["scores"], descending=True)[:_EVAL_MAX_DETS * max(1, len(classes))]
                predictions[i] = [(int(result["labels"][j]), float(result["scores"][j]),
                                   tuple(float(v) for v in result["boxes"][j].tolist()))  # type: ignore[misc]
                                  for j in order.tolist()]
            bar.update(1)
    ground_truth = [[(box.label, box.xyxy) for box in image.boxes] for image in images]
    scores = evaluate_detections(ground_truth, [predictions.get(i, []) for i in range(len(images))], classes,
                                 max_dets=_EVAL_MAX_DETS)
    return scores, (float(np.mean(losses)) if losses else float("nan"))


# --------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------
def train_detector(dataset: DetectionDataset, plan: DetectorPlan, *, out_dir: str | Path, device: str | None = None,
                   resume: bool = False, on_epoch: Callable[[Mapping[str, Any]], None] | None = None,
                   model_factory: ModelFactory | None = None) -> dict[str, Any]:
    """Train, export the best epoch to ``out_dir`` and the last to ``out_dir/last``, and return what happened.

    Raises ``ValueError`` for a run that cannot start: a dataset without training images, a ``resume`` against another
    dataset or other classes, a base model that cannot be loaded. ``model_factory`` replaces the pretrained base
    (tests hand in a tiny RT-DETR).
    """
    plan = plan.validated()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_images = dataset.train[: plan.max_train_images] if plan.max_train_images else dataset.train
    if not train_images:
        raise ValueError("the dataset has no training image")
    _seed_everything(plan.seed)
    dev = _device(device)
    dtype = _amp_dtype(plan, dev)
    model, processor = (model_factory or pretrained_factory)(plan, dataset.id2label)
    model.to(dev)
    _LOG.info("training RT-DETR from %s on %s: %d train / %d val image(s), %d class(es), %d epoch(s), batch %d, "
              "amp %s, device %s", plan.base_model, dataset.root, len(train_images), len(dataset.val),
              len(dataset.classes), plan.epochs, plan.batch, dtype or "off", dev)

    side = working_side(plan.image_size)
    train_samples = DetectionSamples(train_images, processor, augment=plan.augment, max_side=side)
    val_samples = DetectionSamples(dataset.val, processor, augment=False, max_side=side)
    workers = _workers(plan, len(train_images), dev)
    steps_per_epoch = max(1, math.ceil(math.ceil(len(train_images) / plan.batch) / plan.accumulate))
    total_steps = steps_per_epoch * plan.epochs
    warmup = warmup_steps(steps_per_epoch, total_steps)
    optimizer = torch.optim.AdamW(param_groups(model, plan), lr=plan.learning_rate, weight_decay=plan.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)
    ema = ModelEMA(model, decay=plan.ema_decay, warmup=plan.ema_warmup) if plan.ema else None
    scale_rng = random.Random(plan.seed)

    rows: list[dict[str, Any]] = []
    best_fitness, best_epoch, best_scores = -math.inf, 0, None
    start_epoch, resumed_from = 0, 0
    checkpoint_path = out / CHECKPOINT
    if resume:
        if not checkpoint_path.is_file():
            raise ValueError(f"resume asked for, and there is no {CHECKPOINT} in {out}")
        state = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        if list(state["classes"]) != list(dataset.classes) or state["fingerprint"] != dataset.fingerprint:
            raise ValueError(f"{checkpoint_path} was trained on another dataset or other classes "
                             f"({state['classes']}); point out_dir somewhere new to start over")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
            ema.updates = int(state["ema_updates"])
        rows = list(state["rows"])
        best_fitness, best_epoch = float(state["best_fitness"]), int(state["best_epoch"])
        best_scores = state.get("best_scores")
        start_epoch = resumed_from = int(state["epoch"])
        scale_rng.setstate(state["scale_rng"])
        if start_epoch >= plan.epochs:
            _LOG.info("resume: %s already ran all %d epoch(s)", checkpoint_path, plan.epochs)
            return {"nothing_to_resume": True, "epochs": rows, "best_epoch": best_epoch, "best": best_scores,
                    "artifact": _artifact(out, written=(out / "model.safetensors").is_file()),
                    "classes": list(dataset.classes), "resumed_from_epoch": resumed_from}

    stop_augment = plan.stop_augment_epoch()
    bad_steps = 0
    diverged = stopped_early = False
    images_seen = 0
    train_seconds = 0.0
    loader: Batches | None = None
    scales = plan.multiscale_sizes()
    val_loader = (_loader(val_samples, plan, shuffle=False, workers=workers, seed=plan.seed, device=dev)
                  if dataset.val else None)
    _LOG.info("images read at most %d px on their long side%s", side,
              f", in {decode_threads()} thread(s) ahead of each step" if _threaded(workers, dev)
              else (f", by {workers} loader process(es)" if workers else ""))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for epoch in range(start_epoch, plan.epochs):
        epoch_started = time.perf_counter()
        strong = epoch < stop_augment
        if loader is None or train_samples.strong != (plan.augment and strong):
            train_samples.strong = plan.augment and strong
            loader = _loader(train_samples, plan, shuffle=True, workers=workers, seed=plan.seed + 1000 * (epoch + 1),
                             device=dev)
        model.train()
        loss_sum, loss_count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        batches = len(loader)
        with progress_bar(batches, desc=f"epoch {epoch + 1}/{plan.epochs}", unit="batch") as bar:
            for step, batch in enumerate(loader):
                pixels = batch["pixel_values"].to(dev, non_blocking=True)
                if plan.multiscale and strong:
                    size = scale_rng.choice(scales)
                    if size != pixels.shape[-1]:
                        pixels = torch.nn.functional.interpolate(pixels, size=(size, size), mode="bilinear",
                                                                 align_corners=False)
                with torch.autocast(dev.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                    outputs = model(pixel_values=pixels, labels=_to_device(batch["labels"], dev))
                loss = outputs.loss
                if not torch.isfinite(loss):
                    bad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if bad_steps > _MAX_BAD_STEPS:
                        diverged = True
                        break
                    bar.update(1)
                    continue
                bad_steps = 0
                scaler.scale(loss / plan.accumulate).backward()
                loss_sum += float(loss.detach())
                loss_count += 1
                images_seen += pixels.shape[0]
                bar.update(1)
                if step % 10 == 0 or step + 1 == batches:
                    speed = (step + 1) * plan.batch / max(1e-9, time.perf_counter() - epoch_started)
                    bar.set_postfix_str(f"loss {loss_sum / loss_count:.3f}  {speed:.1f} img/s")
                if (step + 1) % plan.accumulate == 0 or step + 1 == batches:
                    scaler.unscale_(optimizer)
                    if plan.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), plan.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
        train_seconds += time.perf_counter() - epoch_started
        if diverged:
            _LOG.error("epoch %d: the loss stayed non-finite for %d step(s) in a row; the run is stopped",
                       epoch + 1, bad_steps)
            break
        judged = ema.module if ema is not None else model
        train_loss = loss_sum / max(1, loss_count)
        if val_loader is not None:
            scores, val_loss = _evaluate(judged, processor, val_loader, dataset.val, dataset.classes, dev, dtype,
                                         desc=f"validation {epoch + 1}/{plan.epochs}")
        else:
            scores, val_loss = None, float("nan")
        # The validation mAP decides; without one (no validation image, or no box in them) the training loss does.
        fitness = scores.map if scores is not None and not math.isnan(scores.map) else -train_loss
        improved = fitness > best_fitness
        if improved:
            best_fitness, best_epoch = fitness, epoch + 1
            best_scores = scores.as_dict() if scores is not None else None
            _export(judged, processor, out)
        row = {
            "epoch": epoch + 1,
            "seconds": round(time.perf_counter() - epoch_started, 3),
            "lr": next((g["lr"] for g in optimizer.param_groups if g.get("name") == "head"),
                       optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "map": scores.map if scores else float("nan"),
            "map50": scores.map50 if scores else float("nan"),
            "map75": scores.map75 if scores else float("nan"),
            "mar100": scores.mar100 if scores else float("nan"),
            "per_class_ap": dict(scores.per_class_ap) if scores else {},
            "strong_augment": bool(train_samples.strong),
            "best": improved,
        }
        rows.append(row)
        _append_results(out / RESULTS, row, dataset.classes, header=epoch == 0 or not (out / RESULTS).is_file())
        _save_checkpoint(checkpoint_path, epoch + 1, model, ema, optimizer, scheduler, scaler, best_fitness,
                         best_epoch, best_scores, rows, dataset, plan, scale_rng)
        _refresh_curves(out, rows, dataset.classes, best_epoch)
        session = rows[resumed_from:]
        left = plan.epochs - (epoch + 1)
        eta = sum(float(r["seconds"]) for r in session) / max(1, len(session)) * left
        best_map = best_scores.get("map") if isinstance(best_scores, dict) else None
        _LOG.info("epoch %d/%d: train loss %.4f, val loss %.4f, mAP %.4f (50: %.4f)%s, %.1f s; best mAP %s (epoch %d)%s",
                  epoch + 1, plan.epochs, row["train_loss"], val_loss, row["map"], row["map50"],
                  ", a new best" if improved else "", row["seconds"],
                  f"{best_map:.4f}" if isinstance(best_map, float) and not math.isnan(best_map) else "-", best_epoch,
                  f"; at most {_duration(eta)} left" if left else "")
        if on_epoch is not None:
            on_epoch(dict(row))
        if plan.patience and dataset.val and epoch + 1 - best_epoch >= plan.patience:
            stopped_early = epoch + 1 < plan.epochs
            if stopped_early:
                _LOG.info("no better validation mAP for %d epoch(s) after epoch %d: stopped", plan.patience,
                          best_epoch)
                break

    final = ema.module if ema is not None else model
    if rows:
        _export(final, processor, out / "last")
    seconds = time.perf_counter() - started
    raw: dict[str, Any] = {
        "epochs": rows,
        "best_epoch": best_epoch if rows else None,
        "best": best_scores,
        "stopped_early": stopped_early,
        "diverged": diverged,
        "nothing_to_resume": False,
        "resumed_from_epoch": resumed_from,
        "classes": list(dataset.classes),
        "device": _device_name(dev),
        "amp": str(dtype).replace("torch.", "") if dtype is not None else "off",
        "seconds": round(seconds, 1),
        "train_images_per_second": round(images_seen / train_seconds, 2) if train_seconds else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2) if dev.type == "cuda" else None,
        "dataset": _dataset_summary(dataset, len(train_images)),
        # Written means this run (or the run it resumed) exported a best epoch; a diverged run keeps the best before it.
        "artifact": _artifact(out, written=best_epoch > 0 and (out / "model.safetensors").is_file()),
    }
    if diverged:
        raw["artifact"]["reason"] = (f"the loss went non-finite at epoch {len(rows) + 1} and stayed there"
                                     + (f"; the best epoch before it ({best_epoch}) is kept" if best_epoch else ""))
    raw["manifest"] = _write_manifest(out, plan, dataset, raw)
    return raw


def evaluate_checkpoint(model_dir: str | Path, dataset: DetectionDataset, *, batch: int = 8,
                        device: str | None = None) -> DetectionScores:
    """mAP of a trained checkpoint folder on the dataset's validation split (its training split if it has none)."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: PLC0415

    dev = _device(device)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir), local_files_only=True).to(dev)
    images = dataset.val or dataset.train
    labels = {v: k for k, v in model.config.id2label.items()}
    if any(name not in labels for name in dataset.classes):
        raise ValueError(f"the checkpoint knows {sorted(labels)} and the dataset names {list(dataset.classes)}")
    plan = DetectorPlan(batch=batch)
    samples = DetectionSamples(images, processor, augment=False)
    scores, _ = _evaluate(model, processor, _loader(samples, plan, shuffle=False, workers=0, seed=plan.seed), images,
                          dataset.classes, dev, _amp_dtype(plan, dev))
    return scores


# --------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------
def _duration(seconds: float) -> str:
    """``seconds`` as an operator reads a wait: ``42 s``, ``7 min``, ``3 h 05 min``."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    hours, minutes = divmod(int(round(seconds / 60)), 60)
    return f"{hours} h {minutes:02d} min"


def _refresh_curves(out: Path, rows: Sequence[Mapping[str, Any]], classes: Sequence[str], best_epoch: int) -> None:
    """Draw ``curves.png`` again from the epochs so far, so a run can be watched rather than waited on.

    Never raises: this runs inside the training loop, and a missing plotting backend or a file a viewer holds open on
    Windows must not cost a run its hours for a picture.
    """
    try:
        from src.models.detection.closed_set.training.report import write_curves  # noqa: PLC0415

        write_curves(rows, out / CURVES, classes=classes, best_epoch=best_epoch, title=out.name)
    except Exception:  # noqa: BLE001 (a picture must never end a training run)
        _LOG.debug("could not draw %s", out / CURVES, exc_info=True)


def _export(model: Any, processor: Any, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(folder))
    processor.save_pretrained(str(folder))


def _artifact(out: Path, *, written: bool) -> dict[str, Any]:
    return {"written": written, "model_dir": str(out), "last_dir": str(out / "last"),
            "checkpoint": str(out / CHECKPOINT), "results_csv": str(out / RESULTS), "curves": str(out / CURVES),
            "manifest": str(out / "manifest.json")}


def _append_results(path: Path, row: Mapping[str, Any], classes: Sequence[str], *, header: bool) -> None:
    columns = ["epoch", "seconds", "lr", "train_loss", "val_loss", "map", "map50", "map75", "mar100"]
    with path.open("w" if header else "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(columns + [f"ap_{name}" for name in classes])
        per_class = row.get("per_class_ap", {})
        writer.writerow([row[c] for c in columns] + [per_class.get(name, "") for name in classes])


def _save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, ema: ModelEMA | None, optimizer: Any,
                     scheduler: Any, scaler: Any, best_fitness: float, best_epoch: int, best_scores: Any,
                     rows: Sequence[Mapping[str, Any]], dataset: DetectionDataset, plan: DetectorPlan,
                     scale_rng: random.Random) -> None:
    state = {"epoch": epoch, "model": model.state_dict(), "ema": ema.module.state_dict() if ema else None,
             "ema_updates": ema.updates if ema else 0, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_fitness": best_fitness,
             "best_epoch": best_epoch, "best_scores": best_scores, "rows": list(rows),
             "classes": list(dataset.classes), "fingerprint": dataset.fingerprint, "plan": plan.as_dict(),
             "scale_rng": scale_rng.getstate()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _dataset_summary(dataset: DetectionDataset, train_used: int) -> dict[str, Any]:
    return {"root": str(dataset.root), "format": dataset.format, "fingerprint": dataset.fingerprint,
            "classes": list(dataset.classes), "train_images": len(dataset.train), "train_images_used": train_used,
            "val_images": len(dataset.val), "val_source": dataset.val_source, "val_fraction": dataset.val_fraction,
            "split_seed": dataset.split_seed, "skipped": list(dataset.skipped)}


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


def _versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for module in ("torch", "torchvision", "transformers"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 (an absent package is recorded as absent)
            versions[module] = "absent"
    return versions


def _write_manifest(out: Path, plan: DetectorPlan, dataset: DetectionDataset, raw: Mapping[str, Any]) -> str:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_predictor": "src.models.detection.closed_set.detector.RtDetrObjectDetector",
        "base_model": plan.base_model,
        "plan": plan.as_dict(),
        "dataset": raw["dataset"],
        "id2label": {str(i): name for i, name in enumerate(dataset.classes)},
        "best_epoch": raw["best_epoch"],
        "epochs_run": len(raw["epochs"]),
        "stopped_early": raw["stopped_early"],
        "diverged": raw["diverged"],
        "metrics": raw["best"],
        "device": raw["device"],
        "amp": raw["amp"],
        "seconds": raw["seconds"],
        "env": _versions(),
    }
    weights = out / "model.safetensors"
    if weights.is_file():
        manifest["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)
