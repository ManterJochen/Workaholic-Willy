"""Training the closed-set detector from code, in the idiom of ``GeneratorTraining``: name a dataset, train, keep the
model and its report.

    from willy import DetectorTraining

    run = DetectorTraining.from_dataset(dataset="D:/data/my_parts", out_dir="models/rtdetr/my_parts", epochs=80)
    print(run.describe())        # what will run, before it costs anything
    print(run.probe())           # what the dataset holds, and what was left out of it
    report = run.train()         # the best epoch lands in out_dir
    print(report)
    run.write_report(report)     # report.json, curves.png and report.html beside the model

The shape mirrors ``GeneratorTraining``: keyword-only factories, one verb, a frozen typed report, and side effects as
separate methods. The dataset is a COCO or YOLO folder (``datasets.py``), all of its classes or those ``classes=``
names; the plan is recipe ``v1`` at a tier, with ``epochs=``, ``batch=``, ``image_size=`` and anything else chosen in
:class:`DetectorPlanOverrides` winning (``plan.py``). The model in ``out_dir`` loads in ``RtDetrObjectDetector`` and
through ``models.detector: "rtdetr"`` with ``models.rtdetr.model_path``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from src.contracts.options import merged_overrides
from src.models.detection.closed_set.training.datasets import (
    DetectionDataset,
    dataset_stats,
    load_dataset,
    write_shapes_dataset,
)
from src.models.detection.closed_set.training.plan import DetectorPlan, DetectorPlanOverrides, build_plan
from src.models.detection.closed_set.training.recipes import TIERS
from src.models.detection.closed_set.training.report import DetectorTrainingReport

__all__ = ["DatasetProbe", "DetectorTraining", "DetectorTrainingContext"]

#: The line separator for the rendered text, built without a backslash escape.
NEWLINE = chr(10)
#: Fewer training boxes than this for one class is worth a line before the run.
_FEW_BOXES = 30


@dataclass(frozen=True, slots=True)
class DatasetProbe:
    """What a dataset holds, before anything is trained: the counts a run's numbers stand on, and what was left out."""

    root: str
    format: str
    classes: tuple[str, ...]
    stats: Mapping[str, Any]
    val_source: str
    val_fraction: float
    skipped: tuple[str, ...]
    warnings: tuple[str, ...]
    fingerprint: str

    @classmethod
    def of(cls, dataset: DetectionDataset) -> "DatasetProbe":
        stats = dataset_stats(dataset)
        warnings: list[str] = []
        train_boxes = stats["train"]["boxes_per_class"]
        val_boxes = stats["val"]["boxes_per_class"]
        for name in dataset.classes:
            if not train_boxes.get(name):
                warnings.append(f"{name}: no training box, so the model cannot learn it")
            elif train_boxes[name] < _FEW_BOXES:
                warnings.append(f"{name}: only {train_boxes[name]} training box(es); a few hundred per class is "
                                f"where fine-tuning usually holds")
            if dataset.val and not val_boxes.get(name):
                warnings.append(f"{name}: no validation box, so its AP is not measured")
        if not dataset.val:
            warnings.append("no validation images: the best epoch is chosen by the training loss and no mAP is "
                            "measured")
        return cls(root=str(dataset.root), format=dataset.format, classes=dataset.classes, stats=stats,
                   val_source=dataset.val_source, val_fraction=dataset.val_fraction, skipped=dataset.skipped,
                   warnings=tuple(warnings), fingerprint=dataset.fingerprint)

    def render(self) -> str:
        train, val = self.stats["train"], self.stats["val"]
        source = {"declared": "its own validation split", "none": "no validation split",
                  "split": f"{self.val_fraction:.0%} split off for validation"}.get(self.val_source, self.val_source)
        lines = [f"  dataset    {self.root} ({self.format.upper()}, {source})",
                 f"  images     {train['images']} train, {val['images']} val "
                 f"({train['images_without_boxes'] + val['images_without_boxes']} without a box)",
                 f"  boxes      {train['boxes']} train, {val['boxes']} val; small {train['small'] + val['small']}, "
                 f"medium {train['medium'] + val['medium']}, large {train['large'] + val['large']}",
                 "  " + f"{'class':24}{'train':>8}{'val':>8}"]
        for name in self.classes:
            lines.append("  " + f"{name[:24]:24}{train['boxes_per_class'].get(name, 0):>8}"
                         f"{val['boxes_per_class'].get(name, 0):>8}")
        for line in self.skipped:
            lines.append(f"  left out   {line}")
        for line in self.warnings:
            lines.append(f"  note       {line}")
        return NEWLINE.join(lines)

    def __str__(self) -> str:
        return self.render()

    def as_dict(self) -> dict[str, Any]:
        return {"root": self.root, "format": self.format, "classes": list(self.classes), "stats": dict(self.stats),
                "val_source": self.val_source, "val_fraction": self.val_fraction, "skipped": list(self.skipped),
                "warnings": list(self.warnings), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class DetectorTrainingContext:
    """Where the data is, where the model goes, and how the run starts; none of it is plan state."""

    dataset: Path
    out_dir: Path | None = None
    format: str = "auto"
    device: str | None = None
    resume: bool = False
    #: The classes to train, by name; ``None`` is every class of the dataset.
    classes: tuple[str, ...] | None = None


@dataclass
class DetectorTraining:
    """Train the closed-set RT-DETR detector on your own labelled images, then use it through
    ``ObjectDetector.from_weights(out_dir)``.

    ```python
    run = DetectorTraining.from_dataset(dataset="data/chess_pieces", out_dir="models/chess", epochs=80)
    print(run.probe())          # what the dataset holds, per class and split; trains nothing
    report = run.train()        # the best epoch in models/chess, the last in models/chess/last
    print(report)
    run.write_report(report)    # report.json, curves.png and report.html beside the model
    ```

    While it trains, a terminal shows a bar over each epoch's batches and the validation, and a line per epoch with
    the best mAP so far and the time left; ``curves.png`` in ``out_dir`` is drawn again after every epoch.

    Build it with :meth:`from_dataset` (a recipe and a tier) or :meth:`from_plan`, then call :meth:`train` once.

    Attributes:
        plan (DetectorPlan): Every training setting, resolved.
        context (DetectorTrainingContext): The dataset, the out dir, the format, the device and ``resume``.
        recipe_notes (Mapping[str, Any]): What the recipe and the tier set, for a caller that logs it (default: {}).
    """

    plan: DetectorPlan
    context: DetectorTrainingContext
    #: What a recipe or tier supplied, and what the caller had already chosen. Empty without one.
    recipe_notes: Mapping[str, Any] = field(default_factory=dict)
    _on_epoch: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False)
    _dataset: DetectionDataset | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_plan(cls, *, dataset: str | Path, plan: DetectorPlan | None = None, out_dir: str | Path | None = None,
                  format: str = "auto", device: str | None = None, resume: bool = False,
                  classes: Sequence[str] | None = None) -> "DetectorTraining":
        """A run from a plan you built yourself, with no recipe resolution.

        Args:
            dataset (str | Path): The dataset folder, COCO or YOLO as CVAT, Label Studio or Roboflow export them.
            plan (DetectorPlan | None): Every setting; ``None`` is a default ``DetectorPlan()`` (default: None).
            out_dir (str | Path | None): Where the model goes: the best epoch at the top, the last in ``out_dir/last``;
                required before ``train()`` (default: None).
            format (str): The dataset's format: ``"coco"``, ``"yolo"`` or ``"auto"``, which tells them apart by their
                files (default: "auto").
            device (str | None): ``"cuda"``, ``"cpu"`` or ``"mps"``; ``None`` is ``WILLY_DEVICE`` or the first of CUDA,
                MPS and the CPU (default: None).
            resume (bool): Go on from the last epoch in ``out_dir``, on the same dataset (default: False).
            classes (Sequence[str] | None): The classes to train, by name, in the dataset's order; the objects of the
                others stay in the images as background. ``None`` is every class (default: None).

        Returns:
            DetectorTraining: The run; nothing is read or trained yet. A class name the dataset does not have is
                refused when it is read, by :meth:`probe` or :meth:`train`.
        """
        folder = Path(dataset)
        if not folder.is_dir():
            raise FileNotFoundError(f"no dataset folder at {folder}")
        chosen = None if classes is None else tuple([classes] if isinstance(classes, str) else classes)
        return cls(plan=(plan or DetectorPlan()).validated(),
                   context=DetectorTrainingContext(dataset=folder, out_dir=Path(out_dir) if out_dir else None,
                                                   format=format, device=device, resume=resume, classes=chosen))

    @classmethod
    def from_dataset(cls, *, dataset: str | Path, recipe: str | None = "v1", tier: str | None = "full",
                     epochs: int | None = None, batch: int | None = None, image_size: int | None = None,
                     classes: Sequence[str] | None = None, overrides: DetectorPlanOverrides | None = None,
                     base: DetectorPlan | None = None, out_dir: str | Path | None = None, format: str = "auto",
                     device: str | None = None, resume: bool = False) -> "DetectorTraining":
        """A run from a dataset folder, a recipe and a tier, plus what you chose explicitly.

        ```python
        DetectorTraining.from_dataset(dataset="data", out_dir="models/chess", epochs=80, batch=8,
                                      image_size=640, classes=["white pawn", "black pawn"])
        ```

        Args:
            dataset (str | Path): The dataset folder, COCO or YOLO; one without a validation split gives
                ``val_fraction`` of its images to one.
            recipe (str | None): The frozen recipe, ``"v1"``: RT-DETR's own (default: "v1").
            tier (str | None): ``"full"``, up to 50 epochs, stopped after 15 without a better validation mAP, the best
                kept; or ``"smoke"``, 2 epochs on 64 images, which proves the chain and says nothing about quality
                (default: "full").
            epochs (int | None): The most epochs; the run still stops after ``patience`` epochs without a better
                validation mAP. ``None`` is the tier's (default: None).
            batch (int | None): Images per batch; fewer where the GPU runs out of memory. ``None`` is 8
                (default: None).
            image_size (int | None): The training size in pixels, a multiple of 32, and the multi-scale sizes scaled
                with it; larger for small parts far from the camera, at a quadratic cost. ``None`` is the recipe's, 640
                (default: None).
            classes (Sequence[str] | None): The classes to train, by name, in the dataset's order; the objects of the
                others stay in the images as background. ``None`` is every class (default: None).
            overrides (DetectorPlanOverrides | None): Every other setting you choose explicitly, the learning rate
                among them; they outrank the recipe and the tier. A setting given here and as an argument is refused
                (default: None).
            base (DetectorPlan | None): The plan the recipe, the tier and the overrides are laid onto; ``None`` the
                defaults (default: None).
            out_dir (str | Path | None): Where the model goes: the best epoch at the top, the last in ``out_dir/last``;
                required before ``train()`` (default: None).
            format (str): The dataset's format: ``"coco"``, ``"yolo"`` or ``"auto"``, which tells them apart by their
                files (default: "auto").
            device (str | None): ``"cuda"``, ``"cpu"`` or ``"mps"``; ``None`` is ``WILLY_DEVICE`` or the first of CUDA,
                MPS and the CPU (default: None).
            resume (bool): Go on from the last epoch in ``out_dir``, on the same dataset (default: False).

        Returns:
            DetectorTraining: The run; nothing is read or trained until :meth:`probe` or :meth:`train`.

        Raises:
            ValueError: An unknown recipe or tier, a setting no run can take, or one given twice.
            FileNotFoundError: The dataset folder is not there.
        """
        overrides = merged_overrides(overrides, DetectorPlanOverrides, epochs=epochs, batch=batch,
                                     image_size=image_size)
        plan, notes = build_plan(recipe=recipe, tier=tier, overrides=overrides, base=base)
        built = cls.from_plan(dataset=dataset, plan=plan, out_dir=out_dir, format=format, device=device,
                              resume=resume, classes=classes)
        built.recipe_notes = notes
        return built

    @staticmethod
    def shapes_dataset(folder: str | Path, *, images: int = 48, seed: int = 0) -> Path:
        """Write a small COCO dataset of three coloured shapes, for proving the chain on a new machine before anything
        is labelled; a model trained on it says nothing about real parts.

        Args:
            folder (str | Path): Where to write it.
            images (int): How many images (default: 48).
            seed (int): The seed the shapes are drawn from (default: 0).

        Returns:
            Path: The dataset folder, ready for :meth:`from_dataset`.
        """
        return write_shapes_dataset(folder, images=images, seed=seed)

    # ------------------------------------------------------------------ opt-in side channels
    def attach_progress_listener(self, listener: Callable[[Mapping[str, Any]], None] | None) -> None:
        """Be told as each epoch finishes, on the training thread.

        Args:
            listener (Callable[[Mapping[str, Any]], None] | None): Called with the epoch's row (its losses and
                validation mAP); ``None`` detaches.
        """
        self._on_epoch = listener

    # ------------------------------------------------------------------ before it costs anything
    def dataset(self) -> DetectionDataset:
        """The dataset as the run will see it, read once and kept.

        Returns:
            DetectionDataset: Its classes, splits and images.

        Raises:
            FileNotFoundError: The folder holds neither a COCO nor a YOLO dataset.
            ValueError: The dataset does not read.
        """
        if self._dataset is None:
            self._dataset = load_dataset(self.context.dataset, format=self.context.format,
                                         val_fraction=self.plan.val_fraction, classes=self.context.classes)
        return self._dataset

    def describe(self) -> str:
        """What this run will do, before it costs anything.

        Returns:
            str: The plan and the dataset, as operator text, ASCII.
        """
        plan = self.plan
        stop = plan.stop_augment_epoch()
        chosen = self.context.classes
        lines = [
            f"  dataset    {self.context.dataset} ({self.context.format})",
            f"  model      {plan.base_model} at {plan.image_size} px, the head rebuilt for "
            + (f"{len(chosen)} chosen class(es): {', '.join(chosen)}; the other objects stay background"
               if chosen is not None else "the dataset's classes"),
            f"  schedule   up to {plan.epochs} epoch(s), batch {plan.batch}"
            + (f" x {plan.accumulate}" if plan.accumulate > 1 else "")
            + f", lr {plan.learning_rate:g} (backbone x{plan.backbone_lr_scale:g}), warm-up then cosine",
            "  keeps      the best epoch by validation mAP@0.5:0.95"
            + (f", stopping after {plan.patience} epoch(s) without a better one" if plan.patience else "")
            + ("; averaged weights (EMA)" if plan.ema else ""),
            "  augment    " + (", ".join(part for part in (
                "colour, zoom-out, IoU crop" if plan.augment else "", "flip" if plan.augment else "",
                f"multi-scale {min(plan.multiscale_sizes())}-{max(plan.multiscale_sizes())} px"
                if plan.multiscale else "") if part)
                or "none") + (f", off from epoch {stop + 1}" if (plan.augment or plan.multiscale)
                              and stop < plan.epochs else ""),
            f"  out        {self.context.out_dir or 'NOT SET: train() refuses without out_dir'}"
            + (" (resuming)" if self.context.resume else ""),
        ]
        if plan.recipe or plan.tier:
            lines.append(f"  recipe     {plan.recipe or 'none'}" + (f"  tier {plan.tier}" if plan.tier else ""))
        if plan.tier in TIERS:
            lines.append(f"  ** {plan.tier.upper()} TIER: {TIERS[plan.tier]['why']}."
                         if plan.tier == "smoke" else f"  tier       {TIERS[plan.tier]['why']}")
        return NEWLINE.join(lines)

    def probe(self) -> DatasetProbe:
        """What the dataset holds, per class and split, and what was left out of it. Trains nothing, no GPU.

        Returns:
            DatasetProbe: The classes, the counts per split, and the images and boxes left out with why; prints as
                itself.
        """
        return DatasetProbe.of(self.dataset())

    # ------------------------------------------------------------------ the one verb
    def train(self) -> DetectorTrainingReport:
        """Train, keep the best epoch in ``out_dir`` and the last in ``out_dir/last``.

        Returns:
            DetectorTrainingReport: What happened: every epoch's row, the best validation mAP and its epoch, and
                ``model_dir`` (``None`` where no model was written); prints as itself. ``report.json`` is written by
                :meth:`write_report`.

        Raises:
            ValueError: A run that cannot start: no ``out_dir``, no training image, a resume against another dataset, a
                base model that cannot be loaded.
        """
        from src.models.detection.closed_set.training.trainer import train_detector  # noqa: PLC0415 (torch)

        if self.context.out_dir is None:
            raise ValueError("no out_dir was given, so there is nowhere to keep the model; pass out_dir=")
        raw = train_detector(self.dataset(), self.plan, out_dir=self.context.out_dir, device=self.context.device,
                             resume=self.context.resume, on_epoch=self._on_epoch)
        return DetectorTrainingReport.from_trainer(raw, self.plan)

    def write_report(self, report: DetectorTrainingReport, path: str | Path | None = None) -> Path:
        """Write ``report.json`` beside the model, and beside it ``report.html``: the curves, AP per class and every
        epoch, one page that opens in any browser. ``curves.png`` is drawn once more from every epoch.

        Args:
            report (DetectorTrainingReport): What :meth:`train` returned.
            path (str | Path | None): Where to write ``report.json``; ``None`` is ``out_dir/report.json``; the page
                goes into the same folder (default: None).

        Returns:
            Path: Where ``report.json`` went.
        """
        import json  # noqa: PLC0415

        from src.models.detection.closed_set.training.report import write_curves, write_html  # noqa: PLC0415

        if path is not None:
            target = Path(path)
        elif self.context.out_dir is not None:
            target = self.context.out_dir / "report.json"
        else:
            raise ValueError("no out_dir was given, so there is nowhere to write report.json; pass an explicit path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
        curves = Path(str(report.artifact.get("curves") or target.parent / "curves.png"))
        name = Path(str(report.artifact.get("model_dir") or target.parent)).name
        if report.epochs:
            try:
                write_curves([row.raw for row in report.epochs], curves, classes=tuple(report.raw.get("classes", ())),
                             best_epoch=report.best_epoch, title=name)
            except Exception:  # noqa: BLE001 (the page still carries its tables)
                pass
        write_html(report, target.with_name("report.html"), curves=curves if curves.is_file() else None, title=name)
        return target
