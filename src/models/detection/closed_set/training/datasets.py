"""Detection datasets for training: a COCO or YOLO folder read into one index, a fixed validation split, and every box
or image that was left out, with the reason.

Two formats, because they are what labelling tools export:

- **COCO.** A folder holding ``train/`` (and ``val/``, ``valid/`` or ``validation/``), each with an annotation file and
  its images, or one folder with the annotation file and the images. The annotation file is ``annotations.json``,
  ``_annotations.coco.json`` (Roboflow) or ``instances_*.json`` (CVAT), beside the images or in ``annotations/``; an
  image is looked for in ``images/``, beside the annotation file and at the dataset root.
- **YOLO.** A ``data.yaml`` naming the classes and the ``train``/``val`` image folders (Ultralytics and Roboflow write
  it), or ``classes.txt`` with ``images/`` and ``labels/``, flat or per split. One ``.txt`` per image under the
  matching ``labels/`` folder, one ``class cx cy w h`` line per box, normalised to the image; a polygon line is read as
  the box around it, and an image without a label file is a background image.

Both end in a :class:`DetectionDataset`: the classes in a fixed order, the train and validation images with their
boxes in pixels, how the validation images were chosen, and a fingerprint of the annotations. A dataset without its
own validation split gives ``val_fraction`` of its images to one, chosen by a hash of each image's path, so the same
images land in validation every time. ``classes=`` keeps only the classes it names, the objects of the others left in
the images as background, and the same images still land in validation. Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VAL_FRACTION",
    "DetectionDataset",
    "LabelledBox",
    "LabelledImage",
    "dataset_stats",
    "load_dataset",
    "split_off",
    "write_shapes_dataset",
]

#: The share of the images a dataset without its own validation split gives to one.
DEFAULT_VAL_FRACTION = 0.15
#: The seed of that split. Changing it moves images between train and validation, so it is stamped in the manifest.
DEFAULT_SPLIT_SEED = 0

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_COCO_NAMES = ("annotations.json", "_annotations.coco.json")
_TRAIN_NAMES = ("train",)
_VAL_NAMES = ("val", "valid", "validation")
_YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
_CLASS_LISTS = ("classes.txt", "obj.names", "classes.names")
#: A box narrower or lower than this, in pixels, after clipping to its image, is not a box.
_MIN_SIDE_PX = 1.0


@dataclass(frozen=True, slots=True)
class LabelledBox:
    """One box: the class's index in :attr:`DetectionDataset.classes` and the corners in pixels (x1, y1, x2, y2)."""

    label: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class LabelledImage:
    """One image with its boxes. ``key`` is its path relative to the dataset root, which the split hashes."""

    path: Path
    width: int
    height: int
    boxes: tuple[LabelledBox, ...]
    key: str


@dataclass(frozen=True, slots=True)
class DetectionDataset:
    """A dataset ready to train on: classes in a fixed order, train and validation images, what was left out."""

    root: Path
    format: str
    classes: tuple[str, ...]
    train: tuple[LabelledImage, ...]
    val: tuple[LabelledImage, ...]
    #: "declared" when the dataset brought its own validation split, "split" when it was cut here, "none" otherwise.
    val_source: str
    val_fraction: float
    split_seed: int
    #: One line per left-out category, image or group of boxes, with the reason.
    skipped: tuple[str, ...]
    fingerprint: str

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.classes))

    @property
    def images(self) -> int:
        return len(self.train) + len(self.val)


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------
def load_dataset(root: str | Path, *, format: str = "auto", val_fraction: float = DEFAULT_VAL_FRACTION,
                 split_seed: int = DEFAULT_SPLIT_SEED, require_boxes: bool = True,
                 classes: Sequence[str] | None = None) -> DetectionDataset:
    """Read a COCO or YOLO dataset folder.

    ``format`` is ``"auto"``, ``"coco"`` or ``"yolo"``. ``classes`` names the classes to keep, by name, as
    Ultralytics' ``classes=`` does: they stay in the dataset's order, the boxes of every other class are left out,
    and the objects under them stay in the images as background. Raises ``FileNotFoundError`` when the folder holds
    neither format, and ``ValueError`` for an annotation file that cannot be read, a class name the dataset does not
    have or, with ``require_boxes``, a dataset with no box at all (``inspect`` passes False, to show what was left out
    instead).
    """
    folder = Path(root)
    if not folder.is_dir():
        raise FileNotFoundError(f"no dataset folder at {folder}")
    if format not in ("auto", "coco", "yolo"):
        raise ValueError(f"unknown dataset format {format!r}; choose from auto, coco, yolo")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be at least 0 and below 1, not {val_fraction}")
    chosen = format if format != "auto" else _detect_format(folder)
    if chosen == "coco":
        classes_read, train, val, skipped, sources = _read_coco(folder)
    else:
        classes_read, train, val, skipped, sources = _read_yolo(folder)
    if classes is None:
        kept_classes = list(classes_read)
    else:
        kept_classes, train, val = _keep_classes(folder, classes_read, classes, train, val, skipped)
    if require_boxes and not any(image.boxes for image in (*train, *val)):
        raise ValueError(f"the dataset at {folder} holds no box"
                         + (f" of the class(es) chosen ({', '.join(kept_classes)})" if classes is not None else "")
                         + ": nothing to train on"
                         + (f" ({len(skipped)} item(s) left out, the first: {skipped[0]})" if skipped else ""))
    val_source = "declared" if val else "none"
    if not val and val_fraction > 0.0:
        kept, held = split_off(train, val_fraction, split_seed)
        train, val = list(kept), list(held)
        val_source = "split" if val else "none"
    fingerprint = _fingerprint(chosen, kept_classes, sources, (*train, *val))
    return DetectionDataset(root=folder, format=chosen, classes=tuple(kept_classes), train=tuple(train),
                            val=tuple(val), val_source=val_source,
                            val_fraction=val_fraction if val_source == "split" else 0.0, split_seed=split_seed,
                            skipped=tuple(skipped), fingerprint=fingerprint)


def _keep_classes(folder: Path, every: Sequence[str], wanted: Sequence[str], train: Sequence[LabelledImage],
                  val: Sequence[LabelledImage],
                  skipped: list[str]) -> tuple[list[str], list[LabelledImage], list[LabelledImage]]:
    """The classes named in ``wanted``, in the dataset's order, and every image with only their boxes, renumbered.

    An image left without a box stays, as a background image. One line in ``skipped`` says what was left out.
    """
    names = list(dict.fromkeys(str(name) for name in ([wanted] if isinstance(wanted, str) else wanted)))
    if not names:
        raise ValueError("classes= names no class; leave it out to train every class of the dataset")
    unknown = [name for name in names if name not in every]
    if unknown:
        raise ValueError(f"the dataset at {folder} has no class {', '.join(repr(name) for name in unknown)}; its "
                         f"classes are {', '.join(repr(name) for name in every)}")
    renumbered = {old: new for new, old in enumerate(i for i, name in enumerate(every) if name in names)}
    left_out: Counter[int] = Counter()

    def keep(images: Sequence[LabelledImage]) -> list[LabelledImage]:
        out = []
        for image in images:
            boxes = []
            for box in image.boxes:
                if box.label in renumbered:
                    boxes.append(LabelledBox(renumbered[box.label], box.xyxy))
                else:
                    left_out[box.label] += 1
            out.append(LabelledImage(path=image.path, width=image.width, height=image.height, boxes=tuple(boxes),
                                     key=image.key))
        return out

    kept_train, kept_val = keep(train), keep(val)
    others = [name for i, name in enumerate(every) if i not in renumbered]
    if others:
        skipped.append(f"{sum(left_out.values())} box(es) of the {len(others)} class(es) not chosen "
                       f"({', '.join(others)}): their objects stay in the images, as background")
    return [every[i] for i in renumbered], kept_train, kept_val


def split_off(images: Sequence[LabelledImage], fraction: float,
              seed: int) -> tuple[tuple[LabelledImage, ...], tuple[LabelledImage, ...]]:
    """``fraction`` of the images, at least one, for validation; the rest for training.

    The order is a hash of each image's key and the seed, not a shuffle: the same image lands on the same side every
    time, and an image added later does not move the others. Fewer than two images cannot be split.
    """
    if len(images) < 2 or fraction <= 0.0:
        return tuple(images), ()
    count = min(len(images) - 1, max(1, round(len(images) * fraction)))
    ranked = sorted(images, key=lambda image: hashlib.sha256(f"{seed}:{image.key}".encode()).hexdigest())
    chosen = {image.key for image in ranked[:count]}
    return (tuple(image for image in images if image.key not in chosen),
            tuple(image for image in images if image.key in chosen))


def _detect_format(folder: Path) -> str:
    if any((folder / name).is_file() for name in _YAML_NAMES):
        return "yolo"
    if _coco_files(folder) or any(_coco_files(folder / name) for name in (*_TRAIN_NAMES, *_VAL_NAMES)):
        return "coco"
    if any((folder / name).is_file() for name in _CLASS_LISTS):
        return "yolo"
    raise FileNotFoundError(
        f"{folder} is neither a COCO nor a YOLO dataset: looked for {', '.join(_COCO_NAMES)} or instances_*.json "
        f"(at the root, in train/ or val/, or in annotations/), {', '.join(_YAML_NAMES)}, and "
        f"{', '.join(_CLASS_LISTS)}")


# --------------------------------------------------------------------------------------------------
# COCO
# --------------------------------------------------------------------------------------------------
def _coco_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    found = [folder / name for name in _COCO_NAMES if (folder / name).is_file()]
    for where in (folder, folder / "annotations"):
        if where.is_dir():
            found += sorted(path for path in where.glob("instances_*.json") if path.is_file())
    return list(dict.fromkeys(found))


def _coco_splits(folder: Path) -> tuple[Path, Path | None]:
    """The train annotation file and the validation one, if the dataset declares it."""
    for train_name in _TRAIN_NAMES:
        files = _coco_files(folder / train_name)
        if files:
            val = next((found[0] for name in _VAL_NAMES if (found := _coco_files(folder / name))), None)
            return files[0], val
    files = _coco_files(folder)
    if not files:
        raise FileNotFoundError(f"no COCO annotation file under {folder}")
    by_split = {path.stem.removeprefix("instances_"): path for path in files if path.stem.startswith("instances_")}
    if "train" in by_split:
        return by_split["train"], next((by_split[name] for name in _VAL_NAMES if name in by_split), None)
    if len(files) > 1:
        raise ValueError(f"{folder} holds {len(files)} COCO annotation files ({', '.join(p.name for p in files)}) and "
                         f"none is named for the train split; put each split in its own folder (train/, val/)")
    return files[0], None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the COCO file {path}: {exc}") from exc
    for key in ("images", "annotations", "categories"):
        if key not in doc:
            raise ValueError(f"the COCO file {path} has no {key!r}")
    return doc


def _read_coco(folder: Path) -> tuple[list[str], list[LabelledImage], list[LabelledImage], list[str], list[Path]]:
    train_file, val_file = _coco_splits(folder)
    docs = {"train": _read_json(train_file)}
    if val_file is not None:
        docs["val"] = _read_json(val_file)
    names: dict[int, str] = {}
    for split, doc in docs.items():
        for category in doc["categories"]:
            cid, name = int(category["id"]), str(category["name"]).strip()
            if names.get(cid, name) != name:
                raise ValueError(f"category id {cid} is {names[cid]!r} in one split and {name!r} in the {split} "
                                 f"split; one export must use one category list")
            names[cid] = name
    if not names:
        raise ValueError(f"the COCO file {train_file} declares no categories")
    used = Counter(int(a["category_id"]) for doc in docs.values() for a in doc["annotations"]
                   if not int(a.get("iscrowd", 0)) and int(a["category_id"]) in names)
    skipped: list[str] = []
    for cid in sorted(names):
        if not used[cid]:
            skipped.append(f"category {names[cid]!r} (id {cid}) has no box in any split and is left out")
    kept = [cid for cid in sorted(names) if used[cid]]
    index = {cid: i for i, cid in enumerate(kept)}
    classes = [names[cid] for cid in kept]
    train = _coco_images(docs["train"], train_file, folder, index, skipped, "train")
    val = _coco_images(docs["val"], val_file, folder, index, skipped, "val") if val_file is not None else []
    return classes, train, val, skipped, [train_file] + ([val_file] if val_file is not None else [])


def _coco_images(doc: dict[str, Any], ann_file: Path, root: Path, index: dict[int, int], skipped: list[str],
                 split: str) -> list[LabelledImage]:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in doc["annotations"]:
        by_image.setdefault(int(ann["image_id"]), []).append(ann)
    declared = {int(image["id"]) for image in doc["images"]}
    orphans = sum(len(anns) for iid, anns in by_image.items() if iid not in declared)
    if orphans:
        skipped.append(f"{split}: {orphans} box(es) name an image the file does not declare")
    images: list[LabelledImage] = []
    missing: list[str] = []
    crowd = clipped = degenerate = unknown = 0
    for image in doc["images"]:
        name = str(image["file_name"])
        path = _find_image(name, ann_file, root)
        if path is None:
            missing.append(name)
            continue
        width, height = int(image.get("width") or 0), int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = _image_size(path)
        boxes: list[LabelledBox] = []
        for ann in by_image.get(int(image["id"]), []):
            if int(ann.get("iscrowd", 0)):
                crowd += 1
                continue
            label = index.get(int(ann["category_id"]))
            if label is None:
                unknown += 1
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            box, was_clipped = _clip((x, y, x + w, y + h), width, height)
            if box is None:
                degenerate += 1
                continue
            clipped += was_clipped
            boxes.append(LabelledBox(label, box))
        images.append(LabelledImage(path=path, width=width, height=height, boxes=tuple(boxes),
                                    key=_key(path, root)))
    if missing:
        skipped.append(f"{split}: {len(missing)} image(s) not found, the first {missing[0]!r} (looked in images/, "
                       f"beside {ann_file.name} and at the dataset root)")
    _box_notes(skipped, split, crowd=crowd, unknown=unknown, degenerate=degenerate, clipped=clipped)
    return images


def _find_image(name: str, ann_file: Path, root: Path) -> Path | None:
    here = ann_file.parent
    split_dir = here.parent if here.name == "annotations" else here
    for base in (split_dir / "images", split_dir, here, root / "images", root):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------------------------------
# YOLO
# --------------------------------------------------------------------------------------------------
def _read_yolo(folder: Path) -> tuple[list[str], list[LabelledImage], list[LabelledImage], list[str], list[Path]]:
    yaml_file = next((folder / name for name in _YAML_NAMES if (folder / name).is_file()), None)
    skipped: list[str] = []
    if yaml_file is not None:
        classes, train_dirs, val_dirs = _yolo_yaml(yaml_file, folder)
        sources = [yaml_file]
    else:
        list_file = next((folder / name for name in _CLASS_LISTS if (folder / name).is_file()), None)
        if list_file is None:
            raise FileNotFoundError(f"no {' or '.join(_YAML_NAMES)} and no {' or '.join(_CLASS_LISTS)} in {folder}")
        classes = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_dirs, val_dirs = _yolo_folders(folder)
        sources = [list_file]
    if not classes:
        raise ValueError(f"the YOLO dataset at {folder} names no class")
    train = _yolo_images(train_dirs, folder, len(classes), skipped, "train", sources)
    val = _yolo_images(val_dirs, folder, len(classes), skipped, "val", sources)
    return classes, train, val, skipped, sources


def _yolo_yaml(yaml_file: Path, folder: Path) -> tuple[list[str], list[Path], list[Path]]:
    import yaml  # noqa: PLC0415 (only a YOLO dataset needs it)

    doc = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    names = doc.get("names")
    if isinstance(names, dict):
        classes = [str(names[key]).strip() for key in sorted(names, key=int)]
    elif isinstance(names, list):
        classes = [str(name).strip() for name in names]
    else:
        raise ValueError(f"{yaml_file} has no 'names' list or mapping")
    count = doc.get("nc")
    if count is not None and int(count) != len(classes):
        raise ValueError(f"{yaml_file} says nc: {count} and names {len(classes)} class(es)")
    base = yaml_file.parent / str(doc["path"]) if doc.get("path") else yaml_file.parent

    def inside(text: str) -> str:
        """Roboflow writes ``../train/images`` beside a data.yaml that sits in the dataset root."""
        while text.startswith(("../", "./", "..\\", ".\\")):
            text = text.split("/", 1)[1] if "/" in text[:3] else text.split("\\", 1)[1]
        return text

    def resolve(entry: Any, fallbacks: Iterable[str]) -> list[Path]:
        entries = entry if isinstance(entry, list) else ([entry] if entry else [])
        found: list[Path] = []
        for item in entries:
            text = str(item)
            for candidate in (base / text, yaml_file.parent / text, folder / inside(text),
                              folder / Path(text).parent.name / Path(text).name):
                if candidate.exists():
                    found.append(candidate)
                    break
        if not found:
            for name in fallbacks:
                for candidate in (folder / name / "images", folder / "images" / name):
                    if candidate.is_dir():
                        found.append(candidate)
                        break
                if found:
                    break
        return found

    return classes, resolve(doc.get("train"), _TRAIN_NAMES), resolve(doc.get("val"), _VAL_NAMES)


def _yolo_folders(folder: Path) -> tuple[list[Path], list[Path]]:
    for name in _TRAIN_NAMES:
        if (folder / name / "images").is_dir():
            val = next(([folder / v / "images"] for v in _VAL_NAMES if (folder / v / "images").is_dir()), [])
            return [folder / name / "images"], val
    if (folder / "images" / "train").is_dir():
        val = next(([folder / "images" / v] for v in _VAL_NAMES if (folder / "images" / v).is_dir()), [])
        return [folder / "images" / "train"], val
    if (folder / "images").is_dir():
        return [folder / "images"], []
    raise FileNotFoundError(f"no images/ folder in {folder}")


def _list_images(entry: Path) -> list[Path]:
    if entry.is_file() and entry.suffix.lower() == ".txt":
        listed = [line.strip() for line in entry.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else (entry.parent / path) for path in map(Path, listed)]
    return sorted(path for path in entry.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file())


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _yolo_images(entries: Sequence[Path], root: Path, n_classes: int, skipped: list[str], split: str,
                 sources: list[Path]) -> list[LabelledImage]:
    images: list[LabelledImage] = []
    bad_lines = unknown = degenerate = clipped = unreadable = 0
    first_bad = ""
    for entry in entries:
        for path in _list_images(entry):
            if not path.is_file():
                unreadable += 1
                continue
            try:
                width, height = _image_size(path)
            except (OSError, ValueError):
                unreadable += 1
                continue
            label_file = _label_path(path)
            boxes: list[LabelledBox] = []
            if label_file.is_file():
                sources.append(label_file)
                for number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
                    values = line.split()
                    if not values:
                        continue
                    try:
                        label = int(float(values[0]))
                        coords = [float(v) for v in values[1:]]
                    except ValueError:
                        coords = []
                        label = -1
                    if len(coords) == 4:
                        cx, cy, w, h = coords
                        corners = ((cx - w / 2) * width, (cy - h / 2) * height,
                                   (cx + w / 2) * width, (cy + h / 2) * height)
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        xs, ys = coords[0::2], coords[1::2]
                        corners = (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)
                    else:
                        bad_lines += 1
                        first_bad = first_bad or f"{label_file.name}:{number}"
                        continue
                    if not 0 <= label < n_classes:
                        unknown += 1
                        continue
                    box, was_clipped = _clip(corners, width, height)
                    if box is None:
                        degenerate += 1
                        continue
                    clipped += was_clipped
                    boxes.append(LabelledBox(label, box))
            images.append(LabelledImage(path=path, width=width, height=height, boxes=tuple(boxes),
                                        key=_key(path, root)))
    if unreadable:
        skipped.append(f"{split}: {unreadable} image(s) could not be read")
    if bad_lines:
        skipped.append(f"{split}: {bad_lines} label line(s) are neither 'class cx cy w h' nor a polygon, the first "
                       f"{first_bad}")
    _box_notes(skipped, split, crowd=0, unknown=unknown, degenerate=degenerate, clipped=clipped)
    return images


# --------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------
def _clip(xyxy: tuple[float, float, float, float], width: int,
          height: int) -> tuple[tuple[float, float, float, float] | None, int]:
    x1, y1, x2, y2 = xyxy
    cx1, cy1 = min(max(x1, 0.0), float(width)), min(max(y1, 0.0), float(height))
    cx2, cy2 = min(max(x2, 0.0), float(width)), min(max(y2, 0.0), float(height))
    if cx2 - cx1 < _MIN_SIDE_PX or cy2 - cy1 < _MIN_SIDE_PX:
        return None, 0
    moved = max(abs(cx1 - x1), abs(cy1 - y1), abs(cx2 - x2), abs(cy2 - y2)) > 1.0
    return (cx1, cy1, cx2, cy2), int(moved)


def _box_notes(skipped: list[str], split: str, *, crowd: int, unknown: int, degenerate: int, clipped: int) -> None:
    if crowd:
        skipped.append(f"{split}: {crowd} crowd region(s) (iscrowd 1) are not boxes to learn and are left out")
    if unknown:
        skipped.append(f"{split}: {unknown} box(es) name a class the dataset does not declare")
    if degenerate:
        skipped.append(f"{split}: {degenerate} box(es) are under {_MIN_SIDE_PX:.0f} px wide or high inside the "
                       f"image and are left out")
    if clipped:
        skipped.append(f"{split}: {clipped} box(es) reached past the image edge and were cut to it (kept)")


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image  # noqa: PLC0415 (only an image without its size in the annotations needs it)

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _key(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _fingerprint(fmt: str, classes: Sequence[str], sources: Sequence[Path], images: Iterable[LabelledImage]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{fmt}\n{json.dumps(list(classes))}\n".encode())
    for source in sorted(set(sources), key=lambda p: p.as_posix()):
        digest.update(hashlib.sha256(source.read_bytes()).hexdigest().encode())
    for key in sorted(image.key for image in images):
        digest.update(key.encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------------
# What a dataset holds
# --------------------------------------------------------------------------------------------------
def dataset_stats(dataset: DetectionDataset) -> dict[str, Any]:
    """Counts per class and split, box sizes in COCO's small/medium/large bands, and image sizes."""

    def split_stats(images: Sequence[LabelledImage]) -> dict[str, Any]:
        per_class = Counter(box.label for image in images for box in image.boxes)
        areas = [(box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1]) for image in images for box in image.boxes]
        return {
            "images": len(images),
            "images_without_boxes": sum(1 for image in images if not image.boxes),
            "boxes": sum(per_class.values()),
            "boxes_per_class": {dataset.classes[i]: per_class.get(i, 0) for i in range(len(dataset.classes))},
            "small": sum(1 for a in areas if a < 32 ** 2),
            "medium": sum(1 for a in areas if 32 ** 2 <= a < 96 ** 2),
            "large": sum(1 for a in areas if a >= 96 ** 2),
        }

    sizes = [(image.width, image.height) for image in (*dataset.train, *dataset.val)]
    return {
        "train": split_stats(dataset.train),
        "val": split_stats(dataset.val),
        "image_sizes": {"min": [min(w for w, _ in sizes), min(h for _, h in sizes)],
                        "max": [max(w for w, _ in sizes), max(h for _, h in sizes)]} if sizes else {},
    }


def write_shapes_dataset(folder: str | Path, *, images: int = 48, seed: int = 0) -> Path:
    """A small COCO dataset of three coloured shapes on noisy backgrounds, with exact boxes. Returns the folder.

    For proving the chain on a new machine before anything is labelled: a model trained on it says nothing about
    real parts. Every call with the same arguments writes the same files.
    """
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageFilter  # noqa: PLC0415

    root = Path(folder)
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    categories = [{"id": 1, "name": "red_block"}, {"id": 2, "name": "green_disc"}, {"id": 3, "name": "blue_wedge"}]
    target = root / "images"
    target.mkdir(parents=True, exist_ok=True)
    doc: dict[str, list[dict[str, Any]]] = {"images": [], "annotations": [], "categories": categories}
    for number in range(images):
        width, height = 640, 480
        background = (noise.random((height, width, 3)) * 60 + rng.randint(40, 140)).astype("uint8")
        image = Image.fromarray(background).filter(ImageFilter.GaussianBlur(1))
        draw = ImageDraw.Draw(image)
        placed: list[tuple[int, int, int]] = []
        for _ in range(rng.randint(1, 4)):
            category = rng.choice(categories)["id"]
            side = rng.randint(48, 120)
            for _attempt in range(30):
                x, y = rng.randint(0, width - side - 1), rng.randint(0, height - side - 1)
                if all(x + side < px or px + ps < x or y + side < py or py + ps < y for px, py, ps in placed):
                    break
            placed.append((x, y, side))
            shade = rng.randint(0, 55)
            if category == 1:
                draw.rectangle([x, y, x + side, y + side], fill=(200 + shade, 30, 30))
            elif category == 2:
                draw.ellipse([x, y, x + side, y + side], fill=(30, 180 + shade, 40))
            else:
                draw.polygon([(x, y + side), (x + side, y + side), (x + side // 2, y)], fill=(30, 50, 200 + shade))
            doc["annotations"].append({"id": len(doc["annotations"]) + 1, "image_id": number + 1,
                                       "category_id": category, "bbox": [x, y, side, side],
                                       "area": side * side, "iscrowd": 0})
        name = f"shapes_{number:04d}.png"
        image.save(target / name)
        doc["images"].append({"id": number + 1, "file_name": name, "width": width, "height": height})
    (root / "annotations.json").write_text(json.dumps(doc), encoding="utf-8")
    return root
