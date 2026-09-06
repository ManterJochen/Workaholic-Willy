"""Make a written scene visible to a human, without touching what the model gets fed.

The dataset's depth and instance PNGs are correct and unviewable, and those are not in tension. Depth
is uint16 millimetres and instances are uint16 object ids, so a 300 mm surface renders at 0.5 %
brightness and object 3 at 0.005 %: in any viewer both are black. A folder of black PNGs reads as a
broken render, and a dataset nobody can look at is a dataset nobody can check.

So the fix is a second set of images rather than a change to the first. Rescaling the stored depth to
look nice would throw away the millimetres the depth is for, and quantising ids to visible greys would
destroy the identity they encode. These previews are lossy, derived, and never read back.

The RGB panel carries the labels drawn on it, which is the point of the whole exercise: every
automated check in this package compares numbers against numbers, and a box drawn on a picture is the
one check a person can make at a glance. A mask set shifted by one object is obvious here in a second
and invisible to the pixel counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, PREVIEW_LOG_FILE

__all__ = ["PreviewReport", "preview_dataset", "preview_scene", "write_how_to_read"]

logger = create_logger("datagen.preview", PREVIEW_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Distinct, saturated, and one per object at the object counts the shipped families produce (at
#: most 13). Cycled by id, so a scene of more than 20 objects repeats a colour; the background is
#: reserved as dark grey so "nothing here" never reads as an object.
_INSTANCE_COLOURS = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)
_BACKGROUND_GREY = 40
#: Gap between panels in the strip, so a black frame is distinguishable from a missing one.
_GUTTER_PX = 6


@dataclass(frozen=True, slots=True)
class PreviewReport:
    scenes: int
    written: int
    skipped: tuple[str, ...]

    def summary(self) -> str:
        head = f"{self.written} preview(s) from {self.scenes} scene(s)"
        if not self.skipped:
            return f"OK: {head}"
        return f"{head}; {len(self.skipped)} skipped\n  " + "\n  ".join(self.skipped[:20])


def _colourise_depth(
    depth_mm: np.ndarray, *, scale: tuple[float, float] | None = None,
) -> np.ndarray:
    """Depth to a colour image, normalised to ``scale`` or to its own valid range. 0 stays black.

    Per-image normalisation by default, not a fixed one: a wrist view and an oblique view span very
    different distances and each needs its full contrast, and the numbers a consumer should trust are
    in the uint16 file next to this one. Every panel is stamped with the range it used, so a reader is
    never guessing what the colours mean; that stamp is what makes per-image scaling honest rather
    than misleading.

    ``scale`` forces a shared range, which is how the noisy depth is drawn against the clean one.
    """
    import cv2  # noqa: PLC0415

    valid = depth_mm > 0
    out = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    near, far = scale if scale is not None else (
        float(depth_mm[valid].min()), float(depth_mm[valid].max()))
    span = max(far - near, 1e-6)
    # Inverted so near is bright: the thing the camera is looking at should be the thing that stands
    # out, and a bin's far floor dominating the image is the opposite of useful.
    scaled = np.zeros(depth_mm.shape, dtype=np.uint8)
    normalised = np.clip((depth_mm[valid] - near) / span, 0.0, 1.0)  # a shared scale can be exceeded
    scaled[valid] = (255.0 * (1.0 - normalised)).astype(np.uint8)
    colour_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    coloured = cv2.applyColorMap(scaled, colour_map)
    coloured[~valid] = 0
    return coloured


def _colourise_instances(instance_map: np.ndarray) -> np.ndarray:
    """Object ids to distinct colours. Background is grey, never black, so an empty map is obvious."""
    out = np.full((*instance_map.shape, 3), _BACKGROUND_GREY, dtype=np.uint8)
    for value in np.unique(instance_map):
        if value == 0:
            continue
        colour = _INSTANCE_COLOURS[(int(value) - 1) % len(_INSTANCE_COLOURS)]
        out[instance_map == value] = colour[::-1]  # cv2 writes BGR
    return out


def _draw_labels(rgb: np.ndarray, objects: list[dict]) -> np.ndarray:
    """The labels, on the picture. The one check a person can make faster than a computer."""
    import cv2  # noqa: PLC0415

    canvas = np.ascontiguousarray(rgb.copy())
    for label in objects:
        box = label.get("bbox_xyxy")
        if box is None:
            continue
        index = int(label["instance_id"])
        colour = tuple(int(c) for c in _INSTANCE_COLOURS[index % len(_INSTANCE_COLOURS)][::-1])
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        # The id and how much of the object is actually visible; an occlusion figure next to the box
        # it belongs to is how a person spots a mask that has been handed to the wrong object.

        text = f"{index}:{float(label.get('visibility', 0.0)) * 100:.0f}%"
        cv2.putText(canvas, text, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return canvas


def _caption(panel: np.ndarray, text: str) -> np.ndarray:
    """A label strip under a panel, so a reader can tell which image in a strip is which."""
    import cv2  # noqa: PLC0415

    strip = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([panel, strip])


def preview_scene(scene_dir: Path, out_dir: Path | None = None) -> tuple[int, list[str]]:
    """Write the viewable images for one scene. Returns ``(views done, skipped reasons)``.

    Per view: a ``_view.png`` twin for each unviewable file, plus the combined ``_preview.png`` strip.
    The twins exist because a folder where three of five images are black is a folder that reads as
    broken, and any single file should open and show what it holds.

    They land in ``out_dir``: one flat directory for the whole dataset, names carrying the scene id.
    The point of these files is to be skimmed, and skimming a dataset means opening one folder and
    scrolling rather than one folder per scene. It also leaves each scene directory holding nothing
    but the data a consumer reads.

    The originals are never touched. ``_depth.png`` stays uint16 millimetres and ``_instances.png``
    stays uint16 ids, because those are the dataset: rescaling depth to look nice discards the
    millimetres it exists to carry, and recolouring ids destroys the identity the labels are keyed on.
    Viewable and lossless cannot be the same file here, so they are two files.
    """
    import json  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    from datagen.render.writer import decode_depth_png  # noqa: PLC0415

    payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    destination = Path(out_dir) if out_dir is not None else scene_dir
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{scene_dir.name}_" if destination != scene_dir else ""
    written, skipped = 0, []
    for view in payload["views"]:
        name = view["name"]
        rgb_path = scene_dir / f"{name}_rgb.png"
        if not rgb_path.exists():
            skipped.append(f"{scene_dir.name}/{name}: no rgb ({view.get('outcome')})")
            continue
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            # A file that exists and will not decode. Said out loud rather than skipped silently:
            # a truncated PNG is a real way for a long run to lose data.
            skipped.append(f"{scene_dir.name}/{name}: {rgb_path.name} exists but will not decode")
            continue
        panels = [_caption(_draw_labels(rgb, view["objects"]),
                           f"{name} rgb + {len(view['objects'])} label(s)")]

        depth_path = scene_dir / f"{name}_depth.png"
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
        scale: tuple[float, float] | None = None
        if depth_raw is not None:
            depth = decode_depth_png(depth_raw)
            valid = depth[depth > 0]
            if valid.size:
                scale = (float(valid.min()), float(valid.max()))
            span = f"{valid.min():.0f}-{valid.max():.0f} mm" if valid.size else "no valid depth"
            panel = _caption(_colourise_depth(depth), f"depth  {span}  (near = bright)")
            cv2.imwrite(str(destination / f"{stem}{name}_depth_view.png"), panel)
            panels.append(panel)

        noisy_path = scene_dir / f"{name}_depth_noisy.png"
        noisy_raw = cv2.imread(str(noisy_path), cv2.IMREAD_UNCHANGED) if noisy_path.exists() else None
        if noisy_raw is not None:
            noisy = decode_depth_png(noisy_raw)
            # Coloured against the clean depth's range, not its own. Sharing the scale is what makes
            # the two panels comparable; the sensor model drops pixels and shifts others, and if
            # each image normalised itself the difference would read as a change of scene.
            valid = noisy[noisy > 0]
            dropped = int(np.count_nonzero((depth_raw > 0) & (noisy_raw == 0))) if depth_raw is not None else 0
            span = f"{valid.min():.0f}-{valid.max():.0f} mm" if valid.size else "no valid depth"
            panel = _caption(_colourise_depth(noisy, scale=scale),
                             f"depth_noisy  {span}  {dropped} px dropped")
            cv2.imwrite(str(destination / f"{stem}{name}_depth_noisy_view.png"), panel)
            panels.append(panel)

        mask_path = scene_dir / f"{name}_instances.png"
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.exists() else None
        if mask_raw is not None:
            instances = np.asarray(mask_raw)
            count = int(np.unique(instances).size) - (1 if 0 in np.unique(instances) else 0)
            panel = _caption(_colourise_instances(instances), f"instances  {count} object(s)")
            cv2.imwrite(str(destination / f"{stem}{name}_instances_view.png"), panel)
            panels.append(panel)

        height = max(panel.shape[0] for panel in panels)
        gutter = np.zeros((height, _GUTTER_PX, 3), dtype=np.uint8)
        strip: list[np.ndarray] = []
        for panel in panels:
            padded = np.zeros((height, panel.shape[1], 3), dtype=np.uint8)
            padded[:panel.shape[0]] = panel
            strip.extend([padded, gutter])
        cv2.imwrite(str(destination / f"{stem}{name}_preview.png"), np.hstack(strip[:-1]))
        written += 1
    return written, skipped


#: Written into every dataset the previews touch. A folder full of black PNGs reads as a broken run,
#: and the explanation belongs where the confusion happens rather than in a README nobody opens.
_HOW_TO_READ = """\
HOW TO READ THIS DATASET
========================

Some PNGs here look BLACK in an image viewer. That is correct, and it is not a broken render.

  <view>_depth.png        uint16, values are MILLIMETRES   (0 = invalid / no return)
  <view>_depth_noisy.png  the same, with the sensor model applied
  <view>_instances.png    uint16, values are OBJECT IDS    (0 = background)

A 300 mm surface stored as 300 out of a 16-bit range of 65535 is 0.5% brightness; object id 3 is
0.005%. Both are essentially black on screen while holding exactly the right numbers. They are stored
this way on purpose: rescaling the depth to look nice would throw away the millimetres it exists to
carry, and recolouring the ids would destroy the identity the labels are keyed on.

To LOOK at a scene, open the preview/ folder beside this file. Same data, coloured, lossy, never read
back by anything; one flat directory for the whole dataset, so 900 images are one scroll rather than
300 folders:

  preview/<scene>_<view>_preview.png            rgb + labelled boxes | depth | noisy depth | instances
  preview/<scene>_<view>_depth_view.png         depth alone, colour-mapped, near = bright
  preview/<scene>_<view>_depth_noisy_view.png   drawn on the CLEAN depth's scale, so both compare
  preview/<scene>_<view>_instances_view.png     one distinct colour per object id

Every coloured panel is stamped with the range it was scaled to, so the colours always state their
own meaning. Regenerate them at any time with:

    python -m datagen preview --name <this dataset>
"""


def write_how_to_read(root: Path) -> Path:
    """Drop the note at a dataset root. Its own function because both producers owe it.

    ``build --preview`` renders the previews scene by scene and never calls
    :func:`preview_dataset`, so without a separate entry point a built dataset would carry viewable
    images and no explanation of why the files beside them are black.
    """
    path = Path(root) / "HOW_TO_READ.txt"
    path.write_text(_HOW_TO_READ, encoding="utf-8")
    return path


def preview_dataset(root: Path) -> PreviewReport:
    """Every written scene under ``root/scenes``, plus the note explaining the unviewable files."""
    scenes_dir = Path(root) / "scenes"
    if not scenes_dir.is_dir():
        logger.error("no scenes directory under %s; nothing to preview", root)
        return PreviewReport(0, 0, (f"no scenes directory under {root}",))
    write_how_to_read(Path(root))
    out_dir = Path(root) / "preview"
    scenes = written = 0
    skipped: list[str] = []
    for scene_dir in sorted(scenes_dir.iterdir()):
        if not (scene_dir / "scene.json").exists():
            continue
        scenes += 1
        count, reasons = preview_scene(scene_dir, out_dir)
        written += count
        skipped.extend(reasons)
    # Once for the dataset. A skipped view is a view whose picture is missing or will not decode,
    # which is the exact defect this module exists to make visible, so it is a warning rather than a
    # count nobody reads, and the reasons come along because "12 skipped" cannot be acted on.

    logger.info("%d preview(s) from %d scene(s) -> %s", written, scenes, out_dir)
    if skipped:
        logger.warning("%d view(s) had no viewable image: %s", len(skipped), "; ".join(skipped[:5]))
    return PreviewReport(scenes, written, tuple(skipped))
