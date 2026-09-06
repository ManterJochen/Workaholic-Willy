"""Drive the real camera adapter over datagen scenes, against a fake streamer and fake models.

``RealSenseVisionPerceptionSource`` is the adapter every real-hardware pick goes through. One of its
two robustness fixes runs on every frame, top-referenced depth over holes; the other,
mask-fill-to-box, is a selectable policy whose default is ``NONE``, so a mask is filled only when a
caller asks for it. Neither is a proven answer on real d435 depth, and nothing here can make one.

What the corpus does offer are the two properties that make a replay worth running:

  * depth shaped like a real sensor. Every datagen view ships a noised copy alongside the exact one:
    axial noise proportional to z^2, edge dropout at depth discontinuities, random holes, disparity
    quantisation, range clipping to 0 (``datagen/render/depth_noise.py``, ``REALSENSE_D435``);
  * masks a real detector produced. ``predict-masks`` runs GroundingDINO + SAM2 over every view and
    writes them next to the ground truth.

That exercises everything above the sensor, which is where the code is. It does not validate the
sensor itself: noise magnitude, specular and dark dropout and auto-exposure are physical.

Why this lives in datagen: ``datagen`` imports ``src``; ``src`` never imports ``datagen``. A replay
harness under ``src/`` would invert that dependency and make the robot library depend on the data
generator, the same reason ``datagen/rl/perception.py`` lives here.

One thing this cannot exercise faithfully. When the axis-aligned policy is selected,
``_maybe_fill_to_detection_box`` compares the mask against the detector's box, and ``predict-masks``
does not store boxes. The box here is derived from the predicted mask's own extent, so the "SAM2
truncated one end and GroundingDINO's box did not" case cannot occur. The other half is measurable
exactly: the rule is an area ratio, so it also fires on a mask that is complete but simply not
rectangular, which is what the probe counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import CAMERA_PROBE_LOG_FILE, DATAGEN_LOG_DIR

__all__ = ["DatagenRGBDStreamer", "ReplayBackend", "ViewProbe", "probe_dataset"]

logger = create_logger("datagen.eval.camera_replay", CAMERA_PROBE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

_MASK_OFFSET = 1
_RENDERED = {"ok", "rendered"}


# ------------------------------------------------------------------- the streamer shim
@dataclass
class DatagenRGBDStreamer:
    """One datagen view, presented the way ``RealSenseRGBDStreamer`` presents a frame.

    ``.color`` is BGR uint8 and ``.depth`` uint16 millimetres with 0 for a hole: the contract the
    adapter reads, so the adapter cannot tell this from the real streamer.
    """

    scene_dir: Path
    view_name: str
    intrinsics: np.ndarray
    depth_source: str = "noisy"
    grabs: int = 0

    def grab(self) -> Any:
        from PIL import Image

        self.grabs += 1
        suffix = "depth_noisy" if self.depth_source == "noisy" else "depth"
        depth = np.asarray(Image.open(self.scene_dir / f"{self.view_name}_{suffix}.png"))
        rgb = np.asarray(Image.open(self.scene_dir / f"{self.view_name}_rgb.png").convert("RGB"))
        bgr = np.ascontiguousarray(rgb[..., ::-1])           # the adapter takes OpenCV BGR
        return _RGBD(color=bgr, depth=depth.astype(np.uint16))

    def get_intrinsics(self) -> np.ndarray:
        return np.asarray(self.intrinsics, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _RGBD:
    color: np.ndarray
    depth: np.ndarray


# --------------------------------------------------------------- the detector/segmenter shim
@dataclass(frozen=True, slots=True)
class _Det:
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _Seg:
    """``acquire()`` calls ``dataclasses.replace(seg, ...)``; a frozen dataclass supports it."""

    mask: np.ndarray
    label: str


@dataclass(frozen=True, slots=True)
class _Perceived:
    detection: _Det
    segmentation: _Seg


@dataclass(frozen=True, slots=True)
class ReplayBackend:
    """Replays the stored GroundingDINO + SAM2 masks as the backend seam the adapter composes.

    The box is the mask's own axis-aligned extent (see the module docstring's caveat).
    """

    scene_dir: Path
    view_name: str
    objects: tuple[dict[str, Any], ...]
    mask_source: str = "pred"

    def perceive(self, image_bgr: np.ndarray, prompt: str) -> tuple[_Perceived, ...]:
        from PIL import Image

        name = "pred_instances" if self.mask_source == "pred" else "instances"
        instances = np.asarray(Image.open(self.scene_dir / f"{self.view_name}_{name}.png"))
        out: list[_Perceived] = []
        for entry in self.objects:
            mask = instances == (int(entry["instance_id"]) + _MASK_OFFSET)
            if not mask.any():
                continue                       # the detector missed it: absent, not empty
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            label = str(entry.get("asset_id") or f"instance_{entry['instance_id']}")
            out.append(_Perceived(_Det(box), _Seg(mask.astype(np.uint8), label)))
        return tuple(out)


# ----------------------------------------------------------------------- the measurement
@dataclass(frozen=True, slots=True)
class ViewProbe:
    """What one view says about the adapter."""

    scene_id: str
    family: str
    view: str
    objects_in_manifest: int
    objects_perceived: int
    segmentations_out: int
    #: Fraction of pixels inside each predicted mask that the noisy depth dropped to 0.
    hole_fractions: tuple[float, ...]
    #: Masks whose depth is entirely holes: where the adapter leaves raw depth on purpose.
    all_hole_masks: int
    #: Masks the 0.8 area rule would replace with their bounding box.
    fill_fired: int
    #: Centroid shift that replacement causes, in millimetres at the mask's own depth.
    fill_shift_mm: tuple[float, ...]
    error: str = ""


def _centroid_shift_mm(mask: np.ndarray, box, depth_mm: np.ndarray, K: np.ndarray) -> float:
    """How far the grasp anchor moves if this mask is replaced by its box, in mm."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    x0, y0, x1, y1 = (int(round(float(c))) for c in box)
    cx_m, cy_m = float(xs.mean()), float(ys.mean())
    cx_b, cy_b = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
    vals = depth_mm[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    z = float(np.median(vals))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    return float(np.hypot((cx_b - cx_m) * z / fx, (cy_b - cy_m) * z / fy))


def _probe_view(scene_dir: Path, view: dict[str, Any], scene_id: str, family: str,
                *, depth_source: str, mask_source: str) -> ViewProbe:
    from src.robot.perception import RealSenseVisionPerceptionSource

    name = str(view["name"])
    K = np.asarray(view["intrinsics"], dtype=np.float64)
    objects = tuple(view.get("objects", ()))
    streamer = DatagenRGBDStreamer(scene_dir, name, K, depth_source=depth_source)
    backend = ReplayBackend(scene_dir, name, objects, mask_source=mask_source)

    holes: list[float] = []
    shifts: list[float] = []
    all_holes = fired = 0
    try:
        perceived = backend.perceive(np.zeros((1, 1, 3), np.uint8), "")
        raw_depth = streamer.grab().depth.astype(np.float64)
        for obj in perceived:
            mask = np.asarray(obj.segmentation.mask).astype(bool)
            n = int(mask.sum())
            if n:
                zeros = int((raw_depth[mask] == 0).sum())
                holes.append(zeros / n)
                if zeros == n:
                    all_holes += 1
                x0, y0, x1, y1 = (int(round(float(c))) for c in obj.detection.box)
                box_area = max(1, (x1 - x0) * (y1 - y0))
                if n < 0.8 * box_area:
                    fired += 1
                    shifts.append(_centroid_shift_mm(mask, obj.detection.box, raw_depth, K))

        # The real adapter, unmodified, on top of the two shims.
        source = RealSenseVisionPerceptionSource(
            streamer=streamer, backend=backend, prompt="", warmup_grabs=0,
        )
        frame = source.acquire()
        out = len(frame.segmentations)
        error = ""
    except Exception as exc:  # a failure here is the finding
        # Error per failing view, not per view: the adapter raising is the whole point of the probe,
        # it is returned rather than raised, and a run that fails on every view has an exit code but
        # no record of which frame first broke it.
        logger.error("%s/%s: the adapter raised; %s: %s", scene_id, name, type(exc).__name__, exc)
        out, error = 0, f"{type(exc).__name__}: {exc}"
        perceived = ()

    return ViewProbe(
        scene_id=scene_id, family=family, view=name,
        objects_in_manifest=len(objects), objects_perceived=len(perceived),
        segmentations_out=out, hole_fractions=tuple(holes), all_hole_masks=all_holes,
        fill_fired=fired, fill_shift_mm=tuple(shifts), error=error,
    )


def probe_dataset(dataset_dir: Path | str, *, limit: int | None = None,
                  depth_source: str = "noisy", mask_source: str = "pred") -> list[ViewProbe]:
    """Run the real adapter over every rendered view of every scene."""
    scenes = sorted((Path(dataset_dir) / "scenes").iterdir())
    if limit is not None:
        scenes = scenes[:limit]
    logger.info("probing %s: %d scene(s), depth=%s, masks=%s",
                dataset_dir, len(scenes), depth_source, mask_source)
    probes: list[ViewProbe] = []
    for scene_dir in scenes:
        manifest = scene_dir / "scene.json"
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        spec = payload.get("spec", {})
        scene_id = str(spec.get("scene_id") or scene_dir.name)
        family = str(spec.get("family") or "")
        for view in payload.get("views", ()):
            if str(view.get("outcome")) not in _RENDERED:
                continue
            probes.append(_probe_view(scene_dir, view, scene_id, family,
                                      depth_source=depth_source, mask_source=mask_source))
    # The two numbers the probe exists for, once. `render_report` prints the full distribution; what
    # is worth keeping is how many views the real adapter survived and how often the fill rule fired.
    errors = sum(1 for probe in probes if probe.error)
    fired = sum(probe.fill_fired for probe in probes)
    masks = sum(len(probe.hole_fractions) for probe in probes)
    logger.info("probed %d view(s): %d adapter error(s), mask-fill would fire on %d of %d mask(s)",
                len(probes), errors, fired, masks)
    return probes


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def render_report(probes: list[ViewProbe]) -> str:
    """The probe as text. A section with nothing measured says so rather than printing a zero."""
    lines: list[str] = []
    errs = [p for p in probes if p.error]
    holes = [h for p in probes for h in p.hole_fractions]
    shifts = [s for p in probes for s in p.fill_shift_mm]
    masks = len(holes)
    fired = sum(p.fill_fired for p in probes)
    all_holes = sum(p.all_hole_masks for p in probes)
    missed = sum(p.objects_in_manifest - p.objects_perceived for p in probes)
    in_manifest = sum(p.objects_in_manifest for p in probes)

    lines.append(f"views probed          {len(probes)}")
    lines.append(f"adapter errors        {len(errs)}"
                 + ("" if not errs else f"   <-- {errs[0].scene_id}/{errs[0].view}: {errs[0].error}"))
    lines.append(f"objects in manifest   {in_manifest}")
    lines.append(f"  not detected        {missed}"
                 + (f"   ({missed / in_manifest:.1%})" if in_manifest else ""))
    lines.append(f"  masks measured      {masks}")
    lines.append("")
    lines.append("DEPTH HOLES inside a predicted mask (sensor-shaped depth)")
    if holes:
        lines.append(f"  median              {np.median(holes):.1%}")
        lines.append(f"  p95                 {_pct(holes, 95):.1%}")
        lines.append(f"  max                 {max(holes):.1%}")
        lines.append(f"  ALL-hole masks      {all_holes}"
                     + (f"   ({all_holes / masks:.2%}; the adapter leaves raw depth here, on purpose)"
                        if masks else ""))
    else:
        lines.append("  no masks measured")
    lines.append("")
    lines.append("MASK-FILL-TO-BOX, the 0.8 area rule")
    if masks:
        lines.append(f"  would fire on       {fired} of {masks} masks   ({fired / masks:.1%})")
    if shifts:
        lines.append(f"  centroid shift      median {np.median(shifts):.1f} mm"
                     f"   p95 {_pct(shifts, 95):.1f} mm   max {max(shifts):.1f} mm")
    else:
        lines.append("  centroid shift      not measured (never fired)")
    return "\n".join(lines)
