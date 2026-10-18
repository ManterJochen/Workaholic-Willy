"""The stereo depth this repository matches, against the D415's own depth of the same infrared pair.

    python scripts/stereo/measure_on_the_d415_ir_pair.py --views <folder of views files>
    python scripts/stereo/measure_on_the_d415_ir_pair.py --views <folder> --every 4 --alternatives
    python scripts/stereo/measure_on_the_d415_ir_pair.py --views <folder> --alternatives --json d415_stereo.json

A pick that a camera recorded for research (``realsense.record_for_research``) keeps, per look, both infrared images of
the D415, which the device hands out rectified, the depth its chip matched on them before any filter ran, in the left
infrared camera's frame, the two lenses and the extrinsics between them, and every part's mask
(``src/robot/execution/record_views.py``). So every look is a calibrated stereo pair with a reference depth on the
very pixels this repository's matcher reads: the pair goes through ``StereoDepth`` exactly as a webcam pair does in
the pick path (``StereoDepthStreamer``), its calibration built from the device's own lenses by ``cv.stereoRectify``,
and the depth it gives is set against the chip's.

The chip is a reference, not the truth: it matches the same images with its own census matcher and ramps its own depth
over a part's edge. What it does tell apart is a depth read about where it reads one from a depth read somewhere else,
and that is what the owner's rule is about: the matcher never invents depth. The pair is lit by the D415's own
projector wherever the camera recorded with it on, as the owner's did (``emitter_enabled`` in the views file's
``camera`` record), and its dots give every surface something to match: a webcam pair has no projector, and reads less
of a plain surface than this. Reported per configuration, over every look read (``--every`` reads every n-th):

* **valid**: the share of the frame that holds a depth; the chip's own beside it. The guard holds a fixed camera with
  less than half its frame measured for blind (``live_world.min_depth_coverage``), and **blind** counts the looks
  that would be.
* **cover**: the share of the pixels the chip measured that hold a depth too.
* **wrong far** and **wrong near**: of the pixels both measured, the share read more than max(10 mm, 3 %) beyond the
  chip's depth, which hides what stands in front, or nearer, which is a surface where none stands.
* **parts** and **mat**: the median and the 90th percentile of |depth - chip| on the parts (each mask eroded 4 px)
  and on the mat they lie on (the hull of every mask grown 40 px, without 8 px about each part).
* **flying**: of the pixels within 4 px of a part's edge that hold a depth, the share whose depth lies more than
  max(5 mm, 1 %) from every depth the chip measured within 2 px of it, a depth no surface there stands at.
* **ms**: the median time of one frame, the pair rectified and matched, on this machine's threads.

``--alternatives`` runs the matcher once more per row below, the measured settings with one thing changed, and the
search of the block that left with ``stereomatcher.yaml``; ``nearer than the search`` asks what the camera makes of a
surface nearer than it searches, with a search of 64 disparities that reads no nearer than 778 mm, in front of parts
at 640 to 780 mm.
Nothing is written but the ``--json`` file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calibration.stereo.config import CalibrationResult  # noqa: E402
from src.calibration.stereo.sub_modules.depth import NEAR_LIMIT_MM, StereoDepth, working_depth_range_mm  # noqa: E402
from src.calibration.stereo.sub_modules.disparity import (  # noqa: E402
    MEASURED_SETTINGS,
    DisparityComputer,
    DisparitySearch,
    MatcherSettings,
)
from src.calibration.stereo.sub_modules.rectifier import StereoRectifier  # noqa: E402

#: The owner's cell's workspace box, x, y and z from and to, BASE millimetres (its robot.yaml, 2026-10-15): its
#: diagonal is where a search ending at the workspace would end for the wrist camera these views were taken with.
_OWNERS_WORKSPACE_MM = ((-470.0, 400.0), (-900.0, -200.0), (16.0, 800.0))
#: Where a depth counts as read wrong: further from the chip's than this many millimetres and this share of it.
_WRONG_MM, _WRONG_SHARE = 10.0, 0.03
#: Where an edge pixel's depth counts as flying: further than this from every chip depth within 2 px.
_FLYING_MM, _FLYING_SHARE, _FLYING_REACH_PX = 5.0, 0.01, 2
#: The search of ``nearer than the search``: 64 disparities, which read no nearer than 778 mm at the D415's numbers.
_SHORT_SEARCH = DisparitySearch(0, 64)


@dataclass(frozen=True, slots=True)
class Look:
    """One look of a views file: the pair, the chip's depth of it in millimetres, its calibration and its parts."""

    name: str
    left: np.ndarray
    right: np.ndarray
    chip_mm: np.ndarray
    calibration: CalibrationResult
    parts: np.ndarray
    edges: np.ndarray
    mat: np.ndarray


@dataclass(slots=True)
class Tally:
    """What one configuration read over every look."""

    looks: int = 0
    blind: int = 0
    pixels: int = 0
    valid: int = 0
    chip_valid: int = 0
    covered: int = 0
    judged: int = 0
    wrong_far: int = 0
    wrong_near: int = 0
    edge_judged: int = 0
    flying: int = 0
    seconds: list[float] = field(default_factory=list)
    parts_err: list[np.ndarray] = field(default_factory=list)
    mat_err: list[np.ndarray] = field(default_factory=list)
    nearer_px: int = 0
    nearer_read: int = 0
    nearer_wrong: int = 0

    def row(self) -> dict[str, Any]:
        parts = np.concatenate(self.parts_err) if self.parts_err else np.zeros(0)
        mat = np.concatenate(self.mat_err) if self.mat_err else np.zeros(0)

        def pct(values: np.ndarray, q: float) -> float:
            return float(np.percentile(values, q)) if values.size else float("nan")

        out = {
            "looks": self.looks, "blind": self.blind, "valid": self.valid / max(1, self.pixels),
            "chip_valid": self.chip_valid / max(1, self.pixels),
            "cover": self.covered / max(1, self.chip_valid), "wrong_far": self.wrong_far / max(1, self.judged),
            "wrong_near": self.wrong_near / max(1, self.judged), "parts_median_mm": pct(parts, 50),
            "parts_p90_mm": pct(parts, 90), "mat_median_mm": pct(mat, 50), "mat_p90_mm": pct(mat, 90),
            "flying": self.flying / max(1, self.edge_judged),
            "ms": 1000.0 * float(np.median(self.seconds)) if self.seconds else float("nan"),
        }
        if self.nearer_px:
            out.update(nearer_px=self.nearer_px, nearer_read=self.nearer_read / self.nearer_px,
                       nearer_wrong=self.nearer_wrong / self.nearer_px)
        return out


def _calibration(lenses: np.ndarray, distortion: np.ndarray, extrinsics: np.ndarray, size: tuple[int, int]
                 ) -> CalibrationResult:
    """The pair's stereo calibration from the device's lenses: rectified by ``cv.stereoRectify``, as the calibrator
    does, the right camera placed by the left to right extrinsics."""
    left_to_right = np.asarray(extrinsics[1], dtype=np.float64)
    rotation, translation = left_to_right[:3, :3], left_to_right[:3, 3]
    rect_l, rect_r, proj_l, proj_r, q, _, _ = cv.stereoRectify(
        lenses[0], distortion[0], lenses[1], distortion[1], size, rotation, translation, alpha=0.0)
    return CalibrationResult(camL=lenses[0], distL=distortion[0], rectL=rect_l, projL=proj_l, camR=lenses[1],
                             distR=distortion[1], rectR=rect_r, projR=proj_r, Q=q, frame_size=size)


def _masks_in_the_left_ir(data: Any, index: int, chip_mm: np.ndarray) -> np.ndarray:
    """Every part's mask, recorded in the colour frame, carried into the left infrared camera's through the chip's
    depth, a hole read at the median about it."""
    height, width = chip_mm.shape
    packed = data[f"seg_masks_{index}"]
    if packed.shape[0] == 0:
        return np.zeros((0, height, width), dtype=bool)
    colour_masks = np.unpackbits(packed, axis=1, count=height * width).reshape(-1, height, width).astype(bool)
    lens = data[f"ir_intrinsics_{index}"][0]
    to_colour = data[f"ir_extrinsics_{index}"][0]
    colour_lens = data[f"intrinsics_{index}"]
    depth = chip_mm.astype(np.float32)
    filled = cv.medianBlur(depth, 5)
    reference = float(np.median(depth[depth > 0])) if np.any(depth > 0) else 700.0
    depth = np.where(depth > 0, depth, np.where(filled > 0, filled, reference)).astype(np.float64)
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([(us - lens[0, 2]) / lens[0, 0] * depth, (vs - lens[1, 2]) / lens[1, 1] * depth, depth], -1)
    points = points @ to_colour[:3, :3].T + to_colour[:3, 3]
    u = np.rint(colour_lens[0, 0] * points[..., 0] / points[..., 2] + colour_lens[0, 2]).astype(np.int64)
    v = np.rint(colour_lens[1, 1] * points[..., 1] / points[..., 2] + colour_lens[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return colour_masks[:, np.clip(v, 0, height - 1), np.clip(u, 0, width - 1)] & inside[None]


def _regions(masks: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The parts' insides, the band within 4 px of their edges, and the mat about them."""
    kernel = np.ones((3, 3), np.uint8)
    parts = np.zeros(shape, np.uint8)
    for mask in masks:
        parts[mask] = 1
    inside = cv.erode(parts, kernel, iterations=4) > 0
    edges = (cv.dilate(parts, kernel, iterations=4) > 0) & ~inside
    mat = np.zeros(shape, dtype=bool)
    if parts.any():
        hull = np.zeros(shape, np.uint8)
        cv.fillConvexPoly(hull, cv.convexHull(cv.findNonZero(parts)), 1)
        mat = (cv.dilate(hull, kernel, iterations=40) > 0) & ~(cv.dilate(parts, kernel, iterations=8) > 0)
    return inside, edges, mat


def looks(folder: Path, every: int) -> Iterator[Look]:
    """Every look with an infrared pair in the views files of ``folder``, every ``every``-th of them."""
    seen = 0
    calibrations: dict[bytes, CalibrationResult] = {}
    for path in sorted(folder.glob("*.npz")):
        with zipfile.ZipFile(path) as archive:
            names = {name[:-4] for name in archive.namelist()}
        indices = sorted(int(m.group(1)) for name in names if (m := re.fullmatch(r"ir_left_(\d+)", name)))
        if not indices:
            continue
        with np.load(path, allow_pickle=False) as data:
            units_mm = float(data["camera_depth_units_m"]) * 1000.0 if "camera_depth_units_m" in names else 1.0
            for index in indices:
                seen += 1
                if (seen - 1) % every:
                    continue
                left, right = data[f"ir_left_{index}"], data[f"ir_right_{index}"]
                chip = data[f"raw_depth_{index}"].astype(np.float64) * units_mm
                chip[chip >= 60000.0] = 0.0  # the chip's own saturated reads: no depth
                key = b"".join(np.ascontiguousarray(data[f"ir_{what}_{index}"]).tobytes()
                               for what in ("intrinsics", "distortion", "extrinsics"))
                size = (int(left.shape[1]), int(left.shape[0]))
                if key not in calibrations:
                    calibrations[key] = _calibration(data[f"ir_intrinsics_{index}"], data[f"ir_distortion_{index}"],
                                                     data[f"ir_extrinsics_{index}"], size)
                masks = _masks_in_the_left_ir(data, index, data[f"raw_depth_{index}"])
                inside, edges, mat = _regions(masks, chip.shape)
                yield Look(f"{path.stem}#{index}", left, right, chip, calibrations[key], inside, edges, mat)


def _depth_over(computer: DisparityComputer, focal_baseline: float, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The depth of a rectified pair through ``computer``, millimetres, 0 where nothing was read: ``StereoDepth``'s
    before it is rounded to the ``uint16`` a frame holds, so the errors below are the matcher's own."""
    disparity, valid = computer.compute(left, right)
    depth = np.zeros(disparity.shape, dtype=np.float64)
    depth[valid] = focal_baseline / disparity[valid].astype(np.float64)
    depth[depth > 65535.0] = 0.0
    return depth


def _explained(depth: np.ndarray, chip: np.ndarray) -> np.ndarray:
    """Where a depth lies within max(5 mm, 1 %) of some depth the chip measured within 2 px of the pixel."""
    best = np.full(depth.shape, np.inf)
    reach = _FLYING_REACH_PX
    padded = np.pad(np.where(chip > 0, chip, np.nan), reach, constant_values=np.nan)
    height, width = depth.shape
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            shifted = padded[reach + dy:reach + dy + height, reach + dx:reach + dx + width]
            with np.errstate(invalid="ignore"):
                best = np.fmin(best, np.abs(depth - shifted))
    return best <= np.maximum(_FLYING_MM, _FLYING_SHARE * depth)


def _tally(tally: Tally, look: Look, depth: np.ndarray, seconds: float) -> None:
    chip = look.chip_mm
    valid, chip_valid = depth > 0, chip > 0
    both = valid & chip_valid
    error = depth - chip
    limit = np.maximum(_WRONG_MM, _WRONG_SHARE * chip)
    tally.looks += 1
    tally.blind += int(float(valid.mean()) < 0.5)
    tally.pixels += depth.size
    tally.valid += int(valid.sum())
    tally.chip_valid += int(chip_valid.sum())
    tally.covered += int(both.sum())
    tally.judged += int(both.sum())
    tally.wrong_far += int((both & (error > limit)).sum())
    tally.wrong_near += int((both & (error < -limit)).sum())
    tally.parts_err.append(np.abs(error[both & look.parts]))
    tally.mat_err.append(np.abs(error[both & look.mat]))
    edge = valid & look.edges
    tally.edge_judged += int(edge.sum())
    tally.flying += int((edge & ~_explained(depth, chip)).sum())
    tally.seconds.append(seconds)


def _configurations(alternatives: bool) -> dict[str, tuple[MatcherSettings, Any]]:
    """Each row: its settings, and its search (a range of depths, or a search of its own)."""
    rows: dict[str, tuple[MatcherSettings, Any]] = {"measured (the pick path)": (MEASURED_SETTINGS, None)}
    if not alternatives:
        return rows
    replace = dataclasses.replace
    diagonal = math.dist(*zip(*_OWNERS_WORKSPACE_MM))
    rows.update({
        "full matcher, not 3-way": (replace(MEASURED_SETTINGS, three_way=False), None),
        "P1, P2 x3 (the block before)": (replace(MEASURED_SETTINGS, smoothness_p1=24, smoothness_p2=96), None),
        "block 3": (replace(MEASURED_SETTINGS, block_size=3), None),
        "block 7": (replace(MEASURED_SETTINGS, block_size=7), None),
        "uniqueness 10": (replace(MEASURED_SETTINGS, uniqueness_ratio=10), None),
        "uniqueness 20": (replace(MEASURED_SETTINGS, uniqueness_ratio=20), None),
        "no left padding": (replace(MEASURED_SETTINGS, pad_left=False), None),
        "no left-right check": (replace(MEASURED_SETTINGS, left_right_px=None), None),
        "flat blocks kept": (replace(MEASURED_SETTINGS, refuse_flat_blocks=False), None),
        "no hole or step margin": (replace(MEASURED_SETTINGS, hole_margin_px=0, step_margin_px=0), None),
        "hole and step margin 3": (replace(MEASURED_SETTINGS, hole_margin_px=3, step_margin_px=3), None),
        "search ends at the workspace diagonal": (MEASURED_SETTINGS, (NEAR_LIMIT_MM, diagonal)),
        "stereomatcher.yaml, without its WLS": (MatcherSettings(
            block_size=7, smoothness_p1=24, smoothness_p2=96, uniqueness_ratio=10, speckle_window_size=50,
            speckle_range=1, disp12_max_diff=1, three_way=True, pad_left=False, left_right_px=None, near_end_px=0,
            refuse_flat_blocks=False, hole_margin_px=0, step_margin_px=0), DisparitySearch(0, 320)),
        "nearer than the search (64 disparities)": (MEASURED_SETTINGS, _SHORT_SEARCH),
    })
    return rows


def measure(folder: Path, *, every: int, alternatives: bool) -> dict[str, dict[str, Any]]:
    """Every configuration's row over the looks of ``folder`` (the module docstring)."""
    rows = _configurations(alternatives)
    tallies = {name: Tally() for name in rows}
    computers: dict[tuple[str, bytes], tuple[DisparityComputer, float]] = {}
    rectifiers: dict[bytes, StereoRectifier] = {}
    for count, look in enumerate(looks(folder, every), start=1):
        lens_key = look.calibration.projR.tobytes() + look.calibration.projL.tobytes()
        rectifier = rectifiers.setdefault(lens_key, StereoRectifier(look.calibration))
        for name, (settings, search) in rows.items():
            key = (name, lens_key)
            if key not in computers:
                if isinstance(search, DisparitySearch):
                    probe = StereoDepth(look.calibration, working_depth_range_mm(), settings=settings)
                    computers[key] = (DisparityComputer(search, settings), probe.focal_px * probe.baseline_mm)
                else:
                    span = search if search is not None else working_depth_range_mm()
                    depth = StereoDepth(look.calibration, span, settings=settings)
                    computers[key] = (DisparityComputer(depth.search, settings), depth.focal_px * depth.baseline_mm)
            computer, focal_baseline = computers[key]
            started = time.perf_counter()
            rect_left, rect_right = rectifier.rectify(look.left, look.right)
            depth_mm = _depth_over(computer, focal_baseline, rect_left, rect_right)
            seconds = time.perf_counter() - started
            tally = tallies[name]
            _tally(tally, look, depth_mm, seconds)
            if isinstance(search, DisparitySearch) and search is _SHORT_SEARCH:
                nearest = focal_baseline / float(search.max_disparity - settings.near_end_px)
                nearer = (look.chip_mm > 0) & (look.chip_mm < nearest)
                read = nearer & (depth_mm > 0)
                limit = np.maximum(_WRONG_MM, _WRONG_SHARE * look.chip_mm)
                tally.nearer_px += int(nearer.sum())
                tally.nearer_read += int(read.sum())
                tally.nearer_wrong += int((read & (np.abs(depth_mm - look.chip_mm) > limit)).sum())
        if count % 25 == 0:
            print(f"  {count} looks read", flush=True)
    return {name: tally.row() for name, tally in tallies.items()}


def _render(rows: dict[str, dict[str, Any]], threads: int) -> str:
    head = (f"{'configuration':40s} {'valid':>6s} {'blind':>5s} {'cover':>6s} {'wr.far':>7s} {'wr.near':>7s} "
            f"{'parts mm':>11s} {'mat mm':>11s} {'flying':>7s} {'ms':>6s}")
    lines = [head, "-" * len(head)]
    for name, row in rows.items():
        if "nearer_px" in row:
            lines.append(f"{name:40s} of {row['nearer_px']} pixels nearer than the search, {100 * row['nearer_read']:.1f}% "
                         f"hold a depth and {100 * row['nearer_wrong']:.1f}% a wrong one ({row['ms']:.0f} ms)")
            continue
        lines.append(
            f"{name:40s} {100 * row['valid']:5.1f}% {row['blind']:5d} {100 * row['cover']:5.1f}% "
            f"{100 * row['wrong_far']:6.2f}% {100 * row['wrong_near']:6.2f}% {row['parts_median_mm']:4.1f} / "
            f"{row['parts_p90_mm']:4.1f} {row['mat_median_mm']:4.1f} / {row['mat_p90_mm']:4.1f} "
            f"{100 * row['flying']:6.2f}% {row['ms']:6.0f}")
    first = next(iter(rows.values()))
    lines.append(f"\n{first['looks']} looks; the chip measured {100 * first['chip_valid']:.1f}% of the frame; "
                 f"{threads} OpenCV threads. parts and mat: median / p90 of |depth - chip|.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_on_the_d415_ir_pair.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--views", required=True, type=Path, help="a folder of views files recorded for research")
    parser.add_argument("--every", type=int, default=1, help="read every n-th look (default: every look)")
    parser.add_argument("--alternatives", action="store_true", help="measure the alternatives as well")
    parser.add_argument("--json", type=Path, default=None, help="write the rows to this file")
    args = parser.parse_args(argv)
    if not args.views.is_dir():
        print(f"no folder at {args.views}")
        return 2
    rows = measure(args.views, every=max(1, int(args.every)), alternatives=bool(args.alternatives))
    if not rows or not next(iter(rows.values()))["looks"]:
        print(f"no look with an infrared pair in {args.views}: the views files of a rig that records for research "
              "hold them (realsense.record_for_research)")
        return 2
    print(_render(rows, cv.getNumThreads()))
    if args.json is not None:
        args.json.write_text(json.dumps({"views": str(args.views), "every": args.every, "threads": cv.getNumThreads(),
                                         "rows": rows}, indent=2), encoding="utf-8")
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
