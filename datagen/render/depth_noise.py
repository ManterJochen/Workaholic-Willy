"""Making perfect depth imperfect: the largest sim2real gap left after light transport.

Isaac's depth is exact. A real stereo depth camera is not, and its error is not random: it grows with
the square of distance, it fails at depth discontinuities, it drops out on specular and dark
surfaces, and it quantises to disparity steps. A model trained on exact depth learns none of that and
falls over on the first real frame.

So every view is written twice: the exact depth, which is ground truth and the source of the 3-D
labels, and a noised copy shaped like a real sensor. Training uses the noisy one and the labels stay
exact, so the cost of the noise is measurable rather than something a robot cell discovers.

Scope. These are the published characteristics of the d435 class (axial noise proportional to z^2,
edge dropout, subpixel disparity quantisation), implemented from the equations and not calibrated
against a physical device. It is an approximation of how the error behaves, not a claim about how
large it is on any particular unit. Calibrating it needs a real camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DepthSensorModel", "REALSENSE_D435", "SENSOR_MODELS", "inject_depth_noise"]


@dataclass(frozen=True, slots=True)
class DepthSensorModel:
    """Error characteristics of a stereo depth camera. Distances in millimetres."""

    name: str
    #: Axial (along-ray) noise: sigma_z = coeff * z^2. The dominant term for stereo, and the reason a
    #: bin picked at 400 mm is a different problem from a pallet scanned at 2 m.
    axial_noise_coeff_per_mm: float
    #: Depth discontinuities are where stereo matching fails: pixels within this many px of an edge in
    #: the depth image are dropped. This is what makes real object boundaries ragged.
    edge_dropout_px: int
    #: Fraction of remaining valid pixels dropped at random (dark, specular, and unmatched regions).
    hole_fraction: float
    #: Disparity quantisation: real depth arrives in disparity steps, so depth resolution degrades with
    #: distance in visible bands rather than smoothly.
    baseline_mm: float
    focal_px: float
    subpixel_steps: float
    #: Anything outside the working range is invalid (0), like a real device reports.
    min_range_mm: float
    max_range_mm: float


#: Intel RealSense d435 class: the camera family this repo's RGB-D driver targets.
REALSENSE_D435 = DepthSensorModel(
    name="realsense_d435",
    axial_noise_coeff_per_mm=1.0e-6,   # ~= 0.16 mm at 400 mm, ~= 4 mm at 2 m
    edge_dropout_px=2,
    hole_fraction=0.01,
    baseline_mm=50.0,
    focal_px=640.0,
    subpixel_steps=8.0,
    min_range_mm=105.0,
    max_range_mm=10000.0,
)


def _edge_mask(depth_mm: np.ndarray, width_px: int) -> np.ndarray:
    """Pixels within ``width_px`` of a depth discontinuity, by dilating a gradient threshold.

    A plain box dilation rather than a morphology call: this module stays dependency-free numpy so it
    runs anywhere, and a separable max filter is exactly that.
    """
    if width_px <= 0:
        return np.zeros_like(depth_mm, dtype=bool)
    gy, gx = np.gradient(depth_mm)
    # A discontinuity is a jump large relative to the local depth: 2% of the distance, so the same
    # physical step counts at 400 mm and at 2 m.
    edges = (np.hypot(gx, gy) > 0.02 * np.maximum(depth_mm, 1.0))
    dilated = edges.copy()
    for _ in range(width_px):
        padded = np.pad(dilated, 1, mode="edge")
        dilated = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:] | dilated
        )
    return dilated


def inject_depth_noise(
    depth_mm: np.ndarray, model: DepthSensorModel, rng: np.random.Generator,
) -> np.ndarray:
    """Return a noised copy of ``depth_mm`` (mm, 0 = invalid). The input is never modified.

    Order matters and follows the physics: quantise to disparity first, because that is what the
    sensor measures, then add axial noise, then remove what the sensor could not have matched (edges,
    random holes, out-of-range). Adding noise after dropping would put noise on pixels the sensor
    never reported.
    """
    depth = np.asarray(depth_mm, dtype=np.float64).copy()
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return np.zeros_like(depth)

    # 1) disparity quantisation: d = f*B/z, rounded to subpixel steps, then back to depth
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = model.focal_px * model.baseline_mm / np.where(valid, depth, np.inf)
        quantised = np.round(disparity * model.subpixel_steps) / model.subpixel_steps
        depth = np.where(
            valid & (quantised > 0.0),
            model.focal_px * model.baseline_mm / np.where(quantised > 0.0, quantised, np.inf),
            depth,
        )

    # 2) axial noise, growing with the square of distance
    sigma = model.axial_noise_coeff_per_mm * depth ** 2
    depth = np.where(valid, depth + rng.normal(0.0, 1.0, depth.shape) * sigma, depth)

    # 3) what the sensor could not match: edges, random holes, out of range
    invalid = _edge_mask(np.where(valid, depth, 0.0), model.edge_dropout_px)
    if model.hole_fraction > 0.0:
        invalid |= rng.random(depth.shape) < model.hole_fraction
    invalid |= (depth < model.min_range_mm) | (depth > model.max_range_mm)
    depth[invalid | ~valid] = 0.0
    return depth


#: The registry `render.depth_noise.sensor` selects from. `isaac.py` looks the sensor up here rather
#: than naming a model directly, so the field a caller sets is the field the renderer reads. One
#: legal value today; a second sensor is then a config change rather than a code change.
SENSOR_MODELS: "dict[str, DepthSensorModel]" = {"realsense_d435": REALSENSE_D435}
