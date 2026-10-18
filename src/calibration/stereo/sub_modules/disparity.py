"""Disparity of a rectified pair: OpenCV's semi-global matcher at settings measured once and fixed (the owner, 2026-10-17).

A stereo pair is a depth camera the pick path takes like an RGB-D one (``StereoDepth``), and the guard judges every
motion against that depth, so what this module hands out is held to one rule above all: it never INVENTS depth. A
pixel it cannot match is a hole, 0 to the RGB-D frame and unknown to every reader. Nothing fills one from its
neighbours (no WLS filter) or from earlier frames (no temporal average): a filled hole can hide an obstacle, and a depth
borrowed from a frame before the arm moved is a surface where nothing stands now. Both left with ``stereomatcher.yaml``
on 2026-10-17, and so did every other knob of that block: the search is derived (:func:`disparity_search`) and the rest
is the one :data:`MEASURED_SETTINGS`.

Measured on the D415's own infrared pair against the chip's depth of the same images, the 306 looks of the owner's
cell of 2026-10-15 that kept both (``scripts/stereo/measure_on_the_d415_ir_pair.py``, its table in
``src/camera/README.md``), each choice below against the same settings with that one thing changed. A pixel is read
wrong where both measured it and it lies more than max(10 mm, 3 %) from the chip, too far or too near. The pair was lit
by the D415's projector, whose dots give every surface something to match; a webcam pair has none and reads less of a
plain surface:

* ``MODE_SGBM_3WAY``: as close to the chip on the parts as the full matcher and closer on the mat (a 90th percentile of
  5.8 mm against 6.9), in a quarter of its time.
* ``P1 = 8 * block**2`` and ``P2 = 32 * block**2``, OpenCV's rule for one channel. The block before this one multiplied
  both by three channels on a grey image, which smooths three times as hard: four times as many pixels read too near
  (3.0 % against 0.7 %), and half again as many at a part's edge at a depth no surface there stands at.
* A block of 5 pixels, a uniqueness of 15 %, speckles up to 100 pixels within 2 disparity levels: of every setting
  measured, the fewest pixels read wrong that keep nine looks in ten holding a depth on half their frame, the share the
  guard asks of a fixed camera's (``live_world.min_depth_coverage``). The median look holds one on 54 %, and the 31
  under the half, nearly all a pick's first look, between 46 and 50 %. A block of 3 reads half as much of the frame
  (28 % against 56 %); one of 7 reads 64 % of it and leaves no look under the half, with half again as many pixels too
  near (1.1 % against 0.7 %), and is the measured way over it where a fixed pair turns out blind. Uniqueness 20 reads a
  sixth fewer pixels wrong and leaves 54 looks under it.
* The pair padded on its left by the search's width before it is matched. A matcher looks for each left pixel's match
  up to the search's width to its left, so without the padding the left band that wide holds no depth at all, a
  quarter of a 1280 pixel frame for a search of 336. Padded, a pixel whose match lies in the right image is matched
  there, and one whose match would lie in the padding is no match: 56 % of the frame read instead of 34 %, a tenth of a
  percent more of it wrong.
* A left-right check of its own, 1 pixel: the pair matched again flipped, right to left, and a pixel kept only where
  the two maps agree. OpenCV's ``disp12MaxDiff`` is on at 1 as well, and alone it lets half again as many pixels too
  far through (0.57 % against 0.36 %); the second match doubles the time.
* No disparity at the search's near end: what stands nearer than the search is matched to a disparity inside it, nearly
  half of the time (44 %) to one of its last two.
* No disparity where the left image's block is flat, every grey level in it the same: clipped dark or bright, or a
  surface with nothing on it to match. Whatever the matcher reads there it carried in from the neighbours, the
  background's depth over a box without texture standing on it. The D415's images are dark (a mean grey of 26), and 4 %
  of their blocks are flat: keeping them reads 6 points more of the frame and a fifth more of it wrong (1.3 % against
  1.1 %).
* Every pixel within 2 pixels of a hole, or of a step of more than one disparity between two neighbours, is a hole too.
  The pixels read wrong cluster there (an occluded rim, a fattened edge, the fringe of a surface without texture), and
  taking these out leaves a third of them (1.1 % against 3.3 %) for a sixth of the frame (56 % against 73 %).

``compute`` returns ``(disparity_float32, valid_mask_bool)``, the disparity in pixels and NaN where it is not valid;
that shape is a stable contract. OpenCV marks an unmatched pixel ``(minDisparity - 1) * 16``, which is not below zero
once ``minDisparity`` is above 1, so validity is read against the search, never as ``disparity > 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.helpers import validate_image_pair_shapes

__all__ = ["MEASURED_SETTINGS", "DisparityComputer", "DisparitySearch", "MatcherSettings", "disparity_search"]

#: OpenCV's matcher reads disparities in sixteenths of a pixel, and takes its search width in whole sixteens.
_SUBPIXEL: Final = 16

#: How far past each end of its depths a search reaches, in pixels: the matcher fits its sub-pixel parabola through a
#: disparity's two neighbours, so a surface standing exactly at an end is still read.
_END_MARGIN_PX: Final = 1


@dataclass(frozen=True, slots=True)
class DisparitySearch:
    """Which disparities the matcher tries: ``min_disparity`` up to, not including, :attr:`max_disparity`.

    Attributes:
        min_disparity (int): The smallest disparity tried, pixels, 0 or more; 0 reaches every depth out to infinity.
        num_disparities (int): How many are tried, a positive multiple of 16.
    """

    min_disparity: int
    num_disparities: int

    def __post_init__(self) -> None:
        if self.min_disparity < 0:
            raise StereoCalibrationError(f"min_disparity must be 0 or more, got {self.min_disparity}")
        if self.num_disparities <= 0 or self.num_disparities % _SUBPIXEL:
            raise StereoCalibrationError(
                f"num_disparities must be a positive multiple of {_SUBPIXEL}, got {self.num_disparities}")

    @property
    def max_disparity(self) -> int:
        """One past the largest disparity tried, pixels: also the width of the band a match is looked for in."""
        return self.min_disparity + self.num_disparities


def disparity_search(focal_px: float, baseline_mm: float, near_mm: float, far_mm: float = math.inf) -> DisparitySearch:
    """The disparities a pair of ``focal_px`` and ``baseline_mm`` tries to read every depth from ``near_mm`` to
    ``far_mm``.

    A surface at depth ``Z`` stands at disparity ``focal_px * baseline_mm / Z`` in a rectified pair, so the near end sets
    how wide the search is and the far end where it starts. Each end reaches a pixel further, the near end one more
    for the disparity there that is never kept (``MatcherSettings.near_end_px``), and the width is rounded up to the
    sixteen OpenCV takes it in; an infinite ``far_mm`` starts the search at 0. Nothing nearer than ``near_mm`` and
    nothing beyond ``far_mm`` is read right: the matcher gives such a surface a disparity inside the search, a wrong
    depth, so the ends are where the camera's view ends, not where the work does.

    Args:
        focal_px (float): The rectified pair's focal length, pixels.
        baseline_mm (float): The distance between the two cameras' centres, millimetres.
        near_mm (float): The nearest depth to read, millimetres, above 0.
        far_mm (float): The farthest depth to read, millimetres, beyond ``near_mm``; infinite reads to infinity
            (default: infinity).

    Returns:
        DisparitySearch: The search, ``min_disparity`` 0 or more and ``num_disparities`` a multiple of 16.

    Raises:
        StereoCalibrationError: A focal length or a baseline that is not finite and above 0, or a range that does not
            hold ``0 < near_mm < far_mm``.
    """
    for name, value in (("focal_px", focal_px), ("baseline_mm", baseline_mm)):
        if not (math.isfinite(value) and value > 0.0):
            raise StereoCalibrationError(f"{name} must be finite and above 0, got {value!r}")
    if not (math.isfinite(near_mm) and 0.0 < near_mm < far_mm):
        raise StereoCalibrationError(f"the depth range must hold 0 < near_mm < far_mm, got {near_mm!r} to {far_mm!r}")
    focal_baseline = float(focal_px) * float(baseline_mm)
    nearest = math.ceil(focal_baseline / float(near_mm)) + _END_MARGIN_PX
    farthest = 0 if math.isinf(far_mm) else max(0, math.floor(focal_baseline / float(far_mm)) - _END_MARGIN_PX)
    width = nearest + 2 - farthest  # the nearest kept, and the one place past it that is never kept
    return DisparitySearch(farthest, int(math.ceil(width / _SUBPIXEL)) * _SUBPIXEL)


@dataclass(frozen=True, slots=True)
class MatcherSettings:
    """How a pair is matched and what of it is kept: :data:`MEASURED_SETTINGS`, the one every pair is matched with.

    A value of its own so that the measurement which chose it runs its alternatives through this module's code, one
    field changed at a time; nothing in the pick path builds another.

    Attributes:
        block_size (int): The matching window's side, pixels, odd.
        smoothness_p1 (int): The penalty of a one-level disparity change between neighbours, per pixel of the window.
        smoothness_p2 (int): The penalty of a larger change, per pixel of the window.
        uniqueness_ratio (int): By how many percent the best match beats the second before it is kept.
        speckle_window_size (int): The largest patch of like disparity cleared as a speckle, pixels; 0 clears none.
        speckle_range (int): How far, in disparity levels, one speckle's disparity may vary.
        disp12_max_diff (int): OpenCV's own left-right check, pixels.
        three_way (bool): ``MODE_SGBM_3WAY``; ``False`` is the full matcher.
        pad_left (bool): Pad the pair on its left by the search's width before it is matched.
        left_right_px (float | None): The check against the pair matched flipped, pixels; ``None`` skips it.
        near_end_px (int): How many of the search's nearest disparities are never kept.
        refuse_flat_blocks (bool): No disparity where every grey level of the left image's block is the same.
        hole_margin_px (int): Every pixel within this many of a hole is a hole too.
        step_margin_px (int): Every pixel within this many of a step is a hole too.
        step_px (float): A step: two neighbours more than this many pixels of disparity apart.
    """

    block_size: int = 5
    smoothness_p1: int = 8
    smoothness_p2: int = 32
    uniqueness_ratio: int = 15
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    three_way: bool = True
    pad_left: bool = True
    left_right_px: Optional[float] = 1.0
    near_end_px: int = 1
    refuse_flat_blocks: bool = True
    hole_margin_px: int = 2
    step_margin_px: int = 2
    step_px: float = 1.0

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.block_size % 2 == 0:
            raise StereoCalibrationError(f"block_size must be a positive odd number, got {self.block_size}")
        if not 0 <= self.smoothness_p1 <= self.smoothness_p2:
            raise StereoCalibrationError("the smoothness penalties must hold 0 <= p1 <= p2")
        for name in ("uniqueness_ratio", "speckle_window_size", "speckle_range", "near_end_px", "hole_margin_px",
                     "step_margin_px"):
            if getattr(self, name) < 0:
                raise StereoCalibrationError(f"{name} must be 0 or more")
        if self.left_right_px is not None and not self.left_right_px >= 0.0:
            raise StereoCalibrationError("left_right_px must be 0 or more, or None")


#: The settings every pair is matched with, chosen on the D415's infrared pair (the module docstring).
MEASURED_SETTINGS: Final = MatcherSettings()


class DisparityComputer:
    """Disparity maps of rectified pairs over one search.

    Args:
        search (DisparitySearch): The disparities tried (:func:`disparity_search`).
        settings (MatcherSettings): How a pair is matched; the measured ones unless a measurement compares others
            (default: MEASURED_SETTINGS).
    """

    def __init__(self, search: DisparitySearch, settings: MatcherSettings = MEASURED_SETTINGS) -> None:
        self._search = search
        self._settings = settings
        block = int(settings.block_size)
        self._matcher = cv.StereoSGBM_create(  # type: ignore[attr-defined]
            minDisparity=int(search.min_disparity),
            numDisparities=int(search.num_disparities),
            blockSize=block,
            P1=int(settings.smoothness_p1) * block * block,
            P2=int(settings.smoothness_p2) * block * block,
            uniquenessRatio=int(settings.uniqueness_ratio),
            speckleWindowSize=int(settings.speckle_window_size),
            speckleRange=int(settings.speckle_range),
            disp12MaxDiff=int(settings.disp12_max_diff),
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY if settings.three_way else cv.STEREO_SGBM_MODE_SGBM,
        )

    @property
    def search(self) -> DisparitySearch:
        """The disparities this computer tries."""
        return self._search

    @property
    def settings(self) -> MatcherSettings:
        """How this computer matches a pair."""
        return self._settings

    def compute(self, rect_left_bgr: np.ndarray, rect_right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The disparity of a rectified pair, in the left image.

        Args:
            rect_left_bgr (np.ndarray): The rectified left image, BGR or grey, ``uint8``.
            rect_right_bgr (np.ndarray): The rectified right image, of the same size.

        Returns:
            tuple[np.ndarray, np.ndarray]: The disparity, ``float32`` pixels and NaN where it is not valid, and the
                ``bool`` mask of where it is valid.

        Raises:
            StereoCalibrationError: The two images have incompatible shapes, or one is neither grey nor BGR.
        """
        left = np.asarray(rect_left_bgr)
        right = np.asarray(rect_right_bgr)
        try:
            validate_image_pair_shapes(left, right)
        except Exception as exc:
            raise StereoCalibrationError("cannot compute disparity for incompatible image pair") from exc
        grey_left, grey_right = _grey(left), _grey(right)
        settings, search = self._settings, self._search
        pad = search.max_disparity if settings.pad_left else 0

        raw = self._match(grey_left, grey_right, pad)
        disparity = raw.astype(np.float32) * np.float32(1.0 / _SUBPIXEL)
        # OpenCV marks no match (minDisparity - 1) * 16, above 0 once minDisparity is above 1; and a disparity of 0 is a
        # depth at infinity, which no reader can place.
        valid = (raw >= search.min_disparity * _SUBPIXEL) & (raw > 0)
        if pad:
            columns = np.arange(raw.shape[1], dtype=np.float32)[None, :]
            valid &= columns - disparity >= 0.0  # a match inside the padding is a match with nothing
        if settings.near_end_px:
            valid &= disparity < np.float32(search.max_disparity - settings.near_end_px)
        if settings.refuse_flat_blocks:
            valid &= ~_flat(grey_left, int(settings.block_size))
        if settings.left_right_px is not None:
            flipped = self._match(np.ascontiguousarray(grey_right[:, ::-1]), np.ascontiguousarray(grey_left[:, ::-1]),
                                  pad)
            valid &= _left_right_agree(disparity, flipped[:, ::-1], search, float(settings.left_right_px))
        valid = _away_from_holes_and_steps(disparity, valid, settings)
        disparity[~valid] = np.nan
        return disparity, valid

    def _match(self, first: np.ndarray, second: np.ndarray, pad: int) -> np.ndarray:
        """OpenCV's ``int16`` map of ``first`` against ``second``, both padded on their left by ``pad`` columns first."""
        if pad:
            first = cv.copyMakeBorder(first, 0, 0, pad, 0, cv.BORDER_CONSTANT, value=0)
            second = cv.copyMakeBorder(second, 0, 0, pad, 0, cv.BORDER_CONSTANT, value=0)
        return np.asarray(self._matcher.compute(first, second))[:, pad:]


def _grey(image: np.ndarray) -> np.ndarray:
    """A grey ``uint8`` image: a grey one as it is, a BGR one converted."""
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise StereoCalibrationError(f"expected a grey or a BGR image, shape (H, W) or (H, W, 3), got {image.shape}")
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def _flat(grey: np.ndarray, block: int) -> np.ndarray:
    """Where every grey level of the ``block`` x ``block`` window about a pixel is the same."""
    kernel = np.ones((block, block), dtype=np.uint8)
    return cv.dilate(grey, kernel, borderType=cv.BORDER_REPLICATE) == cv.erode(grey, kernel, borderType=cv.BORDER_REPLICATE)


def _left_right_agree(disparity: np.ndarray, raw_right: np.ndarray, search: DisparitySearch,
                      tolerance_px: float) -> np.ndarray:
    """Where the left map's disparity lands on a right pixel whose own disparity, the pair matched flipped, agrees.

    ``raw_right`` is the right image's map in its own columns: its pixel ``x`` matches the left one at ``x + d``.
    """
    height, width = disparity.shape
    right = raw_right.astype(np.float32) * np.float32(1.0 / _SUBPIXEL)
    right_valid = (raw_right >= search.min_disparity * _SUBPIXEL) & (raw_right > 0)
    with np.errstate(invalid="ignore"):
        landing = np.rint(np.arange(width, dtype=np.float32)[None, :] - disparity)
    inside = np.isfinite(landing) & (landing >= 0) & (landing < width)
    column = np.where(inside, landing, 0).astype(np.int64)
    row = np.arange(height)[:, None]
    return inside & right_valid[row, column] & (np.abs(disparity - right[row, column]) <= tolerance_px)


def _away_from_holes_and_steps(disparity: np.ndarray, valid: np.ndarray, settings: MatcherSettings) -> np.ndarray:
    """``valid`` without every pixel within its margin of a hole or of a step between two neighbours."""
    near = np.zeros(valid.shape, dtype=bool)
    if settings.step_margin_px:
        step = np.zeros(valid.shape, dtype=bool)
        across = valid[:, 1:] & valid[:, :-1] & (np.abs(disparity[:, 1:] - disparity[:, :-1]) > settings.step_px)
        step[:, 1:] |= across
        step[:, :-1] |= across
        down = valid[1:, :] & valid[:-1, :] & (np.abs(disparity[1:, :] - disparity[:-1, :]) > settings.step_px)
        step[1:, :] |= down
        step[:-1, :] |= down
        near |= _within(step, settings.step_margin_px)
    if settings.hole_margin_px:
        near |= _within(~valid, settings.hole_margin_px)
    return valid & ~near


def _within(seeds: np.ndarray, margin_px: int) -> np.ndarray:
    """Every pixel within ``margin_px`` of a ``seeds`` pixel, the square about it; the frame's border is no seed."""
    kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), dtype=np.uint8)
    return cv.dilate(seeds.astype(np.uint8), kernel, borderType=cv.BORDER_CONSTANT, borderValue=0) > 0
