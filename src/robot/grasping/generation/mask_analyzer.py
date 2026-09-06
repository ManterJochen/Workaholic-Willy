"""
MaskAnalyzer: extract geometric features from a segmentation mask.

Given a binary ``(H, W)`` mask, this analyser computes the inputs the
:class:`GraspCalculator` needs:

* Centroid (from raw image moments).
* Principal / minor axis (PCA on mask pixels).
* Major / minor extent (length of the mask along each axis, in pixels).
* Oriented bounding box from ``cv.minAreaRect``.
* Axis-aligned bounding box from ``cv.boundingRect``.

Optional pre-processing
-----------------------
A small morphological close can be applied before contour extraction to
collapse single-pixel holes in noisy SAM2 masks. Disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE
from src.utility.log_cfg import create_logger


@dataclass(frozen=True, slots=True)
class MaskAnalysis:
    """2D geometric analysis of a segmentation mask (pixel coordinates)."""

    centroid_xy: tuple[float, float]
    principal_axis: np.ndarray   # (2,) unit vector
    minor_axis: np.ndarray       # (2,) unit vector
    extent_major_px: float
    extent_minor_px: float
    orientation_deg: float       # CCW from +x, range (-90, 90]
    obb_corners: np.ndarray      # (4, 2)
    obb_angle_deg: float
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int


class MaskAnalyzer:
    """Stateless analyser. Call :meth:`analyze` with a binary mask.

    Parameters
    ----------
    min_area_px : int
        Masks smaller than this are rejected (returns ``None``).
    morph_close_kernel_px : int
        If > 0, apply a morphological close with an elliptical kernel
        inscribed in a box of this size before contour extraction.
        Useful for noisy masks.
    """

    def __init__(
        self,
        min_area_px: int = 50,
        morph_close_kernel_px: int = 0,
    ) -> None:
        self.min_area_px = max(1, int(min_area_px))
        self.morph_close_kernel_px = max(0, int(morph_close_kernel_px))
        self.logger = create_logger(
            "MaskAnalyzer", ROBOT_LOG_FILE, log_dir=ROBOT_LOG_DIR,
        )

    def analyze(self, mask: np.ndarray) -> MaskAnalysis | None:
        """Return a :class:`MaskAnalysis` or ``None`` if the mask is unusable."""
        if mask.ndim != 2:
            raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")

        binary: np.ndarray = (mask > 0).astype(np.uint8) * 255

        if self.morph_close_kernel_px > 0:
            k = self.morph_close_kernel_px
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
            binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

        area = int(np.count_nonzero(binary))
        if area < self.min_area_px:
            self.logger.debug(
                "Mask rejected: area=%d < min_area_px=%d", area, self.min_area_px,
            )
            return None

        contours, _ = cv.findContours(
            binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        contour = max(contours, key=cv.contourArea)
        if cv.contourArea(contour) < self.min_area_px:
            return None

        rect = cv.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
        obb_corners = cv.boxPoints(rect).astype(np.float64)
        obb_angle = float(rect[2])

        moments = cv.moments(binary)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        ys, xs = np.nonzero(binary)
        pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        principal, minor, ext_major, ext_minor, angle_deg = self._pca(pts)

        x0, y0, w, h = cv.boundingRect(contour)
        bbox_xyxy = (int(x0), int(y0), int(x0 + w), int(y0 + h))

        return MaskAnalysis(
            centroid_xy=(float(cx), float(cy)),
            principal_axis=principal,
            minor_axis=minor,
            extent_major_px=ext_major,
            extent_minor_px=ext_minor,
            orientation_deg=angle_deg,
            obb_corners=obb_corners,
            obb_angle_deg=obb_angle,
            bbox_xyxy=bbox_xyxy,
            area_px=area,
        )

    @staticmethod
    def _pca(
        pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """PCA on Nx2 pixel coordinates.

        Returns
        -------
        principal, minor : (2,) unit vectors (major and minor axis directions)
        ext_major, ext_minor : float extents along those axes (pixels)
        angle_deg : orientation of the major axis, CCW from +x, range (-90, 90]
        """
        mean = pts.mean(axis=0)
        centered = pts - mean

        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]  # descending
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        principal = eigenvectors[:, 0]
        minor = eigenvectors[:, 1]

        # Defensive: for a degenerate (single-row) mask minor would be zero.
        if np.linalg.norm(minor) < 1e-12:
            minor = np.array([-principal[1], principal[0]], dtype=np.float64)

        # NumPy 2.4 can emit a spurious RuntimeWarning for 2D matvec under
        # warnings-as-errors; explicit dot components are equivalent here.
        proj_major = centered[:, 0] * principal[0] + centered[:, 1] * principal[1]
        proj_minor = centered[:, 0] * minor[0] + centered[:, 1] * minor[1]
        ext_major = float(proj_major.max() - proj_major.min())
        ext_minor = float(proj_minor.max() - proj_minor.min())

        angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
        if angle_deg > 90.0:
            angle_deg -= 180.0
        elif angle_deg <= -90.0:
            angle_deg += 180.0

        return principal, minor, ext_major, ext_minor, angle_deg
