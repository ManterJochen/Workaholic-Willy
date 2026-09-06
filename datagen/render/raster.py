"""Depth and instance masks without a GPU, a driver, a compiler or a display.

The rendering half of every gl-free backend. Rendering needs no engine: the corpus carries no RGB,
only points, normals, instance ids, view counts and grasp labels, so what a backend owes is depth in
millimetres and a per-object silhouette. Both are pure geometry, and a z-buffer over projected
triangles produces them exactly.

It adds nothing to install. `trimesh` is already in `requirements.txt`; everything below is numpy.
That matters more than speed: the reason a second backend exists at all is that Isaac Sim cannot be
installed in many companies, and an alternative that needs a Vulkan runtime, a system OpenGL or a
large binary compiled against msvc would be the same wall in smaller form. MuJoCo's own headless path
defaults to glfw, i.e. hardware OpenGL, with `osmesa` available on Linux only, which is why the
physics engine and the renderer are separated here rather than taken as a pair.

What it is not: a beauty pass. No light transport, no materials, no RGB. The path-traced route stays
Isaac's, for the preview strips and anything that needs colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RasterView", "rasterise"]

#: Anything nearer than this in CAMERA z is treated as behind the lens. Triangles crossing the plane
#: are dropped whole rather than clipped: a partially-clipped triangle needs vertex interpolation to
#: stay perspective-correct, and at these working distances (0.5-1 m) a triangle that reaches within a
#: millimetre of the sensor is a broken scene, not a view to salvage.
_NEAR_MM = 1.0


@dataclass(frozen=True, slots=True)
class RasterView:
    """One rendered view.

    ``depth_mm`` is CAMERA-frame z, the convention `camera.unproject_to_base` reads. Not ray length,
    which differs by up to the tangent of the half-field and would look plausible while placing every
    point wrongly.
    """

    depth_mm: np.ndarray
    instance_map: np.ndarray

    @property
    def hit(self) -> np.ndarray:
        return self.depth_mm > 0.0


def rasterise(bodies: dict[int, tuple[np.ndarray, np.ndarray]],
              camera_to_base: np.ndarray, intrinsics: np.ndarray,
              resolution: tuple[int, int]) -> RasterView:
    """Z-buffer every body's triangles into a depth image and an instance map.

    ``bodies`` maps an instance id to ``(vertices_mm, faces)`` in BASE. Ids are written into the
    instance map as they are, and 0 means nothing was hit, so an id of 0 must not be used for a body.
    The caller owns that, and `datagen`'s layout already numbers from 1.

    Deterministic by construction: no sampling, no acceleration structure, and no iteration over a
    dict whose order could change the result. Ties are broken by depth, and the bodies are visited in
    sorted id order, so an exact tie resolves the same way on every run.
    """
    width, height = int(resolution[0]), int(resolution[1])
    depth = np.full((height, width), np.inf, dtype=np.float64)
    instances = np.zeros((height, width), dtype=np.int32)

    transform = np.asarray(camera_to_base, dtype=np.float64)
    base_to_camera = np.linalg.inv(transform)
    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    for instance_id, (vertices_mm, faces) in sorted(bodies.items()):
        vertices = np.asarray(vertices_mm, dtype=np.float64).reshape(-1, 3)
        if not len(vertices):
            continue
        camera = (base_to_camera[:3, :3] @ vertices.T).T + base_to_camera[:3, 3]
        for face in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
            triangle = camera[face]
            if np.any(triangle[:, 2] <= _NEAR_MM):
                continue
            pixels = np.stack([triangle[:, 0] / triangle[:, 2] * fx + cx,
                               triangle[:, 1] / triangle[:, 2] * fy + cy], axis=1)
            low = np.maximum(np.floor(pixels.min(axis=0)).astype(int), 0)
            high = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + 1, [width, height])
            if np.any(high <= low):
                continue

            columns, rows = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]))
            (x0, y0), (x1, y1), (x2, y2) = pixels
            area = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(area) < 1e-12:                       # degenerate on screen: an edge, no interior
                continue
            alpha = ((y1 - y2) * (columns - x2) + (x2 - x1) * (rows - y2)) / area
            beta = ((y2 - y0) * (columns - x2) + (x0 - x2) * (rows - y2)) / area
            gamma = 1.0 - alpha - beta
            inside = (alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0)
            if not inside.any():
                continue

            # Perspective-correct: interpolate 1/z linearly in screen space, not z. Interpolating z
            # directly looks right on a fronto-parallel face and bows every oblique one, which is
            # precisely the view this corpus is built from.
            reciprocal = alpha / triangle[0, 2] + beta / triangle[1, 2] + gamma / triangle[2, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                z = 1.0 / reciprocal
            usable = inside & np.isfinite(z) & (z > _NEAR_MM)
            if not usable.any():
                continue

            at_rows, at_columns = rows[usable], columns[usable]
            candidate = z[usable]
            nearer = candidate < depth[at_rows, at_columns]
            if not nearer.any():
                continue
            depth[at_rows[nearer], at_columns[nearer]] = candidate[nearer]
            instances[at_rows[nearer], at_columns[nearer]] = int(instance_id)

    depth[~np.isfinite(depth)] = 0.0
    return RasterView(depth_mm=depth, instance_map=instances)
