"""The contact-based grasping primitives."""

from __future__ import annotations

from .antipodal import find_antipodal_pairs
from .contact_pair import ContactPair
from .contact_point import ContactPoint
from .dense_sampler import (
    SurfaceSamples,
    dense_surface_samples,
    scene_collision_cloud,
)

__all__ = [
    "ContactPair",
    "ContactPoint",
    "SurfaceSamples",
    "dense_surface_samples",
    "find_antipodal_pairs",
    "scene_collision_cloud",
]
