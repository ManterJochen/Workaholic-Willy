"""Rendering: the half that needs a GPU, and the numerics that do not.

Only :mod:`datagen.render.isaac` imports Isaac. Everything else here (the depth-sensor model, the
label extraction, the viewpoint policy, the material sampler and the dataset writer) is numpy and
filesystem, so it runs off a GPU box. That split is deliberate: these modules decide what the labels
mean, and a labelling bug that can only be caught on a GPU box is a labelling bug that ships.
"""

from __future__ import annotations

from datagen.render.depth_noise import REALSENSE_D435, DepthSensorModel, inject_depth_noise
from datagen.render.labels import ObjectLabel, bbox_from_mask, build_object_labels, visibility
from datagen.render.engine import ENGINES, SceneEngine, build_engine
from datagen.render.materials import MaterialParams, sample_material
from datagen.render.result import SceneRenderResult, ViewRender
from datagen.render.views import UnreachablePolicy, ViewOutcome, resolve_wrist_viewpoint
from datagen.render.writer import DatasetWriter, SceneRecord, encode_depth_png

__all__ = [
    "ENGINES",
    "REALSENSE_D435",
    "DatasetWriter",
    "DepthSensorModel",
    "SceneEngine",
    "SceneRenderResult",
    "ViewRender",
    "build_engine",
    "MaterialParams",
    "ObjectLabel",
    "SceneRecord",
    "UnreachablePolicy",
    "ViewOutcome",
    "bbox_from_mask",
    "build_object_labels",
    "encode_depth_png",
    "inject_depth_noise",
    "resolve_wrist_viewpoint",
    "sample_material",
    "visibility",
]
