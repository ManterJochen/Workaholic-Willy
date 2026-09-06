"""Assets: what may be placed in a scene, and under which licence it may be rendered at all."""

from __future__ import annotations

from datagen.assets.licensing import audit_asset_rows
from datagen.assets.manifest import AssetManifest, AssetRecord
from datagen.assets.procedural import ProceduralAsset, sample_procedural_asset

__all__ = [
    "AssetManifest",
    "AssetRecord",
    "ProceduralAsset",
    "audit_asset_rows",
    "sample_procedural_asset",
]
