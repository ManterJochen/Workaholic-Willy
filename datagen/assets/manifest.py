"""The asset manifest: every mesh that may reach a pixel, with the licence that lets it.

A dataset's licence is not a property of the generator; it is the union of the licences of
everything rendered into it, because an image is a derivative work of every mesh in it. So the
manifest is not bookkeeping. It is the artefact that makes a dataset shareable or not, and it is
audited before a run rather than after, when the only remedy is re-rendering.

Two outputs, both required:

* :meth:`AssetManifest.audit` is fail-closed. Non-commercial, unknown, or forbidden-source rows are
  refused with the reason, and a CC-BY row without an attribution string is refused too, because
  that licence obliges naming the author wherever the work or a derivative appears.
* :meth:`AssetManifest.attribution_text` is the attribution file shipped next to every dataset. It
  is what turns "we complied" into something a reader can check.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any
from dataclasses import dataclass, field

from datagen.assets.licensing import audit_asset_rows

__all__ = ["AssetManifest", "AssetRecord", "ManifestAuditError"]


class ManifestAuditError(ValueError):
    """The manifest contains an asset that may not be rendered. Carries every problem, not the first."""


@dataclass(frozen=True, slots=True)
class AssetRecord:
    """One asset the generator may place. ``kind`` + ``extent_mm`` are what the renderer authors from."""

    asset_id: str
    source: str          # "procedural" | "gso" | "objaverse" | "ycb"
    license: str         # "own" | "CC0-1.0" | "CC-BY-4.0" | ...
    kind: str
    extent_mm: tuple[float, float, float]
    mass_kg: float
    #: Author / title / URL. Required by :func:`audit_asset_rows` for any attribution-obliging licence;
    #: empty is correct only for ``own`` and CC0.
    attribution: str = ""
    #: Path to a mesh on disk, for sourced assets. ``None`` for procedural (the renderer builds it).
    mesh_path: str | None = None
    #: The primitives a composite object is built from; empty for everything else.
    #:
    #: Here rather than in the scene spec because `ObjectPlacement` says it outright: geometry lives
    #: in the manifest. And because the labeller reconstructs the manifest from the provenance stamp,
    #: a structure recorded here reaches every labelling run for free. Without it, `scene_geometry`
    #: would derive a mug's shape from the tail of its asset id and label it as a cuboid.
    parts: tuple[Any, ...] = ()
    tags: tuple[str, ...] = ()

    def as_row(self) -> dict[str, str]:
        return {
            "id": self.asset_id,
            "source": self.source,
            "license": self.license,
            "attribution": self.attribution,
            "kind": self.kind,
        }


@dataclass(slots=True)
class AssetManifest:
    """The set of assets one dataset may draw from."""

    records: list[AssetRecord] = field(default_factory=list)

    def add(self, record: AssetRecord) -> None:
        self.records.append(record)

    def extend(self, records: Iterable[AssetRecord]) -> None:
        self.records.extend(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[AssetRecord]:
        return iter(self.records)

    def get(self, asset_id: str) -> AssetRecord:
        for record in self.records:
            if record.asset_id == asset_id:
                return record
        raise KeyError(
            f"asset {asset_id!r} is not in the manifest; a scene referencing it cannot be rendered, "
            f"and could not have been licence-audited either"
        )

    def problems(self) -> list[str]:
        """Every licensing problem, as strings. Empty means the manifest may be rendered."""
        return audit_asset_rows(record.as_row() for record in self.records)

    def audit(self) -> None:
        """Raise unless every asset may legally be rendered and shipped. Call before generating."""
        problems = self.problems()
        if problems:
            raise ManifestAuditError(
                "the asset manifest contains assets that may not be rendered:\n  "
                + "\n  ".join(problems)
                + "\n\nEvery rendered image is a derivative work of the meshes in it, so an asset that "
                "cannot be shipped cannot be rendered either. Fix the rows or drop the assets."
            )

    def attribution_text(self, *, dataset_name: str) -> str:
        """The attribution file: every asset that could reach a pixel, grouped by licence.

        Includes ``own`` and CC0 rows even though neither obliges attribution: a reader checking
        compliance needs to see the whole set, not only the obliged parts.
        """
        by_license: dict[str, list[AssetRecord]] = {}
        for record in sorted(self.records, key=lambda r: (r.license.lower(), r.asset_id)):
            by_license.setdefault(record.license, []).append(record)
        lines = [
            f"ATTRIBUTION: {dataset_name}",
            "",
            "Every asset below was rendered into one or more images in this dataset. A rendered image",
            "is a derivative work of the meshes it contains, so these attributions travel with the",
            "images, not only with the meshes.",
            "",
        ]
        for license_id, records in by_license.items():
            lines.append(f"## {license_id}  ({len(records)} asset(s))")
            for record in records:
                suffix = f", {record.attribution}" if record.attribution else ""
                lines.append(f"  {record.asset_id}  [{record.source}]{suffix}")
            lines.append("")
        return "\n".join(lines)
