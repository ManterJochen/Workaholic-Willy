"""Which assets a trained model has never seen: the list a held-out dataset is drawn from.

An evaluation set whose fold groups all appear in the training corpus cannot report a held-out
number, whatever the number says. The eval ladder's reference dataset is in that state, so its
`deep` row can never be a held-out number. Every model raises the question again, and answering it
by hand each time is how the answer stops being checked, so it is a command.

The denominator is the whole point. Counting mesh files on disk answers a different question:
`MeshAssetBank` filters on `max_extent_mm` (the `assets.max_mesh_extent_mm` config key the CLI
forwards) and on closability into a solid, so only the survivors can be placed. A corpus can use
every one of them, in which case usable and unseen is zero. This module counts what the bank yields,
because an asset the pipeline cannot place is not an asset a dataset can be built from.

The unit is the trainer's own `asset_group`: real meshes by name, generated ones by shape family,
because two `proc_00002_pouch` in two scenes are two different pouches and grouping them by name
would group by an accident of ordering.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["HeldOutReport", "format_report", "held_out_assets", "trained_groups"]


@dataclass(frozen=True, slots=True)
class HeldOutReport:
    """What the model has seen, what the bank can place, and the intersection that matters."""

    #: `source -> [asset_id]`, the assets that are both placeable and unseen.
    unseen: dict[str, list[str]]
    #: `source -> count` of mesh files present, before any bank filter.
    on_disk: dict[str, int]
    #: `source -> count` the bank actually yields.
    usable: dict[str, int]
    #: `source -> count` of usable assets the corpus trained on.
    trained: dict[str, int]
    #: Fold groups the corpus covered, all sources and procedural families together.
    trained_group_count: int
    corpus: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_unseen(self) -> int:
        return sum(len(ids) for ids in self.unseen.values())


def trained_groups(corpus_root: Path | str) -> set[str]:
    """Every fold group a corpus's scenes placed, via the trainer's own grouping function.

    Walks `**/scenes/*/scene.json`, so it works on a single dataset and on a sharded corpus alike.
    Objects live under `spec.objects`: reading a top-level `objects` key finds nothing in every
    scene, which is a silent empty answer rather than an error.
    """
    from src.robot.grasping.deep.corpus.index import asset_group  # noqa: PLC0415

    groups: set[str] = set()
    for scene in Path(corpus_root).glob("**/scenes/*/scene.json"):
        try:
            payload = json.loads(scene.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):                  # pragma: no cover (unreadable scene)
            continue
        for placed in (payload.get("spec") or {}).get("objects", []):
            asset_id = placed.get("asset_id")
            if asset_id:
                groups.add(asset_group(asset_id))
    return groups


def held_out_assets(corpus_root: Path | str, *, sources: Iterable[str] = ("gso", "ycb"),
                    max_extent_mm: float = 180.0) -> HeldOutReport:
    """The placeable assets whose fold group the corpus never covered."""
    from src.robot.grasping.deep.corpus.index import asset_group  # noqa: PLC0415

    from datagen.assets.library import MeshLibrary  # noqa: PLC0415
    from datagen.assets.meshes import MeshAssetBank  # noqa: PLC0415

    seen = trained_groups(corpus_root)
    library = MeshLibrary()
    bank = MeshAssetBank(library, max_extent_mm=max_extent_mm)

    unseen: dict[str, list[str]] = {}
    on_disk: dict[str, int] = {}
    usable: dict[str, int] = {}
    trained: dict[str, int] = {}
    for source in sources:
        on_disk[source] = len(library.available(source))
        placeable = [asset.asset_id for asset in bank.load(source)]
        usable[source] = len(placeable)
        fresh = [asset_id for asset_id in placeable if asset_group(asset_id) not in seen]
        unseen[source] = sorted(fresh)
        trained[source] = len(placeable) - len(fresh)

    warnings: list[str] = []
    if not seen:
        warnings.append(
            f"no scenes found under {corpus_root}; every asset would read as unseen, which is the "
            f"most dangerous possible answer. Check the path before using this list.")
    if seen and sum(len(ids) for ids in unseen.values()) == 0:
        warnings.append(
            "ZERO placeable assets are unseen: this corpus used every mesh the bank can yield. A "
            "held-out dataset needs NEW meshes; `python -m datagen.assets.fetch --list` for what "
            "is available, and note that only a fraction of any download survives the bank filters.")
    for source, count in usable.items():
        if on_disk[source] and not count:
            warnings.append(
                f"{source}: {on_disk[source]} file(s) on disk and NONE placeable; the bank filtered "
                f"all of them (extent > {max_extent_mm:.0f} mm, or not closable into a solid).")

    return HeldOutReport(unseen=unseen, on_disk=on_disk, usable=usable, trained=trained,
                         trained_group_count=len(seen), corpus=str(corpus_root),
                         warnings=tuple(warnings))


def format_report(report: HeldOutReport) -> str:
    lines = [
        f"corpus {report.corpus}: {report.trained_group_count} fold group(s) covered",
        "",
        f"  {'source':8} {'on disk':>9} {'placeable':>10} {'trained':>9} {'UNSEEN':>8}",
    ]
    for source in sorted(report.unseen):
        # `.get`, not `[...]`: a report function that raises destroys the very information it was
        # called to convey. A partially-filled report is a caller's problem to see, not a reason to
        # lose the rest of the table.
        lines.append(f"  {source:8} {report.on_disk.get(source, 0):9d} "
                     f"{report.usable.get(source, 0):10d} "
                     f"{report.trained.get(source, 0):9d} {len(report.unseen[source]):8d}")
    lines.append(f"  {'TOTAL':8} {sum(report.on_disk.values()):9d} {sum(report.usable.values()):10d} "
                 f"{sum(report.trained.values()):9d} {report.total_unseen:8d}")
    lines.append(
        "\n  PLACEABLE is the denominator, not `on disk`. An asset the bank filters out cannot be "
        "put in a scene, so counting files answers a different question than the one being asked.")
    for warning in report.warnings:
        lines.append(f"\n  ! {warning}")
    return "\n".join(lines)
