"""Steps 1 and 2 of the diagnosis: is the ground truth where the network can see it?

Two questions, in the order they have to be answered, before any training question is worth asking.

Does the ground truth sit on the cloud at all? The graspability label is not placed at a grasp
centre. It is placed where a labelled grasp contacted the surface, and a point is marked graspable
when a contact lands within `graspability_radius_mm` of it. So the label can only exist where the
cloud has points. If contacts routinely land far from every point, the field is empty in exactly the
places the net is supposed to fire.

Does the ground truth survive the downsampling? This network is an encoder-decoder: it samples
2048, then 512, then 128 farthest points, and then interpolates back so that every one of the 8192
input points gets a prediction. The labels are therefore never misaligned. What can go wrong is the
information flow: a graspable point contributes to a set-abstraction level only if it falls inside
the ball query around one of that level's centres. A point outside every ball at level one is a point
whose evidence enters the trunk only through interpolation from neighbours that may not be graspable.

The architecture differs here from the one it is modelled on. The reference baseline predicts at
the M sampled points, so its output head is `M x (2 + V)`; this one decodes back to N and predicts
per input point. The two make different demands on sampling, and only one of them is measured here.

Nothing in this module trains, loads or writes a model. It reads a corpus and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

__all__ = [
    "LevelReach",
    "SamplingReport",
    "format_sampling_report",
    "sampling_report",
]

#: Millimetres per metre, because the sample is in metres and every threshold here is in millimetres.
_M_TO_MM = 1000.0


@dataclass(frozen=True, slots=True)
class LevelReach:
    """What one set-abstraction level can and cannot see of the supervised points."""

    index: int
    centres: int
    radius_mm: float
    #: Fraction of graspable points that fall inside at least one ball at this level.
    covered: float
    #: Distance from a graspable point to its nearest centre, in millimetres.
    median_mm: float
    p95_mm: float
    max_mm: float
    #: Fraction of all points covered, as the control. A level that covers graspable points at the
    #: same rate as everything else is not selecting against them.
    covered_all: float


@dataclass(frozen=True, slots=True)
class SamplingReport:
    """One sample, measured end to end."""

    scene: str
    points: int
    contacts: int
    graspable_points: int
    graspable_fraction: float
    #: Distance from each labelled contact to the nearest point in the raw scene cloud, in mm.
    #: This asks whether the label is placeable at all, before any sampling.
    contact_to_raw_median_mm: float
    contact_to_raw_p95_mm: float
    contact_to_raw_max_mm: float
    contacts_orphaned_raw: int
    #: The same against the sampled cloud the network actually receives. The gap between the
    #: two is what the subsampling costs.
    contact_to_sample_median_mm: float
    contact_to_sample_p95_mm: float
    contact_to_sample_max_mm: float
    contacts_orphaned_sample: int
    labelling_radius_mm: float
    levels: tuple[LevelReach, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


def _nearest_distance_mm(query_mm: np.ndarray, cloud_mm: np.ndarray) -> np.ndarray:
    """Distance from every query point to its nearest neighbour in the cloud, in millimetres.

    Chunked rather than one big matrix. A 8192 x 8192 float64 distance matrix is 512 MB, and this is
    called once per sample in a sweep over hundreds of them.
    """
    if not len(query_mm) or not len(cloud_mm):
        return np.zeros(len(query_mm), dtype=np.float64)
    out = np.empty(len(query_mm), dtype=np.float64)
    step = max(1, 2_000_000 // max(1, len(cloud_mm)))
    for start in range(0, len(query_mm), step):
        chunk = query_mm[start:start + step]
        diff = chunk[:, None, :] - cloud_mm[None, :, :]
        out[start:start + step] = np.sqrt((diff * diff).sum(-1)).min(axis=1)
    return out


def _farthest_point_sample(points_mm: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Indices of `count` farthest points, by the greedy rule.

    Start somewhere, then repeatedly take the point farthest from everything chosen so far. This
    is a corpus diagnostic with no counterpart in the shipped network: the serialized backbone in
    `net` samples no farthest points, so nothing here has to match it, and the loop that calls it
    runs only when a caller passes a non-empty `levels`, which the CLI does not. Written in numpy
    rather than against a torch tensor on a device, because it has to run on a machine with no GPU
    and no torch import: a dataset question should not require the training stack to answer it.
    """
    total = len(points_mm)
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    count = min(count, total)
    chosen = np.empty(count, dtype=np.int64)
    # Seeded rather than always index 0, so a sweep does not measure one arbitrary starting corner.
    rng = np.random.default_rng(seed)
    current = int(rng.integers(total))
    best = np.full(total, np.inf)
    for step in range(count):
        chosen[step] = current
        diff = points_mm - points_mm[current]
        dist = (diff * diff).sum(-1)
        best = np.minimum(best, dist)
        current = int(best.argmax())
    return chosen


def sampling_report(sample: dict[str, Any], *, scene: str,
                    levels: "Sequence[tuple[int, float, int]]",
                    labelling_radius_mm: float,
                    contacts_mm: np.ndarray | None = None,
                    raw_points_mm: np.ndarray | None = None,
                    seed: int = 0) -> SamplingReport:
    """Measure one built sample against the network's sampling schedule.

    `levels` is the network's own `(count, radius_m, cap)` tuple list, so this reports on whatever
    architecture is configured rather than on a copy of its numbers.

    The contacts and the sample are in different frames. `build_sample` returns `points_m` in a
    local frame: xy centred on the cloud mean, z shifted by the support height, while the contacts
    come out of the scene in the original frame. Compared without that move, essentially every
    contact reads as orphaned and an empty label field is reported where the measurement is at
    fault. The contacts are moved into the sample frame here, using the offsets the sample already
    carries for putting a prediction back where it came from.
    """
    points_mm = np.asarray(sample["points_m"], dtype=np.float64) * _M_TO_MM
    graspability = np.asarray(sample["graspability"], dtype=np.float64)
    graspable = graspability > 0.5
    notes: list[str] = []

    raw_median = raw_p95 = raw_max = float("nan")
    sample_median = sample_p95 = sample_max = float("nan")
    orphan_raw = orphan_sample = 0
    contacts = 0
    if contacts_mm is not None and len(contacts_mm):
        contacts = len(contacts_mm)
        raw_contacts = np.asarray(contacts_mm, dtype=np.float64)

        if raw_points_mm is not None and len(raw_points_mm):
            # Both sides in the original frame, which is where the label is actually computed.
            far = _nearest_distance_mm(raw_contacts,
                                       np.asarray(raw_points_mm, dtype=np.float64))
            raw_median = float(np.median(far))
            raw_p95 = float(np.percentile(far, 95))
            raw_max = float(far.max())
            # A contact further than the labelling radius from every point marks nothing. It
            # is a ground-truth grasp contributing no supervision at all, and it is invisible in
            # every aggregate metric because the field it should have written to stays zero.
            orphan_raw = int((far > labelling_radius_mm).sum())
            if orphan_raw:
                notes.append(f"{orphan_raw} of {contacts} contacts have no RAW point within "
                             f"{labelling_radius_mm:.0f} mm")

        # Into the sample frame, with the offsets the sample carries for exactly this purpose.
        centre_xy = np.asarray(sample.get("centre_xy_mm", np.zeros(2)), dtype=np.float64)
        support = float(sample.get("support_height_mm", 0.0))
        local = raw_contacts.copy()
        local[:, :2] -= centre_xy
        local[:, 2] -= support
        near = _nearest_distance_mm(local, points_mm)
        sample_median = float(np.median(near))
        sample_p95 = float(np.percentile(near, 95))
        sample_max = float(near.max())
        orphan_sample = int((near > labelling_radius_mm).sum())
        if orphan_sample and orphan_sample != orphan_raw:
            notes.append(f"{orphan_sample} of {contacts} contacts are orphaned AFTER sampling "
                         f"(raw: {orphan_raw})")

    reaches: list[LevelReach] = []
    current = points_mm
    for index, (count, radius_m, _cap) in enumerate(levels):
        picked = _farthest_point_sample(current, count, seed=seed + index)
        centres = current[picked]
        radius_mm = float(radius_m) * _M_TO_MM
        if graspable.any() and index == 0:
            to_centre = _nearest_distance_mm(points_mm[graspable], centres)
            all_to_centre = _nearest_distance_mm(points_mm, centres)
        else:
            # Beyond level one the balls are drawn around the previous level's points, so the honest
            # question is still "can a graspable input point reach a centre", measured from the
            # original cloud rather than from the subsample.
            to_centre = (_nearest_distance_mm(points_mm[graspable], centres)
                         if graspable.any() else np.zeros(0))
            all_to_centre = _nearest_distance_mm(points_mm, centres)
        reaches.append(LevelReach(
            index=index,
            centres=len(centres),
            radius_mm=radius_mm,
            covered=float((to_centre <= radius_mm).mean()) if len(to_centre) else float("nan"),
            median_mm=float(np.median(to_centre)) if len(to_centre) else float("nan"),
            p95_mm=float(np.percentile(to_centre, 95)) if len(to_centre) else float("nan"),
            max_mm=float(to_centre.max()) if len(to_centre) else float("nan"),
            covered_all=float((all_to_centre <= radius_mm).mean()),
        ))
        current = centres

    if not graspable.any():
        notes.append("no graspable point in this sample at all, so every reach figure is empty")

    return SamplingReport(
        scene=scene,
        points=len(points_mm),
        contacts=contacts,
        graspable_points=int(graspable.sum()),
        graspable_fraction=float(graspable.mean()) if len(graspable) else 0.0,
        contact_to_raw_median_mm=raw_median,
        contact_to_raw_p95_mm=raw_p95,
        contact_to_raw_max_mm=raw_max,
        contacts_orphaned_raw=orphan_raw,
        contact_to_sample_median_mm=sample_median,
        contact_to_sample_p95_mm=sample_p95,
        contact_to_sample_max_mm=sample_max,
        contacts_orphaned_sample=orphan_sample,
        labelling_radius_mm=labelling_radius_mm,
        levels=tuple(reaches),
        notes=tuple(notes),
    )


def format_sampling_report(reports: "Sequence[SamplingReport]") -> str:
    """A human-readable roll-up. One block of headline numbers, then the per-level table."""
    if not reports:
        return "no samples measured"
    lines: list[str] = []
    lines.append(f"{len(reports)} sample(s)")
    lines.append("")

    def summarise(values: list[float], label: str, unit: str = "mm") -> str:
        clean = [v for v in values if not np.isnan(v)]
        if not clean:
            return f"  {label:<44} (nothing to measure)"
        return (f"  {label:<44} median {np.median(clean):7.2f} {unit}   "
                f"p95 {np.percentile(clean, 95):7.2f} {unit}   max {max(clean):7.2f} {unit}")

    lines.append("Step 1: does the ground truth sit on the cloud?")
    total_contacts = sum(r.contacts for r in reports)
    radius = reports[0].labelling_radius_mm
    lines.append("  against the RAW scene cloud, which is where the label is computed:")
    lines.append(summarise([r.contact_to_raw_median_mm for r in reports],
                           "  contact to nearest raw point, per-sample median"))
    lines.append(summarise([r.contact_to_raw_max_mm for r in reports],
                           "  contact to nearest raw point, per-sample worst"))
    orphan_raw = sum(r.contacts_orphaned_raw for r in reports)
    raw_share = (orphan_raw / total_contacts * 100.0) if total_contacts else 0.0
    lines.append(f"  {'  contacts labelling NOTHING (raw)':<44} {orphan_raw} of "
                 f"{total_contacts} ({raw_share:.2f} %), further than {radius:.0f} mm from "
                 f"every point")
    lines.append("")
    lines.append("  against the SAMPLED cloud the network receives:")
    lines.append(summarise([r.contact_to_sample_median_mm for r in reports],
                           "  contact to nearest sampled point, per-sample median"))
    lines.append(summarise([r.contact_to_sample_max_mm for r in reports],
                           "  contact to nearest sampled point, per-sample worst"))
    orphan_sample = sum(r.contacts_orphaned_sample for r in reports)
    sample_share = (orphan_sample / total_contacts * 100.0) if total_contacts else 0.0
    lines.append(f"  {'  contacts labelling NOTHING (sampled)':<44} {orphan_sample} of "
                 f"{total_contacts} ({sample_share:.2f} %)")
    lines.append("")

    graspable = [r.graspable_fraction * 100.0 for r in reports]
    empty = sum(1 for r in reports if r.graspable_points == 0)
    lines.append(f"  {'graspable points per sample':<44} median {np.median(graspable):6.2f} %   "
                 f"min {min(graspable):6.2f} %   max {max(graspable):6.2f} %")
    lines.append(f"  {'samples with NO graspable point at all':<44} {empty} of {len(reports)}")
    lines.append("")

    lines.append("Step 2: does it survive the downsampling?")
    lines.append(f"  {'level':<6}{'centres':>9}{'radius':>10}{'covered':>10}{'covered(all)':>14}"
                 f"{'median':>10}{'p95':>10}{'max':>10}")
    depth = len(reports[0].levels)
    for index in range(depth):
        rows = [r.levels[index] for r in reports if index < len(r.levels)]
        if not rows:
            continue
        cov = [x.covered for x in rows if not np.isnan(x.covered)]
        cov_all = [x.covered_all for x in rows if not np.isnan(x.covered_all)]
        med = [x.median_mm for x in rows if not np.isnan(x.median_mm)]
        p95 = [x.p95_mm for x in rows if not np.isnan(x.p95_mm)]
        mx = [x.max_mm for x in rows if not np.isnan(x.max_mm)]
        lines.append(
            f"  {index + 1:<6}{rows[0].centres:>9}{rows[0].radius_mm:>9.0f}mm"
            f"{(np.mean(cov) * 100 if cov else float('nan')):>9.1f}%"
            f"{(np.mean(cov_all) * 100 if cov_all else float('nan')):>13.1f}%"
            f"{(np.median(med) if med else float('nan')):>9.1f}mm"
            f"{(np.median(p95) if p95 else float('nan')):>9.1f}mm"
            f"{(np.median(mx) if mx else float('nan')):>9.1f}mm")
    lines.append("")
    lines.append("  covered      = graspable points within the level's ball radius of some centre")
    lines.append("  covered(all) = the same for EVERY point, as the control. If the two match, the")
    lines.append("                 sampling is not selecting against graspable points.")

    seen: set[str] = set()
    for report in reports:
        for note in report.notes:
            key = note.split(",")[0]
            if key not in seen:
                seen.add(key)
                lines.append(f"  note: {note}")
    return "\n".join(lines)


def write_visual(sample: dict[str, Any], contacts_mm: np.ndarray | None,
                 out_path: str | Path, *, scene: str = "") -> Path:
    """Draw one sample: the cloud, the graspable points, and the labelled contacts.

    Three orthographic views rather than one perspective render. A perspective view of a point cloud
    hides exactly the thing being checked, which is whether a contact floats off the surface.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    points_mm = np.asarray(sample["points_m"], dtype=np.float64) * _M_TO_MM
    graspable = np.asarray(sample["graspability"], dtype=np.float64) > 0.5
    contacts = (np.asarray(contacts_mm, dtype=np.float64)
                if contacts_mm is not None and len(contacts_mm) else np.zeros((0, 3)))
    if len(contacts):
        # The same frame move as `sampling_report`. Drawn in the original frame, every cross lands
        # off the edge of the plot.
        contacts = contacts.copy()
        contacts[:, :2] -= np.asarray(sample.get("centre_xy_mm", np.zeros(2)), dtype=np.float64)
        contacts[:, 2] -= float(sample.get("support_height_mm", 0.0))

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
    for axis, (a, b, label) in zip(axes, ((0, 1, "x / y"), (0, 2, "x / z"), (1, 2, "y / z")),
                                   strict=True):
        axis.scatter(points_mm[~graspable, a], points_mm[~graspable, b], s=1.0,
                     c="#b0b6bd", linewidths=0, label="cloud")
        axis.scatter(points_mm[graspable, a], points_mm[graspable, b], s=3.0,
                     c="#1f6feb", linewidths=0, label="graspable label")
        if len(contacts):
            axis.scatter(contacts[:, a], contacts[:, b], s=22.0, marker="x",
                         c="#d1242f", linewidths=1.1, label="labelled contact")
        axis.set_xlabel(label)
        axis.set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper right", fontsize=8, markerscale=3)
    figure.suptitle(
        f"{scene}   {len(points_mm)} points, {int(graspable.sum())} graspable, "
        f"{len(contacts)} contacts")
    figure.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    plt.close(figure)
    return out
