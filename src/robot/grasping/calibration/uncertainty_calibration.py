"""Production uncertainty calibration tool.

Given a JSONL replay of records carrying the seven typed signal channels plus a
``label`` in ``[0, 1]``, where 1 means success, it fits a deterministic and
monotone-preserving :class:`UncertaintyCalibration`: per-channel quantile bins,
then the mean label per bin, then an isotonic Pool-Adjacent-Violators map on the
``[0, 1]`` domain the runtime fusion expects. Channel weights are not learned
here and stay at the ``UncertaintyWeights()`` defaults. The fit is fully
deterministic, with no prng, a stable sort order and pure-Python floats.

CLI:

    python -m src.robot.grasping.calibration.uncertainty_calibration \\
        --replay <replay.jsonl> \\
        --out artifact.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.robot.grasping.constants import (
    UNCERTAINTY_CALIBRATION_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannel,
    UncertaintyMonotoneMap,
    UncertaintyWeights,
)

#: The fit silently drops unlabelled/out-of-range samples per channel, so the
#: per-channel sample counts are the only way to tell a real fit from an
#: identity map produced by an empty channel.
logger = create_grasping_logger(
    "UncertaintyCalibration", UNCERTAINTY_CALIBRATION_LOG_FILE
)

__all__ = ["fit_uncertainty_calibration", "main", "usable_samples"]


_NUM_BINS = 5


def _pav_non_decreasing(values: Sequence[float]) -> list[float]:
    """Pool-Adjacent-Violators: a non-decreasing sequence of the same length (deterministic).

    Each output position holds the mean of the pooled block it ended up in, so ties stay stable.
    Blocks carry ``(mean, count)``; adjacent violations are merged until the sequence is monotone,
    then each block is re-expanded to its original width.
    """
    blocks: list[tuple[float, int]] = [(float(v), 1) for v in values]
    i = 1
    while i < len(blocks):
        if blocks[i][0] < blocks[i - 1][0]:
            (v0, n0), (v1, n1) = blocks[i - 1], blocks[i]
            blocks[i - 1] = ((v0 * n0 + v1 * n1) / (n0 + n1), n0 + n1)
            del blocks[i]
            i = max(1, i - 1)
        else:
            i += 1
    expanded: list[float] = []
    for value, count in blocks:
        expanded.extend([value] * count)
    return expanded


def _fit_channel_map(
    samples: list[tuple[float, float]],
) -> UncertaintyMonotoneMap:
    """Fit a monotone PWL map ``channel_value -> label_mean`` (quantile-bin + PAV)."""

    # Stable sort keeps ties deterministic.
    samples = sorted(samples, key=lambda p: p[0])
    n = len(samples)
    if n == 0:
        return UncertaintyMonotoneMap.identity()
    # Build bins of equal record count.
    bin_size = max(1, n // _NUM_BINS)
    bins: list[list[tuple[float, float]]] = []
    i = 0
    while i < n:
        bins.append(samples[i : i + bin_size])
        i += bin_size
    # If the last bin is small, merge into previous.
    if len(bins) > 1 and len(bins[-1]) < bin_size // 2:
        bins[-2].extend(bins[-1])
        bins.pop()
    # Per-bin breakpoint (mean x) and value (mean label).
    bp_raw = [
        sum(x for x, _ in b) / len(b) for b in bins
    ]
    vs_raw = [
        sum(y for _, y in b) / len(b) for b in bins
    ]
    vs_mono = _pav_non_decreasing(vs_raw)
    # Force breakpoints strictly increasing by collapsing duplicates.
    bp: list[float] = []
    vs: list[float] = []
    for x, y in zip(bp_raw, vs_mono):
        if bp and x <= bp[-1]:
            bp[-1] = (bp[-1] + x) / 2.0
            vs[-1] = max(vs[-1], y)
        else:
            bp.append(x)
            vs.append(y)
    # Anchor endpoints at 0 and 1 so the map covers ``[0, 1]``.
    if bp[0] > 0.0:
        bp.insert(0, 0.0)
        vs.insert(0, vs[0])
    if bp[-1] < 1.0:
        bp.append(1.0)
        vs.append(vs[-1])
    vs = [min(1.0, max(0.0, v)) for v in vs]
    if len(bp) < 2:
        return UncertaintyMonotoneMap.identity()
    return UncertaintyMonotoneMap(
        breakpoints=tuple(bp), values=tuple(vs),
    )


def usable_samples(
    records: Iterable[Mapping[str, object]],
) -> dict[UncertaintyChannel, list[tuple[float, float]]]:
    """Every ``(channel_value, label)`` pair the fit can actually use, per channel.

    Lifted out of :func:`fit_uncertainty_calibration` so that the count of usable samples has
    exactly one implementation. It is the number that separates a real fit from an identity map
    produced by an empty channel, and a caller that had to parse it out of a log line or re-derive
    the rule would be holding a second answer to one question.

    Three ways a record contributes nothing, all silent by design: no ``label``, a label outside
    ``[0, 1]``, or a channel value that is absent, non-numeric, or out of range. The replay fixture
    in this repository carries ``null`` for individual channels on individual records, so partial
    contribution is the normal case rather than an error.
    """
    per_channel: dict[UncertaintyChannel, list[tuple[float, float]]] = {
        ch: [] for ch in UncertaintyChannel
    }
    for rec in records:
        label = rec.get("label")
        if label is None:
            continue
        try:
            lab = float(label)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not 0.0 <= lab <= 1.0:
            continue
        for ch in UncertaintyChannel:
            raw = rec.get(ch.value)
            if raw is None:
                continue
            try:
                xv = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not 0.0 <= xv <= 1.0:
                continue
            per_channel[ch].append((xv, lab))
    return per_channel


def fit_uncertainty_calibration(
    records: Iterable[Mapping[str, object]],
    *,
    calibration_id: str | None = None,
) -> UncertaintyCalibration:
    """Fit a monotone calibration from a deterministic record stream (records need a ``label`` in ``[0, 1]``; unlabelled ones are skipped)."""

    records = list(records)
    per_channel = usable_samples(records)
    maps: dict[UncertaintyChannel, UncertaintyMonotoneMap] = {}
    for ch, samples in per_channel.items():
        maps[ch] = _fit_channel_map(samples)
    # One aggregated line, not one per channel inside the fit loop.
    logger.info(
        "Fitted uncertainty calibration id=%s from %d records; usable samples per channel: %s",
        calibration_id,
        len(records),
        ", ".join(f"{ch.value}={len(s)}" for ch, s in per_channel.items()),
    )
    empty = [ch.value for ch, s in per_channel.items() if not s]
    if empty:
        logger.warning(
            "Channels with no usable sample fell back to the identity map: %s",
            ", ".join(empty),
        )
    return UncertaintyCalibration(
        weights=UncertaintyWeights(),
        maps=maps,
        calibration_id=calibration_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.robot.grasping.calibration.uncertainty_calibration",
        description="Fit a monotone uncertainty calibration from a JSONL replay.",
    )
    parser.add_argument("--replay", required=True, help="Path to JSONL replay records.")
    parser.add_argument("--out", required=True, help="Path to write the JSON artifact.")
    parser.add_argument(
        "--calibration-id", default=None,
        help="Optional opaque calibration identifier embedded in the artifact.",
    )
    args = parser.parse_args(argv)
    # The handler is the noun. The fixture label rule is `LabelPolicy.FEASIBILITY_MARGIN_FIXTURE`,
    # passed explicitly here, so a Python caller can see it, choose it, or refuse it. Everybody else
    # gets the `REQUIRE` default; `fits.py` sets out why the two doors deliberately disagree.
    #
    # The import sits inside the function because `fits` imports this module for the fit itself, and
    # a top-level import here would close that loop.
    from src.robot.grasping.calibration.fits import LabelPolicy, UncertaintyFit

    report = UncertaintyFit.from_jsonl(
        args.replay,
        label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        calibration_id=args.calibration_id,
    ).fit()
    if not report.artifact_available:
        # What this CLI emits for a missing replay: one stderr line and exit 2.
        logger.error("Replay file not found: %s", Path(args.replay))
        print(report.detail, file=sys.stderr)
        return report.exit_code
    # stdout stays the single `wrote <path>` line, so nothing that parses this CLI moves. The
    # account the fit gives of itself, meaning invented labels, identity channels and the two
    # channels the runtime weights at 0.0, goes to the log, which for a tool that prints one line is
    # where a person looks.
    logger.info("%s", report.render())
    out_path = report.write(args.out)
    logger.info(
        # Stat'ed rather than `len(payload)`. `write_text` translates newlines on Windows, where
        # the length of the string is not the size of the file.
        "Wrote calibration artifact %s (%d bytes)", out_path, out_path.stat().st_size,
    )
    print(f"wrote {out_path}")
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
