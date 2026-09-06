"""Fitting a calibration as a noun: what went in, what came out, and what came out inert.

The CLI can invent the labels, and the fit cannot tell.
``uncertainty_calibration.main`` reads a replay, and if no record carries a ``label`` it derives one
for every record from ``feasibility_margin > 0.5`` before handing them to
:func:`~src.robot.grasping.calibration.uncertainty_calibration.fit_uncertainty_calibration`. The fit
then sees a labelled corpus, and nothing downstream can tell that corpus apart from one an operator
labelled by hand: not the artifact, not the log, not the exit code. This module is where a Python
caller can ask for that rule or refuse it, because :class:`LabelPolicy` names it instead of an
argparse handler holding it.

Two consequences follow from that rule, and the report names both. ``feasibility_margin`` is then
fitted against labels derived from ``feasibility_margin``, so its map is the label rule reading
itself back and is indistinguishable in the artifact from a channel that predicts well. And the
``or 0.0`` coalesce turns a record whose margin is missing into a label of 0, so some negatives mean
the channel was absent rather than that the grasp failed.

Two of the seven fitted channels are multiplied by zero. ``UncertaintyWeights()`` ships
``topology_risk=0.0`` and ``semantic_confidence=0.0``, so whatever those maps learn contributes
nothing to the fused score, however well they fitted. The weights are in the artifact, so this is
not hidden, but a report is where somebody reading a fit looks for it.

Nothing here duplicates the fit. The fit itself is ``fit_uncertainty_calibration``, called once and
unchanged, and the usable-sample rule is ``usable_samples``, lifted out of that function rather than
written a second time.

The identity map has two opposite causes, which is why both numbers are reported.
``_fit_channel_map`` returns the identity for a channel with no usable sample, and a channel whose
values separate the labels perfectly also fits to exactly ``breakpoints=(0, 1), values=(0, 1)``,
which is the identity. Nothing learned and everything learned are the same object. So
``identity_channels`` is derived from the maps and ``starved_channels`` from the counts, and only
the pair distinguishes them: ``NOTHING_LEARNED`` is decided on starvation and never on identity,
because deciding it on identity would file the best possible fit as the worst.

The bytes and exit codes of the CLI do not move. ``main`` prints ``wrote <path>`` and returns 0 for
the fixture path and 2 for a missing replay. The verdicts here, an invented label set or a fit where
nothing was learned, are readable through :meth:`UncertaintyFitReport.render` and turn into a
non-zero exit only under ``strict=True``, which the CLI does not pass. That mirrors
:class:`~src.robot.grasping.rl.datasets.LeakageAudit`, where the same lever is the only thing that
can call a passing audit vacuous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.contracts import UNSET, Maybe, resolve
from src.robot.grasping.calibration.uncertainty_calibration import (
    fit_uncertainty_calibration,
    usable_samples,
)
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannel,
    UncertaintyMonotoneMap,
    UncertaintyWeights,
)

__all__ = [
    "ChannelFit",
    "FitVerdict",
    "LabelPolicy",
    "UncertaintyFit",
    "UncertaintyFitReport",
]

#: The channel the fixture label rule reads. Named once so the rule and the warning about the rule
#: cannot drift apart.
_FIXTURE_LABEL_CHANNEL = "feasibility_margin"


class LabelPolicy(StrEnum):
    """Where the ``label`` comes from when a replay does not carry one.

    ``REQUIRE`` is the default for every Python caller, and it is the opposite of what the CLI
    does. That asymmetry is deliberate and it is the whole point: a tool run from a shell against a
    shipped fixture should still produce a well-formed artifact, and a library call that silently
    invented its own ground truth would be the more dangerous of the two.
    """

    #: Records must carry their own ``label``. A replay with none is refused, not labelled.
    REQUIRE = "require"

    #: The fixture path. If, and only if, no record carries a ``label``, derive one from
    #: ``feasibility_margin > 0.5``. This is the rule ``uncertainty_calibration.main`` applies,
    #: including its condition: a replay where even one record is labelled is used as it stands and
    #: the rest of its records stay unlabelled.
    FEASIBILITY_MARGIN_FIXTURE = "feasibility_margin_fixture"


class FitVerdict(StrEnum):
    """What the fit was, in the terms somebody deciding whether to ship it would use."""

    #: Records carried their own labels and at least one channel learned a non-identity map.
    FITTED = "fitted"

    #: The labels were invented by :attr:`LabelPolicy.FEASIBILITY_MARGIN_FIXTURE`. The artifact is
    #: well-formed and it is not a calibration of anything that happened in a cell.
    FITTED_FROM_INVENTED_LABELS = "fitted_from_invented_labels"

    #: Not one channel had a usable sample, so all seven maps are the identity. The artifact is a
    #: no-op the runtime will load and apply without complaint.
    NOTHING_LEARNED = "nothing_learned"

    #: ``LabelPolicy.REQUIRE`` and not one record carried a ``label``.
    NO_LABELS = "no_labels"

    #: The replay file does not exist.
    REPLAY_MISSING = "replay_missing"

    #: The replay file exists and is not line-delimited JSON.
    REPLAY_MALFORMED = "replay_malformed"


@dataclass(frozen=True, slots=True)
class ChannelFit:
    """One channel's outcome: how much data it had, what it produced, and whether it counts.

    ``span`` is ``max(values) - min(values)`` of the fitted map. It answers the question the sample
    count cannot: a channel with plenty of samples whose map barely moves across the whole input
    range is not distinguishing anything, and it looks identical to a good fit in every other
    summary.
    """

    channel: str
    samples: int
    span: float
    identity: bool
    weight: float

    @property
    def inert(self) -> bool:
        """Fitted and then multiplied by zero. Two of the seven ship this way."""
        return self.weight == 0.0

    @property
    def starved(self) -> bool:
        """No record contributed a usable pair, so nothing was fitted at all."""
        return self.samples == 0

    @property
    def identity_cause(self) -> str:
        """Why the identity, since the map alone cannot say. ``_fit_channel_map`` returns the
        identity for an empty channel, and a real fit over a perfectly separating channel lands on
        exactly ``breakpoints=(0, 1), values=(0, 1)``, so the best possible outcome and the worst
        possible outcome are the same object. The sample count is the only discriminator, which is
        why this report carries it.
        """
        if not self.identity:
            return ""
        return "no usable sample" if self.starved else "fitted, and it came out as the identity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "samples": self.samples,
            "span": self.span,
            "identity": self.identity,
            "identity_cause": self.identity_cause,
            "weight": self.weight,
            "inert": self.inert,
            "starved": self.starved,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyFitReport:
    """What one fit produced, as one frozen object with an exit code on it."""

    verdict: FitVerdict
    label_policy: LabelPolicy
    source: str
    total_records: int
    labelled_records: int
    invented_labels: int
    strict: bool = False
    channels: tuple[ChannelFit, ...] = ()
    calibration: UncertaintyCalibration | None = None
    detail: str = ""

    # ---------------------------------------------------------------- derived
    @property
    def identity_channels(self) -> tuple[str, ...]:
        """Channels whose map came out as the identity, derived from the maps rather than the counts."""
        return tuple(c.channel for c in self.channels if c.identity)

    @property
    def starved_channels(self) -> tuple[str, ...]:
        """Channels no record could contribute to. They are the half of :attr:`identity_channels`
        that is a problem, the other half being a fit that happened to land on the identity."""
        return tuple(c.channel for c in self.channels if c.starved)

    @property
    def inert_channels(self) -> tuple[str, ...]:
        """Channels the runtime weights at 0.0, however well they fitted."""
        return tuple(c.channel for c in self.channels if c.inert)

    @property
    def ok(self) -> bool:
        """Did a usable artifact come out of this? Under ``strict``, "usable" is a higher bar."""
        if self.verdict in (FitVerdict.FITTED,):
            return True
        if self.verdict in (
            FitVerdict.FITTED_FROM_INVENTED_LABELS,
            FitVerdict.NOTHING_LEARNED,
        ):
            return not self.strict
        return False

    @property
    def exit_code(self) -> int:
        """0 fit, 2 the replay could not be read, 3 a refusal or a strict rejection.

        ``NOTHING_LEARNED`` and ``FITTED_FROM_INVENTED_LABELS`` exit 0 without ``strict``, because
        the CLI exits 0 for both, an artifact is genuinely written, and moving the code would break
        every caller that reads it. :meth:`render` is where those two verdicts are said out loud.
        """
        if self.verdict is FitVerdict.REPLAY_MISSING:
            return 2
        if self.verdict is FitVerdict.REPLAY_MALFORMED:
            return 2
        if self.verdict is FitVerdict.NO_LABELS:
            return 3
        return 0 if self.ok else 3

    @property
    def artifact_available(self) -> bool:
        """Whether :meth:`write` has anything to write. A non-zero exit does not answer this: a
        strict rejection still has a fitted calibration, it just should not be shipped."""
        return self.calibration is not None

    # ---------------------------------------------------------------- output
    def render(self) -> str:
        lines = [
            "uncertainty calibration fit",
            f"  verdict         : {self.verdict.value}",
            f"  source          : {self.source}",
            f"  label policy    : {self.label_policy.value}",
            f"  records         : {self.total_records} "
            f"({self.labelled_records} labelled, {self.invented_labels} invented)",
        ]
        if self.detail:
            lines.append(f"  detail          : {self.detail}")
        if self.channels:
            lines.append("  channels (samples / span of the fitted map / runtime weight):")
            width = max(len(c.channel) for c in self.channels)
            for c in self.channels:
                marks = []
                if c.identity:
                    marks.append(f"IDENTITY - {c.identity_cause}")
                if c.inert:
                    marks.append("INERT - weight 0.0, the runtime discards this map")
                mark = ("   <- " + "; ".join(marks)) if marks else ""
                lines.append(
                    f"    {c.channel.ljust(width)}  n={c.samples:<5d} span={c.span:.3f} "
                    f"weight={c.weight:.1f}{mark}"
                )
        if self.verdict is FitVerdict.FITTED_FROM_INVENTED_LABELS:
            lines.append(
                f"  WARNING: no record carried a label, so every label was derived from "
                f"{_FIXTURE_LABEL_CHANNEL} > 0.5. The {_FIXTURE_LABEL_CHANNEL} map is therefore "
                f"fitted against itself, and a missing {_FIXTURE_LABEL_CHANNEL} reads as a failure."
            )
        if self.verdict is FitVerdict.NOTHING_LEARNED:
            lines.append(
                "  WARNING: not one channel had a usable sample, so all seven are the identity map. "
                "The artifact is well-formed and applying it changes no score."
            )
        if self.inert_channels and self.verdict not in (
            FitVerdict.NO_LABELS, FitVerdict.REPLAY_MISSING, FitVerdict.REPLAY_MALFORMED,
        ):
            lines.append(
                "  NOTE: " + ", ".join(self.inert_channels) + " ship at weight 0.0, so those fits "
                "do not reach the fused score. Channel weights are not learned here."
            )
        lines.append(f"  exit code       : {self.exit_code}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The wire half. ``artifact`` is the calibration's own serialiser, never a second one."""
        return {
            "verdict": self.verdict.value,
            "label_policy": self.label_policy.value,
            "source": self.source,
            "total_records": self.total_records,
            "labelled_records": self.labelled_records,
            "invented_labels": self.invented_labels,
            "strict": self.strict,
            "channels": [c.to_dict() for c in self.channels],
            "identity_channels": list(self.identity_channels),
            "starved_channels": list(self.starved_channels),
            "inert_channels": list(self.inert_channels),
            "detail": self.detail,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "artifact": None if self.calibration is None else self.calibration.to_artifact(),
        }

    def write(self, path: str | Path) -> Path:
        """Write the artifact, in the bytes the CLI writes.

        ``write_text`` and not ``write_bytes``, deliberately. On Windows that translates the
        newlines, so the file is larger than ``len(payload)``, and switching to bytes would move
        every byte of every artifact on that platform. The size reported back is stat'ed after the
        write rather than measured before it, so it is the size of the file.
        """
        if self.calibration is None:
            raise ValueError(
                f"nothing to write: the fit ended in {self.verdict.value} and produced no "
                "calibration. Check `artifact_available` first."
            )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.calibration.to_artifact(), indent=2, sort_keys=True))
        return out


@dataclass(frozen=True, slots=True)
class UncertaintyFit:
    """A replay plus a labelling policy. One verb: :meth:`fit`."""

    records: tuple[Mapping[str, object], ...]
    label_policy: LabelPolicy
    calibration_id: str | None
    strict: bool
    source: str
    load_error: tuple[FitVerdict, str] | None = None

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, object]],
        *,
        label_policy: Maybe[LabelPolicy] = UNSET,
        calibration_id: Maybe[str | None] = UNSET,
        strict: Maybe[bool] = UNSET,
        source: Maybe[str] = UNSET,
    ) -> "UncertaintyFit":
        """The Python door: records already in hand.

        The records are not mutated. Writing ``label`` onto each dict in place is invisible when the
        caller is argparse and a surprise when the caller owns the list, so the fixture label rule
        builds a new dict per record instead.
        """
        return cls(
            records=tuple(records),
            label_policy=resolve("label_policy", label_policy, LabelPolicy.REQUIRE),
            calibration_id=resolve("calibration_id", calibration_id, None),
            strict=resolve("strict", strict, False),
            source=resolve("source", source, "records passed in Python"),
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        label_policy: Maybe[LabelPolicy] = UNSET,
        calibration_id: Maybe[str | None] = UNSET,
        strict: Maybe[bool] = UNSET,
    ) -> "UncertaintyFit":
        """The file door: resolve the path into records, then call the Python door.

        One builder. Every default lives in :meth:`from_records`; this method declares none of
        them, so the two doors cannot disagree about what ``strict`` or ``label_policy`` mean.

        A file that is missing or malformed becomes a verdict rather than an exception, because both
        are ordinary operator conditions and both used to reach the caller as a traceback or as an
        exit code with nothing attached.
        """
        p = Path(path)
        failure: tuple[FitVerdict, str] | None = None
        records: list[Mapping[str, object]] = []
        try:
            with p.open(encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            failure = (FitVerdict.REPLAY_MISSING, f"replay file not found: {p}")
        except (OSError, json.JSONDecodeError) as exc:
            failure = (FitVerdict.REPLAY_MALFORMED, f"could not read {p}: {exc}")
        fit = cls.from_records(
            records,
            label_policy=label_policy,
            calibration_id=calibration_id,
            strict=strict,
            source=str(p),
        )
        # `replace` rather than a second constructor: the failure is not something a caller supplies,
        # and adding it to `from_records` would put a parameter on the public door that only this
        # method is ever allowed to pass.
        return fit if failure is None else replace(fit, load_error=failure)

    # ---------------------------------------------------------------- the verb
    def fit(self) -> UncertaintyFitReport:
        """Fit the calibration, or say precisely why there is none."""
        if self.load_error is not None:
            verdict, detail = self.load_error
            return self._report(verdict, detail=detail)

        labelled = sum(1 for r in self.records if "label" in r)
        records: tuple[Mapping[str, object], ...] = self.records
        invented = 0
        if labelled == 0 and self.label_policy is LabelPolicy.FEASIBILITY_MARGIN_FIXTURE:
            records = tuple(
                # A new dict per record: the same rule `main` applies, without writing into the
                # caller's data. The `or 0.0` coalesce is kept verbatim, including its consequence,
                # which is that a null margin becomes a label of 0.
                {**r, "label": 1.0 if (r.get(_FIXTURE_LABEL_CHANNEL) or 0.0) > 0.5 else 0.0}  # type: ignore[operator]
                for r in self.records
            )
            invented = len(records)
        elif labelled == 0:
            return self._report(
                FitVerdict.NO_LABELS,
                detail=(
                    f"not one of {len(self.records)} records carries a 'label', and the policy is "
                    f"{LabelPolicy.REQUIRE.value}. Pass "
                    f"label_policy=LabelPolicy.{LabelPolicy.FEASIBILITY_MARGIN_FIXTURE.name} to "
                    f"accept the fixture rule, or label the replay."
                ),
            )

        calibration = fit_uncertainty_calibration(records, calibration_id=self.calibration_id)
        samples = usable_samples(records)
        weights = calibration.weights.to_dict()
        identity = UncertaintyMonotoneMap.identity()
        channels = tuple(
            ChannelFit(
                channel=ch.value,
                samples=len(samples[ch]),
                span=(max(calibration.maps[ch].values) - min(calibration.maps[ch].values)),
                # Derived from the map. A channel with samples can still collapse to the
                # identity, and a count-based answer would call that one a success.
                identity=calibration.maps[ch] == identity,
                weight=float(weights.get(ch.value, 0.0)),
            )
            for ch in UncertaintyChannel
        )
        # Starved, not `identity`. A run where every channel produced the identity map because every
        # channel was perfectly separable is the best fit this tool can produce, and calling it
        # "nothing learned" would be exactly backwards.
        if all(c.starved for c in channels):
            verdict = FitVerdict.NOTHING_LEARNED
        elif invented:
            verdict = FitVerdict.FITTED_FROM_INVENTED_LABELS
        else:
            verdict = FitVerdict.FITTED
        return self._report(
            verdict,
            channels=channels,
            calibration=calibration,
            labelled=labelled,
            invented=invented,
        )

    # ---------------------------------------------------------------- internal
    def _report(
        self,
        verdict: FitVerdict,
        *,
        detail: str = "",
        channels: tuple[ChannelFit, ...] = (),
        calibration: UncertaintyCalibration | None = None,
        labelled: int = 0,
        invented: int = 0,
    ) -> UncertaintyFitReport:
        return UncertaintyFitReport(
            verdict=verdict,
            label_policy=self.label_policy,
            source=self.source,
            total_records=len(self.records),
            labelled_records=labelled,
            invented_labels=invented,
            strict=self.strict,
            channels=channels,
            calibration=calibration,
            detail=detail,
        )


def default_weights() -> UncertaintyWeights:
    """The weights every fit here ships with. Re-exported so a reader of a report can see that the
    two zeros in it are a runtime default rather than something this fit chose."""
    return UncertaintyWeights()
