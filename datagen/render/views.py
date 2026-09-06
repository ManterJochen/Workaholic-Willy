"""What to do when the arm cannot reach a wrist viewpoint.

The wrist camera is the only view that has to be earned: the arm must physically get there, and on a
random hemisphere it often cannot. The answer depends on the rig.

When the wrist is the only view, the viewpoint is resampled until one is reachable. Skipping would
leave the scene with no image at all, which is not a datapoint.

When other views exist, the wrist view is skipped, the scene is kept, and the reason is recorded. The
other cameras still produced usable data, and resampling would bias the viewpoint distribution
towards whatever the arm finds easy: a dataset whose wrist views are only ever the comfortable angles
teaches a model that the awkward ones do not happen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum

__all__ = ["UnreachablePolicy", "ViewOutcome", "resolve_wrist_viewpoint", "unreachable_policy"]


class UnreachablePolicy(StrEnum):
    """What an unreachable wrist viewpoint causes."""

    RESAMPLE = "resample"
    SKIP_VIEW = "skip_view"


class ViewOutcome(StrEnum):
    """How a view ended up. Recorded per view, so a thin dataset explains itself."""

    RENDERED = "rendered"
    #: Reachable only after several re-draws: the viewpoint distribution is biased for this scene,
    #: and the record says so.
    RENDERED_AFTER_RESAMPLE = "rendered_after_resample"
    SKIPPED_UNREACHABLE = "skipped_unreachable"


def unreachable_policy(views: Sequence[str]) -> UnreachablePolicy:
    """Resample only when the wrist view is the whole rig; otherwise skip it."""
    others = [view for view in views if view != "wrist"]
    return UnreachablePolicy.SKIP_VIEW if others else UnreachablePolicy.RESAMPLE


def resolve_wrist_viewpoint(
    views: Sequence[str],
    sample: Callable[[int], object],
    reachable: Callable[[object], bool],
    *,
    max_attempts: int = 12,
) -> tuple[object | None, ViewOutcome, int]:
    """Find a reachable wrist viewpoint, or give up the way the policy says.

    Returns ``(viewpoint, outcome, attempts)``. ``sample(attempt)`` draws a candidate and takes the
    attempt number, so the caller can keep its own generator deterministic: a resample must be
    reproducible like everything else in this generator.
    """
    first = sample(0)
    if reachable(first):
        return first, ViewOutcome.RENDERED, 1
    if unreachable_policy(views) is UnreachablePolicy.SKIP_VIEW:
        return None, ViewOutcome.SKIPPED_UNREACHABLE, 1
    for attempt in range(1, max(1, max_attempts)):
        candidate = sample(attempt)
        if reachable(candidate):
            return candidate, ViewOutcome.RENDERED_AFTER_RESAMPLE, attempt + 1
    return None, ViewOutcome.SKIPPED_UNREACHABLE, max(1, max_attempts)
