"""Optional semantic / task-aware policy for grasp selection."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SemanticDecision",
    "SemanticPolicy",
]


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    """Outcome of evaluating a candidate against a :class:`SemanticPolicy`.

    ``reason`` is a stable lowercase denial code, :data:`None` exactly
    when ``accept`` is :data:`True`.
    """

    accept: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticPolicy:
    """Allow-list / deny-list / confidence-threshold policy for grasps.

    ``allow_labels`` (when non-empty) admits only listed labels;
    ``deny_labels`` always rejects (deny wins on conflict);
    ``min_label_confidence`` (``None`` disables) gates the segmentation
    score.
    """

    allow_labels: frozenset[str] = field(default_factory=frozenset)
    deny_labels: frozenset[str] = field(default_factory=frozenset)
    min_label_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow_labels", frozenset(self.allow_labels))
        object.__setattr__(self, "deny_labels", frozenset(self.deny_labels))
        if self.min_label_confidence is not None:
            value = float(self.min_label_confidence)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "min_label_confidence must be in [0.0, 1.0]; "
                    f"got {self.min_label_confidence}"
                )
            object.__setattr__(self, "min_label_confidence", value)

    @property
    def is_noop(self) -> bool:
        """``True`` when the policy imposes no constraints at all."""
        return (
            not self.allow_labels
            and not self.deny_labels
            and self.min_label_confidence is None
        )

    def evaluate(
        self, label: str | None, score: float | None
    ) -> SemanticDecision:
        """Decide whether a candidate with ``(label, score)`` is accepted.

        First failing gate wins (deterministic reason code):
        ``deny_labels``, then ``allow_labels`` (a ``None`` label is not
        in the allow-list), then ``min_label_confidence`` (missing
        scores fail closed).
        """
        if label is not None and label in self.deny_labels:
            return SemanticDecision(False, "label_in_deny_list")
        if self.allow_labels:
            if label is None or label not in self.allow_labels:
                return SemanticDecision(False, "label_not_in_allow_list")
        if self.min_label_confidence is not None:
            if score is None or float(score) < self.min_label_confidence:
                return SemanticDecision(False, "label_confidence_below_threshold")
        return SemanticDecision(True, None)
