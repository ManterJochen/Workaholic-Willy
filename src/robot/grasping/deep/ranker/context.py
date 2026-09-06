"""Loading the learned ranker once per cell, so a pick never touches a file.

The pick loop holds one of these or ``None``. ``None`` is the default and means nothing is scored,
which is the byte-identical path. Building one reads the artifact and resolves the spec once; a pick
that opened a 600 KB JSON would be paying for an opinion it is not even allowed to act on.

Never raises when the artifact is missing or wrong: `from_config` returns ``None`` and logs why. A
cell configured to watch a ranker it cannot load runs its picks exactly as it would without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.robot.grasping.constants import (
    DEEP_RANKER_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.deep.ranker.features import FeatureSpec, spec_named
from src.robot.grasping.deep.ranker.runtime import GbtRanker, load_ranker

__all__ = ["DeepRankerContext"]

logger = create_grasping_logger("DeepRankerContext", DEEP_RANKER_LOG_FILE)


@dataclass(frozen=True, slots=True)
class DeepRankerContext:
    """A loaded ranker and the contract it was fitted under, ready to score."""

    ranker: GbtRanker
    spec: FeatureSpec

    @classmethod
    def from_config(cls, config: Any) -> "DeepRankerContext | None":
        """Build from a ``GraspingDeepRankerConfig``, or ``None`` with a logged reason.

        The spec and the artifact are checked for agreement at build time, not at the first pick, so
        a configuration error costs one log line instead of a refusal on every attempt.
        """
        if config is None or not getattr(config, "enabled", False):
            return None
        artifact = Path(getattr(config, "artifact_dir", "")) / f"{config.spec}.json"
        if not artifact.is_file():
            logger.warning(
                "deep_ranker is enabled but %s does not exist; nothing will be scored. "
                "Regenerate it with `python -m datagen train-ranker`.", artifact)
            return None
        try:
            spec = spec_named(str(config.spec))
            ranker = load_ranker(artifact)
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("deep_ranker could not load %s: %s: %s; nothing will be scored",
                           artifact, type(exc).__name__, exc)
            return None
        if tuple(ranker.features) != tuple(spec.features):
            # Refused rather than reordered. A model whose columns are permuted against its spec
            # scores confidently and means nothing, and there is no way to tell from the number.
            logger.warning(
                "deep_ranker artifact %s was fitted on %s but spec %r declares %s; refusing to "
                "score a vector the model was never fitted on",
                artifact, list(ranker.features), spec.name, list(spec.features))
            return None
        logger.info("deep_ranker ready: spec %r, %d trees, sha %s (shadow; changes no decision)",
                    spec.name, len(ranker.trees), ranker.sha256[:16])
        return cls(ranker=ranker, spec=spec)
