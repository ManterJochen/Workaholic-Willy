"""When are two rows the same grasp? One answer, used by everything that has to agree on it.

Two places need this and they must not disagree. The corpus builder dedupes candidates so that one
grasp proposed by four eval configs weights an object once rather than four times. The physics
sampler dedupes before drawing, because `grasp_eval.jsonl` is appended across runs and holds one
copy per evaluation of a grasp, so a uniform draw over lines would weight a grasp by how many times
someone happened to re-evaluate it.

Without that, a shaken trial can be a later copy of a grasp whose first copy is the one the corpus
kept, and the two cannot be joined: the physics result then belongs to a row that carries no
features, and the trial is spent for nothing.

The quantisation is deliberately coarse, 1 mm and about 0.6 deg. Two candidates closer than that are
the same grasp for every purpose here: the jaw cannot tell them apart and neither can the label.
Exact float equality would call them different and put both in the corpus.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DEDUPE_DIRECTION", "DEDUPE_POSITION_MM", "grasp_identity"]

#: A grasp is a duplicate of another when position agrees to this and both directions to ~0.6 deg.
DEDUPE_POSITION_MM = 1.0
DEDUPE_DIRECTION = 0.01


def grasp_identity(row: dict[str, Any]) -> tuple:
    """The quantised identity of one candidate row. Equal identity means the same grasp.

    ``view`` participates, and it has to: the same grasp seen from two cameras is two observations
    with two different point clouds, so they are two training rows even though the pose is one. A
    row with no view, and the analytic reference's own labels carry none, falls into the empty view,
    which keeps this function total rather than raising on a file it was not written for.
    """
    position = np.asarray(row["position_mm"], dtype=np.float64) / DEDUPE_POSITION_MM
    approach = np.asarray(row["approach"], dtype=np.float64) / DEDUPE_DIRECTION
    axis = np.asarray(row["closing_axis"], dtype=np.float64) / DEDUPE_DIRECTION
    return (row["scene_id"], row.get("view", ""), row["instance_id"],
            *np.round(position).astype(int), *np.round(approach).astype(int),
            *np.round(axis).astype(int))
