"""Opt-in numpy tier of the Tier-0 online accumulator (``robot.rl.runtime_tier='numpy'``).

Lazily imported by :func:`online_state.make_online_accumulator`, never on the default import
path: numpy stays off the ship path and its pure-stdlib guarantee holds. For the diagonal-LinUCB
update the numpy tier is an exact mirror of the stdlib accumulator: the deltas are
integer/float-exact, so it snapshots to a byte-identical :class:`online_state.OnlineState` with the
same content and chain hash. It exercises the tier seam and cross-checks that switching tiers does
not change results; it is not a stronger learner.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .online_state import (
    ONLINE_TIER_NUMPY,
    OnlineState,
    OnlineStateError,
    _content_hash,
    fold_update_chain,
)


class NumpyLinUCBOnlineAccumulator:
    """numpy-backed diagonal-LinUCB online accumulator (exact mirror of the stdlib tier)."""

    tier: str = ONLINE_TIER_NUMPY

    def __init__(self, *, dim: int, actions: Sequence[str]) -> None:
        if dim <= 0:
            raise OnlineStateError(f"dim must be positive; got {dim}")
        self._dim = int(dim)
        self._actions = tuple(actions)
        if len(set(self._actions)) != len(self._actions):
            raise OnlineStateError("actions must be unique")
        self._delta_A = {a: np.zeros(self._dim, dtype=np.float64) for a in self._actions}
        self._delta_b = {a: np.zeros(self._dim, dtype=np.float64) for a in self._actions}
        self._delta_support = {a: np.zeros(self._dim, dtype=np.int64) for a in self._actions}
        self._n_updates = 0
        self._high_water = 0
        self._chain = ""

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def high_water_mark(self) -> int:
        return self._high_water

    def observe(self, *, onehot: Sequence[int], action: str, reward: float) -> None:
        if action not in self._delta_A:
            raise OnlineStateError(f"action={action!r} not in {self._actions!r}")
        if len(onehot) != self._dim:
            raise OnlineStateError(f"onehot length {len(onehot)} != dim {self._dim}")
        x = np.asarray([int(v) for v in onehot], dtype=np.int64)
        if not np.all((x == 0) | (x == 1)):
            raise OnlineStateError("onehot entries must be 0 or 1")
        r = float(reward)
        self._delta_A[action] += x.astype(np.float64)  # x_i^2 == x_i for a one-hot
        self._delta_b[action] += r * x.astype(np.float64)
        self._delta_support[action] += x
        self._n_updates += 1
        self._high_water = max(self._high_water, self._n_updates)
        self._chain = fold_update_chain(
            self._chain, action=action, reward=r, onehot=onehot
        )

    def snapshot(self) -> OnlineState:
        dA = {a: tuple(float(v) for v in self._delta_A[a]) for a in self._actions}
        db = {a: tuple(float(v) for v in self._delta_b[a]) for a in self._actions}
        ds = {a: tuple(int(v) for v in self._delta_support[a]) for a in self._actions}
        content = _content_hash(
            actions=self._actions,
            dim=self._dim,
            delta_A=dA,
            delta_b=db,
            delta_support=ds,
            n_updates=self._n_updates,
            high_water_mark=self._high_water,
            update_chain_hash=self._chain,
            still_shadow=True,
        )
        return OnlineState(
            actions=self._actions,
            dim=self._dim,
            delta_A_diag_per_action=dA,
            delta_b_per_action=db,
            delta_support_per_action=ds,
            n_updates=self._n_updates,
            high_water_mark=self._high_water,
            update_chain_hash=self._chain,
            content_hash=content,
            still_shadow=True,
        )


__all__ = ("NumpyLinUCBOnlineAccumulator",)
