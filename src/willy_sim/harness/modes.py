"""Grasp-mode selection for the Isaac sim runners.

One place that maps a demo-mode string to the typed :class:`GraspMode`, so every pick runner and the
mode-matrix harness read the modes the same way:

* ``easy``: the deterministic trust path.
* ``auto``: the automatic sampler. The decision gate it ran before every grasp left on 2026-10-16 with
  ``robot.grasping.decision``.
* ``dense_clutter``: the profile switches the sampler to the dense point-cloud one and enables itself on a
  multi-object frame. Meaningful only on a clutter scene.

``closed_loop`` and ``dense_autonomous`` were removed on 2026-09-29 with the two-scan pre-grasp
refinement they ran; a runner asked for either refuses with the mode to name instead.

Every heavy import is local, so this module imports where ``isaacsim`` is absent.
"""

from __future__ import annotations

from typing import Any

#: The modes the single-object scenes can meaningfully run.
DEMO_MODES: tuple[str, ...] = ("easy", "auto")
#: Modes the dense-clutter scene can run.
DENSE_MODES: tuple[str, ...] = ("easy", "auto", "dense_clutter")


def resolve_demo_mode(mode: Any) -> Any:
    """Map a demo-mode string (or a ``GraspMode``) to the typed :class:`GraspMode`.

    Accepts ``easy`` / ``auto`` / ``dense_clutter`` (and ``-`` separators). A mode removed on purpose
    raises ``ValueError`` with the sentence that names its replacement, and anything else raises
    ``ValueError`` too, so a typo never silently runs the wrong sampler.
    """

    from src.config.schema._removed import REMOVED_GRASP_MODES
    from src.robot.execution.autonomous_grasp.config import GraspMode

    if isinstance(mode, GraspMode):
        return mode
    key = str(mode).strip().lower().replace("-", "_")
    table = {
        "easy": GraspMode.EASY,
        "auto": GraspMode.AUTO,
        "dense_clutter": GraspMode.DENSE_CLUTTER,
    }
    if key in REMOVED_GRASP_MODES:
        raise ValueError(f"demo mode {mode!r} is removed on purpose: {REMOVED_GRASP_MODES[key]}")
    if key not in table:
        raise ValueError(
            f"unsupported demo mode {mode!r}; supported: {DENSE_MODES}"
        )
    return table[key]


__all__ = ["DEMO_MODES", "DENSE_MODES", "resolve_demo_mode"]
