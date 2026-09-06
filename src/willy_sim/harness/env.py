"""Centralized parsing of the ``WILLY_*`` runner environment knobs.

:class:`RunnerEnv` parses the 17 runner knobs once at boot into a typed, frozen struct, so a call
site reads ``env.field`` rather than ``os.environ.get(...)``. That puts every knob and its default
in one place, and it is where the two context-dependent defaults are computed: ``ycb_cam_z``
depends on ``vision``, and ``c1_standoff_mm`` defaults to the eye-in-hand ``view_height_mm``.

Scope is the runner layer only. The two perception debug-trace flags (``WILLY_TRACE_DEPTH`` and
``WILLY_TRACE_VISION``) are read where they are used, in :mod:`src.willy_sim.perception`, so that
the runners which build the same perception sources without any other knob need no
:class:`RunnerEnv`.

With no ``WILLY_*`` set, every field takes the documented default and the cell is the shipped one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import RUNNER_ENV_LOG_FILE, WILLY_SIM_LOG_DIR

__all__ = ["RunnerEnv"]

_TRUE_TOKENS = ("1", "true", "yes")

#: The struct is a value object and logs nothing itself. The one fact it cannot carry is whether any
#: knob was set at all, and every field below moves a number that changes a result, so a run made
#: with shell overrides would otherwise be indistinguishable afterwards from one made clean.
_LOG = create_logger("RunnerEnv", log_file=RUNNER_ENV_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)


@dataclass(frozen=True, slots=True)
class RunnerEnv:
    """Parsed ``WILLY_*`` runner knobs (built once via :meth:`from_env`)."""

    # YCB scene selection (read in _ycb_specs)
    ycb_scene: str            # WILLY_YCB_SCENE (stripped and lowered; "" selects the demo scene)
    ycb_solo: bool            # WILLY_YCB_SOLO (truthiness)
    ycb_target_y: float | None  # WILLY_YCB_TARGET_Y (override target spawn Y; None keeps it)
    # Failure-injection substrate (read in _apply_rl3_substrate)
    rl3_substrate: str        # WILLY_RL3_SUBSTRATE (stripped/lowered)
    rl3_big_mm: float         # WILLY_RL3_BIG_MM (oversized-target size; used only when substrate set)
    # YCB grasp/scene tuning (read in build_service)
    ycb_cam_z: float          # WILLY_YCB_CAM_Z (context default: 1200 if vision else 1800)
    ycb_penetration: float    # WILLY_YCB_PENETRATION (top-grasp penetration mm)
    ycb_detector: str         # WILLY_YCB_DETECTOR (HF model id for the vision path)
    ycb_close: float | None   # WILLY_YCB_CLOSE (fixed close-width override; None stays adaptive)
    ycb_squeeze: float        # WILLY_YCB_SQUEEZE (adaptive close squeeze margin)
    ik_debug: bool            # WILLY_IK_DEBUG (per-candidate IK cond/min_sv/joint_margin log)
    # Wrist tracker (read in the eye-in-hand build_service)
    c1_world_space_tracker: bool  # WILLY_C1_WORLD_SPACE_TRACKER (membership 1/true/yes)
    c1_standoff_mm: float     # WILLY_C1_STANDOFF_MM (context default: the EIH view_height_mm)
    # opt-in debug traces
    trace_moves: bool         # WILLY_TRACE_MOVES (per-move target/closing/status)
    trace_cands: bool         # WILLY_TRACE_CANDS (all ranked candidates pre-execution)
    trace_calc: bool          # WILLY_TRACE_CALC (per-pick calculator telemetry + geometry)
    trace_g12: bool           # WILLY_TRACE_G12 (approach-sweep obstacle cloud and per-candidate verdict)

    @classmethod
    def from_env(cls, *, vision: bool, view_height_mm: float) -> RunnerEnv:
        """Parse the runner knobs from ``os.environ``.

        ``vision`` selects the ``ycb_cam_z`` default: 1200 with real vision, 1800 with ground-truth
        masks. ``view_height_mm`` is the ``c1_standoff_mm`` default, the wrist view height of the
        eye-in-hand cell.
        """
        get = os.environ.get
        _ty = get("WILLY_YCB_TARGET_Y")
        _close = get("WILLY_YCB_CLOSE")
        # Reported from ``os.environ`` rather than from the parsed fields, so no second copy of the
        # defaults has to be kept in step. Deliberately every ``WILLY_*``, not only this struct's
        # knobs: ``WILLY_COAL_PREFIX`` decides whether the mesh guard can exist at all, and the two
        # perception trace flags are read elsewhere. The line records what was in this shell.
        _set = sorted(k for k in os.environ if k.startswith("WILLY_"))
        _LOG.info(
            "runner knobs (vision=%s view_height_mm=%.1f): %s", vision, view_height_mm,
            ", ".join(f"{k}={os.environ[k]!r}" for k in _set) if _set
            else "no WILLY_* set; the documented defaults",
        )
        return cls(
            ycb_scene=get("WILLY_YCB_SCENE", "").strip().lower(),
            ycb_solo=bool(get("WILLY_YCB_SOLO")),
            ycb_target_y=float(_ty) if _ty is not None else None,
            rl3_substrate=get("WILLY_RL3_SUBSTRATE", "").strip().lower(),
            rl3_big_mm=float(get("WILLY_RL3_BIG_MM", "120.0")),
            ycb_cam_z=float(get("WILLY_YCB_CAM_Z", "1200.0" if vision else "1800.0")),
            ycb_penetration=float(get("WILLY_YCB_PENETRATION", "12.0")),
            ycb_detector=get("WILLY_YCB_DETECTOR", "IDEA-Research/grounding-dino-base"),
            ycb_close=float(_close) if _close is not None else None,
            ycb_squeeze=float(get("WILLY_YCB_SQUEEZE", "11.0")),
            ik_debug=bool(get("WILLY_IK_DEBUG")),
            c1_world_space_tracker=get("WILLY_C1_WORLD_SPACE_TRACKER", "").strip().lower() in _TRUE_TOKENS,
            c1_standoff_mm=float(get("WILLY_C1_STANDOFF_MM", str(view_height_mm))),
            trace_moves=bool(get("WILLY_TRACE_MOVES")),
            trace_cands=bool(get("WILLY_TRACE_CANDS")),
            trace_calc=bool(get("WILLY_TRACE_CALC")),
            trace_g12=bool(get("WILLY_TRACE_G12")),
        )
