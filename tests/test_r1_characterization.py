"""R1.0 — characterization golden for the grasping core (the safety net for the R1 decomposition).

The R1 keystone aggressively decomposes GraspCalculator (compute() 571 lines) and BinPickingOrchestrator
(_execute_pick 342 lines) into strategy objects. compute()/_execute_pick are NOT directly golden-pinned
elsewhere (only via ~38 behaviour-specific unit tests + the on-box picks), so this module snapshots their
FULL observable output (candidate geometry + ordering + telemetry + metadata keys) on fixed inputs and asserts
it against a committed golden. Any candidate-order, score, telemetry, or metadata drift during R1.1/R1.2 fails
here immediately, with a readable diff.

Regenerate the golden INTENTIONALLY (only when a behaviour change is approved) with:
    WILLY_R1_UPDATE_GOLDEN=1 python -m pytest tests/test_r1_characterization.py

REGENERATED ONCE, 2026-08-26 -- and the golden was ENCODING A DEFECT.
`_candidate_generator` assigned `width_mm = extent_minor_px` unconditionally, ABOVE the branch that
picks the closing axis. That branch closes across the MINOR direction for an elongated silhouette but
along the PRINCIPAL (MAJOR) one for a near-square mask, so on a near-square mask the commanded width
belonged to the wrong axis and BOTH CONTACTS LANDED INSIDE THE OBJECT. The `planar_box` fixtures are
55 x 45 px (aspect 0.818, the near-square branch), and this golden pinned `grip_width_mm = 45.0` -- the
minor extent, i.e. the defect. After the repair the same fixtures command 55.0, and `score` moves with
it (0.77691 -> 0.79098) because the score reads the width.

The regeneration diff was read field by field before accepting it: those two values on the three
`planar_box_*` cases and nothing else -- `pick_loop` unchanged, every other case unchanged. The
property itself is now pinned directly by `tests/test_silhouette_width_axis.py`, which asserts against
the real generator instead of a snapshot, so a future drift here has a second witness that says WHY.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping import GraspCalculator
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PerceptionFrame

_GOLDEN = Path(__file__).parent / "data" / "r1_characterization" / "grasping_core_golden.json"


# --------------------------------------------------------------------------- snapshot helpers
def _jsonable(v: object) -> object:
    """Coerce a telemetry/metadata value to a stable JSON-able form (rounded floats; type-name for objects)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, (int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return f"<{type(v).__name__}>"


def _snap_candidate(g: object) -> dict:
    return {
        "position_mm": [round(float(x), 3) for x in np.asarray(g.position)],  # type: ignore[attr-defined]
        "approach": [round(float(x), 4) for x in np.asarray(g.approach)],  # type: ignore[attr-defined]
        "axis": [round(float(x), 4) for x in np.asarray(g.axis)],  # type: ignore[attr-defined]
        "grip_width_mm": round(float(g.grip_width_mm), 3),  # type: ignore[attr-defined]
        "score": round(float(g.score), 5),  # type: ignore[attr-defined]
        "frame": getattr(getattr(g, "frame", None), "value", None),
        "label": getattr(g, "label", None),
        "metadata_keys": sorted(str(k) for k in (getattr(g, "metadata", None) or {})),
    }


# Telemetry keys whose VALUE is non-deterministic (wall-clock timing); the golden pins their PRESENCE +
# ordering but normalizes the flaky value so the safety net never goes flaky-red under CPU load.
_VOLATILE_TELEMETRY_KEYS = frozenset({"dense_runtime_ms"})


def _snap_grasps(grasps: list, telemetry: dict) -> dict:
    tel = {str(k): _jsonable(v) for k, v in sorted((telemetry or {}).items())}
    for k in _VOLATILE_TELEMETRY_KEYS:
        if k in tel:
            tel[k] = "<volatile>"
    return {
        "n_candidates": len(grasps),
        "candidates": [_snap_candidate(g) for g in grasps],
        "telemetry": tel,
    }


def _camera_matrix() -> np.ndarray:
    return np.array([[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _calculator() -> GraspCalculator:
    return GraspCalculator(min_grip_width_mm=1.0, max_grip_width_mm=200.0, max_candidates=3,
                           camera_matrix=_camera_matrix())


def _box_seg(shape: tuple[int, int] = (24, 24)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return SimpleNamespace(mask=mask, label="box")


# --------------------------------------------------------------------------- compute() scenarios
def _compute_snapshots() -> dict:
    """Run compute() across fixed scenarios that span the main paths; return name -> snapshot."""
    out: dict[str, dict] = {}
    seg = _box_seg()
    depth = np.full((24, 24), 1000.0, dtype=np.float64)

    calc = _calculator()
    out["planar_box_mm"] = _snap_grasps(
        calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm"), calc.last_telemetry or {}
    )

    calc = _calculator()
    out["planar_box_dense"] = _snap_grasps(
        calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm", dense_sampling=True),
        calc.last_telemetry or {},
    )

    calc = _calculator()
    cam_to_base = Transform(
        translation_mm=np.array([100.0, -50.0, 300.0], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        from_frame=Frame.CAMERA, to_frame=Frame.BASE,
    )
    out["planar_box_base_frame"] = _snap_grasps(
        calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm", camera_to_base=cam_to_base),
        calc.last_telemetry or {},
    )

    calc = _calculator()
    empty = SimpleNamespace(mask=np.zeros((24, 24), dtype=np.uint8), label="empty")
    out["empty_mask"] = _snap_grasps(
        calc.compute(empty, depth, pixel_to_mm=5.0, unit="mm"), calc.last_telemetry or {}
    )
    return out


# --------------------------------------------------------------------------- pick_loop scenario
class _FakeArm:
    def __init__(self) -> None:
        self.moved: list = []

    def move_to(self, pose: object) -> bool:  # noqa: ANN001
        self.moved.append(pose)
        return True


class _FakePerception:
    def __init__(self, frames: list) -> None:
        self._frames, self._i = frames, 0

    def acquire(self) -> object:
        f = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return f


def _pick_loop_snapshot() -> dict:
    """Characterize the orchestrator: a real GraspCalculator on a fixed perception frame -> PickReport."""
    seg = _box_seg((32, 32))
    depth = np.full((32, 32), 1000.0, dtype=np.float64)
    frame = PerceptionFrame(depth_map=depth, intrinsics=_camera_matrix(), segmentations=(seg,))
    orch = BinPickingOrchestrator(
        arm=_FakeArm(),  # type: ignore[arg-type]
        calculator=_calculator(),
        perception=_FakePerception([frame]),  # type: ignore[arg-type]
    )
    report = orch.run()
    pr = report
    return {
        "outcome": getattr(getattr(pr, "outcome", None), "value", str(getattr(pr, "outcome", None))),
        "target_index": getattr(pr, "target_index", None),
        "attempts": len(getattr(pr, "attempts", ()) or ()),
        "reasons": [str(r) for r in (getattr(pr, "reasons", ()) or ())],
    }


def _all_snapshots() -> dict:
    return {"compute": _compute_snapshots(), "pick_loop": _pick_loop_snapshot()}


class GraspingCoreCharacterizationTests(unittest.TestCase):
    """Pins the FULL observable output of compute() + the orchestrator; the R1 decomposition must reproduce it."""

    def test_matches_golden(self) -> None:
        snaps = _all_snapshots()
        if os.environ.get("WILLY_R1_UPDATE_GOLDEN"):
            _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            _GOLDEN.write_text(json.dumps(snaps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.skipTest("golden regenerated (WILLY_R1_UPDATE_GOLDEN)")
        self.assertTrue(_GOLDEN.exists(), f"missing golden {_GOLDEN}; regenerate with WILLY_R1_UPDATE_GOLDEN=1")
        golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        # Compare section-by-section for readable diffs on a regression.
        self.assertEqual(snaps["compute"], golden["compute"], "compute() output drifted vs the R1 golden")
        self.assertEqual(snaps["pick_loop"], golden["pick_loop"], "orchestrator output drifted vs the R1 golden")


if __name__ == "__main__":
    unittest.main()
