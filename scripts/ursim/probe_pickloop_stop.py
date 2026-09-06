"""The same protective stop one layer up: a real arm, the real pick loop, and its verdict.

The unit tests pin the logic against fakes. This runs the same path with `URRobotArm` connected to
URSim and the controller genuinely stopped, which is the only way to know the capability check
fires where it matters. Perception and the calculator are stand-ins, because URSim has no camera,
but the arm, the execution policy and the motion are real.

The stand-ins have no library twin and should not grow one. `src/robot/grasping/scene.py` takes a
point cloud and returns ranked grasps, which is the opposite of what this needs: a controlled,
known-reachable target 10 mm above the current TCP. `src/robot/execution/cell.py` is the wrong
altitude for every file in this directory: its preflight blocks on a camera-to-base resolver that
URSim has no camera to satisfy, and building it would construct perception the container cannot
supply. These are driver probes; `Cell` is a pick cell.

This is the only probe here that defaults to the UR3e chain, because the singular height it drives
to is the one `probe_protective_stop.py` tabulates per model, and the UR3e reaches the shorter one.

Run (URSim up, in REMOTE control):  python scripts/ursim/probe_pickloop_stop.py

Exit codes: 0 the loop reported the cell rather than a failed grasp, 1 it did not, or no stop was
provoked, 2 no controller reachable, or a profile that is not a UR cell.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root. `scripts/examples/_common.py` does the same insert for the examples.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# And this directory, so the singular-height table is read from the probe that explains it rather
# than declared a second time here. Two declarations of one height is how the two files came to
# disagree by 10 mm. `scripts/examples/pipeline/80_full_pipeline.py` imports its siblings the same way.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
from probe_protective_stop import SINGULAR_Z_MM  # noqa: E402

from src.config.loader import active_profile, load_robot_config  # noqa: E402
from src.robot.core import RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome  # noqa: E402
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy  # noqa: E402
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult  # noqa: E402
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint  # noqa: E402
from src.robot.grasping.types.perception import PerceptionFrame  # noqa: E402

#: The chain this probe needs when the operator has not named one. It is resolved here rather than
#: written into `WILLY_PROFILE`, which would leak process-global state into whatever runs next. It
#: is not left to the loader's own default either: the base tree points `ur.ip` at a plant address,
#: and a probe that deliberately provokes a protective stop must never reach a real cell.
_DEFAULT_PROFILE = "ursim,ursim_ur3"

_OK, _FAILED, _BAD_REQUEST = 0, 1, 2


@dataclass
class _Seg:
    mask: np.ndarray


class _Perception:
    """A fixed frame with one square mask. URSim has no camera, so nothing here is perceived."""

    def acquire(self) -> PerceptionFrame:
        m = np.zeros((8, 8), dtype=bool)
        m[2:6, 2:6] = True
        return PerceptionFrame(
            depth_map=np.full((8, 8), 400.0),
            intrinsics=np.array([[500.0, 0, 4.0], [0, 500.0, 4.0], [0, 0, 1.0]]),
            segmentations=(_Seg(mask=m),),
        )


class _Calculator:
    """One candidate at a fixed base-frame point, so the loop's verdict is about the cell."""

    render_debug_images = False

    def __init__(self, pos: np.ndarray) -> None:
        self._pos = np.asarray(pos, dtype=float)

    def compute_result(self, *_a: Any, **_k: Any) -> GraspResult:
        gp = GraspPoint(position=self._pos, approach=np.array([0.0, 0.0, -1.0]),
                        axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9,
                        frame=GraspFrame.BASE)
        return GraspResult(candidates=(gp,), top_score=0.9)


def _provoke(arm: URRobotArm, model: str) -> None:
    """Drive to the singular height through the raw connection, below the guard.

    `URConnection.moveL` is a published seam and it sits below `SafetyPreflight`, which is the
    point: `arm.move()` gates the same target and would refuse it, so the controller would never
    reach the state this probe measures. Nothing needs to be nulled to get here. `_preflight` is
    not consulted on this path at all, and clearing it would protect nothing while risking an
    ungated arm if the body raised before the restore.
    """
    z_m = SINGULAR_Z_MM.get(model, SINGULAR_Z_MM["ur5e"]) / 1000.0
    try:
        arm.connection.moveL([0.0, 0.0, z_m, 0.0, 3.14, 0.0], 0.9, 1.2)
    except Exception as exc:  # noqa: BLE001
        print(f"   raw moveL: {str(exc)[:60]}")
    time.sleep(3.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python scripts/ursim/probe_pickloop_stop.py",
        description="Run the real pick loop against a genuinely protective-stopped controller and "
                    "check that it reports the cell rather than a failed grasp.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain; defaults to the UR3e URSim chain")
    args = ap.parse_args(argv)

    chain = args.profile if args.profile is not None else (active_profile() or _DEFAULT_PROFILE)
    cfg = load_robot_config(profile=chain)
    built = create_arm(RobotVendor.from_string(cfg.vendor), config=cfg)
    if not isinstance(built, URRobotArm):
        print(f"profile {chain!r} builds {type(built).__name__}; this probe reads UR-only seams")
        return _BAD_REQUEST
    arm = built
    try:
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"arm.connect() failed: {type(exc).__name__}: {exc}")
        print("no controller reachable: is URSim up and in REMOTE control?")
        return _BAD_REQUEST

    try:
        here = arm.get_tcp_pose().position_mm
        print(f"connected on {chain!r}; TCP at {[round(float(v), 1) for v in here]}")

        orch = BinPickingOrchestrator(
            arm=arm,
            # The orchestrator's `perception` field is a Protocol and `_Perception` satisfies it.
            # Its `calculator` field names the concrete `GraspCalculator` instead, so no stand-in
            # can satisfy it structurally even though the loop only calls `compute_result` and
            # reads `render_debug_images`.
            calculator=_Calculator(  # type: ignore[arg-type]
                np.asarray(here, dtype=float) + np.array([0.0, 0.0, 10.0])),
            perception=_Perception(),
            policy=GraspExecutionPolicy(arm=arm, gripper=None, standoff_mm=20.0, retreat_mm=20.0),
            max_attempts=1,
        )

        print("\n=== A. healthy controller ===")
        rep = orch.run()
        reasons = rep.attempts[-1].reasons if rep.attempts else ()
        print(f"   outcome={rep.outcome.value}  reasons={[str(r) for r in reasons]}")

        print("\n=== B. provoke a real protective stop ===")
        _provoke(arm, cfg.ur.model)
        st = arm.get_robot_status()
        print(f"   controller now: safety={st.safety_mode.value} "
              f"protective={st.protective_stopped} operational={st.is_operational}")
        if not st.is_stopped:
            # Asserting against a healthy cell would print FAIL for the wrong reason, and the
            # reader would go looking for a bug in the loop that the loop does not have.
            print("\n   NO STOP WAS PROVOKED: the move did not trip the controller's safety "
                  "system, so there is nothing for section C to measure.")
            return _FAILED

        print("\n=== C. the same pick, with the cell stopped ===")
        rep = orch.run()
        reasons = rep.attempts[-1].reasons if rep.attempts else ()
        print(f"   outcome={rep.outcome.value}")
        print(f"   reasons={[str(r) for r in reasons]}")
        # Compared against the enum members rather than against their spellings, so a renamed
        # value is a type error here rather than a silent pass.
        ok = (rep.outcome == PickOutcome.CONTROLLER_NOT_OPERATIONAL
              and GraspFailureReason.CONTROLLER_NOT_OPERATIONAL in reasons)
        print(f"\n   {'PASS' if ok else 'FAIL'}: the loop reports the cell, not a failed grasp")
        return _OK if ok else _FAILED
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 (teardown must not mask what the probe measured)
            pass
        print("PICKLOOP_STOP_DONE")


if __name__ == "__main__":
    raise SystemExit(main())
