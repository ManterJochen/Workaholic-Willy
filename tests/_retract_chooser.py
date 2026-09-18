"""Run ``scripts/curobo/choose_ur_retract.py`` into a temporary table with its three judges standing in (not a test).

The chooser needs Coal for the exact meshes, a GPU sidecar for the planner and minutes of search. What a customer
depends on around those judges is plain code: which hands and plates a run judges, what it refuses before starting
anything, and what it writes into the table. The registry, the bundles, the maps, the merge and the YAML stay real;
the exact judge calls every pose clear except the folded elbow, the planner admits everything, and the rule returns
the anchor.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
CUROBO = REPO / "scripts" / "curobo"
TABLE = REPO / "src" / "robot" / "safety" / "planning" / "robot" / "ur_retract.yaml"


def chooser() -> Any:
    if str(CUROBO) not in sys.path:
        sys.path.insert(0, str(CUROBO))
    spec = importlib.util.spec_from_file_location("_chooser_under_test", CUROBO / "choose_ur_retract.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Run:
    code: int
    out: str
    table: Path
    judges_built: list = field(default_factory=list)
    planners_built: list = field(default_factory=list)


def run(argv: list[str], *, maps: Path | None = None, planner_refuses: tuple = (), moved: tuple = ()) -> Run:
    """``main(argv)`` into a temporary copy of the committed table. ``planner_refuses`` holds (arm, hand) pairs whose
    planner cannot be asked; ``moved`` holds (arm, hand) pairs whose rule answers a pose one wrist step off the anchor."""
    module = chooser()
    folder = Path(tempfile.mkdtemp(prefix="retract_chooser_"))
    table = folder / "ur_retract.yaml"
    shutil.copy(TABLE, table)
    judges: list = []
    planners: list = []

    class FakeExact:
        engine = "coal"

        def __init__(self, arm: str, hand: str, plate_mm: float, placement: Any) -> None:
            judges.append((arm, hand, plate_mm))

        def nearest(self, joints: list) -> tuple:
            return (0.0, "forearm|upper_arm") if list(joints) == list(module.FOLDED) else (40.0, "wrist_1|gripper")

        def lowest_z_mm(self, joints: list) -> float:
            return 200.0

    class FakePlanner:
        def __init__(self, arm: str, hand: str, plate_mm: float, rotation: Any, margin: float) -> None:
            if (arm, hand) in planner_refuses:
                raise RuntimeError(f"no descriptor for {arm} on this box")
            planners.append((arm, hand, plate_mm))

        def admits(self, poses: list) -> list:
            return [True] * len(poses)

        def depth_at(self, pose: list) -> tuple:
            return None, None

        def close(self) -> None:
            pass

    def fake_choose(arm: str, anchor: Any, hands: list, **_: Any) -> Any:
        retract = list(anchor)
        steps = (0, 0, 0, 0, 0)
        if (arm, hands[0].hand) in moved:
            retract[4] += 0.25
            steps = (0, 0, 0, 1, 0)
        return SimpleNamespace(anchor=tuple(anchor), steps=steps, retract=retract, tried=1)

    out = io.StringIO()
    patches = [
        mock.patch.object(module, "TABLE", table),
        mock.patch.object(module, "ExactJudge", FakeExact),
        mock.patch.object(module, "PlannerJudge", FakePlanner),
        mock.patch.object(module, "choose", fake_choose),
        mock.patch.object(module, "planner_envelope_rad", lambda: ((-6.0,) * 6, (6.0,) * 6)),
    ]
    if maps is not None:
        patches.append(mock.patch.object(module, "MAPS", maps))
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        stack.enter_context(contextlib.redirect_stdout(out))
        stack.enter_context(contextlib.redirect_stderr(out))
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
            out.write(str(exc.code))
    return Run(code=int(code), out=out.getvalue(), table=table, judges_built=judges, planners_built=planners)
