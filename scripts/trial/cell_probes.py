"""Cell probes for the customer chain trial: the numbers a named hand brought, and an arm that builds.

    .venv/Scripts/python.exe scripts/trial/cell_probes.py derived --profile ur10e,acme_dims_ur10e
    .venv/Scripts/python.exe scripts/trial/cell_probes.py build-arm --profile ur10e,acme_dims_ur10e

A trial instrument, not product code. The planner start is the product's own,
``python -m src.robot.execution.real_cell --start-planner``, and the trial runs that.

``derived`` loads the chain and holds every one of the thirteen keys a named hand determines to its registry file.
``build-arm`` constructs the UR driver from the chain without connecting, which
resolves the hand, its sphere map and its plates and builds the guard pipeline, and says what it built.

Exit codes: 0 yes, 1 no with the sentence, 2 the question cannot be asked.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass(frozen=True)
class DerivedReport:
    profile: str
    hand: str
    #: (robot key, loaded value, registry value) for every key that differs.
    differing: tuple[tuple[str, Any, Any], ...]
    checked: int
    refusal: str = ""

    @property
    def exit_code(self) -> int:
        return 0 if not (self.differing or self.refusal) else 1

    def render(self) -> str:
        if self.refusal:
            return f"{self.profile}: {self.refusal}"
        if not self.differing:
            return f"{self.profile}: all {self.checked} hand keys are {self.hand}'s registry numbers"
        lines = [f"{self.profile}: {len(self.differing)} of {self.checked} hand keys are not {self.hand}'s"]
        lines += [f"  robot.{key} = {loaded!r}, the registry says {wanted!r}" for key, loaded, wanted in self.differing]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "hand": self.hand, "checked": self.checked, "refusal": self.refusal,
                "differing": [list(row) for row in self.differing], "exit_code": self.exit_code}


def derived(profile: str, *, data_dir: "str | None" = None) -> DerivedReport:
    from src.config.hand_numbers import HandNumbers
    from src.config.loader import ConfigError, load_robot_config, reload_config

    reload_config()
    try:
        robot = load_robot_config(data_dir, profile=profile)
    except ConfigError as exc:
        return DerivedReport(profile=profile, hand="", differing=(), checked=0, refusal=str(exc))
    if not robot.gripper.model:
        return DerivedReport(profile=profile, hand="", differing=(), checked=0,
                             refusal="robot.gripper.model is unset, so no hand supplies any number")
    numbers = HandNumbers.from_registry(model=robot.gripper.model)
    differing = []
    for key, field, wanted in numbers.values:
        node: Any = robot
        for part in key.split("."):
            node = getattr(node, part)
        loaded = str(node) if field == "kind" else float(node)
        if loaded != wanted:
            differing.append((key, loaded, wanted))
    return DerivedReport(profile=profile, hand=robot.gripper.model, differing=tuple(differing),
                         checked=len(numbers.values))


def build_arm(profile: str, *, data_dir: "str | None" = None) -> "tuple[int, str]":
    from src.config.loader import ConfigError, load_robot_config, reload_config

    reload_config()
    try:
        cfg = load_robot_config(data_dir, profile=profile)
        from src.robot.drivers import create_arm

        arm: Any = create_arm("ur", config=cfg)
    except ConfigError as exc:
        return 1, f"{profile}: REFUSED at build: {exc}"
    guard = arm.safety_preflight
    hand = guard.planner_hand(arm) if guard is not None else None
    return 0, (f"{profile}: built {type(arm).__name__} for {cfg.ur.model}, hand {getattr(hand, 'model', hand)}, "
               f"self collision backend {cfg.safety.self_collision.backend}, not connected")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Cell probes for the customer chain trial.")
    verbs = parser.add_subparsers(dest="verb", required=True)
    for name in ("derived", "build-arm"):
        verb = verbs.add_parser(name)
        verb.add_argument("--profile", required=True)
        verb.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    if args.verb == "derived":
        report = derived(args.profile, data_dir=args.data_dir)
        print(report.render())
        return report.exit_code
    code, said = build_arm(args.profile, data_dir=args.data_dir)
    print(said)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
