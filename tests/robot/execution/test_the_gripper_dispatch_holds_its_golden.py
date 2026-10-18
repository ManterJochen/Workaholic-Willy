"""What every hand-kept gripper dispatch says today, pinned before P0g slice 8 replaces them with one spec per vendor.

The verdict (``robot_parts.gripper_driver_verdict``), the build (``build_gripper``), the desk's three gripper rows
(``real_cell.preflight``) and the console's connect warnings (``api.lifecycle.motion_warnings``) each branch on
``robot.gripper.vendor`` by hand. Slice 8 makes them lookups on a frozen
``GripperDriver`` spec beside each driver, and must not change one answer: this golden was generated from the code
before that slice (P0g slice 3) and committed alone, so the slice is compared against what the old code said, not
against what the new code says about itself.

A built gripper is recorded by its type, the core gripper Protocols it satisfies, and every attribute it holds, with
the arm it was handed, its config and its loggers named rather than printed, so the golden holds on any machine.

Regenerate INTENTIONALLY (an approved change only): WILLY_P0G_UPDATE_GOLDEN=1 python -m pytest
tests/robot/execution/test_the_gripper_dispatch_holds_its_golden.py
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import logging
import os
import unittest
from pathlib import Path
from typing import Any

from src.config.schema.robot import RobotConfig
from src.robot.core import gripper as core_gripper
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.drivers.dummy.arm import DummyRobotArm

_GOLDEN = Path(__file__).parents[2] / "data" / "p0g_gripper_dispatch" / "gripper_dispatch_golden.json"

#: Every gripper block the dispatches branch on: each vendor, and each jaw_io and vacuum wiring they read.
_GRIPPERS: dict[str, dict[str, Any]] = {
    "none": {"vendor": "none"},
    "dummy": {"vendor": "dummy"},
    "robotiq": {"vendor": "robotiq"},
    "vacuum, ejector only": {"vendor": "vacuum", "vacuum": {"vacuum_output_pin": 0}},
    "vacuum, blow-off and switch on the tool bank": {"vendor": "vacuum", "vacuum": {
        "vacuum_output_pin": 0, "blow_off_output_pin": 1, "vacuum_ok_input_pin": 1, "io_port": "tool"}},
    "jaw_io, single solenoid": {"vendor": "jaw_io", "jaw_io": {"close_output_pin": 4, "io_port": "standard"}},
    "jaw_io, single solenoid, end-stop switches": {"vendor": "jaw_io", "jaw_io": {
        "close_output_pin": 4, "part_present_input_pin": 5, "closed_confirm_input_pin": 6, "io_port": "standard"}},
    "jaw_io, single solenoid, opens blind on connect": {"vendor": "jaw_io", "jaw_io": {
        "close_output_pin": 4, "open_on_connect_without_feedback": True, "io_port": "standard"}},
    "jaw_io, single solenoid that asks where its jaws stand": {"vendor": "jaw_io", "jaw_io": {
        "close_output_pin": 4, "confirm_open_at_start": True, "io_port": "standard"}},
    "jaw_io, double solenoid on the configurable bank": {"vendor": "jaw_io", "jaw_io": {
        "actuation": "double_solenoid", "close_output_pin": 4, "open_output_pin": 5, "io_port": "configurable"}},
    "jaw_io, toggle on the tool bank": {"vendor": "jaw_io", "jaw_io": {
        "actuation": "single_toggle", "close_output_pin": 0, "io_port": "tool"}},
    "onrobot": {"vendor": "onrobot"},
    "onrobot, own address and force": {"vendor": "onrobot", "onrobot": {
        "host": "10.0.0.7", "port": 5020, "unit_id": 66, "default_force_n": 40.0, "use_fingertip_offset": True}},
    "franka_hand (reserved)": {"vendor": "franka_hand"},
    "schunk (reserved)": {"vendor": "schunk"},
    #: The schema refuses this name, so it reaches the verdict only past validation (``model_copy``), as a caller that
    #: builds its config object by hand hands it in.
    "an unknown vendor, past the schema": {"vendor": "robotiq", "past the schema": "acme"},
}

#: The trees a gripper sits in: a UR that names its controller, a UR that names none, and an arm with no controller.
_TREES: dict[str, dict[str, Any]] = {
    "ur at 192.168.1.10": {"vendor": "ur", "ur": {"ip": "192.168.1.10", "model": "ur5e"}},
    "ur with no address": {"vendor": "ur", "ur": {"model": "ur5e"}},
    "dummy": {"vendor": "dummy"},
}


class _SwitchingDummy(DummyRobotArm):
    """An arm that is not a UR and switches pins: the case where the IO branches decide by the handle alone."""

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        pass

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return False

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return False

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        pass


def _config(tree: str, gripper: str) -> RobotConfig:
    # A UR arm's exact-mesh guard reads the hand by its name and refuses a tree that names none, so every block names
    # the one hand both command tapes name; the dispatches branch on the vendor, never on this.
    block = {"model": "robotiq_2f85", **_GRIPPERS[gripper]}
    forced = block.pop("past the schema", None)
    cfg = RobotConfig.model_validate({**_TREES[tree], "gripper": block})
    if forced is None:
        return cfg
    return cfg.model_copy(update={"gripper": cfg.gripper.model_copy(update={"vendor": forced})})


def _arms(tree: str, cfg: RobotConfig) -> dict[str, Any]:
    arms: dict[str, Any] = {"dummy arm": DummyRobotArm(), "a dummy that switches pins": _SwitchingDummy()}
    if cfg.vendor == "ur":
        from src.robot.drivers.ur.arm import URRobotArm

        arms["the tree's UR arm"] = URRobotArm(cfg)
    return arms


def _plain(value: Any, *, arm: Any = None, cfg: Any = None) -> Any:
    """A value as JSON with nothing machine- or run-dependent in it."""
    if arm is not None and value is arm:
        return "<the arm it was handed>"
    if cfg is not None and value is cfg:
        return "<robot.gripper>"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, Path):
        return f"path:{value.name}"
    if isinstance(value, logging.Logger):
        return "<logger>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__,
                **{f.name: _plain(getattr(value, f.name), arm=arm, cfg=cfg) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {str(k): _plain(v, arm=arm, cfg=cfg) for k, v in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return {type(value).__name__: [_plain(v, arm=arm, cfg=cfg) for v in items]}
    if inspect.isroutine(value):
        return f"<routine {getattr(value, '__qualname__', type(value).__name__)}>"
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _protocols() -> dict[str, type]:
    return {name: value for name, value in sorted(vars(core_gripper).items())
            if inspect.isclass(value) and getattr(value, "_is_runtime_protocol", False)}


def _gripper(gripper: Any, *, arm: Any, cfg: Any) -> dict[str, Any]:
    return {
        "type": f"{type(gripper).__module__}.{type(gripper).__qualname__}",
        "satisfies": [name for name, proto in _protocols().items() if isinstance(gripper, proto)],
        "holds": {name: _plain(value, arm=arm, cfg=cfg) for name, value in sorted(vars(gripper).items())},
    }


def _verdicts() -> dict[str, Any]:
    from src.robot.execution.robot_parts import gripper_driver_verdict

    out: dict[str, Any] = {}
    for gripper in _GRIPPERS:
        cfg = _config("ur at 192.168.1.10", gripper)
        for vendor in ("ur", "dummy", "sim", "franka"):
            for io in (True, False):
                for ip in ("192.168.1.10", None):
                    verdict = gripper_driver_verdict(cfg, arm_vendor=vendor, arm_has_digital_io=io, arm_ip=ip)
                    out[f"{gripper} | arm {vendor}, io {io}, ip {ip}"] = _plain(verdict)
    return out


def _builds_desk_and_console() -> dict[str, Any]:
    from api.lifecycle import motion_warnings
    from src.robot.execution.real_cell import preflight as desk
    from src.robot.execution.robot_parts import build_gripper

    builds: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    for tree in _TREES:
        for gripper in _GRIPPERS:
            cfg = _config(tree, gripper)
            key = f"{gripper} | {tree}"
            rows[key] = {
                "gripper driver": _plain(desk._gripper_driver_row(cfg)),
                "end-effector wiring": _plain(desk._end_effector_wiring_row(cfg)),
                "jaw travel time": _plain(desk._jaw_travel_time_row(cfg)),
            }
            for label, arm in _arms(tree, cfg).items():
                built = build_gripper(cfg, arm=arm)
                builds[f"{key} | {label}"] = {
                    "gripper": _gripper(built, arm=arm, cfg=cfg.gripper),
                    "console warnings": _plain(motion_warnings(cfg, built)),
                }
    return {"build": builds, "desk": rows}


def _snapshot() -> dict[str, Any]:
    return {"verdict": _verdicts(), **_builds_desk_and_console()}


class TheGripperDispatchHoldsItsGoldenTests(unittest.TestCase):
    def test_every_dispatch_says_what_it_said_before_slice_8(self) -> None:
        snapshot = _snapshot()
        if os.environ.get("WILLY_P0G_UPDATE_GOLDEN"):
            _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            _GOLDEN.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            self.skipTest("golden regenerated (WILLY_P0G_UPDATE_GOLDEN)")
        self.assertTrue(_GOLDEN.exists(), f"missing golden {_GOLDEN}; regenerate with WILLY_P0G_UPDATE_GOLDEN=1")
        golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(sorted(golden), sorted(snapshot))
        for section in golden:
            with self.subTest(section=section):
                self.assertEqual(sorted(golden[section]), sorted(snapshot[section]), "a case came or went")
                for case, said in golden[section].items():
                    self.assertEqual(said, snapshot[section][case], case)

    def test_the_matrix_reaches_every_branch(self) -> None:
        """The positive control: a golden over a matrix that misses a branch pins nothing about that branch."""
        from src.robot.grippers import SubstitutionReason

        snapshot = _snapshot()
        reasons = {v["reason"] for v in snapshot["verdict"].values()}
        self.assertEqual({None, *(f"SubstitutionReason.{r.name}" for r in SubstitutionReason)}, reasons)
        built = {entry["gripper"]["type"].rsplit(".", 1)[-1] for entry in snapshot["build"].values()}
        self.assertLessEqual({"NullGripper", "DummyGripper", "VacuumGripper", "JawIOGripper"}, built)
        self.assertGreaterEqual(sum(1 for e in snapshot["build"].values() if e["console warnings"]["tuple"]), 1)


if __name__ == "__main__":
    unittest.main()
