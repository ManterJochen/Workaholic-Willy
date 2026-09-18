"""docs/runbooks/your_own_gripper.md walks the customer chain, every command parses, and the trial plan covers it.

Customer chain lanes C1k and C5f (merged by the lane decisions of 2026-09-17). A runbook is only as true as the scripts
it names: a flag renamed in a parser, a rotation spelled with commas where the parser takes four numbers, or a step moved
before the file it reads, and the customer meets a traceback the documentation promised away. So every command is run
through the parser of the script it names, the order is held to the order the inputs exist in, and the trial plan that
executes the runbook in a copy of the tree has to reference every step or carry a written reason not to.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shlex
import sys
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
_RUNBOOK = _REPO / "docs" / "runbooks" / "your_own_gripper.md"
_PLAN = _REPO / "scripts" / "trial" / "customer_hand_trial.json"

#: The order the chain's inputs exist in: each step reads what an earlier one wrote.
_ORDER = ("registry-file", "registry-validate", "dims-write", "mesh-write", "usd-write", "spheres-write",
          "retract-choose", "retract-check", "evidence-gate", "cell-layer", "verify-desk", "verify-doctor",
          "verify-planner-start", "verify-ik-desk")


def _runner():
    name = "_trial_run_runbook_for_the_runbook"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "trial" / "run_runbook.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _steps(text: "str | None" = None):
    return _runner().parse_runbook(_RUNBOOK.read_text(encoding="utf-8") if text is None else text)


def _order_refusals(steps) -> list[str]:
    position = {step.id: index for index, step in enumerate(steps)}
    missing = [name for name in _ORDER if name not in position]
    late = [f"{a} after {b}" for a, b in zip(_ORDER, _ORDER[1:])
            if a in position and b in position and position[a] > position[b]]
    return [f"missing {name}" for name in missing] + late


class _Parsed(Exception):
    """Raised after a successful parse, so nothing past the parser runs."""


def _sample_bindings() -> dict[str, str]:
    plan = json.loads(_PLAN.read_text(encoding="utf-8"))
    bound = {"ROOT": "D:/copy", "LOGS": "D:/logs", "USD": "D:/assets/hand.usd", "BODIES": "base,left,right"}
    bound.update(plan["bindings"])
    for phase in plan["phases"]:
        bound.update(phase.get("bindings") or {})
    return bound


def _parses(command: str) -> "tuple[bool, str]":
    """Whether ``command`` gets through the parser of the script or module it names, and what happened otherwise."""
    words = shlex.split(command, posix=True)
    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):  # noqa: ANN001, ANN202
        original(self, args, namespace)
        raise _Parsed

    if words[1] == "-m":
        module = importlib.import_module(f"{words[2]}.__main__")
        argv = words[3:]
    else:
        path = _REPO / words[1]
        name = f"_runbook_parse_{path.stem}"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(path.parent))
            try:
                sys.modules[name] = module
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(path.parent))
        module = sys.modules[name]
        argv = words[2:]
    with mock.patch.object(argparse.ArgumentParser, "parse_args", parse_then_stop), \
            mock.patch.object(sys, "argv", [words[1], *argv]):
        try:
            module.main(argv)
        except _Parsed:
            return True, ""
        except SystemExit as exc:
            return False, f"SystemExit({exc.code!r})"
    return False, "main returned without parsing"


class TheRunbookWalksTheChainTests(unittest.TestCase):
    def test_the_runbook_walks_the_chain_in_order(self) -> None:
        self.assertEqual(_order_refusals(_steps()), [])

    def test_the_order_check_sees_a_swapped_step(self) -> None:
        """⭐ THE CONTROL: the map moved after the chooser, which reads it, is named."""
        steps = _steps()
        moved = [step for step in steps if step.id != "spheres-write"]
        at = next(index for index, step in enumerate(moved) if step.id == "retract-choose") + 1
        moved.insert(at, next(step for step in steps if step.id == "spheres-write"))
        self.assertIn("spheres-write after retract-choose", _order_refusals(moved))

    def test_every_chooser_and_gate_command_carries_the_cells_declared_rotation(self) -> None:
        by_id = {step.id: step for step in _steps()}
        for name in ("retract-choose", "retract-check", "evidence-gate"):
            with self.subTest(step=name):
                self.assertIn("--tool-rotation-xyzw $ROT", _runner().command_of(by_id[name]))


class EveryCommandParsesTests(unittest.TestCase):
    def test_every_project_venv_command_in_it_parses(self) -> None:
        runner = _runner()
        bound = _sample_bindings()
        commands = [runner.substitute(runner.command_of(step), bound) for step in _steps() if step.kind == "bash"]
        self.assertGreaterEqual(len(commands), 15)
        for command in commands:
            with self.subTest(command=command[:90]):
                ok, said = _parses(command)
                self.assertTrue(ok, said)

    def test_a_flag_no_parser_defines_is_caught(self) -> None:
        """⭐ THE CONTROL: the patch does not swallow a parser error."""
        ok, said = _parses("python scripts/grippers/write_hand_from_dimensions.py acme_2f --no-such-flag")
        self.assertFalse(ok)
        self.assertIn("2", said)
        ok, _ = _parses("python scripts/curobo/choose_ur_retract.py ur10e --tool-rotation-xyzw 0,0,0,1")
        self.assertFalse(ok)


class ThePlanCoversTheRunbookTests(unittest.TestCase):
    @staticmethod
    def _plan_refusals(steps, plan: dict) -> list[str]:
        said = []
        used = {step.get("runbook") for phase in plan["phases"] for step in phase["steps"]}
        said += [f"runbook step {step.id} is in no phase and carries no trial-skip reason"
                 for step in steps if step.id not in used and not step.skip]
        hidden: list[str] = []
        for phase in plan["phases"]:
            for step in phase["steps"]:
                if "hide" in step:
                    hidden.append(step["hide"])
                    continue
                if "restore" in step:
                    if step["restore"] in hidden:
                        hidden.remove(step["restore"])
                    continue
                expect = step.get("expect") or {}
                name = step.get("runbook") or step.get("id")
                if not any(expect.get(key) for key in ("stdout_has", "stderr_has", "files_exist")):
                    said.append(f"plan step {name} in {phase['name']} expects no text and no file")
                if "run" in step or ("runbook" in step and not step["runbook"].endswith(("-file", "-layer"))):
                    if "exit" not in expect:
                        said.append(f"plan step {name} in {phase['name']} expects no exit code")
        said += [f"{path} is hidden and never restored" for path in hidden]
        return said

    def test_the_plan_covers_the_runbook_and_every_plan_step_expects_something(self) -> None:
        steps = _steps()
        plan = json.loads(_PLAN.read_text(encoding="utf-8"))
        self.assertEqual(self._plan_refusals(steps, plan), [])
        runner = _runner().Runner(runbook_path=str(_RUNBOOK), steps=steps, plan=plan, root=Path("D:/copy"),
                                  logs=Path("D:/logs"))
        runner.check()
        index = json.loads((_REPO / "docs" / "runbooks" / "runbooks_index.json").read_text(encoding="utf-8"))
        self.assertIn("docs/runbooks/your_own_gripper.md", {entry["path"] for entry in index["runbooks"]})

    def test_a_plan_missing_a_step_or_an_expectation_is_named(self) -> None:
        """⭐ THE CONTROL."""
        steps = _steps()
        plan = json.loads(_PLAN.read_text(encoding="utf-8"))
        for phase in plan["phases"]:
            phase["steps"] = [step for step in phase["steps"] if step.get("runbook") != "spheres-ruler"]
        plan["phases"][0]["steps"].append({"id": "nothing", "run": "python -c pass", "expect": {"exit": 0}})
        said = self._plan_refusals(steps, plan)
        self.assertTrue(any("spheres-ruler" in line for line in said), said)
        self.assertTrue(any("nothing" in line for line in said), said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
