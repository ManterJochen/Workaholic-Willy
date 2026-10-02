"""The push's URSim probe drives an arm, so it never reaches a real cell (fix plan, Track P).

``scripts/ursim/probe_push_on_the_mat.py`` runs a cell's own tree against URSim: it copies the tree, adds one layer
of its own (``ursim_mat``: ``robot.ur.ip`` 127.0.0.1 and nothing else), and loads the chain it is given. A chain that
does not name that layer, and a chain whose address resolves to anything but 127.0.0.1 (a later layer naming the
cell's controller again), are refused before anything connects: no RTDE interface, no dashboard socket, no arm.
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import socket
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "ursim" / "probe_push_on_the_mat.py"


def _probe() -> Any:
    spec = importlib.util.spec_from_file_location("_probe_push_on_the_mat", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Reached(AssertionError):
    """Something tried to reach a controller."""


class TheProbeRefusesBeforeAnythingConnectsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = _probe()
        self.dir = Path(tempfile.mkdtemp(prefix="push_probe_"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.tree = self.dir / "cell" / "config"
        shutil.copytree(REPO / "config", self.tree)
        self.reached: list[str] = []

    def _run(self, *argv: str) -> tuple[int, str]:
        """The probe's main, with every door to a controller replaced by one that records and raises."""

        def reached(name: str) -> Any:
            def door(*_args: Any, **_kwargs: Any) -> Any:
                self.reached.append(name)
                raise _Reached(name)
            return door

        fake_receive = types.ModuleType("rtde_receive")
        fake_receive.RTDEReceiveInterface = reached("RTDEReceiveInterface")  # type: ignore[attr-defined]
        fake_control = types.ModuleType("rtde_control")
        fake_control.RTDEControlInterface = reached("RTDEControlInterface")  # type: ignore[attr-defined]
        out = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"rtde_receive": fake_receive,
                                                              "rtde_control": fake_control}))
            stack.enter_context(mock.patch.object(socket, "create_connection", reached("socket.create_connection")))
            stack.enter_context(mock.patch.object(socket.socket, "connect", reached("socket.connect")))
            stack.enter_context(mock.patch("src.robot.drivers.ur.arm.URRobotArm.connect",
                                           reached("URRobotArm.connect")))
            stack.enter_context(redirect_stdout(out))
            code = self.probe.main(["--tree", str(self.tree), "--work", str(self.dir / "work"), *argv])
        return code, out.getvalue()

    def test_a_chain_without_the_ursim_layer_is_refused(self) -> None:
        code, said = self._run("--profile", "")
        self.assertEqual(2, code, said)
        self.assertIn(self.probe.URSIM_LAYER, said)
        self.assertEqual([], self.reached, "the probe reached for a controller before it refused")

    def test_a_chain_that_names_the_cell_again_after_the_layer_is_refused(self) -> None:
        (self.tree / "robot" / "robot.cellip.yaml").write_text(
            "robot:\n  ur:\n    ip: \"169.254.111.231\"\n", encoding="utf-8")
        code, said = self._run("--profile", f"{self.probe.URSIM_LAYER},cellip")
        self.assertEqual(2, code, said)
        self.assertIn("169.254.111.231", said)
        self.assertIn("127.0.0.1", said)
        self.assertEqual([], self.reached, "the probe reached for a controller before it refused")

    def test_a_layer_of_the_same_name_in_the_tree_cannot_aim_it_elsewhere(self) -> None:
        """The probe writes its own layer over whatever the copied tree held under that name."""
        (self.tree / "robot" / f"robot.{self.probe.URSIM_LAYER}.yaml").write_text(
            "robot:\n  ur:\n    ip: \"10.0.0.7\"\n", encoding="utf-8")
        work = self.dir / "work"
        copied = self.probe.copy_the_tree(self.tree, work)
        self.assertIsNone(self.probe.refusal_of(copied, self.probe.URSIM_LAYER))
        self.assertIn("10.0.0.7", (self.tree / "robot" / f"robot.{self.probe.URSIM_LAYER}.yaml").read_text(
            encoding="utf-8"), "the cell's own tree was written to; the probe works on a copy")

    def test_the_ursim_layer_alone_passes_and_names_only_the_address(self) -> None:
        """The control: without it the refusals above could be refusing every chain."""
        copied = self.probe.copy_the_tree(self.tree, self.dir / "work")
        self.assertIsNone(self.probe.refusal_of(copied, self.probe.URSIM_LAYER))
        layer = (copied / "robot" / f"robot.{self.probe.URSIM_LAYER}.yaml").read_text(encoding="utf-8")
        body = [line.strip() for line in layer.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(["robot:", "ur:", 'ip: "127.0.0.1"'], body)

    def test_an_existing_work_directory_is_refused_before_anything_is_copied(self) -> None:
        work = self.dir / "work"
        work.mkdir()
        (work / "left_over.txt").write_text("from an earlier run", encoding="utf-8")
        code, said = self._run("--profile", self.probe.URSIM_LAYER)
        self.assertEqual(2, code, said)
        self.assertEqual([], self.reached)
        self.assertEqual(["left_over.txt"], sorted(p.name for p in work.iterdir()))


if __name__ == "__main__":
    unittest.main()
