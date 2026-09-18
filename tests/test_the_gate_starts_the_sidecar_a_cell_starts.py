"""The matrix gate starts the sidecar a cell's driver starts: its payload spheres and its plates (customer chain lane C3g).

A file written with ``--attach N`` measured a sidecar with no attach spheres, so its composed hash could never admit the
cell it names (finding 12 of the audit of 2026-09-17); and a plate with a cross section, which a cell's planner adds as a
body, could not be told to the gate at all (finding 20).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_CUROBO = Path(__file__).resolve().parents[1] / "scripts" / "curobo"


def _gate() -> Any:
    if str(_CUROBO) not in sys.path:
        sys.path.insert(0, str(_CUROBO))
    spec = importlib.util.spec_from_file_location("_gate_for_the_sidecar", _CUROBO / "matrix_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Started(Exception):
    """Raised by the recording client once it has seen what the gate hands it."""


class TheSidecarGetsWhatACellGivesItTests(unittest.TestCase):
    def _client_kwargs(self, attach: int) -> dict:
        seen: dict = {}

        class Recording:
            def __init__(self, **kwargs: Any) -> None:
                seen.update(kwargs)
                raise _Started()

        gate = _gate()
        with mock.patch("src.robot.safety.planning.curobo_client.CuroboPlanClient", Recording), \
                self.assertRaises(_Started):
            gate.measure(arm="ur5e", hand="robotiq_2f85", coupling_mm=0.0, planner_margin_mm=4.0, attach_spheres=attach,
                         guard_margin_mm=10.0, rotation_xyzw=[0.0, 0.0, 0.0, 1.0], random_n=2, seed=0)
        return seen

    def test_the_attach_spheres_it_records_are_the_ones_it_starts_with(self) -> None:
        self.assertEqual(self._client_kwargs(4)["attach_spheres"], 4)

    def test_no_attach_spheres_is_zero_and_not_left_out(self) -> None:
        """⭐ THE CONTROL: the default combination hands the sidecar what it always has."""
        self.assertEqual(self._client_kwargs(0)["attach_spheres"], 0)


class APlateWithACrossSectionTests(unittest.TestCase):
    _PLATE = [{"name": "adapter", "thickness_mm": 20.0, "cross_section_mm": [31.5, 31.5]}]

    def test_the_gate_cell_has_the_bodies_the_customer_cell_has(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.hand import planner_hand

        gate = _gate()
        measured = planner_hand(gate._robot_config("ur5e", "robotiq_hande", 0.0, 4.0, [0.0, 0.0, 0.0, 1.0], plates=self._PLATE))
        cell = planner_hand(RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e"},
            "gripper": {"model": "robotiq_hande", "coupling_plates": self._PLATE,
                        "tool_frame": {"source": "polyscope", "offset_mm": [0.0, 0.0, 155.75],
                                       "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        }))
        self.assertTrue(cell.coupling_boxes)
        self.assertEqual(measured.coupling_boxes, cell.coupling_boxes)
        self.assertEqual(measured.coupling_mm, cell.coupling_mm)

    def test_the_command_line_spells_it(self) -> None:
        gate = _gate()
        seen: dict = {}

        class _Result:
            path = Path("x.json")

            def render(self) -> str:
                return "x"

        def recorder(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return _Result()

        with mock.patch.object(gate, "measure", recorder), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(["--arm", "ur5e", "--hand", "robotiq_hande", "--planner-margin-mm", "4",
                                        "--plate", "adapter:20:31.5,31.5", "--tool-rotation-xyzw", "0", "0", "0", "1"]), 0)
        self.assertEqual(seen["plates"], self._PLATE)

    def test_a_plate_beside_a_coupling_or_malformed_is_refused(self) -> None:
        gate = _gate()
        base = ["--arm", "ur5e", "--hand", "robotiq_hande", "--planner-margin-mm", "4"]
        cases = (["--plate", "adapter:20", "--coupling-mm", "20"], ["--plate", "adapter:0"], ["--plate", "adapter:20:31.5"],
                 ["--plate", "a:b"])
        for extra in cases:
            with self.subTest(extra=extra), mock.patch.object(gate, "measure", side_effect=AssertionError("measured")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                gate.main([*base, *extra])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
