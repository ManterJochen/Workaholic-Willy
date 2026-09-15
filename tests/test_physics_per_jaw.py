"""A per-jaw MuJoCo referee: the cell models the hand the labels were made for, and each jaw shakes into its own files.

Lane (i) D6. The MuJoCo cell modelled one 85 mm hand: its sanity limit, `restore`, the width fallback and the open-jaw
control all read that aperture, so a Hand-E label shaken there would be judged by a jaw 35 mm wider than the hand, and a
Hand-E jaw blown open to 60 mm would read as sane. The cell now takes a `JawModel` (`labels.jaw_model_for`) and models
its aperture and pad size; the module constants keep the default cell's values. The shake reads and writes the jaw's
own files and refuses a jaw on Isaac, whose cell loads the 2F-85's USD.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from datagen.grasps.labels import SceneGeometry
from datagen.grasps.shapes import Solid

mujoco = None
try:  # pragma: no cover - the import is the thing under test on a machine that has it
    import mujoco  # noqa: F811
except (ImportError, OSError):  # pragma: no cover - OSError too, see tests/test_physics_mujoco.py
    pass

_HANDE = "robotiq_hande"


def _block(half_mm: float = 10.0) -> SceneGeometry:
    return SceneGeometry(
        scene_id="test", family="alone",
        objects={0: Solid(kind="box", half_extent_mm=np.full(3, half_mm), rotation=np.eye(3),
                          centre_mm=np.array([0.0, 0.0, half_mm]), instance_id=0, asset_id="block", mass_kg=0.2)},
        walls=())


def _hande_cell() -> Any:
    from datagen.grasps.labels import jaw_model_for
    from datagen.grasps.physics_mujoco import PhysicsCell

    return PhysicsCell(jaw=jaw_model_for(_HANDE))  # type: ignore[call-arg]


@unittest.skipIf(mujoco is None, "mujoco is not installed on this machine")
class AHandECellModelsTheHandETests(unittest.TestCase):
    def test_a_jaw_read_past_the_hande_is_refused(self) -> None:
        with _hande_cell() as cell:
            cell.build_scene(_block())
            cell._teleport_jaw(49.99)                            # noqa: SLF001
            self.assertEqual(cell.jaw_fault(), "")
            cell._teleport_jaw(60.0)                             # noqa: SLF001
            self.assertIn("past the 49.99 mm hand", cell.jaw_fault())

    def test_the_default_cell_still_reads_60_mm_as_sane(self) -> None:
        """The control: 60 mm is inside the 85 mm hand's limit, so the refusal above is the Hand-E's own."""
        from datagen.grasps.physics_mujoco import PhysicsCell

        with PhysicsCell() as cell:
            cell.build_scene(_block())
            cell._teleport_jaw(60.0)                             # noqa: SLF001
            self.assertEqual(cell.jaw_fault(), "")

    def test_restore_opens_the_jaw_to_the_hande(self) -> None:
        with _hande_cell() as cell:
            geometry = _block()
            cell.build_scene(geometry)
            cell.restore(geometry)
            self.assertAlmostEqual(cell._jaw_opening_mm(), 49.99, delta=1.0)  # noqa: SLF001

    def test_the_pads_are_the_hande_pads(self) -> None:
        """Half sizes in metres. The thickness is the harness's; the width across the binormal is the Hand-E's 29.24 mm,
        and the default pad (8, 20, 30) is gone."""
        with _hande_cell() as cell:
            xml = cell._gripper_xml()                            # noqa: SLF001
        self.assertIn('size="0.00400 0.01462 ', xml)
        self.assertNotIn("0.00400 0.01000 0.01500", xml)

    def test_the_four_controls_pass_on_a_hande_cell(self) -> None:
        with _hande_cell() as cell:
            controls = cell.run_controls()
        self.assertTrue(controls["jaw_is_solid"], controls)
        self.assertTrue(controls["positive_held"], controls)
        self.assertFalse(controls["negative_held"], controls)
        self.assertTrue(controls["repeat_held"], controls)


class _FakeCell:
    """Stands in for the MuJoCo cell: records how it was built and passes its controls. No scene is shaken."""

    built: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).built = kwargs

    def __enter__(self) -> "_FakeCell":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run_controls(self) -> dict[str, Any]:
        return {"jaw_is_solid": True, "positive_held": True, "negative_held": False, "repeat_held": True}


class AJawShakeUsesItsOwnFilesTests(unittest.TestCase):
    def _run(self, **kwargs: Any) -> tuple[Path, mock.MagicMock]:
        from datagen.grasps import physics

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        _FakeCell.built = {}
        with mock.patch.object(physics, "sample_trials", return_value=[]) as draw, \
             mock.patch.object(physics, "scene_assets", return_value=SimpleNamespace(mesh_collision="sdf")), \
             mock.patch("datagen.grasps.physics_mujoco.PhysicsCell", _FakeCell):
            physics.run_physics_sample(root, engine="mujoco", **kwargs)
        return root, draw

    def test_a_jaw_shake_draws_from_and_writes_to_the_jaws_files(self) -> None:
        from datagen.grasps.labels import jaw_model_for

        root, draw = self._run(jaw=_HANDE, out_name=f"grasp_physics_jaw_{_HANDE}.jsonl")
        self.assertEqual(draw.call_args.kwargs["labels"], f"grasps_jaw_{_HANDE}.jsonl")
        self.assertEqual(_FakeCell.built["jaw"], jaw_model_for(_HANDE))
        first = (root / f"grasp_physics_jaw_{_HANDE}.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(json.loads(first)["controls"]["jaw"], _HANDE)

    def test_the_corpus_shake_draws_the_corpus_file_with_the_default_cell(self) -> None:
        """The control."""
        _root, draw = self._run()
        self.assertEqual(draw.call_args.kwargs.get("labels", "grasps.jsonl"), "grasps.jsonl")
        self.assertNotIn("jaw", _FakeCell.built)

    def test_the_controls_record_the_jaw(self) -> None:
        root, _draw = self._run()
        first = (root / "grasp_physics.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(json.loads(first)["controls"]["jaw"], "2f85")

    def test_a_jaw_on_isaac_refuses_before_anything_loads(self) -> None:
        from datagen.grasps import physics

        with mock.patch.object(physics, "scene_assets") as assets, self.assertRaises(ValueError) as caught:
            physics.run_physics_sample(Path("unused"), engine="isaac", jaw=_HANDE,
                                       out_name=f"grasp_physics_jaw_{_HANDE}.jsonl")
        assets.assert_not_called()
        self.assertIn("2F-85", str(caught.exception))

    def test_a_jaw_shake_may_not_write_another_file(self) -> None:
        from datagen.grasps import physics

        with mock.patch.object(physics, "scene_assets") as assets, self.assertRaises(ValueError):
            physics.run_physics_sample(Path("unused"), engine="mujoco", jaw=_HANDE)
        assets.assert_not_called()


class TheSamplingAndTheCommandTakeTheJawTests(unittest.TestCase):
    def test_the_sampling_names_the_jaws_file_and_passes_the_jaw(self) -> None:
        from datagen.grasps.service import PhysicsSampling

        seen: dict[str, Any] = {}

        def capture(root: Any, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {"trials": 0, "drawn": 0, "refused": 0, "by_source": {}}

        with mock.patch("datagen.grasps.physics.run_physics_sample", side_effect=capture):
            report = PhysicsSampling.from_dataset("d", engine="mujoco").sample(jaw=_HANDE)  # type: ignore[call-arg]
        self.assertEqual(seen["out_name"], f"grasp_physics_jaw_{_HANDE}.jsonl")
        self.assertEqual(seen["jaw"], _HANDE)
        self.assertEqual(report.out_name, f"grasp_physics_jaw_{_HANDE}.jsonl")

    def test_the_command_passes_the_jaw(self) -> None:
        from datagen import __main__ as cli

        with mock.patch.object(cli, "_cmd_physics_sample", return_value=0) as handler:
            cli.main(["physics-sample", "--name", "v5_s0", "--jaw", _HANDE])
        self.assertEqual(handler.call_args.kwargs["jaw"], _HANDE)


if __name__ == "__main__":
    unittest.main()
