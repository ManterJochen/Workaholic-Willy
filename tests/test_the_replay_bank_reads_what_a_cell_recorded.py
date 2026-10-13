"""The replay bank reads what a cell recorded, and its gate says where a change admits fewer grasps.

The bank (``scripts/replay_bank``, the owner, 2026-10-12) judges the cell's recorded picks again with today's code and
sets a run beside the one before: a lever or a port that admits fewer grasps on any recorded pick exits 1 there, before
a cell finds out. Its recordings lie outside the repository (``D:/dev/willy_replay_bank``); these pin the parts that
need none:

* a pick written by ``record_views`` is read back look by look, its joints off the look's label, its depth's holes
  (65535) read as no depth, and its target's mask projected from its fused cloud where the look carries no
  segmentation; the segmentations a research recording carries come back as written;
* a refusal is named by what met what: a finger at a box the camera saw, at the support, a joint limit;
* the gate passes a run that admits as many grasps or more on every pick and fails one with fewer on any.

The cell itself (the UR driver on the bench's instant controller, its guard, the camera world, the calculator) runs in
``python scripts/replay_bank/run.py`` on the bank, a few minutes for its 125 picks.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.execution.record_views import _packed_masks, record_views
from tests._wrist_views import K
from tests.test_a_kept_scene_follows_only_what_stands_where_it_stood import Part, shoot

_REPO = Path(__file__).resolve().parents[1]


def _script(name: str) -> Any:
    folder = _REPO / "scripts" / "replay_bank"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bank = _script("bank")
run = _script("run")

#: The look's label as the pick loop writes it: its joints in degrees.
_LABEL = "(-75.1, -83.0, -72.4, -137.3, 96.0, -76.3)"


def _recorded(folder: Path) -> "tuple[Path, Any]":
    """One look of a grey cube on the bench, straight down, written as the cell writes a pick."""
    shot = shoot([Part("grey", (0.0, -600.0))])
    depth = shot.depth.copy()
    depth[:4, :4] = 65535.0
    frame = SimpleNamespace(rgb=shot.bgr[..., ::-1], depth_map=depth, surface_depth_map=depth, intrinsics=K,
                            tool_pose=None)
    look = SimpleNamespace(label=_LABEL, name=f"EIH_Cam@{_LABEL} deg", frame=frame, camera_to_base=shot.camera)
    from src.robot.grasping.multiview.scene_geometry import to_base_mm

    cloud = to_base_mm(shot.hit == 0, shot.depth, K, shot.camera)
    path = record_views([look], target_cloud_base_mm=cloud, directory=folder / "2026-10-12" / "views", name="pick-x")
    assert path is not None
    return path, shot


class APickIsReadBackTests(unittest.TestCase):
    def test_every_look_comes_back_with_its_joints_its_depth_and_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path, shot = _recorded(Path(folder))
            pick = bank.load_pick(path)

        self.assertEqual("2026-10-12", pick.day)
        (look,) = pick.looks
        np.testing.assert_allclose(np.degrees(look.joints_rad), (-75.1, -83.0, -72.4, -137.3, 96.0, -76.3))
        self.assertEqual(0.0, float(look.depth_mm[:4, :4].max()), "a hole the camera wrote as 65535 is no depth")
        np.testing.assert_allclose(look.camera_to_base, shot.camera)
        self.assertEqual((), look.masks)
        mask = pick.target_mask(look)
        cube = shot.hit == 0
        self.assertGreater(float((mask & cube).sum()) / float(cube.sum()), 0.95)
        self.assertLess(float((mask & ~cube).sum()) / float(cube.sum()), 0.05)

    def test_the_segmentations_a_research_recording_carries_come_back_as_written(self) -> None:
        shot = shoot([Part("grey", (-60.0, -600.0)), Part("grey", (60.0, -560.0))])
        segmentations = [SimpleNamespace(mask=(shot.hit == index)) for index in range(2)]

        masks = bank._unpacked(_packed_masks(segmentations, shot.hit.shape), shot.hit.shape)

        self.assertEqual(2, len(masks))
        for index, mask in enumerate(masks):
            np.testing.assert_array_equal(shot.hit == index, mask)


class ARefusalIsNamedByWhatMetWhatTests(unittest.TestCase):
    def test_a_finger_a_support_and_a_joint_limit(self) -> None:
        said = {
            "rfinger|fixture:seen_00: mesh distance 1.764 mm < 2.000 mm": "rfinger at a box the camera saw",
            "lfinger|fixture:seen_s03_support3: mesh distance -0.4 mm < 2.000 mm (a solid of support3)":
                "a finger or the hand at the support",
            "joint 6 at -361.0 deg is past its limit": "a joint limit",
        }
        for message, kind in said.items():
            with self.subTest(message=message):
                self.assertEqual(kind, bank._cause(message))


def _runs(before: "dict[str, int]", now: "dict[str, int]") -> "tuple[int, str]":
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "before.jsonl"
        path.write_text("".join(json.dumps({"pick": pick, "admitted": admitted}) + "\n"
                                for pick, admitted in before.items()), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run._compare([{"pick": pick, "admitted": admitted} for pick, admitted in now.items()], path)
    return code, out.getvalue()


class TheGateTests(unittest.TestCase):
    def test_as_many_grasps_or_more_on_every_pick_passes(self) -> None:
        code, said = _runs({"a": 3, "b": 0}, {"a": 3, "b": 2})
        self.assertEqual(0, code)
        self.assertIn("1 pick(s) with more grasps admitted, 0 with fewer", said)

    def test_fewer_on_any_pick_fails_and_names_it(self) -> None:
        code, said = _runs({"a": 3, "b": 2}, {"a": 5, "b": 1})
        self.assertEqual(1, code)
        self.assertIn("b: 2 -> 1 admitted", said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
