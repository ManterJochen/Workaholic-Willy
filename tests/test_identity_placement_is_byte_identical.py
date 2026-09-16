"""The hand geometry every consumer places today, captured before the hand is placed by a declared rotation (UM lane S02).

B2 derives one rotation from ``robot.gripper.tool_frame`` and B3 composes the hand as its own body. The sim's Isaac
declaration derives exactly the identity, so every array below has to come out the same bytes afterwards; that is the
owner's UM7 control and the only thing that can see a transposed rotation or a plate added along the wrong axis on the
cells that work today. Captured on the unchanged tree (2026-09-15), for the three registry hands on ur5e and ur3e, under
the sim's declared tool frame and an undeclared one:

* the self filter's hand spheres, and the carried part capsule and box hung from their tip;
* the planner's tool0 spheres for each committed map, placed as the builder places them (the Hand-E one 20 mm plate out);
* for six arms and each hand, a sha256 of every (name, vertices, faces, frame) the exact mesh guard would be handed,
  taken with the collision engine patched out, so it runs where Coal does not.

⚠ The ``guard_meshes`` half covers the ARM arrays too, so it moves whenever a bundle is re-baked, and that is not a
B2 or B3 regression. It was re-blessed on 2026-09-16, when every arm bundle was re-baked from Universal Robots' own
collision STLs (B5 S6 to S8): four arms' geometry moved by 0.5 to 65.0 mm where Isaac's URDFs disagreed with UR's own
description, and the other two kept identical names, dtypes, shapes and frames while their vertex ORDER changed,
because a different reader lists them in its own order. The halves this golden is actually about, ``self_filter`` and
``planner_tool0``, were byte identical across that change.

⚠ ``self_filter`` and ``planner_tool0`` were re-blessed on 2026-09-16 as well, and that one IS a change of
geometry: B6 refitted every committed hand map from its own collision bundle. Measured with one ruler
(``scripts/curobo/_mesh_body.py``) against the same meshes, before and after:

===============  ==========================================  =========================================
hand             before                                      after
===============  ==========================================  =========================================
robotiq_2f85     36 spheres, HOLE 18.5 mm, reach 24.7 mm     142 spheres, no hole, reach 12.0 mm
robotiq_hande    38 spheres, HOLE 16.5 mm, reach 23.4 mm      70 spheres, no hole, reach 12.0 mm
schunk_egu50     46 spheres, HOLE 19.0 mm, reach 22.9 mm     129 spheres, no hole, reach 12.0 mm
===============  ==========================================  =========================================

Every hand this stack ships had a 16 to 19 mm blind spot, in the planner AND in the perception self filter,
which reads the same map. After the refit the filter masks the whole hand and about half as much of the empty
space around it. What this golden still protects is B2 and B3: a transposed rotation or a plate added along
the wrong axis would move these arrays too, and nothing else here changed.

Floats are kept as ``repr`` strings: the claim is byte identity, not closeness. Regenerate only on purpose:

    .venv/Scripts/python.exe tests/test_identity_placement_is_byte_identical.py --write
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest
from typing import Any
from unittest import mock

import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # pragma: no cover - direct runs with --write
    sys.path.insert(0, str(_ROOT))

_GOLDEN = _ROOT / "tests" / "data" / "b2b3_hand_geometry_golden.json"
_HANDS = ("robotiq_2f85", "robotiq_hande", "schunk_egu50")
_FILTER_ARMS = ("ur5e", "ur3e")
_GUARD_ARMS = ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e")
_SIM_FRAME = {"source": "willy", "offset_mm": [0.0, 132.0, 0.0], "rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678]}
_FRAMES = {"sim": _SIM_FRAME, "undeclared": {"source": "undeclared"}}
_PLATES = {"robotiq_hande": [20.0]}


def _r(values: Any) -> Any:
    if isinstance(values, (list, tuple)):
        return [_r(v) for v in values]
    return repr(float(values))


def _planner_hand(hand: str, frame: dict[str, Any]) -> Any:
    from src.config.schema.robot import RobotConfig
    from src.robot.safety.planning.hand import planner_hand

    gripper: dict[str, Any] = {"model": hand, "tool_frame": frame}
    if hand in _PLATES:
        gripper["coupling_plates_mm"] = _PLATES[hand]
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


def _filter_capture() -> dict[str, Any]:
    from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres, payload_capsule

    out: dict[str, Any] = {}
    for frame_name, frame in _FRAMES.items():
        for arm in _FILTER_ARMS:
            for hand_name in _HANDS:
                hand = _planner_hand(hand_name, frame)
                spheres = hand_spheres(hand, arm)
                key = f"{frame_name}/{arm}/{hand_name}"
                if spheres is None:
                    out[key] = None
                    continue
                capsule = payload_capsule(hand, spheres, length_mm=120.0, lateral_margin_mm=10.0)
                # Captured before S07 as carried_part_box(spheres, ...); S07 gave it the hand, whose placement now
                # says which way the part hangs. The numbers are compared, so the call may change and the bytes not.
                dims, centre = carried_part_box(hand, spheres, grip_width_mm=60.0, length_mm=120.0,
                                                lateral_margin_mm=10.0)
                out[key] = {
                    "spheres": [[_r(s.start_mm), _r(s.radius_mm), s.frame] for s in spheres],
                    "payload_capsule": [_r(capsule.start_mm), _r(capsule.end_mm), _r(capsule.radius_mm), capsule.frame],
                    "carried_part_box": [_r(dims), _r(centre)],
                }
    return out


def _planner_capture() -> dict[str, Any]:
    path = _ROOT / "scripts" / "curobo" / "_gripper_placement.py"
    spec = importlib.util.spec_from_file_location("_gripper_placement_for_golden", path)
    assert spec is not None and spec.loader is not None
    placement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(placement)
    maps = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"
    out: dict[str, Any] = {}
    for hand_name in _HANDS:
        data = yaml.safe_load((maps / f"{hand_name}_gripper_spheres.yml").read_text(encoding="utf-8"))
        origin = data["_provenance"]["origin"]
        coupling = sum(_PLATES[hand_name]) if origin == placement.MOUNTING_FACE else None
        placed = placement.place_tool0_spheres(data["collision_spheres"]["tool0"], origin=origin, coupling_mm=coupling)
        out[hand_name] = [[_r(s["center"]), _r(s["radius"])] for s in placed]
    return out


class _Captured:
    """Stands in for MeshSelfCollisionBackend and keeps exactly what the guard would have been handed."""

    last: dict[str, Any] | None = None

    def __init__(self, adapter: Any, meshes: dict[str, Any]) -> None:
        type(self).last = meshes


def _guard_capture() -> dict[str, Any]:
    import numpy as np

    from src.robot.safety import _fcl_self_collision as fcl

    out: dict[str, Any] = {}
    variants = {"robotiq_2f85": (None, 0.0), "robotiq_hande": ("robotiq_hande", 20.0), "schunk_egu50": ("schunk_egu50", 0.0)}
    with mock.patch.object(fcl, "import_collision_engine", return_value=(object(), "coal")), \
            mock.patch.object(fcl, "_EngineAdapter", lambda mod, kind: None), \
            mock.patch.object(fcl, "MeshSelfCollisionBackend", _Captured):
        for arm in _GUARD_ARMS:
            for hand_name, (mesh_name, coupling) in variants.items():
                key = f"{arm}/{hand_name}"
                status = fcl.mesh_backend_status(arm, None, mesh_name)
                _Captured.last = None
                fcl.make_backend(arm, None, mesh_name, coupling)
                if _Captured.last is None:
                    out[key] = {"status": status}
                    continue
                digest = hashlib.sha256()
                parts = []
                for name in sorted(_Captured.last):
                    verts, faces, frame = _Captured.last[name]
                    verts, faces = np.asarray(verts), np.asarray(faces)
                    digest.update(name.encode("utf-8"))
                    digest.update(verts.tobytes())
                    digest.update(faces.tobytes())
                    digest.update(int(frame).to_bytes(8, "little", signed=True))
                    parts.append([name, str(verts.dtype), list(verts.shape), str(faces.dtype), list(faces.shape), int(frame)])
                out[key] = {"status": status, "sha256": digest.hexdigest(), "parts": parts}
    return out


def capture() -> dict[str, Any]:
    return {"self_filter": _filter_capture(), "planner_tool0": _planner_capture(), "guard_meshes": _guard_capture()}


class TheHandIsWhereItWasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        cls.now = capture()

    def test_the_self_filter_hand_is_byte_identical(self) -> None:
        self.assertEqual(self.now["self_filter"], self.golden["self_filter"])

    def test_the_planner_tool0_spheres_are_byte_identical(self) -> None:
        self.assertEqual(self.now["planner_tool0"], self.golden["planner_tool0"])

    def test_the_guard_is_handed_the_same_meshes(self) -> None:
        self.assertEqual(self.now["guard_meshes"], self.golden["guard_meshes"])

    def test_the_golden_covers_what_it_claims(self) -> None:
        """The control against a golden that captured nothing: every filter cell and every guard pairing is present,
        and at least one pairing per hand reached the guard."""
        self.assertEqual(len(self.golden["self_filter"]), len(_FRAMES) * len(_FILTER_ARMS) * len(_HANDS))
        self.assertEqual(len(self.golden["guard_meshes"]), len(_GUARD_ARMS) * len(_HANDS))
        for hand_name in _HANDS:
            with self.subTest(hand=hand_name):
                self.assertTrue(any("sha256" in v for k, v in self.golden["guard_meshes"].items()
                                    if k.endswith(f"/{hand_name}")))
                self.assertTrue(self.golden["planner_tool0"][hand_name])


if __name__ == "__main__":  # pragma: no cover
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_bytes((json.dumps(capture(), indent=1, sort_keys=True) + "\n").encode("utf-8"))
        print(f"wrote {_GOLDEN}")
    else:
        unittest.main()
