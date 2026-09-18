"""One hand, one name, no default: the guard reads its hand from ``robot.gripper.model``.

Step 4f (owner, Step 4 Q5 and Q10). The registry name is the only name a hand has. The self collision guard derives
its mesh bundle and its coupling plate from it, so ``safety.self_collision.collision_mesh_variant`` and
``coupling_mm`` leave the schema, and a cell whose guard reads hand geometry refuses to build while the name is
unset. A KUKA, a dummy and the rehearsal read no hand geometry and build as before.

No test here needs Coal or python-fcl: where the guard would build its exact mesh backend, the arguments it asks
for are captured instead.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import yaml
from pydantic import ValidationError

from src.config import ConfigError, load_robot_config
from src.config.grippers import available_grippers, load_gripper
from src.config.schema.robot import (
    RobotConfig,
    RobotSafetyConfig,
    SelfCollisionSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.contracts import UNSET
from src.robot.execution.real_cell.preflight import CheckStatus
from tests.test_sim_mount_is_derived import _ScratchTree

_ROOT = Path(__file__).resolve().parents[1]
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"
_FCL = "src.robot.safety._fcl_self_collision.make_backend"


def _robot(model: str | None = "robotiq_2f85", plates: list[float] | None = None) -> RobotConfig:
    """A UR tree naming ``model``, with the payload guard off so nothing else about the tree is under test."""
    gripper: dict[str, Any] = {"model": model}
    if plates is not None:
        gripper["coupling_plates"] = [{"name": f"plate_{i}", "thickness_mm": float(mm)} for i, mm in enumerate(plates)]
    return RobotConfig.model_validate(
        {"vendor": "ur", "safety": {"payload": {"enforce": False}}, "gripper": gripper}
    )


def _ur_arm(model: str = "ur5e") -> SimpleNamespace:
    return SimpleNamespace(capabilities=SimpleNamespace(vendor="ur", model=model))


def _guard(preflight: Any) -> Any:
    from src.robot.safety.self_collision import SelfCollisionGuard

    (guard,) = [g for g in preflight._guards if isinstance(g, SelfCollisionGuard)]
    return guard


def _origin(name: str) -> str:
    text = (_MAPS / f"{name}_gripper_spheres.yml").read_text(encoding="utf-8")
    return str(yaml.safe_load(text)["_provenance"]["origin"])


class _Captured:
    """Stands in for ``make_backend``: records what was asked for and builds nothing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, model: str, mesh_dir: str | None = None, mesh_name: str | None = None, coupling_mm: float = 0.0,
        *, placement: Any = None, coupling_boxes: Any = (),
    ) -> None:
        # ⛔ A STUB POORER THAN THE THING IT STANDS FOR cannot fail the way the thing fails, and one richer
        # cannot either. Every argument the guard passes is recorded here, so adding one to `make_backend`
        # without teaching this stub is a TypeError rather than a silently untested path.
        self.calls.append({"model": model, "mesh_name": mesh_name, "coupling_mm": coupling_mm,
                           "coupling_boxes": tuple(coupling_boxes)})
        return None


class PlannerHandTests(unittest.TestCase):
    """The resolver: a registry name in; the sphere map, its origin, the guard's bundle and the plate out."""

    def test_the_2f85_is_the_hand_every_arm_bundle_carries(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(_robot("robotiq_2f85"))
        self.assertEqual(hand.model, "robotiq_2f85")
        self.assertEqual(hand.sphere_map, _MAPS / "robotiq_2f85_gripper_spheres.yml")
        self.assertTrue(hand.sphere_map.is_file())
        self.assertFalse((_MAPS / "ur5e_gripper_spheres.yml").exists(), "the 2F-85 map is named after the hand")
        self.assertEqual(hand.origin, "flange")
        self.assertIsNone(hand.guard_variant, "every committed arm bundle already carries the 2F-85")
        self.assertEqual(hand.coupling_mm, 0.0)
        self.assertEqual(hand.jaw, load_gripper("robotiq_2f85", aliases=False).jaw)

    def test_the_hande_starts_at_its_mounting_face_and_is_its_own_bundle(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(_robot("robotiq_hande", plates=[20.0]))
        self.assertEqual(hand.sphere_map, _MAPS / "robotiq_hande_gripper_spheres.yml")
        self.assertEqual(hand.origin, "mounting_face")
        self.assertEqual(hand.guard_variant, "robotiq_hande")
        self.assertEqual(hand.coupling_mm, 20.0)

    def test_the_egu50_is_its_own_bundle_and_starts_at_the_flange(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(_robot("schunk_egu50"))
        self.assertEqual((hand.origin, hand.guard_variant, hand.coupling_mm), ("flange", "schunk_egu50", 0.0))

    def test_an_unset_hand_is_unset_and_not_a_2f85(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        self.assertIs(planner_hand(_robot(None)), UNSET)

    def test_an_unknown_hand_is_refused_with_the_registry_sentence(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        with self.assertRaises(ConfigError) as caught:
            planner_hand(_robot("robotiq_3f"))
        self.assertIn("no gripper 'robotiq_3f'", str(caught.exception))

    def test_a_short_name_is_refused_naming_the_model(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        with self.assertRaises(ConfigError) as caught:
            planner_hand(_robot("2f85"))
        self.assertIn("'robotiq_2f85'", str(caught.exception))

    def test_the_bundle_the_guard_reads_carries_the_hand_the_planner_map_describes(self) -> None:
        """The property behind the derivation: guard and planner model one hand, for every registry hand."""
        from src.robot.safety.planning.environment import hand_mesh_bundle
        from src.robot.safety.planning.hand import planner_hand

        hands = available_grippers()
        self.assertGreaterEqual(len(hands), 3, hands)
        for name in hands:
            with self.subTest(hand=name):
                plates = [20.0] if _origin(name) == "mounting_face" else None
                hand = planner_hand(_robot(name, plates=plates))
                bundle = hand_mesh_bundle(hand.model)
                committed = yaml.safe_load(hand.sphere_map.read_text(encoding="utf-8"))
                # Matched by what the map SAYS it was fitted from, not by re-running a fit: the committed
                # maps are cover fits since B6 and refitting one costs minutes. The property is unchanged,
                # and the `source` field is itself held to the bundle being present by test_gripper_spheres.
                source = committed["_provenance"]["source"]
                self.assertEqual(
                    bundle.name, source,
                    f"the guard composes {bundle.name} for {name} and {hand.sphere_map.name} was fitted from "
                    f"{source}, so the guard and the planner model different hands",
                )
                self.assertTrue(committed["collision_spheres"]["tool0"],
                                f"{hand.sphere_map.name} holds no spheres at all, so the planner models no hand")


class TheCouplingPlatesTests(unittest.TestCase):
    """Q10: the plates between the flange and a hand's own mounting face, summed into the guard coupling."""

    def test_the_plates_sum_into_the_coupling(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        self.assertEqual(planner_hand(_robot("robotiq_hande", plates=[12.0, 8.0])).coupling_mm, 20.0)

    def test_the_shipped_hande_layer_declares_its_plate(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(load_robot_config(profile="hande"))
        self.assertEqual((hand.model, hand.coupling_mm), ("robotiq_hande", 20.0))

    def test_a_mounting_face_hand_with_no_plates_is_refused(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        with self.assertRaises(ConfigError) as caught:
            planner_hand(_robot("robotiq_hande"))
        self.assertIn("robot.gripper.coupling_plates", str(caught.exception))
        self.assertIn("mounting_face", str(caught.exception))

    def test_no_plate_said_out_loud_is_an_answer(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        self.assertEqual(planner_hand(_robot("robotiq_hande", plates=[])).coupling_mm, 0.0)

    def test_a_flange_hand_with_a_plate_is_refused(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        with self.assertRaises(ConfigError) as caught:
            planner_hand(_robot("robotiq_2f85", plates=[5.0]))
        self.assertIn("robot.gripper.coupling_plates", str(caught.exception))
        self.assertIn("flange", str(caught.exception))

    def test_a_negative_plate_is_refused_by_the_schema(self) -> None:
        # The bound, not the key: before the key existed the same call was refused as an unknown field.
        # UM8 tightened it from ge to gt: a plate is a body now, and a slab of no thickness places nothing
        # and collides with nothing, so it is a line somebody should delete rather than a zero to carry.
        with self.assertRaises(ValidationError) as caught:
            _robot("robotiq_hande", plates=[-1.0])
        self.assertEqual([e["type"] for e in caught.exception.errors()], ["greater_than"])

    def test_the_key_is_unset_by_default(self) -> None:
        self.assertIsNone(RobotConfig().gripper.coupling_plates)


class AnUnsetHandRefusesTests(unittest.TestCase):
    """Where hand geometry is read, an unset name refuses at build (Q5), on a UR and on a non UR double."""

    def test_an_unset_hand_refuses_where_hand_geometry_is_read(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        with self.assertRaises(ConfigError) as caught:
            URRobotArm(_robot(None))
        self.assertIn("robot.gripper.model", str(caught.exception))
        self.assertIn("ur5e", str(caught.exception))

    def test_a_named_hand_builds_and_the_preflight_says_which(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(_robot("robotiq_2f85"))
        preflight = arm.safety_preflight
        assert preflight is not None
        self.assertEqual(preflight.planner_hand(arm).model, "robotiq_2f85")

    def test_the_sim_preflight_builder_refuses_it_and_a_non_ur_arm_gets_the_configured_hand(self) -> None:
        from src.willy_sim.config import load_sim_config, require_robot, sim_safety_preflight

        cfg = load_sim_config()
        robot = require_robot(cfg)
        unnamed = cfg.model_copy(update={"robot": robot.model_copy(
            update={"gripper": robot.gripper.model_copy(update={"model": None})}
        )})
        with self.assertRaises(ConfigError) as caught:
            sim_safety_preflight(unnamed)
        self.assertIn("robot.gripper.model", str(caught.exception))

        isaac = SimpleNamespace(capabilities=SimpleNamespace(vendor="isaac-sim", model="isaac-sim"))
        preflight = sim_safety_preflight(cfg)
        hand = preflight.planner_hand(isaac)
        self.assertEqual(hand.model, "robotiq_2f85")
        self.assertIs(_guard(preflight).hand, hand)

    def test_a_model_from_either_source_is_enough_to_refuse(self) -> None:
        from src.robot.safety.preflight import SafetyPreflight

        declared = RobotSafetyConfig.model_validate({"self_collision": {"kinematics_model": "ur5e"}})
        with self.assertRaises(ConfigError):
            SafetyPreflight.from_safety_config(declared, WorkspaceLimitsConfig())
        with self.assertRaises(ConfigError):
            SafetyPreflight.from_safety_config(RobotSafetyConfig(), WorkspaceLimitsConfig(), arm_model="ur3e")


class WhatReadsNoHandGeometryBuildsTests(unittest.TestCase):
    """Controls, green before and after: none of these resolves an arm model for an exact mesh guard."""

    def test_a_kuka_arm_builds_with_no_hand(self) -> None:
        from src.robot.drivers.kuka.arm import KukaRobotArm

        tree = RobotConfig.model_validate({"vendor": "kuka", "kuka": {"controller_ip": "10.8.8.8"}})
        self.assertIsNone(tree.gripper.model)
        KukaRobotArm(tree)

    def test_a_dummy_robot_builds_with_no_hand(self) -> None:
        from src.robot.execution.robot import Robot

        Robot.from_config(RobotConfig.model_validate({"vendor": "dummy", "gripper": {"vendor": "dummy"}}))

    def test_a_capsule_guard_and_a_stood_down_guard_build_with_no_hand(self) -> None:
        from src.robot.safety.preflight import SafetyPreflight

        for block in ({"backend": "capsule", "kinematics_model": "ur5e"},
                      {"enforce": False, "kinematics_model": "ur5e"}):
            with self.subTest(**block):
                SafetyPreflight.from_safety_config(
                    RobotSafetyConfig.model_validate({"self_collision": block}), WorkspaceLimitsConfig(),
                )


class TheTwoHandKeysAreRefusedAtLoadTests(_ScratchTree):
    def test_a_tree_that_still_writes_the_variant_is_refused_naming_the_hand(self) -> None:
        self._layer("variantkey", 'robot:\n  safety:\n    self_collision:\n      collision_mesh_variant: "robotiq_hande"\n')
        message = self._load_error("variantkey")
        self.assertIn("robot.safety.self_collision.collision_mesh_variant", message)
        self.assertIn("removed on purpose", message)
        self.assertIn("robot.gripper.model", message)

    def test_a_tree_that_still_writes_the_coupling_is_refused_naming_the_plates(self) -> None:
        self._layer("couplingkey", "robot:\n  safety:\n    self_collision:\n      coupling_mm: 20.0\n")
        message = self._load_error("couplingkey")
        self.assertIn("robot.safety.self_collision.coupling_mm", message)
        self.assertIn("removed on purpose", message)
        self.assertIn("robot.gripper.coupling_plates", message)

    def test_the_schema_refuses_both_keys(self) -> None:
        for key, value in (("collision_mesh_variant", "robotiq_hande"), ("coupling_mm", 20.0)):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                SelfCollisionSafetyConfig.model_validate({key: value})


class TheGuardTakesItsHandFromTheRobotConfigTests(unittest.TestCase):
    def test_the_hande_profile_builds_a_guard_on_the_hande_bundle_with_its_plate(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        arm = URRobotArm(load_robot_config(profile="hande"))
        guard = _guard(arm.safety_preflight)
        captured = _Captured()
        with mock.patch(_FCL, captured):
            guard.exact_mesh_engine(arm)
        self.assertEqual(captured.calls, [{"model": "ur5e", "mesh_name": "robotiq_hande", "coupling_mm": 20.0, "coupling_boxes": ()}])
        self.assertEqual(guard.hand.model, "robotiq_hande")

    def test_a_guard_built_directly_keeps_the_arm_bundle(self) -> None:
        """Control, green before and after: the guard's own unit tests build it with no hand."""
        from src.robot.safety.self_collision import SelfCollisionGuard

        guard = SelfCollisionGuard(SelfCollisionSafetyConfig())
        captured = _Captured()
        with mock.patch(_FCL, captured):
            guard.exact_mesh_engine(_ur_arm())
        self.assertEqual(captured.calls, [{"model": "ur5e", "mesh_name": None, "coupling_mm": 0.0, "coupling_boxes": ()}])

    def test_a_guard_built_directly_has_no_hand(self) -> None:
        from src.robot.safety.self_collision import SelfCollisionGuard

        self.assertIs(SelfCollisionGuard(SelfCollisionSafetyConfig()).hand, UNSET)

    def test_the_continuous_monitor_is_built_on_the_same_hand_and_plate(self) -> None:
        from src.robot.safety.continuous_monitor import ContinuousCollisionMonitor, ContinuousGuardProfile

        captured = _Captured()
        with mock.patch("src.robot.safety.continuous_monitor.make_backend", captured):
            ContinuousCollisionMonitor.from_model(
                "ur5e", 180.0, (), ContinuousGuardProfile(enabled=True, margin_mm=8.0),
                variant="robotiq_hande", coupling_mm=20.0,
            )
        self.assertEqual(captured.calls, [{"model": "ur5e", "mesh_name": "robotiq_hande", "coupling_mm": 20.0, "coupling_boxes": ()}])


class TheRealCellPreflightNamesTheHandTests(unittest.TestCase):
    @staticmethod
    def _rows(robot: RobotConfig) -> list[Any]:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        return [c for c in run_config_preflight(robot, curobo_available=False).checks if c.name == "hand"]

    def test_the_real_cell_preflight_blocks_an_unset_hand(self) -> None:
        (row,) = self._rows(_robot(None))
        self.assertEqual(row.status, CheckStatus.BLOCK)
        self.assertIn("robot.gripper.model", row.detail)
        self.assertIn("robotiq_2f85", row.fix, "the fix names what the registry holds")

    def test_a_named_hand_is_reported_with_its_origin_and_plate(self) -> None:
        (row,) = self._rows(_robot("robotiq_hande", plates=[20.0]))
        self.assertEqual(row.status, CheckStatus.OK)
        for part in ("robotiq_hande", "mounting_face", "20"):
            self.assertIn(part, row.detail)

    def test_an_unknown_hand_blocks_with_the_registry_sentence(self) -> None:
        (row,) = self._rows(_robot("robotiq_3f"))
        self.assertEqual(row.status, CheckStatus.BLOCK)
        self.assertIn("no gripper 'robotiq_3f'", row.detail)

    def test_a_dummy_cell_is_not_blocked_for_a_hand_it_never_reads(self) -> None:
        (row,) = self._rows(RobotConfig.model_validate({"vendor": "dummy", "gripper": {"vendor": "dummy"}}))
        self.assertEqual(row.status, CheckStatus.OK)


class TheDoctorAsksForTheHandTests(unittest.TestCase):
    def test_an_unset_hand_is_missing_and_names_the_key(self) -> None:
        from src.robot.safety.planning import doctor

        (probe,) = doctor._probe_gripper("ur5e", None)
        self.assertIs(probe.status, doctor.ProbeStatus.MISSING)
        self.assertIn("robot.gripper.model", probe.detail + probe.remedy)

    def test_the_2f85_is_probed_on_the_arm_bundle_and_its_own_map(self) -> None:
        from src.robot.safety.planning import doctor

        probes = doctor._probe_gripper("ur5e", "robotiq_2f85")
        details = [p.detail for p in probes]
        self.assertTrue(any(d.endswith("ur5e_collision_meshes.npz") for d in details), details)
        self.assertTrue(any(d.endswith("robotiq_2f85_gripper_spheres.yml") for d in details), details)
        self.assertTrue(all(p.status is doctor.ProbeStatus.OK for p in probes), probes)

    def test_the_hande_is_probed_on_its_own_hand_bundle(self) -> None:
        from src.robot.safety.planning import doctor

        probes = doctor._probe_gripper("ur5e", "robotiq_hande")
        details = [p.detail for p in probes]
        self.assertTrue(any(d.endswith("robotiq_hande_hand_meshes.npz") for d in details), details)


class EveryPreflightBuildNamesTheHandTests(unittest.TestCase):
    """Every construction of the preflight in the library passes the hand, except where no arm model can resolve."""

    #: A KUKA resolves no arm model and reads no bundle, so its preflight has no hand to take.
    _READS_NO_HAND = frozenset({"src/robot/drivers/kuka/arm.py"})

    def test_every_site_passes_the_hand(self) -> None:
        sites: list[tuple[str, int, set[str | None]]] = []
        for path in sorted((_ROOT / "src").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "from_safety_config(" not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "from_safety_config":
                    sites.append((path.relative_to(_ROOT).as_posix(), node.lineno, {k.arg for k in node.keywords}))
        self.assertGreaterEqual(len(sites), 5, sites)
        for rel, line, keywords in sites:
            if rel in self._READS_NO_HAND:
                continue
            with self.subTest(site=f"{rel}:{line}"):
                self.assertIn("hand", keywords)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
