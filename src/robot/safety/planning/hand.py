"""The hand a cell's planner and guard model, derived from the one name the cell gives it.

``robot.gripper.model`` is the only name a hand has. Everything the motion stack needs to know about
the geometry at the end of the arm follows from that name and from committed files, so no second key
can name another hand:

* the registry file ``config/grippers/<model>.yaml``, for the jaw;
* the committed sphere map ``robot/<model>_gripper_spheres.yml`` beside this file, for the planner,
  and the ``origin`` its provenance declares;
* the guard's mesh bundle: the arm's own for the hand every committed arm bundle carries, the hand's
  per-arm bundle otherwise (``environment.collision_mesh_bundle`` composes the arm in);
* the coupling, the sum of ``robot.gripper.coupling_plates_mm``;
* the approach axis, flange +Y in every committed model, which a declared tool frame has to agree
  with.

An unset name is :data:`UNSET`, never the 2F-85. A caller that reads hand geometry refuses it, and
:func:`unset_hand_refusal` words that refusal once for every such caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import yaml

from src.config.grippers import available_grippers, load_gripper
from src.config.loader import ConfigError
from src.contracts import UNSET, Maybe, chosen
from src.geometry.quaternion import to_rotation_matrix

from .robot.gripper_spheres import FLANGE, MOUNTING_FACE

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.grippers.gripper_schema import ParallelJawSpec
    from src.config.schema.robot import RobotConfig, SelfCollisionSafetyConfig

__all__ = [
    "APPROACH_TOLERANCE_DEG",
    "ARM_BUNDLE_HAND",
    "HAND_APPROACH_IN_TOOL0",
    "PlannerHand",
    "approach_refusal",
    "declared_approach",
    "descriptor_refusal",
    "guard_variant_for",
    "hand_geometry_model",
    "planner_hand",
    "sphere_map_path",
    "unset_hand_refusal",
]

#: The hand every committed ``{arm}_collision_meshes.npz`` carries. Those bundles were baked from
#: composed UR assets with the Robotiq 2F-85 bolted on, so for this hand the guard reads the arm's own
#: bundle and for any other hand the hand's bundle. The bundle the guard reads for every registry hand
#: is held against that hand's committed sphere map, so this constant cannot name a hand the bundles
#: do not carry.
ARM_BUNDLE_HAND: Final = "robotiq_2f85"

#: The approach axis of every committed hand model, in tool0 axes, which are DH frame 6's. Measured on
#: the Isaac cell: DH frame 6, Lula's tool0 and cuRobo's tool0 are one frame, and the physical 2F-85
#: lies along its +Y, 0.03 degrees off. The bundles' finger midpoints lie along it as well, for every
#: registry hand. By the UR convention a hand on a real flange approaches along +Z instead, and no
#: committed model describes that hand yet.
HAND_APPROACH_IN_TOOL0: Final = (0.0, 1.0, 0.0)

#: How far a declared tool frame's approach may lean from the hand model's before the two describe
#: different hands, in degrees: the tolerance a declared UR tool frame is held to against its own
#: offset.
APPROACH_TOLERANCE_DEG: Final = 1.0

_MAPS: Final = Path(__file__).resolve().parent / "robot"


@dataclass(frozen=True)
class PlannerHand:
    """One resolved hand: what the planner reads, what the guard loads, and where it sits on the flange."""

    #: The registry name, ``robot.gripper.model``.
    model: str
    #: The committed tool0 sphere map the on-box descriptor builder reads.
    sphere_map: Path
    #: Where the map's numbers start: ``flange`` (the arm asset placed the hand) or ``mounting_face``.
    origin: str
    #: The per-hand bundle the exact-mesh guard loads, or ``None`` where the arm's own bundle carries
    #: this hand.
    guard_variant: str | None
    #: The plates between the flange and the hand's mounting face, summed; 0.0 for a flange hand.
    coupling_mm: float
    #: The registry jaw.
    jaw: ParallelJawSpec
    #: The approach axis ``robot.gripper.tool_frame`` declares, in flange axes, or ``None`` where it is
    #: undeclared.
    declared_approach: tuple[float, float, float] | None = None

    @property
    def approach_disagreement_deg(self) -> float | None:
        """Degrees between the declared approach and the hand model's, or ``None`` with no frame declared."""
        if self.declared_approach is None:
            return None
        return _angle_deg(self.declared_approach, HAND_APPROACH_IN_TOOL0)


def sphere_map_path(model: str) -> Path:
    """The committed sphere map for the hand called ``model``, named after the hand, so no table maps names."""
    return _MAPS / f"{model}_gripper_spheres.yml"


def guard_variant_for(model: str) -> str | None:
    """The bundle name the exact-mesh guard composes with the arm, or ``None`` for the arm bundles' hand."""
    return None if model == ARM_BUNDLE_HAND else model


def _angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    cosine = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _axis_name(axis: tuple[float, float, float]) -> str:
    """``+Z`` for an axis within the tolerance of one, the rounded vector otherwise."""
    for index, name in enumerate("XYZ"):
        for sign in (1.0, -1.0):
            unit = [0.0, 0.0, 0.0]
            unit[index] = sign
            if _angle_deg(axis, (unit[0], unit[1], unit[2])) <= APPROACH_TOLERANCE_DEG:
                return f"{'+' if sign > 0 else '-'}{name}"
    return str([round(float(v), 3) for v in axis])


def declared_approach(robot_cfg: RobotConfig) -> tuple[float, float, float] | None:
    """The approach axis the cell's tool frame declares, in flange axes, or ``None`` where it is undeclared."""
    frame = robot_cfg.gripper.tool_frame
    if frame.source == "undeclared":
        return None
    column = to_rotation_matrix(np.asarray(frame.rotation_quat_xyzw, dtype=np.float64))[:, 2]
    return (float(column[0]), float(column[1]), float(column[2]))


def approach_refusal(hand: PlannerHand) -> str | None:
    """Why a hand whose declared approach and model disagree cannot be modelled, or ``None`` where they agree."""
    degrees = hand.approach_disagreement_deg
    if hand.declared_approach is None or degrees is None or degrees <= APPROACH_TOLERANCE_DEG:
        return None
    return (
        f"robot.gripper.tool_frame declares the approach along flange {_axis_name(hand.declared_approach)}, "
        f"and the committed model of {hand.model}, which the self collision guard and the planner check "
        f"against, holds the hand along flange {_axis_name(HAND_APPROACH_IN_TOOL0)}: {degrees:.0f} degrees "
        f"apart. On the Isaac cell the hand, its tool frame and the model agree, measured. A hand bolted to "
        f"a real UR flange approaches along +Z by the UR convention, so this cell's guard and planner would "
        f"model the hand {degrees:.0f} degrees from where it is. The cell refuses to build until the hand's "
        f"axis is measured on it and a model that holds it is committed."
    )


def descriptor_refusal(provenance: "object", hand: PlannerHand, *, arm: str) -> str | None:
    """Why the planner must not plan with a descriptor, or ``None`` where it was built for this arm and hand.

    ``provenance`` is the descriptor's ``_provenance`` block, as ``scripts/curobo/build_ur_config.py``
    writes it and the sidecar reports it when ready. A descriptor that says nothing about its hand
    refuses, as does one built for another arm, another hand, or another plate than
    ``robot.gripper.coupling_plates_mm`` sums to. There is no fallback to another file.
    """
    rebuild = (
        f"Rebuild it on the box: ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py {arm} "
        f"--gripper {hand.model}" + (f" --coupling-mm {hand.coupling_mm:g}" if hand.coupling_mm else "")
    )
    if not isinstance(provenance, dict):
        return (
            f"the cuRobo descriptor this planner loaded has no _provenance, so nothing says which hand it "
            f"models, and this cell names {hand.model!r}. A descriptor built before its hand was recorded "
            f"cannot be told from one for another hand. {rebuild}"
        )
    built_arm = str(provenance.get("arm", ""))
    built_hand = str(provenance.get("gripper_key", ""))
    built_plate = float(provenance.get("coupling_mm") or 0.0)
    if built_arm.lower() != arm.lower():
        return (
            f"the cuRobo descriptor this planner loaded was built for the {built_arm!r} arm, and this cell "
            f"drives {arm!r}: the planner would route another robot's links. {rebuild}"
        )
    if built_hand != hand.model:
        return (
            f"the cuRobo descriptor this planner loaded models the hand {built_hand!r}, and this cell names "
            f"{hand.model!r}: the planner would route another hand's geometry. {rebuild}"
        )
    if abs(built_plate - float(hand.coupling_mm)) > 1e-6:
        return (
            f"the cuRobo descriptor this planner loaded places {hand.model} {built_plate:g} mm from the "
            f"flange, and robot.gripper.coupling_plates_mm sums to {hand.coupling_mm:g} mm: the planner "
            f"would model the hand {abs(float(hand.coupling_mm) - built_plate):g} mm from where it is. "
            f"{rebuild}"
        )
    return None


def hand_geometry_model(
    self_collision: SelfCollisionSafetyConfig, arm_model: Maybe[str] = UNSET,
) -> str | None:
    """The arm model on which this guard would read hand geometry, or ``None`` where it reads none.

    Hand geometry is read by the exact-mesh guard (``backend: fcl``) once a model resolves, and the
    order is the guard's own (``SelfCollisionGuard.model_for``): ``kinematics_model`` first, then the
    arm the driver knows it is. A stood-down guard, the capsule proxy, and a cell with no resolvable
    arm read none.
    """
    if not self_collision.enforce or self_collision.backend != "fcl":
        return None
    if self_collision.kinematics_model is not None:
        return str(self_collision.kinematics_model)
    if chosen(arm_model):
        return str(arm_model)
    return None


def unset_hand_refusal(model: str) -> tuple[str, str]:
    """What is wrong with a cell that reads hand geometry on ``model`` and names no hand, and the fix."""
    try:
        known = ", ".join(available_grippers()) or "no hand"
    except ConfigError as exc:
        known = f"nothing readable ({exc})"
    what = (
        f"robot.gripper.model is unset, and this cell's self collision guard reads hand geometry: it checks "
        f"exact meshes (safety.self_collision.backend fcl) on the {model} arm, and which hand those meshes "
        f"are is the hand's name."
    )
    fix = (
        f"Name the hand in the cell profile, robot.gripper.model: <registry name>; the registry holds "
        f"{known}. The base robot.yaml leaves it unset on purpose, so no overlay inherits a hand."
    )
    return what, fix


def planner_hand(robot_cfg: RobotConfig, *, data_dir: str | Path | None = None) -> Maybe[PlannerHand]:
    """The hand ``robot_cfg`` names, resolved, or :data:`UNSET` when it names none.

    Raises ``ConfigError`` for a name the registry does not hold (the registry's own sentence), a hand
    with no committed sphere map, a mounting-face hand with no plates declared, and a flange hand with
    a plate.
    """
    gripper = robot_cfg.gripper
    if gripper.model is None:
        return UNSET
    spec = load_gripper(gripper.model, data_dir=data_dir, aliases=False)
    sphere_map = sphere_map_path(spec.model)
    if not sphere_map.is_file():
        raise ConfigError(
            f"robot.gripper.model is {spec.model!r} and no sphere map describes that hand: {sphere_map} is "
            f"absent, so neither the planner nor the guard can model it. Fit it from the hand's baked bundle "
            f"with python -m src.robot.safety.planning.robot.build_gripper_spheres --variant <bundle> "
            f"--out {sphere_map.name}, or from its mesh with --mesh."
        )
    provenance = (yaml.safe_load(sphere_map.read_text(encoding="utf-8")) or {}).get("_provenance") or {}
    origin = provenance.get("origin")
    if origin not in (FLANGE, MOUNTING_FACE):
        raise ConfigError(
            f"{sphere_map.name} declares origin {origin!r}, and a sphere map starts at the {FLANGE!r} or at "
            f"the hand's {MOUNTING_FACE!r}. Where the hand sits on the flange follows from it, so it is not "
            f"guessed."
        )
    plates = gripper.coupling_plates_mm
    coupling = None if plates is None else float(sum(plates))
    if origin == MOUNTING_FACE and coupling is None:
        raise ConfigError(
            f"robot.gripper.model is {spec.model!r}, whose sphere map {sphere_map.name} starts at the hand's "
            f"own {MOUNTING_FACE}, and robot.gripper.coupling_plates_mm is unset, so where the hand sits on "
            f"the flange is unknown and this will not guess it. Measure every plate between the flange and "
            f"the hand's mounting face and write them, for example coupling_plates_mm: [20.0], or [] for a "
            f"hand bolted straight to the flange. It is the same bench measurement as the plate term in "
            f"robot.gripper.tool_frame.offset_mm."
        )
    if origin == FLANGE and coupling not in (None, 0.0):
        raise ConfigError(
            f"robot.gripper.model is {spec.model!r}, whose sphere map {sphere_map.name} already sits at the "
            f"{FLANGE}: the arm asset it was fitted from placed the hand where it is bolted. "
            f"robot.gripper.coupling_plates_mm adds {coupling:g} mm to that, which would move the hand away "
            f"from where it was measured. Delete the plates, or fit a map whose origin is {MOUNTING_FACE}."
        )
    return PlannerHand(
        model=spec.model,
        sphere_map=sphere_map,
        origin=str(origin),
        guard_variant=guard_variant_for(spec.model),
        coupling_mm=coupling or 0.0,
        jaw=spec.jaw,
        declared_approach=declared_approach(robot_cfg),
    )
