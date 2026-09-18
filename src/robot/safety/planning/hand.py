"""The hand a cell's planner and guard model, derived from the one name the cell gives it.

``robot.gripper.model`` is the only name a hand has. Everything the motion stack needs to know about
the geometry at the end of the arm follows from that name and from committed files, so no second key
can name another hand:

* the registry file ``config/grippers/<model>.yaml``, for the jaw;
* the committed sphere map ``robot/<model>_gripper_spheres.yml`` beside this file, for the planner,
  and the ``origin`` its provenance declares;
* the guard's mesh bundle: the arm's own for the hand every committed arm bundle carries, the hand's
  own bundle otherwise (``environment.compose_collision_meshes`` composes the arm in);
* the coupling, the thicknesses of ``robot.gripper.coupling_plates`` summed;
* the hand model's own approach axis, flange +Y in every committed model, which every placement
  turns from;
* the placement, the one rotation that puts the hand model on the flange, derived once from the
  declared tool frame (``_hand_placement``).

An unset name is :data:`UNSET`, never the 2F-85. A caller that reads hand geometry refuses it, and
:func:`unset_hand_refusal` words that refusal once for every such caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

from src.config.grippers import available_grippers, load_gripper, tree_hand_refusal
from src.config.loader import ConfigError
from src.contracts import UNSET, Maybe, chosen

from ._curobo_body_links import HAND_LINK
from ._declared_body import Box, stacked_boxes
from ._hand_bundle import MODEL_APPROACH
from ._hand_placement import TOLERANCE_DEG, HandPlacement, PlacementRefused
from .environment import HandProvenance, hand_mesh_bundle, hand_provenance, hand_writers_sentence, sphere_map_writer_sentence
from .robot.gripper_spheres import FLANGE, MOUNTING_FACE

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.grippers.gripper_schema import ParallelJawSpec
    from src.config.schema.robot import RobotConfig, SelfCollisionSafetyConfig

    from .body_link import HandLink
    from .curobo_client import SidecarIdentity

__all__ = [
    "APPROACH_TOLERANCE_DEG",
    "ARM_BUNDLE_HAND",
    "HAND_APPROACH_IN_TOOL0",
    "PlannerHand",
    "declared_placement",
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
#: registry hand. A hand bolted to a real UR flange approaches along +Z by the UR convention, and the
#: placement derived from the declared tool frame turns the model there: this is the axis every
#: placement turns from, not an axis a cell has to declare. It is ``_hand_bundle.MODEL_APPROACH``,
#: which the loader holds every hand bundle to.
HAND_APPROACH_IN_TOOL0: Final = MODEL_APPROACH

#: How far a committed model's fingers may lean from :data:`HAND_APPROACH_IN_TOOL0`, in degrees: the
#: tolerance ``_hand_placement`` snaps a declared frame within, so the model axis and the derivation
#: keep one tolerance.
APPROACH_TOLERANCE_DEG: Final = TOLERANCE_DEG

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
    #: Where the hand model sits on tool0, derived from the declared tool frame; :data:`UNSET` where
    #: the declaration places no hand, and then ``placement_refusal`` says why. A refused placement is
    #: not readable as a rotation.
    placement: Maybe[HandPlacement] = UNSET
    #: Why the declared tool frame places no hand, or ``None`` where it places one.
    placement_refusal: str | None = None
    #: What the hand's own bundle says about where its geometry came from, or ``None`` where the hand
    #: has no bundle of its own to ask.
    provenance: HandProvenance | None = None
    #: The plates this cell measured across, as boxes in the hand model's own axes, stacked from the
    #: flange. Empty where no plate declared a cross section, which is every shipped cell.
    coupling_boxes: tuple[Box, ...] = ()
    #: The plates that hold the hand out and became no body, by name, so a reader can say which ones.
    coupling_undeclared: tuple[str, ...] = ()
    #: Every declared plate as the cell wrote it, flange first: (name, thickness_mm, cross_section_mm
    #: or None). A refusal spells these as ``matrix_gate.py --plate`` so the command measures this
    #: cell's plates.
    plates: tuple[tuple[str, float, tuple[float, float] | None], ...] = ()

    @property
    def models_an_envelope(self) -> bool:
        """Whether this hand's geometry is an envelope built from its registry block rather than a scan.

        An envelope has equal standing and is not refused anywhere. It is reported because it costs
        room: the two shipped hands that could be measured needed 5.73 mm and 9.50 mm of inflation
        before an envelope enclosed the body it stands for, and that is room the planner and the
        guard both give away.
        """
        return self.provenance is not None and self.provenance.from_dimensions


def sphere_map_path(model: str) -> Path:
    """The committed sphere map for the hand called ``model``, named after the hand, so no table maps names."""
    return _MAPS / f"{model}_gripper_spheres.yml"


def guard_variant_for(model: str) -> str | None:
    """The bundle name the exact-mesh guard composes with the arm, or ``None`` for the arm bundles' hand."""
    return None if model == ARM_BUNDLE_HAND else model


def declared_placement(robot_cfg: RobotConfig) -> tuple[Maybe[HandPlacement], str | None]:
    """Where the cell's declared tool frame puts the hand model on the flange, or why it puts it nowhere.

    An undeclared frame gives the undeclared placement, the hand model's own axes; the real driver
    refuses to connect with it anyway. A declaration ``_hand_placement`` refuses gives :data:`UNSET`
    and its sentence, so no reader can take a refused placement for a rotation. That sentence is the
    only refusal a declared frame meets: an approach along +Z, which a real UR flange declares, is
    placed rather than refused.
    """
    frame = robot_cfg.gripper.tool_frame
    if frame.source == "undeclared":
        return HandPlacement.undeclared(), None
    try:
        return HandPlacement.from_quaternion_xyzw(frame.rotation_quat_xyzw), None
    except PlacementRefused as exc:
        return UNSET, f"robot.gripper.tool_frame places no hand model: {exc}"


def descriptor_refusal(identity: "SidecarIdentity", link: "HandLink", *, arm: str) -> str | None:
    """Why the planner must not plan with what its sidecar loaded, or ``None`` for this arm's
    descriptor with this cell's hand added.

    ``identity`` is what the sidecar reported when it became ready. A descriptor that says nothing of
    itself refuses, as does one built for another arm. A descriptor carries no hand, so one that names
    a hand, or does not say it carries none, is a per-hand file from an earlier build and refuses by
    name; and a sidecar that added no body link called ``hand`` would plan without the hand this cell
    carries. There is no fallback to another file. The plate and the placement are in the link the
    client sent, so nothing about them is compared here.
    """
    build = f"Build the arm on the box: ext_deps/curobo_env/python.exe scripts/curobo/build_ur_config.py {arm}"
    provenance = identity.provenance
    if not isinstance(provenance, dict):
        return (
            f"the cuRobo descriptor this planner loaded has no _provenance, so nothing says which arm it describes or "
            f"that no hand is in it, and this cell adds {link.model} to it. A descriptor built before its provenance "
            f"was recorded cannot be told from one for another robot. {build}"
        )
    built_arm = str(provenance.get("arm", ""))
    if built_arm.lower() != arm.lower():
        return (
            f"the cuRobo descriptor this planner loaded was built for the {built_arm!r} arm, and this cell drives "
            f"{arm!r}: the planner would route another robot's links. {build}"
        )
    carried = provenance.get("gripper_key")
    if carried or provenance.get("carries_hand") is not False:
        what = f"models the hand {carried!r}" if carried else "does not say it carries no hand"
        return (
            f"the cuRobo descriptor this planner loaded {what}, and this cell adds {link.model} as a body link: the "
            f"planner would model the hand twice, or a hand that is not there. It is a per hand file from an earlier "
            f"build, and a descriptor is built per arm. {build}"
        )
    if HAND_LINK not in identity.body_names:
        if identity.body_names:
            said = f"reported the body links {', '.join(identity.body_names)}"
        else:
            said = "reported no body link" if chosen(identity.bodies) else "said nothing about body links"
        return (
            f"the cuRobo sidecar {said}, and this cell's hand {link.model} is added as the body link {HAND_LINK!r}: the "
            f"planner would plan without a hand. A sidecar older than body links cannot add one; restart the planner"
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


def wrist_body_reader(robot_cfg: Any) -> str | None:
    """Why this cell reads a wrist camera's geometry, as a phrase, or ``None`` where it reads none.

    A cell reads it when its UR planner is cuRobo, which carries the camera as a body link, or when
    its self-collision guard reads hand geometry (:func:`hand_geometry_model`), which holds the
    camera's parts beside the hand's. A cell that reads neither models no hand and so no camera, and
    resolves no wrist body.
    """
    vendor = str(getattr(robot_cfg, "vendor", "") or "")
    ur = getattr(robot_cfg, "ur", None)
    if vendor == "ur" and getattr(ur, "motion_planner", None) == "curobo":
        return "robot.ur.motion_planner is curobo"
    arm_model: Maybe[str] = str(ur.model) if vendor == "ur" and ur is not None else UNSET
    self_collision = getattr(getattr(robot_cfg, "safety", None), "self_collision", None)
    if self_collision is None:
        return None
    model = hand_geometry_model(self_collision, arm_model)
    if model is not None:
        return f"the exact mesh guard reads hand geometry on the {model} arm"
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


#: What wrote every map the planner may read: the cover fit, which leaves no hole by construction.
COVER_FIT_WRITER: Final = "scripts/curobo/fit_cover_spheres.py"


def cover_fit_refusal(provenance: dict, model: str) -> str | None:
    """Why a sphere map is not a cover fit of ``model``'s own bundle, or ``None`` where it is.

    Held where a hand resolves, so a grid fit with holes, or a map fitted from another hand, never
    reaches the planner and the self filter of a cell that names it.
    """
    writer = provenance.get("generated_by")
    if writer != COVER_FIT_WRITER:
        return (
            f"was written by {writer!r}, not by the cover fit {COVER_FIT_WRITER}: a grid fit left every "
            f"shipped hand a hole of 16.5 to 19.0 mm, a pose the planner calls clear while the hand is there"
        )
    source, own = provenance.get("source"), f"{model}_hand_meshes.npz"
    if source != own:
        return f"was fitted from {source!r}, and a map of {model} is fitted from {own}"
    rows = {str(row.get("body")): row for row in provenance.get("bodies") or [] if isinstance(row, dict)}
    for part in ("gripper", "lfinger", "rfinger"):
        if part not in rows:
            return f"records no {part} in its bodies, so nothing says the {part} is covered"
        uncovered = rows[part].get("fresh_uncovered_max_mm")
        if uncovered is None or float(uncovered) >= 0.0:
            return (
                f"leaves {part} uncovered by {uncovered} mm on fresh surface samples, and a cover fit holds "
                f"every sample inside a sphere"
            )
    return None


def planner_hand(robot_cfg: RobotConfig, *, data_dir: str | Path | None = None) -> Maybe[PlannerHand]:
    """The hand ``robot_cfg`` names, resolved, or :data:`UNSET` when it names none.

    Raises ``ConfigError`` for a name the registry does not hold (the registry's own sentence), a hand
    with no committed sphere map, a mounting-face hand with no plates declared, and a flange hand with
    a plate.

    ``data_dir`` is the cell's config tree, ``None`` for the repository's. The hand always resolves
    from the repository's registry, because its sphere map, bundle, retract rows and evidence were
    written from it; a tree whose own ``grippers/`` describes the named hand differently, or describes
    a hand the repository does not, is refused first, naming both files.
    """
    gripper = robot_cfg.gripper
    if gripper.model is None:
        return UNSET
    refusal = tree_hand_refusal(gripper.model, data_dir=data_dir)
    if refusal is not None:
        raise ConfigError(refusal)
    spec = load_gripper(gripper.model, aliases=False)
    sphere_map = sphere_map_path(spec.model)
    if not sphere_map.is_file():
        # Why this cell needs the file depends on its planner: a UR ik cell reads nothing of the map
        # after construction, and the build reads its origin to decide the plates.
        if str(getattr(getattr(robot_cfg, "ur", None), "motion_planner", "")) == "curobo":
            why = ("its planner adds the hand as a body link from this map when it starts, and the build reads "
                   "the map's origin to decide whether robot.gripper.coupling_plates are required or refused")
        else:
            why = ("no planner starts on this cell, and the build still reads the map's origin to decide "
                   "whether robot.gripper.coupling_plates are required or refused")
        body = "" if hand_mesh_bundle(spec.model).is_file() else f" It has no body yet either: {hand_writers_sentence(spec.model)}."
        raise ConfigError(
            f"robot.gripper.model is {spec.model!r} and no sphere map describes that hand: {sphere_map} is "
            f"absent, and {why}.{body} Then {sphere_map_writer_sentence(spec.model)}."
        )
    provenance = (yaml.safe_load(sphere_map.read_text(encoding="utf-8")) or {}).get("_provenance") or {}
    origin = provenance.get("origin")
    if origin not in (FLANGE, MOUNTING_FACE):
        raise ConfigError(
            f"{sphere_map.name} declares origin {origin!r}, and a sphere map starts at the {FLANGE!r} or at "
            f"the hand's {MOUNTING_FACE!r}. Where the hand sits on the flange follows from it, so it is not "
            f"guessed."
        )
    refused = cover_fit_refusal(provenance, spec.model)
    if refused is not None:
        raise ConfigError(f"{sphere_map.name}: {refused} {sphere_map_writer_sentence(spec.model).capitalize()}.")
    coupling = gripper.coupling_mm
    if origin == MOUNTING_FACE and coupling is None:
        raise ConfigError(
            f"robot.gripper.model is {spec.model!r}, whose sphere map {sphere_map.name} starts at the hand's "
            f"own {MOUNTING_FACE}, and robot.gripper.coupling_plates is unset, so where the hand sits on "
            f"the flange is unknown and this will not guess it. Measure every plate between the flange and "
            f"the hand's mounting face and write them, for example coupling_plates: "
            f"[{{name: adapter, thickness_mm: 20.0}}], or [] for a hand bolted straight to the flange. It is "
            f"the same bench measurement as the plate term in robot.gripper.tool_frame.offset_mm."
        )
    if origin == FLANGE and coupling not in (None, 0.0):
        told = hand_provenance(spec.model, getattr(robot_cfg.safety.self_collision, "mesh_dir", None))
        remedy = (
            f" This hand's body was written from its registry numbers as a flange hand: if its grasp_centre_mm "
            f"was measured from the hand's own mounting face, write it again with "
            f"scripts/grippers/write_hand_from_dimensions.py {spec.model} --origin mounting_face and refit its "
            f"map with scripts/curobo/fit_cover_spheres.py --hand {spec.model} --write."
        ) if told is not None and told.from_dimensions else ""
        raise ConfigError(
            f"robot.gripper.model is {spec.model!r}, whose sphere map {sphere_map.name} already sits at the "
            f"{FLANGE}: the arm asset it was fitted from placed the hand where it is bolted. "
            f"robot.gripper.coupling_plates adds {coupling:g} mm to that, which would move the hand away "
            f"from where it was measured. Delete the plates, or fit a map whose origin is {MOUNTING_FACE}.{remedy}"
        )
    # The stack grows along the hand model's own approach, from the flange. A mounting-face hand carries
    # its plates as bodies where it declared a cross section; a flange hand has no plates at all, so there
    # is nothing to stack.
    boxes, undeclared = stacked_boxes(
        [(plate.name, plate.thickness_mm, plate.cross_section_mm) for plate in (gripper.coupling_plates or ())],
        axis=HAND_APPROACH_IN_TOOL0.index(1.0),
    )
    placement, placement_refusal = declared_placement(robot_cfg)
    return PlannerHand(
        model=spec.model,
        sphere_map=sphere_map,
        origin=str(origin),
        guard_variant=guard_variant_for(spec.model),
        coupling_mm=coupling or 0.0,
        jaw=spec.jaw,
        placement=placement,
        placement_refusal=placement_refusal,
        provenance=hand_provenance(spec.model, getattr(robot_cfg.safety.self_collision, "mesh_dir", None)),
        coupling_boxes=tuple(boxes),
        coupling_undeclared=undeclared,
        plates=tuple(
            (str(plate.name), float(plate.thickness_mm),
             None if plate.cross_section_mm is None else (float(plate.cross_section_mm[0]), float(plate.cross_section_mm[1])))
            for plate in (gripper.coupling_plates or ())
        ),
    )
