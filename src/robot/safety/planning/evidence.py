"""What admits one arm to one hand: a file somebody measured, named after the combination it measured.

Which UR arm can plan with which gripper is a measurement. The retract table knows one pose per pair, which is enough
to start a sidecar and says nothing about the rest of the space, and hand bundles carry no list of the arms they may
be composed onto. Neither is evidence that the planner's model and the exact-mesh guard describe the same robot for
that pair, and that is the claim a cell makes every time it plans.

The key is the combination, not the pair. One arm and one hand disagree at different plates, placements and planner
margins, so all of them are in the file name: at a 10 mm margin the UR5 and the UR10 have no pose at all and at 4 mm
they do, with the same arm, the same hand and the same meshes.

An unreported hash is not a hash that matched. A sidecar too old to say what it loaded reports :data:`UNSET`, and a
check keyed on hashes refuses that rather than skipping the comparison. Skipping is what an admission list inherited
from the bundles amounts to: it answers for pairings nobody measured, and four of the pairs such a list admitted are
pairs cuRobo refuses to plan from at all.

Both drivers call :func:`planner_evidence_refusal` where a planner starts, and the real cell checklist and the doctor
read the file half through :func:`desk_evidence_refusal` before anything starts. ``scripts/curobo/matrix_gate.py``
writes the files into ``robot/evidence``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Final

from src.contracts import UNSET, Maybe, chosen

__all__ = [
    "CombinationEvidence",
    "Criterion",
    "EVIDENCE_DIR",
    "Measured",
    "REQUIRED_CRITERION",
    "desk_evidence_refusal",
    "evidence_path",
    "evidence_refusal",
    "guard_sha256",
    "planner_evidence_refusal",
]


#: Beside the committed sphere maps, and a data directory rather than a package: nothing imports from it.
EVIDENCE_DIR: Final = Path(__file__).resolve().parent / "robot" / "evidence"

#: What writes a file that is missing, quoted in the refusal so an operator has the next command rather than a fact.
_WRITER: Final = "scripts/curobo/matrix_gate.py"


class Criterion(IntEnum):
    """How much a measurement proved. Ordered, so raising the bar later is a value and not a new mechanism.

    Two rungs, and the ladder is deliberately short: a rung nobody measures is a capability this repository would then
    claim. ``NONE`` is a combination that was measured and did not meet the bar, which is different from one nobody
    measured at all; that one has no file.
    """

    NONE = 0
    B1 = 1

    def __str__(self) -> str:  # pragma: no cover (trivial)
        return self.name.lower()


#: The rung a cell must reach before it may plan with a combination. Raising it invalidates every file that recorded
#: a lower one, which is the point: the files say what they proved, not what the reader hoped.
REQUIRED_CRITERION: Final = Criterion.B1


@dataclass(frozen=True)
class Measured:
    """What the gate counted over one combination's pose set.

    ``false_clears`` is here and is not part of :data:`REQUIRED_CRITERION`. The sphere model reaches past the body by
    construction, because the cover fit places each link at a stated reach, so it calls poses collisions that the
    exact meshes clear, and a rung asking the two to agree everywhere would be a rung nothing can reach. It is recorded
    because the number is the price of the fit and somebody should be able to watch it move.
    """

    #: How many configurations were judged, the retract first (``_matrix_gate.pose_set``).
    poses: int
    #: Which ones. Two runs at different seeds judged different robots' worth of space, and a count alone cannot
    #: tell them apart: a file recording 1,483 poses says nothing about whether they were these 1,483.
    poses_sha256: str
    #: Whether the planner accepts the pose its own sidecar would start from. Without this it never becomes ready.
    retract_clear: bool
    #: How many of them the planner refused.
    refused: int
    #: How many the planner cleared and the exact meshes did not.
    false_clears: int
    #: Poses the planner called a collision without naming a pair, or named one at a depth that disagrees. A model
    #: contradicting its own report is the one thing a verdict cannot be read through.
    attribution_disagreements: int
    #: The calibration, in millimetres: the exact clearance at this combination's own retract, measured again now,
    #: against what the committed retract table recorded for it. Both came from the same ruler on the same geometry,
    #: so a difference means something moved between the table and now. It is the cheapest check there is on whether
    #: the gate is talking about the robot it thinks it is, and it is the shape of check that catches a frame defect
    #: in a sphere fit.
    #:
    #: Its floor is not zero and cannot be: the table records clearances to three decimals, so the comparison carries
    #: up to 5e-4 mm of the table's own rounding, measured at 1.5e-4 mm on the ur5e with the 2F-85 and 2.6e-4 mm on
    #: the ur3e with the Hand-E. The tolerance below is set above that floor and far below anything a moved body could
    #: produce.
    control_mm: float

    def render(self) -> str:
        return (f"{self.poses:,} poses ({self.poses_sha256[:8]}), {self.refused:,} refused, "
                f"{self.false_clears:,} false clears, "
                f"{self.attribution_disagreements} attribution disagreements, control {self.control_mm:.3f} mm, "
                f"retract {'clear' if self.retract_clear else 'refused'}")

    def to_dict(self) -> dict[str, Any]:
        return {"poses": int(self.poses), "poses_sha256": self.poses_sha256,
                "retract_clear": bool(self.retract_clear), "refused": int(self.refused),
                "false_clears": int(self.false_clears),
                "attribution_disagreements": int(self.attribution_disagreements),
                "control_mm": float(self.control_mm)}

    @classmethod
    def from_dict(cls, held: "dict[str, Any]") -> Measured:
        return cls(poses=int(held["poses"]), poses_sha256=str(held["poses_sha256"]),
                   retract_clear=bool(held["retract_clear"]),
                   refused=int(held["refused"]), false_clears=int(held["false_clears"]),
                   attribution_disagreements=int(held["attribution_disagreements"]),
                   control_mm=float(held["control_mm"]))


#: How far the control may be from zero and still be zero, in millimetres. Above the 5e-4 the retract table's own
#: three decimals put into the comparison, and far below what a body that moved would produce: the smallest real
#: displacement met across the UR arm bundles was 0.5 mm, a mesh origin Isaac's URDFs omit and UR declares.
_CONTROL_TOLERANCE_MM: Final = 1e-3


def guard_sha256(parts: "dict[str, tuple[Any, Any, int]]") -> str:
    """One hash over the arrays the exact-mesh guard holds, from ``_fcl_self_collision.composed_parts``.

    Over the placed arrays rather than the bundle files: a hand turned by a declared tool frame, a plate added to a
    mounting-face map, and the coupling bodies a declared plate becomes all move the geometry without touching a
    bundle. The file names and their bytes would read identical through every one of those.

    Name, frame, dtype, shape and bytes of each array, in sorted order, so it does not depend on how a dict was built.
    """
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(parts):
        vertices, faces, frame = parts[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(int(frame)).encode("utf-8"))
        for array in (vertices, faces):
            held = np.ascontiguousarray(array)
            digest.update(f"{held.dtype}{held.shape}".encode("utf-8"))
            digest.update(held.tobytes())
    return digest.hexdigest()


def evidence_path(
    *,
    arm: str,
    hand: str,
    coupling_mm: float,
    approach: str,
    closing: str,
    planner_margin_mm: float,
    attach_spheres: int,
    evidence_dir: "Path | None" = None,
) -> Path:
    """Where the file for one combination lives. The name is the combination, so no table maps one to the other."""
    if not approach or not closing:
        raise ValueError(
            f"{arm} with {hand} has no evidence path, because its declared tool frame places the hand nowhere "
            f"(approach {approach!r}, closing {closing!r}). A path here would suggest the combination could be "
            f"measured, and a placement nobody can derive cannot be."
        )
    folder = EVIDENCE_DIR if evidence_dir is None else Path(evidence_dir)
    return folder / (f"{arm}_{hand}_c{float(coupling_mm):g}mm_{approach}{closing}"
                     f"_m{float(planner_margin_mm):g}mm_a{int(attach_spheres)}.json")


@dataclass(frozen=True)
class CombinationEvidence:
    """One measured combination: what was measured, what it was measured on, and what it proved.

    The hashes are what tie the file to a robot rather than to a name. ``composed_sha256`` covers the whole config
    the planner loaded, so the hand's link, the coupling bodies, the margin and the attach slots are all inside it: a
    cell that bolts a tool changer on changes this and needs its own file. ``guard_sha256`` covers the arrays the
    exact-mesh guard composed, for the same reason on the other side.
    """

    arm: str
    hand: str
    coupling_mm: float
    approach: str
    closing: str
    planner_margin_mm: float
    attach_spheres: int
    #: What the exact-mesh guard kept clear while this was measured. A cell at another one measured something else.
    guard_margin_mm: float
    composed_sha256: str
    urdf_sha256: str
    arm_descriptor_sha256: str
    hand_map_sha256: str
    guard_sha256: str
    measured: Measured

    @property
    def criterion(self) -> Criterion:
        """The rung this measurement reaches. Derived, never stored: a stored one could disagree with its numbers."""
        clean = (self.measured.retract_clear
                 and self.measured.attribution_disagreements == 0
                 and abs(self.measured.control_mm) <= _CONTROL_TOLERANCE_MM)
        return Criterion.B1 if clean else Criterion.NONE

    @property
    def path(self) -> Path:
        """The file this record belongs in, derived from the record, so a copy into another name is detectable."""
        return evidence_path(arm=self.arm, hand=self.hand, coupling_mm=self.coupling_mm, approach=self.approach,
                             closing=self.closing, planner_margin_mm=self.planner_margin_mm,
                             attach_spheres=self.attach_spheres)

    def render(self) -> str:
        return (f"{self.arm} with {self.hand} at {self.coupling_mm:g} mm, {self.approach}{self.closing}, planner "
                f"margin {self.planner_margin_mm:g} mm, guard {self.guard_margin_mm:g} mm: {self.criterion} "
                f"({self.measured.render()})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "hand": self.hand, "coupling_mm": float(self.coupling_mm),
            "approach": self.approach, "closing": self.closing,
            "planner_margin_mm": float(self.planner_margin_mm), "attach_spheres": int(self.attach_spheres),
            "guard_margin_mm": float(self.guard_margin_mm),
            "composed_sha256": self.composed_sha256, "urdf_sha256": self.urdf_sha256,
            "arm_descriptor_sha256": self.arm_descriptor_sha256, "hand_map_sha256": self.hand_map_sha256,
            "guard_sha256": self.guard_sha256,
            "criterion": str(self.criterion),
            "measured": self.measured.to_dict(),
        }

    @classmethod
    def from_dict(cls, held: "dict[str, Any]") -> CombinationEvidence:
        """The record a file holds. ``criterion`` in the file is a reading, not an input: it is derived again here."""
        return cls(
            arm=str(held["arm"]), hand=str(held["hand"]), coupling_mm=float(held["coupling_mm"]),
            approach=str(held["approach"]), closing=str(held["closing"]),
            planner_margin_mm=float(held["planner_margin_mm"]), attach_spheres=int(held["attach_spheres"]),
            guard_margin_mm=float(held["guard_margin_mm"]),
            composed_sha256=str(held["composed_sha256"]), urdf_sha256=str(held["urdf_sha256"]),
            arm_descriptor_sha256=str(held["arm_descriptor_sha256"]),
            hand_map_sha256=str(held["hand_map_sha256"]), guard_sha256=str(held["guard_sha256"]),
            measured=Measured.from_dict(held["measured"]),
        )

    def evidence_bytes(self) -> bytes:
        """The file's exact bytes.

        No timestamps and no float spelling beyond what the numbers carry, so a second run over an unchanged tree
        writes the same file and a diff is a real change.
        """
        return (json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n").encode("utf-8")

    def without(self, **over: Any) -> CombinationEvidence:
        """A copy with fields replaced, for a check that has to make one field wrong on purpose."""
        return replace(self, **over)


def gate_command(
    *,
    arm: str,
    hand: str,
    coupling_mm: float,
    approach: str,
    closing: str,
    planner_margin_mm: float,
    guard_margin_mm: float,
    attach_spheres: int,
    plates: "Maybe[tuple]" = UNSET,
) -> str:
    """The whole ``matrix_gate.py`` command that measures exactly this combination.

    The rotation is the declared tool frame that places the hand ``approach`` + ``closing``; ``plates``, where given,
    are spelled one ``--plate`` each with their cross sections, and otherwise the coupling is ``--coupling-mm``.
    Pasted, it writes the file a refusal looked for, and parsed back it names every number the cell asked for.
    """
    from ._hand_placement import placement_quaternion_xyzw

    parts = [".venv/Scripts/python.exe", _WRITER, "--arm", arm, "--hand", hand]
    if chosen(plates) and plates:
        for name, thickness, section in plates:
            spec = f"{name}:{float(thickness):g}"
            if section is not None:
                spec += f":{float(section[0]):g},{float(section[1]):g}"
            parts += ["--plate", spec]
    elif float(coupling_mm):
        parts += ["--coupling-mm", f"{float(coupling_mm):g}"]
    rotation = placement_quaternion_xyzw(approach, closing)
    parts += ["--planner-margin-mm", f"{float(planner_margin_mm):g}", "--guard-margin-mm", f"{float(guard_margin_mm):g}",
              "--attach", str(int(attach_spheres)), "--tool-rotation-xyzw", *(repr(v) for v in rotation), "--write"]
    return " ".join(parts)


def _hash_refusal(name: str, wanted: "Maybe[str]", recorded: str, evidence: CombinationEvidence, command: str) -> "str | None":
    """Why one hash does not admit this cell, or ``None``. UNSET is a refusal and never a skip."""
    if not chosen(wanted):
        return (f"{evidence.path.name} was measured against a {name} of {recorded[:12]}..., and this sidecar did not "
                f"say what it loaded. A hash nobody reported is not a hash that matched, so this combination is not "
                f"admitted here. Start a sidecar that reports its own identity.")
    if str(wanted) != recorded:
        return (f"{evidence.path.name} was measured on {name} {recorded[:12]}... and this cell has {str(wanted)[:12]}"
                f"..., so the robot it proves is not the robot in front of it. Measure it again: {command}")
    return None


def _read(
    *,
    arm: str,
    hand: str,
    coupling_mm: float,
    approach: str,
    closing: str,
    planner_margin_mm: float,
    attach_spheres: int,
    evidence_dir: "Path | None",
    command: str,
) -> "tuple[CombinationEvidence | None, str | None]":
    """The file for one combination, or why there is none to read. Shared by the desk and the start."""
    try:
        path = evidence_path(arm=arm, hand=hand, coupling_mm=coupling_mm, approach=approach, closing=closing,
                             planner_margin_mm=planner_margin_mm, attach_spheres=attach_spheres,
                             evidence_dir=evidence_dir)
    except ValueError as refused:
        return None, str(refused)

    if not path.is_file():
        return None, (f"nothing has measured {arm} with {hand} at a {float(coupling_mm):g} mm plate, "
                      f"{approach}{closing}, a {float(planner_margin_mm):g} mm planner margin and "
                      f"{int(attach_spheres)} attach slots: {path.name} is not there. Measure it and commit the file it "
                      f"writes: {command}")
    try:
        evidence = CombinationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as unreadable:  # noqa: BLE001 (a file nobody can read admits nothing, whatever broke it)
        return None, (f"{path.name} cannot be read as a combination record ({type(unreadable).__name__}: "
                      f"{unreadable}). Measure it again rather than repairing it by hand: {command}")
    if evidence.path.name != path.name:
        return None, (f"{path.name} holds a record that names {evidence.path.name}, so it was copied rather than "
                      f"measured for this combination. Measure this one: {command}")
    return evidence, None


def _file_refusal(
    evidence: CombinationEvidence, *, hand_map_sha256: str, guard_sha256: str, guard_margin_mm: float, command: str,
) -> "str | None":
    """What the file says about the geometry a cell computes without a sidecar: its hand map, its guard, its rung."""
    name = evidence.path.name
    if hand_map_sha256 != evidence.hand_map_sha256:
        return (f"{name} was measured on hand_map_sha256 {evidence.hand_map_sha256[:12]}... and this cell reads "
                f"{hand_map_sha256[:12]}..., so the hand it proves is not the hand in front of it. Measure it again: {command}")
    if guard_sha256 != evidence.guard_sha256:
        return (f"{name} was measured on guard_sha256 {evidence.guard_sha256[:12]}... and this cell composes "
                f"{guard_sha256[:12]}..., so the body the exact mesh guard judges moved since. That is what a plate "
                f"becoming a body does, and it is why it moves the evidence. Measure it again: {command}")
    if abs(float(guard_margin_mm) - evidence.guard_margin_mm) > 1e-9:
        return (f"{name} was measured against a guard keeping {evidence.guard_margin_mm:g} mm clear and this cell "
                f"keeps {float(guard_margin_mm):g} mm, so what it proved is not what this cell asks. Measure it at this "
                f"cell's margin: {command}")
    if evidence.criterion < REQUIRED_CRITERION:
        return (f"{name} measured {evidence.arm} with {evidence.hand} and it did not reach {REQUIRED_CRITERION}: "
                f"{evidence.measured.render()}. This combination is not admitted until that is fixed and measured "
                f"again.")
    return None


def evidence_refusal(
    *,
    arm: str,
    hand: str,
    coupling_mm: float,
    approach: str,
    closing: str,
    planner_margin_mm: float,
    attach_spheres: int,
    guard_margin_mm: float,
    composed_sha256: "Maybe[str]",
    urdf_sha256: "Maybe[str]",
    arm_descriptor_sha256: "Maybe[str]",
    hand_map_sha256: str,
    guard_sha256: str,
    evidence_dir: "Path | None" = None,
    plates: "Maybe[tuple]" = UNSET,
) -> "str | None":
    """Why this cell may not plan with this combination, in one sentence, or ``None`` where it may.

    Both halves: the file (which rung, which hand map, which guard geometry, which guard margin) and what the running
    sidecar reports it loaded. Every branch names what it found and what to run, because the operator reading it is
    holding a cell that will not come up and needs the next command, not a verdict.
    """
    command = gate_command(arm=arm, hand=hand, coupling_mm=coupling_mm, approach=approach, closing=closing,
                           planner_margin_mm=planner_margin_mm, guard_margin_mm=guard_margin_mm,
                           attach_spheres=attach_spheres, plates=plates)
    evidence, refused = _read(arm=arm, hand=hand, coupling_mm=coupling_mm, approach=approach, closing=closing,
                              planner_margin_mm=planner_margin_mm, attach_spheres=attach_spheres,
                              evidence_dir=evidence_dir, command=command)
    if evidence is None:
        return refused
    for name, wanted, recorded in (
        ("composed_sha256", composed_sha256, evidence.composed_sha256),
        ("urdf_sha256", urdf_sha256, evidence.urdf_sha256),
        ("arm_descriptor_sha256", arm_descriptor_sha256, evidence.arm_descriptor_sha256),
    ):
        said = _hash_refusal(name, wanted, recorded, evidence, command)
        if said is not None:
            return said
    return _file_refusal(evidence, hand_map_sha256=hand_map_sha256, guard_sha256=guard_sha256,
                         guard_margin_mm=guard_margin_mm, command=command)


def _geometry(hand: Any, *, arm: str, mesh_dir: "str | None") -> "tuple[dict[str, Any] | None, str | None]":
    """The parts of a combination a cell derives from its own hand and guard, the same way the matrix gate did."""
    from src.robot.safety._fcl_self_collision import composed_parts

    from ._curobo_body_links import BodyLinkError
    from .body_link import HandLink

    if not chosen(hand.placement):
        return None, (f"{hand.model} has no evidence to look up, because its declared tool frame places it nowhere: "
                      f"{hand.placement_refusal}")
    try:
        link = HandLink.from_hand(hand)
    except BodyLinkError as exc:
        return None, str(exc)
    parts = composed_parts(arm, mesh_dir, hand.guard_variant, float(hand.coupling_mm),
                           placement=hand.placement, coupling_boxes=hand.coupling_boxes)
    return {
        "hand": hand.model, "coupling_mm": float(hand.coupling_mm),
        "approach": hand.placement.approach, "closing": hand.placement.closing,
        "hand_map_sha256": link.spheres_sha256, "guard_sha256": guard_sha256(parts),
    }, None


def planner_evidence_refusal(
    identity: Any,
    hand: Any,
    link: Any,
    *,
    arm: str,
    planner_margin_mm: float,
    guard_margin_mm: float,
    attach_spheres: int,
    mesh_dir: "str | None" = None,
    evidence_dir: "Path | None" = None,
) -> "str | None":
    """Why a started sidecar may not be kept for this cell, or ``None``. Both drivers call this, and only this.

    Takes what a driver holds (the sidecar's identity, the resolved hand, the numbers its guard keeps) rather than a
    whole config: the sim driver has no ``RobotConfig``, and a second derivation per driver is how two drivers end up
    admitting different robots. ``link`` is the hand link the sidecar was given; its sphere hash has to be the hand
    map the file recorded, or the file is speaking about another hand.
    """
    geometry, refused = _geometry(hand, arm=arm, mesh_dir=mesh_dir)
    if geometry is None:
        return refused
    given = getattr(link, "spheres_sha256", None)
    if given is not None and given != geometry["hand_map_sha256"]:
        return (f"the hand link this sidecar was given ({given[:12]}...) is not the hand this cell resolves "
                f"({geometry['hand_map_sha256'][:12]}...), so no file can speak for the pair.")
    return evidence_refusal(
        arm=arm, hand=geometry["hand"], coupling_mm=geometry["coupling_mm"], approach=geometry["approach"],
        closing=geometry["closing"], planner_margin_mm=float(planner_margin_mm), attach_spheres=int(attach_spheres),
        guard_margin_mm=float(guard_margin_mm),
        composed_sha256=identity.composed_sha256, urdf_sha256=identity.urdf_sha256,
        arm_descriptor_sha256=identity.arm_descriptor_sha256,
        hand_map_sha256=geometry["hand_map_sha256"], guard_sha256=geometry["guard_sha256"],
        evidence_dir=evidence_dir, plates=tuple(getattr(hand, "plates", ())),
    )


def desk_evidence_refusal(
    robot_cfg: Any, *, evidence_dir: "Path | None" = None, data_dir: "str | Path | None" = None,
) -> "tuple[str | None, CombinationEvidence | None]":
    """The file half of :func:`planner_evidence_refusal`, for a cell that has not started anything yet.

    A desk has no sidecar, so the three hashes a running sidecar reports about what it loaded cannot be checked here
    and are not pretended to be. What can be checked is everything the cell computes from its own config: which
    file, which rung, which hand map, which guard geometry, which guard margin. Returns the refusal and, where there
    is none, the file that admits the cell, so a reader can say what admitted it.

    ``data_dir`` is the cell's config tree, handed to :func:`~.hand.planner_hand`, so a tree that describes its hand
    differently from the repository is refused here too rather than admitted by the repository's evidence.
    """
    from src.config.loader import ConfigError

    from .hand import planner_hand
    from .margin import declared_planner_margin
    from .reservation import PlannerReservation

    self_collision = robot_cfg.safety.self_collision
    arm = str(getattr(self_collision, "kinematics_model", None) or robot_cfg.ur.model)
    declared = declared_planner_margin(self_collision)
    if not chosen(declared):
        return ("safety.self_collision.planner_margin_mm is undeclared, so there is no combination to look up: the "
                "margin is part of what a file measured"), None
    try:
        hand = planner_hand(robot_cfg, data_dir=data_dir)
    except ConfigError as exc:
        return str(exc), None
    if not chosen(hand):
        return "robot.gripper.model is unset, so there is no hand to look up a combination for", None
    geometry, refused = _geometry(hand, arm=arm, mesh_dir=getattr(self_collision, "mesh_dir", None))
    if geometry is None:
        return refused, None
    attach = PlannerReservation.from_config(robot_cfg=robot_cfg).sphere_slots
    command = gate_command(arm=arm, hand=geometry["hand"], coupling_mm=geometry["coupling_mm"],
                           approach=geometry["approach"], closing=geometry["closing"], planner_margin_mm=float(declared),
                           guard_margin_mm=float(self_collision.min_distance_mm), attach_spheres=int(attach),
                           plates=tuple(hand.plates))
    evidence, refused = _read(arm=arm, hand=geometry["hand"], coupling_mm=geometry["coupling_mm"],
                              approach=geometry["approach"], closing=geometry["closing"],
                              planner_margin_mm=float(declared), attach_spheres=int(attach),
                              evidence_dir=evidence_dir, command=command)
    if evidence is None:
        return refused, None
    said = _file_refusal(evidence, hand_map_sha256=geometry["hand_map_sha256"],
                         guard_sha256=geometry["guard_sha256"],
                         guard_margin_mm=float(self_collision.min_distance_mm), command=command)
    return (said, None) if said is not None else (None, evidence)
