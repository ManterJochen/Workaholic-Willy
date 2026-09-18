"""The hand as its own link of the planner's robot, derived from the resolved hand and from where it sits.

The hand stays its committed map, in the hand model's own frame M (x closing, y approach out of the mounting face), and
the placement the cell's declared tool frame gives (:class:`~._hand_placement.HandPlacement`) becomes the fixed
transform of a link named ``hand`` under ``tool0``:

    p_tool0 = R_TM (p_model + plate along model y) = R_TM p_model + plate * (R_TM e_y),

so the rotation is R_TM and the translation is the plate along the placed approach. That is where the exact mesh guard
puts the hand's vertices (``_fcl_self_collision.place_hand_vertices``), and like the guard a flange map takes no plate.
The arithmetic is ``_curobo_body_links.hand_body_link``, shared with the scripts in the cuRobo environment.

The placement is required. A hand whose declared frame places it nowhere has no link, and the refusal says why: UNSET
read as the identity would model a real UR hand a quarter turn from where it is, so it is refused and never implied.

The hand ignores ``tool0``, ``wrist_3_link`` and ``wrist_2_link``: the frame a descriptor holds an ignore table for, and
the pairs the guard skips. ``wrist_1_link`` stays checked. Its buffer is 0, and the margin raises it like every link's.

A wrist camera is :class:`WristBody`: its registry housing and the rig's bracket, grown by the rig's margin and placed
on tool0 by the recorded flange to TCP times the calibrated CAMERA to TOOL. One noun gives the planner its link, the
exact mesh guard its parts and the self filter its spheres, so the three cannot hold three different cameras.
:func:`wrist_body_refusal` is the planner start's check that the sidecar loaded exactly those spheres and that they
hold every point of every box, which is what the combination evidence, keyed without the camera, relies on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from src.contracts import chosen

from ._curobo_body_links import (
    HAND_IGNORE,
    HAND_JOINT,
    HAND_LINK,
    HAND_PARENT,
    WRIST_BODY_PREFIX,
    WRIST_BODY_REACH_MM,
    BodyLinkError,
    canonical_sha256,
    hand_body_link,
    wrist_body_link,
)
from ._declared_body import Box, box_spheres, boxes_to_parts, cover_refusal, inflated
from ._hand_placement import HandPlacement
from .hand import PlannerHand

__all__ = ["HAND_IGNORE", "HAND_JOINT", "HAND_PARENT", "HandLink", "WristBody", "coupling_bodies",
           "wrist_body_refusal"]

#: DH frame 6, where tool0 sits and where the exact mesh guard hangs every part on the flange.
_TOOL0_FRAME = 6
#: How far a reported fixed transform may differ from the one sent before the sidecar holds another camera.
_TRANSFORM_TOLERANCE = 1e-12


def coupling_bodies(hand: PlannerHand) -> "list[dict[str, Any]]":
    """The plates a cell measured across, as bodies to send beside the hand, or an empty list.

    Empty is the answer for every shipped cell, and it is not a silence: a plate that declared only a
    thickness still holds the hand out, and ``PlannerHand.coupling_undeclared`` names it so the checklist
    can say which one. What is refused here is inventing a width for it.

    Both drivers call this rather than each composing the list, so a cell's planner and its guard cannot
    end up holding different plates.
    """
    from ._curobo_body_links import coupling_body_link

    if not chosen(hand.placement):
        raise BodyLinkError(
            f"{hand.model} cannot carry plates on a placement its declared tool frame refuses: "
            f"{hand.placement_refusal}"
        )
    body = coupling_body_link(boxes=hand.coupling_boxes, rotation=hand.placement.rotation)
    return [] if body is None else [body]


@dataclass(frozen=True)
class HandLink:
    """One hand as a fixed link under tool0: its map's spheres in the hand's own frame, and where that frame sits."""

    #: The registry name, ``robot.gripper.model``.
    model: str
    placement: HandPlacement
    #: Where the map's numbers start: ``flange`` or ``mounting_face``.
    origin: str
    #: The plates between the flange and the hand, summed; only a mounting face map is moved by them.
    coupling_mm: float
    #: The map's spheres, verbatim: (centre, radius) in metres, in the hand model's frame.
    spheres: tuple[tuple[tuple[float, ...], float], ...]
    #: cuRobo's ``fixed_transform``: ``(x, y, z, qw, qx, qy, qz)``, metres, in tool0.
    fixed_transform: tuple[float, ...]
    #: :func:`~._curobo_body_links.canonical_sha256` of the spheres as they are sent.
    spheres_sha256: str

    @classmethod
    def from_hand(cls, hand: PlannerHand) -> HandLink:
        """The link ``hand`` is, or :class:`BodyLinkError` where its placement or its map gives none."""
        if not chosen(hand.placement):
            raise BodyLinkError(
                f"{hand.model} cannot be a link of the planner's robot, because its declared tool frame places it "
                f"nowhere: {hand.placement_refusal}"
            )
        document = yaml.safe_load(hand.sphere_map.read_text(encoding="utf-8")) or {}
        entries = (document.get("collision_spheres") or {}).get("tool0") or []
        if not entries:
            raise BodyLinkError(
                f"{hand.sphere_map.name} holds no spheres under collision_spheres.tool0, so {hand.model} has no body to "
                "plan with"
            )
        placement = hand.placement
        body = hand_body_link(
            spheres=entries, rotation=placement.rotation, approach_in_tool0=placement.approach_in_tool0,
            origin=hand.origin, coupling_mm=hand.coupling_mm,
        )
        return cls(
            model=hand.model,
            placement=placement,
            origin=hand.origin,
            coupling_mm=float(hand.coupling_mm),
            spheres=tuple((tuple(sphere["center"]), sphere["radius"]) for sphere in body["spheres"]),
            fixed_transform=tuple(body["fixed_transform"]),
            spheres_sha256=canonical_sha256(body["spheres"]),
        )

    def render(self) -> str:
        x, y, z = (1000.0 * v for v in self.fixed_transform[:3])
        return (
            f"hand link {self.model}: {len(self.spheres)} spheres under {HAND_PARENT} at ({x:g}, {y:g}, {z:g}) mm, "
            f"approach {self.placement.approach}, closing {self.placement.closing}, ignoring {', '.join(HAND_IGNORE)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """The body in the shape ``apply_body_links`` reads, and what it was derived from."""
        return {
            "link": HAND_LINK,
            "parent": HAND_PARENT,
            "joint": HAND_JOINT,
            "fixed_transform": list(self.fixed_transform),
            "spheres": [{"center": list(centre), "radius": radius} for centre, radius in self.spheres],
            "slots": 0,
            "buffer_m": 0.0,
            "ignore": list(HAND_IGNORE),
            "model": self.model,
            "origin": self.origin,
            "coupling_mm": self.coupling_mm,
            "spheres_sha256": self.spheres_sha256,
            "placement": self.placement.to_dict(),
        }


def _rigid(matrix: np.ndarray, what: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise BodyLinkError(f"{what} is not a finite 4x4 transform")
    rotation = array[:3, :3]
    if (not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)):
        raise BodyLinkError(f"{what} is not a rigid transform")
    return array


@dataclass(frozen=True)
class WristBody:
    """One wrist camera as a body on tool0: its grown boxes in the colour optical frame, and where that frame sits.

    Built from the camera registry's housing, the rig's bracket and margin, the rig's calibrated CAMERA to TOOL and the
    flange to TCP that calibration recorded, so the body and the pick frame are placed from the same numbers and go
    stale together.
    """

    rig_id: str
    #: The camera registry name, ``camera.cameras.rigs[<id>].body.model``.
    model: str
    margin_mm: float
    #: The housing and, where the rig declares one, the bracket, each already grown by ``margin_mm``, all named after
    #: the link, in the colour optical frame.
    boxes: tuple[Box, ...]
    #: The optical frame on tool0 as a 4x4 in millimetres: recorded flange to TCP times CAMERA to TOOL.
    placement_mm: tuple[tuple[float, ...], ...]
    #: Where the record came from: ``willy`` or ``polyscope``.
    record_source: str
    #: The calibration artifact the placement was read from.
    artifact_path: str
    #: The flange to TCP the calibration recorded, 4x4 in millimetres.
    record_mm: tuple[tuple[float, ...], ...] = ()
    #: How far the cell's frame may lie from the record before this body is refused; ``None`` where not given.
    record_tolerance_mm: "float | None" = None
    record_tolerance_deg: "float | None" = None

    @classmethod
    def from_parts(
        cls,
        *,
        rig_id: str,
        spec: Any,
        bracket: Any,
        margin_mm: float,
        camera_to_tool: Any,
        flange_to_tcp: Any,
        artifact_path: str = "",
        record_tolerance_mm: "float | None" = None,
        record_tolerance_deg: "float | None" = None,
    ) -> "WristBody":
        """The body ``spec`` (a ``CameraBodySpec``) and ``bracket`` (an ``OpticalBox`` or ``None``) make on this flange.

        ``camera_to_tool`` is the calibration's ``Transform(CAMERA -> TOOL)``, ``flange_to_tcp`` its ``FlangeToTcp``
        record. Refuses a margin that is not above 0 and a placement that is not a rigid transform.
        """
        from src.geometry import Frame

        link = f"{WRIST_BODY_PREFIX}{rig_id}"
        if camera_to_tool.from_frame is not Frame.CAMERA or camera_to_tool.to_frame is not Frame.TOOL:
            raise BodyLinkError(
                f"the calibration of rig {rig_id!r} is {camera_to_tool.from_frame.value} to "
                f"{camera_to_tool.to_frame.value}, and a wrist camera's body is placed by CAMERA to TOOL"
            )
        margin = float(margin_mm)
        if not margin > 0.0:
            raise BodyLinkError(f"the body of rig {rig_id!r} has a margin of {margin:g} mm, and it must be above 0")
        placement = _rigid(np.asarray(flange_to_tcp.matrix(), dtype=np.float64)
                           @ np.asarray(camera_to_tool.to_matrix(), dtype=np.float64),
                           f"the placement of rig {rig_id!r} (flange to TCP times CAMERA to TOOL)")
        declared = [spec.housing] + ([] if bracket is None else [bracket])
        boxes = tuple(
            inflated(Box(name=link, centre_mm=tuple(float(v) for v in box.centre_mm),  # type: ignore[arg-type]
                         half_extents_mm=tuple(float(v) / 2.0 for v in box.size_mm)), by_mm=margin)  # type: ignore[arg-type]
            for box in declared
        )
        return cls(rig_id=str(rig_id), model=str(spec.model), margin_mm=margin, boxes=boxes,
                   placement_mm=tuple(tuple(float(v) for v in row) for row in placement),
                   record_source=str(flange_to_tcp.source), artifact_path=str(artifact_path),
                   record_mm=tuple(tuple(float(v) for v in row) for row in np.asarray(flange_to_tcp.matrix())),
                   record_tolerance_mm=None if record_tolerance_mm is None else float(record_tolerance_mm),
                   record_tolerance_deg=None if record_tolerance_deg is None else float(record_tolerance_deg))

    def record_refusal(self, source: str, frame: "np.ndarray | None") -> "str | None":
        """Why the flange to TCP the cell applies now, ``frame`` on a ``source`` cell, is not this body's record, or None.

        The body and the pick frame were both placed from the record, so a frame that moved since makes both stale. On
        a ``willy`` cell the record was written from the declared numbers and has to be them; on a ``polyscope`` cell
        the frame is derived at connect and may lie within the rig's record tolerances.
        """
        from src.calibration.rig_calibration import flange_to_tcp_record_refusal

        return flange_to_tcp_record_refusal(
            self.rig_id, record_source=self.record_source, record_mm=self.record_mm, source=source, frame=frame,
            record_tolerance_mm=self.record_tolerance_mm, record_tolerance_deg=self.record_tolerance_deg,
        )

    @property
    def link_name(self) -> str:
        return f"{WRIST_BODY_PREFIX}{self.rig_id}"

    def placement(self) -> np.ndarray:
        """The optical frame on tool0, 4x4, millimetres."""
        return np.asarray(self.placement_mm, dtype=np.float64)

    def spheres_by_box(self) -> "list[list[dict[str, Any]]]":
        """Each box's sphere fill, in metres, in the optical frame, as the planner link carries them."""
        return [box_spheres(box, reach_mm=WRIST_BODY_REACH_MM) for box in self.boxes]

    def link(self) -> "dict[str, Any]":
        """The body in the shape ``apply_body_links`` reads, and what it was derived from."""
        placement = self.placement()
        body = wrist_body_link(rig_id=self.rig_id, boxes=self.boxes, rotation=placement[:3, :3].tolist(),
                               translation_m=(placement[:3, 3] / 1000.0).tolist())
        return {**body, "rig_id": self.rig_id, "model": self.model, "margin_mm": self.margin_mm,
                "boxes": [box.to_dict() for box in self.boxes]}

    def guard_parts(self) -> "dict[str, np.ndarray]":
        """The boxes as one exact mesh guard part on DH frame 6, vertices placed on tool0."""
        parts = boxes_to_parts(list(self.boxes), frame=_TOOL0_FRAME)
        placement = self.placement()
        key = f"{self.link_name}__v"
        vertices = parts[key]
        parts[key] = vertices @ placement[:3, :3].T + placement[:3, 3]
        return parts

    def envelope_spheres_mm(self) -> "list[tuple[tuple[float, float, float], float]]":
        """Every sphere of the fill on tool0, in millimetres, for the self filter."""
        placement = self.placement()
        out: list[tuple[tuple[float, float, float], float]] = []
        for spheres in self.spheres_by_box():
            for sphere in spheres:
                centre = placement[:3, :3] @ (np.asarray(sphere["center"], dtype=np.float64) * 1000.0) + placement[:3, 3]
                out.append(((float(centre[0]), float(centre[1]), float(centre[2])), float(sphere["radius"]) * 1000.0))
        return out

    def cover_refusal(self) -> "str | None":
        """Why this body's own fill leaves a hole in one of its boxes, or ``None``."""
        for box, spheres in zip(self.boxes, self.spheres_by_box()):
            refusal = cover_refusal(box, spheres)
            if refusal is not None:
                return refusal
        return None

    def render(self) -> str:
        sizes = "; ".join(" x ".join(f"{2.0 * v:g}" for v in box.half_extents_mm) for box in self.boxes)
        count = sum(len(spheres) for spheres in self.spheres_by_box())
        return (f"wrist camera {self.link_name}: {self.model}, boxes grown by {self.margin_mm:g} mm to {sizes} mm, "
                f"{count} spheres at {WRIST_BODY_REACH_MM:g} mm reach, placed from {self.artifact_path or 'a calibration'}"
                f" (flange to TCP recorded on {self.record_source})")

    def to_dict(self) -> "dict[str, Any]":
        return {"rig_id": self.rig_id, "link": self.link_name, "model": self.model, "margin_mm": self.margin_mm,
                "boxes": [box.to_dict() for box in self.boxes], "placement_mm": [list(row) for row in self.placement_mm],
                "record_source": self.record_source, "artifact_path": self.artifact_path}


def wrist_body_refusal(identity: Any, bodies: "Sequence[WristBody]") -> "str | None":
    """Why the started sidecar does not hold exactly ``bodies`` with a cover proven complete, or ``None``.

    ``identity`` is the sidecar's ``SidecarIdentity``. With no bodies, a sidecar that reports none passes and one that
    reports any is refused. With bodies, in this order: a sidecar that reports no wrist bodies, rows whose hash is not
    ``wrist_bodies_sha256``, another set of links, a link whose parent, fixed transform, spheres or ignore list is not
    the one sent, and a sent fill that leaves a hole in its box. The combination evidence is keyed without the camera,
    so this proof is what stands in for measuring it.
    """
    reported = getattr(identity, "wrist_bodies", None)
    rows = tuple(reported) if chosen(reported) and reported is not None else None
    if not bodies:
        if rows:
            return (f"the sidecar loaded the wrist camera(s) {[row.get('link') for row in rows]} and this cell carries "
                    "none, so the planner holds a camera that is not this cell's. Restart the planner.")
        return None
    if rows is None:
        return ("this cell carries the wrist camera(s) "
                f"{[body.link_name for body in bodies]} and the sidecar reported no wrist bodies, so nothing says the "
                "planner holds them: a sidecar older than wrist bodies plans as if the arm carried no camera. Update "
                "curobo_planner_server.py beside this client and restart the planner.")
    reported_sha = getattr(identity, "wrist_bodies_sha256", None)
    if not chosen(reported_sha) or reported_sha != canonical_sha256(list(rows)):
        return ("the sidecar's wrist body rows do not hash to the wrist_bodies_sha256 it reported, so what it says it "
                "loaded cannot be read. Restart the planner.")
    sent = {body.link_name: body for body in bodies}
    held = {str(row.get("link")): row for row in rows}
    if set(sent) != set(held):
        return (f"the sidecar loaded the wrist camera(s) {sorted(held)} and this cell sent {sorted(sent)}, so the "
                "planner holds a camera that is not this cell's. Restart the planner.")
    for name, body in sent.items():
        row, link = held[name], body.link()
        if row.get("parent") != link["parent"]:
            return (f"the sidecar loaded {name} with parent {row.get('parent')!r} and this cell sent "
                    f"{link['parent']!r}, so the planner holds a camera that is not this cell's. Restart the planner.")
        transform = [float(v) for v in row.get("fixed_transform") or ()]
        if len(transform) != 7 or any(abs(a - float(b)) > _TRANSFORM_TOLERANCE
                                      for a, b in zip(transform, link["fixed_transform"])):
            return (f"the sidecar loaded {name} with fixed_transform {transform} and this cell sent "
                    f"{link['fixed_transform']}, so the planner holds a camera that is not this cell's. Restart the "
                    "planner.")
        if row.get("spheres_sha256") != canonical_sha256(link["spheres"]):
            return (f"the sidecar loaded {name} with spheres_sha256 {str(row.get('spheres_sha256'))[:12]}... and this "
                    f"cell sent {canonical_sha256(link['spheres'])[:12]}..., so the planner holds a camera that is not "
                    "this cell's. Restart the planner.")
        if list(row.get("ignore") or ()) != list(link["ignore"]):
            return (f"the sidecar loaded {name} ignoring {row.get('ignore')} and this cell sent {link['ignore']}, so "
                    "the planner skips a pair this camera must be checked against. Restart the planner.")
        refusal = body.cover_refusal()
        if refusal is not None:
            return (f"wrist camera {name} ({body.model}): {refusal}. The fill is written by "
                    "_declared_body.box_spheres, which leaves none: this is a defect, not a cell setting.")
    return None
