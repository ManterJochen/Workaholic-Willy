"""SelfCollisionGuard, the link, tool and fixture self-collision guard.

There are two backends.

``capsule`` is the default and is always available. It approximates the arm as a chain
of capsules, one per link, plus a tool capsule rooted at the TCP and a base column
capsule. Fixtures are the axis-aligned boxes declared in :class:`FixtureBoxConfig`.
The distance is closed-form, so the per-move cost stays under a millisecond.

* On an arm whose reported model the arm registry describes as a UR (a real UR, and the
  Isaac arm, which reports the UR it drives) the arm-against-arm capsule chain comes from
  that description's DH rows (:mod:`_ur_kinematics`), and every non-adjacent link pair is
  checked against the others and against every fixture.
* On any other arm (the dummy) that chain is unavailable, so the guard checks the tool
  against the base and the tool against the fixtures only.

``fcl`` is exact mesh-against-mesh distance over the bundled per-model collision
meshes, with Coal preferred and python-fcl as the fallback, in
:mod:`_fcl_self_collision`. It removes two measured errors of the capsule proxy, the
skipped-wrist false negative and the fat-radius false positive, by checking every link
pair more than one DH frame apart exactly. Any UR the arm registry describes and with a
committed ``{model}_collision_meshes.npz`` qualifies: seven do today
(``safety/data/bundles.json``), and ur20 has link lengths and no bundle. Where the engine or the bundle is missing the guard falls back to the capsule
path and logs that fallback with a reason token, because the capsule proxy is
materially coarser and a cell that swaps one for the other in silence is a
safety-relevant regression.

Configuration
-------------
* ``min_distance_mm`` is the smallest allowed signed distance between any monitored
  shape pair, and anything below it is a collision.
* ``perceived_min_distance_mm`` is that distance for a box a camera saw, which the exact
  mesh guard judges as the planner holds it, turned; it is already grown by the camera
  world's margin (``SelfCollisionGuard.set_perceived_fixtures``).
* ``link_radii_mm`` is the per-link capsule radius. Omitted, the guard uses 60 mm, a
  typical UR link cross-section.
* ``fixtures`` is a list of :class:`FixtureBoxConfig`.
"""

from __future__ import annotations

from collections.abc import Sequence

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from src.contracts import UNSET, Maybe, chosen

from ._capsule import (
    AxisAlignedBox,
    Capsule,
    capsule_box_distance_mm,
    capsule_capsule_distance_mm,
)
from ._ur_kinematics import (
    arm_link_transforms_mm,
    arm_link_transforms_mm_many,
    described_ur_model,
    ur_link_origins_mm,
)
from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import SelfCollisionSafetyConfig

    from .planning.band import ExactPairs
    from .planning.hand import PlannerHand

__all__ = ["SelfCollisionGuard"]


_LOGGER = logging.getLogger(__name__)

# Default capsule radii and lengths, used where the operator overrides none. They
# match a typical UR link of 60 mm cross-section and a 70 mm gripper envelope.
_DEFAULT_LINK_RADIUS_MM = 60.0
_DEFAULT_TOOL_RADIUS_MM = 70.0
_DEFAULT_TOOL_LENGTH_MM = 150.0
_DEFAULT_BASE_RADIUS_MM = 80.0
_DEFAULT_BASE_HEIGHT_MM = 150.0


def _hand_identity(hand: "Maybe[PlannerHand]") -> "tuple[str, tuple[object, ...]] | None":
    """What makes two hands one hand for a capsule's tool dimensions: its name and the plates that hold it out."""
    return (hand.model, tuple(hand.plates)) if chosen(hand) else None


class SelfCollisionGuard:
    """Self-collision guard.

    Stateless across calls. :meth:`SafetyPreflight.from_safety_config` constructs it
    when ``safety.self_collision.enforce`` is ``True``.
    """

    name = "self_collision"

    def __init__(
        self, config: "SelfCollisionSafetyConfig", *, hand: "Maybe[PlannerHand]" = UNSET,
        built_for: "Maybe[str | None]" = UNSET,
    ) -> None:
        self._config = config
        #: The arm model the pipeline was built for, as ``SafetyPreflight.from_safety_config`` names it, ``None`` where
        #: it names an arm the registry does not describe as a UR, ``UNSET`` for a guard built directly (P0e E6). The
        #: hand refusal at build was decided for this arm, so the guard judges no other (:meth:`refusal_for`).
        self._built_for = built_for
        # The hand this guard models, resolved from robot.gripper.model: its variant picks
        # the mesh bundle and its coupling moves a mounting-face hand onto the flange.
        # UNSET where the guard is built directly, which keeps the arm bundle's own hand;
        # SafetyPreflight.from_safety_config refuses to build a guard that reads hand
        # geometry with no hand named.
        self._hand = hand
        # A refused placement is not an identity. A declared frame the derivation refuses leaves
        # the hand's placement unset, and a guard built on it would place the hand by the model's
        # own axes instead. The cell refuses at build; this refuses a guard built directly, which
        # the sim runners and the probes do.
        if chosen(hand) and not chosen(hand.placement):
            raise ValueError(
                hand.placement_refusal
                or f"{hand.model} has no placement on the flange, so no exact mesh guard can model where it is"
            )
        self._min_distance_mm = float(config.min_distance_mm)
        # A box a camera saw is its surface grown by the camera world's margin, so it is kept at its own,
        # smaller distance (SelfCollisionSafetyConfig.perceived_min_distance_mm). A config without the key,
        # a stand-in built by hand, keeps it at min_distance_mm, the stricter side.
        held = getattr(config, "perceived_min_distance_mm", None)
        self._perceived_min_distance_mm = self._min_distance_mm if held is None else float(held)
        # What the planner keeps clear, so it stops proposing configurations this guard
        # rejects. It is separate from _min_distance_mm on purpose; SelfCollisionConfig
        # says why.
        self._planner_margin_mm = float(getattr(config, 'planner_margin_mm', 0.0) or 0.0)
        self._backend = config.backend
        # The exact-mesh backend, built lazily and only for backend='fcl'. ``None``
        # means not yet built, or unavailable.
        self._fcl_backend: object | None = None
        self._fcl_backend_built = False
        #: The arm model the exact-mesh backend was built for, once it is. One model per guard (P0d D5).
        self._fcl_model: str | None = None
        #: A wrist camera's bodies (``body_link.WristBody``), handed in before the exact-mesh backend
        #: is built.
        self._wrist_bodies: tuple[object, ...] = ()
        #: Why the exact-mesh backend is available or not once built. The tokens are defined
        #: by ``_fcl_self_collision.mesh_backend_status``, and ``None`` means it is not built
        #: yet. It is surfaced so a degraded cell is inspectable rather than silent.
        #:
        #: Deliberately not re-listed here. This comment named four of the tokens and the
        #: function could return five, having grown two since the sentence was written. A list
        #: beside something that already knows the answer only stays true for a while.
        self._fcl_status: str | None = None
        # Tool and base capsule geometry, from config, defaulting to the constants above.
        self._tool_length_mm = float(getattr(config, "tool_length_mm", _DEFAULT_TOOL_LENGTH_MM))
        self._tool_radius_mm = float(getattr(config, "tool_radius_mm", _DEFAULT_TOOL_RADIUS_MM))
        # The tool model. "capsule", the default, is a bounding cylinder along the
        # approach axis. "finger" is a thin capsule along the grasp closing axis, which
        # is the faithful footprint of a descending 2F-85 finger.
        self._tool_model = str(getattr(config, "tool_model", "capsule"))
        self._tool_finger_radius_mm = float(getattr(config, "tool_finger_radius_mm", 16.0))
        self._tool_finger_span_mm = float(getattr(config, "tool_finger_span_mm", 150.0))
        self._base_radius_mm = float(getattr(config, "base_radius_mm", _DEFAULT_BASE_RADIUS_MM))
        self._base_height_mm = float(getattr(config, "base_height_mm", _DEFAULT_BASE_HEIGHT_MM))
        self._declared_fixtures = tuple(
            AxisAlignedBox(
                center_mm=np.asarray(fx.center_mm, dtype=np.float64),
                half_extents_mm=np.asarray(fx.half_extents_mm, dtype=np.float64),
                name=str(fx.name),
            )
            for fx in config.fixtures
        )
        #: Obstacles a camera saw, set before a motion and cleared when nobody can vouch for them.
        #: Separate from the declared list so a perceived box can never overwrite a measured one, and
        #: so clearing them cannot take the bench with it.
        self._perceived_fixtures: tuple[AxisAlignedBox, ...] = ()
        #: Whether a path is judged whole before it is judged sample by sample
        #: (``SelfCollisionSafetyConfig.whole_path_judge``): :meth:`exact_pairs` then answers the mesh-first questions of
        #: a whole path at once too. The path gate reads the same key (``SafetyPreflight.from_safety_config``).
        self._whole_path_judge = bool(getattr(config, "whole_path_judge", False))

    @property
    def _fixtures(self) -> "tuple[AxisAlignedBox, ...]":
        """Everything this guard must keep the arm away from: declared first, then perceived.

        Declared first so an operator reading a refusal meets the box they wrote down before the box
        a camera inferred, and so the numbering of the unnamed ones is stable while the perceived
        list changes underneath.
        """
        return self._declared_fixtures + self._perceived_fixtures

    def set_perceived_fixtures(self, boxes: "Sequence[AxisAlignedBox]") -> None:
        """Tell this guard about obstacles a camera saw, or clear them with an empty sequence.

        The planner and this guard have to be looking at the same cell. A planner routing around a
        tote the guard cannot see produces the worst of both: a path that avoids the tote and a gate
        that would have allowed one straight through it, so nothing in the stack is holding the line.

        Each box carries the turned box the planner takes (``AxisAlignedBox.turned``), and the exact
        mesh guard judges that one: the two authorities hold the same box, where the axis-aligned box
        around it was up to 1.41 times as wide per side and refused bins placed at a turn beside the
        base (2026-09-30). The capsule proxy cannot turn a box and judges the enclosure, the safe side.
        A seen box is its surface grown by the camera world's margin, so the exact guard keeps it at
        ``perceived_min_distance_mm`` rather than ``min_distance_mm``; the schema refuses a cell whose
        two numbers fall short of the step a path is sampled at.

        Never call this with obstacles nobody can vouch for. An empty sequence means the cell is back
        to its declared geometry, which is the honest state when the cameras cannot answer.
        """
        self._perceived_fixtures = tuple(boxes)

    def set_finger_opening(self, width_mm: float | None) -> None:
        """Hold the hand's fingers ``width_mm`` apart, between their inner faces, where the hand was told to open no
        wider (lever G, 2026-10-13); ``None`` holds them as baked, fully open, as before. Kept for the backend where it is
        not built yet. Only a caller that told the hand that width, and read it back, may say so: the guard then judges
        the fingers where they stand."""
        self._finger_opening_mm = None if width_mm is None else float(width_mm)
        if self._fcl_backend_built and self._fcl_backend is not None:
            setter = getattr(self._fcl_backend, "set_finger_opening", None)
            if callable(setter):
                setter(self._finger_opening_mm)

    @property
    def finger_opening_mm(self) -> float | None:
        """How wide this guard holds the fingers open (:meth:`set_finger_opening`); ``None`` fully open, as baked."""
        return getattr(self, "_finger_opening_mm", None)

    @property
    def perceived_fixtures(self) -> "tuple[AxisAlignedBox, ...]":
        """The boxes a camera saw that this guard holds now, as :meth:`set_perceived_fixtures` last handed them.

        The planner's band admission sets exactly these aside in the planner's world, by name, where only they refuse
        it, and has this guard judge the refused samples with them (``planning.band``, the owner's Option 1).
        """
        return self._perceived_fixtures

    # ------------------------------------------------------------------
    # Capsule construction
    # ------------------------------------------------------------------

    def _link_radius(self, axis_idx: int) -> float:
        radii = self._config.link_radii_mm
        if radii is None:
            return _DEFAULT_LINK_RADIUS_MM
        if axis_idx < len(radii):
            return float(radii[axis_idx])
        return _DEFAULT_LINK_RADIUS_MM

    def _base_capsule(self) -> Capsule:
        """The fixed base column, from the floor to the mounting flange."""
        return Capsule(
            p0=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            p1=np.array([0.0, 0.0, self._base_height_mm], dtype=np.float64),
            radius_mm=self._base_radius_mm,
        )

    def _tool_capsule(self, ctx: SafetyContext) -> Capsule | None:
        """The tool capsule, rooted at the commanded TCP pose.

        ``tool_model='capsule'``, the default, runs a capsule ``tool_length_mm`` from
        the TCP back toward the flange, which is a rotation-invariant bounding
        cylinder. ``tool_model='finger'`` instead lays a thin capsule along the closing
        axis of the grasp, which is the faithful parallel-jaw footprint.
        """
        pose = ctx.target_pose
        if pose is None:
            return None
        tcp_mm = np.asarray(pose.position_mm, dtype=np.float64)
        from src.geometry.quaternion import to_rotation_matrix
        R = to_rotation_matrix(pose.quaternion_xyzw)
        if self._tool_model == "finger":
            # The descending fingers of a 2F-85 as a thin capsule along the closing
            # axis, which is R[:, 0] under the [closing, binormal, approach]
            # convention. Measured at about 27 mm wide perpendicular to closing, so a
            # half-width near 13.5 against the r=70 bounding cylinder, and about
            # 150 mm fingertip to fingertip at full open. It is rotation-aware: it
            # clears the wall the thin side faces and rejects the open-span wall.
            closing = R[:, 0]
            half = 0.5 * self._tool_finger_span_mm
            return Capsule(
                p0=tcp_mm - closing * half,
                p1=tcp_mm + closing * half,
                radius_mm=self._tool_finger_radius_mm,
            )
        # The default is one capsule from the TCP back along the tool local -Z, from
        # the grasp centre toward the flange, because that is where the material of the
        # tool is. The direction is load-bearing: a tool protruding from the flange is
        # the right picture for a flange-rooted pose, but ``ctx.target_pose`` is the
        # commanded pose and every pose this stack commands is the grasp centre. Run
        # forward from there, the capsule would model 150 mm of solid tool on the far
        # side of the grasp point. On a top-down grasp at (300, 0, 37) it reaches
        # (300, 0, -113) at r=70, which is 113 mm of phantom material below the table
        # where the object and the table are, while the real 2F-85 body between z 37
        # and z 169 goes unmodelled: inventing collisions where the gripper is not and
        # missing them where it is.
        #
        # The small overhang of the fingertips past the grasp centre is not modelled.
        # Bounding it needs the flange-to-TCP offset from the tool-frame contract, so
        # the omission is stated rather than guessed at.
        tool_z = R[:, 2]
        flange_side_mm = tcp_mm - tool_z * self._tool_length_mm
        return Capsule(p0=tcp_mm, p1=flange_side_mm, radius_mm=self._tool_radius_mm)

    def _ur_arm_capsules(self, ctx: SafetyContext) -> list[Capsule] | None:
        """Per-link capsules for the arm in hand, keyed by the model it reports, or ``None`` where none derive.

        The model is :meth:`model_for`'s, so capsules, meshes and the path sampler's reach answer from one rule: a
        model the arm registry describes as a UR. The Isaac arm reports the UR it drives and gets real arm-against-arm
        capsules; until P0d D5 it answered ``isaac-sim`` and needed ``self_collision.kinematics_model`` to opt in.
        """
        if ctx.target_joints is None or ctx.arm is None:
            return None
        model = self.model_for(ctx.arm)
        if model is None:
            return None
        origins_mm = ur_link_origins_mm(
            model, ctx.target_joints.values,
        )
        if origins_mm is None or len(origins_mm) < 2:
            return None
        # Reconcile the bundled DH base frame to the system base frame, where the
        # base, tool and fixture capsules live, by rotating the arm-link origins about
        # +Z. The Isaac UR5e USD base_link is measured 180 deg off the official UR DH
        # base, and the default of 0.0 leaves an existing cell unchanged.
        yaw_deg = float(self._config.kinematics_base_yaw_deg)
        if yaw_deg != 0.0:
            origins_mm = self._rotate_origins_z(origins_mm, yaw_deg)
        capsules: list[Capsule] = []
        for i in range(len(origins_mm) - 1):
            capsules.append(
                Capsule(
                    p0=origins_mm[i],
                    p1=origins_mm[i + 1],
                    radius_mm=self._link_radius(i),
                )
            )
        return capsules

    @staticmethod
    def _rotate_origins_z(origins_mm: list[np.ndarray], yaw_deg: float) -> list[np.ndarray]:
        """Rotate base-frame origins about +Z by ``yaw_deg`` to reconcile the base frame.

        It is a pure rotation of the base frame, so it preserves every inter-link
        distance, leaving arm-against-arm checks unchanged, and only relocates the
        chain relative to the base, tool and fixture capsules.
        """
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        return [
            np.array([c * o[0] - s * o[1], s * o[0] + c * o[1], o[2]], dtype=np.float64)
            for o in origins_mm
        ]

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        refused = self.refusal_for(ctx.arm)
        if refused is not None:
            return SafetyDecision.reject(self.name, SafetyReason.UNAVAILABLE, message=refused)
        if self._backend == "fcl":
            # Exact mesh-against-mesh self-collision over the real collision meshes. It
            # returns a decision when it runs, and ``None`` when the engine or the mesh
            # bundle is absent or the model does not derive, which falls through to the
            # capsule path.
            dec = self._evaluate_fcl(ctx)
            if dec is not None:
                return dec

        base = self._base_capsule()
        tool = self._tool_capsule(ctx)
        arm_links = self._ur_arm_capsules(ctx)

        # Build the full capsule registry for the pairwise checks, keeping the names so
        # a rejection report can say which pair collided.
        labelled: list[tuple[str, Capsule]] = [("base", base)]
        if arm_links is not None:
            for i, cap in enumerate(arm_links):
                labelled.append((f"link_{i}", cap))
        # The link the tool is bolted to: the last one built, link_5 on a UR (P0f F9d).
        last_link = f"link_{len(arm_links) - 1}" if arm_links else None
        if tool is not None:
            labelled.append(("tool", tool))

        # ---- capsule vs capsule (skip adjacent arm links) -----------
        for i in range(len(labelled)):
            name_i, cap_i = labelled[i]
            for j in range(i + 1, len(labelled)):
                name_j, cap_j = labelled[j]
                if self._should_skip_pair(name_i, name_j, last_link=last_link):
                    continue
                d = capsule_capsule_distance_mm(cap_i, cap_j)
                if d < self._min_distance_mm:
                    return SafetyDecision.reject(
                        self.name,
                        SafetyReason.SELF_COLLISION,
                        message=(
                            f"{name_i} vs {name_j}: signed distance "
                            f"{d:.3f} mm < {self._min_distance_mm:.3f} mm"
                        ),
                        detail={
                            "pair": f"{name_i}|{name_j}",
                            "signed_distance_mm": f"{d:.6f}",
                            "min_distance_mm": (
                                f"{self._min_distance_mm:.6f}"
                            ),
                        },
                    )

        # ---- capsule vs fixture --------------------------------------
        for k, fixture in enumerate(self._fixtures):
            fname = fixture.name or f"fixture_{k}"
            for name_i, cap_i in labelled:
                d = capsule_box_distance_mm(cap_i, fixture)
                if d < self._min_distance_mm:
                    return SafetyDecision.reject(
                        self.name,
                        SafetyReason.SELF_COLLISION,
                        message=(
                            f"{name_i} vs fixture {fname!r}: signed "
                            f"distance {d:.3f} mm < "
                            f"{self._min_distance_mm:.3f} mm"
                        ),
                        detail={
                            "pair": f"{name_i}|fixture:{fname}",
                            "signed_distance_mm": f"{d:.6f}",
                            "min_distance_mm": (
                                f"{self._min_distance_mm:.6f}"
                            ),
                        },
                    )

        return SafetyDecision.accept(self.name)

    @property
    def min_distance_mm(self) -> float:
        """The margin this guard keeps between any two bodies, in mm.

        Public because it is also the step a sampled path has to keep: a path sampled more
        coarsely than the margin can pass through a body between two samples and be
        accepted by both.
        """
        return float(self._min_distance_mm)

    @property
    def mesh_dir(self) -> str | None:
        """The folder this guard reads its bundles from, or ``None`` for the committed one.

        Public because the combination evidence hashes the geometry the guard composes from it, and
        a cell pointed at another folder is judging another robot.
        """
        held = getattr(self._config, "mesh_dir", None)
        return None if held is None else str(held)

    @property
    def hand(self) -> "Maybe[PlannerHand]":
        """The hand this guard models, or ``UNSET`` where it keeps the arm bundle's own."""
        return self._hand

    def mount_refusal(self, hand: "Maybe[PlannerHand]") -> str | None:
        """Why this guard cannot judge with ``hand`` mounted, or ``None``: the build's own refusals (a hand not named, a
        frame that places none), and a capsule guard handed another hand than the one it was built with (P0i i5).

        ⛔ THE CAPSULE MODELS THE TREE'S TOOL (found by review, 2026-09-26). Its tool is a capsule of
        ``safety.self_collision``'s ``tool_length_mm``, ``tool_radius_mm`` and finger dimensions, set for the hand the
        tree names and read from nothing else, so another hand would be judged with the first one's dimensions. A
        guard on ``fcl`` reads the hand's own geometry and takes it; where its engine is missing it falls back to the
        same capsule, a cell the doctor already reports as degraded.
        """
        from src.robot.safety.planning.hand import guard_hand_refusal, hand_geometry_model

        built_for = self._built_for if chosen(self._built_for) else None
        refusal = guard_hand_refusal(self._config, hand, built_for)
        if refusal is not None:
            return refusal
        if chosen(hand) and not chosen(hand.placement):
            return (hand.placement_refusal
                    or f"{hand.model} has no placement on the flange, so no exact mesh guard can model where it is")
        if hand_geometry_model(self._config, built_for if built_for else UNSET) is None:
            held, wanted = _hand_identity(self._hand), _hand_identity(hand)
            if held != wanted:
                return (
                    f"this cell's self collision guard models the tool as a capsule from safety.self_collision's tool "
                    f"dimensions, set for {held[0] if held else 'no hand'} and changed by no mount, so it would judge "
                    f"{wanted[0] if wanted else 'no hand'} with them: build the cell with the tool it carries"
                )
        return None

    def mount(self, hand: "Maybe[PlannerHand]") -> None:
        """Model ``hand`` from now on; the exact mesh backend is built again for it at the next judged path. What
        ``PlannedArm.mount_tool`` reaches, the one path a hand takes after the build (P0i i5).

        Refused with :meth:`mount_refusal`'s reason (``ToolChangeRefused``), with nothing changed. The wrist bodies this
        guard holds stay: they hang on the arm, not the hand.
        """
        refusal = self.mount_refusal(hand)
        if refusal is not None:
            from src.robot.safety.planning.mounted_tool import ToolChangeRefused

            raise ToolChangeRefused(refusal)
        self._hand = hand
        # The opening the old hand was told is not this hand's (lever G): its fingers stand as baked, fully open,
        # until it is told a width of its own.
        self._finger_opening_mm = None
        self._fcl_backend = None
        self._fcl_backend_built = False
        self._fcl_model = None
        self._fcl_status = None

    @property
    def base_yaw_deg(self) -> float:
        """The yaw, in degrees, this guard turns the bundled DH base by to place the arm."""
        return float(self._config.kinematics_base_yaw_deg)

    def model_for(self, arm: "object | None") -> str | None:
        """The robot model this guard judges ``arm`` as: the model the arm reports, where the arm registry describes a
        UR by that name (``_ur_kinematics.described_ur_model``), else ``None``.

        The arm is the only source since P0d D5. ``safety.self_collision.kinematics_model`` used to stand in for it,
        because the Isaac arm answered ``isaac-sim``; it now reports the UR it drives (``robot.sim.robot_model``), and
        a key that could only ever repeat that name goes. The vendor does not enter the rule: an arm that reports a
        model nobody describes, the dummy's ``dummy-6dof`` or a KUKA's, gets no UR geometry whatever its vendor says.

        Public because more than the meshes depend on it. A sampled path is bounded by how far a joint can swing a
        point of this same arm, so the two answers have to be the same robot. An arm that could not name itself cost
        aurora's M1 gate on 2026-09-12: the Isaac arm answered `isaac-sim`, no reach derived, and every motion in the
        sim was refused as unsampleable. ``tests/test_joint_path_gate.py`` keeps that case.
        """
        if self.refusal_for(arm) is not None:
            return None
        return self._reported_model(arm)

    @staticmethod
    def _reported_model(arm: "object | None") -> str | None:
        capabilities = getattr(arm, "capabilities", None)
        return described_ur_model(getattr(capabilities, "model", None))

    def refusal_for(self, arm: "object | None") -> str | None:
        """Why this guard will not judge ``arm``, or ``None`` where it will (P0e E6).

        The pipeline decided at build, for the arm it was built for, whether a hand is named and where it sits, and
        which hand geometry the exact-mesh guard reads. An arm that reports another UR than that one would be judged
        by a guard whose hand refusal never asked about it. Until P0e a pipeline built with no arm model was that case
        for every arm reporting a UR: no "robot.gripper.model is unset", and the arm bundle's own hand judged instead.
        An arm that reports no UR is not refused here; it gets no UR geometry at all, as before.
        """
        if not chosen(self._built_for):
            return None
        reported = self._reported_model(arm)
        expected = described_ur_model(self._built_for) if self._built_for else None
        if reported is None or reported == expected:
            return None
        built = expected if expected is not None else "no arm the arm registry describes as a UR"
        return (f"the self-collision guard was built for {built} and the arm reports {reported}: its hand refusal was "
                f"decided for the arm it was built for, so it judges no other. Build the safety preflight with the "
                f"model this arm reports (arm_model=arm.capabilities.model), from the tree the arm was built from.")

    def set_wrist_bodies(self, bodies: "Sequence[object]") -> None:
        """Hold a wrist camera's bodies (``body_link.WristBody``) in the exact-mesh backend.

        Only before the backend is built. It is built once, on the first judged path or joint
        evaluation, and a body handed in after that would be a camera the guard says it holds and
        does not. An empty sequence clears them.
        """
        if self._fcl_backend_built:
            raise RuntimeError(
                "the exact mesh guard was already built, so a wrist camera handed in now would not be in it: hand the "
                "cell's wrist bodies to the arm before its first judged path"
            )
        self._wrist_bodies = tuple(bodies)

    @property
    def wrist_bodies(self) -> "tuple[object, ...]":
        """The wrist camera bodies this guard holds or will hold, empty for an arm that carries no camera."""
        return self._wrist_bodies

    def _exact_mesh_backend(self, model: str) -> object | None:
        """Build the exact-mesh backend once for ``model`` and cache it, ``None`` included.

        One model per guard. The model comes from the arm each call is asked about (P0d D5), and the cache holds one
        arm's meshes: a guard asked about a second model would place the first arm's links on the second arm's DH
        chain. No driver shares a preflight between arms, so a second model is refused rather than rebuilt.
        """
        if self._fcl_backend_built and self._fcl_model is not None and model != self._fcl_model:
            raise RuntimeError(
                f"this self collision guard holds the exact meshes of the {self._fcl_model} arm and was asked about a "
                f"{model}: build one safety preflight per arm"
            )
        if not self._fcl_backend_built:  # build the BVH models once, caching None too
            from ._fcl_self_collision import make_backend, mesh_backend_status
            hand = self._hand
            variant = hand.guard_variant if chosen(hand) else None
            coupling_mm = hand.coupling_mm if chosen(hand) else 0.0
            # Where the hand sits on the flange, as the hand resolved it. A refused placement never
            # reaches a guard, because the cell refuses to build first, and ``None`` is only a guard
            # built without a hand.
            placement = hand.placement if chosen(hand) and chosen(hand.placement) else None
            # The plates the cell measured across, as bodies. They sit between the flange and the
            # hand's mounting face, nearer the wrist than the hand does.
            boxes = hand.coupling_boxes if chosen(hand) else ()
            # A wrist camera's parts, passed only when the arm carries one, so a cell without a
            # camera calls the builder with the same arguments. A camera is bolted to the flange,
            # which is frame dof of this arm (P0f F9).
            wrist: dict = {}
            if self._wrist_bodies:
                from src.config.arms import load_arm

                flange = load_arm(model.lower()).dof
                for body in self._wrist_bodies:
                    wrist.update(body.guard_parts(frame=flange))  # type: ignore[attr-defined]
            carried: dict = {"wrist_parts": wrist} if wrist else {}
            # How wide the baked fingers stand open, the hand's own stroke, for a told finger opening (lever G).
            stroke = float(hand.jaw.aperture_mm) if chosen(hand) else None
            self._fcl_backend = make_backend(
                model, self._config.mesh_dir, variant, coupling_mm=coupling_mm, placement=placement,
                coupling_boxes=boxes, baked_opening_mm=stroke, **carried,
            )
            self._fcl_backend_built = True
            self._fcl_model = model
            told = getattr(self, "_finger_opening_mm", None)
            if told is not None and self._fcl_backend is not None:
                self._fcl_backend.set_finger_opening(told)  # type: ignore[attr-defined]
            # The config asked for the exact-mesh backend. Falling back to the coarser
            # capsule proxy in silence would look like a robot that cannot grasp
            # anything rather than a missing safety authority, because the proxy
            # over-rejects at _DEFAULT_LINK_RADIUS_MM. So this says it once and records
            # the reason, which lets telemetry and the boot banner show a degraded
            # cell.
            self._fcl_status = mesh_backend_status(model, self._config.mesh_dir, variant)
            # An envelope built from a registry block is guarded like any scanned hand, and a reader
            # must not have to guess which of the two it holds: the two hands that could be measured
            # needed 5.73 mm and 9.50 mm of inflation before an envelope enclosed them, so a cell run
            # this way has that much less room and no way to see it in the geometry itself.
            if chosen(hand) and hand.models_an_envelope and hand.provenance is not None:
                _LOGGER.info("SelfCollisionGuard: %s", hand.provenance.render())
            if self._fcl_backend is None:
                _LOGGER.warning(
                    "SelfCollisionGuard: backend='fcl' requested for model %r but the exact-mesh backend is "
                    "unavailable (%s); running the capsule fallback for this cell.", model, self._fcl_status,
                )
        return self._fcl_backend

    def exact_mesh_engine(self, arm: "object | None" = None) -> str | None:
        """Which exact-mesh engine this guard would run for ``arm``: ``"coal"``, ``"fcl"`` or ``None``.

        ``None`` means every verdict this guard gives is the capsule proxy: either the
        config asked for the proxy, or no robot model derives, or the engine and the baked
        meshes are not on this box. The proxy bounds the arm links with capsules and cannot
        see the gripper on a joint-only context at all, so it is a coarser authority and
        not a quieter one.

        Public because a caller about to judge a whole path has to know which of the two it
        is getting before it asks, and a caller that would rather refuse than be told a
        fiction can only do that if the answer is reachable. Building the backend is the
        only way to answer, so this builds it, once, through the same cache the evaluation
        uses.
        """
        if self._backend != "fcl":
            return None
        model = self.model_for(arm)
        if model is None:
            return None
        backend = self._exact_mesh_backend(model)
        if backend is None:
            return None
        return str(getattr(backend, "engine", "fcl"))

    def exact_pairs(self, arm: "object | None" = None) -> "ExactPairs | None":
        """This guard's own pair rule and exact distances for ``arm``, in the planner's link names, or ``None``.

        ``None`` wherever the exact mesh backend does not run: the capsule proxy decides nothing the planner refused.
        Where it runs, the rule is the backend's :meth:`~._fcl_self_collision.MeshSelfCollisionBackend.checks`, the
        one :meth:`evaluate` skips pairs by, so a planner pair left to this guard (``planning.band``) is one it judges.
        """
        if self._backend != "fcl":
            return None
        model = self.model_for(arm)
        if model is None:
            return None
        backend = self._exact_mesh_backend(model)
        if backend is None:
            return None
        from .planning.band import ExactPairs

        yaw = float(self._config.kinematics_base_yaw_deg)

        def distance(joints: "Sequence[float]", part_a: str, part_b: str) -> "float | None":
            transforms = arm_link_transforms_mm(model, np.asarray([float(v) for v in joints], dtype=np.float64), arm)
            if transforms is None:
                return None
            return float(backend.distance_mm(transforms, yaw, part_a, part_b))  # type: ignore[attr-defined]

        lowest_of = getattr(backend, "lowest_mm", None)

        def lowest(joints: "Sequence[float]") -> "float | None":
            transforms = arm_link_transforms_mm(model, np.asarray([float(v) for v in joints], dtype=np.float64), arm)
            if transforms is None or lowest_of is None:
                return None
            return float(lowest_of(transforms, yaw))

        # The robot's own base, which no part of this guard holds (``planning.band``): where its shape is known, how near a
        # part past the shoulder comes to it.
        from .planning.self_envelope import ROBOT_BASES

        shape = ROBOT_BASES.get(str(model).lower())
        near_of = getattr(backend, "base_near_mm", None)
        within = float(self._min_distance_mm)

        def base(joints: "Sequence[float]") -> "tuple[str, float] | None":
            transforms = arm_link_transforms_mm(model, np.asarray([float(v) for v in joints], dtype=np.float64), arm)
            if transforms is None or shape is None or near_of is None:
                return None
            part, gap = near_of(transforms, yaw, radius_mm=shape.radius_mm, top_mm=shape.top_mm, within_mm=within)
            return str(part), float(gap)

        # The whole-path judge (whole_path_judge): the same two questions asked of every configuration of a path at once.
        # ExactPairs.first_low and first_near_base hand only the configurations these may refuse to lowest and base,
        # which say why; without them every configuration is asked of lowest and base in turn, as before.
        lowest_many_of = getattr(backend, "lowest_mm_many", None)
        near_base_from_of = getattr(backend, "first_near_base", None)

        def lows(configs: "Sequence[Sequence[float]]") -> "np.ndarray | None":
            transforms = arm_link_transforms_mm_many(model, np.asarray(configs, dtype=np.float64), arm)
            if transforms is None or lowest_many_of is None:
                return None
            return np.asarray(lowest_many_of(transforms, yaw), dtype=np.float64)

        def near_base_from(configs: "Sequence[Sequence[float]]", within_mm: float) -> "int | None":
            transforms = arm_link_transforms_mm_many(model, np.asarray(configs, dtype=np.float64), arm)
            if transforms is None or shape is None or near_base_from_of is None:
                return None
            return int(near_base_from_of(transforms, yaw, radius_mm=shape.radius_mm, top_mm=shape.top_mm,
                                         within_mm=within_mm))

        whole = self._whole_path_judge
        return ExactPairs(checks=backend.checks, frames=backend.part_frames,  # type: ignore[attr-defined]
                          distance=distance, min_distance_mm=self._min_distance_mm,
                          lowest=lowest if callable(lowest_of) else None,
                          base=base if shape is not None and callable(near_of) else None,
                          lows=lows if whole and callable(lowest_of) and callable(lowest_many_of) else None,
                          near_base_from=(near_base_from if whole and shape is not None and callable(near_of)
                                          and callable(near_base_from_of) else None))

    def first_suspect(self, arm: "object | None", configs: np.ndarray, until: int) -> "int | None":
        """The first of ``configs``, ``(N, dof)`` in radians, :meth:`evaluate` might refuse on ``arm``, never later than
        the first it does refuse, found over the whole path at once; ``until`` where none before it may.

        ``None`` wherever :meth:`evaluate` would not run the exact mesh backend on these configurations, the capsule
        proxy, no arm, no model, no engine or no bundle, a configuration the DH chain cannot place: then the path gate
        judges every sample one at a time (``SafetyPreflight.gate_joint_path``). Otherwise every part is placed at every
        sample in one pass (``arm_link_transforms_mm_many``, bit for bit the chain :meth:`evaluate` places one sample
        with), and the backend is asked with what :meth:`evaluate` hands it: the arm against itself and the declared
        fixtures at ``min_distance_mm``, the boxes a camera saw at ``perceived_min_distance_mm``
        (``MeshSelfCollisionBackend.first_suspect``, which says how a sample is passed).
        """
        if self._backend != "fcl" or arm is None:
            return None
        model = self.model_for(arm)
        if model is None:
            return None
        backend = self._exact_mesh_backend(model)
        judge = getattr(backend, "first_suspect", None)
        if not callable(judge):
            return None
        transforms = arm_link_transforms_mm_many(model, configs, arm)
        if transforms is None:
            return None
        return int(judge(
            transforms, float(self._config.kinematics_base_yaw_deg), declared=self._declared_fixtures,
            min_distance_mm=self._min_distance_mm, seen=self._perceived_fixtures,
            seen_min_distance_mm=self._perceived_min_distance_mm, until=until,
        ))

    def _evaluate_fcl(self, ctx: SafetyContext) -> SafetyDecision | None:
        """Exact mesh self-collision. ``None`` where it cannot run, which falls back to capsules."""
        if ctx.target_joints is None or ctx.arm is None:
            return None
        model = self.model_for(ctx.arm)
        if model is None:
            return None
        backend = self._exact_mesh_backend(model)
        if backend is None:
            return None  # no engine or no mesh bundle, so fall back to capsules
        # Every link by the model's nominal table, and the flange, with the hand, its fingers and a wrist camera on it,
        # where the arm's controller puts it, where the arm says (KnowsItsFlange, the owner, 2026-10-12): the camera's
        # boxes stand in the controller's world, so the hand is judged where it stands among them.
        transforms = arm_link_transforms_mm(model, ctx.target_joints.values, ctx.arm)
        if transforms is None:
            return None
        # broadphase=True culls a pair whose bounding spheres cannot be within the margin
        # before the exact query runs. The spheres bound the meshes, so a culled pair
        # cannot violate and the verdict is the brute one; the continuous monitor has run
        # it that way all along. Over 300 random configurations, with two declared fixtures
        # and without, it agreed with the brute verdict every time and cost 2.179 ms
        # against 5.812 ms and 1.864 ms against 4.898 ms per configuration. A judged path
        # pays this per sample.
        #
        # The arm against itself and the declared fixtures at min_distance_mm, then the boxes a camera
        # saw, each turned as the planner holds it, at perceived_min_distance_mm: in that order, so a
        # refusal names the same first pair it always did.
        yaw = float(self._config.kinematics_base_yaw_deg)
        limit = self._min_distance_mm
        seen: tuple[AxisAlignedBox, ...] = ()
        hit = backend.evaluate(  # type: ignore[attr-defined]
            transforms, yaw, self._declared_fixtures, limit, broadphase=True,
        )
        if hit is None and self._perceived_fixtures:
            limit, seen = self._perceived_min_distance_mm, self._perceived_fixtures
            hit = backend.evaluate(  # type: ignore[attr-defined]
                transforms, yaw, seen, limit, broadphase=True, arm_pairs=False,
            )
        if hit is None:
            return SafetyDecision.accept(self.name)
        pair, dmm = hit
        engine = getattr(backend, "engine", "fcl")  # 'coal' preferred, 'fcl' the fallback
        where = _fixture_detail(pair, seen or self._declared_fixtures, seen=bool(seen))
        # What the world that built a seen box knows of it beyond its geometry (AxisAlignedBox.note).
        note = where.get("box_note", "")
        return SafetyDecision.reject(
            self.name,
            SafetyReason.SELF_COLLISION,
            message=f"{pair}: mesh distance {dmm:.3f} mm < {limit:.3f} mm" + (f" ({note})" if note else ""),
            detail={
                "pair": pair,
                "signed_distance_mm": f"{dmm:.6f}",
                "min_distance_mm": f"{limit:.6f}",
                "backend": engine,
                # Where the arm stood, which no refusal said before (2026-09-30), and where the box stood.
                "joints_deg": "(" + ", ".join(
                    f"{math.degrees(float(v)):.1f}" for v in ctx.target_joints.values) + ")",
                **where,
            },
        )

    @property
    def perceived_min_distance_mm(self) -> float:
        """The distance this guard keeps from a box a camera saw, in mm; the box is already grown by the world's margin."""
        return float(self._perceived_min_distance_mm)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_skip_pair(name_a: str, name_b: str, *, last_link: str | None) -> bool:
        """Skip the pairs that always overlap by construction.

        Adjacent links share an end-point. The non-adjacent wrist links, ``link_3``,
        ``link_4`` and ``link_5`` on a UR-style spherical wrist, are physically too
        close for a capsule approximation with realistic radii, so any pair within two
        indices of each other is excluded to keep out spurious wrist-against-wrist
        rejections. A real collision in that region needs the exact-mesh backend.
        """
        if name_a.startswith("link_") and name_b.startswith("link_"):
            ia = int(name_a.split("_", 1)[1])
            ib = int(name_b.split("_", 1)[1])
            if abs(ia - ib) <= 2:
                return True
        if {name_a, name_b} == {"base", "link_0"}:
            return True
        # link_1 starts at the shoulder joint, which sits on top of the base column by
        # construction, so its capsule start-point always lies inside the base capsule
        # envelope. The link_1 capsule cannot swing into the base, because it pivots
        # about the base axis, so this pair is skipped as well. A real upper-arm
        # collision is caught by link_2, the upper-arm shaft.
        if {name_a, name_b} == {"base", "link_1"}:
            return True
        # The wrist-mounted tool inevitably overlaps the last link, whichever index the arm gives it.
        if last_link is not None and {name_a, name_b} == {"tool", last_link}:
            return True
        return False


def _fixture_detail(pair: str, fixtures: "Sequence[AxisAlignedBox]", *, seen: bool) -> dict[str, str]:
    """Where the fixture a refusal names stands: its centre, sizes, turn and footprint corners, in mm and degrees.

    ``seen`` says the fixture is a box a camera saw rather than one declared. Only that word for a fixture the refusal
    cannot pick out by its name, an unnamed one or one of two sharing a name; nothing for a pair of the arm alone.
    """
    if "|fixture:" not in pair:
        return {}
    kind = {"fixture": "seen" if seen else "declared"}
    name = pair.split("|fixture:", 1)[1]
    named = [box for box in fixtures if (box.name or "fixture") == name]
    if len(named) != 1:
        return kind
    box = named[0]
    turned = box.turned
    half = np.asarray(box.half_extents_mm if turned is None else turned.half_extents_mm, dtype=np.float64)
    yaw = 0.0 if turned is None else float(turned.yaw_rad)
    cx, cy, cz = (float(v) for v in box.center_mm)
    c, s = math.cos(yaw), math.sin(yaw)
    corners = " ".join(
        f"({cx + u * c - v * s:.1f}, {cy + u * s + v * c:.1f})"
        for u, v in ((-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]), (-half[0], half[1]))
    )
    # A support surface's solid tilts with the surface as the camera read it: its z is the box that encloses it, and
    # the refusal says by how much it tilts.
    rotation = None if turned is None else getattr(turned, "rotation", None)
    tilt: dict[str, str] = {}
    low, high = cz - half[2], cz + half[2]
    if rotation is not None:
        matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        reach_z = float(np.abs(matrix[2]) @ half)
        low, high = cz - reach_z, cz + reach_z
        tilt = {"box_tilt_deg": f"{math.degrees(math.acos(max(-1.0, min(1.0, float(matrix[2, 2]))))):.2f}"}
    note = str(getattr(box, "note", "") or "")
    return {
        **kind,
        "box_centre_mm": f"({cx:.1f}, {cy:.1f}, {cz:.1f})",
        "box_size_mm": f"{2.0 * half[0]:.1f} x {2.0 * half[1]:.1f} x {2.0 * half[2]:.1f}",
        "box_yaw_deg": f"{math.degrees(yaw):.1f}",
        **tilt,
        "box_corners_mm": f"{corners}, z {low:.1f} to {high:.1f}",
        **({"box_note": note} if note else {}),
    }
