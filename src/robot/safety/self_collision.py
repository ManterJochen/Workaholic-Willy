"""SelfCollisionGuard, the link, tool and fixture self-collision guard.

There are two backends.

``capsule`` is the default and is always available. It approximates the arm as a chain
of capsules, one per link, plus a tool capsule rooted at the TCP and a base column
capsule. Fixtures are the axis-aligned boxes declared in :class:`FixtureBoxConfig`.
The distance is closed-form, so the per-move cost stays under a millisecond.

* On a UR the arm-against-arm capsule chain comes from the bundled DH table in
  :mod:`_ur_kinematics`, and every non-adjacent link pair is checked against the others
  and against every fixture.
* On another vendor, KUKA or the sim, that chain is unavailable, so the guard checks
  the tool against the base and the tool against the fixtures only. Arm-against-arm
  coverage needs a DH or URDF entry, which is not bundled for KUKA.

``fcl`` is exact mesh-against-mesh distance over the bundled per-model collision
meshes, with Coal preferred and python-fcl as the fallback, in
:mod:`_fcl_self_collision`. It removes two measured errors of the capsule proxy, the
skipped-wrist false negative and the fat-radius false positive, by checking every link
pair more than one DH frame apart exactly. Any UR model with a bundled DH chain and a
committed ``{model}_collision_meshes.npz`` qualifies, and ``ur5e`` and ``ur3e`` ship
one. Where the engine or the bundle is missing the guard falls back to the capsule
path and logs that fallback with a reason token, because the capsule proxy is
materially coarser and a cell that swaps one for the other in silence is a
safety-relevant regression.

Configuration
-------------
* ``min_distance_mm`` is the smallest allowed signed distance between any monitored
  shape pair, and anything below it is a collision.
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

from ._capsule import (
    AxisAlignedBox,
    Capsule,
    capsule_box_distance_mm,
    capsule_capsule_distance_mm,
)
from ._ur_kinematics import ur_link_origins_mm, ur_link_transforms_mm
from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import SelfCollisionSafetyConfig

__all__ = ["SelfCollisionGuard"]


_LOGGER = logging.getLogger(__name__)

# Default capsule radii and lengths, used where the operator overrides none. They
# match a typical UR link of 60 mm cross-section and a 70 mm gripper envelope.
_DEFAULT_LINK_RADIUS_MM = 60.0
_DEFAULT_TOOL_RADIUS_MM = 70.0
_DEFAULT_TOOL_LENGTH_MM = 150.0
_DEFAULT_BASE_RADIUS_MM = 80.0
_DEFAULT_BASE_HEIGHT_MM = 150.0


class SelfCollisionGuard:
    """Self-collision guard.

    Stateless across calls. :meth:`SafetyPreflight.from_safety_config` constructs it
    when ``safety.self_collision.enforce`` is ``True``.
    """

    name = "self_collision"

    def __init__(self, config: "SelfCollisionSafetyConfig") -> None:
        self._config = config
        self._min_distance_mm = float(config.min_distance_mm)
        # What the planner keeps clear, so it stops proposing configurations this guard
        # rejects. It is separate from _min_distance_mm on purpose; SelfCollisionConfig
        # says why.
        self._planner_margin_mm = float(getattr(config, 'planner_margin_mm', 0.0) or 0.0)
        self._backend = config.backend
        # The exact-mesh backend, built lazily and only for backend='fcl'. ``None``
        # means not yet built, or unavailable.
        self._fcl_backend: object | None = None
        self._fcl_backend_built = False
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

        The boxes are axis-aligned here while the planner takes them turned. That is deliberate and
        it goes in the safe direction: an axis-aligned box enclosing a turned one is bigger, so this
        guard is never more permissive than the planner. It can be stricter, and a diagonal part is
        where that will show.

        Never call this with obstacles nobody can vouch for. An empty sequence means the cell is back
        to its declared geometry, which is the honest state when the cameras cannot answer.
        """
        self._perceived_fixtures = tuple(boxes)

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
        """Per-link capsules for an arm with UR kinematics, or ``None`` where none derive.

        An explicit ``self_collision.kinematics_model`` selects the bundled DH table
        and overrides the vendor gate, which lets a non-UR vendor that is physically a
        UR arm, such as a UR5e in the Isaac sim, opt into real arm-against-arm
        capsules. Without it only a genuine ``vendor == "ur"`` arm qualifies, keyed by
        its own model.
        """
        if ctx.target_joints is None or ctx.arm is None:
            return None
        model = self._config.kinematics_model
        if model is None:
            if ctx.arm.capabilities.vendor != "ur":
                return None
            model = ctx.arm.capabilities.model
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
        if tool is not None:
            labelled.append(("tool", tool))

        # ---- capsule vs capsule (skip adjacent arm links) -----------
        for i in range(len(labelled)):
            name_i, cap_i = labelled[i]
            for j in range(i + 1, len(labelled)):
                name_j, cap_j = labelled[j]
                if self._should_skip_pair(name_i, name_j):
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

    def _evaluate_fcl(self, ctx: SafetyContext) -> SafetyDecision | None:
        """Exact mesh self-collision. ``None`` where it cannot run, which falls back to capsules."""
        if ctx.target_joints is None or ctx.arm is None:
            return None
        model = self._config.kinematics_model
        if model is None:
            if ctx.arm.capabilities.vendor != "ur":
                return None
            model = ctx.arm.capabilities.model
        if not self._fcl_backend_built:  # build the BVH models once, caching None too
            from ._fcl_self_collision import make_backend, mesh_backend_status
            variant = getattr(self._config, "collision_mesh_variant", None)
            self._fcl_backend = make_backend(
                model, self._config.mesh_dir, variant,
                coupling_mm=float(getattr(self._config, "coupling_mm", 0.0) or 0.0),
            )
            self._fcl_backend_built = True
            # The config asked for the exact-mesh backend. Falling back to the coarser
            # capsule proxy in silence would look like a robot that cannot grasp
            # anything rather than a missing safety authority, because the proxy
            # over-rejects at _DEFAULT_LINK_RADIUS_MM. So this says it once and records
            # the reason, which lets telemetry and the boot banner show a degraded
            # cell.
            self._fcl_status = mesh_backend_status(model, self._config.mesh_dir, variant)
            if self._fcl_backend is None:
                _LOGGER.warning(
                    "SelfCollisionGuard: backend='fcl' requested for model %r but the exact-mesh backend is "
                    "unavailable (%s); running the capsule fallback for this cell.", model, self._fcl_status,
                )
        backend = self._fcl_backend
        if backend is None:
            return None  # no engine or no mesh bundle, so fall back to capsules
        transforms = ur_link_transforms_mm(model, ctx.target_joints.values)
        if transforms is None:
            return None
        hit = backend.evaluate(  # type: ignore[attr-defined]
            transforms, float(self._config.kinematics_base_yaw_deg), self._fixtures, self._min_distance_mm
        )
        if hit is None:
            return SafetyDecision.accept(self.name)
        pair, dmm = hit
        engine = getattr(backend, "engine", "fcl")  # 'coal' preferred, 'fcl' the fallback
        return SafetyDecision.reject(
            self.name,
            SafetyReason.SELF_COLLISION,
            message=f"{pair}: mesh distance {dmm:.3f} mm < {self._min_distance_mm:.3f} mm",
            detail={
                "pair": pair,
                "signed_distance_mm": f"{dmm:.6f}",
                "min_distance_mm": f"{self._min_distance_mm:.6f}",
                "backend": engine,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_skip_pair(name_a: str, name_b: str) -> bool:
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
        # The wrist-mounted tool inevitably overlaps the last link.
        if {name_a, name_b} == {"tool", "link_5"}:
            return True
        return False
