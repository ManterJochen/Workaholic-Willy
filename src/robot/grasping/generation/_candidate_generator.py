"""Generates ranked-input grasp poses (silhouette / geometry-first / dense) for the calculator."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

from src.geometry import Frame
from src.robot.grasping.constants import (
    CANDIDATE_GENERATOR_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.contacts import (
    ContactPair,
    SurfaceSamples,
    find_antipodal_pairs,
)
from src.robot.grasping.geometry import estimate_surface_normals
from src.robot.grasping.planning import GraspPose, generate_grasp_poses
from src.robot.grasping.scoring import certify_contact_pair, width_fit_score

from src.robot.grasping.generation._camera_geometry import _SharedGeometryUtil

if TYPE_CHECKING:
    from src.robot.grasping.generation.mask_analyzer import MaskAnalysis


#: The chain of counts from points to pairs to poses is recorded nowhere else.
#: One line per generation call, never per pair or per pose: the inner loops
#: here run into the thousands.
logger = create_grasping_logger("GraspCandidateGenerator", CANDIDATE_GENERATOR_LOG_FILE)


#: Antipodal-pair acceptance gates passed to :func:`find_antipodal_pairs` (geometry-first + dense).
_ANTIPODAL_NORMAL_OPPOSITION_THRESHOLD: float = 0.75
_ANTIPODAL_AXIS_ALIGNMENT_THRESHOLD: float = 0.45
#: Dense-cloud graspability weight at/below which a point is treated as invalid (occlusion-edge guard).
_DENSE_GRASPABILITY_MIN: float = 1e-3
#: Silhouette-fallback antipodal-score blend weights (width + depth-confidence + axis-consistency; sum 1.0).
_SILHOUETTE_WIDTH_WEIGHT: float = 0.45
_SILHOUETTE_DEPTH_WEIGHT: float = 0.35
_SILHOUETTE_AXIS_WEIGHT: float = 0.20
#: A silhouette this elongated (minor/major aspect at/below this) closes across its minor (short) axis so the
#: gripper spans the narrow, gripper-fitting dimension; near-square (cube-like, above this) closes along the
#: principal (major) axis. See ``silhouette_contact_poses``.
_ELONGATED_ASPECT_MAX: float = 0.7


def _mask_non_convexity(mask: np.ndarray) -> float | None:
    """Return ``1 - solidity`` (contour_area / convex_hull_area) for the largest contour in ``mask``.

    A convex blob yields solidity 1.0 (non-convexity 0.0); a U-/L-shape yields lower solidity. Returns
    :data:`None` when no contour can be extracted (empty mask, degenerate hull).
    """
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    binary = (arr > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return None
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv.contourArea)
    area = float(cv.contourArea(contour))
    if area <= 0.0:
        return None
    hull = cv.convexHull(contour)
    hull_area = float(cv.contourArea(hull))
    if hull_area <= 0.0:
        return None
    solidity = float(np.clip(area / hull_area, 0.0, 1.0))
    return 1.0 - solidity


class _GraspCandidateGenerator:
    """Generates ranked-input grasp poses (silhouette / geometry-first / dense)."""

    def __init__(
        self,
        *,
        geom: _SharedGeometryUtil,
        approach_cam: np.ndarray,
        min_grip_mm: float,
        max_grip_mm: float,
        max_candidates: int,
        max_geometry_points: int,
        normal_radius_mm: float,
        axis_end_frac: float,
        dense_non_convexity_threshold: float,
        friction_coefficient: float | None,
        dense_overran_getter: Callable[[], bool],
        oblique_approach: bool = False,
        oblique_tilt_deg: float = 30.0,
        oblique_azimuths: int = 4,
    ) -> None:
        self._geom = geom
        self.approach_cam = approach_cam
        # Oblique/multi-angle approach (default-off): each contact pair also spawns poses tilted off the
        # top-down ``approach_cam`` (``oblique_tilt_deg`` / ``oblique_azimuths``), reaching gap-threading
        # grasps a single top-down axis cannot express. Off means just ``[approach_cam]`` and a single
        # generate_grasp_poses call.
        self._oblique_approach = bool(oblique_approach)
        self._oblique_tilt_deg = float(oblique_tilt_deg)
        self._oblique_azimuths = int(oblique_azimuths)
        self.min_grip_mm = min_grip_mm
        self.max_grip_mm = max_grip_mm
        self.max_candidates = max_candidates
        self.max_geometry_points = max_geometry_points
        self.normal_radius_mm = normal_radius_mm
        self.axis_end_frac = axis_end_frac
        self.dense_non_convexity_threshold = dense_non_convexity_threshold
        self.friction_coefficient = friction_coefficient
        # Reads the facade-owned cross-call dense-overrun latch (single source of truth; no setter here).
        self._dense_overran_getter = dense_overran_getter

    def _approach_axes(self) -> list[np.ndarray]:
        """Approach axes for generation: just ``[approach_cam]``, or plus oblique-tilted variants."""
        base = np.asarray(self.approach_cam, dtype=np.float64)
        if not self._oblique_approach or self._oblique_azimuths < 1 or self._oblique_tilt_deg <= 0.0:
            return [base]
        # Orthonormal (u, v) basis of the plane perpendicular to the base axis, then tilt base toward each azimuth.
        seed = np.array([1.0, 0.0, 0.0]) if abs(float(base[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = seed - base * float(np.dot(seed, base))
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        v = np.cross(base, u)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        tilt = np.deg2rad(self._oblique_tilt_deg)
        axes = [base]
        for k in range(self._oblique_azimuths):
            az = 2.0 * np.pi * float(k) / float(self._oblique_azimuths)
            side = float(np.cos(az)) * u + float(np.sin(az)) * v
            tilted = float(np.cos(tilt)) * base + float(np.sin(tilt)) * side
            axes.append(tilted / max(float(np.linalg.norm(tilted)), 1e-12))
        return axes

    def _poses_from_pairs(self, pairs: list[ContactPair], *, max_poses: int) -> list[GraspPose]:
        """Generate poses for ``pairs`` at every approach axis, unioned."""
        axes = self._approach_axes()
        if len(axes) == 1:
            return generate_grasp_poses(
                pairs, preferred_approach_axis=axes[0], frame=Frame.CAMERA, max_poses=max_poses
            )
        out: list[GraspPose] = []
        for ax in axes:
            out.extend(
                generate_grasp_poses(pairs, preferred_approach_axis=ax, frame=Frame.CAMERA, max_poses=max_poses)
            )
        return out

    def decide_dense_sampling(
        self,
        *,
        dense_sampling: bool | None,
        mask: np.ndarray,
        other_object_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
        telemetry: dict,
    ) -> bool:
        """Decide whether to run the dense geometry sampler this call.

        * Explicit ``True`` / ``False`` is always respected.
        * ``None`` triggers heuristic auto:

          1. If the previous dense call exceeded the runtime budget,
             skip dense (for the life of this calculator, until an explicit override).
          2. Enable dense when ``other_object_masks`` is non-empty
             (clutter signal).
          3. Enable dense when the target mask is markedly non-convex,
             i.e. ``1 - solidity`` exceeds
             :attr:`dense_non_convexity_threshold`.

        The decision and contributing signals are recorded in
        ``telemetry`` under ``dense_decision`` / ``dense_auto_reason`` /
        ``mask_non_convexity``.
        """
        if dense_sampling is True or dense_sampling is False:
            telemetry["dense_decision"] = bool(dense_sampling)
            telemetry["dense_auto_reason"] = "explicit"
            logger.debug("Dense sampling %s (explicit)", "on" if dense_sampling else "off")
            return bool(dense_sampling)

        if self._dense_overran_getter():
            telemetry["dense_decision"] = False
            telemetry["dense_auto_reason"] = "degraded_after_overrun"
            # Not a cooldown, whatever it was called. The latch is cleared in exactly one place,
            # inside the dense branch that this return has just skipped, so nothing that happens
            # later can clear it: once set it holds for the life of this calculator, and only an
            # explicit `dense_sampling=True` reaches the branch that resets it. The word "cooldown"
            # promised a recovery that does not exist.
            #
            # And it is reachable now. The 150 ms budget was set against a loop that rejected every
            # pair before constructing one, because the perception producer flattened the depth
            # under each mask and a plane has no opposing normals. A measured surface makes the
            # dense path do the work the budget was never tested against.
            logger.warning(
                "Dense sampling skipped: a previous dense call overran its budget, and this "
                "calculator stays on the weaker silhouette path until it is rebuilt or a caller "
                "passes dense_sampling=True"
            )
            return False

        if other_object_masks:
            telemetry["dense_decision"] = True
            telemetry["dense_auto_reason"] = "clutter"
            return True

        non_convexity = _mask_non_convexity(mask)
        if non_convexity is not None:
            telemetry["mask_non_convexity"] = float(non_convexity)
            if non_convexity > self.dense_non_convexity_threshold:
                telemetry["dense_decision"] = True
                telemetry["dense_auto_reason"] = "non_convex_mask"
                logger.debug(
                    "Dense sampling on: mask non-convexity %.3f > %.3f",
                    float(non_convexity),
                    self.dense_non_convexity_threshold,
                )
                return True

        telemetry["dense_decision"] = False
        telemetry["dense_auto_reason"] = "default_silhouette"
        logger.debug("Dense sampling off: default silhouette path")
        return False

    def apply_force_closure_certificates(
        self,
        pairs: list[ContactPair],
        poses: list[GraspPose],
    ) -> int:
        """Stamp Coulomb force-closure certificates onto each pose (no-op unless friction is set); returns the certified count.

        ``pairs`` and ``poses`` must share input order (the :func:`generate_grasp_poses` contract).
        """
        if self.friction_coefficient is None:
            return 0
        certified = 0
        mu = self.friction_coefficient
        for pair, pose in zip(pairs, poses):
            cert = certify_contact_pair(pair, mu)
            pose.metadata["force_closure"] = cert.to_dict()
            pose.metadata["force_closure_certified"] = cert.is_certified
            if cert.is_certified:
                # Override the heuristic marker only when a real
                # analytical certificate exists. Non-certified poses
                # remain "heuristic": they are not proven.
                pose.metadata["confidence_kind"] = "analytical"
                certified += 1
        # Aggregated after the loop, never inside it. An "analytical" confidence_kind
        # downstream is only as good as this ratio; 0 certified at a configured mu
        # means every pose stays heuristic.
        logger.debug(
            "Force closure (mu=%.3f): %d/%d pose(s) certified analytically",
            float(mu),
            certified,
            len(poses),
        )
        return certified

    def geometry_first_poses(
        self,
        points_mm: np.ndarray,
        *,
        analysis: MaskAnalysis,
        depth_confidence: float,
    ) -> list[GraspPose]:
        normals = estimate_surface_normals(
            points_mm,
            radius_mm=self.normal_radius_mm,
            min_neighbors=6,
            max_neighbors=64,
        )
        if normals.valid_count < 2:
            logger.info(
                "Geometry-first: no candidates; only %d point(s) carry a valid "
                "surface normal (need 2)",
                normals.valid_count,
            )
            return []
        pairs = find_antipodal_pairs(
            points_mm,
            normals,
            min_width_mm=self.min_grip_mm,
            max_width_mm=self.max_grip_mm,
            normal_opposition_threshold=_ANTIPODAL_NORMAL_OPPOSITION_THRESHOLD,
            axis_alignment_threshold=_ANTIPODAL_AXIS_ALIGNMENT_THRESHOLD,
            max_pairs=self.max_candidates,
        )
        poses = self._poses_from_pairs(pairs, max_poses=self.max_candidates)
        for index, pose in enumerate(poses):
            pose.metadata.setdefault("label", f"antipodal_{index}")
            pose.metadata.setdefault("planner", "geometry_first")
            pose.metadata.setdefault("depth_confidence", round(float(depth_confidence), 3))
            pose.metadata.setdefault("axis_consistency", round(float(pose.metadata.get("axis_alignment", 0.0)), 3))
            pose.metadata.setdefault("mask_area_px", analysis.area_px)
            # The antipodal planner score is a heuristic stability proxy, not a calibrated probability.
            pose.metadata.setdefault("confidence_kind", "heuristic")
        # Analytical force-closure certification (no-op when ``friction_coefficient`` is unset).
        self.apply_force_closure_certificates(pairs, poses)
        log = logger.info if poses else logger.warning
        log(
            "Geometry-first: %d valid normal(s) -> %d antipodal pair(s) -> %d pose(s) "
            "(grip %.1f-%.1f mm, cap %d)",
            normals.valid_count,
            len(pairs),
            len(poses),
            self.min_grip_mm,
            self.max_grip_mm,
            self.max_candidates,
        )
        return poses

    def dense_geometry_poses(
        self,
        samples: SurfaceSamples,
        *,
        analysis: MaskAnalysis,
        depth_confidence: float,
    ) -> list[GraspPose]:
        """Build antipodal poses from a graspability-weighted dense cloud.

        The dense sampler already returns normals and a per-point
        graspability weight, so this path skips the local normal
        estimation ``geometry_first_poses`` performs and passes the
        precomputed values straight into ``find_antipodal_pairs`` via
        its ``valid_mask`` argument (downweighted points are invalid).
        """
        if samples.is_empty:
            logger.info(
                "Dense: no candidates; the surface sampler returned an empty cloud"
            )
            return []
        # Treat graspability == 0 as invalid so the antipodal search
        # never anchors on occlusion-edge points.
        valid_mask = (
            np.asarray(samples.graspability, dtype=np.float32) > _DENSE_GRASPABILITY_MIN
        ) & np.asarray(samples.normals.valid_mask, dtype=bool)
        if not np.any(valid_mask):
            # Every dense point was either occlusion-edge (graspability ~ 0) or had
            # no usable normal: a mask/depth problem upstream, not a gripper-fit one.
            logger.warning(
                "Dense: no candidates; all %d sampled point(s) failed the "
                "graspability/normal validity gate",
                int(samples.size),
            )
            return []
        pairs = find_antipodal_pairs(
            samples.points_mm,
            samples.normals,
            min_width_mm=self.min_grip_mm,
            max_width_mm=self.max_grip_mm,
            normal_opposition_threshold=_ANTIPODAL_NORMAL_OPPOSITION_THRESHOLD,
            axis_alignment_threshold=_ANTIPODAL_AXIS_ALIGNMENT_THRESHOLD,
            max_pairs=self.max_candidates,
            valid_mask=valid_mask,
        )
        poses = self._poses_from_pairs(pairs, max_poses=self.max_candidates)
        for index, pose in enumerate(poses):
            pose.metadata.setdefault("label", f"dense_{index}")
            pose.metadata.setdefault("planner", "dense_surface")
            pose.metadata.setdefault("depth_confidence", round(float(depth_confidence), 3))
            pose.metadata.setdefault("mask_area_px", analysis.area_px)
            pose.metadata.setdefault("dense_sampling_points", int(samples.size))
            # Dense scores are heuristic; flag them so downstream policy never treats them as calibrated.
            pose.metadata.setdefault("confidence_kind", "heuristic")
        # Analytical force-closure certification (no-op when ``friction_coefficient`` is unset).
        self.apply_force_closure_certificates(pairs, poses)
        log = logger.info if poses else logger.warning
        log(
            "Dense: %d sampled point(s), %d valid -> %d antipodal pair(s) -> %d pose(s)",
            int(samples.size),
            int(np.count_nonzero(valid_mask)),
            len(pairs),
            len(poses),
        )
        return poses

    def limit_geometry_points(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=np.float64)
        if points.shape[0] <= self.max_geometry_points:
            return points
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            num=self.max_geometry_points,
            dtype=np.int64,
        )
        return points[np.unique(indices)]

    def silhouette_contact_poses(
        self,
        *,
        analysis: MaskAnalysis,
        depth_map: np.ndarray,
        median_depth_mm: float,
        pixel_to_mm: float,
        point_3d_cam: np.ndarray | None,
        scale_to_mm: float,
        depth_confidence: float,
        penetration_mm: float = 0.0,
        up_cam: np.ndarray | None = None,
    ) -> list[GraspPose]:
        # ``penetration_mm`` descends every silhouette candidate below its referenced surface (a larger
        # camera-frame depth is deeper into the object) so the fingers wrap it instead of resting on the
        # surface. Default 0.0 leaves every candidate unchanged (``x + 0.0 == x``). It is applied to the
        # back-projected depth so the candidate's x/y stay consistent with the deeper z; it has no effect
        # when the centre is supplied directly via ``point_3d_cam`` (never on the live pick path).
        center = self._geom.center_point_mm(
            analysis,
            depth_map,
            median_depth_mm + penetration_mm,
            point_3d_cam,
            scale_to_mm,
        )
        aspect = float(
            np.clip(
                analysis.extent_minor_px / max(analysis.extent_major_px, 1e-9),
                0.0,
                1.0,
            )
        )
        # The closing axis is the finger-separation direction: ``make_synthetic_pair`` places the two
        # contacts at ``center +/- closing_axis * width/2`` and ships ``distance_mm = width_mm``, so the
        # width passed in must be the extent of the axis that closes. An elongated silhouette (aspect at
        # or below ``_ELONGATED_ASPECT_MAX``) closes across the short (minor) axis, perpendicular to the
        # principal (major) axis, so the fingers sit on the two long sides and span the narrow,
        # gripper-fitting dimension; a near-square one closes along the principal (major) axis and takes
        # the major extent. Pairing the minor extent with a major-axis close would put both contacts
        # inside the object and understate the span the gripper has to close on; taking the major extent
        # there means a candidate whose span exceeds the aperture is filtered rather than commanded too
        # narrow. Measured on 503 ground-truth instance masks from 30 `v1_proof` scenes with the real
        # `MaskAnalyzer`: 40.0 % take the near-square branch, where the major-minor extent gap is a
        # median 10.7 px, 15.0 % of the major extent; the silhouette rung's MAE is 10.72 mm against the
        # true span, bias -7.50 mm, with 72.6 % of candidates commanded narrower than the object against
        # 39.4 % on the `sfe_fused` rung.
        #
        # Closing along the long axis is too wide for the gripper. ``align_closing_to_base_x`` hides
        # that, because the blind-IK base-X yaw rewrites the close before exec; a planner that owns the
        # motion (cuRobo, align off) closes the long axis faithfully and there is no grasp: the on-edge
        # sugar box goes 5/5 -> 0/5 under cuRobo with a forearm|finger Coal-reject.
        principal_2d = analysis.principal_axis
        if aspect <= _ELONGATED_ASPECT_MAX:
            closing_2d = np.array([-principal_2d[1], principal_2d[0]], dtype=np.float64)
            width_mm = float(analysis.extent_minor_px * pixel_to_mm)
        else:
            closing_2d = principal_2d
            width_mm = float(analysis.extent_major_px * pixel_to_mm)
        closing_axis = self._geom.principal_to_3d_axis(closing_2d)
        # Level it into the support plane. principal_to_3d_axis lands the axis in the image plane, which
        # is the plane a grasp closes in only for a camera aimed straight down the support normal. On a
        # tilted camera it is not, and 98.9 % of real jaw grasps close horizontally. See
        # _SharedGeometryUtil.level_axis_to_support_plane for the measurement. ``up_cam`` None (no
        # CAMERA->BASE transform, or the caller disabled it) leaves the axis unchanged.
        if up_cam is not None:
            closing_axis = self._geom.level_axis_to_support_plane(closing_axis, up_cam)
        contacts = [
            self.make_synthetic_pair(
                center,
                closing_axis,
                width_mm,
                label="centroid",
                depth_confidence=depth_confidence,
                axis_consistency=1.0,
                aspect=aspect,
                analysis=analysis,
            )
        ]

        offset_px = analysis.extent_major_px * self.axis_end_frac
        for sign, label in ((1.0, "axis_end_pos"), (-1.0, "axis_end_neg")):
            px = analysis.centroid_xy[0] + sign * offset_px * analysis.principal_axis[0]
            py = analysis.centroid_xy[1] + sign * offset_px * analysis.principal_axis[1]
            depth_at_point = self._geom.sample_depth_mm(depth_map, px, py, scale_to_mm)
            if depth_at_point is None:
                continue
            point = self._geom.pixel_depth_to_3d(
                px, py, depth_at_point + penetration_mm, depth_map
            )
            contacts.append(
                self.make_synthetic_pair(
                    point,
                    closing_axis,
                    width_mm,
                    label=label,
                    depth_confidence=depth_confidence,
                    axis_consistency=1.0,
                    aspect=aspect,
                    analysis=analysis,
                )
            )

        pairs = [pair for pair in contacts if self.min_grip_mm <= pair.distance_mm <= self.max_grip_mm]
        poses = self._poses_from_pairs(pairs, max_poses=self.max_candidates)
        for pose in poses:
            pose.metadata.setdefault("planner", "silhouette_contact_fallback")
            # Silhouette scores are heuristic geometric proxies, not calibrated probabilities.
            pose.metadata.setdefault("confidence_kind", "heuristic")
        if not pairs:
            # The most common silhouette dead end, and the width is the whole
            # explanation: the extent of the closing axis, minor on an elongated
            # mask and major on a near-square one, does not fit between the jaws.
            logger.warning(
                "Silhouette: no candidates; all %d synthetic pair(s) fell outside "
                "the grip range (width %.1f mm, allowed %.1f-%.1f mm)",
                len(contacts),
                width_mm,
                self.min_grip_mm,
                self.max_grip_mm,
            )
        else:
            logger.info(
                "Silhouette: %d/%d synthetic pair(s) within grip range -> %d pose(s) "
                "(width %.1f mm, aspect %.2f)",
                len(pairs),
                len(contacts),
                len(poses),
                width_mm,
                aspect,
            )
        # Synthetic silhouette pairs are perfectly antipodal, so a configured friction coefficient
        # certifies them by construction.
        self.apply_force_closure_certificates(pairs, poses)
        return poses

    def make_synthetic_pair(
        self,
        center_mm: np.ndarray,
        closing_axis: np.ndarray,
        width_mm: float,
        *,
        label: str,
        depth_confidence: float,
        axis_consistency: float,
        aspect: float,
        analysis: MaskAnalysis,
    ) -> ContactPair:
        axis = self._geom.unit(closing_axis)
        half_width = 0.5 * float(width_mm)
        point_a = center_mm - half_width * axis
        point_b = center_mm + half_width * axis
        width_score = self.width_fit(width_mm)
        score = float(
            np.clip(
                _SILHOUETTE_WIDTH_WEIGHT * width_score
                + _SILHOUETTE_DEPTH_WEIGHT * depth_confidence
                + _SILHOUETTE_AXIS_WEIGHT * axis_consistency,
                0.0,
                1.0,
            )
        )
        return ContactPair(
            point_a=point_a,
            point_b=point_b,
            normal_a=-axis,
            normal_b=axis,
            distance_mm=float(width_mm),
            antipodal_score=score,
            axis_alignment=float(np.clip(axis_consistency, 0.0, 1.0)),
            normal_opposition=1.0,
            metadata={
                "label": label,
                "depth_confidence": round(float(depth_confidence), 3),
                "axis_consistency": round(float(axis_consistency), 3),
                "aspect_ratio": round(aspect, 3),
                "mask_area_px": analysis.area_px,
                "extent_major_px": round(float(analysis.extent_major_px), 1),
                "extent_minor_px": round(float(analysis.extent_minor_px), 1),
                "orientation_deg": round(float(analysis.orientation_deg), 1),
                "stability": round(float(depth_confidence), 3),
            },
        )

    def width_fit(self, grip_width_mm: float) -> float:
        return width_fit_score(
            grip_width_mm,
            self.min_grip_mm,
            self.max_grip_mm,
        )
