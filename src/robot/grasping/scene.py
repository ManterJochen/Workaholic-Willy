"""Grasps on a point cloud, with no robot, no cell and no configuration.

The smallest useful thing this package does. A caller with a segmented cloud and a support height
wants ranked grasps, and everything needed is pure already: `generate_support_footprint_grasps`
takes a bare BASE-frame array, takes a fully defaulted frozen jaw, and returns frozen candidates.
This module is the name for that question; without it the only way in is a `GraspCalculator` built
with a camera matrix and a config block for a cell that does not exist.

A `Scene` that only called the analytic primitive would be a safety defect. A cell whose config
says `grasping.calculator: deep` would receive analytic geometry, silently, under the learned
generator's name; `calculator_factory` refuses exactly that and fails closed. A check that sweeps
for modules constructing `GraspCalculator` by name cannot see this file either, because the
primitive is reached directly.

So the answer is two doors and a stamped result:

    Scene.from_cloud(...)         no config exists, so no selector is being ignored. The result
                                  says ``generator="geometric"`` and cannot be mistaken.
    Scene.from_robot_config(...)  a config exists and supplies the support height and the jaw. The
                                  generator selector is not read here, so the result still says
                                  ``generator="geometric"``.

The result names its generator either way, and that is what makes the first door safe: a silent
substitution is only dangerous while it is silent, and `SceneGrasps.generator` is on every report
and in every `to_dict`. A number filed under the wrong generator is how a stack gets judged on work
it did not do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from src.contracts import UNSET, Maybe, chosen

from src.robot.grasping.generation.support_footprint import (
    SupportFootprintCandidate,
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["Scene", "SceneGrasps"]

#: What produced a set of candidates. Spelled the way the config key
#: `robot.grasping.calculator` spells it, so a reader comparing a report against a YAML file is
#: comparing like with like rather than translating.
_GEOMETRIC = "geometric"


@dataclass(frozen=True, slots=True)
class SceneGrasps:
    """Ranked grasps, and which generator produced them.

    ``generator`` is not decoration. It is the field that stops analytic output being read as
    learned output, which is the one way this class could mislead. Every report and every wire form
    carries it.
    """

    generator: str
    candidates: tuple[SupportFootprintCandidate, ...]
    #: How many points the target cloud holds. An empty result over a sparse cloud and an empty
    #: result over a dense one are different findings, and this count is what separates them.
    points: int

    @property
    def best(self) -> SupportFootprintCandidate | None:
        return self.candidates[0] if self.candidates else None

    def render(self) -> str:
        """The ranked list, for a person. ASCII, no trailing newline, no arguments."""
        head = (f"  {len(self.candidates)} grasp(s) from the {self.generator} generator "
                f"over {self.points} point(s)")
        if not self.candidates:
            # An empty result is an answer. The primitive returns nothing when the cloud is too
            # sparse to reconstruct or admits no legal grasp, and refusing beats proposing a grasp
            # that goes under the support surface. Saying so beats an empty block.
            return "\n".join((
                head,
                "  none. The cloud was too sparse to reconstruct, or every candidate would have "
                "gone under the support.",
            ))
        lines = [head, f"  {'#':<3}{'score':<8}{'width':<9}{'clearance':<11}position (mm)"]
        for index, candidate in enumerate(self.candidates):
            x, y, z = (float(v) for v in candidate.position_mm[:3])
            lines.append(
                f"  {index:<3}{candidate.score:<8.3f}{candidate.grip_width_mm:<9.1f}"
                f"{candidate.clearance_mm:<11.1f}({x:.0f}, {y:.0f}, {z:.0f})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, `json.dumps`-safe with no custom encoder.

        The numpy arrays on each candidate become lists of floats here.
        `dataclasses.asdict` would hand a consumer an `ndarray`, which fails on
        the first `json.dumps`.
        """
        return {
            "generator": self.generator,
            "points": self.points,
            "count": len(self.candidates),
            "candidates": [
                {
                    "score": float(c.score),
                    "position_mm": [float(v) for v in c.position_mm],
                    "approach": [float(v) for v in c.approach],
                    "closing_axis": [float(v) for v in c.closing_axis],
                    "grip_width_mm": float(c.grip_width_mm),
                    "contact_angle_rad": float(c.contact_angle_rad),
                    "clearance_mm": float(c.clearance_mm),
                }
                for c in self.candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class Scene:
    """A segmented target cloud on a support surface, in BASE millimetres.

        import numpy as np
        from src.robot.grasping.scene import Scene

        grasps = Scene.from_cloud(cloud_base_mm, support_height_mm=0.0).grasps()
        print(grasps.render())

    BASE frame, millimetres, and the class does not guess. A camera-frame cloud produces confident
    nonsense here rather than an error, because the numbers are the right shape and the wrong
    origin.
    """

    target_points_base_mm: np.ndarray
    #: Height of the surface the object rests on, in BASE millimetres. The support is what makes a
    #: footprint grasp decidable, so this field has no default even though the primitive defaults
    #: it to 0.0. The caller states it.
    support_height_mm: float
    #: Observed geometry, the neighbours' depth points. Dilated to cover the gaps between samples.
    obstacle_points_base_mm: np.ndarray | None = None
    #: Declared geometry, today the container walls. Not dilated, because it is already exact and
    #: dilating it would forbid a band of the bin a gripper can legally reach.
    rigid_obstacle_points_base_mm: np.ndarray | None = None
    jaw: SupportFootprintJaw | None = None

    @classmethod
    def from_cloud(
        cls,
        target_points_base_mm: "np.ndarray | Sequence[Sequence[float]]",
        *,
        support_height_mm: float,
        obstacle_points_base_mm: "np.ndarray | Sequence[Sequence[float]] | None" = None,
        rigid_obstacle_points_base_mm: "np.ndarray | Sequence[Sequence[float]] | None" = None,
        jaw: SupportFootprintJaw | None = None,
    ) -> "Scene":
        """A cloud and a support height. No robot, no cell, no configuration.

        This door is the analytic generator and says so in its result. There is no config here, so
        there is no `grasping.calculator` selector to honour and nothing is being ignored. A caller
        who has a config and wants it honoured uses :meth:`from_robot_config` instead, and a caller
        who mixes them up finds out from `SceneGrasps.generator` rather than from a number that
        looks plausible.
        """
        return cls(
            target_points_base_mm=np.asarray(target_points_base_mm, dtype=float),
            support_height_mm=float(support_height_mm),
            obstacle_points_base_mm=(None if obstacle_points_base_mm is None
                                     else np.asarray(obstacle_points_base_mm, dtype=float)),
            rigid_obstacle_points_base_mm=(None if rigid_obstacle_points_base_mm is None
                                           else np.asarray(rigid_obstacle_points_base_mm,
                                                           dtype=float)),
            jaw=jaw,
        )

    @classmethod
    def from_robot_config(
        cls,
        robot_config: "RobotConfig",
        target_points_base_mm: "np.ndarray | Sequence[Sequence[float]]",
        *,
        obstacle_points_base_mm: "np.ndarray | Sequence[Sequence[float]] | None" = None,
        rigid_obstacle_points_base_mm: "np.ndarray | Sequence[Sequence[float]] | None" = None,
    ) -> "Scene":
        """The same scene, with the support height and the jaw taken from a cell's configuration.

        The one-builder rule: this resolves config into arguments and calls :meth:`from_cloud`. It
        is not a second construction path, so the two doors cannot drift.

        It does not yet select the generator. `grasping.calculator: deep` is honoured by
        `build_calculator`, which needs a camera matrix and an artifact and belongs to a cell rather
        than to a bare cloud. Until that is wired here, this door resolves geometry from config and
        the generator stays geometric, which `SceneGrasps.generator` states on every result.
        """
        support = robot_config.grasping.support
        return cls.from_cloud(
            target_points_base_mm,
            support_height_mm=float(support.height_mm),
            obstacle_points_base_mm=obstacle_points_base_mm,
            rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm,
            jaw=SupportFootprintJaw.from_model(
                aperture_mm=float(robot_config.gripper.max_width_mm),
                min_width_mm=float(robot_config.gripper.min_width_mm),
            ),
        )

    def grasps(
        self,
        *,
        max_candidates: "Maybe[int]" = UNSET,
        palm_aware: "Maybe[bool]" = UNSET,
    ) -> SceneGrasps:
        """Ranked grasps in BASE millimetres, best first.

        An empty result is an answer, not a failure: the primitive returns nothing when the cloud is
        too sparse to reconstruct or admits no legal grasp, and refusing beats proposing a grasp that
        goes under the support surface.

        Neither knob is defaulted here. `generate_support_footprint_grasps` declares
        ``max_candidates=12`` and ``palm_aware=False``; repeating those numbers in this signature
        would be a second declaration of one fact, and the drift would be invisible because both
        sides are valid values of the right type. An omitted knob is not forwarded at all.
        """
        chosen_options: dict[str, Any] = {}
        if chosen(max_candidates):
            chosen_options["max_candidates"] = max_candidates
        if chosen(palm_aware):
            chosen_options["palm_aware"] = palm_aware
        candidates = generate_support_footprint_grasps(
            self.target_points_base_mm,
            support_height_mm=self.support_height_mm,
            jaw=self.jaw,
            obstacle_points_base_mm=self.obstacle_points_base_mm,
            rigid_obstacle_points_base_mm=self.rigid_obstacle_points_base_mm,
            **chosen_options,
        )
        return SceneGrasps(
            generator=_GEOMETRIC,
            candidates=tuple(candidates),
            points=int(self.target_points_base_mm.shape[0])
            if self.target_points_base_mm.ndim == 2 else 0,
        )
