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
    Scene.from_robot_config(...)  a config exists and supplies the support, the jaw, the floor
                                  and the inflation, each built as the cell's pick path builds it.
                                  The generator selector is not read here, so the result still
                                  says ``generator="geometric"``.

The result names its generator either way, and that is what makes the first door safe: a silent
substitution is only dangerous while it is silent, and `SceneGrasps.generator` is on every report
and in every `to_dict`. A number filed under the wrong generator is how a stack gets judged on work
it did not do.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from src.contracts import UNSET, Maybe, chosen

from src.robot.grasping.collision import resolve_support_plane
from src.robot.grasping.generation.support_footprint import (
    DEFAULT_MAX_CANDIDATES,
    SupportFootprintCandidate,
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)
from src.robot.grasping.geometry.closing_axis import (
    CANDIDATES_THE_AXIS_CHOOSES_AMONG,
    ClosingAxis,
    ClosingAxisLike,
    closing_along,
    closing_axis_of,
    closing_axis_said,
    natural_closing_axis_of,
    same_closing_axis,
    turned_nearer,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["SUPPORT_READ_ERROR_MM", "Scene", "SceneGrasps"]

#: What produced a set of candidates. Spelled the way the config key
#: `robot.grasping.calculator` spells it, so a reader comparing a report against a YAML file is
#: comparing like with like rather than translating.
_GEOMETRIC = "geometric"

#: How far a single view's target cloud has to reach along the support normal before its lowest point
#: may raise the support, millimetres. The pick loop's ``_MIN_CLOUD_EXTENT_FOR_SUPPORT_MM``, restated
#: rather than imported because importing the pick loop here would put it on the import path of every
#: bare-cloud caller and of the locator; ``tests/test_a_scene_plans_the_trees_hand.py`` holds the two
#: equal. Below it the cloud is one face seen head-on, its lowest point is that face, and the plane
#: would land on top of the object.
_MIN_CLOUD_EXTENT_FOR_SUPPORT_MM = 25.0

#: How far above the support a target point still counts as the support on a real cell, millimetres:
#: the floor :meth:`Scene.from_robot_config` plans with.
#:
#: A real table reads as high as the camera's hand-eye and depth error, and a table point counted as
#: part widens the footprint the jaw has to span. Five, the figure the shipped
#: ``safety.planning_world.perceived.plane_clearance_mm`` gives the bench for the same errors, and a
#: choice rather than a measurement; above the stage's own ``DEFAULT_FLOOR_MARGIN_MM``, which was
#: measured in a simulator whose calibration is exact. It is not that key: a cell that sinks its slab
#: below the bench raises the key by the sink (robot.yaml), and read as a floor over the real table a
#: sink of 50 mm left a 40 mm cube with no grasp (review of 2026-09-23). A cell path that builds its
#: own calculator passes this number too.
SUPPORT_READ_ERROR_MM = 5.0

#: How many of the part's measured points must reach down to the height :attr:`Scene.part_bottom_mm` lowers the part's
#: bottom to: that height is the 20th lowest point, not the lowest. One stray reading under the table (a flying pixel at
#: the foot, multipath on a shiny bench) would otherwise lower the bottom as far as it read low, and the set-down would
#: drop the part from that far over the target. Fewer points than this lower nothing. The count a target's top is read
#: from (``locator._MIN_TOP_POINTS``) for the same reason; a choice, not a measurement: a part's foot seen by a D415 at
#: working range carries hundreds of points.
_PART_BOTTOM_POINTS = 20


def _has_seen_the_support(target: np.ndarray, normal: Sequence[float]) -> bool:
    """Whether a target cloud reaches far enough along the support normal to have seen what it stands on."""
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] == 0:
        return False
    unit = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(unit))
    if length < 1e-12:
        return False
    finite = target[np.isfinite(target).all(axis=1)]
    if finite.shape[0] == 0:
        return False
    along = finite @ (unit / length)
    return float(along.max() - along.min()) >= _MIN_CLOUD_EXTENT_FOR_SUPPORT_MM


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
    #: What the closing axis the caller asked for (``Scene.grasps(closing_axis=...)``) left out of what the generator
    #: found, and why, in one sentence; ``""`` when no axis was asked for or it left nothing out.
    withheld: str = ""

    @property
    def best(self) -> SupportFootprintCandidate | None:
        return self.candidates[0] if self.candidates else None

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The ranked list, for a person. ASCII, no trailing newline, no arguments."""
        head = (f"  {len(self.candidates)} grasp(s) from the {self.generator} generator "
                f"over {self.points} point(s)")
        if not self.candidates and self.withheld:
            # The generator found grasps and none closes along the axis asked for: that is the answer, not the cloud.
            return "\n".join((head, f"  none. {self.withheld}"))
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
        if self.withheld:
            lines.insert(1, f"  {self.withheld}")
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
        the first `json.dumps`. ``withheld`` is there only where a closing axis left something out.
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
            **({"withheld": self.withheld} if self.withheld else {}),
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
    #: How far above the support a target point still counts as the support, millimetres. Unset leaves
    #: the primitive's own floor, ``DEFAULT_FLOOR_MARGIN_MM``.
    floor_margin_mm: "Maybe[float]" = UNSET
    #: How far every footprint face is pushed outward, millimetres. Unset leaves the primitive's own.
    inflate_mm: "Maybe[float]" = UNSET
    #: What :attr:`declared_support_height_mm` reads: set by :meth:`from_robot_config`, ``None`` where the caller's
    #: ``support_height_mm`` is the only support stated.
    _declared_support_mm: "float | None" = field(default=None, repr=False)
    #: Whether the target cloud is what two or more looks of a wrist camera measured of the part, fused, so that
    #: :attr:`part_bottom_mm` may lower the part's bottom to where they measured it: set by ``Located.scene`` for object
    #: 0 of a look around, ``False`` for one view.
    _bottom_from_looks: bool = field(default=False, repr=False)
    #: The closing axis a look around judged this part's grasp along (``Locator.look_around(closing_axis=...)``, ``None``
    #: for the best grasp of any), set by ``Located.scene`` for object 0 of a look around that judged it: what
    #: :meth:`grasps` takes when asked for none, and the only axis it takes, so the grasp gripped is the grasp judged.
    #: ``UNSET`` for a scene no look around judged, which takes any axis it is asked for.
    _judged_along: "Maybe[ClosingAxis | None]" = field(default=UNSET, repr=False)
    #: How the cell's hand and camera naturally stand (``robot.natural_closing_axis``), set by :meth:`from_robot_config`:
    #: where no closing axis is asked for, :meth:`grasps` turns every grasp the way round nearer it. ``None``: none is.
    _natural: "ClosingAxis | None" = field(default=None, repr=False)

    @property
    def declared_support_height_mm(self) -> float:
        """The support the cell DECLARES, in BASE millimetres: a lower bound on what the part stands on.

        :attr:`support_height_mm` is what the grasps are planned on. :meth:`from_robot_config` resolves it from the
        declared height (the container floor where ``grasping.support.container.floor_height_mm`` is given, the table
        ``grasping.support.height_mm`` otherwise) raised to the target's lowest SEEN point, and never lowers it. That
        point is the part's base only where the camera saw down to it: an occluder in front, a mask that stops short,
        or a tilted view that misses the foot of the near face leave it above the base by as much as they hid. So the
        planned support is an UPPER bound on the part's base, and this is the LOWER one: no part stands below the
        surface it is declared on.

        A held part's hang below the grasp is measured from here (``Located.set_down(i, grasp=, part_bottom_mm=
        scene.declared_support_height_mm)``), so what the camera missed of the part's foot goes to more air when it is
        set down, never into what it is set down on. The cost: a part taken off a raised block, or off another part,
        hangs lower than that and drops the block's height further. A caller that knows where the part stood passes
        that instead.

        From a bare cloud (:meth:`from_cloud`) it is the ``support_height_mm`` the caller stated. Read-only; along the
        support normal, as :attr:`support_height_mm` is, which on a level table is BASE Z.
        """
        return float(self.support_height_mm if self._declared_support_mm is None else self._declared_support_mm)

    @property
    def part_bottom_mm(self) -> float:
        """Where the part's bottom stood, as the LOWER bound a set-down measures its hang from, BASE Z millimetres.

        :attr:`declared_support_height_mm`, lowered to where two or more looks of a wrist camera measured the part's
        foot, fused (a ``Located.scene`` of a look around's object 0), where they measured it BELOW the declared
        support, and never raised. ``Located.set_down(i, grasp=, part_bottom_mm=scene.part_bottom_mm)`` hangs the part
        from here.

        Why only lowered (the owner, 2026-09-29, addendum 7.6): the part's bottom from all the looks may be used only as
        a lower bound, and only where it reads at or below the declared support plus its read error
        (:data:`SUPPORT_READ_ERROR_MM`); elsewhere the declared value stands. A bottom read more than the read error
        above the support is a foot the looks did not see, or a raised block, and no view tells the two apart. A
        bottom read within the read error above it is the declared support read with that error: it says the part
        stands on it, and is no lower bound on where, since a camera never sees below a part's base; taken, it would
        shorten the hang by up to the read error and eat the air the part is set down with. A bottom read below the
        declared support says the part stood lower than the cell declares (a table or a container floor declared a
        few millimetres high), which the declared value, no lower bound there, would press into the target by as much:
        the hang grows to where the looks measured the foot, and the part lands with more air, never less. That height
        is the :data:`_PART_BOTTOM_POINTS`-th lowest point of the cloud, not the lowest, so a stray reading under the
        table drops nothing; a point read low only adds air. So every error still goes toward more air, and nothing
        is pressed into what the part is set down on.

        One view keeps the declared support as before: a fixed camera's (a cell with no wrist camera sets down as it
        did), a wrist camera's that stopped at its first look, and a scene built from a cloud alone, which says
        nothing of looks. The measurement is the extra looks' to give. From a bare cloud the declared support is the
        one the caller stated.
        """
        declared = self.declared_support_height_mm
        if not self._bottom_from_looks:
            return declared
        points = np.asarray(self.target_points_base_mm, dtype=float).reshape(-1, 3)
        heights = points[np.isfinite(points).all(axis=1), 2]
        if heights.size < _PART_BOTTOM_POINTS:
            return declared
        foot = float(np.partition(heights, _PART_BOTTOM_POINTS - 1)[_PART_BOTTOM_POINTS - 1])
        return foot if foot < declared else declared

    @classmethod
    def from_cloud(
        cls,
        target_points_base_mm: "np.ndarray | Sequence[Sequence[float]]",
        *,
        support_height_mm: float,
        obstacle_points_base_mm: "np.ndarray | Sequence[Sequence[float]] | None" = None,
        rigid_obstacle_points_base_mm: "np.ndarray | Sequence[Sequence[float]] | None" = None,
        jaw: SupportFootprintJaw | None = None,
        floor_margin_mm: "Maybe[float]" = UNSET,
        inflate_mm: "Maybe[float]" = UNSET,
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
            floor_margin_mm=(float(floor_margin_mm) if chosen(floor_margin_mm) else UNSET),
            inflate_mm=(float(inflate_mm) if chosen(inflate_mm) else UNSET),
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
        """The same scene, with the support, the jaw, the floor and the inflation taken from a cell's
        configuration, each the way the cell's pick path takes it.

        The one-builder rule: this resolves config into arguments and calls :meth:`from_cloud`. It
        is not a second construction path, so the two doors cannot drift.

        * The jaw is :meth:`SupportFootprintJaw.from_robot_config`: the fingers of
          ``grasping.gripper_geometry``, the stroke of ``robot.gripper`` and the clearance of
          ``grasping.support.min_clearance_mm``. Until 2026-09-23 only the stroke was read here, and a
          Hand-E cell planned 2F-85 fingers; see that method.
        * The support is ``resolve_support_plane`` over the declared height (or the container floor),
          raised to the target's own lowest point when the cloud reaches 25 mm along the support
          normal and ``refine_from_target`` is on. That is the pick loop's rule, so a part standing on
          another part is planned on what it stands on. It never lowers the declared height, which the scene
          keeps as :attr:`declared_support_height_mm`, the lower bound a set-down measures a part's hang from.
        * The floor is :data:`SUPPORT_READ_ERROR_MM` over the support, so a table that reads a few
          millimetres high through hand-eye or depth error is the table here too, and not the start of
          the part. Not ``safety.planning_world.perceived.plane_clearance_mm``: that is measured from
          the planner's slab, which a cell may sink below the table, and raised by the sink.
        * The inflation is ``grasping.geometry.inflate_mm``, which the cell hands its calculator.
        * The natural orientation is ``robot.natural_closing_axis``: where no closing axis is asked for, :meth:`grasps`
          turns every grasp the way round nearer it, as the cell's pick loop turns its own.

        It does not yet select the generator. `grasping.calculator: deep` is honoured by
        `build_calculator`, which needs a camera matrix and an artifact and belongs to a cell rather
        than to a bare cloud. Until that is wired here, this door resolves geometry from config and
        the generator stays geometric, which `SceneGrasps.generator` states on every result.

        A hand that is not a parallel jaw is refused with ``ValueError``.
        """
        support = robot_config.grasping.support
        target = np.asarray(target_points_base_mm, dtype=float)
        resolution = resolve_support_plane(
            declared_height_mm=float(support.height_mm),
            container_floor_mm=support.container.floor_height_mm,
            normal=support.normal,
            target_clouds_base_mm=([target] if _has_seen_the_support(target, support.normal) else None),
            refine_from_target=bool(support.refine_from_target),
        )
        scene = cls.from_cloud(
            target,
            support_height_mm=resolution.height_mm,
            obstacle_points_base_mm=obstacle_points_base_mm,
            rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm,
            jaw=SupportFootprintJaw.from_robot_config(robot_config),
            floor_margin_mm=SUPPORT_READ_ERROR_MM,
            inflate_mm=float(robot_config.grasping.geometry.inflate_mm),
        )
        return replace(scene, _declared_support_mm=float(resolution.declared_mm),
                       _natural=natural_closing_axis_of(robot_config))

    def grasps(
        self,
        *,
        max_candidates: "Maybe[int]" = UNSET,
        palm_aware: "Maybe[bool]" = UNSET,
        closing_axis: "Maybe[ClosingAxisLike]" = UNSET,
    ) -> SceneGrasps:
        """Ranked grasps in BASE millimetres, best first.

        An empty result is an answer, not a failure: the primitive returns nothing when the cloud is
        too sparse to reconstruct or admits no legal grasp, and refusing beats proposing a grasp that
        goes under the support surface.

        Neither knob is defaulted here. `generate_support_footprint_grasps` declares
        ``max_candidates=12`` and ``palm_aware=False``; repeating those numbers in this signature
        would be a second declaration of one fact, and the drift would be invisible because both
        sides are valid values of the right type. An omitted knob is not forwarded at all.

        ``closing_axis`` takes only the grasps that close along the axis named (the owner's "choose, don't twist",
        2026-09-30): within ``CLOSING_AXIS_TOLERANCE_DEG`` (30 degrees) of it either way round, each turned half a turn
        about its approach where that puts its closing axis the named way round. A name ``Pose.tool_down`` takes
        (``"-y"`` on the owner's cell) or an orientation, a quaternion or a BASE ``Pose``, whose tool +X is the heading
        (``src.robot.grasping.geometry.closing_axis``). They are chosen among the best
        ``CANDIDATES_THE_AXIS_CHOOSES_AMONG`` (36) the generator finds, as the cell's pick loop chooses (among
        ``max_candidates`` where that is more), and the best ``max_candidates`` of them that close along it come back.
        What it left out is said in :attr:`SceneGrasps.withheld`, and a result it left empty says so.

        Asked for no axis, a scene from :meth:`from_robot_config` whose cell names how its hand and camera naturally
        stand (``robot.natural_closing_axis``) turns every grasp, of its two equivalent wrist turns (half a turn about its
        approach: the same two contact faces, the jaws swapped), to the one whose closing axis lies nearer that
        direction at its place. None is left out or tilted.

        The scene of a part a look around judged (``Located.scene(0, ...)`` after ``Locator.look_around``) takes the axis
        the looks judged its grasp along when asked for none, and refuses (``ValueError``) any other, the best grasp of
        any axis included, since its grasp is one the looks did not judge: its early stop, its jaw faces.
        """
        wanted = self._closing_axis_asked(closing_axis)
        chosen_options: dict[str, Any] = {}
        if chosen(max_candidates):
            chosen_options["max_candidates"] = max_candidates
        if chosen(palm_aware):
            chosen_options["palm_aware"] = palm_aware
        if chosen(self.floor_margin_mm):
            chosen_options["floor_margin_mm"] = self.floor_margin_mm
        if chosen(self.inflate_mm):
            chosen_options["inflate_mm"] = self.inflate_mm
        cap = int(chosen_options.get("max_candidates", DEFAULT_MAX_CANDIDATES))
        if wanted is not None:
            chosen_options["max_candidates"] = max(CANDIDATES_THE_AXIS_CHOOSES_AMONG, cap)
        candidates = generate_support_footprint_grasps(
            self.target_points_base_mm,
            support_height_mm=self.support_height_mm,
            jaw=self.jaw,
            obstacle_points_base_mm=self.obstacle_points_base_mm,
            rigid_obstacle_points_base_mm=self.rigid_obstacle_points_base_mm,
            **chosen_options,
        )
        withheld = ""
        if wanted is not None:
            candidates, withheld = _closing_along(candidates, wanted, cap)
        elif self._natural is not None:
            candidates = _closing_toward(candidates, self._natural)
        return SceneGrasps(
            generator=_GEOMETRIC,
            candidates=tuple(candidates),
            points=int(self.target_points_base_mm.shape[0])
            if self.target_points_base_mm.ndim == 2 else 0,
            withheld=withheld,
        )

    def _closing_axis_asked(self, closing_axis: "Maybe[ClosingAxisLike]") -> "ClosingAxis | None":
        """The closing axis :meth:`grasps` chooses along: the one asked for, else the one a look around judged this
        part's grasp along (:attr:`_judged_along`); ``None`` for any. Refused where a look around judged the grasp along
        another axis, or along any, since the grasp chosen would be one the looks did not judge."""
        asked = closing_axis_of(closing_axis) if chosen(closing_axis) else None
        judged = self._judged_along
        if chosen(judged):
            if asked is None:
                return judged
            points = np.asarray(self.target_points_base_mm, dtype=float).reshape(-1, 3)
            x_mm, y_mm = (float(value) for value in np.median(points[:, :2], axis=0)) if len(points) else (0.0, 0.0)
            if judged is None or not same_closing_axis(asked, judged, x_mm, y_mm):
                along = "of any closing axis" if judged is None else f"closing along {judged}"
                raise ValueError(
                    f"the look around judged this part's grasp {along}, and a grasp closing along {asked} is one it "
                    "did not judge (its early stop, its jaw faces): name that axis to look_around(..., closing_axis="
                    "...), or ask the scene for none and take the grasp the looks judged")
        return asked


def _closing_toward(
    candidates: "Sequence[SupportFootprintCandidate]", natural: ClosingAxis,
) -> "list[SupportFootprintCandidate]":
    """Every candidate, in its order, turned half a turn about its approach where that puts its closing axis nearer the
    natural direction at its place (``turned_nearer``): the same two contact faces, the jaws swapped. None is left out."""
    return [replace(candidate, closing_axis=-candidate.closing_axis)
            if turned_nearer(natural, candidate.position_mm, candidate.closing_axis) else candidate
            for candidate in candidates]


def _closing_along(
    candidates: "Sequence[SupportFootprintCandidate]", wanted: ClosingAxis, cap: int,
) -> "tuple[list[SupportFootprintCandidate], str]":
    """The best ``cap`` of ``candidates`` that close along ``wanted``, each signed along it (``closing_along``), and what
    the axis left out, said (``closing_axis_said``)."""
    kept: list[SupportFootprintCandidate] = []
    whys: list[str] = []
    for candidate in candidates:
        signed, why = closing_along(wanted, candidate.position_mm, candidate.closing_axis)
        if signed is None:
            whys.append(why)
        elif float(np.dot(signed, candidate.closing_axis)) < 0.0:
            kept.append(replace(candidate, closing_axis=signed))
        else:
            kept.append(candidate)
    return kept[:max(cap, 0)], closing_axis_said(wanted, whys, len(candidates))
