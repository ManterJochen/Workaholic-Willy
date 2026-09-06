"""Native dense-clutter pick runner for Isaac Sim.

``run_fused_pick.run_clutter_gate`` localizes overhead and then picks with the wrist camera, which leaves
the dense grasp features inert. This runner drives a native dense pick instead: one overhead camera sees
every object in a single frame (:class:`MultiObjectGroundTruthPerceptionSource`), the orchestrator runs
``GraspMode.DENSE_CLUTTER``, and a hard label-target (``service.set_target_label``) makes it grasp the
prompted object while the others stay neighbour clutter. That multi-object frame is what the dense
features need: the DENSE_CLUTTER sampler auto-enables on the neighbours, and the corridor-risk producer
and the swept-volume approach validator read the same frame. The gate verifies that the right object
lifted and that no distractor was disturbed.

Builds on the overhead path (``StaticCameraToBaseResolver``, ground-truth depth plus grasp-lift).
On-box only: needs Isaac. Run with Isaac's bundled python:

    <isaac-sim>\\python.bat -m src.willy_sim.run_dense_pick --runs 10 --prompt "the red cube"
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.willy_sim.harness.depth_noise import DepthNoiseConfig
    from src.robot.grasping.geometry.filters import CloudOutlierConfig

from src.config.schema.robot import SimObjectConfig
from src.utility.log_cfg import create_logger
from src.willy_sim.constants import SAFETY_GUARDS_LOG_FILE, WILLY_SIM_LOG_DIR
from src.willy_sim.harness.bootstrap import bootstrap_sim_cell
from src.willy_sim.harness.gate import GateResult, gate_passed
from src.willy_sim.config import require_robot
from src.willy_sim.harness.ik_service import ArmBackedIKService
from src.willy_sim.harness.instrumentation import write_pick_artifacts, write_run_result
from src.willy_sim.harness.modes import mode_service_kwargs, resolve_demo_mode
from src.willy_sim.perception import MultiObjectGroundTruthPerceptionSource
from src.willy_sim.harness.randomizer import DomainRandomizer, RandomizationConfig
from src.willy_sim.run_fused_pick import _clutter_specs, _select_by_prompt
from src.willy_sim.harness.env import RunnerEnv
from src.willy_sim.scene import SceneAppearance

#: Not a log for this runner: the pick narrative stays on stdout, where it is attributable to one
#: run. It belongs to ``wire_safety_guards`` below, which every runner's ``build_service`` calls and
#: which happens to live in this file. The guard a run asks for and the guard it gets are different
#: facts, and neither the request nor a mismatch survives the shell, so the request is written here.
_GUARD_LOG = create_logger("SimSafetyGuards", log_file=SAFETY_GUARDS_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)


def _blocking_specs(blocker_offset_mm: float = 35.0) -> list:
    """The three-cube clutter plus a blocker cube in the red target's +X approach column.

    Red sits at -Y, so its top-down close yaws to base +X: the descent and the jaw sweep run through the
    +X column beside it. A blocker there blocks red's approach, so the swept-volume validator
    (``enable_g12``) refuses or avoids it, and it sits in red's grasp corridor, so the corridor-risk
    producer (``enable_g4``) measures real blockage. Green and blue close along -X and keep their own
    +/-X columns clear, so only the red pick is affected: prompting red is the blocked case and prompting
    green the clear one.
    """
    specs = _clutter_specs()  # red(-95) green(+5) blue(+105) at x=450
    specs.append(
        SimObjectConfig(
            name="orange blocker",
            color=(0.95, 0.55, 0.10),
            position_mm=(450.0 + float(blocker_offset_mm), -95.0, 25.0),
        )
    )
    return specs


def _cylinder_specs() -> list:
    """The three-object clutter with the red target as a cylinder, for a three-finger centric gripper.

    A round body is the natural object for a centric gripper such as the Schunk EZU-35: the three fingers
    wrap it, where a flat cube gives only point and edge contacts. Green and blue stay cubes as neighbour
    clutter.

    This scene does not produce a passing pick. The mount, the drive and the equal-force grip work, but
    the free cylinder rolls: the open wide fingers brush the round body and it rolls away before the
    close. The scene exists to reproduce that. Default-off, reached only via
    ``build_service(cylinder=True)``.
    """
    specs = _clutter_specs()  # red(-95) green(+5) blue(+105) at x=450
    specs[0] = specs[0].model_copy(update={
        "shape": "cylinder",
        # diameter 40 mm (the 3 fingers wrap it; inside the EZU-35 grip range 11-81). Height matches the
        # 30 mm cube so the grasp lands on the centre (z~25). A taller cylinder's grasp rises to the top
        # edge and the gripper rams it: the move times out and the body tunnels through the table.
        "size_mm": (40.0, 40.0, 30.0),
        "position_mm": (450.0, -95.0, 25.0),
        "mass_kg": 0.1,  # modest mass; the round body still rolls on descent contact, so the pick fails
        # here. The equal three-finger grip works through the follower target-sync; the roll does not.
        "static_friction": 2.0, "dynamic_friction": 1.8,  # grip the smooth round side
    })
    return specs


_PILE_SHAPES = ("cube", "cylinder")


def _pile_specs(
    *,
    count: int = 8,
    seed: int = 0,
    center_xy_mm: tuple[float, float] = (450.0, 0.0),
    spread_mm: float = 50.0,
    drop_base_z_mm: float = 42.0,
    drop_step_mm: float = 16.0,
    shapes: str = "mixed",
) -> list:
    """Dense-clutter pile scene: ``count`` varied rigid primitives dropped into a touching heap.

    Mixed boxes and cylinders at varied graspable sizes (32-52 mm, inside the 2F-85 span), each in a
    distinct palette colour so a '<colour> box/cylinder' label lets the ground-truth or vision perception
    name every object. They spawn in a tight cluster over ``center_xy_mm``, staggered in Z
    (``drop_base_z_mm + drop_step_mm*i``), so they fall and settle into a genuinely touching, overlapping
    pile with real occlusion and neighbour contact rather than into well-separated cubes. Seeded numpy
    keeps it deterministic and reproducible off-box. The spread and drop knobs are the pile-density dial.
    Default-off: ``build_service`` calls this only when ``pile=True``, so the plain clutter pick is
    unchanged.
    """
    from src.willy_sim.harness.randomizer import DISTRACTOR_PALETTE

    rng = np.random.default_rng(int(seed))
    cx, cy = float(center_xy_mm[0]), float(center_xy_mm[1])
    palette = list(DISTRACTOR_PALETTE)
    rng.shuffle(palette)
    specs = []
    for i in range(int(count)):
        cname, crgb = palette[i % len(palette)]
        # shapes: "mixed" alternates box and cylinder for varied clutter; "cube" or "cylinder" pins one
        # shape. A cylinder is inherently hard for a parallel jaw because it rolls on descent, so "cube"
        # isolates clutter difficulty from the shape-versus-gripper mismatch.
        shape = shapes if shapes in _PILE_SHAPES else _PILE_SHAPES[i % len(_PILE_SHAPES)]
        foot = float(rng.uniform(32.0, 52.0))  # footprint (box: x extent / cylinder: diameter)
        foot_y = float(rng.uniform(32.0, 52.0)) if shape == "cube" else foot  # cylinder = square footprint
        height = float(rng.uniform(32.0, 52.0))
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = float(rng.uniform(0.0, float(spread_mm)))  # tight cluster: the objects overlap and touch
        px = cx + rad * float(np.cos(ang))
        py = cy + rad * float(np.sin(ang))
        pz = float(drop_base_z_mm) + float(drop_step_mm) * i  # staggered drop: they fall into a pile, not a grid
        label = f"{cname} {'box' if shape == 'cube' else 'cylinder'}"
        specs.append(
            SimObjectConfig(
                name=label, color=crgb, shape=shape,
                size_mm=(foot, foot_y, height),
                position_mm=(px, py, pz),
                mass_kg=0.08, static_friction=1.2, dynamic_friction=1.0,
            )
        )
    return specs


def _make_bin_fixtures(
    *,
    center_xy_mm: tuple[float, float] = (450.0, 0.0),
    half_width_mm: float = 95.0,
    half_width_y_mm: float | None = None,
    height_mm: float = 120.0,
    thickness_mm: float = 12.0,
) -> list:
    """Four axis-aligned bin or tray walls (``FixtureBoxConfig``) enclosing the clutter, in the robot BASE
    frame (roughly world here, base at the world origin).

    This list is the single source for both the physical ``FixedCuboid`` prims in the scene and the
    ``SelfCollisionGuard`` fixtures in safety, so the prim and the guard fixture agree by construction and
    an approach into a wall is self-collision-rejected. Walls rise from the table (z=0) to ``height_mm``.
    ``half_width_mm`` is the bin inner half-extent in x, and in y as well when ``half_width_y_mm`` is
    None, which makes the bin square. Passing a separate ``half_width_y_mm`` authors the real KLT
    rectangle (about 90 narrow by 140 long) so the finger tool's open span (base X) and its thin side
    (base Y) face different walls.
    """
    from src.config.schema.robot.safety_schema import FixtureBoxConfig

    cx, cy = float(center_xy_mm[0]), float(center_xy_mm[1])
    wx = float(half_width_mm)
    wy = float(half_width_y_mm) if half_width_y_mm is not None else wx
    t = float(thickness_mm) / 2.0  # wall half-thickness
    hz = float(height_mm) / 2.0  # wall half-height (centered at z=hz -> base on the table z=0)
    spx = wx + 2.0 * t  # outer half-span (x) so the side walls meet the end walls at the corners
    spy = wy + 2.0 * t  # outer half-span (y)
    return [
        FixtureBoxConfig(name="bin_wall_pos_x", center_mm=(cx + wx + t, cy, hz), half_extents_mm=(t, spy, hz)),
        FixtureBoxConfig(name="bin_wall_neg_x", center_mm=(cx - wx - t, cy, hz), half_extents_mm=(t, spy, hz)),
        FixtureBoxConfig(name="bin_wall_pos_y", center_mm=(cx, cy + wy + t, hz), half_extents_mm=(spx, t, hz)),
        FixtureBoxConfig(name="bin_wall_neg_y", center_mm=(cx, cy - wy - t, hz), half_extents_mm=(spx, t, hz)),
    ]


_YCB_ROOT = "/Isaac/Props/YCB/Axis_Aligned_Physics/"
# The non-physics YCB directory: geometry only, no collider and no rigid body. ``scene/build.py`` authors
# a convexHull collider on these at runtime through ``SimObjectConfig.usd_collision_approximation``, so
# they become graspable rigid bodies beyond the pre-rigged 003-006 under Axis_Aligned_Physics.
_YCB_ROOT_NONPHYS = "/Isaac/Props/YCB/Axis_Aligned/"
# 90 deg about X (WXYZ): the soup can is authored lying with its cylinder axis along Y. Rotating about X
# lifts that axis from Y to vertical Z, so the can settles upright: stable, no roll, symmetric top-down so
# any diameter close is trackable, and it lifts without torque or slip.
_UPRIGHT_X90 = (0.70710678, 0.70710678, 0.0, 0.0)
# 90 deg about Y (WXYZ): a box that settles flat puts its thin ~38 mm axis vertical, so top-down sees the
# wide ~89 mm face, too wide for the 85 mm gripper. Standing it on edge turns the 38 mm axis horizontal,
# graspable top-down across flat faces with no cylinder escape. Trades stability (taller) for graspability.
_EDGE_Y90 = (0.70710678, 0.0, 0.70710678, 0.0)
_YCB_OBJECTS = [
    # (name, usd, position_mm, orientation_wxyz). Spawned low, a small drop, so each settles at or near its
    # spawn orientation without tumbling. The sugar box is the target: on edge it presents a ~38 mm
    # graspable short axis with flat faces and no cylinder escape. It is long, about 175 mm along +/-Y, so
    # the camera nadir sits over it and stays parallax-free while the distractors flank it in X, a
    # different column, where they neither interpenetrate the long box at settle nor block its vertical
    # top-down descent at x=450. The can and the cracker are distractors and are never grasped. The solo
    # env knob keeps only the first object, the box.
    ("sugar box", "004_sugar_box.usd", (450.0, -120.0, 60.0), _EDGE_Y90),
    ("tomato soup can", "005_tomato_soup_can.usd", (580.0, -60.0, 60.0), _UPRIGHT_X90),
    # "red cracker box": GroundingDINO merges objects that share a word, so "sugar box" and "cracker box"
    # come back as one detection with the cracker absorbed and unnamed. A colour adjective disambiguates
    # (003 is the red Cheez-It against the yellow and white sugar box) without the tokenizer trouble of a
    # bare "cracker". Distractor; never grasped.
    ("red cracker box", "003_cracker_box.usd", (320.0, -120.0, 60.0), None),
]

# Generalization scene: the same objects and labels in a different arrangement (target moved, distractors
# swapped sides), with a different table colour and different lighting, both set in ``build_service``. If
# GroundingDINO and SAM2 still detect and name all three and the box still lifts here, the pipeline is not
# overfit to the one tuned scene. The box stays on edge at the camera nadir; the distractors flank in X.
_YCB_OBJECTS_SCENE2 = [
    ("sugar box", "004_sugar_box.usd", (450.0, -100.0, 60.0), _EDGE_Y90),
    ("tomato soup can", "005_tomato_soup_can.usd", (330.0, -40.0, 60.0), _UPRIGHT_X90),
    ("red cracker box", "003_cracker_box.usd", (575.0, -150.0, 60.0), None),
]
# Appearance change against the default gray table at dome 150 and key 300: a warm wood table and a
# dimmer, more keyed light balance. Detection stays robust to it. The grasp is robustified separately by
# the detection-box mask fill in ``MultiObjectVisionPerceptionSource``, which keeps the grasp centred when
# the SAM2 mask of a long box is incomplete.
#
# That fill is off by default: ``DEFAULT_MASK_COMPLETION`` is ``NONE``, because filling costs top-1
# accuracy over a corpus of object views. This scene was tuned with the fill on, and this runner is the
# only on-box caller of that default, so the two policies are compared here, paired, in one session. The
# scene clears its bar without the fill, so nothing is pinned to it. Pass ``--mask-completion`` to re-open
# the comparison.
_SCENE2_TABLE_COLOR = (0.50, 0.37, 0.24)
_SCENE2_DOME_INTENSITY = 105.0
_SCENE2_KEY_INTENSITY = 410.0


# Per-target tuned close width (mm) for known graspable objects. A fixed value is steadier than the
# general adaptive close (grip_w - squeeze), which inherits the vision mask's grip_w noise on a thin
# object. Untuned targets fall to the adaptive close in ``build_service``. The key is the object's spawn
# name, which is also its target label.
_YCB_CLOSE_BY_TARGET = {
    "sugar box": 33.0,    # on-edge short axis about 38 mm
    "pudding box": 33.0,  # on-edge short axis about 38.5 mm, so the same close as the sugar box
    "gelatin box": 25.0,  # on-edge short axis about 30.0 mm, so a tighter close
}

# Collider-authored compact boxes from the non-physics directory: geometry only, with a convexHull
# authored at runtime. Placed on edge (``_EDGE_Y90``, like the sugar box) so the thin short axis is the
# top-down graspable axis: pudding 38.5 mm, gelatin 30.0 mm, both well inside the 85 mm 2F-85 span.
# Selected by ``WILLY_YCB_SCENE`` (pudding|gelatin|extra); unset leaves the demo scene unchanged.
_YCB_EXTRA_BOXES: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    "pudding box": ("008_pudding_box.usd", _EDGE_Y90),
    "gelatin box": ("009_gelatin_box.usd", _EDGE_Y90),
}

# The per-episode cube re-colour palette lives in ``harness/randomizer.py`` as ``DISTRACTOR_PALETTE``.
# ``DomainRandomizer`` owns colour randomization there alongside position, orientation and lighting.

# The failure-taxonomy tokens returned below are the ones ``derive_outcome_class`` in
# ``src/robot/grasping/rl/dataset.py`` recognises.


def _failure_taxonomy_from_sim(outcome: str, pick_outcome: str, sim_lifted: bool) -> str | None:
    """A failure-taxonomy token derived from the sim ground truth, or None on a real success.

    ``derive_outcome_class`` in ``src/robot/grasping/rl/dataset.py`` checks
    ``extra['failure_taxonomy_class']`` first, so stamping this makes the dataset's outcome class reflect
    the real lift (``sim_lifted``) instead of the pipeline's self-reported ``final_outcome``, which reads
    ``'succeeded'`` even for a slip. Returns None when the pick really lifted, leaving the record a
    success. Otherwise it maps to the nearest token: no candidates becomes an empty-air grasp, a collision
    outcome becomes a collision rejection, and grasped but not lifted becomes a slip after grasp.
    """
    if sim_lifted:
        return None
    if outcome == "no_valid_grasp" or pick_outcome in ("rescanned_exhausted", "no_perception"):
        return "empty_air_grasp"
    if "collision" in str(outcome):
        return "collision_rejection"
    return "slip_after_grasp"


def _ycb_extra_specs(scene_sel: str) -> list:
    """``SimObjectConfig`` entries for the collider-authored compact boxes (008 pudding, 009 gelatin).

    ``pudding`` or ``gelatin`` places that box alone at the camera nadir, which is the grasp-gate target.
    ``extra`` places both, pudding at the nadir and gelatin flanking, for a geometry-diverse scene and a
    non-degenerate offline-training dataset. Each spec opts into runtime collider authoring (convexHull)
    and uses the non-physics asset root, standing on edge (``_EDGE_Y90``) so the thin axis is the top-down
    graspable axis.
    """
    nadir = (450.0, -120.0, 60.0)  # camera-nadir target slot (the sugar-box target position)
    flank = (580.0, -60.0, 60.0)   # 2nd-object slot (the soup-can position)
    layout = {
        "pudding": [("pudding box", nadir)],
        "gelatin": [("gelatin box", nadir)],
        "extra": [("pudding box", nadir), ("gelatin box", flank)],
    }[scene_sel]
    return [
        SimObjectConfig(
            name=name, usd_asset_path=_YCB_ROOT_NONPHYS + _YCB_EXTRA_BOXES[name][0],
            position_mm=pos, orientation_wxyz=_YCB_EXTRA_BOXES[name][1],
            usd_collision_approximation="convexHull",
        )
        for (name, pos) in layout
    ]


def _ycb_specs(scene2: bool, env: RunnerEnv) -> list:
    """YCB clutter: real textured YCB objects (referenced USDs) instead of procedural cubes.

    ``scene2`` selects the generalization layout, a different arrangement of the same objects.
    ``env.ycb_scene`` in {pudding, gelatin, extra} selects the collider-authored compact boxes instead;
    the default empty value keeps the demo scene.
    """
    if env.ycb_scene in ("pudding", "gelatin", "extra"):
        return _ycb_extra_specs(env.ycb_scene)
    base = _YCB_OBJECTS_SCENE2 if scene2 else _YCB_OBJECTS
    objects = base[:1] if env.ycb_solo else base  # solo: just the target
    specs = [
        SimObjectConfig(name=n, usd_asset_path=_YCB_ROOT + usd, position_mm=pos, orientation_wxyz=orient)
        for (n, usd, pos, orient) in objects
    ]
    if env.ycb_target_y is not None:  # diagnostic: override the first object's spawn Y (reachability probe)
        x, _, z = specs[0].position_mm
        specs[0] = specs[0].model_copy(update={"position_mm": (float(x), env.ycb_target_y, float(z))})
    return specs


# GSO roster: graspable Aletheia scanned objects (minimum dimension <= 60 mm) spread as a clutter scene.
# The default set spans compact toys and tools. ``WILLY_GSO_ROSTER`` (comma-separated labels) overrides it
# so successive collect runs draw different object sets, which is the object diversity the leakage audit
# needs. Only labels present in the curated GSO catalogue survive; an all-invalid roster falls back to the
# default.
_GSO_DEFAULT_ROSTER = ("toy rhino", "toy dinosaur", "toy lion", "screwdriver")


def _gso_specs() -> list:
    """Clutter scene from the Aletheia GSO catalogue: real scanned objects (toys, tools) as clutter.

    The roster (``WILLY_GSO_ROSTER``, comma-separated labels; default a curated graspable set) picks which
    objects spawn, so successive collect runs vary the ``object_set`` the leakage audit reads. Objects sit
    in a Y-column at x=450, which is top-down reachable; ``--prompt`` selects the target and the rest are
    clutter and occlusion. Real meshes at randomized poses genuinely fail sometimes, which gives the mixed
    per-target outcomes the ranking needs. Requires the one-time on-box USD conversion
    (``python -m src.willy_sim.gso_assets --convert``). The import is lazy and Isaac-free so the module
    stays importable off-box; the call fires only under ``--gso``.
    """
    from src.willy_sim.gso_assets import GSO_BY_LABEL, gso_object_spec

    raw = os.environ.get("WILLY_GSO_ROSTER")
    labels = [s.strip() for s in raw.split(",")] if raw else list(_GSO_DEFAULT_ROSTER)
    labels = [lbl for lbl in labels if lbl in GSO_BY_LABEL] or list(_GSO_DEFAULT_ROSTER)
    n = len(labels)
    step = 70.0  # ~70 mm apart clears the ~35-56 mm footprints; symmetric about y=0 (matches the cube spread)
    ys = [(i - (n - 1) / 2.0) * step for i in range(n)]
    specs = []
    for lbl, y in zip(labels, ys, strict=True):
        part = GSO_BY_LABEL[lbl]
        spec = gso_object_spec(part, position_mm=(450.0, y, 60.0), mass_kg=0.15)
        # A friction grasp, with no rigid-carry attachment here unlike the bin-clearing demo, needs the
        # collider to match the visual mesh. The default boundingCube box is wider than a thin figurine, so
        # the jaw stops on the box, never clamps the object and it slips on lift. A convexHull collider
        # tracks the mesh, so the jaw closes on the real geometry: a good grasp holds and a bad one slips,
        # which is the marginal outcome mix the ranking needs. Clutter and round objects keep boundingCube.
        if part.graspable:
            spec = spec.model_copy(update={
                "usd_collision_approximation": "convexHull",
                "static_friction": 2.0, "dynamic_friction": 1.8,
            })
        specs.append(spec)
    return specs


def _build_g6_recovery(agitate_amplitude_mm: float = 30.0, center_mm=(300.0, 0.0, 400.0),  # noqa: ANN001, ANN202
                       contact_depth_mm: float = 0.0, sweep_offset_mm: float = 0.0):
    """A recovery setup that permits the container-agitate motion on a clutter jam (ALL_COLLIDED).

    Reuses the real recovery machinery: the same ``run_recovery_loop`` and
    ``RecoveryOrchestrator(bypass_strategies)`` the service's ``_run_with_recovery`` uses. Opt-in via
    ``--g6`` only; without it the pick path is unchanged. Agitate passes four gates. First a profile
    allow-list: a copy of the DENSE_CLUTTER profile with "container_agitate" added, because the locked
    profiles omit it and are not mutated. Second a policy allow-list of CONTAINER_AGITATE alone, so the
    dispatcher picks it directly for ALL_COLLIDED. Third a ``FixtureEnvelope`` with a non-zero agitate
    amplitude, which is the envelope clamp and the safety backstop. Fourth the trigger ALL_COLLIDED
    itself, which the calculator produces when ``rejected_collision`` is greater than zero. Returns
    ``(profile, policy, orchestrator)``.
    """
    from dataclasses import replace

    from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
    from src.robot.grasping.recovery.policy import (
        ContainerAgitateStrategy,
        FixtureEnvelope,
        SceneRecoveryAction,
        SceneRecoveryPolicy,
    )
    from src.robot.grasping.recovery.orchestrator import (
        RecoveryDispatcher,
        RecoveryOrchestrator,
    )

    base = _profile_for(GraspMode.DENSE_CLUTTER)
    profile = replace(
        base, recovery_allowed_actions=(*base.recovery_allowed_actions, "container_agitate")
    )
    fixture = FixtureEnvelope(
        # The agitate safety envelope is a bounded box around the recovery start pose. The agitate runs
        # wherever the arm sits after the failed pick, which is the parked pose because the jam refusal
        # commands no motion. The caller centres the envelope on that live TCP so the +/- amplitude shake
        # stays inside it while any out-of-box waypoint is still refused.
        center_mm=(float(center_mm[0]), float(center_mm[1]), float(center_mm[2])),
        half_extents_mm=(250.0, 200.0, 250.0),
        max_agitate_amplitude_mm=float(agitate_amplitude_mm),
        # Greater than zero makes the executor run a contact redistribute: descend, then a directed +X
        # corridor sweep that moves a movable blocker out of the approach column instead of shaking the
        # air. The default 0 leaves the air-shake.
        agitate_contact_depth_mm=float(contact_depth_mm),
        agitate_sweep_offset_mm=float(sweep_offset_mm),
    )
    policy = SceneRecoveryPolicy(
        enabled=True,
        allowed_actions=(SceneRecoveryAction.CONTAINER_AGITATE,),
        max_recovery_actions=2,
        fixture=fixture,
        apply_modes=("dense_clutter",),
    )
    # Use the real ``ContainerAgitateStrategy``, not ``bypass_strategies``: the strategy reads the
    # fixture's amplitude into the plan, while a bypass plan carries amplitude 0 and the executor then
    # refuses with "agitate_disabled".
    orchestrator = RecoveryOrchestrator(
        dispatcher=RecoveryDispatcher(),
        strategies={SceneRecoveryAction.CONTAINER_AGITATE: ContainerAgitateStrategy()},
        bypass_strategies=False,
    )
    return profile, policy, orchestrator


class _G6RecoveryAdapter:
    """A ``run_recovery_loop`` view over an ``AutonomousGraspReport``, mirroring the service's adapter.

    It adds one mapping: an all-sweeps-blocked refusal from the approach validator
    (``pick_report.outcome=approach_path_blocked``) is a collision jam, so it is surfaced as
    :attr:`GraspFailureReason.ALL_COLLIDED`, the jam signal the recovery dispatcher routes to
    container-agitate. The calculator's own ``rejected_collision`` fires only for a neighbour inside the
    closed-jaw collision box, that is for touching cubes, so the sweep block is the deterministic jam
    trigger.
    """

    __slots__ = ("report", "outcome", "failure_reasons")

    def __init__(self, report) -> None:  # noqa: ANN001
        from src.robot.grasping.types.feedback import GraspFailureReason

        self.report = report
        self.outcome = report.outcome
        psr = getattr(report, "pick_report", None)
        reasons = tuple(psr.attempts[-1].reasons) if (psr is not None and psr.attempts) else ()
        po = getattr(getattr(psr, "outcome", None), "value", None)
        if po == "approach_path_blocked" and GraspFailureReason.ALL_COLLIDED not in reasons:
            reasons = (*reasons, GraspFailureReason.ALL_COLLIDED)
        self.failure_reasons = reasons


def wire_safety_guards(
    arm: object,
    *,
    natural_aim: bool = True,
    continuous_guard: bool = False,
    continuous_guard_margin_mm: float = 8.0,
    fixtures: Sequence[object] = (),
    kinematics_model: str = "ur5e",
    kinematics_base_yaw_deg: float = 180.0,
    collision_mesh_variant: str | None = None,
) -> None:
    """Owner-set the safety guards onto a sim arm; every runner's build calls this.

    ``natural_aim`` makes IK prefer the natural, non-self-colliding branch and defaults to on. It is
    paired with the Coal mesh self-collision standard (``robot.yaml`` ``self_collision.backend: fcl``):
    without natural-aim the default IK branch self-penetrates ``forearm|gripper`` at 0 mm, which Isaac
    physics does not see because its self-collision is off but which is real on hardware, and the Coal
    guard correctly rejects it. With natural-aim the reach-down grasps clear. Coal catches and natural-aim
    avoids, so the two ship together. ``continuous_guard``, the per-waypoint collision-avoidance monitor,
    stays opt-in. The mesh backend needs Coal or python-fcl, with ``WILLY_COAL_PREFIX`` set on Windows; if
    it is unavailable the guard logs a warning. This is software collision avoidance, not certified
    functional safety.
    """
    # Unconditional, one line per build: what this run asked for. The request has to be on file for the
    # "requested but off" warning below to mean anything.
    _GUARD_LOG.info(
        "guards requested: natural_aim=%s continuous_guard=%s (margin %.1f mm, %d fixture(s))",
        natural_aim, continuous_guard, continuous_guard_margin_mm, len(fixtures),
    )
    if natural_aim:
        arm._natural_aim_seed = True  # type: ignore[attr-defined]
    if not continuous_guard:
        return
    from src.robot.safety.continuous_monitor import (
        ContinuousCollisionMonitor,
        ContinuousGuardProfile,
    )

    _fix = list(fixtures) if fixtures else []
    # Pass the model, yaw and gripper variant the one-shot guard was built with. Hard-coded values would
    # watch a --gripper-mount run with the baked 2F-85 meshes while the guard used the mounted gripper's,
    # and would watch a ur3e cell with ur5e link lengths.
    mon = ContinuousCollisionMonitor.from_model(
        kinematics_model,
        kinematics_base_yaw_deg,
        _fix,
        ContinuousGuardProfile(enabled=True, margin_mm=continuous_guard_margin_mm),
        variant=collision_mesh_variant,
    )
    if mon is None:  # requested but no mesh backend (set WILLY_COAL_PREFIX): guard off, warn
        # The run continues unguarded with everything else identical, so this line is the only
        # difference between a guarded and an unguarded measurement and must outlive the console.
        _GUARD_LOG.warning(
            "continuous guard REQUESTED but no mesh backend (set WILLY_COAL_PREFIX?) -> guard OFF; "
            "this run proceeds unguarded",
        )
        print("[continuous-guard] WARN: mesh backend unavailable (set WILLY_COAL_PREFIX?) -> guard OFF",
              flush=True)
        return
    arm._continuous_monitor = mon  # type: ignore[attr-defined]
    _GUARD_LOG.info(
        "continuous guard ON: engine=%s margin=%.1f mm fixtures=%d (bucket-1 software avoidance, "
        "NOT certified safety)", mon.engine, continuous_guard_margin_mm, len(_fix),
    )
    print(f"[continuous-guard] ON engine={mon.engine} margin={continuous_guard_margin_mm}mm "
          f"fixtures={len(_fix)} (bucket-1 software avoidance, NOT certified safety)", flush=True)


def build_service(
    *,
    prompt: str = "the red cube",
    headless: bool = True,
    data_dir: str | None = None,
    mode: str = "dense_clutter",
    # Feature flags; all default off, which leaves the plain dense pick unchanged:
    ycb: bool = False,                # real YCB objects (referenced USDs) instead of procedural cubes
    gso: bool = False,                # real Aletheia GSO scanned objects (toys, tools) as the clutter scene.
    #                                   Varied real meshes at randomized poses genuinely fail sometimes, which
    #                                   gives the mixed per-target outcomes the ranking and candidate policies
    #                                   need; a clean cube pick almost always succeeds and is degenerate. Roster
    #                                   via WILLY_GSO_ROSTER so successive collect runs draw different
    #                                   object_sets. Needs the one-time on-box `gso_assets --convert`. Default-off.
    vision: bool = False,             # real vision (GroundingDINO detect_all + SAM2) over ground-truth masks
    mask_completion: "str | None" = None,  # compare mask-completion policies for `vision`. None keeps the
    #                                   shipped default. This runner is the only on-box user of that default,
    #                                   and this scene was tuned before the default became 'none'; see the
    #                                   block comment above _SCENE2_TABLE_COLOR.
    scene2: bool = False,             # generalization layout with a different floor and lighting
    blocking: bool = False,           # add a blocker in the red target's +X approach column
    cylinder: bool = False,           # the red target is a cylinder (the EZU-35 three-finger centric grip)
    pile: bool = False,               # a dense-clutter pile of varied primitives, the real-clutter
    pile_count: int = 8,              #        baseline scene. Default-off leaves the plain pick unchanged.
    pile_seed: int = 0,
    pile_spread_mm: float = 50.0,     #        pile-density dial; smaller is tighter and more piled.
    pile_shapes: str = "mixed",       #        "mixed"|"cube"|"cylinder"; "cube" is the jaw-friendly scene.
    pile_top_ref: bool = False,       #        top-referenced grasp depth for tall objects; off is centre-ref,
    #                                          the reliable default here. Keeping it a knob lets the harness
    #                                          anchor to the known-good config and add features one at a time.
    enable_g4: bool = False,          # corridor-risk producer: stamp corridor_blockage_confidence per candidate
    enable_g12: bool = False,         # approach and retreat swept-volume validation (refuse or avoid)
    g4_rerank: bool = False,          # wire the uncertainty-rerank consumer; implies the producer above.
    #                                   from_components leaves it None, so nothing else turns it on here
    g4_rerank_weight: float = 0.3,    # subtractive penalty weight, schema-clamped to [0, 0.5]
    oblique_approach: bool = False,   # generate grasps at several approach angles, top-down plus tilts,
    oblique_tilt_deg: float = 30.0,   #        so an oblique gap-threading grasp is proposed where a top-down
    oblique_azimuths: int = 4,        #        one rams. Default-off leaves the pick unchanged.
    enable_bin: bool = False,         # enclose the clutter in 4 bin walls (prims + SelfCollisionGuard fixtures)
    near_wall: bool = False,          # shift the target toward the +X wall, so a grasp there must be rejected
    bin_half_width_mm: float = 190.0,  # inner half-extent in x and y, clearing the tool capsule (r=70) + clutter
    bin_half_width_y_mm: float | None = None,  # separate Y half-extent, authoring the KLT rectangle over a square
    bin_height_mm: float = 50.0,      # wall height; low so the arm links clear it on a top-down descent while
    #                                   the low tool (grasp z~37) still reaches a near wall and is rejected
    near_wall_offset_mm: float = 120.0,  # +X shift of the target for the near-wall reject case
    self_collision_arm: bool = False,  # enable the UR-DH arm-link capsules and the 180 deg base-frame
    #                                    reconcile, so the guard checks the arm links against the bin walls and
    #                                    not only the tool. The Isaac cell is a UR5e but its vendor is not 'ur',
    #                                    so this opt-in is required; default-off keeps the tool-only guard.
    finger_tool: bool = False,         # tool_model='finger' models the 2F-85 descending fingers as a thin
    #                                    capsule along the grasp closing axis instead of the r=70 bounding
    #                                    cylinder, a rotation-aware bin-wall footprint. Default-off keeps r=70.
    self_collisions_phys: bool = False,  # opt-in PhysX arm-versus-self collision, so a self-colliding IK branch
    #                                    physically fails in the sim too. Default-off leaves physics as it is.
    natural_aim: bool = True,            # IK prefers the natural shoulder-pan branch over the park-pi flipped
    #                                    self-colliding one. Paired with the Coal self-collision standard:
    #                                    without it the default branch self-penetrates forearm|gripper at
    #                                    0 mm, which Coal rejects.
    continuous_guard: bool = False,      # opt-in continuous collision-avoidance guard: a per-waypoint mesh
    #                                    check, a margin and a fail-safe. Default-off. Needs WILLY_COAL_PREFIX.
    continuous_guard_margin_mm: float = 8.0,  # stop at this clearance before contact; keep it under the ~19 mm
    #                                    natural-grasp wrist gap.
    collect: bool = False,            # wire the full producer stack for episode collection: the shadow
    #                                   router, the success-probability context, the corridor-risk producer
    #                                   and service recovery, so the logged records carry
    #                                   rl_candidate_features and recovery_actions for offline training
    recovery: bool = False,           # enable the service recovery loop through effective_config, which
    #                                   populates recovery_actions
    blocker_offset_mm: float = 35.0,  # blocker distance along +X from the red target
    approach_margin_mm: float = 20.0,  # swept-volume collision margin; larger flags farther blockers
    # Pick-policy tunings. standoff_mm is 150, not the 50 that `run_m1_pick` uses: from the overhead
    # park pose the pre-grasp must be high, so the move from park comes straight down and converges. A
    # low pre-grasp at z+50 does not converge from park; the standoff move times out even near the centre.
    pre_open_width_mm: float = 80.0,
    close_width_mm: float = 25.0,
    adaptive_close: bool = False,            # opt-in adaptive close (grip_w - squeeze, per grasp) for the
    #                                          clutter path, which generalizes across box sizes; the YCB path
    #                                          already has it. Default-off keeps the fixed close_width_mm.
    adaptive_close_squeeze_mm: float = 11.0,  # squeeze margin for the clutter adaptive close, as for YCB
    grasp_lift_mm: float = 12.0,
    standoff_mm: float = 150.0,
    planner_owns_approach: bool = False,        # opt-in: drive only the grasp goal so cuRobo plans the full
    #                                             descent from park to grasp, rather than the standoff plus
    #                                             interpolated waypoints. Off by default, because the standoff
    #                                             keeps a clean vertical final approach where cuRobo's free
    #                                             path can brush a side obstacle; useful only where the
    #                                             vertical approach itself is blocked.
    motion_planner: "str | None" = None,        # None takes the SimRobotConfig default, currently "curobo";
    #                                             an explicit value such as "ik" overrides it per runner.
    #                                             The effective planner is resolved below from the config
    #                                             value and a build-time cuRobo-env availability check, and
    #                                             both arm._motion_planner and align_closing_to_base_x follow
    #                                             it: "curobo" drops the base-X align, because cuRobo plans
    #                                             the natural free yaw and the base-X yaw is a blind-IK
    #                                             workaround that, in a tight bin, rotates the corridor-clear
    #                                             close back into the +X blocker.
    real_klt_bin: tuple[float, ...] | None = None,  # reference the real small_KLT.usd visual mesh as the bin
    #                                             look, over the fixture physics; an optional 4th entry is
    #                                             the Z scale.
    curobo_bin_world: bool = False,             # register the bin walls into cuRobo's collision world so the
    #                                             planner threads the gripper into the bin around the walls,
    #                                             instead of planning blind and rejecting afterwards through
    #                                             Coal. Default-off leaves the planner blind to the walls.
    retreat_mm: float = 100.0,
    retreat_steps: int = 1,                 # chunk the post-close lift into N settled steps, which damps
    #                                         the retreat pendulum on wide or heavy boxes. Default 1 is one lift.
    gripper_mount: "str | None" = None,     # mount a standalone vendor gripper (e.g. "schunk_egu50")
    #                                         on the wrist instead of the baked 2F-85. None keeps the 2F-85.
    depth_source: str = "gt",               # "rendered" feeds the calculator real, non-uniform depth
    grasp_depth_reference: str = "centre",  # calculator depth reference (centre|top)
    depth_noise: "DepthNoiseConfig | None" = None,            # synthetic sensor depth noise on the sim
    #                                                          depth map; None leaves the depth untouched
    cloud_outlier_filter: "CloudOutlierConfig | None" = None,  # the point-cloud outlier filter
    overhang_shelf: "tuple[tuple[float, float, float], tuple[float, float, float]] | None" = None,
    #                  (center_mm, half_extents_mm) of a physical shelf or ledge prim above the approach. It
    #                  renders in depth and is segmented into other_object_mask, so the corridor analyzer
    #                  sees 3D structure a flat cell cannot show. None leaves the scene without a shelf.
    depth_band_mm: float = 0.0,             # above 0 enables the calculator robust depth-band filter
    clutter_sizes: "dict[str, tuple[float, float, float]] | None" = None,  # per-label cube size override
):
    """Boot a multi-object overhead scene and wire the native dense pick service.

    Returns ``(service, arm, gripper, handles, cfg, target_idx, target_label)``.
    """
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
    # Reach the calculator through the factory, not by name, so `robot.grasping.calculator: deep`
    # reaches this runner and the six entry points that share its `build_service`.
    #
    # Under `deep` this site refuses, on purpose. It passes `ik_service` unconditionally and
    # `corridor_risk_per_candidate` whenever the corridor-risk producer or collection is on, plus the
    # rendered-depth penetration descend and the outlier filter. The learned generator honours none of
    # those, and a dense gate run without the reachability filter would report a different experiment
    # under the same name. The factory names what it cannot carry rather than dropping it.
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.grasping.types.perception import PerceptionSource

    env = RunnerEnv.from_env(vision=vision, view_height_mm=0.0)  # the WILLY_* knobs, parsed once
    # Multi-object clutter scene through the runner's objects_override; the single-object runners keep
    # theirs. The spec and camera-override computation need no config, so they run before the shared boot.
    specs = (
        _pile_specs(count=pile_count, seed=pile_seed, spread_mm=pile_spread_mm, shapes=pile_shapes) if pile
        else _gso_specs() if gso
        else _ycb_specs(scene2, env) if ycb
        else _cylinder_specs() if cylinder
        else _blocking_specs(blocker_offset_mm) if blocking
        else _clutter_specs()
    )
    if clutter_sizes:  # varied cube sizes for stacking, where a wide base gives a stable pyramid
        specs = [s.model_copy(update={"size_mm": tuple(clutter_sizes[s.name])}) if s.name in clutter_sizes else s
                 for s in specs]
    target_idx = _select_by_prompt(specs, prompt)
    specs = _apply_rl3_substrate(specs, target_idx, env)  # RL3 failure-injection (default-off)
    # Default-off: enclose the clutter in 4 bin walls. The same FixtureBoxConfig list is the single source
    # for the physical wall prims in scene_kwargs below and for the SelfCollisionGuard fixtures, owner-set
    # onto the arm preflight after boot. `near_wall` shifts the target toward the +X wall so a top-down
    # grasp's tool capsule enters the wall and the guard rejects it; the un-shifted, centred target is the
    # case where the guard must not reject.
    _bin_fixtures = (
        _make_bin_fixtures(
            half_width_mm=bin_half_width_mm, half_width_y_mm=bin_half_width_y_mm, height_mm=bin_height_mm
        ) if enable_bin else None
    )
    if enable_bin and near_wall:
        _t = specs[target_idx]
        _tx, _ty, _tz = (float(c) for c in _t.position_mm)
        specs[target_idx] = _t.model_copy(
            update={"position_mm": (_tx + float(near_wall_offset_mm), _ty, _tz)}
        )
    # Put the overhead camera's nadir over the target so its single top-down view is parallax-free: it
    # sees the top rather than a side wall, giving a centred, holding grasp. Cubes keep the config camera.
    _cam_override = None
    if ycb:  # camera nadir over the target: a parallax-free, centred grasp
        _tx, _ty, _ = specs[target_idx].position_mm
        # The target sits at the camera nadir, so it is parallax-free at any height and the height is tuned
        # for the perception in use. Ground-truth masks are pixel-exact, so z=1800 still holds the box and
        # both X-flank distractors in one frame. Real vision, a base detector with a colour-disambiguated
        # prompt, is height-robust across z=1200 to 1600; above about 1600 the objects get too small for
        # GroundingDINO and detection collapses, and the pick then fails safe with no_valid_grasp rather
        # than grasping a tiny blob. The default 1200 sits inside the robust band. The grasp depth is
        # height-invariant either way, and the height is env-tunable.
        _cam_override = (float(_tx), float(_ty), env.ycb_cam_z)
    # The generalization scene uses a different table colour and lighting, which is the appearance change
    # that shows the pipeline is not overfit to one scene; the first scene keeps the defaults (None).
    # Applies only on the ycb path.
    _s2 = ycb and scene2
    # The shared boot prefix builds the fail-closed arm, session, scene and gripper; this runner forwards
    # its multi-object, camera-nadir and appearance overrides to it as scene_kwargs.
    _appearance = SceneAppearance(
        table_color=_SCENE2_TABLE_COLOR, dome_intensity=_SCENE2_DOME_INTENSITY,
        key_intensity=_SCENE2_KEY_INTENSITY,
    ) if _s2 else None
    # An optional overhang shelf: a physical prim that renders in depth and is also a perception target,
    # segmented into other_object_mask, so the corridor analyzer's mask path sees 3D structure. It is not a
    # SelfCollisionGuard fixture; the safety fixtures stay `_bin_fixtures` and the shelf is a perception
    # obstacle only. `_scene_walls` is the bin walls, if any, plus the shelf, in prim-index order.
    _scene_walls = list(_bin_fixtures) if _bin_fixtures else []
    _shelf_prim = None
    if overhang_shelf is not None:
        from src.config.schema.robot.safety_schema import FixtureBoxConfig
        from src.willy_sim.scene import BIN_WALL_PRIM
        _shelf_prim = f"{BIN_WALL_PRIM}_{len(_scene_walls)}"
        _scene_walls.append(FixtureBoxConfig(name="overhang_shelf",
                            center_mm=overhang_shelf[0], half_extents_mm=overhang_shelf[1]))
    cell = bootstrap_sim_cell(
        data_dir, headless=headless, gripper_mount=gripper_mount, scene_kwargs={
            "objects_override": specs, "camera_position_mm": _cam_override, "appearance": _appearance,
            # Diagnostic: WILLY_BIN_NO_PRIMS skips the physical wall prims while keeping the safety
            # fixtures, which separates a physical arm-versus-wall collision from a preflight cause.
            "bin_walls": (None if os.getenv("WILLY_BIN_NO_PRIMS") else (_scene_walls or None)),
            "enable_self_collisions": self_collisions_phys,  # opt-in PhysX arm-versus-self collision
            "real_klt_bin": real_klt_bin,  # the real small_KLT visual mesh over the fixture physics
        },
    )
    arm, gripper, handles, cfg, sim = cell.arm, cell.gripper, cell.handles, cell.cfg, cell.sim
    # Resolve the effective planner: the explicit parameter, else the config default. A build-time
    # availability check then degrades "curobo" to "ik" when the cuRobo environment is absent, so the
    # per-planner config below, the align flag and the arm planner, matches the planner that will really
    # run. The driver's move-time fallback catches only a present but broken environment.
    from src.robot.safety.planning import curobo_env_available
    _requested_planner = motion_planner if motion_planner is not None else arm._motion_planner
    _effective_planner = _requested_planner
    if _requested_planner == "curobo" and not curobo_env_available():
        _effective_planner = "ik"
        print("[motion-planner] cuRobo env absent -> building the 'ik' (blind) path for this run.", flush=True)
    arm._motion_planner = _effective_planner
    # Register the bin walls, the same FixtureBoxConfig list, into cuRobo's collision world, so the planner
    # takes the gripper into the bin around the walls instead of meeting them only as a physical prim and a
    # later Coal rejection. Default-off; it needs --curobo-bin-world, the curobo planner and --bin
    # together, and without it the planner stays blind to the walls. Cuboids are BASE frame, metres, WXYZ.
    if curobo_bin_world and _effective_planner == "curobo" and _bin_fixtures:
        _cub = [
            {
                "name": _fx.name,
                "dims_m": [2.0 * float(_fx.half_extents_mm[0]) / 1000.0,
                           2.0 * float(_fx.half_extents_mm[1]) / 1000.0,
                           2.0 * float(_fx.half_extents_mm[2]) / 1000.0],
                "pose": [float(_fx.center_mm[0]) / 1000.0, float(_fx.center_mm[1]) / 1000.0,
                         float(_fx.center_mm[2]) / 1000.0, 1.0, 0.0, 0.0, 0.0],
            }
            for _fx in _bin_fixtures
        ]
        _cub.append({"name": "floor", "dims_m": [2.0, 2.0, 0.05], "pose": [0.0, 0.0, -0.026, 1.0, 0.0, 0.0, 0.0]})
        _n_world = arm.set_curobo_world(_cub)
        print(f"[curobo-world] registered {_n_world}/{len(_cub)} obstacles (bin walls + floor) -> cuRobo plans "
              f"AROUND the walls (bin 'really real' for the planner)", flush=True)
    _dwell = cell.dwell  # the steady-state gate
    # Owner-set the opt-in natural-aim IK seed and the continuous collision-avoidance guard; both default off.
    # The guard's fixtures = the bin walls (if wired) so an in-motion approach into a wall is also caught.
    _cg_fix = (list(_bin_fixtures) if (_bin_fixtures is not None
               and not os.getenv("WILLY_BIN_NO_FIXTURES")) else [])
    # The monitor is built with the same geometry the one-shot guard gets below. Hard-coded values would
    # watch a --gripper-mount run with the baked 2F-85 meshes while the guard used the mounted gripper's,
    # and would watch a ur3e cell with ur5e link lengths.
    wire_safety_guards(arm, natural_aim=natural_aim, continuous_guard=continuous_guard,
                       continuous_guard_margin_mm=continuous_guard_margin_mm, fixtures=_cg_fix,
                       kinematics_model="ur5e",  # must track the _sc_update below
                       kinematics_base_yaw_deg=180.0,
                       collision_mesh_variant=gripper_mount or None)
    # Owner-set the SelfCollisionGuard fixtures onto the arm preflight. `bootstrap_sim_cell` built the
    # preflight from the config, which ships an empty fixtures list, so the bin walls are injected here and
    # an approach into a wall is self-collision-rejected. Without --bin the preflight is unchanged.
    _wire_bin = _bin_fixtures is not None and not os.getenv("WILLY_BIN_NO_FIXTURES")  # diagnostic: prims only
    _sc_update: dict[str, object] = {}
    if _bin_fixtures is not None and _wire_bin:
        _sc_update["fixtures"] = list(_bin_fixtures)
    if gripper_mount:  # a mounted gripper makes the Coal guard load that gripper's collision meshes,
        _sc_update["collision_mesh_variant"] = gripper_mount  # not the baked 2F-85's (the variant npz)
    if self_collision_arm:
        # Opt the sim cell, a UR5e whose vendor is not 'ur', into the bundled UR-DH arm-link capsules and
        # the 180 deg base-frame reconcile, so the guard checks the arm links against the bin-wall fixtures
        # and not only the tool capsule. Arm-versus-arm distance is rotation-invariant; the reconcile only
        # aligns arm against base, tool and wall.
        _sc_update["kinematics_model"] = "ur5e"
        _sc_update["kinematics_base_yaw_deg"] = 180.0
    if finger_tool:
        # A faithful 2F-85 finger footprint: a thin capsule along the grasp closing axis instead of the
        # r=70 bounding cylinder. The defaults (radius 16 mm, span 150 mm) ship in the schema; this only
        # flips the model.
        _sc_update["tool_model"] = "finger"
    if _sc_update:
        from src.robot.safety.preflight import SafetyPreflight

        _bin_sc = cell.robot.safety.self_collision.model_copy(update=_sc_update)
        _safety = cell.robot.safety.model_copy(update={"self_collision": _bin_sc})
        arm._preflight = SafetyPreflight.from_safety_config(_safety, cell.robot.workspace_limits)
        print(f"[self-collision] arm_capsules={self_collision_arm} tool_model={'finger' if finger_tool else 'capsule'} "
              f"bin_walls={'yes' if _wire_bin else 'no'} (half_width_x={bin_half_width_mm:.0f} "
              f"half_width_y={bin_half_width_y_mm if bin_half_width_y_mm is not None else bin_half_width_mm:.0f}mm, "
              f"near_wall={near_wall})", flush=True)
        # Surface the self-collision reject so the per-run output separates a bin-wall reject from any other
        # failure, since all of them collapse to pick_outcome='execution_failed'. Wrap the owned guard's
        # evaluate; the closure stashes the offending pair on the arm. `reason` keeps the fixture-only pair,
        # `any` keeps any reject pair, including a link against link one.
        arm._bin_reject = {"reason": None, "any": None}  # type: ignore[attr-defined]
        for _g in getattr(arm._preflight, "_guards", ()):
            if type(_g).__name__ == "SelfCollisionGuard":
                _orig_eval = _g.evaluate

                def _bin_traced(ctx, _orig=_orig_eval, _store=arm._bin_reject):  # type: ignore[attr-defined] # noqa: ANN001, ANN202
                    dec = _orig(ctx)
                    if not dec.accepted:
                        _pair = str(dec.detail.get("pair", ""))
                        _store["any"] = _pair
                        if "fixture:" in _pair:
                            _store["reason"] = dec.detail.get("pair")
                    return dec

                _g.evaluate = _bin_traced  # type: ignore[method-assign]
                break

    # Off-axis clutter needs the convention-correct overhead extrinsic. `handles.camera_to_base` is a
    # hand-built grasp transform 180 deg in-plane from the true one, so it flips off-centre positions about
    # the camera centre. That is invisible at Y=0 and wrong for clutter, where the grasp then lands on a
    # distractor. Fit the true CV-optical extrinsic with Umeyama, as `run_fused_pick.run_clutter_gate`
    # does; that reads rendered depth at a near clip of 0.05 rather than 1.0, from a clean overhead view
    # with the arm parked so it never occludes perception. Ground-truth depth still overlays the solid
    # per-object grasp depth, and the 0.05 near clip is also what makes the overhead RGB render at all, so
    # the grasp-viewer frames show the real cubes instead of black.
    from src.willy_sim.scene import camera_to_base_ground_truth
    from src.robot.core import JointPositions

    overhead_cam = handles.camera
    park_q = np.asarray(sim.park_joint_positions, dtype=np.float64)
    for _fn in (lambda: overhead_cam.add_distance_to_image_plane_to_frame(),
                lambda: overhead_cam.set_clipping_range(0.05, 1.0e6)):
        try:
            _fn()
        except Exception:  # noqa: BLE001 (annotator/clip best-effort)
            pass
    arm.move_joint(JointPositions(park_q))  # clean overhead view for the extrinsic fit + every perceive
    app = getattr(arm.session, "app", None)
    t_overhead_true = None
    for attempt in range(6):
        for _ in range(sim.scene_setup.render_warmup_steps if attempt == 0 else 12):
            arm.session.step(render=True)
            if app is not None:
                app.update()
        try:
            t_overhead_true, _ = camera_to_base_ground_truth(overhead_cam)
            break
        except Exception as exc:  # noqa: BLE001 (depth not ready yet; pump + retry)
            if attempt == 5:
                raise
            print(f"overhead true-extrinsic retry {attempt + 1}/5 ({exc})", flush=True)
    assert t_overhead_true is not None  # the loop re-raised on the 6th failure otherwise (narrow for mypy)
    print(f"overhead TRUE extrinsic T_cam_to_BASE mm: "
          f"{np.round(np.asarray(t_overhead_true.to_matrix())[:3, 3], 1)}", flush=True)

    # Multi-mask ground-truth perception: one labelled segmentation per object, so the orchestrator sees
    # every object in one frame. `object_specs` carries (prim_path, spec.name), which are the prompt labels.
    # For a tall object the grasp is referenced to the object top plus a penetration, so a tall YCB box's
    # grasp rises near its top and the gripper body clears the top instead of driving into it. The cube
    # path leaves this off, and the value is None when ycb is off.
    # For the pile scene the top reference is opt-in through `pile_top_ref`. It helps tall objects, which a
    # centre-referenced grasp rams, but on short or flat objects the centre reference is the reliable path
    # here, so the harness can anchor to the known-good config and add the top reference deliberately.
    _ycb_pen = env.ycb_penetration if ycb else (12.0 if (pile and pile_top_ref) else None)
    # A "rendered" depth source feeds the calculator the real, non-uniform rendered depth: the ground-truth
    # source skips its uniform centre-depth stamp through ground_truth_depth=False. The calculator's
    # depth-distribution levers, the top reference and the depth band, only do work that way; under the
    # uniform stamp they are no-ops. This applies to the ground-truth path only, because the vision source
    # has its own always-on top reference. The default "gt" keeps ground_truth_depth=True.
    _render_depth = depth_source == "rendered"
    if _render_depth and vision:
        raise SystemExit(
            "--depth-source rendered is GT-path only (not --vision); the vision source has its own "
            "always-on top-reference (a follow-up converges it)."
        )
    _depth_band_mm = depth_band_mm if depth_band_mm > 0.0 else None
    # In rendered mode the calculator owns the penetration descend below the referenced top surface,
    # because the perception override is skipped, so it reuses the ycb penetration of about 12 mm. In gt
    # mode this stays 0.0 and the perception override still does the descend.
    _calc_pen_mm = float(_ycb_pen) if (_render_depth and _ycb_pen is not None) else 0.0
    perception: PerceptionSource  # both sources satisfy the acquire()->PerceptionFrame Protocol
    if vision:
        # Real perception: overhead RGB into GroundingDINO detect_all, with a multi-phrase prompt over the
        # object labels, then SAM2 per box, giving one labelled segmentation per object with
        # top-referenced depth. HF-offline is set by default so transformers does not ping the hub at boot,
        # because Kit closes the httpx client and the boot then crashes. The detector must run fp32 in sim,
        # which comes from cfg.models. `object_labels` are the spawn names, which is what makes the target
        # match canonical.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from src.willy_sim.perception import MultiObjectVisionPerceptionSource
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
        from src.models.segmentation.realtime.segmenter import Sam2Segmenter

        _labels = [label for (_prim, label) in handles.object_specs]
        # Use a bigger GroundingDINO for the dense-vision path than the tiny config default: it detects
        # small and cluttered objects better across camera heights. It is isolated here so the runners that
        # take the config default are untouched. The base model is cached locally and the id is
        # env-overridable; a blank override falls back to the config model_id.
        _det_model = env.ycb_detector
        _det_cfg = cfg.models.objectdetector.model_copy(update={"model_id": _det_model}) if _det_model \
            else cfg.models.objectdetector
        # The mask policy is explicit here rather than inherited, so a gate run says in its own arguments
        # which transform produced its number. `None` keeps the shipped default; the argparse help and the
        # block comment above `_SCENE2_TABLE_COLOR` say why this scene has a stake in it.
        from src.robot.perception.mask_completion import (
            DEFAULT_MASK_COMPLETION,
            MaskCompletion,
        )

        _policy = (
            DEFAULT_MASK_COMPLETION if mask_completion is None else MaskCompletion(mask_completion)
        )
        print(f"=== MASK COMPLETION: {_policy.value} ===", flush=True)
        perception = MultiObjectVisionPerceptionSource(
            camera=overhead_cam,
            detector=GroundingDinoObjectDetector(_det_cfg),
            segmenter=Sam2Segmenter(cfg.models.segmenter),
            prompt=prompt, object_labels=_labels, session=arm.session,
            warmup_steps=max(25, sim.scene_setup.render_warmup_steps),
            grasp_top_penetration_mm=float(_ycb_pen if _ycb_pen is not None else 12.0),
            mask_completion=_policy,
        )
    else:
        _gt_targets = list(handles.object_specs)
        if _shelf_prim is not None:  # segment the shelf so it becomes an other_object_mask neighbour
            _gt_targets.append((_shelf_prim, "overhang_shelf"))
        perception = MultiObjectGroundTruthPerceptionSource(
            camera=overhead_cam, targets=_gt_targets, session=arm.session,
            warmup_steps=sim.scene_setup.render_warmup_steps, ground_truth_depth=not _render_depth,
            grasp_lift_mm=grasp_lift_mm, grasp_top_penetration_mm=_ycb_pen,  # both ignored when rendered
        )
    if depth_noise is not None and depth_noise.enabled:  # wrap perception with synthetic depth noise
        from src.willy_sim.harness.depth_noise import NoisyDepthPerceptionSource
        perception = NoisyDepthPerceptionSource(perception, depth_noise)
        print(f"[depth-noise] ON gaussian={depth_noise.gaussian_sigma_mm}mm "
              f"speckle={depth_noise.speckle_fraction:.0%}@{depth_noise.speckle_sigma_mm}mm (H3.9a sim sensor model)",
              flush=True)
    calculator = build_calculator(
        cell.robot,
        camera_matrix=np.asarray(overhead_cam.get_intrinsics_matrix(), dtype=np.float64),
        max_grip_width_mm=cell.robot.gripper.max_width_mm,
        min_grip_width_mm=cell.robot.gripper.min_width_mm,
        # Reject near-singular grasps, such as the UR5e base-Y top-down close that solves IK but executes
        # tens of mm off, so the calculator selects the reachable base-X antipodal grasp.
        # Passing the joint-limit config lets the IK service fill IKQualityMetrics.joint_margin_deg with a
        # real value for the feasibility scorer and the success-probability feature. The leverage is gated:
        # the feasibility joint_margin is disabled by default, the sim's +/-360 limits rarely bind, and the
        # success model is not trained on it. The metric is real telemetry either way.
        ik_service=ArmBackedIKService(
            arm, joint_limits_config=cell.robot.safety.joint_limits,
            debug=env.ik_debug,  # log per-candidate cond, min_sv and joint_margin
        ),
        # The corridor-risk producer stamps corridor_blockage_confidence per candidate: real per-candidate
        # approach-corridor risk, measured from the neighbour masks. Collection turns it on as well, so the
        # logged rl_candidate_features carry a real per_candidate_uncertainty.
        corridor_risk_per_candidate=(enable_g4 or collect or g4_rerank),  # the rerank implies the producer
        oblique_approach=oblique_approach,  # multi-angle approach generation; default off
        oblique_tilt_deg=oblique_tilt_deg,
        oblique_azimuths=oblique_azimuths,
        # In "gt" mode these stay "centre" and None; in "rendered" mode they are the real levers
        # (grasp_depth_reference=top and depth_band_mm), paired with the rendered-depth passthrough.
        grasp_depth_reference=grasp_depth_reference,
        grasp_top_penetration_mm=_calc_pen_mm,
        depth_band_mm=_depth_band_mm,
        cloud_outlier_filter=cloud_outlier_filter,  # default None leaves the cloud unfiltered
    )
    resolver = StaticCameraToBaseResolver(transform=t_overhead_true)
    # Close width. Adaptive by default for YCB: close = measured grip_w minus squeeze, so any object is
    # clamped without per-object tuning; the thin sugar box, about 44 mm grip_w and about 33 close, and a
    # wider box both hold. `WILLY_YCB_CLOSE` forces a fixed width, and the cube runners pass an explicit
    # close. The squeeze margin is env-tunable and defaults to 11, which gives about 33 for the sugar box.
    _ycb_close_fixed = env.ycb_close
    if not ycb:
        if adaptive_close or gso:  # adapt the close to the grasp width (grip_w - squeeze) so varied real
            #                        GSO objects (25-56 mm) each clamp without per-object tuning
            effective_close_width = None        # the policy computes close = grip_w - squeeze per grasp
            effective_close_squeeze = adaptive_close_squeeze_mm
        else:
            effective_close_width = close_width_mm  # default: fixed 25mm (tuned for 30mm) = byte-identical
            effective_close_squeeze = 1.0
    elif _ycb_close_fixed is not None:  # explicit fixed override
        effective_close_width = _ycb_close_fixed
        effective_close_squeeze = 1.0
    elif specs[target_idx].name in _YCB_CLOSE_BY_TARGET:
        # A per-target tuned close for known objects. A fixed value is steadier than the adaptive one,
        # which inherits the vision grip_w noise on a thin object. Untuned targets fall to the general
        # adaptive close below.
        effective_close_width = _YCB_CLOSE_BY_TARGET[specs[target_idx].name]
        effective_close_squeeze = 1.0
    else:
        effective_close_width = None  # General fallback: adaptive close = grip_w - squeeze (per grasp)
        effective_close_squeeze = env.ycb_squeeze
    policy = GraspExecutionPolicy(
        arm=arm, gripper=gripper, standoff_mm=standoff_mm, retreat_mm=retreat_mm,
        retreat_steps=retreat_steps,         # the default 1 is a single lift
        pre_open_width_mm=pre_open_width_mm, close_width_mm=effective_close_width,
        close_squeeze_mm=effective_close_squeeze,
        require_base_frame_grasp=True,
        require_steady_before_motion=(_dwell.require_steady_before_motion if _dwell else False),
        steady_timeout_s=(_dwell.steady_timeout_s if _dwell else 5.0),
        # Yaw a close that points near base Y onto the base-X line so the overhead move converges; a base-Y
        # close times out. A natural base-X close is left alone, and the cubes are symmetric.
        # cuRobo plans the natural free-yaw close and has no base-Y IK timeout, so the base-X yaw is
        # dropped there: in a tight bin it would rotate the corridor-clear close back into the +X blocker.
        # Keyed on the effective planner, so a build degraded to ik keeps the base-X align on.
        align_closing_to_base_x=(_effective_planner != "curobo"),
        # Opt-in, default off: when set, cuRobo owns the full descent from park to grasp as a single goal.
        # The default keeps the standoff and interpolated waypoints, so the final approach is a clean
        # vertical descent that does not brush a side obstacle.
        planner_owns_approach=planner_owns_approach,
    )
    grasp_mode = resolve_demo_mode(mode)
    service = AutonomousGraspService.from_components(
        arm=arm, calculator=calculator, perception=perception, gripper=gripper,
        mode=grasp_mode, frame_resolver=resolver, policy=policy, max_attempts=3,
        **mode_service_kwargs(grasp_mode),
    )
    # A hard label-target: grasp only the prompted object. The others stay neighbour clutter, which the
    # DENSE_CLUTTER sampler and the corridor-risk and approach-validation features consume.
    target_label = handles.object_specs[target_idx][1]
    service.set_target_label(target_label)
    # Wire the dense-mode swept-volume approach validator onto the orchestrator. `from_components` does not
    # set these carriers, though `from_robot_config` and `apply_orchestrator_overlays` would, so they are
    # set directly: the validator, the dense apply-mode, and the mode_label the per-pick gate reads.
    if enable_g12:
        from src.robot.grasping.motion.trajectory_safety import ApproachPathPolicy

        orch = service.runtime.orchestrator
        orch.approach_path_policy = ApproachPathPolicy(
            standoff_mm=standoff_mm, retreat_mm=retreat_mm,
            num_approach_samples=6, num_retreat_samples=4,
            collision_margin_mm=float(approach_margin_mm),
        )
        orch._approach_path_modes = frozenset({"dense_clutter"})
        orch.mode_label = "dense_clutter"
        if env.trace_g12:  # debug: log the obstacle cloud size + per-candidate verdict
            _oc = orch._obstacle_cloud_in_candidate_frame
            _sc = orch._approach_sweep_is_clear

            def _oc_traced(result, _oc=_oc):  # noqa: ANN001, ANN202
                c = _oc(result)
                print(f"[g12] cloud={'None' if c is None else len(c)} candidates={len(result.candidates)} "
                      f"pending_cam2base={'set' if orch._pending_camera_to_base is not None else 'None'}", flush=True)
                return c

            def _sc_traced(cand, cloud, _sc=_sc):  # noqa: ANN001, ANN202
                clear = _sc(cand, cloud)
                print(f"[g12]  cand pos={np.round(np.asarray(cand.position), 1)} grip_w={cand.grip_width_mm:.0f} "
                      f"clear={clear}", flush=True)
                return clear

            orch._obstacle_cloud_in_candidate_frame = _oc_traced  # type: ignore[method-assign]
            orch._approach_sweep_is_clear = _sc_traced  # type: ignore[method-assign]
    # Wire the uncertainty-rerank consumer directly. `from_components` does not set it; only
    # `from_robot_config` and `apply_orchestrator_overlays` would, and that path has no live caller here.
    # The rerank fires only when the whole gate ladder in `_rerank.py` holds, so all three carriers are set:
    #   (1) uncertainty_rerank_config, enabled, with weight above 0 and the mode among its modes,
    #   (2) every candidate stamped with corridor_blockage_confidence, from the producer forced on above,
    #   (3) a non-shadow ShadowSuccessContext, mode_label among its modes and lifecycle not "shadow".
    # That context is also the success-probability carrier, so wiring it co-activates the synthetic
    # success annotation. That is metadata only and reorders nothing, because ranking_blend_config stays
    # None. The context is built directly, with lifecycle "active" meaning this rerank may influence the
    # pick, from the committed synthetic model, bypassing the loader's promotion gate, which guards only
    # the try_load_* path.
    if g4_rerank:
        from src.robot.grasping.scoring.success_probability import (
            ShadowSuccessContext,
            UncertaintyRerankConfig,
            load_success_probability_model,
        )

        orch = service.runtime.orchestrator
        orch.uncertainty_rerank_config = UncertaintyRerankConfig(
            enabled=True, weight=float(g4_rerank_weight),
            modes=("dense_clutter", "dense_autonomous"),
        )
        _sm_cfg = getattr(getattr(cfg.robot, "grasping", None), "success_model", None)
        _artifact_dir = getattr(_sm_cfg, "artifact_dir", "assets/models/success_probability/v1")
        orch.shadow_success_context = ShadowSuccessContext(
            model=load_success_probability_model(_artifact_dir),
            version_label="v1", lifecycle_phase="active", mode_label="dense_clutter",
        )
        orch.mode_label = "dense_clutter"
        print(f"[g4-rerank] consumer wired (weight={float(g4_rerank_weight):.2f}, lifecycle=active)", flush=True)
    if collect or recovery:
        _wire_collection_carriers(service, cfg, grasp_mode, collect=collect, recovery=recovery)
    return service, arm, gripper, handles, cfg, target_idx, target_label


def _wire_collection_carriers(service, cfg, grasp_mode, *, collect: bool, recovery: bool) -> None:  # noqa: ANN001
    """Wire the producer stack onto a ``from_components`` service for episode collection.

    Collected records then carry the signals offline training needs. ``collect`` wires the shadow router,
    which fills ``rl_candidate_features``, and the success-probability context, which fills a real
    ``shadow_predicted_success_probability``. ``recovery`` enables the service recovery loop through
    ``effective_config.recovery_orchestrator``, which fills ``recovery_actions``, without the fusion
    overlay, because that overlay accumulates across reset-picks and starves candidates. Every step is
    guarded and lazy: a failure degrades to no carrier and never breaks the deterministic pick.
    """
    orch = service.runtime.orchestrator
    if collect:
        try:
            from types import SimpleNamespace

            from src.robot.execution.autonomous_grasp.shadow import maybe_build_shadow_router

            router = maybe_build_shadow_router(SimpleNamespace(rl=SimpleNamespace(mode="rl_shadow")))
            if router is not None:
                # Set both: the service field, which `_maybe_attach_shadow_router` reads to splice
                # rl_candidate_features into the record, and the orchestrator field, which the ranking
                # shadow reads to populate `_candidate_log`. `from_components` leaves both None.
                service.shadow_router = router
                orch.shadow_router = router
                print(f"[collect] shadow_router wired (candidate={router.candidate_policy is not None}, "
                      f"ranking={router.ranking_policy is not None})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[collect] shadow_router wiring skipped: {exc}", flush=True)
        try:
            from src.robot.grasping.scoring.success_probability import (
                try_load_shadow_success_context,
            )

            sm_cfg = getattr(getattr(cfg.robot, "grasping", None), "success_model", None)
            if sm_cfg is not None:
                ctx = try_load_shadow_success_context(
                    enabled=True, artifact_dir=str(sm_cfg.artifact_dir),
                    mode_label=getattr(grasp_mode, "value", str(grasp_mode)), version_label="v1",
                )
                if ctx is not None:
                    orch.shadow_success_context = ctx
                    print("[collect] U2 shadow_success_context wired", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[collect] U2 success-context wiring skipped: {exc}", flush=True)
    if recovery:
        try:
            from dataclasses import replace as _dc_replace

            from src.robot.execution.autonomous_grasp.builders import build_effective_config

            grasping_cfg = getattr(cfg.robot, "grasping", None)
            eff = (
                build_effective_config(grasping_cfg, resolved_mode=grasp_mode, resolved_max_attempts=3)
                if grasping_cfg is not None else None
            )
            if eff is not None:
                # Enable the service recovery loop for dense_clutter without applying the orchestrator
                # overlays: the fusion overlay accumulates BASE-frame voxels across reset-picks and starves
                # candidates. Fixture-free recovery actions only, because NUDGE_TARGET and
                # CONTAINER_AGITATE are physical and need a FixtureEnvelope the service-config recovery
                # path does not supply. next_target is allowed and is one of those actions.
                ro = _dc_replace(
                    eff.recovery_orchestrator, enabled=True,
                    allowed_actions=("rescan", "next_viewpoint", "next_target"),
                    apply_modes=("dense_clutter", "auto"), max_actions=3,
                )
                service.effective_config = _dc_replace(eff, recovery_orchestrator=ro)
                print("[collect] service recovery enabled (recovery_orchestrator; no fusion overlay)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[collect] recovery wiring skipped: {exc}", flush=True)


def _apply_rl3_substrate(specs: list, target_idx: int, env: RunnerEnv) -> list:
    """Failure-injection substrate, off by default, that makes the service recovery loop fire.

    The failures the sim produces on its own, slips and approach blocks, do not trigger the loop's allowed
    actions of rescan, next_viewpoint and next_target, so this engineers failures that do, and
    ``recovery_actions`` then populate. ``WILLY_RL3_SUBSTRATE`` takes ``unreachable``, an oversized target
    that yields NO_CANDIDATES_GENERATED and so next_viewpoint or rescan; ``occlusion``, an occluder over
    the target that yields an empty mask; or ``both``. The recoveries do not succeed in sim, because
    re-perceiving cannot fix an ungraspable target, so the recovery policy abstains with
    degenerate_single_action. This resolves "recovery never fires", not "recovery succeeds".
    """

    rl3 = env.rl3_substrate
    if rl3 not in ("unreachable", "occlusion", "both") or not specs:
        return specs
    if rl3 in ("unreachable", "both"):
        big = env.rl3_big_mm  # wider than the 85 mm 2F-85 maximum, so no valid grasp exists
        _bx, _by, bz = specs[target_idx].size_mm
        specs[target_idx] = specs[target_idx].model_copy(update={"size_mm": (big, big, float(bz))})
    if rl3 in ("occlusion", "both"):
        ox, oy, oz = specs[target_idx].position_mm
        specs.append(SimObjectConfig(
            name="occluder", color=(0.15, 0.15, 0.15),
            position_mm=(float(ox), float(oy), float(oz) + 140.0), size_mm=(90.0, 90.0, 20.0),
        ))
    return specs


def _scene_family_id(*, kind: str, target_label: str) -> str:
    """A stable, coarse scene-family id: the scene kind plus the target object's identity.

    It is independent of the per-episode ``scene_id``. Pair-grouping in ``rl/train_ranking._group_key``
    and the leakage-safe split both use this granule: every episode of the same target object forms one
    group, so success and failure pairs form across episodes, and a group-aware split keeps each family on
    one side, so no object identity leaks between train and test and the object_set overlap audit passes
    honestly. The generalization layout and the randomized appearance are within-family diversity of the
    same object and deliberately not a separate family. ``kind`` is the scene granule (cube|ycb|gso), so a
    GSO real-object family never collides with a same-named cube or ycb family.
    """
    slug = "_".join(str(target_label).strip().lower().split()) or "object"
    return f"{kind}:{slug}"


def _build_randomizer(
    *, randomize: bool, seed: int | None, rand_seed: int | None,
    jitter_mm: float, collect: bool, ycb: bool, gso: bool = False,
) -> tuple[RandomizationConfig, DomainRandomizer]:
    """Select the domain-randomization config and build its randomizer.

    ``--randomize`` gives the full set: position, orientation, colour and lighting. ``--seed`` alone gives
    position, and colour as well when collecting cube scenes. Neither leaves randomization disabled and
    the reset at the home poses. Registers the Replicator light randomizers once when lighting is on, and
    prints the enabled axes. Returns ``(rand_cfg, randomizer)``.
    """
    if randomize:
        rand_cfg = RandomizationConfig.full(
            seed=rand_seed if rand_seed is not None else (seed or 0), jitter_mm=jitter_mm
        )
    elif seed is not None:
        rand_cfg = RandomizationConfig.legacy(
            seed=seed, jitter_mm=jitter_mm, recolor=(collect and not ycb and not gso)
        )
    else:
        rand_cfg = RandomizationConfig.disabled()
    randomizer = DomainRandomizer(rand_cfg)
    if rand_cfg.any_lighting:  # register the Replicator light randomizers once (lights exist post scene-build)
        randomizer.setup_lighting(dome_path="/World/Lighting/Dome", key_path="/World/Lighting/Key")
    if rand_cfg.enabled:
        print(f"[s2] domain randomization ON (seed={rand_cfg.seed}) axes: "
              f"pos={rand_cfg.position_jitter_mm}mm yaw={rand_cfg.orientation_jitter_deg}deg "
              f"color={rand_cfg.randomize_color} (numpy/OmniPBR) dome={rand_cfg.dome_intensity_range} "
              f"key={rand_cfg.key_intensity_range} (Replicator)", flush=True)
    return rand_cfg, randomizer


def _install_trace_hooks(service, arm, env: RunnerEnv) -> None:  # noqa: ANN001 (lazy Isaac objects)
    """Opt-in debug monkeypatches over the arm and the orchestrator.

    ``env.trace_moves`` logs each commanded move's target, closing axis and status; ``env.trace_cands``
    dumps every ranked candidate before execution. With neither set this does nothing.
    """
    if env.trace_moves:  # debug: log each commanded move's target + closing + status
        _orig_move = arm.move

        def _traced_move(pose, *a, **k):  # noqa: ANN001, ANN202
            res = _orig_move(pose, *a, **k)
            st = getattr(getattr(res, "status", None), "value", res)
            q = np.asarray(getattr(pose, "quaternion_xyzw", [0, 0, 0, 1]), dtype=np.float64)
            x, y, z, w = q
            cl = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
            print(f"[trace-move] {st} tgt={np.round(np.asarray(getattr(pose, 'position_mm', [0, 0, 0])), 1)} "
                  f"closing={np.round(cl, 2)}", flush=True)
            return res

        arm.move = _traced_move  # type: ignore[method-assign]

    if env.trace_cands:  # dump all ranked candidates pre-execution (selector design)
        orch_c = service.runtime.orchestrator
        _orig_exec = orch_c._execute

        def _traced_exec(result, _orig=_orig_exec):  # noqa: ANN001, ANN202
            cands = getattr(result, "candidates", ()) or ()
            print(f"[cands] n={len(cands)}", flush=True)
            for j, c in enumerate(cands):
                gw = getattr(c, "grip_width_mm", None)
                sc = getattr(c, "score", None)
                print(f"[cands]  #{j} pos={np.round(np.asarray(c.position), 1)} "
                      f"grip_w={gw:.0f} score={sc:.3f} axis={np.round(np.asarray(c.axis), 2)}", flush=True)
            return _orig(result)

        orch_c._execute = _traced_exec  # type: ignore[method-assign]


def _build_g6_runtime(  # noqa: ANN202 (returns the opaque recovery-loop tuple consumed by run_gate)
    *, enable_g6: bool, homes: list, target_idx: int, agitate_amplitude_mm: float,
    redistribute_depth_mm: float, redistribute_offset_mm: float,
):
    """Build the agitate-hover pose and the recovery-loop tuple, or None.

    Without ``--g6`` this returns None and ``run_gate``'s ``_do_pick`` calls ``service.pick()`` directly.
    The hover sits 200 mm over the target so the agitate, clamped by the envelope and gated by
    ``SafetyPreflight``, stays inside the workspace. Returns
    ``(run_recovery_loop, hover_pose, profile, policy, orchestrator)`` or None.
    """
    if not enable_g6:
        return None
    from src.geometry import Frame, Pose
    from src.robot.grasping.motion.execution_policy import _quaternion_from_axes
    from src.robot.grasping.recovery.orchestrator import run_recovery_loop

    _hover_pos = np.asarray(homes[target_idx][0], dtype=np.float64) * 1000.0 + np.array([0.0, 0.0, 200.0])
    _hover_pose = Pose(
        position_mm=_hover_pos,
        quaternion_xyzw=_quaternion_from_axes(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        frame=Frame.BASE, label="g6_agitate_hover",
    )
    print(f"[g6] agitate hover (over target) TCP mm={np.round(_hover_pos, 1)}", flush=True)
    return (run_recovery_loop, _hover_pose, *_build_g6_recovery(
        agitate_amplitude_mm, center_mm=tuple(_hover_pos),
        contact_depth_mm=redistribute_depth_mm, sweep_offset_mm=redistribute_offset_mm,
    ))


def run_gate(runs: int = 10, *, prompt: str = "the red cube", headless: bool = True,
             data_dir: str | None = None, mode: str = "dense_clutter",
             record_log: str | None = None, debug_frames: str | None = None,
             blocking: bool = False, enable_g4: bool = False, enable_g12: bool = False,
             g4_rerank: bool = False, g4_rerank_weight: float = 0.3,
             enable_bin: bool = False, near_wall: bool = False, bin_half_width_mm: float = 190.0,
             bin_half_width_y_mm: float | None = None,
             bin_height_mm: float = 50.0, near_wall_offset_mm: float = 120.0,
             self_collision_arm: bool = False, finger_tool: bool = False, place_probe: bool = False,
             curobo_bin_world: bool = False,
             blocker_offset_mm: float = 35.0, approach_margin_mm: float = 20.0,
             enable_g6: bool = False, agitate_amplitude_mm: float = 30.0, ycb: bool = False,
             gso: bool = False,
             vision: bool = False, scene2: bool = False, collect: bool = False,
             recovery: bool = False, recovery_variety: "str | None" = None,
             seed: int | None = None,
             jitter_mm: float = 20.0, phase: str | None = None,
             randomize: bool = False, rand_seed: int | None = None,
             redistribute_depth_mm: float = 0.0, redistribute_offset_mm: float = 0.0,
             depth_source: str = "gt", grasp_depth_reference: str = "centre",
             depth_band_mm: float = 0.0,
             motion_planner: "str | None" = None, planner_owns_approach: bool = False,
             mask_completion: "str | None" = None) -> GateResult:
    """Run ``runs`` native dense picks.

    A pick passes only when the prompted object lifts at least the gate threshold and no distractor does.
    ``record_log``, ``debug_frames`` and, through ``main``, ``result_json`` are opt-in artifacts. The
    grasp-viewer can fire here because overhead RGB is present, so ``--debug-frames`` dumps an annotated
    multi-object frame per pick. The scene and feature flags are opt-in and off by default: ``blocking``
    adds a blocker in the red target's approach column, ``enable_g4`` stamps per-candidate corridor risk,
    and ``enable_g12`` runs the swept-volume approach validator.
    """
    service, arm, gripper, handles, cfg, target_idx, target_label = build_service(
        prompt=prompt, headless=headless, data_dir=data_dir, mode=mode,
        blocking=blocking, enable_g4=enable_g4, enable_g12=enable_g12,
        g4_rerank=g4_rerank, g4_rerank_weight=g4_rerank_weight,
        enable_bin=enable_bin, near_wall=near_wall, bin_half_width_mm=bin_half_width_mm,
        bin_half_width_y_mm=bin_half_width_y_mm,
        bin_height_mm=bin_height_mm, near_wall_offset_mm=near_wall_offset_mm,
        self_collision_arm=self_collision_arm, finger_tool=finger_tool, curobo_bin_world=curobo_bin_world,
        blocker_offset_mm=blocker_offset_mm, approach_margin_mm=approach_margin_mm, ycb=ycb, gso=gso,
        vision=vision, mask_completion=mask_completion, scene2=scene2, collect=collect,
        recovery=recovery,
        depth_source=depth_source, grasp_depth_reference=grasp_depth_reference,
        depth_band_mm=depth_band_mm,
        motion_planner=motion_planner, planner_owns_approach=planner_owns_approach,
    )
    env = RunnerEnv.from_env(vision=vision, view_height_mm=0.0)  # the WILLY_* knobs, parsed once
    calc = service.runtime.orchestrator.calculator  # read the corridor telemetry after each pick
    if debug_frames:
        service.enable_debug_image_rendering()
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions

    robot = require_robot(cfg)
    gate = robot.sim.scene_setup.gate
    park_q = np.asarray(robot.sim.park_joint_positions, dtype=np.float64)
    prims = [p for (p, _label) in handles.object_specs]
    objs = [SingleRigidPrim(p) for p in prims]
    homes = [o.get_world_pose() for o in objs]
    run_id = f"dense-{int(time.time())}"
    # Domain randomization: the three-way selection and the Replicator light setup live in _build_randomizer.
    rand_cfg, randomizer = _build_randomizer(
        randomize=randomize, seed=seed, rand_seed=rand_seed, jitter_mm=jitter_mm, collect=collect, ycb=ycb,
        gso=gso,
    )
    print(f"=== NATIVE DENSE pick: prompt={prompt!r} -> target={target_label!r} ({prims[target_idx]}); "
          f"{len(objs)} objects, mode={mode} ===", flush=True)
    _install_trace_hooks(service, arm, env)  # the WILLY_TRACE_MOVES and WILLY_TRACE_CANDS opt-in hooks

    # Wrap each pick in the real recovery loop, permitting container-agitate on a clutter jam, which the
    # approach validator reports as all sweeps blocked and the adapter maps to ALL_COLLIDED. Without --g6
    # the loop is skipped and `service.pick()` runs directly. The agitate motion is envelope-clamped and
    # gated by SafetyPreflight. The executor oscillates from the live TCP, and a jam refusal commands no
    # motion, so the arm sits at the park pose, which is behind the base and outside the reachable
    # workspace; SafetyPreflight then correctly rejects the agitate. So the pick callback parks first, for
    # the clean overhead perceive, then moves to a hover over the target that is inside the workspace
    # before returning, and the agitate shakes over the bin. The oscillation is the recovery action.
    _g6 = _build_g6_runtime(
        enable_g6=enable_g6, homes=homes, target_idx=target_idx,
        agitate_amplitude_mm=agitate_amplitude_mm,
        redistribute_depth_mm=redistribute_depth_mm, redistribute_offset_mm=redistribute_offset_mm,
    )

    def _tgt_xyz_mm():  # noqa: ANN202 (the target object's current world pose in mm (diagnosis))
        return np.round(np.asarray(objs[target_idx].get_world_pose()[0]) * 1000.0, 1)

    # Runner-local next_target recovery. The core recovery's next_target does nothing here, because it
    # re-picks the same locked target and fails the same way. This records a next_target action, switches
    # `service.set_target_label` to a graspable neighbour and re-picks, which produces the genuine
    # recovered_success episodes the recovery policy trains on; its reward is
    # `recovery_actions[-1].outcome == "recovered_success"`. The scene is an oversized target that yields
    # no candidates plus normal-size neighbours. Active only with --collect-recovery next_target.
    _orig_target_label = target_label
    _neighbor_items = [(j, lbl) for j, (_p, lbl) in enumerate(handles.object_specs) if j != target_idx]
    _recov: dict = {"acts": [], "final_idx": target_idx}

    def _rep_ok(rep) -> bool:  # noqa: ANN001 (report.outcome == succeeded)
        return getattr(getattr(rep, "outcome", None), "value", None) == "succeeded"

    def _do_pick():  # noqa: ANN202 (returns (report, trail_or_None))
        if recovery_variety == "next_target":
            _recov["acts"], _recov["final_idx"] = [], target_idx
            service.set_target_label(_orig_target_label)
            rep = service.pick()
            if not _rep_ok(rep):
                _pr = getattr(rep, "pick_report", None)
                reason = getattr(getattr(_pr, "outcome", None), "value", None) or "no_candidates_generated"
                for alt_idx, alt_lbl in _neighbor_items:
                    _recov["acts"].append({
                        "action": "next_target", "executed": True, "failure_reason": reason,
                        "outcome": "completed", "plan_reason": "runner:next_target", "step_result": "completed"})
                    service.set_target_label(alt_lbl)
                    gripper.open()  # clean re-perceive from park before the neighbour re-pick, as in
                    arm.move_to_joints(JointPositions(park_q))  # _pick_then_hover, so the pick starts fresh
                    rep = service.pick()
                    _recov["final_idx"] = alt_idx
                    if _rep_ok(rep):
                        break
            service.set_target_label(_orig_target_label)  # reset for the next episode
            return rep, None
        if _g6 is None:
            if env.trace_calc:
                print(f"[target] pre-pick pos_mm={_tgt_xyz_mm()}", flush=True)
            rep = service.pick()
            if env.trace_calc:
                print(f"[target] post-pick pos_mm={_tgt_xyz_mm()}", flush=True)
            return rep, None
        run_loop, hover_pose, prof, pol, orch_r = _g6

        def _pick_then_hover():  # park to perceive, pick, then hover over the bin so the agitate is in-workspace
            arm.move_to_joints(JointPositions(park_q))  # park for the clean overhead perceive (each attempt)
            report = service.pick()
            arm.move(hover_pose)
            return _G6RecoveryAdapter(report)

        final, trail = run_loop(
            pick=_pick_then_hover, profile=prof, policy=pol,
            orchestrator=orch_r, frame_acquirer=lambda: None, arm=arm,
        )
        return final.report, trail

    results = []
    for i in range(runs):
        # Clean reset: release the jaw, park the arm for an unoccluded overhead perceive, then re-place and
        # settle every object. The typed joint move resets the continuity reference across that jump.
        gripper.open()
        arm.move_to_joints(JointPositions(park_q))
        # Domain randomization: `DomainRandomizer` samples the per-episode pose in numpy, deterministically,
        # and applies it through set_world_pose, while colour and lighting go through Replicator. When it is
        # disabled, `sample_pose` returns the home poses unchanged and the visual pass does nothing, so the
        # reset lands back at the home poses.
        _home_pos_mm = [tuple(np.asarray(hp, dtype=np.float64) * 1000.0) for (hp, _hq) in homes]
        _home_quat = [tuple(np.asarray(hq, dtype=np.float64)) for (_hp, hq) in homes]
        _ep_pose = randomizer.sample_pose(i, _home_pos_mm, _home_quat)
        for o, _pos_mm, _quat in zip(objs, _ep_pose.positions_mm, _ep_pose.orientations_wxyz):
            o.set_world_pose(position=np.asarray(_pos_mm, dtype=np.float64) / 1000.0,
                             orientation=np.asarray(_quat, dtype=np.float64))
            try:
                o.set_linear_velocity(np.zeros(3))
                o.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001
                pass
        # Colour: a numpy-sampled palette applied through OmniPBR, with names that match the colour. Skipped
        # on YCB, where the objects are real textured meshes, so object_set keeps the spawn names there.
        _color_names: list[str | None] = (
            randomizer.apply_colors(objs, i, target_idx)
            if (rand_cfg.any_color and not ycb) else [None] * len(objs)
        )
        _light = randomizer.apply_lighting(i) if rand_cfg.any_lighting else {}
        # object_set provenance: re-coloured distractor names keep the leakage audit non-vacuous while the
        # target keeps its identity. Falls back to the spawn names when colour is off and on YCB.
        _ep_object_set: list[str] | None = None
        if any(c is not None for c in _color_names):
            _ep_object_set = [
                (f"{_color_names[j]} cube" if _color_names[j] is not None else lbl)
                for j, (_p, lbl) in enumerate(handles.object_specs)
            ]
        arm.session.step_n(120 if (ycb or gso) else 30)  # a real YCB or GSO mesh needs longer to settle
        z0 = [float(np.asarray(o.get_world_pose()[0])[2]) for o in objs]
        # The movable blocker is the last spec, appended by `_blocking_specs`. Track its XY before and after
        # the pick and recovery, so the measurement shows whether the contact redistribute moved it.
        blk_xy0 = np.asarray(objs[-1].get_world_pose()[0], dtype=np.float64)[:2] * 1000.0 if blocking else None
        if i == 0:  # confirm the objects settled on the table, and the target's settled orientation
            tpos, tquat = objs[target_idx].get_world_pose()
            print(f"[settled] object z0 mm={[round(z * 1000.0, 1) for z in z0]} | target={target_label!r} "
                  f"pos_mm={np.round(np.asarray(tpos) * 1000.0, 1)} quat_wxyz={np.round(np.asarray(tquat), 3)}",
                  flush=True)
        report, trail = _do_pick()
        if env.trace_calc:  # diagnose: candidate breakdown + the executed grasp geometry
            print(f"[calc] last_telemetry={dict(calc.last_telemetry or {})}", flush=True)
            _pr = getattr(report, "pick_report", None)
            _eg = getattr(_pr, "executed_grasp", None)
            _cands = getattr(_eg, "candidates", None) or ()
            if _cands:
                _c = _cands[0]
                print(f"[grasp] pos={np.round(np.asarray(_c.position), 1)} "
                      f"grip_w={getattr(_c, 'grip_width_mm', None)} score={getattr(_c, 'score', None)}", flush=True)
            try:
                print(f"[grasp] gripper_width_after_pick={gripper.get_width_mm():.1f}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        rec_actions = [e.plan_action.value for e in trail.entries] if trail is not None else []
        rec_terminal = trail.terminal_reason if trail is not None else None
        rec_outcomes = [e.outcome for e in trail.entries] if trail is not None else []
        rec_agitate_ok = bool(
            trail is not None
            and any(e.plan_action.value == "container_agitate" and e.executed for e in trail.entries)
        )
        z1 = [float(np.asarray(o.get_world_pose()[0])[2]) for o in objs]
        lifts = [(z1[k] - z0[k]) * 1000.0 for k in range(len(objs))]
        # A next_target recovery scores the switched object, the graspable neighbour the runner re-picked,
        # rather than the oversized original target, so the lift and the success reflect the recovered pick.
        _score_idx = _recov["final_idx"] if recovery_variety == "next_target" else target_idx
        target_lift = lifts[_score_idx]
        distractor_lifts = [lifts[k] for k in range(len(objs)) if k != _score_idx]
        max_distractor = max(distractor_lifts) if distractor_lifts else 0.0
        target_lifted = target_lift >= gate.lift_threshold_mm
        distractor_disturbed = max_distractor >= gate.lift_threshold_mm
        passed = bool(target_lifted and not distractor_disturbed)
        outcome = getattr(report, "outcome", None)
        outcome_value = getattr(outcome, "value", outcome)
        succeeded = outcome_value == "succeeded"
        # The top-level outcome collapses several reasons into "execution_failed". The precise reason, such
        # as approach_path_blocked, sits on the pick_report, so it is surfaced here for the measurement.
        pr = getattr(report, "pick_report", None)
        pick_outcome = getattr(getattr(pr, "outcome", None), "value", None)
        feat: dict = {"outcome": outcome_value, "pick_outcome": pick_outcome}
        if recovery_variety == "next_target" and _recov["acts"]:
            # Stamp the runner-local next_target trail onto the report so the record serializer writes
            # trainable recovery_actions. The sim ground truth decides: recovered_success only when the
            # switched-to neighbour lifted, which `target_lifted` above measures, otherwise
            # recovered_failure.
            from dataclasses import replace as _dc_replace
            _nt = [dict(a) for a in _recov["acts"]]
            _nt[-1]["outcome"] = "recovered_success" if target_lifted else "recovered_failure"
            _nt[-1]["sim_recovered"] = bool(target_lifted)
            report = _dc_replace(report, recovery_actions=tuple(_nt))
            feat["recovery_actions"] = [a["action"] for a in _nt]
            feat["recovered_success"] = bool(target_lifted)
        if place_probe and _g6 is None and target_lifted:  # does the friction grasp survive a lateral
            # transport to a place pose, or slip? Transport the held object +X at the retreat height, then
            # lower and release, measuring how far the object followed the gripper against how far it lagged.
            from src.geometry import Frame, Pose
            _DX, _PZ = 200.0, 60.0  # +X transport (mm) at retreat height; release height above the table (mm)
            tcp0 = arm.get_tcp_pose()
            obj0 = np.asarray(objs[target_idx].get_world_pose()[0], dtype=np.float64) * 1000.0
            px, py, pz = float(tcp0.position_mm[0]) + _DX, float(tcp0.position_mm[1]), float(tcp0.position_mm[2])
            _q = np.asarray(tcp0.quaternion_xyzw, dtype=np.float64)
            mv = arm.move(Pose(position_mm=np.array([px, py, pz]), quaternion_xyzw=_q.copy(),
                               frame=Frame.BASE, label="place-transit"))
            obj1 = np.asarray(objs[target_idx].get_world_pose()[0], dtype=np.float64) * 1000.0
            transported = float(np.linalg.norm(obj1[:2] - obj0[:2]))  # how far the object moved with the gripper
            z_drop = float(obj0[2] - obj1[2])                          # slipped down during transit?
            feat["place_move_ok"] = getattr(getattr(mv, "status", None), "value", None) == "executed"
            feat["place_obj_transport_mm"] = round(transported, 1)
            feat["place_z_drop_mm"] = round(z_drop, 1)
            feat["place_held"] = bool(transported > 0.6 * _DX and z_drop < 40.0)
            arm.move(Pose(position_mm=np.array([px, py, _PZ]), quaternion_xyzw=_q.copy(),
                          frame=Frame.BASE, label="place-down"))
            gripper.open()
            arm.session.step_n(60)
            obj2 = np.asarray(objs[target_idx].get_world_pose()[0], dtype=np.float64) * 1000.0
            feat["place_land_err_mm"] = round(float(np.linalg.norm(obj2[:2] - np.array([px, py]))), 1)
            print(f"[place-probe] move_ok={feat['place_move_ok']} transported={transported:.1f}mm "
                  f"z_drop={z_drop:.1f}mm held={feat['place_held']} land_err={feat['place_land_err_mm']:.1f}mm",
                  flush=True)
        if enable_g4:  # the measured per-candidate corridor risk for this pick
            tele = calc.last_telemetry or {}
            feat["corridor_max_blockage"] = tele.get("corridor_max_blockage_confidence")
            feat["corridor_blocked"] = tele.get("corridor_blocked_count")
            feat["corridor_analyzed"] = tele.get("corridor_candidates_analyzed")
        if g4_rerank:  # did the corridor risk demote the geometric top-1?
            rrt = getattr(pr, "uncertainty_rerank_telemetry", None)
            feat["rerank_applied"] = getattr(rrt, "applied", None)
            feat["rerank_skip"] = getattr(rrt, "skip_reason", None)
            feat["rerank_top1_changed"] = getattr(rrt, "top1_changed", None)
            feat["rerank_winner_uncertainty"] = getattr(rrt, "winner_uncertainty", None)
        if enable_bin:  # did a bin-wall self-collision reject the approach this pick?
            _br = getattr(arm, "_bin_reject", None)
            feat["bin_wall_reject"] = _br.get("reason") if isinstance(_br, dict) else None
            if isinstance(_br, dict):
                _br["reason"] = None  # reset for the next pick
        if self_collision_arm:  # any self-collision reject pair, link|link included, for the no-false-reject check
            _br = getattr(arm, "_bin_reject", None)
            feat["sc_reject"] = _br.get("any") if isinstance(_br, dict) else None
            if isinstance(_br, dict):
                _br["any"] = None  # reset for the next pick
        if enable_g6:  # which recovery actions ran and executed, and the loop's terminal reason
            feat["recovery_actions"] = rec_actions
            feat["recovery_outcomes"] = rec_outcomes
            feat["recovery_terminal"] = rec_terminal
            feat["agitated"] = "container_agitate" in rec_actions
            feat["agitate_executed"] = rec_agitate_ok
            if blk_xy0 is not None:  # did the contact redistribute move the blocker?
                blk_xy1 = np.asarray(objs[-1].get_world_pose()[0], dtype=np.float64)[:2] * 1000.0
                feat["blocker_disp_mm"] = round(float(np.linalg.norm(blk_xy1 - blk_xy0)), 1)
                # The sim ground truth decides: a recovery counts as a success only if the target lifted.
                # A pipeline 'recovered_success' that slipped is not a real recovery.
                feat["recovered_success"] = rec_terminal == "recovered_success" and target_lifted
            # The recovery loop returns the trail separately, so report.recovery_actions is empty here.
            # Serialize the per-step trail onto the report so the record serializer writes trainable
            # recovery_actions, applying the same sim ground truth: demote a pipeline 'recovered_success'
            # to 'recovered_failure' when the target did not lift, so the reward reflects a genuine lift.
            if trail is not None:
                from dataclasses import replace as _dc_replace
                from src.robot.grasping.recovery.orchestrator import (
                    recovery_actions_from_trail,
                )

                _acts = [dict(a) for a in recovery_actions_from_trail(trail)]
                if _acts and _acts[-1].get("outcome") == "recovered_success" and not target_lifted:
                    _acts[-1]["outcome"] = "recovered_failure"
                    _acts[-1]["sim_recovered"] = False
                report = _dc_replace(report, recovery_actions=tuple(_acts))
        results.append({"run": i, "succeeded": succeeded, "target_lift_mm": round(target_lift, 1),
                        "max_distractor_lift_mm": round(max_distractor, 1), "passed": passed, **feat})
        msg = (f"RUN {i}: target_lift={target_lift:.1f} max_distractor={max_distractor:.1f} "
               f"succeeded={succeeded} outcome={outcome_value} pick_outcome={pick_outcome} passed={passed}")
        if enable_g4:
            msg += (f" | G4 max_blockage={feat.get('corridor_max_blockage')} "
                    f"blocked={feat.get('corridor_blocked')}/{feat.get('corridor_analyzed')}")
        if g4_rerank:
            msg += (f" | RERANK applied={feat.get('rerank_applied')} top1_changed={feat.get('rerank_top1_changed')} "
                    f"skip={feat.get('rerank_skip')} win_unc={feat.get('rerank_winner_uncertainty')}")
        if enable_bin:
            msg += f" | BIN wall_reject={feat.get('bin_wall_reject')}"
        if self_collision_arm:
            msg += f" | SC reject={feat.get('sc_reject')}"
        if place_probe:
            msg += (f" | PLACE transported={feat.get('place_obj_transport_mm')}mm "
                    f"z_drop={feat.get('place_z_drop_mm')}mm held={feat.get('place_held')} "
                    f"land_err={feat.get('place_land_err_mm')}mm")
        if enable_g6:
            msg += (f" | G6 agitated={feat.get('agitated')} executed={rec_agitate_ok} "
                    f"outcomes={rec_outcomes} terminal={rec_terminal} blocker_disp={feat.get('blocker_disp_mm')}")
        print(msg, flush=True)
        # Leakage and stratification fields for build-dataset, which make the leakage audits non-vacuous
        # and the splits stratified. A unique scene_id per episode keeps any train and test split
        # leakage-safe; object_set, camera_pose_hash, timestamp and attempt_index feed the audits in
        # `rl/leakage.py`. Written only on the collect path.
        _leak: dict = {}
        if collect:
            import hashlib

            _leak = {
                "scene_id": f"{run_id}-s{seed}-ep{i:04d}",
                # The coarse scene family, scene kind plus target identity, for pair-grouping and the
                # leakage-safe group-aware split; scene_id stays unique per episode for the camera audit.
                "scene_family_id": _scene_family_id(
                    kind=("gso" if gso else "ycb" if ycb else "cube"), target_label=str(target_label)),
                "camera_pose_hash": hashlib.sha256(
                    f"{target_label}|{mode}|{run_id}".encode()
                ).hexdigest()[:16],
                # Per-episode re-coloured cube names give a diverse object_set, so the leakage audit is
                # non-vacuous. Falls back to the spawn names, as on YCB, where object_set legitimately
                # repeats and the audit then flags the sim's limited objects.
                "object_set": _ep_object_set or [lbl for (_p, lbl) in handles.object_specs],
                # An integer-index timestamp makes the time_bleed audit skip it: a stratified-shuffled pack
                # has no time-order leakage. Same convention as the canonical replay packs.
                "timestamp": float(i),
                "attempt_index": i,
                # The OPE reward (compute_record_reward in rl/ope.py) requires extra.cycle_time_s, so it is
                # sourced from the pick's wall time and the collected pack is OPE-ready.
                "cycle_time_s": float(getattr(report, "telemetry", {}).get("attempt_wall_time_s", 0.0) or 0.0),
            }
            # The outcome label comes from the sim ground truth (sim_lifted), so derive_outcome_class sees
            # a real success and failure mix instead of the pipeline's self-report, which says 'succeeded'
            # even for a slip.
            _ftc = _failure_taxonomy_from_sim(str(outcome_value), str(pick_outcome), bool(target_lifted))
            if _ftc is not None:
                _leak["failure_taxonomy_class"] = _ftc
            if phase is not None:
                _leak["phase"] = phase
            # Stamp the randomization parameters that were actually applied, so the dataset is reproducible
            # and auditable from the records alone, not only by re-seeding.
            if rand_cfg.enabled:
                _leak["randomization"] = randomizer.provenance(i, _ep_pose, _color_names, _light)
            # Surface the recovery evidence in the fields
            # `train_sequencing._derive_action_from_successor` reads, extra.recovery_attempted and
            # extra.recovery_action, so a successor that carries a recovery maps to the recover action
            # rather than to grasp or reobserve. The record already carries top-level recovery_actions;
            # this mirrors the last one into extra. It is additive and only on sim records, so the
            # canonical packs, which have no such field, are unchanged. The trainer is deliberately not
            # taught to read top-level recovery_actions, because that would alter the derivation for
            # replay_dense_canonical_v1.
            _ra = getattr(report, "recovery_actions", ()) or ()
            if _ra and isinstance(_ra[-1], dict):
                _leak["recovery_attempted"] = True
                _leak["recovery_action"] = str(_ra[-1].get("action"))
        write_pick_artifacts(
            report, attempt_id=f"{run_id}-{i:04d}",
            record_log=record_log, debug_dir=debug_frames,
            debug_png=service.last_debug_image_png,
            # Merge the active features' signals from `feat`, the approach-validator pick_outcome, the
            # corridor_* values and the recovery_* values, into the record's extra, so the
            # GraspAttemptRecord carries each fired feature's trace. `report.telemetry` does not carry
            # corridor_* or pick_outcome on its own.
            extra={"sim_runner": "dense", "sim_mode": mode, "sim_prompt": prompt, "sim_target": target_label,
                   "sim_lift_mm": round(target_lift, 2), "sim_lifted": bool(target_lifted),
                   "sim_max_distractor_lift_mm": round(max_distractor, 2), "sim_gate_passed": passed,
                   **_leak, **feat},
        )
    n_pass = sum(1 for r in results if r["passed"])
    gate_ok = gate_passed(n_pass, runs, gate.pass_fraction)
    print(f"DENSE GATE: {n_pass}/{runs} passed -> gate_passed={gate_ok}", flush=True)
    return GateResult(
        runs=runs, passed=n_pass, results=results, gate_passed=gate_ok, target=target_label,
    )


def run_build_stack(
    *, headless: bool = True, data_dir: str | None = None,
    order: tuple[str, ...] = ("red cube", "green cube", "blue cube"),
    base_xy_mm: tuple[float, float] = (600.0, 0.0),
    sizes: "dict[str, tuple[float, float, float]] | None" = None,
    place_offset_mm: float = 0.0,
    record_path: str | None = None,
) -> int:
    """Pick the labelled cubes in ``order`` and place them into a tower at ``base_xy_mm``.

    Reuses the friction grasp with ``move`` and ``open`` and no attachment constraint; ``--place-probe``
    is the check that the grasp survives a 200 mm lateral transport. ``set_target_label`` restricts each
    pick to one cube, which controls the order. ``place_offset_mm`` is the per-cube +X stagger: 0 gives
    a centred, aligned, stable tower, and a value larger than the cube half-width gives a leaning
    staircase whose top cube overhangs its support and topples. That is the aligned against misaligned
    contrast, built on the 30x30x50 grasp that works. Returns 0 only when every cube ends on the base
    with a strictly increasing Z.
    """
    from src.geometry import Frame, Pose
    from src.robot.core import JointPositions

    service, arm, gripper, handles, cfg, _ti, _tl = build_service(
        headless=headless, data_dir=data_dir, clutter_sizes=sizes,
    )
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]  # after SimulationApp boots

    park_q = np.asarray(require_robot(cfg).sim.park_joint_positions, dtype=np.float64)
    objs = {label: SingleRigidPrim(prim) for (prim, label) in handles.object_specs}
    bx, by = base_xy_mm
    cube_h = {label: (float(sizes[label][2]) if sizes and label in sizes else 50.0) for label in objs}
    # Place the cubes about 110 mm apart; a tighter cluster lets the open jaw smash a neighbour on approach
    # and fling it. Red at -95 and green at +5 are the known-good spots. Blue moves off its spawn at +Y=105,
    # which needs a shoulder-pan flip that trips the IK-jump guard at 3.62 rad against a 3.50 limit, to a
    # clear, flip-free, reachable spot. This routine defines its own scene, and each cube spawns at half
    # its own height so it sits on the table.
    cluster = {"red cube": (450.0, -95.0), "green cube": (450.0, 5.0), "blue cube": (560.0, -110.0)}

    # Optional MP4: film a cinematic side three-quarter view of the whole build by monkey-patching
    # session.step to grab the cine cam each step, as run_dense_demo does; capture 1 of every 5 steps to
    # keep the clip short.
    _frames: list[np.ndarray] = []
    _rec_on = {"v": False}
    if record_path:
        from isaacsim.core.utils.viewports import set_camera_view  # type: ignore[import-not-found]
        from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]
        _session, _app, _n = arm.session, getattr(arm.session, "app", None), {"v": 0}
        _cine = Camera(prim_path="/World/Cameras/Cinematic",
                       position=np.array([2.05, -1.82, 1.68], dtype=np.float64), resolution=(1100, 760))
        _cine.initialize()
        _cine.add_distance_to_image_plane_to_frame()
        try:
            _cine.set_clipping_range(0.05, 1.0e6)
        except Exception:  # noqa: BLE001
            pass
        set_camera_view(eye=[2.05, -1.82, 1.68], target=[0.55, -0.03, 0.08],
                        camera_prim_path="/World/Cameras/Cinematic")
        _orig_step = _session.step

        def _cap_step(dt=None, *, render=False):  # noqa: ANN001, ANN202 (film the cine cam each step)
            _orig_step(dt, render=True)
            if _app is not None:
                _app.update()
            if _rec_on["v"]:
                _n["v"] += 1
                if _n["v"] % 5 == 0:
                    _fr = _cine.get_current_frame()
                    _rgb = _fr.get("rgb") if isinstance(_fr, dict) else None
                    if _rgb is not None:
                        _frames.append(np.asarray(_rgb)[..., :3].copy())

        _session.step = _cap_step  # type: ignore[assignment]
        for _ in range(60):
            _session.step(render=True)

    _mode = "ALIGNED tower (stable)" if place_offset_mm == 0.0 else f"STAGGERED +{place_offset_mm:.0f}mm (expect topple)"
    print(f"=== H3.3 BUILD-STACK [{_mode}]: order={order} base_xy={base_xy_mm} ===", flush=True)
    gripper.open()
    arm.move_to_joints(JointPositions(park_q))
    for label, o in objs.items():  # place into the reachable cluster + zero velocity + settle (clean home)
        cx, cy = cluster.get(label, tuple(float(v) for v in np.asarray(o.get_world_pose()[0])[:2] * 1000.0))
        o.set_world_pose(position=np.array([cx, cy, cube_h[label] / 2.0], dtype=np.float64) / 1000.0,
                         orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        try:
            o.set_linear_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
    arm.session.step_n(60)

    def _placed(n: int, tag: str) -> None:  # topple diagnosis: the already-placed cubes' xyz (mm) per sub-step
        st = {order[k]: np.round(np.asarray(objs[order[k]].get_world_pose()[0]) * 1000.0, 1).tolist() for k in range(n)}
        print(f"  [diag] {tag}: {st}", flush=True)

    _rec_on["v"] = True  # start filming the build (the reset is done; the cubes are at their clean home)
    cum = 0.0  # cumulative stack height (mm): each cube's centre sits on the cubes already placed
    for i, label in enumerate(order):
        report = None
        for _try in range(2):  # one retry: a transient drive miss shouldn't abort the whole stack
            arm.move_to_joints(JointPositions(park_q))  # clean overhead perceive each step
            service.set_target_label(label)
            report = service.pick()
            if getattr(getattr(report, "outcome", None), "value", None) == "succeeded":
                break
        if getattr(getattr(report, "outcome", None), "value", None) != "succeeded":
            print(f"[stack] {label}: pick FAILED ({getattr(report, 'outcome', None)}) after retry -> abort", flush=True)
            break
        # Centre the cube, not the TCP, on the base in XY: an off-centre grasp leans the stack and the lean
        # topples the third cube. Z is not corrected, because the grasp Z-offset measured at the retreat
        # does not hold during the descent, so correcting Z releases too high and the cube bounces away.
        cube_xy = np.asarray(objs[label].get_world_pose()[0], dtype=np.float64)[:2] * 1000.0
        tcp = arm.get_tcp_pose()
        off = cube_xy - np.asarray(tcp.position_mm, dtype=np.float64)[:2]  # cube = tcp + off
        q = np.asarray(tcp.quaternion_xyzw, dtype=np.float64)  # top-down holding orientation
        tx, ty = bx - float(off[0]) + i * place_offset_mm, by - float(off[1])  # +X stagger: a leaning staircase
        center_z = cum + cube_h[label] / 2.0  # this cube's centre sits on the cumulative stack height
        _placed(i, f"{label} picked (pre-place)")
        above = Pose(position_mm=np.array([tx, ty, center_z + 80.0]), quaternion_xyzw=q.copy(),
                     frame=Frame.BASE, label="place-above")
        arm.move(above)  # transport over the stack at height (held by friction)
        _placed(i, "after place-above")
        arm.move(Pose(position_mm=np.array([tx, ty, center_z + 2.0]), quaternion_xyzw=q.copy(),
                      frame=Frame.BASE, label="place-down"))  # descend to a gentle release just above the stack top
        _placed(i, "after place-down")
        gripper.open()
        arm.session.step_n(80)  # let the placed cube + stack settle before the next pick
        _placed(i, "after release")
        # Retreat only between picks: lifting the open 2F-85 jaw off the cube nudges it, and on a tall,
        # fragile tower of same-size cubes that nudge topples the lot. A one- or two-cube base survives the
        # vertical lift, and the top cube needs no retreat because no pick follows it, so it is left settled.
        if i < len(order) - 1:
            arm.move(above, linear=True)  # vertical clear for the next pick (clean on the 1-/2-cube base)
        pos = np.asarray(objs[label].get_world_pose()[0]) * 1000.0
        print(f"[stack] {label}: placed #{i} h={cube_h[label]:.0f} target_center_z={center_z:.0f} "
              f"tx,ty=({tx:.0f},{ty:.0f}) off=({off[0]:.0f},{off[1]:.0f}) -> pos_mm={np.round(pos, 1)}", flush=True)
        cum += cube_h[label]

    finals = {label: np.asarray(o.get_world_pose()[0]) * 1000.0 for label, o in objs.items()}
    on_base = {lbl: p for lbl, p in finals.items()
               if lbl in order and float(np.linalg.norm(p[:2] - np.array([bx, by]))) < 40.0}
    zs = sorted(float(p[2]) for p in on_base.values())
    tower = bool(len(on_base) == len(order) and all(zs[k + 1] - zs[k] > 20.0 for k in range(len(zs) - 1)))
    print(f"=== BUILD-STACK {'OK' if tower else 'FAIL'}: {len(on_base)}/{len(order)} on base, "
          f"tower_height={(max(zs) + cube_h[order[-1]] / 2.0) if zs else 0.0:.0f}mm, "
          f"Zs={[round(z, 1) for z in zs]} ===", flush=True)
    for label in order:
        print(f"  {label}: pos_mm={np.round(finals[label], 1)}", flush=True)
    if record_path:
        for _ in range(40):  # hold on the finished stack (still filming) before stitching
            arm.session.step(render=True)
        _rec_on["v"] = False
        if _frames:
            from pathlib import Path

            import cv2  # type: ignore[import-not-found]
            _out = Path(record_path)
            _out.parent.mkdir(parents=True, exist_ok=True)
            _h, _w = _frames[0].shape[:2]
            _wr = cv2.VideoWriter(str(_out), cv2.VideoWriter_fourcc(*"mp4v"), 18.0, (_w, _h))  # type: ignore[attr-defined]
            for _f in _frames:
                _wr.write(np.ascontiguousarray(_f[..., ::-1]))  # RGB -> BGR
            _wr.release()
            for _i, _idx in enumerate(np.linspace(0, len(_frames) - 1, 4).astype(int)):
                cv2.imwrite(str(_out.with_name(f"{_out.stem}_key{_i}.png")),
                            np.ascontiguousarray(_frames[int(_idx)][..., ::-1]))
            print(f"=== MP4: {len(_frames)} frames @18fps ({len(_frames) / 18.0:.1f}s) -> {_out} (+4 PNGs) ===", flush=True)
        else:
            print("=== MP4: no frames captured (cine RGB never populated) ===", flush=True)
    return 0 if tower else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy native dense-clutter pick (Isaac).")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--prompt", type=str, default="the red cube")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-headless", action="store_true", help="alias for --gui")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--mode", type=str, default="dense_clutter",
                    choices=["easy", "auto", "dense_clutter", "dense_autonomous"],
                    help="P2: grasp mode (dense_clutter = the native dense path; dense_autonomous = C2 full "
                         "perceive->refine->verify->recover loop; easy/auto for comparison)")
    ap.add_argument("--record-log", type=str, default=None,
                    help="P0: append one GraspAttemptRecord JSONL line per pick to this path")
    ap.add_argument("--debug-frames", type=str, default=None,
                    help="P0: dump the grasp-point overlay PNG per pick into this dir")
    ap.add_argument("--result-json", type=str, default=None,
                    help="P1: write the per-run result dict (for the mode-matrix harness) to this path")
    ap.add_argument("--blocking", action="store_true",
                    help="P3: add a blocker in the red target's +X approach column (for G4/G12)")
    ap.add_argument("--g4", action="store_true",
                    help="P3: enable the G4 corridor-risk producer (stamp + measure per-candidate blockage)")
    ap.add_argument("--g12", action="store_true",
                    help="P3: enable the G12 approach/retreat swept-volume validator (refuse/avoid)")
    ap.add_argument("--g4-rerank", action="store_true",
                    help="H2.1a: wire the G4 uncertainty-rerank CONSUMER (implies --g4) so a blocked candidate "
                         "is DEMOTED by its corridor risk; co-activates the synthetic U2 context (metadata only)")
    ap.add_argument("--g4-rerank-weight", type=float, default=0.3,
                    help="H2.1a: G4 rerank subtractive penalty weight (schema-clamped [0, 0.5]); tune on-box")
    ap.add_argument("--bin", action="store_true",
                    help="H3.1: enclose the clutter in 4 bin/tray walls (FixedCuboid prims + SelfCollision fixtures)")
    ap.add_argument("--near-wall", action="store_true",
                    help="H3.1: shift the target toward the +X wall -> a top-down grasp's tool capsule enters the "
                         "wall -> self-collision REJECT (the goal-1 proof). Without it the centred target is the "
                         "no-false-reject case (>=8/10).")
    ap.add_argument("--bin-half-width", type=float, default=190.0,
                    help="H3.1b: bin inner half-extent (mm) in x (and y unless --bin-half-width-y is set)")
    ap.add_argument("--bin-half-width-y", type=float, default=None,
                    help="H3.9d: SEPARATE bin inner half-extent (mm) in y -> author the KLT RECTANGLE (e.g. "
                         "--bin-half-width 140 --bin-half-width-y 90 puts the long axis along base X / the open-span)")
    ap.add_argument("--finger-tool", action="store_true",
                    help="H3.9d: tool_model='finger' -> model the 2F-85 descending fingers as a thin capsule along "
                         "the grasp CLOSING axis (measured radius 16mm, span 150mm) instead of the r=70 bounding "
                         "cylinder; the faithful, rotation-aware bin-wall footprint. Default-off byte-identical.")
    ap.add_argument("--bin-height", type=float, default=50.0,
                    help="H3.1b: wall height (mm); LOW so the arm links clear it while the low tool still hits a "
                         "near-wall (the reject proof)")
    ap.add_argument("--near-wall-offset", type=float, default=120.0,
                    help="H3.1b: +X shift (mm) of the target for the near-wall reject case")
    ap.add_argument("--self-collision-arm", action="store_true",
                    help="H3.1c: enable the UR-DH arm-link capsules (L2.7 hook) + the measured 180deg base-frame "
                         "reconcile so the guard checks ARM links (not just the tool) vs the bin walls. The sim is "
                         "a UR5e but vendor!='ur' so this opt-in is required; default-off byte-identical.")
    ap.add_argument("--curobo-bin-world", action="store_true",
                    help="H3.2 re-open: register the --bin walls into cuRobo's collision WORLD so the planner "
                         "ACTIVELY threads the gripper into the bin AROUND the walls (the bin becomes 'really real' "
                         "for the planner, not just a physical prim + post-hoc Coal reject). Needs --bin + curobo; "
                         "default-off byte-identical.")
    ap.add_argument("--place-probe", action="store_true",
                    help="H3.3 discriminance: after a successful pick, transport the held object +200mm laterally "
                         "then lower+release; measure whether the friction grasp survives the lateral transport "
                         "(held) or slips. The make-or-break read-only check before building a PLACE capability.")
    ap.add_argument("--build-stack", action="store_true",
                    help="H3.3: pick the 3 cubes IN ORDER (red->green->blue) and PLACE them into a tower at a base "
                         "(pick-and-place assembly). Reuses the friction grasp+move+open. Add --unordered to stagger.")
    ap.add_argument("--unordered", action="store_true",
                    help="H3.3: stagger each placed cube +20mm in X -> a leaning staircase whose top cube overhangs "
                         "its support and TOPPLES (the placement-based ordered-matters contrast vs the aligned tower).")
    ap.add_argument("--record", type=str, default=None,
                    help="H3.3: record a cinematic MP4 of the build to this path (e.g. logs/demo/build_stack.mp4).")
    ap.add_argument("--blocker-offset", type=float, default=35.0,
                    help="P3: blocker distance (mm) along +X from the red target (tune on-box)")
    ap.add_argument("--approach-margin", type=float, default=20.0,
                    help="P3: G12 collision margin (mm); larger flags farther blockers (conservative)")
    ap.add_argument("--g6", action="store_true",
                    help="P3: enable G6 container-agitate recovery on a clutter jam (ALL_COLLIDED)")
    ap.add_argument("--agitate-amplitude", type=float, default=30.0,
                    help="P3: G6 container-agitate amplitude (mm), envelope-clamped")
    ap.add_argument("--redistribute-depth-mm", type=float, default=0.0,
                    help="C3: G6 contact redistribute: descend this far toward the object layer + sweep the "
                         "approach corridor to push a movable blocker out (default 0 = legacy air-shake)")
    ap.add_argument("--redistribute-offset-mm", type=float, default=0.0,
                    help="C3: offset the contact sweep start +X off the target (so the descent clears the target)")
    ap.add_argument("--ycb", action="store_true",
                    help="P2.D: real YCB objects (referenced USDs) instead of procedural cubes")
    ap.add_argument("--gso", action="store_true",
                    help="RL-diversity: real Aletheia GSO scanned objects (toys/tools) as the clutter scene -> "
                         "varied real objects that genuinely fail sometimes = MIXED per-target outcomes for the RL "
                         "ranking/candidate policies. Roster via WILLY_GSO_ROSTER (comma-separated labels). Needs the "
                         "one-time on-box `python -m src.willy_sim.gso_assets --convert`.")
    ap.add_argument("--vision", action="store_true",
                    help="P2.D.2: REAL vision (GroundingDINO detect_all + SAM2) instead of GT instance masks")
    ap.add_argument("--mask-completion", type=str, default=None,
                    choices=["none", "axis_aligned_box", "oriented_box"],
                    help="A/B the mask-completion policy for --vision (ignored without it). Omit to take "
                         "the shipped default. THIS RUNNER IS THE ONLY ON-BOX USER OF THAT DEFAULT: only "
                         "MultiObjectVisionPerceptionSource calls complete_mask, so m2/attribute/fused "
                         "(IsaacVisionPerceptionSource) and the GT runners are structurally unaffected. "
                         "The default moved to 'none' on 2026-08-20 after eval-grasps measured the fill "
                         "costing 7.5 pp of top-1 over 1365 object-views; but this scene's tuning "
                         "PREDATES that, so the two must be compared here, paired, in one session.")
    ap.add_argument("--scene2", action="store_true",
                    help="P5.1: generalization scene (different arrangement + table colour + lighting; not-overfit)")
    ap.add_argument("--collect", action="store_true",
                    help="P4.3: wire the RL producer stack (V2/V3 shadow + U2 success-context + G4) so logged "
                         "records carry rl_candidate_features for offline RL (use with --record-log)")
    ap.add_argument("--recovery", action="store_true",
                    help="P4.3: enable the SERVICE recovery loop (-> recovery_actions in the logged records)")
    ap.add_argument("--collect-recovery", type=str, default=None, choices=["next_target"],
                    help="A (recovery variety): runner-local NEXT_TARGET recovery for V6 data; an RL3-oversized "
                         "target fails, the runner switches to a graspable neighbour + re-picks -> GENUINE "
                         "recovered_success episodes (use with WILLY_RL3_SUBSTRATE=unreachable --collect --record-log)")
    ap.add_argument("--seed", type=int, default=None,
                    help="P4.3: seed per-episode position jitter for dataset diversity (None -> no jitter)")
    ap.add_argument("--jitter-mm", type=float, default=20.0,
                    help="P4.3: per-episode object X/Y position jitter amplitude (mm)")
    ap.add_argument("--phase", type=str, default=None, choices=["train", "test"],
                    help="P4.3: tag collected records with a train/test phase (leakage-safe 2-phase split)")
    ap.add_argument("--randomize", action="store_true",
                    help="S2: full domain randomization (position+orientation+colour+lighting) via the unified "
                         "DomainRandomizer (colour/lighting through Isaac Replicator). Default-off byte-identical.")
    ap.add_argument("--rand-seed", type=int, default=None,
                    help="S2: seed for --randomize (defaults to --seed, else 0); per-episode seed = rand_seed+i")
    ap.add_argument("--depth-source", type=str, default="gt", choices=["gt", "rendered"],
                    help="H1.0R: 'rendered' feeds the calculator the REAL (non-uniform) rendered depth "
                         "(GT source skips its uniform centre stamp; GT path only, not --vision). "
                         "Default 'gt' = the uniform GT stamp (byte-identical).")
    ap.add_argument("--grasp-depth-reference", type=str, default="centre", choices=["centre", "top"],
                    help="H1.0R/H1.1a: calculator silhouette depth reference (top = near-surface quantile).")
    ap.add_argument("--depth-band-mm", type=float, default=0.0,
                    help="H1.0R-a: >0 enables the calculator robust depth-band filter (keep [top, top+band]; "
                         "rejects mask-edge table bleed under rendered depth). 0 = off (byte-identical).")
    ap.add_argument("--motion-planner", type=str, default=None, choices=["ik", "rmpflow", "curobo"],
                    help="cuRobo-everywhere: override the approach-phase planner for the whole gate. None (default) "
                         "= the SimRobotConfig default (currently 'ik', byte-identical). 'curobo' drives the "
                         "process-isolated cuRobo planner (auto-falls-back to 'ik' if the cuRobo env is absent) and "
                         "runs the Fix-C Coal-mesh preflight on cuRobo's PLANNED final config.")
    ap.add_argument("--planner-owns-approach", action="store_true",
                    help="with --motion-planner curobo: drive ONLY the grasp goal so cuRobo plans the full "
                         "collision-aware approach (no blind straight-line pre-grasp). Default-off.")
    args = ap.parse_args()
    if args.build_stack:  # pick-and-place assembly, separate from the per-run pick gate
        _hl, _dd = not (args.gui or args.no_headless), args.data_dir
        # Ordered gives a centred, aligned, stable tower; --unordered staggers each cube +20 mm into a
        # leaning staircase whose top cube overhangs its support and topples. That is the aligned against
        # misaligned contrast, built on the 30x30x50 grasp; varied sizes hit a grasp-generation wall.
        raise SystemExit(run_build_stack(
            headless=_hl, data_dir=_dd, place_offset_mm=(20.0 if args.unordered else 0.0),
            record_path=args.record,
        ))
    result = run_gate(mask_completion=args.mask_completion,
                      runs=args.runs, prompt=args.prompt, headless=not (args.gui or args.no_headless),
                      data_dir=args.data_dir, mode=args.mode,
                      record_log=args.record_log, debug_frames=args.debug_frames,
                      blocking=args.blocking, enable_g4=args.g4, enable_g12=args.g12,
                      g4_rerank=args.g4_rerank, g4_rerank_weight=args.g4_rerank_weight,
                      enable_bin=args.bin, near_wall=args.near_wall, bin_half_width_mm=args.bin_half_width,
                      bin_half_width_y_mm=args.bin_half_width_y,
                      bin_height_mm=args.bin_height, near_wall_offset_mm=args.near_wall_offset,
                      self_collision_arm=args.self_collision_arm, finger_tool=args.finger_tool,
                      place_probe=args.place_probe, curobo_bin_world=args.curobo_bin_world,
                      blocker_offset_mm=args.blocker_offset, approach_margin_mm=args.approach_margin,
                      enable_g6=args.g6, agitate_amplitude_mm=args.agitate_amplitude, ycb=args.ycb,
                      gso=args.gso,
                      vision=args.vision, scene2=args.scene2, collect=args.collect, recovery=args.recovery,
                      recovery_variety=args.collect_recovery,
                      seed=args.seed, jitter_mm=args.jitter_mm, phase=args.phase,
                      randomize=args.randomize, rand_seed=args.rand_seed,
                      redistribute_depth_mm=args.redistribute_depth_mm,
                      redistribute_offset_mm=args.redistribute_offset_mm,
                      depth_source=args.depth_source, grasp_depth_reference=args.grasp_depth_reference,
                      depth_band_mm=args.depth_band_mm,
                      motion_planner=args.motion_planner, planner_owns_approach=args.planner_owns_approach)
    write_run_result(args.result_json, result.to_dict(), scene="dense", mode=args.mode, prompt=args.prompt)


if __name__ == "__main__":
    main()
