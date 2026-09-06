"""The attribute scene: four objects where the noun alone is never enough.

The scene is a 2x2 factorial:

    red cube   |   blue cube   |   red cylinder   |   blue cylinder

"red" matches two objects and "cube" matches two objects, so only the conjunction picks one. That is
the attribute-binding problem, and it is the first thing that separates a phrase grounder from a VLM.
The four objects are otherwise identical: same 30 mm footprint, same 50 mm height, same mass, friction
and material. The grasp mechanics are therefore constant, and what differs between two runs is the
grounding and nothing else.

Two measurements, deliberately separate:

* ``--mode grounding`` (cheap, no motion) asks each route to ground each prompt and scores the answer
  against projected ground truth. It is the number that compares routes, because a slipped grasp or an
  IK failure cannot contaminate it.
* ``--mode pick`` runs the full pick and scores the lift of the intended object, not any lift. A run
  that confidently lifts the blue cylinder when the prompt said "der rote Wuerfel" is the failure this
  route exists to prevent, so it is counted as its own outcome rather than as a pass.

The prompt table is half German on purpose. GroundingDINO is English-only in practice, the operator
language in this cell is German, and the router sends any German prompt to the VLM for that reason
(``RouteReason.NON_ENGLISH``). To measure the rule, force ``--route simple`` and ``--route vlm`` over
the same eight prompts and compare the two numbers.

On-box only (needs Isaac plus the GroundingDINO and SAM2 weights, and Qwen for the VLM route):

    <isaac-sim>\\python.bat -m src.willy_sim.run_attribute_pick --mode grounding --route vlm
    <isaac-sim>\\python.bat -m src.willy_sim.run_attribute_pick --mode pick --route vlm \\
        --prompt "der rote Wuerfel" --runs 10
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.schema.robot.sim_schema import SimObjectConfig
from src.willy_sim.harness.cli import add_cell_arguments, cell_profile_kwargs
from src.willy_sim.harness.gate import GateResult
from src.willy_sim.harness.instrumentation import cell_identity, write_run_result

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.app import ModelsConfig

__all__ = [
    "ATTRIBUTE_OBJECTS",
    "PROMPTS",
    "attribute_specs",
    "grounded_index",
    "models_for_route",
    "project_base_points_to_image",
    "target_for_prompt",
]

#: The 2x2 factorial: {red, blue} x {cube, cylinder}, all 30 mm wide and 50 mm tall, in one row at
#: x = 450 mm, 90 mm apart. That leaves a 60 mm clear gap between neighbouring surfaces, and the 80 mm
#: pre-open jaw reaches only 25 mm past the target's surface, so it never touches a neighbour. The
#: colours are the RGB triples the clutter scene uses: they render distinctly under the sim's lighting,
#: and a subtler palette would measure the renderer rather than the grounding.
ATTRIBUTE_OBJECTS: tuple[tuple[str, str, tuple[float, float, float], float], ...] = (
    ("red cube", "cube", (0.90, 0.12, 0.12), -135.0),
    ("blue cube", "cube", (0.12, 0.22, 0.90), -45.0),
    ("red cylinder", "cylinder", (0.90, 0.12, 0.12), 45.0),
    ("blue cylinder", "cylinder", (0.12, 0.22, 0.90), 135.0),
)

#: Object geometry, shared by all four so the grasp is the same problem every time.
OBJECT_X_MM = 450.0
OBJECT_SIZE_MM = (30.0, 30.0, 50.0)   # cylinder reads this as (diameter, diameter, height)

#: The measurement set: every object addressed once in English and once in German. Each prompt names
#: exactly one object and needs both attributes to do it; no prompt here can be answered by a single
#: word. The German half is the decisive axis: it is the language the cell is operated in, and it is
#: what the router's NON_ENGLISH rule sends to the VLM.
PROMPTS: tuple[tuple[str, int, str], ...] = (
    ("the red cube", 0, "en"),
    ("the blue cube", 1, "en"),
    ("the red cylinder", 2, "en"),
    ("the blue cylinder", 3, "en"),
    ("der rote Würfel", 0, "de"),
    ("der blaue Würfel", 1, "de"),
    ("der rote Zylinder", 2, "de"),
    ("der blaue Zylinder", 3, "de"),
)

#: How far a grounded box centre may sit from an object's projected top centre and still count as
#: "that object". The objects are 90 mm apart, which is ~55 px in this overhead view, so 28 px is
#: half that spacing: the box is closer to one object than to any other, or it grounded neither.
GROUNDING_TOLERANCE_PX = 28.0

#: Umlauts folded to their ASCII digraphs, for two reasons. A Windows console is cp1252, so printing
#: the umlaut spelling of "Wuerfel" into a redirected on-box log raises UnicodeEncodeError and kills a
#: run that was otherwise fine. And folding lets an operator on a keyboard that will not produce the
#: umlaut type "der rote Wuerfel" and still hit the measured prompt.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def _ascii(text: str) -> str:
    """``text`` with umlauts folded, for printing only. The prompt itself is never folded."""
    return text.translate(_FOLD)


def _lookup(prompt: str) -> tuple[str, int, str] | None:
    """The measured table row for ``prompt``, matched umlaut-insensitively, or ``None``.

    The row's own spelling is what is handed to the model, so a grounder always sees the same string
    whichever spelling the operator typed.
    """
    folded = _ascii(prompt).strip().lower()
    for row in PROMPTS:
        if _ascii(row[0]).strip().lower() == folded:
            return row
    return None


def target_for_prompt(prompt: str) -> int | None:
    """Which object the prompt names, or ``None`` when the prompt is not in the measured table."""
    row = _lookup(prompt)
    return None if row is None else row[1]


def attribute_specs() -> list[SimObjectConfig]:
    """The four scene objects as ``SimObjectConfig`` (``objects_override`` for the scene builder)."""
    return [
        SimObjectConfig(
            name=name, shape=shape, color=color,
            size_mm=OBJECT_SIZE_MM, position_mm=(OBJECT_X_MM, y, OBJECT_SIZE_MM[2] / 2.0),
        )
        for (name, shape, color, y) in ATTRIBUTE_OBJECTS
    ]


def models_for_route(models: ModelsConfig, route: str) -> ModelsConfig:
    """The models config with ``pipeline`` forced onto one route, without editing the config tree.

    ``simple`` and ``vlm`` pin the route so the same prompt can be sent down both and the two answers
    compared. ``auto`` turns the router on and lets it choose, which is the shipping configuration;
    under ``auto`` a German prompt never reaches the phrase grounder, so the two routes cannot be
    compared on it.
    """
    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        raise ValueError(
            "this runner needs a models.pipeline block (it selects the perception route). The sim "
            "config inherits one from config/models/object.yaml."
        )
    if route == "simple":
        zero_shot, router_enabled = pipeline.zero_shot.model_copy(update={"backend": "grounded_sam"}), False
    elif route == "vlm":
        zero_shot, router_enabled = pipeline.zero_shot.model_copy(update={"backend": "vlm"}), False
    elif route == "auto":
        zero_shot, router_enabled = pipeline.zero_shot.model_copy(update={"backend": "vlm"}), True
    else:
        raise ValueError(f"unknown route {route!r} (expected 'simple', 'vlm' or 'auto')")
    updated = pipeline.model_copy(update={
        "zero_shot": zero_shot,
        "router": pipeline.router.model_copy(update={"enabled": router_enabled}),
    })
    return models.model_copy(update={"pipeline": updated})


def project_base_points_to_image(
    points_base_mm: np.ndarray, camera_to_base: np.ndarray, intrinsics: np.ndarray,
) -> np.ndarray:
    """Project BASE-frame points (mm, Nx3) into pixels with the CV-optical extrinsic and intrinsics.

    Pure numpy, with no Isaac import, so the projection runs off-box. ``camera_to_base`` is the 4x4
    the overhead extrinsic fit returns, mapping CAMERA to BASE in mm; it is inverted here because the
    projection needs BASE to CAMERA. Points behind the camera return NaN rather than a mirrored pixel,
    which is the correct answer for a point that is not in view.
    """
    pts = np.atleast_2d(np.asarray(points_base_mm, dtype=np.float64))
    base_to_cam = np.linalg.inv(np.asarray(camera_to_base, dtype=np.float64))
    cam = (base_to_cam @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T[:, :3]
    k = np.asarray(intrinsics, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2]
        v = k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]
    out = np.stack([u, v], axis=1)
    out[cam[:, 2] <= 0.0] = np.nan
    return out


def grounded_index(
    box: Sequence[float], object_pixels: np.ndarray, *, tolerance_px: float = GROUNDING_TOLERANCE_PX,
) -> int | None:
    """Which scene object a grounded box points at, or ``None`` when it points at none of them.

    ``None`` is a real outcome and is reported as such: a box drawn around the table, around two
    objects at once, or around the arm is not the closest object, and scoring it as one would credit a
    grounder for missing. The tolerance is a fraction of the object spacing, so the answer is
    unambiguous.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    centre = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
    distances = np.linalg.norm(np.asarray(object_pixels, dtype=np.float64) - centre, axis=1)
    if not np.isfinite(distances).any():
        return None
    best = int(np.nanargmin(distances))
    return best if float(distances[best]) <= tolerance_px else None


def _acquire_bgr(camera: Any, session: Any, *, warmup_steps: int) -> np.ndarray:
    """One rendered overhead BGR frame (the same acquisition the vision source performs)."""
    app = getattr(session, "app", None)
    for _ in range(max(1, warmup_steps)):
        session.step(render=True)
        if app is not None:
            app.update()
    frame = camera.get_current_frame()
    rgba = frame.get("rgb") if isinstance(frame, Mapping) else None
    if rgba is None:
        raise RuntimeError("the overhead camera produced no RGB frame")
    return np.ascontiguousarray(np.asarray(rgba)[..., :3][..., ::-1])


def _true_overhead_extrinsic(camera: Any, session: Any, *, warmup_steps: int) -> Any:
    """The CV-optical CAMERA to BASE fit, retried until the depth annotator has filled.

    Not ``handles.camera_to_base``: that hand-built transform is 180 deg in-plane from true, which is
    invisible at y = 0 and mirrors every off-centre object about the camera centre. This scene is
    entirely off-centre, so using it would send every grasp to the object's mirror image and the
    scoring would blame the grounder.
    """
    from src.willy_sim.scene import camera_to_base_ground_truth

    # The fit reads depth, so the annotator has to be attached and the near clip small before the
    # first pump. Without the annotator, `get_depth()` hands back a buffer with no image shape and the
    # fit fails with an unpacking error that says nothing about the cause. The vision source does this
    # in its constructor; here the fit runs before any source exists.
    for setup in (lambda: camera.add_distance_to_image_plane_to_frame(),
                  lambda: camera.set_clipping_range(0.05, 1.0e6)):
        try:
            setup()
        except Exception:  # noqa: BLE001 (best-effort: the API varies by Isaac version)
            pass
    app = getattr(session, "app", None)
    last: Exception | None = None
    for attempt in range(6):
        for _ in range(warmup_steps if attempt == 0 else 12):
            session.step(render=True)
            if app is not None:
                app.update()
        try:
            transform, _rmse = camera_to_base_ground_truth(camera)
            return transform
        except Exception as exc:  # noqa: BLE001 (depth not ready yet; pump the render loop and retry)
            last = exc
            print(f"overhead true-extrinsic retry {attempt + 1}/5 ({exc})", flush=True)
    raise RuntimeError(f"overhead extrinsic fit never converged: {last}")


def build_cell(
    *, headless: bool = True, data_dir: str | None = None,
    cell_kwargs: Mapping[str, Any] | None = None,
    continuous_guard: bool = False, continuous_guard_margin_mm: float = 8.0,
) -> Any:
    """Boot the Isaac cell with the four attribute objects on the table."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell

    cell = bootstrap_sim_cell(
        data_dir, headless=headless,
        scene_kwargs={"objects_override": attribute_specs()},
        **(cell_kwargs or {}),
    )
    from src.willy_sim.run_dense_pick import wire_safety_guards

    wire_safety_guards(
        cell.arm, natural_aim=False, continuous_guard=continuous_guard,
        continuous_guard_margin_mm=continuous_guard_margin_mm, fixtures=(),
    )
    return cell


def _object_pixels(cell: Any, camera_to_base: Any) -> np.ndarray:
    """Each object's top centre projected into the overhead image: the grounding ground truth.

    The top centre rather than the body centre, because that is the surface the camera sees.
    """
    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    tops = []
    for prim_path, _label in cell.handles.object_specs:
        position_m = np.asarray(SingleRigidPrim(prim_path).get_world_pose()[0], dtype=np.float64)
        tops.append(position_m * 1000.0 + np.array([0.0, 0.0, OBJECT_SIZE_MM[2] / 2.0]))
    return project_base_points_to_image(
        np.asarray(tops), np.asarray(camera_to_base.to_matrix(), dtype=np.float64),
        np.asarray(cell.handles.camera.get_intrinsics_matrix(), dtype=np.float64),
    )


def run_grounding(
    *, route: str = "vlm", headless: bool = True, data_dir: str | None = None,
    prompts: Sequence[tuple[str, int, str]] = PROMPTS,
    cell_kwargs: Mapping[str, Any] | None = None, save_frame: str | None = None,
) -> dict[str, Any]:
    """Ask one route to ground every prompt on one scene, scored against projected ground truth.

    No motion: the arm parks once, the scene is rendered once, and every prompt is answered on the
    same image. That keeps the grasp, the IK and the physics out of a number that is about perception,
    and it makes two routes comparable because they saw the same pixels.
    """
    from src.models.factory import build_perception
    from src.robot.core import JointPositions

    cell = build_cell(headless=headless, data_dir=data_dir, cell_kwargs=cell_kwargs)
    arm, sim = cell.arm, cell.sim
    camera = cell.handles.camera
    warmup = max(25, sim.scene_setup.render_warmup_steps)

    arm.move_to_joints(JointPositions(np.asarray(sim.park_joint_positions, dtype=np.float64)))
    extrinsic = _true_overhead_extrinsic(camera, arm.session, warmup_steps=warmup)
    pixels = _object_pixels(cell, extrinsic)
    for (name, _shape, _color, _y), pixel in zip(ATTRIBUTE_OBJECTS, pixels):
        print(f"  ground truth: {name:15} -> pixel ({pixel[0]:6.1f}, {pixel[1]:6.1f})", flush=True)

    backend = build_perception(models_for_route(cell.cfg.models, route))
    bgr = _acquire_bgr(camera, arm.session, warmup_steps=warmup)
    if save_frame:
        # The frame is the evidence: any claim about what a grounder can tell apart is a claim about
        # these pixels, so the annotated frame and the raw frame are both written out.
        import cv2  # type: ignore[import-not-found]

        from pathlib import Path

        annotated = bgr.copy()
        for (name, _shape, _color, _y), pixel in zip(ATTRIBUTE_OBJECTS, pixels):
            if np.isfinite(pixel).all():
                centre = (int(pixel[0]), int(pixel[1]))
                cv2.drawMarker(annotated, centre, (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
                cv2.putText(annotated, name, (centre[0] - 45, centre[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        out = Path(save_frame)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        cv2.imwrite(str(out.with_name(f"{out.stem}_raw{out.suffix}")), bgr)
        print(f"  overhead frame -> {out} (+ _raw)", flush=True)

    rows: list[dict[str, Any]] = []
    for prompt, target, language in prompts:
        started = time.perf_counter()
        try:
            perceived = backend.perceive(bgr, prompt)
            error = None
        except Exception as exc:  # noqa: BLE001 (a refusal IS a result; record it and keep measuring)
            perceived, error = (), f"{type(exc).__name__}: {exc}"
        elapsed_s = time.perf_counter() - started
        best = max(perceived, key=lambda o: o.detection.score, default=None)
        got = None if best is None else grounded_index(best.detection.box, pixels)
        decision = getattr(backend, "last_decision", None)
        row = {
            "prompt": prompt, "language": language, "target": target,
            "target_name": ATTRIBUTE_OBJECTS[target][0],
            "grounded": got, "grounded_name": None if got is None else ATTRIBUTE_OBJECTS[got][0],
            "correct": got == target, "detections": len(perceived),
            "seconds": round(elapsed_s, 3),
            "route": None if decision is None else str(decision.route),
            "route_reason": None if decision is None else str(decision.reason),
            "error": error,
        }
        rows.append(row)
        verdict = "OK  " if row["correct"] else "WRONG" if got is not None else "NONE "
        print(f"{verdict} [{language}] {_ascii(prompt)!r:28} -> {row['grounded_name']}"
              f"  (want {row['target_name']}, {elapsed_s:.2f}s"
              f"{'' if row['route'] is None else ', ' + str(row['route_reason'])})", flush=True)

    correct = sum(1 for r in rows if r["correct"])
    by_language = {
        lang: sum(1 for r in rows if r["language"] == lang and r["correct"])
        for lang in sorted({r["language"] for r in rows})
    }
    totals = {
        lang: sum(1 for r in rows if r["language"] == lang)
        for lang in sorted({r["language"] for r in rows})
    }
    print(f"\nGROUNDING [{route}]: {correct}/{len(rows)} correct "
          + " ".join(f"({lang} {by_language[lang]}/{totals[lang]})" for lang in by_language), flush=True)
    return {
        "route": route, "correct": correct, "prompts": len(rows),
        "by_language": {lang: [by_language[lang], totals[lang]] for lang in by_language},
        "results": rows, **cell_identity(cell),
    }


def _service_for(
    cell: Any, backend: Any, prompt: str, *, extrinsic: Any,
    standoff_mm: float, close_width_mm: float,
) -> Any:
    """One perception source and one service per prompt, over the same already loaded backend.

    A service is cheap under the analytic generator, which carries no weights; the backend is not.
    Under ``calculator: deep`` the service does carry a torch model and a CUDA context, so it is cheap
    only on the geometric path. Rebuilding just the source is what lets one Isaac boot measure several
    prompts without reloading Qwen for each.
    """
    from src.robot.execution.autonomous_grasp import AutonomousGraspService
    # Build through the factory, never by naming `GraspCalculator`: a cell that asks for
    # `robot.grasping.calculator: deep` would otherwise get the analytic stack in silence. The
    # selector defaults to `geometric`, and that branch is `GraspCalculator(**kwargs)` verbatim.
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver
    from src.willy_sim.harness.modes import mode_service_kwargs, resolve_demo_mode
    from src.willy_sim.perception import IsaacVisionPerceptionSource

    perception = IsaacVisionPerceptionSource(
        camera=cell.handles.camera, backend=backend, prompt=prompt, session=cell.arm.session,
        warmup_steps=max(25, cell.sim.scene_setup.render_warmup_steps),
    )
    calculator = build_calculator(
        cell.robot,
        camera_matrix=np.asarray(cell.handles.camera.get_intrinsics_matrix(), dtype=np.float64),
        max_grip_width_mm=cell.robot.gripper.max_width_mm,
        min_grip_width_mm=cell.robot.gripper.min_width_mm,
        # Analytic-only. The learned generator decodes its own closing axis; the factory logs that
        # it is ignoring this rather than dropping it in silence.
        isotropic_radial_closing=bool(getattr(cell.robot.grasping, "isotropic_radial_closing", False)),
    )
    policy = GraspExecutionPolicy(
        arm=cell.arm, gripper=cell.gripper, standoff_mm=standoff_mm, retreat_mm=100.0,
        pre_open_width_mm=80.0, close_width_mm=close_width_mm, require_base_frame_grasp=True,
        require_steady_before_motion=(cell.dwell.require_steady_before_motion if cell.dwell else False),
        steady_timeout_s=(cell.dwell.steady_timeout_s if cell.dwell else 5.0),
    )
    mode = resolve_demo_mode("easy")
    return AutonomousGraspService.from_components(
        arm=cell.arm, calculator=calculator, perception=perception, gripper=cell.gripper,
        mode=mode, frame_resolver=StaticCameraToBaseResolver(transform=extrinsic),
        policy=policy, max_attempts=2, **mode_service_kwargs(mode),
    )


def run_picks(
    runs: int = 10, *, prompt: str = "der rote Würfel", route: str = "vlm", headless: bool = True,
    data_dir: str | None = None, cell_kwargs: Mapping[str, Any] | None = None,
    standoff_mm: float = 150.0, close_width_mm: float = 25.0, record_log: str | None = None,
) -> GateResult:
    """Pick the prompted object out of the four, ``runs`` times, scored on the intended object's lift.

    Three outcomes, kept apart because they mean different things: the intended object lifted (pass);
    a different object lifted, the confident wrong grasp, which nothing downstream can detect; or
    nothing lifted, a grasp or grounding miss that at least fails visibly.
    """
    # Every isaacsim import stays below build_cell: Kit's plugin system registers ``isaacsim.core``
    # only once SimulationApp exists, so hoisting this to the top of the function raises
    # ModuleNotFoundError before the cell has booted.
    from src.models.factory import build_perception
    from src.robot.core import JointPositions
    from src.willy_sim.harness.instrumentation import write_pick_artifacts

    row = _lookup(prompt)
    if row is None:
        raise SystemExit(
            f"prompt {_ascii(prompt)!r} is not in the measured table, so there is no ground truth "
            "for it.\nKnown prompts: " + ", ".join(repr(_ascii(text)) for text, _i, _l in PROMPTS)
        )
    prompt, target = row[0], row[1]
    cell = build_cell(headless=headless, data_dir=data_dir, cell_kwargs=cell_kwargs)
    arm, gripper, sim = cell.arm, cell.gripper, cell.sim
    gate = sim.scene_setup.gate
    park_q = np.asarray(sim.park_joint_positions, dtype=np.float64)
    warmup = max(25, sim.scene_setup.render_warmup_steps)

    arm.move_to_joints(JointPositions(park_q))
    extrinsic = _true_overhead_extrinsic(cell.handles.camera, arm.session, warmup_steps=warmup)
    backend = build_perception(models_for_route(cell.cfg.models, route))
    service = _service_for(cell, backend, prompt, extrinsic=extrinsic,
                           standoff_mm=standoff_mm, close_width_mm=close_width_mm)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    prims = [SingleRigidPrim(path) for path, _label in cell.handles.object_specs]
    homes = [(np.asarray(p.get_world_pose()[0]), np.asarray(p.get_world_pose()[1])) for p in prims]
    print(f"=== ATTRIBUTE pick: route={route} prompt={_ascii(prompt)!r} -> "
          f"'{ATTRIBUTE_OBJECTS[target][0]}' among {len(prims)} objects ===", flush=True)

    run_id = f"attr-{int(time.time())}"
    results: list[dict[str, Any]] = []
    for i in range(runs):
        gripper.open()
        arm.move_to_joints(JointPositions(park_q))
        for prim, (home_pos, home_quat) in zip(prims, homes):
            prim.set_world_pose(position=home_pos, orientation=home_quat)
            try:
                prim.set_linear_velocity(np.zeros(3))
                prim.set_angular_velocity(np.zeros(3))
            except Exception:  # noqa: BLE001 (velocity-zero is best-effort)
                pass
        arm.session.step_n(25)
        z0 = [float(np.asarray(p.get_world_pose()[0])[2]) for p in prims]

        report = service.pick()
        lifts = [
            (float(np.asarray(p.get_world_pose()[0])[2]) - z) * 1000.0 for p, z in zip(prims, z0)
        ]
        lifted = [idx for idx, mm in enumerate(lifts) if mm >= gate.lift_threshold_mm]
        passed = lifted == [target]
        wrong = bool(lifted) and target not in lifted
        outcome = getattr(getattr(report, "outcome", None), "value", getattr(report, "outcome", None))
        route_taken = getattr(backend, "last_decision", None)
        results.append({
            "run": i, "outcome": outcome, "passed": passed, "wrong_object": wrong,
            "lift_mm": [round(mm, 1) for mm in lifts],
            "lifted": [ATTRIBUTE_OBJECTS[idx][0] for idx in lifted],
        })
        print(f"RUN {i}: outcome={outcome} lifted={results[-1]['lifted']} "
              f"passed={passed}{' WRONG-OBJECT' if wrong else ''} "
              f"lift_mm={results[-1]['lift_mm']}"
              f"{'' if route_taken is None else ' route=' + route_taken.describe()}", flush=True)
        write_pick_artifacts(
            report, attempt_id=f"{run_id}-{i:04d}", record_log=record_log,
            extra={**cell_identity(cell), "sim_runner": "attribute", "sim_prompt": prompt,
                   "sim_route": route, "sim_target": ATTRIBUTE_OBJECTS[target][0],
                   "sim_lift_mm": round(lifts[target], 2), "sim_lifted": bool(passed),
                   "sim_wrong_object": wrong, "sim_gate_passed": passed},
        )

    n_pass = sum(1 for r in results if r["passed"])
    n_wrong = sum(1 for r in results if r["wrong_object"])
    print(f"ATTRIBUTE GATE [{route}]: {n_pass}/{runs} picked the intended object "
          f"({n_wrong} wrong-object grasps)", flush=True)
    degraded = getattr(arm, "curobo_degraded", False)
    reason = str(getattr(arm, "curobo_degraded_reason", "") or "") if degraded else None
    return GateResult(
        runs=runs, passed=n_pass, results=results, target=ATTRIBUTE_OBJECTS[target][0],
        distractor_lifts=n_wrong, planner_degraded=reason,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy P5 attribute-scene pick / grounding (Isaac).")
    ap.add_argument("--mode", type=str, default="grounding", choices=["grounding", "pick"],
                    help="grounding: score every prompt on one rendered frame, no motion. "
                         "pick: run the full pick and score the INTENDED object's lift.")
    ap.add_argument("--route", type=str, default="vlm", choices=["simple", "vlm", "auto"],
                    help="force the perception route (simple=GroundingDINO, vlm=Qwen) or let the "
                         "router choose (auto). Forcing is what makes the two comparable.")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--prompt", type=str, default="der rote Würfel",
                    help="pick mode: which of the measured prompts to run")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--standoff-mm", type=float, default=150.0)
    ap.add_argument("--close-width-mm", type=float, default=25.0)
    ap.add_argument("--record-log", type=str, default=None,
                    help="append one GraspAttemptRecord JSONL line per pick (default off)")
    ap.add_argument("--result-json", type=str, default=None)
    ap.add_argument("--save-frame", type=str, default=None,
                    help="grounding mode: write the overhead frame the routes were scored on "
                         "(annotated with the projected ground truth) plus the raw frame")
    add_cell_arguments(ap)
    args = ap.parse_args()

    if args.mode == "grounding":
        result = run_grounding(route=args.route, headless=not args.gui, data_dir=args.data_dir,
                               cell_kwargs=cell_profile_kwargs(args), save_frame=args.save_frame)
        write_run_result(args.result_json, result, scene="attribute", mode="grounding")
        return
    gate = run_picks(runs=args.runs, prompt=args.prompt, route=args.route, headless=not args.gui,
                     data_dir=args.data_dir, cell_kwargs=cell_profile_kwargs(args),
                     standoff_mm=args.standoff_mm, close_width_mm=args.close_width_mm,
                     record_log=args.record_log)
    write_run_result(args.result_json, gate.to_dict(), scene="attribute", mode="pick",
                     prompt=args.prompt, route=args.route)


if __name__ == "__main__":
    main()
