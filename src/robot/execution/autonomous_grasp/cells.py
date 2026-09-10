"""Build a whole cell in one call: the real one, and the desk rehearsal.

    service = build_real_cell(robot_cfg, prompt="a box")   # a physical cell
    service = build_rehearsal_cell(robot_cfg)              # a desk: dummy arm, no camera

`from_robot_config` is the composition root: it builds the vendor arm, the gripper branch, the
safety preflight, the frame resolver from the calibration artifact, the sub-policies and record
logging, all from a validated `RobotConfig`. But `calculator` and `perception` are required keyword
arguments with no default, because config alone cannot describe a camera adapter that needs a
per-run text prompt and two models on a GPU. These helpers supply both and hand the result to the
root, so a cell built from config takes one call and the construction lives in the package that
owns it.

What this deliberately does not do: make `calculator` / `perception` optional on
`from_robot_config`. That would buy the same single call and pay for it with a root that opens a
camera when a caller forgets an argument. The root keeps its explicit contract; the convenience
lives beside it, under a name that says what it costs.

Torch is not imported until a real cell is actually built. `from_robot_config` is torch-free and
stays that way: the models factory is imported inside `build_real_cell`, so a sim runner or a
rehearsal never pays for it.

Anything with its own components, such as the Isaac runners and `datagen`'s probes, keeps calling
`from_robot_config` or `from_components` directly. These helpers are the convenience for the
config-driven case, not a replacement for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema import AppConfig
    from src.config.schema.robot import RobotConfig

    from .service import AutonomousGraspService

__all__ = ["CellBuildRefused", "build_real_cell", "build_real_components", "build_rehearsal_cell",
           "build_rehearsal_components"]


class CellBuildRefused(RuntimeError):
    """This config cannot produce a cell, and the reason is in the message.

    An `Exception`, and that is the whole point. These refusals used to be `SystemExit`, which is a
    `BaseException`, so neither caller that was written to handle them could: `api/routers/cell.py`
    catches `Exception` and answers 422 `build_refused`, and `real_cell/__main__.py` catches
    `Exception` and returns its config exit code. Both were bypassed, so a refusal the operator was
    meant to read as "your config says X" arrived at the console as a 500 with no message and at the
    bench as a bare traceback.

    It went unnoticed because no shipped profile could reach one: the cell took the first rig with
    `source: rgbd` in list order and there was always such a rig, even switched off. Naming the
    primary camera made the refusals reachable, on the base profile and on tiltcam, which turned a
    latent 500 into a certain one on every profile there is.
    """


def build_rehearsal_components(robot_cfg: "RobotConfig") -> tuple[Any, Any, Any, Any, Any]:
    """``(calculator, perception, frame_resolver, multi_camera, camera_calculators)`` for a desk.

    Kept as its own function beside the cell builder so a caller that wants the pieces, to swap
    one or to inspect one, does not have to take the whole service to get them.
    """
    import numpy as np

    from src.geometry import Frame, Transform
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

    from .rehearsal import RehearsalPerceptionSource, rehearsal_intrinsics

    perception = RehearsalPerceptionSource()
    # Through the factory, not `GraspCalculator` by name, so `robot.grasping.calculator`
    # (geometric | deep) selects the stack and the factory's fail-closed refusal can fire. Naming
    # the analytic class directly lets a cell ask for `deep` and silently get the analytic stack.
    # The selector is a string, not a boolean `enabled` flag, and a selector is as capable of
    # being inert as a switch.
    calculator = build_calculator(
        robot_cfg,
        camera_matrix=rehearsal_intrinsics(),
        max_grip_width_mm=robot_cfg.gripper.max_width_mm,
        min_grip_width_mm=robot_cfg.gripper.min_width_mm,
        # Which geometry stage proposes candidates, from config. The default
        # ('support_footprint') needs the BASE support plane the pick loop resolves and the
        # CAMERA->BASE transform the resolver below provides; without either it stands down to the
        # silhouette stage on its own and says so in telemetry.
        support_footprint_geometry=(
            robot_cfg.grasping.geometry.stage == "support_footprint"),
        support_footprint_inflate_mm=robot_cfg.grasping.geometry.inflate_mm,
    )
    # A nadir camera 800 mm above the workspace centre, looking straight down. Stands in for the
    # artifact a real eye-to-hand calibration writes.
    resolver = StaticCameraToBaseResolver(transform=Transform(
        translation_mm=np.array([400.0, 0.0, 800.0]),
        quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
        from_frame=Frame.CAMERA, to_frame=Frame.BASE,
    ))
    # A desk has no cameras at all, so it has no other cameras and no per-camera calculators. The
    # last two slots are here so the two component builders keep the same shape and a reader is not
    # left wondering which one is missing something.
    return calculator, perception, resolver, None, None


def build_real_components(robot_cfg: "RobotConfig", prompt: str, *,
                          app_config: "Maybe[AppConfig]" = UNSET,
                          ) -> tuple[Any, Any, Any, Any, Any]:
    """``(calculator, perception, frame_resolver=None, multi_camera, camera_calculators)``.

    The resolver stays ``None`` on purpose: ``from_robot_config`` builds it from the calibration
    artifact named in config, and refuses if there is none. Passing one here would paper over exactly
    the misconfiguration the preflight is there to surface.

    ``multi_camera`` is the other cameras of a fused cell (``None`` for every single-camera one). It
    cannot come from ``from_robot_config``: the composition root is deliberately unable to open a
    device, so the cameras are opened here and handed in, exactly as ``perception`` is. See
    :func:`_build_multi_camera_rig` for what it does and does not include.

    The ``app_config`` argument is which tree, and it used not to be askable. Measured 2026-09-10: a
    caller resolves the robot half itself, through ``load_robot_config(data_dir, profile=...)``,
    because ``real_cell`` has a ``--profile`` and a ``--data-dir``. This function then read the
    camera half with a bare ``load_config()``, and a no-argument load takes ``profile=UNSET``, which
    falls back to ``WILLY_PROFILE``, and ``data_dir=None``, which falls back to the checkout's own
    tree. So one command built an arm from the flag and cameras from the environment. Reproduced
    three ways: the same chain through ``--profile`` and through ``WILLY_PROFILE`` produced two
    different refusals, a scratch tree that could build was refused quoting this repository's rigs,
    and ``--data-dir`` had no environment equivalent at all, so a console pointed at a deployment
    tree got that tree's arm and this checkout's cameras.

    ``UNSET`` means the caller did not choose a tree, and then the default load is the honest
    answer, which is what every caller did before and still does. The defect was never that a
    default existed; it was that a caller who had chosen could not say so.
    """
    import numpy as np

    from src.config import load_config
    from src.camera.orchestration.frame_provider import FrameProvider
    from src.models.perception_spec import PerceptionSpec
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.perception import RealSenseVisionPerceptionSource

    app_cfg = app_config if chosen(app_config) else load_config()
    # The cell's camera is named, not inferred. This took the first rig with `source: rgbd` by list
    # position and did not consult `enabled`, while the schema's own two-camera serial check used
    # `source == "rgbd" and enabled` and this file's fusion guard used a third rule. Three predicates
    # over one list, disagreeing. On the shipped base profile the consequence was concrete: its only
    # RGB-D rig ships `enabled: false`, and this opened it anyway, so the failure arrived at the
    # device instead of at the config that caused it.
    rigs = list(app_cfg.camera.cameras.rigs)
    rig_id = app_cfg.camera.cameras.primary_rig_id
    primary = next((r for r in rigs if r.rig_id == rig_id), None)
    if primary is None:  # the schema checks this, so reaching it means the config was built by hand
        raise CellBuildRefused(
            f"camera.cameras.primary_rig_id names {rig_id!r}, which is not in camera.cameras.rigs "
            f"({sorted(r.rig_id for r in rigs)})."
        )
    if not getattr(primary, "enabled", True):
        raise CellBuildRefused(
            f"camera.cameras.primary_rig_id names {rig_id!r} and that rig has `enabled: false`. A "
            "cell cannot run on a camera its own config declares off. Switch it on, or name a rig "
            "that is on."
        )
    if getattr(primary, "source", None) != "rgbd":
        # The RGB-D rigs are named, not just demanded. Telling someone to "name the RGB-D rig" is
        # telling them to supply the one fact they have just demonstrated they do not have, and this
        # function is holding the list while it says it.
        depth_rigs = sorted(r.rig_id for r in rigs if getattr(r, "source", None) == "rgbd")
        raise CellBuildRefused(
            f"camera.cameras.primary_rig_id names {rig_id!r}, a {getattr(primary, 'source', '?')!r} "
            "rig. A grasp cell needs an RGB-D primary: the grasp is synthesised from its depth. "
            + (f"The RGB-D rigs here are {depth_rigs}. Name one of those instead, and prove it "
               "standalone with `python -m src.robot.perception`."
               if depth_rigs else
               "There is no RGB-D rig in camera.cameras.rigs at all, so this profile cannot build a "
               "grasp cell until one is added.")
        )
    # Every camera goes through the frame provider, the one class that owns rig identity, per-rig
    # lifecycle and open/release bookkeeping. The provider is told about every rig and opens
    # exactly one: constructing a streamer touches no device, so knowing a rig costs nothing and
    # holding it is a deliberate act.
    #
    # `provider.rig(...)` hands back a `RigHandle` shaped exactly like the streamer this adapter
    # duck-types against, so nothing in `robot.perception` has to know about camera orchestration,
    # and the handle can reach this rig and no other. `handle.release()` gives back this rig only,
    # so the console's teardown (`execution.lifecycle.release_perception` calling
    # `perception.close()`) closes exactly the device it opened.
    #
    # Which rig is primary: whichever `camera.cameras.primary_rig_id` names, checked above. The
    # grasp is synthesised from this camera and every other camera confirms what it sees, so moving
    # the primary moves every measured number this cell has.
    provider = FrameProvider(rigs)
    provider.open_rig(rig_id)
    try:
        return _build_on_open_cameras(
            robot_cfg, app_cfg, prompt=prompt, provider=provider, rig_id=rig_id, np=np,
            perception_spec=PerceptionSpec, build_calculator=build_calculator,
            vision_source=RealSenseVisionPerceptionSource,
        )
    except BaseException:
        # A refused build gives the camera back. Everything below the open can raise, and one of
        # those raises is deliberate: `_preload()` exists so a corrupt artifact is refused at build
        # time rather than at 3 a.m. Without this, the refusal it was added to produce left the
        # RealSense claimed by a dead process, and the operator who fixed the artifact and ran again
        # met "device busy", which names neither the cause nor the first failure. `release` never
        # raises and is idempotent, so it cannot replace the exception being reported.
        provider.release()
        raise


def _build_on_open_cameras(robot_cfg: "RobotConfig", app_cfg: Any, *, prompt: str, provider: Any,
                           rig_id: str, np: Any, perception_spec: Any, build_calculator: Any,
                           vision_source: Any) -> tuple[Any, Any, Any, Any, Any]:
    """The half of :func:`build_real_components` that runs with a device already held.

    Split out for the `try` above rather than for its own sake: wrapping the tail in place would
    have indented sixty lines of load-bearing commentary, and a diff that moves every line is a diff
    whose one real change nobody can see. The imports are passed in because they are function-local
    by design, so that a rehearsal, a test or a sim runner never pays for torch.
    """
    handle = provider.rig(rig_id)
    # The stack is built once and shared by every camera. It is the expensive part of a cell (two
    # models on the GPU) and it is stateless per call, so a four-camera rig costs four inference
    # passes and one set of weights, not four sets.
    #
    # It goes through `PerceptionSpec`, so the whole `models.pipeline` block (kind, VLM backend,
    # OneFormer segmenter, prompt router) selects the stack and `build_perception`'s fail-closed
    # refusals are reachable from here. Reading only `models.detector` and
    # `models.segmenter_backend` would build GroundingDINO plus SAM2 whatever the pipeline block
    # says, with no error and no log line. On the shipped values the two resolve to the same pair.
    spec = perception_spec.from_config(app_cfg.models)
    backend = spec.build()
    perception = vision_source(
        streamer=handle,
        backend=backend,
        prompt=prompt,
    )
    intrinsics = np.asarray(handle.get_intrinsics(), dtype=np.float64)
    multi_camera = _build_multi_camera_rig(
        robot_cfg, app_cfg, provider=provider, primary_rig_id=rig_id,
        backend=backend, prompt=prompt,
    )
    # Through the factory, as in `build_rehearsal_components`. This is the path a physical cell
    # takes, so it is the one where a silently ignored `calculator: deep` files the analytic
    # stack's numbers under the learned generator's name.
    calculator = build_calculator(
        robot_cfg,
        camera_matrix=intrinsics,
        max_grip_width_mm=robot_cfg.gripper.max_width_mm,
        min_grip_width_mm=robot_cfg.gripper.min_width_mm,
        support_footprint_geometry=(
            robot_cfg.grasping.geometry.stage == "support_footprint"),
        support_footprint_inflate_mm=robot_cfg.grasping.geometry.inflate_mm,
    )
    # Load the weights here, while the caller can still refuse; `preload` exists for exactly that.
    # `_compute_result` catches everything, because the calculator protocol forbids raising, so a
    # corrupt or mismatched artifact otherwise reaches the operator as an empty candidate list
    # rather than as a refusal at build time.
    #
    # The physical path only. Other callers build calculators they may never call, and eager
    # loading there would pay for weights nobody uses. This function already opens cameras and
    # loads two models onto the GPU; a `.pt` file is the cheapest thing it touches.
    #
    # Duck-typed: the analytic calculator has no `preload`, and that is the normal case.
    _preload = getattr(calculator, "preload", None)
    if callable(_preload):
        _preload()
    calculators = _build_camera_calculators(
        robot_cfg, app_cfg, provider=provider, primary_rig_id=rig_id, primary=calculator,
        multi_camera=multi_camera,
    )
    return calculator, perception, None, multi_camera, calculators


def _build_camera_calculators(
    robot_cfg: "RobotConfig", app_cfg: Any, *, provider: Any, primary_rig_id: str,
    primary: Any, multi_camera: Any,
) -> "dict[str, Any] | None":
    """One calculator per camera, or None when one is enough.

    A GraspCalculator is bound to one camera's intrinsics at construction: it back-projects the mask
    with them, and every candidate position it returns is in that camera's frame. So an object seen
    only by a second camera cannot be computed with the primary's calculator. It would not fail. It
    would produce a plausible grasp at the wrong place, using the right pixels with the wrong focal
    length and the wrong principal point, and nothing downstream can tell that pose from a good one.

    Returns None unless the cell can actually promote an object, which keeps every existing cell on
    exactly the object it built before. The gripper limits are read once and shared: a gripper is a
    fact about the cell and not about a camera, so K calculators differ in their lens and in nothing
    else.
    """
    geometry_cfg = getattr(getattr(getattr(robot_cfg, "grasping", None), "fusion", None),
                           "geometry", None)
    if multi_camera is None or geometry_cfg is None:
        return None
    if not bool(getattr(geometry_cfg, "promote_unmatched", False)):
        # Fusion without promotion never needs a second calculator: every object it computes was
        # seen by the primary, because an object the primary did not see is not in the list.
        return None

    import numpy as np  # noqa: PLC0415

    from src.robot.grasping.calculator_factory import build_calculator  # noqa: PLC0415

    calculators: dict[str, Any] = {primary_rig_id: primary}
    for cam_id in getattr(multi_camera, "sources", {}):
        handle = provider.rig(cam_id)
        matrix = handle.get_intrinsics()
        if matrix is None:
            # Refused rather than defaulted. A camera whose intrinsics are unknown can still fuse a
            # surface onto somebody else's object, and it cannot own one: the pick loop would fall
            # back to the primary's lens and say so, once, in a warning nobody reads at 3 a.m.
            raise CellBuildRefused(
                f"camera {cam_id!r} can promote an object (grasping.fusion.geometry."
                "promote_unmatched is true) and its intrinsics are unavailable, so a grasp "
                "synthesised from it would be placed with the primary camera's lens. Calibrate it, "
                "or set promote_unmatched: false and let it confirm objects rather than introduce "
                "them."
            )
        calculators[cam_id] = build_calculator(
            robot_cfg,
            camera_matrix=np.asarray(matrix, dtype=np.float64),
            max_grip_width_mm=robot_cfg.gripper.max_width_mm,
            min_grip_width_mm=robot_cfg.gripper.min_width_mm,
            support_footprint_geometry=(
                robot_cfg.grasping.geometry.stage == "support_footprint"),
            support_footprint_inflate_mm=robot_cfg.grasping.geometry.inflate_mm,
        )
    return calculators


def _build_multi_camera_rig(robot_cfg: "RobotConfig", app_cfg: Any, *, provider: Any,
                            primary_rig_id: str, backend: Any,
                            prompt: str) -> Any:
    """The other cameras of a fused cell, as a `MultiCameraPerceptionSource`, or None.

    `apply_orchestrator_overlays` wires `fusion_geometry_config` and the per-camera resolver map
    from config, but the rig itself is a live device handle a caller must pass in. This is what
    builds it, so a physical cell that names calibrated cameras fuses them instead of naming three
    cameras, loading three artifacts fail-closed, and still running single-view.

    The primary is deliberately not in here. `fuse_scene_geometry` takes the primary view as its
    first four arguments and the rest as `other_views`; a primary that appears in both is fused
    with a second copy of itself, and on hardware each extra observation costs a full detect plus
    segment pass. The Isaac runner does include its primary in the map; that is a redundancy rather
    than an error, and it is left as it is because changing it would move a measured number
    without a measurement.

    Returns None when fusion geometry is off or names no camera besides the primary, which keeps
    every existing cell byte-identical.

    Never run on hardware. No physical multi-camera cell has been built by this function.
    """
    from src.robot.grasping.types.perception import MappedCameraRig
    from src.robot.perception import RealSenseVisionPerceptionSource

    grasping_cfg = getattr(robot_cfg, "grasping", None)
    fusion_cfg = getattr(grasping_cfg, "fusion", None)
    geometry_cfg = getattr(fusion_cfg, "geometry", None)
    if geometry_cfg is None or not bool(getattr(geometry_cfg, "enabled", False)):
        return None
    cameras = getattr(fusion_cfg, "cameras", None) or {}
    wanted = [cam_id for cam_id, cam in sorted(cameras.items())
              if bool(getattr(cam, "enabled", True)) and cam_id != primary_rig_id]
    if not wanted:
        return None

    # Fail-closed, and at build time. A camera named in `fusion.cameras` with no matching RGB-D rig
    # still gets a resolver, because its artifact loads, and therefore joins
    # `configured_camera_ids`; at runtime it is a camera that was expected and delivered nothing,
    # which under `refuse` stops every pick and under `degrade` warns on every pick. This says it
    # once instead, while an operator is watching.
    known = {r.rig_id for r in app_cfg.camera.cameras.rigs
             if getattr(r, "source", None) == "rgbd"}
    unknown = [cam_id for cam_id in wanted if cam_id not in known]
    if unknown:
        raise CellBuildRefused(
            f"grasping.fusion.cameras names {unknown} but camera.cameras.rigs has no RGB-D rig with "
            f"that id (known: {sorted(known)}). A fused cell expects every configured camera to "
            "deliver a frame, so this would refuse or warn on every pick instead of once here."
        )

    sources = {}
    for cam_id in wanted:
        provider.open_rig(cam_id)
        sources[cam_id] = RealSenseVisionPerceptionSource(
            streamer=provider.rig(cam_id),
            # The same backend object, not a second build. Every camera in a fused cell grounds
            # with one set of weights; the cost of an extra camera is an inference pass.
            backend=backend,
            prompt=prompt,
        )
    return MappedCameraRig(sources)


def build_real_cell(robot_cfg: "RobotConfig", *, prompt: str = "object",
                    app_config: "Maybe[AppConfig]" = UNSET,
                    **overrides: Any) -> "AutonomousGraspService":
    """A complete cell for physical hardware, from config, in one call.

    Opens one RGB-D camera and loads two models onto the GPU. It does not connect the arm: the
    caller owns that lifecycle, deliberately, because connecting a gripper drives digital I/O and
    the order matters (arm first).

    ``overrides`` are forwarded to :meth:`AutonomousGraspService.from_robot_config`, so a caller
    that needs a mode, a policy or a live device handle still has one.

    ``app_config`` is the camera half's tree, and it must be the tree ``robot_cfg`` came from. See
    :func:`build_real_components` for what happened while the two could disagree.
    """
    from src.config import load_config

    from .service import AutonomousGraspService

    # Resolved once, here, and handed to both readers below. It used to be read here a second time
    # with a comment explaining that the loader caches, so the two reads return the same object.
    # That comment was true and it was not about the right thing: two calls agreeing with each
    # other says nothing about whether either is the tree the caller asked for, and they were not.
    app_cfg = app_config if chosen(app_config) else load_config()
    calculator, perception, resolver, multi_camera, calculators = build_real_components(
        robot_cfg, prompt, app_config=app_cfg)
    rig_id = app_cfg.camera.cameras.primary_rig_id
    service = AutonomousGraspService.from_robot_config(
        robot_cfg, calculator=calculator, perception=perception, frame_resolver=resolver,
        multi_camera_perception=multi_camera,
        camera_calculators=calculators,
        # Which camera is the primary, told to the root rather than inferred there. It cannot read
        # it: the key lives in the camera section and the root is handed a `RobotConfig`. Without it
        # the primary is expected among the cameras a fused pick waits for, and it delivers through
        # `perception` instead, so it would be missing on every pick.
        primary_camera_id=rig_id,
        **overrides,
    )
    _wire_live_planner_world(robot_cfg, service, perception)
    return service


def _wire_live_planner_world(robot_cfg: "RobotConfig", service: Any, perception: Any) -> None:
    """Give the arm something that can say what the cell looks like right now, if it asked for one.

    Here rather than inside the service, because this is the only place that holds all three
    pieces: the camera the components builder opened, the transform the calibration produced,
    and the arm the service built. The layers below cannot reach across those and are not
    supposed to.

    A cell that did not ask gets nothing and behaves exactly as before. A cell that asked and
    cannot have it, which today means a wrist camera, is told at build time rather than at the
    first motion: an eye-in-hand transform changes between the shutter and the moment the
    geometry is built, and a world displaced by the arm travel is worse than no world at all.
    """
    import logging

    from .live_world import (
        LiveWorldUnavailable,
        RigDepthSource,
        build_live_planner_world,
        static_camera_to_base_mm,
    )

    arm = getattr(getattr(service, "runtime", None), "orchestrator", None)
    arm = getattr(arm, "arm", None)
    setter = getattr(arm, "set_live_planner_world", None)
    streamer = getattr(perception, "streamer", None)
    resolver = getattr(getattr(service, "runtime", None), "orchestrator", None)
    resolver = getattr(resolver, "frame_resolver", None)
    if not callable(setter) or streamer is None or resolver is None:
        return

    try:
        transform = static_camera_to_base_mm(resolver)
    except LiveWorldUnavailable as exc:
        logging.getLogger(__name__).warning("no live planner world for this cell: %s", exc)
        return

    world = build_live_planner_world(
        robot_cfg,
        [(getattr(streamer, "rig_id", "camera"), RigDepthSource(streamer), transform)],
    )
    if world is None:
        return
    setter(world)
    logging.getLogger(__name__).info(
        "live planner world wired: %d camera(s), refreshed before every plan", len(world.cameras)
    )


def build_rehearsal_cell(robot_cfg: "RobotConfig",
                         **overrides: Any) -> "AutonomousGraspService":
    """A complete cell for a desk rehearsal: no camera, no models, and a dummy arm.

    The same wiring the real cell gets: `from_robot_config`, the connect order, the pick loop,
    safety, record logging. What it does not do is pretend to be perception. The scene is one flat
    box at a known place, so what this exercises is the wiring. Grasp quality is measured in Isaac
    and on the bench, never here.

    The vendor swap lives here, not in the caller. A function called `build_rehearsal_cell` builds
    a rehearsal cell, so every caller gets the dummy arm and the arm is not an argument a caller
    has to remember. A swap done outside this function lets a terminal and a browser disagree about
    what the cell is, and the browser's connect then reaches for a physical controller.

    Idempotent: a caller that already swapped the vendor gets the same result. The CLI still swaps
    it, because its preflight and its banner need the swapped config.
    """
    from .service import AutonomousGraspService

    robot_cfg = robot_cfg.model_copy(update={"vendor": "dummy"})
    calculator, perception, resolver, _no_cameras, _one_lens = (
        build_rehearsal_components(robot_cfg))
    return AutonomousGraspService.from_robot_config(
        robot_cfg, calculator=calculator, perception=perception, frame_resolver=resolver,
        **overrides,
    )
