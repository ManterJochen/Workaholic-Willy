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
    from pathlib import Path

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


def _refuse_a_tree_hand(robot_cfg: "RobotConfig", data_dir: "str | Path | None") -> None:
    """Raise :class:`CellBuildRefused` when the tree at ``data_dir`` describes the named hand differently.

    The hand's body, sphere map, retract rows and evidence were written from the repository's
    registry, so a cell run from a tree whose own copy disagrees would plan with one hand while its
    calculator reads another.
    """
    from src.config.grippers import tree_hand_refusal

    if data_dir is None:
        return  # the repository's own tree: nothing to compare
    model = robot_cfg.gripper.model
    refusal = tree_hand_refusal(model, data_dir=data_dir) if model else None
    if refusal is not None:
        raise CellBuildRefused(refusal)


def _scene_kwargs(robot_cfg: "RobotConfig") -> dict[str, Any]:
    """What the cell's calculator reads beside the part it grasps: the parts the prompt did not name, by the planner
    world's rules from the tree, and the tilted approaches beside the vertical ones.

    Every cell's since 2026-10-17, as the owner's cell ran them (``robot.grasping.scene_obstacles`` and
    ``side_approaches`` until then); a calculator a script builds by hand reads neither, as before (the fix plan's
    contracts 5 and 8).
    """
    from src.robot.grasping.generation.scene_obstacles import SceneObstacleRules

    return {"scene_obstacles": SceneObstacleRules.from_robot_config(robot_cfg), "side_approaches": True}


def build_rehearsal_components(robot_cfg: "RobotConfig", *, data_dir: "str | Path | None" = None,
                               ) -> tuple[Any, Any, Any, Any]:
    """``(calculator, perception, frame_resolver, multi_camera)`` for a desk.

    Kept as its own function beside the cell builder so a caller that wants the pieces, to swap
    one or to inspect one, does not have to take the whole service to get them.

    ``data_dir`` is the config tree the cell came from, whose gripper registry answers for its hand.
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
        data_dir=data_dir,
        camera_matrix=rehearsal_intrinsics(),
        max_grip_width_mm=robot_cfg.gripper.max_width_mm,
        min_grip_width_mm=robot_cfg.gripper.min_width_mm,
        # The support-footprint stage needs the BASE support plane the pick loop resolves and the CAMERA->BASE
        # transform the resolver below provides; without either a call offers nothing and says so in telemetry.
        **_scene_kwargs(robot_cfg),
    )
    # A nadir camera 800 mm above the workspace centre, looking straight down. Stands in for the
    # artifact a real eye-to-hand calibration writes.
    resolver = StaticCameraToBaseResolver(transform=Transform(
        translation_mm=np.array([400.0, 0.0, 800.0]),
        quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
        from_frame=Frame.CAMERA, to_frame=Frame.BASE,
    ))
    # A desk has no cameras at all, so it has no other cameras. The last slot is here so the two
    # component builders keep the same shape and a reader is not left wondering which one is missing
    # something.
    return calculator, perception, resolver, None


def build_real_components(robot_cfg: "RobotConfig", prompt: str, *,
                          app_config: "Maybe[AppConfig]" = UNSET,
                          data_dir: "str | Path | None" = None,
                          ) -> tuple[Any, Any, Any, Any]:
    """``(calculator, perception, frame_resolver=None, multi_camera)``.

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
    from src.camera.orchestration.camera import Camera, CameraRefused
    from src.models.perception_spec import PerceptionSpec
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.perception import RealSenseVisionPerceptionSource

    app_cfg = app_config if chosen(app_config) else load_config(data_dir)
    # The cell's camera is named, not inferred. This took the first rig with `source: rgbd` by list
    # position and did not consult `enabled`, while the schema's own two-camera serial check used
    # `source == "rgbd" and enabled` and this file's fusion guard used a third rule. Three predicates
    # over one list, disagreeing (the fusion's is `camera_fusion.fused_cameras` since 2026-10-16). On the shipped base profile the consequence was concrete: its only
    # RGB-D rig ships `enabled: false`, and this opened it anyway, so the failure arrived at the
    # device instead of at the config that caused it.
    #
    # The refusals (a rig the section does not configure, one switched off, a stereo pair whose
    # stereo calibration is not written yet) are the camera noun's, so the cell, the calibration CLI
    # and the exerciser say one thing. A stereo primary searches its depth from where the robot's
    # workspace box begins (`working_depth_range_mm`), so the box is handed over with the section.
    #
    # Which rig is primary: whichever `camera.cameras.primary_rig_id` names. The grasp is
    # synthesised from this camera and every other camera confirms what it sees, so moving the
    # primary moves every measured number this cell has.
    try:
        primary = Camera.from_config(app_cfg.camera, workspace=robot_cfg.motion.workspace)
    except CameraRefused as refused:
        raise CellBuildRefused(str(refused)) from refused
    rig_id = primary.rig_id
    # Every camera has one owner. A `Camera` is the only opener of its device in this process: a
    # second opener is refused naming the holder, and every grab through any of its handles runs
    # under the rig's lock. `camera.handle()` is shaped exactly like the streamer the perception
    # adapter duck-types against, so nothing in `robot.perception` has to know about camera
    # orchestration, and `handle.release()` gives back this rig only, so the console's teardown
    # (`execution.lifecycle.release_perception` calling `perception.close()`) closes exactly the
    # device it opened.
    cameras: dict[str, Any] = {rig_id: primary}
    primary.open()
    try:
        return _build_on_open_cameras(
            robot_cfg, app_cfg, prompt=prompt, cameras=cameras, rig_id=rig_id, np=np, data_dir=data_dir,
            perception_spec=PerceptionSpec, build_calculator=build_calculator,
            vision_source=RealSenseVisionPerceptionSource,
        )
    except BaseException:
        # A refused build gives the cameras back. Everything below the open can raise, and one of
        # those raises is deliberate: `_preload()` exists so a corrupt artifact is refused at build
        # time rather than at 3 a.m. Without this, the refusal it was added to produce left the
        # RealSense claimed by a dead process, and the operator who fixed the artifact and ran again
        # met "device busy", which names neither the cause nor the first failure. `release` never
        # raises and is idempotent, so it cannot replace the exception being reported.
        for camera in cameras.values():  # every camera this build opened, extras included
            camera.release()
        raise


def _build_on_open_cameras(robot_cfg: "RobotConfig", app_cfg: Any, *, prompt: str,
                           cameras: "dict[str, Any]",
                           rig_id: str, np: Any, perception_spec: Any, build_calculator: Any,
                           vision_source: Any, data_dir: "str | Path | None" = None,
                           ) -> tuple[Any, Any, Any, Any]:
    """The half of :func:`build_real_components` that runs with a device already held.

    Split out for the `try` above rather than for its own sake: wrapping the tail in place would
    have indented sixty lines of load-bearing commentary, and a diff that moves every line is a diff
    whose one real change nobody can see. The imports are passed in because they are function-local
    by design, so that a rehearsal, a test or a sim runner never pays for torch.
    """
    handle = cameras[rig_id].handle()
    # The stack is built once and shared by every camera. It is the expensive part of a cell (two
    # models on the GPU) and it is stateless per call, so a four-camera rig costs four inference
    # passes and one set of weights, not four sets.
    #
    # It goes through `PerceptionSpec`, so `models.pipeline` (the VLM, or RT-DETR's classes)
    # selects the stack, every model reads its weights under `models.weights_root`, and
    # `build_perception`'s fail-closed refusals are reachable from here.
    spec = perception_spec.from_config(app_cfg.models)
    backend = spec.build()
    perception = vision_source(
        streamer=handle,
        backend=backend,
        prompt=prompt,
    )
    intrinsics = np.asarray(handle.get_intrinsics(), dtype=np.float64)
    multi_camera = _build_multi_camera_rig(
        robot_cfg, app_cfg, cameras=cameras, primary_rig_id=rig_id,
        backend=backend, prompt=prompt,
    )
    # Through the factory, as in `build_rehearsal_components`. This is the path a physical cell
    # takes, so it is the one where a silently ignored `calculator: deep` files the analytic
    # stack's numbers under the learned generator's name.
    calculator = build_calculator(
        robot_cfg,
        data_dir=data_dir,
        camera_matrix=intrinsics,
        max_grip_width_mm=robot_cfg.gripper.max_width_mm,
        min_grip_width_mm=robot_cfg.gripper.min_width_mm,
        **_scene_kwargs(robot_cfg),
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
    return calculator, perception, None, multi_camera


def _build_multi_camera_rig(robot_cfg: "RobotConfig", app_cfg: Any, *, cameras: "dict[str, Any]",
                            primary_rig_id: str, backend: Any,
                            prompt: str) -> Any:
    """The other cameras of a fused cell, as a `MultiCameraPerceptionSource`, or None.

    `apply_orchestrator_overlays` wires the per-camera resolver map from config, but the rig itself
    is a live device handle a caller must pass in. This is what builds it, so a physical cell whose
    camera section declares calibrated cameras fuses them instead of loading their artifacts
    fail-closed and still running single-view. The cameras are `camera_fusion.fused_cameras`: every
    enabled rig that declares its calibration, RGB-D or stereo, where two or more do (the owner,
    2026-10-16: no switch).

    The primary is deliberately not in here. `fuse_scene_geometry` takes the primary view as its
    first four arguments and the rest as `other_views`; a primary that appears in both is fused
    with a second copy of itself, and on hardware each extra observation costs a full detect plus
    segment pass. The Isaac runner does include its primary in the map; that is a redundancy rather
    than an error, and it is left as it is because changing it would move a measured number
    without a measurement.

    Returns None where the cell fuses no camera besides the primary, which keeps every one-camera
    cell byte-identical.

    Never run on hardware. No physical multi-camera cell has been built by this function.
    """
    from src.camera.orchestration.camera import Camera, CameraRefused
    from src.robot.execution.camera_fusion import fused_cameras
    from src.robot.grasping.types.perception import MappedCameraRig
    from src.robot.perception import RealSenseVisionPerceptionSource

    wanted = [cam_id for cam_id in fused_cameras(app_cfg.camera.cameras.rigs, primary_rig_id)
              if cam_id != primary_rig_id]
    if not wanted:
        return None

    sources = {}
    for cam_id in wanted:
        # Through the camera noun, as the primary is opened, so the cell says the camera's own
        # refusals. An opened camera joins `cameras`, so a later refusal gives it back too.
        try:
            camera = Camera.from_config(app_cfg.camera, rig_id=cam_id, workspace=robot_cfg.motion.workspace)
        except CameraRefused as refused:
            raise CellBuildRefused(f"the fused camera {cam_id!r}: {refused}") from refused
        camera.open()
        cameras[cam_id] = camera
        sources[cam_id] = RealSenseVisionPerceptionSource(
            streamer=camera.handle(),
            # The same backend object, not a second build. Every camera in a fused cell grounds
            # with one set of weights; the cost of an extra camera is an inference pass.
            backend=backend,
            prompt=prompt,
        )
    return MappedCameraRig(sources)


def build_real_cell(robot_cfg: "RobotConfig", *, prompt: str = "object",
                    app_config: "Maybe[AppConfig]" = UNSET,
                    data_dir: "str | Path | None" = None,
                    **overrides: Any) -> "AutonomousGraspService":
    """A complete cell for physical hardware, from config, in one call.

    Opens the primary RGB-D camera, every fused camera and every other calibrated rig a live
    planner world takes, and loads two models onto the GPU. It does not connect the arm: the
    caller owns that lifecycle, deliberately, because connecting a gripper drives digital I/O and
    the order matters (arm first).

    ``overrides`` are forwarded to :meth:`AutonomousGraspService.from_robot_config`, so a caller
    that needs a mode, a policy or a live device handle still has one.

    ``app_config`` is the camera half's tree, and it must be the tree ``robot_cfg`` came from. See
    :func:`build_real_components` for what happened while the two could disagree. Left unset, the
    tree at ``data_dir`` is read, as ``Cell.preflight`` and ``Cell.abilities`` read it.

    ``data_dir`` is that tree's root, whose gripper registry answers for the cell's hand; ``None``
    is the repository's tree. A tree that describes the named hand differently from the
    repository's registry is refused first, before a camera opens.
    """
    from src.config import load_config
    from src.robot.execution.abilities import Ability, AbilityStatus, row_of
    from src.robot.execution.wrist_bodies import WristBodies

    from .service import AutonomousGraspService

    # Resolved once, here, and handed to both readers below. It used to be read here a second time
    # with a comment explaining that the loader caches, so the two reads return the same object.
    # That comment was true and it was not about the right thing: two calls agreeing with each
    # other says nothing about whether either is the tree the caller asked for, and they were not.
    # Left unset, the camera half of the tree at `data_dir`, the one the desk's rows and the checklist read
    # (`Cell.abilities`, `Cell.preflight`): the build asks the same pick row they ask, so reading another tree's
    # cameras here let the two answer differently (review of P0i i3). The chain is still the environment's.
    app_cfg = app_config if chosen(app_config) else load_config(data_dir)
    # What the desk's pick row reads refuses the build before any camera opens, in the row's words (P0i i3): a
    # tree that describes the named hand differently from the repository's registry, a wrist camera body this
    # cell cannot place, and on a cell that plans with cuRobo a calibrated camera that yields no live world,
    # since the pick service declines nothing and every motion would be refused. A refusal here is one nobody
    # has to give a device back for.
    row = row_of(Ability.PICK, arm=None, gripper=None, robot_config=robot_cfg, camera=app_cfg.camera,
                 data_dir=data_dir)
    if row.status is AbilityStatus.REFUSED:
        # The chain the refusal was read from, where there was one: a wrist body's calibration error, for the log.
        raise CellBuildRefused(row.reason) from row.cause
    # The wrist cameras the arm carries, which the row has just placed.
    wrist = WristBodies.from_config(robot_cfg, app_cfg.camera, data_dir=data_dir)
    calculator, perception, resolver, multi_camera = build_real_components(
        robot_cfg, prompt, app_config=app_cfg, data_dir=data_dir)
    rig_id = app_cfg.camera.cameras.primary_rig_id
    try:
        service = AutonomousGraspService.from_robot_config(
            robot_cfg, calculator=calculator, perception=perception, frame_resolver=resolver,
            multi_camera_perception=multi_camera,
            # Which camera is the primary, told to the root rather than inferred there. It cannot
            # read it: the key lives in the camera section and the root is handed a `RobotConfig`.
            # Without it the primary is expected among the cameras a fused pick waits for, and it
            # delivers through `perception` instead, so it would be missing on every pick.
            primary_camera_id=rig_id,
            # Each camera's calibration is declared on its rig, so the root reads the camera section
            # of the same tree for the primary's resolver and a fused cell's resolver map.
            camera=app_cfg.camera,
            # What a task takes where it says nothing itself, whether the parts are critical among it.
            app_runtime=app_cfg.runtime,
            **overrides,
        )
        _stamp_wrist_pick_frames(robot_cfg, service, perception)
        # Before the world, because the world's self filter reads the bodies the arm holds.
        _wire_wrist_bodies(service, wrist)
        _wire_live_planner_world(robot_cfg, service, perception, app_cfg=app_cfg)
        _write_records_in_the_background(service)
    except BaseException:
        # A refused build gives the cameras back past the components as well.
        # `build_real_components` gives back what it opened when it fails itself, but the root, the
        # stamp and the world wiring all run afterwards with every camera held, and a refusal there
        # would leave the devices claimed until the objects were collected, so the next build would
        # meet a busy device. The wiring gives back the cameras it opened before it raises.
        _give_back(perception, multi_camera)
        raise
    return service


def _write_records_in_the_background(service: Any) -> None:
    """What keeps a pick of a real cell from waiting on its records: they are written in the background
    (``AutonomousGraspService.write_records_in_the_background``). Nothing here moves or opens a device."""
    write_later = getattr(service, "write_records_in_the_background", None)
    if callable(write_later):
        write_later(True)


def _plans_with_curobo(robot_cfg: Any) -> bool:
    """Whether this tree's planner is cuRobo, where every motion needs a world or a decline.

    Read off the tree's selection (``RobotConfig.arm``), which answers the planner only for a vendor whose driver reads
    ``robot.motion.planner``, so a dummy or a sim tree never counts.
    """
    selected = getattr(robot_cfg, "arm", None)
    return selected is not None and selected.planner == "curobo"


def _wire_wrist_bodies(service: Any, wrist: Any) -> None:
    """Hand the service's arm this cell's wrist camera bodies, or refuse an arm that cannot hold them."""
    from src.robot.execution.wrist_bodies import WristBodyRequired

    orchestrator: Any = getattr(getattr(service, "runtime", None), "orchestrator", None)
    try:
        wrist.hand_to(getattr(orchestrator, "arm", None))
    except (WristBodyRequired, ValueError, RuntimeError) as exc:
        raise CellBuildRefused(str(exc)) from exc


def _stamp_wrist_pick_frames(robot_cfg: Any, service: Any, perception: Any) -> None:
    """Give every wrist camera's pick source the arm's TCP reader, its rig's tolerance and attempts.

    Decided by the resolver the root built, because the resolver is what reads the tool pose: an
    eye-in-hand resolver composes it into every grasp, and a fixed camera's resolver never asks. So
    the primary is bound when the root's resolver is eye in hand, and so is every fused source whose
    resolver in ``camera_frame_resolvers`` is. A wrist cell whose frames go unstamped places each
    grasp by where the tool is when the grasp is resolved, not by where it was when the camera saw
    the part. A stamped frame carries the pose at its shutter, and one taken while the arm moved
    beyond the rig's shutter tolerance is taken again up to
    ``shutter_motion.FRESH_FRAME_ATTEMPTS`` more times before the pick stops.

    Each rig's calibration also has to have been solved against the flange to TCP this cell
    declares, whether or not the rig declares a body: a wrist frame is placed by CAMERA to TOOL, and
    a calibration made against another tool frame puts every grasp off by the difference.
    """
    from src.calibration.rig_calibration import flange_to_tcp_refusal
    from src.robot.core.shutter_motion import FRESH_FRAME_ATTEMPTS
    from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver

    orchestrator: Any = getattr(getattr(service, "runtime", None), "orchestrator", None)
    sources = []
    if isinstance(getattr(orchestrator, "frame_resolver", None), EyeInHandFrameResolver):
        sources.append(perception)
    resolvers = getattr(orchestrator, "camera_frame_resolvers", None) or {}
    fused = getattr(getattr(orchestrator, "multi_camera_perception", None), "sources", None) or {}
    for cam_id, source in fused.items():
        if source is not perception and isinstance(resolvers.get(cam_id), EyeInHandFrameResolver):
            sources.append(source)
    if not sources:
        return
    attempts = FRESH_FRAME_ATTEMPTS
    for source in sources:
        calibration = source.streamer.camera.calibration()
        stale = flange_to_tcp_refusal(calibration, robot_cfg.gripper.tool_frame)
        if stale is not None:
            raise CellBuildRefused(stale)
        source.stamp_tool_pose_with(
            orchestrator.arm.get_tcp_pose,
            motion_tolerance=(float(calibration.shutter_motion_tolerance_mm),
                              float(calibration.shutter_motion_tolerance_deg)),
            attempts=attempts,
        )


def _wire_live_planner_world(robot_cfg: "RobotConfig", service: Any, perception: Any, *, app_cfg: Any) -> None:
    """Give the arm a world built from every camera that can say what the cell looks like, if asked.

    Here rather than inside the service, because this is the only place that holds all three
    pieces: the cameras the components builder opened, the calibration each rig declares, and the
    arm the service built. The layers below cannot reach across those and are not supposed to.

    Which rigs feed the world is `CameraWorldPlan`'s answer: every enabled rig that declares its
    calibration, RGB-D or stereo, the primary first. The primary and the fused cameras are already open and are
    taken as they are. Every other planned rig is opened here, held on the orchestrator as
    `planner_world_cameras`, and given back with the rest on teardown. A cell that did not ask
    opens nothing more and logs nothing. A camera that cannot be opened or placed refuses the build
    naming it, after the cameras opened here are given back.
    """
    from src.camera.orchestration.camera import Camera
    from src.robot.constants import create_robot_logger
    from src.robot.core.camera_world import holds_a_planner_world
    from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldWiring, OpenedCameras

    log = create_robot_logger(__name__, "grasp_cells.log")
    orchestrator: Any = getattr(getattr(service, "runtime", None), "orchestrator", None)
    arm = getattr(orchestrator, "arm", None)
    # An arm that can hold a live world is the one to hand it to; a world handed to anything else is a world nobody
    # refreshes.
    holder = holds_a_planner_world(arm)
    if holder is None:
        return
    section = app_cfg.camera.cameras
    plan = CameraWorldPlan.from_config(robot_cfg, section.rigs, primary_rig_id=section.primary_rig_id)
    if not plan.rig_ids:
        if plan.asked:
            log.warning("no live planner world for this cell: %s", plan.render())
        return

    fused = getattr(getattr(orchestrator, "multi_camera_perception", None), "sources", None) or {}
    owners: dict[str, Any] = {}
    opened: list[Any] = []
    for rig_id in plan.rig_ids:
        try:
            if rig_id == section.primary_rig_id:
                owners[rig_id] = perception.streamer.camera
            elif rig_id in fused:
                owners[rig_id] = fused[rig_id].streamer.camera
            else:
                camera = Camera.from_config(app_cfg.camera, rig_id=rig_id, workspace=robot_cfg.motion.workspace)
                camera.open()
                opened.append(camera)
                owners[rig_id] = camera
        except Exception as exc:
            _give_back(OpenedCameras(tuple(opened)))
            raise CellBuildRefused(
                f"the live planner world cannot open camera {rig_id!r}: {exc}. A cell that enables "
                "safety.planning_world takes every enabled rig that declares its calibration, so free that "
                "camera or switch its rig off"
            ) from exc
    reader = getattr(arm, "get_tcp_pose", None)
    try:
        wiring = CameraWorldWiring.from_cameras(
            robot_cfg, plan=plan, cameras=owners, tool_pose=reader if callable(reader) else UNSET)
    except Exception as exc:
        _give_back(OpenedCameras(tuple(opened)))
        raise CellBuildRefused(f"the live planner world cannot place its cameras: {exc}") from exc

    if wiring.world is None:
        _give_back(OpenedCameras(tuple(opened)))
        if _plans_with_curobo(robot_cfg):
            # The plan named cameras and they built no world (a stereo answer, a wrist rig with no
            # tool reader). On a cuRobo cell that refuses every motion, so the build is refused.
            raise CellBuildRefused(
                f"this cell plans with cuRobo and its calibrated cameras built no live world: {wiring.reason}")
        log.warning("%s", wiring.render())
        return
    if opened:
        orchestrator.planner_world_cameras = OpenedCameras(tuple(opened))
    holder.set_live_planner_world(wiring.world)
    if wiring.planner == "curobo":
        log.info("%s", wiring.render())
    else:
        log.warning("%s", wiring.render())


def _give_back(*holders: Any) -> None:
    """Close every holder that can be closed, and never raise.

    This runs while a refusal is on its way out, so a camera that would not close must not replace
    the exception being reported.
    """
    for holder in holders:
        close = getattr(holder, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:  # noqa: BLE001 (the refusal being reported outranks a camera that would not close)
            pass


def build_rehearsal_cell(robot_cfg: "RobotConfig", *, data_dir: "str | Path | None" = None,
                         app_config: "Maybe[AppConfig]" = UNSET, **overrides: Any) -> "AutonomousGraspService":
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

    A tree that describes the named hand differently from the repository's registry is refused
    first, with the same sentence the real cell and the desk give.

    ``app_config`` is the camera half of the same tree, which a caller holding both halves hands in (the CLI
    and the console do). A rehearsal opens no camera, and still reads the calibrations its grasping block
    names: a cell that fuses two cameras was refused at this build because no camera section reached it
    (found 2026-09-26 on ``eth2``), so handed its camera half, its rehearsal builds on the calibrations its rigs
    declare, or is refused naming the one that does not load, and runs single view on the synthetic scene,
    saying so. Left unset, no camera half is read: ``robot_cfg`` does not say which chain it came from, and a
    second load here would read another tree's cameras.
    """
    from .service import AutonomousGraspService

    _refuse_a_tree_hand(robot_cfg, data_dir)

    camera: dict[str, Any] = ({"camera": app_config.camera, "app_runtime": app_config.runtime} if chosen(app_config)
                              else {})
    robot_cfg = robot_cfg.model_copy(update={"vendor": "dummy"})
    calculator, perception, resolver, _no_cameras = build_rehearsal_components(robot_cfg, data_dir=data_dir)
    return AutonomousGraspService.from_robot_config(
        robot_cfg, calculator=calculator, perception=perception, frame_resolver=resolver,
        **camera, **overrides,
    )
