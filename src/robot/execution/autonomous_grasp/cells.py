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

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

    from .service import AutonomousGraspService

__all__ = ["build_real_cell", "build_real_components", "build_rehearsal_cell",
           "build_rehearsal_components"]


def build_rehearsal_components(robot_cfg: "RobotConfig") -> tuple[Any, Any, Any, Any]:
    """``(calculator, perception, frame_resolver, multi_camera=None)`` for a desk rehearsal.

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
    # A desk has no cameras at all, so it has no other cameras either. The fourth slot is here so
    # the two component builders keep the same shape and a reader is not left wondering which one is
    # missing something.
    return calculator, perception, resolver, None


def build_real_components(robot_cfg: "RobotConfig", prompt: str) -> tuple[Any, Any, Any, Any]:
    """``(calculator, perception, frame_resolver=None, multi_camera)`` for a physical cell.

    The resolver stays ``None`` on purpose: ``from_robot_config`` builds it from the calibration
    artifact named in config, and refuses if there is none. Passing one here would paper over exactly
    the misconfiguration the preflight is there to surface.

    ``multi_camera`` is the other cameras of a fused cell (``None`` for every single-camera one). It
    cannot come from ``from_robot_config``: the composition root is deliberately unable to open a
    device, so the cameras are opened here and handed in, exactly as ``perception`` is. See
    :func:`_build_multi_camera_rig` for what it does and does not include.
    """
    import numpy as np

    from src.config import load_config
    from src.camera.orchestration.frame_provider import FrameProvider
    from src.models.perception_spec import PerceptionSpec
    from src.robot.grasping.calculator_factory import build_calculator
    from src.robot.perception import RealSenseVisionPerceptionSource

    app_cfg = load_config()
    rgbd = [r for r in app_cfg.camera.cameras.rigs if getattr(r, "source", None) == "rgbd"]
    if not rgbd:
        raise SystemExit(
            "no RGB-D rig in camera.cameras.rigs; the real cell needs one. Enable the realsense rig "
            "in the camera config, and prove it standalone with `python -m src.robot.perception`."
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
    # Which rig is primary: the first RGB-D one. The grasp is synthesised from this camera and
    # every other camera confirms what it sees, so moving the primary would move every measured
    # number this cell has.
    provider = FrameProvider(list(app_cfg.camera.cameras.rigs))
    rig_id = rgbd[0].rig_id
    provider.open_rig(rig_id)
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
    spec = PerceptionSpec.from_config(app_cfg.models)
    backend = spec.build()
    perception = RealSenseVisionPerceptionSource(
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
    return calculator, perception, None, multi_camera


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
        raise SystemExit(
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
                    **overrides: Any) -> "AutonomousGraspService":
    """A complete cell for physical hardware, from config, in one call.

    Opens one RGB-D camera and loads two models onto the GPU. It does not connect the arm: the
    caller owns that lifecycle, deliberately, because connecting a gripper drives digital I/O and
    the order matters (arm first).

    ``overrides`` are forwarded to :meth:`AutonomousGraspService.from_robot_config`, so a caller
    that needs a mode, a policy or a live device handle still has one.
    """
    from .service import AutonomousGraspService

    calculator, perception, resolver, multi_camera = build_real_components(robot_cfg, prompt)
    return AutonomousGraspService.from_robot_config(
        robot_cfg, calculator=calculator, perception=perception, frame_resolver=resolver,
        multi_camera_perception=multi_camera,
        **overrides,
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
    calculator, perception, resolver, _no_cameras = build_rehearsal_components(robot_cfg)
    return AutonomousGraspService.from_robot_config(
        robot_cfg, calculator=calculator, perception=perception, frame_resolver=resolver,
        **overrides,
    )
