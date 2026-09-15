"""Isaac-Sim driver + scene config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel

from ._ur_models import UR_MODEL_KEYS  # re-exported: the sim cell and the real UR cell share one list


class SimCameraSchema(StrictModel):
    """Per-camera entry inside :class:`SimConfig.cameras`.

    Pydantic mirror of the driver-side dataclass ``src.robot.drivers.sim.config.SimCameraConfig``,
    which the runtime builds from this model at arm-construction time. Kept minimal: a new knob
    lands on the driver dataclass first and mirrors here afterwards.
    """

    prim_path: str = Field(default="", min_length=0)
    mounting_mode: Literal["eye_to_hand", "eye_in_hand"] = "eye_to_hand"
    # --- Isaac authoring extras (read by src.willy_sim, not the bare driver) ---
    near_clip_m: float | None = Field(default=None, gt=0.0)
    resolution: tuple[int, int] | None = None
    # Horizontal field of view (deg) the lens is authored from, meaning the FOV of the real sensor
    # this camera stands in for; an Intel RealSense D435's RGB stream is 69.4 deg. None keeps
    # Isaac's default lens, which is narrower and frames less of the table than the real cell does,
    # a silent sim2real gap. See willy_sim/scene/cameras.py (D435_RGB_HFOV_DEG).
    hfov_deg: float | None = Field(default=None, gt=0.0, lt=180.0)
    # Fixed (eye-to-hand) cameras: world position in mm (None -> authored elsewhere).
    position_mm: tuple[float, float, float] | None = None
    # Eye-in-hand cameras: mount in the wrist-link local frame (mm) + look-at aim + up hint.
    mount_offset_mm: tuple[float, float, float] | None = None
    mount_aim_mm: tuple[float, float, float] | None = None
    mount_up_hint: tuple[float, float, float] | None = None


class SimObjectConfig(StrictModel):
    """Graspable object authored into the sim scene (willy_sim scene-authoring)."""

    name: str = "cube"  # identity handle for prompt-based selection among clutter ("the green cube")
    # Procedural primitive shape, ignored when usd_asset_path is set. "cube" is a DynamicCuboid
    # with size_mm as the box extents. "cylinder" is a DynamicCylinder with size_mm as (diameter,
    # diameter, height), the round body a 3-finger centric gripper like the Schunk EZU-35 wraps.
    shape: str = "cube"
    size_mm: tuple[float, float, float] = (30.0, 30.0, 50.0)
    position_mm: tuple[float, float, float] = (450.0, 0.0, 25.0)
    mass_kg: float = Field(default=0.05, gt=0.0)
    static_friction: float = Field(default=1.5, ge=0.0)
    dynamic_friction: float = Field(default=1.3, ge=0.0)
    color: tuple[float, float, float] = (0.9, 0.2, 0.1)  # RGB 0-1; the VL colour disambiguation handle
    usd_asset_path: str | None = None  # relative /Isaac/... USD asset (YCB mesh); None -> DynamicCuboid
    # Spawn orientation (world, WXYZ); None is identity. A referenced YCB authored lying down, such
    # as the soup can, is spawned upright so it settles stable and reliably graspable. Ignored for
    # procedural cubes.
    orientation_wxyz: tuple[float, float, float, float] | None = None
    # Collision approximation for a referenced mesh without physics: the /Isaac/Props/YCB/Axis_Aligned/
    # variants carry no collider and no rigid-body, unlike Axis_Aligned_Physics/. None authors nothing,
    # leaving the pre-rigged Axis_Aligned_Physics YCB untouched. "convexHull" (Isaac's own recipe for
    # 003-006) and "convexDecomposition" author UsdPhysics on the mesh at runtime, making it a
    # graspable rigid body. Ignored for procedural cubes.
    usd_collision_approximation: str | None = None


class SimTableConfig(StrictModel):
    """Static table authored into the sim scene."""

    size_mm: tuple[float, float, float] = (800.0, 800.0, 400.0)
    position_mm: tuple[float, float, float] = (450.0, 0.0, -200.0)


class SimMarkerConfig(StrictModel):
    """Dedicated hand-eye calibration marker. Marker length and ArUco dictionary come from
    ``camera.hand_eye.eye_in_hand``, the single source of truth; this block holds placement and
    kind."""

    kind: Literal["none", "flat", "aruco"] = "none"
    position_mm: tuple[float, float, float] = (450.0, -150.0, 6.0)
    flat_size_mm: tuple[float, float, float] = (80.0, 80.0, 10.0)
    aruco_plate_size_mm: float = Field(default=72.0, gt=0.0)
    aruco_marker_id: int = Field(default=0, ge=0)


class SimViewpointConfig(StrictModel):
    """Eye-in-hand look-at viewpoint pattern (hemisphere around the marker)."""

    radii_mm: list[float] = Field(default_factory=lambda: [235.0, 275.0])
    elevations_deg: list[float] = Field(default_factory=lambda: [16.0, 30.0])
    azimuths_deg: list[float] = Field(
        default_factory=lambda: [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    )


class SimGateConfig(StrictModel):
    """Pick/lift gate thresholds for the sim validation runners."""

    lift_threshold_mm: float = Field(default=50.0, gt=0.0)
    pass_fraction: float = Field(default=0.8, gt=0.0, le=1.0)


class SimSceneConfig(StrictModel):
    """Scene-authoring and experiment parameters, read by src.willy_sim only."""

    object: SimObjectConfig = Field(default_factory=SimObjectConfig)
    objects: list[SimObjectConfig] = Field(default_factory=list)  # multi-object clutter; empty -> [object]
    table: SimTableConfig = Field(default_factory=SimTableConfig)
    marker: SimMarkerConfig = Field(default_factory=SimMarkerConfig)
    eih_viewpoints: SimViewpointConfig = Field(default_factory=SimViewpointConfig)
    gate: SimGateConfig = Field(default_factory=SimGateConfig)
    render_warmup_steps: int = Field(default=20, ge=0)


class SimConfig(StrictModel):
    """Isaac-Sim (or pure-Python mock) driver settings.

    Read only when ``robot.vendor == "sim"``. Mirrors the driver dataclass
    ``src.robot.drivers.sim.config.SimRobotConfig``, so YAML edits stay validated on hosts with no
    Isaac SDK installed.
    """

    backend: Literal["isaac"] = "isaac"
    enabled: bool = False
    mock_mode: bool = False
    # Which UR model this sim cell drives: "ur5e" (the default), "ur3e" or "ur10e". It selects the
    # Lula config, the Isaac USD the scene builder references and the driver's cuRobo {key}.yml.
    # The safety block's kinematics_model has to match it. See sim/robot_models.py.
    robot_model: str = "ur5e"
    scene: str | None = None
    robot_prim_path: str | None = None
    gripper_prim_path: str | None = None
    home_joint_positions: list[float] | None = None
    cameras: dict[str, SimCameraSchema] = Field(default_factory=dict)
    step_dt_s: float = Field(default=1.0 / 60.0, gt=0.0, le=1.0)
    settle_timeout_s: float = Field(default=5.0, gt=0.0, le=120.0)
    headless: bool = True
    # --- Isaac scene-authoring extras (read by src.willy_sim, not forwarded to the bare
    # driver dataclass; the "robot.sim" namespace owns the whole simulated work-cell). ---
    assets_root: str | None = None              # Isaac asset pack root (carb asset_root + scene build)
    # No key here names the gripper. Which one Isaac puts on the arm, a baked USD variant or a mounted
    # standalone asset, is derived from the hand, ``robot.gripper.model``, and ``robot_model`` above
    # (``willy_sim.grippers.sim_mount_for``). ``gripper_mount`` and ``gripper_variant`` are not keys,
    # because each was a second name for the hand (src/config/schema/_removed.py).
    # Which suction cup to mount for a suction pick: a key in willy_sim.grippers.SUCTION_CUPS, such
    # as "slim" for a finer cup that reaches tighter gaps. None, the default, uses the standard cup.
    # Only the suction runners read it; the jaw cells ignore it.
    suction_cup: str | None = None
    park_joint_positions: list[float] | None = None  # arm parked clear of an overhead cam for perception
    scene_setup: SimSceneConfig = Field(default_factory=SimSceneConfig)

    @field_validator("robot_model")
    @classmethod
    def _known_robot_model(cls, v: str) -> str:
        """Reject an unknown or mistyped model at config validation (<1 s), not ~60 s into an Isaac boot.

        Without this, ``ur3``, ``UR3E `` and ``ur-3e`` validate fine and raise only inside
        ``ur_model_spec``, when the driver brings up the articulation (drivers/sim/arm.py) or the
        scene builder references the USD.
        """
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown robot_model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
