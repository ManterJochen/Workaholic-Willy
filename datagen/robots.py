"""Which robot may stand in a generated scene, and what it has to have declared first.

The arm is not decoration. It is this cell's dominant occluder, it carries the eye-in-hand camera,
and its reach decides whether the objects in a scene are ones a robot could ever pick. A generator
that places an arm it only half knows produces a dataset whose extrinsics, occlusion and
reachability are all wrong, and none of that is visible until a model has trained on it.

So this module is a contract, checked before anything renders:

* a robot is a key in the stack's own registry (:mod:`src.robot.drivers.sim.robot_models`),
  which is what ties one string to the DH table, the self-collision model and the cuRobo config;
* every fact the generator needs must be present, and a missing one is named: all of them at once,
  before Isaac boots, rather than one per failed run;
* the wrist camera mount must be measured, never assumed. Without it the wrist view is refused and
  the other views still render, because an invented mount writes a wrong extrinsic into every label
  of every scene while looking perfectly normal.

The measured mount is imported from the module that owns the measurement,
:mod:`src.willy_sim.scene.cameras`, rather than copied here. One number, one home; a copy is a
number that can drift without anyone touching it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.robot.drivers.sim.robot_models import URModelSpec, ur_model_spec
from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, ROBOTS_LOG_FILE
# Imported at module level rather than lazily: `datagen.render.camera` has no top-level datagen
# imports, so there is no cycle here to break, and it is cheap enough that deferring it buys
# nothing.
from datagen.render.camera import look_at_camera_to_base

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import numpy as np

__all__ = [
    "RobotDefinition",
    "RobotNotUsable",
    "WristCameraMount",
    "resolve_robot",
    "workspace_overshoot_mm",
]

logger = create_logger("datagen.robots", ROBOTS_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Isaac ships this gripper and the pick stack is validated with it, so a bare model gets it
#: attached rather than standing there as a naked flange: two datasets would otherwise show
#: different robots and a model could learn the silhouette instead of the task.
_DEFAULT_GRIPPER = "robotiq_2f85"
#: Clearance kept inside the reach sphere. At the very edge an arm is near-singular and has almost no
#: orientation freedom, so "just barely inside" is not usefully reachable for a top-down look.
_REACH_MARGIN_MM = 40.0


@dataclass(frozen=True, slots=True)
class WristCameraMount:
    """Where the eye-in-hand camera sits on the wrist link, in that link's own frame (mm).

    Every field here is a measurement, not a convention. ``aim_target_mm`` is a point rather than a
    direction because that is the form the measurement was taken and validated in.
    """

    offset_mm: tuple[float, float, float]
    aim_target_mm: tuple[float, float, float]
    up_hint: tuple[float, float, float]
    near_clip_m: float
    #: Where the measurement comes from, carried into the dataset so a consumer can trace it.
    source: str

    def camera_to_link(self) -> "np.ndarray":
        """4x4 CAMERA(CV optical) -> wrist link, from the measured offset and aim point.

        The same look-at construction the rest of the generator projects with
        (:mod:`datagen.render.camera`), applied in the link's frame instead of the base's.
        Deliberately not a second implementation: a mount built with one handedness and projected
        with another is a 180 deg roll that looks perfectly consistent and is wrong everywhere.
        """

        return look_at_camera_to_base(self.offset_mm, self.aim_target_mm, self.up_hint)


@dataclass(frozen=True, slots=True)
class RobotDefinition:
    """Everything the generator needs about one robot, complete by construction.

    There is no partially-valid instance: :func:`resolve_robot` either returns a whole one or raises
    with the full list of what is missing. A definition that is almost complete is the one that
    renders a whole dataset before anybody notices.
    """

    key: str
    usd_relpath: str
    wrist_link_name: str
    gripper: str
    gripper_is_baked: bool
    max_reach_mm: float
    max_payload_kg: float
    shoulder_height_mm: float
    workspace_center_mm: tuple[float, float]
    workspace_half_extents_mm: tuple[float, float]
    #: ``None`` means the mount was never measured for this model: the wrist view is refused, the rest
    #: of the rig still renders. Never a fallback to another robot's mount.
    wrist_camera_mount: WristCameraMount | None

    @property
    def can_render_wrist_view(self) -> bool:
        return self.wrist_camera_mount is not None


class RobotNotUsable(RuntimeError):
    """A robot the generator refuses to place, with every missing fact named at once."""


def declared_mount(spec: object) -> WristCameraMount | None:
    """Turn a config's `arm.wrist_camera_mount` into the value object, or None when unset.

    Takes a duck, not a `WristMountSpec`. `datagen.robots` is imported by `datagen.config`, so
    naming the config type here would close the loop. The five attributes are read by name, and
    anything carrying them works.
    """
    if spec is None:
        return None
    return WristCameraMount(
        offset_mm=tuple(float(v) for v in spec.offset_mm),          # type: ignore[attr-defined,arg-type]
        aim_target_mm=tuple(float(v) for v in spec.aim_target_mm),  # type: ignore[attr-defined,arg-type]
        up_hint=tuple(float(v) for v in spec.up_hint),              # type: ignore[attr-defined,arg-type]
        near_clip_m=float(spec.near_clip_m),                        # type: ignore[attr-defined]
        source=str(spec.source),                                    # type: ignore[attr-defined]
    )


def _measured_mount(key: str) -> WristCameraMount | None:
    """The measured eye-in-hand mount for ``key``, or ``None`` if nobody has measured one.

    Imported from the module that owns the measurement rather than restated here. The UR e-series
    share the wrist-3 flange geometry, but sharing a flange is not the same as having been measured,
    so only the model the calibration ran on is listed. Adding a model here means running
    ``run_eih_calibrate`` for it first; see :class:`RobotNotUsable`'s message.
    """
    from src.willy_sim.scene.cameras import (  # noqa: PLC0415 (keeps the import off the cheap path)
        WRIST_CAM_AIM_TARGET_MM,
        WRIST_CAM_NEAR_CLIP_M,
        WRIST_CAM_OFFSET_MM,
        WRIST_CAM_UP_HINT,
    )

    if key != "ur5e":
        return None
    return WristCameraMount(
        offset_mm=tuple(float(v) for v in WRIST_CAM_OFFSET_MM),  # type: ignore[arg-type]
        aim_target_mm=tuple(float(v) for v in WRIST_CAM_AIM_TARGET_MM),  # type: ignore[arg-type]
        up_hint=tuple(float(v) for v in WRIST_CAM_UP_HINT),  # type: ignore[arg-type]
        near_clip_m=float(WRIST_CAM_NEAR_CLIP_M),
        source="willy_sim.scene.cameras (measured on-box 2026-06-07, EIH pick 10/10)",
    )


def _shoulder_height_mm(key: str) -> float | None:
    """``d1``, the shoulder height that centres the reach sphere.

    ``None`` when the model has no DH table.
    """
    from src.robot.safety._ur_kinematics import UR_DH_TABLES_M  # noqa: PLC0415

    table = UR_DH_TABLES_M.get(key.lower())
    return float(table[0].d_m * 1000.0) if table else None


def _missing_facts(key: str, spec: URModelSpec, shoulder_mm: float | None) -> list[str]:
    """Every fact this robot is missing, as actionable lines. Empty means it may be placed."""
    missing: list[str] = []
    if not spec.usd_relpath:
        missing.append("usd_relpath, the arm asset to reference (URModelSpec.usd_relpath)")
    if not spec.wrist_link_name:
        missing.append("wrist_link_name, the link the tool and eye-in-hand camera hang from")
    if spec.max_reach_mm <= 0.0:
        missing.append("max_reach_mm, the datasheet reach, used to refuse unreachable workspaces")
    if shoulder_mm is None:
        missing.append(
            f"a DH row in src.robot.safety._ur_kinematics.UR_DH_TABLES_M['{key}'], without which "
            f"there is no forward kinematics, so neither the posed configuration nor the "
            f"self-collision check can be computed"
        )
    return missing


def resolve_robot(model: str, declared_mount: "WristCameraMount | None" = None,
                  ) -> RobotDefinition:
    """The complete definition for ``model``, or a refusal naming everything that is missing.

    Raises :class:`RobotNotUsable` rather than returning something partial, and lists all the gaps
    in one message rather than one per failed run. An unknown key is refused the same way, with the
    known keys listed.
    """
    try:
        spec = ur_model_spec(model)
    except ValueError as exc:
        raise RobotNotUsable(
            f"{exc}\n"
            f"To add one: register a URModelSpec (USD path, wrist link, reach, reachable workspace), "
            f"add its DH row to UR_DH_TABLES_M, build its cuRobo {model}.yml, then measure its "
            f"eye-in-hand mount with run_eih_calibrate. `python -m datagen verify-robot {model}` "
            f"checks all of it on-box before any dataset is built."
        ) from exc

    shoulder_mm = _shoulder_height_mm(spec.key)
    missing = _missing_facts(spec.key, spec, shoulder_mm)
    if missing:
        raise RobotNotUsable(
            f"robot {spec.key!r} cannot be placed in a scene; {len(missing)} fact(s) missing:\n  - "
            + "\n  - ".join(missing)
            + "\nNothing was rendered. Fix these first; `python -m datagen verify-robot "
            + f"{spec.key}` proves the result on-box."
        )

    # A declared mount wins, and it is the only way a non-ur5e cell gets an eye-in-hand view.
    # `_measured_mount` answers for `ur5e` alone, because the four constants it returns came from
    # running the calibration on that arm and the e-series sharing a flange is not the same as a
    # mount having been measured. That gate stays. Declaring the mount in config is what lets a
    # customer who has run their own calibration say so without editing a file inside `src/`, the
    # package datagen may only read from.
    mount = declared_mount or _measured_mount(spec.key)
    if mount is None:
        # Not an error and not silent either: the run continues and every wrist view is refused later.
        # Reading "0 wrist views" in a finished dataset without this line sends the reader to the
        # renderer for something that was decided here, before Isaac booted.
        logger.warning("%s has no MEASURED wrist-camera mount; wrist views will be refused "
                       "(run run_eih_calibrate for it, never borrow another model's)", spec.key)
    logger.info("resolved %s: reach %.0f mm, payload %.1f kg, shoulder %.0f mm, gripper %s%s",
                spec.key, float(spec.max_reach_mm), float(spec.max_payload_kg),
                float(shoulder_mm or 0.0),
                spec.baked_gripper_variant or _DEFAULT_GRIPPER,
                " (baked)" if spec.baked_gripper_variant is not None else "")
    return RobotDefinition(
        key=spec.key,
        usd_relpath=spec.usd_relpath,
        wrist_link_name=spec.wrist_link_name,
        gripper=spec.baked_gripper_variant or _DEFAULT_GRIPPER,
        gripper_is_baked=spec.baked_gripper_variant is not None,
        max_reach_mm=float(spec.max_reach_mm),
        max_payload_kg=float(spec.max_payload_kg),
        shoulder_height_mm=float(shoulder_mm or 0.0),
        workspace_center_mm=spec.workspace_center_mm,
        workspace_half_extents_mm=spec.workspace_half_extents_mm,
        wrist_camera_mount=mount,
    )


def workspace_overshoot_mm(
    definition: RobotDefinition,
    center_mm: tuple[float, float],
    half_extents_mm: tuple[float, float],
    table_height_mm: float = 0.0,
) -> float:
    """How far the worst corner of a workspace lies outside the reach sphere, in mm.

    0.0 means it fits. The corners, not the centre: a workspace whose centre is comfortable and
    whose far corner is out of reach produces scenes in which some objects can never be picked, and
    nothing about the images says which ones. Measured from the shoulder, because that is where the
    reach sphere is centred, and with `_REACH_MARGIN_MM` of clearance because the very edge of the
    sphere is near-singular rather than usefully reachable.
    """
    limit = max(0.0, definition.max_reach_mm - _REACH_MARGIN_MM)
    worst = 0.0
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            x = center_mm[0] + sign_x * half_extents_mm[0]
            y = center_mm[1] + sign_y * half_extents_mm[1]
            z = table_height_mm - definition.shoulder_height_mm
            worst = max(worst, math.sqrt(x * x + y * y + z * z) - limit)
    return max(0.0, worst)
