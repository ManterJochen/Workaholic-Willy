"""The Universal Robots model registry for the Isaac sim driver.

This stack targets more than one UR, a UR5e cell and a UR3e for hardware validation, so
the model is a config choice: :attr:`SimRobotConfig.robot_model` selects a key here and
the driver and the scene builder derive the model-specific parts from it.

The canonical model key, "ur5e" or "ur3e", is the one string the whole stack shares:

  * the safety self-collision DH table, ``UR_DH_TABLES_M`` in
    :mod:`src.robot.safety._ur_kinematics`;
  * the cuRobo robot config filename, ``{key}.yml``, built on-box by
    ``scripts/curobo/build_{key}_config.py``;
  * the ``kinematics_model`` of the safety config.

This registry adds the two Isaac-specific parts that key cannot express: the Lula
supported-config name, which is capitalised as "UR3e", and the USD relative path in the
Isaac asset pack, which is the combined arm and gripper asset.

It is import-safe, being pure data with no ``isaacsim`` import, so it loads anywhere the
rest of the config does.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "URModelSpec",
    "curobo_robot_yml",
    "ur_model_spec",
]


@dataclass(frozen=True, slots=True)
class URModelSpec:
    """The model-specific parts the sim driver and the scene builder need.

    ``key`` is the canonical lower-case model key, the same string as the safety
    ``kinematics_model`` and the ``{key}.yml`` cuRobo config. ``lula_key`` is the Isaac
    supported-config name passed to
    ``interface_config_loader.load_supported_lula_kinematics_solver_config`` and
    ``load_supported_motion_policy_config``. ``usd_relpath`` is appended to the Isaac
    assets root to reference the combined arm USD, with any baked gripper.
    """

    key: str
    lula_key: str
    usd_relpath: str
    #: The ``Gripper`` USD variant the asset of this model bakes in, or ``None`` where the
    #: asset ships bare. The Isaac ur5e and ur10e assets bake a Robotiq 2F-85, and
    #: ``ur3e.usd`` bakes no gripper at all, verified at byte level: it carries neither a
    #: ``Gripper`` variant set nor a ``Robotiq`` token. A bare model therefore requires
    #: ``SimConfig.gripper_mount``, or the cell comes up as a bare 6-DoF arm and fails
    #: much later with a driven joint 'finger_joint' missing from the gripper dof_names.
    baked_gripper_variant: str | None = None
    #: The vendor-datasheet reach in mm, which is the radius of the working sphere about
    #: the shoulder, and the payload cap in kg. They reject a scene or pose the arm cannot
    #: physically touch, or a mass it cannot lift, at config time rather than after an
    #: Isaac boot and an inscrutable IK failure. Swapping a UR5e cell to a UR3e shrinks
    #: the reach from 850 to 500 mm, which is why the UR5e-tuned scenes, with bins at 515
    #: to 644 mm, are unreachable and have to be re-anchored.
    max_reach_mm: float = 850.0
    max_payload_kg: float = 5.0
    #: The link the eye-in-hand camera and the tool hang from, relative to the robot root
    #: prim. Every UR e-series shares ``wrist_3_link``. It is a field rather than a
    #: constant because the wrist link is the first thing a non-UR arm changes, and a
    #: wrong guess mounts the camera on the wrong body without erroring: the image simply
    #: moves with something else.
    wrist_link_name: str = "wrist_3_link"
    #: A workspace this model can reach, in BASE-frame mm as a centre and half-extents,
    #: which is the patch a scene generator places objects in. It is stated per model
    #: rather than derived, which makes it inspectable, and CI asserts that each one fits
    #: inside the reach sphere of its model, so an explicit value cannot quietly become a
    #: wrong one. The UR5e pair is the measured, validated pick patch the sim runners use,
    #: and the UR3e pair is pulled in towards the base because a reach of 500 mm rather
    #: than 850 mm puts the UR5e patch outside the sphere.
    workspace_center_mm: tuple[float, float] = (450.0, 0.0)
    workspace_half_extents_mm: tuple[float, float] = (150.0, 150.0)


# The UR e-series models Isaac ships a USD for, verified on-box: the ur3e and ur5e asset
# directories exist under ``.../Isaac/Robots/UniversalRobots/``. Each key also has a
# bundled DH row in ``_ur_kinematics`` and a buildable ``{key}.yml`` cuRobo config.
# Extend it as more UR variants are validated.
_UR_MODELS: dict[str, URModelSpec] = {
    # ur3e ships bare, with no baked gripper, so a ur3e cell sets gripper_mount.
    "ur3e": URModelSpec("ur3e", "UR3e", "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd",
                        max_reach_mm=500.0, max_payload_kg=3.0,
                        # Computed rather than guessed: a guessed 330 +- 110 comes out
                        # 18.3 mm over. 500 mm of reach less a 40 mm dexterity margin,
                        # about a 151.9 mm shoulder, leaves 434 mm of horizontal radius
                        # at table height, and the far corner of this patch sits at
                        # 412 mm. The UR5e patch reaches 600 mm and overshoots by
                        # 177 mm, which is why a UR3e cell cannot inherit it.
                        workspace_center_mm=(300.0, 0.0),
                        workspace_half_extents_mm=(100.0, 100.0)),
    "ur5e": URModelSpec("ur5e", "UR5e", "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
                        baked_gripper_variant="Robotiq_2f_85", max_reach_mm=850.0, max_payload_kg=5.0),
    "ur10e": URModelSpec("ur10e", "UR10e", "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd",
                         baked_gripper_variant="Robotiq_2f_85", max_reach_mm=1300.0, max_payload_kg=12.5),
}


def ur_model_spec(model: str) -> URModelSpec:
    """The :class:`URModelSpec` for ``model``, case-insensitively. An unknown key raises ``ValueError``."""
    spec = _UR_MODELS.get(model.lower())
    if spec is None:
        raise ValueError(
            f"unsupported sim robot_model {model!r}; known: {sorted(_UR_MODELS)}. Add a URModelSpec "
            "(Lula key + Isaac USD relpath) and build the matching cuRobo {key}.yml to support it."
        )
    return spec


def curobo_robot_yml(model: str) -> str:
    """The cuRobo robot-config filename for ``model``, built on-box by ``scripts/curobo/build_{key}_config.py``."""
    return f"{ur_model_spec(model).key}.yml"
