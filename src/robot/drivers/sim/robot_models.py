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
    #: The vendor-datasheet reach in mm, which is the radius of the working sphere about
    #: the shoulder, and the payload cap in kg. They reject a scene or pose the arm cannot
    #: physically touch, or a mass it cannot lift, at config time rather than after an
    #: Isaac boot and an inscrutable IK failure. Swapping a UR5e cell to a UR3e shrinks
    #: the reach from 850 to 500 mm, which is why the UR5e-tuned scenes, with bins at 515
    #: to 644 mm, are unreachable and have to be re-anchored.
    #:
    #: Required, where they used to default to the UR5e figures. An entry that omitted them
    #: claimed an 850 mm sphere and a 5 kg payload in silence, which is generous for a UR3
    #: and mean for a UR10. The loader cannot know an arm's envelope, so it demands it.
    max_reach_mm: float
    max_payload_kg: float
    #: A workspace this model can reach, in BASE-frame mm as a centre and half-extents,
    #: which is the patch a scene generator places objects in. It is stated per model
    #: rather than derived, which makes it inspectable, and CI asserts that each one fits
    #: inside the reach sphere of its model, so an explicit value cannot quietly become a
    #: wrong one.
    #:
    #: Two of the six are measured and four are derived, and each entry says which. A patch
    #: describes where to place objects on a cell somebody has stood up, and only the UR5e
    #: and UR3e cells exist. The other four are the UR5e patch scaled by the reach ratio,
    #: which keeps the same fraction of the working sphere instead of inventing a number.
    #:
    #: Required for the same reason as the two above. Inheriting the UR5e patch puts a
    #: 618 mm far corner on the 434 mm horizontal radius of a UR3, which the test catches,
    #: and wastes half a UR10, which nothing catches.
    workspace_center_mm: tuple[float, float]
    workspace_half_extents_mm: tuple[float, float]
    #: The ``Gripper`` USD variant to select on the asset of this model, or ``None`` where
    #: the asset offers no Robotiq at all. Measured across all six assets on 2026-09-09 by
    #: opening each stage and listing its variant sets, because the comment that stood here
    #: said the ur5e and ur10e assets bake in a 2F-85, and that is not what they do:
    #:
    #:     ur3, ur3e, ur5   no Gripper variant set at all
    #:     ur5e             Gripper: [None, Robotiq_2f_85]                 ships selected None
    #:     ur10             Gripper: [Long_Suction, None, Short_Suction]   no Robotiq at all
    #:     ur10e            Gripper: [None, Robotiq_2f_140, Robotiq_2f_85] ships selected None
    #:
    #: So the variant exists and ships de-selected, and this field is what selects it. Four
    #: of the six cannot offer a Robotiq, and those require ``SimConfig.gripper_mount``, or
    #: the cell comes up as a bare 6-DoF arm and fails much later with a driven joint
    #: 'finger_joint' missing from the gripper dof_names.
    baked_gripper_variant: str | None = None
    #: The link the eye-in-hand camera and the tool hang from, relative to the robot root
    #: prim. Every UR e-series shares ``wrist_3_link``. It is a field rather than a
    #: constant because the wrist link is the first thing a non-UR arm changes, and a
    #: wrong guess mounts the camera on the wrong body without erroring: the image simply
    #: moves with something else.
    wrist_link_name: str = "wrist_3_link"


# Every UR from the UR3 to the UR10, e-series and CB-series, verified on-box: each asset
# directory exists under ``.../Isaac/Robots/UniversalRobots/``, each key has a DH row in
# ``_ur_kinematics``, a joint-limit row, an entry in ``UR_MODEL_KEYS`` and a buildable
# ``{key}.yml``. ``tests/test_ur_model_family.py`` compares all four key sets, so they cannot
# drift in either direction. ur16e, ur20 and ur30 are deliberately absent: Isaac ships assets
# and this repository has no DH row for them, so admitting them would invent kinematics.
_UR_MODELS: dict[str, URModelSpec] = {
    # UR3, CB-series. Bare asset, with no Gripper variant set at all, so a cell sets gripper_mount.
    "ur3": URModelSpec(
        "ur3", "UR3", "/Isaac/Robots/UniversalRobots/ur3/ur3.usd",
        max_reach_mm=500.0, max_payload_kg=3.0,
        # Derived, not measured: the UR5e patch scaled by 500/850. The far corner sits at
        # 361.4 mm against a 434.2 mm horizontal radius at table height, so 72.8 mm of margin.
        workspace_center_mm=(260.0, 0.0), workspace_half_extents_mm=(90.0, 90.0)),
    # ur3e ships bare, with no Gripper variant set, so a ur3e cell sets gripper_mount.
    "ur3e": URModelSpec(
        "ur3e", "UR3e", "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd",
        max_reach_mm=500.0, max_payload_kg=3.0,
        # Measured on a standing cell, and computed rather than guessed: a guessed 330 +- 110
        # comes out 18.3 mm over. 500 mm of reach less a 40 mm dexterity margin, about a
        # 151.9 mm shoulder, leaves 434 mm of horizontal radius at table height, and the far
        # corner of this patch sits at 412 mm. The UR5e patch reaches 600 mm and overshoots by
        # 177 mm, which is why a UR3e cell cannot inherit it.
        workspace_center_mm=(300.0, 0.0), workspace_half_extents_mm=(100.0, 100.0)),
    # UR5, CB-series. The same a2 as the UR5e and 73 mm less shoulder.
    "ur5": URModelSpec(
        "ur5", "UR5", "/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        max_reach_mm=850.0, max_payload_kg=5.0,
        # Derived: the same patch as the UR5e, because the reach is the same. The far corner
        # sits at 618.5 mm against 805.1 mm of horizontal radius, so 186.6 mm of margin. The CB
        # shoulder is 73 mm lower, which is why the radius differs from the UR5e one.
        workspace_center_mm=(450.0, 0.0), workspace_half_extents_mm=(150.0, 150.0)),
    "ur5e": URModelSpec(
        "ur5e", "UR5e", "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        max_reach_mm=850.0, max_payload_kg=5.0,
        # Measured: the validated pick patch the sim runners use, and the reference every
        # derived patch is scaled from. Far corner 618.5 mm against 793.5 mm of radius.
        workspace_center_mm=(450.0, 0.0), workspace_half_extents_mm=(150.0, 150.0),
        baked_gripper_variant="Robotiq_2f_85"),
    # UR10, CB-series. Its Gripper variant set holds only suction tools and no Robotiq, so a jaw
    # cell sets gripper_mount exactly as the bare models do.
    "ur10": URModelSpec(
        "ur10", "UR10", "/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
        max_reach_mm=1300.0, max_payload_kg=10.0,
        # Derived: the UR5e patch scaled by 1300/850. Far corner 948.3 mm against 1253.6 mm of
        # radius, so 305.2 mm of margin. Conservative on purpose: this uses about three quarters
        # of what the arm can reach, and a cell that wants the rest should measure it.
        workspace_center_mm=(690.0, 0.0), workspace_half_extents_mm=(230.0, 230.0)),
    "ur10e": URModelSpec(
        "ur10e", "UR10e", "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd",
        max_reach_mm=1300.0, max_payload_kg=12.5,
        # Derived, and it used to be worse than derived: this entry declared 1300 mm of reach and
        # then inherited the UR5e patch by default, so it claimed a 618 mm corner on an arm that
        # reaches 1247 mm. Far corner 948.3 mm now, against 1247.0 mm of radius.
        workspace_center_mm=(690.0, 0.0), workspace_half_extents_mm=(230.0, 230.0),
        baked_gripper_variant="Robotiq_2f_85"),
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
