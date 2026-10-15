"""The Universal Robots model registry for the Isaac sim driver.

This stack targets more than one UR, a UR5e cell and a UR3e for hardware validation, so
the model is a config choice: :attr:`SimRobotConfig.robot_model` selects a key here and
the driver and the scene builder derive the model-specific parts from it.

The canonical model key, "ur5e" or "ur3e", is the one string the whole stack shares:

  * the arm registry, ``config/arms/<model>.yaml``, which holds each arm's joints and its sim
    block;
  * the cuRobo robot config filename, ``willy_{key}.yml``, built on-box by
    ``scripts/arms/descriptor.py``;
  * the model the Isaac arm reports as ``capabilities.model``, which the self-collision guard judges it as.

Each spec is built from its arm's description (P0d D3d; the hand-kept ``_UR_MODELS`` went in D4) and adds the
Isaac-specific parts that key cannot express: the Lula supported-config name, which is capitalised as "UR3e", and the
USD relative path in the Isaac asset pack, which is the combined arm and gripper asset.

It is import-safe, being pure data with no ``isaacsim`` import, so it loads anywhere the
rest of the config does.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

__all__ = [
    "URModelSpec",
    "simulable_models",
    "ur_model_spec",
]


@dataclass(frozen=True, slots=True)
class URModelSpec:
    """The model-specific parts the sim driver and the scene builder need.

    ``key`` is the canonical lower-case model key, the same string as the arm's
    ``capabilities.model`` and the ``{key}.yml`` cuRobo config. ``lula_key`` is the Isaac
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
    #: Required, with no default. An entry that omitted them would claim an 850 mm sphere and
    #: a 5 kg payload in silence, which is generous for a UR3 and mean for a UR10. The loader
    #: cannot know an arm's envelope, so it demands it.
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
    #: the asset offers no Robotiq at all. Measured across all six assets by opening each
    #: stage and listing its variant sets, because neither the ur5e nor the ur10e asset bakes
    #: a 2F-85 in:
    #:
    #:     ur3, ur3e, ur5   no Gripper variant set at all
    #:     ur5e             Gripper: [None, Robotiq_2f_85]                 ships selected None
    #:     ur10             Gripper: [Long_Suction, None, Short_Suction]   no Robotiq at all
    #:     ur10e            Gripper: [None, Robotiq_2f_140, Robotiq_2f_85] ships selected None
    #:
    #: So the variant exists and ships de-selected, and this field is what selects it. Four
    #: of the six cannot offer a Robotiq, and on those the sim mounts the hand standalone
    #: (``sim_mount_for``), or the cell comes up as a bare 6-DoF arm and fails much later
    #: with a driven joint 'finger_joint' missing from the gripper dof_names.
    baked_gripper_variant: str | None = None
    #: The registry name of the hand ``baked_gripper_variant`` selects, or ``None`` where
    #: nothing is baked. The sim mount is derived from it
    #: (``willy_sim.grippers.sim_mount_for``): a cell whose hand is the one this asset bakes
    #: selects the variant, and any other hand is mounted standalone.
    baked_hand: str | None = None
    #: The link the eye-in-hand camera and the tool hang from, relative to the robot root
    #: prim. Every UR e-series shares ``wrist_3_link``. It is a field rather than a
    #: constant because the wrist link is the first thing a non-UR arm changes, and a
    #: wrong guess mounts the camera on the wrong body without erroring: the image simply
    #: moves with something else.
    wrist_link_name: str = "wrist_3_link"


@functools.lru_cache(maxsize=None)
def _described_spec(key: str) -> URModelSpec | None:
    """The spec an arm's description gives a simulator, or ``None`` where it gives none.

    ``None`` for an arm the registry does not describe, for a non-UR description, and for a description with no ``sim``
    block: ur16e and ur20 are described and have no Isaac asset, which is the answer the hand-kept table gave.
    """
    from src.config import ConfigError  # noqa: PLC0415 (the config layer is below the drivers; lazy for import cost)
    from src.config.arms import load_arm  # noqa: PLC0415

    try:
        description = load_arm(key)
    except ConfigError:
        return None
    sim = description.sim
    if description.vendor != "ur" or sim is None or description.reach_mm is None or description.payload_kg is None:
        return None
    return URModelSpec(
        key=description.model, lula_key=sim.lula_key, usd_relpath=sim.usd_relpath,
        max_reach_mm=description.reach_mm, max_payload_kg=description.payload_kg,
        workspace_center_mm=sim.workspace_center_mm, workspace_half_extents_mm=sim.workspace_half_extents_mm,
        baked_gripper_variant=sim.baked_gripper_variant, baked_hand=sim.baked_hand, wrist_link_name=sim.wrist_link,
    )


def simulable_models() -> list[str]:
    """Every arm a simulator here can stand up, sorted: the described arms that carry a ``sim`` block."""
    from src.config.arms import available_arms  # noqa: PLC0415

    return [model for model in available_arms() if _described_spec(model) is not None]


def ur_model_spec(model: str) -> URModelSpec:
    """The :class:`URModelSpec` for ``model``, case-insensitively; ``ValueError`` for an arm no simulator here can build.

    Built from the arm registry since P0d D3d, from the description's ``sim`` block and its reach and payload. Every
    reader of the spec (the sim driver, the scene builder, the gripper mount, the reach helper, the ETH calibration, the
    bootstrap banner and datagen's robot resolver) goes through this function, so none of them moved: only where the
    numbers come from did.
    """
    spec = _described_spec(model.lower())
    if spec is None:
        raise ValueError(
            f"unsupported sim robot_model {model!r}; a simulator here can stand up: {simulable_models()}. Give the arm "
            f"a sim block in config/arms/{model.lower()}.yaml (Lula key, Isaac USD relpath, wrist link, workspace) and "
            f"build its cuRobo descriptor to support it."
        )
    return spec
