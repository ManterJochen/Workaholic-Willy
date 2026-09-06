"""Structural ``Protocol`` stubs for the Isaac Sim runtime surface.

Every Isaac and Lula handle the sim driver and the ``willy_sim`` runners hold,
``SingleArticulation``, ``ArticulationSubset``, ``LulaKinematicsSolver``, ``RmpFlow``,
``ArticulationMotionPolicy``, ``Camera`` and ``SingleRigidPrim``, comes from a lazy
``isaacsim.*`` import. mypy therefore sees each of them as ``Any`` at the construction
site while the field or local that stores it is annotated ``object`` or
``object | None``, so every method call lands as ``[attr-defined]``.

These Protocols type exactly the member surface this code calls. They import no
``isaacsim`` and only numpy, so the module stays importable on a host without Isaac. A
``Protocol`` is purely structural and is never runtime-checked here: assigning an
``Any``-typed Isaac handle into an ``IsaacArticulation | None`` field changes nothing at
runtime, because this is annotation only.

The method signatures are deliberately permissive, using ``*args`` and ``**kwargs``
where the vendor call shape varies: ``apply_action`` is called both positionally with an
``ArticulationAction`` and with ``joint_positions=``. Return types are pinned where the
caller relies on them, which is the ``np.ndarray`` joint reads and the FK and IK tuples.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

__all__ = [
    "IsaacArmSubset",
    "IsaacArticulation",
    "IsaacCamera",
    "IsaacKinematicsSolver",
    "IsaacMotionPolicy",
    "IsaacRigidPrim",
    "IsaacRmpFlow",
]


class IsaacArticulation(Protocol):
    """``SingleArticulation``, the combined 12-DoF arm and gripper articulation."""

    dof_names: list[str]

    def initialize(self) -> None: ...
    def get_joint_positions(self) -> np.ndarray: ...
    def get_joint_velocities(self) -> np.ndarray: ...
    def apply_action(self, *args: Any, **kwargs: Any) -> None: ...


class IsaacArmSubset(Protocol):
    """``ArticulationSubset`` addressing the 6 named arm joints by name."""

    def get_joint_positions(self) -> np.ndarray: ...
    def get_joint_velocities(self) -> np.ndarray: ...
    def set_joint_positions(self, positions: Any) -> None: ...
    def apply_action(self, *args: Any, **kwargs: Any) -> None: ...


class IsaacKinematicsSolver(Protocol):
    """``LulaKinematicsSolver``, the native FK and IK for the UR5e."""

    def get_all_frame_names(self) -> list[str]: ...
    def compute_forward_kinematics(
        self, frame: str, joints: Any
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def compute_inverse_kinematics(
        self, frame: str, target_pos: Any, target_quat: Any, **kwargs: Any
    ) -> tuple[Any, bool]: ...


class IsaacRmpFlow(Protocol):
    """``RmpFlow``, the reactive Cartesian motion policy."""

    def set_end_effector_target(self, **kwargs: Any) -> None: ...
    def update_world(self) -> None: ...
    def add_obstacle(self, obstacle: Any, static: bool = ...) -> None: ...


class IsaacMotionPolicy(Protocol):
    """``ArticulationMotionPolicy``, which binds an RmpFlow to an articulation."""

    def get_next_articulation_action(self, dt: float) -> Any: ...


class IsaacCamera(Protocol):
    """``isaacsim.sensors.camera.Camera``, the overhead or wrist sensor."""

    def initialize(self) -> None: ...
    def get_intrinsics_matrix(self) -> np.ndarray: ...
    def get_current_frame(self) -> Any: ...
    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    def add_distance_to_image_plane_to_frame(self) -> None: ...
    def set_clipping_range(self, near: float, far: float) -> None: ...


class IsaacRigidPrim(Protocol):
    """``SingleRigidPrim``, a handle on a graspable dynamic object."""

    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    def set_world_pose(self, *args: Any, **kwargs: Any) -> None: ...
    def set_linear_velocity(self, velocity: Any) -> None: ...
    def set_angular_velocity(self, velocity: Any) -> None: ...
    def apply_physics_material(self, material: Any) -> None: ...
