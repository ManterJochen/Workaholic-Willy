"""The arm registry schema: one validated description per arm.

The files live in ``config/arms/`` and :mod:`src.config.arms` loads them.
"""

from .arm_schema import ArmFlangeSpec, ArmJointSpec, ArmSimSpec, ArmSpec, ArmUpstreamSpec

__all__ = ["ArmFlangeSpec", "ArmJointSpec", "ArmSimSpec", "ArmSpec", "ArmUpstreamSpec"]
