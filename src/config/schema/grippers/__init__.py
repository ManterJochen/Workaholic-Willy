"""The gripper registry schema: one validated description per hand.

The files live in ``config/grippers/`` and :mod:`src.config.grippers` loads them.
"""

from .gripper_schema import MODEL_NAME_PATTERN, GripperSpec, ParallelJawSpec

__all__ = ["MODEL_NAME_PATTERN", "GripperSpec", "ParallelJawSpec"]
