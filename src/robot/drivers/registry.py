"""The driver registry, the single entry point for selecting a vendor-specific
:class:`~src.robot.core.RobotArm` implementation.

Design
------
The registry stores lazy factory callables rather than driver classes, which matters
for two reasons:

* It keeps :mod:`src.robot.core` and the registry itself free of vendor-SDK imports,
  which the architecture requires and CI enforces. A vendor SDK import, such as
  ``rtde_control``, ``franky`` or ``rclpy``, happens inside the factory body the first
  time a caller asks for that vendor.
* It lets a driver fail to load gracefully on a host without its SDK: the
  :func:`create_arm` call raises a clear :class:`RobotConnectionError` and every other
  vendor stays usable.

Public surface
--------------

* :func:`register_arm_driver` registers a driver, decorator-style. :mod:`.ur` and
  :mod:`.dummy` use it here, and a KUKA or Franka driver does the same in its own
  ``__init__``.
* :func:`create_arm` turns a vendor into a :class:`RobotArm`.
* :func:`available_vendors` lists the registered vendor strings.
* :func:`is_vendor_registered` probes one vendor.
* :func:`unregister_arm_driver` undoes a registration.

A worked example::

    from src.robot.core import RobotVendor
    from src.robot.drivers import create_arm, register_arm_driver

    @register_arm_driver(RobotVendor.SIM)
    def _make_sim(**kwargs):
        from .sim.arm import SimArm
        return SimArm(**kwargs)

    arm = create_arm(RobotVendor.SIM, dof=7)
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from src.robot.constants import DRIVER_REGISTRY_LOG_FILE, create_robot_logger
from src.robot.core import RobotArm, RobotConnectionError, RobotVendor

__all__ = [
    "ArmDriverFactory",
    "available_vendors",
    "create_arm",
    "is_vendor_registered",
    "register_arm_driver",
    "unregister_arm_driver",
]


# A driver factory is any callable that returns a fully constructed
# :class:`RobotArm` from vendor-specific kwargs. Its body is the only place that may
# import the vendor SDK.
ArmDriverFactory = Callable[..., RobotArm]


_REGISTRY: dict[RobotVendor, ArmDriverFactory] = {}
_LOCK = threading.RLock()

# Registration happens at driver-package import and instantiation once per cell boot,
# so neither is a hot path. Which vendor this process ended up driving is the first
# thing anybody asks of a log that starts with a robot moving unexpectedly.
logger = create_robot_logger("ArmDriverRegistry", DRIVER_REGISTRY_LOG_FILE)


def _coerce(vendor: RobotVendor | str) -> RobotVendor:
    if isinstance(vendor, RobotVendor):
        return vendor
    return RobotVendor.from_string(vendor)


def register_arm_driver(
    vendor: RobotVendor | str,
    *,
    overwrite: bool = False,
) -> Callable[[ArmDriverFactory], ArmDriverFactory]:
    """Register a factory under ``vendor``.

    Use as a decorator::

        @register_arm_driver(RobotVendor.UR)
        def _make_ur(**kwargs) -> RobotArm:
            from src.robot.drivers.ur.arm import URRobotArm
            return URRobotArm(kwargs["config"])

    Registering a vendor that already has a factory raises :class:`ValueError`.
    ``overwrite=True`` replaces it instead.
    """
    key = _coerce(vendor)

    def _decorator(factory: ArmDriverFactory) -> ArmDriverFactory:
        if not callable(factory):
            raise TypeError(
                f"register_arm_driver: factory must be callable, got {type(factory).__name__}"
            )
        with _LOCK:
            if key in _REGISTRY and not overwrite:
                raise ValueError(
                    f"register_arm_driver: vendor {key.value!r} is already registered; "
                    "pass overwrite=True to replace."
                )
            _REGISTRY[key] = factory
        logger.debug(
            "registered arm driver: vendor=%s factory=%s%s",
            key.value, getattr(factory, "__qualname__", repr(factory)),
            " (overwrite)" if overwrite else "",
        )
        return factory

    return _decorator


def unregister_arm_driver(vendor: RobotVendor | str) -> None:
    """Remove a registered factory. It does nothing if the vendor has none."""
    key = _coerce(vendor)
    with _LOCK:
        _REGISTRY.pop(key, None)


def is_vendor_registered(vendor: RobotVendor | str) -> bool:
    """``True`` only where a factory is registered under ``vendor``."""
    key = _coerce(vendor)
    with _LOCK:
        return key in _REGISTRY


def available_vendors() -> list[str]:
    """Sorted snapshot of the currently registered vendor strings."""
    with _LOCK:
        return sorted(v.value for v in _REGISTRY)


def create_arm(vendor: RobotVendor | str, **kwargs) -> RobotArm:
    """Instantiate the driver registered under ``vendor``.

    Every keyword argument is forwarded to the factory verbatim, and each driver
    documents its own kwargs in its ``__init__.py`` under :mod:`src.robot.drivers`.

    Raises
    ------
    ValueError
        If ``vendor`` is not a known :class:`RobotVendor`.
    RobotConnectionError
        If the vendor is recognised and no factory is registered, meaning the driver
        package is missing or its SDK failed to import.
    JointLimitTableMissing
        If the built arm enforces the joint-limit guard and no per-axis table resolves for
        its own vendor and model. Measured on the shipped ``web`` profile: such a cell used
        to build and then refuse every motion as ``controller_rejected``, naming the
        controller for a missing config key. See
        :func:`src.robot.safety.joint_limits.assert_joint_limit_table_available`.
    """
    key = _coerce(vendor)
    with _LOCK:
        factory = _REGISTRY.get(key)
    if factory is None:
        registered = ", ".join(available_vendors()) or "(none)"
        raise RobotConnectionError(
            f"no driver registered for vendor {key.value!r}; "
            f"available: {registered}. Make sure the driver package "
            f"and its SDK are installed."
        )
    arm = factory(**kwargs)
    logger.info(
        "created arm driver: vendor=%s -> %s (kwargs=%s)",
        key.value, type(arm).__name__, sorted(kwargs),
    )
    # The Protocol is runtime_checkable, so it doubles as a structural guard and a
    # buggy factory cannot quietly return the wrong shape.
    if not isinstance(arm, RobotArm):
        raise TypeError(
            f"factory for vendor {key.value!r} returned "
            f"{type(arm).__name__}, which does not satisfy the RobotArm Protocol."
        )
    # A guard with nothing to enforce refuses every motion and blames the controller.
    # Measured 2026-09-10 on the shipped `web` profile: the joint-limit guard resolved no
    # table for `vendor: kuka`, answered UNAVAILABLE for an all-zeros joint target, and the
    # preflight mapped that to `controller_rejected`. Safe, and unusable, and the operator
    # goes to look at the KRC. It is asked here rather than in each driver because the
    # defect is that the next vendor lands the same way and nobody notices, and this is the
    # one line every cell boot passes through. It is imported inside the function so the
    # registry keeps its import discipline, with no safety at module scope.
    from src.robot.safety.joint_limits import assert_joint_limit_table_available

    assert_joint_limit_table_available(arm)
    return arm
