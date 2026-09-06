"""Vendor-specific arm driver packages.

Every driver implements :class:`src.robot.core.RobotArm`. Adding a vendor means
dropping a sibling subpackage here that exposes one class satisfying the Protocol, and
no pipeline, planner or application-layer change.

Registered:

* :mod:`.ur` for Universal Robots, over RTDE.
* :mod:`.kuka` for KUKA, over EthernetKRL, meaning EKI and KRL on TCP with XML.
* :mod:`.dummy`, the pure-Python sim arm for offline development.

Package slots exist for ``franka``, ``ros2`` and ``sim``.

Vendor isolation
----------------
No module outside this package imports a vendor SDK such as ``ur_rtde``, ``franky`` or
``rclpy`` directly. The :mod:`.registry` factories defer an SDK import until
:func:`create_arm` runs, so importing this package on a machine without the UR SDK
works.

Public surface
--------------

* :class:`src.robot.core.RobotVendor` holds the canonical identifiers.
* :func:`register_arm_driver` registers a driver, decorator-style.
* :func:`create_arm` turns a vendor into a :class:`RobotArm`.
* :func:`available_vendors` snapshots the registered vendors.
* :func:`is_vendor_registered` and :func:`unregister_arm_driver` inspect and undo a
  registration.

``drivers_README.md`` is the step-by-step guide to completing a new vendor.
"""

from __future__ import annotations

from src.robot.core import RobotArm, RobotVendor

from .registry import (
    ArmDriverFactory,
    available_vendors,
    create_arm,
    is_vendor_registered,
    register_arm_driver,
    unregister_arm_driver,
)

__all__ = [
    "ArmDriverFactory",
    "RobotVendor",
    "available_vendors",
    "create_arm",
    "is_vendor_registered",
    "register_arm_driver",
    "unregister_arm_driver",
]


# ---------------------------------------------------------------------------
# Built-in driver registration. It is lazy: a factory body imports the SDK.
# ---------------------------------------------------------------------------


@register_arm_driver(RobotVendor.DUMMY)
def _make_dummy(**kwargs) -> RobotArm:
    """Build a :class:`.dummy.DummyRobotArm`.

    Accepted kwargs, all optional::

        dof          : int   = 6
        initial_pose : Pose  = TCP at (400, 0, 300, identity)

    ``model`` is not a constructor kwarg. The capability ``model`` string is derived
    from ``dof`` inside :class:`.dummy.DummyRobotArm`, and passing ``model=`` raises
    ``TypeError``.

    It is pure Python and needs no SDK.
    """
    from .dummy import DummyRobotArm

    # The app-layer factory forwards ``config=...`` whatever the vendor, so an operator
    # switches vendors by editing config alone.
    kwargs.pop("config", None)
    return DummyRobotArm(**kwargs)


@register_arm_driver(RobotVendor.UR)
def _make_ur(**kwargs) -> RobotArm:
    """Build a UR-backed :class:`src.robot.drivers.ur.URRobotArm`.

    Accepted kwargs::

        config : RobotConfig   the full robot config tree, required

    This factory imports the UR driver chain only where UR is selected, and
    :class:`URConnection` defers the ``ur_rtde`` dependency itself until connection
    time.
    """
    from .ur.arm import URRobotArm

    config = kwargs.pop("config", None)
    if config is None:
        raise TypeError("create_arm(RobotVendor.UR) requires a 'config=RobotConfig(...)' kwarg.")
    if kwargs:
        raise TypeError(
            f"create_arm(RobotVendor.UR) got unexpected kwargs: {sorted(kwargs)}"
        )
    return URRobotArm(config)


@register_arm_driver(RobotVendor.KUKA)
def _make_kuka(**kwargs) -> RobotArm:
    """Build a KUKA-backed :class:`src.robot.drivers.kuka.KukaRobotArm`.

    Accepted kwargs::

        config : RobotConfig   the full robot config tree, required

    The KUKA driver speaks EthernetKRL, which is EKI, over TCP with XML to the
    controller-side KRL program shipped under ``config/robot/templates/kuka/``. It
    needs no vendor SDK on this side.
    """
    from .kuka.arm import KukaRobotArm

    config = kwargs.pop("config", None)
    if config is None:
        raise TypeError(
            "create_arm(RobotVendor.KUKA) requires a 'config=RobotConfig(...)' kwarg."
        )
    if kwargs:
        raise TypeError(
            f"create_arm(RobotVendor.KUKA) got unexpected kwargs: {sorted(kwargs)}"
        )
    return KukaRobotArm(config)


@register_arm_driver(RobotVendor.SIM)
def _make_sim(**kwargs) -> RobotArm:
    """Build an Isaac-backed :class:`IsaacRobotArm`.

    Accepted kwargs::

        config : SimRobotConfig   the sim cell config, required

    Construction is import-safe on a host without Isaac, because the SDK is touched only
    when :meth:`IsaacRobotArm.connect` runs, and on such a host ``connect()`` raises
    :class:`src.robot.drivers.sim.IsaacNotAvailableError`. That is why the import is
    deferred: a development machine without Isaac has to keep importing this package.
    """
    from .sim.arm import IsaacRobotArm
    from .sim.config import SimRobotConfig

    config = kwargs.pop("config", None)
    if config is None:
        raise TypeError(
            "create_arm(RobotVendor.SIM) requires a 'config=SimRobotConfig(...)' kwarg."
        )
    if not isinstance(config, SimRobotConfig):
        raise TypeError(
            "create_arm(RobotVendor.SIM) requires config to be a SimRobotConfig; "
            f"got {type(config).__name__}."
        )
    if kwargs:
        raise TypeError(
            f"create_arm(RobotVendor.SIM) got unexpected kwargs: {sorted(kwargs)}"
        )
    return IsaacRobotArm(config)
