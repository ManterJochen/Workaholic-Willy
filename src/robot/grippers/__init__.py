"""Vendor-specific gripper driver modules.

Every driver implements :class:`src.robot.core.Gripper`. Adding a gripper means
dropping a sibling module here that exposes one class satisfying the Protocol, with no
pipeline changes.

Currently shipping:

* :class:`.robotiq.GripperController`, a Robotiq 2F-85, 2F-140 or Hand-E over the
  URCap socket on port 63352. There is no SDK behind it: ``robotiq_gripper`` is not a
  distribution, ``pip index versions`` finds nothing for it, and it is one example
  file from SDU's repository. :mod:`.robotiq_socket` speaks the grammar directly.
* :class:`.onrobot.OnRobotGripper`, an RG2 or RG6 over Modbus TCP through the Compute
  Box, in :mod:`.onrobot_modbus`. Three things are the opposite of the Robotiq: no
  activation stroke, no speed register, and a width that is an opening, so a bigger
  number is more open. RG2 and RG6 only, because the 2FG7 shares the family name and
  not the register map.
* :class:`.vacuum.VacuumGripper` and :class:`.jaw_io.JawIOGripper`, suction and
  parallel jaws over the controller digital I/O, deliberately manufacturer-neutral.
* :class:`.dummy.DummyGripper`, the pure-Python sim gripper.
* :class:`.null.NullGripper`, the explicit no-op for a robot with no gripper attached.

Schunk is validated in sim only: ``GripperVendor.SCHUNK`` runs in Isaac on the
vendor-neutral ``grippers/sim/gripper.IsaacGripper`` with the measured
``SCHUNK_EGU50_PROFILE``, under ``gripper.vendor: schunk`` and
``robot.sim.gripper_mount: schunk_egu50``. A real-hardware Schunk driver, EGK, EGN or
EGU over the Schunk SDK, waits on the hardware, and ``franka_hand`` over libfranka is
planned.

Vendor isolation
----------------
No module outside this package imports the ``robotiq_gripper`` SDK directly, and the
:mod:`.registry` factories defer an SDK import until :func:`create_gripper` runs.

Public surface
--------------

* :class:`src.robot.core.GripperVendor` holds the canonical identifiers.
* :func:`register_gripper_driver` registers a factory, decorator-style.
* :func:`create_gripper` turns a vendor into a :class:`Gripper`.
* :func:`available_gripper_vendors`, :func:`is_gripper_vendor_registered` and
  :func:`unregister_gripper_driver` inspect and undo a registration.

Backwards compatibility
-----------------------
:class:`GripperController` is re-exported here, so an existing import of that name
keeps working.
"""

from __future__ import annotations

from typing import cast

from src.robot.core import Gripper, GripperVendor

from .onrobot_modbus import ModbusError, OnRobotRG, RGStatus
from .robotiq_socket import RobotiqSocket, RobotiqSocketError
from .registry import (
    GripperDriverFactory,
    available_gripper_vendors,
    create_gripper,
    is_gripper_vendor_registered,
    register_gripper_driver,
    unregister_gripper_driver,
)
from .null import GripperSubstitution, SubstitutionReason
from .robotiq import GripperController  # back-compat re-export

__all__ = [
    "ModbusError",
    "OnRobotRG",
    "RGStatus",
    "RobotiqSocket",
    "RobotiqSocketError",
    "Gripper",
    "GripperController",
    "GripperDriverFactory",
    "GripperSubstitution",
    "GripperVendor",
    "SubstitutionReason",
    "available_gripper_vendors",
    "create_gripper",
    "is_gripper_vendor_registered",
    "register_gripper_driver",
    "unregister_gripper_driver",
]


# ---------------------------------------------------------------------------
# Built-in driver registration. It is lazy: a factory body imports the SDK.
# ---------------------------------------------------------------------------


@register_gripper_driver(GripperVendor.ROBOTIQ)
def _make_robotiq(**kwargs) -> Gripper:
    """Build a :class:`.robotiq.GripperController`.

    Required kwargs::

        config : GripperConfig   physical opening limits
        ip     : str             IP of the UR controller: the gripper is
                                 daisy-chained on the tool I/O

    Optional kwargs::

        port           : int   = 63352
        driver_factory : Callable returning a Robotiq-like driver

    The controller imports what it needs; this factory only builds the wrapper.
    """
    from .robotiq import GripperController
    return cast(Gripper, GripperController(**kwargs))


@register_gripper_driver(GripperVendor.ONROBOT)
def _make_onrobot(**kwargs) -> Gripper:
    """Build an :class:`.onrobot.OnRobotGripper`, an RG2 or RG6 over the Compute Box.

    Required kwargs::

        config : GripperConfig   physical opening limits
        host   : str             the Compute Box IP, not the robot's. A Robotiq URCap
                                 daemon runs on the UR controller; an OnRobot box is a
                                 separate device on the network.

    Optional kwargs::

        port                 : int   = 502
        unit                 : int   = 65   chosen by the mounting, not by the tool
        default_force_n      : float = 20.0 newtons natively, not a 0-255 count
        use_fingertip_offset : bool  = False
        client_factory       : Callable returning an OnRobotRG-like client

    RG2 and RG6 only. The 2FG7 shares the family name and not the register map.
    """
    from .onrobot import OnRobotGripper
    return cast(Gripper, OnRobotGripper(**kwargs))


@register_gripper_driver(GripperVendor.DUMMY)
def _make_dummy(**kwargs) -> Gripper:
    """Build the pure-Python sim :class:`.dummy.DummyGripper`.

    Accepted kwargs, all optional::

        min_width_mm     : float = 0.0
        max_width_mm     : float = 150.0
        initial_width_mm : float = max_width_mm
    """
    from .dummy import DummyGripper
    return DummyGripper(**kwargs)


@register_gripper_driver(GripperVendor.VACUUM)
def _make_vacuum(**kwargs) -> Gripper:
    """Build the :class:`.vacuum.VacuumGripper`, suction over the controller's I/O.

    Required kwargs::

        io : SupportsDigitalIO   the I/O source, on a real cell the arm driver,
                                 because the ejector is wired to the controller.

    The optional kwargs mirror :class:`VacuumGripperConfig`, meaning pins, port,
    timings and the vacuum-on width threshold, plus ``config`` for the width limits.

    There is no SDK to import: a vacuum gripper is a pin and, optionally, a switch.
    """
    from .vacuum import VacuumGripper
    return cast(Gripper, VacuumGripper(**kwargs))


@register_gripper_driver(GripperVendor.JAW_IO)
def _make_jaw_io(**kwargs) -> Gripper:
    """Build the :class:`.jaw_io.JawIOGripper`, parallel jaws over the controller's I/O.

    Required kwargs::

        io : SupportsDigitalIO   the I/O source, on a real cell the arm driver,
                                 because the valve is wired to the controller.

    The optional kwargs mirror :class:`JawIOGripperConfig`, meaning actuation, pins,
    port, timings and the closed-below width threshold, plus ``config`` for the width
    limits.

    There is no SDK to import: a solenoid jaw is one or two pins and, optionally, reed
    switches. Same shape as the vacuum factory above, and vendor-neutral for the same
    reason.
    """
    from .jaw_io import JawIOGripper
    return cast(Gripper, JawIOGripper(**kwargs))


@register_gripper_driver(GripperVendor.NONE)
def _make_null(**kwargs) -> Gripper:
    """Build a :class:`.null.NullGripper`, for a cell with no gripper attached.

    Accepted kwargs, all optional::

        min_width_mm : float = 0.0
        max_width_mm : float = 0.0
        substitution : GripperSubstitution | None = None

    ``substitution`` is how a build path says that a real gripper was asked for and
    could not be made, rather than that this cell has no end-effector: the same object,
    very different facts.
    """
    from .null import NullGripper
    return NullGripper(**kwargs)
