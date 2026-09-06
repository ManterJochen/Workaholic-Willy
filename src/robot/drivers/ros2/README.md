# ROS 2 driver: a reserved empty slot

A namespace placeholder for a future ROS 2 and MoveIt bridge. No driver is implemented yet.

One day it would expose a ROS-controlled manipulator behind the vendor-neutral `RobotArm` Protocol,
registered under `RobotVendor.ROS2`. Today the package is an `__init__.py` with an empty `__all__`,
and this README.

Its single job is to hold the namespace and keep the import-discipline promise: when a driver is
built, every `rclpy` and MoveIt import stays lazily contained inside this subpackage. Vendor SDKs
never import at `core` or `registry` top level.

## Selecting it fails fast

With no registered factory, `create_arm("ros2", ...)` raises `RobotConnectionError` from
[`../registry.py`](../registry.py), reporting that no driver is registered for vendor `'ros2'` and
listing the vendors that are. The composition root passes whatever vendor the config names straight
to `create_arm`, so a config that selects `ros2` fails there with that message rather than silently.

`RobotVendor.ROS2` is a declared enum member and no factory is registered for it. Those are
different things: enum membership exists, factory registration does not.

## The unsolved design question

Before this driver can feed the grasping pipeline, one thing has to be decided rather than coded:
per-message frame conversion between the ROS conventions of REP 103 and this stack's convention of
millimetres, XYZW quaternions and frame-tagged poses. Every other driver solves that once, at its
own boundary. A ROS bridge has to solve it per message type, and getting it wrong is the class of
error that produces plausible motion in the wrong place.

The intended future surface is a factory registered with `@register_arm_driver(RobotVendor.ROS2)`,
so that `create_arm(RobotVendor.ROS2, config=...)` returns a `RobotArm`, depending on the
`robot.core` Protocols plus `rclpy` and MoveIt. None of it is built. There is no node lifecycle, no
quality-of-service policy, no message conversion and no `python -m` entry point, and no tests target
this package because there is no code to test.

## See also

- [drivers](../README.md), the driver layer, the registry and the vendor-selection rules
- [ur](../ur/README.md), the working reference arm driver
- [kuka](../kuka/README.md), a real driver that has never run on a controller
- [franka](../franka/README.md), the sibling empty slot
