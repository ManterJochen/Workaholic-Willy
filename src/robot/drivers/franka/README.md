# Franka slot (`src/robot/drivers/franka`)

A reserved place for a Franka Emika arm driver. There is no driver here: the package is an
`__init__.py` that registers nothing, so `robot.vendor: franka` validates as a config value and is then
refused. `Robot.from_tree` refuses it at the readiness gate, and `create_arm("franka")` raises
`RobotConnectionError` naming the vendors that do have a driver (`dummy`, `kuka`, `sim`, `ur`).

The slot exists so that a Franka SDK (`franky` or `libfranka`), once someone writes the driver, is
imported inside this package and nowhere else. Neither is a dependency of this repository, and no
Franka arm has been driven from it. `GripperVendor.FRANKA_HAND` is the matching reserved name on the
gripper side. The steps for turning this slot into a driver are in [drivers](../README.md), and the
[UR driver](../ur/README.md) is the one to model it on.
