# ROS 2 slot (`src/robot/drivers/ros2`)

A reserved place for a ROS 2 and MoveIt bridge. There is no driver and no ROS node here: the package is
an `__init__.py` that registers nothing, so `robot.vendor: ros2` validates as a config value and is then
refused. `Robot.from_tree` refuses it at the readiness gate, and `create_arm("ros2")` raises
`RobotConnectionError` naming the vendors that do have a driver (`dummy`, `kuka`, `sim`, `ur`).

The slot exists so that `rclpy` and MoveIt, once someone writes the bridge, are imported inside this
package and nowhere else. Before that bridge can feed the pick path, one question has to be answered:
how each ROS message type converts between REP 103 conventions and this library's millimetres, XYZW
quaternions and frame-tagged poses. Every other driver converts once at its own boundary; a bridge has
to do it per message type, and a wrong conversion moves the arm plausibly to the wrong place. The steps
for turning this slot into a driver are in [drivers](../README.md).
