# Franka driver: a reserved empty slot

A namespace reserved for a future Franka Emika `RobotArm` driver. No code lives here yet.

The package is an `__init__.py` with an empty `__all__`, and this README. It sits alongside the real
`ur`, `kuka`, `dummy` and `sim` drivers and the fellow-empty [`ros2`](../ros2/README.md) slot.

It exists so a contributor can bring up a Franka arm without inventing the namespace, and so that
future vendor-SDK imports (`franky`, `libfranka`) stay confined here, which is what the
vendor-isolation rule requires.

## Confirm the slot is empty at runtime

```python
from src.robot.core import RobotVendor
from src.robot.drivers import is_vendor_registered, available_vendors

is_vendor_registered(RobotVendor.FRANKA)  # False
"franka" in available_vendors()           # False
```

`create_arm(RobotVendor.FRANKA, ...)` raises `RobotConnectionError` from
[`../registry.py`](../registry.py), reporting that no driver is registered for vendor `'franka'` and
listing the vendors that are.

`RobotVendor.FRANKA` and `GripperVendor.FRANKA_HAND` are reserved enum names, and neither is wired
to a driver or gripper factory. The enum member exists; a registered factory is what is missing.
Those are different things.

## What a real driver would have to add

| | |
| --- | --- |
| `arm.py` | the `RobotArm` Protocol implementation |
| transport | with `franky` or `libfranka` imported inside this package only |
| capability descriptor | joint limits, workspace bounds, payload defaults |
| factory registration | `@register_arm_driver(RobotVendor.FRANKA)` in the parent package |
| tests | Protocol conformance, safety wiring, and the typed motion result |

`franky` and `libfranka` are not dependencies of this repository. There is no SDK pin, and no
hardware has ever been validated.

To build a real arm today, pick a registered vendor: `ur`, `kuka`, `dummy` or `sim`.

## See also

- [drivers](../README.md), the drivers layer and the checklist for turning an empty slot into a
  registered driver
- [ur](../ur/README.md), a real implemented vendor driver to model this one on
- [ros2](../ros2/README.md), the sibling reserved slot
