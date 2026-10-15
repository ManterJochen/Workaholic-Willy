"""The digital-I/O bench: which pin closes the jaws, turned into a measurement, with the arm standing still.

``python -m src.robot.execution.io_bench`` reads, watches, sets, pulses and times a gripper's pins on the arm's
controller, drives the configured jaw_io hand through its driver (``--jaws``) and says where the arm stands
(``--where``); it never moves the arm, and every write is gated behind ``--yes``. It was ``python -m
src.robot.drivers.ur`` until P0g slice 9, which moved it here unchanged but for its name: it builds the cell's gripper
and takes the cell lock, which is composition, and a driver package does not compose. It still refuses a tree whose
arm is not a UR.

:class:`Bench` is the library twin, one verb over an action value, with no default that permits a write; the ``pins``
functions under it have no gate at all, which is why the twin exists.
"""

from .bench import Bench, BenchReading, BenchVerdict, Measure, Pulse, Read, Set, Watch, where_lines

__all__ = ["Bench", "BenchReading", "BenchVerdict", "Measure", "Pulse", "Read", "Set", "Watch", "where_lines"]
