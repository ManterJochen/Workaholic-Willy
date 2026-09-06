"""The structured per-attempt records and the per-stage latency tracking.

:mod:`outcome_logging` serialises a completed pick into the frozen
:class:`GraspAttemptRecord`, which is the telemetry contract the offline analysis tail
consumes, and :mod:`latency_tracker` records the per-stage wall-clock spans that feed
the runtime latency gate.
"""
