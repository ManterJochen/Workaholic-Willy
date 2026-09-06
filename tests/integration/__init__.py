"""Isaac Sim integration tests (Windows RTX workstation only).

Everything here is gated behind the ``isaac`` pytest marker and auto-skips on any
machine where ``isaacsim`` is not importable (the MacBook / CI), so the default
``pytest tests/`` run stays green everywhere. The real bodies are filled in on the
workstation per the milestone gates in ``docs/ISAAC_VALIDATION_PLATFORM.md``.
"""
