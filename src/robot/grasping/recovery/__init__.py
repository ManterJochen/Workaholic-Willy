"""Recover from a failed or blocked pick (nudge, agitate, re-target).

:mod:`policy` holds the typed recovery actions, the operator-bounded policy,
and the safety-gated motion executor; :mod:`orchestrator` turns failure
reasons into actions, enforces anti-loop / budget limits, and runs the bounded
closed-loop retry; :mod:`trail_serialize` renders the recovery trail to JSONL
telemetry.
"""
