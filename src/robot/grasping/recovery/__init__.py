"""Recover from a failed or blocked pick (push, agitate, skip the failed part, rescan).

:mod:`policy` holds the typed recovery actions, the operator-bounded policy,
and the safety-gated motion executor; :mod:`orchestrator` turns failure
reasons into actions, enforces anti-loop / budget limits, and runs the bounded
recovery loop around a pick; :mod:`trail_serialize` renders the recovery trail to JSONL
telemetry.

Three pure helpers serve the push and the skip, which run inside the pick attempt
and the service rather than in the loop (owner, 2026-09-29): :mod:`push_planner`
plans one push of the failed part from the points the camera saw, or refuses with a
reason; :mod:`push_budgets` counts pushes (1 per part, 2 per pick, 5 per campaign);
:mod:`exclusion_zones` remembers the parts a rescan skips for ``next_target``.
None of them moves the arm or imports robot code.
"""
