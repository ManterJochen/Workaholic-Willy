"""Prompt routing: decide which perception stack a prompt needs, before any weights load.

    from src.models.routing import route, Route

    decision = route("greif den kaputten Würfel")
    decision.route      # Route.VLM
    decision.reason     # RouteReason.NON_ENGLISH
    decision.describe() # 'vlm (non_english)'

See [README.md](README.md) for the rules and their vocabulary limits.
"""

from __future__ import annotations

from .decision import PromptRouter, PromptSignals, Route, RouteDecision, RouteReason
from .normalize import normalize_simple_prompt
from .rules import (
    MAX_SIMPLE_WORDS,
    MULTI_ATTRIBUTE_THRESHOLD,
    RuleBasedRouter,
    analyse,
    route,
)

__all__ = [
    "Route",
    "RouteReason",
    "PromptSignals",
    "RouteDecision",
    "PromptRouter",
    "RuleBasedRouter",
    "analyse",
    "route",
    "normalize_simple_prompt",
    "MAX_SIMPLE_WORDS",
    "MULTI_ATTRIBUTE_THRESHOLD",
]
