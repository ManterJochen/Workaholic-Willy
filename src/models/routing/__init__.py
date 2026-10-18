"""What the command reader reads off a phrase before the detector is asked: its language and its quantifier.

    from src.models.routing import analyse

    analyse("greif den kaputten Würfel").non_english   # True: a phrase the detector gets is English
    analyse("all screws").has_quantifier                # True: all/every belongs in the task's scope

A vocabulary, not a model: no weights, no I/O, the same answer every time. It is what is left of the prompt router
that once sent a phrase to GroundingDINO or to the VLM (removed on 2026-10-17 with GroundingDINO): the VLM grounds
every phrase now, and the command reader (``src/models/vlm/command.py``) still holds the phrases it hands the
detector to English words with no quantifier. See [README.md](README.md) for the vocabularies and their limits.
"""

from __future__ import annotations

from .rules import PromptSignals, analyse

__all__ = ["PromptSignals", "analyse"]
