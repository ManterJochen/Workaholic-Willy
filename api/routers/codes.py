"""The console's vocabulary: every typed code it answers with, served whole.

The browser translates codes, not the backend's English sentences, so it needs the list of every code to translate.
``frontend/src/api/codes.ts`` mirrors this answer and the generated types check the mirror at compile time;
``tests/test_api_contract.py`` checks it against ``api/codes.py``.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.codes import catalog
from api.schemas import CodesOut

router = APIRouter(tags=["codes"])


@router.get("/codes", response_model=CodesOut, summary="Every typed code the console answers with")
def get_codes() -> CodesOut:
    """Run kinds, stop codes and their classes, event types, refusal codes, readiness lights and blockers, the jaws
    question's stages and choices, and the command reader's notes. Moves nothing, reads no cell."""
    return CodesOut.model_validate(catalog())
