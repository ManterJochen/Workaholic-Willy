"""Reading the config with its provenance, and writing the handful of values a person measures.

Two endpoints and one rule. ``GET`` answers what a value is and which layer set it; a layered profile
chain makes that genuinely hard to answer by opening files. ``PATCH`` writes a bench measurement
through the same validators the YAML goes through, as one transaction, and refuses anything else by
name.

The refusal list is not a limitation to be lifted later. Limits and thresholds in this tree are written
next to the comment that carries the evidence for them, and a form that writes the number without
showing the comment is a form that invites changing a value nobody remembers the reason for.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.cell import Console, RunLocked, console
from api.constants import API_LOG_DIR, ROUTER_CONFIG_LOG_FILE
from api.routers.preflight import to_wire_for
from api.schemas import ConfigPatchOut, ConfigValueOut, LayerOut, WritableOut
from src.utility.log_cfg import create_logger

router = APIRouter(prefix="/config", tags=["config"])

#: This file writes YAML. Everything below is therefore an audit trail rather than diagnostics: which
#: keys changed, to what, in which file, and, when the write did not happen, why.
logger = create_logger("ConfigRouter", ROUTER_CONFIG_LOG_FILE, log_dir=API_LOG_DIR)


@router.get("/writable", response_model=list[WritableOut], summary="What may be set from here")
def get_writable() -> list[WritableOut]:
    """The guided-write form, described by the library rather than duplicated in the frontend.

    The ``measure`` text is what the operator needs and what a type annotation cannot give them: not
    "float, kg" but "weigh the whole assembly including the coupling plate and the hoses".
    """
    from src.config.edit import WRITABLE

    return [
        WritableOut(key=w.path, label=w.label, measure=w.measure, unit=w.unit) for w in WRITABLE
    ]


@router.get("/explain", response_model=ConfigValueOut, summary="A value and where it came from")
def get_explain(
    cell: Annotated[Console, Depends(console)],
    key: Annotated[str, Query(description="Dotted config path, e.g. robot.safety.payload.mass_kg")],
) -> ConfigValueOut:
    # One call, so the console and the terminal cannot answer differently: `explain_in` does the read
    # and reports `has_value` itself. A key that is set to None is a real answer here, not a miss.
    from src.config.explain import explain_in

    detail = explain_in(cell.config(), key, cell.root, cell.layers)
    found = detail.has_value
    value = detail.value

    if not detail.known and not found:
        logger.info(
            "explain: %r is not a config key (suggested: %s).",
            key, ", ".join(detail.suggestions) or "nothing close",
        )
        # An unknown key is a client bug, not a server error, and the schema's own near-misses are the
        # most useful thing to hand back: they are how a typo becomes a correction in one round trip.
        raise HTTPException(
            status_code=404,
            detail={
                "code": "unknown_key",
                "message": f"{key} is not a config key.",
                "detail": {"suggestions": list(detail.suggestions)},
            },
        )

    return ConfigValueOut(
        key=key,
        value=None if not found else value,
        type=detail.type_summary or None,
        default=detail.default,
        doc=detail.doc or None,
        doc_scope=detail.doc_scope or None,
        source=detail.set_in or None,
        tier=detail.tier or None,
        layers=[
            LayerOut(location=layer.location, raw=layer.raw, winner=layer.winner)
            for layer in detail.layers
        ],
        # The YAML comment above the winning line. Shown because in this tree it usually carries the
        # justification for the number, and a console that hides it makes every value look arbitrary.
        why=detail.comment or None,
        writable=_writable_key(key),
        text=detail.render(),
    )


def _writable_key(key: str) -> bool:
    from src.config.edit import writable

    return writable(key) is not None


@router.patch("", response_model=ConfigPatchOut, summary="Write measured values (all or none)")
def patch_config(
    cell: Annotated[Console, Depends(console)],
    values: dict[str, Any],
) -> ConfigPatchOut:
    """Write a group of measurements as one transaction.

    A group, not a key, because the schema has cross-field rules that no single write can satisfy: a
    tool frame that declares an owner while its transform is still identity is rejected, so the three
    tool-frame keys only ever validate together.
    """
    from src.config.edit import WriteRefused
    from src.config.tree import ConfigTree

    try:
        cell.require_idle()
    except RunLocked as locked:
        logger.warning(
            "Config write of %s refused: run %s owns the cell.",
            ", ".join(sorted(values)), locked.run_id,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "code": "run_active",
                "message": str(locked),
                "detail": {"run_id": locked.run_id},
            },
        ) from locked

    if not values:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_patch", "message": "no values given.", "detail": {}},
        )

    # `connected=` has to reach `ConfigTree.write`, which has no default for it. `set_keys` declares
    # `connected: bool = False`, the permissive value, and `robot.ur.ip` and `robot.kuka.controller_ip`
    # carry `requires_disconnected=True` because they decide which machine receives every motion. A
    # caller that leaves it out repoints the live cell from the browser with nothing in the result
    # saying the guard did not run: a default that means "no guard" is a guard nobody has to switch off.
    #
    # `root`, `layers` and `profile` do not travel separately either. `ConfigTree` derives the layers
    # from the chain, so a write cannot land in a file that a different chain then validates and leave
    # the tree unloadable while reporting success.
    result = ConfigTree(
        root=cell.root, profile=cell.profile, layers=tuple(cell.layers)
    ).write(values, connected=cell.session.connected)
    if not result.applied:
        logger.warning(
            "Config write refused (%s) on key %s: %s",
            result.refused, result.refused_key, result.message,
        )
        # Every `WriteRefused` member needs a status here, `CELL_CONNECTED` included: a member with no
        # entry turns a correct refusal into a KeyError and a 500, and a guard that becomes a server
        # error the first time it fires is not a guard.
        status = {
            WriteRefused.UNKNOWN_KEY: 404,
            WriteRefused.NOT_WRITABLE: 403,
            WriteRefused.INVALID_VALUE: 422,
            WriteRefused.NO_TARGET: 409,
            WriteRefused.CELL_CONNECTED: 409,
        }[result.refused]  # type: ignore[index]
        raise HTTPException(
            status_code=status,
            detail={
                "code": str(result.refused),
                "message": result.message,
                "detail": {"key": result.refused_key, "keys": list(result.keys)},
            },
        )

    # The values, not just the key names: this is a bench measurement being written into the tree that
    # decides how a robot moves, and "the payload changed at 14:02" is only half an answer. Nothing in
    # the writable set is a credential: it is masses, offsets, serials and controller addresses.
    logger.info(
        "Config written: %s -> %s.",
        ", ".join(f"{k}={v!r}" for k, v in sorted(result.values.items())),
        ", ".join(str(path) for path in result.files),
    )
    return ConfigPatchOut(
        applied=True,
        values=result.values,
        files=[str(path) for path in result.files],
        # Re-rendered because one measurement routinely clears more than one checklist row, and an
        # operator working down that list needs to see it move rather than remember to refresh.
        preflight=to_wire_for(cell.robot(), profile=cell.profile),
    )
