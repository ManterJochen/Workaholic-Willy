"""The cell checklist, in a browser.

The whole endpoint is one call to ``run_config_preflight`` plus a serialisation. That is the point: the
browser and ``python -m src.robot.execution.real_cell --check`` must give the same verdicts, row for
row, and the only way to guarantee that is for there to be one implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Depends

from api.cell import Console, console
from api.schemas import PreflightCheckOut, PreflightOut, Status

if TYPE_CHECKING:  # pragma: no cover
    from src.config.schema.robot.robot_schema import RobotConfig
    from src.robot.execution.real_cell.preflight import PreflightReport

router = APIRouter(tags=["preflight"])


def to_wire(report: "PreflightReport", *, profile: str | None, vendor: str) -> PreflightOut:
    """A :class:`PreflightReport` as wire types. Shared with the config router, which re-renders it."""
    # The counts come from the report, not from a second derivation over the wire objects: `n_warn`
    # and `n_bench` are numbers `PreflightReport` states about itself, and its `to_dict` keeps them
    # reachable without this module, so a library caller wanting a preflight as data never has to
    # import fastapi.
    #
    # `profile` and `vendor` still come from the caller, because neither is on the report and
    # neither is derivable from it: the chain is a property of how the config was loaded, the vendor
    # of the config itself. A verdict without its chain is unreadable, which is why they are here.
    data = report.to_dict()
    return PreflightOut(
        checks=[
            PreflightCheckOut(
                name=c["name"], status=cast(Status, c["status"]), detail=c["detail"], fix=c["fix"]
            )
            for c in cast("list[dict[str, str]]", data["checks"])
        ],
        ok=cast(bool, data["ok"]),
        n_blocking=cast(int, data["n_blocking"]),
        n_warn=cast(int, data["n_warn"]),
        n_bench=cast(int, data["n_bench"]),
        profile=profile,
        vendor=vendor,
    )


def to_wire_for(robot: "RobotConfig", *, profile: str | None) -> PreflightOut:
    """Run the checklist and serialise it. The one path both routers use.

    Named ``to_wire``, not ``render``. Both functions here mean serialise-to-wire and return a
    ``PreflightOut``, while the convention fixes ``render()`` as "describe yourself to a person, as
    text, taking no arguments". `PreflightReport.render()` is that method and it is one attribute
    away, so one verb may not carry both meanings on adjacent lines.
    """
    from src.robot.execution.real_cell.preflight import run_config_preflight

    return to_wire(run_config_preflight(robot), profile=profile, vendor=str(robot.vendor))


@router.get("/preflight", response_model=PreflightOut, summary="Is this cell runnable?")
def get_preflight(cell: Annotated[Console, Depends(console)]) -> PreflightOut:
    return to_wire_for(cell.robot(), profile=cell.profile)
