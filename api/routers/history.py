"""What happened: the runs this session drove, and the records that outlive it.

Every response says which of the two it came from. They are not interchangeable: runs are rich and
die with the process; records survive restarts and include what the CLI runner wrote, but the production
serializer fills six of the twelve blocks on a default attempt, so the target and the candidate set are
not stored. The executed pose does survive, in ``execution.executed_grasp`` and, since 2026-09-10, in
``selected_grasp``; the refinement trail survives only on an attempt that ran the refiner, which no
shipped config turns on. A view that mixed them silently would let an operator conclude that something
is stored which is not.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from api.cell import Console, console
from api.history import load_records, records_csv, rollup, runs_csv
from api.schemas import RecordOut, RollupOut, RunOut

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/kpis", response_model=RollupOut, summary="KPIs over the logged records")
def get_kpis(cell: Annotated[Console, Depends(console)]) -> RollupOut:
    """Rolled up with the same function the offline gate uses.

    A console that computed its own success rate would eventually disagree with
    ``python -m src.robot.grasping.replay --records``, and then nobody could say which number
    was real.
    """
    records = load_records(cell.record_log_path)
    summary = rollup(records, source=str(cell.record_log_path))
    return RollupOut(
        total_attempts=summary.total_attempts,
        kpis=summary.kpis,
        unmeasurable=summary.unmeasurable,
        outcomes=summary.outcomes,
        source=summary.source,
        record_log_path=str(cell.record_log_path),
        record_log_exists=cell.record_log_path.exists(),
    )


@router.get("/records", response_model=list[RecordOut], summary="Logged grasp attempts, newest first")
def get_records(
    cell: Annotated[Console, Depends(console)], limit: int = 200
) -> list[RecordOut]:
    records = load_records(cell.record_log_path)
    return [
        RecordOut(
            timestamp=record.timestamp,
            attempt_id=record.attempt_id,
            mode=str(record.mode),
            final_outcome=str(record.final_outcome),
            # The whole free-form bag, verbatim. It carries the live `safety_rejected` signal and the
            # robot provenance stamp, and re-shaping it here would be a second contract to keep in
            # step with the frozen one.
            extra=dict(record.extra or {}),
        )
        for record in reversed(records[-limit:])
    ]


@router.get("/runs.csv", response_class=PlainTextResponse, summary="Runs as CSV")
def get_runs_csv(cell: Annotated[Console, Depends(console)], limit: int = 500) -> PlainTextResponse:
    """One row per run: the report-level view, for someone writing up a bring-up day."""
    body = runs_csv(cell.registry.recent(limit))
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="willy-runs.csv"'},
    )


@router.get("/records.csv", response_class=PlainTextResponse, summary="Grasp attempts as CSV")
def get_records_csv(cell: Annotated[Console, Depends(console)]) -> PlainTextResponse:
    """One row per attempt: the analysis view.

    Only the blocks the serializer actually populates get columns. An empty ``selected_grasp`` column on
    every row would suggest the data exists and happened to be missing, rather than that it is never
    written on the live path.
    """
    body = records_csv(load_records(cell.record_log_path))
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="willy-attempts.csv"'},
    )


@router.get("/runs", response_model=list[RunOut], summary="This session's runs (in memory)")
def get_session_runs(
    cell: Annotated[Console, Depends(console)], limit: int = 50
) -> list[RunOut]:
    """The rich view, and the perishable one: these live in memory and go when the server does."""
    return [RunOut(**run.to_dict()) for run in cell.registry.recent(limit)]
