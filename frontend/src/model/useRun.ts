/**
 * The run on screen, and how it survives a reload.
 *
 * A run lives on the server and outlives the tab, so the view is never kept in the browser: it is REBUILT, by the
 * same four steps every time (build plan 4.6):
 *
 * 1. the run id: `active_run_id` from the shared cell poll, or the run a click just started (`follow(run)`);
 * 2. its record: `GET /v1/runs/{id}`, drawn at once as a snapshot;
 * 3. its events: `followRun(id, since 0)`, the whole run replayed, then live;
 * 4. into the pure reducer (`runModel.reduce`).
 *
 * After a `gap` the record is fetched again and its counters replace what the replay could not count. When the run
 * ends, its summary goes into the conversation (`chat.archiveRun`) and its stream is closed; the view stays on screen
 * until the next run starts.
 *
 * When the cell poll stops naming the followed run while its view still runs, the record is asked once: the stream
 * may only be late (the poll can see the end first), and then the record's state is drawn and the stream stays open
 * for the run's own last events; or the server restarted and forgot the run (`404`), and then the view says the run
 * is no longer known (`run.lost`) and its stream, which a restarted server would never feed, is closed.
 *
 * The run says when the hand changes (a grasp, a place, a put back), and the cell is read again at once on each: the
 * top bar's hand chip, a toggle's counted jaws, would otherwise lag up to a whole cell poll behind the count.
 *
 * Nothing here starts, stops or restarts a run. `follow` only watches a run something else started.
 */

import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useReducer, useState, type ReactNode } from 'react'

import { ApiError, api, type RunOut } from '../api/client'
import { followRun, type StreamState } from '../api/events'
import { archiveRun } from './chat'
import { EMPTY_RUN, hasEnded, reduce, type RunAction, type RunView } from './runModel'
import { useCell } from './useCell'

export interface RunModel {
  readonly view: RunView
  /** The run stream's socket: `live`, `connecting`, `closed`, `failed`. Not the run's state. */
  readonly streamState: StreamState
  /** Watch a run that a click just started (the 202 answer of task, restart, home, pick, teach, planner). */
  readonly follow: (run: RunOut) => void
  /** A local action for the view (a fetched record). Never sent to the server. */
  readonly dispatch: (action: RunAction) => void
}

const INERT: RunModel = {
  view: EMPTY_RUN,
  streamState: 'closed',
  follow: () => undefined,
  dispatch: () => undefined,
}

const RunContext = createContext<RunModel | null>(null)

/** A run's events after which the hand holds something else: the cell (the hand chip) is read again at once. */
const HAND_CHANGES: ReadonlySet<string> = new Set(['pick_result', 'task.placed', 'task.put_back'])

export function RunProvider({ children }: { children?: ReactNode }) {
  const { cell, refresh } = useCell()
  const [view, dispatch] = useReducer(reduce, EMPTY_RUN)
  const [streamState, setStreamState] = useState<StreamState>('closed')
  /** The run being followed: set by a click's answer, or by the cell's active run. */
  const [target, setTarget] = useState<RunOut | null>(null)

  const follow = useCallback((run: RunOut) => {
    setTarget((current) => (current?.id === run.id ? current : run))
  }, [])

  // Follow whatever the cell says is running: after a reload, or a run started from another tab or window.
  const active = cell?.active_run_id ?? null
  const targetId = target?.id ?? null
  useEffect(() => {
    if (!active || active === targetId) return
    let cancelled = false
    api
      .run(active)
      .then((run) => {
        if (!cancelled) follow(run)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [active, targetId, follow])

  // One stream per followed run: its record first, then every event from seq 0.
  useEffect(() => {
    if (!target) return
    const runId = target.id
    dispatch({ type: 'run.snapshot', run: target })
    let done = false
    const stop = followRun(
      runId,
      {
        onEvent: (event) => {
          if (done) return
          dispatch(event)
          // The cell poll takes every such ask that comes while one read is out as one more read, so a burst of them
          // (a long run replayed after a reload) asks the cell once or twice, not once per event.
          if (HAND_CHANGES.has(event.type)) refresh()
          if (event.type === 'gap') {
            // The counters the replay could not count come from the server's record, never from a guess.
            api
              .run(runId)
              .then((record) => {
                if (!done) dispatch({ type: 'run.snapshot', run: record })
              })
              .catch(() => undefined)
          }
          if (event.type === 'run_finished') {
            archiveRun(event.data as unknown as RunOut)
            // The run is over and its stream has nothing more to say; the view stays until the next run.
            done = true
            stop()
          }
        },
        onState: setStreamState,
      },
      { sinceSeq: 0 },
    )
    return () => {
      done = true
      stop()
    }
    // Keyed on the run id: a newer record of the same run must not reopen its stream.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetId])

  // The cell stopped naming the followed run while its view still runs: ask its record once per change of what the
  // cell names. An answer is drawn as a snapshot (the stream stays open: its run_finished draws the cards); a 404 is a
  // server that no longer knows the run, so the view says so and the stream is closed. Any other failure (no server
  // at all) changes nothing: the cell chip already says "no server", and the next answer of the poll asks again.
  const polled = cell !== null
  const watching = targetId !== null && view.runId === targetId && !hasEnded(view.phase)
  useEffect(() => {
    if (!watching || !polled || targetId === null || active === targetId) return
    let cancelled = false
    api
      .run(targetId)
      .then((record) => {
        if (!cancelled) dispatch({ type: 'run.snapshot', run: record })
      })
      .catch((err: unknown) => {
        if (cancelled || !(err instanceof ApiError) || err.httpStatus !== 404) return
        dispatch({ type: 'run.lost', runId: targetId })
        setTarget((current) => (current?.id === targetId ? null : current))
      })
    return () => {
      cancelled = true
    }
  }, [watching, polled, active, targetId])

  const value = useMemo<RunModel>(() => ({ view, streamState, follow, dispatch }), [view, streamState, follow])
  return createElement(RunContext.Provider, { value }, children)
}

/** The run on screen. Without a provider: the empty view, and a `follow` that does nothing. */
export function useRun(): RunModel {
  return useContext(RunContext) ?? INERT
}
