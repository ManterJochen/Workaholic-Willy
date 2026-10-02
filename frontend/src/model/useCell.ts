/**
 * The cell, shared: ONE poll of `GET /v1/cell` every 2 s for the whole console, the ready bar's answer, the cell's
 * facts, and the cell's own event stream.
 *
 * Before this, the shell and the Pick screen each polled the cell on their own timer; every screen that needs the cell
 * now reads this one model instead (`useCell()`), so the top bar's chips, the ready bar and the stop card can never
 * disagree about which state they saw.
 *
 * * `cell` is the poll, the authority on every gate the console draws (`recovery`, `halted`, `hand`, `jaws_question`).
 *   A failed poll keeps the last answer and sets `error`: "no server" is said, never drawn as an empty cell.
 * * `readiness` is polled on the same tick. A server that does not build it yet (`501 not_built_yet`) is not asked
 *   again until the cell's state changes, so a contract-stage server's log is not flooded with one refusal per tick.
 * * `facts` are read once after a build and once after a connect: they describe what was built. A read that failed is
 *   asked again on the next tick of the poll, until it answers (the halt button's words and its e-stop alarm read
 *   them); a server that has no facts at all (`404`, `501`) is not asked again in the same state.
 * * `telemetry` is read once per connection, for the provenance of the cell chip (simulated arm, URSim, controller).
 * * `stream` is the cell stream, replayed from seq 0 on load (`cellModel`), so the jaws dialog opens the moment a
 *   question is asked. Every burst of cell events also asks the poll for a fresh `CellOut` right away.
 *
 * Nothing here moves anything or answers anything: it only reads.
 */

import {
  createContext,
  createElement,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import { ApiError, api, type CellFactsOut, type CellOut, type ReadinessOut, type TelemetryOut } from '../api/client'
import { followCell, type StreamState } from '../api/events'
import { CELL_RESET, EMPTY_CELL_STREAM, reduceCell, type CellStreamView } from './cellModel'

/** How often the shared cell poll runs. */
export const CELL_POLL_MS = 2000

export interface CellModel {
  readonly cell: CellOut | null
  /** The last poll failed: `cell` is the answer before it. */
  readonly error: ApiError | null
  readonly readiness: ReadinessOut | null
  readonly readinessError: ApiError | null
  readonly facts: CellFactsOut | null
  readonly factsError: ApiError | null
  readonly telemetry: TelemetryOut | null
  readonly stream: CellStreamView
  readonly streamState: StreamState
  /** Wall-clock milliseconds of the last successful poll. */
  readonly loadedAt: number | null
  readonly connected: boolean
  /** Poll now: after an action that changed the cell (build, connect, acknowledge, an answer). */
  readonly refresh: () => void
}

const INERT: CellModel = {
  cell: null,
  error: null,
  readiness: null,
  readinessError: null,
  facts: null,
  factsError: null,
  telemetry: null,
  stream: EMPTY_CELL_STREAM,
  streamState: 'closed',
  loadedAt: null,
  connected: false,
  refresh: () => undefined,
}

const CellContext = createContext<CellModel | null>(null)

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

/** A refusal that means "this server does not have it": asking again on every tick would only fill its log. */
function notThere(error: ApiError): boolean {
  return error.code === 'not_built_yet' || error.httpStatus === 404 || error.httpStatus === 501
}

/** What was read for one state of the cell: facts and provenance belong to the cell that was built or connected. */
interface ReadFor<T> {
  readonly state: string
  readonly data: T | null
  readonly error: ApiError | null
}

export function CellProvider({ children, pollMs = CELL_POLL_MS }: { children?: ReactNode; pollMs?: number }) {
  const [cell, setCell] = useState<CellOut | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [readiness, setReadiness] = useState<ReadinessOut | null>(null)
  const [readinessError, setReadinessError] = useState<ApiError | null>(null)
  const [factsRead, setFactsRead] = useState<ReadFor<CellFactsOut> | null>(null)
  const [telemetryRead, setTelemetryRead] = useState<ReadFor<TelemetryOut> | null>(null)
  const [stream, dispatchCell] = useReducer(reduceCell, EMPTY_CELL_STREAM)
  const [streamState, setStreamState] = useState<StreamState>('connecting')
  const [loadedAt, setLoadedAt] = useState<number | null>(null)
  /** One poll per tick: the timer, `refresh()` and a burst of cell events each move it on. */
  const [tick, setTick] = useState(0)
  /** Another read of the facts, after one that failed: moved on by the poll. */
  const [factsTry, setFactsTry] = useState(0)
  /** The last read of the facts failed, and may be asked again (read only in effects). */
  const factsFailed = useRef(false)
  // Read only in effects and callbacks, never while rendering.
  /** The cell state in which the server said it has no ready bar. */
  const readinessOffIn = useRef<string | null | undefined>(undefined)
  /** A poll is out; a tick that comes meanwhile is remembered, not run beside it. */
  const inFlight = useRef(false)
  const again = useRef(false)
  const alive = useRef(true)

  const refresh = useCallback(() => setTick((n) => n + 1), [])

  useEffect(() => {
    alive.current = true
    const id = setInterval(() => setTick((n) => n + 1), pollMs)
    return () => {
      alive.current = false
      clearInterval(id)
    }
  }, [pollMs])

  // The poll. One at a time: a slow cell PC whose answers take longer than a tick still gets every answer drawn (a
  // newer poll never throws an older answer away), and a tick that came meanwhile runs once the answer is in.
  useEffect(() => {
    if (inFlight.current) {
      again.current = true
      return
    }
    inFlight.current = true
    if (factsFailed.current) {
      factsFailed.current = false
      setFactsTry((n) => n + 1)
    }
    const poll = async () => {
      let state: string | null = null
      try {
        const next = await api.cell()
        if (!alive.current) return
        state = next.state
        setCell(next)
        setError(null)
        setLoadedAt(Date.now())
      } catch (err) {
        if (!alive.current) return
        setError(asApiError(err))
      }
      if (readinessOffIn.current !== undefined && readinessOffIn.current === state) return
      try {
        const answer = await api.readiness()
        if (!alive.current) return
        readinessOffIn.current = undefined
        setReadiness(answer)
        setReadinessError(null)
      } catch (err) {
        if (!alive.current) return
        const refused = asApiError(err)
        setReadinessError(refused)
        setReadiness(null)
        if (notThere(refused)) readinessOffIn.current = state
      }
    }
    void poll().finally(() => {
      inFlight.current = false
      if (again.current && alive.current) {
        again.current = false
        setTick((n) => n + 1)
      }
    })
  }, [tick])

  // The cell stream: replayed from the start (again after every reconnect, the view reset first), then live. A burst
  // of cell events asks for a fresh CellOut once.
  useEffect(() => {
    let soon: ReturnType<typeof setTimeout> | null = null
    const stop = followCell({
      onEvent: (event) => {
        dispatchCell(event)
        if (soon) return
        soon = setTimeout(() => {
          soon = null
          setTick((n) => n + 1)
        }, 150)
      },
      onState: setStreamState,
      onReset: () => dispatchCell(CELL_RESET),
    })
    return () => {
      if (soon) clearTimeout(soon)
      stop()
    }
  }, [])

  // What belongs to one state of the cell: facts after a build and after a connect (again after a read that failed);
  // provenance once per connection.
  const state = cell?.state ?? null
  useEffect(() => {
    factsFailed.current = false
    if (state !== 'built' && state !== 'connected') return
    let cancelled = false
    api
      .facts()
      .then((data) => {
        if (!cancelled) setFactsRead({ state, data, error: null })
      })
      .catch((err: unknown) => {
        if (cancelled) return
        const refused = asApiError(err)
        setFactsRead({ state, data: null, error: refused })
        // Asked again on the next tick, unless this server has no facts to give.
        factsFailed.current = !notThere(refused)
      })
    return () => {
      cancelled = true
    }
  }, [state, factsTry])
  useEffect(() => {
    if (state !== 'connected') return
    let cancelled = false
    api
      .status(false)
      .then((data) => {
        if (!cancelled) setTelemetryRead({ state, data, error: null })
      })
      .catch((err: unknown) => {
        if (!cancelled) setTelemetryRead({ state, data: null, error: asApiError(err) })
      })
    return () => {
      cancelled = true
    }
  }, [state])

  // Derived, never reset from an effect: what was read for another state of the cell is not this cell's.
  const facts = factsRead && factsRead.state === state ? factsRead.data : null
  const factsError = factsRead && factsRead.state === state ? factsRead.error : null
  const telemetry = telemetryRead && telemetryRead.state === state && state === 'connected' ? telemetryRead.data : null

  const value = useMemo<CellModel>(
    () => ({
      cell,
      error,
      readiness,
      readinessError,
      facts,
      factsError,
      telemetry,
      stream,
      streamState,
      loadedAt,
      connected: cell?.state === 'connected',
      refresh,
    }),
    [cell, error, readiness, readinessError, facts, factsError, telemetry, stream, streamState, loadedAt, refresh],
  )
  return createElement(CellContext.Provider, { value }, children)
}

/** The shared cell model. Without a provider: an inert model that knows nothing (a bare component test). */
export function useCell(): CellModel {
  return useContext(CellContext) ?? INERT
}
