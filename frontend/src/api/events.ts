/**
 * The live run stream.
 *
 * The backend's contract, which this file exists to honour rather than reinvent:
 *
 * * every event carries a `seq`, monotonic within a run, starting at 1;
 * * reconnecting with `since_seq=<last rendered>` replays everything after it;
 * * if the client slept longer than the ring buffer, the server sends a `gap` frame FIRST, saying how
 *   many events are gone forever.
 *
 * That last one is the reason this is a hand-written socket and not a two-line `new WebSocket`. A UI
 * that silently renders a short replay draws a run that never had those steps -- an operator reading
 * it would conclude the cell skipped a stage it actually performed. So `gap` is surfaced as an event
 * in the list, visibly, and never swallowed.
 *
 * `keepalive` frames are the opposite: they carry no run information and exist only to prove the
 * socket is alive. They are consumed here and turned into a timestamp the UI can show as "last heard
 * from the cell", never into a row.
 */

export type Severity = 'info' | 'warn' | 'error' | 'success'

/** One thing that happened, addressed to both a person and a program. */
export interface RunEvent {
  type: string
  run_id: string
  seq: number
  ts: number
  severity: Severity
  /** A plain sentence, written by the backend. The console never composes its own. */
  human: string
  step: string
  step_index: number | null
  step_total: number | null
  /** The machine payload. Never a stringified version of `human`. */
  data: Record<string, unknown>
}

/**
 * A hole in the record, rendered as loudly as anything else.
 *
 * Synthesised locally from the server's `gap` frame so the list has one element type. `dropped` is the
 * server's count, not a guess.
 */
export interface GapEvent extends RunEvent {
  type: 'gap'
  dropped: number
}

export type StreamState = 'connecting' | 'live' | 'closed' | 'failed'

export interface StreamHandlers {
  onEvent: (event: RunEvent) => void
  onState: (state: StreamState, detail?: string) => void
  /** Fired on every frame including keepalives, so a UI can show liveness without inventing rows. */
  onHeartbeat?: (at: number) => void
}

function socketUrl(runId: string, sinceSeq: number): string {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${window.location.host}/v1/events?run_id=${encodeURIComponent(runId)}&since_seq=${sinceSeq}`
}

/**
 * Follow one run until it is closed or the caller stops it.
 *
 * Reconnects automatically, resuming from the last `seq` actually delivered to `onEvent` -- not from
 * the last one received, and not from zero. Resuming from zero would replay the whole run into a list
 * that already holds it; resuming from a seq that was received but not rendered would lose a row.
 *
 * Returns a stop function. Calling it is idempotent and suppresses reconnection.
 */
export function followRun(runId: string, handlers: StreamHandlers): () => void {
  let cursor = 0
  let stopped = false
  let socket: WebSocket | null = null
  let retry: ReturnType<typeof setTimeout> | null = null
  // Backs off so a backend that is down does not become a reconnect storm, but stays short enough
  // that an operator watching a bring-up does not think the console has given up.
  let backoffMs = 500

  const open = () => {
    if (stopped) return
    handlers.onState('connecting')
    let ws: WebSocket
    try {
      ws = new WebSocket(socketUrl(runId, cursor))
    } catch (err) {
      handlers.onState('failed', String(err))
      schedule()
      return
    }
    socket = ws

    ws.onopen = () => {
      backoffMs = 500
      handlers.onState('live')
    }

    ws.onmessage = (message) => {
      handlers.onHeartbeat?.(Date.now())
      let frame: Record<string, unknown>
      try {
        frame = JSON.parse(message.data as string)
      } catch {
        return // an unparseable frame is a backend bug; dropping it beats crashing the console
      }
      const type = String(frame.type ?? '')
      if (type === 'keepalive') return
      if (type === 'gap') {
        // Not a run event -- it has no seq of its own -- so it is given the cursor it interrupts and
        // shown in place. It must never be filtered out as "not a real event".
        handlers.onEvent({
          type: 'gap',
          run_id: runId,
          seq: cursor,
          ts: Date.now() / 1000,
          severity: 'warn',
          human: String(frame.human ?? 'Events were lost before this point.'),
          step: '',
          step_index: null,
          step_total: null,
          data: { dropped: frame.dropped ?? null },
          dropped: Number(frame.dropped ?? 0),
        } as GapEvent)
        return
      }
      const event = frame as unknown as RunEvent
      if (typeof event.seq !== 'number') return
      // Guard against a duplicate replay after a reconnect: the server is asked for `> cursor`, but a
      // race at the boundary is cheaper to drop here than to de-duplicate in every screen.
      if (event.seq <= cursor) return
      cursor = event.seq
      handlers.onEvent(event)
    }

    ws.onerror = () => {
      // Deliberately quiet: `onclose` always follows, and reporting both makes one dropped connection
      // look like two failures in the UI.
    }

    ws.onclose = () => {
      socket = null
      if (stopped) {
        handlers.onState('closed')
        return
      }
      handlers.onState('closed')
      schedule()
    }
  }

  const schedule = () => {
    if (stopped || retry) return
    retry = setTimeout(() => {
      retry = null
      open()
    }, backoffMs)
    backoffMs = Math.min(backoffMs * 2, 5000)
  }

  open()

  return () => {
    stopped = true
    if (retry) clearTimeout(retry)
    retry = null
    socket?.close()
    socket = null
  }
}
