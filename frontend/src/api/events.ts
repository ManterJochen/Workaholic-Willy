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
 *
 * One stream belongs to no run: `cell` (`CELL_STREAM`) carries the jaws question, a brake with no run, the recovery
 * record and "the cell is clear". It is read exactly like a run (`followCell`). The payloads the contract adds
 * (build plan 1.4) are typed below by event type, so a reducer reads `event.data.part` rather than guessing; every
 * field stays optional, because a payload is only as complete as the server that sent it.
 */

import type { RunOut, TaskPlanOut } from './client'
import { CELL_STREAM, type EventType, type JawsChoice, type JawsStage, type RunKind, type StopCode } from './codes'

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
  /**
   * Fired before a stream that replays from the start on a reconnect (`fromStartOnReconnect`) replays it: whatever the
   * caller built from the earlier connection is to be thrown away, because the replay rebuilds all of it.
   */
  onReset?: () => void
}

function socketUrl(runId: string, sinceSeq: number): string {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${window.location.host}/v1/events?run_id=${encodeURIComponent(runId)}&since_seq=${sinceSeq}`
}

export interface FollowOptions {
  /**
   * The last seq the caller already holds; the server replays everything after it. 0 (the default) replays the
   * whole run, which is how a reloaded page rebuilds its view.
   */
  sinceSeq?: number
  /**
   * Replay from seq 0 on every reconnect, after `onReset`, instead of resuming after the last seq delivered. For a
   * stream whose numbering can start over under the same key: the cell stream, after the server restarted. Resuming
   * at the old cursor would drop every new event as "already seen".
   */
  fromStartOnReconnect?: boolean
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
export function followRun(runId: string, handlers: StreamHandlers, options: FollowOptions = {}): () => void {
  let cursor = Math.max(0, Math.floor(options.sinceSeq ?? 0))
  let stopped = false
  let socket: WebSocket | null = null
  let retry: ReturnType<typeof setTimeout> | null = null
  let opens = 0
  // Backs off so a backend that is down does not become a reconnect storm, but stays short enough
  // that an operator watching a bring-up does not think the console has given up.
  let backoffMs = 500

  const open = () => {
    if (stopped) return
    if (opens > 0 && options.fromStartOnReconnect) {
      // The whole stream again, from its first event: the caller forgets what it built, then rebuilds it.
      cursor = 0
      handlers.onReset?.()
    }
    opens += 1
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

/**
 * Follow the cell's own stream: the events that belong to no run. From the beginning, so a page opened while a jaws
 * question waits replays the question (and whatever answered it) and ends up in the right state; and from the
 * beginning again after every reconnect (`onReset` first), because a restarted server numbers this stream from 1
 * again under the same key, which a resume after the old cursor would silently skip. The stream is small (bounded
 * by the server's ring), so a full replay costs nothing worth saving.
 */
export function followCell(handlers: StreamHandlers, options: FollowOptions = {}): () => void {
  return followRun(CELL_STREAM, handlers, { fromStartOnReconnect: true, ...options })
}

// ── the payloads the contract adds (build plan 1.4), by event type ───────────────────────────────────────────

/** Where a camera place's target was found, and how big it is. */
export interface TargetData {
  label?: string
  score?: number
  centre_mm?: number[]
  rim_mm?: number
  footprint_mm?: number[]
  opening_mm?: number[] | null
  look?: string | null
  seen_at?: number
  overlay?: string | null
}

/** One pick's result: the fields the console has always had, and those the contract adds, each only where said. */
export interface PickResultData {
  outcome?: string
  succeeded?: boolean
  reason?: string
  looks?: string[]
  looks_fused?: string[]
  jaw_faces_seen?: boolean[]
  generated_view_deg?: number
  hand_eye_gap_mm?: number
  refused_look?: string
  part?: number | null
  pick?: number
  hold_measured?: boolean
  controller_stopped?: boolean
  gripper_fault?: string
  needs_person?: boolean | string
  grasp_pose?: { frame?: string; position_mm?: number[]; quaternion_xyzw?: number[] } | null
  object_centre_mm?: number[] | null
  both_faces?: boolean
  pushes?: number
  push_stopped?: boolean
  fused_views?: string[]
  found_nothing?: boolean
  only_excluded?: boolean
  detector_failed?: boolean
  hand_eye_warn_mm?: number
  views_file?: string | null
  overlay?: string | null
}

/** The fields every `pick.<stage>` event may carry (`api/runs.py` `_payload`), plus the stage extras. */
export interface PickStageData {
  attempt?: number
  attempt_total?: number
  segmentation_count?: number
  candidate_count?: number
  target_index?: number
  score?: number
  position_mm?: number[]
  reasons?: string[]
  action?: string
  outcome?: string
  route?: string
  motion_status?: string
  motion_message?: string
  look?: string
  push?: string
  push_reason?: string
  push_mm?: number
  overlay?: string | null
}

/** A jaws question as the cell stream publishes it. It has no default: unanswered, it is refused. */
export interface JawsQuestionData {
  question_id?: string
  stage?: JawsStage
  at?: 'connect' | 'check'
  where?: string
  reason?: string
  choices?: JawsChoice[]
  attempt?: number
  of?: number
  why_again?: string
  expires_at?: number
  text?: string
}

/** Each event type's `data`. Exhaustive over `EventType`: a type the server adds fails `tsc` until it is named here. */
export interface EventDataMap {
  run_started: {
    kind?: RunKind
    plan?: TaskPlanOut | null
    restart_of?: string | null
    to?: string
    name?: string
    label?: string
    role?: string
    prompt?: string
    requested_picks?: number
    push_mm?: number
  }
  run_countdown: { seconds_left?: number; because?: 'teach' | 'jaws_opened' }
  run_stop_requested: { scope?: 'after_part' | 'between_attempts' | 'before_motion' | 'disconnect'; reason?: string }
  run_halt_requested: { reason?: string; in_motion?: boolean; requested_at?: number; braking?: boolean }
  run_error: { error?: string; stop_code?: StopCode }
  run_finished: Partial<RunOut>
  'pick.pick_started': PickStageData
  'pick.attempt_started': PickStageData
  'pick.perceived': PickStageData
  'pick.ranked': PickStageData
  'pick.no_candidate': PickStageData
  'pick.executing': PickStageData
  'pick.attempt_finished': PickStageData
  'pick.pick_finished': PickStageData
  'pick.cancelled': PickStageData
  pick_result: PickResultData
  'task.pose_screened': { pose?: string; role?: 'place' | 'return'; verdict?: string; detail?: string; nearby_deg?: number[] | null }
  'task.survey_started': { phrase?: string; looks?: string[] }
  'task.target_found': { target?: TargetData; look?: string; parts_seen?: number }
  'task.target_missing': { phrase?: string; looks_tried?: string[] | number }
  'task.part_started': { part?: number; of?: number | null }
  'task.nothing_found': { part?: number; empty_in_a_row?: number; only_excluded?: boolean }
  'task.carry_started': { to_look?: string }
  'task.target_checked': { moved_mm?: number; followed?: boolean; target?: TargetData }
  'task.target_lost': { look?: string; why?: 'not_seen' | 'moved_too_far' | 'footprint_changed' }
  'task.drop_planned': { kind?: 'pose' | 'camera'; pose_mm?: number[]; rim_mm?: number | null; hang_mm?: number; air_mm?: number | null; verdict?: string }
  'task.place_started': { place?: string }
  'task.placed': { outcome?: string; no_sensor?: boolean; line_out_refused?: boolean }
  'task.place_failed': { outcome?: string; message?: string }
  'task.put_back': { outcome?: string }
  'task.return_started': { to?: string; note?: string }
  'task.returned': { to?: string; note?: string }
  'task.return_failed': { status?: string; message?: string }
  'task.part_finished': { part?: number; placed?: boolean; duration_s?: number }
  'home.started': { to?: string; note?: string }
  'home.arrived': { to?: string; note?: string }
  'home.refused': { status?: string; message?: string }
  'teach.free': Record<string, never>
  'teach.say': { line?: string }
  'teach.outside': { sentence?: string }
  'teach.inside': Record<string, never>
  'teach.time_warning': { seconds_left?: number }
  'teach.holding_when_still': { because?: 'heartbeat' | 'time_limit' }
  'teach.holding': Record<string, never>
  'teach.screening': Record<string, never>
  'teach.saved': { name?: string; label?: string; verdict?: string }
  'teach.refused': { verdict?: string; detail?: string; nearby_deg?: number[] | null }
  'teach.not_saved': { message?: string }
  'planner.starting': Record<string, never>
  'planner.ready': { loaded?: boolean }
  'planner.failed': { refusal?: string }
  'wave.started': { swings?: number; swing_deg?: number }
  'wave.done': { swings?: number }
  'wave.refused': { status?: string; message?: string; moved?: boolean }
  'cell.jaws_question': JawsQuestionData
  'cell.jaws_answered': { question_id?: string; choice?: JawsChoice }
  'cell.jaws_ended': { question_id?: string; outcome?: 'open' | 'refused' | 'no_answer' | 'cancelled'; refusal?: string; detached?: boolean }
  'cell.halted': { reason?: string; in_motion?: boolean }
  'cell.recovery': { run_id?: string; kind?: RunKind; stop_code?: StopCode; at?: number }
  'cell.recovery_ended': { run_id?: string; by?: 'restart' | 'home'; ended_by?: string }
  'cell.acknowledged': { cleared?: string[]; jaws_emptied?: boolean }
  'cell.planner': { state?: string }
}

/** `EventDataMap` names every event type and no other: a compile-time check, nothing at runtime. */
type Exact<A, B> = [Exclude<A, B>, Exclude<B, A>] extends [never, never] ? true : false
const EVERY_EVENT_TYPE_IS_TYPED: Exact<keyof EventDataMap, EventType> = true
void EVERY_EVENT_TYPE_IS_TYPED

/** An event whose `data` is typed by its `type`. */
export type TypedEvent<T extends EventType> = Omit<RunEvent, 'type' | 'data'> & { type: T; data: EventDataMap[T] }

/** Narrow an event to its typed payload. The payload's fields stay optional: read them defensively. */
export function isEventOf<T extends EventType>(event: RunEvent, type: T): event is RunEvent & TypedEvent<T> {
  return event.type === type
}
