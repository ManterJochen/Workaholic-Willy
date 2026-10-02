/**
 * The cell's own stream (`WS /v1/events?run_id=cell`), reduced: what belongs to no run.
 *
 * The jaws question, a brake pressed with no run, the recovery record written, cleared and ended, the planner's
 * state. `GET /v1/cell` remains the authority on every one of them (`jaws_question`, `halted`, `recovery`), and the
 * gates are the server's; this stream is what makes the console react in the same instant instead of at the next
 * poll, and what the conversation says about the cell.
 *
 * A question is never answered here and never defaulted: this model only says that one waits, with exactly the
 * choices the server offered, and forgets it when it is answered, ends, or expires.
 */

import type { JawsChoice, JawsStage, RunKind, StopCode } from '../api/codes'
import { CELL_STREAM, isStopCode } from '../api/codes'
import type { RunEvent } from '../api/events'
import { jawsChoiceMsg, stopMsg, whereMsg } from '../i18n/codes'
import type { MessageKey, Msg, Params } from '../i18n/types'
import type { ChatKey, ChatLine, Level, Tone } from './runModel'

export interface JawsQuestionView {
  readonly id: string
  readonly stage: JawsStage
  readonly at: 'connect' | 'check'
  /** The bank and pin, as a person reads it on the pendant: `tool output 0`. */
  readonly where: string
  readonly reason: string
  /** Exactly what the server offers. There is no default: a click names one of these, or nothing is sent. */
  readonly choices: readonly JawsChoice[]
  readonly attempt: number
  readonly of: number
  readonly whyAgain: string
  /** Unix seconds; unanswered by then, the question is refused, never "open". */
  readonly expiresAt: number | null
  /** The server's own words for the question, for the tech view. */
  readonly text: string
  readonly askedAt: number
}

export interface CellStreamView {
  readonly question: JawsQuestionView | null
  readonly lastAnswer: { readonly id: string; readonly choice: JawsChoice; readonly at: number } | null
  readonly lastEnded: {
    readonly id: string
    readonly outcome: 'open' | 'refused' | 'no_answer' | 'cancelled'
    readonly refusal: string
    readonly detached: boolean
    readonly at: number
  } | null
  /** A brake pressed while no run was active. */
  readonly halted: { readonly reason: string; readonly inMotion: boolean; readonly at: number } | null
  readonly recovery: {
    readonly runId: string
    readonly kind: RunKind | null
    readonly stopCode: StopCode | null
    readonly at: number
    /** When a person first confirmed the cell is clear, after the stop (a later "Backen leer" confirms it again). */
    readonly acknowledgedAt: number | null
  } | null
  readonly recoveryEnded: { readonly runId: string; readonly by: string; readonly endedBy: string | null; readonly at: number } | null
  readonly acknowledged: { readonly cleared: readonly string[]; readonly jawsEmptied: boolean; readonly at: number } | null
  readonly planner: string | null
  readonly lines: readonly ChatLine[]
  readonly lastSeq: number
}

export const EMPTY_CELL_STREAM: CellStreamView = {
  question: null,
  lastAnswer: null,
  lastEnded: null,
  halted: null,
  recovery: null,
  recoveryEnded: null,
  acknowledged: null,
  planner: null,
  lines: [],
  lastSeq: 0,
}

/**
 * Forget everything: the stream is about to be replayed from seq 0 (a reconnect, maybe to a restarted server whose
 * numbering started over). Never sent anywhere; `followCell` asks for it through `onReset`.
 */
export const CELL_RESET = { type: 'cell.reset', reset: true } as const

export type CellAction = RunEvent | typeof CELL_RESET

const CHOICES: readonly JawsChoice[] = ['open', 'closed', 'open_now', 'abort']

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function m(key: MessageKey, params?: Params): Msg {
  return params === undefined ? { key } : { key, params }
}

function line(view: CellStreamView, event: RunEvent, msg: Msg<ChatKey>, tone: Tone, level: Level, id?: string): CellStreamView {
  const said: ChatLine = {
    id: id ?? `cell:${event.seq}`,
    runId: CELL_STREAM,
    seq: event.seq,
    at: event.ts,
    type: event.type,
    msg,
    human: event.human ?? '',
    tone,
    level,
    // The cell's lines belong to no part of any run.
    part: null,
    data: (event.data ?? {}) as Readonly<Record<string, unknown>>,
  }
  return { ...view, lines: [...view.lines, said] }
}

/** Where, in the words a person reads on the pendant. The server says `tool output 0`; that is kept as it is. */
function whereOf(data: Record<string, unknown>): string {
  return str(data.where) || '—'
}

/**
 * The next cell view after one event of the cell stream. Events of runs, and events already applied, change nothing;
 * a reset starts over, for a replay from seq 0.
 */
export function reduceCell(view: CellStreamView, action: CellAction): CellStreamView {
  if ('reset' in action) return EMPTY_CELL_STREAM
  const event = action
  if (event.run_id !== CELL_STREAM) return view
  if (event.type !== 'gap' && event.seq <= view.lastSeq) return view
  const data = (event.data ?? {}) as Record<string, unknown>
  const base: CellStreamView = event.type === 'gap' ? view : { ...view, lastSeq: event.seq }
  switch (event.type) {
    case 'cell.jaws_question': {
      const stage: JawsStage = data.stage === 'open_now' ? 'open_now' : 'where'
      const choices = Array.isArray(data.choices)
        ? (data.choices as unknown[]).filter((c): c is JawsChoice => (CHOICES as readonly unknown[]).includes(c))
        : []
      const question: JawsQuestionView = {
        id: str(data.question_id),
        stage,
        at: data.at === 'check' ? 'check' : 'connect',
        where: whereOf(data),
        reason: str(data.reason),
        choices,
        attempt: num(data.attempt) ?? 1,
        of: num(data.of) ?? 3,
        whyAgain: str(data.why_again),
        expiresAt: num(data.expires_at),
        text: str(data.text),
        askedAt: event.ts,
      }
      const where = whereMsg(question.where)
      const msg = stage === 'open_now'
        ? m('event.cell.jaws_question.open_now', { where })
        : m('event.cell.jaws_question', { where })
      return line({ ...base, question }, event, msg, 'warn', 'demo')
    }
    case 'cell.jaws_answered': {
      const id = str(data.question_id)
      const choice = (CHOICES as readonly unknown[]).includes(data.choice) ? (data.choice as JawsChoice) : null
      const next: CellStreamView = {
        ...base,
        question: base.question && base.question.id === id ? null : base.question,
        lastAnswer: choice ? { id, choice, at: event.ts } : base.lastAnswer,
      }
      return line(next, event, m('event.cell.jaws_answered', { choice: choice ? jawsChoiceMsg(choice) : '—' }), 'info', 'tech')
    }
    case 'cell.jaws_ended': {
      const id = str(data.question_id)
      const raw = str(data.outcome)
      const outcome = raw === 'open' || raw === 'refused' || raw === 'no_answer' || raw === 'cancelled' ? raw : 'refused'
      const next: CellStreamView = {
        ...base,
        // Any question still showing ends with it: an ended exchange leaves nothing to answer.
        question: null,
        lastEnded: { id, outcome, refusal: str(data.refusal), detached: data.detached === true, at: event.ts },
      }
      return line(next, event, m(`event.cell.jaws_ended.${outcome}`), outcome === 'open' ? 'ok' : 'warn', 'demo')
    }
    case 'cell.halted': {
      const next: CellStreamView = { ...base, halted: { reason: str(data.reason), inMotion: data.in_motion === true, at: event.ts } }
      return line(next, event, m('event.cell.halted'), 'error', 'demo')
    }
    case 'cell.recovery': {
      const code = isStopCode(data.stop_code) ? data.stop_code : null
      const next: CellStreamView = {
        ...base,
        recovery: {
          runId: str(data.run_id),
          kind: (str(data.kind) || null) as RunKind | null,
          stopCode: code,
          at: num(data.at) ?? event.ts,
          acknowledgedAt: null,
        },
        recoveryEnded: null,
      }
      // The stop card says this in full wherever a person reads the cockpit; the tech view keeps the stream's line.
      return line(next, event, m('event.cell.recovery', { title: code ? stopMsg(code) : '—' }), 'error', 'tech')
    }
    case 'cell.acknowledged': {
      const cleared = Array.isArray(data.cleared) ? (data.cleared as unknown[]).filter((c): c is string => typeof c === 'string') : []
      const jawsEmptied = data.jaws_emptied === true
      const next: CellStreamView = {
        ...base,
        acknowledged: { cleared, jawsEmptied, at: event.ts },
        recovery: base.recovery ? { ...base.recovery, acknowledgedAt: base.recovery.acknowledgedAt ?? event.ts } : null,
        halted: null,
      }
      // Say what the person confirmed. Every acknowledge says the cell is clear (`cell_clear: true`), and re-stamps a
      // stop record already cleared, so "Backen leer" after "Zelle ist frei" names only the hand: the cell was said
      // clear a moment ago, and an empty hand is the news (a safety fact of its own). A first confirmation that also
      // empties the hand says both.
      const clearsNow =
        cleared.includes('halted') ||
        cleared.includes('needs_person') ||
        (cleared.includes('recovery') && (base.recovery === null || base.recovery.acknowledgedAt === null))
      const said: Msg<ChatKey> = !jawsEmptied
        ? m('event.cell.acknowledged')
        : { key: clearsNow ? 'ck.cell.clearAndEmpty' : 'ck.cell.jawsEmptied' }
      return line(next, event, said, 'info', 'demo')
    }
    case 'cell.recovery_ended': {
      const next: CellStreamView = {
        ...base,
        recovery: null,
        recoveryEnded: { runId: str(data.run_id), by: str(data.by), endedBy: str(data.ended_by) || null, at: event.ts },
      }
      return line(next, event, m('event.cell.recovery_ended'), 'ok', 'demo')
    }
    case 'cell.planner': {
      const state = str(data.state) || null
      return line({ ...base, planner: state }, event, m('event.cell.planner', { state: state ?? '—' }), 'info', 'tech')
    }
    case 'gap':
      // A gap carries the cursor it interrupts, which is the seq of the line before it: it gets an id of its own.
      return line(base, event, m('chat.gap', { dropped: num(data.dropped) ?? 0 }), 'warn', 'tech', `cell:gap:${view.lines.length}`)
    default:
      return line(base, event, m('common.raw', { text: event.human || event.type }), 'info', 'tech')
  }
}

export function replayCell(actions: Iterable<CellAction>, from: CellStreamView = EMPTY_CELL_STREAM): CellStreamView {
  let view = from
  for (const action of actions) view = reduceCell(view, action)
  return view
}

/**
 * The question to show now, or none. A question past its expiry is not offered even if the stream has not said it
 * ended yet: the server refuses it at that moment, and a click on a dead question would only be refused in turn.
 */
export function pendingQuestion(view: CellStreamView, nowS: number): JawsQuestionView | null {
  const q = view.question
  if (!q) return null
  if (q.expiresAt !== null && nowS > q.expiresAt) return null
  return q
}
