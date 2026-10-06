/**
 * One run, as the cockpit draws it: a pure `reduce(view, event)` over the server's event stream.
 *
 * **Replay is the whole design.** A run outlives the browser, and a reloaded page rebuilds its view by replaying the
 * run from seq 0 (`useRun`). So nothing here may depend on WHEN an event arrived, on a clock, or on anything outside
 * the event and the view: the same events in the same order always give the same view, which is what lets the tests
 * feed it the contract's event logs and assert the picture a person would have seen.
 *
 * **Structure from typed fields, words from codes.** The timeline, the cards and the statistics come from the events'
 * machine half (`data`); every chat line is a message KEY with parameters (`i18n`), never a sentence composed here,
 * and every line keeps the backend's own English `human` for the tech view. The model never translates.
 *
 * **What it refuses to invent.** A gap in the stream is drawn as a gap, and the counters are then taken from the
 * server's run record (`run.snapshot`), never extrapolated; an event it already applied is ignored, so a reconnect
 * replay never draws a step twice; a step it never heard of stays idle rather than being assumed done; a push is
 * announced as pushed only when the pick loop says it ran (`push: "pushed"`), never for one that stopped or was
 * refused; and a run the server no longer knows (`run.lost`, a restarted server) stops being drawn as running.
 *
 * The halt has one state no event can set: "unconfirmed". `haltState` derives it from a clock the caller passes, so
 * the reducer stays pure: only where the arm brakes a move in flight, and only after 1.5 s without a confirmation
 * while the arm was moving, does the cockpit raise "press the e-stop" (build plan 4.2). A run that ENDED confirms its
 * halt whatever it ended on: its thread commands nothing more once `run_finished` is out.
 */

import type { CellFactsOut, HaltStateOut, RunOut, TaskPlanOut } from '../api/client'
import { STOP_CLASS_OF, isStopCode, type RunKind, type StopClass, type StopCode } from '../api/codes'
import type { RunEvent } from '../api/events'
import type { CockpitKey } from '../cockpit/i18n'
import { isRefusalCode, outcomeMsg, refusalMsg, runKindMsg, stopMsg } from '../i18n/codes'
import type { MessageKey, Msg, Params } from '../i18n/types'

/**
 * What a chat line may say: a key of the shared core, or of the cockpit's own catalog, the one area that draws the
 * conversation (a line the core has no words for yet, such as "Backen leer" confirmed).
 */
export type ChatKey = MessageKey | CockpitKey

// ── the view ──────────────────────────────────────────────────────────────────────────────────────────────────

export type StepId = 'look' | 'detect' | 'grasp' | 'place' | 'return'

/** The task's steps in the plan's order (OD 2: look -> detect -> grasp -> place -> return). */
export const STEPS: readonly StepId[] = ['look', 'detect', 'grasp', 'place', 'return']

/** `warn` is "seen, nothing usable" (a detect with no candidate), `skipped` a step the pick never reached. */
export type StepState = 'idle' | 'active' | 'done' | 'warn' | 'failed' | 'skipped'

export interface StepView {
  readonly id: StepId
  readonly state: StepState
  /** A sub-chip: "Blick 2", "Versuch 2/5". */
  readonly note: Msg | null
}

/**
 * `lost` is a run the server no longer knows (it restarted): how it ended cannot be said here, only that it is no
 * longer running as far as the console can tell.
 */
export type Phase = 'idle' | 'countdown' | 'running' | 'stopping' | 'halting' | 'finished' | 'cancelled' | 'failed' | 'lost'

/**
 * Where a stop acts: a task's "stop after this part", a pick run's Stop before its next attempt (plan risk 16), or a
 * Home run's stop before its one move is sent (`before_motion`: a move already under way runs to its end).
 */
export type StopScope = 'after_part' | 'between_attempts' | 'before_motion'

export type Tone = 'info' | 'ok' | 'warn' | 'error'

/** `demo` lines are read by everyone; `tech` lines only in the tech view. */
export type Level = 'demo' | 'tech'

export interface ChatLine {
  /** Unique within the conversation: `<run>:<seq>`, or `<run>:<name>` for a line that updates in place. */
  readonly id: string
  readonly runId: string
  readonly seq: number
  /** Unix seconds, the server's clock. */
  readonly at: number
  /** The event type the line says, `''` for the console's own lines. */
  readonly type: string
  readonly msg: Msg<ChatKey>
  /** The backend's English sentence, word for word, for the tech view. `''` for the console's own lines. */
  readonly human: string
  readonly tone: Tone
  readonly level: Level
  /**
   * The part the line belongs to, which the chat draws as that part's card; `null` for the run's own lines (its start,
   * its stops, its end, a return that is no part's) and for the cell's.
   */
  readonly part: number | null
  /** The event's machine half, as the server sent it, for the tech view to open. `{}` for the console's own lines. */
  readonly data: Readonly<Record<string, unknown>>
}

export interface PickView {
  readonly pick: number
  readonly part: number | null
  readonly at: number
  readonly outcome: string
  readonly succeeded: boolean
  readonly reason: string
  readonly looks: readonly string[]
  readonly looksFused: readonly string[]
  /** Jaw 1, jaw 2: whether each contact face of the chosen grasp was seen. */
  readonly faces: readonly boolean[] | null
  readonly handEyeMm: number | null
  readonly handEyeWarnMm: number | null
  /** `false` on a hand that measures nothing: "nicht gemessen (kein Sensor)", never "held". */
  readonly holdMeasured: boolean | null
  readonly bothFaces: boolean
  readonly pushes: number
  readonly pushStopped: boolean
  readonly foundNothing: boolean
  readonly onlyExcluded: boolean
  readonly detectorFailed: boolean
  /**
   * A halt (or a cancel) cut the pick short: it ended because a person stopped it, not because it failed to grasp.
   * It is no failure in the success rate.
   */
  readonly cut: boolean
  readonly graspMm: readonly number[] | null
  readonly overlay: string | null
  readonly viewsFile: string | null
}

export interface PartView {
  readonly part: number
  readonly of: number | null
  readonly startedAt: number
  readonly picks: readonly PickView[]
  /** `null` while the part is not finished. */
  readonly placed: boolean | null
  /**
   * A person stopped the run while this part was in hand (a halt, a cancel, a Disconnect): it was not placed, and it
   * is no failure in the success rate, as a pick a halt cut short is not.
   */
  readonly cut: boolean
  readonly durationS: number | null
  /** How far the part was pushed free, summed over its pushes. */
  readonly pushedMm: number
  /** The last grasp overlay of this part, for its card's thumbnail. */
  readonly overlay: string | null
}

export interface CurrentView {
  readonly part: number | null
  readonly of: number | null
  readonly attempt: number | null
  readonly attemptTotal: number | null
  readonly look: string | null
  /** How many looks this attempt has taken so far ("Blick 2"). */
  readonly lookIndex: number
  readonly step: StepId | null
  /** The push of the part in hand, for the "30 mm geschoben, erneut geschaut" sub-chip. */
  readonly pushMm: number | null
}

export interface CountdownView {
  readonly state: 'none' | 'active' | 'done' | 'cancelled'
  readonly secondsLeft: number | null
  readonly because: string | null
}

export interface SurveyView {
  readonly state: 'none' | 'active' | 'done' | 'failed'
  readonly phrase: string | null
  readonly looks: readonly string[]
  readonly look: string | null
  readonly partsSeen: number | null
}

export interface TargetView {
  readonly label: string
  readonly score: number | null
  readonly centreMm: readonly number[] | null
  readonly rimMm: number | null
  readonly footprintMm: readonly number[] | null
  readonly openingMm: readonly number[] | null
  readonly look: string | null
  readonly seenAt: number | null
  readonly overlay: string | null
}

/** The stop card of a problem stop (build plan 1.3.5): the arm stands where it stopped. */
export interface StopCardView {
  readonly runId: string
  readonly kind: RunKind | null
  readonly stopCode: StopCode
  readonly stopClass: StopClass
  /** The backend's sentence, for the tech view. */
  readonly error: string
  readonly part: number | null
  readonly step: StepId | null
  readonly at: number
  readonly holding: boolean
}

/** The ask card of a question stop: the run returned, and the person chooses what next. */
export interface AskCardView {
  readonly runId: string
  readonly stopCode: StopCode
  readonly part: number | null
  /** Why a kept target was lost: `not_seen`, `moved_too_far` or `footprint_changed`. */
  readonly why: string | null
  readonly target: TargetView | null
  readonly error: string
}

export interface StatsView {
  /** Parts released at the place. */
  readonly placed: number
  /** Picks the run made, the empty looks included. */
  readonly picks: number
  readonly succeeded: number
  /** Picks that saw nothing matching (or only the task's own kept-out parts): how "until empty" ends. */
  readonly emptyLooks: number
  /** Picks a halt or a cancel cut short: a person stopped them, they did not fail. */
  readonly cut: number
  /**
   * The picks the success rate is of. A pick run: the picks that had something to grasp and were let finish. A task:
   * those of them whose part's end is known, a grasp that failed at once, a gripped part once it is placed or not; a
   * part a person stopped in the jaws is left out (`PartView.cut`). History sums this count, so its rate is this one.
   */
  readonly rated: number
  /**
   * Placed (a task) or grasped (a pick run) over `rated`. An empty look is how "until empty" ends, a part on its way is
   * not decided yet, and a pick or a part a person halted was stopped by a person: none of them is a failure, so none
   * is in the denominator, and the rate never dips while a part is carried. `null` before the first rated pick.
   */
  readonly successRate: number | null
  /** The median time of a placed part, in seconds. */
  readonly medianPartS: number | null
  readonly outcomes: Readonly<Record<string, number>>
  readonly pushes: number
}

export interface HaltView {
  /** `requested` from the run's own event; `confirmed` once the run stopped on it, or ended on any other code. */
  readonly state: 'none' | 'requested' | 'confirmed'
  readonly requestedAt: number | null
  readonly inMotion: boolean
  readonly braking: boolean
  readonly reason: string
}

/** What the halt button shows: the view's state, plus `unconfirmed` (see `haltState`). */
export type HaltState = 'none' | 'requested' | 'confirmed' | 'unconfirmed'

export interface OverlayPin {
  readonly kind: 'grasp' | 'target'
  readonly url: string
  /** When the overlay was captured: "wie entschieden 12:31:05", the arm has moved since. */
  readonly at: number
  readonly part: number | null
  readonly pick: number | null
  readonly look: string | null
}

export type TeachState =
  | 'freeing'
  | 'free'
  | 'holding_when_still'
  | 'holding'
  | 'screening'
  | 'saved'
  | 'refused'
  | 'not_saved'

export interface TeachView {
  readonly name: string
  readonly label: string
  readonly role: string
  readonly state: TeachState
  readonly outside: boolean
  /** The TCP left the workspace box at least once during the session. */
  readonly wasOutside: boolean
  readonly secondsLeft: number | null
  readonly verdict: string | null
  readonly lines: readonly string[]
}

export interface RunView {
  readonly runId: string | null
  readonly kind: RunKind | null
  readonly plan: TaskPlanOut | null
  readonly phase: Phase
  readonly restartOf: string | null
  readonly startedAt: number | null
  readonly finishedAt: number | null
  /** A Home run's target (`home` or a pose name). */
  readonly to: string | null
  /** A pick run's prompt. */
  readonly prompt: string | null
  readonly parts: readonly PartView[]
  readonly picks: readonly PickView[]
  readonly current: CurrentView
  readonly timeline: readonly StepView[]
  readonly countdown: CountdownView
  readonly survey: SurveyView
  readonly chat: readonly ChatLine[]
  readonly stopCard: StopCardView | null
  readonly askCard: AskCardView | null
  readonly stats: StatsView
  /**
   * What the program believes is in the jaws, as the run says it. The stop card's gate is the server's word
   * (`part_still_held`), which counts the stop record's belief for a hand that can say nothing itself.
   */
  readonly holding: boolean
  readonly halt: HaltView
  readonly target: TargetView | null
  readonly targetCheck: { readonly movedMm: number | null; readonly followed: boolean } | null
  /** Why the kept target was lost at the drop, for the ask card the run ends on. */
  readonly lostWhy: string | null
  readonly overlays: readonly OverlayPin[]
  readonly gaps: { readonly count: number; readonly dropped: number }
  /** "Stop after this part" was asked. */
  readonly stopRequested: boolean
  /** Where the asked stop acts, for the words of the `stopping` phase (`phaseMsg`). `null` while none was asked. */
  readonly stopScope: StopScope | null
  readonly stopCode: StopCode | ''
  readonly stopClass: StopClass | ''
  readonly error: string
  /** After a done or operator end, Willy asks "Was soll ich als Nächstes tun?" and the input takes the focus. */
  readonly nextQuestion: boolean
  readonly teach: TeachView | null
  readonly lastSeq: number
  /** The server's record of the run, as the last snapshot or `run_finished` said it. */
  readonly run: RunOut | null
}

const IDLE_TIMELINE: readonly StepView[] = STEPS.map((id) => ({ id, state: 'idle' as StepState, note: null }))

const NO_CURRENT: CurrentView = {
  part: null,
  of: null,
  attempt: null,
  attemptTotal: null,
  look: null,
  lookIndex: 0,
  step: null,
  pushMm: null,
}

const NO_STATS: StatsView = {
  placed: 0,
  picks: 0,
  succeeded: 0,
  emptyLooks: 0,
  cut: 0,
  rated: 0,
  successRate: null,
  medianPartS: null,
  outcomes: {},
  pushes: 0,
}

/** The view before any run. */
export const EMPTY_RUN: RunView = {
  runId: null,
  kind: null,
  plan: null,
  phase: 'idle',
  restartOf: null,
  startedAt: null,
  finishedAt: null,
  to: null,
  prompt: null,
  parts: [],
  picks: [],
  current: NO_CURRENT,
  timeline: IDLE_TIMELINE,
  countdown: { state: 'none', secondsLeft: null, because: null },
  survey: { state: 'none', phrase: null, looks: [], look: null, partsSeen: null },
  chat: [],
  stopCard: null,
  askCard: null,
  stats: NO_STATS,
  holding: false,
  halt: { state: 'none', requestedAt: null, inMotion: false, braking: false, reason: '' },
  target: null,
  targetCheck: null,
  lostWhy: null,
  overlays: [],
  gaps: { count: 0, dropped: 0 },
  stopRequested: false,
  stopScope: null,
  stopCode: '',
  stopClass: '',
  error: '',
  nextQuestion: false,
  teach: null,
  lastSeq: 0,
  run: null,
}

// ── actions ───────────────────────────────────────────────────────────────────────────────────────────────────

/** The server's run record, fetched rather than streamed: at the start of a follow, and after a gap. */
export interface SnapshotAction {
  readonly type: 'run.snapshot'
  readonly run: RunOut
}

/**
 * The server answered that it does not know the followed run (`404` for `GET /v1/runs/{id}`): it restarted, and the
 * run's record and events went with it. The console stops drawing the run as running and says why.
 */
export interface LostAction {
  readonly type: 'run.lost'
  readonly runId: string
}

export type RunAction = RunEvent | SnapshotAction | LostAction

function isSnapshot(action: RunAction): action is SnapshotAction {
  return action.type === 'run.snapshot' && 'run' in action
}

function isLost(action: RunAction): action is LostAction {
  return action.type === 'run.lost' && 'runId' in action
}

/** The run is over: it ended on a stop code, or the server no longer knows it. */
export function hasEnded(phase: Phase): boolean {
  return phase === 'finished' || phase === 'cancelled' || phase === 'failed' || phase === 'lost'
}

// ── small readers: a payload is only as complete as the server that sent it ─────────────────────────────────────

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value !== '' ? value : null
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []
}

function numbers(value: unknown): number[] | null {
  return Array.isArray(value) && value.every((v) => typeof v === 'number') ? (value as number[]) : null
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function m(key: MessageKey, params?: Params): Msg {
  return params === undefined ? { key } : { key, params }
}

/** The picks that had something to grasp and ran their course: a pick run's success rate's denominator. */
function fairPicks(picks: number, emptyLooks: number, cut: number): number {
  return picks - emptyLooks - cut
}

/**
 * A task's picks the success rate is of: every grasp that failed (its part is looked for again: decided at once), and
 * every grasp whose part's end is known, placed or not. A part in the jaws counts once it is down, so the rate never
 * dips while a part is carried, the longest phase on a real cell; a part a person stopped in the jaws (a halt, a
 * cancel, a Disconnect) is no failure, as a pick a halt cut short is not. A grasp whose part this view never heard
 * start counts as it lands.
 */
function ratedPicks(view: RunView): number {
  let rated = 0
  for (const pick of view.picks) {
    if (pick.foundNothing || pick.onlyExcluded || pick.cut) continue
    if (!pick.succeeded) {
      rated += 1
      continue
    }
    const part = pick.part === null ? undefined : view.parts.find((p) => p.part === pick.part)
    if (part !== undefined && (part.placed === null || part.cut)) continue
    rated += 1
  }
  return rated
}

function round(value: number | null, digits = 0): number | null {
  if (value === null) return null
  const f = 10 ** digits
  return Math.round(value * f) / f
}

function median(values: readonly number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

const VERDICTS = ['clear', 'band', 'guard_refused', 'planner_refused', 'unscreened', 'error'] as const

/** A screen verdict in words; one this console does not know is shown as its own words. */
export function verdictMsg(verdict: unknown): Msg {
  if (typeof verdict === 'string' && (VERDICTS as readonly string[]).includes(verdict)) {
    return m(`verdict.${verdict as (typeof VERDICTS)[number]}`)
  }
  return m('common.raw', { text: typeof verdict === 'string' ? verdict.replace(/_/g, ' ') : '—' })
}

// ── building blocks ──────────────────────────────────────────────────────────────────────────────────────────

function withStep(timeline: readonly StepView[], id: StepId, state: StepState, note?: Msg | null): readonly StepView[] {
  return timeline.map((s) => (s.id === id ? { id, state, note: note === undefined ? s.note : note } : s))
}

function stepState(timeline: readonly StepView[], id: StepId): StepState {
  return timeline.find((s) => s.id === id)?.state ?? 'idle'
}

/** The step a stop happened at: the last one, in the plan's order, that failed or was still going. */
function stoppedStep(view: RunView): StepId | null {
  let at: StepId | null = null
  for (const s of view.timeline) if (s.state === 'active' || s.state === 'failed') at = s.id
  return at ?? view.current.step
}

/** The run's own events: they belong to the run, never to the part in hand, whatever part the run is at. */
function runLevel(type: string): boolean {
  return type === '' || type === 'gap' || type.startsWith('run_')
}

/**
 * A return that is no part's: the restart's first motion (no part yet) and the final return after the empty looks
 * (the part being looked for was never placed nor carried). A part's own return comes after its place step.
 */
function standaloneReturn(view: RunView): boolean {
  return view.current.part === null || stepState(view.timeline, 'place') === 'idle'
}

function say(
  view: RunView,
  event: RunEvent,
  msg: Msg<ChatKey>,
  tone: Tone = 'info',
  level: Level = 'demo',
  id?: string,
  part?: number | null,
): RunView {
  const line: ChatLine = {
    id: id ?? `${event.run_id}:${event.seq}`,
    runId: event.run_id,
    seq: event.seq,
    at: event.ts,
    type: event.type,
    msg,
    human: event.human ?? '',
    tone,
    level,
    part: part !== undefined ? part : runLevel(event.type) ? null : view.current.part,
    data: record(event.data),
  }
  const index = view.chat.findIndex((l) => l.id === line.id)
  if (index < 0) return { ...view, chat: [...view.chat, line] }
  const chat = [...view.chat]
  chat[index] = line
  return { ...view, chat }
}

/** A line of the console's own (the next-instruction question): no backend sentence behind it. */
function sayOwn(view: RunView, event: RunEvent, name: string, msg: Msg, tone: Tone = 'info'): RunView {
  const line: ChatLine = {
    id: `${event.run_id}:${name}`,
    runId: event.run_id,
    seq: event.seq,
    at: event.ts,
    type: '',
    msg,
    human: '',
    tone,
    level: 'demo',
    part: null,
    data: {},
  }
  return view.chat.some((l) => l.id === line.id) ? view : { ...view, chat: [...view.chat, line] }
}

function pin(view: RunView, overlay: OverlayPin): RunView {
  if (view.overlays.some((o) => o.url === overlay.url)) return view
  return { ...view, overlays: [...view.overlays, overlay] }
}

/** Where the plan places a part, in the words a person reads: the pose's label, or the target as they said it. */
function placeWords(plan: TaskPlanOut | null, fallback: string | null): string | Msg {
  const place = plan?.place
  if (place?.kind === 'pose') return place.pose_label || place.pose || m('common.defaultPlace')
  if (place?.kind === 'camera') return place.said || place.phrase || fallback || '—'
  return fallback || '—'
}

/** A return target in words: Home, a taught pose's label, or its name. */
function returnWords(plan: TaskPlanOut | null, to: string | null): string | Msg {
  const target = to ?? plan?.return_to ?? 'home'
  if (target === 'home') return m('common.home')
  if (plan && plan.return_to === target && plan.return_label) return plan.return_label
  return target
}

function updatePart(view: RunView, part: number | null, change: (p: PartView) => PartView): RunView {
  if (part === null) return view
  const index = view.parts.findIndex((p) => p.part === part)
  if (index < 0) return view
  const parts = [...view.parts]
  parts[index] = change(parts[index])
  return { ...view, parts }
}

function computeStats(view: RunView): StatsView {
  const picks = view.picks.length
  const succeeded = view.picks.filter((p) => p.succeeded).length
  const emptyLooks = view.picks.filter((p) => p.foundNothing || p.onlyExcluded).length
  const cut = view.picks.filter((p) => p.cut).length
  const placed = view.parts.filter((p) => p.placed === true).length
  const outcomes: Record<string, number> = {}
  for (const p of view.picks) outcomes[p.outcome] = (outcomes[p.outcome] ?? 0) + 1
  const task = view.kind === 'task'
  const real = task ? ratedPicks(view) : fairPicks(picks, emptyLooks, cut)
  const wins = task ? placed : succeeded
  return {
    placed,
    picks,
    succeeded,
    emptyLooks,
    cut,
    rated: real,
    successRate: real > 0 ? Math.min(1, wins / real) : null,
    medianPartS: median(view.parts.filter((p) => p.placed === true && p.durationS !== null).map((p) => p.durationS!)),
    outcomes,
    pushes: view.picks.reduce((sum, p) => sum + p.pushes, 0),
  }
}

/** The counters as the server's record states them: what a view holds after a gap swallowed some of its events. */
function statsFromRecord(view: RunView, run: RunOut): StatsView {
  const outcomes: Record<string, number> = {}
  for (const o of run.outcomes ?? []) outcomes[o] = (outcomes[o] ?? 0) + 1
  const picks = Math.max(view.stats.picks, run.attempted ?? 0)
  const succeeded = Math.max(view.stats.succeeded, run.succeeded ?? 0)
  const placed = Math.max(view.stats.placed, run.parts_placed ?? 0)
  // A task still running may have a part in hand, which its record cannot tell from one that failed: its rate stays
  // as the events last counted it, never a dip drawn from the record. A record of a run that ended is whole.
  const pending = view.kind === 'task' && run.state === 'running'
  const real = pending ? view.stats.rated : fairPicks(picks, view.stats.emptyLooks, view.stats.cut)
  const wins = view.kind === 'task' ? placed : succeeded
  return {
    ...view.stats,
    picks,
    succeeded,
    placed,
    outcomes: Object.keys(outcomes).length > 0 ? outcomes : view.stats.outcomes,
    rated: real,
    successRate: real > 0 && !pending ? Math.min(1, wins / real) : view.stats.successRate,
  }
}

function phaseOf(state: string | undefined, current: Phase): Phase {
  if (state === 'finished' || state === 'cancelled' || state === 'failed') return state
  if (state === 'running') return current === 'idle' ? 'running' : current
  return current
}

function stopOf(code: unknown): { stopCode: StopCode | ''; stopClass: StopClass | '' } {
  return isStopCode(code) ? { stopCode: code, stopClass: STOP_CLASS_OF[code] } : { stopCode: '', stopClass: '' }
}

function stopCardOf(view: RunView, stopCode: StopCode, at: number, error: string, holding: boolean): StopCardView {
  return {
    runId: view.runId ?? '',
    kind: view.kind,
    stopCode,
    stopClass: STOP_CLASS_OF[stopCode],
    error,
    part: view.current.part,
    step: stoppedStep(view),
    at,
    holding,
  }
}

// ── the reducer ───────────────────────────────────────────────────────────────────────────────────────────────

/** Start a fresh view for a new run: nothing of the previous run's parts, cards or timeline survives. */
function fresh(runId: string): RunView {
  return { ...EMPTY_RUN, runId }
}

/** Apply the server's record of the run: kind, plan, the stop, and, after a gap, the counters. */
function applyRecord(view: RunView, run: RunOut): RunView {
  const { stopCode, stopClass } = stopOf(run.stop_code)
  let next: RunView = {
    ...view,
    runId: run.id ?? view.runId,
    kind: run.kind ?? view.kind,
    plan: run.plan ?? view.plan,
    restartOf: run.restart_of ?? view.restartOf,
    startedAt: num(run.started_at) ?? view.startedAt,
    finishedAt: num(run.finished_at) ?? view.finishedAt,
    prompt: view.prompt ?? str(run.prompt),
    phase: phaseOf(run.state, view.phase),
    stopCode: stopCode || view.stopCode,
    stopClass: stopClass || view.stopClass,
    error: run.error || view.error,
    stopRequested: view.stopRequested || run.stop_requested === true,
    run,
  }
  if (run.state === 'finished' || run.state === 'cancelled' || run.state === 'failed') {
    next = { ...next, holding: run.holding === true }
  }
  // The record's counters stand where the events could not count: after a gap, and before the first event of a run
  // followed late (a stopped run read again after a reload) has been replayed. A replayed pick counts afresh.
  if (view.gaps.count > 0 || view.picks.length === 0) next = { ...next, stats: statsFromRecord(next, run) }
  return next
}

/** The cards and the closing lines of a run that ended. */
function finish(view: RunView, event: RunEvent, run: RunOut): RunView {
  let next = applyRecord(view, run)
  const code = next.stopCode
  const cls = next.stopClass
  const at = num(run.finished_at) ?? event.ts
  // A person ended the run: "Sofort anhalten" (also where the run then ended on another code), a stop before the first
  // motion, a Disconnect. A part they stopped in hand did not fail.
  const personStopped =
    code === 'halted' || code === 'cancelled' || code === 'disconnected' || view.halt.state !== 'none' || run.halt_requested === true

  // A part the run never finished (nothing found, a stop) was not placed: the run is over. Where a person stopped it,
  // the part is no failure in the success rate; the rate is counted again with every part's end now known.
  if (next.parts.some((p) => p.placed === null)) {
    next = { ...next, parts: next.parts.map((p) => (p.placed === null ? { ...p, placed: false, cut: personStopped } : p)) }
    if (next.gaps.count === 0 && next.picks.length > 0) next = { ...next, stats: computeStats(next) }
  }
  // A step still going when the run stopped on a problem did not finish: it failed there.
  if (cls === 'problem') {
    next = { ...next, timeline: next.timeline.map((s) => (s.state === 'active' ? { ...s, state: 'failed' } : s)) }
  }
  if (next.countdown.state === 'active') {
    next = { ...next, countdown: { ...next.countdown, state: code === 'cancelled' ? 'cancelled' : 'done' } }
  }
  // The run's thread has ended before `run_finished` is published (api/runs.py `_finish`), so it commands nothing
  // more: a halt still waiting for its confirmation is confirmed by the end, whatever the code (a teach the halt
  // cancelled, a Home move that arrived, a refused return). Left `requested`, it would turn into "press the e-stop"
  // 1.5 s later over an arm nothing drives any more.
  if (code === 'halted' || next.halt.state === 'requested') next = { ...next, halt: { ...next.halt, state: 'confirmed' } }

  next = {
    ...next,
    stopCard: cls === 'problem' && code ? stopCardOf(next, code, at, next.error, next.holding) : null,
    askCard:
      cls === 'ask' && code
        ? {
            runId: next.runId ?? event.run_id,
            stopCode: code,
            part: next.current.part,
            why: next.lostWhy,
            target: next.target,
            error: next.error,
          }
        : null,
  }

  const parts = next.gaps.count > 0 ? Math.max(next.stats.placed, run.parts_placed ?? 0) : next.stats.placed
  const title = code ? stopMsg(code) : m('common.none')
  const refused: unknown = run.refusal?.code
  let end: Msg
  let tone: Tone = 'info'
  const kind = next.kind
  if (cls === 'problem') {
    end = m('event.run_finished.problem', { title })
    tone = 'error'
  } else if (cls === 'ask') {
    end = m('event.run_finished.ask', { title })
    tone = 'warn'
  } else if (code === 'cancelled' && isRefusalCode(refused)) {
    // The library refused the task inside its run, before anything moved (the cell changed after the route's gates):
    // what it refused is the news, not "cancelled" (RunOut.refusal; no run_error and no stop record were written).
    end = refusalMsg(refused)
    tone = 'warn'
  } else if (code === 'cancelled' && /jaws question/i.test(next.error)) {
    // A jaws check began as the run took the cell: the run gave way before its first motion. Answer the question,
    // then start again; never a problem stop. The server types it (RunOut.refusal `jaws_question_pending`, read
    // above, api/runs.py `_give_way_to_the_jaws_question`); the sentence counts only where a record carries none.
    end = refusalMsg('jaws_question_pending')
    tone = 'warn'
  } else if (code === 'cancelled') {
    end = m('event.run_finished.cancelled')
  } else if (kind === 'task' && code === 'stopped_after_part') {
    end = m('event.run_finished.stopped_after_part', { parts })
  } else if (kind === 'task' && (code === 'nothing_left' || code === 'part_limit')) {
    end = m(`event.run_finished.${code}`, { parts })
    tone = 'ok'
  } else if (kind === 'task' && code === 'finished') {
    end = m('event.run_finished.task', { parts })
    tone = 'ok'
  } else if (kind === 'home' && code === 'finished') {
    end = m('event.run_finished.home', { to: returnWords(null, next.to) })
    tone = 'ok'
  } else if (kind === 'wave' && code === 'finished') {
    end = m('event.run_finished.wave')
    tone = 'ok'
  } else if (kind === 'pick' && code === 'finished') {
    end = m('event.run_finished.pick', { succeeded: run.succeeded ?? next.stats.succeeded, attempted: run.attempted ?? next.stats.picks })
    tone = 'ok'
  } else if (code === 'taught') {
    end = m('event.run_finished.taught', { label: next.teach?.label || next.teach?.name || '—' })
    tone = 'ok'
  } else if (code === 'planner_ready') {
    end = m('event.run_finished.planner_ready')
    tone = 'ok'
  } else if (cls === 'teach') {
    end = m('event.run_finished.teach', { title })
    tone = 'warn'
  } else if (cls === 'planner') {
    end = m('event.run_finished.planner', { title })
    tone = 'error'
  } else {
    end = m('event.run_finished', { kind: kind ? runKindMsg(kind) : m('common.none'), title })
  }
  next = say(next, event, end, tone, 'demo')

  const asks = (cls === 'done' || cls === 'operator') && (kind === 'task' || kind === 'pick' || kind === 'home' || kind === 'wave')
  if (asks) next = sayOwn({ ...next, nextQuestion: true }, event, 'next', m('chat.next'))
  return next
}

function startRun(view: RunView, event: RunEvent): RunView {
  const data = record(event.data)
  const kind = (str(data.kind) as RunKind | null) ?? 'pick'
  const plan = (data.plan as TaskPlanOut | null | undefined) ?? null
  let next: RunView = {
    ...fresh(event.run_id),
    kind,
    plan,
    phase: 'running',
    restartOf: str(data.restart_of),
    startedAt: event.ts,
    to: str(data.to),
    prompt: str(data.prompt),
    lastSeq: view.runId === event.run_id ? view.lastSeq : 0,
  }
  if (kind === 'teach') {
    next = {
      ...next,
      teach: {
        name: str(data.name) ?? '',
        label: str(data.label) ?? '',
        role: str(data.role) ?? 'other',
        state: 'freeing',
        outside: false,
        wasOutside: false,
        secondsLeft: null,
        verdict: null,
        lines: [],
      },
    }
  }
  let line: Msg
  if (kind === 'task' && plan?.first_motion === 'return') {
    line = m('event.run_started.restart', { back: returnWords(plan, null) })
  } else if (kind === 'task') {
    line = m('event.run_started.task', {
      what: plan?.object_said || plan?.object || m('common.anything'),
      where: placeWords(plan, null),
      back: returnWords(plan, null),
      // Inside the sentence, in lower case: "Bis leer" with its capital is the switch's label.
      scope: m(plan?.scope === 'until_empty' ? 'scope.inline.until_empty' : 'scope.inline.once'),
    })
  } else if (kind === 'home') {
    line = m('event.run_started.home', { to: returnWords(null, next.to ?? 'home') })
  } else if (kind === 'teach') {
    line = m('event.run_started.teach', { label: next.teach?.label || next.teach?.name || '—' })
  } else if (kind === 'planner') {
    line = m('event.run_started.planner')
  } else if (kind === 'wave') {
    line = m('event.run_started.wave')
  } else {
    line = m('event.run_started.pick', { what: next.prompt || m('common.anything') })
  }
  return say(next, event, line, 'info', 'demo')
}

/** The first motion has begun: a countdown still showing is over. */
function moving(view: RunView): RunView {
  if (view.countdown.state !== 'active') return view
  return { ...view, countdown: { ...view.countdown, state: 'done' }, phase: view.phase === 'countdown' ? 'running' : view.phase }
}

function pickResult(view: RunView, event: RunEvent): RunView {
  const data = record(event.data)
  const succeeded = data.succeeded === true
  const foundNothing = data.found_nothing === true
  const onlyExcluded = data.only_excluded === true
  const part = num(data.part) ?? view.current.part
  // A pick that did not grasp after "Sofort anhalten" was pressed, or that its cancel check ended, was stopped by a
  // person: the halt latches the arm and the pick ends on whatever the refused motion reports (`execution_failed` on
  // the captured logs), never on a grasp that was tried and missed.
  const cut = !succeeded && !foundNothing && !onlyExcluded && (view.halt.state !== 'none' || loopOutcome(str(data.outcome)) === 'cancelled')
  const pick: PickView = {
    pick: num(data.pick) ?? view.picks.length + 1,
    part,
    at: event.ts,
    outcome: str(data.outcome) ?? (succeeded ? 'succeeded' : 'unknown'),
    succeeded,
    reason: str(data.reason) ?? '',
    looks: strings(data.looks),
    looksFused: strings(data.looks_fused ?? data.fused_views),
    faces: Array.isArray(data.jaw_faces_seen) ? (data.jaw_faces_seen as unknown[]).map((v) => v === true) : null,
    handEyeMm: num(data.hand_eye_gap_mm),
    handEyeWarnMm: num(data.hand_eye_warn_mm),
    holdMeasured: typeof data.hold_measured === 'boolean' ? data.hold_measured : null,
    bothFaces: data.both_faces === true,
    pushes: num(data.pushes) ?? 0,
    pushStopped: data.push_stopped === true,
    foundNothing,
    onlyExcluded,
    detectorFailed: data.detector_failed === true,
    cut,
    graspMm: numbers(record(data.grasp_pose).position_mm),
    overlay: str(data.overlay),
    viewsFile: str(data.views_file),
  }
  let next: RunView = { ...view, picks: [...view.picks, pick] }
  next = updatePart(next, part, (p) => ({ ...p, picks: [...p.picks, pick], overlay: pick.overlay ?? p.overlay }))

  if (foundNothing || onlyExcluded) {
    let timeline = next.timeline
    if (stepState(timeline, 'look') === 'active') timeline = withStep(timeline, 'look', 'done')
    timeline = withStep(timeline, 'detect', 'warn')
    timeline = withStep(timeline, 'grasp', 'skipped')
    next = { ...next, timeline }
  } else if (succeeded) {
    let timeline = next.timeline
    for (const id of ['look', 'detect'] as const) if (stepState(timeline, id) !== 'done') timeline = withStep(timeline, id, 'done')
    next = { ...next, timeline: withStep(timeline, 'grasp', 'done'), holding: true }
  } else {
    next = { ...next, timeline: withStep(next.timeline, 'grasp', 'failed') }
  }

  if (pick.overlay) {
    next = pin(next, { kind: 'grasp', url: pick.overlay, at: event.ts, part, pick: pick.pick, look: next.current.look })
  }
  next = { ...next, stats: computeStats(next) }
  if (next.gaps.count > 0 && next.run) next = { ...next, stats: statsFromRecord(next, next.run) }

  // In a task the nothing-found line comes from the task; a pick run says it here.
  const task = next.kind === 'task'
  if (foundNothing || onlyExcluded) return say(next, event, m('event.pick_result.nothing'), 'info', task ? 'tech' : 'demo')
  if (succeeded) {
    const fused = pick.looksFused.length
    return say(next, event, fused > 1 ? m('event.pick_result.fused', { fused }) : m('event.pick_result.succeeded'), 'ok')
  }
  return say(next, event, m('event.pick_result.failed', { outcome: outcomeMsg(pick.outcome) }), 'warn')
}

function teachEvent(view: RunView, event: RunEvent): RunView {
  const data = record(event.data)
  const teach = view.teach ?? {
    name: '',
    label: '',
    role: 'other',
    state: 'freeing' as TeachState,
    outside: false,
    wasOutside: false,
    secondsLeft: null,
    verdict: null,
    lines: [],
  }
  const set = (change: Partial<TeachView>): RunView => ({ ...view, teach: { ...teach, ...change } })
  switch (event.type) {
    case 'teach.free':
      return say(set({ state: 'free' }), event, m('event.teach.free'))
    case 'teach.say': {
      const line = str(data.line) ?? event.human
      return say(set({ lines: [...teach.lines, line] }), event, m('event.teach.say', { line }), 'info', 'tech')
    }
    case 'teach.outside':
      return say(set({ outside: true, wasOutside: true }), event, m('event.teach.outside'), 'warn')
    case 'teach.inside':
      return say(set({ outside: false }), event, m('event.teach.inside'), 'info', 'tech')
    case 'teach.time_warning': {
      const seconds = num(data.seconds_left)
      return say(set({ secondsLeft: seconds }), event, m('event.teach.time_warning', { seconds }), 'warn')
    }
    case 'teach.holding_when_still':
      return say(set({ state: 'holding_when_still' }), event, m('event.teach.holding_when_still'), 'warn')
    case 'teach.holding':
      return say(set({ state: 'holding' }), event, m('event.teach.holding'), 'info', 'tech')
    case 'teach.screening':
      return say(set({ state: 'screening' }), event, m('event.teach.screening'))
    case 'teach.saved': {
      const verdict = str(data.verdict)
      const label = str(data.label) ?? teach.label
      return say(set({ state: 'saved', verdict, label }), event, m('event.teach.saved', { label, verdict: verdictMsg(verdict) }), 'ok')
    }
    case 'teach.refused': {
      const verdict = str(data.verdict)
      return say(set({ state: 'refused', verdict }), event, m('event.teach.refused', { verdict: verdictMsg(verdict) }), 'error')
    }
    case 'teach.not_saved':
      return say(set({ state: 'not_saved' }), event, m('event.teach.not_saved'), 'warn')
    default:
      return view
  }
}

function taskEvent(view: RunView, event: RunEvent): RunView {
  const data = record(event.data)
  const plan = view.plan
  switch (event.type) {
    case 'task.pose_screened': {
      const pose = str(data.pose)
      const label = plan?.place?.pose === pose && plan?.place?.pose_label ? plan.place.pose_label : (pose ?? '—')
      const verdict = str(data.verdict)
      const tone: Tone = verdict === 'clear' || verdict === 'band' ? 'info' : 'warn'
      // Not screened ahead (the screen itself could not judge it): the move judges it when it runs. Never "geprüft:
      // nicht geprüft".
      if (verdict === 'unscreened') return say(view, event, { key: 'ck.pose.unscreened', params: { pose: label } }, tone, 'tech')
      return say(view, event, m('event.task.pose_screened', { pose: label, verdict: verdictMsg(verdict) }), tone, 'tech')
    }
    // The camera's search is said around the operator's own words in quotes: the reader hands them over as they were
    // said, a preposition and all ("in die blaue Kiste"), which no sentence can take as its object.
    case 'task.survey_started': {
      const phrase = str(data.phrase)
      const looks = strings(data.looks)
      const next = moving({ ...view, survey: { ...view.survey, state: 'active', phrase, looks } })
      return say(next, event, { key: 'ck.target.looking', params: { target: plan?.place?.said || phrase || '—', looks: looks.length } })
    }
    case 'task.target_found': {
      const target = targetOf(data.target, str(data.look))
      let next: RunView = {
        ...view,
        survey: { ...view.survey, state: 'done', look: str(data.look) ?? target?.look ?? null, partsSeen: num(data.parts_seen) },
        target,
      }
      if (target?.overlay) {
        next = pin(next, { kind: 'target', url: target.overlay, at: event.ts, part: null, pick: null, look: target.look })
      }
      return say(next, event, { key: 'ck.target.found', params: { score: target?.score ?? null, parts: num(data.parts_seen) ?? 0 } }, 'ok')
    }
    case 'task.target_missing': {
      const next = { ...view, survey: { ...view.survey, state: 'failed' as const } }
      return say(next, event, { key: 'ck.target.missing', params: { target: plan?.place?.said || str(data.phrase) || '—' } }, 'warn')
    }
    case 'task.part_started': {
      const part = num(data.part)
      if (part === null) return view
      const of = num(data.of)
      const known = view.parts.some((p) => p.part === part)
      const parts = known
        ? view.parts
        : [...view.parts, { part, of, startedAt: event.ts, picks: [], placed: null, cut: false, durationS: null, pushedMm: 0, overlay: null }]
      const next: RunView = {
        ...view,
        parts,
        timeline: IDLE_TIMELINE,
        current: { ...NO_CURRENT, part, of, step: 'look', pushMm: known ? view.current.pushMm : null },
      }
      const line = of !== null ? m('event.task.part_started.of', { part, of }) : m('event.task.part_started', { part })
      return say(next, event, line, 'info', known ? 'tech' : 'demo')
    }
    case 'task.nothing_found': {
      const empty = num(data.empty_in_a_row) ?? 1
      const line = data.only_excluded === true
        ? m('event.task.nothing_found.excluded', { empty })
        : m('event.task.nothing_found', { empty })
      return say(view, event, line)
    }
    case 'task.carry_started': {
      const next: RunView = {
        ...moving(view),
        timeline: withStep(view.timeline, 'place', 'active'),
        current: { ...view.current, step: 'place' },
      }
      return say(next, event, m('event.task.carry_started', { look: str(data.to_look) ?? '—' }), 'info', 'tech')
    }
    case 'task.target_checked': {
      const followed = data.followed === true
      const seen = targetOf(data.target, null)
      const next: RunView = {
        ...view,
        targetCheck: { movedMm: num(data.moved_mm), followed },
        target: followed && seen ? seen : view.target,
      }
      return say(next, event, m('event.task.target_checked', { moved: round(num(data.moved_mm)) }), followed ? 'info' : 'warn', 'tech')
    }
    case 'task.target_lost': {
      // The ask card is drawn only when the run ENDS on the question (`finish`); until then the reason rides along.
      const why = str(data.why)
      const next: RunView = { ...view, timeline: withStep(view.timeline, 'place', 'failed'), lostWhy: why }
      const line = why === 'not_seen' || why === 'moved_too_far' || why === 'footprint_changed'
        ? m(`event.task.target_lost.${why}`)
        : m('event.task.target_lost')
      return say(next, event, line, 'warn')
    }
    case 'task.drop_planned': {
      const kind = str(data.kind)
      const line = kind === 'pose'
        ? m('event.task.drop_planned.pose', { hang: round(num(data.hang_mm)) })
        : kind === 'camera'
          ? m('event.task.drop_planned.camera', { air: round(num(data.air_mm)) })
          : m('event.task.drop_planned')
      return say(view, event, line, 'info', 'tech')
    }
    case 'task.place_started': {
      const raw = str(data.place)
      const fallback = raw && raw.includes(':') ? raw.slice(raw.indexOf(':') + 1) : raw
      const next: RunView = {
        ...moving(view),
        timeline: withStep(view.timeline, 'place', 'active'),
        current: { ...view.current, step: 'place' },
      }
      return say(next, event, m('event.task.place_started', { where: placeWords(plan, fallback) }))
    }
    case 'task.placed': {
      let next: RunView = { ...view, timeline: withStep(view.timeline, 'place', 'done'), holding: false }
      next = updatePart(next, view.current.part, (p) => ({ ...p, placed: true }))
      next = { ...next, stats: computeStats(next) }
      const line = data.line_out_refused === true
        ? m('event.task.placed.line_out_refused')
        : data.no_sensor === true
          ? m('event.task.placed.no_sensor')
          : m('event.task.placed')
      return say(next, event, line, data.line_out_refused === true ? 'warn' : 'ok')
    }
    case 'task.place_failed':
      return say({ ...view, timeline: withStep(view.timeline, 'place', 'failed') }, event, m('event.task.place_failed'), 'error')
    case 'task.put_back': {
      const released = str(data.outcome) === 'executed'
      return say({ ...view, holding: released ? false : view.holding }, event, m('event.task.put_back'))
    }
    case 'task.return_started': {
      const next: RunView = {
        ...moving(view),
        timeline: withStep(view.timeline, 'return', 'active'),
        current: { ...view.current, step: 'return' },
      }
      // The restart's first motion, and the final return after the last part, are read by everyone.
      const standalone = view.current.part === null || view.timeline.every((s) => s.state !== 'active')
      const part = standaloneReturn(view) ? null : view.current.part
      return say(next, event, m('event.task.return_started', { to: returnWords(plan, str(data.to)) }), 'info', standalone ? 'demo' : 'tech', undefined, part)
    }
    case 'task.returned':
      return say(
        { ...view, timeline: withStep(view.timeline, 'return', 'done') },
        event,
        m('event.task.returned', { to: returnWords(plan, str(data.to)) }),
        'info',
        'tech',
        undefined,
        standaloneReturn(view) ? null : view.current.part,
      )
    case 'task.return_failed':
      return say({ ...view, timeline: withStep(view.timeline, 'return', 'failed') }, event, m('event.task.return_failed'), 'error')
    case 'task.part_finished': {
      const part = num(data.part) ?? view.current.part
      const placed = data.placed === true
      const durationS = num(data.duration_s)
      let next = updatePart(view, part, (p) => ({ ...p, placed, durationS }))
      next = { ...next, stats: computeStats(next) }
      const line = placed
        ? m('event.task.part_finished.placed', { part: part ?? '—', seconds: round(durationS) })
        : m('event.task.part_finished.not_placed', { part: part ?? '—' })
      return say(next, event, line, placed ? 'ok' : 'warn')
    }
    default:
      return view
  }
}

function targetOf(value: unknown, look: string | null): TargetView | null {
  const t = record(value)
  const label = str(t.label)
  if (label === null) return null
  return {
    label,
    score: num(t.score),
    centreMm: numbers(t.centre_mm),
    rimMm: num(t.rim_mm),
    footprintMm: numbers(t.footprint_mm),
    openingMm: numbers(t.opening_mm),
    look: str(t.look) ?? look,
    seenAt: num(t.seen_at),
    overlay: str(t.overlay),
  }
}

/**
 * The pick loop's own outcome as words: it sends `str()` of its enum (`PickOutcome.NO_PERCEPTION`), the class name
 * included; the tech line says `no perception`. An outcome of the grasp report (`no_target`) is kept as it is.
 */
function loopOutcome(outcome: string | null): string | null {
  if (outcome === null) return null
  const bare = outcome.replace(/^[A-Za-z]*Outcome\./, '')
  return bare === outcome ? outcome : bare.toLowerCase()
}

function pickStage(view: RunView, event: RunEvent): RunView {
  const data = record(event.data)
  switch (event.type) {
    case 'pick.pick_started': {
      const next: RunView = {
        ...moving(view),
        timeline: withStep(view.timeline, 'look', 'active'),
        current: { ...view.current, attemptTotal: num(data.attempt_total), lookIndex: 0, step: 'look' },
      }
      return say(next, event, m('event.pick.pick_started', { total: num(data.attempt_total) }), 'info', 'tech')
    }
    case 'pick.attempt_started': {
      const attempt = (num(data.attempt) ?? 0) + 1
      const total = num(data.attempt_total) ?? view.current.attemptTotal
      let timeline = view.timeline
      // A second attempt looks again from the start: what the first one saw is not this attempt's.
      if (attempt > 1) for (const id of ['look', 'detect', 'grasp'] as const) timeline = withStep(timeline, id, 'idle', null)
      const next: RunView = { ...view, timeline, current: { ...view.current, attempt, attemptTotal: total, lookIndex: 0, look: null } }
      return say(next, event, m('event.pick.attempt_started', { attempt, total }), 'info', 'tech')
    }
    case 'pick.perceived': {
      const look = str(data.look)
      const lookIndex = view.current.lookIndex + 1
      let timeline = withStep(view.timeline, 'look', 'active', m('step.lookN', { n: lookIndex }))
      if (stepState(timeline, 'detect') === 'idle') timeline = withStep(timeline, 'detect', 'active')
      const next: RunView = { ...moving(view), timeline, current: { ...view.current, look, lookIndex, step: 'look' } }
      const count = num(data.segmentation_count) ?? 0
      const line = look ? m('event.pick.perceived', { look, count }) : m('event.pick.perceived.fixed', { count })
      return say(next, event, line, 'info', 'tech')
    }
    case 'pick.ranked': {
      const timeline = withStep(withStep(view.timeline, 'look', 'done'), 'detect', 'done')
      const next: RunView = { ...view, timeline, current: { ...view.current, step: 'detect' } }
      return say(next, event, m('event.pick.ranked', { count: num(data.candidate_count) ?? 0, score: num(data.score) }), 'info', 'tech')
    }
    case 'pick.no_candidate': {
      const next: RunView = { ...view, timeline: withStep(view.timeline, 'detect', 'warn') }
      const reasons = strings(data.reasons).join(', ').replace(/_/g, ' ') || '—'
      return say(next, event, m('event.pick.no_candidate', { reasons }), 'warn', 'tech')
    }
    case 'pick.executing': {
      const attempt = view.current.attempt
      const total = view.current.attemptTotal
      const note = attempt !== null && total !== null ? m('step.attempt', { n: attempt, total }) : null
      let timeline = view.timeline
      for (const id of ['look', 'detect'] as const) if (stepState(timeline, id) === 'active') timeline = withStep(timeline, id, 'done')
      let next: RunView = { ...view, timeline: withStep(timeline, 'grasp', 'active', note), current: { ...view.current, step: 'grasp' } }
      const overlay = str(data.overlay)
      if (overlay) {
        next = pin(next, { kind: 'grasp', url: overlay, at: event.ts, part: view.current.part, pick: view.picks.length + 1, look: view.current.look })
        next = updatePart(next, view.current.part, (p) => ({ ...p, overlay }))
      }
      const at = numbers(data.position_mm) ?? []
      return say(next, event, m('event.pick.executing', { x: round(at[0] ?? null), y: round(at[1] ?? null), z: round(at[2] ?? null) }), 'info', 'tech')
    }
    case 'pick.attempt_finished': {
      const action = str(data.action)
      if (action === 'push') {
        // The pick loop reports EVERY push attempt with action "push": the push that ran (`push: "pushed"`; a
        // hand-written log says it as the outcome), one that stopped where the arm stands (`unsafe_recovery_refused`,
        // `controller_not_operational`), and one refused before anything moved (`refused_*`). Only the first pushed
        // a part and looked again; the others end the pick, and a person decides.
        const code = str(data.push)
        if (code === 'pushed' || (code === null && str(data.outcome) === 'pushed')) {
          const mm = num(data.push_mm)
          let next: RunView = { ...view, current: { ...view.current, pushMm: (view.current.pushMm ?? 0) + (mm ?? 0) } }
          next = updatePart(next, view.current.part, (p) => ({ ...p, pushedMm: p.pushedMm + (mm ?? 0) }))
          return say(next, event, m('event.pick.attempt_finished.push', { mm: round(mm) }), 'info', 'demo')
        }
        const refused = code !== null && code.startsWith('refused_')
        return say(view, event, m(refused ? 'event.pick.attempt_finished.push_refused' : 'event.pick.attempt_finished.push_stopped'), 'warn', 'demo')
      }
      if (action === 'clear_blocker') {
        // The pick loop's "clear the blocker" (the owner, 2026-10-02): a neighbour in the way of every grasp was gripped,
        // set down elsewhere and the arm looked again (`blocker: "set_aside"`), or the clearing stopped once something
        // moved: the arm stands where it stopped, the blocker may be in the hand, and a person decides.
        if (str(data.blocker) === 'set_aside') return say(view, event, m('event.pick.attempt_finished.blocker'), 'info', 'demo')
        return say(view, event, m('event.pick.attempt_finished.blocker_stopped'), 'warn', 'demo')
      }
      let next = view
      if (action === 'look_refused') next = { ...view, timeline: withStep(view.timeline, 'look', 'failed') }
      return say(next, event, m('event.pick.attempt_finished', { action: (action ?? '—').replace(/_/g, ' ') }), 'info', 'tech')
    }
    case 'pick.pick_finished':
      return say(view, event, m('event.pick.pick_finished', { outcome: outcomeMsg(loopOutcome(str(data.outcome))) }), 'info', 'tech')
    case 'pick.cancelled':
      return say(view, event, m('event.pick.cancelled'), 'info', 'tech')
    default:
      return view
  }
}

/**
 * The next view after one event (or a fetched run record). Pure: the same view and event always give the same result.
 *
 * An event of another run starts a new view only when it is that run's `run_started`; anything else of another run is
 * ignored, and an event this view already applied (by seq) is ignored too.
 */
export function reduce(view: RunView, action: RunAction): RunView {
  if (isSnapshot(action)) {
    const base = view.runId !== null && action.run.id !== view.runId ? fresh(action.run.id) : view
    return applyRecord(base.runId === null ? fresh(action.run.id) : base, action.run)
  }
  if (isLost(action)) return lose(view, action.runId)
  const event = action
  if (event.run_id === 'cell') return view

  if (event.type === 'gap') {
    if (view.runId !== null && event.run_id !== view.runId) return view
    const dropped = num(record(event.data).dropped) ?? 0
    const next: RunView = { ...view, runId: view.runId ?? event.run_id, gaps: { count: view.gaps.count + 1, dropped: view.gaps.dropped + dropped } }
    return say(next, event, m('chat.gap', { dropped }), 'warn', 'demo', `${event.run_id}:gap:${next.gaps.count}`)
  }

  if (event.type === 'run_started') {
    if (view.runId === event.run_id && event.seq <= view.lastSeq) return view
    return { ...startRun(view, event), lastSeq: event.seq }
  }
  if (view.runId !== null && event.run_id !== view.runId) return view
  if (view.runId === event.run_id && event.seq <= view.lastSeq) return view

  const base: RunView = { ...view, runId: view.runId ?? event.run_id, lastSeq: Math.max(view.lastSeq, event.seq) }
  const data = record(event.data)
  switch (event.type) {
    case 'run_countdown': {
      const seconds = num(data.seconds_left)
      const next: RunView = {
        ...base,
        phase: 'countdown',
        countdown: { state: 'active', secondsLeft: seconds, because: str(data.because) },
      }
      return say(next, event, m('event.run_countdown', { seconds }), 'warn', 'demo', `${event.run_id}:countdown`)
    }
    case 'run_stop_requested': {
      const scope = str(data.scope)
      const stopping = scope === 'after_part' || scope === 'between_attempts'
      const next: RunView = {
        ...base,
        stopRequested: base.stopRequested || scope === 'after_part',
        // A Home run's stop acts before its one move is sent: no part, so it never reads "stopping after the part".
        stopScope: stopping || scope === 'before_motion' ? scope : base.stopScope,
        phase: stopping && (base.phase === 'running' || base.phase === 'countdown') ? 'stopping' : base.phase,
      }
      // A Home run's stop is said as what it is: the move is not sent, and one already sent runs to its end.
      const line: Msg<ChatKey> = scope === 'after_part' || scope === 'between_attempts' || scope === 'disconnect'
        ? m(`event.run_stop_requested.${scope}`)
        : scope === 'before_motion'
          ? { key: 'ck.stop.beforeMotion' }
          : m('event.run_stop_requested')
      return say(next, event, line, scope === 'disconnect' ? 'warn' : 'info')
    }
    case 'run_halt_requested': {
      const next: RunView = {
        ...base,
        phase: 'halting',
        halt: {
          state: base.halt.state === 'confirmed' ? 'confirmed' : 'requested',
          requestedAt: num(data.requested_at) ?? event.ts,
          inMotion: data.in_motion === true,
          braking: data.braking === true,
          reason: str(data.reason) ?? '',
        },
      }
      return say(next, event, m('event.run_halt_requested'), 'error')
    }
    case 'run_error': {
      const { stopCode, stopClass } = stopOf(data.stop_code)
      const error = str(data.error) ?? base.error
      let next: RunView = { ...base, error, stopCode: stopCode || base.stopCode, stopClass: stopClass || base.stopClass }
      if (stopCode === 'halted') next = { ...next, halt: { ...next.halt, state: 'confirmed' } }
      if (stopClass === 'problem' && stopCode) {
        next = { ...next, stopCard: stopCardOf(next, stopCode, event.ts, error, next.holding) }
      }
      // A problem stop is said once to everyone, by the run's end ("Problem: Angehalten.") and its stop card: this
      // line is the tech view's. A run_error that names no problem (an older server) is still read by everyone.
      const level: Level = stopClass === 'problem' ? 'tech' : 'demo'
      return say(next, event, m('event.run_error', { title: stopCode ? stopMsg(stopCode) : error || '—' }), 'error', level)
    }
    case 'run_finished':
      return finish(base, event, data as unknown as RunOut)
    case 'pick_result':
      return pickResult(base, event)
    // A Home run has one step, the return: its timeline says where that one move stands.
    case 'home.started': {
      const next: RunView = {
        ...moving({ ...base, to: str(data.to) ?? base.to }),
        timeline: withStep(base.timeline, 'return', 'active'),
        current: { ...base.current, step: 'return' },
      }
      return say(next, event, m('event.home.started', { to: returnWords(null, str(data.to) ?? base.to) }), 'info', 'tech')
    }
    case 'home.arrived':
      return say({ ...base, timeline: withStep(base.timeline, 'return', 'done') }, event, m('event.home.arrived', { to: returnWords(null, str(data.to) ?? base.to) }), 'info', 'tech')
    case 'home.refused':
      return say({ ...base, timeline: withStep(base.timeline, 'return', 'failed') }, event, m('event.home.refused'), 'error')
    // A wave at a greeting: its first swing is the run's first motion, and a refused swing is said to everyone.
    case 'wave.started':
      return say(moving(base), event, m('event.wave.started', { swings: num(data.swings) ?? 2, deg: num(data.swing_deg) ?? 15 }), 'info', 'tech')
    case 'wave.done':
      return say(base, event, m('event.wave.done'), 'info', 'tech')
    case 'wave.refused':
      return say(base, event, m('event.wave.refused'), 'error')
    case 'planner.starting':
      return say(base, event, m('event.planner.starting'), 'info', 'tech')
    case 'planner.ready':
      return say(base, event, m('event.planner.ready'), 'ok', 'tech')
    case 'planner.failed':
      return say(base, event, m('event.planner.failed'), 'error')
    default:
      break
  }
  if (event.type.startsWith('pick.')) return pickStage(base, event)
  if (event.type.startsWith('teach.')) return teachEvent(base, event)
  if (event.type.startsWith('task.')) return taskEvent(base, event)
  // An event type this console does not know yet: kept for the tech view, never dropped.
  return say(base, event, m('common.raw', { text: event.human || event.type }), 'info', 'tech')
}

/**
 * The followed run is gone from the server (`run.lost`): no step is in progress any more as far as the console can
 * tell, and one line says so. What the run said before stays on screen; another run's loss, or the loss of a run
 * that already ended, changes nothing, and so does a second report of the same loss.
 */
function lose(view: RunView, runId: string): RunView {
  if (view.runId !== runId || hasEnded(view.phase)) return view
  const timeline = view.timeline.map((s) => (s.state === 'active' ? { ...s, state: 'idle' as StepState } : s))
  const countdown = view.countdown.state === 'active' ? { ...view.countdown, state: 'none' as const } : view.countdown
  const line: ChatLine = {
    id: `${runId}:lost`,
    runId,
    seq: view.lastSeq,
    at: view.chat.at(-1)?.at ?? view.startedAt ?? 0,
    type: '',
    msg: m('chat.lost'),
    human: '',
    tone: 'warn',
    level: 'demo',
    part: null,
    data: {},
  }
  return { ...view, phase: 'lost', timeline, countdown, chat: [...view.chat, line] }
}

/** Replay a list of events (and snapshots) from an empty view, or from `from`. */
export function replay(actions: Iterable<RunAction>, from: RunView = EMPTY_RUN): RunView {
  let view = from
  for (const action of actions) view = reduce(view, action)
  return view
}

/**
 * The run's phase in words. The two stops are kept apart (build plan risk 16): a task stopping after its part says
 * so, a pick run stopping before its next attempt says that, never "after the part".
 */
export function phaseMsg(view: RunView): Msg {
  if (view.phase === 'stopping' && view.stopScope === 'between_attempts') return m('run.phase.stopping.between_attempts')
  return m(`run.phase.${view.phase}`)
}

/**
 * How many looks an attempt is planned to take, where that is known: one with multi-view off (Q11), else the
 * configured looks of a wrist camera (`facts.looks`). `null` where nothing says it (a fixed camera, facts not read).
 */
export function plannedLooks(
  plan: TaskPlanOut | null,
  facts: Pick<CellFactsOut, 'wrist_camera' | 'looks'> | null,
): number | null {
  if (plan?.options?.multi_view === false) return 1
  if (!facts || facts.wrist_camera !== true) return null
  return facts.looks && facts.looks.length > 0 ? facts.looks.length : null
}

/**
 * The look step's note with its total: "Blick 2/3" (build plan 1.4) where `total` is known, "Blick 2" where it is
 * not, and also for a look beyond the total (the one view a pick generates after its configured looks), which is
 * counted but never put over a total it passed. `null` before the attempt's first look.
 */
export function lookNote(view: RunView, total: number | null): Msg | null {
  const n = view.current.lookIndex
  if (n < 1) return null
  return total !== null && n <= total ? m('step.lookOf', { n, total }) : m('step.lookN', { n })
}

/** How long a halt may go unconfirmed while the arm was moving, on an arm that brakes, before the alarm. */
export const HALT_CONFIRM_S = 1.5

/**
 * The halt as the stop buttons show it. `unconfirmed` ("Anhalten nicht bestätigt: Not-Aus drücken") only when the arm
 * brakes a move in flight (`facts.brake.brakes_in_motion`), the halt was pressed while it moved, and no confirmation
 * came within `HALT_CONFIRM_S`. An arm that latches but does not brake stops before its next motion: no false alarm.
 *
 * A confirmation is the run's own (it stopped on `halted`, or ended at all: `reduce`), or the cell's: `cellHalted`
 * (`CellOut.halted`) saying the move in flight was BRAKED (`braked`, or `brake: "braked"`), or that it ran out to its
 * end (`brake: "ran_out"`: nothing after it was sent), for this halt or a later one. The latch alone confirms nothing on
 * an arm that brakes: it is set before the brake.
 *
 * The arm's own `brake: "unconfirmed"` (the brake raised, or the arm was never seen still) is the alarm at once,
 * without waiting out the window, again only where the arm brakes, and WHENEVER it comes in: the brake gives up after
 * 2 s and the halted run ends right after it, so the poll that carries the arm's word often arrives after the run's
 * end has confirmed the halt. The end says the run commands nothing more; the arm says it was not seen to stop. The
 * arm's word wins, for as long as its latch stands (until "Zelle ist frei"), also on a page that has no halt in view.
 */
export function haltState(
  halt: HaltView,
  nowS: number,
  brakesInMotion: boolean,
  cellHalted: HaltStateOut | null = null,
): HaltState {
  const latchOfThisHalt = cellHalted !== null && (halt.requestedAt === null || cellHalted.requested_at + 1 >= halt.requestedAt)
  if (brakesInMotion && latchOfThisHalt && cellHalted.brake === 'unconfirmed') return 'unconfirmed'
  if (halt.state !== 'requested') return halt.state
  const thisHalt = halt.requestedAt !== null && cellHalted !== null && cellHalted.requested_at + 1 >= halt.requestedAt
  if (thisHalt && (cellHalted.braked === true || cellHalted.brake === 'braked' || cellHalted.brake === 'ran_out')) {
    return 'confirmed'
  }
  if (!brakesInMotion || !halt.inMotion || halt.requestedAt === null) return 'requested'
  if (thisHalt && cellHalted.brake === 'unconfirmed') return 'unconfirmed'
  return nowS - halt.requestedAt > HALT_CONFIRM_S ? 'unconfirmed' : 'requested'
}
