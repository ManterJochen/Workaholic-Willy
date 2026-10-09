/**
 * Every typed code the backend answers with, mapped to the message that says it.
 *
 * The codes come from `GET /v1/codes` through `api/codes.ts`, whose unions are generated from the backend's own
 * Literals. Each key family below is a template over one of those unions, and both language files `satisfies` the
 * whole family set: a stop code added on the server fails `tsc` here until it has a German and an English text, and
 * a text for a code that no longer exists fails it too.
 *
 * A code this console does not know (a newer server, a typo in a test double) is never shown as a raw identifier
 * in place of a sentence and never as silence: `refusalMsg` falls back to "Abgelehnt (<code>)", and the screens put
 * the backend's own sentence under it in the tech view.
 *
 * Several messages in a row (a sort's rules, the targets a survey looks for) stay one message until they are said
 * (`listMsg`, `quotedMsg`), so the run model and the card name them without knowing the reader's language.
 */

import {
  REFUSAL_CODES,
  type BlockerCode,
  type CommandNote,
  type EventType,
  type JawsChoice,
  type JawsStage,
  type LightCode,
  type LightId,
  type LightState,
  type RefusalCode,
  type RunKind,
  type StopClass,
  type StopCode,
} from '../api/codes'
import type { Msg, ParamValue } from './types'

/**
 * The outcomes a pick reports (`AutonomousGraspOutcome`, `src/robot/execution/autonomous_grasp/report.py`), plus
 * `refinement_diverged`, which records logged before 2026-09-29 still carry. Not a `/v1/codes` list: an outcome
 * outside it is shown as its own words, never dropped.
 */
export const OUTCOMES = [
  'succeeded',
  'no_target',
  'no_valid_grasp',
  'execution_failed',
  'verification_failed',
  'recovery_exhausted',
  'unsafe_recovery_refused',
  'missing_camera_frame',
  'mode_not_available',
  'decision_fail_closed',
  'decision_recover_pending',
  'uncertainty_fail_closed',
  'drift_blocked_auto',
  'ood_blocked_auto',
  'cancelled',
  'refinement_diverged',
] as const
export type Outcome = (typeof OUTCOMES)[number]

/**
 * The lines an event says in more than one way, by what its data carries: a task that knows how many parts it has,
 * a sort, a push, a found-nothing pick, a place found nowhere, the end of a run by its class and kind. Each is
 * `event.<variant>`.
 */
export const EVENT_VARIANTS = [
  'run_started.task',
  'run_started.sort',
  'run_started.restart',
  'run_started.pick',
  'run_started.home',
  'run_started.teach',
  'run_started.planner',
  'run_started.wave',
  'run_stop_requested.after_part',
  'run_stop_requested.between_attempts',
  'run_stop_requested.disconnect',
  'run_finished.task',
  'run_finished.nothing_left',
  'run_finished.part_limit',
  'run_finished.stopped_after_part',
  'run_finished.cancelled',
  'run_finished.home',
  'run_finished.wave',
  'run_finished.pick',
  'run_finished.taught',
  'run_finished.planner_ready',
  'run_finished.ask',
  'run_finished.teach',
  'run_finished.planner',
  'run_finished.problem',
  'pick.perceived.fixed',
  'pick.perceived.followed',
  'pick.attempt_finished.push',
  'pick.attempt_finished.push_stopped',
  'pick.attempt_finished.push_refused',
  'pick.attempt_finished.blocker',
  'pick.attempt_finished.blocker_stopped',
  'pick_result.succeeded',
  'pick_result.fused',
  'pick_result.failed',
  'pick_result.nothing',
  'task.part_started.of',
  'task.nothing_found.excluded',
  'task.carry_started.over_the_rim',
  'task.target_lost.not_seen',
  'task.target_lost.moved_too_far',
  'task.target_lost.footprint_changed',
  'task.drop_planned.pose',
  'task.drop_planned.camera',
  'task.drop_planned.below',
  'task.placed.no_sensor',
  'task.placed.line_out_refused',
  'task.placed.over_the_rim',
  'task.part_finished.placed',
  'task.part_finished.not_placed',
  'task.target_relocated.nowhere',
  'task.unsorted.unnamed',
  'cell.jaws_question.open_now',
  'cell.jaws_ended.open',
  'cell.jaws_ended.refused',
  'cell.jaws_ended.no_answer',
  'cell.jaws_ended.cancelled',
] as const
export type EventVariant = (typeof EVENT_VARIANTS)[number]

/** Every key of the code catalogs. Both `codes.de.ts` and `codes.en.ts` name each of these exactly once. */
export type CodeKey =
  | `stop.${StopCode}`
  | `stopSay.${StopCode}`
  | `stopClass.${StopClass}`
  | `event.${EventType}`
  | `event.${EventVariant}`
  | `refusal.${RefusalCode}`
  | `light.${LightCode}`
  | `lightId.${LightId}`
  | `lightState.${LightState}`
  | `blocker.${BlockerCode}`
  | `note.${CommandNote}`
  | `runKind.${RunKind}`
  | `jawsStage.${JawsStage}`
  | `jawsChoice.${JawsChoice}`
  | `outcome.${Outcome}`

/** A stop code's short title: "Angehalten", "Nichts mehr da". */
export function stopMsg(code: StopCode): Msg {
  return { key: `stop.${code}` }
}

/** A stop code's sentence: what happened, and where the arm stands. */
export function stopSayMsg(code: StopCode): Msg {
  return { key: `stopSay.${code}` }
}

export function stopClassMsg(stopClass: StopClass): Msg {
  return { key: `stopClass.${stopClass}` }
}

export function runKindMsg(kind: RunKind): Msg {
  return { key: `runKind.${kind}` }
}

export function lightMsg(code: LightCode): Msg {
  return { key: `light.${code}` }
}

export function lightIdMsg(id: LightId): Msg {
  return { key: `lightId.${id}` }
}

export function lightStateMsg(state: LightState): Msg {
  return { key: `lightState.${state}` }
}

export function blockerMsg(code: BlockerCode): Msg {
  return { key: `blocker.${code}` }
}

export function noteMsg(note: CommandNote): Msg {
  return { key: `note.${note}` }
}

export function jawsStageMsg(stage: JawsStage): Msg {
  return { key: `jawsStage.${stage}` }
}

export function jawsChoiceMsg(choice: JawsChoice): Msg {
  return { key: `jawsChoice.${choice}` }
}

export function isRefusalCode(code: unknown): code is RefusalCode {
  return typeof code === 'string' && (REFUSAL_CODES as readonly string[]).includes(code)
}

/**
 * What a refusal says, by its code. An unknown code is "Abgelehnt (<code>)": never the bare identifier, never
 * nothing. `http_404`-style codes (`ApiError` for a body that was not the envelope) count as unknown. `detail`, the
 * envelope's machine half, fills the words that name a number (`text_too_long`: "höchstens {max} Zeichen"); only
 * its plain numbers and strings are taken.
 */
export function refusalMsg(code: string, detail?: Readonly<Record<string, unknown>>): Msg {
  if (!isRefusalCode(code)) return { key: 'common.refused', params: { code } }
  const params: Record<string, string | number> = {}
  for (const [name, value] of Object.entries(detail ?? {})) {
    if (typeof value === 'number' || typeof value === 'string') params[name] = value
  }
  return Object.keys(params).length > 0 ? { key: `refusal.${code}`, params } : { key: `refusal.${code}` }
}

/**
 * Where a hand's output is, as the pendant names it: the server says `tool output 0`; the console says `Tool-DO0`
 * (German) or `tool DO0` (English). Anything it does not recognise is shown as the server said it.
 */
export function whereMsg(where: string | null | undefined): Msg | string {
  const text = (where ?? '').trim()
  const tool = /^tool (?:digital )?output (\d+)$/i.exec(text)
  if (tool) return { key: 'hand.toolOutput', params: { pin: tool[1] } }
  const standard = /^(?:standard |configurable )?(?:digital )?output (\d+)$/i.exec(text)
  if (standard) return { key: 'hand.output', params: { pin: standard[1] } }
  return text || '—'
}

/**
 * Items in a row as one message, so a list of messages stays a message until it is said: `list.rules` joins a sort's
 * rules (" · "), `list.words` words in a sentence (", "). Each item may be a message itself. One item is itself, none
 * a dash.
 */
export function listMsg(items: readonly ParamValue[], join: 'list.rules' | 'list.words'): ParamValue {
  if (items.length === 0) return null
  if (items.length === 1) return items[0]
  return { key: join, params: { first: items[0], rest: listMsg(items.slice(1), join) } }
}

/** Words quoted in the reader's language: „in die blaue“, “in die blaue”. */
export function quotedMsg(text: string): Msg {
  return { key: 'list.quoted', params: { text } }
}

export function isOutcome(outcome: unknown): outcome is Outcome {
  return typeof outcome === 'string' && (OUTCOMES as readonly string[]).includes(outcome)
}

/** A pick outcome in words; one this console does not know is shown as its own words, underscores spaced. */
export function outcomeMsg(outcome: string | null | undefined): Msg {
  if (isOutcome(outcome)) return { key: `outcome.${outcome}` }
  return { key: 'common.raw', params: { text: String(outcome ?? '—').replace(/_/g, ' ') } }
}
