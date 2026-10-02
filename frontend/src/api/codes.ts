/**
 * The console's typed codes, as `GET /v1/codes` serves them (`api/codes.py`).
 *
 * The backend's sentences stay English; the console translates CODES. So every translation table is keyed by one of
 * these unions, and a table that misses a member fails `tsc`, not an operator reading an untranslated word.
 *
 * Two checks keep this file honest:
 *
 * * the types are derived from the generated `schema.d.ts` (`CodesOut`), and each list below goes through `every`,
 *   which accepts it only if it names every member of its union and nothing else, so a code added on the server
 *   fails the build here until it is listed;
 * * `tests/test_api_contract.py` reads these lists and compares them with the StrEnums, in order.
 *
 * One code, one meaning: `not_acknowledged` is Connect's 428 ("read the preview first") and nothing else. The stop
 * card's gate is `cell_not_cleared`.
 */

import type { components } from './schema'

type Codes = components['schemas']['CodesOut']

export type RunKind = Codes['run_kinds'][number]
export type StopCode = Codes['stop_codes'][number]
export type StopClass = Codes['stop_classes'][number]
export type EventType = Codes['event_types'][number]
export type RefusalCode = Codes['refusal_codes'][number]
export type LightId = Codes['light_ids'][number]
export type LightState = Codes['light_states'][number]
export type LightCode = Codes['light_codes'][number]
export type BlockerCode = Codes['blocker_codes'][number]
export type JawsStage = Codes['jaws_stages'][number]
export type JawsChoice = Codes['jaws_choices'][number]
export type CommandNote = Codes['command_notes'][number]

/**
 * `every<U>()(list)` accepts `list` only when it names every member of `U`: a missing member turns the argument's
 * type into `{ missing: ... }`, which no array is, and an extra one is not a `U`. Duplicates are the contract test's.
 */
function every<U extends string>() {
  return <const L extends readonly U[]>(
    list: L & ([Exclude<U, L[number]>] extends [never] ? unknown : { missing: Exclude<U, L[number]> }),
  ): L => list
}

export const RUN_KINDS = every<RunKind>()([
  'pick',
  'task',
  'home',
  'teach',
  'planner',
])

/** The kinds that move the arm by themselves; a problem stop of one of them leaves the recovery record. */
export const MOVING_KINDS = ['pick', 'task', 'home'] as const satisfies readonly RunKind[]

export const STOP_CODES = every<StopCode>()([
  'finished',
  'nothing_left',
  'part_limit',
  'taught',
  'planner_ready',
  'stopped_after_part',
  'cancelled',
  'target_not_found',
  'target_lost',
  'target_unreachable',
  'part_does_not_fit',
  'pose_refused',
  'teach_refused',
  'heartbeat_lost',
  'teach_time_limit',
  'teach_not_saved',
  'planner_failed',
  'halted',
  'controller_stopped',
  'hand_needs_person',
  'recovery_needs_person',
  'part_still_held',
  'return_failed',
  'failed_in_a_row',
  'detector_failed',
  'cell_fault',
  'disconnected',
  'software_error',
])

export const STOP_CLASSES = every<StopClass>()([
  'done',
  'operator',
  'ask',
  'teach',
  'planner',
  'problem',
])

/**
 * What the frontend does with a stop code: `done` and `operator` a neutral line and the next-instruction question,
 * `ask` an ask card, `teach` a teach card, `planner` the planner light, `problem` the stop card.
 */
export const STOP_CLASS_OF: Readonly<Record<StopCode, StopClass>> = {
  finished: 'done',
  nothing_left: 'done',
  part_limit: 'done',
  taught: 'done',
  planner_ready: 'done',
  stopped_after_part: 'operator',
  cancelled: 'operator',
  target_not_found: 'ask',
  target_lost: 'ask',
  target_unreachable: 'ask',
  part_does_not_fit: 'ask',
  pose_refused: 'ask',
  teach_refused: 'ask',
  heartbeat_lost: 'teach',
  teach_time_limit: 'teach',
  teach_not_saved: 'teach',
  planner_failed: 'planner',
  halted: 'problem',
  controller_stopped: 'problem',
  hand_needs_person: 'problem',
  recovery_needs_person: 'problem',
  part_still_held: 'problem',
  return_failed: 'problem',
  failed_in_a_row: 'problem',
  detector_failed: 'problem',
  cell_fault: 'problem',
  disconnected: 'problem',
  software_error: 'problem',
}

export const EVENT_TYPES = every<EventType>()([
  'run_started',
  'run_countdown',
  'run_stop_requested',
  'run_halt_requested',
  'run_error',
  'run_finished',
  'pick.pick_started',
  'pick.attempt_started',
  'pick.perceived',
  'pick.ranked',
  'pick.no_candidate',
  'pick.executing',
  'pick.attempt_finished',
  'pick.pick_finished',
  'pick.cancelled',
  'pick_result',
  'task.pose_screened',
  'task.survey_started',
  'task.target_found',
  'task.target_missing',
  'task.part_started',
  'task.nothing_found',
  'task.carry_started',
  'task.target_checked',
  'task.target_lost',
  'task.drop_planned',
  'task.place_started',
  'task.placed',
  'task.place_failed',
  'task.put_back',
  'task.return_started',
  'task.returned',
  'task.return_failed',
  'task.part_finished',
  'home.started',
  'home.arrived',
  'home.refused',
  'teach.free',
  'teach.say',
  'teach.outside',
  'teach.inside',
  'teach.time_warning',
  'teach.holding_when_still',
  'teach.holding',
  'teach.screening',
  'teach.saved',
  'teach.refused',
  'teach.not_saved',
  'planner.starting',
  'planner.ready',
  'planner.failed',
  'cell.jaws_question',
  'cell.jaws_answered',
  'cell.jaws_ended',
  'cell.halted',
  'cell.recovery',
  'cell.recovery_ended',
  'cell.acknowledged',
  'cell.planner',
])

export const REFUSAL_CODES = every<RefusalCode>()([
  'bad_request',
  'no_robot_configured',
  'no_such_run',
  'not_built_yet',
  'not_built',
  'not_acknowledged',
  'stale_token',
  'cell_busy',
  'no_real_gripper',
  'driver_refused',
  'wrong_state',
  'build_refused',
  'jaws_seam_missing',
  'not_connected',
  'run_active',
  'halted',
  'controller_stopped',
  'cell_not_cleared',
  'restart_required',
  'needs_person',
  'jaws_question_pending',
  'part_still_held',
  'jaws_not_confirmed',
  'route_refused',
  'carried_part_not_modelled',
  'camera_target_unavailable',
  'object_required',
  'prompt_not_routable',
  'target_not_routable',
  'push_distance_refused',
  'unknown_pose',
  'no_place_declared',
  'closing_axis_refused',
  'not_a_task',
  'not_a_pick',
  'not_restartable',
  'jaws_not_open',
  'no_question',
  'question_changed',
  'choice_not_offered',
  'planner_not_used',
  'no_overlay',
  'no_layer',
  'no_hand_guiding',
  'part_in_hand',
  'planner_not_ready',
  'screen_unavailable',
  'payload_changed',
  'name_taken',
  'invalid_name',
  'invalid_label',
  'wrong_token',
  'not_free',
  'vlm_not_loaded',
  'vlm_unavailable',
  'vlm_model_missing',
  'unknown_key',
  'not_writable',
  'invalid_value',
  'no_target',
  'cell_connected',
  'empty_patch',
  'empty_audio',
  'audio_format_unsupported',
  'audio_undecodable',
  'audio_too_long',
  'speech_model_missing',
  'speech_unavailable',
  'transcription_failed',
  'listen_busy',
  'talk_not_pressed',
  'microphone_ended',
  'nothing_recorded',
  'microphone_unavailable',
  'listen_failed',
])

export const LIGHT_IDS = every<LightId>()([
  'robot',
  'cameras',
  'planner',
  'gripper',
  'carried_part',
  'commands',
])

export const LIGHT_STATES = every<LightState>()([
  'ok',
  'wait',
  'blocked',
  'info',
])

export const LIGHT_CODES = every<LightCode>()([
  'connected',
  'not_built',
  'not_connected',
  'halted',
  'controller_stopped',
  'controller_unreadable',
  'live',
  'rehearsal',
  'no_frame',
  'none',
  'ready',
  'starting',
  'off',
  'failed',
  'unplanned',
  'route_refused',
  'open_confirmed',
  'jaws_unknown',
  'jaws_closed',
  'question_pending',
  'modelled',
  'not_modelled',
  'not_applicable',
  'idle',
  'loading',
  'missing',
  'not_configured',
])

export const BLOCKER_CODES = every<BlockerCode>()([
  'run_active',
  'cell_not_cleared',
  'restart_required',
  'needs_person',
  'part_still_held',
  'jaws_question_pending',
])

export const JAWS_STAGES = every<JawsStage>()([
  'where',
  'open_now',
])

/** The jaws question has no default: there is no "open" a click can fall back to. */
export const JAWS_CHOICES = every<JawsChoice>()([
  'open',
  'closed',
  'open_now',
  'abort',
])

export const COMMAND_NOTES = every<CommandNote>()([
  'object_not_in_sentence',
  'place_not_in_sentence',
  'pose_unknown',
  'count_not_supported',
  'retried',
])

/** What models the part the gripper carries now (`CellOut.payload_model`). */
export type PayloadModel = components['schemas']['CellOut']['payload_model']

/**
 * The `payload_model` values that model no part now (`api.schemas.NO_PART_MODELLED`). It is one of the reads of the
 * stop card's "no part held, now" gate (`cockpit/recovery.ts` `partHeld`), whose whole rule is the server's
 * `part_still_held` (the ready bar's blocker; it also counts a stop record's belief for a hand that measures nothing).
 * `not_applicable` (an arm that models no part at all: the dummy, sim) is in it, so the gate opens on the rehearsal
 * cell; `unknown` (a read that failed) is not, so a part nobody can rule out keeps the gate shut.
 */
export const NO_PART_MODELLED = ['none', 'not_applicable'] as const satisfies readonly PayloadModel[]

/** Whether `model` models no part now (`NO_PART_MODELLED`). */
export function noPartModelled(model: PayloadModel): boolean {
  return (NO_PART_MODELLED as readonly PayloadModel[]).includes(model)
}

/** The stream of the events that belong to no run (the jaws question, the recovery record, ...). */
export const CELL_STREAM = 'cell'

/** Whether `code` is a stop code of this console, for a payload typed as a plain string. */
export function isStopCode(code: unknown): code is StopCode {
  return typeof code === 'string' && (STOP_CODES as readonly string[]).includes(code)
}
