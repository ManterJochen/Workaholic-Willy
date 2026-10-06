/**
 * What the audience window says in big words: the step the robot is at, with what, and where it goes; and the counts
 * of the STILL GRINDING card.
 *
 * Pure, from the run model's view (`model/runModel`, the cockpit's own reducer) and the session's runs, so the
 * projector says what the cockpit shows and never a word of its own about the run.
 */

import type { RunOut, TaskPlanOut } from '../api/client'
import { STOP_CLASS_OF } from '../api/codes'
import { stopMsg } from '../i18n/codes'
import type { MessageKey, Msg } from '../i18n/types'
import { hasEnded, type RunView } from '../model/runModel'
import type { DemoKey } from './i18n'

type Key = DemoKey | MessageKey

export interface Caption {
  /** The verb: "Greift:", "Legt ab:", "Fertig:"; `null` where the title says it all ("Angehalten"). */
  readonly verb: Msg<Key> | null
  /** What it is about: the part as the operator said it, the place, the stop's title; `null` for none. */
  readonly what: string | Msg<Key> | null
  /** One quiet line under it, where the verb alone does not say enough. */
  readonly sub: Msg<Key> | null
  /** `warn` is the hands-off countdown before a motion the robot starts by itself: never the lime of a running step. */
  readonly tone: 'idle' | 'run' | 'done' | 'ask' | 'stop' | 'warn'
}

function m(key: Key, params?: Msg['params']): Msg<Key> {
  return params === undefined ? { key } : { key, params }
}

/** The part as the operator said it ("den grünen Würfel"), else the detector's phrase, else "anything". */
export function objectWords(plan: TaskPlanOut | null, prompt: string | null): string | Msg<Key> {
  if (plan) return plan.object_said || plan.object || m('common.anything')
  return prompt || m('common.anything')
}

/** Where the part goes: a taught pose's label, or the target as it was said. */
export function placeWords(plan: TaskPlanOut | null): string | Msg<Key> | null {
  const place = plan?.place
  if (!place) return null
  if (place.kind === 'pose') return place.pose_label || place.pose || m('common.defaultPlace')
  return place.said || place.phrase || null
}

/** The taught poses' labels by name (`GET /v1/poses`): a Home run's record names its pose only by its YAML name. */
export type PoseLabels = ReadonlyMap<string, string>

const NO_LABELS: PoseLabels = new Map()

/** Where the arm goes back to: Home, or a taught pose by its label (the plan's, else the cell's), never its name. */
function returnWords(plan: TaskPlanOut | null, to: string | null, labels: PoseLabels): string | Msg<Key> {
  const target = to ?? plan?.return_to ?? 'home'
  if (target === 'home') return m('common.home')
  if (plan?.return_label && plan.return_to === target) return plan.return_label
  return labels.get(target) || target
}

/**
 * The caption for the run on screen; `connected` false says the cell is down when no run is shown. `labels` names the
 * taught poses, for a Home run whose record knows only the pose's name.
 */
export function captionOf(view: RunView, connected: boolean, labels: PoseLabels = NO_LABELS): Caption {
  if (!view.runId || view.phase === 'idle') {
    return connected
      ? { verb: m('aud.cap.ready'), what: null, sub: m('aud.cap.ready.sub'), tone: 'idle' }
      : { verb: m('aud.cap.notConnected'), what: null, sub: null, tone: 'idle' }
  }
  if (view.phase === 'lost') return { verb: m('aud.cap.lost'), what: null, sub: null, tone: 'idle' }

  if (hasEnded(view.phase)) {
    const code = view.stopCode || null
    const title = code ? stopMsg(code) : null
    const cls = code ? STOP_CLASS_OF[code] : view.stopClass || null
    if (cls === 'problem') {
      // A halt is its own word; any other problem says that it is one.
      return { verb: code === 'halted' ? null : m('aud.cap.problem'), what: title, sub: null, tone: 'stop' }
    }
    if (cls === 'ask') return { verb: m('aud.cap.ask'), what: title, sub: null, tone: 'ask' }
    if (cls === 'done' || cls === 'operator') {
      const verb = cls === 'done' ? m('aud.cap.done') : m('aud.cap.stopped')
      if (view.kind === 'task') return { verb, what: m('aud.cap.placed', { n: view.stats.placed }), sub: null, tone: 'done' }
      if (view.kind === 'home' && code === 'finished') {
        return { verb: m('aud.cap.arrived'), what: returnWords(view.plan, view.to, labels), sub: null, tone: 'done' }
      }
      return { verb, what: title, sub: null, tone: 'done' }
    }
    return { verb: m('aud.cap.stopped'), what: title, sub: null, tone: 'done' }
  }

  if (view.phase === 'halting') return { verb: m('aud.cap.halting'), what: null, sub: null, tone: 'stop' }
  if (view.countdown.state === 'active') {
    return { verb: m('aud.cap.handsOff'), what: `${view.countdown.secondsLeft ?? 3} s`, sub: null, tone: 'warn' }
  }

  switch (view.kind) {
    case 'home':
      return { verb: m('aud.cap.home'), what: returnWords(view.plan, view.to, labels), sub: null, tone: 'run' }
    case 'teach':
      return { verb: m('aud.cap.teach'), what: view.teach?.label || view.teach?.name || null, sub: m('aud.cap.teach.sub'), tone: 'run' }
    case 'planner':
      return { verb: m('aud.cap.planner'), what: null, sub: m('aud.cap.planner.sub'), tone: 'run' }
    case 'wave':
      return { verb: m('aud.cap.wave'), what: null, sub: null, tone: 'run' }
    default:
      break
  }

  const object = objectWords(view.plan, view.prompt)
  if (view.survey.state === 'active') {
    return { verb: m('aud.cap.survey'), what: view.survey.phrase || placeWords(view.plan), sub: null, tone: 'run' }
  }
  switch (view.current.step) {
    case 'look':
      return { verb: m('aud.cap.look'), what: object, sub: null, tone: 'run' }
    case 'detect':
      return { verb: m('aud.cap.detect'), what: object, sub: null, tone: 'run' }
    case 'grasp':
      return { verb: m('aud.cap.grasp'), what: object, sub: null, tone: 'run' }
    case 'place':
      return { verb: m('aud.cap.place'), what: placeWords(view.plan) ?? object, sub: null, tone: 'run' }
    case 'return':
      return { verb: m('aud.cap.return'), what: returnWords(view.plan, view.to, labels), sub: null, tone: 'run' }
    default:
      return { verb: m('aud.cap.start'), what: object, sub: null, tone: 'run' }
  }
}

/** The local calendar day of a moment, as `YYYY-M-D`. */
function day(unixSeconds: number): string {
  const d = new Date(unixSeconds * 1000)
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
}

export interface Grind {
  /** Parts placed by today's tasks (the one on screen counted live). */
  readonly partsToday: number
  /** Seconds today's tasks, picks and moves ran. */
  readonly workS: number
}

const WORK = new Set(['task', 'pick', 'home'])

/** The STILL GRINDING counts: today's placed parts and working time, from the session's runs and the run on screen. */
export function grindOf(runs: readonly RunOut[], view: RunView, nowS: number): Grind {
  const today = day(nowS)
  let partsToday = 0
  let workS = 0
  for (const run of runs) {
    if (day(run.started_at) !== today || !WORK.has(run.kind)) continue
    const live = view.runId === run.id
    if (run.kind === 'task') partsToday += live ? Math.max(view.stats.placed, run.parts_placed ?? 0) : run.parts_placed ?? 0
    const end = run.finished_at ?? (live && view.finishedAt ? view.finishedAt : nowS)
    workS += Math.max(0, end - run.started_at)
  }
  return { partsToday, workS }
}
