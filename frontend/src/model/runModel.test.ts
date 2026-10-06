/**
 * The run model replays what the server said into what the cockpit draws, and nothing else.
 *
 * The fixtures are the console's event logs (`frontend/src/test/fixtures/*.json`), captured by
 * `scripts/console/capture_event_log.py` but for `teach_one_pose`, written by hand. Two kinds of assertion live here:
 *
 * * for EVERY fixture and every run in it, the replayed view agrees with the run record the server keeps
 *   (`runs[id]`, what `GET /v1/runs/{id}` answers): the same stop code, class, state, parts placed and held part. A
 *   reducer that drifted from the record would draw a run the server does not remember;
 * * for each scenario, the moments a person reads: the timeline, the countdown, the stop card, the ask card, the
 *   halt state, the statistics, and the gap.
 *
 * No assertion names a run id or a seq: a run is chosen by its record (`runOf`), a moment by the event that makes it
 * (`upTo`), because the captured logs draw their run ids at random. What a scenario CONTAINS (two parts, a push in
 * the second) is the scenario; a regenerated log that tells another story fails here on purpose.
 */

import { describe, expect, it } from 'vitest'

import type { RunEvent } from '../api/events'
import { COCKPIT } from '../cockpit/i18n'
import { translate } from '../i18n'
import backenLeerJson from '../test/fixtures/console_dummy_halted_backen_leer.json'
import homeJson from '../test/fixtures/console_dummy_home_counts_down.json'
import haltedRestartJson from '../test/fixtures/task_halted_then_restart.json'
import targetLostJson from '../test/fixtures/task_camera_target_lost.json'
import twoPartsJson from '../test/fixtures/task_two_parts_nothing_left.json'
import teachJson from '../test/fixtures/teach_one_pose.json'
import { eventsOf, isType, runOf, upTo, type Fixture } from '../test/render'
import {
  EMPTY_RUN,
  STEPS,
  haltState,
  lookNote,
  phaseMsg,
  plannedLooks,
  reduce,
  replay,
  type ChatLine,
  type RunView,
  type StepId,
} from './runModel'

const twoParts = twoPartsJson as unknown as Fixture
const targetLost = targetLostJson as unknown as Fixture
const haltedRestart = haltedRestartJson as unknown as Fixture
const teach = teachJson as unknown as Fixture

const ALL = import.meta.glob<{ default: unknown }>('../test/fixtures/*.json', { eager: true })

function step(view: RunView, id: StepId) {
  const found = view.timeline.find((s) => s.id === id)
  if (!found) throw new Error(`no step ${id}`)
  return found.state
}

function keys(view: RunView): string[] {
  return view.chat.map((line) => line.msg.key)
}

function lineOf(view: RunView, key: string): ChatLine | undefined {
  return view.chat.find((line) => line.msg.key === key)
}

function say(line: ChatLine | undefined, lang: 'de' | 'en' = 'de'): string {
  if (!line) throw new Error('no line')
  return translate(lang, line.msg.key as never, line.msg.params)
}

describe('every fixture', () => {
  const fixtures = Object.entries(ALL).map(([path, mod]) => [path, mod.default as Fixture] as const)

  it('is found by the glob, so a fixture the run track adds is replayed too', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(5)
  })

  it('replays each of its runs into the view the server\'s own record describes', () => {
    for (const [path, fixture] of fixtures) {
      for (const [runId, record] of Object.entries(fixture.runs)) {
        const view = replay(eventsOf(fixture, runId))
        const where = `${path} ${runId}`
        expect(view.runId, where).toBe(runId)
        expect(view.kind, where).toBe(record.kind)
        expect(view.stopCode, where).toBe(record.stop_code)
        expect(view.stopClass, where).toBe(record.stop_class)
        expect(view.phase, where).toBe(record.state)
        expect(view.holding, where).toBe(record.holding)
        expect(view.stats.placed, where).toBe(record.parts_placed)
        expect(view.restartOf, where).toBe(record.restart_of ?? null)
        // A stop card only for a problem, an ask card only for a question: never both, never the wrong one.
        expect(view.stopCard !== null, where).toBe(record.stop_class === 'problem')
        expect(view.askCard !== null, where).toBe(record.stop_class === 'ask')
        // A run that is over never alarms "press the e-stop", whatever it ended on.
        expect(haltState(view.halt, Number.MAX_SAFE_INTEGER, true), where).not.toBe('unconfirmed')
      }
    }
  })

  it('keeps the timeline at five steps in the plan\'s order, and every chat line unique', () => {
    expect(STEPS).toEqual(['look', 'detect', 'grasp', 'place', 'return'])
    for (const [path, fixture] of fixtures) {
      for (const runId of Object.keys(fixture.runs)) {
        const view = replay(eventsOf(fixture, runId))
        expect(view.timeline.map((s) => s.id), path).toEqual(STEPS)
        const ids = view.chat.map((line) => line.id)
        expect(new Set(ids).size, path).toBe(ids.length)
      }
    }
  })

  it('ignores an event it has already applied, so a replay after a reconnect draws nothing twice', () => {
    const events = eventsOf(twoParts, runOf(twoParts))
    const once = replay(events)
    const twice = replay([...events.slice(0, 20), ...events.slice(10)])
    expect(twice.chat.length).toBe(once.chat.length)
    expect(twice.stats).toEqual(once.stats)
    expect(twice.parts).toEqual(once.parts)
  })
})

describe('two parts, then nothing left (until empty, pose place)', () => {
  const runId = runOf(twoParts, (record) => record.kind === 'task')
  const events = eventsOf(twoParts, runId)
  const view = replay(events)

  it('counts two placed parts, two picks that found nothing, and a perfect rate over the real attempts', () => {
    expect(view.stats.placed).toBe(2)
    expect(view.stats.picks).toBe(4)
    expect(view.stats.emptyLooks).toBe(2)
    // A look that found nothing is how "until empty" ENDS; it is not a failed grasp.
    expect(view.stats.successRate).toBe(1)
    expect(view.stats.medianPartS).toBeCloseTo((21.4 + 27.9) / 2, 5)
    expect(view.stats.outcomes).toEqual({ succeeded: 2, no_target: 2 })
  })

  it('keeps one card per part, the second part with its push', () => {
    expect(view.parts.map((p) => [p.part, p.placed])).toEqual([
      [1, true],
      [2, true],
      [3, false],
    ])
    expect(view.parts[0].durationS).toBe(21.4)
    expect(view.parts[1].pushedMm).toBe(30)
    expect(view.parts[2].picks.map((p) => p.foundNothing)).toEqual([true, true])
  })

  it('shows the push as its own line, and looks again after it', () => {
    // The moment after the push: the push's own event, and the next one, the attempt that looks again.
    const pushed = upTo(events, (e) => e.type === 'pick.attempt_finished' && e.data.action === 'push').length
    expect(events[pushed].type).toBe('pick.attempt_started')
    const mid = replay(events.slice(0, pushed + 1))
    expect(mid.current.part).toBe(2)
    expect(mid.current.pushMm).toBe(30)
    expect(step(mid, 'look')).toBe('idle')
    const line = lineOf(mid, 'event.pick.attempt_finished.push')
    expect(line?.msg.params).toEqual({ mm: 30 })
    expect(say(line)).toBe('30 mm geschoben, erneut geschaut.')
  })

  it('pins each grasp overlay once, as it was decided', () => {
    expect(view.overlays.map((o) => [o.kind, o.url, o.part])).toEqual([
      ['grasp', `/v1/runs/${runId}/overlays/1`, 1],
      ['grasp', `/v1/runs/${runId}/overlays/2`, 2],
    ])
  })

  it('walks the timeline: grasp active while moving, place done after the drop, return done at the end', () => {
    expect(step(replay(upTo(events, isType('pick.executing'))), 'grasp')).toBe('active')
    const placed = replay(upTo(events, isType('task.placed')))
    expect(STEPS.map((id) => step(placed, id))).toEqual(['done', 'done', 'done', 'done', 'idle'])
    expect(placed.holding).toBe(false)
    const carrying = replay(upTo(events, isType('pick_result')))
    expect(carrying.holding).toBe(true)
    expect(STEPS.map((id) => step(view, id))).toEqual(['done', 'warn', 'skipped', 'idle', 'done'])
  })

  it('ends with the result and asks for the next instruction', () => {
    expect(view.phase).toBe('finished')
    expect(view.nextQuestion).toBe(true)
    const tail = keys(view).slice(-2)
    expect(tail).toEqual(['event.run_finished.nothing_left', 'chat.next'])
    expect(view.chat.at(-2)?.msg.params).toEqual({ parts: 2 })
  })

  it('opens with the plan in the operator\'s own words: what, where, where after, and the scope', () => {
    const first = view.chat[0]
    expect(first.msg.key).toBe('event.run_started.task')
    expect(first.msg.params?.what).toBe('alle grünen Würfel')
    expect(first.msg.params?.where).toBe('Ablage links')
    expect(first.msg.params?.back).toEqual({ key: 'common.home' })
    // The scope inside the sentence, in lower case: the capital "Bis leer" is the switch's label, not a word mid-line.
    expect(first.msg.params?.scope).toEqual({ key: 'scope.inline.until_empty' })
    expect(say(first)).toBe('Los: alle grünen Würfel → Ablage links → Home, bis leer.')
    expect(say(first, 'en')).toMatch(/, until empty\.$/)
  })

  it('keeps the backend\'s English sentence under every line, for the tech view', () => {
    for (const line of view.chat) {
      if (line.msg.key === 'chat.next') continue
      expect(typeof line.human).toBe('string')
    }
    expect(lineOf(view, 'event.task.placed.no_sensor')?.human).toMatch(/no sensor/)
  })
})

describe('a camera target lost at the drop', () => {
  const runId = runOf(targetLost, (record) => record.stop_code === 'target_lost')
  const view = replay(eventsOf(targetLost, runId))

  it('surveys first, keeps the bin it found and pins its overlay', () => {
    expect(view.survey.state).toBe('done')
    expect(view.survey.phrase).toBe('blue bin')
    expect(view.target?.label).toBe('blue bin')
    expect(view.target?.footprintMm).toEqual([300, 400])
    expect(view.overlays[0]).toMatchObject({ kind: 'target', url: `/v1/runs/${runId}/target/overlay` })
  })

  it('does not follow a bin that moved past the bound, and fails the place', () => {
    expect(view.targetCheck).toEqual({ movedMm: 180, followed: false })
    expect(view.target?.centreMm).toEqual([120, -610, 95])
    expect(lineOf(view, 'event.task.target_lost.moved_too_far')).toBeTruthy()
  })

  it('puts the part back, returns, and asks rather than stopping on a problem', () => {
    expect(view.holding).toBe(false)
    expect(view.parts[0].placed).toBe(false)
    expect(view.stopCard).toBeNull()
    expect(view.askCard).toMatchObject({ runId, stopCode: 'target_lost', part: 1, why: 'moved_too_far' })
    expect(view.nextQuestion).toBe(false)
    expect(step(view, 'return')).toBe('done')
  })
})

describe('halted while grasping, then restarted', () => {
  const stopped = runOf(haltedRestart, (record) => record.stop_class === 'problem')
  const restart = runOf(haltedRestart, (record) => Boolean(record.restart_of))
  const stoppedEvents = eventsOf(haltedRestart, stopped)
  const restartEvents = eventsOf(haltedRestart, restart)

  it('says the halt was asked the moment it was pressed, and confirmed when the run stops on it', () => {
    const pressed = replay(upTo(stoppedEvents, isType('run_halt_requested')))
    expect(pressed.halt.state).toBe('requested')
    expect(pressed.halt.inMotion).toBe(true)
    expect(pressed.phase).toBe('halting')
    const view = replay(stoppedEvents)
    expect(view.halt.state).toBe('confirmed')
  })

  it('draws the stop card where the arm stopped: the part, the step, the code', () => {
    const view = replay(stoppedEvents)
    expect(view.stopCard).toMatchObject({ runId: stopped, kind: 'task', stopCode: 'halted', part: 1, step: 'grasp' })
    expect(view.stopCard?.error).toMatch(/halted the arm/)
    expect(step(view, 'grasp')).toBe('failed')
    expect(view.nextQuestion).toBe(false)
    expect(keys(view)).toContain('event.run_halt_requested')
    expect(lineOf(view, 'event.run_error')?.msg.params).toEqual({ title: { key: 'stop.halted' } })
    expect(keys(view).at(-1)).toBe('event.run_finished.problem')
  })

  it('says a problem stop once in the demo view: the halt as it was pressed, then the end; the stop\'s own line is the tech view\'s', () => {
    const view = replay(stoppedEvents)
    const demo = view.chat.filter((line) => line.level === 'demo').map((line) => line.msg.key)
    expect(demo).toContain('event.run_halt_requested')
    expect(demo).toContain('event.run_finished.problem')
    expect(demo).not.toContain('event.run_error')
    expect(lineOf(view, 'event.run_error')?.level).toBe('tech')
  })

  it('does not count the pick the halt cut short as a failed grasp', () => {
    const view = replay(stoppedEvents)
    expect(view.stats.picks).toBe(1)
    expect(view.stats.cut).toBe(1)
    // No pick had its fair chance: the rate says nothing rather than 0 %.
    expect(view.stats.successRate).toBeNull()
  })

  it('starts the restart with the planned move home, and drops the old run\'s card', () => {
    const both = replay([...stoppedEvents, ...restartEvents])
    expect(both.runId).toBe(restart)
    expect(both.restartOf).toBe(stopped)
    expect(both.stopCard).toBeNull()
    expect(both.chat[0].msg).toEqual({ key: 'event.run_started.restart', params: { back: { key: 'common.home' } } })
    const homing = replay(upTo(restartEvents, isType('task.return_started')))
    expect(step(homing, 'return')).toBe('active')
    expect(homing.current.part).toBeNull()
    expect(both.phase).toBe('finished')
    expect(both.stats.placed).toBe(1)
  })
})

describe('a pose taught by hand', () => {
  const view = replay(eventsOf(teach, runOf(teach, (record) => record.kind === 'teach')))

  it('follows the session to its verdict and says what was saved', () => {
    expect(view.kind).toBe('teach')
    expect(view.teach).toMatchObject({ name: 'drop_left', label: 'Ablage links', role: 'place', state: 'saved', verdict: 'clear' })
    expect(view.teach?.wasOutside).toBe(true)
    expect(view.teach?.outside).toBe(false)
    expect(lineOf(view, 'event.run_finished.taught')?.msg.params).toEqual({ label: 'Ablage links' })
    expect(view.nextQuestion).toBe(false)
  })
})

/** A small event, for the behaviours no fixture covers yet. */
function ev(seq: number, type: string, data: Record<string, unknown> = {}, runId = 'run-x'): RunEvent {
  return { type, run_id: runId, seq, ts: 1000 + seq, severity: 'info', human: type, step: '', step_index: null, step_total: null, data }
}

const PLAN = {
  object: 'green cube',
  object_said: null,
  place: { kind: 'pose', pose: 'drop', pose_label: null },
  return_to: 'home',
  scope: 'once',
  first_motion: 'look',
  countdown: true,
}

/** A task's run record as `run_finished` carries it. */
function ended(kind: string, state: string, stopCode: string, stopClass: string, extra: Record<string, unknown> = {}) {
  return { id: 'run-x', kind, state, stop_code: stopCode, stop_class: stopClass, holding: false, parts_placed: 0, ...extra }
}

describe('the countdown before the first motion', () => {
  it('counts down on one line and is done at the first motion', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan: PLAN }),
      ev(2, 'run_countdown', { seconds_left: 3, because: 'teach' }),
      ev(3, 'run_countdown', { seconds_left: 2, because: 'teach' }),
      ev(4, 'run_countdown', { seconds_left: 1, because: 'teach' }),
    ])
    expect(view.phase).toBe('countdown')
    expect(view.countdown).toEqual({ state: 'active', secondsLeft: 1, because: 'teach' })
    expect(view.chat.filter((l) => l.msg.key === 'event.run_countdown')).toHaveLength(1)
    const moving = reduce(view, ev(5, 'pick.pick_started', { attempt_total: 5 }))
    expect(moving.countdown.state).toBe('done')
    expect(moving.phase).toBe('running')
  })

  it('stopped during the countdown, it ends cancelled with nothing moved', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'home', to: 'home' }),
      ev(2, 'run_countdown', { seconds_left: 3, because: 'jaws_opened' }),
      ev(3, 'run_finished', ended('home', 'cancelled', 'cancelled', 'operator')),
    ])
    expect(view.countdown.state).toBe('cancelled')
    expect(view.phase).toBe('cancelled')
    expect(view.stopCard).toBeNull()
  })
})

describe('the two stops', () => {
  it('a task\'s stop says the run stops after this part, and lets it finish the part', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false } }),
      ev(2, 'task.part_started', { part: 1, of: null }),
      ev(3, 'run_stop_requested', { scope: 'after_part' }),
    ])
    expect(view.phase).toBe('stopping')
    expect(view.stopScope).toBe('after_part')
    expect(view.stopRequested).toBe(true)
    expect(keys(view)).toContain('event.run_stop_requested.after_part')
    expect(translate('de', phaseMsg(view).key as never)).toBe('stoppt nach dem Teil')
  })

  it('a pick run\'s stop says it stops before the next attempt, never "after the part" (two meanings, kept apart)', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'pick', prompt: 'cube' }),
      ev(2, 'run_stop_requested', { scope: 'between_attempts' }),
    ])
    expect(view.phase).toBe('stopping')
    expect(view.stopScope).toBe('between_attempts')
    expect(phaseMsg(view)).toEqual({ key: 'run.phase.stopping.between_attempts' })
    expect(translate('de', phaseMsg(view).key as never)).toBe('stoppt vor dem nächsten Versuch')
    expect(translate('en', phaseMsg(view).key as never)).toBe('stopping before the next attempt')
  })

  it('every other phase reads as itself', () => {
    expect(phaseMsg(EMPTY_RUN)).toEqual({ key: 'run.phase.idle' })
  })
})

describe('a push', () => {
  const begun = [
    ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false } }),
    ev(2, 'task.part_started', { part: 1, of: 1 }),
    ev(3, 'pick.pick_started', { attempt_total: 5 }),
    ev(4, 'pick.attempt_started', { attempt: 0, attempt_total: 5 }),
  ]

  it('that ran says how far it pushed and that the arm looked again, and counts the distance', () => {
    const view = replay([
      ...begun,
      ev(5, 'pick.attempt_finished', { action: 'push', outcome: 'pushed', push: 'pushed', push_mm: 30, push_reason: 'Pushed the part 30 mm.' }),
    ])
    const line = view.chat.at(-1)
    expect(line?.msg).toEqual({ key: 'event.pick.attempt_finished.push', params: { mm: 30 } })
    expect(line?.level).toBe('demo')
    expect(view.current.pushMm).toBe(30)
    expect(view.parts[0].pushedMm).toBe(30)
  })

  // The pick loop reports every push attempt with action "push"; only `push: "pushed"` is a push that ran
  // (src/robot/grasping/loop/pick_loop.py, PickAttempt.push).
  for (const [push, outcome] of [
    ['unsafe_recovery_refused', 'aborted'],
    ['controller_not_operational', 'controller_not_operational'],
  ] as const) {
    it(`that stopped where the arm stands (${push}) is never announced as pushed, and adds no distance`, () => {
      const view = replay([
        ...begun,
        ev(5, 'pick.attempt_finished', { action: 'push', outcome, push, push_reason: 'stopped where the arm stands', push_leg: 'push' }),
      ])
      expect(keys(view)).not.toContain('event.pick.attempt_finished.push')
      const line = view.chat.at(-1)
      expect(line?.msg.key).toBe('event.pick.attempt_finished.push_stopped')
      expect(line?.tone).toBe('warn')
      expect(line?.level).toBe('demo')
      expect(say(line)).not.toMatch(/geschoben, erneut geschaut/)
      expect(say(line, 'en')).not.toMatch(/looked again/)
      expect(view.current.pushMm).toBeNull()
      expect(view.parts[0].pushedMm).toBe(0)
    })
  }

  for (const [push, outcome] of [
    ['refused_controller_stopped', 'controller_not_operational'],
    ['refused_jaws_unknown', 'gripper_fault'],
  ] as const) {
    it(`refused before anything moved (${push}) says nothing moved, and adds no distance`, () => {
      const view = replay([
        ...begun,
        ev(5, 'pick.attempt_finished', { action: 'push', outcome, push, push_reason: 'refused before anything moved' }),
      ])
      expect(keys(view)).not.toContain('event.pick.attempt_finished.push')
      const line = view.chat.at(-1)
      expect(line?.msg.key).toBe('event.pick.attempt_finished.push_refused')
      expect(line?.tone).toBe('warn')
      expect(view.current.pushMm).toBeNull()
      expect(view.parts[0].pushedMm).toBe(0)
    })
  }

  it('whose result the payload does not say is not claimed as a push that ran', () => {
    const view = replay([...begun, ev(5, 'pick.attempt_finished', { action: 'push', outcome: 'aborted' })])
    expect(keys(view)).not.toContain('event.pick.attempt_finished.push')
    expect(view.parts[0].pushedMm).toBe(0)
  })
})

// The pick loop's "clear the blocker" (the owner, 2026-10-02): a neighbour in the way of every grasp is gripped, set
// down elsewhere and the arm looks again (`blocker: "set_aside"`), or the clearing stopped once something moved
// (src/robot/grasping/loop/pick_loop.py, `_set_the_blocker_aside`).
describe('a blocker', () => {
  const begun = [
    ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false } }),
    ev(2, 'task.part_started', { part: 1, of: 1 }),
    ev(3, 'pick.pick_started', { attempt_total: 5 }),
    ev(4, 'pick.attempt_started', { attempt: 0, attempt_total: 5 }),
  ]

  it('set aside is said in the demo view, and is no push', () => {
    const view = replay([
      ...begun,
      ev(5, 'pick.attempt_finished', { action: 'clear_blocker', outcome: 'set_aside', blocker: 'set_aside',
        blocker_at_mm: [-196.0, -705.0, 103.0], set_down: 'a free spot at (-40, -620) mm, 160 mm from the part' }),
    ])
    const line = view.chat.at(-1)
    expect(line?.msg).toEqual({ key: 'event.pick.attempt_finished.blocker' })
    expect(line?.level).toBe('demo')
    expect(line?.tone).not.toBe('warn')
    expect(say(line)).toMatch(/Hindernis/)
    expect(say(line, 'en')).toMatch(/obstacle/)
    expect(keys(view)).not.toContain('event.pick.attempt_finished.push')
    expect(view.parts[0].pushedMm).toBe(0)
  })

  it('whose clearing stopped once something moved says a person decides, and never that it was set aside', () => {
    const view = replay([
      ...begun,
      ev(5, 'pick.attempt_finished', { action: 'clear_blocker', outcome: 'aborted', blocker: 'stopped_where_the_arm_stands',
        blocker_reason: 'the arm stays where it stopped, the blocker may be in the hand' }),
    ])
    const line = view.chat.at(-1)
    expect(line?.msg.key).toBe('event.pick.attempt_finished.blocker_stopped')
    expect(line?.tone).toBe('warn')
    expect(line?.level).toBe('demo')
    expect(keys(view)).not.toContain('event.pick.attempt_finished.blocker')
    expect(say(line)).toMatch(/Person entscheidet/)
    expect(say(line, 'en')).toMatch(/a person decides/)
  })
})

describe('the look step', () => {
  const looking = replay([
    ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false, options: { multi_view: true } } }),
    ev(2, 'task.part_started', { part: 1, of: 1 }),
    ev(3, 'pick.pick_started', { attempt_total: 5 }),
    ev(4, 'pick.attempt_started', { attempt: 0, attempt_total: 5 }),
    ev(5, 'pick.perceived', { look: 'look_1', segmentation_count: 3 }),
  ])

  it('counts the looks of an attempt, and counts again from one in the next attempt', () => {
    const note = (view: RunView) => view.timeline.find((s) => s.id === 'look')?.note
    expect(note(looking)).toEqual({ key: 'step.lookN', params: { n: 1 } })
    const second = reduce(looking, ev(6, 'pick.perceived', { look: 'look_2', segmentation_count: 3 }))
    expect(note(second)).toEqual({ key: 'step.lookN', params: { n: 2 } })
    expect(second.current.lookIndex).toBe(2)
    const again = replay([ev(7, 'pick.attempt_started', { attempt: 1, attempt_total: 5 }), ev(8, 'pick.perceived', { look: 'look_1' })], second)
    expect(note(again)).toEqual({ key: 'step.lookN', params: { n: 1 } })
  })

  it('says the planned total where it is known ("Blick 2/3"), and never puts a look over a total it passed', () => {
    const facts = { wrist_camera: true, looks: ['look_1', 'look_2', 'look_3'] }
    expect(plannedLooks(looking.plan, facts)).toBe(3)
    // Multi-view off is the first configured look only (Q11).
    expect(plannedLooks({ ...looking.plan!, options: { ...looking.plan!.options!, multi_view: false } }, facts)).toBe(1)
    expect(plannedLooks(null, null)).toBeNull()
    expect(plannedLooks(null, { wrist_camera: false, looks: [] })).toBeNull()
    const second = reduce(looking, ev(6, 'pick.perceived', { look: 'look_2' }))
    expect(lookNote(second, 3)).toEqual({ key: 'step.lookOf', params: { n: 2, total: 3 } })
    expect(translate('de', 'step.lookOf', { n: 2, total: 3 })).toBe('Blick 2/3')
    expect(translate('en', 'step.lookOf', { n: 2, total: 3 })).toBe('Look 2/3')
    // The one view a pick generates comes after the configured looks: counted, but over no total it exceeds.
    const fourth = replay([ev(6, 'pick.perceived', { look: 'look_2' }), ev(7, 'pick.perceived', { look: 'look_3' }), ev(8, 'pick.perceived', { look: 'generated' })], looking)
    expect(lookNote(fourth, 3)).toEqual({ key: 'step.lookN', params: { n: 4 } })
    expect(lookNote(EMPTY_RUN, 3)).toBeNull()
  })
})

describe('a gap in the stream', () => {
  it('is shown, and the counters come from the run record afterwards', () => {
    const runId = runOf(twoParts)
    const events = eventsOf(twoParts, runId)
    const gap: RunEvent = { ...ev(30, 'gap', { dropped: 18 }, runId), severity: 'warn' }
    let view = replay([...events.slice(0, 3), gap, ...events.slice(48)])
    expect(view.gaps).toEqual({ count: 1, dropped: 18 })
    expect(lineOf(view, 'chat.gap')?.msg.params).toEqual({ dropped: 18 })
    view = reduce(view, { type: 'run.snapshot', run: twoParts.runs[runId] })
    expect(view.stats.placed).toBe(2)
    expect(view.stats.picks).toBe(4)
    expect(view.stats.outcomes).toEqual({ succeeded: 2, no_target: 2 })
  })
})

describe('a run the server no longer knows', () => {
  const running = replay([
    ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false } }),
    ev(2, 'task.part_started', { part: 1, of: 1 }),
    ev(3, 'pick.pick_started', { attempt_total: 5 }),
    ev(4, 'pick.perceived', { look: 'look_1' }),
  ])

  it('stops running on screen and says so, rather than showing a step in progress forever', () => {
    const lost = reduce(running, { type: 'run.lost', runId: 'run-x' })
    expect(lost.phase).toBe('lost')
    expect(lost.timeline.some((s) => s.state === 'active')).toBe(false)
    const line = lost.chat.at(-1)
    expect(line?.msg.key).toBe('chat.lost')
    expect(line?.tone).toBe('warn')
    expect(line?.level).toBe('demo')
    // What it saw before stays: the part it was on, the lines it said.
    expect(lost.current.part).toBe(1)
    expect(lost.chat.length).toBe(running.chat.length + 1)
    expect(translate('de', phaseMsg(lost).key as never)).toBe('nicht mehr bekannt')
    // Said once, however often the console notices.
    expect(reduce(lost, { type: 'run.lost', runId: 'run-x' })).toBe(lost)
  })

  it('changes nothing for another run, or for a run that has ended', () => {
    expect(reduce(running, { type: 'run.lost', runId: 'run-other' })).toBe(running)
    const over = reduce(running, ev(5, 'run_finished', ended('task', 'finished', 'finished', 'done')))
    expect(reduce(over, { type: 'run.lost', runId: 'run-x' })).toBe(over)
  })
})

describe('the halt as the stop buttons read it', () => {
  const pressed = replay([
    ev(1, 'run_started', { kind: 'task', plan: PLAN }),
    ev(2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: 1002, braking: true }),
  ])

  it('raises "press the e-stop" only where the arm brakes, and only after 1.5 s without a confirmation', () => {
    expect(haltState(pressed.halt, 1002.5, true)).toBe('requested')
    expect(haltState(pressed.halt, 1003.6, true)).toBe('unconfirmed')
    // An arm that latches but does not brake halts before its next motion: no false alarm.
    expect(haltState(pressed.halt, 1010, false)).toBe('requested')
  })

  it('never raises it once the halt is confirmed, or for an arm standing still', () => {
    const confirmed = reduce(pressed, ev(3, 'run_error', { stop_code: 'halted', error: 'halted' }))
    expect(haltState(confirmed.halt, 1010, true)).toBe('confirmed')
    const still = replay([
      ev(1, 'run_started', { kind: 'task', plan: PLAN }),
      ev(2, 'run_halt_requested', { in_motion: false, requested_at: 1002, braking: false }),
    ])
    expect(haltState(still.halt, 1010, true)).toBe('requested')
    expect(haltState(EMPTY_RUN.halt, 1010, true)).toBe('none')
  })

  it('is confirmed by the cell when the arm reports the move in flight braked, and by nothing less', () => {
    const latch = { reason: 'the operator pressed halt now', requested_at: 1002.01, in_motion: true, braked: true, brake_s: 0.04 }
    expect(haltState(pressed.halt, 1010, true, latch)).toBe('confirmed')
    // A latch that has not braked (yet) is not a confirmation on an arm that brakes: the latch is set BEFORE the brake.
    expect(haltState(pressed.halt, 1010, true, { ...latch, braked: false })).toBe('unconfirmed')
    expect(haltState(pressed.halt, 1010, true, null)).toBe('unconfirmed')
    // An older halt's latch confirms nothing about this one.
    expect(haltState(pressed.halt, 1010, true, { ...latch, requested_at: 900 })).toBe('unconfirmed')
  })

  // A run that ends on another code after a halt request has stopped commanding: its thread is over before
  // run_finished is published (api/runs.py `_finish`). The alarm would otherwise stay raised over an arm that stands.
  for (const [what, kind, state, code, cls] of [
    ['a task whose return was refused', 'task', 'failed', 'return_failed', 'problem'],
    ['a Home move that arrived as the halt was pressed', 'home', 'finished', 'finished', 'done'],
    ['a teach, which a halt cancels', 'teach', 'cancelled', 'cancelled', 'operator'],
    ['a run the cell went down under', 'task', 'failed', 'disconnected', 'problem'],
  ] as const) {
    it(`never raises it after the run ended: ${what}`, () => {
      const view = replay([
        ev(1, 'run_started', { kind, plan: kind === 'task' ? PLAN : undefined, to: kind === 'home' ? 'home' : undefined }),
        ev(2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: 1002, braking: true }),
        ...(state === 'failed' ? [ev(3, 'run_error', { stop_code: code, error: 'stopped' })] : []),
        ev(4, 'run_finished', ended(kind, state, code, cls)),
      ])
      expect(view.phase).toBe(state)
      expect(haltState(view.halt, 1100, true)).toBe('confirmed')
    })
  }
})

describe('what the cockpit groups and the tech view opens', () => {
  const runId = runOf(twoParts, (record) => record.kind === 'task')
  const view = replay(eventsOf(twoParts, runId))

  it('files every line under the part it belongs to, and the run\'s own lines under none', () => {
    // The chat draws a card per part and the run's own lines (start, stop, end) between the cards.
    const of = (key: string) => view.chat.filter((line) => line.msg.key === key).map((line) => line.part)
    expect(of('event.run_started.task')).toEqual([null])
    // Part 3 is looked for twice (two empty looks): the second time only the tech view hears it.
    expect(of('event.task.part_started')).toEqual([1, 2, 3, 3])
    expect(of('event.pick.attempt_finished.push')).toEqual([2])
    expect(of('event.task.placed.no_sensor')).toEqual([1, 2])
    expect(of('event.run_finished.nothing_left')).toEqual([null])
    expect(of('chat.next')).toEqual([null])
  })

  it('keeps each event\'s machine half on its line, for the tech view to open', () => {
    const result = view.chat.find((line) => line.type === 'pick_result' && line.part === 1)
    expect(result?.data).toMatchObject({ outcome: 'succeeded', part: 1, overlay: `/v1/runs/${runId}/overlays/1` })
    expect(view.chat.find((line) => line.msg.key === 'chat.next')?.data).toEqual({})
  })

  it('says the pick loop\'s own outcome without its enum\'s name (PickOutcome.NO_PERCEPTION)', () => {
    const finished = view.chat.filter((line) => line.type === 'pick.pick_finished').map((line) => translate('en', line.msg.key as never, line.msg.params))
    expect(finished).toContain('Pick ended: executed.')
    expect(finished).toContain('Pick ended: no perception.')
    expect(finished.join(' ')).not.toMatch(/PickOutcome/)
  })
})

describe('a Home run', () => {
  const homeLog = homeJson as unknown as Fixture
  const events = eventsOf(homeLog, runOf(homeLog))

  it('walks its one step: return active while it moves, done when it arrives', () => {
    const moving = replay(upTo(events, isType('home.started')))
    expect(step(moving, 'return')).toBe('active')
    expect(moving.countdown.state).toBe('done')
    const arrived = replay(events)
    expect(step(arrived, 'return')).toBe('done')
    expect(arrived.nextQuestion).toBe(true)
  })

  it('stopped before its move is sent, says nothing about a part, and stays out of "stopping after the part"', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'home', to: 'home' }),
      ev(2, 'run_countdown', { seconds_left: 3, because: 'teach' }),
      ev(3, 'run_stop_requested', { scope: 'before_motion' }),
    ])
    expect(view.stopScope).toBe('before_motion')
    expect(view.phase).toBe('countdown')
    expect(keys(view)).not.toContain('event.run_stop_requested.after_part')
    // Said as what it is: the move is not sent, and one already sent runs to its end (the cockpit's own line).
    expect(view.chat.at(-1)?.msg.key).toBe('ck.stop.beforeMotion')
    expect(translate('de', 'ck.stop.beforeMotion' as never, undefined, COCKPIT)).toBe('Ich stoppe vor der Fahrt; eine schon gesendete Fahrt endet erst.')
    expect(translate('de', phaseMsg(view).key as never)).not.toMatch(/Teil/)
  })
})

describe('a run that ended before it moved', () => {
  it('names the library\'s refusal (RunOut.refusal) rather than "cancelled", and asks for the next instruction', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false } }),
      ev(2, 'run_finished', ended('task', 'cancelled', 'cancelled', 'operator', {
        error: 'refused before anything moved (camera_target_unavailable, 409): no camera', refusal: { code: 'camera_target_unavailable', status: 409 },
      })),
    ])
    expect(view.stopCard).toBeNull()
    expect(keys(view)).not.toContain('event.run_finished.cancelled')
    expect(keys(view)).toContain('refusal.camera_target_unavailable')
    expect(view.chat.find((line) => line.msg.key === 'refusal.camera_target_unavailable')?.tone).toBe('warn')
    expect(view.nextQuestion).toBe(true)
  })

  it('says a jaws question that began as it started is to be answered first, never as a problem', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false } }),
      ev(2, 'run_finished', ended('task', 'cancelled', 'cancelled', 'operator', {
        error: 'a jaws question began as the run started (a check of the jaws): answer it in the browser, then start again; the run ended before its first motion, and nothing moved',
      })),
    ])
    expect(view.stopCard).toBeNull()
    expect(keys(view)).toContain('refusal.jaws_question_pending')
    expect(keys(view)).not.toContain('event.run_finished.cancelled')
    expect(view.nextQuestion).toBe(true)
  })
})

describe('the arm\'s own word on a brake (HaltStateOut.brake)', () => {
  const pressed = replay([
    ev(1, 'run_started', { kind: 'task', plan: PLAN }),
    ev(2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: 1002, braking: true }),
  ])
  const latch = { reason: 'the operator pressed halt now', requested_at: 1002.01, in_motion: true, braked: false, brake_s: null }

  it('alarms at once when the arm says the brake was not confirmed, where it brakes', () => {
    expect(haltState(pressed.halt, 1002.2, true, { ...latch, brake: 'unconfirmed' })).toBe('unconfirmed')
    // An arm that does not brake raises no alarm, whatever it says.
    expect(haltState(pressed.halt, 1002.2, false, { ...latch, brake: 'unconfirmed' })).toBe('requested')
  })

  it('is confirmed when the arm says it braked, or that the move ran out', () => {
    expect(haltState(pressed.halt, 1010, true, { ...latch, brake: 'braked' })).toBe('confirmed')
    expect(haltState(pressed.halt, 1010, true, { ...latch, brake: 'ran_out' })).toBe('confirmed')
    // Still braking: the 1.5 s window decides, as before.
    expect(haltState(pressed.halt, 1002.5, true, { ...latch, brake: 'pending' })).toBe('requested')
    expect(haltState(pressed.halt, 1010, true, { ...latch, brake: 'pending' })).toBe('unconfirmed')
  })

  // The brake gives up after 2 s and writes "unconfirmed"; the halted run ends right after it, and the poll that
  // carries the arm's word may come in only after `run_finished` (connection.py `_wait_still`).
  it('keeps the alarm when the arm\'s "not confirmed" comes in after the halted run ended', () => {
    const over = replay([
      ev(1, 'run_started', { kind: 'task', plan: PLAN }),
      ev(2, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: 1002, braking: true }),
      ev(3, 'run_error', { stop_code: 'halted', error: 'halted' }),
      ev(4, 'run_finished', ended('task', 'failed', 'halted', 'problem')),
    ])
    expect(over.halt.state).toBe('confirmed')
    expect(haltState(over.halt, 1004.5, true, { ...latch, brake: 'unconfirmed' })).toBe('unconfirmed')
    // A page opened after the run (no halt in view): the latch's own word still alarms while it stands.
    expect(haltState(EMPTY_RUN.halt, 1004.5, true, { ...latch, brake: 'unconfirmed' })).toBe('unconfirmed')
    // Never where the arm does not brake, and an older latch says nothing about this run's halt.
    expect(haltState(over.halt, 1004.5, false, { ...latch, brake: 'unconfirmed' })).toBe('confirmed')
    expect(haltState(over.halt, 1004.5, true, { ...latch, brake: 'unconfirmed', requested_at: 900 })).toBe('confirmed')
  })
})

describe('the success rate', () => {
  it('leaves out a pick a halt cut short, and still counts the parts placed before it', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan: { ...PLAN, countdown: false, scope: 'until_empty' } }),
      ev(2, 'task.part_started', { part: 1, of: null }),
      ev(3, 'pick_result', { outcome: 'succeeded', succeeded: true, part: 1, pick: 1 }),
      ev(4, 'task.placed', { outcome: 'executed', no_sensor: true }),
      ev(5, 'task.part_finished', { part: 1, placed: true, duration_s: 12 }),
      ev(6, 'task.part_started', { part: 2, of: null }),
      ev(7, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: 1007, braking: false }),
      ev(8, 'pick_result', { outcome: 'cancelled', succeeded: false, part: 2, pick: 2 }),
    ])
    expect(view.stats.placed).toBe(1)
    expect(view.stats.picks).toBe(2)
    expect(view.stats.cut).toBe(1)
    expect(view.stats.successRate).toBe(1)
  })

  it('never dips while a part is carried: a gripped part counts once it is placed or not', () => {
    const plan = { ...PLAN, countdown: false, scope: 'until_empty' }
    const first = [
      ev(1, 'run_started', { kind: 'task', plan }),
      ev(2, 'task.part_started', { part: 1, of: null }),
      ev(3, 'pick_result', { outcome: 'succeeded', succeeded: true, part: 1, pick: 1 }),
    ]
    // Part 1 is in the jaws: nothing is decided yet, so the rate says nothing rather than 0 %.
    const carrying = replay(first)
    expect(carrying.stats.successRate).toBeNull()
    expect(carrying.stats.rated).toBe(0)
    const placed = [
      ...first,
      ev(4, 'task.placed', { outcome: 'executed', no_sensor: true }),
      ev(5, 'task.part_finished', { part: 1, placed: true, duration_s: 12 }),
    ]
    expect(replay(placed).stats.successRate).toBe(1)
    // Part 2 is gripped and carried: still 100 %, never "50 % · 1 von 2" while it travels.
    const second = replay([...placed, ev(6, 'task.part_started', { part: 2, of: null }), ev(7, 'pick_result', { outcome: 'succeeded', succeeded: true, part: 2, pick: 2 })])
    expect(second.stats.successRate).toBe(1)
    expect(second.stats.rated).toBe(1)
    // A grasp that failed is decided at once: its part is looked for again.
    const missed = replay([...placed, ev(6, 'task.part_started', { part: 2, of: null }), ev(7, 'pick_result', { outcome: 'execution_failed', succeeded: false, part: 2, pick: 2 })])
    expect(missed.stats.successRate).toBe(0.5)
  })

  it('pins the rate in the middle of the captured two-part task: 100 % while part 2 is carried', () => {
    const events = eventsOf(twoParts, runOf(twoParts, (record) => record.kind === 'task'))
    expect(replay(upTo(events, isType('pick_result'), 1)).stats.successRate).toBeNull()
    expect(replay(upTo(events, isType('pick_result'), 2)).stats.successRate).toBe(1)
  })

  it('leaves out a part a person halted in the jaws, as it leaves out a pick a halt cut short', () => {
    const plan = { ...PLAN, countdown: false, scope: 'until_empty' }
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan }),
      ev(2, 'task.part_started', { part: 1, of: null }),
      ev(3, 'pick_result', { outcome: 'succeeded', succeeded: true, part: 1, pick: 1 }),
      ev(4, 'task.placed', { outcome: 'executed', no_sensor: true }),
      ev(5, 'task.part_finished', { part: 1, placed: true, duration_s: 12 }),
      ev(6, 'task.part_started', { part: 2, of: null }),
      ev(7, 'pick_result', { outcome: 'succeeded', succeeded: true, part: 2, pick: 2 }),
      ev(8, 'task.place_started', { place: 'pose:drop' }),
      ev(9, 'run_halt_requested', { reason: 'the operator pressed halt now', in_motion: true, requested_at: 1009, braking: false }),
      ev(10, 'run_error', { stop_code: 'halted', error: 'halted' }),
      ev(11, 'run_finished', ended('task', 'failed', 'halted', 'problem', { parts_placed: 1, holding: true, attempted: 2, succeeded: 2 })),
    ])
    expect(view.parts.map((p) => [p.part, p.placed, p.cut])).toEqual([
      [1, true, false],
      [2, false, true],
    ])
    // "ohne leere Blicke und angehaltene Griffe": one placed of the one part that was let finish.
    expect(view.stats.successRate).toBe(1)
    expect(view.stats.rated).toBe(1)
    // The captured halt after a grasp (the dummy's "Backen leer" scenario): nothing was let finish, so no rate at all.
    const backenLeer = backenLeerJson as unknown as Fixture
    const stopped = replay(eventsOf(backenLeer, runOf(backenLeer, (record) => record.stop_code === 'halted')))
    expect(stopped.stats.successRate).toBeNull()
  })

  it('still counts a part that was gripped and not placed for a reason of the cell\'s own', () => {
    const plan = { ...PLAN, countdown: false, scope: 'once' }
    const view = replay([
      ev(1, 'run_started', { kind: 'task', plan }),
      ev(2, 'task.part_started', { part: 1, of: 1 }),
      ev(3, 'pick_result', { outcome: 'succeeded', succeeded: true, part: 1, pick: 1 }),
      ev(4, 'task.place_failed', { outcome: 'refused', message: 'the line in was refused' }),
      ev(5, 'run_error', { stop_code: 'part_still_held', error: 'refused before the release' }),
      ev(6, 'run_finished', ended('task', 'failed', 'part_still_held', 'problem', { holding: true, attempted: 1, succeeded: 1 })),
    ])
    expect(view.parts[0].cut).toBe(false)
    expect(view.stats.successRate).toBe(0)
  })
})

describe('a pick run (POST /v1/pick, no place)', () => {
  it('counts its picks without parts and ends with its own line', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'pick' }),
      ev(2, 'pick.pick_started', { attempt_total: 5 }),
      ev(3, 'pick_result', { outcome: 'succeeded', succeeded: true, looks: ['look_1'], looks_fused: ['look_1'] }),
      ev(4, 'run_finished', ended('pick', 'finished', 'finished', 'done', { succeeded: 1, attempted: 1 })),
    ])
    expect(view.parts).toEqual([])
    expect(view.picks).toHaveLength(1)
    expect(lineOf(view, 'event.run_finished.pick')?.msg.params).toEqual({ succeeded: 1, attempted: 1 })
    expect(view.nextQuestion).toBe(true)
  })
})

describe('a wave at a greeting', () => {
  it('says it waves, ends on "Gewinkt" and asks for the next instruction', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'wave' }),
      ev(2, 'wave.started', { swings: 2, swing_deg: 15 }),
      ev(3, 'wave.done', { swings: 2 }),
      ev(4, 'run_finished', ended('wave', 'finished', 'finished', 'done')),
    ])
    expect(view.kind).toBe('wave')
    expect(say(lineOf(view, 'event.run_started.wave'))).toBe('Ich winke dir zu.')
    expect(say(lineOf(view, 'event.wave.started'))).toBe('Das Handgelenk schwingt: 2× 15° zu jeder Seite.')
    expect(say(lineOf(view, 'event.run_finished.wave'))).toBe('Gewinkt.')
    expect(view.nextQuestion).toBe(true)
  })

  it('says a refused swing to everyone', () => {
    const view = replay([
      ev(1, 'run_started', { kind: 'wave' }),
      ev(2, 'wave.started', { swings: 2, swing_deg: 15 }),
      ev(3, 'wave.refused', { status: 'collision', message: 'the guard refused', moved: true }),
      ev(4, 'run_finished', ended('wave', 'failed', 'return_failed', 'problem')),
    ])
    const refused = lineOf(view, 'event.wave.refused')
    expect(refused?.level).toBe('demo')
    expect(say(refused)).toBe('Winken abgelehnt: ein Schwung ist nicht frei.')
    expect(view.stopCard).not.toBeNull()
  })
})
