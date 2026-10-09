/**
 * What Enter may start, as pure rules (the owner, 2026-10-08: Enter starts at once, the mode set beforehand, what and
 * where from the sentence, the settings only where it says nothing).
 *
 * `doubts` says why a reading needs a person's look; `draftFromReading` fills what the sentence did not say from the
 * settings; `cellOffOf` is the one rule of the card's Start and of Enter; `modelToLoad` keeps a task from loading the
 * language model inside its first look; `readTask` reads the settings back, each the default where nothing readable is
 * stored. The one part a sentence singles out (`which`) and where the parts lie (`source`) go from the reading onto the
 * card and into the task, and a selection starts once. `cockpit.test.tsx` pins the same through the screen.
 */

import { afterEach, describe, expect, it } from 'vitest'

import type { CellFactsOut, CellOut, CommandOut, ReadinessOut, TaskPlanOut } from '../api/client'
import { DEFAULT_TASK, placeOf, readTask, type TaskPrefs } from '../model/prefs'
import {
  doubts,
  draftFromPlan,
  draftFromReading,
  groundedOf,
  KNOWN_SENTENCE_MODEL,
  looksOfPlan,
  looksOptions,
  manualDraft,
  taskOf,
  taskOfPlan,
} from './draft'
import { cellOffOf, modelToLoad } from './gate'

const SAID = { text: 'Alle grauen Würfel in die Gelbe Kiste.', source: 'typed', language: 'de' } as const

function reading(over: Partial<CommandOut> = {}): CommandOut {
  return {
    understood: true,
    intent: 'task',
    object: { phrase: 'gray cube', said: 'grauen Würfel', verified: true, route: null },
    place: { phrase: 'yellow bin', said: 'in die Gelbe Kiste', verified: true, route: null },
    place_pose: null,
    scope: 'until_empty',
    count: null,
    return_to: null,
    notes: [],
    reason: '',
    model: { model_id: KNOWN_SENTENCE_MODEL, latency_ms: 0.2, attempts: 0, loaded_now: false, remembered: false },
    raw: '',
    greeting: null,
    startable: true,
    ...over,
  }
}

const ENTER: TaskPrefs = { ...DEFAULT_TASK, start: 'enter', scope: 'once' }

afterEach(() => {
  localStorage.clear()
})

describe('doubts: why a reading waits for the card', () => {
  it('has none for a clean reading', () => {
    expect(doubts(reading())).toEqual([])
  })

  it('names each note, a part not found, a place not found, and no part at all', () => {
    expect(doubts(reading({ notes: ['retried'], startable: false }))).toEqual([{ key: 'note.retried' }])
    expect(doubts(reading({ object: { phrase: 'gray cube', said: null, verified: false, route: null }, startable: false })))
      .toEqual([{ key: 'ck.doubt.object' }])
    expect(doubts(reading({ place: { phrase: 'yellow bin', said: null, verified: false, route: null }, startable: false })))
      .toEqual([{ key: 'ck.doubt.place' }])
    expect(doubts(reading({ object: null, startable: false }))).toEqual([{ key: 'ck.doubt.noPart' }])
  })

  it('trusts no reading the reader itself does not call startable', () => {
    expect(doubts(reading({ startable: false }))).toEqual([{ key: 'ck.doubt.reading' }])
    expect(doubts({ ...reading(), startable: undefined } as unknown as CommandOut)).toEqual([{ key: 'ck.doubt.reading' }])
  })
})

describe('draftFromReading: the settings where the sentence says nothing', () => {
  it('takes the mode from the settings where Enter starts, and the reader\'s where the card does', () => {
    expect(draftFromReading(reading(), SAID, ENTER).scope).toBe('once')
    expect(draftFromReading(reading(), SAID, { ...ENTER, start: 'card' }).scope).toBe('until_empty')
    expect(draftFromReading(reading(), SAID).scope).toBe('until_empty')
  })

  it('keeps the sentence\'s place over the settings\' place', () => {
    const task: TaskPrefs = { ...ENTER, place: { kind: 'pose', pose: 'park' } }
    expect(draftFromReading(reading(), SAID, task).place).toEqual({ kind: 'camera', phrase: 'yellow bin', said: 'in die Gelbe Kiste' })
    expect(draftFromReading(reading({ place: null, place_pose: 'drop_left' }), SAID, task).place).toEqual({ kind: 'pose', pose: 'drop_left' })
    expect(draftFromReading(reading({ place: null }), SAID, task).place).toEqual({ kind: 'pose', pose: 'park' })
  })

  it('ticks "anything" only for a sentence that names no part, and only where the settings say so', () => {
    const anything: TaskPrefs = { ...ENTER, anything: true }
    expect(draftFromReading(reading({ object: null }), SAID, anything).pickAnything).toBe(true)
    expect(draftFromReading(reading(), SAID, anything).pickAnything).toBe(false)
    expect(draftFromReading(reading({ object: null }), SAID, ENTER).pickAnything).toBe(false)
  })

  it('says whether no model was asked, or the answer came from memory', () => {
    expect(draftFromReading(reading(), SAID, ENTER).reading).toMatchObject({ known: true, remembered: false })
    const remembered = reading({ model: { model_id: 'Qwen/Qwen3-VL-8B-Instruct', latency_ms: 0.4, attempts: 1, loaded_now: false, remembered: true } })
    expect(draftFromReading(remembered, SAID, ENTER).reading).toMatchObject({ known: false, remembered: true })
  })
})

describe('the one part singled out and where the parts lie, from the reading to the start (2026-10-08)', () => {
  const SINGLED = reading({ which: 'the gray cube on top of the other one', source: 'on the black mat', scope: 'once' })

  it('come from the reading onto the card, and are empty where it said neither', () => {
    expect(draftFromReading(SINGLED, SAID, ENTER)).toMatchObject({ which: 'the gray cube on top of the other one', source: 'on the black mat' })
    expect(draftFromReading(reading(), SAID, ENTER)).toMatchObject({ which: '', source: '' })
    expect(draftFromReading({ ...reading(), which: undefined, source: undefined } as unknown as CommandOut, SAID)).toMatchObject({ which: '', source: '' })
  })

  it('start one part once, whatever mode the settings or the sentence set', () => {
    expect(draftFromReading(SINGLED, SAID, { ...ENTER, scope: 'until_empty' }).scope).toBe('once')
    expect(draftFromReading({ ...SINGLED, scope: 'until_empty' }, SAID, { ...ENTER, start: 'card' }).scope).toBe('once')
    expect(draftFromReading(reading({ source: 'on the black mat' }), SAID, { ...ENTER, scope: 'until_empty' }).scope).toBe('until_empty')
  })

  it('go out with the task, without their blanks', () => {
    const card = draftFromReading(SINGLED, SAID, ENTER)
    expect(taskOf({ ...card, which: '  the upper gray cube ', source: ' on the black mat  ' })).toMatchObject({
      which: 'the upper gray cube',
      source: 'on the black mat',
    })
  })

  it('stay with a run\'s plan when it is edited or started again', () => {
    const plan = {
      object: 'gray cube',
      object_said: null,
      which: 'the upper gray cube',
      source: 'on the black mat',
      place: { kind: 'pose', pose: 'drop_left' },
      return_to: 'home',
      scope: 'once',
    } as unknown as TaskPlanOut
    expect(draftFromPlan(plan)).toMatchObject({ which: 'the upper gray cube', source: 'on the black mat' })
    expect(taskOfPlan(plan, 'default')).toMatchObject({ which: 'the upper gray cube', source: 'on the black mat' })
    const kept = { ...plan, which: undefined, source: undefined } as unknown as TaskPlanOut
    expect(taskOfPlan(kept)).toMatchObject({ which: '', source: '' })
  })

  it('route the pick as the server\'s guard judges it: the one part alone, else the object where the parts lie', () => {
    expect(groundedOf({ object: 'gray cube', which: ' the upper gray cube ', source: 'on the black mat' })).toBe('the upper gray cube')
    expect(groundedOf({ object: ' gray cube ', which: '', source: ' on the black mat' })).toBe('gray cube on the black mat')
    expect(groundedOf({ object: 'gray cube', which: '  ', source: '' })).toBe('gray cube')
    expect(groundedOf({ object: '', which: '', source: '' })).toBe('')
  })
})

describe('cellOffOf and modelToLoad: whether the cell lets a task start', () => {
  const CELL = { state: 'connected', active_run_id: null, recovery: null } as unknown as CellOut
  const READY = { ready: true, lights: [], blockers: [] } as unknown as ReadinessOut

  it('lets a connected, ready cell start, and says why any other does not', () => {
    expect(cellOffOf(CELL, READY, false)).toEqual([])
    expect(cellOffOf({ ...CELL, state: 'built' } as CellOut, READY, false)).toEqual([{ key: 'refusal.not_connected' }])
    expect(cellOffOf(CELL, READY, true)).toEqual([{ key: 'blocker.run_active' }])
    expect(cellOffOf(CELL, null, false)).toEqual([{ key: 'ck.start.readinessUnknown' }])
    expect(cellOffOf(CELL, { ...READY, ready: false } as ReadinessOut, false)).toEqual([{ key: 'ck.ready.not' }])
  })

  it('asks for the language model first only on a cell that detects with it, until it is loaded', () => {
    const vlm = { detector: { backend: 'vlm' } } as unknown as CellFactsOut
    const commands = (code: string) => ({ ...READY, lights: [{ id: 'commands', state: 'info', code, message: '', blocks: false }] }) as unknown as ReadinessOut
    expect(modelToLoad(vlm, commands('idle'))).toBe(true)
    expect(modelToLoad(vlm, commands('ready'))).toBe(false)
    expect(modelToLoad({ detector: { backend: 'grounded_sam' } } as unknown as CellFactsOut, commands('idle'))).toBe(false)
    expect(modelToLoad(null, commands('idle'))).toBe(false)
  })
})

describe('the task\'s settings, as this browser keeps them', () => {
  it('start with Enter, once, the default place, the card asking for a part, and the looks when needed', () => {
    expect(readTask()).toEqual({ start: 'enter', scope: 'once', place: { kind: 'default' }, anything: false, looks: 'when_needed' })
  })

  it('read back what was stored, and the default for anything they cannot read', () => {
    localStorage.setItem('willy.taskStart', 'card')
    localStorage.setItem('willy.taskScope', 'until_empty')
    localStorage.setItem('willy.taskPlace', 'pose:park')
    localStorage.setItem('willy.taskAnything', 'anything')
    localStorage.setItem('willy.taskLooks', 'every')
    expect(readTask()).toEqual({
      start: 'card',
      scope: 'until_empty',
      place: { kind: 'pose', pose: 'park' },
      anything: true,
      looks: 'every',
    })
    localStorage.setItem('willy.taskStart', 'now')
    localStorage.setItem('willy.taskScope', 'twice')
    localStorage.setItem('willy.taskLooks', 'all of them')
    expect(readTask()).toMatchObject({ start: 'enter', scope: 'once', looks: 'when_needed' })
  })

  it('take a camera place with a phrase only, and never longer than a card\'s field', () => {
    expect(placeOf('camera:yellow bin')).toEqual({ kind: 'camera', phrase: 'yellow bin' })
    expect(placeOf('camera:   ')).toEqual({ kind: 'default' })
    expect(placeOf('pose:')).toEqual({ kind: 'default' })
    expect(placeOf(`camera:${'y'.repeat(300)}`)).toEqual({ kind: 'camera', phrase: 'y'.repeat(200) })
    expect(placeOf('somewhere')).toEqual({ kind: 'default' })
  })
})

describe('the looks, the views and the carry a task starts with (the owner, 2026-10-08 night)', () => {
  const PLAN = {
    object: 'gray cube',
    object_said: null,
    place: { kind: 'camera', phrase: 'yellow bin', said: 'in die Gelbe Kiste' },
    return_to: 'home',
    scope: 'once',
    options: { multi_view: true, every_look: true, both_faces: true, record_views: false, carry: 'over_the_rim' },
  } as unknown as TaskPlanOut

  it('take the looks from the settings, Enter or card, and a card filled by hand too', () => {
    expect(draftFromReading(reading(), SAID, { ...ENTER, looks: 'every' }).options.looks).toBe('every')
    expect(draftFromReading(reading(), SAID, { ...ENTER, start: 'card', looks: 'first' }).options.looks).toBe('first')
    expect(draftFromReading(reading(), SAID).options.looks).toBe('when_needed')
    expect(manualDraft(SAID, 'vlm_unavailable', 'no VLM', { looks: 'every' }).options.looks).toBe('every')
    expect(manualDraft(SAID, 'vlm_unavailable', 'no VLM').options.looks).toBe('when_needed')
  })

  it('send the looks as multi-view and every look, and every task the console starts keeps its views', () => {
    const card = draftFromReading(reading(), SAID, ENTER)
    for (const [looks, options] of [
      ['first', { multi_view: false, every_look: false }],
      ['when_needed', { multi_view: true, every_look: false }],
      ['every', { multi_view: true, every_look: true }],
    ] as const) {
      expect(taskOf({ ...card, options: { ...card.options, looks } }).options).toMatchObject({ ...options, record_views: true })
    }
    expect(looksOptions('every')).toEqual({ multi_view: true, every_look: true })
  })

  it('send the carry to a bin the camera finds only where it was touched, and none to a taught pose', () => {
    const card = draftFromReading(reading(), SAID, ENTER)
    expect(taskOf(card).options?.carry).toBeNull()
    expect(taskOf({ ...card, options: { ...card.options, carryOverTheRim: true } }).options?.carry).toBe('over_the_rim')
    expect(taskOf({ ...card, options: { ...card.options, carryOverTheRim: false } }).options?.carry).toBe('via_the_look')
    const pose = draftFromReading(reading({ place: null, place_pose: 'drop_left' }), SAID, ENTER)
    expect(taskOf({ ...pose, options: { ...pose.options, carryOverTheRim: true } }).options?.carry).toBeNull()
  })

  it('keep a run\'s plan as it ran: its looks, its carry and a program\'s both faces, its views kept', () => {
    expect(draftFromPlan(PLAN).options).toMatchObject({ looks: 'every', carryOverTheRim: true, bothFaces: true })
    expect(taskOfPlan(PLAN).options).toMatchObject({ multi_view: true, every_look: true, both_faces: true, record_views: true, carry: 'over_the_rim' })
    expect(taskOfPlan(PLAN, 'default').options?.carry).toBeNull()
    const first = { ...PLAN, options: { multi_view: false } } as unknown as TaskPlanOut
    expect(looksOfPlan(first.options)).toBe('first')
    expect(taskOfPlan(first).options).toMatchObject({ multi_view: false, every_look: false })
    const kept = { ...PLAN, options: undefined } as unknown as TaskPlanOut
    expect(draftFromPlan(kept).options).toMatchObject({ looks: 'when_needed', carryOverTheRim: null, bothFaces: false })
  })
})
