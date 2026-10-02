/**
 * The jaws question, answered in the browser (build plan 1.6, item 10): the dialog shows the waiting question with
 * exactly the choices the server offers and NEVER a default.
 *
 * A toggle hand has no sensor: every change of its output moves the jaws, and only a person looking at them can say
 * where they stand. So nothing here may answer for that person: no choice is focused or lit when the dialog opens, Enter
 * sends nothing, and a question that expired is not offered (the server has refused it: no answer is never "open").
 * "Jetzt öffnen" is one change of the output that releases what the jaws hold, so it is drawn as the motion it is and
 * the dialog says to hold the part first.
 */

import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fakeSockets, refusal, renderWith, stubApi, type ApiCall } from '../test/render'
import JawsDialog from './JawsDialog'

const HAND = {
  kind: 'toggle',
  driver: 'JawIOGripper',
  where: 'tool output 0',
  connected: true,
  jaws: 'unknown',
  why_unknown: 'nobody said yet',
  no_sensor: true,
  commands_sent: 0,
}

const CELL = {
  state: 'connecting',
  arm: 'URRobotArm',
  gripper: 'JawIOGripper',
  vendor: 'ur',
  profile: 'ur10_cell',
  active_run_id: null,
  needs_person: '',
  halted: null,
  payload_model: 'none',
  recovery: null,
  countdown_due: false,
  jaws_question: true,
  hand: HAND,
}

function question(over: Record<string, unknown> = {}) {
  return {
    question_id: 'q-1',
    stage: 'where',
    at: 'connect',
    where: 'tool output 0',
    reason: 'every change of its output moves them and nothing reads them back',
    choices: ['open', 'closed'],
    attempt: 1,
    of: 3,
    why_again: '',
    expires_at: Date.now() / 1000 + 100,
    text: 'The jaws ... [o]pen / [c]losed',
    ...over,
  }
}

function serve(routes: Record<string, unknown>): ApiCall[] {
  return stubApi({
    '/v1/cell': CELL,
    '/v1/cell/readiness': refusal(501, 'not_built_yet', 'not built yet'),
    '/v1/cell/facts': refusal(501, 'not_built_yet', 'not built yet'),
    ...routes,
  })
}

function answers(calls: ApiCall[]): ApiCall[] {
  return calls.filter((c) => c.method === 'POST' && c.path === '/v1/cell/jaws/answer')
}

beforeEach(() => {
  localStorage.clear()
  fakeSockets()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('the jaws dialog', () => {
  it('offers exactly the choices the server offers, and none of them as a default', async () => {
    const calls = serve({ '/v1/cell/jaws': { hand: HAND, question: question() } })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    await waitFor(() => expect(within(dialog).getByText(/Wo stehen die Backen an Tool-DO0\?/)).toBeTruthy())

    const group = within(dialog).getByRole('group', { name: 'Antworten' })
    const buttons = within(group).getAllByRole('button')
    expect(buttons.map((b) => b.dataset.choice)).toEqual(['open', 'closed'])
    // No default: nothing is focused, lit or pressed when the question opens.
    for (const button of buttons) {
      expect(document.activeElement).not.toBe(button)
      expect(button.getAttribute('aria-pressed')).toBeNull()
      expect(button.className).not.toMatch(/primary/)
    }
    // Enter on the dialog answers nothing.
    fireEvent.keyDown(document.activeElement ?? dialog, { key: 'Enter' })
    fireEvent.keyDown(dialog, { key: 'Enter' })
    expect(answers(calls)).toHaveLength(0)
    // It says why only a person can answer, and that no answer is never "open".
    expect(dialog.textContent).toMatch(/keine Vorauswahl/)
    expect(dialog.textContent).toMatch(/nie „offen“/)
  })

  it('sends the one choice a person clicked, with the question it answers', async () => {
    const calls = serve({
      '/v1/cell/jaws': { hand: HAND, question: question() },
      'POST /v1/cell/jaws/answer': { hand: HAND, question: null },
    })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    const closed = await within(dialog).findByRole('button', { name: /Geschlossen/ })
    fireEvent.click(closed)
    await waitFor(() => expect(answers(calls)).toHaveLength(1))
    expect(answers(calls)[0].body).toEqual({ question_id: 'q-1', choice: 'closed' })
  })

  it('draws "open now" as the one change it is, and says to hold the part first', async () => {
    serve({
      '/v1/cell/jaws': {
        hand: { ...HAND, jaws: 'closed' },
        question: question({ question_id: 'q-2', stage: 'open_now', choices: ['open_now', 'abort'] }),
      },
    })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    const open = await within(dialog).findByRole('button', { name: /Jetzt öffnen/ })
    expect(open.className).toMatch(/danger/)
    expect(within(dialog).getByText('Die Backen sind zu. Jetzt öffnen?')).toBeTruthy()
    expect(within(open).getByText('Jetzt öffnen')).toBeTruthy()
    expect(open.textContent).toMatch(/ein Schaltvorgang an Tool-DO0: die Backen öffnen sich/)
    // Said calmly, once: no capitals shouting "EINE", and the "where" explanation stays with the question where.
    expect(dialog.textContent).not.toMatch(/\bEINE[RN]?\b/)
    expect(dialog.textContent).not.toMatch(/Nur du siehst/)
    // "Abbrechen" weighs as much as any other answer: a filled neutral button, neither red nor lit nor a ghost.
    const abort = within(dialog).getByRole('button', { name: /Abbrechen/ })
    expect(abort.className).not.toMatch(/danger|primary|ghost/)
    expect(dialog.textContent).toMatch(/Halte das Teil fest/)
    expect(document.activeElement).not.toBe(open)
  })

  it('never brings back a question it answered, when a read that left before the answer lands after it', async () => {
    let reads = 0
    let late: (value: unknown) => void = () => undefined
    const calls = serve({
      '/v1/cell/jaws': () => {
        reads += 1
        if (reads === 1) return { hand: HAND, question: question() }
        if (reads === 2) return new Promise((resolve) => (late = resolve))
        return { hand: HAND, question: null }
      },
      'POST /v1/cell/jaws/answer': { hand: { ...HAND, jaws: 'open' }, question: null },
    })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    await within(dialog).findByRole('button', { name: /Offen/ })
    // The read of the next second has left and is still on its way when the person answers.
    await waitFor(() => expect(reads).toBeGreaterThanOrEqual(2), { timeout: 2500 })
    fireEvent.click(within(dialog).getByRole('button', { name: /Offen/ }))
    await waitFor(() => expect(answers(calls)).toHaveLength(1))
    await waitFor(() => expect(screen.queryAllByRole('button', { name: /Offen|Geschlossen/ })).toHaveLength(0))
    await act(async () => {
      late({ hand: HAND, question: question() })
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
    expect(screen.queryAllByRole('button', { name: /Offen|Geschlossen/ })).toHaveLength(0)
    expect(answers(calls)).toHaveLength(1)
  })

  it('offers no answer while the hand acts on one (a flag without a waiting question)', async () => {
    serve({ '/v1/cell/jaws': { hand: HAND, question: null } })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    await waitFor(() => expect(dialog.textContent).toMatch(/führt eine Antwort aus/))
    expect(within(dialog).queryAllByRole('button')).toHaveLength(0)
  })

  it('never offers a question that has expired: the server refused it, it is not "open"', async () => {
    serve({ '/v1/cell/jaws': { hand: HAND, question: question({ expires_at: Date.now() / 1000 - 5 }) } })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    await waitFor(() => expect(dialog.textContent).toMatch(/Abgelaufen/))
    expect(within(dialog).queryAllByRole('button')).toHaveLength(0)
  })

  it('opens from the cell stream at once, before the server is asked, and says when the question expires', async () => {
    serve({ '/v1/cell': { ...CELL, jaws_question: false } })
    const sockets = fakeSockets()
    renderWith(<JawsDialog />)
    await waitFor(() => expect(sockets.find('cell')).toBeTruthy())
    act(() =>
      sockets.find('cell')!.deliver([
        {
          type: 'cell.jaws_question',
          run_id: 'cell',
          seq: 1,
          ts: Date.now() / 1000,
          severity: 'warn',
          human: 'Where do the jaws stand?',
          step: '',
          step_index: null,
          step_total: null,
          data: question({ expires_at: Date.now() / 1000 + 60 }),
        },
      ]),
    )
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    expect(dialog.textContent).toMatch(/läuft ab in (59|60) s/)
    expect(within(dialog).getAllByRole('button').map((b) => b.dataset.choice)).toEqual(['open', 'closed'])
  })

  it('counts down from the moment a question shows, not from when the page was opened', async () => {
    serve({ '/v1/cell': { ...CELL, jaws_question: false } })
    const sockets = fakeSockets()
    renderWith(<JawsDialog />)
    await waitFor(() => expect(sockets.find('cell')).toBeTruthy())
    // The page has been open for an hour when the question comes: the clock moves on an hour.
    const realNow = Date.now.bind(Date)
    const clock = vi.spyOn(Date, 'now').mockImplementation(() => realNow() + 3_600_000)
    try {
      const asked = Date.now() / 1000
      act(() =>
        sockets.find('cell')!.deliver([
          {
            type: 'cell.jaws_question',
            run_id: 'cell',
            seq: 1,
            ts: asked,
            severity: 'warn',
            human: 'Where do the jaws stand?',
            step: '',
            step_index: null,
            step_total: null,
            data: question({ expires_at: asked + 120 }),
          },
        ]),
      )
      const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
      // At most the 120 s the server holds a question, never the hour the page stood open on top of it.
      expect(dialog.textContent).toMatch(/läuft ab in (119|120) s/)
    } finally {
      clock.mockRestore()
    }
  })

  it('says a refused answer in words and keeps the question', async () => {
    serve({
      '/v1/cell/jaws': { hand: HAND, question: question() },
      'POST /v1/cell/jaws/answer': refusal(409, 'question_changed', 'the question changed'),
    })
    renderWith(<JawsDialog />)
    const dialog = await screen.findByRole('alertdialog', { name: 'Backenfrage' })
    fireEvent.click(await within(dialog).findByRole('button', { name: /Offen/ }))
    await waitFor(() => expect(within(dialog).getByRole('alert').textContent).toMatch(/Die Frage hat sich geändert/))
    expect(within(dialog).getAllByRole('button', { name: /Offen|Geschlossen/ })).toHaveLength(2)
  })

  it('is not there when no question waits', async () => {
    serve({ '/v1/cell': { ...CELL, state: 'connected', jaws_question: false } })
    renderWith(<JawsDialog />)
    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
  })
})
