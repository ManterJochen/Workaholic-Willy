/**
 * One confirm host for the console (App mounts it): `await confirm({...})` before Restart and Home.
 *
 * The answer is a promise of a boolean, and every way out of the dialog except the confirm button is `false`: Cancel,
 * Escape, a click beside it, a second confirm asked while one is open, and a console with no host at all. Nothing
 * moves unless a person clicked the button that says it moves.
 */

import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '../i18n'
import { ConfirmProvider, useConfirm, type Confirm } from './confirm'

const HOME = { title: 'Home anfahren?', firstMotion: 'geplante Fahrt nach Home', confirmLabel: 'Home – der Roboter fährt', moves: true }
const RESTART = { ...HOME, title: 'Neustart?', firstMotion: 'geplante Fahrt nach Home, dann der Auftrag', confirmLabel: 'Neustart – der Roboter fährt' }

let confirm: Confirm = async () => false

function Grab() {
  const asked = useConfirm()
  useEffect(() => {
    confirm = asked
  }, [asked])
  return null
}

function host() {
  return render(
    <I18nProvider lang="de">
      <ConfirmProvider>
        <Grab />
      </ConfirmProvider>
    </I18nProvider>,
  )
}

/** A promise's answer if it has one within a moment, else `'pending'`. */
function settled<T>(promise: Promise<T>): Promise<T | 'pending'> {
  return Promise.race([promise, new Promise<'pending'>((resolve) => setTimeout(() => resolve('pending'), 30))])
}

const reactEnv = globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }

afterEach(() => {
  reactEnv.IS_REACT_ACT_ENVIRONMENT = true
  cleanup()
})

describe('the confirm host', () => {
  it('answers true only for the confirm button', async () => {
    host()
    let answer: Promise<boolean> = Promise.resolve(false)
    act(() => {
      answer = confirm(HOME)
    })
    expect(screen.getByRole('dialog', { name: 'Home anfahren?' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Home – der Roboter fährt' }))
    await expect(answer).resolves.toBe(true)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('answers false for Cancel and for Escape', async () => {
    host()
    let first: Promise<boolean> = Promise.resolve(true)
    act(() => {
      first = confirm(HOME)
    })
    fireEvent.click(screen.getByRole('button', { name: 'Abbrechen' }))
    await expect(first).resolves.toBe(false)
    let second: Promise<boolean> = Promise.resolve(true)
    act(() => {
      second = confirm(HOME)
    })
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    await expect(second).resolves.toBe(false)
  })

  it('answers false to an older request when a newer one replaces it', async () => {
    host()
    let older: Promise<boolean> = Promise.resolve(true)
    act(() => {
      older = confirm(HOME)
    })
    act(() => {
      void confirm({ ...HOME, title: 'Neustart?' })
    })
    await expect(older).resolves.toBe(false)
    expect(screen.getByRole('dialog', { name: 'Neustart?' })).toBeTruthy()
  })

  it('never confirms without a host', async () => {
    render(<Grab />)
    await expect(confirm(HOME)).resolves.toBe(false)
  })

  it('never lets a click on one dialog answer the request that replaced it before it was drawn', async () => {
    // The race of a request that arrives from outside an event (a handler that awaited the cell first: React schedules
    // it at a lower priority) while a dialog is open, and a person's click on that dialog (the highest priority) before
    // the new request was drawn. React re-runs a state updater when it rebases a skipped update, so an answer resolved
    // inside one reached the replacing request: a click on "Home" answered a "Restart" nobody saw.
    host()
    let home: Promise<boolean> = Promise.resolve(true)
    act(() => {
      home = confirm(HOME)
    })
    const homeConfirm = screen.getByRole('button', { name: 'Home – der Roboter fährt' })
    reactEnv.IS_REACT_ACT_ENVIRONMENT = false
    const restart = confirm(RESTART)
    homeConfirm.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await new Promise((resolve) => setTimeout(resolve, 60))
    reactEnv.IS_REACT_ACT_ENVIRONMENT = true

    // The replaced request answers no; the replacing one waits for its own dialog, which is now on screen.
    await expect(home).resolves.toBe(false)
    expect(await settled(restart)).toBe('pending')
    const dialog = screen.getByRole('dialog', { name: 'Neustart?' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Neustart – der Roboter fährt' }))
    await expect(restart).resolves.toBe(true)
  })

  it('answers no for a request still open when the host goes away, and for one asked after', async () => {
    const { unmount } = host()
    let open: Promise<boolean> = Promise.resolve(true)
    act(() => {
      open = confirm(HOME)
    })
    const kept = confirm
    unmount()
    await expect(open).resolves.toBe(false)
    await expect(kept(HOME)).resolves.toBe(false)
  })
})
