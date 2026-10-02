/**
 * The two motions that are not Start: Home, and Restart after a problem stop (build plan item 12, 1.3.5, 1.3.6).
 *
 * Each asks first, through the console's one confirm host: the dialog names the first motion (the planned move to the
 * return pose), the 3 s hands-off countdown when one is due, and that a planned move may begin with a short straight
 * leg. Only a click on the dialog's confirm sends the request, once; Cancel, Escape and a click beside it send nothing.
 * The 202 answer is followed at once, so the run is drawn even if it ends before the next cell poll.
 */

import { createElement, useCallback, useState } from 'react'

import { ApiError, api, type RunOut } from '../api/client'
import { useT } from '../i18n'
import { useConfirm } from '../model/confirm'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { COCKPIT } from './i18n'

export interface Motions {
  /** THIS MOVES, after the dialog's confirm: the planned move to Home. */
  home(): Promise<void>
  /** THIS MOVES, after the dialog's confirm: a new run of the stopped one, its first motion the move to `to`. */
  restart(runId: string, to: string, then: string | null): Promise<void>
  readonly busy: boolean
  readonly error: ApiError | null
  clearError(): void
}

export function useMotions(): Motions {
  const t = useT(COCKPIT)
  const confirm = useConfirm()
  const { follow } = useRun()
  const { cell, refresh } = useCell()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const countdown = cell?.countdown_due === true

  const send = useCallback(
    async (request: () => Promise<RunOut>) => {
      setBusy(true)
      setError(null)
      try {
        follow(await request())
        refresh()
      } catch (err: unknown) {
        setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
      } finally {
        setBusy(false)
      }
    },
    [follow, refresh],
  )

  const home = useCallback(async () => {
    const yes = await confirm({
      title: t('ck.confirm.homeTitle'),
      firstMotion: t('ck.confirm.homeMotion'),
      countdown,
      confirmLabel: t('ck.confirm.homeGo'),
      moves: true,
    })
    if (yes) await send(() => api.home('home'))
  }, [confirm, countdown, send, t])

  const restart = useCallback(
    async (runId: string, to: string, then: string | null) => {
      const yes = await confirm({
        title: t('ck.confirm.restartTitle'),
        firstMotion: t('ck.confirm.restartMotion', { to }),
        countdown,
        confirmLabel: t('ck.confirm.restartGo'),
        moves: true,
        // One more of the dialog's notes, in the same list form, so it lines up with them.
        children: then ? createElement('ul', { className: 'dialog-notes ck-dialog-then' }, createElement('li', null, then)) : undefined,
      })
      if (yes) await send(() => api.restart(runId))
    },
    [confirm, countdown, send, t],
  )

  return { home, restart, busy, error, clearError: () => setError(null) }
}
