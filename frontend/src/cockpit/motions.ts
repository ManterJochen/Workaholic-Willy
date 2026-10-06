/**
 * The motions that are not Start: Home, and Restart after a problem stop (build plan item 12, 1.3.5, 1.3.6), and the
 * wave at a greeting.
 *
 * Home and Restart ask first, through the console's one confirm host: the dialog names the first motion (the planned
 * move to the return pose), the 3 s hands-off countdown when one is due, and that a planned move may begin with a short
 * straight leg. Only a click on the dialog's confirm sends the request, once; Cancel, Escape and a click beside it send
 * nothing. The wave asks the same way where the app config says `confirm`, and goes at once where it says `direct`
 * (the owner, 2026-10-06: "Sofort winken"), the greeting the person typed being the click. The 202 answer is followed
 * at once, so the run is drawn even if it ends before the next cell poll.
 */

import { createElement, useCallback, useState } from 'react'

import { ApiError, api, type RunOut } from '../api/client'
import { useT } from '../i18n'
import { useConfirm } from '../model/confirm'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { COCKPIT } from './i18n'

/** What became of a wave: started (its run is followed), declined at the dialog, or refused, with the refusal. */
export type WaveAnswer = 'started' | 'declined' | ApiError

export interface Motions {
  /** THIS MOVES, after the dialog's confirm: the planned move to Home. */
  home(): Promise<void>
  /** THIS MOVES: Willy waves, after the dialog's confirm where `ask`, else at once. The refusal is returned, not kept. */
  wave(ask: boolean): Promise<WaveAnswer>
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

  const wave = useCallback(
    async (ask: boolean): Promise<WaveAnswer> => {
      if (ask) {
        const yes = await confirm({
          title: t('ck.confirm.waveTitle'),
          firstMotion: t('ck.confirm.waveMotion'),
          countdown,
          confirmLabel: t('ck.confirm.waveGo'),
          moves: true,
        })
        if (!yes) return 'declined'
      }
      try {
        follow(await api.wave())
        refresh()
        return 'started'
      } catch (err: unknown) {
        return err instanceof ApiError ? err : new ApiError(0, null, String(err))
      }
    },
    [confirm, countdown, follow, refresh, t],
  )

  return { home, restart, wave, busy, error, clearError: () => setError(null) }
}
