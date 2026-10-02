/**
 * What a person does to bring the cell up, and the little state it needs: build, read the connect preview, connect
 * with its token, disconnect, and say the cell is clear after a latch.
 *
 * One hook for the Setup stepper and the stacked Cell panels, so the preview a person read in one step is the token
 * the Connect step sends, and nothing else ever is. Each action is one click's one request: no retry, no resend, and
 * the server enforces every gate itself. A refusal comes back as the typed `ApiError` for the screen to say.
 *
 * Only `connect` moves anything (brakes release, a gripper may move); it is never called without the token of a
 * preview read in this session, and the token is dropped once used, after a build and after a disconnect, so a
 * second connect needs a second look at what it moves.
 */

import { useCallback, useState } from 'react'

import { ApiError, api, type CellOut, type ConnectPreviewOut } from '../api/client'

/** Whether to rehearse when nobody chose: only on the dummy arm (build plan 4.3: real by default where it is real). */
export function rehearseByDefault(cell: CellOut | null): boolean {
  return cell?.vendor === 'dummy'
}

export type CellBusy = 'build' | 'preview' | 'connect' | 'disconnect' | 'acknowledge'

export interface CellActions {
  readonly busy: CellBusy | null
  readonly error: ApiError | null
  /** The connect preview read for the cell as it is built now; its token is what Connect sends. */
  readonly preview: ConnectPreviewOut | null
  readonly build: (rehearse: boolean) => Promise<void>
  readonly readPreview: () => Promise<void>
  readonly connect: () => Promise<void>
  readonly disconnect: () => Promise<void>
  /** "The cell is clear": a person's word after a latch. Moves nothing; a protective stop is cleared at the pendant. */
  readonly acknowledge: () => Promise<void>
  readonly clearError: () => void
}

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

/** Codes after which the preview's token is no use any more: read the preview again. */
const STALE = new Set(['stale_token', 'not_acknowledged'])

export function useCellActions(onChanged: () => void): CellActions {
  const [busy, setBusy] = useState<CellBusy | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [preview, setPreview] = useState<ConnectPreviewOut | null>(null)

  const act = useCallback(
    async (what: CellBusy, fn: () => Promise<void>) => {
      setBusy(what)
      setError(null)
      try {
        await fn()
      } catch (err) {
        const refused = asApiError(err)
        setError(refused)
        if (STALE.has(refused.code)) setPreview(null)
      } finally {
        setBusy(null)
        onChanged()
      }
    },
    [onChanged],
  )

  const build = useCallback(
    (rehearse: boolean) =>
      act('build', async () => {
        await api.build(rehearse)
        setPreview(null)
      }),
    [act],
  )

  const readPreview = useCallback(
    () =>
      act('preview', async () => {
        setPreview(await api.connectPreview())
      }),
    [act],
  )

  const connect = useCallback(
    () =>
      act('connect', async () => {
        if (!preview) return
        const token = preview.token
        await api.connect(token)
        setPreview(null)
      }),
    [act, preview],
  )

  const disconnect = useCallback(
    () =>
      act('disconnect', async () => {
        await api.disconnect()
        setPreview(null)
      }),
    [act],
  )

  const acknowledge = useCallback(
    () =>
      act('acknowledge', async () => {
        await api.acknowledge(false)
      }),
    [act],
  )

  const clearError = useCallback(() => setError(null), [])

  return { busy, error, preview, build, readPreview, connect, disconnect, acknowledge, clearError }
}
