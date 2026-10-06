/**
 * From a sentence to a running task, with a person between every step (OD 8, OD 22; build plan 4.2).
 *
 * 1. `read(text, source)`: the operator's words go into the conversation as they were said, and to the VLM
 *    (`POST /v1/commands/parse`), which MOVES NOTHING. Its answer becomes the Understood card (`Draft`). Without a reader
 *    (501, or 409 `vlm_not_loaded`) the card opens by hand with the reason; a sentence read as "stop" only says where
 *    the stop buttons are. Reading never starts anything. A greeting ("Hallo Willy") opens no card: Willy greets back
 *    and hands the wave to `greet`, as the app config says (`CommandOut.greeting`): at once where it says `direct`
 *    (the owner, 2026-10-06), after the dialog's confirm where it says `confirm`, not at all where it says `off`.
 * 2. `update(field, change)`: a person corrects the card; each changed field is recorded for the run's record.
 * 3. `start()`: the click on Start, the confirmation. It posts the card once (`POST /v1/task`), follows the 202 answer
 *    and clears the card. A refusal stays on the card, said in the reader's language; nothing is retried.
 *
 * Nothing here runs on a timer or a stream: every request is the direct answer to a person's key or click.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api, type RunOut, type TaskIn } from '../api/client'
import type { Lang } from '../i18n'
import { lightMsg, refusalMsg } from '../i18n/codes'
import type { Msg } from '../i18n/types'
import { say } from '../model/chat'
import { draftFromReading, edit, manualDraft, taskOf, type Draft, type DraftField } from './draft'
import type { WaveAnswer } from './motions'

/** The reader's refusals that open the card by hand (OD 22): there is no reader to ask. */
const NO_READER = new Set(['vlm_unavailable', 'vlm_not_loaded', 'vlm_model_missing'])

/** How long the card waits after a keystroke before it asks how the detector would route the phrase. */
const ROUTE_DEBOUNCE_MS = 350

export type CommandBusy = 'reading' | 'starting' | 'loading' | null

export interface CommandModel {
  readonly draft: Draft | null
  readonly busy: CommandBusy
  /** Why the last Start (or a load) was refused; shown on the card. */
  readonly error: ApiError | null
  /** The last sentence read, for "read it again" after the reader was loaded. */
  readonly lastSaid: { text: string; source: 'typed' | 'spoken' } | null
  read(text: string, source: 'typed' | 'spoken'): Promise<void>
  update(field: DraftField, change: Partial<Draft>): void
  /** Open a card a person fills (an ask card's "another target"). */
  open(draft: Draft): void
  discard(): void
  /** THIS MOVES. Post the card as a task, once. */
  start(): Promise<void>
  /** THIS MOVES. Post a task an ask card's option names, once. */
  startTask(task: TaskIn): Promise<boolean>
  /** "Laden": load the command reader (moves nothing). */
  load(): Promise<void>
}

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

export function useCommand({
  lang,
  follow,
  refresh = () => undefined,
  greet,
}: {
  lang: Lang
  follow: (run: RunOut) => void
  /** Poll the cell now: a run a click just started is then named by every chip at once. */
  refresh?: () => void
  /** THIS MOVES: the wave at a greeting (`Motions.wave`), after the dialog's confirm where `ask`. None: no wave. */
  greet?: (ask: boolean) => Promise<WaveAnswer>
}): CommandModel {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [busy, setBusy] = useState<CommandBusy>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [lastSaid, setLastSaid] = useState<{ text: string; source: 'typed' | 'spoken' } | null>(null)
  /** One request at a time, whatever renders in between: a double Enter is one parse, a double click one task. */
  const inFlight = useRef(false)

  const read = useCallback(
    async (text: string, source: 'typed' | 'spoken') => {
      const words = text.trim()
      if (!words || inFlight.current) return
      inFlight.current = true
      setBusy('reading')
      setError(null)
      setLastSaid({ text: words, source })
      say({ who: 'operator', kind: 'command', text: words, source })
      const said = { text: words, source, language: lang }
      try {
        const out = await api.parse({ text: words, source, language: lang })
        if (out.greeting) {
          // A greeting opens no card: Willy greets back, and waves as the app config says. The wave is its own run,
          // judged swing by swing, with the stop buttons under the image like every other.
          const mode = out.greeting
          setDraft(null)
          const hello = mode === 'direct' && greet ? 'ck.greet.wave' : mode === 'confirm' && greet ? 'ck.greet.ask' : 'ck.greet.hello'
          say({ who: 'willy', kind: 'reply', msg: { key: hello }, tone: 'info' })
          if (mode !== 'off' && greet) {
            void greet(mode === 'confirm').then((answer) => {
              if (answer instanceof ApiError) {
                say({ who: 'willy', kind: 'reply', msg: { key: 'ck.greet.refused', params: { why: refusalWords(answer) } }, tone: 'warn' })
              }
            })
          }
        } else if (out.intent === 'stop') {
          // Speech never stops anything (api/routers/media.py): the buttons do. Nothing is sent.
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.stopHint' }, tone: 'warn' })
          setDraft(null)
        } else if (!out.understood) {
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.notUnderstood' }, tone: 'warn' })
          setDraft(draftFromReading(out, said))
        } else {
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.understood' }, tone: 'info' })
          setDraft(draftFromReading(out, said))
        }
      } catch (err: unknown) {
        const refused = asApiError(err)
        if (NO_READER.has(refused.code)) {
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.manual' }, tone: 'warn' })
          setDraft(manualDraft(said, refused.code, refused.message))
        } else {
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.refused', params: { why: refusalWords(refused) } }, tone: 'error' })
        }
      } finally {
        inFlight.current = false
        setBusy(null)
      }
    },
    [lang, greet],
  )

  const update = useCallback((field: DraftField, change: Partial<Draft>) => {
    setDraft((current) => (current ? edit(current, field, change) : current))
    setError(null)
  }, [])

  const open = useCallback((next: Draft) => {
    setDraft(next)
    setError(null)
  }, [])

  const discard = useCallback(() => {
    setDraft(null)
    setError(null)
  }, [])

  const post = useCallback(
    async (task: TaskIn): Promise<boolean> => {
      if (inFlight.current) return false
      inFlight.current = true
      setBusy('starting')
      setError(null)
      try {
        const run = await api.task(task)
        follow(run)
        refresh()
        return true
      } catch (err: unknown) {
        setError(asApiError(err))
        return false
      } finally {
        inFlight.current = false
        setBusy(null)
      }
    },
    [follow, refresh],
  )

  const start = useCallback(async () => {
    if (!draft) return
    if (await post(taskOf(draft))) setDraft(null)
  }, [draft, post])

  const load = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    setBusy('loading')
    try {
      const status = await api.warmCommands()
      if (status.state === 'ready') say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.loaded' }, tone: 'ok' })
      else say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.loadRefused', params: { why: lightMsg(status.state) } }, tone: 'warn' })
    } catch (err: unknown) {
      const refused = asApiError(err)
      say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.loadRefused', params: { why: refusalWords(refused) } }, tone: 'warn' })
    } finally {
      inFlight.current = false
      setBusy(null)
    }
  }, [])

  // How the detector would route what the card says, asked a moment after the last keystroke: a free text analysis
  // (`/v1/diagnostics/route`), so a German word in a card filled by hand says "braucht das VLM" before Start.
  const object = draft?.edited.includes('object') || draft?.mode === 'manual' ? draft.object.trim() : null
  const phrase = draft?.place.kind === 'camera' && (draft.edited.includes('place') || draft.mode === 'manual') ? draft.place.phrase.trim() : null
  const draftId = draft?.id ?? null
  useEffect(() => {
    if (draftId === null || (!object && !phrase)) return
    let cancelled = false
    const id = setTimeout(() => {
      if (object) {
        api
          .routePreview(object)
          .then((route) => {
            if (!cancelled) setDraft((d) => (d && d.id === draftId && d.object.trim() === object ? { ...d, objectRoute: route } : d))
          })
          .catch(() => undefined)
      }
      if (phrase) {
        api
          .routePreview(phrase)
          .then((route) => {
            if (!cancelled)
              setDraft((d) => (d && d.id === draftId && d.place.kind === 'camera' && d.place.phrase.trim() === phrase ? { ...d, placeRoute: route } : d))
          })
          .catch(() => undefined)
      }
    }, ROUTE_DEBOUNCE_MS)
    return () => {
      cancelled = true
      clearTimeout(id)
    }
  }, [draftId, object, phrase])

  return { draft, busy, error, lastSaid, read, update, open, discard, start, startTask: post, load }
}

/** A refusal in the conversation, in the reader's language: its code's words, or "no answer" for a dead server. */
function refusalWords(error: ApiError): Msg<string> {
  return error.httpStatus === 0 ? { key: 'ck.ready.noServer' } : refusalMsg(error.code)
}
