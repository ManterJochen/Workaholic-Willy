/**
 * From a sentence to a running task: the person's Enter, and where a person must look, a person's click (OD 8, OD 22;
 * build plan 4.2; the owner, 2026-10-08).
 *
 * 1. `read(text, source)`: the operator's words go into the conversation as they were said, and to the reader
 *    (`POST /v1/commands/parse`), which MOVES NOTHING. Without a reader (501, or 409 `vlm_not_loaded`) the card opens
 *    by hand with the reason; a sentence read as "stop" only says where the stop buttons are. A greeting ("Hallo
 *    Willy") opens no card: Willy greets back and hands the wave to `greet`, as the app config says
 *    (`CommandOut.greeting`): at once where it says `direct` (the owner, 2026-10-06), after the dialog's confirm where
 *    it says `confirm`, not at all where it says `off`.
 * 2. **Enter starts** where the settings say so (`TaskPrefs.start === 'enter'`, the default; the owner, 2026-10-08:
 *    Enter is the person's click). A reading the reader calls `startable` (an understood task, no note, its part and
 *    its place found in the sentence) is started at once, `POST /v1/task`, with the mode of the settings, and only
 *    when the server, asked afresh, says the cell is ready (`gate`). Any doubt opens the Understood card instead, and
 *    the chat says the first reason; a refused start opens it with the refusal above Start. The request is held from
 *    the parse to the start, so a double Enter is one parse and one task. A transcript starts the same way, on the
 *    person's Enter after reading it; the text box never sends anything by itself.
 * 3. Otherwise the reading becomes the Understood card (`Draft`), as before: `update(field, change)` records each
 *    field a person corrects, and `start()`, the click on Start, posts the card once, follows the 202 answer and
 *    clears the card. A refusal stays on the card, said in the reader's language; nothing is retried.
 *
 * A sort (the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die blaue") is read and started the same way,
 * every rule at once: the reply names each rule, Enter starts a clean reading with all of them, and a note on any rule
 * opens the card, which starts nothing by itself.
 *
 * Nothing here runs on a timer or a stream: every request is the direct answer to a person's key or click, and the
 * server's gates on `POST /v1/task` stand behind every start.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api, type CellFactsOut, type PosesOut, type RunOut, type TaskIn } from '../api/client'
import type { Lang } from '../i18n'
import { lightMsg, refusalMsg } from '../i18n/codes'
import type { Msg, ParamValue, Params } from '../i18n/types'
import { say } from '../model/chat'
import { DEFAULT_TASK, type TaskPrefs } from '../model/prefs'
import {
  doubts,
  draftFromReading,
  edit,
  firstMotion,
  groundedOf,
  manualDraft,
  MOTION_KEY,
  rulesMsg,
  taskOf,
  type Draft,
  type DraftField,
  type DraftRule,
} from './draft'
import { gate } from './gate'
import type { WaveAnswer } from './motions'

/** The reader's refusals that open the card by hand (OD 22): there is no reader to ask. */
const NO_READER = new Set(['vlm_unavailable', 'vlm_not_loaded', 'vlm_model_missing'])

/** The card's fields that change what the picks ground, and so the pick's route. */
const PICK_FIELDS: ReadonlySet<DraftField> = new Set(['object', 'which', 'source'])

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
  /**
   * Read a sentence. On the person's Enter (`atOnce`, the default) a clean reading starts at once where the settings
   * say so: THIS MAY MOVE. The card's "Satz nochmal lesen" passes `false`: a button that does not say it starts never
   * does.
   */
  read(text: string, source: 'typed' | 'spoken', atOnce?: boolean): Promise<void>
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
  task = DEFAULT_TASK,
  facts = null,
  poses = null,
}: {
  lang: Lang
  follow: (run: RunOut) => void
  /** Poll the cell now: a run a click just started is then named by every chip at once. */
  refresh?: () => void
  /** THIS MOVES: the wave at a greeting (`Motions.wave`), after the dialog's confirm where `ask`. None: no wave. */
  greet?: (ask: boolean) => Promise<WaveAnswer>
  /** What the person set beforehand (Settings, "Auftrag"): whether Enter starts, the mode, the defaults. */
  task?: TaskPrefs
  /** The cell's facts: the first motion Enter names, and whether the cell detects with the language model. */
  facts?: CellFactsOut | null
  /** The taught poses, which a card's place and return are checked against before Enter starts. */
  poses?: PosesOut | null
}): CommandModel {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [busy, setBusy] = useState<CommandBusy>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [lastSaid, setLastSaid] = useState<{ text: string; source: 'typed' | 'spoken' } | null>(null)
  /** One request at a time, whatever renders in between: a double Enter is one parse (and one task), a double click
   *  one task. */
  const inFlight = useRef(false)

  /** THIS MOVES. Post one task and follow it; the refusal, where there is one, stays for the card. No guard: the
   *  caller holds `inFlight`. */
  const send = useCallback(
    async (body: TaskIn): Promise<ApiError | null> => {
      setError(null)
      try {
        const run = await api.task(body)
        follow(run)
        refresh()
        return null
      } catch (err: unknown) {
        const refused = asApiError(err)
        setError(refused)
        return refused
      }
    },
    [follow, refresh],
  )

  const read = useCallback(
    async (text: string, source: 'typed' | 'spoken', atOnce = true) => {
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
          setDraft(draftFromReading(out, said, task))
        } else if (task.start === 'enter' && atOnce) {
          // Enter is the person's click (the owner, 2026-10-08): a clean reading on a ready cell starts now, and any
          // doubt opens the card with its first reason. A sort starts with every rule, or not at all.
          const card = draftFromReading(out, said, task)
          const checked = await gate(card, { facts, poses })
          const why = [...doubts(out), ...checked.why]
          const rules = sortWords(card, poses)
          if (why.length > 0) {
            say({ who: 'willy', kind: 'reply', msg: reply('ck.reply.check', { why: why[0] }, rules), tone: 'warn' })
            setDraft(card)
          } else {
            const motion: Msg<string> = { key: MOTION_KEY[firstMotion(checked.facts)] }
            say({
              who: 'willy',
              kind: 'reply',
              msg: checked.countdown
                ? reply('ck.reply.startingCountdown', { motion, countdown: { key: 'ck.start.countdown' } }, rules)
                : reply('ck.reply.starting', { motion }, rules),
              tone: 'info',
            })
            if (out.scope && out.scope !== card.scope) {
              const scope = { key: card.scope === 'until_empty' ? 'scope.inline.until_empty' : 'scope.inline.once' }
              say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.scopeFromSettings', params: { scope } }, tone: 'info' })
            }
            setBusy('starting')
            const refused = await send(taskOf(card))
            if (refused) {
              say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.notStarted', params: { why: refusalWords(refused) } }, tone: 'warn' })
              setDraft(card)
            } else {
              setDraft(null)
            }
          }
        } else {
          const card = draftFromReading(out, said, task)
          say({ who: 'willy', kind: 'reply', msg: reply('ck.reply.understood', undefined, sortWords(card, poses)), tone: 'info' })
          setDraft(card)
        }
      } catch (err: unknown) {
        const refused = asApiError(err)
        if (NO_READER.has(refused.code)) {
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.manual' }, tone: 'warn' })
          setDraft(manualDraft(said, refused.code, refused.message, task))
        } else {
          say({ who: 'willy', kind: 'reply', msg: { key: 'ck.reply.refused', params: { why: refusalWords(refused) } }, tone: 'error' })
        }
      } finally {
        inFlight.current = false
        setBusy(null)
      }
    },
    [lang, greet, task, facts, poses, send],
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
    async (body: TaskIn): Promise<boolean> => {
      if (inFlight.current) return false
      inFlight.current = true
      setBusy('starting')
      try {
        return (await send(body)) === null
      } finally {
        inFlight.current = false
        setBusy(null)
      }
    },
    [send],
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
  // (`/v1/diagnostics/route`), so a German word in a card filled by hand says "braucht das VLM" before Start. The pick's
  // route is that of what its picks ground, as the server's guard judges it (`groundedOf`): the one part singled out,
  // else the object where the parts lie.
  const object = draft && (draft.mode === 'manual' || draft.edited.some((field) => PICK_FIELDS.has(field))) ? groundedOf(draft) : null
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
            if (!cancelled) setDraft((d) => (d && d.id === draftId && groundedOf(d) === object ? { ...d, objectRoute: route } : d))
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

  // The same for a sort's further rules once a person changed them: each phrase whose route is not known, what the
  // rule's picks ground and what the camera finds for it. A row's edit forgets its route, so it is asked again.
  const ruleKey = JSON.stringify(draft && (draft.mode === 'manual' || draft.edited.includes('rules')) ? routesToAsk(draft.moreRules) : [])
  useEffect(() => {
    const asked = JSON.parse(ruleKey) as RouteAsk[]
    if (draftId === null || asked.length === 0) return
    let cancelled = false
    const id = setTimeout(() => {
      for (const ask of asked) {
        api
          .routePreview(ask.phrase)
          .then((route) => {
            if (cancelled) return
            setDraft((d) => {
              const rule = d && d.id === draftId ? d.moreRules[ask.index] : undefined
              if (!d || !rule || !routesToAsk([rule]).some((still) => still.kind === ask.kind && still.phrase === ask.phrase)) return d
              const moreRules = d.moreRules.map((r, i) => (i !== ask.index ? r : ask.kind === 'object' ? { ...r, objectRoute: route } : { ...r, placeRoute: route }))
              return { ...d, moreRules }
            })
          })
          .catch(() => undefined)
      }
    }, ROUTE_DEBOUNCE_MS)
    return () => {
      cancelled = true
      clearTimeout(id)
    }
  }, [draftId, ruleKey])

  return { draft, busy, error, lastSaid, read, update, open, discard, start, startTask: post, load }
}

/** A refusal in the conversation, in the reader's language: its code's words, or "no answer" for a dead server. */
function refusalWords(error: ApiError): Msg<string> {
  return error.httpStatus === 0 ? { key: 'ck.ready.noServer' } : refusalMsg(error.code, error.detail)
}

/** One route a further rule's row waits for: its index, which phrase, and the phrase as it was asked. */
interface RouteAsk {
  readonly index: number
  readonly kind: 'object' | 'place'
  readonly phrase: string
}

/** The phrases of further rules whose route is not known: what each rule's picks ground, what the camera finds for it. */
function routesToAsk(rules: readonly DraftRule[]): RouteAsk[] {
  const asks: RouteAsk[] = []
  rules.forEach((rule, index) => {
    const grounded = groundedOf(rule)
    if (rule.objectRoute === null && grounded !== '') asks.push({ index, kind: 'object', phrase: grounded })
    if (rule.place.kind === 'camera' && rule.placeRoute === null && rule.place.phrase.trim() !== '') {
      asks.push({ index, kind: 'place', phrase: rule.place.phrase.trim() })
    }
  })
  return asks
}

/** A sort's rules as the reply names them; `null` for a reading of one kind, whose reply is the one it always was. */
function sortWords(card: Draft, poses: PosesOut | null): ParamValue | null {
  return card.moreRules.length > 0 ? rulesMsg(card, poses) : null
}

/** Willy's reply to a reading: a sort's names every rule (`<key>.rules`, the owner, 2026-10-09). */
function reply(
  key: 'ck.reply.understood' | 'ck.reply.check' | 'ck.reply.starting' | 'ck.reply.startingCountdown',
  params: Params | undefined,
  rules: ParamValue | null,
): Msg<string> {
  if (rules !== null) return { key: `${key}.rules`, params: { ...params, rules } }
  return params === undefined ? { key } : { key, params }
}
