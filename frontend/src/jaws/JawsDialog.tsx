/**
 * The jaws question, global (build plan 1.6, item 10): whenever a toggle hand asks where its jaws stand, at Connect or
 * at a check before a Restart, the question is the one thing the console shows above everything else, and it is
 * answered here, in the browser.
 *
 * **Exactly the server's choices, and never a default.** The hand has no sensor: every change of its output moves the
 * jaws and nothing reads them back, so only a person looking at them can say where they stand. The dialog offers the
 * choices the waiting question carries, in its order, and nothing else; none is focused, lit or pre-selected when it
 * opens (the focus goes to the dialog itself, so Enter answers nothing), and it cannot be dismissed into an answer:
 * Escape does nothing. A question nobody answers is refused when it expires, never taken as "open", and the dialog says
 * so; a question past its expiry is not offered at all.
 *
 * **"Jetzt öffnen" is a motion.** It is one switch of the output, which opens the jaws and releases what they hold, so
 * it is drawn in the arming red like everything that arms something (Verbinden, "Arm freigeben"), and the dialog
 * tells the person to hold the part first. The other answers are plain filled buttons of the same size, "Abbrechen"
 * included: none is lit.
 *
 * **Where the question comes from.** `GET /v1/cell/jaws` is the authority (it shows only the question that waits), read
 * every second while one is flagged; the cell stream opens the dialog the moment a question is published, before that
 * read comes back, and closes it the moment the question was answered or ended. A read that left before an answer and
 * lands after it is stale and is dropped (a read generation), and a question this tab answered is never shown again,
 * so an answered question cannot come back between two reads. `CellOut.jaws_question` true with no waiting question
 * means the hand is acting on an answer (the one change and its stroke), or a check is reading the latch and the
 * controller: the dialog says so and offers nothing. The text is built from the stage, the output and the attempt,
 * never from the question's `text` (that is the hand's terminal prompt, with its key letters); the server's own
 * English sentences (`reason`, `why_again`) are details of the tech view.
 *
 * Above every other layer (`jaws.css`): the Diagnostics drawer and a confirm dialog never cover a waiting question.
 */

import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'

import { ApiError, api, type JawQuestionOut } from '../api/client'
import type { JawsChoice } from '../api/codes'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { whereMsg } from '../i18n/codes'
import { Icon } from '../icons'
import { pendingQuestion, type JawsQuestionView } from '../model/cellModel'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { JAWS } from './i18n'
import './jaws.css'

/** How often the waiting question is read again while one is flagged. */
export const JAWS_READ_MS = 1000

/** One question, whichever way it came (the server's read or the cell stream). */
interface Question {
  readonly id: string
  readonly stage: 'where' | 'open_now'
  readonly at: 'connect' | 'check'
  readonly where: string
  readonly reason: string
  readonly choices: readonly JawsChoice[]
  readonly attempt: number
  readonly of: number
  readonly whyAgain: string
  readonly expiresAt: number | null
}

/** The last answer of the server: the question that waited (or none), and when it was read (browser ms). */
interface Served {
  readonly question: JawQuestionOut | null
  readonly at: number
}

function fromServer(q: JawQuestionOut): Question {
  return {
    id: q.question_id,
    stage: q.stage === 'open_now' ? 'open_now' : 'where',
    at: q.at === 'check' ? 'check' : 'connect',
    where: q.where,
    reason: q.reason,
    choices: q.choices,
    attempt: q.attempt,
    of: q.of,
    whyAgain: q.why_again,
    expiresAt: Number.isFinite(q.expires_at) ? q.expires_at : null,
  }
}

function fromStream(q: JawsQuestionView): Question {
  return {
    id: q.id,
    stage: q.stage,
    at: q.at,
    where: q.where,
    reason: q.reason,
    choices: q.choices,
    attempt: q.attempt,
    of: q.of,
    whyAgain: q.whyAgain,
    expiresAt: q.expiresAt,
  }
}

/**
 * The question to show: the server's last read, unless the stream published a newer one since (it opens the dialog
 * before the next read), and never one the stream says was answered or ended.
 */
function pick(served: Served | null, streamQ: JawsQuestionView | null, gone: ReadonlySet<string>): Question | null {
  let q: Question | null
  if (!served) q = streamQ ? fromStream(streamQ) : null
  else if (streamQ && streamQ.id !== served.question?.question_id && streamQ.askedAt * 1000 >= served.at - 1000) {
    q = fromStream(streamQ)
  } else q = served.question ? fromServer(served.question) : null
  return q && !gone.has(q.id) ? q : null
}

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])'

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

export default function JawsDialog() {
  const t = useT(JAWS)
  const { view } = usePrefs()
  const { cell, stream, refresh } = useCell()
  const titleId = useId()
  const askId = useId()
  const dialogRef = useRef<HTMLElement | null>(null)
  const [now, setNow] = useState(() => Date.now() / 1000)
  const [served, setServed] = useState<Served | null>(null)
  const [sending, setSending] = useState<JawsChoice | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  /**
   * Bumped by every answer sent: a read that left before it is older than the answer and is dropped when it lands, so a
   * question already answered cannot come back between two reads (the server refuses a click on it, but it would look
   * offered).
   */
  const generation = useRef(0)
  /** The questions this tab answered and the server took: never shown again, whoever names them. */
  const [answered, setAnswered] = useState<ReadonlySet<string>>(() => new Set())

  const streamQ = pendingQuestion(stream, now)
  const flagged = cell?.jaws_question === true
  const active = flagged || streamQ !== null || sending !== null
  const streamId = streamQ?.id ?? ''

  // The server's read, every second while a question is flagged; again at once when the stream names a new one.
  useEffect(() => {
    if (!active) return
    let stopped = false
    const read = () => {
      const asked = generation.current
      api
        .jaws()
        .then((answer) => {
          if (!stopped && asked === generation.current) setServed({ question: answer.question ?? null, at: Date.now() })
        })
        .catch(() => {
          // A server without the read (or none at all): the stream's question stands alone.
          if (!stopped && asked === generation.current) setServed(null)
        })
    }
    read()
    const id = setInterval(read, JAWS_READ_MS)
    return () => {
      stopped = true
      clearInterval(id)
      setServed(null)
    }
  }, [active, streamId])

  // The clock of the expiry (a question past it is offered no more). The countdown a person reads keeps a clock of its
  // own (`ExpiresIn`), started when the question is shown.
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(id)
  }, [active])

  const gone = new Set<string>(answered)
  if (stream.lastAnswer) gone.add(stream.lastAnswer.id)
  if (stream.lastEnded) gone.add(stream.lastEnded.id)
  const question = active ? pick(served, streamQ, gone) : null
  const expired = question !== null && question.expiresAt !== null && now > question.expiresAt
  const shown: 'question' | 'sending' | 'busy' | null = question
    ? 'question'
    : sending
      ? 'sending'
      : flagged
        ? 'busy'
        : null
  const shownKey = shown === 'question' ? `q:${question?.id}` : shown ?? ''

  // The focus goes to the dialog, never to a choice: Enter answers nothing.
  useEffect(() => {
    if (shownKey) dialogRef.current?.focus()
  }, [shownKey])

  const answer = useCallback(
    async (choice: JawsChoice) => {
      if (!question || sending || expired) return
      setSending(choice)
      setError(null)
      // Every read already on its way is older than this answer.
      generation.current += 1
      try {
        const next = await api.answerJaws(question.id, choice)
        const id = question.id
        setAnswered((before) => new Set(before).add(id))
        generation.current += 1
        setServed({ question: next.question ?? null, at: Date.now() })
      } catch (err) {
        setError(asApiError(err))
        generation.current += 1
        const asked = generation.current
        api
          .jaws()
          .then((read) => {
            if (asked === generation.current) setServed({ question: read.question ?? null, at: Date.now() })
          })
          .catch(() => {
            if (asked === generation.current) setServed(null)
          })
      } finally {
        setSending(null)
        refresh()
      }
    },
    [question, sending, expired, refresh],
  )

  if (!shown) return null

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') {
      // Not a way out: the question waits for one of its own answers, or it expires and is refused.
      event.preventDefault()
      event.stopPropagation()
      return
    }
    if (event.key !== 'Tab' || !dialogRef.current) return
    const items = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE))
    if (items.length === 0) {
      event.preventDefault()
      return
    }
    const first = items[0]
    const last = items[items.length - 1]
    if (event.shiftKey && (document.activeElement === first || document.activeElement === dialogRef.current)) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  const where = question ? whereMsg(question.where) : whereMsg(cell?.hand?.where)

  return (
    <div className="jd-backdrop">
      <section
        ref={dialogRef}
        className="jd-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={askId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <header className="jd-head">
          <Icon name="gripper" size={22} className="jd-icon" />
          <h2 id={titleId}>{t('jd.title')}</h2>
          {question && <span className="jd-context">{t(question.at === 'check' ? 'jd.at.check' : 'jd.at.connect')}</span>}
        </header>

        {/* What is asked, announced when it changes; the answers and the countdown stay outside it. */}
        <div className="jd-ask" id={askId} role="status" aria-label={t('jd.title')}>
          {shown === 'question' && question && (
            <>
              <p className="jd-q">{t(question.stage === 'open_now' ? 'jd.ask.open_now' : 'jd.ask.where', { where })}</p>
              {/* Why only a person can say where the jaws stand belongs to that question; opening needs the part held. */}
              {question.stage === 'open_now' ? (
                <p className="jd-hold">{t('jd.why.open_now')}</p>
              ) : (
                <p className="jd-why">{t('jd.why.where', { where })}</p>
              )}
              <p className="jd-why">{expired ? t('jd.expired') : t('jd.noAnswer')}</p>
            </>
          )}
          {shown === 'sending' && <p className="jd-why">{t('jd.sending')}</p>}
          {shown === 'busy' && <p className="jd-why">{t('jd.busy')}</p>}
        </div>

        {shown === 'question' && question && !expired && (
          <div className="jd-choices" role="group" aria-label={t('jd.choices')}>
            {question.choices.map((choice) => (
              <button
                key={choice}
                type="button"
                data-choice={choice}
                className={choice === 'open_now' ? 'danger big' : 'big'}
                disabled={sending !== null}
                onClick={() => void answer(choice)}
              >
                <span className="jd-choice-label">{t(`jd.label.${choice}`)}</span>
                <small>{t(`jd.choice.${choice}`, { where })}</small>
              </button>
            ))}
          </div>
        )}

        {error && <ErrorBanner error={error} />}

        {question && (
          <footer className="jd-foot">
            {question.expiresAt !== null && !expired && <ExpiresIn key={question.id} expiresAt={question.expiresAt} />}
            {question.attempt > 1 && <span>{t('jd.attempt', { attempt: question.attempt, of: question.of })}</span>}
          </footer>
        )}

        {view === 'tech' && question && (
          <div className="jd-tech">
            {question.reason && <span>{t('jd.tech.reason', { text: question.reason })}</span>}
            {question.whyAgain && <span>{t('jd.tech.again', { text: question.whyAgain })}</span>}
            <span>{t('jd.tech.id', { id: question.id })}</span>
          </div>
        )}
      </section>
    </div>
  )
}

/**
 * "läuft ab in N s", on a clock of its own that starts when the question is shown (keyed by the question, so each
 * question starts it again). The dialog's own clock is only read every second while a question waits, so on a page
 * that stood open for an hour its first second drew the question's 120 s plus that hour ("läuft ab in 3720 s").
 */
function ExpiresIn({ expiresAt }: { expiresAt: number }) {
  const t = useT(JAWS)
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(id)
  }, [])
  const left = Math.max(0, Math.round(expiresAt - now))
  return <span className={left <= 20 ? 'caution' : undefined}>{t('jd.expires', { seconds: left })}</span>
}
