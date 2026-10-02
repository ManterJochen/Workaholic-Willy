/**
 * Teach one pose by freedrive (build plan 1.8, OD 15, Q14): payload -> free -> screening -> verdict.
 *
 * **Before anything is freed** the person names the pose, sees where it is written (the cell's own layer, Q10) and
 * confirms the payload the controller compensates for, because a wrong payload makes a freed arm sink or rise in a
 * person's hands (read again on request: it changes when the tool does). A place pose says where the fingertips go:
 * where the part's bottom is let go; the robot raises every part by its hang (item 7). Only the red "Arm freigeben"
 * button frees the arm, and beside it, while it cannot, the dialog says what is still missing.
 *
 * **While the arm is being freed** (the request is on its way) the dialog cannot be closed: no "Abbrechen", no Escape,
 * because the arm may be free a moment later. If the dialog goes away all the same (a navigation) and the server then
 * frees the arm, the session is held at once (cancel) and never followed: no arm is ever left free without a dialog.
 *
 * **While the arm is free** the dialog polls the session four times a second, and every poll is the browser's
 * heartbeat: without one for 3 s the server holds the arm, but only once it stands still, never while it moves in a
 * person's hands. The same holds at the 5-minute limit; the time left is shown while the arm is free, and from 30 s on
 * it says what happens at 0 (announced once to a screen reader). "Speichern" captures where the arm stands once it is
 * still, then the pose is screened (the collision check and the planner, up to a minute); "Halten" holds at once and
 * saves nothing (the orange of a halt). A tab that closes sends the cancel beacon (`pagehide`), and a dialog that goes
 * away while the arm is free cancels the session too.
 *
 * **The verdict**: a pose the screen clears is written; a refused one is not, and the nearest clear joints are shown,
 * or, with none nearby, that it is taught elsewhere; a session the server ended for another reason (a lapse, the time
 * limit, a Disconnect) says so by its stop code. After a teach the next motion counts down 3 s (item 11). Numbers are
 * the reader's (`-60,1` in German), the label's quotation marks the reader's language's.
 *
 * The teach is a run: it is followed (`useRun().follow`) so the top bar says a teach runs, and every moving route of
 * the server refuses while it does. Where voice output is on (OD 18, off unless switched on), the dialog says the 30 s
 * warning aloud once, with the browser's own speech synthesis: the person's hands are on the arm and their eyes on the
 * tool, and the cockpit, whose voice says a run's key moments, is not on this page.
 */

import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'

import {
  ApiError,
  api,
  sendTeachCancelBeacon,
  type PayloadOut,
  type PosesOut,
  type RunOut,
  type TeachStateOut,
} from '../api/client'
import { isStopCode } from '../api/codes'
import { ErrorBanner } from '../components/ui'
import { useLiveFrame } from '../demo/live'
import { useT, type Lang } from '../i18n'
import { stopMsg, stopSayMsg } from '../i18n/codes'
import { Icon } from '../icons'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { fileName, jointsText, labelRefusal, nameRefusal, readoutText, wristRig } from './format'
import { SETUP } from './i18n'
import VerdictChip from './VerdictChip'

/** How often the session is polled while it lasts: the heartbeat (the server holds after 3 s without one). */
export const TEACH_POLL_MS = 250

/** How often the wrist camera is read during a teach: slower than the cockpit's, the arm is in a person's hands. */
const LIVE_MS = 500

/** The last seconds of the free time, said as a warning. */
const WARN_S = 30

const ENDED = new Set(['saved', 'refused', 'not_saved', 'ended'])

/** What the session's state says to the person holding the arm. */
const STATE_WORDS: Partial<Record<TeachStateOut['state'], `td.state.${'freeing' | 'free' | 'holding_when_still' | 'holding' | 'screening'}`>> = {
  freeing: 'td.state.freeing',
  free: 'td.state.free',
  holding_when_still: 'td.state.holding_when_still',
  holding: 'td.state.holding',
  screening: 'td.state.screening',
}

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

/** The voice's locale for the UI language: the tags the console formats numbers and times with. */
const VOICE_LOCALE: Record<Lang, string> = { de: 'de-DE', en: 'en-GB' }

/**
 * One sentence said with the browser's own speech synthesis (OD 18; no dependency). Never throws: a browser that
 * cannot speak says nothing, and the dialog goes on.
 */
function sayAloud(text: string, lang: Lang): void {
  try {
    if (typeof window.speechSynthesis === 'undefined' || typeof window.SpeechSynthesisUtterance === 'undefined') return
    const utterance = new window.SpeechSynthesisUtterance(text)
    utterance.lang = VOICE_LOCALE[lang]
    window.speechSynthesis.speak(utterance)
  } catch {
    /* no speech in this browser after all */
  }
}

export interface TeachDialogProps {
  /** What `GET /v1/poses` said: the file a pose is written to, and the names already taken. */
  poses: PosesOut
  onClose: () => void
}

interface Session {
  readonly runId: string
  readonly token: string
}

export default function TeachDialog({ poses, onClose }: TeachDialogProps) {
  const t = useT(SETUP)
  const titleId = useId()
  const { view, voiceOut } = usePrefs()
  const { facts, refresh } = useCell()
  const { follow } = useRun()
  const nameRef = useRef<HTMLInputElement | null>(null)

  // ── the form ───────────────────────────────────────────────────────────────────────────────────────────────
  const [name, setName] = useState('')
  const [label, setLabel] = useState('')
  const [role, setRole] = useState<'place' | 'other'>('place')
  const [makeDefault, setMakeDefault] = useState(false)
  const [replace, setReplace] = useState(false)
  const [payloadOk, setPayloadOk] = useState(false)
  const [payload, setPayload] = useState<PayloadOut | null>(null)
  const [payloadError, setPayloadError] = useState<ApiError | null>(null)
  const [payloadTick, setPayloadTick] = useState(0)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)

  // ── the session ────────────────────────────────────────────────────────────────────────────────────────────
  const [session, setSession] = useState<Session | null>(null)
  const [state, setState] = useState<TeachStateOut | null>(null)
  const [record, setRecord] = useState<RunOut | null>(null)
  const [lost, setLost] = useState(false)
  const [acting, setActing] = useState<'save' | 'hold' | null>(null)
  const [wasFree, setWasFree] = useState(false)

  const ended = lost || (state !== null && ENDED.has(state.state))
  const free = session !== null && !ended

  // The payload, read when the dialog opens and again on request (it changes when the tool changes).
  useEffect(() => {
    let cancelled = false
    api
      .teachPayload()
      .then((read) => {
        if (cancelled) return
        setPayload(read)
        setPayloadError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setPayload(null)
        setPayloadError(asApiError(err))
      })
    return () => {
      cancelled = true
    }
  }, [payloadTick])

  useEffect(() => {
    nameRef.current?.focus()
  }, [])

  // The heartbeat: four polls a second while the session lasts, one at a time.
  useEffect(() => {
    if (!session || ended) return
    let stopped = false
    let inFlight = false
    const poll = async () => {
      if (stopped || inFlight) return
      inFlight = true
      try {
        const next = await api.teachState(session.runId, session.token)
        if (stopped) return
        setState(next)
        if (next.state === 'free' || next.state === 'holding_when_still') setWasFree(true)
      } catch (err) {
        if (!stopped && err instanceof ApiError && err.httpStatus === 404) setLost(true)
      } finally {
        inFlight = false
      }
    }
    void poll()
    const id = setInterval(() => void poll(), TEACH_POLL_MS)
    return () => {
      stopped = true
      clearInterval(id)
    }
  }, [session, ended])

  // A tab that closes while the arm is free holds it: the one request a closing page still sends.
  useEffect(() => {
    if (!session || ended) return
    const leave = () => {
      sendTeachCancelBeacon(session.runId, session.token)
    }
    window.addEventListener('pagehide', leave)
    return () => window.removeEventListener('pagehide', leave)
  }, [session, ended])

  // A dialog that goes away while the session lasts holds the arm too.
  const open = useRef<{ session: Session | null; ended: boolean }>({ session: null, ended: false })
  useEffect(() => {
    open.current = { session, ended }
  }, [session, ended])
  /** The dialog is on screen: a free request answered after it went away is held at once (`start`). */
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      const { session: left, ended: done } = open.current
      if (left && !done) void api.teachCancel(left.runId, left.token).catch(() => undefined)
    }
  }, [])

  // Once over: the run's own record says how it ended (a lapse, the time limit, a Disconnect).
  useEffect(() => {
    if (!session || !ended) return
    let cancelled = false
    api
      .run(session.runId)
      .then((run) => {
        if (!cancelled) setRecord(run)
      })
      .catch(() => undefined)
    refresh()
    return () => {
      cancelled = true
    }
  }, [session, ended, refresh])

  const liveFrame = useLiveFrame({ rig: wristRig(facts), maxWidth: 640, intervalMs: LIVE_MS, enabled: free })

  // ── actions ────────────────────────────────────────────────────────────────────────────────────────────────
  const exists = (poses.poses ?? []).some((pose) => pose.name === name)
  const nameBad = name !== '' && nameRefusal(name)
  const labelBad = label !== '' && labelRefusal(label)
  const canFree =
    !starting &&
    name !== '' &&
    !nameRefusal(name) &&
    !labelRefusal(label) &&
    payloadOk &&
    payload !== null &&
    (!exists || replace) &&
    Boolean(poses.target_file)

  const start = async () => {
    if (!canFree) return
    setStarting(true)
    setError(null)
    try {
      const seen = payload?.readable ? { mass_kg: payload.mass_kg ?? null, cog_mm: payload.cog_mm ?? null } : null
      const answer = await api.teach({
        name,
        label: label.trim(),
        role,
        replace: exists && replace,
        make_default_place: role === 'place' && makeDefault,
        payload_seen: seen,
      })
      if (!alive.current) {
        // The dialog went away while the arm was being freed: nobody holds it, so it is held at once, never followed.
        void api.teachCancel(answer.run.id, answer.token).catch(() => undefined)
        return
      }
      follow(answer.run)
      setSession({ runId: answer.run.id, token: answer.token })
    } catch (err) {
      if (!alive.current) return
      const refused = asApiError(err)
      setError(refused)
      if (refused.code === 'payload_changed') {
        setPayloadOk(false)
        setPayloadTick((n) => n + 1)
      }
    } finally {
      if (alive.current) {
        setStarting(false)
        refresh()
      }
    }
  }

  /** Read the controller's payload again (the tool changed): what the person confirmed was the old one. */
  const readPayloadAgain = () => {
    setPayloadOk(false)
    setPayloadTick((n) => n + 1)
  }

  const save = async () => {
    if (!session || acting) return
    setActing('save')
    setError(null)
    try {
      setState(await api.teachCapture(session.runId, session.token))
    } catch (err) {
      setError(asApiError(err))
    } finally {
      setActing(null)
    }
  }

  const hold = async () => {
    if (!session || acting === 'hold') return
    setActing('hold')
    setError(null)
    try {
      setState(await api.teachCancel(session.runId, session.token))
    } catch (err) {
      setError(asApiError(err))
    } finally {
      setActing(null)
    }
  }

  const checkJaws = async () => {
    setError(null)
    try {
      await api.checkJaws()
    } catch (err) {
      setError(asApiError(err))
    } finally {
      refresh()
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    // Escape closes only the form: a session in progress ends with Save or Hold, never with a key, and a free request
    // on its way may free the arm a moment later, so the form stays until it is answered.
    if (event.key !== 'Escape') return
    event.stopPropagation()
    if (!session && !starting) onClose()
  }

  const file = poses.target_file ? (view === 'tech' ? poses.target_file : fileName(poses.target_file)) : ''

  // ── what is drawn ──────────────────────────────────────────────────────────────────────────────────────────
  // The time left counts only while the arm is free: once it is being held (a lapse, the limit, Save, Hold) there is
  // nothing left to save in, so no "noch 0 s" stands beside a Save that cannot be pressed.
  const timeLeft = state?.state === 'free' ? (state.time_left_s ?? null) : null
  const warned = timeLeft !== null && timeLeft <= WARN_S
  const seconds = timeLeft === null ? 0 : Math.max(0, Math.round(timeLeft))
  const left =
    timeLeft === null
      ? null
      : warned
        ? t('td.left.warn', { s: seconds })
        : timeLeft <= 60
          ? t('td.left.short', { s: seconds })
          : t('td.left', { time: t.fmt.duration(timeLeft) })

  // The last 30 s, said aloud once a session where voice output is on: when they begin, never again each second.
  const warnedAloud = useRef<string | null>(null)
  useEffect(() => {
    if (!warned || !session || warnedAloud.current === session.runId) return
    warnedAloud.current = session.runId
    if (voiceOut) sayAloud(t('td.left.alert'), t.lang)
  }, [warned, session, voiceOut, t])

  // What still keeps the red button off, said beside it.
  const missing: string[] = []
  if (name === '' || nameRefusal(name)) missing.push(t('td.missing.name'))
  if (labelRefusal(label)) missing.push(t('td.missing.label'))
  if (!payloadOk || payload === null) missing.push(t('td.missing.payload'))
  if (exists && !replace) missing.push(t('td.missing.replace'))
  if (!poses.target_file) missing.push(t('td.missing.file'))

  const form = (
    <div className="td-form">
      <div className="td-fields">
        <label className="td-field">
          <span className="td-field-name">{t('td.name')}</span>
          <input
            ref={nameRef}
            type="text"
            className="mono"
            value={name}
            maxLength={32}
            autoComplete="off"
            spellCheck={false}
            aria-invalid={nameBad || undefined}
            onChange={(e) => setName(e.target.value.trim())}
          />
          <small className={nameBad ? 'caution' : undefined}>{nameBad ? t('td.name.bad') : t('td.name.help')}</small>
        </label>
        <label className="td-field">
          <span className="td-field-name">{t('td.label')}</span>
          <input
            type="text"
            value={label}
            maxLength={40}
            autoComplete="off"
            aria-invalid={labelBad || undefined}
            onChange={(e) => setLabel(e.target.value)}
          />
          <small className={labelBad ? 'caution' : undefined}>{labelBad ? t('td.label.bad') : t('td.label.help')}</small>
        </label>
      </div>

      <fieldset className="td-role">
        <legend>{t('td.role')}</legend>
        <label className="check">
          <input type="radio" name={`${titleId}-role`} checked={role === 'place'} onChange={() => setRole('place')} />
          {t('td.role.place')}
        </label>
        <label className="check">
          <input type="radio" name={`${titleId}-role`} checked={role === 'other'} onChange={() => setRole('other')} />
          {t('td.role.other')}
        </label>
        {role === 'place' && (
          <label className="check">
            <input type="checkbox" checked={makeDefault} onChange={(e) => setMakeDefault(e.target.checked)} />
            {t('td.makeDefault')}
          </label>
        )}
        {exists && (
          <label className="check">
            <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
            {t('td.replace', { name })}
          </label>
        )}
      </fieldset>

      {role === 'place' && <p className="td-instruction">{t('td.place.instruction')}</p>}

      <section className="td-payload" aria-label={t('td.payload.title')}>
        <h4>{t('td.payload.title')}</h4>
        {payloadError ? (
          <ErrorBanner error={payloadError} onRetry={() => setPayloadTick((n) => n + 1)} />
        ) : !payload ? (
          <p className="small faint">{t('td.payload.reading')}</p>
        ) : payload.readable ? (
          <p className="td-payload-value mono">
            {t('td.payload.value', {
              kg: payload.mass_kg ?? null,
              x: payload.cog_mm?.[0] ?? null,
              y: payload.cog_mm?.[1] ?? null,
              z: payload.cog_mm?.[2] ?? null,
            })}
          </p>
        ) : (
          <p className="td-payload-value">{t('td.payload.unreadable')}</p>
        )}
        <p className="small dim">{t('td.payload.why')}</p>
        {view === 'tech' && payload?.source && <p className="small mono dim">{payload.source}</p>}
        {payload && (
          <div>
            <button type="button" className="ghost td-payload-again" disabled={starting} onClick={readPayloadAgain}>
              {t('td.payload.again')}
            </button>
          </div>
        )}
        <label className="check td-confirm">
          <input
            type="checkbox"
            checked={payloadOk}
            disabled={!payload}
            onChange={(e) => setPayloadOk(e.target.checked)}
          />
          {t('td.payload.confirm')}
        </label>
      </section>

      <p className="td-target">{poses.target_file ? t('td.target', { file }) : t('ps.target.none')}</p>
      <p className="small dim">{t('td.freedrive')}</p>
    </div>
  )

  const sessionView = state && !ended && (
    <div className="td-session">
      <div className="td-live stage hud-frame">
        {liveFrame.src ? (
          <img src={liveFrame.src} alt={t('td.live')} />
        ) : (
          <div className="td-live-none">
            <Icon name="camera" size={26} />
            <span>{t('td.live.none')}</span>
          </div>
        )}
        <span className="td-live-badge">{t('td.live')}</span>
      </div>
      <div className="td-readout">
        <p className={`td-state ${state.state}`} aria-live="polite">
          {t(STATE_WORDS[state.state] ?? 'td.state.holding')}
        </p>
        {state.outside && (
          <div className="banner error td-outside" role="alert">
            <div className="body">{t('td.outside')}</div>
          </div>
        )}
        {left && <p className={`td-left${warned ? ' warn' : ''}`}>{left}</p>}
        {/* Said once to a screen reader, when the last 30 s begin: the seconds under it are not read out each time. */}
        {warned && (
          <p className="sr-only" role="alert">
            {t('td.left.alert')}
          </p>
        )}
        <dl className="kv td-kv">
          <dt>{t('td.joints')}</dt>
          <dd className="mono">{state.joints_deg ? jointsText(state.joints_deg, t.lang) : '—'}</dd>
          <dt>{t('td.tcp')}</dt>
          <dd className="mono">{state.tcp_mm && state.tcp_mm.length >= 3 ? readoutText(state.tcp_mm.slice(0, 3), t.lang) : '—'}</dd>
        </dl>
        {view === 'tech' && (state.lines ?? []).length > 0 && (
          <details className="td-lines">
            <summary>{t('td.lines')}</summary>
            <ul>
              {(state.lines ?? []).slice(-6).map((line, index) => (
                <li key={`${index}-${line}`}>{line}</li>
              ))}
            </ul>
          </details>
        )}
        <div className="td-buttons">
          <button
            type="button"
            className="primary big"
            disabled={state.state !== 'free' || acting !== null}
            aria-describedby={`${titleId}-save`}
            onClick={() => void save()}
          >
            {t('td.save')}
          </button>
          <small id={`${titleId}-save`}>{t('td.save.note')}</small>
          <button
            type="button"
            className="halt big td-hold"
            disabled={acting === 'hold'}
            aria-describedby={`${titleId}-hold`}
            onClick={() => void hold()}
          >
            {t('td.hold')}
          </button>
          <small id={`${titleId}-hold`}>{t('td.hold.note')}</small>
        </div>
      </div>
    </div>
  )

  const code = record?.stop_code && isStopCode(record.stop_code) ? record.stop_code : null
  const verdict = ended && (
    <div className="td-verdict" aria-live="polite">
      {lost ? (
        <p className="td-verdict-title">{t('td.lost')}</p>
      ) : state?.state === 'saved' ? (
        <>
          <p className="td-verdict-title ok">{t('td.saved', { label: label.trim() || name })}</p>
          <div className="td-verdict-line">
            {state.verdict && <VerdictChip verdict={state.verdict} />}
            {file && <span className="small dim">{t('td.saved.where', { file })}</span>}
          </div>
          {role === 'place' && makeDefault && <p className="small">{t('td.saved.default')}</p>}
        </>
      ) : state?.state === 'refused' ? (
        <>
          <p className="td-verdict-title no">{t('td.notSaved')}</p>
          {/* The verdict once, as its chip: what refused the pose. Then where to teach it instead. */}
          <div className="td-verdict-line">
            <VerdictChip verdict={state.verdict || 'error'} />
          </div>
          {state.nearby_deg ? (
            <p className="mono small">{t('td.nearby', { joints: jointsText(state.nearby_deg, t.lang) })}</p>
          ) : (
            <p className="small">{t('td.nearby.none')}</p>
          )}
          {view === 'tech' && state.message && <p className="small mono dim">{state.message}</p>}
        </>
      ) : (
        <>
          <p className="td-verdict-title no">{code ? t.msg(stopMsg(code)) : t('td.notSaved')}</p>
          {code && <p className="small">{t.msg(stopSayMsg(code))}</p>}
          {view === 'tech' && (state?.message || record?.error) && (
            <p className="small mono dim">{state?.message || record?.error}</p>
          )}
        </>
      )}
      {wasFree && !lost && (
        <>
          <p className="small dim">{t('td.heldWhere')}</p>
          <p className="small td-countdown">{t('td.countdown')}</p>
        </>
      )}
    </div>
  )

  return (
    <div className="dialog-backdrop td-backdrop">
      <div className="dialog td-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={onKeyDown}>
        <div className="dialog-head">
          <Icon name="gripper" size={22} className="icon" />
          <h2 id={titleId}>{t('td.title')}</h2>
          {session && label && <span className="td-head-label">{t('td.head.label', { label: label.trim() })}</span>}
        </div>
        <div className="dialog-body">
          {!session && form}
          {session && !state && !ended && <p className="td-state">{t('td.state.freeing')}</p>}
          {sessionView}
          {verdict}
          {error && <ErrorBanner error={error} />}
          {error?.code === 'jaws_not_confirmed' && !session && (
            <div className="actions">
              <button type="button" className="big" onClick={() => void checkJaws()}>
                {t('td.checkJaws')}
              </button>
            </div>
          )}
        </div>
        {!session && (
          <div className="dialog-foot">
            {!starting && missing.length > 0 && (
              <span className="td-missing" id={`${titleId}-missing`}>
                {t('td.missing', { what: missing.join(', ') })}
              </span>
            )}
            <button type="button" className="ghost big" disabled={starting} onClick={onClose}>
              {t('td.cancel')}
            </button>
            <button
              type="button"
              className="danger big"
              disabled={!canFree}
              aria-describedby={!starting && missing.length > 0 ? `${titleId}-missing` : undefined}
              onClick={() => void start()}
            >
              {starting ? t('td.freeing') : t('td.free')}
            </button>
          </div>
        )}
        {session && ended && (
          <div className="dialog-foot">
            <button type="button" className="primary big" onClick={onClose}>
              {t('td.close')}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
