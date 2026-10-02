/**
 * The one prompt box: type it, or say it.
 *
 * ⛔ SPEECH DOES NOT START ANYTHING. The transcription lands in the text field and the operator
 * reads it, corrects it if needed, and sends it to be read; nothing moves until Start on the card
 * that shows what was understood. This mirrors the backend's own rule -- `POST /v1/voice/transcribe`
 * returns TEXT and starts nothing -- and it is there because a spoken command that went straight to
 * motion would mean a misheard word moves an arm. "Pick up the red cube" and "pick up the red cup"
 * differ by one phoneme and by a whole grasp.
 *
 * ⚠ WHAT THIS COMPONENT REFUSES TO HIDE. When the text came from speech it SAYS so, and it keeps
 * saying so until the operator edits it. A console that presents a transcription identically to
 * typing is a console where nobody can tell, afterwards, whether the wrong thing was asked for or
 * the right thing was misheard.
 *
 * The microphone button is a toggle (owner decision 10): a click starts, a second click stops, and a
 * second click while the microphone is still opening ends the recording as soon as it is open. A foot
 * switch that sends a key keeps push to talk (build plan item 21): the talk key (F8 unless `talkKey`
 * or `localStorage['willy.talkKey']` names another) records while it is held down, wherever the focus
 * is. A stored key is taken only where typing never uses it (F1-F24, Pause, Scroll Lock: the rule of
 * `screens/talkKey.ts`, which Settings applies too); any other, stored by hand or by the old console,
 * would take that key from typing, so F8 talks instead. While it records, a level meter says that the
 * microphone hears something. A box that becomes
 * disabled (a run starts, a teach frees the arm) closes the microphone and throws the recording away:
 * nobody is reading what it would propose.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type Ref } from 'react'

import { ApiError, api } from '../api/client'
import { useT } from '../i18n'
import { refusalMsg } from '../i18n/codes'
import { Icon } from '../icons'
import { usePrefs } from '../model/prefs'
import { isTalkKey } from '../screens/talkKey'
import { PROMPT } from './i18n'
import { type ActiveRecording, micUnavailable, startRecording } from './recordWav'
import './prompt.css'

type MicState = 'idle' | 'opening' | 'recording' | 'transcribing'
/** Who started the recording: a click (a click stops it) or the talk key (letting go stops it). */
type Mode = 'click' | 'key'

/** The key a foot switch sends when nothing else is configured. Not a letter: typing must not talk. */
export const DEFAULT_TALK_KEY = 'F8'
/** Where a cell keeps its own talk key, so a switch set to another key needs no rebuild. */
export const TALK_KEY_STORAGE = 'willy.talkKey'

/**
 * Below this a recording is a tap, not a sentence. Whisper would return an empty string or invent a word from room
 * noise, and an invented word is worse than nothing when it becomes a grasp target.
 */
const SHORTEST_S = 0.3

function storedTalkKey(): string {
  try {
    const stored = localStorage.getItem(TALK_KEY_STORAGE)
    // Enter, Shift, a letter: the window-wide listener would take it from typing (it prevents the key's default).
    return stored && isTalkKey(stored) ? stored : DEFAULT_TALK_KEY
  } catch {
    // Private browsing, a file:// origin, a locked-down kiosk profile. The default still works.
    return DEFAULT_TALK_KEY
  }
}

/** A frame's RMS on the meter's 0-100: speech sits around 0.02-0.2, and the square root spreads it across the bar. */
function meterOf(rms: number): number {
  return Math.max(0, Math.min(100, Math.round(Math.sqrt(Math.max(0, rms)) * 140)))
}

export interface PromptInputProps {
  value: string
  onChange: (text: string, source: 'typed' | 'spoken') => void
  /** Enter submits only when this is true -- the screen owns the readiness rules, not this box. */
  canSubmit: boolean
  onSubmit: () => void
  placeholder?: string
  /** True once the text came from speech and has not been edited since. */
  spoken?: boolean
  disabled?: boolean
  /** Said in the box while it is disabled: why it is (a run is active, a teach). */
  disabledReason?: string
  /**
   * The `KeyboardEvent.key` that records while it is held, for a foot switch that sends a key.
   * Unset, it is what `localStorage['willy.talkKey']` names, and F8 when that names nothing.
   */
  talkKey?: string
  /** The text field, for a screen that hands the focus back to it ("What should I do next?"). */
  inputRef?: Ref<HTMLInputElement>
  /** A Send button beside the microphone, for a touch screen; Enter does the same. */
  showSend?: boolean
  /** A request for this text is out: Send and Enter wait for it. */
  busy?: boolean
}

export function PromptInput({
  value,
  onChange,
  canSubmit,
  onSubmit,
  placeholder,
  spoken = false,
  disabled = false,
  disabledReason,
  talkKey,
  inputRef,
  showSend = false,
  busy = false,
}: PromptInputProps) {
  const t = useT(PROMPT)
  const { view } = usePrefs()
  const [mic, setMic] = useState<MicState>('idle')
  const [micError, setMicError] = useState<{ text: string; detail: string } | null>(null)
  const [meter, setMeter] = useState(0)
  // The state machine lives in refs: a click, a key and a microphone that opens asynchronously all read it at once.
  const phase = useRef<MicState>('idle')
  const mode = useRef<Mode>('click')
  /** A stop came while the microphone was still opening: stop as soon as it is open. */
  const stopAsked = useRef(false)
  /** The recording is to be thrown away (the box was disabled, the component went away). */
  const dropped = useRef(false)
  const active = useRef<ActiveRecording | null>(null)
  const alive = useRef(true)
  const unavailable = micUnavailable()
  const key = useMemo(() => talkKey ?? storedTalkKey(), [talkKey])

  const enter = useCallback((next: MicState) => {
    phase.current = next
    if (alive.current) setMic(next)
  }, [])

  // ⛔ A LIVE MICROPHONE MUST NOT SURVIVE THE COMPONENT. Navigating away mid-recording would leave
  // the track open and the browser's recording indicator lit, with nothing on screen to explain it.
  // The console has had exactly this leak before, with cameras: every build opened one and nothing
  // closed one, and it went unnoticed for months because no part of the UI said the device was held.
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      dropped.current = true
      active.current?.cancel()
      active.current = null
    }
  }, [])

  const finish = useCallback(async () => {
    const recording = active.current
    active.current = null
    stopAsked.current = false
    setMeter(0)
    if (!recording) {
      enter('idle')
      return
    }
    enter('transcribing')
    try {
      const { wav, seconds } = await recording.stop()
      if (seconds < SHORTEST_S) {
        if (alive.current) setMicError({ text: t('prompt.tooShort'), detail: '' })
        return
      }
      const { text } = await api.transcribe(wav)
      if (!alive.current) return
      if (!text.trim()) setMicError({ text: t('prompt.nothingHeard'), detail: '' })
      else onChange(text, 'spoken')
    } catch (err: unknown) {
      if (!alive.current) return
      // ⚠ 501 IS NOT A FAILURE THE OPERATOR CAUSED: speech-to-text not installed, or a container this host cannot
      // decode. The code is said in the reader's language; the backend's own sentence (it names the install line) is
      // the detail of the tech view.
      if (err instanceof ApiError) setMicError({ text: t.msg(refusalMsg(err.code)), detail: err.message })
      else setMicError({ text: String(err), detail: '' })
    } finally {
      enter('idle')
    }
  }, [enter, onChange, t])

  const begin = useCallback(
    async (how: Mode) => {
      if (phase.current !== 'idle') return
      mode.current = how
      stopAsked.current = false
      dropped.current = false
      setMicError(null)
      setMeter(0)
      enter('opening')
      let recording: ActiveRecording
      try {
        recording = await startRecording({
          onLevel: (rms) => {
            if (alive.current && phase.current === 'recording') setMeter(meterOf(rms))
          },
        })
      } catch (err: unknown) {
        // A refused permission arrives here. It is the operator's own decision, not a fault, so it is
        // stated and nothing else changes.
        if (alive.current) setMicError({ text: t('prompt.micRefused', { why: err instanceof Error ? err.message : String(err) }), detail: '' })
        enter('idle')
        return
      }
      if (dropped.current) {
        recording.cancel()
        enter('idle')
        return
      }
      active.current = recording
      if (stopAsked.current) {
        // Stopped while the microphone was still opening: the recording ends as soon as it began.
        void finish()
        return
      }
      enter('recording')
    },
    [enter, finish, t],
  )

  /** Stop and transcribe: now when recording, as soon as it is open when still opening. */
  const end = useCallback(() => {
    if (phase.current === 'opening') stopAsked.current = true
    else if (phase.current === 'recording') void finish()
  }, [finish])

  /** Stop and throw it away: the box was disabled while it listened. */
  const drop = useCallback(() => {
    if (phase.current === 'opening') {
      dropped.current = true
      return
    }
    if (phase.current === 'recording') {
      active.current?.cancel()
      active.current = null
      setMeter(0)
      enter('idle')
    }
  }, [enter])

  // The talk key, held anywhere on the page. A key that repeats while held starts nothing new, and a window that
  // loses focus ends a held key's recording, because its key-up would never arrive. A box that becomes disabled
  // closes the microphone, whoever opened it.
  useEffect(() => {
    if (disabled || unavailable !== null) {
      drop()
      return undefined
    }
    const down = (event: KeyboardEvent) => {
      if (event.key !== key) return
      event.preventDefault()
      if (!event.repeat && phase.current === 'idle') void begin('key')
    }
    const up = (event: KeyboardEvent) => {
      if (event.key !== key) return
      event.preventDefault()
      if (mode.current === 'key') end()
    }
    const blur = () => {
      if (mode.current === 'key') end()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [key, begin, end, drop, disabled, unavailable])

  const onMicClick = () => {
    if (phase.current === 'idle') void begin('click')
    else end()
  }

  const listening = mic === 'recording' || mic === 'opening'
  const micName =
    mic === 'transcribing' ? t('prompt.mic.transcribing') : listening ? t('prompt.mic.stop') : t('prompt.mic.speak')
  const why = unavailable !== null ? t(`prompt.noMic.${unavailable}`) : null
  const shown = disabled && disabledReason ? disabledReason : (placeholder ?? t('prompt.placeholder'))

  return (
    <div className={`prompt${listening ? ' is-listening' : ''}`}>
      <div className="prompt-row">
        <input
          ref={inputRef}
          type="text"
          value={value}
          placeholder={shown}
          aria-label={t('prompt.field')}
          className="prompt-field"
          disabled={disabled}
          onChange={(e) => onChange(e.target.value, 'typed')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && canSubmit && !busy) {
              e.preventDefault()
              onSubmit()
            }
          }}
        />
        <span className="prompt-mic-wrap">
          <button
            type="button"
            className={`prompt-mic${listening ? ' on' : ''}`}
            aria-label={micName}
            aria-pressed={listening}
            title={why ?? t('prompt.mic.title', { key })}
            disabled={disabled || mic === 'transcribing' || unavailable !== null}
            onClick={onMicClick}
          >
            <Icon name="mic" size={18} />
            <span className="prompt-mic-label">{micName}</span>
          </button>
          {listening && (
            <span
              className="prompt-level"
              role="meter"
              aria-label={t('prompt.level')}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={meter}
            >
              <span style={{ width: `${meter}%` }} />
            </span>
          )}
        </span>
        {showSend && (
          <button
            type="button"
            className="prompt-send"
            disabled={disabled || busy || !canSubmit}
            onClick={onSubmit}
            title={t('prompt.sendTitle')}
            aria-label={t('prompt.send')}
          >
            <Icon name="chevronRight" size={20} />
          </button>
        )}
      </div>
      {why && <div className="prompt-note dim">{t('prompt.noMic', { why })}</div>}
      {micError && (
        <div className="prompt-note caution" role="status">
          {micError.text}
          {view === 'tech' && micError.detail && <span className="prompt-detail"> {micError.detail}</span>}
        </div>
      )}
      {spoken && value.trim() !== '' && <div className="prompt-note prompt-spoken">{t('prompt.spoken')}</div>}
    </div>
  )
}
