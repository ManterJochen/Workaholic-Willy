/**
 * The one prompt box: type it, or say it.
 *
 * ⛔ SPEECH DOES NOT START ANYTHING. The transcription lands in the text field and the operator
 * presses the same button, with the same acknowledgement, as for a typed prompt. This mirrors the
 * backend's own rule -- `POST /v1/voice/transcribe` returns TEXT and starts nothing -- and it is
 * there because a spoken command that went straight to motion would mean a misheard word moves an
 * arm. "Pick up the red cube" and "pick up the red cup" differ by one phoneme and by a whole grasp.
 *
 * ⚠ WHAT THIS COMPONENT REFUSES TO HIDE. When the text came from speech it SAYS so, and it keeps
 * saying so until the operator edits it. A console that presents a transcription identically to
 * typing is a console where nobody can tell, afterwards, whether the wrong thing was asked for or
 * the right thing was misheard.
 *
 * Push to talk, both ways. The microphone records while the button is held and stops when it is let
 * go or the pointer leaves it. A foot switch that sends a key does the same: the talk key (F8 unless
 * `talkKey` or `localStorage['willy.talkKey']` names another) records while it is held down, wherever
 * the focus is. A click alone records nothing.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, api } from '../api/client'
import { type ActiveRecording, micAvailability, startRecording } from './recordWav'

type MicState = 'idle' | 'recording' | 'transcribing'

/** The key a foot switch sends when nothing else is configured. Not a letter: typing must not talk. */
export const DEFAULT_TALK_KEY = 'F8'
/** Where a cell keeps its own talk key, so a switch set to another key needs no rebuild. */
export const TALK_KEY_STORAGE = 'willy.talkKey'

function storedTalkKey(): string {
  try {
    return localStorage.getItem(TALK_KEY_STORAGE) || DEFAULT_TALK_KEY
  } catch {
    // Private browsing, a file:// origin, a locked-down kiosk profile. The default still works.
    return DEFAULT_TALK_KEY
  }
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
  /**
   * The `KeyboardEvent.key` that records while it is held, for a foot switch that sends a key.
   * Unset, it is what `localStorage['willy.talkKey']` names, and F8 when that names nothing.
   */
  talkKey?: string
}

export function PromptInput({
  value,
  onChange,
  canSubmit,
  onSubmit,
  placeholder = 'the green cube',
  spoken = false,
  disabled = false,
  talkKey,
}: PromptInputProps) {
  const [mic, setMic] = useState<MicState>('idle')
  const [micError, setMicError] = useState<string | null>(null)
  const active = useRef<ActiveRecording | null>(null)
  /** True from the press until the release, including while the microphone is still opening. */
  const holding = useRef(false)
  /** True while a recording is stopped and transcribed; a press then records nothing. */
  const busy = useRef(false)
  const unavailable = micAvailability()
  const key = useMemo(() => talkKey ?? storedTalkKey(), [talkKey])

  // ⛔ A LIVE MICROPHONE MUST NOT SURVIVE THE COMPONENT. Navigating away mid-recording would leave
  // the track open and the browser's recording indicator lit, with nothing on screen to explain it.
  // The console has had exactly this leak before, with cameras: every build opened one and nothing
  // closed one, and it went unnoticed for months because no part of the UI said the device was held.
  useEffect(() => () => active.current?.cancel(), [])

  const finish = useCallback(async () => {
    const recording = active.current
    active.current = null
    if (!recording) return
    busy.current = true
    setMic('transcribing')
    try {
      const { wav, seconds } = await recording.stop()
      if (seconds < 0.3) {
        // A tap rather than a press. Whisper would return an empty string or invent a word from
        // room noise, and an invented word is worse than nothing when it becomes a grasp target.
        setMicError('That was too short to transcribe — hold the button while you speak.')
        return
      }
      const { text } = await api.transcribe(wav)
      if (!text.trim()) {
        setMicError('Nothing was recognised in that recording.')
      } else {
        onChange(text, 'spoken')
      }
    } catch (err: unknown) {
      // ⚠ 501 IS NOT A FAILURE THE OPERATOR CAUSED, and the backend distinguishes it: speech-to-text
      // not installed, or a container this host cannot decode. Both name the install line in their
      // own message, so it is shown as-is rather than replaced with something friendlier and emptier.
      setMicError(err instanceof ApiError ? err.message : String(err))
    } finally {
      busy.current = false
      setMic('idle')
    }
  }, [onChange])

  const begin = useCallback(async () => {
    if (holding.current || busy.current || active.current) return
    holding.current = true
    setMicError(null)
    try {
      active.current = await startRecording()
    } catch (err: unknown) {
      // A refused permission arrives here. It is the operator's own decision, not a fault, so it is
      // stated and nothing else changes.
      holding.current = false
      setMicError(err instanceof Error ? err.message : String(err))
      setMic('idle')
      return
    }
    if (!holding.current) {
      // Let go while the microphone was still opening: the hold is over, so the recording is too.
      void finish()
      return
    }
    setMic('recording')
  }, [finish])

  const end = useCallback(() => {
    if (!holding.current) return
    holding.current = false
    void finish()
  }, [finish])

  // The talk key, held anywhere on the page. A key that repeats while held starts nothing new, and a
  // window that loses focus ends the hold, because its key-up would never arrive. A box that becomes
  // disabled ends a hold too: a disabled button never hears the pointer come up.
  useEffect(() => {
    if (disabled || unavailable !== null) {
      end()
      return undefined
    }
    const down = (event: KeyboardEvent) => {
      if (event.key !== key) return
      event.preventDefault()
      if (!event.repeat) void begin()
    }
    const up = (event: KeyboardEvent) => {
      if (event.key !== key) return
      event.preventDefault()
      end()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', end)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', end)
    }
  }, [key, begin, end, disabled, unavailable])

  const micLabel =
    mic === 'recording'
      ? '● release to stop'
      : mic === 'transcribing'
        ? 'transcribing…'
        : '🎤 hold to speak'

  return (
    <>
      <div className="actions instruction-row">
        <input
          type="text"
          value={value}
          placeholder={placeholder}
          className="grow"
          disabled={disabled}
          onChange={(e) => onChange(e.target.value, 'typed')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && canSubmit) onSubmit()
          }}
        />
        <button
          type="button"
          className={mic === 'recording' ? 'danger' : ''}
          disabled={disabled || mic === 'transcribing' || unavailable !== null}
          title={
            unavailable ??
            `Hold to speak, and let go to stop. Holding ${key} does the same, so a foot switch ` +
              `that sends ${key} works too. The text lands in the box; you press the button.`
          }
          onPointerDown={(e) => {
            if (e.button > 0) return
            e.preventDefault()
            void begin()
          }}
          onPointerUp={end}
          onPointerLeave={end}
          onPointerCancel={end}
          onContextMenu={(e) => e.preventDefault()}
        >
          {micLabel}
        </button>
      </div>
      {unavailable && (
        <div className="small dim">
          No microphone here — {unavailable} Type the prompt instead.
        </div>
      )}
      {micError && <div className="small caution">{micError}</div>}
      {spoken && value.trim() !== '' && (
        <div className="small dim">
          This prompt was <strong>transcribed from speech</strong>, not typed. Read it before you
          start — the arm moves on what was heard.
        </div>
      )}
    </>
  )
}
