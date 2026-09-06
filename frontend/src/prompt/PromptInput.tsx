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
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api } from '../api/client'
import { type ActiveRecording, micAvailability, startRecording } from './recordWav'

type MicState = 'idle' | 'recording' | 'transcribing'

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
}

export function PromptInput({
  value,
  onChange,
  canSubmit,
  onSubmit,
  placeholder = 'the green cube',
  spoken = false,
  disabled = false,
}: PromptInputProps) {
  const [mic, setMic] = useState<MicState>('idle')
  const [micError, setMicError] = useState<string | null>(null)
  const active = useRef<ActiveRecording | null>(null)
  const unavailable = micAvailability()

  // ⛔ A LIVE MICROPHONE MUST NOT SURVIVE THE COMPONENT. Navigating away mid-recording would leave
  // the track open and the browser's recording indicator lit, with nothing on screen to explain it.
  // The console has had exactly this leak before, with cameras: every build opened one and nothing
  // closed one, and it went unnoticed for months because no part of the UI said the device was held.
  useEffect(() => () => active.current?.cancel(), [])

  const begin = useCallback(async () => {
    setMicError(null)
    try {
      active.current = await startRecording()
      setMic('recording')
    } catch (err: unknown) {
      // A refused permission arrives here. It is the operator's own decision, not a fault, so it is
      // stated and nothing else changes.
      setMicError(err instanceof Error ? err.message : String(err))
      setMic('idle')
    }
  }, [])

  const finish = useCallback(async () => {
    const recording = active.current
    active.current = null
    if (!recording) return
    setMic('transcribing')
    try {
      const { wav, seconds } = await recording.stop()
      if (seconds < 0.3) {
        // A tap rather than a press. Whisper would return an empty string or invent a word from
        // room noise, and an invented word is worse than nothing when it becomes a grasp target.
        setMicError('That was too short to transcribe — hold the button while you speak.')
        setMic('idle')
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
      setMic('idle')
    }
  }, [onChange])

  const micLabel =
    mic === 'recording' ? '● stop' : mic === 'transcribing' ? 'transcribing…' : '🎤 speak'

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
          title={unavailable ?? 'Hold to dictate. The text lands in the box; you press the button.'}
          onClick={() => void (mic === 'recording' ? finish() : begin())}
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
