/**
 * The prompt path: one box, two ways in, and one rule that must never bend.
 *
 * ⛔ THE RULE, AND THE FIRST TEST IN THIS FILE. Speech does not start anything. A transcription lands
 * in the text field and the operator presses the same button, with the same acknowledgement, as for
 * a typed prompt. "Pick up the red cube" and "pick up the red cup" differ by one phoneme and by a
 * whole grasp, and a console that shortened that path would be a console where a misheard word moves
 * an arm. The backend holds the same line -- `/v1/voice/transcribe` returns TEXT -- and this pins the
 * other half of it, because the shortcut is exactly the kind of "improvement" a later reader adds.
 *
 * ⚠ The WAV encoder is tested against its own bytes rather than a fixture. What makes it correct is
 * that a decoder can read it, and the header is where that goes wrong -- a wrong byte rate or a
 * wrong data length produces a file every player opens and every decoder mis-reads.
 */

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PromptInput } from './PromptInput'
import { encodeWav } from './recordWav'
import { resetPromptHistory, settled, submitted, usePromptHistory } from './pipeline'

afterEach(() => {
  cleanup()
  resetPromptHistory()
  vi.restoreAllMocks()
})

describe('the WAV encoder', () => {
  it('writes a header a decoder can actually read', () => {
    const samples = new Float32Array(1000).map((_, i) => Math.sin(i / 10))
    const blob = encodeWav(samples, 16000)
    expect(blob.type).toBe('audio/wav')
    // 44-byte header + 2 bytes per sample. A wrong length here is the classic silent defect: the
    // file plays and the decoder reads past or stops short.
    expect(blob.size).toBe(44 + 1000 * 2)
  })

  it('clamps rather than wrapping', async () => {
    // ⚠ Gain control can hand back samples above 1.0. Scaling one of those straight into an int16
    // wraps it to a large NEGATIVE, which turns the loudest word of a sentence into a click.
    const blob = encodeWav(new Float32Array([2.0, -2.0]), 16000)
    const view = new DataView(await blob.arrayBuffer())
    expect(view.getInt16(44, true)).toBe(32767)
    expect(view.getInt16(46, true)).toBe(-32767)
  })

  it('declares the sample rate it was given', async () => {
    const view = new DataView(await encodeWav(new Float32Array(10), 48000).arrayBuffer())
    expect(view.getUint32(24, true)).toBe(48000)
    expect(view.getUint32(28, true)).toBe(48000 * 2)   // byte rate = rate * blockAlign
  })
})

describe('the prompt history', () => {
  function History() {
    const entries = usePromptHistory()
    return (
      <ul>
        {entries.map((e) => (
          <li key={e.id}>
            {e.source}: {e.text} {e.run ? `-> ${e.run.id}` : e.error ? `!! ${e.error.code}` : '...'}
          </li>
        ))}
      </ul>
    )
  }

  it('records HOW the text arrived, not only what it said', () => {
    render(<History />)
    act(() => {
      submitted('the green cube', 'spoken')
    })
    // ⭑ THE WHOLE REASON THE HISTORY EXISTS. When a run goes wrong, "the operator asked for the
    // wrong thing" and "the machine heard the wrong thing" look identical afterwards -- unless
    // something recorded which of the two routes the text came in by.
    expect(screen.getByText(/spoken: the green cube/)).toBeTruthy()
  })

  it('attaches the outcome to the prompt that caused it', () => {
    render(<History />)
    let id = 0
    act(() => {
      id = submitted('the red cup', 'typed')
    })
    act(() => {
      settled(id, { run: { id: 'run-7' } as never })
    })
    expect(screen.getByText(/-> run-7/)).toBeTruthy()
  })
})

describe('PromptInput', () => {
  beforeEach(() => {
    // jsdom has no microphone. `micAvailability()` therefore reports one specific thing, and the
    // component must render a REASON rather than a dead button -- which is what these assert.
  })

  it('says why there is no microphone instead of offering a broken button', () => {
    render(
      <PromptInput value="" onChange={() => undefined} canSubmit onSubmit={() => undefined} />,
    )
    expect(screen.getByText(/No microphone here/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /speak/i }).hasAttribute('disabled')).toBe(true)
  })

  it('does NOT submit when the text changes -- only the operator submits', () => {
    // ⛔ THE RULE. A transcription arriving must never be able to start a run by itself.
    const onSubmit = vi.fn()
    const onChange = vi.fn()
    render(
      <PromptInput value="" onChange={onChange} canSubmit onSubmit={onSubmit} />,
    )
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'the green cube' } })
    expect(onChange).toHaveBeenCalledWith('the green cube', 'typed')
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('submits on Enter only when the screen says it may', () => {
    const onSubmit = vi.fn()
    const { rerender } = render(
      <PromptInput value="x" onChange={() => undefined} canSubmit={false} onSubmit={onSubmit} />,
    )
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    expect(onSubmit).not.toHaveBeenCalled()
    rerender(
      <PromptInput value="x" onChange={() => undefined} canSubmit onSubmit={onSubmit} />,
    )
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('SAYS a prompt was transcribed, and stops saying it once edited', () => {
    // ⚠ A console that presents a transcription identically to typing is a console where nobody can
    // tell, afterwards, whether the wrong thing was asked for or the right thing was misheard.
    const { rerender } = render(
      <PromptInput value="the green cube" onChange={() => undefined} canSubmit spoken
                   onSubmit={() => undefined} />,
    )
    expect(screen.getByText(/transcribed from speech/)).toBeTruthy()
    rerender(
      <PromptInput value="the green cube" onChange={() => undefined} canSubmit spoken={false}
                   onSubmit={() => undefined} />,
    )
    expect(screen.queryByText(/transcribed from speech/)).toBeNull()
  })

  it('says nothing about speech for an empty box', () => {
    // A `spoken` flag left standing over an empty field would claim a transcription that is not there.
    render(
      <PromptInput value="   " onChange={() => undefined} canSubmit spoken
                   onSubmit={() => undefined} />,
    )
    expect(screen.queryByText(/transcribed from speech/)).toBeNull()
  })
})

describe('when the microphone IS available', () => {
  /** A recorder that yields one short buffer, so the whole path runs without hardware. */
  function installFakeMicrophone(seconds: number): { transcribe: ReturnType<typeof vi.fn> } {
    const rate = 16000
    const track = { stop: vi.fn() }
    const stream = { getTracks: () => [track] }
    vi.stubGlobal('navigator', {
      mediaDevices: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    let onaudioprocess: ((event: unknown) => void) | null = null
    const node = {
      connect: vi.fn(),
      disconnect: vi.fn(),
      set onaudioprocess(fn: ((event: unknown) => void) | null) {
        onaudioprocess = fn
      },
      get onaudioprocess() {
        return onaudioprocess
      },
    }
    const gain = { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() }
    vi.stubGlobal(
      'AudioContext',
      class {
        sampleRate = rate
        destination = {}
        createMediaStreamSource = () => ({ connect: vi.fn(), disconnect: vi.fn() })
        createScriptProcessor = () => node
        createGain = () => gain
        close = vi.fn().mockResolvedValue(undefined)
      },
    )
    vi.stubGlobal('isSecureContext', true)
    // Feed the node once the component has attached its handler.
    queueMicrotask(() => {
      const frames = Math.round(seconds * rate)
      onaudioprocess?.({
        inputBuffer: { getChannelData: () => new Float32Array(frames).fill(0.1) },
      })
    })
    return { transcribe: vi.fn() }
  }

  it('refuses a tap instead of inventing a word from room noise', async () => {
    installFakeMicrophone(0.05)
    const onChange = vi.fn()
    render(<PromptInput value="" onChange={onChange} canSubmit onSubmit={() => undefined} />)
    const button = screen.getByRole('button', { name: /speak/i })
    fireEvent.click(button)
    await waitFor(() => expect(screen.getByRole('button', { name: /stop/i })).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /stop/i }))
    // ⚠ Whisper on 50 ms of a loud cell returns an empty string or a word it made up, and an
    // invented word is worse than nothing once it becomes a grasp target.
    await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    expect(onChange).not.toHaveBeenCalled()
  })
})
