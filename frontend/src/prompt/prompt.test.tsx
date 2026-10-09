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
 * The microphone button is a toggle (owner decision 10): a click starts, a second click stops. A foot switch that
 * sends the talk key keeps push to talk: the key records while it is held (build plan item 21).
 *
 * ⚠ The WAV encoder is tested against its own bytes rather than a fixture. What makes it correct is
 * that a decoder can read it, and the header is where that goes wrong -- a wrong byte rate or a
 * wrong data length produces a file every player opens and every decoder mis-reads.
 */

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '../i18n'
import { PromptInput } from './PromptInput'
import { encodeWav, rms } from './recordWav'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
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

describe('the level meter\'s number', () => {
  it('is the root mean square of a frame: silence is 0, a full-scale square is 1', () => {
    expect(rms(new Float32Array(64))).toBe(0)
    expect(rms(new Float32Array(64).fill(1))).toBe(1)
    expect(rms(new Float32Array([0.5, -0.5, 0.5, -0.5]))).toBeCloseTo(0.5, 6)
    expect(rms(new Float32Array(0))).toBe(0)
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

  it('stops typing at the reader\'s 1000 characters, and counts them from 800 on', () => {
    // The owner, 2026-10-08: a long description is never refused at the door with "Die Anfrage ist ungültig.".
    const { rerender } = render(
      <PromptInput value={'x'.repeat(799)} onChange={() => undefined} canSubmit onSubmit={() => undefined} />,
    )
    expect((screen.getByRole('textbox') as HTMLInputElement).maxLength).toBe(1000)
    expect(screen.queryByText(/of 1,000 characters/)).toBeNull()
    rerender(<PromptInput value={'x'.repeat(800)} onChange={() => undefined} canSubmit onSubmit={() => undefined} />)
    // The numbers in the reader's language: "1,000" here, "1.000" in German.
    expect(screen.getByText('800 of 1,000 characters')).toBeTruthy()
  })

  it('says under the box what the screen says Enter does', () => {
    render(
      <PromptInput value="" onChange={() => undefined} canSubmit onSubmit={() => undefined} note="Enter starts at once · Once" />,
    )
    expect(screen.getByText('Enter starts at once · Once')).toBeTruthy()
  })

  it('speaks German under the console\'s provider', () => {
    render(
      <I18nProvider lang="de">
        <PromptInput value="" onChange={() => undefined} canSubmit onSubmit={() => undefined} />
      </I18nProvider>,
    )
    expect(screen.getByText(/Kein Mikrofon hier/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /Sprechen/ })).toBeTruthy()
  })
})

describe('when the microphone IS available', () => {
  /** A recorder the test feeds by hand, so the whole path runs without hardware. */
  function installFakeMicrophone(seconds: number): {
    track: { stop: ReturnType<typeof vi.fn> }
    getUserMedia: ReturnType<typeof vi.fn>
    /** Hand the open recorder `secondsOf` seconds of a steady tone at `level`. */
    feed: (secondsOf: number, level?: number) => void
  } {
    const rate = 16000
    const track = { stop: vi.fn() }
    const stream = { getTracks: () => [track] }
    const getUserMedia = vi.fn().mockResolvedValue(stream)
    vi.stubGlobal('navigator', {
      mediaDevices: { getUserMedia },
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
    const feed = (secondsOf: number, level = 0.1) => {
      const frames = Math.round(secondsOf * rate)
      onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array(frames).fill(level) } })
    }
    // Fed once right away: before the recorder listens, so a tap records nothing.
    queueMicrotask(() => feed(seconds))
    return { track, getUserMedia, feed }
  }

  function box(talkKey?: string) {
    return render(
      <PromptInput value="" onChange={() => undefined} canSubmit onSubmit={() => undefined}
                   talkKey={talkKey} />,
    )
  }

  it('refuses a tap instead of inventing a word from room noise', async () => {
    installFakeMicrophone(0.05)
    const onChange = vi.fn()
    render(<PromptInput value="" onChange={onChange} canSubmit onSubmit={() => undefined} />)
    fireEvent.click(screen.getByRole('button', { name: /speak/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /stop recording/i }))
    // ⚠ Whisper on 50 ms of a loud cell returns an empty string or a word it made up, and an
    // invented word is worse than nothing once it becomes a grasp target.
    await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    expect(onChange).not.toHaveBeenCalled()
  })

  it('a click starts, a second click stops (owner decision 10)', async () => {
    const { getUserMedia, track } = installFakeMicrophone(0.05)
    box()
    fireEvent.click(screen.getByRole('button', { name: /speak/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    expect(track.stop).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /stop recording/i }))
    await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    expect(track.stop).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /speak/i })).toBeTruthy()
  })

  it('keeps recording when the pointer leaves the button: a toggle stops on a click only', async () => {
    const { track } = installFakeMicrophone(0.05)
    box()
    fireEvent.click(screen.getByRole('button', { name: /speak/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    fireEvent.pointerLeave(screen.getByRole('button', { name: /stop recording/i }))
    fireEvent.pointerUp(screen.getByRole('button', { name: /stop recording/i }))
    await act(async () => undefined)
    expect(track.stop).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy()
  })

  it('a second click while the microphone is still opening ends the recording', async () => {
    // The click opens the microphone asynchronously. A second click before it is open must not leave it open.
    const { track } = installFakeMicrophone(0.05)
    box()
    const button = screen.getByRole('button', { name: /speak/i })
    fireEvent.click(button)
    fireEvent.click(button)
    await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    expect(track.stop).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /speak/i })).toBeTruthy()
  })

  it('shows a level meter while it records, and none before', async () => {
    const { feed } = installFakeMicrophone(0.05)
    box()
    expect(screen.queryByRole('meter')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /speak/i }))
    await waitFor(() => expect(screen.getByRole('meter', { name: /level/i })).toBeTruthy())
    act(() => feed(0.2, 0.5))
    await waitFor(() => expect(Number(screen.getByRole('meter', { name: /level/i }).getAttribute('aria-valuenow'))).toBeGreaterThan(0))
  })

  it('puts the transcript in the box as spoken, and never submits it', async () => {
    const { feed } = installFakeMicrophone(0.05)
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ text: 'nimm den grünen Würfel', reason: null, speech: {}, transcript: null }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )))
    const onChange = vi.fn()
    const onSubmit = vi.fn()
    render(<PromptInput value="" onChange={onChange} canSubmit onSubmit={onSubmit} />)
    fireEvent.click(screen.getByRole('button', { name: /speak/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    act(() => feed(1.2))
    fireEvent.click(screen.getByRole('button', { name: /stop recording/i }))
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('nimm den grünen Würfel', 'spoken'))
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('closes the microphone and drops the recording when the box is disabled (a run starts)', async () => {
    const { track } = installFakeMicrophone(0.05)
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    const { rerender } = render(<PromptInput value="" onChange={() => undefined} canSubmit onSubmit={() => undefined} />)
    fireEvent.click(screen.getByRole('button', { name: /speak/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    rerender(<PromptInput value="" onChange={() => undefined} canSubmit onSubmit={() => undefined} disabled />)
    await waitFor(() => expect(track.stop).toHaveBeenCalled())
    expect(fetcher).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /speak/i }).hasAttribute('disabled')).toBe(true)
  })

  it('records while the talk key is held, for a foot switch that sends a key', async () => {
    const { getUserMedia } = installFakeMicrophone(0.05)
    box()
    fireEvent.keyDown(window, { key: 'F8' })
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    // A held key repeats. The repeat is the same hold, not a second recording.
    fireEvent.keyDown(window, { key: 'F8', repeat: true })
    fireEvent.keyUp(window, { key: 'F8' })
    await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    expect(getUserMedia).toHaveBeenCalledTimes(1)
  })

  it('listens for the configured talk key and not the default one', async () => {
    const { getUserMedia } = installFakeMicrophone(0.05)
    box('PageDown')
    fireEvent.keyDown(window, { key: 'F8' })
    await act(async () => undefined)
    expect(getUserMedia).not.toHaveBeenCalled()
    fireEvent.keyDown(window, { key: 'PageDown' })
    await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).toBeTruthy())
    fireEvent.keyUp(window, { key: 'PageDown' })
    await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
  })

  it('takes the talk key a cell stored', async () => {
    const { getUserMedia } = installFakeMicrophone(0.05)
    localStorage.setItem('willy.talkKey', 'F9')
    try {
      box()
      fireEvent.keyDown(window, { key: 'F9' })
      await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1))
      fireEvent.keyUp(window, { key: 'F9' })
      await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    } finally {
      localStorage.removeItem('willy.talkKey')
    }
  })

  it('takes no stored talk key that typing uses: Enter stays Enter, and the default F8 talks', async () => {
    // A key stored by hand, or by the old console, which took any key: Settings refuses it (`screens/talkKey.ts`),
    // and the cockpit must not take it from typing either.
    const { getUserMedia } = installFakeMicrophone(0.05)
    localStorage.setItem('willy.talkKey', 'Enter')
    try {
      box()
      fireEvent.keyDown(window, { key: 'Enter' })
      await act(async () => undefined)
      expect(getUserMedia).not.toHaveBeenCalled()
      fireEvent.keyDown(window, { key: 'F8' })
      await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1))
      fireEvent.keyUp(window, { key: 'F8' })
      await waitFor(() => expect(screen.getByText(/too short to transcribe/)).toBeTruthy())
    } finally {
      localStorage.removeItem('willy.talkKey')
    }
  })

  it('says in its tooltip that a click starts and a click stops, and which key records while held', () => {
    installFakeMicrophone(0.05)
    box('F9')
    const title = screen.getByRole('button', { name: /speak/i }).getAttribute('title') ?? ''
    expect(title).toMatch(/Click to speak, click again to stop/)
    expect(title).toMatch(/Holding F9 records while held/)
    // Enter may start the task where the settings say so (the owner, 2026-10-08): never before the person read it.
    expect(title).toMatch(/nothing moves before you have read it and pressed Enter/)
  })
})
