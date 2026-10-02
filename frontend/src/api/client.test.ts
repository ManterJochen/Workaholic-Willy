/**
 * The client sends exactly what the contract (build plan 1.11) says, and nothing a person did not click.
 *
 * Every route is called through a recording `fetch`, and the method, path, query and body are compared with the
 * contract. The point is not the plumbing: three of these routes move the arm (task, restart, home), and a client
 * that quietly added a flag, reused a stale body or called the wrong verb would be the one place no server-side
 * gate can see.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError, api, overlayUrl, sendTeachCancelBeacon, targetOverlayUrl } from './client'
import { followCell, followRun, isEventOf, type RunEvent } from './events'

interface Call {
  method: string
  url: string
  body: unknown
}

let calls: Call[] = []

function answer(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

beforeEach(() => {
  calls = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string, init?: RequestInit) => {
      calls.push({
        method: init?.method ?? 'GET',
        url: String(input),
        body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined,
      })
      return answer(200, {})
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('the contract routes', () => {
  it('send each method, path and body as the contract names them', async () => {
    const plan = { object: 'green cube', place: { kind: 'pose' as const, pose: null }, return_to: 'home', scope: 'once' as const }
    await api.codes()
    await api.task(plan)
    await api.stopTask('run-1')
    await api.restart('run-1')
    await api.facts()
    await api.readiness()
    await api.acknowledge()
    await api.acknowledge(true)
    await api.brake()
    await api.home()
    await api.home('park')
    await api.startPlanner()
    await api.jaws()
    await api.answerJaws('q-1', 'closed')
    await api.checkJaws()
    await api.live()
    await api.live('wrist_d415', 640)
    await api.poses()
    await api.setDefaultPlace('drop_left')
    await api.setDefaultPlace(null)
    await api.teachPayload()
    await api.teach({ name: 'drop_left', label: 'Ablage links', role: 'place', replace: false, make_default_place: true, payload_seen: null })
    await api.teachState('run-2', 'tok')
    await api.teachCapture('run-2', 'tok')
    await api.teachCancel('run-2', 'tok')
    await api.parse({ text: 'nimm den grünen Würfel', source: 'spoken', language: 'de' })
    await api.commandStatus()
    await api.warmCommands()
    await api.runs()
    await api.runs(1)

    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      'GET /v1/codes',
      'POST /v1/task',
      'POST /v1/task/stop?run_id=run-1',
      'POST /v1/task/restart',
      'GET /v1/cell/facts',
      'GET /v1/cell/readiness',
      'POST /v1/cell/acknowledge',
      'POST /v1/cell/acknowledge',
      'POST /v1/cell/brake',
      'POST /v1/cell/home',
      'POST /v1/cell/home',
      'POST /v1/cell/planner',
      'GET /v1/cell/jaws',
      'POST /v1/cell/jaws/answer',
      'POST /v1/cell/jaws/check',
      'GET /v1/camera/live?max_width=960',
      'GET /v1/camera/live?rig=wrist_d415&max_width=640',
      'GET /v1/poses',
      'PUT /v1/poses/default-place',
      'PUT /v1/poses/default-place',
      'GET /v1/teach/payload',
      'POST /v1/teach',
      'GET /v1/teach/run-2?token=tok',
      'POST /v1/teach/run-2/capture?token=tok',
      'POST /v1/teach/run-2/cancel?token=tok',
      'POST /v1/commands/parse',
      'GET /v1/commands/status',
      'POST /v1/commands/warmup',
      'GET /v1/runs',
      'GET /v1/runs?limit=1',
    ])

    const body = (url: string, nth = 0) => calls.filter((c) => c.url === url)[nth].body
    expect(body('/v1/task')).toEqual(plan)
    expect(body('/v1/task/restart')).toEqual({ run_id: 'run-1' })
    // "The cell is clear" is a person's word, sent as the literal true; "the jaws are empty" only when ticked.
    expect(body('/v1/cell/acknowledge', 0)).toEqual({ cell_clear: true, jaws_empty: false })
    expect(body('/v1/cell/acknowledge', 1)).toEqual({ cell_clear: true, jaws_empty: true })
    expect(body('/v1/cell/home', 0)).toEqual({ to: 'home' })
    expect(body('/v1/cell/home', 1)).toEqual({ to: 'park' })
    expect(body('/v1/cell/jaws/answer')).toEqual({ question_id: 'q-1', choice: 'closed' })
    expect(body('/v1/poses/default-place', 1)).toEqual({ name: null })
    expect(body('/v1/commands/parse')).toEqual({ text: 'nimm den grünen Würfel', source: 'spoken', language: 'de' })
  })

  it('send halt now with no body and nothing else', async () => {
    await api.brake()
    expect(calls).toEqual([{ method: 'POST', url: '/v1/cell/brake', body: undefined }])
  })

  it('keep the stop of a pick run, now able to name its run; the console starts no pick run (programs do, at POST /v1/pick)', async () => {
    await api.stopPick()
    await api.stopPick('run-9')
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual(['POST /v1/pick/stop', 'POST /v1/pick/stop?run_id=run-9'])
    expect('pick' in api).toBe(false)
  })

  it('name the overlay images as their own URLs, never as data drawn over the live picture', () => {
    expect(overlayUrl('run-1f3a9c0e21', 2)).toBe('/v1/runs/run-1f3a9c0e21/overlays/2')
    expect(targetOverlayUrl('run-a84c2e7d05')).toBe('/v1/runs/run-a84c2e7d05/target/overlay')
  })

  it('cancel a teach with a beacon when the page goes away', () => {
    const beacon = vi.fn(() => true)
    vi.stubGlobal('navigator', { ...navigator, sendBeacon: beacon })
    expect(sendTeachCancelBeacon('run-2', 't k')).toBe(true)
    expect(beacon).toHaveBeenCalledWith('/v1/teach/run-2/cancel?token=t%20k')
  })
})

describe('a refusal', () => {
  it('arrives as an ApiError carrying the code, the sentence and the detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => answer(409, { code: 'restart_required', message: 'the stop of run-7 stands', detail: { run_id: 'run-7' } })),
    )
    const error = await api.task({ object: '', place: { kind: 'pose', pose: null }, return_to: 'home', scope: 'once' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).code).toBe('restart_required')
    expect((error as ApiError).httpStatus).toBe(409)
    expect((error as ApiError).detail).toEqual({ run_id: 'run-7' })
  })
})

describe('the event streams', () => {
  let opened: string[] = []

  beforeEach(() => {
    opened = []
    vi.stubGlobal(
      'WebSocket',
      class {
        onopen: (() => void) | null = null
        onmessage: ((m: { data: string }) => void) | null = null
        onerror: (() => void) | null = null
        onclose: (() => void) | null = null
        constructor(url: string) {
          opened.push(url)
        }
        close() {}
      } as unknown as typeof WebSocket,
    )
  })

  it('resume a run from the seq the caller already holds', () => {
    const stop = followRun('run-1', { onEvent: () => undefined, onState: () => undefined }, { sinceSeq: 41 })
    stop()
    expect(opened[0]).toMatch(/\/v1\/events\?run_id=run-1&since_seq=41$/)
  })

  it('read the cell stream from the beginning by default', () => {
    const stop = followCell({ onEvent: () => undefined, onState: () => undefined })
    stop()
    expect(opened[0]).toMatch(/\/v1\/events\?run_id=cell&since_seq=0$/)
  })

  it('replay the cell stream from the start after a reconnect, saying so first, so a restarted server is read', () => {
    // A restarted server numbers the cell stream from 1 again: resuming after the old cursor would drop every new event.
    vi.useFakeTimers()
    const sockets: Array<{ onmessage: ((m: { data: string }) => void) | null; onclose: (() => void) | null }> = []
    vi.stubGlobal(
      'WebSocket',
      class {
        onopen: (() => void) | null = null
        onmessage: ((m: { data: string }) => void) | null = null
        onerror: (() => void) | null = null
        onclose: (() => void) | null = null
        constructor(url: string) {
          opened.push(url)
          sockets.push(this)
        }
        close() {}
      } as unknown as typeof WebSocket,
    )
    const resets: number[] = []
    const seen: number[] = []
    const stop = followCell({ onEvent: (e) => seen.push(e.seq), onState: () => undefined, onReset: () => resets.push(seen.length) })
    const frame = (seq: number) => ({ data: JSON.stringify({ type: 'cell.planner', run_id: 'cell', seq, ts: 0, severity: 'info', human: '', step: '', step_index: null, step_total: null, data: {} }) })
    sockets[0].onmessage?.(frame(41))
    sockets[0].onclose?.()
    vi.advanceTimersByTime(1000)
    expect(opened.at(-1)).toMatch(/run_id=cell&since_seq=0$/)
    expect(resets).toEqual([1])
    sockets[1].onmessage?.(frame(1))
    expect(seen).toEqual([41, 1])
    stop()
    vi.useRealTimers()
  })

  it('resume a run stream after the last seq it delivered, as before', () => {
    vi.useFakeTimers()
    const sockets: Array<{ onmessage: ((m: { data: string }) => void) | null; onclose: (() => void) | null }> = []
    vi.stubGlobal(
      'WebSocket',
      class {
        onopen: (() => void) | null = null
        onmessage: ((m: { data: string }) => void) | null = null
        onerror: (() => void) | null = null
        onclose: (() => void) | null = null
        constructor(url: string) {
          opened.push(url)
          sockets.push(this)
        }
        close() {}
      } as unknown as typeof WebSocket,
    )
    const stop = followRun('run-1', { onEvent: () => undefined, onState: () => undefined })
    sockets[0].onmessage?.({ data: JSON.stringify({ type: 'pick.ranked', run_id: 'run-1', seq: 7, ts: 0, severity: 'info', human: '', step: '', step_index: null, step_total: null, data: {} }) })
    sockets[0].onclose?.()
    vi.advanceTimersByTime(1000)
    expect(opened.at(-1)).toMatch(/run_id=run-1&since_seq=7$/)
    stop()
    vi.useRealTimers()
  })

  it('narrow an event to its typed payload by its type', () => {
    const event: RunEvent = {
      type: 'task.part_started',
      run_id: 'run-1',
      seq: 3,
      ts: 0,
      severity: 'info',
      human: 'Looking for part 1 of 1.',
      step: '',
      step_index: null,
      step_total: null,
      data: { part: 1, of: 1 },
    }
    expect(isEventOf(event, 'task.part_started')).toBe(true)
    expect(isEventOf(event, 'task.placed')).toBe(false)
    if (isEventOf(event, 'task.part_started')) expect(event.data.part).toBe(1)
  })
})
