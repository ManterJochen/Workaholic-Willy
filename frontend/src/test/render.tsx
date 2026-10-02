/**
 * Rendering a piece of the console the way it runs, for tests.
 *
 * Three helpers, so every area tests against the same providers and the same fakes:
 *
 * * `renderWith(ui, { lang, route })` renders under a router and the console's providers (language, preferences, the
 *   shared cell poll, the run on screen). German unless the test says otherwise, because German is what the cell PC
 *   shows; `lang: 'en'` pins the English copy;
 * * `stubApi(routes)` answers `fetch` from a table keyed `"METHOD /path"` (or just `"/path"` for any method) and
 *   RECORDS every call, so a test can assert what was sent, and that nothing that moves was sent unasked. An
 *   unknown route answers the backend's own 404 envelope;
 * * `fakeSockets()` replaces `WebSocket`; each socket the console opens is listed with its URL, and a test feeds it
 *   events (`deliver`) as the server would.
 *
 * And three readers of the contract's event logs (`fixtures/*.json`) that never name a run id or a seq: the run
 * track regenerates those logs on console_dummy, where every run id is drawn at random (`run-<hex>`), so a test that
 * picks a run by its record (`runOf`) and an event by its type and position (`upTo`) survives the regeneration.
 *
 * Deliberately no component is defined in this file (only imported ones are rendered), so it can export plain
 * functions next to JSX without breaking Fast Refresh's rule.
 */

import { render, type RenderResult } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'

import type { RunOut } from '../api/client'
import type { RunEvent } from '../api/events'
import { I18nProvider, type Lang } from '../i18n'
import { PrefsProvider } from '../model/prefs'
import { CellProvider } from '../model/useCell'
import { RunProvider } from '../model/useRun'

export interface RenderOptions {
  /** The UI language. German unless said (the console's default). */
  lang?: Lang
  /** The route the router starts at. */
  route?: string
  /** `false` renders under the router and the language only: no cell poll, no run. */
  providers?: boolean
}

/** Render `ui` as the console would: router, language, preferences, the shared cell, the run on screen. */
export function renderWith(ui: ReactElement, options: RenderOptions = {}): RenderResult {
  const { lang = 'de', route = '/', providers = true } = options
  const inner = providers ? (
    <PrefsProvider>
      <CellProvider>
        <RunProvider>{ui}</RunProvider>
      </CellProvider>
    </PrefsProvider>
  ) : (
    ui
  )
  return render(
    <MemoryRouter initialEntries={[route]}>
      <I18nProvider lang={lang}>{inner}</I18nProvider>
    </MemoryRouter>,
  )
}

export interface ApiCall {
  readonly method: string
  readonly path: string
  /** The query string without the `?`, `''` when there is none. */
  readonly query: string
  readonly body: unknown
}

/** An answer with a status of its own: a refusal, an empty 204. */
export interface Reply {
  readonly reply: true
  readonly status: number
  readonly body: unknown
}

/** An answer with its own status, e.g. `reply(409, { code: 'run_active', message: '…', detail: {} })`. */
export function reply(status: number, body: unknown = {}): Reply {
  return { reply: true, status, body }
}

/** A refusal in the backend's one envelope. */
export function refusal(status: number, code: string, message = '', detail: Record<string, unknown> = {}): Reply {
  return reply(status, { code, message, detail })
}

function isReply(value: unknown): value is Reply {
  return typeof value === 'object' && value !== null && (value as { reply?: unknown }).reply === true
}

export type Route = unknown | ((call: ApiCall) => unknown)

/**
 * Answer `fetch` from `routes` and record every call. A route is keyed `"METHOD /path"` or `"/path"`; its value is
 * the JSON body (200), a `reply(...)`, or a function of the call returning either, or a promise of either (a slow
 * server).
 */
export function stubApi(routes: Record<string, Route>): ApiCall[] {
  const calls: ApiCall[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string, init?: RequestInit) => {
      const url = String(input)
      const [path, query = ''] = url.split('?')
      const method = (init?.method ?? 'GET').toUpperCase()
      let body: unknown
      if (typeof init?.body === 'string') {
        try {
          body = JSON.parse(init.body)
        } catch {
          body = init.body
        }
      }
      const call: ApiCall = { method, path, query, body }
      calls.push(call)
      const route = routes[`${method} ${path}`] ?? routes[path]
      let answer: unknown = typeof route === 'function' ? await (route as (c: ApiCall) => unknown)(call) : route
      if (answer === undefined) answer = refusal(404, 'http_404', 'Not Found')
      const status = isReply(answer) ? answer.status : 200
      const payload = isReply(answer) ? answer.body : answer
      if (status === 204) return new Response(null, { status })
      return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
    }),
  )
  return calls
}

export interface FakeSocketHandle {
  readonly url: string
  /** The `run_id` the socket asked for (`cell` for the cell stream). */
  readonly runId: string
  readonly sinceSeq: number
  readonly closed: boolean
  /** Deliver frames as the server would: each is JSON-encoded and handed to `onmessage`. */
  deliver(frames: readonly unknown[]): void
}

export interface FakeSockets {
  readonly sockets: readonly FakeSocketHandle[]
  /** The newest socket that follows `runId`. */
  find(runId: string): FakeSocketHandle | undefined
}

/** Replace `WebSocket` with fakes the test drives. Each opens on the next tick, as a real one would. */
export function fakeSockets(): FakeSockets {
  const sockets: FakeSocketHandle[] = []
  const fake = class {
    onopen: (() => void) | null = null
    onmessage: ((message: { data: string }) => void) | null = null
    onerror: (() => void) | null = null
    onclose: (() => void) | null = null
    readonly handle: FakeSocketHandle

    constructor(url: string) {
      const params = new URL(url).searchParams
      let closed = false
      const deliver = (frames: readonly unknown[]) => {
        for (const frame of frames) {
          if (closed) return
          this.onmessage?.({ data: JSON.stringify(frame) })
        }
      }
      this.handle = {
        url,
        runId: params.get('run_id') ?? '',
        sinceSeq: Number(params.get('since_seq') ?? 0),
        get closed() {
          return closed
        },
        deliver,
      }
      this.close = () => {
        closed = true
      }
      sockets.push(this.handle)
      setTimeout(() => this.onopen?.(), 0)
    }

    close(): void {}
  }
  vi.stubGlobal('WebSocket', fake as unknown as typeof WebSocket)
  return {
    sockets,
    find: (runId: string) => [...sockets].reverse().find((s) => s.runId === runId),
  }
}

/** One of the contract's event logs: the events in the order the browser saw them, and each run's record. */
export interface Fixture {
  readonly about?: string
  readonly scenario: string
  readonly events: RunEvent[]
  readonly runs: Record<string, RunOut>
}

/**
 * The id of the one run in `fixture` whose record matches (every run when no test is given): a run is chosen by what
 * it is, never by its id. Throws unless exactly one run matches, so a regenerated log that changed the scenario says so.
 */
export function runOf(fixture: Fixture, match: (record: RunOut) => boolean = () => true): string {
  const found = Object.entries(fixture.runs).filter(([, record]) => match(record))
  if (found.length !== 1) throw new Error(`${fixture.scenario}: ${found.length} runs match, not one`)
  return found[0][0]
}

/** One run's events, in order. */
export function eventsOf(fixture: Fixture, runId: string): RunEvent[] {
  return fixture.events.filter((event) => event.run_id === runId)
}

/** The events up to and including the `nth` (1-based) one that matches: a moment chosen by what happened, not by seq. */
export function upTo(events: readonly RunEvent[], match: (event: RunEvent) => boolean, nth = 1): RunEvent[] {
  let seen = 0
  const at = events.findIndex((event) => match(event) && ++seen === nth)
  if (at < 0) throw new Error(`no event number ${nth} matches`)
  return events.slice(0, at + 1)
}

/** `upTo(events, isType('task.placed'))`: the run as it stood at its first placed part. */
export function isType(type: string): (event: RunEvent) => boolean {
  return (event) => event.type === type
}
