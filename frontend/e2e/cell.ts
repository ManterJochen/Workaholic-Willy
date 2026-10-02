/**
 * What the smoke specs read and do through the API, beside the browser: the cell's state, a run's record, and a known
 * starting point (no run, no stop record, the cell down), so each spec starts from the same cell whatever ran before.
 *
 * Nothing here is a shortcut past a person's click in the spec under test: the browser does every step the spec is
 * about; these helpers only bring the shared server back to a starting point and wait for what it reports.
 */

import { expect, type APIRequestContext } from '@playwright/test'

export interface Cell {
  state: string
  active_run_id?: string | null
  halted?: { reason: string } | null
  /** `holding`: the stopped run believed a part was in the jaws. */
  recovery?: { run_id: string; cleared_at?: number | null; at: number; holding?: boolean } | null
  hand?: { kind: string } | null
}

export interface Run {
  id: string
  state: string
  kind: string
  stop_code: string
  restart_of?: string | null
  parts_placed: number
}

export async function cell(request: APIRequestContext): Promise<Cell> {
  const answer = await request.get('/v1/cell')
  expect(answer.ok()).toBeTruthy()
  return (await answer.json()) as Cell
}

export async function run(request: APIRequestContext, id: string): Promise<Run> {
  const answer = await request.get(`/v1/runs/${id}`)
  expect(answer.ok()).toBeTruthy()
  return (await answer.json()) as Run
}

/** Poll until `test` holds for the cell, or fail with what the cell said last. */
export async function untilCell(request: APIRequestContext, test: (c: Cell) => boolean, timeout = 60_000): Promise<Cell> {
  let last: Cell | null = null
  await expect
    .poll(
      async () => {
        last = await cell(request)
        return test(last)
      },
      { timeout, intervals: [250, 500, 1000] },
    )
    .toBe(true)
  return last as unknown as Cell
}

/** Poll until the run has ended. */
export async function untilEnded(request: APIRequestContext, id: string, timeout = 60_000): Promise<Run> {
  let last: Run | null = null
  await expect
    .poll(
      async () => {
        last = await run(request, id)
        return last.state !== 'running'
      },
      { timeout, intervals: [250, 500, 1000] },
    )
    .toBe(true)
  return last as unknown as Run
}

/** Built (rehearsal) and connected, through the API: for a spec whose subject is not the Setup page. */
export async function connected(request: APIRequestContext): Promise<void> {
  let now = await cell(request)
  if (now.state === 'disconnected') {
    expect((await request.post('/v1/cell/build?rehearse=true')).ok()).toBeTruthy()
    now = await cell(request)
  }
  if (now.state !== 'connected') {
    const preview = await (await request.get('/v1/cell/connect-preview')).json()
    expect((await request.post('/v1/cell/connect', { data: { token: preview.token } })).ok()).toBeTruthy()
  }
  await untilCell(request, (c) => c.state === 'connected')
}

/**
 * Back to a known cell: no run, no latch, no stop record, and (with `down`) the cell disconnected. A stop record ends
 * only when a run arrives at its return pose, so one left by an earlier spec is ended the way a person would: the cell
 * declared clear and empty, then Home.
 */
export async function reset(request: APIRequestContext, { down = true } = {}): Promise<void> {
  let now = await cell(request)
  if (now.active_run_id) {
    await request.post(`/v1/task/stop?run_id=${now.active_run_id}`)
    await untilCell(request, (c) => !c.active_run_id)
    now = await cell(request)
  }
  if (now.halted || now.recovery) {
    await request.post('/v1/cell/acknowledge', { data: { cell_clear: true, jaws_empty: true } })
    now = await cell(request)
  }
  if (now.recovery) {
    if (now.state !== 'connected') await connected(request)
    const home = await request.post('/v1/cell/home', { data: { to: 'home' } })
    expect(home.ok()).toBeTruthy()
    await untilCell(request, (c) => !c.recovery && !c.active_run_id)
  }
  if (down && (await cell(request)).state === 'connected') {
    expect((await request.post('/v1/cell/disconnect')).ok()).toBeTruthy()
  }
}
