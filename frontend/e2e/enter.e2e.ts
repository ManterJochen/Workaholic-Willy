/**
 * Enter starts the task (the owner, 2026-10-08: "Enter ist der Klick der Person"), through the browser on console_dummy.
 *
 * The desk has no language model, so a command the server's table does not know opens the card by hand (the smoke
 * spec). A known sentence is read by the table, with no model asked: "Leg den Würfel auf Ablage links" names a part
 * and a taught pose, both found in the sentence, so on a ready cell the person's Enter starts it at once. No card
 * opens, and the chat says the first motion. The setting "Erst die Karte" gives the card back, and Enter then starts
 * nothing.
 */

import { expect, test, type APIRequestContext } from '@playwright/test'

import { connected, reset, untilCell, untilEnded, type Run } from './cell'

/** The task runs the server keeps, newest first. */
async function tasks(request: APIRequestContext): Promise<Run[]> {
  const answer = await request.get('/v1/runs')
  expect(answer.ok()).toBeTruthy()
  return ((await answer.json()) as Run[]).filter((run) => run.kind === 'task')
}

test('a known sentence and Enter start the task at once, with no card', async ({ page, request }) => {
  await reset(request, { down: false })
  await connected(request)
  const before = (await tasks(request)).length
  await page.goto('/')
  await expect(page.getByText(/Enter startet sofort/)).toBeVisible()
  const input = page.getByPlaceholder(/Was soll Willy tun/)
  await input.fill('Leg den Würfel auf Ablage links')
  await input.press('Enter')
  // The desk's rehearsal ends such a task within a second: the run is read from the server's list, not caught live.
  await expect.poll(async () => (await tasks(request)).length, { timeout: 20_000 }).toBe(before + 1)
  const started = (await tasks(request))[0]
  await expect(page.getByText(/Verstanden, ich fange an:/)).toBeVisible()
  await expect(page.getByRole('region', { name: 'Verstanden' })).toHaveCount(0)
  const ended = await untilEnded(request, started.id)
  expect(['finished', 'nothing_left']).toContain(ended.stop_code)
})

test('with "Erst die Karte" set, Enter opens the card and starts nothing', async ({ page, request }) => {
  await reset(request, { down: false })
  await connected(request)
  await page.goto('/settings')
  await page.getByRole('group', { name: 'Start', exact: true }).getByRole('button', { name: 'Erst die Karte' }).click()
  await page.goto('/')
  const input = page.getByPlaceholder(/Was soll Willy tun/)
  await input.fill('Leg den Würfel auf Ablage links')
  await input.press('Enter')
  await expect(page.getByRole('region', { name: 'Verstanden' })).toBeVisible()
  expect((await untilCell(request, () => true)).active_run_id ?? null).toBeNull()
})
