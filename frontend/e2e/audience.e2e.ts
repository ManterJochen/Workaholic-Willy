/**
 * The audience window against a real console_dummy server (build plan 4.3): a read-only mirror of a task, of its stop
 * card after "halt now", and of the Restart that ends it.
 *
 * The run is driven through the API here (the cockpit is not this spec's subject; `smoke.e2e.ts` clicks it): what is
 * proven is that the projector's page follows the run the cell names, shows the stop card (again after a reload, from
 * the console's recovery record), drops it when the Restart's return ends the record, and never offers anything to
 * click.
 */

import { expect, test } from '@playwright/test'

import { connected, reset, untilCell, untilEnded } from './cell'

test('the audience window mirrors a task, its stop card and the restart, and offers nothing to click', async ({ page, request }) => {
  await reset(request, { down: false })
  await connected(request)

  await page.goto('/demo.html')
  const caption = page.getByRole('heading', { level: 1 })
  await expect(caption).toContainText('Bereit')
  await expect(page.getByRole('region', { name: 'STILL GRINDING' })).toContainText('Kaffeepausen')
  await expect(page.getByRole('button')).toHaveCount(0)
  await expect(page.getByRole('link')).toHaveCount(0)

  const started = await request.post('/v1/task', {
    data: { object: '', place: { kind: 'pose', pose: null }, scope: 'until_empty' },
  })
  expect(started.status()).toBe(202)
  const task = (await started.json()) as { id: string }
  await expect(caption).toContainText(/Sucht|Erkennt|Greift|Legt ab|Fährt zurück|Hände weg/)
  await expect(page.getByText('Ziel: Ablage links')).toBeVisible()

  expect((await request.post('/v1/cell/brake')).ok()).toBeTruthy()
  expect((await untilEnded(request, task.id)).stop_code).toBe('halted')
  const card = page.getByRole('alert')
  await expect(card).toContainText('Angehalten')
  await expect(card).toContainText('Bedienplatz')
  await expect(page.getByRole('button')).toHaveCount(0)
  // The card is the one thing to read: no caption repeating it, no joke beside a problem stop.
  await expect(caption).toHaveCount(0)
  await expect(page.getByRole('region', { name: 'STILL GRINDING' })).toHaveCount(0)

  // Opened after the stop, the window shows the card from the console's recovery record.
  await page.reload()
  await expect(page.getByRole('alert')).toContainText('Angehalten')

  // A person's word that the cell is clear and the hand empty: the card says it now waits for Restart or Home.
  expect((await request.post('/v1/cell/acknowledge', { data: { cell_clear: true, jaws_empty: true } })).ok()).toBeTruthy()
  await expect(page.getByRole('alert')).toContainText('Zelle freigegeben: wartet auf Neustart oder Home')
  // Then Restart: its return ends the record.
  const restarted = await request.post('/v1/task/restart', { data: { run_id: task.id } })
  expect(restarted.status()).toBe(202)
  const restart = (await restarted.json()) as { id: string }
  await untilCell(request, (c) => !c.recovery)
  await expect(page.getByRole('alert')).toHaveCount(0)

  expect((await request.post(`/v1/task/stop?run_id=${restart.id}`)).ok()).toBeTruthy()
  expect(['stopped_after_part', 'part_limit', 'nothing_left']).toContain((await untilEnded(request, restart.id)).stop_code)
  await expect(caption).toContainText(/Gestoppt|Fertig/)
  await expect(page.getByRole('region', { name: 'STILL GRINDING' })).toContainText(/Teile heute\s*[1-9]/)
})
