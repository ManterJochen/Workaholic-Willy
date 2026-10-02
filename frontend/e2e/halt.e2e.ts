/**
 * "Sofort anhalten" is one click (build plan 4.2), whatever else is open, and it ends on screen.
 *
 * The tech view's Diagnostics drawer can be opened at any time, a run included. Its backdrop covers the page to close
 * the drawer on a click beside it, and the halt under it was still visible but took two clicks: the first only closed
 * the drawer. Here a task runs on console_dummy, the drawer is open, and ONE ordinary click on the halt (Playwright
 * clicks only what would receive the click, never through another element) latches the arm. The halt's foot is inside
 * the window at the two screen sizes the cell PC and the projector use, with no scrolling.
 */

import { expect, test } from '@playwright/test'

import { connected, reset, untilCell } from './cell'

for (const size of [
  { width: 1366, height: 768 },
  { width: 1920, height: 1080 },
]) {
  test(`one click on "Sofort anhalten" halts with the Diagnostics drawer open, at ${size.width}x${size.height}`, async ({ page, request }) => {
    await reset(request, { down: false })
    await connected(request)
    await page.setViewportSize(size)
    await page.goto('/')
    const halt = page.getByRole('button', { name: /Sofort anhalten/ })
    await expect(halt).toBeEnabled()
    // The demo view, the one shown by default: the halt's foot is inside the window, no scrolling.
    const box = await halt.boundingBox()
    expect(box).not.toBeNull()
    expect(box!.y + box!.height).toBeLessThanOrEqual(size.height)

    // The tech view, where the drawer is offered: switched as a person switches it.
    await page.getByRole('group', { name: 'Ansicht', exact: true }).getByRole('button', { name: 'Technik', exact: true }).click()
    const started = await request.post('/v1/task', { data: { object: '', place: { kind: 'pose', pose: null }, scope: 'until_empty' } })
    expect(started.status()).toBe(202)
    await page.getByRole('button', { name: 'Diagnose' }).click()
    await expect(page.getByRole('dialog', { name: 'Diagnose' })).toBeVisible()

    await halt.click({ timeout: 5_000 })
    // The click landed on the halt, not on the drawer's backdrop: the arm is latched (with the task running or just
    // ended, a halt latches it either way), and the drawer is still open, because nothing closed it.
    await untilCell(request, (c) => Boolean(c.halted), 10_000)
    await expect(page.getByRole('dialog', { name: 'Diagnose' })).toBeVisible()

    await reset(request, { down: false })
  })
}
