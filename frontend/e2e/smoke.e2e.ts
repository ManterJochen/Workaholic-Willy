/**
 * The smoke test of build plan 6.4: connect -> task -> stop card -> "Zelle ist frei" -> Restart, through the browser,
 * on console_dummy with the built bundle (`e2e/serve.mjs` starts the server; see `playwright.config.ts`).
 *
 * Every step a person takes is a click here, on the button that names it: Setup opens at the checklist and goes on to
 * Aufbauen at the stepper, builds, reads the preview and connects; the cockpit takes a command, shows what it
 * understood, and starts only at Start; "Sofort anhalten" halts the task; the stop card (the region named by its title)
 * takes "Zelle ist frei", then "Backen leer" where it offers it (a hand that is not a toggle, once the cell is clear;
 * always where the stopped run believed a part was in the jaws), and Restart asks its confirm dialog, whose first
 * motion is the planned move home. The task runs "until empty",
 * because the dummy's rehearsal scene never empties: about seven seconds, long enough to halt it mid-run.
 *
 * The cockpit's words are the build plan's copy (4.2: "Bis leer", "Start – …", "Sofort anhalten", "Zelle ist frei",
 * "Backen leer", "Neustart", "Nach diesem Teil stoppen"); the cockpit track builds that screen, so a label it words
 * otherwise is changed here, never worked around.
 */

import { expect, test, type Page } from '@playwright/test'

import { reset, untilCell, untilEnded } from './cell'

/** Tick a checkbox that names the step where the cockpit offers one; then press the button that says it. */
async function sayTheCellIsClear(page: Page): Promise<void> {
  const looked = page.getByRole('checkbox', { name: /Zelle ist frei|niemand ist im Arbeitsraum/ })
  if ((await looked.count()) > 0) await looked.first().check()
  await page.getByRole('button', { name: /Zelle ist frei/ }).first().click()
}

test('connect in Setup, start a task in the cockpit, halt it, clear the cell and restart', async ({ page, request }) => {
  await reset(request)

  await test.step('Setup: build (a rehearsal on the dummy arm), read the preview, connect with its token', async () => {
    await page.goto('/setup')
    await expect(page.getByRole('heading', { level: 2, name: 'Einrichten' })).toBeVisible()
    // The stepper opens at the step the cell is at: the checklist for a cell that is down (it is never skipped), the
    // preview for one an earlier spec built. Going to "Aufbauen" from it builds again.
    await page.getByRole('list', { name: 'Schritte' }).getByRole('button', { name: /Aufbauen/ }).click()
    // The dummy arm rehearses by default; a real arm would build for real.
    await expect(page.getByRole('checkbox', { name: /Probe/ })).toBeChecked()
    await page.getByRole('button', { name: /^(Zelle aufbauen|Neu aufbauen)$/ }).click()
    await page.getByRole('button', { name: 'Vorschau lesen' }).click()
    const connect = page.getByRole('button', { name: 'Verbinden – der Roboter bewegt sich' })
    await expect(connect).toBeEnabled()
    await connect.click()
    await expect(page.getByText('Alles bereit.')).toBeVisible()
  })

  let taskId = ''
  await test.step('Cockpit: a command, the Understood card, then Start (nothing moves before it)', async () => {
    await page.getByRole('link', { name: 'Zum Cockpit' }).click()
    const input = page.getByPlaceholder(/Was soll Willy tun/)
    await input.fill('Räum alles auf die Ablage')
    await input.press('Enter')
    await page.getByRole('button', { name: 'Bis leer' }).click()
    // Parsing starts nothing: no run before the click on Start.
    expect((await untilCell(request, () => true)).active_run_id ?? null).toBeNull()
    await page.getByRole('button', { name: /^Start/ }).click()
    taskId = (await untilCell(request, (c) => Boolean(c.active_run_id))).active_run_id as string
  })

  await test.step('Sofort anhalten: the task stops, and the stop card says it was halted', async () => {
    await page.getByRole('button', { name: /Sofort anhalten/ }).click()
    const ended = await untilEnded(request, taskId)
    expect(ended.stop_code).toBe('halted')
    // The stop card itself (a region named by its title), not the top bar's chip that may say the same word.
    await expect(page.getByRole('region', { name: 'Angehalten' })).toBeVisible()
  })

  await test.step('"Zelle ist frei" and "Backen leer", then Restart through its confirm dialog', async () => {
    await sayTheCellIsClear(page)
    const cleared = await untilCell(request, (c) => Boolean(c.recovery?.cleared_at && c.recovery.cleared_at >= c.recovery.at))
    // A hand that is not a toggle is emptied by a person's word, a button of the stop card, offered once the cell is
    // clear. Where the stopped run believed a part was in the jaws (the halt came while one was carried), the card MUST
    // offer it, and Restart waits for it: the server refuses the way back with a part it cannot rule out.
    const empty = page.getByRole('region', { name: 'Angehalten' }).getByRole('button', { name: 'Backen leer' })
    if (cleared.recovery?.holding && cleared.hand?.kind !== 'toggle') await expect(empty).toBeVisible()
    if ((await empty.count()) > 0) {
      await expect(empty.first()).toBeEnabled()
      await empty.first().click()
      await untilCell(request, (c) => c.recovery?.holding !== true)
    }
    await page.getByRole('button', { name: /Neustart/ }).first().click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toContainText(/Home/)
    // The confirm that moves is the arming red; the focus starts on Cancel.
    await dialog.locator('button.primary').click()
    // The restart's first motion is the planned move home; its arrival ends the stop record.
    await untilCell(request, (c) => !c.recovery && Boolean(c.active_run_id))
  })

  await test.step('Nach diesem Teil stoppen: the restarted task places its part and returns', async () => {
    const restart = (await untilCell(request, (c) => Boolean(c.active_run_id))).active_run_id as string
    await page.getByRole('button', { name: /Nach diesem Teil stoppen/ }).click()
    const ended = await untilEnded(request, restart)
    expect(['stopped_after_part', 'part_limit', 'nothing_left']).toContain(ended.stop_code)
  })
})
