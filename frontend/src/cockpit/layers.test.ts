/**
 * The halt stays one click away (build plan 4.2: "Sofort anhalten" is one click, no dialog), whatever else is open.
 *
 * The tech view's Diagnostics drawer can be opened at any time, a run included, and its backdrop covers the whole page
 * to close the drawer on a click beside it. Under that backdrop the halt was still visible but took two clicks: the
 * first only closed the drawer. So the stop row is layered ABOVE the drawer's backdrop (and under the drawer itself,
 * which only covers the right edge), in the wide layout and in the stacked one. The layers that are safety's own stay
 * above the stop row: the confirm dialog of a motion and the jaws question.
 *
 * jsdom draws no layers, so this reads the stylesheets; the browser check is `e2e/halt.e2e.ts`.
 */

import { describe, expect, it } from 'vitest'

import jawsCss from '../jaws/jaws.css?raw'
import shellCss from '../styles.css?raw'
import cockpitCss from './cockpit.css?raw'

/** The bodies of every rule whose selector list is exactly `selector`, in the order they appear. */
function rules(css: string, selector: string): string[] {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const found = [...css.matchAll(new RegExp(`(?:^|[};]|\\*/)\\s*${escaped}\\s*\\{([^}]*)\\}`, 'gm'))].map((m) => m[1])
  if (found.length === 0) throw new Error(`no rule for ${selector}`)
  return found
}

function layer(body: string): number {
  const match = /z-index:\s*(\d+)/.exec(body)
  if (!match) throw new Error(`no z-index in { ${body.trim()} }`)
  return Number(match[1])
}

describe('the stop row', () => {
  const backdrop = layer(rules(shellCss, '.drawer-backdrop')[0])
  const drawer = layer(rules(shellCss, '.drawer')[0])
  const stops = rules(cockpitCss, '.ck-stops').filter((body) => /z-index/.test(body) || /position/.test(body))

  it('lies above the Diagnostics drawer\'s backdrop, so one click on the halt halts, in both layouts', () => {
    // The wide layout's rule and the stacked layout's (below 1200 px) both place it.
    expect(stops.length).toBeGreaterThanOrEqual(2)
    for (const body of stops) {
      expect(body).toMatch(/position:\s*(relative|sticky)/)
      expect(layer(body)).toBeGreaterThan(backdrop)
      // Under the drawer itself: the drawer is read, the stop row is never drawn over it.
      expect(layer(body)).toBeLessThan(drawer)
    }
  })

  it('stays under the confirm dialog of a motion and under the jaws question', () => {
    const confirm = layer(rules(shellCss, '.dialog-backdrop')[0])
    const jaws = layer(rules(jawsCss, '.jd-backdrop')[0])
    for (const body of stops) {
      expect(layer(body)).toBeLessThan(confirm)
      expect(layer(body)).toBeLessThan(jaws)
    }
  })
})
