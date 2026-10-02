/**
 * One outcome, one colour; one stop class, one colour.
 *
 * A refusal (a rule fired, nothing went wrong) is red; a pick that tried and did not manage is amber. Three of the
 * pick's fail-closed outcomes used to fall through to amber, which drew a refusal as a bad grasp.
 */

import { describe, expect, it } from 'vitest'

import { STOP_CODES } from '../api/codes'
import { outcomeTone, runTone, stopTone } from './outcome'

describe('outcomeTone', () => {
  it('draws every fail-closed refusal red, the three that used to read as a bad grasp included', () => {
    for (const outcome of ['uncertainty_fail_closed', 'drift_blocked_auto', 'ood_blocked_auto', 'decision_fail_closed']) {
      expect(outcomeTone(outcome), outcome).toBe('block')
    }
  })

  it('keeps a success green and a failed attempt amber', () => {
    expect(outcomeTone('succeeded')).toBe('ok')
    expect(outcomeTone('no_valid_grasp')).toBe('warn')
    expect(outcomeTone(null)).toBe('idle')
  })
})

describe('stopTone', () => {
  it('gives every stop code a tone by its class: a problem red, a question amber, done green', () => {
    expect(stopTone('halted')).toBe('block')
    expect(stopTone('target_lost')).toBe('warn')
    expect(stopTone('nothing_left')).toBe('ok')
    expect(stopTone('stopped_after_part')).toBe('info')
    expect(stopTone('')).toBe('idle')
    for (const code of STOP_CODES) expect(['ok', 'info', 'warn', 'block']).toContain(stopTone(code))
  })

  it('leaves the run states as they were', () => {
    expect(runTone('failed')).toBe('block')
    expect(runTone('finished')).toBe('ok')
  })
})
