/**
 * The cockpit's and the prompt box's catalogs (build plan 4.5): German by default, English on the switch, and neither
 * may be missing a word, fill other placeholders, or say something a robot cell must not.
 *
 * `tsc` already refuses an English half that lacks a German key; these pin what the type system cannot see: no empty
 * text, the same placeholders in both languages, a brake claimed only where the arm brakes a move in flight (the owner's
 * cell latches without braking, Q1 = A), a halt never called a protective stop, and the e-stop named only to say what
 * is NOT one, or to ask for it.
 */

import { describe, expect, it } from 'vitest'

import promptDe from '../prompt/i18n.de'
import promptEn from '../prompt/i18n.en'
import cockpitDe from './i18n.de'
import cockpitEn from './i18n.en'

const CATALOGS = [
  { name: 'cockpit', de: cockpitDe as Record<string, string>, en: cockpitEn as Record<string, string> },
  { name: 'prompt', de: promptDe as Record<string, string>, en: promptEn as Record<string, string> },
]

function placeholders(text: string): string[] {
  return [...text.matchAll(/\{(\w+)(?:\|[^{}]*)?\}/g)].map((m) => m[1]).sort()
}

describe('the area catalogs', () => {
  it('name the same keys in German and English, each in its own family', () => {
    for (const { name, de, en } of CATALOGS) {
      expect(Object.keys(en).sort(), name).toEqual(Object.keys(de).sort())
      for (const key of Object.keys(de)) expect(key.startsWith(name === 'cockpit' ? 'ck.' : 'prompt.'), key).toBe(true)
    }
  })

  it('leave no text empty, and fill the same placeholders in both languages', () => {
    for (const { name, de, en } of CATALOGS) {
      for (const key of Object.keys(de)) {
        expect(de[key].trim(), `${name} de ${key}`).not.toBe('')
        expect(en[key].trim(), `${name} en ${key}`).not.toBe('')
        expect(placeholders(en[key]), `${name} ${key}`).toEqual(placeholders(de[key]))
      }
    }
  })

  it('claim a brake only in the texts drawn where the arm brakes a move in flight', () => {
    // The last two are drawn only on the arm's own "brake not confirmed", which only an arm that brakes reports.
    const braking = new Set(['ck.haltSubBrake', 'ck.haltTitleBrake', 'ck.facts.brakes', 'ck.haltAlarmSub', 'ck.haltAlarmSubArm', 'ck.stop.haltedUnconfirmed'])
    for (const [key, text] of Object.entries(cockpitDe)) if (/brems/i.test(text)) expect(braking.has(key), key).toBe(true)
    for (const [key, text] of Object.entries(cockpitEn)) if (/brak/i.test(text)) expect(braking.has(key), key).toBe(true)
  })

  it('never call the console\'s halt a protective stop, and name the e-stop only to say what is not one, or to ask for it', () => {
    for (const [key, text] of Object.entries(cockpitDe)) {
      if (/halt|Angehalten|anhalten/i.test(key) || /^Sofort anhalten|Angehalten/.test(text)) expect(text, key).not.toMatch(/Schutzstopp/)
      if (/Not-Aus/.test(text)) expect(text, key).toMatch(/kein Not-Aus|Kein Not-Aus|Not-Aus drücken/)
    }
    for (const [key, text] of Object.entries(cockpitEn)) {
      if (/e-stop|emergency stop/i.test(text) && !/stopSay|step\.pendant|stoppedSub|ck\.stop\.step/.test(key)) {
        expect(text, key).toMatch(/not the e-stop|Not an emergency stop|press the e-stop/)
      }
    }
  })

  it('say "the robot moves" on every control that starts motion', () => {
    for (const key of ['ck.motion.look', 'ck.motion.homeLook', 'ck.motion.grasp', 'ck.motion.unknown', 'ck.confirm.restartGo', 'ck.confirm.homeGo'] as const) {
      expect(cockpitDe[key], key).toMatch(/der Roboter fährt/)
      expect(cockpitEn[key], key).toMatch(/the robot moves/)
    }
  })

  it('keep the verb second after "dann" in German: no "dann der Roboter fährt"', () => {
    for (const [key, text] of Object.entries(cockpitDe)) expect(text, key).not.toMatch(/dann \{motion\}|dann der Roboter fährt/)
  })
})
