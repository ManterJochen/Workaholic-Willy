/**
 * Every catalog of the console: the shell's and the codes' (`src/i18n`), the cockpit's and the prompt box's, and the
 * area catalogs of Setup, the jaws dialog, the screens and the audience window. German and English name the same keys
 * (`tsc` already refuses a missing English key; this also catches an extra one), no text is empty or its own key, and
 * both languages fill the same placeholders, so a parameter the code passes is never dropped in one language. The
 * joke stays in the audience window, the German stays plain, and no catalog shouts.
 */

import { describe, expect, it } from 'vitest'

import cockpitDe from '../cockpit/i18n.de'
import cockpitEn from '../cockpit/i18n.en'
import demoDe from '../demo/i18n.de'
import demoEn from '../demo/i18n.en'
import codesDe from '../i18n/codes.de'
import codesEn from '../i18n/codes.en'
import commonDe from '../i18n/common.de'
import commonEn from '../i18n/common.en'
import jawsDe from '../jaws/i18n.de'
import jawsEn from '../jaws/i18n.en'
import promptDe from '../prompt/i18n.de'
import promptEn from '../prompt/i18n.en'
import screensDe from '../screens/i18n.de'
import screensEn from '../screens/i18n.en'
import setupDe from './i18n.de'
import setupEn from './i18n.en'

const AREAS = [
  { name: 'common', de: commonDe as Record<string, string>, en: commonEn as Record<string, string> },
  { name: 'codes', de: codesDe as Record<string, string>, en: codesEn as Record<string, string> },
  { name: 'cockpit', de: cockpitDe as Record<string, string>, en: cockpitEn as Record<string, string> },
  { name: 'prompt', de: promptDe as Record<string, string>, en: promptEn as Record<string, string> },
  { name: 'setup', de: setupDe as Record<string, string>, en: setupEn as Record<string, string> },
  { name: 'jaws', de: jawsDe as Record<string, string>, en: jawsEn as Record<string, string> },
  { name: 'screens', de: screensDe as Record<string, string>, en: screensEn as Record<string, string> },
  { name: 'demo', de: demoDe as Record<string, string>, en: demoEn as Record<string, string> },
]

/** The placeholder NAMES a text fills, plural forms included, sorted. */
function placeholders(text: string): string[] {
  return [...text.matchAll(/\{(\w+)(?:\|[^{}]*)?\}/g)].map((m) => m[1]).sort()
}

describe('the catalogs', () => {
  it('are every catalog the console has: the shell, the codes, the cockpit, the prompt box and every area', () => {
    expect(AREAS.map((area) => area.name)).toEqual(['common', 'codes', 'cockpit', 'prompt', 'setup', 'jaws', 'screens', 'demo'])
  })

  it('name the same keys in German and in English', () => {
    for (const { name, de, en } of AREAS) expect(Object.keys(en).sort(), name).toEqual(Object.keys(de).sort())
  })

  it('leave no text empty, and no text is its own key', () => {
    for (const { name, de, en } of AREAS) {
      for (const [key, text] of [...Object.entries(de), ...Object.entries(en)]) {
        expect(text.trim(), `${name}: ${key}`).not.toBe('')
        expect(text, `${name}: ${key}`).not.toBe(key)
      }
    }
  })

  it('fill the same placeholders in both languages', () => {
    for (const { name, de, en } of AREAS) {
      for (const key of Object.keys(de)) expect(placeholders(en[key]), `${name}: ${key}`).toEqual(placeholders(de[key]))
    }
  })

  it('keep the joke in the audience window: no other catalog says coffee breaks or raises', () => {
    for (const { name, de, en } of AREAS.filter((area) => area.name !== 'demo')) {
      for (const text of [...Object.values(de), ...Object.values(en)]) {
        expect(text, name).not.toMatch(/Kaffeepause|Gehaltserhöhung|coffee break|raises? asked|STILL GRINDING/i)
      }
    }
  })

  it('speak plain German: no English "Guard" for the collision check, no "bis etwa" for "up to"', () => {
    for (const { name, de } of AREAS) {
      for (const [key, text] of Object.entries(de)) {
        expect(text, `${name}: ${key}`).not.toMatch(/\bGuard\b|bis etwa/)
      }
    }
  })

  it('never shout: no word in capitals for emphasis ("EINE Änderung")', () => {
    for (const { name, de, en } of AREAS) {
      for (const [key, text] of [...Object.entries(de), ...Object.entries(en)]) {
        expect(text, `${name}: ${key}`).not.toMatch(/\b(EINE[RNMS]?|ONE|NIE|NEVER|NICHT|NOT)\b/)
      }
    }
  })
})
