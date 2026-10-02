/**
 * The console speaks German by default and English on a switch, and neither language may be missing a word.
 *
 * `tsc` already refuses an English catalog that lacks a German key (each `en` file is typed by its `de` file); these
 * tests pin what the type system cannot see: that no text is empty, that both languages fill the same placeholders,
 * that every code the backend can send has a sentence in both, and that a component rendered WITHOUT a provider (the
 * existing screen tests do exactly that) still answers in English.
 */

import { act, cleanup, render, screen } from '@testing-library/react'
import { createElement, useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  BLOCKER_CODES,
  COMMAND_NOTES,
  EVENT_TYPES,
  JAWS_CHOICES,
  JAWS_STAGES,
  LIGHT_CODES,
  LIGHT_IDS,
  LIGHT_STATES,
  REFUSAL_CODES,
  RUN_KINDS,
  STOP_CLASSES,
  STOP_CODES,
} from '../api/codes'
import commonDe from './common.de'
import commonEn from './common.en'
import { EVENT_VARIANTS, OUTCOMES, outcomeMsg, refusalMsg, stopMsg } from './codes'
import codesDe from './codes.de'
import codesEn from './codes.en'
import { I18nProvider, LANG_KEY, defineArea, msg, translate, useLang, useT } from './index'

const CATALOGS = [
  { name: 'common', de: commonDe as Record<string, string>, en: commonEn as Record<string, string> },
  { name: 'codes', de: codesDe as Record<string, string>, en: codesEn as Record<string, string> },
]

/** The placeholder NAMES a text fills, plural forms included, sorted. */
function placeholders(text: string): string[] {
  return [...text.matchAll(/\{(\w+)(?:\|[^{}]*)?\}/g)].map((m) => m[1]).sort()
}

afterEach(() => {
  cleanup()
  try {
    localStorage.clear()
  } catch {
    /* jsdom always has it; a browser without it is what storedLang() guards */
  }
})

describe('the catalogs', () => {
  it('name the same keys in German and in English, catalog by catalog', () => {
    for (const { name, de, en } of CATALOGS) {
      expect(Object.keys(en).sort(), name).toEqual(Object.keys(de).sort())
    }
  })

  it('leave no text empty, and no text is its own key', () => {
    for (const { name, de, en } of CATALOGS) {
      for (const [key, text] of [...Object.entries(de), ...Object.entries(en)]) {
        expect(text.trim(), `${name}: ${key}`).not.toBe('')
        expect(text, `${name}: ${key}`).not.toBe(key)
      }
    }
  })

  it('fill the same placeholders in both languages', () => {
    for (const { name, de, en } of CATALOGS) {
      for (const key of Object.keys(de)) {
        expect(placeholders(en[key]), `${name}: ${key}`).toEqual(placeholders(de[key]))
      }
    }
  })

  it('give every code the backend can send a German and an English text', () => {
    const families: Array<[string, readonly string[]]> = [
      ['stop', STOP_CODES],
      ['stopSay', STOP_CODES],
      ['stopClass', STOP_CLASSES],
      ['event', EVENT_TYPES],
      ['event', EVENT_VARIANTS],
      ['refusal', REFUSAL_CODES],
      ['light', LIGHT_CODES],
      ['lightId', LIGHT_IDS],
      ['lightState', LIGHT_STATES],
      ['blocker', BLOCKER_CODES],
      ['note', COMMAND_NOTES],
      ['runKind', RUN_KINDS],
      ['jawsStage', JAWS_STAGES],
      ['jawsChoice', JAWS_CHOICES],
      ['outcome', OUTCOMES],
    ]
    const de = codesDe as Record<string, string>
    const en = codesEn as Record<string, string>
    let expected = 0
    for (const [family, codes] of families) {
      for (const code of codes) {
        expected += 1
        expect(de[`${family}.${code}`], `de ${family}.${code}`).toBeTruthy()
        expect(en[`${family}.${code}`], `en ${family}.${code}`).toBeTruthy()
      }
    }
    // Nothing else hides in the code catalogs: every key there belongs to one family.
    expect(Object.keys(de).length).toBe(expected)
  })

  it('say "halted" and never "Schutzstopp" for a halt, in both languages', () => {
    // A halt is the console's own brake; a protective stop is the controller's. The stop card must never mix them up.
    expect(translate('de', 'stop.halted')).toBe('Angehalten')
    expect(translate('de', 'stop.halted')).not.toMatch(/Schutzstopp/)
    expect(translate('en', 'stop.halted')).toBe('Halted')
  })

  it('never claim a halt braked the arm: the owner\'s cell latches without braking (Q1 = A, brake_on_halt off)', () => {
    // With the brake off, the move in flight ends at its target and only the next one is refused; with it on, the move
    // is braked. What every halt text says must be true of both: nothing was commanded after it, the arm stands.
    const halts = (catalog: Record<string, string>) => Object.entries(catalog).filter(([key]) => /halt/i.test(key))
    for (const [key, text] of [...halts(codesDe), ...halts(commonDe)]) expect(text, `de ${key}`).not.toMatch(/brems/i)
    for (const [key, text] of [...halts(codesEn), ...halts(commonEn)]) expect(text, `en ${key}`).not.toMatch(/brak/i)
    expect(translate('de', 'stopSay.halted')).toMatch(/keine Bewegung mehr befohlen/)
    expect(translate('en', 'stopSay.halted')).toMatch(/no motion was commanded after it/)
  })

  it('say a lost target only what is true wherever it was lost, before a pick as well as at the drop', () => {
    // A fixed camera looks at its bin again before every pick and ends `target_lost` there, with no part in hand and
    // nothing put back; a wrist camera loses it at the drop. `footprint_changed` also means a bin of the same size
    // whose rim stands more than 20 mm off: another bin, another size or height (the task library, 2026-10-01).
    const lost = [
      'stopSay.target_lost', 'event.task.target_lost.not_seen', 'event.task.target_lost.footprint_changed',
    ] as const
    for (const key of lost) {
      expect(translate('de', key), `de ${key}`).not.toMatch(/Ablegen|liegt wieder/)
      expect(translate('en', key), `en ${key}`).not.toMatch(/at the drop|back where/)
    }
    expect(translate('de', 'event.task.target_lost.footprint_changed')).toMatch(/Größe oder Höhe/)
    expect(translate('en', 'event.task.target_lost.footprint_changed')).toMatch(/size or height/)
  })
})

describe('translate', () => {
  it('fills {name} placeholders, and leaves an unpassed one visible', () => {
    expect(translate('en', 'run.part', { part: 'two' })).toBe('part two')
    expect(translate('en', 'run.part')).toBe('part {part}')
  })

  it('reads null as a dash rather than as the word null', () => {
    expect(translate('en', 'run.part', { part: null })).toBe('part —')
    expect(translate('en', 'run.part', { part: null })).not.toContain('null')
  })

  it('chooses the singular or the plural form by the number', () => {
    expect(translate('de', 'event.run_finished.task', { parts: 1 })).toBe('Fertig: 1 Teil abgelegt.')
    expect(translate('de', 'event.run_finished.task', { parts: 7 })).toBe('Fertig: 7 Teile abgelegt.')
    expect(translate('en', 'event.run_finished.task', { parts: 1 })).toBe('Done: 1 part placed.')
    expect(translate('en', 'event.run_finished.task', { parts: 0 })).toBe('Done: 0 parts placed.')
  })

  it('formats a number in the reader\'s language', () => {
    expect(translate('de', 'event.pick.ranked', { count: 14, score: 0.87 })).toContain('0,87')
    expect(translate('en', 'event.pick.ranked', { count: 14, score: 0.87 })).toContain('0.87')
  })

  it('translates a message nested in a parameter', () => {
    const line = translate('de', 'event.run_error', { title: stopMsg('halted') })
    expect(line).toBe('Gestoppt: Angehalten.')
  })

  it('falls back to "Refused (<code>)" for a code it does not know, and to the code\'s own sentence otherwise', () => {
    expect(translate('de', refusalMsg('no_such_thing').key, refusalMsg('no_such_thing').params)).toBe(
      'Abgelehnt (no_such_thing)',
    )
    expect(translate('en', refusalMsg('http_404').key, refusalMsg('http_404').params)).toBe('Refused (http_404)')
    expect(translate('de', refusalMsg('run_active').key)).toBe('Ein Lauf ist aktiv.')
  })

  it('shows an outcome it does not know as its own words', () => {
    const unknown = outcomeMsg('brand_new_outcome')
    expect(translate('en', unknown.key, unknown.params)).toBe('brand new outcome')
    expect(translate('de', outcomeMsg('succeeded').key)).toBe('gegriffen')
  })

  it('answers the key itself for a key no catalog has, so a missing word is visible', () => {
    expect(translate('de', 'no.such.key' as never)).toBe('no.such.key')
  })

  it('answers a dash, never a crash, for a message that is no message (a stale stored line)', () => {
    expect(translate('de', undefined as never)).toBe('—')
    expect(translate('en', 7 as never, { n: 1 })).toBe('—')
  })
})

function Probe() {
  const t = useT()
  const { lang } = useLang()
  return createElement('p', { 'data-lang': lang }, t('nav.cockpit') + ' / ' + t('nav.settings'))
}

describe('useT', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('lang')
  })

  it('answers English when no provider wraps it', () => {
    render(createElement(Probe))
    expect(screen.getByText('Cockpit / Settings')).toBeTruthy()
  })

  it('answers German under the provider when nothing is stored, and <html lang> follows', () => {
    render(createElement(I18nProvider, null, createElement(Probe)))
    expect(screen.getByText('Cockpit / Einstellungen')).toBeTruthy()
    expect(document.documentElement.lang).toBe('de')
  })

  it('reads the stored language, and a switch is remembered', () => {
    localStorage.setItem(LANG_KEY, 'en')
    let setLang: ((lang: 'de' | 'en') => void) | null = null
    function Switch() {
      const set = useLang().setLang
      useEffect(() => {
        setLang = set
      }, [set])
      return null
    }
    render(createElement(I18nProvider, null, createElement(Probe), createElement(Switch)))
    expect(screen.getByText('Cockpit / Settings')).toBeTruthy()
    act(() => setLang!('de'))
    expect(screen.getByText('Cockpit / Einstellungen')).toBeTruthy()
    expect(localStorage.getItem(LANG_KEY)).toBe('de')
    expect(document.documentElement.lang).toBe('de')
  })

  it('lets an area add its own keys while the shared ones still answer', () => {
    const area = defineArea({ 'probe.hello': 'Hallo {name}' }, { 'probe.hello': 'Hello {name}' })
    function AreaProbe() {
      const t = useT(area)
      return createElement('p', null, `${t('probe.hello', { name: 'Willy' })} | ${t('nav.history')}`)
    }
    render(createElement(I18nProvider, { lang: 'de' }, createElement(AreaProbe)))
    expect(screen.getByText('Hallo Willy | Verlauf')).toBeTruthy()
  })

  it('renders a composed message with t.msg', () => {
    function MsgProbe() {
      const t = useT()
      return createElement('p', null, t.msg(msg('event.task.part_started.of', { part: 2, of: 3 })))
    }
    render(createElement(I18nProvider, { lang: 'de' }, createElement(MsgProbe)))
    expect(screen.getByText('Suche Teil 2 von 3.')).toBeTruthy()
  })

  it('renders a malformed message as a dash with t.msg, so one bad line cannot take the page down', () => {
    function BadProbe() {
      const t = useT()
      return createElement('p', null, `[${t.msg({} as never)}]`)
    }
    render(createElement(I18nProvider, { lang: 'de' }, createElement(BadProbe)))
    expect(screen.getByText('[—]')).toBeTruthy()
  })

  it('formats times and durations in the language', () => {
    function FmtProbe() {
      const t = useT()
      return createElement('p', null, `${t.fmt.duration(21.4)} | ${t.fmt.duration(102)} | ${t.fmt.percent(0.75)}`)
    }
    render(createElement(I18nProvider, { lang: 'de' }, createElement(FmtProbe)))
    expect(screen.getByText('21 s | 1:42 min | 75 %')).toBeTruthy()
  })
})
