/**
 * Two languages and no dependency: `I18nProvider`, `useT()`, and the same `translate` outside React.
 *
 * **German by default, English on a switch, English without a provider.** The language lives in
 * `localStorage['willy.lang']` (anything but `en` reads as German) and `<html lang>` follows it, so a screen reader and
 * the browser's own hyphenation agree with the words. A component rendered WITHOUT the provider answers in English:
 * the console's existing screen tests render bare components and pin English sentences, and they keep passing while
 * the areas move to the catalogs one by one.
 *
 * **The text format is the smallest that reads well in both languages.** `{name}` is a parameter; `{n|one|other}` is
 * the singular or the plural by `Intl.PluralRules`, with `#` standing for the number ("1 Teil", "7 Teile"); a number
 * is formatted in the reader's language ("2,1" / "2.1"); a parameter may be a message itself and is translated in
 * turn. Nothing else: no nesting syntax, no HTML.
 *
 * **Written without JSX on purpose.** This is a `.ts` module so the provider, the hooks and the plain functions can
 * live together; a `.tsx` file that exports a component next to functions breaks Fast Refresh (oxlint warns).
 */

import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import codesDe from './codes.de'
import codesEn from './codes.en'
import commonDe from './common.de'
import commonEn from './common.en'
import type { Area, Format, Lang, MessageKey, Msg, ParamValue, Params, Translate } from './types'

export type { Area, Format, Lang, MessageKey, Msg, ParamValue, Params, Translate } from './types'

/** The key the provider and the HTML entry points read; `en` is English, anything else German. */
export const LANG_KEY = 'willy.lang'

/** BCP 47 tags for `Intl`. English is en-GB for its 24-hour clock and day-month dates, which a cell PC in Europe uses. */
const LOCALE: Record<Lang, string> = { de: 'de-DE', en: 'en-GB' }

const CORE: Record<Lang, Readonly<Record<string, string>>> = {
  de: { ...commonDe, ...codesDe },
  en: { ...commonEn, ...codesEn },
}

/** `{name}` or `{name|one|other}`. A form may hold anything but braces and bars. */
const TOKEN = /\{(\w+)(?:\|([^|{}]*)\|([^|{}]*))?\}/g

const numberFormats = new Map<string, Intl.NumberFormat>()
const pluralRules = new Map<Lang, Intl.PluralRules>()

function numberFormat(lang: Lang, digits: number): Intl.NumberFormat {
  const id = `${lang}:${digits}`
  let format = numberFormats.get(id)
  if (!format) {
    format = new Intl.NumberFormat(LOCALE[lang], { maximumFractionDigits: digits })
    numberFormats.set(id, format)
  }
  return format
}

function plural(lang: Lang, n: number): Intl.LDMLPluralRule {
  let rules = pluralRules.get(lang)
  if (!rules) {
    rules = new Intl.PluralRules(LOCALE[lang])
    pluralRules.set(lang, rules)
  }
  return rules.select(n)
}

function isMsg(value: unknown): value is Msg<string> {
  return typeof value === 'object' && value !== null && typeof (value as { key?: unknown }).key === 'string'
}

/**
 * A key's text: the area's, else the core's, else the key itself (a missing word stays visible). Something that is
 * not a key at all (a stale stored line, a payload of the wrong shape) reads as a dash: one bad line never throws.
 */
function lookup(lang: Lang, key: unknown, area?: Area<string>): string {
  if (typeof key !== 'string') return '—'
  return area?.[lang][key] ?? CORE[lang][key] ?? key
}

function render(lang: Lang, value: ParamValue, area?: Area<string>): string | undefined {
  if (value === undefined) return undefined
  if (value === null) return '—'
  if (typeof value === 'number') return numberFormat(lang, 2).format(value)
  if (typeof value === 'string') return value
  if (isMsg(value)) return fill(lang, lookup(lang, value.key, area), value.params, area)
  return String(value)
}

function fill(lang: Lang, text: string, params: Params | undefined, area?: Area<string>): string {
  if (!text.includes('{')) return text
  return text.replace(TOKEN, (whole: string, name: string, one?: string, other?: string) => {
    const value = params?.[name]
    if (one !== undefined && other !== undefined) {
      const n = typeof value === 'number' ? value : Number.NaN
      if (!Number.isFinite(n)) return whole
      const form = plural(lang, n) === 'one' ? one : other
      return form.replace(/#/g, numberFormat(lang, 2).format(n))
    }
    return render(lang, value, area) ?? whole
  })
}

/** One message in one language, outside React (the run model's tests, `document.title`, voice output). */
export function translate(lang: Lang, key: MessageKey, params?: Params): string
export function translate<K extends string>(lang: Lang, key: K, params: Params | undefined, area: Area<K>): string
export function translate(lang: Lang, key: string, params?: Params, area?: Area<string>): string {
  return fill(lang, lookup(lang, key, area), params, area)
}

/** A typed message: the key is checked against the core catalogs at compile time. */
export function msg(key: MessageKey, params?: Params): Msg {
  return params === undefined ? { key } : { key, params }
}

/** An area's catalog. The English half must name exactly the German half's keys (`tsc` says which one is missing). */
export function defineArea<K extends string>(de: Record<K, string>, en: Record<NoInfer<K>, string>): Area<K> {
  return { de, en }
}

function makeFormat(lang: Lang): Format {
  const time = new Intl.DateTimeFormat(LOCALE[lang], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
  const percent = new Intl.NumberFormat(LOCALE[lang], { style: 'percent', maximumFractionDigits: 0 })
  return {
    number: (value, digits = 2) => numberFormat(lang, digits).format(value),
    percent: (ratio) => percent.format(ratio),
    time: (unixSeconds) => time.format(new Date(unixSeconds * 1000)),
    duration: (seconds) => {
      if (!Number.isFinite(seconds) || seconds < 0) return '—'
      const whole = Math.round(seconds)
      if (whole < 60) return `${whole} s`
      const s = String(whole % 60).padStart(2, '0')
      if (whole < 3600) return `${Math.floor(whole / 60)}:${s} min`
      const m = String(Math.floor(whole / 60) % 60).padStart(2, '0')
      return `${Math.floor(whole / 3600)}:${m}:${s} h`
    },
  }
}

const formats = new Map<Lang, Format>()

/** The formatter of one language (numbers, percentages, wall-clock times, durations). */
export function formatFor(lang: Lang): Format {
  let format = formats.get(lang)
  if (!format) {
    format = makeFormat(lang)
    formats.set(lang, format)
  }
  return format
}

function makeTranslate<K extends string>(lang: Lang, area?: Area<K>): Translate<K | MessageKey> {
  const widened = area as Area<string> | undefined
  const t = ((key: string, params?: Params) => fill(lang, lookup(lang, key, widened), params, widened)) as Translate<
    K | MessageKey
  >
  Object.assign(t, {
    msg: (message: Msg<string>) => fill(lang, lookup(lang, message?.key, widened), message?.params, widened),
    lang,
    fmt: formatFor(lang),
  })
  return t
}

/** The language stored in this browser; German unless English was chosen. Never throws (a locked-down kiosk). */
export function storedLang(): Lang {
  try {
    return localStorage.getItem(LANG_KEY) === 'en' ? 'en' : 'de'
  } catch {
    return 'de'
  }
}

interface LangState {
  lang: Lang
  setLang: (lang: Lang) => void
}

const I18nContext = createContext<LangState | null>(null)

const NO_PROVIDER: LangState = { lang: 'en', setLang: () => undefined }

/**
 * The language of everything under it. `lang` fixes the starting language (a test, the audience window); without it
 * the stored choice decides, and German when nothing is stored.
 */
export function I18nProvider({ children, lang: initial }: { children?: ReactNode; lang?: Lang }) {
  const [lang, setState] = useState<Lang>(() => initial ?? storedLang())
  const setLang = useCallback((next: Lang) => {
    setState(next)
    try {
      localStorage.setItem(LANG_KEY, next)
    } catch {
      /* see storedLang() */
    }
  }, [])
  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])
  const value = useMemo(() => ({ lang, setLang }), [lang, setLang])
  return createElement(I18nContext.Provider, { value }, children)
}

/** The reader's language and the switch. Without a provider: English, and a switch that does nothing. */
export function useLang(): LangState {
  return useContext(I18nContext) ?? NO_PROVIDER
}

/**
 * A translator for the shared core (`useT()`), or for an area plus the shared core (`useT(area)`: the area's keys win).
 * Pass a module-level area constant: the translator is memoised on it.
 */
export function useT(): Translate<MessageKey>
export function useT<K extends string>(area: Area<K>): Translate<K | MessageKey>
export function useT(area?: Area<string>): Translate<string> {
  const { lang } = useLang()
  return useMemo(() => makeTranslate(lang, area), [lang, area])
}
