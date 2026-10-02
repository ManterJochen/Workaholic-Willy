/**
 * The console's words, as types.
 *
 * The backend writes every sentence in English and keeps it that way (`human`, `message`, `error`); the console
 * translates CODES. So a message here is a key and its parameters, never a finished sentence: the run model can say
 * "part 2 placed" without knowing which language the person reads, and the same line renders in German on the cell
 * PC and in English in a test.
 *
 * Two kinds of catalog exist, and both are typed by their keys:
 *
 * * the shared core (`common.*.ts` for the shell, `codes.*.ts` for every typed code `GET /v1/codes` serves), which
 *   `useT()` answers without being told anything;
 * * an AREA's own catalog (the cockpit, setup, the screens, the audience window), handed to `useT(area)` and checked
 *   key for key against its German original, so an English text that is missing fails `tsc`, not an operator.
 */

import type commonDe from './common.de'
import type { CodeKey } from './codes'

/** The two languages. German is the console's default; English is the switch, and the answer without a provider. */
export type Lang = 'de' | 'en'

/** A key of the shared core: the shell's words and every typed code. */
export type MessageKey = keyof typeof commonDe | CodeKey

/**
 * A message that is not a sentence yet: a key, and what fills its `{placeholders}`.
 *
 * A parameter may itself be a message (`{title}` filled with the stop code's title), so a line composed in the run
 * model stays language-free all the way down.
 */
export interface Msg<K extends string = MessageKey> {
  readonly key: K
  readonly params?: Params
}

/**
 * What fills a `{placeholder}`. A number is formatted in the reader's language (`2,1` in German, `2.1` in English);
 * `null` reads as a dash; `undefined` leaves the placeholder visible, because a parameter nobody passed is a bug a
 * test should see, not a blank an operator should guess at.
 */
export type ParamValue = string | number | null | undefined | Msg<string>

export type Params = Readonly<Record<string, ParamValue>>

/** One language of one catalog: every key, each with its text. */
export type Catalog<K extends string> = Readonly<Record<K, string>>

/** An area's catalog in both languages. The English half has exactly the German half's keys. */
export interface Area<K extends string> {
  readonly de: Catalog<K>
  readonly en: Catalog<K>
}

/** Numbers, times and durations, in the reader's language (`Intl`, no dependency). */
export interface Format {
  /** `2.1` -> `2,1` (de) or `2.1` (en); at most `digits` decimals, 2 when not said. */
  number(value: number, digits?: number): string
  /** A ratio as a percentage: `0.75` -> `75 %` (de) or `75%` (en). */
  percent(ratio: number): string
  /** Unix SECONDS (the backend's clock) as a wall-clock time, `12:31:05`. */
  time(unixSeconds: number): string
  /** A duration in seconds: `21 s`, `1:42 min`, `1:02:03 h`. */
  duration(seconds: number): string
}

/** A translator: `t(key, params)`, plus `t.msg(message)` for a composed message and the reader's language. */
export interface Translate<K extends string> {
  (key: K, params?: Params): string
  msg(message: Msg<string>): string
  readonly lang: Lang
  readonly fmt: Format
}
