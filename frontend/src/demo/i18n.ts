/** The audience window's catalog, both languages: `useT(DEMO)` answers these keys and the shared core. */

import { defineArea, type MessageKey, type Translate } from '../i18n'
import de from './i18n.de'
import en from './i18n.en'

export const DEMO = defineArea(de, en)

export type DemoKey = keyof typeof de

/** A translator of the audience window's area (its keys, then the shared core's). */
export type DemoT = Translate<DemoKey | MessageKey>
