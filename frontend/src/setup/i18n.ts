/** Setup's catalog, both languages: `useT(SETUP)` answers these keys and the shared core. */

import { defineArea, type MessageKey, type Translate } from '../i18n'
import de from './i18n.de'
import en from './i18n.en'

export const SETUP = defineArea(de, en)

/** A translator of Setup's area (its keys, then the shared core's). */
export type SetupT = Translate<keyof typeof de | MessageKey>
