/** The screens' catalog, both languages: `useT(SCREENS)` answers these keys and the shared core. */

import { defineArea, type MessageKey, type Translate } from '../i18n'
import de from './i18n.de'
import en from './i18n.en'

export const SCREENS = defineArea(de, en)

/** A translator of the screens' area (its keys, then the shared core's). */
export type ScreensT = Translate<keyof typeof de | MessageKey>
