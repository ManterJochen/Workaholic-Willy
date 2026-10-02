/**
 * The prompt area's catalog (build plan 4.5): its German original and its English half, checked key for key.
 * `useT(PROMPT)` answers these keys and the shared core; without a provider it answers in English.
 */

import { defineArea } from '../i18n'
import de from './i18n.de'
import en from './i18n.en'

export const PROMPT = defineArea(de, en)

export type PromptKey = keyof typeof de
