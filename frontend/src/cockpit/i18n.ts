/**
 * The cockpit area's catalog (build plan 4.5): its German original and its English half, checked key for key.
 * `useT(COCKPIT)` answers these keys and the shared core (stop codes, lights, refusals, steps); without a provider it
 * answers in English, which is what a bare component test renders.
 */

import { defineArea } from '../i18n'
import de from './i18n.de'
import en from './i18n.en'

export const COCKPIT = defineArea(de, en)

export type CockpitKey = keyof typeof de
