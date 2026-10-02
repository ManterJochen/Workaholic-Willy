/**
 * Dark by default, light on a switch, and no white flash on the way in.
 *
 * The console is a control room: dark unless a person chose light (OD 3). The choice is stamped on `<html data-theme>`
 * twice, on purpose: by an inline script in each HTML entry before the first paint, and by `theme.ts` after React
 * started. These tests run the inline scripts themselves, so the two can never disagree about the default.
 */

import { act, cleanup, render, screen } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import demoHtml from '../../demo.html?raw'
import indexHtml from '../../index.html?raw'
import { THEME_KEY, applyTheme, readTheme, useTheme } from './theme'

/** The first inline (non-module) script of an HTML entry, as a function the test can run. */
function inlineScript(html: string): () => void {
  const match = /<script>([\s\S]*?)<\/script>/.exec(html)
  if (!match) throw new Error('no inline script')
  return new Function(match[1]) as () => void
}

beforeEach(() => {
  localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
  document.documentElement.removeAttribute('lang')
})

afterEach(() => {
  cleanup()
  localStorage.clear()
})

describe('the theme', () => {
  it('is dark when nothing is stored, and when an old "system" choice is', () => {
    expect(readTheme()).toBe('dark')
    localStorage.setItem(THEME_KEY, 'system')
    expect(readTheme()).toBe('dark')
    localStorage.setItem(THEME_KEY, 'light')
    expect(readTheme()).toBe('light')
  })

  it('is always stamped, so the page never falls back to the operating system\'s choice', () => {
    applyTheme('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    applyTheme('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('toggles dark and light, and remembers the choice', () => {
    function Probe() {
      const { theme, toggle } = useTheme()
      return createElement('button', { onClick: toggle }, theme)
    }
    render(createElement(Probe))
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    act(() => screen.getByText('dark').click())
    expect(screen.getByText('light')).toBeTruthy()
    expect(localStorage.getItem(THEME_KEY)).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })
})

describe('the inline scripts of both pages', () => {
  it('index.html stamps dark before the first paint unless light is stored', () => {
    inlineScript(indexHtml)()
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    localStorage.setItem(THEME_KEY, 'light')
    inlineScript(indexHtml)()
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
    localStorage.setItem(THEME_KEY, 'system')
    inlineScript(indexHtml)()
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })

  it('demo.html stamps dark before the first paint whatever is stored: the audience window is the black stage', () => {
    // The brand's look on a projector (owner, 2026-10-01): `demo/main.tsx` stamps dark too, but only after the first
    // paint, so a light choice here would show the audience one light frame first.
    for (const stored of [null, 'light', 'system']) {
      if (stored === null) localStorage.removeItem(THEME_KEY)
      else localStorage.setItem(THEME_KEY, stored)
      inlineScript(demoHtml)()
      expect(document.documentElement.getAttribute('data-theme'), String(stored)).toBe('dark')
    }
  })

  for (const [name, html] of [['index.html', indexHtml], ['demo.html', demoHtml]] as const) {

    it(`${name} sets the page language before the first paint: German unless English is stored`, () => {
      inlineScript(html)()
      expect(document.documentElement.lang).toBe('de')
      localStorage.setItem('willy.lang', 'en')
      inlineScript(html)()
      expect(document.documentElement.lang).toBe('en')
    })
  }
})
