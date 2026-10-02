/**
 * The console's browser smoke tests (the owner's answer of 2026-09-30: Playwright as a dev dependency, on
 * console_dummy, not in CI at first).
 *
 * They drive the BUILT bundle (`npm run build` first: the server serves `api/static`) through a real
 * `python -m api --profile console_dummy` server that `e2e/serve.mjs` starts for the run, on a scratch copy of the
 * config: the shipped desk profile as it is, with its two poses. The repo's own config is never written.
 *
 *   npm run build && npm run e2e
 *
 * `WILLY_PYTHON` names the interpreter with the project's environment (default `python`); `WILLY_E2E_PORT` the port
 * (default 8761); `WILLY_E2E_URL` points the run at a console that is already up instead of starting one.
 *
 * The specs are `e2e/*.e2e.ts`, never `*.spec.ts` or `*.test.ts`: Vitest collects those, and these are not its tests.
 * The run's output stays under `node_modules/.cache` (ignored by git); the reporter is the plain list.
 */

import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.WILLY_E2E_PORT ?? 8761)
const EXTERNAL = process.env.WILLY_E2E_URL

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.e2e\.ts$/,
  timeout: 180_000,
  expect: { timeout: 20_000 },
  // One cell, one server: the specs share it, so they run one after the other.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  outputDir: './node_modules/.cache/willy-e2e/results',
  use: {
    baseURL: EXTERNAL ?? `http://127.0.0.1:${PORT}`,
    // A control that is not there fails its step in 30 s, not at the end of the test's three minutes.
    actionTimeout: 30_000,
    viewport: { width: 1600, height: 1000 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1600, height: 1000 } } }],
  webServer: EXTERNAL
    ? undefined
    : {
        command: `node e2e/serve.mjs --port ${PORT}`,
        url: `http://127.0.0.1:${PORT}/v1/health`,
        reuseExistingServer: false,
        timeout: 180_000,
        stdout: 'pipe',
        stderr: 'pipe',
      },
})
