import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  workers: 2,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:8791',
    viewport: { width: 1440, height: 1100 },
    colorScheme: 'dark',
    timezoneId: 'America/New_York',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command: 'python3 -m http.server 8791 --bind 127.0.0.1 --directory src/usagedash/static',
    url: 'http://127.0.0.1:8791',
    reuseExistingServer: false,
    timeout: 15_000,
  },
});
