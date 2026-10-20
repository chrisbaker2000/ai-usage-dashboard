import { mkdir } from 'node:fs/promises';

import { expect, test } from '@playwright/test';

import { demoState, mountDemo } from './demo.js';

test('demo forecasts stay before their resets and advice uses every blocking window', () => {
  for (const provider of demoState().providers) {
    const active = provider.accounts.find((account) => account.is_active);
    const used = Math.max(...active.windows.filter((window) => window.blocking).map((window) => window.used_percent));
    expect(provider.advice.summary).toContain(`${100 - used}% headroom`);
    for (const account of provider.accounts) {
      for (const window of account.windows) {
        if (window.pace_exhausts_at !== null) {
          expect(Date.parse(window.pace_exhausts_at)).toBeGreaterThan(Date.parse(window.as_of));
          expect(Date.parse(window.pace_exhausts_at)).toBeLessThan(Date.parse(window.resets_at));
        }
      }
    }
  }
});

test('README screenshot renders the real dashboard with demo accounts', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await mountDemo(page);
  await expect(page.locator('.card')).toHaveCount(6);
  await expect(page.locator('#chart svg')).toBeVisible();
  await expect(page.locator('#load-error')).toBeHidden();
  await expect(page.locator('#providers')).toContainText('personal@example.test');
  if (process.env.USAGEDASH_SCREENSHOT === '1') {
    await mkdir('docs/images', { recursive: true });
    const height =
      Math.ceil(await page.locator('#providers').evaluate((node) => node.getBoundingClientRect().bottom)) + 24;
    await page.setViewportSize({ width: 1440, height });
    for (const theme of ['dark', 'light']) {
      await page.emulateMedia({ colorScheme: theme });
      await page.screenshot({ path: `docs/images/dashboard-${theme}.png` });
    }
  }
  expect(errors).toEqual([]);
});

for (const width of [320, 390, 768, 1440]) {
  for (const colorScheme of ['dark', 'light']) {
    test(`dashboard fits ${width}px in ${colorScheme} mode`, async ({ page }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.emulateMedia({ colorScheme });
      await mountDemo(page);
      await expect(page.locator('.card')).toHaveCount(6);
      await expect(page.locator('#chart svg')).toBeVisible();
      await page.getByText('By model', { exact: true }).click();
      await expect(page.locator('#model-table tbody tr')).toHaveCount(6);
      await expect(page.locator('#model-table')).toContainText('claude-opus-4-6');
      await expect(page.locator('#cost-table tbody tr')).toHaveCount(7);
      await expect(page.locator('#cost-table tbody tr').first()).toContainText('$65');
      await page.getByText('Cost by model', { exact: true }).click();
      await expect(page.locator('#cost-model-table tbody tr')).toHaveCount(6);
      await expect(page.locator('#cost-model-table tbody tr').last()).toContainText('No rate');
      await expect(page.locator('html')).toHaveCSS('color-scheme', colorScheme);
      const background = await page.locator('body').evaluate((node) => getComputedStyle(node).backgroundColor);
      const channels = background.match(/\d+/g).map(Number);
      const brightness = channels.reduce((sum, channel) => sum + channel, 0) / channels.length;
      if (colorScheme === 'dark') expect(brightness).toBeLessThan(128);
      else expect(brightness).toBeGreaterThan(128);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      for (const card of await page.locator('.card').all()) {
        const bounds = await card.boundingBox();
        expect(bounds.x).toBeGreaterThanOrEqual(0);
        expect(bounds.x + bounds.width).toBeLessThanOrEqual(width);
      }
    });
  }
}

test('keyboard navigation reaches the dashboard and account rename saves', async ({ page }) => {
  await mountDemo(page);
  await expect(page.locator('.card')).toHaveCount(6);
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to dashboard' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();
  const account = page.locator('.card').first();
  await account.getByRole('button', { name: 'Rename' }).click();
  await account.getByRole('textbox').fill('Research');
  await account.getByRole('button', { name: 'Save' }).click();
  await expect(account.getByRole('heading', { name: 'Research' })).toBeVisible();
  await expect(account.getByRole('button', { name: 'Rename' })).toBeFocused();
  await expect(page.locator('#load-error')).toBeHidden();
});

test('token filters and refresh use the signed demo API', async ({ page }) => {
  await mountDemo(page);
  await page.getByRole('button', { name: 'Last 24 hours' }).click();
  await expect(page.locator('#chart-title')).toContainText('last 24 hours');
  await page.locator('#measure').selectOption('output');
  await expect(page.locator('#chart-title')).toContainText('output tokens');
  await page.getByRole('button', { name: 'Refresh now' }).click();
  await expect(page.locator('#updated')).toHaveText('Collecting…');
  await expect(page.locator('#load-error')).toBeHidden();
});
