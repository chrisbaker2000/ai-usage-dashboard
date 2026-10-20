import { mkdir } from 'node:fs/promises';

import { expect, test } from '@playwright/test';

import { mountDemo } from './demo.js';

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
