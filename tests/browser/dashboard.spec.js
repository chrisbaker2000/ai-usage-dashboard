import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

import { expect, test } from '@playwright/test';

import { demoState, mountDemo } from './demo.js';

test('social preview screenshot fits the shared-link image dimensions', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 640 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.route('**/social-preview', (route) =>
    route.fulfill({ path: 'tests/browser/social-preview.html', contentType: 'text/html' }),
  );
  await page.goto('/social-preview');
  await expect(page.getByRole('heading', { name: 'Your AI, at a glance.' })).toBeVisible();
  await expect(page.locator('.share-provider')).toHaveCount(3);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(1280);
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBe(640);
  const footer = await page.locator('.share-footer').boundingBox();
  expect(footer.y + footer.height).toBeLessThanOrEqual(640);
  const rendered = await page.screenshot();
  const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
  const sources = {
    source_sha256: sha256(await readFile('tests/browser/social-preview.html')),
    stylesheet_sha256: sha256(await readFile('src/usagedash/static/styles.css')),
  };
  if (process.env.USAGEDASH_SCREENSHOT === '1') {
    await mkdir('docs/images', { recursive: true });
    await writeFile('docs/images/social-preview.png', rendered);
    await writeFile(
      'docs/images/social-preview.json',
      JSON.stringify({ ...sources, image_sha256: sha256(rendered) }, null, 2) + '\n',
    );
  }
  const image = await readFile('docs/images/social-preview.png');
  const manifest = JSON.parse(await readFile('docs/images/social-preview.json', 'utf8'));
  expect(manifest, 'Run npm run screenshot after changing the card or stylesheet').toEqual({
    ...sources,
    image_sha256: sha256(image),
  });
  for (const png of [rendered, image]) {
    expect(png.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
    expect(png.readUInt32BE(16)).toBe(1280);
    expect(png.readUInt32BE(20)).toBe(640);
    expect(png.length).toBeLessThan(1_000_000);
  }
});

function contrastRatio(foreground, background) {
  const luminance = (hex) => {
    const [red, green, blue] = hex
      .slice(1)
      .match(/../g)
      .map((channel) => {
        const value = Number.parseInt(channel, 16) / 255;
        return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
      });
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  };
  const values = [luminance(foreground), luminance(background)];
  return (Math.max(...values) + 0.05) / (Math.min(...values) + 0.05);
}

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
      const palette = await page.locator('html').evaluate((node) => {
        const styles = getComputedStyle(node);
        const read = (name) => styles.getPropertyValue(name).trim();
        return {
          text: ['--text-primary', '--text-secondary', '--text-muted', '--brand'].map(read),
          surfaces: ['--page', '--surface', '--surface-raised'].map(read),
        };
      });
      for (const text of palette.text) {
        for (const surface of palette.surfaces) {
          expect(contrastRatio(text, surface), `${text} on ${surface}`).toBeGreaterThanOrEqual(4.5);
        }
      }
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
