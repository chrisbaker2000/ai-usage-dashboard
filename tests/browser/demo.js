import { createDecipheriv, createHmac, hkdfSync, randomBytes } from 'node:crypto';

import { SIGNED_HEADER, TOKEN, answerSignatureOf, sealedAnswerOf, sha256 } from '../js/fakes/dashboard.js';

const NOW = Date.parse('2026-10-20T15:00:00Z');
const HOUR = 3_600_000;
const DAY = 24 * HOUR;
const iso = (offset = 0) => new Date(NOW + offset).toISOString();

function quota(key, label, percent, resetHours, hours = 168) {
  return {
    key,
    label,
    blocking: true,
    used_percent: percent,
    effective_percent: percent,
    state: percent >= 100 ? 'exhausted' : percent >= 90 ? 'critical' : percent >= 75 ? 'warning' : 'ok',
    resets_at: iso(resetHours * HOUR),
    reset_passed: false,
    as_of: iso(),
    freshness: 'live',
    window_seconds: hours * 3600,
    pace_exhausts_at: percent > 0 && percent < 75 && resetHours > 32 ? iso(32 * HOUR) : null,
  };
}

function account(id, provider, name, plan, slot, windows, active = true) {
  return {
    id,
    label: `${provider} · ${name}`,
    email: `${name.toLowerCase()}@example.test`,
    plan,
    color_slot: slot,
    is_active: active,
    is_live: true,
    windows,
    facts: [],
  };
}

/** Synthetic accounts only: used by browser tests and the public README screenshots. */
export function demoState() {
  const providers = [
    {
      provider: 'claude',
      display_name: 'Claude',
      accounts: [
        account('1', 'Claude', 'Work', 'Max 5x', 2, [
          quota('session', '5-hour session', 12, 3, 5),
          quota('weekly', 'Weekly · all models', 27, 140),
          quota('model', 'Weekly · Fable', 18, 140),
        ]),
        account(
          '2',
          'Claude',
          'Personal',
          'Max 20x',
          1,
          [
            quota('session', '5-hour session', 34, 2, 5),
            quota('weekly', 'Weekly · all models', 100, 85),
            quota('model', 'Weekly · Fable', 56, 85),
          ],
          false,
        ),
      ],
    },
    {
      provider: 'codex',
      display_name: 'Codex',
      accounts: [
        account('3', 'Codex', 'Work', 'Pro', 3, [quota('weekly', 'Weekly', 73, 130)]),
        account('4', 'Codex', 'Personal', 'Pro', 7, [quota('weekly', 'Weekly', 42, 124)], false),
      ],
    },
    {
      provider: 'grok',
      display_name: 'Grok',
      accounts: [
        account('5', 'Grok', 'Work', 'SuperGrok Heavy', 4, [quota('weekly', 'Weekly credits', 45, 133)]),
        account('6', 'Grok', 'Personal', 'SuperGrok Heavy', 5, [quota('weekly', 'Weekly credits', 88, 38)], false),
      ],
    },
  ];
  for (const provider of providers) {
    const active = provider.accounts.find((item) => item.is_active);
    const headroom =
      100 - Math.max(...active.windows.filter((window) => window.blocking).map((window) => window.used_percent));
    provider.advice = {
      action: 'stay',
      summary: `${active.label} has ${headroom}% headroom.`,
      as_of: null,
      next_available_at: null,
    };
  }
  return {
    providers,
    expected_accounts_per_provider: 2,
    collector: { running: false, last_finished_at: iso() },
    history: providers.flatMap((provider) =>
      provider.accounts.flatMap((item) =>
        item.windows.map((window) => ({
          account_id: item.id,
          window_key: window.key,
          steps: Array.from({ length: 15 }, (_, index) => ({
            from: iso((-7 + index / 2) * DAY),
            to: iso(Math.min(0, -7 + (index + 1) / 2) * DAY),
            used_percent: (window.used_percent * index) / 14,
          })),
        })),
      ),
    ),
    sources: providers.flatMap((provider) =>
      provider.accounts.map((account) => ({
        source_id: `${provider.provider}:${account.is_active ? 'working login' : 'standing login'}`,
        kind: account.is_active ? 'direct' : 'cli',
        severity: 'ok',
        message: 'Connected and collecting',
        last_attempt_at: iso(),
        last_success_at: iso(),
      })),
    ),
  };
}

function counts(value) {
  return {
    input: String(value),
    cache_read: String(value * 4),
    cache_write: '0',
    output: String(value),
    reasoning: '0',
    requests: String(Math.round(value / 1000)),
    total: String(value * 6),
  };
}

function totalCounts(rows) {
  return Object.fromEntries(
    Object.keys(counts(0)).map((key) => [key, String(rows.reduce((sum, row) => sum + BigInt(row[key]), 0n))]),
  );
}

function tokenReport(state, range) {
  const bucketCount = range === '24h' ? 24 : range === '30d' ? 30 : 7;
  const interval = range === '24h' ? HOUR : DAY;
  const anchor = range === '24h' ? NOW : Date.parse('2026-10-20T04:00:00Z');
  const buckets = Array.from({ length: bucketCount }, (_, index) => {
    const start = new Date(anchor + (index - bucketCount + 1) * interval).toISOString();
    return { key: range === '24h' ? start : start.slice(0, 10), start };
  });
  const series = state.providers.flatMap((provider) =>
    provider.accounts.map((item, index) => {
      const points = buckets.map((_, day) =>
        counts((12 + ((day * 7 + Number(item.id) * 11) % 35)) * (index + 1) * 10000),
      );
      return {
        provider: provider.provider,
        account_id: item.id,
        label: item.label,
        color_slot: item.color_slot,
        attributed: true,
        points,
        totals: totalCounts(points),
      };
    }),
  );
  const modelNames = { claude: 'claude-opus-4-6', codex: 'gpt-6-sol', grok: 'grok-4.7' };
  const models = series.map((item) => ({
    provider: item.provider,
    account_id: item.account_id,
    model: item.account_id === '6' ? 'unlisted-demo-model' : modelNames[item.provider],
    totals: item.totals,
  }));
  // Why: demo costs match the short-prompt rates recorded in pricing.py on 2026-10-08.
  const rates = {
    claude: [5_000_000n, 500_000n, 25_000_000n],
    codex: [2_000_000n, 200_000n, 10_000_000n],
    grok: [2_000_000n, 500_000n, 6_000_000n],
  };
  const costs = models.map((row) => {
    const rate = rates[row.provider];
    const price =
      (BigInt(row.totals.input) * rate[0] +
        BigInt(row.totals.cache_read) * rate[1] +
        BigInt(row.totals.output) * rate[2]) /
      1_000_000n;
    const unpriced = row.model === 'unlisted-demo-model';
    return {
      provider: row.provider,
      account_id: row.account_id,
      priced_microdollars: unpriced ? '0' : String(price),
      unpriced_tokens: unpriced ? row.totals.total : '0',
      models: [
        {
          model: row.model,
          priced: !unpriced,
          microdollars: unpriced ? null : String(price),
          unpriced_tokens: unpriced ? row.totals.total : '0',
        },
      ],
    };
  });
  return {
    range,
    bucket: range === '24h' ? 'hour' : 'day',
    timezone: 'America/New_York',
    buckets,
    series,
    models,
    provider_totals: state.providers.map((provider) => ({
      provider: provider.provider,
      display_name: provider.display_name,
      totals: totalCounts(series.filter((item) => item.provider === provider.provider).map((item) => item.totals)),
    })),
    costs: {
      priced_on: '2026-10-08',
      currency: 'USD',
      unit: 'microdollar',
      note: 'Estimated at published short-prompt API rates. This is not your subscription charge.',
      accounts: costs,
    },
  };
}

function openRequest(body, method, target, challenge, nonce) {
  const key = hkdfSync(
    'sha256',
    Buffer.from(TOKEN),
    Buffer.from('usagedash'),
    Buffer.from('usagedash/sealed-body/v1'),
    32,
  );
  const decipher = createDecipheriv('aes-256-gcm', key, body.subarray(0, 12));
  decipher.setAAD(Buffer.from(`${method}\n${target}\n${challenge}\n${nonce}`));
  decipher.setAuthTag(body.subarray(-16));
  return JSON.parse(Buffer.concat([decipher.update(body.subarray(12, -16)), decipher.final()]).toString());
}

/** Mounts a signed, sealed demo API without reading a real credential, session file, or database. */
export async function mountDemo(page) {
  const state = demoState();
  const challenges = new Set();
  await page.clock.install({ time: NOW });
  await page.addInitScript((token) => {
    window.__USAGEDASH_TOKEN = token;
  }, TOKEN);
  // The Python static server has no /static prefix; production retains its own stylesheet URL.
  await page.route('**/static/styles.css', (route) =>
    route.fulfill({ path: 'src/usagedash/static/styles.css', contentType: 'text/css' }),
  );
  await page.route('**/static/app.js', (route) =>
    route.fulfill({ path: 'src/usagedash/static/app.js', contentType: 'text/javascript' }),
  );
  for (const name of ['access', 'failures', 'format', 'latest', 'rename']) {
    await page.route(`**/static/${name}.js`, (route) =>
      route.fulfill({ path: `src/usagedash/static/${name}.js`, contentType: 'text/javascript' }),
    );
  }
  await page.route('**/challenge', async (route) => {
    const challenge = randomBytes(32).toString('base64url');
    challenges.add(challenge);
    await route.fulfill({ json: { challenge } });
  });
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const target = url.pathname + url.search;
    const parts = SIGNED_HEADER.exec(request.headers().authorization ?? '');
    if (parts === null || !challenges.delete(parts[1]))
      throw new Error('Demo request must use a fresh signed challenge');
    const [, challenge, nonce, digest] = parts;
    const body = request.postDataBuffer() ?? Buffer.alloc(0);
    const expected = createHmac('sha256', TOKEN)
      .update(`${request.method()}\n${target}\n${sha256(body)}\n${challenge}\n${nonce}`)
      .digest('hex');
    if (digest !== expected) throw new Error('Demo request signature does not match');
    let answer;
    if (url.pathname === '/api/state') answer = state;
    else if (url.pathname === '/api/tokens') answer = tokenReport(state, url.searchParams.get('range'));
    else if (url.pathname === '/api/refresh') answer = { scheduled: true };
    else if (request.method() === 'PATCH' && /^\/api\/accounts\/\d+$/.test(url.pathname)) {
      const item = state.providers
        .flatMap((provider) => provider.accounts)
        .find((item) => item.id === url.pathname.split('/').at(-1));
      if (item === undefined) throw new Error('Demo account is missing');
      item.label = openRequest(body, request.method(), target, challenge, nonce).label;
      answer = { label: item.label };
    } else throw new Error(`Unexpected demo endpoint: ${url.pathname}`);
    const sealed = sealedAnswerOf(TOKEN, challenge, nonce, 200, JSON.stringify(answer));
    await route.fulfill({
      status: 200,
      body: Buffer.from(sealed),
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Usagedash-Response': answerSignatureOf(TOKEN, challenge, nonce, 200, sealed),
      },
    });
  });
  await page.goto('/');
}
