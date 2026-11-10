import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  DAY_MS,
  FULL_PERCENT,
  HOUR_MS,
  MEASURES,
  MINUTE_MS,
  UNATTRIBUTED_LABEL,
  UNATTRIBUTED_SLOT,
  foldSeries,
  formatAgo,
  formatAxisTick,
  formatCompactCount,
  formatCount,
  formatDuration,
  formatRate,
  formatRoundedDollars,
  formatWholeDollars,
  niceCeiling,
  parseCount,
  parseTokenReport,
  quotaUsageReading,
  rateSourceText,
  rateUpdateRefusalText,
  rateUpdateSummary,
  refreshStatusText,
  retailTotal,
  roundedRetailTotal,
  shareOf,
  stepsInEffectFrom,
  topRoundedBar,
  unpricedTokenTotal,
} from '../../src/usagedash/static/format.js';

function counts(total, requests = 1) {
  return {
    input: BigInt(total),
    cache_read: 0n,
    cache_write: 0n,
    output: 0n,
    reasoning: 0n,
    requests: BigInt(requests),
    total: BigInt(total),
  };
}

function countText(total, requests = 1) {
  return {
    input: String(total),
    cache_read: '0',
    cache_write: '0',
    output: '0',
    reasoning: '0',
    requests: String(requests),
    total: String(total),
  };
}

describe('what the token chart counts', () => {
  const bucket = {
    input: 100n,
    cache_read: 900n,
    cache_write: 30n,
    output: 20n,
    reasoning: 5n,
    requests: 3n,
    total: 1050n,
  };

  it('counts all tokens, the tokens that are not cache reads, or the output tokens', () => {
    const counted = Object.fromEntries(
      Object.entries(MEASURES).map(([name, measure]) => [name, measure.value(bucket)]),
    );

    assert.deepEqual(counted, { total: 1050n, no_cache_read: 150n, output: 20n });
  });

  it('names each measure for the chart title', () => {
    assert.deepEqual(
      Object.values(MEASURES).map((measure) => measure.label),
      ['all tokens', 'excluding cache reads', 'output tokens'],
    );
  });

  it('keeps a count past a safe integer exact', () => {
    const cacheReads = 2n ** 53n;
    const mostlyCacheReads = {
      ...bucket,
      input: 1n,
      cache_read: cacheReads,
      cache_write: 0n,
      output: 0n,
      total: cacheReads + 1n,
    };

    assert.equal(MEASURES.total.value(mostlyCacheReads), cacheReads + 1n);
    assert.equal(MEASURES.no_cache_read.value(mostlyCacheReads), 1n);
  });

  it('cannot be changed by the page', () => {
    assert.throws(() => {
      MEASURES.total = MEASURES.output;
    }, TypeError);
  });
});

describe('durations', () => {
  const cases = [
    [0, 'under a minute'],
    [MINUTE_MS - 1, 'under a minute'],
    [MINUTE_MS, '1m'],
    [59 * MINUTE_MS, '59m'],
    [HOUR_MS, '1h'],
    [HOUR_MS + 5 * MINUTE_MS, '1h 5m'],
    [DAY_MS - 1, '23h 59m'],
    [DAY_MS, '1d'],
    [DAY_MS + 59 * MINUTE_MS, '1d'],
    [3 * DAY_MS + 4 * HOUR_MS + 30 * MINUTE_MS, '3d 4h'],
  ];
  for (const [milliseconds, expected] of cases) {
    it(`describes ${milliseconds} ms as ${JSON.stringify(expected)}`, () => {
      assert.equal(formatDuration(milliseconds), expected);
    });
  }
});

describe('elapsed time', () => {
  const now = Date.parse('2026-10-03T14:00:00Z');

  it('says just now within the last minute', () => {
    assert.equal(formatAgo('2026-10-03T13:59:01Z', now), 'just now');
  });

  it('gives the duration once a full minute has passed', () => {
    assert.equal(formatAgo('2026-10-03T13:59:00Z', now), '1m ago');
    assert.equal(formatAgo('2026-10-02T12:00:00Z', now), '1d 2h ago');
  });

  it('never calls an instant after now recent: a clock set back puts readings in the future', () => {
    assert.equal(formatAgo('2026-10-03T14:00:01Z', now), 'under a minute from now');
    assert.equal(formatAgo('2026-10-03T14:05:00Z', now), '5m from now');
    assert.equal(formatAgo('2026-10-04T16:00:00Z', now), '1d 2h from now');
    assert.equal(formatAgo('2026-10-03T14:00:00Z', now), 'just now');
  });
});

describe('axis maximum', () => {
  const cases = [
    [-5, 1],
    [0, 1],
    [1, 1],
    [1.5, 2],
    [3, 4],
    [4.5, 5],
    [7, 8],
    [9, 10],
    [10, 10],
    [11, 20],
    [137, 200],
    [800_001, 1_000_000],
  ];
  for (const [value, expected] of cases) {
    it(`rounds ${value} up to ${expected}`, () => {
      assert.equal(niceCeiling(value), expected);
    });
  }

  it('never returns less than the value it was given', () => {
    for (const value of [0.3, 2.01, 49, 51, 999, 1001, 123_456]) {
      assert.ok(niceCeiling(value) >= value, `ceiling of ${value}`);
    }
  });

  it('rounds an exact count past a safe integer up to a tick that can hold it', () => {
    const past = 2n ** 53n + 1n;

    assert.equal(niceCeiling(past) >= past, true);
    assert.equal(niceCeiling(137n), 200n);
    assert.equal(niceCeiling(0n), 1n);
  });
});

describe('token counts from the API', () => {
  const past = 2n ** 53n + 1n;

  it('reads a decimal string as the same integer the store held', () => {
    assert.equal(parseCount(past.toString()), past);
    assert.equal(parseCount('0'), 0n);
  });

  it('refuses a count a JSON number would already have rounded', () => {
    assert.throws(() => parseCount(Number(past)), TypeError);
    assert.throws(() => parseCount('1.5'), TypeError);
    assert.throws(() => parseCount('-1'), TypeError);
    assert.throws(() => parseCount(''), TypeError);
  });

  it('reads every count in a report, including one past a safe integer', () => {
    const report = parseTokenReport({
      range: '24h',
      series: [
        {
          account_id: '7',
          label: 'Account A',
          points: [countText(past), countText(0, 0)],
          totals: countText(past),
        },
      ],
      models: [{ account_id: '7', model: 'model-x', totals: countText(past) }],
      provider_totals: [{ provider: 'claude', totals: countText(past) }],
      costs: emptyCosts(),
    });

    assert.equal(report.series[0].points[0].input, past);
    assert.equal(report.series[0].totals.total, past);
    assert.equal(report.models[0].totals.input, past);
    assert.equal(report.provider_totals[0].totals.total, past);
    assert.equal(formatCount(report.series[0].points[0].input), '9,007,199,254,740,993');
  });
});

function emptyCosts(accounts = []) {
  return {
    priced_on: '2026-10-08',
    currency: 'USD',
    unit: 'microdollar',
    note: 'Published standard rates. This is not a subscription charge.',
    accounts,
  };
}

function reportWith(costs) {
  return {
    range: '24h',
    series: [],
    models: [],
    provider_totals: [],
    costs,
  };
}

describe('retail cost from the API', () => {
  const account = (pricedMicrodollars, unpricedTokens, models) => ({
    provider: 'claude',
    account_id: '7',
    priced_microdollars: pricedMicrodollars,
    unpriced_tokens: unpricedTokens,
    models,
  });

  it('reads a cost as an exact integer, including one a JSON number cannot hold', () => {
    const huge = (10n ** 19n).toString();
    const report = parseTokenReport(
      reportWith(
        emptyCosts([
          account('00', '0', [
            { model: 'gpt-5-nano', priced: true, microdollars: '00', unpriced_tokens: '0' },
            { model: 'mystery', priced: false, microdollars: null, unpriced_tokens: '0' },
          ]),
          account(huge, '3', [{ model: 'gpt-6-astra', priced: true, microdollars: huge, unpriced_tokens: '0' }]),
        ]),
      ),
    );

    assert.equal(report.costs.accounts[0].priced_microdollars, 0n);
    assert.equal(report.costs.accounts[0].models[0].microdollars, 0n);
    assert.equal(report.costs.accounts[0].models[1].microdollars, null);
    assert.equal(report.costs.accounts[0].models[1].unpriced_tokens, 0n);
    assert.equal(report.costs.accounts[1].priced_microdollars, 10n ** 19n);
    assert.equal(retailTotal(report.costs.accounts), 10n ** 19n);
    assert.equal(unpricedTokenTotal(report.costs.accounts), 3n);
  });

  it('refuses a report with no cost, another currency, or a price that contradicts its flag', () => {
    assert.throws(() => parseTokenReport(reportWith(undefined)), /no retail cost/);
    const wrongCurrency = emptyCosts();
    wrongCurrency.currency = 'EUR';
    assert.throws(() => parseTokenReport(reportWith(wrongCurrency)), /not US dollars in microdollars/);
    const wrongUnit = emptyCosts();
    wrongUnit.unit = 'cent';
    assert.throws(() => parseTokenReport(reportWith(wrongUnit)), /not US dollars in microdollars/);
    assert.throws(
      () =>
        parseTokenReport(
          reportWith(
            emptyCosts([
              account('1', '0', [{ model: 'gpt-5-nano', priced: true, microdollars: null, unpriced_tokens: '0' }]),
            ]),
          ),
        ),
      /priced model has no cost/,
    );
    assert.throws(
      () =>
        parseTokenReport(
          reportWith(
            emptyCosts([
              account('0', '3', [{ model: 'mystery', priced: false, microdollars: '1', unpriced_tokens: '3' }]),
            ]),
          ),
        ),
      /unpriced model has a cost/,
    );
    assert.throws(
      () =>
        parseTokenReport(
          reportWith(
            emptyCosts([
              account('1', '4', [{ model: 'gpt-5-nano', priced: true, microdollars: '1', unpriced_tokens: '4' }]),
            ]),
          ),
        ),
      /priced model has unpriced tokens/,
    );
    assert.throws(
      () =>
        parseTokenReport(
          reportWith(
            emptyCosts([
              account('-1', '0', [{ model: 'gpt-5-nano', priced: true, microdollars: '1', unpriced_tokens: '0' }]),
            ]),
          ),
        ),
      /a cost is not a decimal string/,
    );
  });

  it('rounds a cost to the nearest dollar, half a dollar away from zero', () => {
    assert.equal(formatRoundedDollars(0n), '$0');
    assert.equal(formatRoundedDollars(4_872_644n), '$5');
    assert.equal(formatRoundedDollars(499_999n), '$0');
    assert.equal(formatRoundedDollars(500_000n), '$1');
    assert.equal(formatRoundedDollars(1_499_999n), '$1');
    assert.equal(formatRoundedDollars(1_500_000n), '$2');
    assert.equal(formatRoundedDollars(1_500_000_000n), '$1,500');
    assert.equal(formatRoundedDollars(10n ** 19n), '$10,000,000,000,000');
    assert.equal(formatRoundedDollars(-4_872_644n), '-$5');
    assert.equal(formatRoundedDollars(-500_000n), '-$1');
    assert.equal(formatWholeDollars(0n), '$0');
    assert.throws(() => formatRoundedDollars(1), /a cost is not an exact integer/);
    assert.throws(() => formatRoundedDollars(1.5), /a cost is not an exact integer/);
    assert.throws(() => formatWholeDollars(1), /a cost is not an exact integer/);
  });

  it('adds the account costs the server already rounded, then rounds each account', () => {
    const accounts = [
      { priced_microdollars: 1_500_000n, unpriced_tokens: 0n },
      { priced_microdollars: 500_000n, unpriced_tokens: 4n },
    ];
    const halves = [
      { priced_microdollars: 600_000n, unpriced_tokens: 0n },
      { priced_microdollars: 600_000n, unpriced_tokens: 0n },
    ];

    assert.equal(retailTotal(accounts), 2_000_000n);
    assert.equal(roundedRetailTotal(accounts), 3n);
    assert.equal(roundedRetailTotal(halves), 2n);
    assert.equal(unpricedTokenTotal(accounts), 4n);
    assert.equal(retailTotal([]), 0n);
    assert.equal(roundedRetailTotal([]), 0n);
    assert.equal(unpricedTokenTotal([]), 0n);
  });
});

describe('chart geometry for an exact count', () => {
  it('gives a count the share of the span it is of the maximum', () => {
    assert.equal(shareOf(1n, 4n, 100), 25);
    assert.equal(shareOf(2n ** 53n + 1n, 2n ** 53n + 1n, 100), 100);
    assert.equal(shareOf(0n, 4n, 100), 0);
    assert.equal(shareOf(1n, 0n, 100), 0);
    assert.equal(shareOf(1n, 4n, 0), 0);
  });

  it('labels a fractional tick without rounding the maximum', () => {
    assert.equal(formatAxisTick(10n, 1, 4), '2.5');
    assert.equal(formatAxisTick(200n, 2, 4), '100');
    const past = 2n ** 53n + 1n;
    assert.equal(formatAxisTick(past, 4, 4), formatCompactCount(past));
    assert.equal(formatAxisTick(4001n, 1, 4), formatCompactCount(1000n));
    assert.notEqual(formatCount(past), formatCompactCount(past));
  });
});

describe('what a refresh tells the page', () => {
  it('does not call a spaced attempt a successful collection', () => {
    assert.equal(refreshStatusText(429), 'Refresh attempted moments ago');
    assert.equal(refreshStatusText(202), 'Collecting…');
  });
});

describe('what a rate update tells the page', () => {
  const page = (overrides) => ({ display_name: 'Claude', error: null, changes: [], ...overrides });
  const rates = { input: '8000000', cache_read: '200000', cache_write: '10000000', output: '40000000' };

  it('says which models were newly priced and which changed, with each rate before and after', () => {
    const noCacheWrite = { input: '2000000', cache_read: '500000', cache_write: null, output: '6000000' };
    const summary = rateUpdateSummary({
      pages: [
        page({
          changes: [
            { model: 'claude-fable-5-2', previous: null, fetched: rates },
            {
              model: 'claude-opus-5',
              previous: { input: '5000000', cache_read: '500000', cache_write: '6250000', output: '25000000' },
              fetched: { input: '4000000', cache_read: '500000', cache_write: '5000000', output: '25000000' },
            },
          ],
        }),
        page({
          display_name: 'Grok',
          changes: [
            { model: 'grok-5', previous: null, fetched: noCacheWrite },
            { model: 'grok-4.7', previous: { ...noCacheWrite, cache_write: '125000' }, fetched: noCacheWrite },
          ],
        }),
        page({ display_name: 'Codex' }),
      ],
    });

    assert.equal(
      summary,
      'Rates checked. Claude has 1 newly priced and 1 changed: ' +
        'claude-fable-5-2 (input $8, cache read $0.2, cache write $10, output $40); ' +
        'claude-opus-5 (input $5 → $4, cache write $6.25 → $5). ' +
        'Grok has 1 newly priced and 1 changed: grok-5 (input $2, cache read $0.5, output $6); ' +
        'grok-4.7 (cache write $0.125 → none). Codex has no change.',
    );
  });

  it('writes a rate as exact dollars, down to one microdollar', () => {
    assert.equal(formatRate('0'), '$0');
    assert.equal(formatRate('1'), '$0.000001');
    assert.equal(formatRate('999999'), '$0.999999');
    assert.equal(formatRate('1000000'), '$1');
    assert.equal(formatRate('18750000'), '$18.75');
    assert.equal(formatRate('9007199254740993000000'), '$9007199254740993');
    assert.equal(formatRate(null), 'none');
    for (const rate of [8, '8.5', '-1', '', undefined]) {
      assert.throws(() => formatRate(rate), /a rate is not a decimal string/);
    }
  });

  it('says a vendor whose page was not read was not updated, and why', () => {
    const summary = rateUpdateSummary({
      pages: [page({ display_name: 'Grok', error: 'Firecrawl returned HTTP 402' })],
    });

    assert.equal(summary, 'Rates checked. Grok was not updated (Firecrawl returned HTTP 402).');
  });

  it('refuses a result that does not list its pages', () => {
    for (const result of [null, 'done', {}, { pages: 'none' }]) {
      assert.throws(() => rateUpdateSummary(result), /does not list its pages/);
    }
  });

  it('explains a refusal with the dashboard’s own detail, and by status when there is none', () => {
    assert.equal(rateUpdateRefusalText(409, { detail: 'Store a key.' }), 'Store a key.');
    for (const body of [null, 'text', {}, { detail: '' }, { detail: 7 }]) {
      assert.equal(rateUpdateRefusalText(503, body), 'the dashboard responded 503');
    }
  });

  it('names where a rate came from, and nothing for a model with no rate', () => {
    assert.equal(rateSourceText('fetched'), 'Fetched');
    assert.equal(rateSourceText('listed'), 'Listed');
    for (const source of [null, undefined, '', 'toString', 'other']) {
      assert.equal(rateSourceText(source), '—');
    }
  });
});

describe('bar outline', () => {
  it('rounds the top corners and keeps the base square', () => {
    assert.equal(topRoundedBar(10, 20, 24, 100), 'M10,120 L10,24 Q10,20 14,20 L30,20 Q34,20 34,24 L34,120 Z');
  });

  it('shrinks the radius for a bar shorter than the corner radius', () => {
    assert.equal(topRoundedBar(0, 0, 24, 2), 'M0,2 L0,2 Q0,0 2,0 L22,0 Q24,0 24,2 L24,2 Z');
  });

  it('shrinks the radius for a bar narrower than twice the corner radius', () => {
    assert.equal(topRoundedBar(0, 0, 4, 50), 'M0,50 L0,2 Q0,0 2,0 L2,0 Q4,0 4,2 L4,50 Z');
  });
});

describe('unattributed usage in the chart', () => {
  const buckets = [{ key: 'day-1' }, { key: 'day-2' }];

  it('returns the attributed series unchanged when nothing is unattributed', () => {
    const series = [{ attributed: true, label: 'Account A', points: [counts(1), counts(2)], totals: counts(3, 2) }];

    assert.deepEqual(foldSeries({ buckets, series }), series);
  });

  it('merges unattributed series from every provider into one, listed last', () => {
    const attributed = { attributed: true, label: 'Account A', points: [counts(1), counts(2)], totals: counts(3, 2) };
    const report = {
      buckets,
      series: [
        { attributed: false, points: [counts(10), counts(0, 0)], totals: counts(10) },
        attributed,
        { attributed: false, points: [counts(5), counts(7)], totals: counts(12, 2) },
      ],
    };

    const folded = foldSeries(report);

    assert.equal(folded.length, 2);
    assert.equal(folded[0], attributed);
    assert.equal(folded[1].label, UNATTRIBUTED_LABEL);
    assert.equal(folded[1].color_slot, UNATTRIBUTED_SLOT);
    assert.deepEqual(
      folded[1].points.map((point) => point.total),
      [15n, 7n],
    );
    assert.deepEqual(folded[1].totals, counts(22, 3));
  });
});

describe('the steps of a quota history to draw from the start of a range', () => {
  const at = (hour) => `2026-10-03T${String(hour).padStart(2, '0')}:00:00+00:00`;
  const start = new Date(at(12)).getTime();
  const steps = (...hours) => hours.map((hour) => ({ from: at(hour), used_percent: hour }));
  const levels = (found) => found.map((step) => step.used_percent);

  it('keeps the step in effect when the range starts, and every later one', () => {
    assert.deepEqual(levels(stepsInEffectFrom(steps(8, 10, 13, 14), start)), [10, 13, 14]);
  });

  it('keeps a step that begins exactly at the start as the first one', () => {
    assert.deepEqual(levels(stepsInEffectFrom(steps(10, 12, 14), start)), [12, 14]);
  });

  it('keeps every step when all of them begin after the start', () => {
    assert.deepEqual(levels(stepsInEffectFrom(steps(13, 14), start)), [13, 14]);
  });

  it('keeps only the last step when all of them begin before the start', () => {
    assert.deepEqual(levels(stepsInEffectFrom(steps(8, 9, 10), start)), [10]);
  });

  it('finds nothing in an empty history', () => {
    assert.deepEqual(stepsInEffectFrom([], start), []);
  });
});

describe('quota window percent', () => {
  it('shows a whole number when rounding does not land on a full window', () => {
    assert.deepEqual(quotaUsageReading(0), { label: '0', meterNow: '0', meterText: '0% used' });
    assert.deepEqual(quotaUsageReading(1), { label: '1', meterNow: '1', meterText: '1% used' });
    assert.deepEqual(quotaUsageReading(99.4), { label: '99', meterNow: '99.4', meterText: '99.4% used' });
    assert.deepEqual(quotaUsageReading(100.5), { label: '101', meterNow: '100', meterText: '100.5% used' });
    assert.deepEqual(quotaUsageReading(340.5), { label: '341', meterNow: '100', meterText: '340.5% used' });
  });

  it('says which side of full a reading is on when rounding would hide it', () => {
    assert.equal(quotaUsageReading(99.5).label, 'Under 100');
    assert.equal(quotaUsageReading(99.9).label, 'Under 100');
    assert.equal(quotaUsageReading(100.4).label, 'Over 100');
    assert.deepEqual(quotaUsageReading(FULL_PERCENT), {
      label: '100',
      meterNow: '100',
      meterText: '100% used',
    });
  });

  it('keeps the meter inside 0 to 100 and announces the server percent', () => {
    assert.equal(quotaUsageReading(99.9).meterNow, '99.9');
    assert.equal(quotaUsageReading(99.9).meterText, '99.9% used');
    assert.equal(quotaUsageReading(100.4).meterNow, '100');
    assert.equal(quotaUsageReading(100.4).meterText, '100.4% used');
    assert.equal(quotaUsageReading(340.5).meterNow, '100');
    assert.equal(quotaUsageReading(-1).meterNow, '0');
  });

  it('refuses a percent that is not a finite number', () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, '100', null, undefined]) {
      assert.throws(() => quotaUsageReading(value), TypeError);
    }
  });
});
