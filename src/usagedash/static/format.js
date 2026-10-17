/** Pure formatting and chart arithmetic. Nothing here touches the document, so it runs under `node --test`. */

export const MINUTE_MS = 60_000;
export const HOUR_MS = 60 * MINUTE_MS;
export const DAY_MS = 24 * HOUR_MS;
export const WEEK_MS = 7 * DAY_MS;
export const FULL_PERCENT = 100;
export const UNATTRIBUTED_SLOT = 0;
export const UNATTRIBUTED_LABEL = 'Unattributed (before tracking)';

/**
 * The ways the token chart can count a bucket, by the name the page's control gives each one.
 *
 * Counts are exact integers. "Excluding cache reads" adds up what is not a cache read. It does not
 * subtract cache reads from the total: a total can be larger than any one of its parts is wide, and
 * the subtraction would have to invent a width it was not given.
 */
export const MEASURES = Object.freeze({
  total: { label: 'all tokens', value: (counts) => counts.total },
  no_cache_read: {
    label: 'excluding cache reads',
    value: (counts) => counts.input + counts.cache_write + counts.output,
  },
  output: { label: 'output tokens', value: (counts) => counts.output },
});

const COUNT_FIELDS = ['input', 'cache_read', 'cache_write', 'output', 'reasoning', 'requests', 'total'];
const MICRODOLLARS_PER_DOLLAR = 1_000_000n;
const HALF_DOLLAR = MICRODOLLARS_PER_DOLLAR / 2n;
const EXACT_INTEGER = new Intl.NumberFormat('en');
const COMPACT_INTEGER = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const HTTP_TOO_MANY_REQUESTS = 429;

/**
 * Reads one non-negative decimal integer from the API.
 *
 * The API sends decimal text. A JSON number is exact only up to 2**53, and a stored count or a cost
 * can be larger, so a number would already be wrong by the time it arrived.
 *
 * @param {string} value - Decimal text of a whole number.
 * @param {string} label - What the value is, used when it is refused.
 * @returns {bigint}
 */
function parseDecimal(value, label) {
  if (typeof value !== 'string' || !/^[0-9]+$/.test(value)) {
    throw new TypeError(`a ${label} is not a decimal string`);
  }
  return BigInt(value);
}

/**
 * Reads one token count from the API.
 * @param {string} value - Decimal text of a whole count.
 * @returns {bigint}
 */
export function parseCount(value) {
  return parseDecimal(value, 'token count');
}

/**
 * Reads every field of one counts object as an exact integer.
 * @param {Record<string, string>} counts
 * @returns {Record<string, bigint>}
 */
export function parseCounts(counts) {
  return Object.fromEntries(COUNT_FIELDS.map((field) => [field, parseCount(counts[field])]));
}

/**
 * Reads one model's retail cost. A priced model has a cost and no unpriced tokens.
 * An unpriced model has no cost.
 * @param {object} model
 * @returns {object}
 * @throws {TypeError} When the priced flag, the cost, and the unpriced-token count disagree.
 */
function parseModelCost(model) {
  if (typeof model.priced !== 'boolean') throw new TypeError('a model cost is not priced or unpriced');
  if (model.priced && model.microdollars === null) throw new TypeError('a priced model has no cost');
  if (!model.priced && model.microdollars !== null) throw new TypeError('an unpriced model has a cost');
  const unpricedTokens = parseCount(model.unpriced_tokens);
  if (model.priced && unpricedTokens !== 0n) throw new TypeError('a priced model has unpriced tokens');
  return {
    ...model,
    microdollars: model.microdollars === null ? null : parseDecimal(model.microdollars, 'cost'),
    unpriced_tokens: unpricedTokens,
  };
}

/**
 * Reads the retail-cost block. A report without one, or in any other currency, is refused.
 * @param {object} costs
 * @returns {object}
 */
function parseCosts(costs) {
  if (costs === undefined || costs === null || typeof costs !== 'object') {
    throw new TypeError('a token report has no retail cost');
  }
  if (costs.currency !== 'USD' || costs.unit !== 'microdollar') {
    throw new TypeError('a retail cost is not US dollars in microdollars');
  }
  if (typeof costs.note !== 'string' || typeof costs.priced_on !== 'string' || !Array.isArray(costs.accounts)) {
    throw new TypeError('a retail cost has no accounts');
  }
  return {
    ...costs,
    accounts: costs.accounts.map((account) => ({
      ...account,
      priced_microdollars: parseDecimal(account.priced_microdollars, 'cost'),
      unpriced_tokens: parseCount(account.unpriced_tokens),
      models: account.models.map(parseModelCost),
    })),
  };
}

/**
 * Returns the token report with every count and every cost read as an exact integer.
 * @param {object} report - A token report as the API sent it.
 * @returns {object}
 * @throws {TypeError} When a count, a cost, or the cost block is not the shape the API sends.
 */
export function parseTokenReport(report) {
  const counted = (row) => ({ ...row, totals: parseCounts(row.totals) });
  return {
    ...report,
    series: report.series.map((series) => ({
      ...series,
      points: series.points.map(parseCounts),
      totals: parseCounts(series.totals),
    })),
    models: report.models.map(counted),
    provider_totals: report.provider_totals.map(counted),
    costs: parseCosts(report.costs),
  };
}

/**
 * Groups an exact count the way a table cell does.
 * @param {bigint | number} value
 * @returns {string}
 */
export function formatCount(value) {
  return EXACT_INTEGER.format(value);
}

/**
 * Nearest whole dollar for a microdollar amount. A half dollar rounds away from zero.
 * @param {bigint} microdollars - One US dollar is 1,000,000.
 * @returns {bigint}
 * @throws {TypeError} When `microdollars` is not an exact integer.
 */
export function nearestDollar(microdollars) {
  if (typeof microdollars !== 'bigint') throw new TypeError('a cost is not an exact integer');
  const negative = microdollars < 0n;
  const absolute = negative ? -microdollars : microdollars;
  const dollars = (absolute + HALF_DOLLAR) / MICRODOLLARS_PER_DOLLAR;
  return negative ? -dollars : dollars;
}

/**
 * Formats a whole-dollar amount.
 * @param {bigint} dollars
 * @returns {string} For example `$5` or `$1,500`.
 * @throws {TypeError} When `dollars` is not an exact integer.
 */
export function formatWholeDollars(dollars) {
  if (typeof dollars !== 'bigint') throw new TypeError('a cost is not an exact integer');
  const negative = dollars < 0n;
  const absolute = negative ? -dollars : dollars;
  return `${negative ? '-' : ''}$${EXACT_INTEGER.format(absolute)}`;
}

/**
 * Formats a microdollar cost as the nearest whole dollar.
 * @param {bigint} microdollars - One US dollar is 1,000,000.
 * @returns {string}
 * @throws {TypeError} When `microdollars` is not an exact integer.
 */
export function formatRoundedDollars(microdollars) {
  return formatWholeDollars(nearestDollar(microdollars));
}

/**
 * The retail cost of every account, in microdollars, already rounded per model on the server.
 * @param {Array<{priced_microdollars: bigint}>} accounts
 * @returns {bigint}
 */
export function retailTotal(accounts) {
  return accounts.reduce((total, account) => total + account.priced_microdollars, 0n);
}

/**
 * Whole dollars to show for every account.
 *
 * Each account is rounded on its own and those dollars are added, so the total row matches the
 * rows above it. Rounding the exact combined total can land on a different dollar.
 * @param {Array<{priced_microdollars: bigint}>} accounts
 * @returns {bigint}
 */
export function roundedRetailTotal(accounts) {
  return accounts.reduce((total, account) => total + nearestDollar(account.priced_microdollars), 0n);
}

/**
 * Tokens the dollar total left out, across every account.
 * @param {Array<{unpriced_tokens: bigint}>} accounts
 * @returns {bigint}
 */
export function unpricedTokenTotal(accounts) {
  return accounts.reduce((total, account) => total + account.unpriced_tokens, 0n);
}

/**
 * Groups an exact count the way a tile or an axis does when the full number does not fit.
 * @param {bigint | number} value
 * @returns {string}
 */
export function formatCompactCount(value) {
  return COMPACT_INTEGER.format(value);
}

/**
 * Labels one axis tick of an exact maximum.
 * @param {bigint} maximum - The axis maximum.
 * @param {number} tick - Which tick, from 0 through `tickCount`.
 * @param {number} tickCount - How many spaces there are between the bottom and the top.
 * @returns {string}
 */
export function formatAxisTick(maximum, tick, tickCount) {
  const scaled = (maximum * 10n * BigInt(tick)) / BigInt(tickCount);
  const whole = scaled / 10n;
  const tenth = scaled % 10n;
  if (tenth === 0n || whole >= 1000n) return formatCompactCount(whole);
  return `${whole}.${tenth}`;
}

/**
 * How far across `span` the `value` reaches when `maximum` is the whole span.
 * @param {bigint} value
 * @param {bigint} maximum
 * @param {number} span - The span in pixels.
 * @returns {number} Pixels. A count past 2**53 still takes its share of the span.
 */
export function shareOf(value, maximum, span) {
  if (maximum <= 0n || value <= 0n || span <= 0) return 0;
  const thousandths = (value * BigInt(Math.round(span * 1000))) / maximum;
  return Number(thousandths) / 1000;
}

/**
 * What the page says after asking for a collection.
 *
 * A 429 means an attempt was made recently, including one that failed. It does not mean a collection
 * succeeded.
 * @param {number} status - The refresh response status.
 * @returns {string}
 */
export function refreshStatusText(status) {
  return status === HTTP_TOO_MANY_REQUESTS ? 'Refresh attempted moments ago' : 'Collecting…';
}

const NICE_AXIS_STEPS = [1, 2, 4, 5, 8, 10];
const BAR_CORNER_RADIUS = 4;

/**
 * Describes a span of time in at most two units.
 * @param {number} milliseconds - A non-negative span.
 * @returns {string} For example `under a minute`, `45m`, `2h 5m`, or `3d 4h`.
 */
export function formatDuration(milliseconds) {
  if (milliseconds < MINUTE_MS) return 'under a minute';
  const days = Math.floor(milliseconds / DAY_MS);
  const hours = Math.floor((milliseconds % DAY_MS) / HOUR_MS);
  const minutes = Math.floor((milliseconds % HOUR_MS) / MINUTE_MS);
  if (days > 0) return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
  if (hours > 0) return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  return `${minutes}m`;
}

/**
 * Describes how far from now an instant is.
 * @param {string} iso - An ISO-8601 instant.
 * @param {number} now - The current time in epoch milliseconds.
 * @returns {string} `just now` within the last minute, otherwise a duration followed by `ago`. An
 *   instant after `now`, which a clock set back produces, is a duration followed by `from now`, never
 *   `just now`: the server does not call such a reading live, and the page must not call it recent.
 */
export function formatAgo(iso, now) {
  const elapsed = now - new Date(iso).getTime();
  if (elapsed < 0) return `${formatDuration(-elapsed)} from now`;
  return elapsed < MINUTE_MS ? 'just now' : `${formatDuration(elapsed)} ago`;
}

/**
 * The percent a quota window shows, and the meter values an assistive client can accept.
 *
 * Rounding 99.5 and 100.4 both land on 100, which would draw an open window as full and an overrun as
 * exactly full; those two say which side of the limit they are on. The meter value stays inside 0..100
 * while its text keeps the server's percent.
 *
 * @param {number} effectivePercent - The window's effective percent. Not negative from the API.
 * @returns {{label: string, meterNow: string, meterText: string}}
 * @throws {TypeError} When `effectivePercent` is not a finite number.
 */
export function quotaUsageReading(effectivePercent) {
  if (typeof effectivePercent !== 'number' || !Number.isFinite(effectivePercent)) {
    throw new TypeError('a quota percent is not a finite number');
  }
  const rounded = Math.round(effectivePercent);
  let label = String(rounded);
  if (rounded === FULL_PERCENT && effectivePercent !== FULL_PERCENT) {
    label = effectivePercent < FULL_PERCENT ? 'Under 100' : 'Over 100';
  }
  const meterNow = Math.min(FULL_PERCENT, Math.max(0, effectivePercent));
  return { label, meterNow: String(meterNow), meterText: `${effectivePercent}% used` };
}

/**
 * Rounds an axis maximum up to a value that divides into even ticks.
 * @param {number} value - The largest value to be plotted.
 * @returns {number} A value at or above `value`; 1 when there is nothing to plot.
 */
function niceCeilingExact(value) {
  if (value <= 0n) return 1n;
  const magnitude = 10n ** BigInt(value.toString().length - 1);
  for (const step of NICE_AXIS_STEPS) {
    const ceiling = BigInt(step) * magnitude;
    if (ceiling >= value) return ceiling;
  }
  return 10n * magnitude;
}

export function niceCeiling(value) {
  if (typeof value === 'bigint') return niceCeilingExact(value);
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const step = NICE_AXIS_STEPS.find((candidate) => normalized <= candidate);
  return step * magnitude;
}

/**
 * Builds the path of a bar whose top corners are rounded and whose base is square.
 * @returns {string} SVG path data. The radius shrinks so it never exceeds the bar's height or half its width.
 */
export function topRoundedBar(x, y, width, height) {
  const radius = Math.min(BAR_CORNER_RADIUS, height, width / 2);
  return [
    `M${x},${y + height}`,
    `L${x},${y + radius}`,
    `Q${x},${y} ${x + radius},${y}`,
    `L${x + width - radius},${y}`,
    `Q${x + width},${y} ${x + width},${y + radius}`,
    `L${x + width},${y + height}`,
    'Z',
  ].join(' ');
}

/**
 * Picks the steps of a quota history in effect at or after `start`: every step that begins after it and
 * the last one that begins at or before it, the level the range starts at.
 *
 * @param {Array<{from: string}>} steps - One window's steps, oldest first.
 * @param {number} start - The start of the range, in milliseconds since the epoch.
 * @returns {Array<{from: string}>} The steps to draw, oldest first.
 */
export function stepsInEffectFrom(steps, start) {
  const firstAfter = steps.findIndex((step) => new Date(step.from).getTime() > start);
  const inEffectAtStart = firstAfter === -1 ? steps.length - 1 : firstAfter - 1;
  return steps.slice(Math.max(inEffectAtStart, 0));
}

/**
 * Merges every unattributed series of a token report into one, listed after the attributed ones.
 * @param {{series: Array<object>, buckets: Array<object>}} report - A token report from the API.
 * @returns {Array<object>} The series to draw.
 */
export function foldSeries(report) {
  const attributed = report.series.filter((series) => series.attributed);
  const unattributed = report.series.filter((series) => !series.attributed);
  if (unattributed.length === 0) return attributed;
  const measureKeys = Object.keys(unattributed[0].totals);
  const sum = (items) =>
    Object.fromEntries(measureKeys.map((key) => [key, items.reduce((total, item) => total + item[key], 0n)]));
  return [
    ...attributed,
    {
      label: UNATTRIBUTED_LABEL,
      color_slot: UNATTRIBUTED_SLOT,
      points: report.buckets.map((_, index) => sum(unattributed.map((series) => series.points[index]))),
      totals: sum(unattributed.map((series) => series.totals)),
    },
  ];
}
