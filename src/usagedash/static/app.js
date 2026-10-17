import { RESPONSE_SIGNATURE_HEADER, createAccess } from './access.js';
import { createFailureBoard, recordingOutcome } from './failures.js';
import {
  DAY_MS,
  FULL_PERCENT,
  MEASURES,
  UNATTRIBUTED_LABEL,
  UNATTRIBUTED_SLOT,
  WEEK_MS,
  foldSeries,
  formatAgo,
  formatAxisTick,
  formatCount,
  formatDuration,
  formatRoundedDollars,
  formatWholeDollars,
  niceCeiling,
  parseTokenReport,
  quotaUsageReading,
  refreshStatusText,
  roundedRetailTotal,
  shareOf,
  stepsInEffectFrom,
  topRoundedBar,
  unpricedTokenTotal,
} from './format.js';
import { latestOnly } from './latest.js';
import { createRename, withAccountLabel, withTokenAccountLabel } from './rename.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const STATE_POLL_MS = 30_000;
const TOKEN_POLL_MS = 300_000;
const CSRF_HEADERS = { 'X-Requested-With': 'usagedash' };
const API_ORIGIN_GLOBAL = '__USAGEDASH_ORIGIN';
const DEFAULT_METHOD = 'GET';
const NOT_SIGNED_IN_MESSAGE = 'this browser is not signed in. Run `usagedash open` in a terminal';
const MS_PER_SECOND = 1000;
const CLOCK_WEEKDAY_HORIZON_MS = 6 * DAY_MS;
const REFRESH_FOLLOW_UP_MS = 5000;
const HTTP_UNAUTHORIZED = 401;
const HTTP_TOO_MANY_REQUESTS = 429;

const CHART_HEIGHT = 260;
const CHART_MARGIN = { top: 12, right: 12, bottom: 28, left: 56 };
const MAX_BAR_WIDTH = 24;
const BAR_BAND_FRACTION = 0.7;
const SEGMENT_GAP = 2;
const Y_TICK_COUNT = 4;
const SPARKLINE_WIDTH = 96;
const SPARKLINE_HEIGHT = 24;
const SPARKLINE_PADDING = 4;
const SPARKLINE_MIN_SPAN_PX = 6;
const SPARKLINE_END_RADIUS = 3;
const TOOLTIP_GAP_PX = 8;
const MIN_CHART_WIDTH = 320;
const MAX_AXIS_LABELS = 8;
const AXIS_LABEL_GAP_PX = 8;
const AXIS_LABEL_BASELINE_SHIFT_PX = 4;

const RANGE_LABELS = { '24h': 'last 24 hours', '7d': 'last 7 days', '30d': 'last 30 days' };

const WINDOW_STATE_LABELS = { warning: 'Near limit', critical: 'Critical', exhausted: 'Limit reached' };
const WINDOW_STATE_TONES = { warning: 'warning', critical: 'critical', exhausted: 'critical' };

const ADVICE_CHIPS = {
  stay: { label: 'Stay', tone: 'good' },
  switch: { label: 'Switch', tone: 'warning' },
  verify: { label: 'Verify', tone: 'warning' },
  hold: { label: 'Wait', tone: 'critical' },
  unknown: { label: 'No data', tone: 'neutral' },
};

const SEVERITY_CHIPS = {
  ok: { label: 'OK', tone: 'good' },
  info: { label: 'Info', tone: 'neutral' },
  warning: { label: 'Waiting', tone: 'warning' },
  error: { label: 'Error', tone: 'critical' },
};

const clockTime = new Intl.DateTimeFormat('en', { weekday: 'short', hour: 'numeric', minute: '2-digit' });
const clockDate = new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
const hourLabel = new Intl.DateTimeFormat('en', { hour: 'numeric' });
const dayLabel = new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric' });
const dayTooltipLabel = new Intl.DateTimeFormat('en', { weekday: 'long', month: 'short', day: 'numeric' });
const hourTooltipLabel = new Intl.DateTimeFormat('en', { weekday: 'short', hour: 'numeric', minute: '2-digit' });

const view = {
  state: null,
  tokens: null,
  range: '7d',
  measure: 'total',
};

// Why: the page opened from the bootstrap file is not at the dashboard's origin, so it must name it. The
// page served by the dashboard itself, and the tests, leave the global unset and address paths as they are.
const API_ORIGIN = typeof window[API_ORIGIN_GLOBAL] === 'string' ? window[API_ORIGIN_GLOBAL] : '';
const access = createAccess(window, { origin: API_ORIGIN });

function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.class) node.className = options.class;
  if (options.text !== undefined) node.textContent = options.text;
  for (const [name, value] of Object.entries(options.attrs ?? {})) node.setAttribute(name, value);
  for (const [name, value] of Object.entries(options.data ?? {})) node.dataset[name] = value;
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  return node;
}

function formatClock(iso, now) {
  const moment = new Date(iso);
  return Math.abs(moment.getTime() - now) < CLOCK_WEEKDAY_HORIZON_MS
    ? clockTime.format(moment)
    : clockDate.format(moment);
}

function statusChip({ label, tone }) {
  return el('span', { class: 'status-chip', data: { tone } }, el('span', { class: 'status-dot' }), label);
}

/**
 * Sends one request to the dashboard, signed and sealed for exactly its method, target, and body over the
 * exchange's challenge and nonce; the answer is used only when it carries the dashboard's signature over
 * that exchange and opens. The token itself is never sent.
 *
 * @param {string} path - The path and query, as the dashboard will see them.
 * @param {RequestInit & { body?: string }} [init] - Passed to `fetch`. A body is JSON text or absent.
 * @returns {Promise<{ ok: boolean, status: number, json: () => Promise<unknown> }>} The verified answer.
 *   Rejects, sending nothing, when the page holds no token: an unsigned exchange with whoever holds the
 *   port could only show fabricated data or send a label in clear.
 */
async function request(path, init = {}) {
  const method = (init.method ?? DEFAULT_METHOD).toUpperCase();
  const signed = await access.authorize(method, path, init.body);
  if (signed === null) throw new Error(NOT_SIGNED_IN_MESSAGE);
  const headers = { ...init.headers, ...signed.headers };
  const body = signed.body ?? init.body;
  const response = await fetch(`${API_ORIGIN}${path}`, { ...init, method, headers, body });
  // Why the answer is checked, then opened: whoever holds the port can answer a request; only the
  // dashboard can sign the answer over this request's challenge and nonce, and only the dashboard could
  // have sealed it. An answer not signed so, a recorded one included, is not shown or acted on.
  const sealed = new Uint8Array(await response.arrayBuffer());
  const genuine = await access.verifyAnswer(
    signed.challenge,
    signed.nonce,
    response.status,
    sealed,
    response.headers.get(RESPONSE_SIGNATURE_HEADER),
  );
  if (!genuine) {
    if (response.status === HTTP_UNAUTHORIZED) throw new Error(NOT_SIGNED_IN_MESSAGE);
    throw new Error(`the answer to ${path} did not come from the dashboard (${response.status})`);
  }
  let text;
  try {
    text = await access.openAnswer(signed.challenge, signed.nonce, response.status, sealed);
  } catch {
    throw new Error(`the answer to ${path} could not be opened (${response.status})`);
  }
  return { ok: response.ok, status: response.status, json: async () => JSON.parse(text) };
}

async function fetchJson(path, init) {
  const response = await request(path, init);
  if (!response.ok) throw new Error(`${path} responded ${response.status}`);
  return response.json();
}

function showFailures(messages) {
  const banner = document.getElementById('load-error');
  banner.hidden = messages.length === 0;
  banner.replaceChildren(...messages.map((message) => el('div', { text: message })));
}

const failures = createFailureBoard(showFailures);

const rename = createRename({
  save: (accountId, label) =>
    fetchJson(`/api/accounts/${accountId}`, {
      method: 'PATCH',
      headers: { ...CSRF_HEADERS, 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    }),
  onChange: () => renderProviders(),
  onFailure: (message) => (message === null ? failures.clear('rename') : failures.report('rename', message)),
  onSaved: (accountId, label) => {
    // Why the name is put into the state the page holds: the server has stored it, so it is shown at
    // once and does not wait on the reload below succeeding.
    // Why Rename: the saving redraw removed the control the keyboard was on, so the closed card gets it
    // back on that account's Rename button, unless the keyboard has moved elsewhere meanwhile.
    const keyboardWaits = focusAfterRename !== null && document.activeElement === document.body;
    focusAfterRename = keyboardWaits ? { kind: 'rename', accountId } : null;
    view.state = withAccountLabel(view.state, accountId, label);
    view.tokens = withTokenAccountLabel(view.tokens, accountId, label);
    renderTokens();
    loadState();
    loadTokens();
  },
});

/* ---------- Quota ---------- */

function resetText(quotaWindow, now) {
  if (quotaWindow.resets_at === null) return 'No reset time reported';
  const resetAt = new Date(quotaWindow.resets_at).getTime();
  if (quotaWindow.reset_passed) return `Reset ${formatAgo(quotaWindow.resets_at, now)}`;
  return `Resets in ${formatDuration(resetAt - now)} · ${formatClock(quotaWindow.resets_at, now)}`;
}

function freshnessText(quotaWindow, now) {
  const ago = formatAgo(quotaWindow.as_of, now);
  return quotaWindow.freshness === 'live' ? `as of ${ago}` : `last checked ${ago}`;
}

function sparkline(steps, windowSeconds, now) {
  const span = windowSeconds !== null && windowSeconds * MS_PER_SECOND <= DAY_MS ? DAY_MS : WEEK_MS;
  const start = now - span;
  const plotWidth = SPARKLINE_WIDTH - 2 * SPARKLINE_PADDING;
  const plotHeight = SPARKLINE_HEIGHT - 2 * SPARKLINE_PADDING;
  const x = (time) => SPARKLINE_PADDING + ((Math.max(time, start) - start) / span) * plotWidth;
  const y = (percent) => SPARKLINE_PADDING + (1 - Math.min(percent, FULL_PERCENT) / FULL_PERCENT) * plotHeight;

  const visible = stepsInEffectFrom(steps, start);
  const emptySlot = () => el('span', { class: 'sparkline', attrs: { 'aria-hidden': 'true' } });
  if (visible.length === 0) return emptySlot();

  // Why: a reading holds until the next one arrives, so each value is drawn flat up to the next reading
  // and then jumps. Joining readings with a slope would invent usage between them.
  const commands = [];
  let endX = 0;
  let endY = 0;
  visible.forEach((step, index) => {
    const next = visible[index + 1];
    const heldUntil = new Date(next ? next.from : step.to).getTime();
    endX = x(heldUntil);
    endY = y(step.used_percent);
    commands.push(
      index === 0 ? `M${x(new Date(step.from).getTime()).toFixed(1)},${endY.toFixed(1)}` : `V${endY.toFixed(1)}`,
    );
    commands.push(`H${endX.toFixed(1)}`);
  });
  const startX = x(new Date(visible[0].from).getTime());
  if (endX - startX < SPARKLINE_MIN_SPAN_PX) return emptySlot();

  const svg = svgEl('svg', {
    class: 'sparkline',
    viewBox: `0 0 ${SPARKLINE_WIDTH} ${SPARKLINE_HEIGHT}`,
    role: 'img',
    'aria-label': `Usage over the ${span === DAY_MS ? 'last 24 hours' : 'last 7 days'}`,
  });
  svg.append(svgEl('path', { class: 'sparkline-line', d: commands.join(' ') }));
  svg.append(
    svgEl('circle', { class: 'sparkline-end', cx: endX.toFixed(1), cy: endY.toFixed(1), r: SPARKLINE_END_RADIUS }),
  );
  return svg;
}

/**
 * One quota window: the percent a person reads, and a meter whose value stays inside 0..100.
 *
 * @param {object} quotaWindow - A window from `/api/state`.
 * @param {Array<object>} steps - That window's history steps. Empty when none have been stored.
 * @param {number} now - The current time in epoch milliseconds.
 * @returns {HTMLElement}
 */
export function renderWindow(quotaWindow, steps, now) {
  const usage = quotaUsageReading(quotaWindow.effective_percent);
  const value = el('span', { class: 'window-value' }, `${usage.label}% used`);
  if (quotaWindow.state !== 'ok') {
    value.append(
      statusChip({ label: WINDOW_STATE_LABELS[quotaWindow.state], tone: WINDOW_STATE_TONES[quotaWindow.state] }),
    );
  }

  const fill = el('div', { class: 'meter-fill' });
  fill.style.setProperty('--fill-width', `${Math.min(100, Math.max(0, quotaWindow.effective_percent))}%`);
  const meter = el(
    'div',
    {
      class: 'meter',
      data: { state: quotaWindow.state },
      attrs: {
        role: 'meter',
        'aria-label': `${quotaWindow.label} used`,
        'aria-valuemin': '0',
        'aria-valuemax': '100',
        'aria-valuenow': usage.meterNow,
        'aria-valuetext': usage.meterText,
      },
    },
    fill,
  );

  const row = el(
    'div',
    { class: 'window' },
    el('div', { class: 'window-top' }, el('span', { class: 'window-label', text: quotaWindow.label }), value),
    el('div', { class: 'window-middle' }, meter, sparkline(steps, quotaWindow.window_seconds, now)),
    el(
      'div',
      { class: 'window-bottom' },
      el('span', { text: resetText(quotaWindow, now) }),
      el('span', { text: freshnessText(quotaWindow, now) }),
    ),
  );

  if (quotaWindow.reset_passed) {
    row.append(
      el('div', {
        class: 'window-note',
        text: `Was ${quotaUsageReading(quotaWindow.used_percent).label}% before the reset. Shown as 0% until this account is read again.`,
      }),
    );
  }
  if (quotaWindow.pace_exhausts_at !== null) {
    const remaining = new Date(quotaWindow.pace_exhausts_at).getTime() - now;
    row.append(
      el('div', {
        class: 'window-note',
        text: `At the current pace this limit is reached in about ${formatDuration(Math.max(remaining, 0))}.`,
      }),
    );
  }
  return row;
}

/** Builds the rename form from the rename's own state, so it can be built again at any time. */
function renameForm(edit) {
  const isSaving = edit.phase === 'saving';
  const input = el('input', {
    attrs: { type: 'text', maxlength: '60', required: '', 'aria-label': 'Account name' },
    data: { focus: 'name' },
  });
  input.value = edit.draft;
  const save = el('button', {
    class: 'button',
    text: isSaving ? 'Saving…' : 'Save',
    attrs: { type: 'submit' },
    data: { focus: 'save' },
  });
  const cancel = el('button', {
    class: 'button',
    text: 'Cancel',
    attrs: { type: 'button' },
    data: { focus: 'cancel' },
  });
  // Why everything is disabled while saving: the name being saved is the one that was submitted, and
  // nothing typed or pressed until the answer arrives could be honored.
  for (const control of [input, save, cancel]) control.disabled = isSaving;
  const form = el('form', { class: 'rename-form' }, input, save, cancel);
  input.addEventListener('input', () => rename.type(input.value));
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const captured = renameControlFocus(edit.accountId);
    const held = focusAfterRename;
    if (captured !== null) focusAfterRename = captured;
    rename.submit();
    // A blank name never leaves editing, so the capture must not move focus on the next redraw.
    if (rename.state().phase !== 'saving') focusAfterRename = held;
  });
  cancel.addEventListener('click', () => {
    const held = focusAfterRename;
    focusAfterRename = { kind: 'rename', accountId: edit.accountId };
    if (!rename.cancel()) focusAfterRename = held;
  });
  return form;
}

function renameInput() {
  return document.querySelector('.rename-form input');
}

/** True when `node` is `root` or drawn inside it. */
function containsNode(root, node) {
  let current = node;
  while (current !== null && current !== undefined) {
    if (current === root) return true;
    current = current.parentNode ?? null;
  }
  return false;
}

/**
 * The keyboard control inside `root` that a redraw is about to throw away.
 *
 * A poll replaces the account cards and the chart even while someone is using them. Remembering only the
 * rename field's caret left Rename, Save, Cancel, and the selected token bucket behind.
 *
 * @param {HTMLElement | null} root - The element whose contents are about to be replaced.
 * @returns {object | null} The control's kind, and the account id, bucket key, or caret when it has one.
 */
function rememberedFocus(root) {
  const active = document.activeElement;
  if (root === null || active === null || active === undefined || active === document.body) return null;
  if (!containsNode(root, active)) return null;
  const kind = active.dataset?.focus;
  if (typeof kind !== 'string' || kind === '') return null;
  return {
    kind,
    accountId: active.dataset.account,
    bucket: active.dataset.bucket,
    start: active.selectionStart,
    end: active.selectionEnd,
  };
}

/**
 * The replacement for a control a redraw removed.
 * @param {HTMLElement} root
 * @param {{kind: string, accountId: string | undefined, bucket: string | undefined}} remembered
 * @returns {HTMLElement | null}
 */
function replacementControl(root, remembered) {
  let found = null;
  const visit = (node) => {
    if (found !== null || node === null || typeof node !== 'object') return;
    const kind = node.dataset?.focus;
    const sameAccount = remembered.kind !== 'rename' || node.dataset.account === remembered.accountId;
    const sameBucket = remembered.kind !== 'bucket' || node.dataset.bucket === remembered.bucket;
    if (kind === remembered.kind && sameAccount && sameBucket) found = node;
    for (const child of node.children ?? []) visit(child);
  };
  visit(root);
  return found;
}

/**
 * Puts the keyboard focus back on the control a redraw replaced.
 * A control that is gone, or disabled, is left alone.
 * @param {HTMLElement | null} root
 * @param {ReturnType<typeof rememberedFocus>} remembered
 * @returns {boolean} True when focus was moved.
 */
function restoreRememberedFocus(root, remembered) {
  if (remembered === null || root === null) return false;
  const target = replacementControl(root, remembered);
  if (target === null || target.disabled) return false;
  target.focus({ preventScroll: true });
  if (remembered.kind === 'name' && Number.isInteger(remembered.start) && Number.isInteger(remembered.end)) {
    target.setSelectionRange(remembered.start, remembered.end);
  }
  return true;
}

/**
 * Focus owed after Cancel, a failed Save, or a saved one.
 *
 * Saving disables the focused control at once, so that redraw cannot focus it, and the redraw that
 * reopens or closes the form would otherwise find the keyboard on the page body. The handoff waits
 * until an enabled replacement exists, and is dropped if the keyboard has moved elsewhere meanwhile.
 * Cancel's and a saved rename's replacement is that account's Rename button.
 * @type {{kind: string, accountId: string, start?: number, end?: number} | null}
 */
let focusAfterRename = null;

/**
 * The rename control the keyboard is on, when a save should return to it.
 * @param {string} accountId
 * @returns {{kind: string, accountId: string, start: number, end: number} | null}
 */
function renameControlFocus(accountId) {
  const active = document.activeElement;
  if (active === null || active === undefined || active === document.body) return null;
  const kind = active.dataset?.focus;
  if (kind !== 'name' && kind !== 'save' && kind !== 'cancel') return null;
  // A button has no caret. Reading selectionStart is only meaningful on the name field.
  const start = kind === 'name' ? active.selectionStart : 0;
  const end = kind === 'name' ? active.selectionEnd : 0;
  return { kind, accountId, start, end };
}

/**
 * @param {object} state
 * @param {string} accountId
 * @returns {boolean}
 */
function accountIsOnScreen(state, accountId) {
  return state.providers.some((provider) => provider.accounts.some((account) => account.id === accountId));
}

/**
 * Moves the keyboard to the control a rename transition left behind, once that control is enabled.
 * A handoff is dropped when the keyboard has moved to another control meanwhile, or when the account is
 * no longer on screen, so it can neither override a choice nor land on a different account.
 * @param {HTMLElement} container
 * @param {object} state
 */
function applyFocusAfterRename(container, state) {
  if (focusAfterRename === null) return;
  const active = document.activeElement;
  if (active !== null && active !== undefined && active !== document.body) {
    focusAfterRename = null;
    return;
  }
  if (!accountIsOnScreen(state, focusAfterRename.accountId)) {
    focusAfterRename = null;
    return;
  }
  if (restoreRememberedFocus(container, focusAfterRename)) focusAfterRename = null;
}

function accountBadge(account) {
  if (account.is_active) return statusChip({ label: 'Active', tone: 'good' });
  if (account.is_live) return statusChip({ label: 'Logged in', tone: 'good' });
  return statusChip({ label: 'Logged out', tone: 'neutral' });
}

function standingLoginHint(providerKey) {
  return el(
    'span',
    {},
    'Run ',
    el('code', { text: `usagedash login ${providerKey}` }),
    ' to track it continuously without switching your working login.',
  );
}

function renderAccount(account, providerKey, history, now) {
  const header = el(
    'div',
    { class: 'card-header' },
    el('span', { class: 'swatch', data: { slot: account.color_slot } }),
  );
  const edit = rename.state();
  if (edit.phase !== 'idle' && edit.accountId === account.id) {
    header.append(renameForm(edit));
  } else {
    const open = el('button', {
      class: 'button button-quiet',
      text: 'Rename',
      attrs: { type: 'button' },
      data: { focus: 'rename', account: account.id },
    });
    // Why disabled while another name is being saved: one rename is under way at a time.
    open.disabled = edit.phase === 'saving';
    open.addEventListener('click', () => {
      if (rename.open(account.id, account.label)) renameInput().select();
    });
    header.append(
      el('h3', { text: account.label, attrs: { title: account.label } }),
      accountBadge(account),
      el('span', { class: 'spacer' }),
      open,
    );
  }

  const meta = [account.plan, account.email !== account.label ? account.email : null].filter(Boolean).join(' · ');
  const card = el(
    'article',
    { class: 'card' },
    header,
    el('div', { class: 'card-meta', text: meta || 'Plan unknown' }),
  );

  if (account.windows.length === 0) {
    card.append(
      el('div', { class: 'empty' }, 'No quota reading for this account yet. ', standingLoginHint(providerKey)),
    );
  }
  for (const quotaWindow of account.windows) {
    const steps = history.get(`${account.id}:${quotaWindow.key}`) ?? [];
    card.append(renderWindow(quotaWindow, steps, now));
  }
  if (account.facts.length > 0) {
    const facts = el('ul', { class: 'facts' });
    for (const fact of account.facts) {
      const reset = fact.resets_at ? ` · resets ${formatClock(fact.resets_at, now)}` : '';
      facts.append(el('li', { text: `${fact.label}: ${fact.value}${reset}` }));
    }
    card.append(facts);
  }
  return card;
}

function placeholderCard(provider) {
  return el(
    'article',
    { class: 'card card-placeholder' },
    el('h3', { text: 'Account not seen yet' }),
    el(
      'p',
      {},
      `Your other ${provider.display_name} account has not been logged in here. `,
      standingLoginHint(provider.provider),
    ),
  );
}

function adviceText(advice, now) {
  let text = advice.summary;
  if (advice.as_of !== null) text += ` Last read ${formatAgo(advice.as_of, now)}.`;
  if (advice.next_available_at !== null) {
    const wait = new Date(advice.next_available_at).getTime() - now;
    text += ` First reset in ${formatDuration(Math.max(wait, 0))} (${formatClock(advice.next_available_at, now)}).`;
  }
  return text;
}

function renderProviders() {
  const state = view.state;
  if (state === null) return;
  // Why the cards are rebuilt even while a name is being edited: they show quota and advice, which
  // must not go stale for as long as a form is open. The form is rebuilt from the rename's own
  // state, and the control that had the keyboard focus gets it back.
  const container = document.getElementById('providers');
  const remembered = rememberedFocus(container);
  const now = Date.now();
  const history = new Map(state.history.map((series) => [`${series.account_id}:${series.window_key}`, series.steps]));
  const sections = state.providers.map((provider) => {
    const cards = el('div', { class: 'cards' });
    for (const account of provider.accounts) cards.append(renderAccount(account, provider.provider, history, now));
    for (let index = provider.accounts.length; index < state.expected_accounts_per_provider; index += 1) {
      cards.append(placeholderCard(provider));
    }
    return el(
      'div',
      { class: 'provider' },
      el(
        'div',
        { class: 'provider-header' },
        el('h3', { text: provider.display_name }),
        el(
          'div',
          { class: 'advice' },
          statusChip(ADVICE_CHIPS[provider.advice.action]),
          el('span', { text: adviceText(provider.advice, now) }),
        ),
      ),
      cards,
    );
  });
  container.replaceChildren(...sections);
  restoreRememberedFocus(container, remembered);
  applyFocusAfterRename(container, state);
}

function renderSources() {
  const now = Date.now();
  const items = view.state.sources.map((source) => {
    const [providerKey, ...pathParts] = source.source_id.split(':');
    const success = source.last_success_at
      ? `last success ${formatAgo(source.last_success_at, now)}`
      : 'no success yet';
    return el(
      'li',
      { class: 'source' },
      statusChip(SEVERITY_CHIPS[source.severity]),
      el('span', { class: 'source-name', text: `${providerKey} ${source.kind}` }),
      el('span', { class: 'source-message', text: source.message ?? pathParts.join(':') }),
      el('span', { class: 'muted', text: success }),
    );
  });
  document.getElementById('sources').replaceChildren(...items);
}

function renderHeader() {
  const { collector } = view.state;
  const updated = document.getElementById('updated');
  if (collector.running) updated.textContent = 'Collecting…';
  else if (collector.last_finished_at)
    updated.textContent = `Collected ${formatAgo(collector.last_finished_at, Date.now())}`;
  else updated.textContent = 'Waiting for the first collection';
}

/* ---------- Token usage ---------- */

function bucketAxisLabel(bucket, unit) {
  const moment = new Date(bucket.start);
  return unit === 'hour' ? hourLabel.format(moment) : dayLabel.format(moment);
}

function bucketTooltipLabel(bucket, unit) {
  const moment = new Date(bucket.start);
  return unit === 'hour' ? hourTooltipLabel.format(moment) : dayTooltipLabel.format(moment);
}

function showTooltip(anchor, title, rows) {
  const tooltip = document.getElementById('tooltip');
  tooltip.replaceChildren(
    el('div', { class: 'tooltip-title', text: title }),
    ...rows.map((row) =>
      el(
        'div',
        { class: 'tooltip-row' },
        el('span', { class: 'tooltip-key', data: { slot: row.slot } }),
        el('span', { class: 'tooltip-value', text: formatCount(row.value) }),
        el('span', { class: 'tooltip-name', text: row.label }),
      ),
    ),
  );
  tooltip.hidden = false;
  const anchorBox = anchor.getBoundingClientRect();
  const tooltipBox = tooltip.getBoundingClientRect();
  const left = Math.min(
    window.innerWidth - tooltipBox.width - TOOLTIP_GAP_PX,
    Math.max(TOOLTIP_GAP_PX, anchorBox.right + TOOLTIP_GAP_PX),
  );
  const top = Math.min(
    window.innerHeight - tooltipBox.height - TOOLTIP_GAP_PX,
    Math.max(TOOLTIP_GAP_PX, anchorBox.top),
  );
  tooltip.style.setProperty('left', `${left}px`);
  tooltip.style.setProperty('top', `${top}px`);
}

function hideTooltip() {
  document.getElementById('tooltip').hidden = true;
}

function renderChart(report, series, measure) {
  const container = document.getElementById('chart');
  const remembered = rememberedFocus(container);
  // Why before the replacement: removing a focused bucket does not always blur it, so the tooltip
  // would keep that bucket's old counts over the chart that replaced it.
  hideTooltip();
  const bucketTotals = report.buckets.map((_, index) =>
    series.reduce((total, item) => total + measure.value(item.points[index]), 0n),
  );
  if (bucketTotals.every((total) => total === 0n)) {
    container.replaceChildren(el('div', { class: 'empty', text: 'No token usage recorded in this range.' }));
    restoreRememberedFocus(container, remembered);
    return;
  }

  const width = Math.max(container.clientWidth, MIN_CHART_WIDTH);
  const plotWidth = width - CHART_MARGIN.left - CHART_MARGIN.right;
  const plotHeight = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom;
  const peak = bucketTotals.reduce((highest, total) => (total > highest ? total : highest), 0n);
  const yMax = niceCeiling(peak);
  const band = plotWidth / report.buckets.length;
  const barWidth = Math.min(MAX_BAR_WIDTH, band * BAR_BAND_FRACTION);
  const baselineY = CHART_MARGIN.top + plotHeight;
  const labelEvery = Math.ceil(report.buckets.length / MAX_AXIS_LABELS);

  const svg = svgEl('svg', {
    viewBox: `0 0 ${width} ${CHART_HEIGHT}`,
    height: CHART_HEIGHT,
    role: 'img',
    'aria-label': document.getElementById('chart-title').textContent,
  });

  for (let tick = 0; tick <= Y_TICK_COUNT; tick += 1) {
    const y = baselineY - (plotHeight * tick) / Y_TICK_COUNT;
    svg.append(
      svgEl('line', {
        class: tick === 0 ? 'baseline' : 'gridline',
        x1: CHART_MARGIN.left,
        x2: width - CHART_MARGIN.right,
        y1: y,
        y2: y,
      }),
    );
    const label = svgEl('text', {
      x: CHART_MARGIN.left - AXIS_LABEL_GAP_PX,
      y: y + AXIS_LABEL_BASELINE_SHIFT_PX,
      'text-anchor': 'end',
    });
    label.textContent = formatAxisTick(yMax, tick, Y_TICK_COUNT);
    svg.append(label);
  }

  report.buckets.forEach((bucket, index) => {
    const bandX = CHART_MARGIN.left + band * index;
    const barX = bandX + (band - barWidth) / 2;
    const visible = series.filter((item) => measure.value(item.points[index]) > 0n);
    let cursorY = baselineY;
    visible.forEach((item, position) => {
      const fullHeight = shareOf(measure.value(item.points[index]), yMax, plotHeight);
      const isTop = position === visible.length - 1;
      const height = Math.max(1, isTop ? fullHeight : fullHeight - SEGMENT_GAP);
      const y = cursorY - height;
      const mark = isTop
        ? svgEl('path', { d: topRoundedBar(barX, y, barWidth, height) })
        : svgEl('rect', { x: barX, y, width: barWidth, height });
      mark.setAttribute('class', 'segment-mark');
      mark.dataset.slot = item.color_slot;
      svg.append(mark);
      cursorY -= fullHeight;
    });

    if (index % labelEvery === 0) {
      const label = svgEl('text', {
        x: bandX + band / 2,
        y: CHART_HEIGHT - AXIS_LABEL_GAP_PX,
        'text-anchor': 'middle',
      });
      label.textContent = bucketAxisLabel(bucket, report.bucket);
      svg.append(label);
    }

    const title = bucketTooltipLabel(bucket, report.bucket);
    const rows = [
      ...visible
        .map((item) => ({ slot: item.color_slot, label: item.label, value: measure.value(item.points[index]) }))
        .reverse(),
      { slot: UNATTRIBUTED_SLOT, label: 'Total', value: bucketTotals[index] },
    ];
    const hit = svgEl('rect', {
      class: 'hit-band',
      x: bandX,
      y: CHART_MARGIN.top,
      width: band,
      height: plotHeight,
      tabindex: 0,
      'aria-label': `${title}: ${formatCount(bucketTotals[index])} tokens`,
    });
    hit.dataset.focus = 'bucket';
    hit.dataset.bucket = bucket.key;
    for (const eventName of ['pointerenter', 'focus'])
      hit.addEventListener(eventName, () => showTooltip(hit, title, rows));
    for (const eventName of ['pointerleave', 'blur']) hit.addEventListener(eventName, hideTooltip);
    svg.append(hit);
  });

  container.replaceChildren(svg);
  restoreRememberedFocus(container, remembered);
}

function renderLegend(series) {
  const legend = document.getElementById('legend');
  if (series.length < 2) {
    legend.replaceChildren();
    return;
  }
  legend.replaceChildren(
    ...series.map((item) =>
      el(
        'span',
        { class: 'legend-item' },
        el('span', { class: 'swatch', data: { slot: item.color_slot } }),
        item.label,
      ),
    ),
  );
}

function renderTiles(report, measure) {
  const rangeLabel = RANGE_LABELS[report.range];
  const grandTotal = report.provider_totals.reduce((total, item) => total + measure.value(item.totals), 0n);
  const tiles = [
    {
      label: 'All providers',
      value: grandTotal,
      requests: report.provider_totals.reduce((total, item) => total + item.totals.requests, 0n),
    },
    ...report.provider_totals.map((item) => ({
      label: item.display_name,
      value: measure.value(item.totals),
      requests: item.totals.requests,
    })),
  ];
  document.getElementById('token-tiles').replaceChildren(
    ...tiles.map((tile) =>
      el(
        'div',
        { class: 'tile' },
        el('div', { class: 'tile-label', text: `${tile.label} · ${rangeLabel}` }),
        el('div', {
          class: 'tile-value',
          text: formatCount(tile.value),
          attrs: { title: formatCount(tile.value) },
        }),
        el('div', { class: 'tile-detail', text: `${formatCount(tile.requests)} requests` }),
      ),
    ),
  );
}

function numericCell(value) {
  return el('td', { class: 'numeric', text: formatCount(value) });
}

function accountCell(label, providerName, slot) {
  return el(
    'td',
    {},
    el(
      'div',
      { class: 'account-cell' },
      el('span', { class: 'swatch', data: { slot } }),
      el('span', { text: label }),
      el('span', { class: 'muted', text: providerName }),
    ),
  );
}

function renderTables(report) {
  const providerNames = new Map(report.provider_totals.map((item) => [item.provider, item.display_name]));
  const seriesByKey = new Map(report.series.map((series) => [`${series.provider}:${series.account_id}`, series]));

  const rows = report.series.map((series) =>
    el(
      'tr',
      {},
      accountCell(series.label, providerNames.get(series.provider), series.color_slot),
      numericCell(series.totals.input),
      numericCell(series.totals.cache_read),
      numericCell(series.totals.cache_write),
      numericCell(series.totals.output),
      numericCell(series.totals.total),
      numericCell(series.totals.requests),
    ),
  );
  document.querySelector('#token-table tbody').replaceChildren(...rows);

  const modelRows = report.models.map((row) => {
    const series = seriesByKey.get(`${row.provider}:${row.account_id}`);
    return el(
      'tr',
      {},
      accountCell(
        series?.label ?? UNATTRIBUTED_LABEL,
        providerNames.get(row.provider),
        series?.color_slot ?? UNATTRIBUTED_SLOT,
      ),
      el('td', { text: row.model }),
      numericCell(row.totals.input),
      numericCell(row.totals.cache_read),
      numericCell(row.totals.cache_write),
      numericCell(row.totals.output),
      numericCell(row.totals.total),
    );
  });
  document.querySelector('#model-table tbody').replaceChildren(...modelRows);
}

function moneyCell(microdollars) {
  return el('td', { class: 'numeric', text: formatRoundedDollars(microdollars) });
}

function rateGapCell(tokens) {
  return el('td', { class: 'numeric', text: tokens === 0n ? '0' : formatCount(tokens) });
}

function modelCostCell(model) {
  if (model.microdollars === null) return el('td', { class: 'numeric muted', text: 'No rate' });
  return moneyCell(model.microdollars);
}

function renderCosts(report) {
  const costs = report.costs;
  const providerNames = new Map(report.provider_totals.map((item) => [item.provider, item.display_name]));
  const seriesByKey = new Map(report.series.map((series) => [`${series.provider}:${series.account_id}`, series]));
  const accountColumn = (provider, accountId) => {
    const series = seriesByKey.get(`${provider}:${accountId}`);
    return accountCell(
      series?.label ?? UNATTRIBUTED_LABEL,
      providerNames.get(provider),
      series?.color_slot ?? UNATTRIBUTED_SLOT,
    );
  };

  document.getElementById('cost-note').textContent = costs.note;
  const rows = costs.accounts.map((account) =>
    el(
      'tr',
      {},
      accountColumn(account.provider, account.account_id),
      moneyCell(account.priced_microdollars),
      rateGapCell(account.unpriced_tokens),
    ),
  );
  if (costs.accounts.length > 0) {
    rows.push(
      el(
        'tr',
        {},
        el('th', { attrs: { scope: 'row' }, text: 'All accounts' }),
        el('td', { class: 'numeric', text: formatWholeDollars(roundedRetailTotal(costs.accounts)) }),
        rateGapCell(unpricedTokenTotal(costs.accounts)),
      ),
    );
  }
  document.querySelector('#cost-table tbody').replaceChildren(...rows);

  const modelRows = costs.accounts.flatMap((account) =>
    account.models.map((model) =>
      el(
        'tr',
        {},
        accountColumn(account.provider, account.account_id),
        el('td', { text: model.model }),
        modelCostCell(model),
        rateGapCell(model.unpriced_tokens),
      ),
    ),
  );
  document.querySelector('#cost-model-table tbody').replaceChildren(...modelRows);
}

function renderTokens() {
  const report = view.tokens;
  if (report === null) return;
  const measure = MEASURES[view.measure];
  const series = foldSeries(report);
  const unit = report.bucket === 'hour' ? 'hour' : 'day';
  document.getElementById('chart-title').textContent =
    `Tokens per ${unit} by account · ${measure.label} · ${RANGE_LABELS[report.range]}`;
  renderTiles(report, measure);
  renderLegend(series);
  renderChart(report, series, measure);
  renderTables(report);
  renderCosts(report);
}

/* ---------- Loading ---------- */

const describeLoadFailure = (error) => `Could not load dashboard data: ${error.message}`;

/**
 * Shows one `/api/state` payload.
 *
 * The account cards are rebuilt, including a rename form that is open. A control inside those cards that
 * had the keyboard focus is focused again. Cancel and a saved rename move focus to that account's Rename
 * button, and a failed save back to the control that was in use, unless the keyboard moved meanwhile.
 *
 * @param {object} state - The dashboard state returned by `/api/state`.
 */
export function showDashboardState(state) {
  view.state = state;
  renderHeader();
  renderSources();
  renderProviders();
}

const loadState = recordingOutcome(
  latestOnly(
    () => fetchJson('/api/state'),
    (state) => showDashboardState(state),
  ),
  failures,
  'state',
  describeLoadFailure,
);

const loadTokenReport = latestOnly(
  () => {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return fetchJson(`/api/tokens?range=${view.range}&tz=${encodeURIComponent(zone)}`);
  },
  (report) => {
    view.tokens = parseTokenReport(report);
    renderTokens();
  },
);

let newestTokenReport = 0;

/**
 * Loads the token report and keeps the refreshing indicator up until the newest of these loads settles.
 *
 * @returns {Promise<boolean>} True when this call's report was shown, false when a newer call superseded
 *   it. Rejects when this call was still the newest and the load failed.
 */
export async function showTokenReport() {
  const body = document.getElementById('token-body');
  newestTokenReport += 1;
  const ticket = newestTokenReport;
  body.classList.add('is-refreshing');
  try {
    return await loadTokenReport();
  } finally {
    // Why: this call already knows its own result in the microtask before this runs, so a newer load can
    // have started. Only the newest load may take the indicator down.
    if (ticket === newestTokenReport) body.classList.remove('is-refreshing');
  }
}

const loadTokens = recordingOutcome(showTokenReport, failures, 'tokens', describeLoadFailure);

async function requestRefresh(button) {
  button.disabled = true;
  try {
    const response = await request('/api/refresh', { method: 'POST', headers: CSRF_HEADERS });
    if (!response.ok && response.status !== HTTP_TOO_MANY_REQUESTS)
      throw new Error(`refresh responded ${response.status}`);
    failures.clear('refresh');
    document.getElementById('updated').textContent = refreshStatusText(response.status);
  } catch (error) {
    failures.report('refresh', `Refresh failed: ${error.message}`);
  } finally {
    setTimeout(() => {
      button.disabled = false;
      loadState();
      loadTokens();
    }, REFRESH_FOLLOW_UP_MS);
  }
}

function start() {
  for (const button of document.querySelectorAll('.segment')) {
    button.addEventListener('click', () => {
      view.range = button.dataset.range;
      for (const other of document.querySelectorAll('.segment'))
        other.setAttribute('aria-pressed', String(other === button));
      loadTokens();
    });
  }
  document.getElementById('measure').addEventListener('change', (event) => {
    view.measure = event.target.value;
    renderTokens();
  });
  const refresh = document.getElementById('refresh');
  refresh.addEventListener('click', () => requestRefresh(refresh));
  window.addEventListener('resize', renderTokens);

  loadState();
  loadTokens();
  setInterval(loadState, STATE_POLL_MS);
  setInterval(loadTokens, TOKEN_POLL_MS);
}

if (document.getElementById('token-body') !== null) start();
