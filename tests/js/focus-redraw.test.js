import assert from 'node:assert/strict';
import { before, describe, it } from 'node:test';

import { TOKEN, dashboard, flush } from './fakes/dashboard.js';

/**
 * A document that can hold keyboard focus. The page is not started: `#token-body` is absent until the
 * tests mount it, which is after this module has been imported.
 */
const mounted = new Map();

function connected(node) {
  let current = node;
  while (current !== null && current !== undefined) {
    for (const root of mounted.values()) if (current === root) return true;
    current = current.parentNode ?? null;
  }
  return false;
}

function element(tag) {
  const classes = new Set();
  const node = {
    tag,
    id: tag,
    className: '',
    dataset: {},
    attrs: {},
    children: [],
    parentNode: null,
    disabled: false,
    hidden: false,
    value: '',
    selectionStart: 0,
    selectionEnd: 0,
    clientWidth: 800,
    listeners: {},
    style: { setProperty() {} },
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
    },
    setAttribute(name, value) {
      node.attrs[name] = value;
    },
    addEventListener(type, fn) {
      node.listeners[type] = fn;
    },
    append(...children) {
      for (const child of children) {
        if (child === null || child === undefined) continue;
        if (typeof child !== 'object') {
          node.children.push(child);
          continue;
        }
        child.parentNode = node;
        node.children.push(child);
      }
    },
    replaceChildren(...children) {
      for (const child of node.children) {
        if (child !== null && typeof child === 'object') child.parentNode = null;
      }
      node.children = [];
      node.append(...children);
      const active = document.activeElement;
      if (active !== null && active !== document.body && !connected(active)) document.activeElement = document.body;
    },
    focus() {
      document.activeElement = node;
      document.focused.push(node);
    },
    select() {
      node.selectionStart = 0;
      node.selectionEnd = node.value.length;
      node.focus();
    },
    setSelectionRange(start, end) {
      node.selectionStart = start;
      node.selectionEnd = end;
    },
  };
  return node;
}

function mount(id) {
  const node = element('div');
  node.id = id;
  mounted.set(id, node);
  return node;
}

function hasClass(node, className) {
  return typeof node.className === 'string' && node.className.split(/\s+/).includes(className);
}

function ancestorHasClass(node, className) {
  let current = node;
  while (current !== null && current !== undefined) {
    if (hasClass(current, className)) return true;
    current = current.parentNode ?? null;
  }
  return false;
}

function walk(visit) {
  const visitNode = (node) => {
    if (node === null || typeof node !== 'object') return;
    visit(node);
    for (const child of node.children ?? []) visitNode(child);
  };
  for (const root of mounted.values()) visitNode(root);
}

globalThis.window = {
  location: { hash: '', pathname: '/', search: '' },
  history: { replaceState() {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  addEventListener() {},
  innerWidth: 1200,
  innerHeight: 800,
  __USAGEDASH_TOKEN: TOKEN,
};
globalThis.fetch = () => Promise.reject(new Error('fetch during import'));
globalThis.document = {
  body: element('body'),
  activeElement: null,
  getElementById: (id) => mounted.get(id) ?? null,
  querySelector(selector) {
    if (
      selector === '#token-table tbody' ||
      selector === '#model-table tbody' ||
      selector === '#cost-table tbody' ||
      selector === '#cost-model-table tbody'
    ) {
      return mounted.get(selector) ?? null;
    }
    if (selector === '.rename-form input') {
      let found = null;
      walk((node) => {
        if (found === null && node.tag === 'input' && ancestorHasClass(node, 'rename-form')) found = node;
      });
      return found;
    }
    return null;
  },
  querySelectorAll: () => [],
  createElement: (tag) => element(tag),
  createElementNS: (_namespace, tag) => element(tag),
};
document.activeElement = document.body;
document.focused = [];

const { showDashboardState, showTokenReport } = await import('../../src/usagedash/static/app.js');

const LARGE_ACCOUNT_ID = '9007199254740993';

function account(id, label) {
  return {
    id,
    label,
    color_slot: '1',
    is_active: true,
    is_live: true,
    plan: 'Pro',
    email: `${label}@example.com`,
    windows: [],
    facts: [],
  };
}

function dashboardState() {
  return {
    expected_accounts_per_provider: 2,
    history: [],
    sources: [],
    collector: { running: false, last_finished_at: null },
    providers: [
      {
        provider: 'claude',
        display_name: 'Claude',
        advice: { action: 'stay', summary: 'Room left.', as_of: null, next_available_at: null },
        accounts: [account(LARGE_ACCOUNT_ID, 'Account A'), account('8', 'Account B')],
      },
    ],
  };
}

function counts(total) {
  return {
    input: '1',
    cache_read: '0',
    cache_write: '0',
    output: '1',
    reasoning: '0',
    requests: '1',
    total,
  };
}

function tokenReport(total = '2') {
  const row = counts(total);
  return {
    range: '7d',
    bucket: 'day',
    timezone: 'UTC',
    buckets: [
      { key: '2026-10-05', start: '2026-10-05T00:00:00Z' },
      { key: '2026-10-06', start: '2026-10-06T00:00:00Z' },
    ],
    series: [
      {
        provider: 'claude',
        account_id: '7',
        label: 'Account A',
        color_slot: '1',
        attributed: true,
        points: [row, row],
        totals: row,
      },
    ],
    models: [],
    provider_totals: [{ provider: 'claude', display_name: 'Claude', totals: row }],
    costs: {
      priced_on: '2026-10-08',
      currency: 'USD',
      unit: 'microdollar',
      note: 'Published standard rates. This is not a subscription charge.',
      accounts: [],
    },
  };
}

function controls(kind) {
  const found = [];
  walk((node) => {
    if (node.dataset?.focus === kind) found.push(node);
  });
  return found;
}

function click(node) {
  node.focus();
  node.listeners.click?.({ preventDefault() {} });
}

async function drawTokens(report) {
  dashboard('', () => ({ status: 200, body: JSON.stringify(report) }));
  assert.equal(await showTokenReport(), true);
}

/** A dashboard whose answer to the save waits until the test rejects it. */
function pendingSave() {
  const pending = {};
  dashboard(
    '',
    () =>
      new Promise((_resolve, reject) => {
        pending.reject = reject;
      }),
  );
  return pending;
}

/** A save asks for a challenge, signs, sends, verifies, and opens: it is waited for by what it changes. */
const settleSave = () => flush(() => controls('save').length === 0);
const formReopened = () => flush(() => controls('save')[0]?.disabled === false);

function idleCards() {
  const cancel = controls('cancel')[0];
  if (cancel !== undefined && cancel.disabled === false) cancel.listeners.click();
  showDashboardState(dashboardState());
  document.activeElement = document.body;
}

function openRename() {
  idleCards();
  click(controls('rename').find((button) => button.dataset.account === LARGE_ACCOUNT_ID));
}

describe('keyboard focus across redraws', () => {
  before(() => {
    mount('providers');
    mount('sources');
    mount('updated');
    mount('load-error');
    mount('token-body');
    mount('token-tiles');
    mount('legend');
    mount('chart');
    mount('chart-title');
    mount('tooltip');
    mount('#token-table tbody');
    mount('#model-table tbody');
    mount('cost-note');
    mount('#cost-table tbody');
    mount('#cost-model-table tbody');
  });

  it('keeps Rename focused on the same account when the cards are drawn again', () => {
    idleCards();
    const rename = controls('rename').find((button) => button.dataset.account === LARGE_ACCOUNT_ID);
    rename.focus();

    showDashboardState(dashboardState());

    const focused = document.activeElement;
    assert.equal(focused.dataset.focus, 'rename');
    assert.equal(focused.dataset.account, LARGE_ACCOUNT_ID);
    assert.notEqual(focused, rename);
  });

  it('keeps Save focused when the cards are drawn again', () => {
    openRename();
    const save = controls('save')[0];
    save.focus();

    showDashboardState(dashboardState());

    assert.equal(document.activeElement.dataset.focus, 'save');
    assert.notEqual(document.activeElement, save);
    assert.equal(document.activeElement.disabled, false);
  });

  it('keeps Cancel focused when the cards are drawn again', () => {
    openRename();
    const cancel = controls('cancel')[0];
    cancel.focus();

    showDashboardState(dashboardState());

    assert.equal(document.activeElement.dataset.focus, 'cancel');
    assert.notEqual(document.activeElement, cancel);
  });

  it('puts the rename caret back in the same place', () => {
    openRename();
    const input = document.querySelector('.rename-form input');
    input.focus();
    input.setSelectionRange(1, 4);

    showDashboardState(dashboardState());

    const restored = document.activeElement;
    assert.equal(restored, document.querySelector('.rename-form input'));
    assert.equal(restored.selectionStart, 1);
    assert.equal(restored.selectionEnd, 4);
  });

  it('does not focus Save once the save has disabled it', async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    const pending = pendingSave();
    save.parentNode.listeners.submit({ preventDefault() {} });

    assert.equal(document.activeElement, document.body);
    assert.equal(controls('save')[0].disabled, true);

    await flush(() => pending.reject !== undefined);
    pending.reject(new Error('stopped'));
    await formReopened();
    controls('cancel')[0].listeners.click();
  });

  it('focuses the closed account Rename button when Cancel closes the form', () => {
    openRename();
    const cancel = controls('cancel')[0];
    cancel.focus();

    cancel.listeners.click();

    const focused = document.activeElement;
    assert.equal(focused.dataset.focus, 'rename');
    assert.equal(focused.dataset.account, LARGE_ACCOUNT_ID);
    assert.equal(controls('cancel').length, 0);
  });

  it('focuses Rename when Cancel closes a form whose name field had the keyboard', () => {
    openRename();
    document.querySelector('.rename-form input').focus();

    controls('cancel')[0].listeners.click();

    assert.equal(document.activeElement.dataset.focus, 'rename');
    assert.equal(document.activeElement.dataset.account, LARGE_ACCOUNT_ID);
  });

  it('focuses Save again when a failed save reopens the form', async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    const pending = pendingSave();
    save.parentNode.listeners.submit({ preventDefault() {} });
    assert.equal(controls('save')[0].disabled, true);
    assert.equal(document.activeElement, document.body);

    await flush(() => pending.reject !== undefined);
    pending.reject(new Error('stopped'));
    await formReopened();

    const focused = document.activeElement;
    assert.equal(focused.dataset.focus, 'save');
    assert.equal(focused.disabled, false);
    showDashboardState(dashboardState());
    assert.equal(document.activeElement.dataset.focus, 'save');
    assert.notEqual(document.activeElement, focused);
  });

  it('puts the caret back when a failed save reopens the name field', async () => {
    openRename();
    const input = document.querySelector('.rename-form input');
    input.focus();
    input.setSelectionRange(1, 4);
    const pending = pendingSave();
    input.parentNode.listeners.submit({ preventDefault() {} });

    await flush(() => pending.reject !== undefined);
    pending.reject(new Error('stopped'));
    await formReopened();

    const restored = document.activeElement;
    assert.equal(restored, document.querySelector('.rename-form input'));
    assert.equal(restored.selectionStart, 1);
    assert.equal(restored.selectionEnd, 4);
  });

  it('does not pull focus back after a blank name is rejected', () => {
    openRename();
    const input = document.querySelector('.rename-form input');
    input.value = '   ';
    input.listeners.input();
    input.focus();
    input.parentNode.listeners.submit({ preventDefault() {} });
    document.activeElement = document.body;

    showDashboardState(dashboardState());

    assert.equal(document.activeElement, document.body);
    assert.equal(controls('save')[0].disabled, false);
  });

  it('does not focus another account when the account being saved leaves the page', async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    const pending = pendingSave();
    save.parentNode.listeners.submit({ preventDefault() {} });
    await flush(() => pending.reject !== undefined);
    const without = dashboardState();
    without.providers[0].accounts = [account('8', 'Account B')];
    showDashboardState(without);

    pending.reject(new Error('stopped'));
    await flush(() => controls('rename').find((button) => button.dataset.account === '8')?.disabled === false);
    document.focused = [];
    controls('rename')
      .find((button) => button.dataset.account === '8')
      .listeners.click();

    assert.equal(document.activeElement.dataset.focus, 'name');
    assert.equal(
      document.focused.some((node) => node.dataset.focus === 'save'),
      false,
    );
  });

  it('leaves focus where the user moved it while the save was pending, when the save fails', async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    const pending = pendingSave();
    save.parentNode.listeners.submit({ preventDefault() {} });
    await flush(() => pending.reject !== undefined);
    const measure = mount('measure');
    measure.focus();

    pending.reject(new Error('stopped'));
    await formReopened();

    assert.equal(document.activeElement, measure);
    controls('cancel')[0].listeners.click();
  });

  it("returns focus to the account's Rename button when a save succeeds", async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    dashboard('', () => ({ status: 200, body: '{"label": "Renamed"}' }));
    save.parentNode.listeners.submit({ preventDefault() {} });

    await settleSave();

    assert.equal(document.activeElement.dataset.focus, 'rename');
    assert.equal(document.activeElement.dataset.account, LARGE_ACCOUNT_ID);
  });

  it('leaves focus where the user moved it while the save was pending, when the save succeeds', async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    let answerSave;
    dashboard(
      '',
      () =>
        new Promise((resolve) => {
          answerSave = () => resolve({ status: 200, body: '{"label": "Renamed"}' });
        }),
    );
    save.parentNode.listeners.submit({ preventDefault() {} });
    await flush(() => answerSave !== undefined);
    const measure = mount('measure');
    measure.focus();

    answerSave();
    await settleSave();

    assert.equal(document.activeElement, measure);
  });

  it('does not focus a later form after a save succeeds', async () => {
    openRename();
    const save = controls('save')[0];
    save.focus();
    dashboard('', (url) =>
      String(url).includes('/api/accounts/')
        ? { status: 200, body: '{"label": "Renamed"}' }
        : Promise.reject(new Error('no follow-up')),
    );
    save.parentNode.listeners.submit({ preventDefault() {} });
    await settleSave();
    document.focused = [];
    click(controls('rename').find((button) => button.dataset.account === LARGE_ACCOUNT_ID));

    assert.equal(
      document.focused.some((node) => node.dataset.focus === 'save'),
      false,
    );
    assert.equal(document.activeElement.dataset.focus, 'name');
  });

  it('leaves focus where it is when nothing inside the cards has it', () => {
    idleCards();

    showDashboardState(dashboardState());

    assert.equal(document.activeElement, document.body);
  });

  it('keeps the selected token bucket focused when the chart is drawn again', async () => {
    await drawTokens(tokenReport());
    const bucket = controls('bucket').find((band) => band.dataset.bucket === '2026-10-06');
    bucket.focus();

    await drawTokens(tokenReport());

    const focused = document.activeElement;
    assert.equal(focused.dataset.focus, 'bucket');
    assert.equal(focused.dataset.bucket, '2026-10-06');
    assert.notEqual(focused, bucket);
  });

  it('drops the bucket focus and its tooltip when that bucket is no longer drawn', async () => {
    const bucket = controls('bucket').find((band) => band.dataset.bucket === '2026-10-06');
    bucket.focus();
    const tooltip = document.getElementById('tooltip');
    tooltip.hidden = false;

    await drawTokens(tokenReport('0'));

    assert.equal(document.activeElement, document.body);
    assert.equal(controls('bucket').length, 0);
    assert.equal(tooltip.hidden, true);
  });

  it('hides the tooltip when the focused bucket is left out of the next chart', async () => {
    await drawTokens(tokenReport());
    const bucket = controls('bucket').find((band) => band.dataset.bucket === '2026-10-06');
    bucket.focus();
    const tooltip = document.getElementById('tooltip');
    tooltip.hidden = false;
    const report = tokenReport();
    report.buckets = report.buckets.slice(0, 1);

    await drawTokens(report);

    assert.equal(tooltip.hidden, true);
    assert.equal(document.activeElement, document.body);
    assert.equal(
      controls('bucket').some((band) => band.dataset.bucket === '2026-10-06'),
      false,
    );
  });

  it('does not move focus from a bucket to an account button', () => {
    showDashboardState(dashboardState());
    return drawTokens(tokenReport()).then(() => {
      const bucket = controls('bucket').find((band) => band.dataset.bucket === '2026-10-05');
      bucket.focus();

      showDashboardState(dashboardState());

      assert.equal(document.activeElement, bucket);
    });
  });

  it('does not move focus from Rename to a chart bucket', async () => {
    idleCards();
    const rename = controls('rename').find((button) => button.dataset.account === '8');
    rename.focus();

    await drawTokens(tokenReport());

    assert.equal(document.activeElement, rename);
    assert.equal(rename.dataset.account, '8');
  });
});
