import assert from 'node:assert/strict';
import { before, describe, it } from 'node:test';

import { TOKEN, dashboard, flush } from './fakes/dashboard.js';

/**
 * A document with only the nodes the token report touches. The page is not started: `#token-body` is
 * absent until the tests ask for it, which is after this module has been imported.
 */
const elements = new Map();

function element(id) {
  const classes = new Set();
  const node = {
    id,
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
    },
    className: '',
    dataset: {},
    hidden: false,
    children: [],
    attributes: {},
    setAttribute(name, value) {
      node.attributes[name] = value;
    },
    append(...nodes) {
      node.children.push(...nodes);
    },
    replaceChildren(...nodes) {
      node.children = nodes;
    },
    addEventListener() {},
    onText: null,
    _text: '',
  };
  Object.defineProperty(node, 'textContent', {
    get() {
      return node._text;
    },
    set(value) {
      node._text = value;
      node.onText?.();
    },
  });
  return node;
}

function mount(id) {
  const node = element(id);
  elements.set(id, node);
  return node;
}

globalThis.window = {
  location: { hash: '', pathname: '/', search: '' },
  history: { replaceState() {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  addEventListener() {},
  __USAGEDASH_TOKEN: TOKEN,
};

const importFetches = [];
globalThis.fetch = (...args) => {
  importFetches.push(args);
  return Promise.reject(new Error('fetch during import'));
};

globalThis.document = {
  getElementById: (id) => elements.get(id) ?? null,
  querySelector: (selector) => elements.get(selector) ?? null,
  querySelectorAll: () => [],
  createElement: (tag) => element(tag),
  createElementNS: (_namespace, tag) => element(tag),
};

const { showTokenReport } = await import('../../src/usagedash/static/app.js');

const EMPTY_COSTS = {
  priced_on: '2026-10-08',
  currency: 'USD',
  unit: 'microdollar',
  note: 'Published standard API rates as of 2026-10-08. This is not a subscription charge.',
  accounts: [],
};

const EMPTY_REPORT = {
  range: '7d',
  bucket: 'day',
  timezone: 'UTC',
  buckets: [],
  series: [],
  models: [],
  provider_totals: [],
  costs: EMPTY_COSTS,
};

const okReport = () => ({ status: 200, body: JSON.stringify(EMPTY_REPORT) });
const failedReport = () => ({ status: 500, body: '{}' });

describe('importing the page', () => {
  it('does not start polling when the page markup is absent', () => {
    assert.equal(importFetches.length, 0);
  });
});

describe('the refreshing indicator', () => {
  /** @type {ReturnType<typeof element>} */
  let body;
  /** @type {ReturnType<typeof element>} */
  let chartTitle;

  before(() => {
    body = mount('token-body');
    mount('token-tiles');
    mount('legend');
    mount('chart');
    chartTitle = mount('chart-title');
    mount('tooltip');
    mount('#token-table tbody');
    mount('#model-table tbody');
    mount('cost-note');
    mount('#cost-table tbody');
    mount('#cost-model-table tbody');
  });

  it('takes the indicator down when the only load finishes', async () => {
    chartTitle.onText = null;
    dashboard('', okReport);

    assert.equal(await showTokenReport(), true);
    assert.equal(body.classList.contains('is-refreshing'), false);
  });

  it('leaves the indicator up when an older load finishes before the newer one', async () => {
    chartTitle.onText = null;
    let releaseOlder;
    let releaseNewer;
    let calls = 0;
    dashboard('', () => {
      calls += 1;
      if (calls === 1)
        return new Promise((resolve) => {
          releaseOlder = () => resolve(okReport());
        });
      return new Promise((resolve) => {
        releaseNewer = () => resolve(okReport());
      });
    });

    // Why in turn: two loads started together reach the dashboard in either order.
    const older = showTokenReport();
    await flush(() => calls === 1);
    const newer = showTokenReport();
    await flush(() => calls === 2);
    releaseOlder();

    assert.equal(await older, false);
    assert.equal(body.classList.contains('is-refreshing'), true);
    releaseNewer();
    assert.equal(await newer, true);
    assert.equal(body.classList.contains('is-refreshing'), false);
  });

  it('does not let an older load that already has its report clear a newer load', async () => {
    let releaseNewer;
    let calls = 0;
    /** @type {Promise<boolean> | null} */
    let newer = null;
    chartTitle.onText = () => {
      if (newer === null) newer = showTokenReport();
    };
    dashboard('', () => {
      calls += 1;
      if (calls === 1) return okReport();
      return new Promise((resolve) => {
        releaseNewer = () => resolve(okReport());
      });
    });

    assert.equal(await showTokenReport(), true);
    await flush(() => calls >= 2);
    assert.equal(calls, 2);
    assert.equal(body.classList.contains('is-refreshing'), true);

    releaseNewer();
    assert.equal(await newer, true);
    assert.equal(body.classList.contains('is-refreshing'), false);
  });

  it('takes the indicator down when the newest load fails', async () => {
    chartTitle.onText = null;
    dashboard('', failedReport);

    await assert.rejects(showTokenReport(), /\/api\/tokens\?range=7d.*responded 500/);
    assert.equal(body.classList.contains('is-refreshing'), false);
  });

  it('leaves the indicator up when an older load fails and a newer one is still loading', async () => {
    chartTitle.onText = null;
    let failOlder;
    let releaseNewer;
    let calls = 0;
    dashboard('', () => {
      calls += 1;
      if (calls === 1)
        return new Promise((resolve) => {
          failOlder = () => resolve(failedReport());
        });
      return new Promise((resolve) => {
        releaseNewer = () => resolve(okReport());
      });
    });

    const older = showTokenReport();
    await flush(() => calls === 1);
    const newer = showTokenReport();
    await flush(() => calls === 2);
    failOlder();

    assert.equal(await older, false);
    assert.equal(body.classList.contains('is-refreshing'), true);
    releaseNewer();
    assert.equal(await newer, true);
    assert.equal(body.classList.contains('is-refreshing'), false);
  });
});

function countText(total = 0) {
  return {
    input: String(total),
    cache_read: '0',
    cache_write: '0',
    output: '0',
    reasoning: '0',
    requests: '0',
    total: String(total),
  };
}

function cellLabel(cell) {
  const parts = [];
  const walk = (node) => {
    if (node.textContent) parts.push(node.textContent);
    node.children.forEach(walk);
  };
  walk(cell);
  return parts.join(' ');
}

describe('the retail cost section', () => {
  before(() => {
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

  it('shows no all-accounts row when the range has no usage', async () => {
    dashboard('', okReport);

    assert.equal(await showTokenReport(), true);
    assert.equal(elements.get('#cost-table tbody').children.length, 0);
    assert.equal(elements.get('#cost-model-table tbody').children.length, 0);
    assert.match(elements.get('cost-note').textContent, /not a subscription charge/);
  });

  it('rounds each account to the nearest dollar and does not reprice it from its model rows', async () => {
    const report = {
      range: '7d',
      bucket: 'day',
      timezone: 'UTC',
      buckets: [],
      series: [
        {
          provider: 'claude',
          account_id: '7',
          label: 'Work',
          color_slot: 1,
          attributed: true,
          points: [],
          totals: countText(),
        },
        {
          provider: 'grok',
          account_id: '8',
          label: 'Standing',
          color_slot: 2,
          attributed: true,
          points: [],
          totals: countText(),
        },
      ],
      models: [],
      provider_totals: [
        { provider: 'claude', display_name: 'Claude', totals: countText() },
        { provider: 'grok', display_name: 'Grok', totals: countText() },
      ],
      costs: {
        priced_on: '2026-10-08',
        currency: 'USD',
        unit: 'microdollar',
        note: 'Priced for a short prompt. grok-4.7-build is priced as grok-4.7. This is not a subscription charge.',
        accounts: [
          {
            provider: 'claude',
            account_id: '7',
            priced_microdollars: '1500000',
            unpriced_tokens: '0',
            models: [{ model: 'claude-opus-4-7', priced: true, microdollars: '1000000', unpriced_tokens: '0' }],
          },
          {
            provider: 'grok',
            account_id: '8',
            priced_microdollars: '500000',
            unpriced_tokens: '1000',
            models: [{ model: 'grok-4.7-build', priced: false, microdollars: null, unpriced_tokens: '1000' }],
          },
        ],
      },
    };
    dashboard('', () => ({ status: 200, body: JSON.stringify(report) }));

    assert.equal(await showTokenReport(), true);

    const accountRows = elements.get('#cost-table tbody').children.map((tableRow) => tableRow.children.map(cellLabel));
    assert.deepEqual(accountRows, [
      ['Work Claude', '$2', '0'],
      ['Standing Grok', '$1', '1,000'],
      ['All accounts', '$3', '1,000'],
    ]);
    const footerLabel = elements.get('#cost-table tbody').children[2].children[0];
    assert.equal(footerLabel.id, 'th');
    assert.equal(footerLabel.attributes.scope, 'row');

    const modelRows = elements.get('#cost-model-table tbody').children;
    assert.deepEqual(
      modelRows.map((tableRow) => tableRow.children.map(cellLabel)),
      [
        ['Work Claude', 'claude-opus-4-7', '$1', '0'],
        ['Standing Grok', 'grok-4.7-build', 'No rate', '1,000'],
      ],
    );
    assert.equal(modelRows[1].children[2].className, 'numeric muted');
    assert.equal(elements.get('cost-note').textContent, report.costs.note);
  });
});
