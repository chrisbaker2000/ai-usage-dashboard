import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

/**
 * A document that records the attributes and children `renderWindow` writes. The page is not started:
 * `#token-body` is absent, so importing the page does not poll.
 */
function element(tag) {
  const node = {
    tag,
    className: '',
    dataset: {},
    attrs: {},
    children: [],
    style: { setProperty() {} },
    setAttribute(name, value) {
      node.attrs[name] = value;
    },
    append(...nodes) {
      node.children.push(...nodes);
    },
    addEventListener() {},
  };
  return node;
}

globalThis.window = {
  location: { hash: '', pathname: '/', search: '' },
  history: { replaceState() {} },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  addEventListener() {},
};
globalThis.fetch = () => Promise.reject(new Error('fetch during import'));
globalThis.document = {
  getElementById: () => null,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: (tag) => element(tag),
  createElementNS: (_namespace, tag) => element(tag),
};

const { renderWindow } = await import('../../src/usagedash/static/app.js');

const NOW = Date.parse('2026-10-06T00:00:00Z');

function quotaWindow(effectivePercent, overrides = {}) {
  return {
    label: '5 hour',
    key: '5h',
    effective_percent: effectivePercent,
    used_percent: effectivePercent,
    state: 'critical',
    resets_at: '2026-10-06T05:00:00Z',
    reset_passed: false,
    as_of: '2026-10-06T00:00:00Z',
    freshness: 'live',
    window_seconds: 5 * 60 * 60,
    pace_exhausts_at: null,
    ...overrides,
  };
}

function texts(node) {
  if (typeof node === 'string') return [node];
  const own = typeof node.textContent === 'string' && node.textContent !== '' ? [node.textContent] : [];
  return [...own, ...node.children.flatMap(texts)];
}

function meter(row) {
  const found = [];
  const visit = (node) => {
    if (typeof node === 'string') return;
    if (node.className === 'meter') found.push(node);
    node.children.forEach(visit);
  };
  visit(row);
  assert.equal(found.length, 1);
  return found[0];
}

describe('a quota window near the limit', () => {
  it('says a reading just under the limit is under 100, and the meter stays under its maximum', () => {
    const row = renderWindow(quotaWindow(99.9), [], NOW);
    assert.ok(texts(row).includes('Under 100% used'));
    assert.equal(meter(row).attrs['aria-valuenow'], '99.9');
    assert.equal(meter(row).attrs['aria-valuemax'], '100');
    assert.equal(meter(row).attrs['aria-valuetext'], '99.9% used');
  });

  it('says a reading just over the limit is over 100, and the meter does not pass its maximum', () => {
    const row = renderWindow(quotaWindow(100.4), [], NOW);
    assert.ok(texts(row).includes('Over 100% used'));
    assert.equal(meter(row).attrs['aria-valuenow'], '100');
    assert.equal(meter(row).attrs['aria-valuetext'], '100.4% used');
  });

  it('shows a large overrun as its rounded percent and clamps the meter to 100', () => {
    const row = renderWindow(quotaWindow(340.5, { state: 'exhausted' }), [], NOW);
    assert.ok(texts(row).includes('341% used'));
    assert.equal(meter(row).attrs['aria-valuenow'], '100');
    assert.equal(meter(row).attrs['aria-valuetext'], '340.5% used');
    assert.equal(Number(meter(row).attrs['aria-valuenow']) <= Number(meter(row).attrs['aria-valuemax']), true);
  });

  it('shows an exactly full window as 100', () => {
    const row = renderWindow(quotaWindow(100, { state: 'exhausted' }), [], NOW);
    assert.ok(texts(row).includes('100% used'));
    assert.equal(meter(row).attrs['aria-valuenow'], '100');
    assert.equal(meter(row).attrs['aria-valuetext'], '100% used');
  });

  it('uses the same side-of-full reading for the percent from before a reset', () => {
    const row = renderWindow(quotaWindow(0, { state: 'ok', reset_passed: true, used_percent: 100.4 }), [], NOW);
    assert.ok(texts(row).some((text) => text.includes('Was Over 100% before the reset')));
  });
});
