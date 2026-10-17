import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { latestOnly } from '../../src/usagedash/static/latest.js';

/** A load whose requests are answered by the test, in whatever order it chooses. */
function controllableLoad() {
  const pending = [];
  const load = () => new Promise((resolve, reject) => pending.push({ resolve, reject }));
  return { load, pending };
}

describe('overlapping loads', () => {
  it('shows only the newest call when an older request is answered last', async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const run = latestOnly(load, (result) => shown.push(result));

    const older = run();
    const newer = run();
    pending[1].resolve('24h report');
    pending[0].resolve('30d report');

    assert.deepEqual(await Promise.all([older, newer]), [false, true]);
    assert.deepEqual(shown, ['24h report']);
  });

  it('shows the newest call when requests are answered in the order they were made', async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const run = latestOnly(load, (result) => shown.push(result));

    const older = run();
    const newer = run();
    pending[0].resolve('first');
    pending[1].resolve('second');

    assert.deepEqual(await Promise.all([older, newer]), [false, true]);
    assert.deepEqual(shown, ['second']);
  });

  it('shows every call that finishes before the next one starts', async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const run = latestOnly(load, (result) => shown.push(result));

    const first = run();
    pending[0].resolve('first');
    assert.equal(await first, true);
    const second = run();
    pending[1].resolve('second');
    assert.equal(await second, true);

    assert.deepEqual(shown, ['first', 'second']);
  });
});

describe('failed loads', () => {
  it('reports the failure of the newest call', async () => {
    const { load, pending } = controllableLoad();
    const run = latestOnly(load, () => assert.fail('a failed load must not be shown'));

    const call = run();
    pending[0].reject(new Error('server unreachable'));

    await assert.rejects(call, /server unreachable/);
  });

  it('does not report the failure of a call that a newer one superseded', async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const run = latestOnly(load, (result) => shown.push(result));

    const older = run();
    const newer = run();
    pending[1].resolve('current');
    pending[0].reject(new Error('stale failure'));

    assert.deepEqual(await Promise.all([older, newer]), [false, true]);
    assert.deepEqual(shown, ['current']);
  });

  it('does not show an older result after the newest call failed', async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const run = latestOnly(load, (result) => shown.push(result));

    const older = run();
    const newer = run();
    pending[1].reject(new Error('newest failed'));
    await assert.rejects(newer, /newest failed/);
    pending[0].resolve('stale success');

    assert.equal(await older, false);
    assert.deepEqual(shown, []);
  });
});
