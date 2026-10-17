import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { SIGNED_HEADER, TOKEN, dashboard, digestOf, flush } from './fakes/dashboard.js';

const OTHER_TOKEN = `synthetic-other-token-${'z'.repeat(21)}`;
const ORIGIN = 'http://127.0.0.1:8787';

/**
 * @param {{ token?: string, origin?: string }} globals - What the bootstrap file would set on `window`.
 * @returns {{ loadErrors: () => string[] }} A reader of what the page put in its error banner.
 */
function installDocument({ token, origin } = {}) {
  /** @type {Map<string, ReturnType<typeof element>>} */
  const nodes = new Map();
  const element = () => ({
    className: '',
    textContent: '',
    dataset: {},
    hidden: false,
    children: [],
    classList: { add() {}, remove() {}, contains: () => false },
    setAttribute() {},
    addEventListener() {},
    append() {},
    replaceChildren(...children) {
      this.children = children;
    },
  });
  for (const id of ['token-body', 'measure', 'refresh', 'load-error', 'updated']) nodes.set(id, element());
  globalThis.document = {
    body: element(),
    getElementById: (id) => nodes.get(id) ?? null,
    querySelectorAll: () => [],
    querySelector: () => null,
    createElement: () => element(),
    createElementNS: () => element(),
  };
  globalThis.window = {
    location: { hash: '', pathname: '/', search: '' },
    history: { replaceState() {} },
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    addEventListener() {},
    innerWidth: 1200,
    innerHeight: 800,
  };
  if (token !== undefined) globalThis.window.__USAGEDASH_TOKEN = token;
  if (origin !== undefined) globalThis.window.__USAGEDASH_ORIGIN = origin;
  return { loadErrors: () => nodes.get('load-error').children.map((child) => child.textContent) };
}

/**
 * Every API call carries a signature over its own method and target and over a challenge this dashboard
 * issued, each challenge used once; the challenge requests carried nothing.
 */
function assertSignedForThemselves(board, origin) {
  const used = new Set();
  for (const call of board.api) {
    assert.ok(call.url.startsWith(origin), `${call.url} does not start with ${JSON.stringify(origin)}`);
    const target = call.url.slice(origin.length);
    const parts = SIGNED_HEADER.exec(call.authorization ?? '');
    assert.ok(parts !== null, `${call.method} ${call.url} is not signed: ${call.authorization}`);
    const [, challenge, nonce, digest] = parts;
    assert.ok(board.issued.has(challenge), 'signed over a challenge the dashboard did not issue');
    assert.equal(used.has(challenge), false, 'a challenge was used twice');
    assert.equal(used.has(nonce), false, 'a nonce was used twice');
    used.add(challenge).add(nonce);
    assert.equal(digest, digestOf(TOKEN, call.method, target, challenge, nonce));
    assert.equal(call.authorization.includes(TOKEN), false);
    assert.equal(call.url.includes(TOKEN), false);
  }
  assert.ok(board.challengeRequests.every((request) => request.authorization === null));
  assert.equal(board.challengeRequests.length, board.api.length);
}

const serverError = () => ({ status: 500, body: '{"detail": "synthetic"}' });

describe('the page signs every request for exactly that request', { concurrency: false }, () => {
  it('asks the dashboard origin for a challenge, then signs its own method and target over it, never sending the token', async () => {
    const page = installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, serverError);
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=origin');
    await flush(() => page.loadErrors().some((message) => message.includes('responded 500')));

    assert.ok(board.api.length >= 2);
    assert.ok(board.api.every((call) => call.url.startsWith(`${ORIGIN}/api/`)));
    assertSignedForThemselves(board, ORIGIN);
    assert.equal(
      board.api.some((call) => call.authorization !== null && call.authorization.startsWith('Bearer')),
      false,
    );
    assert.ok(page.loadErrors().some((message) => message.includes('responded 500')));
  });

  it('signs a later poll over a fresh challenge', async () => {
    installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, serverError);
    const scheduled = [];
    globalThis.setInterval = (callback) => {
      scheduled.push(callback);
      return scheduled.length;
    };

    await import('../../src/usagedash/static/app.js?signed=again');
    await flush(() => board.api.length >= 2);
    const firstChallenges = new Set(board.api.map((call) => SIGNED_HEADER.exec(call.authorization)[1]));
    assert.equal(firstChallenges.size, board.api.length);
    assert.equal(scheduled.length, 2);

    board.api.length = 0;
    board.challengeRequests.length = 0;
    scheduled[0]();
    await flush(() => board.api.length >= 1);

    assert.equal(board.api.length, 1);
    assertSignedForThemselves(board, ORIGIN);
    assert.equal(firstChallenges.has(SIGNED_HEADER.exec(board.api[0].authorization)[1]), false);
  });

  it('uses relative paths when the page is served by the dashboard itself and no origin is set', async () => {
    installDocument({ token: TOKEN });
    const board = dashboard('', serverError);
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=relative');
    await flush(() => board.api.length >= 2);

    assert.ok(board.api.length >= 2);
    assert.ok(board.api.every((call) => call.url.startsWith('/api/')));
    assertSignedForThemselves(board, '');
  });

  it("does not use an answer that does not carry the dashboard's own signature, and says so", async () => {
    const page = installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, () => ({ status: 200, body: '{"providers": []}' }), { forge: true });
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=forged');
    await flush(() => page.loadErrors().some((message) => message.includes('did not come from')));

    assert.ok(board.api.length >= 2);
    assert.ok(page.loadErrors().some((message) => message.includes('did not come from the dashboard (200)')));
  });

  it("opens the dashboard's sealed answer and uses it", async () => {
    const page = installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, () => ({ status: 200, body: '{"providers": []}' }));
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=sealed');
    await flush(() => board.api.length >= 2);
    await flush();

    assert.ok(board.api.length >= 2);
    assert.equal(
      page.loadErrors().some((message) => /did not come from|could not be opened/.test(message)),
      false,
    );
    assert.ok(board.api.every((call) => call.body === null || call.body === undefined));
  });

  it('does not use an answer a listener recorded earlier and replays behind the same re-issued challenge', async () => {
    const page = installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, () => ({ status: 200, body: '{"providers": []}' }), { replay: true });
    const scheduled = [];
    globalThis.setInterval = (callback) => {
      scheduled.push(callback);
      return scheduled.length;
    };

    await import('../../src/usagedash/static/app.js?signed=replay');
    await flush(() => board.api.length >= 2);
    const firstErrors = page.loadErrors().filter((message) => message.includes('did not come from'));
    board.api.length = 0;
    scheduled[0]();
    await flush(() => page.loadErrors().length > firstErrors.length);

    assert.ok(board.api.length >= 1);
    assert.ok(page.loadErrors().some((message) => message.includes('did not come from the dashboard (200)')));
    assert.ok(page.loadErrors().length > firstErrors.length);
  });

  it('does not use an answer it cannot open, and says so', async () => {
    const page = installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, () => ({ status: 200, body: '{"providers": []}' }), { sealWith: OTHER_TOKEN });
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=unopenable');
    await flush(() => page.loadErrors().some((message) => message.includes('could not be opened')));

    assert.ok(board.api.length >= 2);
    assert.ok(page.loadErrors().some((message) => message.includes('could not be opened (200)')));
  });

  it('sends no request at all when the dashboard issues no challenge, and says so', async () => {
    const page = installDocument({ token: TOKEN, origin: ORIGIN });
    const board = dashboard(ORIGIN, serverError);
    globalThis.fetch = async (url) => {
      board.challengeRequests.push({ url: String(url), authorization: null });
      return { ok: false, status: 404, json: async () => ({}) };
    };
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=nochallenge');
    await flush();

    assert.equal(board.api.length, 0);
    assert.ok(board.challengeRequests.length >= 2);
    assert.ok(page.loadErrors().some((message) => message.includes('did not issue a challenge')));
  });

  it('sends nothing at all without a token, whatever the listener would answer, and tells the user to sign in', async () => {
    const page = installDocument({ origin: ORIGIN });
    const board = dashboard(ORIGIN, () => ({ status: 200, body: '{"providers": [{"fabricated": true}]}' }), {
      forge: true,
    });
    globalThis.setInterval = () => 0;

    await import('../../src/usagedash/static/app.js?signed=none');
    await flush();

    assert.equal(board.api.length, 0);
    assert.equal(board.challengeRequests.length, 0);
    assert.ok(page.loadErrors().some((message) => message.includes('not signed in')));
  });
});
