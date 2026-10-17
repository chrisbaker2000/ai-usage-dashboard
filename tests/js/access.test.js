import assert from 'node:assert/strict';
import { createCipheriv, createDecipheriv, createHash, createHmac, hkdfSync, randomBytes } from 'node:crypto';
import { describe, it } from 'node:test';

import {
  SEALED_CONTENT_TYPE,
  answerSealContext,
  bodyDigest,
  bootstrapToken,
  createAccess,
  forgetStoredToken,
  freshNonce,
  issuedChallenge,
  requestSealContext,
  requestSignature,
  responseSignature,
  seal,
  signatureMessage,
  unseal,
} from '../../src/usagedash/static/access.js';

const STORAGE_KEY = 'usagedash.access';
const TOKEN_GLOBAL = '__USAGEDASH_TOKEN';
const TOKEN = `synthetic-access-token-${'x'.repeat(20)}`;
const OTHER_TOKEN = `synthetic-second-token-${'y'.repeat(20)}`;
const CHALLENGE = `synthetic-challenge-${'c'.repeat(23)}`;
const OTHER_CHALLENGE = `synthetic-challenge-${'d'.repeat(23)}`;
const NONCE = `synthetic-nonce-${'n'.repeat(27)}`;
const OTHER_NONCE = `synthetic-nonce-${'m'.repeat(27)}`;
const ORIGIN = 'http://127.0.0.1:8787';
const SIGNED_HEADER = /^Usagedash ([A-Za-z0-9_-]{43,256})\.([A-Za-z0-9_-]{43,256})\.([0-9a-f]{64})$/;
const RENAME_BODY = '{"label":"Renamed"}';
const STATE_BODY = '{"providers": []}';
const sha256 = (text) => createHash('sha256').update(text).digest('hex');
const EMPTY_BODY_HASH = sha256('');
const SEAL_INFO = 'usagedash/sealed-body/v1';
const NONCE_BYTES = 12;
const TAG_BYTES = 16;

/**
 * @param {string} token
 * @returns {Buffer} The key the dashboard derives: HKDF-SHA256 of the token with the fixed salt and info.
 */
function sealKeyOf(token) {
  return Buffer.from(hkdfSync('sha256', Buffer.from(token), Buffer.from('usagedash'), Buffer.from(SEAL_INFO), 32));
}

/**
 * Seals as the dashboard does, with Node's own AES-256-GCM.
 *
 * @param {string} token
 * @param {string} plaintext
 * @param {string} context
 * @param {Buffer} [nonce]
 * @returns {Uint8Array}
 */
function sealWithReference(token, plaintext, context, nonce = randomBytes(NONCE_BYTES)) {
  const cipher = createCipheriv('aes-256-gcm', sealKeyOf(token), nonce);
  cipher.setAAD(Buffer.from(context));
  const ciphertext = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
  return new Uint8Array(Buffer.concat([nonce, ciphertext, cipher.getAuthTag()]));
}

/**
 * Opens as the dashboard does, with Node's own AES-256-GCM. Throws when the bytes do not open.
 *
 * @param {string} token
 * @param {Uint8Array} sealed
 * @param {string} context
 * @returns {string}
 */
function openWithReference(token, sealed, context) {
  const bytes = Buffer.from(sealed);
  const decipher = createDecipheriv('aes-256-gcm', sealKeyOf(token), bytes.subarray(0, NONCE_BYTES));
  decipher.setAAD(Buffer.from(context));
  decipher.setAuthTag(bytes.subarray(bytes.length - TAG_BYTES));
  return Buffer.concat([
    decipher.update(bytes.subarray(NONCE_BYTES, bytes.length - TAG_BYTES)),
    decipher.final(),
  ]).toString('utf8');
}

function memoryStorage(initial = {}) {
  const items = new Map(Object.entries(initial));
  let reads = 0;
  let writes = 0;
  return {
    getItem: (key) => {
      reads += 1;
      return items.has(key) ? items.get(key) : null;
    },
    setItem: (key, value) => {
      writes += 1;
      items.set(key, value);
    },
    removeItem: (key) => items.delete(key),
    items,
    reads: () => reads,
    writes: () => writes,
  };
}

function brokenStorage() {
  const failing = () => {
    throw new Error('storage is disabled');
  };
  return { getItem: failing, setItem: failing, removeItem: failing };
}

/**
 * @param {{ token?: unknown, storage?: object }} [options] - `token` is what the bootstrap file set, if anything.
 */
function browser({ token, storage = memoryStorage() } = {}) {
  const environment = { localStorage: storage, location: { hash: '', pathname: '/', search: '' }, replaced: [] };
  environment.history = { replaceState: (_state, _title, url) => environment.replaced.push(url) };
  if (token !== undefined) environment[TOKEN_GLOBAL] = token;
  return environment;
}

/** A window whose `localStorage` property throws when read, as a browser that blocks storage can. */
function browserThatBlocksStorage(options = {}) {
  const environment = browser(options);
  Object.defineProperty(environment, 'localStorage', {
    get() {
      throw new Error('The operation is insecure.');
    },
  });
  return environment;
}

/**
 * @param {string} token
 * @param {string} method
 * @param {string} target
 * @param {string} bodyHash
 * @param {string} challenge
 * @returns {string}
 */
function digestOf(token, method, target, bodyHash, challenge, nonce) {
  return createHmac('sha256', token).update(`${method}\n${target}\n${bodyHash}\n${challenge}\n${nonce}`).digest('hex');
}

/**
 * @param {string} token
 * @param {string} challenge
 * @param {number} status
 * @param {string} body
 * @returns {string} The signature the dashboard puts on its answer.
 */
function answerSignatureOf(token, challenge, nonce, status, body) {
  return createHmac('sha256', token)
    .update(`${challenge}\n${nonce}\n${status}\n${sha256(body)}`)
    .digest('hex');
}

/**
 * A dashboard that issues the given challenges in order and records what it was asked.
 *
 * @param {...string} challenges
 * @returns {{ fetch: typeof fetch, asked: () => Array<{ url: string, init: RequestInit | undefined }> }}
 */
function issuing(...challenges) {
  const asked = [];
  const pending = [...challenges];
  return {
    asked: () => asked,
    fetch: async (url, init) => {
      asked.push({ url: String(url), init });
      return { ok: true, status: 200, json: async () => ({ challenge: pending.shift() }) };
    },
  };
}

/**
 * @param {{ headers: Record<string, string>, challenge: string } | null} signed - What `authorize` resolved to.
 * @returns {{ challenge: string, digest: string }}
 */
function partsOf(signed) {
  assert.ok(signed !== null, 'nothing was signed');
  const match = SIGNED_HEADER.exec(signed.headers.Authorization);
  assert.ok(match !== null, `not a signed header: ${signed.headers.Authorization}`);
  assert.equal(match[1], signed.challenge);
  assert.equal(match[2], signed.nonce);
  return { challenge: match[1], nonce: match[2], digest: match[3] };
}

describe('the token the bootstrap file sets', () => {
  it('is held when it is well formed', () => {
    const access = createAccess(browser({ token: TOKEN }));

    assert.equal(access.isSignedIn(), true);
    assert.equal(bootstrapToken(browser({ token: TOKEN })), TOKEN);
  });

  for (const token of [undefined, null, 42, '', 'short', 'x'.repeat(42), `${TOKEN} `, `${TOKEN.slice(0, -1)}é`]) {
    it(`is ignored when it is ${JSON.stringify(token)}, and nothing is asked of the dashboard`, async () => {
      const dashboard = issuing(CHALLENGE);
      const access = createAccess(browser({ token }), { fetch: dashboard.fetch });

      assert.equal(access.isSignedIn(), false);
      assert.equal(await access.authorize('GET', '/api/state'), null);
      assert.deepEqual(dashboard.asked(), []);
      assert.equal(bootstrapToken(browser({ token })), null);
    });
  }

  it('is accepted at either length bound', () => {
    assert.equal(bootstrapToken(browser({ token: 'A'.repeat(43) })), 'A'.repeat(43));
    assert.equal(bootstrapToken(browser({ token: 'z'.repeat(256) })), 'z'.repeat(256));
    assert.equal(bootstrapToken(browser({ token: 'z'.repeat(257) })), null);
  });
});

describe('signing a request', () => {
  const fixedNonce = { nonce: () => NONCE };

  it('asks the dashboard for a challenge with nothing attached, then signs method, target, body, challenge, and its own nonce', async () => {
    const dashboard = issuing(CHALLENGE);
    const access = createAccess(browser({ token: TOKEN }), { fetch: dashboard.fetch, ...fixedNonce });

    const signed = await access.authorize('GET', '/api/state');

    assert.deepEqual(dashboard.asked(), [{ url: '/challenge', init: undefined }]);
    assert.deepEqual(signed, {
      headers: {
        Authorization: `Usagedash ${CHALLENGE}.${NONCE}.${digestOf(TOKEN, 'GET', '/api/state', EMPTY_BODY_HASH, CHALLENGE, NONCE)}`,
      },
      challenge: CHALLENGE,
      nonce: NONCE,
      body: undefined,
    });
  });

  it('draws a fresh well-formed nonce for every request when none is supplied', async () => {
    const access = createAccess(browser({ token: TOKEN }), { fetch: issuing(CHALLENGE, CHALLENGE).fetch });

    const first = partsOf(await access.authorize('GET', '/api/state'));
    const second = partsOf(await access.authorize('GET', '/api/state'));

    assert.notEqual(first.nonce, second.nonce);
    assert.match(first.nonce, /^[A-Za-z0-9_-]{43}$/);
    assert.match(freshNonce(), /^[A-Za-z0-9_-]{43}$/);
    assert.notEqual(freshNonce(), freshNonce());
  });

  it('asks at the dashboard origin it was given', async () => {
    const dashboard = issuing(CHALLENGE);
    const access = createAccess(browser({ token: TOKEN }), { fetch: dashboard.fetch, origin: ORIGIN });

    await access.authorize('GET', '/api/state');

    assert.deepEqual(
      dashboard.asked().map((call) => call.url),
      [`${ORIGIN}/challenge`],
    );
  });

  it('never puts the token in the header or in the challenge request', async () => {
    const dashboard = issuing(CHALLENGE);
    const access = createAccess(browser({ token: TOKEN }), { fetch: dashboard.fetch });

    const { Authorization } = (await access.authorize('GET', '/api/state')).headers;

    assert.equal(Authorization.includes(TOKEN), false);
    assert.equal(Authorization.startsWith('Bearer'), false);
    assert.match(Authorization, SIGNED_HEADER);
    assert.equal(JSON.stringify(dashboard.asked()).includes(TOKEN), false);
  });

  it('signs over the challenge the dashboard issued for that request, a fresh one each time', async () => {
    const dashboard = issuing(CHALLENGE, OTHER_CHALLENGE);
    const access = createAccess(browser({ token: TOKEN }), { fetch: dashboard.fetch, ...fixedNonce });

    const first = partsOf(await access.authorize('GET', '/api/state'));
    const second = partsOf(await access.authorize('GET', '/api/state'));

    assert.equal(first.challenge, CHALLENGE);
    assert.equal(second.challenge, OTHER_CHALLENGE);
    assert.notEqual(first.digest, second.digest);
    assert.equal(dashboard.asked().length, 2);
  });

  it('signs the query as part of the target', async () => {
    const target = '/api/tokens?range=7d&tz=America%2FNew_York';
    const access = createAccess(browser({ token: TOKEN }), { fetch: issuing(CHALLENGE).fetch, ...fixedNonce });

    const { digest } = partsOf(await access.authorize('GET', target));

    assert.equal(digest, digestOf(TOKEN, 'GET', target, EMPTY_BODY_HASH, CHALLENGE, NONCE));
  });

  it('produces a different digest for another method, target, body, challenge, nonce, or token', async () => {
    const digest = async (token, method, target, body, challenge, nonce) =>
      partsOf(
        await createAccess(browser({ token }), { fetch: issuing(challenge).fetch, nonce: () => nonce }).authorize(
          method,
          target,
          body,
        ),
      ).digest;
    const reference = await digest(TOKEN, 'GET', '/api/state', undefined, CHALLENGE, NONCE);

    assert.notEqual(await digest(TOKEN, 'POST', '/api/state', undefined, CHALLENGE, NONCE), reference);
    assert.notEqual(await digest(TOKEN, 'GET', '/api/tokens', undefined, CHALLENGE, NONCE), reference);
    assert.notEqual(await digest(TOKEN, 'GET', '/api/state', RENAME_BODY, CHALLENGE, NONCE), reference);
    assert.notEqual(await digest(TOKEN, 'GET', '/api/state', undefined, OTHER_CHALLENGE, NONCE), reference);
    assert.notEqual(await digest(TOKEN, 'GET', '/api/state', undefined, CHALLENGE, OTHER_NONCE), reference);
    assert.notEqual(await digest(OTHER_TOKEN, 'GET', '/api/state', undefined, CHALLENGE, NONCE), reference);
  });

  it('joins the five parts of the message with line breaks, in the order the server reads them', () => {
    assert.equal(
      signatureMessage('PATCH', '/api/accounts/7', EMPTY_BODY_HASH, CHALLENGE, NONCE),
      `PATCH\n/api/accounts/7\n${EMPTY_BODY_HASH}\n${CHALLENGE}\n${NONCE}`,
    );
  });

  it('computes the same digest as a reference HMAC-SHA256', async () => {
    assert.equal(
      await requestSignature(TOKEN, 'PATCH', '/api/accounts/7', EMPTY_BODY_HASH, CHALLENGE, NONCE),
      digestOf(TOKEN, 'PATCH', '/api/accounts/7', EMPTY_BODY_HASH, CHALLENGE, NONCE),
    );
  });

  it('signs the sealed body exactly as it will be sent, and no body as no bytes', async () => {
    const access = createAccess(browser({ token: TOKEN }), {
      fetch: issuing(CHALLENGE, CHALLENGE).fetch,
      ...fixedNonce,
    });

    const signed = await access.authorize('PATCH', '/api/accounts/7', RENAME_BODY);
    const withBody = partsOf(signed).digest;
    const without = partsOf(await access.authorize('PATCH', '/api/accounts/7')).digest;

    assert.ok(signed.body instanceof Uint8Array);
    assert.equal(signed.headers['Content-Type'], SEALED_CONTENT_TYPE);
    assert.equal(
      withBody,
      digestOf(TOKEN, 'PATCH', '/api/accounts/7', sha256(Buffer.from(signed.body)), CHALLENGE, NONCE),
    );
    assert.equal(without, digestOf(TOKEN, 'PATCH', '/api/accounts/7', EMPTY_BODY_HASH, CHALLENGE, NONCE));
    assert.equal(await bodyDigest(undefined), EMPTY_BODY_HASH);
    assert.equal(await bodyDigest(RENAME_BODY), sha256(RENAME_BODY));
    assert.equal(await bodyDigest('é'), sha256(Buffer.from('é', 'utf8')));
    assert.equal(await bodyDigest(new Uint8Array([1, 2, 3])), sha256(Buffer.from([1, 2, 3])));
  });

  it('refuses to send a body that is not text, and asks for no challenge', async () => {
    const dashboard = issuing(CHALLENGE);
    const access = createAccess(browser({ token: TOKEN }), { fetch: dashboard.fetch });

    await assert.rejects(access.authorize('POST', '/api/refresh', new Uint8Array([1, 2, 3])), TypeError);
    await assert.rejects(bodyDigest({ label: 'Renamed' }), TypeError);
    assert.deepEqual(dashboard.asked(), []);
  });
});

describe('sealing a request body', () => {
  const context = requestSealContext('PATCH', '/api/accounts/7', CHALLENGE, NONCE);

  it('seals the body for this method, target, challenge, and nonce under the key derived from the token', async () => {
    const access = createAccess(browser({ token: TOKEN }), { fetch: issuing(CHALLENGE).fetch, nonce: () => NONCE });

    const signed = await access.authorize('PATCH', '/api/accounts/7', RENAME_BODY);

    assert.equal(openWithReference(TOKEN, signed.body, context), RENAME_BODY);
    assert.equal(Buffer.from(signed.body).includes(Buffer.from('Renamed')), false);
    assert.equal(signed.body.length, NONCE_BYTES + Buffer.byteLength(RENAME_BODY) + TAG_BYTES);
  });

  it('uses a nonce of its own for each seal', async () => {
    const first = await seal(TOKEN, RENAME_BODY, context);
    const second = await seal(TOKEN, RENAME_BODY, context);

    assert.notDeepEqual(first, second);
    assert.notDeepEqual(first.subarray(0, NONCE_BYTES), second.subarray(0, NONCE_BYTES));
    assert.equal(openWithReference(TOKEN, first, context), RENAME_BODY);
    assert.equal(openWithReference(TOKEN, second, context), RENAME_BODY);
  });

  it('seals so that nothing but the same token and the same context opens it', async () => {
    const sealed = await seal(TOKEN, RENAME_BODY, context);
    const elsewhere = [
      requestSealContext('PATCH', '/api/accounts/8', CHALLENGE, NONCE),
      requestSealContext('PATCH', '/api/accounts/7', OTHER_CHALLENGE, NONCE),
      requestSealContext('PATCH', '/api/accounts/7', CHALLENGE, OTHER_NONCE),
      requestSealContext('POST', '/api/accounts/7', CHALLENGE, NONCE),
      answerSealContext(CHALLENGE, NONCE, 200),
    ];

    assert.throws(() => openWithReference(OTHER_TOKEN, sealed, context));
    for (const other of elsewhere) assert.throws(() => openWithReference(TOKEN, sealed, other), other);
  });

  it('builds the contexts one part per line, as the dashboard reads them', () => {
    assert.equal(
      requestSealContext('PATCH', '/api/accounts/7?x=1', CHALLENGE, NONCE),
      `PATCH\n/api/accounts/7?x=1\n${CHALLENGE}\n${NONCE}`,
    );
    assert.equal(answerSealContext(CHALLENGE, NONCE, 404), `${CHALLENGE}\n${NONCE}\n404`);
  });
});

describe("verifying the dashboard's answer", () => {
  const access = () => createAccess(browser({ token: TOKEN }));
  const genuine = answerSignatureOf(TOKEN, CHALLENGE, NONCE, 200, STATE_BODY);

  it('computes the signature the dashboard puts on an answer: challenge, nonce, status, and body digest, keyed by the token', async () => {
    assert.equal(await responseSignature(TOKEN, CHALLENGE, NONCE, 200, STATE_BODY), genuine);
    assert.equal(
      await responseSignature(TOKEN, CHALLENGE, NONCE, 503, ''),
      answerSignatureOf(TOKEN, CHALLENGE, NONCE, 503, ''),
    );
  });

  it("accepts an answer signed by the dashboard for this request's challenge, nonce, status, and body", async () => {
    assert.equal(await access().verifyAnswer(CHALLENGE, NONCE, 200, STATE_BODY, genuine), true);
  });

  for (const [description, challenge, nonce, status, body, signature] of [
    ['carries no signature', CHALLENGE, NONCE, 200, STATE_BODY, null],
    ['carries an empty signature', CHALLENGE, NONCE, 200, STATE_BODY, ''],
    ['is signed for another challenge', OTHER_CHALLENGE, NONCE, 200, STATE_BODY, genuine],
    ['is signed for another nonce', CHALLENGE, OTHER_NONCE, 200, STATE_BODY, genuine],
    ['is signed for another status', CHALLENGE, NONCE, 201, STATE_BODY, genuine],
    ['has a body other than the one signed', CHALLENGE, NONCE, 200, '{"providers": [1]}', genuine],
    [
      'is signed with another token',
      CHALLENGE,
      NONCE,
      200,
      STATE_BODY,
      answerSignatureOf(OTHER_TOKEN, CHALLENGE, NONCE, 200, STATE_BODY),
    ],
    [
      'has one character of its signature changed',
      CHALLENGE,
      NONCE,
      200,
      STATE_BODY,
      `${genuine.slice(0, -1)}${genuine.endsWith('0') ? '1' : '0'}`,
    ],
    ['has a signature of another length', CHALLENGE, NONCE, 200, STATE_BODY, genuine.slice(1)],
  ]) {
    it(`refuses an answer that ${description}`, async () => {
      assert.equal(await access().verifyAnswer(challenge, nonce, status, body, signature), false);
    });
  }

  it('verifies nothing when no token is held', async () => {
    assert.equal(await createAccess(browser()).verifyAnswer(CHALLENGE, NONCE, 200, STATE_BODY, genuine), false);
  });

  it('verifies the signature over the sealed bytes exactly as received', async () => {
    const sealed = sealWithReference(TOKEN, STATE_BODY, answerSealContext(CHALLENGE, NONCE, 200));
    const overSealed = answerSignatureOf(TOKEN, CHALLENGE, NONCE, 200, Buffer.from(sealed));

    assert.equal(await access().verifyAnswer(CHALLENGE, NONCE, 200, sealed, overSealed), true);
    assert.equal(await access().verifyAnswer(CHALLENGE, NONCE, 200, sealed, genuine), false);
    assert.equal(await access().verifyAnswer(CHALLENGE, NONCE, 200, sealed.subarray(1), overSealed), false);
  });
});

describe("opening the dashboard's answer", () => {
  const access = () => createAccess(browser({ token: TOKEN }));
  const context = answerSealContext(CHALLENGE, NONCE, 200);

  it('opens an answer the dashboard sealed for this challenge, nonce, and status', async () => {
    const sealed = sealWithReference(TOKEN, STATE_BODY, context);

    assert.equal(await access().openAnswer(CHALLENGE, NONCE, 200, sealed), STATE_BODY);
    assert.equal(await unseal(TOKEN, sealed, context), STATE_BODY);
    assert.equal(await unseal(TOKEN, sealWithReference(TOKEN, '', context), context), '');
  });

  for (const [description, challenge, nonce, status, sealed] of [
    [
      'was sealed for another challenge',
      OTHER_CHALLENGE,
      NONCE,
      200,
      () => sealWithReference(TOKEN, STATE_BODY, context),
    ],
    ['was sealed for another nonce', CHALLENGE, OTHER_NONCE, 200, () => sealWithReference(TOKEN, STATE_BODY, context)],
    ['was sealed for another status', CHALLENGE, NONCE, 201, () => sealWithReference(TOKEN, STATE_BODY, context)],
    [
      'was sealed under another token',
      CHALLENGE,
      NONCE,
      200,
      () => sealWithReference(OTHER_TOKEN, STATE_BODY, context),
    ],
    [
      'has one byte of its ciphertext changed',
      CHALLENGE,
      NONCE,
      200,
      () => {
        const altered = sealWithReference(TOKEN, STATE_BODY, context);
        altered[NONCE_BYTES] ^= 1;
        return altered;
      },
    ],
    [
      'is too short to hold a nonce and a tag',
      CHALLENGE,
      NONCE,
      200,
      () => new Uint8Array(NONCE_BYTES + TAG_BYTES - 1),
    ],
    ['is not bytes at all', CHALLENGE, NONCE, 200, () => STATE_BODY],
  ]) {
    it(`refuses an answer that ${description}`, async () => {
      await assert.rejects(access().openAnswer(challenge, nonce, status, sealed()), /could not be opened/);
    });
  }

  it('opens nothing when no token is held', async () => {
    const sealed = sealWithReference(TOKEN, STATE_BODY, context);

    await assert.rejects(createAccess(browser()).openAnswer(CHALLENGE, NONCE, 200, sealed), /no token/);
  });
});

describe('an answer recorded from an earlier exchange', () => {
  /** What a listener can record: the dashboard's sealed, signed answer over CHALLENGE and NONCE. */
  const recorded = () => {
    const sealed = sealWithReference(TOKEN, STATE_BODY, answerSealContext(CHALLENGE, NONCE, 200));
    return { sealed, signature: answerSignatureOf(TOKEN, CHALLENGE, NONCE, 200, Buffer.from(sealed)) };
  };

  it('is refused by a later request made over the same re-issued challenge, because its nonce is new', async () => {
    const { sealed, signature } = recorded();
    const access = createAccess(browser({ token: TOKEN }), { fetch: issuing(CHALLENGE).fetch });
    const later = await access.authorize('GET', '/api/state');

    assert.equal(later.challenge, CHALLENGE);
    assert.notEqual(later.nonce, NONCE);
    assert.equal(await access.verifyAnswer(later.challenge, later.nonce, 200, sealed, signature), false);
    await assert.rejects(access.openAnswer(later.challenge, later.nonce, 200, sealed), /could not be opened/);
  });

  it('is refused by a freshly opened page as well: nothing about freshness is remembered, each request is new', async () => {
    const { sealed, signature } = recorded();
    const fresh = createAccess(browser({ token: TOKEN }), { fetch: issuing(CHALLENGE).fetch });
    const later = await fresh.authorize('PATCH', '/api/accounts/7', RENAME_BODY);

    assert.equal(await fresh.verifyAnswer(later.challenge, later.nonce, 200, sealed, signature), false);
  });

  it('still fits the request it answered', async () => {
    const { sealed, signature } = recorded();
    const access = createAccess(browser({ token: TOKEN }), { fetch: issuing(CHALLENGE).fetch, nonce: () => NONCE });
    const same = await access.authorize('GET', '/api/state');

    assert.equal(await access.verifyAnswer(same.challenge, same.nonce, 200, sealed, signature), true);
    assert.equal(await access.openAnswer(same.challenge, same.nonce, 200, sealed), STATE_BODY);
  });
});

describe('a dashboard that does not issue a challenge', () => {
  const answering = (response) => async () => response;

  it('is not sent a request when it cannot be reached', async () => {
    const access = createAccess(browser({ token: TOKEN }), {
      fetch: async () => {
        throw new Error('offline');
      },
    });

    await assert.rejects(access.authorize('GET', '/api/state'), /offline/);
  });

  it('is not sent a request when it answers with a failure', async () => {
    const access = createAccess(browser({ token: TOKEN }), {
      fetch: answering({ ok: false, status: 404, json: async () => ({}) }),
    });

    await assert.rejects(access.authorize('GET', '/api/state'), /did not issue a challenge \(404\)/);
  });

  for (const body of [{}, { challenge: 42 }, { challenge: 'short' }, { challenge: `${CHALLENGE}!` }, null, 'text']) {
    it(`is not sent a request when it answers ${JSON.stringify(body)}`, async () => {
      const fetchImpl = answering({ ok: true, status: 200, json: async () => body });

      await assert.rejects(issuedChallenge(fetchImpl, ''), /did not issue a challenge/);
      await assert.rejects(
        createAccess(browser({ token: TOKEN }), { fetch: fetchImpl }).authorize('GET', '/api/state'),
        /did not issue a challenge/,
      );
    });
  }

  it('is not sent a request when its answer is not JSON', async () => {
    const fetchImpl = answering({
      ok: true,
      status: 200,
      json: async () => {
        throw new SyntaxError('not JSON');
      },
    });

    await assert.rejects(issuedChallenge(fetchImpl, ''), SyntaxError);
  });

  it('returns the challenge when the answer is one', async () => {
    assert.equal(await issuedChallenge(issuing(CHALLENGE).fetch, ORIGIN), CHALLENGE);
  });
});

describe('a token an earlier page stored', () => {
  it('is removed when the page starts, and is never read as a credential', () => {
    const storage = memoryStorage({ [STORAGE_KEY]: TOKEN });

    const access = createAccess(browser({ storage }));

    assert.equal(storage.items.has(STORAGE_KEY), false);
    assert.equal(storage.reads(), 0);
    assert.equal(access.isSignedIn(), false);
  });

  it('is removed when the bootstrap token is held too, and the bootstrap token is not written in its place', () => {
    const storage = memoryStorage({ [STORAGE_KEY]: OTHER_TOKEN });

    const access = createAccess(browser({ token: TOKEN, storage }));

    assert.equal(storage.items.has(STORAGE_KEY), false);
    assert.equal(storage.writes(), 0);
    assert.equal(access.isSignedIn(), true);
  });

  it('leaves other stored items alone', () => {
    const storage = memoryStorage({ [STORAGE_KEY]: TOKEN, 'usagedash.range': '7d' });

    createAccess(browser({ storage }));

    assert.deepEqual([...storage.items.keys()], ['usagedash.range']);
  });

  it('reports that nothing could be removed when storage is unavailable, and the page still signs in', async () => {
    assert.equal(forgetStoredToken(browser({ storage: brokenStorage() })), false);

    const access = createAccess(browser({ token: TOKEN, storage: brokenStorage() }), {
      fetch: issuing(CHALLENGE).fetch,
    });

    assert.equal(access.isSignedIn(), true);
    assert.match((await access.authorize('GET', '/api/state')).headers.Authorization, SIGNED_HEADER);
  });

  it('does not stop the page when reading the storage property itself throws', () => {
    assert.equal(forgetStoredToken(browserThatBlocksStorage()), false);

    const access = createAccess(browserThatBlocksStorage({ token: TOKEN }));

    assert.equal(access.isSignedIn(), true);
  });

  it('reports a removal that storage accepted', () => {
    assert.equal(forgetStoredToken(browser()), true);
  });
});

describe('the address bar', () => {
  it('is not read: a fragment that carries an access parameter does not sign the page in (hand-off by fragment was removed)', async () => {
    const environment = browser();
    environment.location.hash = `#access=${TOKEN}`;

    const access = createAccess(environment);

    assert.equal(access.isSignedIn(), false);
    assert.equal(await access.authorize('GET', '/api/state'), null);
    assert.deepEqual(environment.replaced, []);
  });
});
