import { createCipheriv, createHash, createHmac, hkdfSync, randomBytes } from 'node:crypto';

/** A dashboard for the page tests: issues challenges, seals and signs answers, with Node's own primitives. */

export const TOKEN = `synthetic-page-token-${'t'.repeat(22)}`;
export const SIGNED_HEADER = /^Usagedash ([A-Za-z0-9_-]{43,256})\.([A-Za-z0-9_-]{43,256})\.([0-9a-f]{64})$/;
export const sha256 = (data) => createHash('sha256').update(data).digest('hex');
export const EMPTY_BODY_HASH = sha256('');

/** The digest of a request with no body, which is every request the page makes on its own. */
export function digestOf(token, method, target, challenge, nonce) {
  return createHmac('sha256', token)
    .update(`${method}\n${target}\n${EMPTY_BODY_HASH}\n${challenge}\n${nonce}`)
    .digest('hex');
}

/** The signature the dashboard puts on its answer, over the sealed bytes as sent. */
export function answerSignatureOf(token, challenge, nonce, status, body) {
  return createHmac('sha256', token)
    .update(`${challenge}\n${nonce}\n${status}\n${sha256(body)}`)
    .digest('hex');
}

/** An answer sealed as the dashboard seals it: AES-256-GCM under HKDF-SHA256 of the token, bound to the exchange. */
export function sealedAnswerOf(token, challenge, nonce, status, body) {
  const key = hkdfSync(
    'sha256',
    Buffer.from(token),
    Buffer.from('usagedash'),
    Buffer.from('usagedash/sealed-body/v1'),
    32,
  );
  const iv = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', Buffer.from(key), iv);
  cipher.setAAD(Buffer.from(`${challenge}\n${nonce}\n${status}`));
  const ciphertext = Buffer.concat([cipher.update(body, 'utf8'), cipher.final()]);
  return new Uint8Array(Buffer.concat([iv, ciphertext, cipher.getAuthTag()]));
}

/**
 * Installs `globalThis.fetch` as a dashboard at `origin` and records what it was asked.
 *
 * @param {string} origin - What the page is expected to prefix its paths with.
 * @param {(url: string, init?: RequestInit) => { status: number, body: string } | Promise<{ status: number, body: string }>} answer -
 *   The API answer, by request; a rejected promise fails the fetch.
 * @param {{ forge?: boolean, sealWith?: string, replay?: boolean }} [options] - `forge` leaves the signature off
 *   every answer; `sealWith` seals under another token; `replay` records the first exchange and from then on
 *   re-issues its challenge and replays its answer, as a listener could.
 */
export function dashboard(origin, answer, { forge = false, sealWith = TOKEN, replay = false } = {}) {
  const api = [];
  const challengeRequests = [];
  const issued = new Set();
  let recording = null;
  globalThis.fetch = async (url, init) => {
    const text = String(url);
    const authorization = init?.headers?.Authorization ?? null;
    if (text === `${origin}/challenge`) {
      challengeRequests.push({ url: text, authorization });
      const challenge = recording?.challenge ?? randomBytes(32).toString('base64url');
      issued.add(challenge);
      return { ok: true, status: 200, json: async () => ({ challenge }) };
    }
    api.push({ url: text, method: init?.method ?? 'GET', authorization, body: init?.body ?? null });
    const { status, body } = await answer(text, init);
    const parts = SIGNED_HEADER.exec(authorization ?? '');
    const [challenge, nonce] = parts === null ? [null, null] : [parts[1], parts[2]];
    let sent =
      challenge === null ? new Uint8Array(Buffer.from(body)) : sealedAnswerOf(sealWith, challenge, nonce, status, body);
    let signature = forge || challenge === null ? null : answerSignatureOf(TOKEN, challenge, nonce, status, sent);
    if (replay && challenge !== null) {
      if (recording === null) recording = { challenge, sent, signature };
      else ({ sent, signature } = recording);
    }
    return {
      ok: status >= 200 && status < 300,
      status,
      headers: { get: (name) => (name === 'X-Usagedash-Response' ? signature : null) },
      arrayBuffer: async () => sent.buffer.slice(sent.byteOffset, sent.byteOffset + sent.byteLength),
      text: async () => body,
      json: async () => JSON.parse(body),
    };
  };
  return { api, challengeRequests, issued };
}

/**
 * Lets the page's asynchronous work (signing, a challenge round trip) run: with `until`, macrotask turns
 * pass until it holds, up to a bound that fails loudly; without, a fixed few pass, for when nothing should happen.
 *
 * @param {(() => boolean) | null} [until]
 */
export async function flush(until = null) {
  const settle = 25;
  const limit = until === null ? settle : 2000;
  for (let turn = 0; turn < limit; turn += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (until !== null && turn >= settle && until()) return;
  }
  if (until !== null) throw new Error('the page did not reach the awaited state in time');
}
