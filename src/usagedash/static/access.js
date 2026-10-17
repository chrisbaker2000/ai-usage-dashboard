/**
 * The page's side of signing in.
 *
 * The token reaches the page as a global the owner-only file `usagedash open` writes sets before this module
 * runs; it is never stored, never put in a URL, never sent. For each request the page asks the dashboard for a
 * challenge, adds a nonce of its own, and sends `Authorization: Usagedash <challenge>.<nonce>.<digest>`, the
 * digest being the HMAC-SHA256 of the method, the target, a SHA-256 of the body, the challenge, and the nonce,
 * keyed by the token. The dashboard signs its answer over the challenge, the nonce, the status, and the body
 * (`X-Usagedash-Response`), and the page uses an answer only when that signature is its own, so a process that
 * merely holds the port can neither answer as the dashboard nor replay an answer it recorded. Every body is
 * sealed as well, AES-256-GCM under a key both sides derive from the token (`seal`, `unseal`).
 *
 * A token an earlier version of this page kept in `localStorage` is removed.
 */

const STORAGE_KEY = 'usagedash.access';
const TOKEN_GLOBAL = '__USAGEDASH_TOKEN';
const AUTHORIZATION_SCHEME = 'Usagedash';
const CHALLENGE_PATH = '/challenge';
export const RESPONSE_SIGNATURE_HEADER = 'X-Usagedash-Response';
export const SEALED_CONTENT_TYPE = 'application/vnd.usagedash.sealed';
const SEAL_SALT = 'usagedash';
const SEAL_INFO = 'usagedash/sealed-body/v1';
const SEAL_NONCE_BYTES = 12;
const SEAL_TAG_BYTES = 16;
const SEAL_KEY_BITS = 256;
const NONCE_BYTES = 32;
const TOKEN_FORMAT = /^[A-Za-z0-9_-]{43,256}$/;
const CHALLENGE_FORMAT = /^[A-Za-z0-9_-]{43,256}$/;

/**
 * Removes the token an earlier version of this page stored, if any. Nothing is read.
 *
 * @param {Pick<Window, 'localStorage'>} browserWindow - The window whose storage to clear.
 * @returns {boolean} False when storage is unavailable, in which case nothing was stored there either.
 */
export function forgetStoredToken(browserWindow) {
  try {
    browserWindow.localStorage.removeItem(STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

/**
 * @param {Record<string, unknown>} browserWindow - The window the bootstrap file set its globals on.
 * @returns {string | null} The well-formed token the file set, or null when there is none.
 */
export function bootstrapToken(browserWindow) {
  const value = browserWindow[TOKEN_GLOBAL];
  return typeof value === 'string' && TOKEN_FORMAT.test(value) ? value : null;
}

/**
 * @param {ArrayBuffer} bytes
 * @returns {string} Lower-case hex.
 */
function hex(bytes) {
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * @param {string | Uint8Array | undefined} body - The body exactly as it is sent or was received, or nothing:
 *   text, or the bytes of a sealed body.
 * @returns {Promise<string>} Lower-case hex SHA-256 of the body's bytes (a string's UTF-8); of no bytes when
 *   there is no body.
 * @throws {TypeError} When the body is anything else: it would be sent as bytes this function did not hash.
 */
export async function bodyDigest(body) {
  const bytes =
    body === undefined ? new Uint8Array() : typeof body === 'string' ? new TextEncoder().encode(body) : body;
  if (!(bytes instanceof Uint8Array)) throw new TypeError('a body is text, or the bytes of a sealed body');
  return hex(await crypto.subtle.digest('SHA-256', bytes));
}

/**
 * @param {string} token - The token both sides hold. It is not sent anywhere by this function.
 * @returns {Promise<CryptoKey>} The AES-256-GCM key bodies are sealed under: HKDF-SHA256 of the token with
 *   this page's fixed salt and info, the same derivation the dashboard makes.
 */
async function sealKey(token) {
  const encoder = new TextEncoder();
  const material = await crypto.subtle.importKey('raw', encoder.encode(token), 'HKDF', false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    { name: 'HKDF', hash: 'SHA-256', salt: encoder.encode(SEAL_SALT), info: encoder.encode(SEAL_INFO) },
    material,
    { name: 'AES-GCM', length: SEAL_KEY_BITS },
    false,
    ['encrypt', 'decrypt'],
  );
}

/**
 * @returns {string} A random value for one request, 43 URL-safe characters, so the answer to it fits no other.
 */
export function freshNonce() {
  const bytes = crypto.getRandomValues(new Uint8Array(NONCE_BYTES));
  return btoa(String.fromCharCode(...bytes))
    .replaceAll('+', '-')
    .replaceAll('/', '_')
    .replace(/=+$/, '');
}

/**
 * @param {string} method - The request method, upper case.
 * @param {string} target - The path and query exactly as the request will carry them.
 * @param {string} challenge - The challenge the dashboard issued for this request.
 * @param {string} nonce - The nonce the page chose for it.
 * @returns {string} What a sealed request body is bound to besides the key, one part per line.
 */
export function requestSealContext(method, target, challenge, nonce) {
  return `${method}\n${target}\n${challenge}\n${nonce}`;
}

/**
 * @param {string} challenge - The challenge the request was signed over.
 * @param {string} nonce - The nonce the page chose for the request.
 * @param {number} status - The answer's status code.
 * @returns {string} What a sealed answer is bound to besides the key, one part per line.
 */
export function answerSealContext(challenge, nonce, status) {
  return `${challenge}\n${nonce}\n${status}`;
}

/**
 * Seals `plaintext` for `context`: a fresh random 96-bit nonce, then the AES-256-GCM ciphertext and its tag.
 *
 * @param {string} token - The token the key derives from.
 * @param {string} plaintext - The text to seal.
 * @param {string} context - `requestSealContext` or `answerSealContext`; authenticated, not carried.
 * @returns {Promise<Uint8Array>} The sealed bytes, which open only under the same token for the same context.
 */
export async function seal(token, plaintext, context) {
  const encoder = new TextEncoder();
  const nonce = crypto.getRandomValues(new Uint8Array(SEAL_NONCE_BYTES));
  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: nonce, additionalData: encoder.encode(context) },
      await sealKey(token),
      encoder.encode(plaintext),
    ),
  );
  const sealed = new Uint8Array(nonce.length + ciphertext.length);
  sealed.set(nonce);
  sealed.set(ciphertext, nonce.length);
  return sealed;
}

/**
 * Opens what `seal` sealed under the same token for the same context.
 *
 * @param {string} token - The token the key derives from.
 * @param {Uint8Array} sealed - The sealed bytes exactly as received.
 * @param {string} context - What they were sealed for.
 * @returns {Promise<string>} The text that was sealed.
 * @throws {Error} When the bytes are too short to be a sealed body, were sealed under another token or for
 *   another context, or have been altered.
 */
export async function unseal(token, sealed, context) {
  if (!(sealed instanceof Uint8Array) || sealed.length < SEAL_NONCE_BYTES + SEAL_TAG_BYTES) {
    throw new Error('the sealed body could not be opened');
  }
  const encoder = new TextEncoder();
  try {
    const plaintext = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: sealed.subarray(0, SEAL_NONCE_BYTES), additionalData: encoder.encode(context) },
      await sealKey(token),
      sealed.subarray(SEAL_NONCE_BYTES),
    );
    return new TextDecoder().decode(plaintext);
  } catch {
    throw new Error('the sealed body could not be opened');
  }
}

/**
 * The text a request signature covers, one part per line. The server builds the same text from the
 * request it received and the challenge it issued.
 *
 * @param {string} method - The request method, upper case.
 * @param {string} target - The path and query exactly as the request will carry them.
 * @param {string} bodyHash - `bodyDigest` of the body exactly as the request will carry it.
 * @param {string} challenge - The challenge the dashboard issued for this request.
 * @param {string} nonce - The nonce the page chose for it.
 * @returns {string}
 */
export function signatureMessage(method, target, bodyHash, challenge, nonce) {
  return `${method}\n${target}\n${bodyHash}\n${challenge}\n${nonce}`;
}

/**
 * @param {string} token - The key. It is not sent anywhere by this function.
 * @param {string} method - The request method, upper case.
 * @param {string} target - The path and query exactly as the request will carry them.
 * @param {string} bodyHash - `bodyDigest` of the body exactly as the request will carry it.
 * @param {string} challenge - The challenge the dashboard issued for this request.
 * @param {string} nonce - The nonce the page chose for it.
 * @returns {Promise<string>} Lower-case hex HMAC-SHA256 of `signatureMessage`, keyed by `token`.
 */
export async function requestSignature(token, method, target, bodyHash, challenge, nonce) {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey('raw', encoder.encode(token), { name: 'HMAC', hash: 'SHA-256' }, false, [
    'sign',
  ]);
  const message = encoder.encode(signatureMessage(method, target, bodyHash, challenge, nonce));
  return hex(await crypto.subtle.sign('HMAC', key, message));
}

/**
 * Asks the dashboard for the challenge the next request must be signed over.
 *
 * The request carries nothing: no token, no signature. Whatever process answers can only hand out
 * values of its own; a signature over one of those fits that process and nothing else.
 *
 * @param {typeof fetch} fetchImpl - The fetch implementation to ask with.
 * @param {string} origin - The dashboard origin, or '' from the page the dashboard serves itself.
 * @returns {Promise<string>} A well-formed challenge.
 * @throws {Error} When no challenge was issued: the listener did not answer, answered with a failure, or
 *   answered with something that is not a challenge. The request is then not sent.
 */
export async function issuedChallenge(fetchImpl, origin) {
  const response = await fetchImpl(`${origin}${CHALLENGE_PATH}`);
  if (response.ok !== true) throw new Error(`the dashboard did not issue a challenge (${response.status})`);
  /** @type {unknown} */
  const body = await response.json();
  const challenge =
    body !== null && typeof body === 'object' ? /** @type {{ challenge?: unknown }} */ (body).challenge : undefined;
  if (typeof challenge !== 'string' || !CHALLENGE_FORMAT.test(challenge)) {
    throw new Error('the dashboard did not issue a challenge');
  }
  return challenge;
}

/**
 * The signature the dashboard puts on its answer to a signed request.
 *
 * @param {string} token - The key.
 * @param {string} challenge - The challenge the request was signed over.
 * @param {string} nonce - The nonce the page chose for the request.
 * @param {number} status - The answer's status code.
 * @param {string | Uint8Array} body - The answer's body exactly as received: the sealed bytes.
 * @returns {Promise<string>} Lower-case hex HMAC-SHA256 of the challenge, the nonce, the status, and the
 *   body's SHA-256, one per line, keyed by `token`.
 */
export async function responseSignature(token, challenge, nonce, status, body) {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey('raw', encoder.encode(token), { name: 'HMAC', hash: 'SHA-256' }, false, [
    'sign',
  ]);
  const message = encoder.encode(`${challenge}\n${nonce}\n${status}\n${await bodyDigest(body)}`);
  return hex(await crypto.subtle.sign('HMAC', key, message));
}

/**
 * @param {string} expected - The signature this page computed.
 * @param {string | null} presented - The signature the answer carried, if any.
 * @returns {boolean} Whether the two are the same length and the same characters, compared whole.
 */
function signaturesMatch(expected, presented) {
  if (presented === null || presented.length !== expected.length) return false;
  let difference = 0;
  for (let index = 0; index < expected.length; index += 1) {
    difference |= expected.charCodeAt(index) ^ presented.charCodeAt(index);
  }
  return difference === 0;
}

/**
 * Takes the token the bootstrap file set, if any, and forgets one an earlier page stored.
 *
 * @param {Pick<Window, 'localStorage'> & Record<string, unknown>} browserWindow - The window to sign in. Its
 *   `localStorage` is only touched inside a guarded operation: a browser that blocks storage can make the
 *   property itself throw.
 * @param {{ fetch?: typeof fetch, origin?: string, nonce?: () => string }} [options] - `fetch` replaces the
 *   fetch the challenge is asked with; `origin` is the dashboard origin, '' from the served page; `nonce`
 *   replaces the draw of each request's nonce (tests).
 * @returns {{
 *   isSignedIn: () => boolean,
 *   authorize: (method: string, target: string, body?: string) => Promise<{ headers: Record<string, string>, challenge: string, nonce: string, body: Uint8Array | undefined } | null>,
 *   verifyAnswer: (challenge: string, nonce: string, status: number, body: string | Uint8Array, signature: string | null) => Promise<boolean>,
 *   openAnswer: (challenge: string, nonce: string, status: number, sealed: Uint8Array) => Promise<string>,
 * }} `authorize` asks for a challenge, draws a nonce, seals the body (if any) for that method, target,
 *   challenge, and nonce, and resolves to the headers for one request (the `Authorization` header signed over
 *   exactly that, and the sealed body's content type), the challenge, the nonce, and the sealed body to send;
 *   null, asking nothing, when no token is held; rejects when no challenge was issued or the body is not
 *   text. `verifyAnswer` reports whether an answer's signature is the dashboard's own for that exchange,
 *   status, and body as received (false without a token). `openAnswer` opens an answer sealed for that
 *   exchange and status, and rejects when it was not.
 */
export function createAccess(browserWindow, options = {}) {
  forgetStoredToken(browserWindow);
  const token = bootstrapToken(browserWindow);
  const fetchImpl = options.fetch ?? ((...args) => globalThis.fetch(...args));
  const origin = options.origin ?? '';
  const drawNonce = options.nonce ?? freshNonce;
  return {
    isSignedIn: () => token !== null,
    authorize: async (method, target, body) => {
      if (token === null) return null;
      if (body !== undefined && typeof body !== 'string') throw new TypeError('a request body must be text');
      const challenge = await issuedChallenge(fetchImpl, origin);
      const nonce = drawNonce();
      const sealedBody =
        body === undefined ? undefined : await seal(token, body, requestSealContext(method, target, challenge, nonce));
      const digest = await requestSignature(token, method, target, await bodyDigest(sealedBody), challenge, nonce);
      /** @type {Record<string, string>} */
      const headers = { Authorization: `${AUTHORIZATION_SCHEME} ${challenge}.${nonce}.${digest}` };
      if (sealedBody !== undefined) headers['Content-Type'] = SEALED_CONTENT_TYPE;
      return { headers, challenge, nonce, body: sealedBody };
    },
    verifyAnswer: async (challenge, nonce, status, body, signature) => {
      if (token === null) return false;
      return signaturesMatch(await responseSignature(token, challenge, nonce, status, body), signature);
    },
    openAnswer: async (challenge, nonce, status, sealed) => {
      if (token === null) throw new Error('no token is held');
      return unseal(token, sealed, answerSealContext(challenge, nonce, status));
    },
  };
}
