"""The dashboard's own access token, the signature a request carries instead of it, and the page that holds it.

The token is created once, kept in a file only this user can read, and never travels. For each request
the page asks the process it is talking to for a challenge, adds a nonce of its own, and signs method,
target, body digest, challenge, and nonce with the token (``request_signature``). The server accepts a
signature only over a challenge it issued and has not seen used (``RequestVerifier``), and signs its
answer over the challenge, the nonce, and the status, so a request is good once, towards the one process
that issued its challenge, and a recorded answer fits no later request. Every body is sealed as well
(``BodySealer``). Nothing here depends on the wall clock.

The page is an owner-only local file written by ``usagedash open`` (``write_bootstrap_page``), carrying
the token as a JSON-encoded string and the dashboard's own code inline, so neither comes from whatever
process is listening on the port.
"""

import base64
import contextlib
import dataclasses
import enum
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from usagedash.errors import ConfigError, PageAssemblyError, SealedBodyError
from usagedash.private import PRIVATE_FILE_MODE, ensure_private_directory, restrict_to_owner

CREDENTIAL_FILE_NAME = "access-token"
BOOTSTRAP_PAGE_NAME = "open-dashboard.html"
STATIC_DIR = Path(__file__).parent / "static"
PAGE_SHELL_NAME = "index.html"
PAGE_MODULE_NAME = "app.js"
PAGE_STYLESHEET_NAME = "styles.css"
SHELL_STYLESHEET_TAG = '<link rel="stylesheet" href="/static/styles.css" />'
SHELL_MODULE_TAG = '<script type="module" src="/static/app.js"></script>'
ORIGIN_GLOBAL = "__USAGEDASH_ORIGIN"
TOKEN_GLOBAL = "__USAGEDASH_TOKEN"  # noqa: S105 - the name of the JavaScript global that holds the token, not a value
DASHBOARD_HOST = "127.0.0.1"
"""The one name the page may address the dashboard by.

Why not ``localhost``: a browser may resolve it to ``::1``, where another process can listen on the
dashboard's port while the dashboard holds ``127.0.0.1``, receive the page's signed requests, and relay
them. The dashboard binds ``127.0.0.1`` only, so the page connects there only.
"""
SIGNATURE_SCHEME = "Usagedash"
RESPONSE_SIGNATURE_HEADER = "X-Usagedash-Response"
SEALED_CONTENT_TYPE = "application/vnd.usagedash.sealed"
"""The media type of a sealed body: a nonce, then AES-256-GCM ciphertext and its tag."""
SEAL_NONCE_BYTES = 12
SEAL_TAG_BYTES = 16
SEAL_KEY_BYTES = 32
_SEAL_SALT = b"usagedash"
_SEAL_INFO = b"usagedash/sealed-body/v1"
_HKDF_MAX_BLOCKS = 255
"""The header that carries ``response_signature`` on every answer to an accepted signed request."""
CHALLENGE_BYTES = 32
NONCE_BYTES = 32
CHALLENGE_TTL_SECONDS = 60
"""How long an issued challenge waits for its request. The page uses one within milliseconds."""
MAX_OUTSTANDING_CHALLENGES = 256
"""More unused challenges than this is not the page, which asks for one request at a time."""
ACCESS_TOKEN_BYTES = 32
TEMPORARY_SUFFIX_BYTES = 8
MAX_TOKEN_FILE_BYTES = 1024
_TOKEN_FORMAT = re.compile(r"[A-Za-z0-9_-]{43,256}")
_CHALLENGE_FORMAT = re.compile(r"[A-Za-z0-9_-]{43,256}")
_DIGEST_FORMAT = re.compile(r"[0-9a-f]{64}")
_SIBLING_IMPORT = re.compile(r"from '\./([A-Za-z0-9_-]+\.js)'")
"""An import of a module beside this one, as the page's modules write them (single quotes, one directory)."""
_LEFTOVER_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*['"`]\.\.?/""")
"""Any import of a relative path that ``_SIBLING_IMPORT`` did not rewrite. The browser could not load it."""


def is_well_formed_token(token: str) -> bool:
    """Report whether ``token`` is one this module could have created.

    That is at least the 43 characters ``secrets.token_urlsafe(32)`` produces, all from its alphabet. A
    shorter token could be guessed, and any other character might not survive a JavaScript string or an
    HMAC key unchanged.
    """
    return _TOKEN_FORMAT.fullmatch(token) is not None


def _write_private_temporary(directory: Path, name: str, content: str) -> Path:
    """Write ``content`` to a new owner-only file beside ``name`` and return its path.

    Raises:
        OSError: if the file cannot be created or written. Nothing is left behind.
    """
    temporary = directory / f".{name}.{secrets.token_hex(TEMPORARY_SUFFIX_BYTES)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            restrict_to_owner(handle.fileno(), PRIVATE_FILE_MODE)
            handle.write(content)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stored_token(path: Path) -> str | None:
    """Return the stored token, or None when no token file exists yet.

    The file is made owner-only every time it is read: a file restored from a backup keeps whatever
    permissions it arrived with. A symbolic link is refused.

    Raises:
        OSError: if the file exists but cannot be opened, made owner-only, or read.
        ConfigError: if the file holds anything other than one well-formed token.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as handle:
        restrict_to_owner(handle.fileno(), PRIVATE_FILE_MODE)
        content = handle.read(MAX_TOKEN_FILE_BYTES + 1)
    token = content.decode("ascii", errors="replace").strip()
    # Why the length is checked on its own: only the first bytes were read. A file longer than that
    # holds something after them, and a token followed by anything else is not one token.
    if len(content) > MAX_TOKEN_FILE_BYTES or not is_well_formed_token(token):
        raise ConfigError(f"{path} holds no usable token; delete it and run `usagedash open` to create a new one")
    return token


def _create_token_if_absent(path: Path) -> None:
    temporary = _write_private_temporary(path.parent, path.name, secrets.token_urlsafe(ACCESS_TOKEN_BYTES))
    try:
        # Why: linking fails when the file already exists. That means another caller created the token
        # first, and theirs is the stored one this caller must use, so the failure is the expected outcome.
        with contextlib.suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        temporary.unlink()


def ensure_access_token(data_dir: Path) -> str:
    """Return the dashboard's access token, creating it on first use.

    The server and ``usagedash open`` both call this, possibly at once. The file is created by linking a
    fully written temporary file into place, which fails if the file exists, so every caller returns the
    one stored token and none sees a partly written file.

    Raises:
        ConfigError: if the data directory cannot be made private, or the token file cannot be read or
            created, or holds anything other than one well-formed token.
    """
    ensure_private_directory(data_dir)
    path = data_dir / CREDENTIAL_FILE_NAME
    try:
        token = _stored_token(path)
        if token is None:
            _create_token_if_absent(path)
            token = _stored_token(path)
    except OSError as error:
        raise ConfigError(f"cannot read or create {path}: {type(error).__name__}") from error
    if token is None:
        raise ConfigError(f"{path} disappeared while it was being created; run `usagedash open` again")
    return token


def token_matches(presented: str, expected: str) -> bool:
    """Compare two tokens in constant time."""
    return secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def body_digest(body: bytes) -> str:
    """Return the lower-case hex SHA-256 of a request body. An absent body is the empty body."""
    return hashlib.sha256(body).hexdigest()


def is_well_formed_challenge(challenge: str) -> bool:
    """Report whether ``challenge`` is one ``RequestVerifier.issue`` could have made: 43 to 256 URL-safe characters.

    A nonce the page chooses (``fresh_nonce``) has the same shape.
    """
    return _CHALLENGE_FORMAT.fullmatch(challenge) is not None


def fresh_nonce() -> str:
    """Return the random value the page adds to one request, so the answer to it fits no other."""
    return secrets.token_urlsafe(NONCE_BYTES)


def signature_message(method: str, target: str, body_hash: str, challenge: str, nonce: str) -> bytes:
    """Return the bytes a request signature covers, one part per line.

    ``target`` is the path and query exactly as the request line carries them, ``body_hash`` is
    ``body_digest`` of the body as sent, ``challenge`` the value the dashboard issued for this request,
    and ``nonce`` the value the page chose for it.
    """
    return f"{method}\n{target}\n{body_hash}\n{challenge}\n{nonce}".encode()


def request_signature(token: str, method: str, target: str, body_hash: str, challenge: str, nonce: str) -> str:
    """Return the lower-case hex HMAC-SHA256 of ``signature_message``, keyed by ``token``.

    The page computes this with the token it holds and the server with its own copy. The digest reveals
    nothing of the key.
    """
    message = signature_message(method, target, body_hash, challenge, nonce)
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def response_signature(token: str, challenge: str, nonce: str, status_code: int, body: bytes) -> str:
    """Return the lower-case hex HMAC-SHA256 the dashboard puts on its answer to a signed request.

    The message is the challenge, the nonce, the status code, and ``body_digest`` of the answer's body,
    one per line. A process holding the port but not the token cannot sign an answer, and an answer it
    recorded earlier was signed over another nonce.
    """
    message = f"{challenge}\n{nonce}\n{status_code}\n{body_digest(body)}".encode()
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def hkdf_sha256(input_key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """Derive ``length`` bytes from ``input_key`` as RFC 5869 does with SHA-256; the page derives the same.

    Raises:
        ValueError: if ``length`` is not between 1 and 255 hash lengths.
    """
    digest_size = hashlib.sha256().digest_size
    if not 0 < length <= _HKDF_MAX_BLOCKS * digest_size:
        raise ValueError(f"an HKDF-SHA256 output is 1 to {_HKDF_MAX_BLOCKS * digest_size} bytes")
    pseudo_random_key = hmac.new(salt, input_key, hashlib.sha256).digest()
    output = bytearray()
    block = b""
    counter = 1
    while len(output) < length:
        block = hmac.new(pseudo_random_key, block + info + bytes([counter]), hashlib.sha256).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:length])


def seal_key(token: str) -> bytes:
    """Return the key bodies are sealed under: HKDF-SHA256 of the token, with this package's fixed salt and info."""
    return hkdf_sha256(token.encode("utf-8"), _SEAL_SALT, _SEAL_INFO, SEAL_KEY_BYTES)


def request_seal_context(method: str, target: str, challenge: str, nonce: str) -> bytes:
    """Return what a sealed request body is bound to besides the key: method, target, challenge, nonce, one per line."""
    return f"{method}\n{target}\n{challenge}\n{nonce}".encode()


def answer_seal_context(challenge: str, nonce: str, status_code: int) -> bytes:
    """Return what a sealed answer is bound to besides the key: the exchange's challenge and nonce, and the status."""
    return f"{challenge}\n{nonce}\n{status_code}".encode()


class BodySealer:
    """Seals bodies, and opens them, with AES-256-GCM under the key derived from the token.

    A sealed body is a fresh random 96-bit nonce followed by the ciphertext and its 128-bit tag. The
    context it was sealed for (``request_seal_context`` or ``answer_seal_context``) is authenticated with
    it but not carried in it, so a sealed body opens only where it was sealed for. The seal adds
    confidentiality; who sent and who answered is still established by the signatures over the sealed bytes.
    """

    def __init__(self, token: str) -> None:
        self._cipher = AESGCM(seal_key(token))

    def seal(self, plaintext: bytes, context: bytes) -> bytes:
        """Return ``plaintext`` sealed for ``context``, under a nonce used for this call alone."""
        nonce = secrets.token_bytes(SEAL_NONCE_BYTES)
        return nonce + self._cipher.encrypt(nonce, plaintext, context)

    def unseal(self, sealed: bytes, context: bytes) -> bytes:
        """Return the plaintext of ``sealed``, which must have been sealed under this key for ``context``.

        Raises:
            SealedBodyError: if ``sealed`` is too short to hold a nonce and a tag, was sealed under another
                key or for another context, or has been altered.
        """
        if len(sealed) < SEAL_NONCE_BYTES + SEAL_TAG_BYTES:
            raise SealedBodyError("the sealed body is too short to hold a nonce and a tag")
        nonce, ciphertext = sealed[:SEAL_NONCE_BYTES], sealed[SEAL_NONCE_BYTES:]
        try:
            return self._cipher.decrypt(nonce, ciphertext, context)
        except InvalidTag as error:
            raise SealedBodyError("the sealed body does not open under this key for this context") from error


@dataclasses.dataclass(frozen=True, slots=True)
class SignedAuthorization:
    """The three parts of an ``Authorization: Usagedash <challenge>.<nonce>.<digest>`` header."""

    challenge: str
    nonce: str
    digest: str

    @classmethod
    def parse(cls, credentials: str) -> "SignedAuthorization | None":
        """Return the parts of ``credentials`` (the header value after the scheme), or None if they are not well formed.

        The challenge and the nonce are 43 to 256 URL-safe characters each; the digest is 64 lower-case
        hex characters.
        """
        parts = credentials.strip().split(".")
        if len(parts) != 3:
            return None
        challenge, nonce, digest = parts
        if not is_well_formed_challenge(challenge) or not is_well_formed_challenge(nonce):
            return None
        if _DIGEST_FORMAT.fullmatch(digest) is None:
            return None
        return cls(challenge, nonce, digest)

    def header(self) -> str:
        """Return the ``Authorization`` header value that carries these parts."""
        return f"{SIGNATURE_SCHEME} {self.challenge}.{self.nonce}.{self.digest}"


def sign_request(
    token: str, method: str, target: str, challenge: str, body: bytes = b"", nonce: str | None = None
) -> str:
    """Return the ``Authorization`` header value for one request, signed with ``token`` over ``challenge`` and
    ``nonce``.

    ``body`` is the request body exactly as it will be sent. A ``nonce`` left out is drawn fresh.

    Raises:
        ValueError: if ``challenge`` or ``nonce`` is not well formed.
    """
    chosen = fresh_nonce() if nonce is None else nonce
    if not is_well_formed_challenge(challenge):
        raise ValueError("a challenge is 43 to 256 URL-safe characters")
    if not is_well_formed_challenge(chosen):
        raise ValueError("a nonce is 43 to 256 URL-safe characters")
    digest = request_signature(token, method, target, body_digest(body), challenge, chosen)
    return SignedAuthorization(challenge, chosen, digest).header()


class SignatureOutcome(enum.Enum):
    """What ``RequestVerifier.verify`` found. Each value is the reason a refusal is recorded under."""

    ACCEPTED = "accepted"
    NOT_SIGNED = "no signature"
    MALFORMED = "the signature is malformed"
    UNKNOWN_CHALLENGE = "the challenge was not issued by this process, or has been used"
    WRONG = "the signature does not match"


class RequestVerifier:
    """Issues the challenge each request must be signed over, and checks the signature that comes back.

    A challenge is held for ``ttl_seconds`` of this process's own monotonic clock, or until a signature
    over it is accepted; one another process issued, one already accepted, or one issued too long ago is
    refused. At most ``capacity`` are held, the oldest forgotten first, so a token-less flood of requests
    for challenges only makes the page's next request fail and retry. A signature that does not match
    leaves its challenge held. Safe to use from several threads.
    """

    def __init__(
        self,
        token: str,
        *,
        ttl_seconds: float = CHALLENGE_TTL_SECONDS,
        capacity: int = MAX_OUTSTANDING_CHALLENGES,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._token = token
        self._ttl_seconds = ttl_seconds
        self._capacity = capacity
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._issued: dict[str, float] = {}

    @property
    def outstanding(self) -> int:
        """How many issued challenges are currently held."""
        with self._lock:
            return len(self._issued)

    def issue(self) -> str:
        """Return a fresh challenge and hold it for one request."""
        with self._lock:
            self._forget_expired()
            while len(self._issued) >= self._capacity:
                del self._issued[next(iter(self._issued))]
            challenge = secrets.token_urlsafe(CHALLENGE_BYTES)
            self._issued[challenge] = self._monotonic()
            return challenge

    def verify(self, authorization: str | None, method: str, target: str, body_hash: str) -> SignatureOutcome:
        """Check the ``Authorization`` header of a request, and spend its challenge if it is accepted.

        ``method``, ``target``, and ``body_hash`` are what the request actually asked for. ``NOT_SIGNED``
        means the header is absent or carries another scheme; every other outcome but ``ACCEPTED`` is a
        refusal.
        """
        scheme, _, credentials = (authorization or "").partition(" ")
        if scheme.lower() != SIGNATURE_SCHEME.lower():
            return SignatureOutcome.NOT_SIGNED
        signed = SignedAuthorization.parse(credentials)
        if signed is None:
            return SignatureOutcome.MALFORMED
        with self._lock:
            self._forget_expired()
            if signed.challenge not in self._issued:
                return SignatureOutcome.UNKNOWN_CHALLENGE
            expected = request_signature(self._token, method, target, body_hash, signed.challenge, signed.nonce)
            if not hmac.compare_digest(expected, signed.digest):
                return SignatureOutcome.WRONG
            del self._issued[signed.challenge]
            return SignatureOutcome.ACCEPTED

    def _forget_expired(self) -> None:
        """Drop the challenges issued longer ago than the lifetime. Called with the lock held."""
        now = self._monotonic()
        for challenge in [c for c, issued_at in self._issued.items() if now - issued_at > self._ttl_seconds]:
            del self._issued[challenge]


def dashboard_origin(dashboard_url: str) -> str:
    """Return the origin (``http://127.0.0.1:port``) of the dashboard at ``dashboard_url``.

    Raises:
        ValueError: if the URL is not plain HTTP to ``DASHBOARD_HOST`` with an explicit port and nothing
            but an optional trailing slash after it. The dashboard listens nowhere else, and the origin
            is written into the page's content-security policy and its script.
    """
    parts = urlsplit(dashboard_url)
    if (
        parts.scheme != "http"
        or parts.hostname != DASHBOARD_HOST
        or parts.port is None
        or parts.username is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"the dashboard URL must be http://{DASHBOARD_HOST}:<port>/")
    return f"http://{DASHBOARD_HOST}:{parts.port}"


def _module_data_url(source: str) -> str:
    return "data:text/javascript;base64," + base64.b64encode(source.encode("utf-8")).decode("ascii")


def _inline_module(name: str, static_dir: Path, inlined: dict[str, str], importers: tuple[str, ...] = ()) -> str:
    """Return the ``data:`` URL of module ``name``, its sibling imports replaced by their own ``data:`` URLs.

    The module's text is otherwise unchanged. A sibling imported from two modules is inlined once and
    gets the same URL each time, which the browser loads once.

    Raises:
        PageAssemblyError: if the module is missing, imports itself through others, or imports a path
            this cannot rewrite.
    """
    if name in inlined:
        return inlined[name]
    if name in importers:
        raise PageAssemblyError(f"{name} imports itself through {' -> '.join(importers)}")
    try:
        source = (static_dir / name).read_text(encoding="utf-8")
    except OSError as error:
        raise PageAssemblyError(f"cannot read the page module {name}: {type(error).__name__}") from error

    def inline_sibling(match: re.Match[str]) -> str:
        return f"from '{_inline_module(match.group(1), static_dir, inlined, (*importers, name))}'"

    rewritten = _SIBLING_IMPORT.sub(inline_sibling, source)
    if _LEFTOVER_IMPORT.search(rewritten) is not None:
        raise PageAssemblyError(f"{name} imports a path the page cannot inline")
    inlined[name] = _module_data_url(rewritten)
    return inlined[name]


def _csp_hash(inline_text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(inline_text.encode("utf-8")).digest()).decode("ascii") + "'"


def _javascript_string(value: str) -> str:
    """Write ``value`` as a JavaScript string literal that cannot end the script element it sits in."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e")


def _replace_once(shell: str, tag: str, replacement: str) -> str:
    if shell.count(tag) != 1:
        raise PageAssemblyError(f"{PAGE_SHELL_NAME} does not hold exactly one {tag}")
    return shell.replace(tag, replacement)


def assemble_bootstrap_page(origin: str, token: str, static_dir: Path = STATIC_DIR) -> str:
    """Return the dashboard page as one HTML document that loads nothing from the dashboard but its data.

    The served shell with its stylesheet and its module inlined, the modules as ``data:`` URLs
    (``_inline_module``): a browser refuses a module loaded from a ``file:`` URL, and one served by the
    listener would run with the token in reach. The token and the origin are set as JSON-encoded
    globals before the module runs; a content-security policy allows exactly those scripts, that
    stylesheet, and connections to ``origin``.

    Raises:
        PageAssemblyError: if the shell or a module is not in the shape this expects.
    """
    try:
        shell = (static_dir / PAGE_SHELL_NAME).read_text(encoding="utf-8")
        stylesheet = (static_dir / PAGE_STYLESHEET_NAME).read_text(encoding="utf-8")
    except OSError as error:
        raise PageAssemblyError(f"cannot read the page shell: {type(error).__name__}") from error
    if "</style" in stylesheet:
        raise PageAssemblyError(f"{PAGE_STYLESHEET_NAME} cannot be inlined")
    globals_script = (
        f"window.{ORIGIN_GLOBAL} = {_javascript_string(origin)};\n"
        f"window.{TOKEN_GLOBAL} = {_javascript_string(token)};\n"
    )
    module_script = f"import {_javascript_string(_inline_module(PAGE_MODULE_NAME, static_dir, {}))};\n"
    policy = (
        f"default-src 'none'; script-src {_csp_hash(globals_script)} {_csp_hash(module_script)} data:; "
        f"style-src {_csp_hash(stylesheet)}; connect-src {origin}; base-uri 'none'; form-action 'none'"
    )
    head = f'<meta http-equiv="Content-Security-Policy" content="{policy}" />\n    <style>{stylesheet}</style>'
    scripts = f'<script>{globals_script}</script>\n    <script type="module">{module_script}</script>'
    page = _replace_once(shell, SHELL_STYLESHEET_TAG, head)
    return _replace_once(page, SHELL_MODULE_TAG, scripts)


def write_bootstrap_page(data_dir: Path, dashboard_url: str, token: str) -> Path:
    """Write the owner-only page that is the dashboard, holding ``token``, and return its path.

    Raises:
        ValueError: if ``token`` is not well formed or ``dashboard_url`` is not the dashboard's.
        PageAssemblyError: if the package's static files are not in the shape the page is built from.
        ConfigError: if the data directory cannot be made private or the page cannot be written.
    """
    if not is_well_formed_token(token):
        raise ValueError("the page can only carry a token created by ensure_access_token")
    page = assemble_bootstrap_page(dashboard_origin(dashboard_url), token)
    ensure_private_directory(data_dir)
    path = data_dir / BOOTSTRAP_PAGE_NAME
    try:
        temporary = _write_private_temporary(data_dir, BOOTSTRAP_PAGE_NAME, page)
        try:
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
    except OSError as error:
        raise ConfigError(f"cannot write {path}: {type(error).__name__}") from error
    return path
