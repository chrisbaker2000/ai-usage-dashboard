import base64
import hashlib
import hmac
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from usagedash import access
from usagedash.access import (
    CHALLENGE_TTL_SECONDS,
    SEAL_NONCE_BYTES,
    SEAL_TAG_BYTES,
    SEALED_CONTENT_TYPE,
    BodySealer,
    RequestVerifier,
    SignatureOutcome,
    SignedAuthorization,
    answer_seal_context,
    assemble_bootstrap_page,
    body_digest,
    dashboard_origin,
    ensure_access_token,
    fresh_nonce,
    hkdf_sha256,
    is_well_formed_challenge,
    is_well_formed_token,
    request_seal_context,
    request_signature,
    response_signature,
    seal_key,
    sign_request,
    signature_message,
    token_matches,
    write_bootstrap_page,
)
from usagedash.errors import ConfigError, PageAssemblyError, SealedBodyError

DASHBOARD_URL = "http://127.0.0.1:8787/"
ORIGIN = "http://127.0.0.1:8787"
TOKEN = "synthetic-access-token-" + "x" * 20
OTHER_TOKEN = "synthetic-second-token-" + "y" * 20
CHALLENGE = "synthetic-challenge-" + "c" * 23
OTHER_CHALLENGE = "synthetic-challenge-" + "d" * 23
NONCE = "synthetic-nonce-" + "n" * 27
OTHER_NONCE = "synthetic-nonce-" + "m" * 27
EMPTY_BODY = hashlib.sha256(b"").hexdigest()
RENAME_BODY = b'{"label":"Renamed"}'
CONCURRENT_CALLERS = 8
STATIC_DIR = Path(access.__file__).parent / "static"
SHELL = (
    "<!doctype html>\n<html><head>\n"
    '    <link rel="stylesheet" href="/static/styles.css" />\n'
    '</head><body><div id="providers"></div>\n'
    '    <script type="module" src="/static/app.js"></script>\n'
    "</body></html>\n"
)
DATA_URL = re.compile(r"data:text/javascript;base64,([A-Za-z0-9+/=]+)")
CSP_HASH = re.compile(r"'sha256-([A-Za-z0-9+/=]+)'")


def leftover_temporary_files(directory: Path) -> list[str]:
    return [path.name for path in directory.iterdir() if path.name.endswith(".tmp")]


def reference_digest(token: str, method: str, target: str, body_hash: str, challenge: str, nonce: str = NONCE) -> str:
    message = f"{method}\n{target}\n{body_hash}\n{challenge}\n{nonce}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


class FakeMonotonic:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def verifier_and_clock(**options: int) -> tuple[RequestVerifier, FakeMonotonic]:
    clock = FakeMonotonic()
    return RequestVerifier(TOKEN, monotonic=clock, **options), clock


def signed_for(verifier: RequestVerifier, method: str = "GET", target: str = "/api/state", body: bytes = b"") -> str:
    """A header the page would send to ``verifier``: signed over a challenge it just issued."""
    return sign_request(TOKEN, method, target, verifier.issue(), body)


def decoded(data_url: str) -> str:
    match = DATA_URL.fullmatch(data_url)
    assert match is not None, data_url[:60]
    return base64.b64decode(match.group(1)).decode()


def csp_hash_of(inline_text: str) -> str:
    return base64.b64encode(hashlib.sha256(inline_text.encode()).digest()).decode()


def inline_blocks(page: str) -> dict[str, str]:
    """The text of each inline block: the plain script, the module script, the stylesheet, and the policy."""
    plain = re.search(r"<script>(.*?)</script>", page, re.S)
    module = re.search(r'<script type="module">(.*?)</script>', page, re.S)
    style = re.search(r"<style>(.*?)</style>", page, re.S)
    policy = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]*)" />', page)
    assert plain and module and style and policy
    return {"plain": plain.group(1), "module": module.group(1), "style": style.group(1), "policy": policy.group(1)}


def synthetic_static_dir(
    tmp_path: Path, modules: dict[str, str], shell: str = SHELL, stylesheet: str = "body{}"
) -> Path:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(shell)
    (static / "styles.css").write_text(stylesheet)
    for name, source in modules.items():
        (static / name).write_text(source)
    return static


class TestTokenFormat:
    def test_a_created_token_is_well_formed(self, tmp_path: Path) -> None:
        assert is_well_formed_token(ensure_access_token(tmp_path))

    @pytest.mark.parametrize("token", [TOKEN, "A" * 43, "a-b_c" + "0" * 38, "z" * 256])
    def test_forty_three_or_more_url_safe_characters_are_well_formed(self, token: str) -> None:
        assert is_well_formed_token(token)

    @pytest.mark.parametrize(
        "token",
        [
            "",
            "short",
            "x" * 42,
            "z" * 257,
            TOKEN + " ",
            TOKEN + "\n",
            " " + TOKEN,
            TOKEN[:20] + " " + TOKEN[20:],
            TOKEN[:-1] + "é",
            TOKEN[:-1] + '"',
            TOKEN[:-1] + "<",
            TOKEN[:-1] + "#",
            TOKEN[:-1] + "=",
        ],
    )
    def test_a_token_that_is_guessable_or_would_not_survive_a_string_or_header_is_not(self, token: str) -> None:
        assert not is_well_formed_token(token)


class TestRequestSignature:
    def test_is_the_hmac_of_method_target_body_challenge_and_nonce_keyed_by_the_token(self) -> None:
        digest = request_signature(TOKEN, "GET", "/api/state", EMPTY_BODY, CHALLENGE, NONCE)

        assert digest == reference_digest(TOKEN, "GET", "/api/state", EMPTY_BODY, CHALLENGE, NONCE)
        assert len(digest) == 64
        assert TOKEN not in digest

    def test_the_message_is_the_five_parts_one_per_line(self) -> None:
        assert signature_message("PATCH", "/api/accounts/7", EMPTY_BODY, CHALLENGE, NONCE) == (
            f"PATCH\n/api/accounts/7\n{EMPTY_BODY}\n{CHALLENGE}\n{NONCE}".encode()
        )

    @pytest.mark.parametrize(
        ("token", "method", "target", "body_hash", "challenge", "nonce"),
        [
            (OTHER_TOKEN, "GET", "/api/state", EMPTY_BODY, CHALLENGE, NONCE),
            (TOKEN, "POST", "/api/state", EMPTY_BODY, CHALLENGE, NONCE),
            (TOKEN, "GET", "/api/tokens", EMPTY_BODY, CHALLENGE, NONCE),
            (TOKEN, "GET", "/api/state?range=7d", EMPTY_BODY, CHALLENGE, NONCE),
            (TOKEN, "GET", "/api/state", hashlib.sha256(RENAME_BODY).hexdigest(), CHALLENGE, NONCE),
            (TOKEN, "GET", "/api/state", EMPTY_BODY, OTHER_CHALLENGE, NONCE),
            (TOKEN, "GET", "/api/state", EMPTY_BODY, CHALLENGE, OTHER_NONCE),
        ],
    )
    def test_any_other_token_method_target_body_challenge_or_nonce_gives_another_digest(
        self, token: str, method: str, target: str, body_hash: str, challenge: str, nonce: str
    ) -> None:
        assert request_signature(token, method, target, body_hash, challenge, nonce) != request_signature(
            TOKEN, "GET", "/api/state", EMPTY_BODY, CHALLENGE, NONCE
        )

    def test_the_answer_signature_is_the_hmac_of_challenge_nonce_status_and_body_digest(self) -> None:
        body = b'{"status": "ok"}'

        digest = response_signature(TOKEN, CHALLENGE, NONCE, 200, body)

        message = f"{CHALLENGE}\n{NONCE}\n200\n{hashlib.sha256(body).hexdigest()}".encode()
        assert digest == hmac.new(TOKEN.encode(), message, hashlib.sha256).hexdigest()
        assert digest != response_signature(TOKEN, OTHER_CHALLENGE, NONCE, 200, body)
        assert digest != response_signature(TOKEN, CHALLENGE, OTHER_NONCE, 200, body)
        assert digest != response_signature(TOKEN, CHALLENGE, NONCE, 404, body)
        assert digest != response_signature(TOKEN, CHALLENGE, NONCE, 200, b"{}")
        assert digest != response_signature(OTHER_TOKEN, CHALLENGE, NONCE, 200, body)
        assert TOKEN not in digest

    def test_a_fresh_nonce_is_well_formed_and_never_repeats(self) -> None:
        drawn = {fresh_nonce() for _ in range(64)}

        assert len(drawn) == 64
        assert all(is_well_formed_challenge(nonce) and len(nonce) == 43 for nonce in drawn)

    def test_the_body_digest_is_the_sha256_of_the_bytes_and_of_no_bytes_for_no_body(self) -> None:
        assert body_digest(b"") == EMPTY_BODY
        assert body_digest(RENAME_BODY) == hashlib.sha256(RENAME_BODY).hexdigest()
        assert body_digest(RENAME_BODY) != body_digest(b'{"label": "Renamed"}')


class TestSignedAuthorization:
    def test_parses_the_three_parts_and_writes_them_back(self) -> None:
        digest = reference_digest(TOKEN, "GET", "/api/state", EMPTY_BODY, CHALLENGE)

        parsed = SignedAuthorization.parse(f"{CHALLENGE}.{NONCE}.{digest}")

        assert parsed == SignedAuthorization(CHALLENGE, NONCE, digest)
        assert parsed.header() == f"Usagedash {CHALLENGE}.{NONCE}.{digest}"

    def test_signing_a_request_gives_a_header_that_parses_and_matches_the_digest(self) -> None:
        header = sign_request(TOKEN, "PATCH", "/api/accounts/7", CHALLENGE, RENAME_BODY, NONCE)
        drawn = sign_request(TOKEN, "PATCH", "/api/accounts/7", CHALLENGE, RENAME_BODY)

        scheme, _, credentials = header.partition(" ")
        parsed = SignedAuthorization.parse(credentials)
        assert scheme == "Usagedash"
        assert parsed is not None
        assert (parsed.challenge, parsed.nonce) == (CHALLENGE, NONCE)
        assert parsed.digest == reference_digest(TOKEN, "PATCH", "/api/accounts/7", body_digest(RENAME_BODY), CHALLENGE)
        assert TOKEN not in header
        own_nonce = SignedAuthorization.parse(drawn.partition(" ")[2])
        assert own_nonce is not None
        assert own_nonce.nonce != NONCE
        assert len(own_nonce.nonce) == 43

    @pytest.mark.parametrize("value", ["", "a" * 42, "a" * 257, "a" * 42 + "!", "a" * 42 + ".", "a" * 42 + " "])
    def test_a_challenge_or_nonce_of_the_wrong_shape_cannot_be_signed_over(self, value: str) -> None:
        with pytest.raises(ValueError, match="a challenge is 43 to 256"):
            sign_request(TOKEN, "GET", "/api/state", value)
        with pytest.raises(ValueError, match="a nonce is 43 to 256"):
            sign_request(TOKEN, "GET", "/api/state", CHALLENGE, nonce=value)
        assert not is_well_formed_challenge(value)

    @pytest.mark.parametrize(
        "credentials",
        [
            "",
            "abc",
            CHALLENGE,
            f"{CHALLENGE}.{'0' * 64}",
            f"{CHALLENGE}.{NONCE}.{'0' * 64}.extra",
            f"{'a' * 42}.{NONCE}.{'0' * 64}",
            f"{'a' * 257}.{NONCE}.{'0' * 64}",
            f"{'a' * 42}!.{NONCE}.{'0' * 64}",
            f"{CHALLENGE}.{'a' * 42}.{'0' * 64}",
            f"{CHALLENGE}.{'a' * 42}!.{'0' * 64}",
            f"{CHALLENGE}.{NONCE}.{'0' * 63}",
            f"{CHALLENGE}.{NONCE}.{'0' * 65}",
            f"{CHALLENGE}.{NONCE}.{'A' * 64}",
            f"{CHALLENGE}.{NONCE}.{'g' * 64}",
        ],
    )
    def test_anything_else_is_not_a_signature(self, credentials: str) -> None:
        assert SignedAuthorization.parse(credentials) is None

    def test_a_challenge_and_a_nonce_at_either_length_bound_parse(self) -> None:
        assert SignedAuthorization.parse(f"{'a' * 43}.{'b' * 43}.{'0' * 64}") is not None
        assert SignedAuthorization.parse(f"{'z' * 256}.{'y' * 256}.{'0' * 64}") is not None


class TestBodySealing:
    """Bodies travel sealed under a key both sides derive from the token, bound to the exchange they belong to."""

    CONTEXT = request_seal_context("PATCH", "/api/accounts/7", CHALLENGE, NONCE)

    def test_the_key_derivation_is_hkdf_sha256_rfc_5869_test_case_1(self) -> None:
        input_key = bytes.fromhex("0b" * 22)
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")

        assert hkdf_sha256(input_key, salt, info, 42).hex() == (
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        )

    @pytest.mark.parametrize("length", [0, -1, 255 * 32 + 1])
    def test_a_length_the_derivation_cannot_produce_is_refused(self, length: int) -> None:
        with pytest.raises(ValueError, match="1 to 8160 bytes"):
            hkdf_sha256(b"key", b"salt", b"info", length)

    def test_the_longest_derivation_is_produced_whole(self) -> None:
        assert len(hkdf_sha256(b"key", b"salt", b"info", 255 * 32)) == 255 * 32

    def test_the_seal_key_derives_from_the_token_with_this_package_s_salt_and_info(self) -> None:
        token = "synthetic-access-token-" + "x" * 20

        assert seal_key(token) == hkdf_sha256(token.encode(), b"usagedash", b"usagedash/sealed-body/v1", 32)
        assert len(seal_key(token)) == 32
        assert seal_key(token) != seal_key("synthetic-second-token-" + "y" * 20)

    def test_a_sealed_body_opens_under_the_same_key_for_the_same_context_only(self) -> None:
        sealer = BodySealer("synthetic-access-token-" + "x" * 20)
        other = BodySealer("synthetic-second-token-" + "y" * 20)
        sealed = sealer.seal(b'{"label": "Renamed"}', self.CONTEXT)

        assert sealer.unseal(sealed, self.CONTEXT) == b'{"label": "Renamed"}'
        assert b"Renamed" not in sealed
        with pytest.raises(SealedBodyError, match="does not open"):
            sealer.unseal(sealed, request_seal_context("PATCH", "/api/accounts/8", CHALLENGE, NONCE))
        with pytest.raises(SealedBodyError, match="does not open"):
            sealer.unseal(sealed, request_seal_context("PATCH", "/api/accounts/7", CHALLENGE, OTHER_NONCE))
        with pytest.raises(SealedBodyError, match="does not open"):
            sealer.unseal(sealed, answer_seal_context(CHALLENGE, NONCE, 200))
        with pytest.raises(SealedBodyError, match="does not open"):
            other.unseal(sealed, self.CONTEXT)

    def test_an_empty_body_seals_and_opens(self) -> None:
        sealer = BodySealer("synthetic-access-token-" + "x" * 20)
        sealed = sealer.seal(b"", self.CONTEXT)

        assert len(sealed) == SEAL_NONCE_BYTES + SEAL_TAG_BYTES
        assert sealer.unseal(sealed, self.CONTEXT) == b""

    def test_each_seal_uses_a_nonce_of_its_own(self) -> None:
        sealer = BodySealer("synthetic-access-token-" + "x" * 20)

        first = sealer.seal(b"same", self.CONTEXT)
        second = sealer.seal(b"same", self.CONTEXT)

        assert first != second
        assert first[:SEAL_NONCE_BYTES] != second[:SEAL_NONCE_BYTES]
        assert sealer.unseal(first, self.CONTEXT) == sealer.unseal(second, self.CONTEXT) == b"same"

    @pytest.mark.parametrize("altered_at", [0, SEAL_NONCE_BYTES, -1])
    def test_a_sealed_body_altered_anywhere_does_not_open(self, altered_at: int) -> None:
        sealer = BodySealer("synthetic-access-token-" + "x" * 20)
        sealed = bytearray(sealer.seal(b"data", self.CONTEXT))
        sealed[altered_at] ^= 0x01

        with pytest.raises(SealedBodyError, match="does not open"):
            sealer.unseal(bytes(sealed), self.CONTEXT)

    @pytest.mark.parametrize("length", [0, 1, SEAL_NONCE_BYTES + SEAL_TAG_BYTES - 1])
    def test_bytes_too_short_to_be_a_sealed_body_do_not_open(self, length: int) -> None:
        with pytest.raises(SealedBodyError, match="too short"):
            BodySealer("synthetic-access-token-" + "x" * 20).unseal(b"x" * length, self.CONTEXT)

    def test_the_contexts_are_one_part_per_line_in_the_order_the_page_builds_them(self) -> None:
        challenge = "synthetic-challenge-" + "c" * 23

        assert request_seal_context("PATCH", "/api/accounts/7?x=1", challenge, NONCE) == (
            f"PATCH\n/api/accounts/7?x=1\n{challenge}\n{NONCE}".encode()
        )
        assert answer_seal_context(challenge, NONCE, 404) == f"{challenge}\n{NONCE}\n404".encode()

    def test_the_sealed_media_type_is_this_package_s_own(self) -> None:
        assert SEALED_CONTENT_TYPE == "application/vnd.usagedash.sealed"


class TestRequestVerifier:
    def test_issues_a_fresh_well_formed_challenge_each_time_and_holds_it(self) -> None:
        verifier = RequestVerifier(TOKEN)

        first = verifier.issue()
        second = verifier.issue()

        assert is_well_formed_challenge(first)
        assert len(first) == 43
        assert first != second
        assert verifier.outstanding == 2

    def test_accepts_a_signature_over_an_issued_challenge_for_its_own_request_and_spends_the_challenge(self) -> None:
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier)

        outcome = verifier.verify(header, "GET", "/api/state", EMPTY_BODY)

        assert outcome is SignatureOutcome.ACCEPTED
        assert verifier.outstanding == 0
        assert verifier.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.UNKNOWN_CHALLENGE

    @pytest.mark.parametrize("scheme", ["Usagedash", "usagedash", "USAGEDASH"])
    def test_the_scheme_is_matched_whatever_its_case(self, scheme: str) -> None:
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier).replace("Usagedash", scheme, 1)

        assert verifier.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.ACCEPTED

    @pytest.mark.parametrize("authorization", [None, "", "Bearer " + TOKEN, "Basic abc", TOKEN, "Token x.y"])
    def test_a_header_of_another_scheme_or_none_is_not_a_signature(self, authorization: str | None) -> None:
        assert (
            RequestVerifier(TOKEN).verify(authorization, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.NOT_SIGNED
        )

    @pytest.mark.parametrize(
        "credentials", ["", "abc", CHALLENGE, f"{CHALLENGE}.{'0' * 64}", f"{CHALLENGE}.{NONCE}.{'0' * 63}", TOKEN]
    )
    def test_a_signature_of_the_wrong_shape_is_malformed(self, credentials: str) -> None:
        verifier = RequestVerifier(TOKEN)
        verifier.issue()

        assert (
            verifier.verify(f"Usagedash {credentials}", "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.MALFORMED
        )
        assert verifier.outstanding == 1

    def test_a_challenge_this_process_never_issued_is_refused_however_well_it_is_signed(self) -> None:
        verifier = RequestVerifier(TOKEN)
        header = sign_request(TOKEN, "GET", "/api/state", CHALLENGE)

        assert verifier.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.UNKNOWN_CHALLENGE

    def test_a_challenge_another_process_issued_is_refused(self) -> None:
        """A signature captured by a process that held the port fits that process and no other."""
        other_process = RequestVerifier(TOKEN)
        this_process = RequestVerifier(TOKEN)
        header = signed_for(other_process)

        assert this_process.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.UNKNOWN_CHALLENGE
        assert other_process.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.ACCEPTED

    @pytest.mark.parametrize(
        ("method", "target", "body_hash"),
        [
            ("POST", "/api/state", EMPTY_BODY),
            ("GET", "/api/tokens", EMPTY_BODY),
            ("GET", "/api/state?range=7d", EMPTY_BODY),
            ("GET", "/api/state/", EMPTY_BODY),
            ("get", "/api/state", EMPTY_BODY),
            ("GET", "/api/state", hashlib.sha256(RENAME_BODY).hexdigest()),
        ],
    )
    def test_a_signature_for_one_request_does_not_fit_another_and_leaves_its_challenge_held(
        self, method: str, target: str, body_hash: str
    ) -> None:
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier)

        assert verifier.verify(header, method, target, body_hash) is SignatureOutcome.WRONG
        assert verifier.outstanding == 1
        assert verifier.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.ACCEPTED

    def test_a_signature_for_a_target_with_a_query_fits_only_that_exact_query(self) -> None:
        target = "/api/tokens?range=24h&tz=America%2FNew_York"
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier, target=target)

        assert (
            verifier.verify(header, "GET", "/api/tokens?range=24h&tz=America/New_York", EMPTY_BODY)
            is SignatureOutcome.WRONG
        )
        assert (
            verifier.verify(header, "GET", "/api/tokens?tz=America%2FNew_York&range=24h", EMPTY_BODY)
            is SignatureOutcome.WRONG
        )
        assert verifier.verify(header, "GET", target, EMPTY_BODY) is SignatureOutcome.ACCEPTED

    def test_a_signature_covers_the_body_exactly_as_sent(self) -> None:
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier, "PATCH", "/api/accounts/7", RENAME_BODY)
        verdicts = [
            verifier.verify(header, "PATCH", "/api/accounts/7", body_digest(b'{"label":"Hijacked"}')),
            verifier.verify(header, "PATCH", "/api/accounts/7", body_digest(b'{"label": "Renamed"}')),
            verifier.verify(header, "PATCH", "/api/accounts/7", EMPTY_BODY),
            verifier.verify(header, "PATCH", "/api/accounts/7", body_digest(RENAME_BODY)),
        ]

        assert verdicts == [SignatureOutcome.WRONG] * 3 + [SignatureOutcome.ACCEPTED]

    def test_a_signature_by_another_token_is_wrong_and_the_challenge_stays_held(self) -> None:
        verifier = RequestVerifier(TOKEN)
        challenge = verifier.issue()

        outcome = verifier.verify(
            sign_request(OTHER_TOKEN, "GET", "/api/state", challenge), "GET", "/api/state", EMPTY_BODY
        )

        assert outcome is SignatureOutcome.WRONG
        assert verifier.outstanding == 1

    def test_a_tampered_digest_is_wrong(self) -> None:
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier)
        flipped = header[:-1] + ("0" if header[-1] != "0" else "1")

        assert verifier.verify(flipped, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.WRONG

    def test_a_challenge_is_spent_whatever_request_it_is_spent_on(self) -> None:
        verifier = RequestVerifier(TOKEN)
        challenge = verifier.issue()
        verifier.verify(sign_request(TOKEN, "GET", "/api/state", challenge), "GET", "/api/state", EMPTY_BODY)

        outcome = verifier.verify(
            sign_request(TOKEN, "GET", "/api/tokens", challenge), "GET", "/api/tokens", EMPTY_BODY
        )

        assert outcome is SignatureOutcome.UNKNOWN_CHALLENGE

    def test_a_challenge_lives_for_its_lifetime_on_the_monotonic_clock_and_then_is_forgotten(self) -> None:
        verifier, clock = verifier_and_clock()
        header = signed_for(verifier)
        clock.now += CHALLENGE_TTL_SECONDS

        still_held = verifier.verify(
            sign_request(TOKEN, "GET", "/api/tokens", verifier.issue()), "GET", "/api/tokens", EMPTY_BODY
        )
        assert still_held is SignatureOutcome.ACCEPTED
        assert verifier.outstanding == 1
        clock.now += 0.001

        assert verifier.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.UNKNOWN_CHALLENGE
        assert verifier.outstanding == 0

    def test_expired_challenges_are_dropped_when_a_new_one_is_issued(self) -> None:
        verifier, clock = verifier_and_clock()
        verifier.issue()
        verifier.issue()
        clock.now += CHALLENGE_TTL_SECONDS + 1

        verifier.issue()

        assert verifier.outstanding == 1

    def test_a_full_memory_forgets_the_oldest_challenge_and_nothing_else(self) -> None:
        verifier = RequestVerifier(TOKEN, capacity=2)
        first = verifier.issue()
        second = verifier.issue()

        third = verifier.issue()

        assert verifier.outstanding == 2
        assert verifier.verify(sign_request(TOKEN, "GET", "/api/state", first), "GET", "/api/state", EMPTY_BODY) is (
            SignatureOutcome.UNKNOWN_CHALLENGE
        )
        assert verifier.verify(sign_request(TOKEN, "GET", "/api/state", second), "GET", "/api/state", EMPTY_BODY) is (
            SignatureOutcome.ACCEPTED
        )
        assert verifier.verify(sign_request(TOKEN, "GET", "/api/state", third), "GET", "/api/state", EMPTY_BODY) is (
            SignatureOutcome.ACCEPTED
        )

    def test_expired_challenges_make_room_before_a_live_one_is_forgotten(self) -> None:
        verifier, clock = verifier_and_clock(capacity=2)
        verifier.issue()
        clock.now += CHALLENGE_TTL_SECONDS + 1
        live = verifier.issue()

        verifier.issue()

        assert verifier.outstanding == 2
        assert verifier.verify(sign_request(TOKEN, "GET", "/api/state", live), "GET", "/api/state", EMPTY_BODY) is (
            SignatureOutcome.ACCEPTED
        )

    def test_concurrent_presentations_of_one_signature_are_accepted_exactly_once(self) -> None:
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier)
        everyone_ready = threading.Barrier(CONCURRENT_CALLERS)

        def present() -> SignatureOutcome:
            everyone_ready.wait(timeout=5)
            return verifier.verify(header, "GET", "/api/state", EMPTY_BODY)

        with ThreadPoolExecutor(max_workers=CONCURRENT_CALLERS) as pool:
            outcomes = list(pool.map(lambda _: present(), range(CONCURRENT_CALLERS)))

        assert outcomes.count(SignatureOutcome.ACCEPTED) == 1
        assert outcomes.count(SignatureOutcome.UNKNOWN_CHALLENGE) == CONCURRENT_CALLERS - 1

    def test_the_wall_clock_is_not_consulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting the system clock back or forward must change nothing about which signatures are accepted."""
        verifier = RequestVerifier(TOKEN)
        header = signed_for(verifier)

        def no_wall_clock(*_args: object, **_kwargs: object) -> float:
            raise AssertionError("the verifier read the wall clock")

        monkeypatch.setattr(access.time, "time", no_wall_clock)

        assert verifier.verify(header, "GET", "/api/state", EMPTY_BODY) is SignatureOutcome.ACCEPTED


class TestTokenCreation:
    def test_the_token_is_created_once_in_a_file_only_the_user_can_read(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"

        token = ensure_access_token(data_dir)

        token_file = data_dir / "access-token"
        assert len(token) == 43
        assert token_file.read_text() == token
        assert token_file.stat().st_mode & 0o777 == 0o600
        assert data_dir.stat().st_mode & 0o777 == 0o700
        assert ensure_access_token(data_dir) == token
        assert leftover_temporary_files(data_dir) == []

    def test_an_existing_token_is_returned_without_being_rewritten(self, tmp_path: Path) -> None:
        token_file = tmp_path / "access-token"
        token_file.write_text(f"{TOKEN}\n")
        modified = token_file.stat().st_mtime_ns

        assert ensure_access_token(tmp_path) == TOKEN
        assert token_file.stat().st_mtime_ns == modified

    @pytest.mark.parametrize("existing_mode", [0o644, 0o666, 0o640])
    def test_an_existing_token_file_other_accounts_could_read_is_made_private_when_it_is_read(
        self, tmp_path: Path, existing_mode: int
    ) -> None:
        token_file = tmp_path / "access-token"
        token_file.write_text(TOKEN)
        token_file.chmod(existing_mode)

        assert ensure_access_token(tmp_path) == TOKEN
        assert token_file.stat().st_mode & 0o777 == 0o600

    def test_a_token_file_that_is_a_symbolic_link_is_refused_and_what_it_points_at_is_left_alone(
        self, tmp_path: Path
    ) -> None:
        """A token reached through a link would live outside the directory that was made private."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.write_text(TOKEN)
        elsewhere.chmod(0o644)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "access-token").symlink_to(elsewhere)

        with pytest.raises(ConfigError, match=r"cannot read or create .*access-token: OSError"):
            ensure_access_token(data_dir)

        assert elsewhere.stat().st_mode & 0o777 == 0o644

    def test_a_dangling_link_where_the_token_file_should_be_is_refused_and_nothing_is_created_through_it(
        self, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "access-token").symlink_to(tmp_path / "not-there")

        with pytest.raises(ConfigError, match=r"cannot read or create .*access-token"):
            ensure_access_token(data_dir)

        assert not (tmp_path / "not-there").exists()
        assert leftover_temporary_files(data_dir) == []

    @pytest.mark.skipif(sys.platform != "darwin", reason="access-control lists of this kind are a macOS feature")
    def test_an_existing_token_file_s_grant_to_other_accounts_is_removed_when_it_is_read(self, tmp_path: Path) -> None:
        token_file = tmp_path / "access-token"
        token_file.write_text(TOKEN)
        token_file.chmod(0o600)
        subprocess.run(["/bin/chmod", "+a", "everyone allow read", str(token_file)], check=True)

        assert ensure_access_token(tmp_path) == TOKEN

        listing = subprocess.run(["/bin/ls", "-le", str(token_file)], capture_output=True, text=True, check=True)
        assert listing.stdout.splitlines()[1:] == []

    def test_a_data_directory_other_accounts_could_reach_is_made_private(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_dir.chmod(0o755)

        ensure_access_token(data_dir)

        assert data_dir.stat().st_mode & 0o777 == 0o700

    def test_callers_racing_to_create_the_token_all_get_the_one_that_is_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every caller is held until all of them have found no token, so each one tries to create it."""
        all_found_no_token = threading.Barrier(CONCURRENT_CALLERS)
        real_create = access._create_token_if_absent

        def create_after_everyone_looked(path: Path) -> None:
            all_found_no_token.wait(timeout=5)
            real_create(path)

        monkeypatch.setattr(access, "_create_token_if_absent", create_after_everyone_looked)

        with ThreadPoolExecutor(max_workers=CONCURRENT_CALLERS) as pool:
            tokens = list(pool.map(lambda _: ensure_access_token(tmp_path), range(CONCURRENT_CALLERS)))

        assert set(tokens) == {(tmp_path / "access-token").read_text()}
        assert leftover_temporary_files(tmp_path) == []

    def test_a_caller_that_loses_the_race_keeps_the_winner_s_token(self, tmp_path: Path) -> None:
        winner = tmp_path / "access-token"
        winner.write_text(TOKEN)

        access._create_token_if_absent(winner)

        assert winner.read_text() == TOKEN
        assert leftover_temporary_files(tmp_path) == []

    @pytest.mark.parametrize(
        "content",
        [
            b"",
            b"\n",
            b"   \n",
            b"short-token",
            b"x" * 42,
            b"x" * 2000,
            TOKEN.encode() + b" " * 2000,
            TOKEN.encode() + b"\n" * access.MAX_TOKEN_FILE_BYTES + OTHER_TOKEN.encode(),
            TOKEN.encode() + b" " * (access.MAX_TOKEN_FILE_BYTES + 1 - len(TOKEN)),
            TOKEN.encode() + b"\n" + OTHER_TOKEN.encode(),
            TOKEN.encode()[:-1] + b"\xff",
            TOKEN.encode()[:20] + b"\x00" + TOKEN.encode()[20:],
        ],
    )
    def test_a_file_that_does_not_hold_one_well_formed_token_is_an_error_that_does_not_quote_it(
        self, tmp_path: Path, content: bytes
    ) -> None:
        (tmp_path / "access-token").write_bytes(content)

        with pytest.raises(ConfigError, match="holds no usable token; delete it") as raised:
            ensure_access_token(tmp_path)

        assert "short-token" not in str(raised.value)
        assert TOKEN not in str(raised.value)

    def test_a_token_followed_only_by_blank_space_up_to_the_size_limit_is_accepted(self, tmp_path: Path) -> None:
        """One byte more is refused: only the limit is read, so what follows it has not been looked at."""
        padded = TOKEN.encode() + b"\n" * (access.MAX_TOKEN_FILE_BYTES - len(TOKEN))
        (tmp_path / "access-token").write_bytes(padded)

        assert len(padded) == access.MAX_TOKEN_FILE_BYTES
        assert ensure_access_token(tmp_path) == TOKEN

    def test_a_token_file_that_cannot_be_read_is_a_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "access-token").mkdir()

        with pytest.raises(ConfigError, match=r"cannot read or create .*access-token: IsADirectoryError"):
            ensure_access_token(tmp_path)

    def test_a_failed_write_is_a_config_error_and_leaves_no_temporary_file_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_to_open(descriptor: int, *args: object, **kwargs: object) -> None:
            os.close(descriptor)
            raise OSError("disk full")

        monkeypatch.setattr(access.os, "fdopen", fail_to_open)

        with pytest.raises(ConfigError, match=r"cannot read or create .*access-token: OSError"):
            ensure_access_token(tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_a_token_file_removed_while_it_is_being_created_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(access, "_create_token_if_absent", lambda _path: None)

        with pytest.raises(ConfigError, match="disappeared while it was being created"):
            ensure_access_token(tmp_path)

    def test_a_data_directory_that_cannot_be_made_private_is_a_config_error(self, tmp_path: Path) -> None:
        in_the_way = tmp_path / "data"
        in_the_way.write_text("not a directory")

        with pytest.raises(ConfigError, match="a private directory"):
            ensure_access_token(in_the_way)


class TestTokenComparison:
    @pytest.mark.parametrize(
        ("presented", "matches"),
        [(TOKEN, True), ("", False), (TOKEN + "x", False), (TOKEN[:-1], False), (TOKEN.upper(), False), ("é", False)],
    )
    def test_only_the_exact_token_matches(self, presented: str, matches: bool) -> None:
        assert token_matches(presented, TOKEN) is matches


class TestDashboardOrigin:
    @pytest.mark.parametrize(
        ("url", "origin"),
        [
            ("http://127.0.0.1:8787/", "http://127.0.0.1:8787"),
            ("http://127.0.0.1:8787", "http://127.0.0.1:8787"),
            ("http://127.0.0.1:9000/", "http://127.0.0.1:9000"),
        ],
    )
    def test_the_origin_of_a_dashboard_url_on_the_loopback_address(self, url: str, origin: str) -> None:
        assert dashboard_origin(url) == origin

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "https://127.0.0.1:8787/",
            "http://127.0.0.1/",
            "http://localhost:8787/",
            "http://LOCALHOST:8787/",
            "http://127.0.0.2:8787/",
            "http://[::1]:8787/",
            "http://example.com:8787/",
            "http://127.0.0.1.example.com:8787/",
            "http://user@127.0.0.1:8787/",
            "http://127.0.0.1:8787/dashboard",
            "http://127.0.0.1:8787/?range=7d",
            "http://127.0.0.1:8787/#access=x",
            'http://127.0.0.1:8787/?a="><script>alert(1)</script>',
            "file:///tmp/open-dashboard.html",
        ],
    )
    def test_anything_but_plain_http_to_the_loopback_address_with_a_port_is_refused(self, url: str) -> None:
        """localhost is refused too: a browser may resolve it to ::1, where another process can listen."""
        with pytest.raises(ValueError, match=re.escape("must be http://127.0.0.1:<port>/")):
            dashboard_origin(url)


class TestBootstrapPage:
    def test_the_page_holds_the_token_once_as_a_javascript_string_and_never_in_a_url(self, tmp_path: Path) -> None:
        page = write_bootstrap_page(tmp_path / "data", DASHBOARD_URL, TOKEN)

        content = page.read_text()
        assert page == tmp_path / "data" / "open-dashboard.html"
        assert f'window.__USAGEDASH_TOKEN = "{TOKEN}";' in content
        assert 'window.__USAGEDASH_ORIGIN = "http://127.0.0.1:8787";' in content
        assert content.count(TOKEN) == 1
        assert "#access=" not in content
        assert "?access=" not in content
        assert 'http-equiv="refresh"' not in content
        assert "Bearer" not in content

    def test_the_page_loads_nothing_from_the_dashboard_but_carries_its_modules_inline(self) -> None:
        page = assemble_bootstrap_page(ORIGIN, TOKEN)

        assert 'src="/static/' not in page
        assert 'href="/static/' not in page
        assert page.count("<script") == 2
        blocks = inline_blocks(page)
        imported = DATA_URL.search(blocks["module"])
        assert imported is not None
        assert blocks["module"] == f'import "{imported.group(0)}";\n'
        app = decoded(imported.group(0))
        assert "from './" not in app
        assert "createAccess(window, { origin: API_ORIGIN })" in app
        for sibling in ("access.js", "failures.js", "format.js", "latest.js", "rename.js"):
            sibling_url = access._module_data_url((STATIC_DIR / sibling).read_text())
            assert app.count(f"from '{sibling_url}'") == 1, sibling
        assert (
            decoded(access._module_data_url((STATIC_DIR / "access.js").read_text()))
            == (STATIC_DIR / "access.js").read_text()
        )

    def test_the_module_is_the_served_module_with_only_its_sibling_imports_rewritten(self) -> None:
        served = (STATIC_DIR / "app.js").read_text()
        page = assemble_bootstrap_page(ORIGIN, TOKEN)
        inlined = decoded(DATA_URL.search(inline_blocks(page)["module"]).group(0))

        restored = re.sub(
            r"from 'data:text/javascript;base64,([A-Za-z0-9+/=]+)'",
            lambda match: (
                "from './"
                + next(
                    name
                    for name in ("access.js", "failures.js", "format.js", "latest.js", "rename.js")
                    if base64.b64decode(match.group(1)).decode() == (STATIC_DIR / name).read_text()
                )
                + "'"
            ),
            inlined,
        )

        assert restored == served

    def test_the_stylesheet_is_inlined(self) -> None:
        page = assemble_bootstrap_page(ORIGIN, TOKEN)

        assert inline_blocks(page)["style"] == (STATIC_DIR / "styles.css").read_text()
        assert '<link rel="stylesheet"' not in page

    def test_the_content_security_policy_allows_the_inline_blocks_the_data_modules_and_the_dashboard_only(self) -> None:
        page = assemble_bootstrap_page(ORIGIN, TOKEN)
        blocks = inline_blocks(page)

        directives = dict(directive.strip().split(" ", 1) for directive in blocks["policy"].split(";"))

        assert directives["default-src"] == "'none'"
        assert directives["connect-src"] == ORIGIN
        assert directives["base-uri"] == "'none'"
        assert directives["form-action"] == "'none'"
        assert (
            directives["script-src"]
            == f"'sha256-{csp_hash_of(blocks['plain'])}' 'sha256-{csp_hash_of(blocks['module'])}' data:"
        )
        assert directives["style-src"] == f"'sha256-{csp_hash_of(blocks['style'])}'"
        assert "unsafe-inline" not in blocks["policy"]
        assert "http:" not in directives["script-src"]

    def test_the_markup_is_the_served_page_s_with_its_two_tags_replaced(self) -> None:
        page = assemble_bootstrap_page(ORIGIN, TOKEN)

        without_head = re.sub(
            r'<meta http-equiv="Content-Security-Policy" content="[^"]*" />\n\s*<style>.*?</style>',
            '<link rel="stylesheet" href="/static/styles.css" />',
            page,
            flags=re.S,
        )
        restored = re.sub(
            r"<script>.*?</script>\n\s*<script type=\"module\">.*?</script>",
            '<script type="module" src="/static/app.js"></script>',
            without_head,
            flags=re.S,
        )

        assert restored == (STATIC_DIR / "index.html").read_text()

    def test_a_string_that_could_end_a_script_element_is_escaped(self) -> None:
        assert access._javascript_string("</script><!--") == '"\\u003c/script\\u003e\\u003c!--"'
        assert access._javascript_string(TOKEN) == f'"{TOKEN}"'

    def test_the_page_is_readable_only_by_the_user(self, tmp_path: Path) -> None:
        page = write_bootstrap_page(tmp_path / "data", DASHBOARD_URL, TOKEN)

        assert page.stat().st_mode & 0o777 == 0o600
        assert page.parent.stat().st_mode & 0o777 == 0o700
        assert leftover_temporary_files(page.parent) == []

    def test_writing_again_replaces_the_page(self, tmp_path: Path) -> None:
        write_bootstrap_page(tmp_path, DASHBOARD_URL, TOKEN)
        page = write_bootstrap_page(tmp_path, "http://127.0.0.1:9000/", OTHER_TOKEN)

        content = page.read_text()
        assert OTHER_TOKEN in content
        assert TOKEN not in content
        assert "connect-src http://127.0.0.1:9000;" in content
        assert "8787" not in content
        assert page.stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize("token", ["", "short", 'a"><script>alert(1)</script>' + "x" * 40, TOKEN + "\n"])
    def test_only_a_well_formed_token_is_ever_written_into_the_page(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ValueError, match="can only carry a token created by ensure_access_token"):
            write_bootstrap_page(tmp_path, DASHBOARD_URL, token)

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("url", ["https://127.0.0.1:8787/", "http://example.com:8787/", "http://127.0.0.1:8787/x"])
    def test_a_url_that_is_not_the_dashboard_s_is_refused_and_nothing_is_written(
        self, tmp_path: Path, url: str
    ) -> None:
        with pytest.raises(ValueError, match=re.escape("must be http://127.0.0.1:<port>/")):
            write_bootstrap_page(tmp_path, url, TOKEN)

        assert list(tmp_path.iterdir()) == []

    def test_a_page_that_cannot_be_put_in_place_is_a_config_error_and_leaves_no_temporary_file(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "open-dashboard.html").mkdir()

        with pytest.raises(ConfigError, match=r"cannot write .*open-dashboard\.html: "):
            write_bootstrap_page(tmp_path, DASHBOARD_URL, TOKEN)

        assert leftover_temporary_files(tmp_path) == []

    def test_a_page_that_cannot_be_written_at_all_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_to_open(descriptor: int, *args: object, **kwargs: object) -> None:
            os.close(descriptor)
            raise OSError("disk full")

        monkeypatch.setattr(access.os, "fdopen", fail_to_open)

        with pytest.raises(ConfigError, match=r"cannot write .*open-dashboard\.html: OSError"):
            write_bootstrap_page(tmp_path, DASHBOARD_URL, TOKEN)

        assert list(tmp_path.iterdir()) == []


class TestPageAssembly:
    """The assembler against synthetic static files, so each shape it refuses is exercised."""

    def test_a_module_imported_from_two_places_is_inlined_once_under_one_url(self, tmp_path: Path) -> None:
        static = synthetic_static_dir(
            tmp_path,
            {
                "app.js": "import { a } from './a.js';\nimport { b } from './b.js';\na(); b();\n",
                "a.js": "import { c } from './c.js';\nexport const a = c;\n",
                "b.js": "import { c } from './c.js';\nexport const b = c;\n",
                "c.js": "export const c = () => 1;\n",
            },
        )

        page = assemble_bootstrap_page(ORIGIN, TOKEN, static)

        app = decoded(DATA_URL.search(inline_blocks(page)["module"]).group(0))
        a_url, b_url = DATA_URL.findall(app)
        c_in_a = DATA_URL.findall(decoded(f"data:text/javascript;base64,{a_url}"))
        c_in_b = DATA_URL.findall(decoded(f"data:text/javascript;base64,{b_url}"))
        assert c_in_a == c_in_b
        assert decoded(f"data:text/javascript;base64,{c_in_a[0]}") == "export const c = () => 1;\n"

    def test_a_multi_line_import_is_rewritten(self, tmp_path: Path) -> None:
        static = synthetic_static_dir(
            tmp_path,
            {
                "app.js": "import {\n  one,\n  two,\n} from './pair.js';\n",
                "pair.js": "export const one = 1, two = 2;\n",
            },
        )

        app = decoded(DATA_URL.search(inline_blocks(assemble_bootstrap_page(ORIGIN, TOKEN, static))["module"]).group(0))

        assert app.startswith("import {\n  one,\n  two,\n} from 'data:text/javascript;base64,")
        assert "./pair.js" not in app

    def test_a_missing_module_is_an_assembly_error(self, tmp_path: Path) -> None:
        static = synthetic_static_dir(tmp_path, {"app.js": "import { x } from './missing.js';\n"})

        with pytest.raises(PageAssemblyError, match=re.escape("cannot read the page module missing.js")):
            assemble_bootstrap_page(ORIGIN, TOKEN, static)

    def test_modules_that_import_each_other_are_an_assembly_error(self, tmp_path: Path) -> None:
        static = synthetic_static_dir(
            tmp_path,
            {
                "app.js": "import './a.js'; import { a } from './a.js';\n",
                "a.js": "import { b } from './b.js';\n",
                "b.js": "import { a } from './a.js';\n",
            },
        )

        with pytest.raises(PageAssemblyError, match=re.escape("a.js imports itself through app.js -> a.js -> b.js")):
            assemble_bootstrap_page(ORIGIN, TOKEN, static)

    @pytest.mark.parametrize(
        "source",
        [
            "import './side-effect.js';\n",
            'import { x } from "./double.js";\n',
            "const x = await import('./dynamic.js');\n",
            "import { x } from '../outside.js';\n",
            "import { x } from './sub/dir.js';\n",
        ],
    )
    def test_an_import_the_page_cannot_inline_is_an_assembly_error(self, tmp_path: Path, source: str) -> None:
        static = synthetic_static_dir(tmp_path, {"app.js": source})

        with pytest.raises(PageAssemblyError, match=re.escape("app.js imports a path the page cannot inline")):
            assemble_bootstrap_page(ORIGIN, TOKEN, static)

    @pytest.mark.parametrize(
        "shell",
        [
            SHELL.replace('<script type="module" src="/static/app.js"></script>', ""),
            SHELL.replace('<link rel="stylesheet" href="/static/styles.css" />', ""),
            SHELL + '<script type="module" src="/static/app.js"></script>',
        ],
    )
    def test_a_shell_without_exactly_one_stylesheet_tag_and_one_module_tag_is_an_assembly_error(
        self, tmp_path: Path, shell: str
    ) -> None:
        static = synthetic_static_dir(tmp_path, {"app.js": "export {};\n"}, shell=shell)

        with pytest.raises(PageAssemblyError, match=re.escape("index.html does not hold exactly one")):
            assemble_bootstrap_page(ORIGIN, TOKEN, static)

    def test_a_stylesheet_that_would_end_its_element_is_an_assembly_error(self, tmp_path: Path) -> None:
        static = synthetic_static_dir(
            tmp_path, {"app.js": "export {};\n"}, stylesheet="body{} </style><script>1</script>"
        )

        with pytest.raises(PageAssemblyError, match=re.escape("styles.css cannot be inlined")):
            assemble_bootstrap_page(ORIGIN, TOKEN, static)

    def test_a_missing_shell_is_an_assembly_error(self, tmp_path: Path) -> None:
        with pytest.raises(PageAssemblyError, match="cannot read the page shell: FileNotFoundError"):
            assemble_bootstrap_page(ORIGIN, TOKEN, tmp_path / "nowhere")
