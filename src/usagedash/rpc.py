"""Talk newline-delimited JSON to a vendor CLI running in its own headless protocol mode.

Standing logins are read this way: the CLI authenticates with its own stored session and refreshes its own
tokens, so this package never uses or writes a token for those accounts.
"""

import contextlib
import json
import os
import select
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from usagedash.errors import ConfigError, ProviderRequestError
from usagedash.executables import checked_command, execution_environment

READ_CHUNK_BYTES = 65536
MAX_LINE_BYTES = 8 * 1024 * 1024
"""The longest line a vendor CLI may send. Every message read here is a few kilobytes; a Claude session's
first message, which lists the tools it has, is the largest and still far below this."""
POLL_SECONDS = 0.25
MIN_ERROR_CODE = -(2**31)
MAX_ERROR_CODE = 2**31 - 1
TERMINATE_GRACE_SECONDS = 5

Message = dict[str, Any]


class JsonLineProcess:
    """A child process that exchanges one JSON object per line on stdin and stdout.

    Use as a context manager; the process is always stopped on exit, including after an error or timeout.
    Prepare optional private helpers here so failures stay within each source's read boundary.
    ``fallback_name`` is the vendor command looked up when the recorded executable file is gone.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        service: str,
        helper_directory: Path | None = None,
        fallback_name: str | None = None,
    ):
        self._service = service
        self._buffer = bytearray()
        try:
            environment = execution_environment(env, cwd, helper_directory=helper_directory)
            self._process = subprocess.Popen(
                checked_command(command, environment, cwd, fallback_name=fallback_name),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                cwd=cwd,
            )
        except ConfigError as error:
            raise ProviderRequestError(f"cannot start {service}: {error}") from error
        except OSError as error:
            raise ProviderRequestError(f"could not start {service}: {type(error).__name__}") from error
        stdin, stdout = self._process.stdin, self._process.stdout
        if stdin is None or stdout is None:
            # Why: Popen always supplies both streams when asked for pipes. The check exists to give the
            # two attributes below a non-optional type; if it ever fired, the child must not be left running.
            self._process.kill()
            raise ProviderRequestError(f"{service} started without its input and output streams")
        self._stdin = stdin
        self._stdout = stdout

    def __enter__(self) -> Self:
        return self

    @property
    def pid(self) -> int:
        """The child's process id, which a CLI writes on each record of its own log."""
        return self._process.pid

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        # Why: closing flushes anything still buffered. If the child has already exited that flush fails,
        # but send() has already reported it, and raising here would replace that error with this one.
        with contextlib.suppress(BrokenPipeError):
            self._stdin.close()
        self._stdout.close()
        self._process.terminate()
        try:
            self._process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def send(self, message: Mapping[str, Any]) -> None:
        """Write one message.

        Raises:
            ProviderRequestError: if the process has already closed its input.
        """
        try:
            self._stdin.write(json.dumps(message).encode("utf-8") + b"\n")
            self._stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise ProviderRequestError(f"{self._service} exited before accepting a request") from error

    def receive(self, matches: Callable[[Message], bool], deadline: float) -> Message:
        """Return the first message satisfying ``matches``, skipping notifications and non-JSON output.

        ``deadline`` is a ``time.monotonic()`` instant.

        What has been read but not yet ended by a newline is held in one growing buffer, never copied
        whole on each read, and it may not grow past ``MAX_LINE_BYTES``: a CLI that writes without end
        on one line would otherwise take all the memory there is before the deadline came.

        Raises:
            ProviderRequestError: if the process exits, the deadline passes, or a line grows past
                ``MAX_LINE_BYTES`` first.
        """
        stdout = self._stdout
        while True:
            while (end := self._buffer.find(b"\n")) != -1:
                self._refuse_a_line_of(end)
                line = bytes(self._buffer[:end])
                del self._buffer[: end + 1]
                message = _parse_object(line)
                if message is not None and matches(message):
                    return message
            self._refuse_a_line_of(len(self._buffer))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderRequestError(f"{self._service} did not answer in time")
            ready, _, _ = select.select([stdout], [], [], min(POLL_SECONDS, remaining))
            if not ready:
                continue
            chunk = os.read(stdout.fileno(), READ_CHUNK_BYTES)
            if chunk == b"":
                raise ProviderRequestError(f"{self._service} exited before answering")
            self._buffer += chunk

    def _refuse_a_line_of(self, length: int) -> None:
        """Refuse a line of ``length`` bytes, ended or not, that is longer than ``MAX_LINE_BYTES``.

        Checked for every line that ends as well as for what has not ended yet: a line whose newline
        arrives in the read that takes it past the limit is otherwise ended, and parsed, before any
        check of what is left.

        Raises:
            ProviderRequestError: if it is longer.
        """
        if length > MAX_LINE_BYTES:
            raise ProviderRequestError(f"{self._service} sent a line longer than {MAX_LINE_BYTES} bytes")


def _parse_object(line: bytes) -> Message | None:
    if line.strip() == b"":
        return None
    try:
        parsed = json.loads(line)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def jsonrpc_result(process: JsonLineProcess, request: Mapping[str, Any], deadline: float, *, service: str) -> Message:
    """Send a JSON-RPC request and return its ``result`` object.

    Raises:
        ProviderRequestError: if the peer answers with an error, a non-object result, or not at all.
    """
    request_id = request["id"]
    process.send(request)
    response = process.receive(
        lambda message: message.get("id") == request_id and ("result" in message or "error" in message), deadline
    )
    if response.get("error") is not None:
        raise ProviderRequestError(f"{service} rejected {request['method']} ({_error_code(response['error'])})")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ProviderRequestError(f"{service} returned no result for {request['method']}")
    return result


def _error_code(error: object) -> str:
    """Describe a JSON-RPC error by its numeric code and nothing else.

    The error's message is the peer's own text and can quote an account, a path, or a credential. What
    is raised here is stored against its source and shown on the page, so it carries the code only.
    """
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, bool) or not isinstance(code, int) or not MIN_ERROR_CODE <= code <= MAX_ERROR_CODE:
        return "no error code given"
    return f"error code {code}"


def child_environment(
    base: Mapping[str, str], vendor_prefixes: tuple[str, ...], settings: Mapping[str, str]
) -> dict[str, str]:
    """Build the environment a vendor CLI runs in to act as the login stored in one home.

    Everything inherited is kept except the vendor's own variables, which are dropped by prefix, and
    ``settings`` are then applied. A vendor variable inherited from the shell, or from an enclosing
    session of the same CLI, can name another home, another config, an API key, or a credential. The
    CLI would then act as something other than the login in the home it is pointed at, and its answer
    would be recorded against the identity read from that home: another account's usage under this
    account's name.
    """
    environment = {name: value for name, value in base.items() if not name.startswith(vendor_prefixes)}
    environment.update(settings)
    return environment
