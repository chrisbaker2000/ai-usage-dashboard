"""A stand-in for a vendor CLI's headless protocol mode, driven by environment variables.

``FAKE_PROTOCOL`` selects the dialect (``claude``, ``codex``, ``grok``). ``FAKE_SCENARIO`` selects the
behavior. ``FAKE_RECORD`` names a file where every received message and the relevant environment are
appended, so a test can assert what the dashboard actually sent. ``FAKE_WORKSPACE`` is the ChatGPT
workspace id the Codex dialect reports as logged in, sent exactly as given, blank values included.
``FAKE_WORKSPACE_ABSENT`` makes it report no workspace id at all. In the ``usage_answer_from_env`` scenario the
Claude dialect answers ``get_usage`` with the JSON object in ``FAKE_USAGE_ANSWER``.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

PROTOCOL = os.environ.get("FAKE_PROTOCOL", "")
SCENARIO = os.environ.get("FAKE_SCENARIO", "ok")
RECORD = os.environ.get("FAKE_RECORD")
WORKSPACE = None if os.environ.get("FAKE_WORKSPACE_ABSENT") else os.environ.get("FAKE_WORKSPACE", "workspace-b")

CLAUDE_RATE_LIMITS = {
    "five_hour": {"utilization": 6, "resets_at": "2026-10-03T18:20:00+00:00"},
    "limits": [
        {"kind": "session", "percent": 6, "resets_at": "2026-10-03T18:20:00+00:00", "scope": None},
        {"kind": "weekly_all", "percent": 54, "resets_at": "2026-10-06T15:00:00+00:00", "scope": None},
    ],
}
CODEX_SNAPSHOT = {
    "limitId": "codex",
    "limitName": None,
    "primary": {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 1791580444},
    "secondary": {"usedPercent": 15, "windowDurationMins": 300, "resetsAt": 1791060000},
    "credits": {"hasCredits": True, "unlimited": False, "balance": "500.5"},
    "planType": "team",
    "rateLimitReachedType": None,
}
GROK_CONFIG = {
    "creditUsagePercent": 31.0,
    "currentPeriod": {
        "type": "USAGE_PERIOD_TYPE_WEEKLY",
        "start": "2026-09-27T16:00:00+00:00",
        "end": "2026-10-04T16:00:00+00:00",
    },
    "onDemandCap": {"val": 0},
    "onDemandUsed": {"val": 0},
    "prepaidBalance": {"val": 0},
}
# The answer for an account that has used nothing this period, in the shape seen live on 2026-10-03.
GROK_RESULT_WITH_NOTHING_USED = {
    "config": {
        "currentPeriod": {
            "type": "USAGE_PERIOD_TYPE_WEEKLY",
            "start": "2026-10-01T14:49:41.842876+00:00",
            "end": "2026-10-08T14:49:41.842876+00:00",
        },
        "onDemandCap": {"val": 0},
        "onDemandUsed": {"val": 0},
        "prepaidBalance": {"val": 11970},
        "isUnifiedBillingUser": True,
        "billingPeriodStart": "2026-10-01T14:49:41.842876+00:00",
        "billingPeriodEnd": "2026-10-08T14:49:41.842876+00:00",
    },
    "subscription_tier": "SuperGrok Heavy",
}


def stamped_with_this_process(logged: str) -> str:
    """Give each JSON record that names no process this one's id, as the real CLI writes on its own records."""
    lines = []
    for line in logged.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except ValueError:
            lines.append(line)
            continue
        if isinstance(record, dict) and "pid" not in record:
            record["pid"] = os.getpid()
            line = json.dumps(record) + ("\n" if line.endswith("\n") else "")
        lines.append(line)
    return "".join(lines)


def record(entry: dict) -> None:
    if RECORD:
        with open(RECORD, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def reply(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def change_the_login() -> None:
    """Replace or remove a file while a request is being answered, as a login changing mid-read does."""
    target = os.environ.get("FAKE_CHANGE_FILE")
    if target is None:
        return
    content = os.environ.get("FAKE_CHANGE_CONTENT")
    if content is None:
        Path(target).unlink()
    else:
        Path(target).write_text(content)


def handle_claude(message: dict) -> None:
    request = message.get("request", {})
    request_id = message.get("request_id")
    subtype = request.get("subtype")

    def respond(body: dict | None, outcome: str = "success", error: str | None = None) -> None:
        reply(
            {
                "type": "control_response",
                "response": {"subtype": outcome, "request_id": request_id, "response": body, "error": error},
            }
        )

    if subtype == "initialize":
        reply({"type": "system", "subtype": "hook_started"})
        respond({"account": {"organization": "Synthetic Org", "subscriptionType": "Claude Max"}})
    elif subtype == "get_usage":
        change_the_login()
        if SCENARIO == "usage_rejected":
            respond(None, outcome="error", error="not signed in")
        elif SCENARIO == "usage_without_body":
            respond(None)
        elif SCENARIO == "no_limits":
            respond({"subscription_type": None, "rate_limits_available": False, "rate_limits": None})
        elif SCENARIO == "usage_answer_from_env":
            respond(json.loads(os.environ["FAKE_USAGE_ANSWER"]))
        else:
            respond({"subscription_type": "max", "rate_limits_available": True, "rate_limits": CLAUDE_RATE_LIMITS})


def handle_codex(message: dict, state: dict) -> None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return
    if method == "initialize":
        reply({"id": request_id, "result": {"codexHome": os.environ.get("CODEX_HOME")}})
    elif method == "account/read":
        reply({"method": "account/updated", "params": {}})
        if SCENARIO == "logged_out":
            reply({"id": request_id, "result": {"account": None, "requiresOpenaiAuth": True}})
            return
        refreshing = bool(message.get("params", {}).get("refreshToken"))
        if SCENARIO == "reshaped_account" or (SCENARIO == "reshaped_account_on_refresh" and refreshing):
            reply({"id": request_id, "result": {"account": {"type": 5}}})
            return
        if SCENARIO == "account_without_result":
            reply({"id": request_id, "result": None})
            return
        if SCENARIO == "account_error":
            reply({"id": request_id, "error": {"code": -32000, "message": "account service unavailable"}})
            return
        if refreshing:
            state["refreshed"] = True
        reply(
            {
                "id": request_id,
                "result": {
                    "account": {"type": "chatgpt", "email": "user-b@example.test", "planType": "team"},
                    "workspaceRouting": None if WORKSPACE is None else {"chatgptAccountId": WORKSPACE},
                },
            }
        )
    elif method == "account/rateLimits/read":
        if SCENARIO in ("stale_token", "reshaped_account_on_refresh") and not state.get("refreshed"):
            reply({"id": request_id, "error": {"code": -32000, "message": "token expired"}})
            return
        if SCENARIO == "always_failing":
            reply({"id": request_id, "error": {"code": -32000, "message": "backend unavailable"}})
            return
        snapshot = dict(CODEX_SNAPSHOT)
        if SCENARIO == "limit_reached":
            snapshot["rateLimitReachedType"] = "rate_limit_reached"
        reply(
            {
                "id": request_id,
                "result": {
                    "accountId": WORKSPACE,
                    "rateLimits": snapshot,
                    "rateLimitsByLimitId": {"codex": snapshot},
                },
            }
        )


def handle_grok(message: dict) -> None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        reply({"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": 1}})
        reply({"jsonrpc": "2.0", "method": "_x.ai/models/update", "params": {}})
    elif method == "_x.ai/billing":
        change_the_login()
        logged = os.environ.get("FAKE_GROK_LOG")
        if logged is not None:
            log = Path(os.environ["GROK_HOME"]) / "logs" / "unified.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as handle:
                handle.write(stamped_with_this_process(logged).encode())
        if SCENARIO == "billing_error":
            error = json.loads(os.environ.get("FAKE_BILLING_ERROR", "null"))
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": error or {"code": -32603, "message": "billing unavailable"},
                }
            )
        elif SCENARIO == "error_without_message":
            reply({"jsonrpc": "2.0", "id": request_id, "error": "nope"})
        elif SCENARIO == "null_result":
            reply({"jsonrpc": "2.0", "id": request_id, "result": None})
        elif SCENARIO == "reshaped":
            reply({"jsonrpc": "2.0", "id": request_id, "result": {"config": {"unexpected": True}}})
        elif SCENARIO == "nothing_used":
            reply({"jsonrpc": "2.0", "id": request_id, "result": GROK_RESULT_WITH_NOTHING_USED})
        else:
            reply({"jsonrpc": "2.0", "id": request_id, "result": {"config": GROK_CONFIG}})


def main() -> None:
    record(
        {
            "argv": sys.argv[1:],
            "env": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("CLAUDE", "ANTHROPIC", "CODEX_", "OPENAI_", "GROK_", "XAI_"))
            },
            "user": os.environ.get("USER"),
        }
    )
    if SCENARIO == "exit_immediately":
        return
    if SCENARIO == "silent":
        time.sleep(30)
        return
    if SCENARIO == "endless_line":
        chunk = "x" * 65536
        while True:
            sys.stdout.write(chunk)
            sys.stdout.flush()
    if SCENARIO == "ignores_termination":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sys.stdout.write('{"ready": true}\n')
        sys.stdout.flush()
        time.sleep(30)
        return
    sys.stdout.write('startup banner that is not JSON\n\n   \n[1, 2]\n"text"\n')
    sys.stdout.flush()
    state: dict = {}
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        record({"received": message})
        if PROTOCOL == "claude":
            handle_claude(message)
        elif PROTOCOL == "codex":
            handle_codex(message, state)
        elif PROTOCOL == "grok":
            handle_grok(message)


main()
