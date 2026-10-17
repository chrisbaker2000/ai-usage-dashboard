"""Factory functions for synthetic provider payloads and credential files, in the shapes the CLIs use."""

import base64
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tests.factories import PERSONA_EMAIL, WEEK


def make_jwt(claims: dict[str, Any]) -> str:
    """Build an unsigned JWT-shaped string carrying ``claims``."""
    return jwt_with_raw_payload(json.dumps(claims))


def jwt_with_raw_payload(payload: str) -> str:
    """Build an unsigned JWT-shaped string whose payload is exactly ``payload``, which need not be valid JSON."""

    def encode(text: str) -> str:
        return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")

    return f"{encode(json.dumps({'alg': 'none'}))}.{encode(payload)}.signature"


def claude_keychain_blob(
    access_token: str = "synthetic-claude-token",
    expires_at: datetime | None = None,
    subscription_type: str = "max",
    rate_limit_tier: str = "default_claude_max_20x",
) -> str:
    oauth: dict[str, Any] = {
        "accessToken": access_token,
        "refreshToken": "synthetic-refresh",
        "subscriptionType": subscription_type,
        "rateLimitTier": rate_limit_tier,
    }
    if expires_at is not None:
        oauth["expiresAt"] = int(expires_at.timestamp() * 1000)
    return json.dumps({"claudeAiOauth": oauth, "mcpOAuth": {}})


def write_claude_account_file(
    path: Path,
    organization_uuid: str = "org-a",
    organization_name: str = f"{PERSONA_EMAIL}'s Organization",
    email: str = PERSONA_EMAIL,
) -> None:
    path.write_text(
        json.dumps(
            {
                "oauthAccount": {
                    "organizationUuid": organization_uuid,
                    "organizationName": organization_name,
                    "emailAddress": email,
                }
            }
        )
    )


def claude_profile_payload(
    organization_uuid: str = "org-a",
    organization_name: str = f"{PERSONA_EMAIL}'s Organization",
    organization_type: str = "claude_max",
    rate_limit_tier: str = "default_claude_max_20x",
    email: str = PERSONA_EMAIL,
) -> dict[str, Any]:
    return {
        "account": {"email": email, "uuid": "acct-a"},
        "organization": {
            "uuid": organization_uuid,
            "name": organization_name,
            "organization_type": organization_type,
            "rate_limit_tier": rate_limit_tier,
        },
    }


NO_DOLLAR_AMOUNTS: dict[str, Any] = {
    "limit_dollars": None,
    "used_dollars": None,
    "remaining_dollars": None,
    "locked_reason": None,
}
"""What the usage API sends in every limit entry that is not a credit pool (shape seen live 2026-10-03)."""


def claude_usage_payload(now: datetime, session_percent: float = 12, weekly_percent: float = 52) -> dict[str, Any]:
    session_reset = (now + timedelta(hours=4, seconds=59, microseconds=887958)).isoformat()
    weekly_reset = (now + timedelta(days=3)).isoformat()
    return {
        "five_hour": {"utilization": session_percent, "resets_at": session_reset, **NO_DOLLAR_AMOUNTS},
        "seven_day": {"utilization": weekly_percent, "resets_at": weekly_reset, **NO_DOLLAR_AMOUNTS},
        "seven_day_opus": None,
        "extra_usage": {"is_enabled": False, "monthly_limit": 10000, "used_credits": 0.0, "decimal_places": 2},
        "promo_pool": {
            "utilization": 0.0,
            "resets_at": (now + timedelta(days=30)).isoformat(),
            "limit_dollars": 250,
            "used_dollars": 12.5,
            "remaining_dollars": 237.5,
            "locked_reason": None,
        },
        "limits": [
            {
                "kind": "session",
                "group": "session",
                "percent": session_percent,
                "resets_at": session_reset,
                "scope": None,
            },
            {
                "kind": "weekly_all",
                "group": "weekly",
                "percent": weekly_percent,
                "resets_at": weekly_reset,
                "scope": None,
            },
            {
                "kind": "weekly_scoped",
                "group": "weekly",
                "percent": 42,
                "resets_at": weekly_reset,
                "scope": {"model": {"id": None, "display_name": "Fable"}, "surface": None},
            },
        ],
        "model_scoped": [{"display_name": "Fable", "utilization": 42, "resets_at": weekly_reset}],
    }


def codex_auth_file(
    path: Path,
    account_id: str = "workspace-a",
    email: str = PERSONA_EMAIL,
    plan: str = "pro",
    access_expires_at: datetime | None = None,
    access_token: str | None = None,
) -> None:
    id_token = make_jwt(
        {
            "email": email,
            "https://api.openai.com/auth": {"chatgpt_plan_type": plan, "chatgpt_account_id": account_id},
        }
    )
    if access_token is None:
        claims = {} if access_expires_at is None else {"exp": int(access_expires_at.timestamp())}
        access_token = make_jwt(claims)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "id_token": id_token,
                    "access_token": access_token,
                    "refresh_token": "synthetic-refresh",
                    "account_id": account_id,
                },
            }
        )
    )


def codex_usage_payload(
    now: datetime,
    primary_percent: float = 38,
    primary_seconds: int = WEEK,
    secondary: dict[str, Any] | None = None,
    account_id: str = "workspace-a",
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "email": PERSONA_EMAIL,
        "plan_type": "pro",
        "rate_limit": {
            "allowed": True,
            "limit_reached": False,
            "primary_window": {
                "used_percent": primary_percent,
                "limit_window_seconds": primary_seconds,
                "reset_at": int((now + timedelta(days=6)).timestamp()),
            },
            "secondary_window": secondary,
        },
        "code_review_rate_limit": None,
        "additional_rate_limits": None,
        "credits": {"has_credits": True, "unlimited": False, "balance": "61180.0843600000"},
    }


def grok_auth_file(
    path: Path,
    user_id: str = "grok-user-a",
    email: str = PERSONA_EMAIL,
    create_time: datetime | None = None,
    expires_at: datetime | None = None,
) -> None:
    entry: dict[str, Any] = {
        "key": "synthetic-grok-token",
        "refresh_token": "synthetic-refresh",
        "user_id": user_id,
        "email": email,
        "principal_id": user_id,
    }
    if create_time is not None:
        entry["create_time"] = create_time.isoformat()
    if expires_at is not None:
        entry["expires_at"] = expires_at.isoformat()
    path.write_text(json.dumps({"https://auth.example.test::client": entry}))


def grok_billing_log_line(
    timestamp: datetime, now: datetime, percent: float = 85.0, tier: str = "SuperGrok Heavy", pid: int | None = None
) -> bytes:
    """A billing record as the Grok CLI logs it. ``pid`` names the process that wrote it; left out, the fake CLI
    that appends it names itself."""
    record: dict[str, Any] = {
        "ts": timestamp.isoformat().replace("+00:00", "Z"),
        "src": "shell",
        "lvl": "info",
        "msg": "billing: fetched credits config",
        "ctx": {"config": grok_credits_config(now, percent), "subscriptionTier": tier},
    }
    if pid is not None:
        record["pid"] = pid
    return json.dumps(record).encode() + b"\n"


def grok_credits_config(
    now: datetime, percent: float = 85.0, period_type: str = "USAGE_PERIOD_TYPE_WEEKLY"
) -> dict[str, Any]:
    return {
        "creditUsagePercent": percent,
        "currentPeriod": {
            "type": period_type,
            "start": (now - timedelta(days=6)).isoformat(),
            "end": (now + timedelta(days=1)).isoformat(),
        },
        "onDemandCap": {"val": 0},
        "onDemandUsed": {"val": 0},
        "prepaidBalance": {"val": 0},
    }
