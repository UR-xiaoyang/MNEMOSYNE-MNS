"""Authentication primitives for the local MNEMOSYNE HTTP API."""

from __future__ import annotations

import hmac


def authorization_is_valid(configured_key: str, authorization: str | None) -> bool:
    """Accept all requests when no key is configured; otherwise require exact Bearer auth."""
    if not configured_key:
        return True
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        return False
    return hmac.compare_digest(authorization[len(prefix):], configured_key)
