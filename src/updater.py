"""Lightweight auto-update checker for IonFlow Pipeline.

Queries the GitHub Releases API (unauthenticated, rate-limited to 60 req/h)
and returns a human-readable message **only** when a newer version is available.
If the network is unreachable, the API errors out, or the local version is
already up-to-date, the function returns ``None`` — it is designed to never
raise.

Usage
-----
    from src.updater import check_for_updates

    msg = check_for_updates()   # blocking, ~1-2 s on first call
    if msg:
        print(msg)
"""
from __future__ import annotations

import re
import urllib.request
import json
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────
_REPO_OWNER = "Emanuel-963"
_REPO_NAME = "ubiquitous-jougen"
_API_URL = (
    f"https://api.github.com/repos/{_REPO_OWNER}/{_REPO_NAME}/releases/latest"
)
_TIMEOUT_S = 5

# Pulled from pyproject.toml at build time; kept here as single source
# so the module is self-contained.
_LOCAL_VERSION = "0.1.0"

_TAG_RE = re.compile(r"v?(\d+(?:\.\d+)*)")


def _parse_version(tag: str) -> tuple:
    """Extract a numeric tuple from a tag like ``v0.2.1``."""
    m = _TAG_RE.search(tag)
    if m is None:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))


def check_for_updates() -> Optional[str]:
    """Return a user-friendly message if a newer release exists, else ``None``."""
    try:
        req = urllib.request.Request(
            _API_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        tag = data.get("tag_name", "")
        remote = _parse_version(tag)
        local = _parse_version(_LOCAL_VERSION)

        if remote > local:
            url = data.get("html_url", f"https://github.com/{_REPO_OWNER}/{_REPO_NAME}/releases")
            return (
                f"🆕 Nova versão disponível: {tag}  (atual: v{_LOCAL_VERSION})\n"
                f"   Download: {url}"
            )

        return None  # already up-to-date

    except Exception:
        # Network failure, rate-limit, JSON error, etc. — fail silently.
        return None
