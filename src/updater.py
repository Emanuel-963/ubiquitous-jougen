"""Lightweight auto-update checker for IonFlow Pipeline.

Queries the GitHub Releases API (unauthenticated, rate-limited to 60 req/h)
and returns a human-readable message **only** when a newer version is available.
If the network is unreachable, the API errors out, or the local version is
already up-to-date, the function returns ``None`` — it is designed to never
raise.

Usage
-----
    from src.updater import check_for_updates, get_latest_release, download_release

    info = get_latest_release()
    if info and info.is_newer_than_local():
        zip_path = download_release(info, Path("/tmp/ionflow_update.zip"))
"""
from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src import __version__ as _LOCAL_VERSION

# ── Constants ────────────────────────────────────────────────────────
_REPO_OWNER = "Emanuel-963"
_REPO_NAME = "ubiquitous-jougen"
_API_URL = f"https://api.github.com/repos/{_REPO_OWNER}/{_REPO_NAME}/releases/latest"
_TIMEOUT_S = 5

_TAG_RE = re.compile(r"v?(\d+(?:\.\d+)*)")


def _parse_version(tag: str) -> tuple:
    """Extract a numeric tuple from a tag like ``v0.2.1``."""
    m = _TAG_RE.search(tag)
    if m is None:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))


@dataclass
class ReleaseInfo:
    """Parsed metadata for a single GitHub release."""

    tag: str
    html_url: str
    zipball_url: str
    body: str

    def is_newer_than_local(self) -> bool:
        """Return True when this release is newer than the installed version."""
        return _parse_version(self.tag) > _parse_version(_LOCAL_VERSION)


def get_latest_release() -> Optional[ReleaseInfo]:
    """Query the GitHub Releases API and return the latest release.

    Returns ``None`` on any network / API error (fail-silent design).
    """
    try:
        req = urllib.request.Request(
            _API_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return ReleaseInfo(
            tag=data.get("tag_name", ""),
            html_url=data.get(
                "html_url",
                f"https://github.com/{_REPO_OWNER}/{_REPO_NAME}/releases",
            ),
            zipball_url=data.get("zipball_url", ""),
            body=data.get("body", ""),
        )
    except Exception:
        return None


def check_for_updates() -> Optional[str]:
    """Return a user-friendly message if a newer release exists, else ``None``."""
    info = get_latest_release()
    if info is None:
        return None
    if info.is_newer_than_local():
        return (
            f"🆕 Nova versão disponível: {info.tag}  (atual: v{_LOCAL_VERSION})\n"
            f"   Download: {info.html_url}"
        )
    return None


def download_release(
    info: ReleaseInfo,
    dest_path: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download the release ZIP to *dest_path*.

    Parameters
    ----------
    info:
        ReleaseInfo returned by ``get_latest_release()``.
    dest_path:
        Target file path for the downloaded ZIP.
    on_progress:
        Optional ``callback(bytes_downloaded, total_bytes)``.
        *total_bytes* may be 0 when Content-Length is unavailable.

    Returns
    -------
    Path
        The path where the ZIP was saved.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        info.zipball_url,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        with open(dest_path, "wb") as fh:
            while True:
                block = resp.read(65536)  # 64 KiB chunks
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                if on_progress:
                    on_progress(downloaded, total)

    return dest_path


def extract_release(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the release ZIP and return the top-level directory inside it.

    GitHub source ZIPs always contain a single top-level folder
    (``owner-repo-<sha>/``).  This folder is returned so the caller can
    locate the extracted sources.

    Parameters
    ----------
    zip_path:
        Path to the downloaded ZIP file.
    dest_dir:
        Directory where the ZIP will be extracted.

    Returns
    -------
    Path
        Path to the top-level directory extracted from the ZIP.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
        names = zf.namelist()
        if names:
            top = names[0].split("/")[0]
            return dest_dir / top

    return dest_dir
