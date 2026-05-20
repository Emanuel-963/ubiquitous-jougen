"""Auto-update for IonFlow Pipeline — silent in-place installer edition.

Strategy
--------
* **Frozen exe (Inno Setup install)**: looks for the Windows installer asset
  ``IonFlow_Pipeline_Setup_X.Y.Z.exe`` in the GitHub release, downloads it to
  a temp folder and runs it silently (``/VERYSILENT /SUPPRESSMSGBOXES
  /NORESTART``).  The Inno Setup installer detects the existing installation
  path and replaces all program files in-place, then relaunches the app.
  The current process exits immediately after starting the installer.

* **Source / git clone**: cannot auto-patch binaries; surfaces a dialog with
  the exact commands ``git pull && pip install -e .`` to copy-paste.

Usage
-----
    from src.updater import check_for_updates, get_latest_release

    info = get_latest_release()
    if info and info.is_newer_than_local():
        print(info.installer_asset_url)  # URL do .exe, or None
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from src import __version__ as _LOCAL_VERSION

# ── Constants ────────────────────────────────────────────────────────
_REPO_OWNER = "Emanuel-963"
_REPO_NAME = "ubiquitous-jougen"
_API_URL = f"https://api.github.com/repos/{_REPO_OWNER}/{_REPO_NAME}/releases/latest"
_TIMEOUT_S = 8

_TAG_RE = re.compile(r"v?(\d+(?:\.\d+)*)")


def _parse_version(tag: str) -> tuple:
    """Extract a numeric tuple from a tag like ``v0.2.1``."""
    m = _TAG_RE.search(tag)
    if m is None:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))


def is_frozen() -> bool:
    """Return True when running as a PyInstaller frozen executable."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


@dataclass
class ReleaseInfo:
    """Parsed metadata for a single GitHub release."""

    tag: str
    html_url: str
    zipball_url: str
    body: str
    assets: List[dict] = field(default_factory=list)

    def is_newer_than_local(self) -> bool:
        """Return True when this release is newer than the installed version."""
        return _parse_version(self.tag) > _parse_version(_LOCAL_VERSION)

    @property
    def installer_asset_url(self) -> Optional[str]:
        """URL of the Windows .exe installer asset, or None if not published."""
        for asset in self.assets:
            name: str = asset.get("name", "")
            if name.lower().endswith(".exe") and "setup" in name.lower():
                return asset.get("browser_download_url")
        return None

    @property
    def installer_asset_size(self) -> int:
        """File size in bytes of the installer asset, or 0 if unknown."""
        for asset in self.assets:
            name: str = asset.get("name", "")
            if name.lower().endswith(".exe") and "setup" in name.lower():
                return int(asset.get("size", 0))
        return 0


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
            assets=data.get("assets", []),
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


def download_asset(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download any release asset (installer .exe, zip, etc.) to *dest_path*.

    Parameters
    ----------
    url:
        Direct download URL (``browser_download_url`` from GitHub assets or
        ``zipball_url`` for source archives).
    dest_path:
        Target file path for the downloaded file.
    on_progress:
        Optional ``callback(bytes_downloaded, total_bytes)``.
        *total_bytes* may be 0 when Content-Length is unavailable.

    Returns
    -------
    Path
        The path where the file was saved.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": f"IonFlow/{_LOCAL_VERSION}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
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


def launch_installer_and_exit(installer_path: Path) -> None:
    """Run the Inno Setup installer silently and exit the current process.

    The installer detects the existing installation directory (via AppId in
    the registry) and replaces all program files in-place.  After the
    installer finishes, it relaunches ``IonFlow_Pipeline.exe`` automatically
    (the ``[Run]`` section in ionflow_setup.iss has no ``skipifsilent``).

    This function never returns — it always calls ``sys.exit(0)``.
    """
    import subprocess

    subprocess.Popen(  # noqa: S603 — controlled internal use only
        [
            str(installer_path),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
        ],
        close_fds=True,
    )
    sys.exit(0)


# ── Legacy helpers kept for backward compatibility ────────────────────

def download_release(
    info: "ReleaseInfo",
    dest_path: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download the release source ZIP (legacy — prefer ``download_asset``)."""
    return download_asset(info.zipball_url, dest_path, on_progress)


def extract_release(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the release source ZIP and return the top-level directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
        names = zf.namelist()
        if names:
            top = names[0].split("/")[0]
            return dest_dir / top

    return dest_dir
