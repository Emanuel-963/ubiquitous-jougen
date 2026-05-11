"""
IonFlow Pipeline — License Manager
===================================
Implements a simple offline freemium gate.

Tiers
-----
- free  : up to FREE_FILE_LIMIT files per pipeline run
- pro   : unlimited files, activated with a license key

Key format
----------
    IONFLOW-PRO-<SERIAL8>-<MAC10>

Where:
    SERIAL8  — 8 uppercase alphanumeric chars (unique per customer)
    MAC10    — first 10 hex chars of HMAC-SHA256(_SECRET, SERIAL8.encode())

Keys are generated offline with ``scripts/generate_license_key.py``.

Storage
-------
Active license key is persisted to ``~/.ionflow/license.key``.
"""

from __future__ import annotations

import hashlib
import hmac
import string
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREE_FILE_LIMIT: int = 5
"""Maximum files processed per run in the free tier."""

_SECRET: bytes = b"ionflow-license-v1-xK9mPqRzWn4s"
"""
Shared secret used for HMAC key validation.
This is embedded in the binary; it is NOT a cryptographic secret — it provides
a simple deterrent against casual key forgery, not strong protection.
"""

_SERIAL_CHARS: str = string.ascii_uppercase + string.digits
_SERIAL_LEN: int = 8
_MAC_LEN: int = 10  # hex nibbles


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LicenseError(Exception):
    """Base class for license errors."""


class LicenseLimitError(LicenseError):
    """Raised when the free-tier file limit is exceeded."""


class LicenseKeyError(LicenseError):
    """Raised when an invalid key is provided."""


# ---------------------------------------------------------------------------
# Low-level helpers (module-level so the generator script can import them)
# ---------------------------------------------------------------------------


def _compute_mac(serial: str) -> str:
    """Return the expected MAC for *serial* (uppercase hex, _MAC_LEN chars)."""
    return (
        hmac.new(_SECRET, serial.upper().encode("ascii"), hashlib.sha256)
        .hexdigest()[:_MAC_LEN]
        .upper()
    )


def generate_key(serial: str) -> str:
    """
    Build a valid license key from *serial*.

    Parameters
    ----------
    serial:
        Exactly 8 uppercase alphanumeric characters.  Any lowercase letters
        are automatically folded to uppercase.

    Returns
    -------
    str
        A key of the form ``IONFLOW-PRO-<SERIAL8>-<MAC10>``.

    Raises
    ------
    ValueError
        If *serial* has an invalid length or contains illegal characters.
    """
    serial = serial.upper()
    if len(serial) != _SERIAL_LEN:
        raise ValueError(
            f"Serial must be exactly {_SERIAL_LEN} characters, got {len(serial)}"
        )
    if not all(c in _SERIAL_CHARS for c in serial):
        raise ValueError(f"Serial must contain only A-Z and 0-9, got {serial!r}")
    mac = _compute_mac(serial)
    return f"IONFLOW-PRO-{serial}-{mac}"


def validate_key(key: str) -> bool:
    """
    Return True if *key* is a well-formed, authentic IonFlow Pro license key.

    This is a pure function — it does **not** mutate any state.
    """
    key = key.strip().upper()
    prefix = "IONFLOW-PRO-"
    if not key.startswith(prefix):
        return False
    rest = key[len(prefix) :]  # "SERIAL8-MAC10"
    parts = rest.rsplit("-", 1)
    if len(parts) != 2:
        return False
    serial, mac = parts
    if len(serial) != _SERIAL_LEN or len(mac) != _MAC_LEN:
        return False
    expected = _compute_mac(serial)
    return hmac.compare_digest(mac, expected)


# ---------------------------------------------------------------------------
# LicenseManager — singleton
# ---------------------------------------------------------------------------


class LicenseManager:
    """
    Singleton that tracks the current license tier and persists the key.

    Usage
    -----
    >>> from src.license_manager import LicenseManager
    >>> mgr = LicenseManager.get()
    >>> mgr.tier          # "free" or "pro"
    >>> mgr.activate("IONFLOW-PRO-ABCD1234-XXXXXXXXXX")
    True
    >>> mgr.check_file_limit(10)  # raises LicenseLimitError on free tier
    """

    _instance: LicenseManager | None = None

    def __init__(self) -> None:
        self._tier: str = "free"
        self._key: str | None = None
        self._load_persisted_key()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "LicenseManager":
        """Return the global LicenseManager instance (created on first call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (mainly useful in tests)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _key_path() -> Path:
        config_dir = Path.home() / ".ionflow"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "license.key"

    def _load_persisted_key(self) -> None:
        try:
            raw = self._key_path().read_text(encoding="utf-8").strip()
            if validate_key(raw):
                self._key = raw.upper()
                self._tier = "pro"
        except Exception:
            pass

    def _persist_key(self, key: str) -> None:
        try:
            self._key_path().write_text(key, encoding="utf-8")
        except Exception:
            pass

    def _delete_persisted_key(self) -> None:
        try:
            self._key_path().unlink(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tier(self) -> str:
        """``"free"`` or ``"pro"``."""
        return self._tier

    @property
    def is_pro(self) -> bool:
        """True when a valid Pro license is active."""
        return self._tier == "pro"

    @property
    def active_key(self) -> str | None:
        """The currently active license key, or None."""
        return self._key

    def activate(self, key: str) -> bool:
        """
        Try to activate *key*.

        Returns True on success, False if the key is invalid.
        The key is persisted to disk on success.
        """
        key = key.strip().upper()
        if not validate_key(key):
            return False
        self._key = key
        self._tier = "pro"
        self._persist_key(key)
        return True

    def deactivate(self) -> None:
        """Remove the active license and revert to the free tier."""
        self._key = None
        self._tier = "free"
        self._delete_persisted_key()

    def check_file_limit(self, n_files: int) -> None:
        """
        Raise :class:`LicenseLimitError` if *n_files* exceeds the free-tier cap.

        Does nothing when a Pro license is active.

        Parameters
        ----------
        n_files:
            Number of files about to be processed.

        Raises
        ------
        LicenseLimitError
            When ``tier == "free"`` and ``n_files > FREE_FILE_LIMIT``.
        """
        if self.is_pro:
            return
        if n_files > FREE_FILE_LIMIT:
            raise LicenseLimitError(
                f"A versão gratuita suporta até {FREE_FILE_LIMIT} ficheiros "
                f"por execução (detectados: {n_files}).\n\n"
                f"Active uma licença Pro em Configurações → 🔑 Licença "
                f"para processar ficheiros ilimitados."
            )

    def status_label(self) -> str:
        """Human-readable status string for display in the UI."""
        if self.is_pro:
            serial = (self._key or "").split("-")[2] if self._key else "?"
            return f"✅ Pro activo  [serial: {serial}]"
        return f"🆓 Versão gratuita  (máx. {FREE_FILE_LIMIT} ficheiros/execução)"
