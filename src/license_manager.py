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

LAB_SEAT_LIMIT: int = 5
"""Maximum concurrent users in the Lab tier."""

# ---------------------------------------------------------------------------
# Per-tier HMAC secrets (NOT cryptographic — casual forgery deterrent only)
# ---------------------------------------------------------------------------

_SECRET_PRO: bytes = b"ionflow-license-v1-xK9mPqRzWn4s"
_SECRET_LAB: bytes = b"ionflow-lab-v1-tY7nBkLvWm2pXqRs"
_SECRET_OEM: bytes = b"ionflow-oem-v1-zQ3cDjHuEa6wFgNt"

# Keep backwards-compatible alias
_SECRET: bytes = _SECRET_PRO

_SERIAL_CHARS: str = string.ascii_uppercase + string.digits
_SERIAL_LEN: int = 8
_MAC_LEN: int = 10  # hex nibbles

# Valid tier prefixes
_TIER_SECRETS: dict[str, bytes] = {
    "PRO": _SECRET_PRO,
    "LAB": _SECRET_LAB,
    "OEM": _SECRET_OEM,
}


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


def _compute_mac(serial: str, tier: str = "PRO") -> str:
    """Return the expected MAC for *serial* and *tier* (uppercase hex, _MAC_LEN chars)."""
    secret = _TIER_SECRETS.get(tier.upper(), _SECRET_PRO)
    return (
        hmac.new(secret, serial.upper().encode("ascii"), hashlib.sha256)
        .hexdigest()[:_MAC_LEN]
        .upper()
    )


def generate_key(serial: str, tier: str = "PRO") -> str:
    """
    Build a valid license key from *serial* and *tier*.

    Parameters
    ----------
    serial:
        Exactly 8 uppercase alphanumeric characters.  Any lowercase letters
        are automatically folded to uppercase.
    tier:
        One of ``"PRO"``, ``"LAB"``, ``"OEM"`` (case-insensitive).
        Defaults to ``"PRO"`` for backwards compatibility.

    Returns
    -------
    str
        A key of the form ``IONFLOW-<TIER>-<SERIAL8>-<MAC10>``.

    Raises
    ------
    ValueError
        If *serial* has an invalid length or contains illegal characters,
        or if *tier* is unknown.
    """
    tier = tier.upper()
    if tier not in _TIER_SECRETS:
        raise ValueError(f"Unknown tier {tier!r}. Must be one of {list(_TIER_SECRETS)}")
    serial = serial.upper()
    if len(serial) != _SERIAL_LEN:
        raise ValueError(
            f"Serial must be exactly {_SERIAL_LEN} characters, got {len(serial)}"
        )
    if not all(c in _SERIAL_CHARS for c in serial):
        raise ValueError(f"Serial must contain only A-Z and 0-9, got {serial!r}")
    mac = _compute_mac(serial, tier)
    return f"IONFLOW-{tier}-{serial}-{mac}"


def validate_key(key: str) -> bool:
    """
    Return True if *key* is a well-formed, authentic IonFlow license key
    for any tier (Pro, Lab, or OEM).

    This is a pure function — it does **not** mutate any state.
    """
    return key_tier(key) is not None


def key_tier(key: str) -> str | None:
    """Return the tier string (``"PRO"``, ``"LAB"``, ``"OEM"``) for a valid key,
    or ``None`` if the key is invalid or malformed."""
    key = key.strip().upper()
    prefix = "IONFLOW-"
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]  # "TIER-SERIAL8-MAC10"
    parts = rest.split("-", 2)
    if len(parts) != 3:
        return None
    tier, serial, mac = parts
    if tier not in _TIER_SECRETS:
        return None
    if len(serial) != _SERIAL_LEN or len(mac) != _MAC_LEN:
        return None
    expected = _compute_mac(serial, tier)
    if not hmac.compare_digest(mac, expected):
        return None
    return tier


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
    >>> mgr.tier          # "free", "pro", "lab", or "oem"
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
            t = key_tier(raw)
            if t is not None:
                self._key = raw.upper()
                self._tier = t.lower()
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
        """``"free"``, ``"pro"``, ``"lab"``, or ``"oem"``."""
        return self._tier

    @property
    def is_pro(self) -> bool:
        """True when a Pro *or higher* license is active."""
        return self._tier in ("pro", "lab", "oem")

    @property
    def is_lab(self) -> bool:
        """True when a Lab *or higher* license is active."""
        return self._tier in ("lab", "oem")

    @property
    def is_oem(self) -> bool:
        """True when an OEM license is active."""
        return self._tier == "oem"

    @property
    def active_key(self) -> str | None:
        """The currently active license key, or None."""
        return self._key

    def activate(self, key: str) -> bool:
        """
        Try to activate *key*.

        Returns True on success, False if the key is invalid.
        The detected tier (pro/lab/oem) is set automatically.
        The key is persisted to disk on success.
        """
        t = key_tier(key)
        if t is None:
            return False
        self._key = key.strip().upper()
        self._tier = t.lower()
        self._persist_key(self._key)
        return True

    def deactivate(self) -> None:
        """Remove the active license and revert to the free tier."""
        self._key = None
        self._tier = "free"
        self._delete_persisted_key()

    def check_file_limit(self, n_files: int) -> None:
        """
        Raise :class:`LicenseLimitError` if *n_files* exceeds the free-tier cap.

        Does nothing when any paid license is active.

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
        if self._tier == "oem":
            serial = (self._key or "").split("-")[2] if self._key else "?"
            return f"✅ OEM activo  [serial: {serial}]"
        if self._tier == "lab":
            serial = (self._key or "").split("-")[2] if self._key else "?"
            return (
                f"✅ Lab activo  [serial: {serial}]  (até {LAB_SEAT_LIMIT} utilizadores)"
            )
        if self._tier == "pro":
            serial = (self._key or "").split("-")[2] if self._key else "?"
            return f"✅ Pro activo  [serial: {serial}]"
        return f"🆓 Versão gratuita  (máx. {FREE_FILE_LIMIT} ficheiros/execução)"
