"""Internationalisation (i18n) support for IonFlow Pipeline.

Usage
-----
    from src.i18n import tr, set_language, get_language, LANGUAGES

    # Legacy mode -- Portuguese key returns translated value:
    set_language("en")
    label = tr("Rodar Pipeline EIS")  # -> "Run EIS Pipeline"

    # New section-key mode -- dotted key lookup:
    label = tr("ui.run_eis")          # -> "Run EIS Pipeline"

    # Convenience for a specific section:
    label = tr_section("plots", "frequency_axis")  # -> "Frequency (Hz)"

    # List available languages:
    from src.i18n import LANGUAGES     # ("pt", "en", "es")

Design decisions
----------------
* **Backward compatible**: ``tr("Portuguese string")`` still works exactly as
  before.  When the active language is ``"pt"`` the key is returned as-is.
* **JSON-based string files**: translations live in
  ``src/i18n_strings/{pt,en,es}.json``, organised by section
  (``ui``, ``pipeline``, ``ai``, ``reports``, ``columns``, ``plots``,
  ``diagnostics``, ``cli``, ``empty``, ``log``, ``knowledge``).
* **Dotted-key lookup**: ``tr("section.key")`` for structured access.
* **Missing keys never crash**: a missing key always returns the key itself.
* **Thread-safe**: language switching is protected by a ``threading.Lock``.
* **Lazy loading**: JSON files are read once on first ``tr()`` call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =====================================================================
# Public constants
# =====================================================================

LANGUAGES: Tuple[str, ...] = ("pt", "en", "es")
"""Supported language codes."""

SECTIONS: Tuple[str, ...] = (
    "ui",
    "pipeline",
    "ai",
    "reports",
    "columns",
    "plots",
    "diagnostics",
    "cli",
    "empty",
    "log",
    "knowledge",
)
"""Top-level sections in each language JSON file."""

# =====================================================================
# Module state
# =====================================================================

_lock = threading.Lock()
_current: str = "pt"

# Populated by _ensure_loaded():
# _flat[lang] = {"ui.run_eis": "Run EIS Pipeline", ...}
_flat: Dict[str, Dict[str, str]] = {}

# _legacy[lang] = {"Rodar Pipeline EIS": "Run EIS Pipeline", ...}
# Built so tr("Portuguese string") still works for en/es.
_legacy: Dict[str, Dict[str, str]] = {}

# _raw[lang] = original nested dict from JSON
_raw: Dict[str, Dict[str, Any]] = {}

_loaded: bool = False

# =====================================================================
# Internal -- loading & flattening
# =====================================================================

_STRINGS_DIR = Path(__file__).resolve().parent / "i18n_strings"


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Recursively flatten a nested dict to dotted-key -> string."""
    result: Dict[str, str] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, full_key))
        else:
            result[full_key] = str(value)
    return result


def _load_language(lang: str) -> Dict[str, Any]:
    """Load a language JSON file; return empty dict on failure."""
    path = _STRINGS_DIR / f"{lang}.json"
    if not path.exists():
        logger.warning("i18n strings file not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load i18n strings for '%s': %s", lang, exc)
        return {}


def _build_legacy_table(
    pt_flat: Dict[str, str], other_flat: Dict[str, str],
) -> Dict[str, str]:
    """Build a Portuguese-value -> other-language-value map.

    For each dotted key present in both dicts, maps
    ``pt_flat[key]`` -> ``other_flat[key]``.
    """
    legacy: Dict[str, str] = {}
    for key, pt_value in pt_flat.items():
        if key in other_flat:
            legacy[pt_value] = other_flat[key]
    return legacy


def _ensure_loaded() -> None:
    """Lazy-load all JSON files on first use."""
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        for lang in LANGUAGES:
            raw = _load_language(lang)
            _raw[lang] = raw
            _flat[lang] = _flatten(raw)

        # Build legacy reverse tables for non-Portuguese languages
        pt_flat = _flat.get("pt", {})
        for lang in LANGUAGES:
            if lang == "pt":
                _legacy[lang] = {}
                continue
            _legacy[lang] = _build_legacy_table(pt_flat, _flat[lang])

        _loaded = True


def reload_strings() -> None:
    """Force reload of all language JSON files.

    Useful after editing the JSON files at runtime (e.g. GUI editor).
    """
    global _loaded
    with _lock:
        _loaded = False
        _flat.clear()
        _legacy.clear()
        _raw.clear()
    _ensure_loaded()


# =====================================================================
# Public API
# =====================================================================


def set_language(lang: str) -> None:
    """Set the active language.  Must be one of ``LANGUAGES``."""
    global _current
    code = lang.lower().strip()[:2]
    if code not in LANGUAGES:
        code = "pt"
    with _lock:
        _current = code


def get_language() -> str:
    """Return the current language code (``"pt"``, ``"en"`` or ``"es"``)."""
    return _current


def get_languages() -> Tuple[str, ...]:
    """Return all supported language codes."""
    return LANGUAGES


def tr(key: str) -> str:
    """Translate *key* to the current language.

    Supports two lookup modes:

    1. **Dotted key** (``"section.key"``): looks up in the flattened JSON
       table for the current language.
    2. **Legacy Portuguese string**: when the current language is not
       ``"pt"``, tries to find the Portuguese string in the legacy reverse
       table and returns its translation.

    If no translation is found, *key* is returned unchanged -- the app
    never crashes because of a missing translation.

    Examples
    --------
    >>> set_language("en")
    >>> tr("ui.run_eis")
    'Run EIS Pipeline'
    >>> tr("Rodar Pipeline EIS")       # legacy mode
    'Run EIS Pipeline'
    >>> set_language("pt")
    >>> tr("Rodar Pipeline EIS")       # Portuguese returns key as-is
    'Rodar Pipeline EIS'
    """
    _ensure_loaded()
    lang = _current

    # Fast path: Portuguese returns key unchanged for legacy calls
    if lang == "pt":
        # But dotted keys should still resolve from pt.json
        if "." in key:
            return _flat.get("pt", {}).get(key, key)
        return key

    # Try dotted-key lookup first
    flat_table = _flat.get(lang, {})
    if key in flat_table:
        return flat_table[key]

    # Try legacy (Portuguese string -> translated) lookup
    legacy_table = _legacy.get(lang, {})
    if key in legacy_table:
        return legacy_table[key]

    # Fallback: return key unchanged
    return key


def tr_section(section: str, key: str) -> str:
    """Translate a key within a specific section.

    Equivalent to ``tr(f"{section}.{key}")``.

    Parameters
    ----------
    section : str
        Top-level section name (e.g. ``"ui"``, ``"plots"``).
    key : str
        Key within the section.

    Returns
    -------
    str
        Translated string, or the dotted key if not found.
    """
    return tr(f"{section}.{key}")


def available_keys(section: Optional[str] = None) -> List[str]:
    """Return all dotted keys for a section, or all keys if *section* is None.

    Useful for tooling, introspection and tests.
    """
    _ensure_loaded()
    pt_flat = _flat.get("pt", {})
    if section is None:
        return sorted(pt_flat.keys())
    prefix = f"{section}."
    return sorted(k for k in pt_flat if k.startswith(prefix))


def get_section(section: str, lang: Optional[str] = None) -> Dict[str, str]:
    """Return all key->value pairs for a section in the given language.

    Parameters
    ----------
    section : str
        Section name (e.g. ``"ui"``).
    lang : str | None
        Language code.  ``None`` -> current language.

    Returns
    -------
    dict[str, str]
        Keys are *short* (without the section prefix).
    """
    _ensure_loaded()
    lang = lang or _current
    flat_table = _flat.get(lang, {})
    prefix = f"{section}."
    plen = len(prefix)
    return {k[plen:]: v for k, v in flat_table.items() if k.startswith(prefix)}


def missing_keys(lang: str) -> List[str]:
    """Return dotted keys present in ``pt.json`` but missing from *lang*.

    Useful for detecting incomplete translations.
    """
    _ensure_loaded()
    pt_keys = set(_flat.get("pt", {}).keys())
    lang_keys = set(_flat.get(lang, {}).keys())
    return sorted(pt_keys - lang_keys)


def translation_coverage(lang: str) -> float:
    """Return fraction of ``pt.json`` keys covered by *lang* (0.0-1.0)."""
    _ensure_loaded()
    pt_keys = set(_flat.get("pt", {}).keys())
    if not pt_keys:
        return 1.0
    lang_keys = set(_flat.get(lang, {}).keys())
    return len(pt_keys & lang_keys) / len(pt_keys)
