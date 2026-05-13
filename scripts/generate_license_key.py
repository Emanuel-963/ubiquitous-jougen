#!/usr/bin/env python
"""
IonFlow Pipeline — License Key Generator
=========================================
Run this script (privately, as the developer) to generate license keys
for customers.

Usage
-----
    python scripts/generate_license_key.py [SERIAL] [TIER]

SERIAL   8-char alphanumeric.  Omit for a random serial.
TIER     One of PRO (default), LAB, OEM.

Examples
--------
    # Generate a random Pro key
    python scripts/generate_license_key.py

    # Generate a specific Pro key
    python scripts/generate_license_key.py CUST0001

    # Generate a Lab key
    python scripts/generate_license_key.py CUST0001 LAB

    # Generate an OEM key
    python scripts/generate_license_key.py OEM00001 OEM

Output
------
    Serial:  CUST0001
    Tier:    PRO
    Key:     IONFLOW-PRO-CUST0001-3F8A2C4B9D

Security note
-------------
Keep this script and the _SECRET* constants in src/license_manager.py
**private** — they are equivalent to a master password.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.license_manager import (  # noqa: E402
    _SERIAL_CHARS,
    _SERIAL_LEN,
    _TIER_SECRETS,
    generate_key,
    validate_key,
)


def random_serial() -> str:
    return "".join(random.choices(_SERIAL_CHARS, k=_SERIAL_LEN))


def main() -> None:
    serial = sys.argv[1].strip().upper() if len(sys.argv) > 1 else random_serial()
    tier = sys.argv[2].strip().upper() if len(sys.argv) > 2 else "PRO"

    if tier not in _TIER_SECRETS:
        print(
            f"ERROR: Unknown tier {tier!r}. Must be one of {list(_TIER_SECRETS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        key = generate_key(serial, tier)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Sanity check
    assert validate_key(key), "BUG: generated key failed validation!"

    print(f"Serial:  {serial}")
    print(f"Tier:    {tier}")
    print(f"Key:     {key}")
    print()
    print("Send the 'Key' line to the customer.")
    print("They enter it in Settings → 🔑 Licença → Activar.")


if __name__ == "__main__":
    main()
