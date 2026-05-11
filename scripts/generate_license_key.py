#!/usr/bin/env python
"""
IonFlow Pipeline — License Key Generator
=========================================
Run this script (privately, as the developer) to generate license keys
for customers.

Usage
-----
    python scripts/generate_license_key.py [SERIAL]

If SERIAL is omitted, a random 8-character serial is generated.

Examples
--------
    # Generate a random key
    python scripts/generate_license_key.py

    # Generate a key with a specific serial (e.g. for customer #0001)
    python scripts/generate_license_key.py CUST0001

Output
------
    Serial:  CUST0001
    Key:     IONFLOW-PRO-CUST0001-3F8A2C4B9D

Security note
-------------
Keep this script and the _SECRET constant in src/license_manager.py
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
    generate_key,
    validate_key,
)


def random_serial() -> str:
    return "".join(random.choices(_SERIAL_CHARS, k=_SERIAL_LEN))


def main() -> None:
    if len(sys.argv) > 1:
        serial = sys.argv[1].strip().upper()
    else:
        serial = random_serial()

    try:
        key = generate_key(serial)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Sanity check
    assert validate_key(key), "BUG: generated key failed validation!"

    print(f"Serial:  {serial}")
    print(f"Key:     {key}")
    print()
    print("Send the 'Key' line to the customer.")
    print("They enter it in Settings → 🔑 Licença → Activar.")


if __name__ == "__main__":
    main()
