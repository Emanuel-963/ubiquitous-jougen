#!/usr/bin/env python3
"""
IonFlow Pipeline — License Validation Server (LIC-05)
======================================================
A minimal Flask server that validates IonFlow license keys over HTTPS.

Run locally
-----------
    pip install flask
    python scripts/license_server.py

Run with Gunicorn (production)
------------------------------
    pip install gunicorn
    gunicorn "scripts.license_server:create_app()" --bind 0.0.0.0:5001

API
---
POST /api/v1/validate
    Body (JSON): {"key": "IONFLOW-PRO-ABCD1234-XXXXXXXXXX"}
    Response  : {"valid": true, "tier": "pro", "version": "0.4.0"}

GET  /api/v1/ping
    Response  : {"status": "ok", "version": "0.4.0"}

Environment variables
---------------------
IONFLOW_SERVER_PORT   : TCP port (default 5001)
IONFLOW_DEBUG         : "1" to enable Flask debug mode (dev only)

Security notes
--------------
- Keys are validated locally using HMAC — no database is needed.
- No personally identifiable information is stored or logged.
- For production, place behind a reverse proxy (nginx/caddy) with TLS.
- Rate-limit the /validate endpoint to prevent brute-force attempts
  (e.g. nginx limit_req or a WAF rule).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this script directly from the repo root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from flask import Flask, jsonify, request
except ImportError as exc:
    raise SystemExit(
        "Flask is required: pip install flask\n" f"Original error: {exc}"
    ) from exc

from src import __version__  # noqa: E402
from src.license_manager import key_tier  # noqa: E402

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/api/v1/ping")
    def ping():
        return jsonify({"status": "ok", "version": __version__})

    # ------------------------------------------------------------------
    # Key validation
    # ------------------------------------------------------------------

    @app.post("/api/v1/validate")
    def validate():
        data = request.get_json(silent=True) or {}
        raw_key = data.get("key", "")

        if not isinstance(raw_key, str) or not raw_key.strip():
            return jsonify({"valid": False, "error": "missing_key"}), 400

        key = raw_key.strip()
        tier = key_tier(key)
        if tier is None:
            return jsonify({"valid": False, "tier": None, "version": __version__}), 200

        return (
            jsonify(
                {
                    "valid": True,
                    "tier": tier.lower(),
                    "version": __version__,
                }
            ),
            200,
        )

    # ------------------------------------------------------------------
    # Batch validation (for Lab/OEM seat accounting)
    # ------------------------------------------------------------------

    @app.post("/api/v1/validate/batch")
    def validate_batch():
        data = request.get_json(silent=True) or {}
        keys = data.get("keys", [])

        if not isinstance(keys, list):
            return jsonify({"error": "keys must be a list"}), 400

        # Limit batch size to prevent abuse
        if len(keys) > 100:
            return jsonify({"error": "batch_too_large", "max": 100}), 400

        results = []
        for raw_key in keys:
            if not isinstance(raw_key, str):
                results.append({"key": str(raw_key), "valid": False, "tier": None})
                continue
            t = key_tier(raw_key.strip())
            results.append(
                {
                    "key": raw_key.strip(),
                    "valid": t is not None,
                    "tier": t.lower() if t else None,
                }
            )

        return jsonify({"results": results, "version": __version__}), 200

    return app


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("IONFLOW_SERVER_PORT", 5001))
    debug = os.environ.get("IONFLOW_DEBUG", "0") == "1"

    print(f"IonFlow License Server v{__version__}  —  http://127.0.0.1:{port}")
    print("Endpoints:")
    print(f"  GET  http://127.0.0.1:{port}/api/v1/ping")
    print(f"  POST http://127.0.0.1:{port}/api/v1/validate")
    print(f"  POST http://127.0.0.1:{port}/api/v1/validate/batch")

    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=debug)
