"""Simple helper to create repository labels via GitHub API.

Usage:
    GITHUB_TOKEN=<token> python scripts/create_labels.py

This script will create a set of suggested labels for PR triage.
"""
import os
import sys
import requests

REPO = "Emanuel-963/ubiquitous-jougen"
LABELS = [
    {"name": "type: enhancement", "color": "a2eeef"},
    {"name": "type: bug", "color": "d73a4a"},
    {"name": "status: needs review", "color": "c2e0c6"},
    {"name": "status: ready to merge", "color": "0e8a16"},
    {"name": "ci: failing", "color": "fbca04"},
]

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    print("Please set GITHUB_TOKEN environment variable with repo scope")
    sys.exit(1)

headers = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}

for label in LABELS:
    r = requests.post(f"https://api.github.com/repos/{REPO}/labels", json=label, headers=headers)
    if r.status_code in (200, 201):
        print("Created label:", label["name"])
    elif r.status_code == 422:
        print("Label already exists:", label["name"])
    else:
        print("Failed to create label:", label["name"], r.status_code, r.text)
