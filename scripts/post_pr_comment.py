"""Post or update a PR comment with local follow-ups.

Usage:
    GITHUB_TOKEN=<token> python scripts/post_pr_comment.py [--pr PR] [--repo owner/repo]

The script posts or updates a single comment marked with `<!-- precommit-status -->`.
If no --pr is given it will try to detect an open PR for the current branch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional

import requests

REPO_DEFAULT = "Emanuel-963/ubiquitous-jougen"
MARKER = "<!-- precommit-status -->"


def get_token(cli_token: Optional[str]) -> str:
    token = cli_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: Please set GITHUB_TOKEN or pass --token")
        sys.exit(1)
    return token


def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(
            cmd, universal_newlines=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return ""


def detect_pr_by_branch(repo: str, token: str) -> Optional[int]:
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None
    if not branch:
        return None

    owner, _name = repo.split("/", 1)
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/pulls?head={owner}:{branch}&state=open"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None
    prs = r.json()
    if isinstance(prs, list) and len(prs) == 1:
        return int(prs[0]["number"])
    return None


def find_existing_comment(repo: str, pr: int, token: str) -> Optional[Dict]:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("Failed to list comments:", r.status_code, r.text)
        return None
    for c in r.json():
        if MARKER in c.get("body", ""):
            return c
    return None


def post_comment(repo: str, pr: int, token: str, body: str) -> Optional[Dict]:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
    r = requests.post(url, json={"body": body}, headers=headers)
    if r.status_code in (200, 201):
        return r.json()
    print("Failed to post comment:", r.status_code, r.text)
    return None


def update_comment(repo: str, comment_id: int, token: str, body: str) -> Optional[Dict]:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/issues/comments/{comment_id}"
    r = requests.patch(url, json={"body": body}, headers=headers)
    if r.status_code == 200:
        return r.json()
    print("Failed to update comment:", r.status_code, r.text)
    return None


def build_message(branch: str) -> str:
    lines = [
        "Hello! I pushed formatting and `.gitignore` updates to branch ",
        f"`{branch}`.",
        "",
        "- Reformatted `scripts/create_labels.py` (black/isort)",
        "- Added `.gitignore` entries for local Python 3.11 embed and wrapper",
        "",
        "Note: pre-commit hooks failed to create cache environments due to a",
        "PermissionError during environment creation; to fix locally,",
        "either:",
        "",
        "1) Run PowerShell as Administrator and run:",
        "   - Remove pre-commit cache directory (run as admin)",
        "   - `pre-commit install`",
        "   - `pre-commit run --all-files`",
        "",
        "2) Or change permissions on the pre-commit cache directory so your user",
        "   can write there.",
        "",
        "Also ensure dev dependencies are installed and run tests:",
        "- `pip install -r requirements-dev.txt`",
        "- `python -m pytest -q`",
        "",
        MARKER,
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post or update a PR comment with follow-ups"
    )
    parser.add_argument("--pr", type=int, help="Pull request number (optional)")
    parser.add_argument(
        "--repo", default=REPO_DEFAULT, help="Repository in owner/repo form"
    )
    parser.add_argument(
        "--token",
        help="GitHub token (optional). If provided it overrides GITHUB_TOKEN env var",
    )
    args = parser.parse_args(argv)

    token = get_token(args.token)

    pr_number = args.pr
    if pr_number is None:
        pr_number = detect_pr_by_branch(args.repo, token)
        if pr_number is None:
            print(
                "Could not detect PR for the current branch. Provide "
                "--pr <number> and try again."
            )
            return 2

    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "<branch>"

    body = build_message(branch)

    existing = find_existing_comment(args.repo, pr_number, token)
    if existing:
        updated = update_comment(args.repo, existing["id"], token, body)
        if updated:
            print(f"Updated comment: {updated.get('html_url')}")
            return 0
        return 3

    posted = post_comment(args.repo, pr_number, token, body)
    if posted:
        print(f"Created comment: {posted.get('html_url')}")
        return 0
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
