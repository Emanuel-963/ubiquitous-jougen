"""Update PR body and optionally request reviewers.

Usage:
    python scripts/update_pr.py --pr 1 --body-file docs/ONE_PAGER.md \
        --reviewers user1,user2 --token <token>

If `--body-file` is provided, it will replace the PR body.
Use `--append` to append to existing body.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import requests

REPO_DEFAULT = "Emanuel-963/ubiquitous-jougen"


def get_token(cli_token: Optional[str]) -> str:
    token = cli_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: Please set GITHUB_TOKEN or pass --token")
        sys.exit(1)
    return token


def update_pr_body(repo: str, pr: int, token: str, body: str) -> bool:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}"
    r = requests.patch(url, json={"body": body}, headers=headers)
    if r.status_code == 200:
        print("Updated PR body")
        return True
    print("Failed to update PR body:", r.status_code, r.text)
    return False


def request_reviewers(repo: str, pr: int, token: str, reviewers: list[str]) -> bool:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}/requested_reviewers"
    r = requests.post(url, json={"reviewers": reviewers}, headers=headers)
    if r.status_code in (201, 200):
        print("Requested reviewers:", reviewers)
        return True
    print("Failed to request reviewers:", r.status_code, r.text)
    return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--token")
    p.add_argument("--body-file", help="Path to markdown file to set as PR body")
    p.add_argument(
        "--append", action="store_true", help="Append to existing body if present"
    )
    p.add_argument("--reviewers", help="Comma-separated list of reviewers to request")
    args = p.parse_args(argv)

    token = get_token(args.token)

    body = None
    if args.body_file:
        if not os.path.exists(args.body_file):
            print("body-file not found:", args.body_file)
            return 2
        body = open(args.body_file, encoding="utf-8").read()

    if args.append and body is not None:
        # fetch existing
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }
        r = requests.get(
            f"https://api.github.com/repos/{args.repo}/pulls/{args.pr}", headers=headers
        )
        if r.status_code == 200:
            existing = r.json().get("body", "")
            body = existing + "\n\n" + body
        else:
            print("Failed to fetch existing PR body to append")

    if body is not None:
        if not update_pr_body(args.repo, args.pr, token, body):
            return 3

    if args.reviewers:
        reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
        if reviewers:
            if not request_reviewers(args.repo, args.pr, token, reviewers):
                return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
