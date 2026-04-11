Branch protection recommendations (manual API commands)

Recommended protection rules for `main`:
- Require pull request reviews before merging (1 or 2 reviewers)
- Require status checks to pass (CI job: test-and-lint)
- Require branches to be up to date before merging
- Require signed commits (optional)

Use the GitHub REST API to configure (replace TOKEN with a personal access token with admin:repo scope):

curl -X PUT \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/Emanuel-963/ubiquitous-jougen/branches/main/protection \
  -d '{
    "required_status_checks": {
      "strict": true,
      "contexts": ["test-and-lint"]
    },
    "enforce_admins": false,
    "required_pull_request_reviews": {
      "required_approving_review_count": 1
    },
    "restrictions": null
  }'

Notes:
- You need admin permissions on the repository to set branch protection.
- The `contexts` entry must match the job name from your workflow (test-and-lint). Adjust if necessary.
