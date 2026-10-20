# Maintaining the project

## GitHub configuration

The public repository uses these settings, verified on 2026-10-20:

- An active `Protect main` ruleset requires a PR, passing `test` and `browser` checks from GitHub
  Actions, and resolved review threads. The branch must be up to date. Force pushes and branch
  deletion are blocked, with no bypass actors. The approval count is zero for a solo maintainer;
  PRs and checks are still required.
- Secret scanning, push protection, private vulnerability reporting, Dependabot alerts, and
  Dependabot security updates are enabled.
- CodeQL default setup scans Python, JavaScript/TypeScript, and GitHub Actions with the default suite.
- Actions default to read-only permissions and cannot approve PRs. All external contributors need
  approval before their fork PR workflows run.
- Merged branches are deleted automatically. Squash merges use the PR title and body.

`.github/dependabot.yml` schedules weekly updates for Python (`uv`), development npm packages, and
GitHub Actions. Keep action references pinned to full commit SHAs with a readable version comment.
The static dashboard has no npm runtime dependencies.

When changing CI job names, update the required status checks in the ruleset at the same time.

## Before sharing a release

1. Run the checks in `CONTRIBUTING.md`, including browser tests for a page change.
2. Build the distribution with `uv build` and check that the wheel contains the static page assets.
3. Review dependency alerts and the latest CodeQL run. Investigate every open security finding.
4. Scan reachable Git history with `gitleaks git . --log-opts=--all --redact=100`.
5. Regenerate README screenshots with `npm run screenshot` when the design changes. These use only
   synthetic accounts. Never commit a live screenshot, private bootstrap page, database, or login home.
6. Merge through a PR, then create a version tag and GitHub release with changes and compatibility notes.

The repository is the installation source. No package registry publishing or hosted dashboard is configured.
