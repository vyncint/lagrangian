# Releasing lagrangian

One page, copy-pasteable. Maintainers only.

## Prerequisites

All of these are in place; this is what to check if a release fails.

- **Publishing is authenticated by crates.io Trusted Publishing.**
  crates.io → `lagrangian` → Settings → Trusted Publishing has a GitHub
  publisher for repository `vyncint/lagrangian`, workflow `release.yml` and
  environment `release`. The `publish` job holds `id-token: write`, and
  `rust-lang/crates-io-auth-action` exchanges the job's OIDC token for a
  publish token that lives for the job and is revoked at its end. No
  publish token is stored: `0.1.0`, the first release, went out with an API
  token, since Trusted Publishing links to a crate that already exists, and
  that token was deleted before `0.1.1`.
- **The `release` environment deploys only from tags matching `v*`**
  (Settings → Environments → release → Deployment branches and tags), so a
  publish token can never be minted from a branch.
- **Only a maintainer can push such a tag.** The `release tags` ruleset
  limits creating, moving and deleting `refs/tags/v*` to the admin role.

## What may break

`lagrangian` is pre-1.0. A `0.x` minor release may break; a patch release
may not. What dependents compile against is the trait surface —
`Lagrangian`, `Controlled`, `SceneMap` and `Scalar` as a bound — and a change
to it is always a `- **Breaking:**` bullet under **Changed** or **Removed**
in `CHANGELOG.md`. `Scalar` is sealed, so adding a method to it is not a
break; the `ad` module is exempt from semver until 1.0
([CLAUDE.md](../CLAUDE.md), "API Stability Rules").

There is no semver gate in CI yet. Before a release, run
`cargo semver-checks -p lagrangian` against the previous release by hand
(step 3 below); a gate is a `semver` job in `ci.yml` with its line in
CONTRIBUTING §1.

## Cutting vX.Y.Z

```sh
# 0. Green main: every job of the latest CI run on the commit that will be
#    tagged, and a look at the latest canary run.
gh run list --workflow ci.yml --branch main --limit 1
gh run list --workflow canary.yml --limit 1

# 1. Bump the version: workspace.package.version in the root Cargo.toml,
#    then refresh the lockfile. Two more copies of the number have no gate:
#    the bug report form's placeholder, and the README's install snippets
#    when the minor version changes.
git switch -c chore/release-vX.Y.Z
$EDITOR Cargo.toml .github/ISSUE_TEMPLATE/bug_report.yml README.md
cargo check

# 2. Move the CHANGELOG section: rename [Unreleased] to [X.Y.Z] - YYYY-MM-DD,
#    open a new empty [Unreleased] above it, and keep the links at the foot
#    of the file in step:
#
#      [Unreleased]: …/compare/vX.Y.Z...HEAD     <- was vPREV...HEAD
#      [X.Y.Z]:      …/compare/vPREV...vX.Y.Z    <- new line, under it
#
#    Release notes are the section plus the link definitions it names:
.github/scripts/extract-changelog.sh X.Y.Z

# 3. Re-measure. Every limit in the physics suite is about ten times a
#    measured baseline; PLAN.md §7.1 must describe the tree being released.
cargo test --workspace --all-features --release -- --nocapture | grep '^measured'
tools/preflight.sh
cargo semver-checks -p lagrangian --baseline-version PREV   # cargo install cargo-semver-checks

# 4. Check what will be uploaded. This builds the package from the tarball.
cargo publish -p lagrangian --locked --dry-run

# 5. Land it.
git commit -s -am "chore: release vX.Y.Z"
gh pr create --fill          # squash-merge once required-green passes

# 6. Tag the squash-merged commit on main. release.yml takes it from here:
#    verify the tag, re-run every CI gate, publish, cut the GitHub Release.
git switch main && git pull --ff-only
git tag -s vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

A published version cannot be replaced, only yanked. Step 4 is the last
point at which a mistake costs nothing.

## If something fails mid-release

- **`verify-version` failed**: the tag and the manifest disagree. Delete the
  tag (`git push origin :vX.Y.Z`), fix the version through a pull request,
  tag again.
- **`ci` failed**: nothing was published. Fix on `main` through a pull
  request, delete the tag, tag the new commit.
- **`publish` failed**: nothing was published unless the log says
  otherwise. If the `auth` step failed, or `cargo publish` answered 403,
  the Trusted Publishing link does not match this run: on crates.io the
  repository, the workflow file name and the environment must be exactly
  `vyncint/lagrangian`, `release.yml` and `release`. Fix the link and
  re-run the failed job; the tag is still right.
- **`github-release` failed**: the crate *is* on crates.io, and a version
  cannot be published twice. Do not delete the tag; re-run the failed job,
  or create the release by hand from
  `.github/scripts/extract-changelog.sh vX.Y.Z`.
