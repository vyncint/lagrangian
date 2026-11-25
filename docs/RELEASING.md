# Releasing lagrangian

One page, copy-pasteable. Maintainers only. Publishing is a maintainer's
action and never an agent's.

## Prerequisites

- Publishing auth is **crates.io Trusted Publishing**: crates.io →
  `lagrangian` → Settings → Trusted Publishing → GitHub, repository
  `vyncint/lagrangian`, workflow `release.yml`, environment `release`.
  No token or secret is stored for publishing. **Not linked yet** — see
  [Bootstrapping](#bootstrapping-the-first-publish).
- The publish job runs in the **`release` GitHub environment**, which must
  be restricted to `v*` tags, so an OIDC publish token can never be minted
  from a branch. **Not created yet.**

## What may break

`lagrangian` is pre-1.0. A `0.x` minor release may break; a patch release
may not. What dependents compile against is the trait surface —
`Lagrangian`, `Controlled`, `SceneMap` and `Scalar` as a bound — and a change
to it is always a `- **Breaking:**` bullet under **Changed** or **Removed**
in `CHANGELOG.md`. `Scalar` is sealed, so adding a method to it is not a
break; the `ad` module is exempt from semver until 1.0
([CLAUDE.md](../CLAUDE.md), "API Stability Rules").

There is no automated semver gate yet: `cargo-semver-checks` compares
against a published release, and none exists. Add the gate, as a `semver`
job in `ci.yml` with its line in CONTRIBUTING §1, in the first pull request
after the first publish.

## Cutting vX.Y.Z

```sh
# 0. Green main: every job of the latest CI run on the commit that will be
#    tagged, and a look at the latest canary run.
gh run list --workflow ci.yml --branch main --limit 1
gh run list --workflow canary.yml --limit 1

# 1. Bump the version: workspace.package.version in the root Cargo.toml,
#    then refresh the lockfile. The bug report form's version placeholder is
#    one more copy of the number.
git switch -c chore/release-vX.Y.Z
$EDITOR Cargo.toml .github/ISSUE_TEMPLATE/bug_report.yml
cargo update -p lagrangian

# 2. Move the CHANGELOG section: rename [Unreleased] to [X.Y.Z] - YYYY-MM-DD,
#    open a new empty [Unreleased] above it, and keep the compare links at
#    the foot of the file in step:
#
#      [Unreleased]: …/compare/vX.Y.Z...HEAD     <- was vPREV...HEAD
#      [X.Y.Z]:      …/compare/vPREV...vX.Y.Z    <- new line, under it
#
#    Release notes are the section plus the link definitions it names:
.github/scripts/extract-changelog.sh X.Y.Z

# 3. Re-measure. Every threshold in the physics suite is about ten times a
#    measured baseline; PLAN.md §7 must describe the tree being released.
cargo test --workspace --all-features
tools/preflight.sh

# 4. Land it.
git commit -s -am "chore: release vX.Y.Z"
gh pr create --fill          # squash-merge once required-green passes

# 5. Tag the squash-merged commit on main. release.yml takes it from here:
#    verify the tag, re-run every CI gate, publish, cut the GitHub Release.
git switch main && git pull --ff-only
git tag -s vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

## Bootstrapping the first publish

Trusted Publishing is configured against a crate that already exists, so the
very first publish of a new name cannot use it. When termlens-cli was
bootstrapped this way, the first publish went out with a one-shot API token
and the link was added afterwards; check the current crates.io
documentation before assuming the rule still holds.

1. `cargo publish -p lagrangian --locked` from a clean checkout of the
   tagged commit, with a crates.io API token scoped to publishing new crates
   and revoked immediately afterwards.
2. Link Trusted Publishing as described under Prerequisites, and create the
   `release` environment restricted to `v*` tags.
3. Every later release goes through `release.yml`.

[PLAN.md](../PLAN.md) §1 explains why the name should be reserved by a small
working release and not by an empty placeholder.

## If something fails mid-release

- **`verify-version` failed**: the tag and the manifest disagree. Delete the
  tag (`git push origin :vX.Y.Z`), fix the version through a pull request,
  tag again.
- **`ci` failed**: nothing was published. Fix on `main` through a pull
  request, delete the tag, tag the new commit.
- **`publish` failed**: nothing was published unless the log says
  otherwise. An authentication failure almost always means the Trusted
  Publishing link or the `release` environment is missing.
- **`github-release` failed**: the crate *is* on crates.io, and a version
  cannot be published twice. Do not delete the tag; re-run the failed job,
  or create the release by hand from
  `.github/scripts/extract-changelog.sh vX.Y.Z`.
