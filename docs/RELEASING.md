# Releasing lagrangian

One page, copy-pasteable. Maintainers only.

## Prerequisites

- Publishing auth is a **crates.io API token** held as the repository secret
  `CARGO_REGISTRY_TOKEN`. `release.yml` passes it to `cargo publish`.
  `0.1.0`, the crate's first release, had to go out this way: Trusted
  Publishing is linked to a crate that already exists.
- The publish job runs in the **`release` GitHub environment**. GitHub
  creates it the first time the workflow runs, with no rules. Restrict it to
  `v*` tags (Settings → Environments → release → Deployment branches and
  tags), so the token can never be used from a branch.

## Moving to Trusted Publishing

Worth doing before the second release: it replaces a long-lived secret with a
token that lives for one job.

1. crates.io → `lagrangian` → Settings → Trusted Publishing → GitHub:
   repository `vyncint/lagrangian`, workflow `release.yml`, environment
   `release`.
2. In `release.yml`'s `publish` job, grant `id-token: write` beside
   `contents: read`, and replace the publish step with:

   ```yaml
   - uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9 # v1.0.6
     id: auth
   - run: cargo publish -p lagrangian --locked
     env:
       CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}
   ```

3. Delete the `CARGO_REGISTRY_TOKEN` secret and revoke the token on
   crates.io.
4. Update SECURITY.md's "Release integrity" bullet and this section.

## What may break

`lagrangian` is pre-1.0. A `0.x` minor release may break; a patch release
may not. What dependents compile against is the trait surface —
`Lagrangian`, `Controlled`, `SceneMap` and `Scalar` as a bound — and a change
to it is always a `- **Breaking:**` bullet under **Changed** or **Removed**
in `CHANGELOG.md`. `Scalar` is sealed, so adding a method to it is not a
break; the `ad` module is exempt from semver until 1.0
([CLAUDE.md](../CLAUDE.md), "API Stability Rules").

There is no automated semver gate yet. `cargo-semver-checks` compares
against a published release, and `0.1.0` is the first, so the gate can be
added now: a `semver` job in `ci.yml` with its line in CONTRIBUTING §1.

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
  otherwise. A 403 means the token is missing, expired, or not allowed to
  publish this crate: the `CARGO_REGISTRY_TOKEN` secret, or its scopes on
  crates.io. Fix the secret and re-run the failed job; the tag is still
  right.
- **`github-release` failed**: the crate *is* on crates.io, and a version
  cannot be published twice. Do not delete the tag; re-run the failed job,
  or create the release by hand from
  `.github/scripts/extract-changelog.sh vX.Y.Z`.
