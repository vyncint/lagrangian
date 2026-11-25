# Contributing to lagrangian

Thanks for your interest! This document covers everything you need to get a
change from your editor into `main`.

> **This project follows the contributor pattern of its siblings** — the same
> commit rules, the same DCO, the same AI policy, the same CI and release
> shape: [termlens](https://github.com/vyncint/termlens),
> [mossaic](https://github.com/vyncint/mossaic),
> [launchbound](https://github.com/vyncint/launchbound),
> [reconverge](https://github.com/vyncint/reconverge). Learn it once.

## Where to start

- Read [PLAN.md](PLAN.md) for what the crate is, what it is not (§6), and the
  measured baselines behind every numeric threshold (§7).
- Open a draft PR early — it is the cheapest way to confirm direction before
  you write much.
- Two things trip people up here: commits need `git commit -s` for the
  [DCO](#5-developer-certificate-of-origin-dco), and a test threshold is
  never loosened without recording the new measurement
  ([§3](#3-testing-policy)).

## 1. Dev setup

```sh
git clone https://github.com/vyncint/lagrangian
cd lagrangian
cargo test --workspace --all-features   # the whole suite: unit + physics verification + doctests, `serde` included
```

`rust-toolchain.toml` pins stable and installs the two extra targets the
`no-std` gates compile for. `--all-features` matters: `serde` is off by
default, and without it the round-trip test never builds. That is the `test`
job. CI gates on more than the test job, and every gate is reproducible
locally — run these before pushing and nothing in CI should surprise you.
[`tools/preflight.sh`](tools/preflight.sh) runs the whole list below in order
and prints one line per gate; `--fast` narrows it to fmt, clippy and the test
suite. A gate whose optional tooling is not installed — `cargo-deny`, `pipx`,
the 1.85 toolchain, a cross target — is skipped by name rather than failed.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets -- -D warnings    # default features: what `cargo add lagrangian` gives you
cargo clippy -p lagrangian --all-targets --no-default-features --features libm -- -D warnings          # the library alone: no std, no allocator
cargo clippy -p lagrangian --all-targets --no-default-features --features libm,alloc -- -D warnings
.github/scripts/check-readme-links.sh README.md CONTRIBUTING.md .github/PULL_REQUEST_TEMPLATE.md docs/*.md   # every in-repo link target exists; the packaged README and the pull-request template may not use relative ones
.github/scripts/check-dependency-tree.sh                 # zero runtime dependencies by default, `libm` alone without std
cargo test --workspace                                   # default features
cargo test -p lagrangian --features serde
cargo test -p lagrangian --no-default-features --features libm          # the same physics on libm's sin and cos, no allocator
cargo test -p lagrangian --no-default-features --features libm,alloc
cargo bench -p benches --no-run                          # the benchmarks build; none is run, since a shared runner's time is not a baseline
cargo check -p lagrangian --no-default-features --features libm
cargo check -p lagrangian --no-default-features --features libm,alloc
cargo check -p lagrangian --no-default-features --features libm --target thumbv7em-none-eabihf          # bare metal: proves no std and no allocator
cargo check -p lagrangian --no-default-features --features libm,alloc --target thumbv7em-none-eabihf
cargo check -p lagrangian --target wasm32-unknown-unknown                                                # the browser
.github/scripts/check-ci-gates-listed.sh                  # every single-line CI gate is documented here
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features
.github/scripts/check-package-contents.sh                 # the published crate ships its README and both licenses, and nothing outside the allowlist
.github/scripts/check-forbid-unsafe.sh                    # SECURITY.md's "no unsafe" statement matches the code
cargo deny check                          # cargo install cargo-deny
pipx run zizmor==1.29.0 --persona=pedantic .github/   # workflow audit; needs GH_TOKEN for the online checks
cargo +1.85 check --workspace --locked --all-targets    # the MSRV: `rust-version` in Cargo.toml
cargo +1.85 check -p lagrangian --all-features --all-targets --locked           # …and every optional feature at the same floor
cargo +1.85 check -p lagrangian --no-default-features --features libm --all-targets --locked
tools/link-gate-selftest/run.sh                         # the link gate can fail, with no extra tooling…
tools/attribution-gate-selftest/run.sh                  # …and the AI-attribution gate, likewise
tools/dco-gate-selftest/run.sh                          # …and the DCO gate, its one-commit squash exemption included
tools/preflight-selftest/run.sh                         # …and the runner accounts for each gate it runs, failures included
tools/gates-listed-selftest/run.sh                      # …and the gates-listed gate, named, quoted and folded gates included
tools/release-notes-selftest/run.sh                     # …and the release notes carry the link definitions their section uses
```

The list is written out here rather than left as a pointer because a first
PR that fails on `cargo fmt` after following the setup to the letter is an
avoidable bad first experience. If it ever disagrees with
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), the workflow is the
source of truth and this list is the bug — and the `gates-listed` CI job
(`.github/scripts/check-ci-gates-listed.sh`) fails when the two disagree,
so the bug cannot sit here unnoticed.

There is nothing to install beyond a Rust toolchain. The core crate has no
runtime dependencies, and the one dev-dependency is `serde_json`.

## 2. Project layout

- `crates/lagrangian/` — the published library: the sealed `Scalar` trait
  and forward-mode AD types, the equations-of-motion engine, the `Stepper`
  and its solvers, `Segment<N>` and `Trajectory<N>`, and the built-in
  benchmark systems. `tests/` is the physics verification suite, and
  `examples/svg_showcase.rs` draws figures with no dependencies.
- `benches/` — the benchmarks, with a timing loop and no dependencies:
  `cargo bench -p benches`, or `cargo bench -p benches -- steppers` for
  those whose name contains a word. Not published.
- [PLAN.md](PLAN.md) — positioning, scope and non-goals, and §7, the table of
  measured baselines every test threshold is derived from, and the measured
  speed.
- [PROMPT.md](PROMPT.md) — the implementation specification.
- [CLAUDE.md](CLAUDE.md) — the architectural rules and the mathematical and
  IEEE-754 invariants. **Read it before touching `ad.rs`, `mechanics.rs`,
  `linalg.rs` or `solver/`.**
- [AGENTS.md](AGENTS.md) — the brief for coding agents working *on* the
  crate.
- `tools/` — contributor tooling. `preflight.sh` runs §1's gate list in
  order, one line per gate (`--fast` for fmt, clippy and the tests), and
  each `*-selftest/run.sh` proves that a gate — or the runner itself — goes
  red when it should, before a green run of it is trusted.
- `.github/` — CI, commit policy enforcement, release automation.

## 3. Testing policy

- Every feature or bug fix lands with tests. No exceptions; a numerical
  library whose claims are untested is a liability.
- **Thresholds come from measurement.** A physics test asserts a number that
  was measured on this implementation, at about ten times the measured
  value, and [PLAN.md §7](PLAN.md) records the baseline. The step from
  "what I expected" to "what it does" is where five of the first plan's
  targets turned out to be unreachable as written.
- **Never loosen a threshold to make a test pass.** If a change moves a
  measured value, record the new measurement and the reason in PLAN.md §7 in
  the same pull request. A threshold that drifts without a record is a
  claim nobody can check.
- **Knots and mid-step samples get separate thresholds.** The step
  controller does not bound the interpolant's error
  ([CLAUDE.md](CLAUDE.md), invariant 7), so one number for both is either
  loose at knots or false between them.
- **A test must hold on `libm` too.** The `features` job runs the suite with
  `--no-default-features --features libm`, where `sin` and `cos` come from a
  different library and differ in the last bits. A threshold that passes
  only with the platform's math is a claim about the platform.
- **Every quantitative claim in the README names its test.** No "guarantees"
  and no "machine precision" without a number and the name of the test or
  benchmark that reproduces it.
- **The next compilers run once a week.** `canary.yml` runs clippy with
  `-D warnings`, the suite and rustdoc with `-D warnings` on beta and
  nightly every Monday, and on demand from the Actions tab. It is not
  required, so it never blocks a pull request; a red canary means the next
  toolchain breaks something here. It also runs the randomized robustness
  harness for ten minutes, from seeds that change every run.
- **Random and extreme inputs are a test.** `tests/robustness.rs` pushes
  random and extreme values (`NaN`, `±∞`, subnormals, `±1e300`, zero and
  negative masses) through every public entry point, checks that no call
  panics, hangs or returns an `Ok` that breaks its documented promise, and
  names the seed and the case of anything it finds. Every `cargo test`
  runs it for two seconds; set `LAGRANGIAN_ROBUSTNESS_SECONDS` for longer
  and `LAGRANGIAN_ROBUSTNESS_SEED` to start from a reported seed.

## 4. Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, `refactor:`, `perf:` —
scope optional (`feat(solver): …`). Subject line: imperative mood,
≤ 72 characters.

This is a convention, and review holds it: no CI job checks the type or
the length of a commit subject. The two commit rules CI does enforce, on
every commit of a pull request, are the sign-off
([§5](#5-developer-certificate-of-origin-dco)) and no AI attribution
([§6](#6-ai-tooling-policy)).

## 5. Developer Certificate of Origin (DCO)

Every commit must be signed off:

```sh
git commit -s
```

This appends `Signed-off-by: Your Name <you@example.com>` and certifies you
wrote the change or otherwise have the right to submit it under the project
license — the [Developer Certificate of Origin](https://developercertificate.org),
the same lightweight model the Linux kernel uses. The sign-off email must
match the commit author email; CI enforces this on every commit in a PR.

**There is no CLA. DCO only.** You keep your copyright.

Forgot to sign off? `git commit --amend -s` for the last commit, or
`git rebase --signoff main` for a whole branch, then force-push.

One exception, and it is GitHub's rather than ours: a pull request
**squash-merged through the web UI** has its author email rewritten by GitHub
*after* the sign-off was written, so an exact match is impossible by
construction. Such a commit must carry a sign-off, but is not matched against
an author it did not choose. The commits that went into the PR were already
checked, address and all, on the branch.

GitHub also *writes* that message, and it drops the trailers of the commits it
squashed whenever the branch contained a merge commit — pressing **Update
branch** is enough to cause it. The check therefore exempts exactly one
commit — the tip of a push to main. **Keep your branch up to date by
rebasing, not merging:**

```sh
git fetch origin && git rebase origin/main
git push --force-with-lease
```

## 6. AI tooling policy

**AI assistance is welcome here — use whatever helps.** This project was
built with it. There is an [AGENTS.md](AGENTS.md) briefing coding agents on
the layout, the commands, and the house style.

**AI attribution is not welcome.** No `Co-Authored-By` trailer naming an
assistant, model or vendor; no "Generated with …" footer; no robot emoji; no
bot identity as author or committer, save the one carve-out below. Whoever
opens the pull request is the author of record, takes responsibility under the
DCO, and the history should say so — a tool cannot certify the DCO, which is
the whole point of it.

This is enforced, not requested: `commit-policy.yml` runs
[`check-no-ai-attribution.sh`](.github/scripts/check-no-ai-attribution.sh) and
[`check-dco.sh`](.github/scripts/check-dco.sh) over every commit in a pull
request. Run them yourself first — both take a range:

```sh
.github/scripts/check-dco.sh main..HEAD
.github/scripts/check-no-ai-attribution.sh main..HEAD
```

If a check fails, rewrite the message rather than arguing with it:

```sh
git commit --amend            # the last commit
git rebase -i main            # several, marking each `reword`
git push --force-with-lease
```

`.claude/settings.json` turns co-author trailers off for agents that read
repository settings. That is a courtesy; the check in CI is the boundary.
Contributions authored *by* an autonomous account are not accepted.

**One carve-out: a named dependency bot.** Dependabot is exempt from the
*identity* half of the check and from nothing else. The message rules still
apply in full. Adding another bot means naming it in
`check-no-ai-attribution.sh`: the allowlist is a list on purpose, so that
widening it is a visible decision.

## 7. PR flow

- Branch from `main`; name branches `feat/…`, `fix/…`, `docs/…`, `ci/…`.
- PRs are **squash-merged**. A pull request of one commit lands on `main`
  under that commit's subject, and one of several under the PR title,
  unless the maintainer sets another subject at merge — so keep both in
  Conventional Commit form. Branches are deleted on merge.
- Required checks: `required-green`, which needs every CI job — `fmt`,
  `clippy`, `test`, `features`, `no-std`, `gates-listed`, `msrv`, `docs`,
  `deny` and `zizmor` — plus `commit-policy` (DCO + attribution). The
  `main` ruleset enforces them, along with a pull request for every change,
  squash merges and linear history; nobody can push to `main` directly or
  bypass it. Tags matching `v*` publish a release, and only a maintainer
  can create one.
- Workflows may use GitHub's own actions and the four third-party actions
  they already pin by commit (`dtolnay/rust-toolchain`,
  `Swatinem/rust-cache`, `EmbarkStudios/cargo-deny-action`,
  `rust-lang/crates-io-auth-action`). A new one is a repository setting to
  change as well as a line to add.
- **Every change lands with a test, and the test must be able to fail.** If
  you add a guard, break it once and watch it go red before you commit.
- **Say what you did not do.** A PR that lists what it left out and why is
  worth more than one implying completeness. An honest gap is cheap; a false
  claim is expensive.
- Review: small, focused PRs get reviewed faster. Update `CHANGELOG.md`
  under `[Unreleased]` for any user-facing change.
- **The trait surface is what dependents compile against.** `Lagrangian`,
  `Controlled`, `SceneMap` and `Scalar` (as a bound) change only with a
  version bump and a `- **Breaking:**` bullet in the CHANGELOG. `Scalar` is
  sealed so that adding a method is not a break; the `ad` module is exempt
  from semver until 1.0 ([CLAUDE.md](CLAUDE.md), "API Stability Rules").

## 8. Release process

Releases are cut by maintainers only; the checklist lives in
[docs/RELEASING.md](docs/RELEASING.md).
