# Working on lagrangian

Instructions for coding agents — and useful to humans. **lagrangian** is a
zero-algebra equations-of-motion engine: the user writes `L = T - V` in plain
generic Rust, and the crate returns exact accelerations, momenta, energy and
linearization through stack-allocated forward-mode automatic differentiation,
with a streaming `Stepper` and a scrubbable `Trajectory<N>` built on top.

This file is the brief on layout, commands and house rules. `CLAUDE.md`
holds the architectural rules and the mathematical invariants and is not
repeated here. `CONTRIBUTING.md` is the full contributor document and wins
wherever it and this file disagree.

## Layout

- `crates/lagrangian/` — the published library.
  - `scalar.rs` the sealed `Scalar` trait, `ad.rs` the dual-number types,
    `linalg.rs` the stack-allocated solvers.
  - `mechanics.rs` the engine: `Lagrangian`, `Controlled`, `eom_terms`,
    `accelerations`, energy, `linearize`, `SceneMap`, `velocity_of`.
  - `segment.rs` the `Copy` quintic Hermite piece, `solver/` the `Stepper`
    and its integrators, `trajectory.rs` (behind `alloc`) the scrubbable
    collection of segments, `systems.rs` the built-in benchmark systems.
  - `tests/` — the physics verification suite. `examples/svg_showcase.rs`
    draws figures with no dependencies.
- `PLAN.md` — positioning, non-goals (§6), and the measured baselines every
  test threshold is derived from (§7).
- `PROMPT.md` — the implementation specification.
- `CLAUDE.md` — **read it before touching `ad.rs`, `mechanics.rs`,
  `linalg.rs` or `solver/`.**

## Build and test

```sh
cargo test --workspace --all-features       # the whole suite
cargo test -p lagrangian --test energy      # one verification file
tools/preflight.sh --fast                   # fmt, clippy and the tests
tools/preflight.sh                          # every gate CI runs, one line each
```

Stable toolchain, MSRV 1.85 (checked in CI against the committed lockfile).

## Things that will bite you here

- **The crate must build with no `std` and no allocator.** A `Vec`, a
  `format!` or a `std::` path outside the `alloc`-gated modules compiles on
  your machine and fails the `no-std` job. `cargo check -p lagrangian
  --no-default-features --features libm --target thumbv7em-none-eabihf` is
  the fast way to find out.
- **Zero runtime dependencies is a gate, not a wish.**
  `check-dependency-tree.sh` fails if the default build resolves anything
  but the crate itself.
- **A threshold is a measurement.** Do not write the number you expect.
  Measure, set the threshold at about ten times that, and record the
  baseline in `PLAN.md` §7. Never loosen one without recording why.
- **`0.0 * inf` is `NaN`, and systems start at rest.** The `powi` and
  `hypot` rules in `CLAUDE.md` exist because the obvious derivative formulas
  produce `NaN` at `dq = 0`. Every AD change is tested at rest.
- **The suite also runs on `libm`.** The last bits of `sin` and `cos`
  differ there, so a test that compares floats exactly, or that passes by a
  hair, fails in the `features` job.

## The rules that will fail CI

Two, and they are the same in every one of these repositories.
`commit-policy.yml` checks both on every commit of a pull request.

1. **DCO sign-off.** `git commit -s`, and the `Signed-off-by:` email must
   match the commit author's. Forgot? `git commit --amend -s --no-edit`, or
   `git rebase --signoff main` for a branch.
2. **No AI attribution.** See below — this one is about you, and it is the
   rule most likely to catch an agent out.

Run them yourself before pushing; both scripts take a commit range:

```sh
.github/scripts/check-dco.sh main..HEAD
.github/scripts/check-no-ai-attribution.sh main..HEAD
```

The commit style is a convention, not a gate. No CI job checks the type
or the length of a commit subject, so review is what holds it:

- **Conventional Commits.** `feat:`, `fix:`, `docs:`, `test:`, `ci:`,
  `chore:`, `refactor:`, `perf:` — imperative mood, subject line of 72
  characters or fewer, scope optional (`fix(solver): …`).

## Using AI here

**You are welcome.** This project was built with AI assistance and says so
in its CONTRIBUTING. Use whatever helps.

**You are not a contributor.** Do not add yourself to the history:

- no `Co-Authored-By:` trailer naming an assistant, a model, or a vendor,
- no "Generated with …" footer, no robot emoji,
- no bot account as author or committer — save a named dependency bot, which
  is exempt from the identity rule only and still checked on its message.

The human who opens the pull request is the author of record and takes
responsibility for the change under the DCO. That is what the sign-off
certifies, and it cannot be certified by a tool. `.claude/settings.json`
turns co-author trailers off for agents that read it; the check in CI is the
boundary, and it reads every commit in the range.

If CI catches one, the fix is to rewrite the message, not to argue with it:

```sh
git commit --amend            # the last commit
git rebase -i main            # several, marking each `reword`
git push --force-with-lease
```

## What good work looks like here

These repositories share a house style, and it is stricter than most:

- **Evidence over assertion.** A claim about accuracy says what was measured,
  on which system, with which solver and tolerance. "Drift 2.5e-12 at
  rtol 1e-12 over t in [0, 20]" is the standard; "conserves energy" is not.
- **Every change lands with a test**, and the test must be able to fail. If
  you add a guard, prove it catches the thing — break it once and watch it go
  red before you commit.
- **Comments say *why*, never *what*.** The diff shows what. A comment earns
  its place by recording the reason, the alternative rejected, or the failure
  that motivated the line.
- **Say what you did not do.** A pull request that lists what it left out and
  why is worth more than one that implies completeness. If something is
  unverified, say so — an honest gap is cheap and a false claim is expensive.
- **Documentation is checked, not maintained.** The README's Rust blocks are
  compiled and run as doctests, and its numbers name the tests that
  reproduce them. Do not break that pattern by hand-editing a number.

## Pull requests

Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `ci/…`). PRs are
**squash-merged**. A PR of one commit lands on `main` under that commit's
subject, and a PR of several under the PR title, unless the maintainer sets
another at merge — so write both as Conventional Commits. Update
`CHANGELOG.md` under `[Unreleased]` for anything user-facing.

## Releasing

Tag `vX.Y.Z` on `main`; `release.yml` gates, publishes to crates.io, and
cuts the GitHub Release. A published version cannot be taken back, so a
release is cut only when the maintainer asks for that release. See
`docs/RELEASING.md`.
