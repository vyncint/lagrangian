# Security Policy

`lagrangian` is a numerical library. It reads no files, opens no sockets,
spawns no processes and parses no input formats: it evaluates functions the
caller wrote, on numbers the caller supplied. It is not a security boundary
and is not meant to process untrusted input. That said, a bug that lets
ordinary numeric input corrupt memory, loop without bound, or abort a process
that asked for a `Result` is treated as a security issue.

## Posture

What the project does continuously, enforced by required CI on every change:

- **The crate contains no `unsafe`.** The crate root carries
  `#![forbid(unsafe_code)]`, which no inner `#[allow]` can override, so a
  block anywhere in the library is a compile error rather than a review
  comment. `.github/scripts/check-forbid-unsafe.sh` holds this paragraph to
  the code: it fails if the attribute is removed or if this sentence is.
- **No runtime dependencies by default.** With the default `std` feature the
  crate resolves nothing but itself; without `std` it resolves `libm` and
  nothing else; `serde` is opt-in. `.github/scripts/check-dependency-tree.sh`
  asserts each of those trees exactly, so a new dependency is a visible
  decision and not a side effect.
- **Random and extreme inputs** (`tests/robustness.rs`): `NaN`, `±∞`,
  subnormals, `±1e300` and nonsensical masses go through every public
  entry point, and the run fails on a panic, a call that does not return,
  or an `Ok` that breaks its documented promise. Every CI run spends two
  seconds on it and the weekly canary ten minutes, from new seeds each
  week. Against 0.1.0 it fails within a minute, on the hang of #4.
- **Dependency policy** (`cargo-deny` job): RUSTSEC advisories, yanked
  crates, license allowlist, and crates.io-only sources — on every PR, with
  weekly grouped Dependabot updates.
- **Workflow security** (`zizmor` job at `--persona=pedantic`): every
  GitHub Action is pinned to a full commit SHA, checkouts don't persist
  credentials, the workflow token is read-only by default with write scopes
  granted per job, and inputs reach shell steps only via environment
  variables. Accepted findings are documented in `.github/zizmor.yml`.
- **Release integrity**: releases re-run the full CI gates before anything
  is published. Publishing uses a crates.io API token held as a repository
  secret, because a crate's first publish cannot use Trusted Publishing;
  moving to short-lived OIDC tokens and deleting the secret is the next
  step in `docs/RELEASING.md`.
- **Provenance**: every commit requires a DCO sign-off, and CI rejects
  bot-authored commits except Dependabot's, which is exempt from the
  identity rule only: its messages are checked like any other
  (CONTRIBUTING.md, AI tooling policy).

Resource exhaustion, and what bounds it. Each figure is in the rustdoc of
the item named, and each bound has a test:

- **Steps per call.** `simulate` and `Trajectory::extend` take at most
  `SimConfig::max_steps` steps (default 1,000,000) and then report
  `MaxStepsExceeded`. An end time that is not finite is refused with
  `InvalidConfig` before any step is taken.
- **Memory per call.** Those calls keep one `Segment<N>` per step, of
  `8·(6N + 2)` bytes: at most 107 MiB at `N = 2` and 748 MiB at `N = 16`
  with the default budget. Lower `max_steps` when the span or the system
  comes from someone else. The engine, the `Stepper` and a `Segment<N>`
  allocate nothing.
- **Work per step.** One `Stepper::step` makes at most 1000 attempts, and
  the step size never exceeds `f64::MAX`; a step size that collapses reports
  `StepSizeUnderflow` instead of spinning. Each Newton iteration and each
  root-finding call has a fixed iteration cap.
- **Work per query.** `Trajectory::crossings_with` looks at no more than
  2¹⁶ points inside each step, and `Trajectory::energy_drift` takes no more
  than 2²⁰ samples, whatever count is asked for.
- **Stack.** Nothing allocates, so the engine's working set is on the
  stack and grows with the square of `N`: 189 KiB for `accelerations` at
  `N = 32` in a release build, 292 KiB in a debug one. The crate
  documentation has the table by operation and `N`, and `tests/stack.rs`
  holds it to the code. An overflow aborts the process on an operating
  system and traps on `wasm32`, but on bare metal with `cortex-m-rt`'s
  default layout it can run into static data silently: the one
  memory-safety hazard `#![forbid(unsafe_code)]` cannot rule out. Size the
  stack from the table there, and put it below the static data
  (`flip-link`).
- **Non-finite values.** A state that becomes `NaN` or infinite is reported
  as `NonFiniteState` rather than propagated, and no step returns a segment
  that overflows between its ends.
- **Data read with the `serde` feature.** A file that no solver could have
  written is refused: times or an initial state that are not finite, a
  duration that is not positive and finite, a polynomial that overflows
  between the ends of its segment, or segments that do not join. How much
  input to read at all is the deserializer's choice, not this crate's.

A `Stepper` driven by a loop of the caller's has no step budget: the loop
is the caller's, and so is its bound.

What is *not* bounded is the caller's own code: a `Lagrangian` implementation
that loops forever or panics does so inside the caller's process, as any
function would.

## Supported versions

| Version        | Supported          |
| -------------- | ------------------ |
| latest 0.x     | :white_check_mark: |
| older releases | :x:                |

Only the most recent release receives fixes. Pre-1.0, fixes ship as a new 0.x
release rather than backports.

## Reporting a vulnerability

**Please do not open a public issue for suspected vulnerabilities.**

- Preferred: use GitHub
  [Private Vulnerability Reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  — the "Report a vulnerability" button under the Security tab.
- Interim / fallback: email the maintainer at <vyncint@icloud.com> with the
  subject line `[SECURITY] lagrangian`.

You should receive an acknowledgement within 72 hours.

## Disclosure

We follow **90-day coordinated disclosure**: we ask that you give us up to 90
days from your report to publish a fix before any public disclosure. In
practice we aim to be much faster. We will credit reporters in the advisory
and CHANGELOG unless you prefer otherwise.

There is **no bug bounty** for this project.
