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
- **Dependency policy** (`cargo-deny` job): RUSTSEC advisories, yanked
  crates, license allowlist, and crates.io-only sources — on every PR, with
  weekly grouped Dependabot updates.
- **Workflow security** (`zizmor` job at `--persona=pedantic`): every
  GitHub Action is pinned to a full commit SHA, checkouts don't persist
  credentials, the workflow token is read-only by default with write scopes
  granted per job, and inputs reach shell steps only via environment
  variables. Accepted findings are documented in `.github/zizmor.yml`.
- **Release integrity**: releases re-run the full CI gates, and publishing
  uses crates.io Trusted Publishing (short-lived OIDC tokens) rather than a
  stored secret.
- **Provenance**: every commit requires a DCO sign-off, and CI rejects
  bot-authored commits except Dependabot's, which is exempt from the
  identity rule only: its messages are checked like any other
  (CONTRIBUTING.md, AI tooling policy).

Resource-exhaustion notes for the paranoid: every loop the caller's numbers
can reach is bounded. An integration stops at `max_steps` and reports
`MaxStepsExceeded`; a step size that collapses reports `StepSizeUnderflow`
instead of spinning; each Newton iteration and each root-finding call has a
fixed iteration cap; a state that becomes `NaN` or infinite is reported as
`NonFiniteState` rather than propagated. The engine, the `Stepper` and a
`Segment<N>` allocate nothing. A `Trajectory<N>` grows by one segment per
accepted step and therefore by at most `max_steps` segments per call.

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
