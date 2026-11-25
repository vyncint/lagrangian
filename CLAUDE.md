# CLAUDE.md — `lagrangian` Workspace Guidelines

## Project Overview
`lagrangian` is a zero-algebra equations-of-motion engine for analytical mechanics in Rust. Users implement `Lagrangian<const N: usize>` ($L = T - V$) in plain generic Rust arithmetic. The engine extracts exact Euler-Lagrange equations via stack-allocated forward-mode automatic differentiation and exposes them as plain functions (`eom_terms`, `accelerations`, `mass_matrix`, `momentum`, `energy`, `linearize`) that any ODE solver, game loop or controller can call.

Built on the engine: a streaming `Stepper`, adaptive (`Dopri5`) and variational symplectic solvers, `Copy` quintic Hermite `Segment<N>`s, and a continuous, $C^2$-smooth, scrubbable `Trajectory<N>`.

`PLAN.md` holds the positioning, roadmap and measured baselines. `PROMPT.md` holds the implementation specification. **[AGENTS.md](AGENTS.md)** is the brief on layout, commands and house rules for every coding agent, and `CONTRIBUTING.md` is the full contributor document; this file holds the architectural rules and the mathematical invariants, and nothing here is repeated there.

The short version of the house rules: Conventional Commits, `git commit -s` for the DCO, and **no AI attribution in the history** — you are welcome to use AI here, but the human opening the pull request is the author of record. Publishing to crates.io is the maintainer's action, never an agent's.

## Workspace Layout
- `crates/lagrangian` — Core library (`#![no_std]`, `#![forbid(unsafe_code)]`; `alloc` is an optional feature). `tests/` is the physics verification suite. `README.md` and the two `LICENSE-*` files in it are symlinks to the repository root so the published crate carries them.
- `crates/lagrangian/examples/svg_showcase.rs` — Headless scientific SVG generator (zero dependencies). It lives inside the package because Cargo ignores `examples/*.rs` at a virtual workspace root.
- `crates/bevy_lagrangian` — Bevy plugin, published separately.
- `examples/egui_playground` — Interactive native & WASM (`Trunk.toml`) playground. Workspace member, `publish = false`.
- `examples/ode_adapters` — `accelerations` as the right-hand side for `ode_solvers` and `diffsol`. Workspace member, `publish = false`.
- `benches` — Criterion benchmarks. Workspace member, `publish = false`.
- `.github/`, `tools/` — CI, commit-policy enforcement, release automation and the local gate runner, sharing one pattern with the maintainer's other repositories.

Only `crates/lagrangian` exists so far; the other members are planned (`PLAN.md` §8).

## Hard Architectural Rules
1. **`#![no_std]` Core, `alloc` Optional:** `crates/lagrangian` MUST compile with `--no-default-features --features libm` and no `alloc`. That build includes the engine functions, `Stepper`, `Segment<N>` and all solvers. `Trajectory<N>`, `simulate()` and `crossings()` sit behind the `alloc` feature. `std` implies `alloc`. Enabling neither `std` nor `libm` is a `compile_error!`.
2. **Minimal Core Dependencies:** ZERO runtime dependencies when the default `std` feature is enabled; ONLY `libm` when `std` is disabled; optional `serde` behind a feature flag. Never add `nalgebra`, `ndarray`, or graphics/windowing crates to `crates/lagrangian`. serde does not implement its traits for `[T; N]` with const-generic `N` (checked against serde 1.0.229), so the `serde` feature uses hand-written impls, never a helper crate. The only dev-dependency allowed in the core crate is `serde_json` for the round-trip test.
3. **Zero Heap Allocations in Inner Loops:** All AD types (`Dual<T>`, `Grad<N>`, `Taylor2`), state vectors `[f64; N]`, mass matrices `[[f64; N]; N]` and `Segment<N>` MUST be `Copy` and stack-allocated using const generics. `alloc::vec::Vec` is only for storing `Trajectory<N>` segments and event results.
4. **No `unsafe`:** Keep `#![forbid(unsafe_code)]` in `crates/lagrangian`.
5. **Engine First:** Built-in solvers consume only the public engine functions. Anything a built-in solver can compute must be reachable by an external solver. `simulate()` is a loop over `Stepper`, never a separate code path.
6. **No `generic_const_exprs`:** Stable Rust cannot express `[f64; 2 * N]`. Return block matrices (`a_q`, `a_dq`, `b`) instead of a dense $2N \times 2N$ state matrix.

## API Stability Rules
1. **`Scalar` is sealed.** Users bound on it and never implement it, so methods can be added without a breaking release.
2. **`lagrangian::ad` is public but exempt from semver** until `1.0`. User code writes `S: Scalar` and never names `Dual`, `Grad<N>` or `Taylor2`.
3. **`#[non_exhaustive]`** on every public enum and configuration struct (`SimError`, `Solver`, `SimConfig`, `CrossingDirection`).
4. **`f64` only** in the public numeric surface for `0.x`.
5. **MSRV:** edition 2024, `rust-version = "1.85"`. Raise it only in a minor release.

## Critical Mathematical & IEEE-754 Invariants
1. **Safe `powi` on AD Types:** ALWAYS implement `Scalar::powi(n: i32)` on AD types using binary exponentiation by squaring via `*` and `.recip()`. NEVER compute derivatives of `powi` via `n * (n - 1) * x.powi(n - 2)` or `exp(n * ln(x))` — starting from rest (`dq = 0.0`) causes `0.0 * inf = NaN`, and negative velocities cause `ln(-x) = NaN`.
2. **Safe `hypot(0.0, 0.0)`:** Guard `hypot(x, y)` at `(0.0, 0.0)` so derivatives evaluate cleanly to `0.0` rather than `0.0 / 0.0 = NaN`. Do NOT mask `sqrt` at `0.0`: its derivative is infinite by mathematics. Document that norms should use `hypot`.
3. **Convective Directional AD:** Extract Euler-Lagrange equations in $N + 2$ passes of `lagrangian` using `Grad<N>` and `Dual<Grad<N>>`. The convective vector $C$ is the directional derivative of $p = \nabla_{\dot{q}} L$ along $(\delta q = \dot{q}, \delta \dot{q} = 0, \delta t = 1)$, obtained in 1 pass. Each pass is $O(N)$ wide, so total cost is $O(N^2) \cdot \text{cost}(L)$: a constant-factor win over scalar hyperduals, never described as asymptotic. Cross-check against scalar `HyperDual` (`Dual<Dual<f64>>`) in unit tests to a relative tolerance of `1e-12`, not bit-for-bit (symmetrizing $M$ changes the last bits).
4. **Mass Matrix Solver:** Symmetrize $M_{ij} = 0.5 * (M_{ij} + M_{ji})$. Solve $M\ddot{q} = \text{RHS}$ with unrolled $N = 1, 2, 3$ fast paths and stack-allocated partial-pivoting LU (`PA = LU`) for general $N$. NEVER assume $M$ is positive definite (General Relativity geodesic metrics have Lorentzian signature $(-, +, +, +)$). The general LU path MUST be exercised by a test with $N \ge 4$ and an indefinite matrix; the unrolled paths MUST agree with it on random symmetric indefinite matrices.
5. **Legendre Energy:** Always compute $p = \nabla_{\dot{q}} L$, $E = p \cdot \dot{q} - L$, $V_{\text{eff}}(q, t) = -L(q, 0, t)$, and $T_{\text{eff}} = E - V_{\text{eff}}$. Never assume $T = \frac{1}{2}\dot{q}^T M \dot{q}$. Normalize relative drift by $\max(|E_0|, \max_t T_{\text{eff}})$, never by $|E_0|$ alone, which can be exactly zero.
6. **$C^2$ Quintic Hermite `Segment<N>`:** Store 6 coefficients per DOF per segment derived from $(q_k, \dot{q}_k, \ddot{q}_k)$ and $(q_{k+1}, \dot{q}_{k+1}, \ddot{q}_{k+1})$ so $\dot{q}(t) \equiv q'(t)$ and $\ddot{q}(t) \equiv q''(t)$ are continuous across step knots. The one exception is a knot where the input or a system parameter changed: $\ddot{q}$ jumps there, and the cached FSAL derivative MUST be recomputed.
7. **Interpolant Error Is Separate From Step Error:** The step controller does not bound mid-step error. Mid-step velocity carries $\mathcal{O}(h^5)$ truncation error, measured at 10–25× the knot error on a chaotic double pendulum. Tests set separate thresholds for knots and for mid-step samples.
8. **Variational Step Solves For Increments:** Newton's unknown is $\Delta q = q_{k+1} - q_k$, with midpoint $q_k + \Delta q / 2$ and velocity $\Delta q / h$. NEVER form $(q_{k+1} - q_k) / h$ from stored positions: it amplifies roundoff by $|q| / h$ and costs about three orders of magnitude in momentum conservation.

## Claims & Threshold Discipline
1. Every quantitative claim in `README.md` or rustdoc names the test or benchmark that reproduces it. No "guarantees" and no "machine precision" without a number and a test.
2. Test thresholds come from measurement, set at about 10× the measured value. `PLAN.md` §7 records the baselines.
3. NEVER loosen a threshold to make a test pass without recording the new measurement and the reason in `PLAN.md` §7.

## Required Verification Commands
`rust-toolchain.toml` pins stable and installs the two cross targets (`thumbv7em-none-eabihf`, `wasm32-unknown-unknown`) on first use.

Run these checks at the end of every implementation phase:
```bash
cargo test --workspace
cargo test -p lagrangian --features serde
cargo clippy --workspace --all-targets -- -D warnings
cargo check -p lagrangian --no-default-features --features libm
cargo check -p lagrangian --no-default-features --features libm,alloc
cargo check -p lagrangian --no-default-features --features libm --target thumbv7em-none-eabihf
cargo check -p lagrangian --target wasm32-unknown-unknown
```

Before pushing, run every gate CI runs: `tools/preflight.sh` (one line per gate; `--fast` for fmt, clippy and tests). The list it runs is `CONTRIBUTING.md` §1, and the `gates-listed` CI job fails if that list and `.github/workflows/ci.yml` disagree.

Unpublished workspace members that pull in a GUI or a game engine are slow to build. Use `-p lagrangian` while iterating, and the full workspace commands at the end of a phase.
