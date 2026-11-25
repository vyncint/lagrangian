# `lagrangian` — Architectural & Product Plan

> Revised 2026-11-25 after a registry and demand check and a numerical feasibility prototype.
> Figures marked *measured* come from those checks and are listed in §7. Thresholds marked *provisional* must be re-baselined against the real implementation.

## 1. Thesis & Ecosystem Positioning

`lagrangian` is a **zero-algebra equations-of-motion engine**. The user writes only $L = T - V$ in plain generic Rust; the crate returns the exact mass matrix, generalized accelerations, momenta, energy and linearization through stack-allocated forward-mode automatic differentiation.

On top of that engine it ships a streaming `Stepper`, an adaptive and a variational symplectic integrator, and a continuous, $C^2$-smooth, scrubbable `Trajectory<N>`.

**The engine is the product; the integrators are batteries included.** Everything the built-in solvers compute must also be reachable from an external ODE solver, a game loop, or a controller. That is what lets other crates depend on this one.

### Demand Evidence (crates.io API, 2026-11-25)

There is no incumbent. A crates.io search for "euler-lagrange" returns 0 crates, and GitHub has 5 Rust repositories across the searches "lagrangian mechanics", "euler-lagrange" and "variational integrator", none above 2 stars. That confirms the gap and also means demand is unproven.

| Crate | Downloads, last 90 days | Crates depending on it |
| --- | --- | --- |
| `ode_solvers` | 393k | 17 |
| `diffsol` | 50k | 7 |
| `differential-equations` | 6.8k | 4 |
| `num-dual` | 192k | 33 |
| `rapier3d` | 697k | 74 |
| `avian3d` | 216k | 70 |
| `argmin` | 1.54M | 93 |
| `egui` | 10.9M | 1,466 |
| `bevy` | 2.2M | 2,216 |
| `ranim` | 127 | 1 |

Scientific solver crates are consumed by applications, not by libraries: the leading ODE crate has 17 dependents. The crates.io keyword `robotics` has 1,020 crates and `simulation` 1,305, against 13 for `mechanics`.

### Success Metric

A realistic ceiling for this scope is **tens of dependent crates**, plus application users who never show up as reverse dependencies. Dependents can come through three channels, and the plan must serve each:

1. **Model crates** that implement `Lagrangian<N>` for their own systems. They need a small, stable trait surface (§3).
2. **Integration crates** for interactive frameworks, starting with `bevy_lagrangian`.
3. **Solver and control crates** that consume `accelerations` and `linearize` without using the built-in integrators.

### Why Narrower Alternatives Were Rejected
- **Pure "scrubbable ODE trajectory" crate:** collides directly with `ode_solvers` and `differential-equations`, which already provide DOPRI5, dense output and event root-finding.
- **Pure "physics vector primitives" (springs, pulleys, field lines on `kurbo`):** serves authors of static diagrams who already use TikZ or Typst/CeTZ, and carries no simulation intelligence.

### Ecosystem Comparison

| Tool | What it solves | Relationship to `lagrangian` |
| --- | --- | --- |
| **`rapier` / `avian`** | Rigid-body game physics with contacts. Rapier also offers reduced-coordinate multibody joints. | Different job. `lagrangian` has no contacts; it takes arbitrary generalized coordinates from one scalar function and gives exact energy, phase space and scrubbable trajectories. |
| **`ode_solvers` / `diffsol` / `differential-equations`** | General-purpose ODE solvers. | **Complement, not competitor.** They need $\ddot{q} = f(q, \dot{q}, t)$ derived by hand; `lagrangian::accelerations` supplies it. Adapter examples ship in the repository. |
| **`num-dual`** | Generic dual numbers. | Lower-level building block. AD stays internal here to keep the core dependency-free. |
| **SymPy mechanics / ModelingToolkit** | Symbolic derivation of equations of motion. | Heavy runtimes; no native or WASM-first Rust counterpart. |
| **`lagrangian`** | **Zero-algebra generalized mechanics.** | Write only $L$; get exact equations of motion, integrate with the built-in solvers or your own, scrub $C^2$ trajectories. |

### Prior Art to Cite in the README

Readers will raise these, so name them first:
- **SICM / `scmutils`** (Sussman & Wisdom) and its Clojure port **Emmy**: `Lagrange-equations` from a Lagrangian function. The closest spiritual ancestor; "SICM's `Lagrange-equations` for Rust, at native speed" is a legitimate hook.
- **Lagrangian Neural Networks** (Cranmer et al., 2020): the same $\ddot{q} = M^{-1}(\nabla_q L - \dots)$ formula obtained by autodiff in JAX.
- **GeometricIntegrators.jl**: variational integrators in Julia.
- **MuJoCo, Pinocchio, Drake**: reduced-coordinate rigid-body dynamics with $O(N)$ algorithms. `lagrangian` does not compete with them on large articulated robots (§6).

### Crate Name Availability (checked 2026-11-25)

- **`lagrangian` is free** on crates.io, PyPI and npm. So are `bevy_lagrangian`, `lagrangian-core`, `lagrangian-derive` and `lagrangian-macros`.
- **`lagrange` was taken on 2026-06-25** by an empty `0.0.0` with no repository, described as "Massively parallel runtime for computing Lagrangian systems". Names in this space are being claimed.
- Alternatives still free: `variational`, `symplectic`, `euler-lagrange`, `least-action`, `maupertuis`. Keep `lagrangian`: `symplectic` would misdescribe a crate whose default solver is DOPRI5, and `variational` is too broad.
- **The word is overloaded.** Searching crates.io for "lagrangian" returns 87 crates led by optimization, video codecs and interpolation. The crate description and keywords must say "Lagrangian mechanics".
- **Reserve by shipping.** Publish `0.0.1` as soon as Phase 2 passes verification (working engine and `Stepper`), not an empty placeholder and not as late as `v0.1.0`.

---

## 2. Nine Architectural & Mathematical Ideas

### 1. Convective Directional AD ($N + 2$ Passes)
From the Euler-Lagrange equation:
$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = Q_i$$

Expanding the total time derivative yields a linear system for generalized accelerations $\ddot{q}$:
$$\sum_{j=1}^{N} \underbrace{\frac{\partial^2 L}{\partial \dot{q}_i \partial \dot{q}_j}}_{M_{ij}(q, \dot{q}, t)} \ddot{q}_j = \frac{\partial L}{\partial q_i} + Q_i(q, \dot{q}, t) - \underbrace{\left(\sum_{j=1}^{N} \frac{\partial^2 L}{\partial \dot{q}_i \partial q_j} \dot{q}_j + \frac{\partial^2 L}{\partial \dot{q}_i \partial t}\right)}_{C_i(q, \dot{q}, t)}$$

The entire Coriolis/centrifugal + explicit-time vector $C \in \mathbb{R}^N$ is a **single directional derivative** of generalized momentum $p = \nabla_{\dot{q}} L$ along the convective tangent $(\delta q = \dot{q}, \delta \dot{q} = 0, \delta t = 1)$. Using stack-allocated `Grad<N>` and `Dual<Grad<N>>`, the equations of motion are extracted in **$N + 2$ passes** of `lagrangian`. Scalar `HyperDual` (`Dual<Dual<f64>>`) is retained as an independent reference oracle in tests.

**Cost, stated honestly:** each pass is $O(N)$ wide, so the total is $O(N^2) \cdot \text{cost}(L)$, the same order as any Hessian method. The gain over $N^2$ scalar hyperdual passes is a constant factor. Do not advertise it as an asymptotic win; the benchmark crate must quantify it. Candidate optimization to benchmark before adopting: a single-pass second-order jet that evaluates each transcendental primal once instead of $N + 2$ times.

### 2. Exact Energy & Phase Space via Legendre Transform ($E = p \cdot \dot{q} - L$)
Systems such as a **Bead on a Rotating Hoop** or a **charged particle in a magnetic field** contain $\dot{q}^0$ centrifugal terms or $\dot{q}^1$ vector-potential terms in $L$. Computing naive $\frac{1}{2}\dot{q}^T M \dot{q} + V$ produces spurious energy oscillations. The **Legendre transform (Jacobi integral)**
$$p_i = \frac{\partial L}{\partial \dot{q}_i}, \qquad E(q, \dot{q}, t) = p \cdot \dot{q} - L(q, \dot{q}, t), \qquad V_{\text{eff}}(q, t) = -L(q, 0, t), \qquad T_{\text{eff}} = E - V_{\text{eff}}$$
is the quantity the continuous dynamics conserve for every autonomous conservative Lagrangian, with zero extra traits from the user.

Using it means any measured drift is integrator error and not a bookkeeping artifact. It does not by itself make the drift small: that depends on the solver and tolerance (§7).

Relative drift is normalized by $\max(|E_0|, \max_t T_{\text{eff}})$, never by $|E_0|$ alone. A double pendulum released from horizontal has $E_0 = 0$ with the natural potential zero.

### 3. $C^2$-Smooth Quintic Hermite Segments
DOPRI5's FSAL (First Same As Last) property yields exact $(q, \dot{q}, \ddot{q})$ at both endpoints of every accepted step at zero extra cost. A 6-coefficient **quintic Hermite spline** per DOF gives:
- $\mathcal{O}(h^6)$ position accuracy
- Exact kinematic identities: $\dot{q}(t) \equiv q'(t)$ and $\ddot{q}(t) \equiv q''(t)$
- $C^0$-continuous acceleration across adaptive step boundaries (*measured* mismatch at knots: $10^{-14}$)
- Half the storage of separate polynomials for $q$ and $\dot{q}$

**Interpolation accuracy is not governed by the step controller.** Mid-step velocity carries the interpolant's own $\mathcal{O}(h^5)$ truncation error. *Measured* on a chaotic double pendulum, mid-step energy error is 10–25× the error at knots. Tests must set separate thresholds for knots and mid-step samples.

Continuity of $\ddot{q}$ holds between changes of input or system parameters. At a knot where the input changes, $\ddot{q}$ jumps, as it does physically.

### 4. True Symplectic Variational Integrator (`Solver::Variational4`)
Standard Velocity Verlet is **not** symplectic for non-separable systems like the double pendulum, where $M(q)$ depends on $q$. **Marsden-West Midpoint Discrete Euler-Lagrange** composed with **Yoshida's 4th-order symmetric triple step** preserves the symplectic form and conserves the momentum of every cyclic coordinate up to floating-point roundoff, under the same `Segment<N>` output as `Solver::Dopri5`.

**Newton must solve for the increment $\Delta q = q_{k+1} - q_k$, not for $q_{k+1}$.** Differencing stored positions amplifies roundoff by $|q|/h$. *Measured* on a Kepler orbit, angular-momentum drift is $4.5 \times 10^{-14}$ after $10^4$ steps and $2.3 \times 10^{-12}$ after $10^6$ with the increment form, against $8.5 \times 10^{-11}$ and $3.9 \times 10^{-8}$ without it. Roundoff-limited conservation grows slowly with step count; never promise a fixed $10^{-14}$.

Energy error is bounded and 4th-order: *measured* on the chaotic double pendulum, the band is $8.3 \times 10^{-4}$, $4.7 \times 10^{-5}$ and $1.2 \times 10^{-6}$ for $dt = 0.01, 0.005, 0.002$, with no growth between the first and second half of $10^4$ steps.

### 5. Zero-Algebra World Kinematics (`Taylor2`) & `velocity_of` Combinator
- **`SceneMap` + `Taylor2`:** Passing `Taylor2(q[i], dq[i], 0.5 * ddq[i])` ($c_0 + c_1 \tau + c_2 \tau^2 \pmod{\tau^3}$) through a user's position mapping yields 2D/3D Cartesian `[pos, vel, acc]` in a single call, capturing centripetal, tangential, Coriolis and moving-pivot accelerations.
- **`velocity_of`:** Uses `Dual<S>` inside `lagrangian` so users can define systems like a **Triple Pendulum** or **Kapitza's Vibrating Inverted Pendulum** purely from Cartesian position expressions without expanding $\dot{x}^2 + \dot{y}^2$ manually.

### 6. Partial-Pivoting LU Solver (Unlocking General Relativity Geodesics)
In addition to unrolled solvers for $N = 1, 2, 3$, solving $M \ddot{q} = F$ with stack-allocated **Gaussian elimination with partial pivoting ($PA = LU$)** rather than SPD-only Cholesky supports **indefinite** symmetric mass matrices. This unlocks short simulations of light bending and relativistic precession around a **Schwarzschild black hole**.

`SchwarzschildGeodesic` uses all four coordinates $(t, r, \theta, \varphi)$ with signature $(-, +, +, +)$, so it exercises the general-$N$ LU path. An equatorial $N = 3$ model would only ever hit the unrolled $3 \times 3$ path.

### 7. Event Root-Finding & Poincaré Sections
Brent's method on the quintic Hermite segments yields zero crossings (`traj.crossings(...)`), nonlinear period extraction and Poincaré sections with no solver reruns.

The roots are accurate to $10^{-13}$ **with respect to the interpolant**. Accuracy against the true solution is bounded by the integration tolerance: *measured* pendulum period error is $1.6 \times 10^{-9}$ at default tolerances and $1.5 \times 10^{-12}$ at `rtol = 1e-12`.

Detection by sign change at knots misses an even number of crossings inside one step, so each segment is also sampled at interior points.

### 8. Streaming `Stepper` and `Copy` Segments
`Stepper::step()` advances one accepted step and returns a `Segment<N>`: a `Copy`, stack-allocated quintic Hermite piece that can be sampled directly. `Trajectory<N>` is a collection of segments, and `simulate()` is a loop over `Stepper`.

This is required for interactive use (live control forces, parameter sliders, endless simulations) and it makes `alloc` optional, so the engine and stepper run on targets with no heap.

### 9. Control Inputs & Linearization
`Controlled<N, M>` adds $M$ external inputs (motor torque, thrust) held constant across a step. `linearize` returns $\partial\ddot{q}/\partial q$, $\partial\ddot{q}/\partial\dot{q}$ and $\partial\ddot{q}/\partial u$, the blocks of the state-space pair $(A, B)$ used by LQR and MPC.

Classic control benchmarks (cart-pole, acrobot, pendulum swing-up) are all Lagrangian systems, and this is the audience the `robotics` keyword count points at.

---

## 3. Public API & Stability Rules

Dependents compile against a small surface, so it must be hard to break by accident.

- **Stable in `0.x`:** `Lagrangian<N>`, `Controlled<N, M>`, `SceneMap`, `Scalar` (as a bound), the engine functions (`eom_terms`, `accelerations`, `mass_matrix`, `momentum`, `energy`, `linearize`), `Stepper`, `Segment<N>`, `State<N>`, `Trajectory<N>`, `SimConfig`, `Solver`, `SimError`.
- **`Scalar` is sealed.** Users bound on it but cannot implement it, so methods can be added without a breaking release.
- **`lagrangian::ad` is public but exempt from semver** until `1.0`. Users write `S: Scalar` and never name `Dual`, `Grad<N>` or `Taylor2`.
- **`#[non_exhaustive]`** on every public enum and configuration struct.
- **`f64` only** in the public numeric surface for `0.x`. Accuracy targets of $10^{-10}$ are meaningless in `f32`, and WASM has native `f64`. Because the AD types are semver-exempt, a float-generic core stays possible later.
- **MSRV:** edition 2024 with `rust-version = "1.85"`, raised only in minor releases.
- **License:** `MIT OR Apache-2.0`, the Rust ecosystem default. *Owner to confirm.*

---

## 4. Workspace Layout

```text
lagrangian/
├── Cargo.toml                       # Virtual workspace manifest (resolver = "3", shared lints)
├── AGENTS.md / CONTRIBUTING.md      # Agent brief and contributor document (commit rules, gate list)
├── CLAUDE.md                        # Persistent architectural & math invariants
├── PLAN.md                          # This plan
├── PROMPT.md                        # Master implementation prompt for Claude Code
├── .github/                         # CI, commit policy (DCO, no AI attribution), release automation
├── tools/                           # preflight.sh (every CI gate, locally) and the gates' self-tests
├── docs/RELEASING.md                # Release checklist
├── crates/
│   ├── lagrangian/                  # Core crate, published (#![no_std], #![forbid(unsafe_code)])
│   │   ├── Cargo.toml               # features: std (default, implies alloc), alloc, libm, serde
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── scalar.rs            # Sealed Scalar trait + f64 impl (IEEE-754 powi & hypot guards)
│   │   │   ├── ad.rs                # Dual<T>, HyperDual, Grad<N>, Taylor2 (100% Copy, stack-only)
│   │   │   ├── linalg.rs            # Unrolled N=1,2,3 + stack-allocated partial-pivoting LU
│   │   │   ├── mechanics.rs         # Lagrangian, Controlled, engine functions, linearize, SceneMap, velocity_of
│   │   │   ├── segment.rs           # Segment<N>, State<N> (Copy, no alloc)
│   │   │   ├── solver/
│   │   │   │   ├── mod.rs           # SimConfig, Solver, Stepper; simulate() behind `alloc`
│   │   │   │   ├── dopri5.rs        # Dormand-Prince 5(4) FSAL + adaptive PI step controller
│   │   │   │   └── variational.rs   # Marsden-West midpoint DEL + Yoshida 4th-order
│   │   │   ├── trajectory.rs        # `alloc`: Trajectory<N>, Cursor, crossings
│   │   │   └── systems.rs           # Built-in benchmark systems
│   │   ├── tests/                   # Physics verification suite
│   │   └── examples/
│   │       └── svg_showcase.rs      # Zero-dependency SVG figures
│   └── bevy_lagrangian/             # Bevy plugin, published separately
├── examples/
│   ├── egui_playground/             # Workspace member, publish = false (native + WASM via Trunk)
│   └── ode_adapters/                # Workspace member, publish = false (ode_solvers, diffsol)
└── benches/                         # Workspace member, publish = false (criterion)
```

Cargo does not build `examples/*.rs` placed at a virtual workspace root. A loose example must live inside a package (`crates/lagrangian/examples/`), and anything with external dependencies must be its own workspace member.

---

## 5. Launch Strategy (`v0.1.0`)

### Three Launch Pillars
1. **An engine anyone can call.** Core is `no_std`, `wasm32`-friendly and renderer-agnostic. `ode_adapters` proves it works as the right-hand side for `ode_solvers` and `diffsol`.
2. **A live WASM playground on line 3 of `README.md`.** `examples/egui_playground`, built with Trunk for GitHub Pages: scrub chaotic double pendulums, Kapitza pendulums and Schwarzschild geodesics, push a cart-pole with a live control force, and watch energy-drift plots next to the 15-line Rust definitions.
3. **`bevy_lagrangian`.** A plugin that advances a `Stepper` in `FixedUpdate` and writes `SceneMap` output to `Transform`s. Bevy has 2,216 dependent crates; this is the main channel for reverse dependencies.

`svg_showcase` stays as the zero-dependency figure generator. The `ranim` recipe is **dropped from v0.1**: `ranim` had 127 downloads in the last 90 days and 1 dependent. `eval_normalized()` stays in the API, since any animation timeline can use it.

### Launch Credibility
- **Every quantitative claim in the README names the test or benchmark that reproduces it.** No "guarantees", no "machine precision" without a number and a test.
- **Benchmarks before claims.** `benches/` compares AD-extracted accelerations against hand-derived equations for the double pendulum, and measures scaling on an $N$-link chain for $N = 2, 4, 8, 16$.
- **A "Limitations" section** in the README listing the non-goals in §6.
- **A "Prior art" section** citing the work in §1.
- **crates.io metadata:** description starting "Lagrangian mechanics engine"; keywords `physics`, `mechanics`, `simulation`, `robotics`, `autodiff`; categories `science`, `simulation`, `mathematics`, `no-std`, `wasm`. Use `include` so `PROMPT.md` and `CLAUDE.md` are not shipped in the published crate.

---

## 6. Scope: v0.1 Non-Goals & Roadmap

| Missing in v0.1 | Workaround today | Planned |
| --- | --- | --- |
| Holonomic constraints (Lagrange multipliers, closed loops) | Choose minimal coordinates | v0.2 |
| Events that change state during integration (impacts, bounces) | `Stepper::set_state` between steps; `crossings` after the fact | v0.2 |
| `Hamiltonian<N>` trait ($H(q, p)$ with Hamilton's equations) | Use the Legendre functions | v0.2 |
| Interpolation error control (defect check per step) | Tighten `rtol` | v0.2, if benchmarks justify the extra evaluation |
| Stiff or implicit solver | Feed `accelerations` to an external stiff solver | v0.3 |
| Backward-in-time integration | None | Unscheduled |
| Runtime-sized $N$ | None; $N$ is a const generic | Not planned |
| `f32` | None | Not planned for `0.x` |
| Contacts and collision detection | Use `rapier` or `avian` | Never |

**Target envelope:** small systems, roughly $N \le 16$. One extraction costs $O(N^2) \cdot \text{cost}(L)$ plus an $O(N^3)$ solve. The benchmark crate must confirm the envelope before the README states it.

---

## 7. Measured Baselines (prototype, 2026-11-25)

Prototype: IEEE doubles, hand-derived equations of motion, Dormand-Prince 5(4) with a plain integral controller, quintic Hermite coefficients exactly as specified in `PROMPT.md`. `atol = rtol / 100` throughout. Energy figures are relative drift.

| Case | Setup | Measured |
| --- | --- | --- |
| Double pendulum energy, `Dopri5` | $m = l = 1$, $g = 9.81$, $q_0 = (2.2, 2.8)$ at rest, $t \in [0, 20]$ | `rtol=1e-10`: knots $5.0 \times 10^{-10}$, mid-step $5.6 \times 10^{-9}$<br>`rtol=1e-12`: knots $2.5 \times 10^{-12}$, mid-step $5.7 \times 10^{-11}$<br>`rtol=1e-13`: knots $2.2 \times 10^{-13}$, mid-step $3.3 \times 10^{-11}$ |
| Same, $t \in [0, 100]$ | `rtol=1e-10` | knots $1.6 \times 10^{-9}$, mid-step $1.7 \times 10^{-8}$ |
| Pendulum period vs elliptic integral | $l = 1$, $\theta_0 = 2.4$, first period from upward crossings | `rtol=1e-9`: $1.6 \times 10^{-9}$<br>`rtol=1e-12`: $1.5 \times 10^{-12}$ |
| Coupled oscillator vs normal modes | $k = 1$, $k_c = 0.5$, $x_0 = (1, 0)$ at rest, $t \in [0, 20]$, max error in $q$ including mid-step | `rtol=1e-9`: $1.8 \times 10^{-9}$<br>`rtol=1e-12`: $1.7 \times 10^{-12}$ |
| Rotating hoop Jacobi integral | $R = 1$, $\Omega = 5$, $\theta_0 = 0.3$ at rest, $t \in [0, 20]$ | `rtol=1e-9`: $1.1 \times 10^{-8}$<br>`rtol=1e-12`: $1.1 \times 10^{-11}$ |
| Kepler $p_\theta$, `Variational4` | $GM = 1$, $r_0 = 1$, $\dot{\theta}_0 = 1.2$, $dt = 0.02$ | Newton on $\Delta q$: $4.5 \times 10^{-14}$ at $10^4$ steps, $2.3 \times 10^{-12}$ at $10^6$<br>Newton on $q_{k+1}$: $8.5 \times 10^{-11}$, $3.9 \times 10^{-8}$ |
| Double pendulum energy band, `Variational4` | Same initial state, $10^4$ steps | $dt = 0.01$: $8.3 \times 10^{-4}$<br>$dt = 0.005$: $4.7 \times 10^{-5}$<br>$dt = 0.002$: $1.2 \times 10^{-6}$ |
| Continuity at knots | Double pendulum, `rtol=1e-10` | $q$: 0, $\dot{q}$: $2.2 \times 10^{-16}$, $\ddot{q}$: $1.0 \times 10^{-14}$ |

What the baselines established:
- The original targets were not reachable as written. Double pendulum drift below $10^{-10}$ at `rtol=1e-10` fails by 5× at knots and 56× mid-step. Period, oscillator and hoop targets fail at default tolerances by 160×, 18× and 1,000×. The $10^{-14}$ momentum target fails by 4.5× in the best case.
- Mid-step energy error is the interpolant's truncation error. Building the Hermite coefficients from the integrator's increments instead of differenced positions made no difference (tested), so that is not a fix.
- At `rtol=1e-13` the mid-step figure stops improving near $3 \times 10^{-11}$. The cause was not identified.
- Not prototyped: Schwarzschild, Kapitza, cart-pole linearization. Their thresholds in `PROMPT.md` are uncalibrated.

**Rule:** once the real implementation exists, replace every provisional threshold with about 10× the measured value and update this table. Never loosen a threshold without recording the measurement and the reason here.

---

## 8. Execution Phases

- **Phase 1:** Workspace scaffold, sealed `Scalar`, AD (`Dual`, `Grad<N>`, `Taylor2`), `linalg`, AD oracle and LU tests.
- **Phase 2:** Engine (`Lagrangian<N>`, `eom_terms`, `accelerations`, Legendre energy and momentum, `SceneMap`, `velocity_of`), `Segment<N>`, `Stepper`, `Dopri5`, `Trajectory<N>` (`Cursor`, `crossings`). **Milestone: owner publishes `0.0.1` to reserve the name.**
- **Phase 3:** `Variational2` / `Variational4`, `Controlled<N, M>` and `linearize`, built-in systems, full physics verification suite, thresholds re-baselined into §7.
- **Phase 4:** `benches`, `svg_showcase`, `ode_adapters`, `egui_playground` (native + WASM), flagship `README.md`.
- **Phase 5:** `bevy_lagrangian`, then `v0.1.0`.
