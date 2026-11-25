# Master Prompt for Claude Code: Implementing `lagrangian`

You are a Principal Numerical Physicist and Rust Library Architect. Design and implement **`lagrangian`** (`v0.1.0`), a zero-algebra equations-of-motion engine in Rust. It takes ONLY a user's `L = T - V` (Lagrangian) written in plain generic Rust arithmetic and returns exact accelerations, momenta, energy and linearization. On top of that engine it provides a streaming `Stepper` and a continuous, $C^2$-smooth, scrubbable `Trajectory<N>` ready for interactive simulations, WASM playgrounds, scientific SVG figures, game engines and controllers.

Read `PLAN.md` first for positioning, scope and the measured baselines behind every numeric threshold below. Follow `CLAUDE.md` at all times.

## Core Philosophy & Non-Negotiable Constraints

1. **Zero-Algebra UX:** Users never derive Euler-Lagrange equations, Jacobians, Coriolis terms, or mass matrices by hand. They implement one trait (`Lagrangian<const N: usize>`) with one required method (`lagrangian<S: Scalar>`).
2. **Engine First:** The equations-of-motion functions are public API, usable with no built-in solver. Built-in solvers call only those public functions. `simulate()` is a loop over `Stepper`, never a second code path.
3. **`#![no_std]` Core, `alloc` Optional, Renderer-Agnostic:**
   - Features of `crates/lagrangian`: `std` (default, implies `alloc`), `alloc`, `libm`, `serde`.
   - Without `alloc`: engine functions, `Stepper`, `Segment<N>`, all solvers, built-in systems.
   - With `alloc`: `Trajectory<N>`, `simulate()`, `crossings()`.
   - Enabling neither `std` nor `libm` is a `compile_error!` with a helpful message. With both enabled, prefer `std`.
   - Runtime dependencies: **ZERO** with the default `std` feature; ONLY `libm` with `default-features = false`; optional `serde`. No `nalgebra`, no `ndarray`, no graphics crates in the core.
   - serde does not implement `Serialize` / `Deserialize` for `[T; N]` with const-generic `N` (verified against serde 1.0.229). Write the impls for `Segment<N>`, `State<N>` and `Trajectory<N>` by hand. Do not add a helper crate.
   - Compile time of `crates/lagrangian` from scratch must stay under ~1 second.
4. **Zero Heap Allocation in the Inner Integration Loop:** All automatic differentiation types, generalized vectors `[f64; N]`, mass matrices `[[f64; N]; N]` and `Segment<N>` must be `Copy` and stack-allocated using const generics. Heap (`alloc::vec::Vec`) is used ONLY to store the segments of `Trajectory<N>` and event results.
5. **Toolchain & Safety:** Check `rustc --version` first. Use Rust Edition 2024 and declare `rust-version = "1.85"`. Enforce `#![forbid(unsafe_code)]` in `crates/lagrangian`, and ensure `cargo clippy --workspace --all-targets -- -D warnings` passes cleanly. Stable Rust only: no `generic_const_exprs`, so never write `[f64; 2 * N]`.
6. **API Stability:** Seal `Scalar` (private supertrait) so methods can be added later. Mark every public enum and configuration struct `#[non_exhaustive]`. The `ad` module is public but documented as exempt from semver until `1.0`. The public numeric surface is `f64` only.
7. **Claims Discipline:** Every quantitative claim in `README.md` or rustdoc names the test or benchmark that reproduces it. Thresholds marked *provisional* below must be re-measured on the real implementation, set to about 10× the measured value, and recorded in `PLAN.md` §7.

---

## Mathematical Specification (Read Carefully Before Coding)

### 1. Automatic Extraction of Equations of Motion (Euler-Lagrange)
For a system with $N$ generalized coordinates $q \in \mathbb{R}^N$, velocities $\dot{q} \in \mathbb{R}^N$, time $t \in \mathbb{R}$, and optional non-conservative generalized forces $Q(q, \dot{q}, t) \in \mathbb{R}^N$:

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = Q_i$$

Expanding the total time derivative yields the linear system for generalized accelerations $\ddot{q} \in \mathbb{R}^N$:

$$\sum_{j=1}^{N} \underbrace{\frac{\partial^2 L}{\partial \dot{q}_i \partial \dot{q}_j}}_{M_{ij}(q, \dot{q}, t)} \ddot{q}_j = \frac{\partial L}{\partial q_i} + Q_i(q, \dot{q}, t) - \underbrace{\left(\sum_{j=1}^{N} \frac{\partial^2 L}{\partial \dot{q}_i \partial q_j} \dot{q}_j + \frac{\partial^2 L}{\partial \dot{q}_i \partial t}\right)}_{C_i(q, \dot{q}, t)}$$

**Convective Directional AD:**
The entire Coriolis/centrifugal + explicit-time vector $C \in \mathbb{R}^N$ on the RHS is a **single directional derivative** of the generalized momentum $p(q, \dot{q}, t) = \nabla_{\dot{q}} L$ along the convective tangent $(\delta q = \dot{q}, \delta \dot{q} = 0, \delta t = 1)$.
- Implement stack-allocated forward-mode AD types implementing `Scalar`:
  1. `Dual<T>` (`val: T, eps: T`) — generic nested dual number (`Dual<f64>`, `Dual<Dual<f64>>` = `HyperDual`, `Dual<Grad<N>>`, and deeper nestings for linearization).
  2. `Grad<const N: usize>` (`val: f64, d: [f64; N]`) — 1st-order multivariate gradient jet.
  3. `Taylor2` (`c0: f64, c1: f64, c2: f64`) — univariate truncated Taylor polynomial $c_0 + c_1 \tau + c_2 \tau^2 \pmod{\tau^3}$ for fast forward kinematics.
- Using `Grad<N>` and `Dual<Grad<N>>`, extract the equations of motion in **$N + 2$ passes** of `lagrangian`:
  - **1 pass** of `Grad<N>` seeded on $q$: yields $L$ and $\nabla_q L \in \mathbb{R}^N$.
  - **1 pass** of `Dual<Grad<N>>` with inner `Grad<N>` seeded on $\dot{q}$ ($e_1 \dots e_N$) and outer `Dual` ($\epsilon$) seeded along $(q' = \dot{q}, \dot{q}' = 0, t' = 1)$: yields generalized momentum $p = \nabla_{\dot{q}} L \in \mathbb{R}^N$, $\partial_t L$ (via `eps.val - dot(grad_q_L, dq)`), AND the entire convective vector $C \in \mathbb{R}^N$ in **one pass**.
  - **$N$ passes** of `Dual<Grad<N>>` (with inner `Grad<N>` on $\dot{q}$ and outer `Dual` on $\dot{q}_j$ for $j = 0..N$; symmetrize $M_{ij} = \frac{1}{2}(M_{ij} + M_{ji})$ to eliminate floating-point asymmetry): yields the exact Hessian mass matrix $M \in \mathbb{R}^{N \times N}$.
- Each pass is $O(N)$ wide, so total cost is $O(N^2) \cdot \text{cost}(L)$. This is a constant-factor gain over $N^2$ scalar hyperdual passes, not an asymptotic one. Describe it that way everywhere.
- Keep scalar `HyperDual` (`Dual<Dual<f64>>`) as an independent reference oracle in unit tests to cross-check `M`, `C`, `p` and $\nabla_q L$ on random inputs to a relative tolerance of `1e-12`. Do not require bit-for-bit equality: symmetrization changes the last bits.

**Public engine API (`mechanics.rs`, no `alloc`):**
```rust
pub struct EomTerms<const N: usize> {
    pub lagrangian: f64,            // L
    pub mass: [[f64; N]; N],        // M, symmetrized
    pub grad_q: [f64; N],           // ∇_q L
    pub momentum: [f64; N],         // p = ∇_q̇ L
    pub convective: [f64; N],       // C
    pub force: [f64; N],            // Q
    pub dl_dt: f64,                 // ∂L/∂t
}

pub fn eom_terms<L: Lagrangian<N>, const N: usize>(sys: &L, q: [f64; N], dq: [f64; N], t: f64) -> EomTerms<N>;
pub fn accelerations<L: Lagrangian<N>, const N: usize>(sys: &L, q: [f64; N], dq: [f64; N], t: f64) -> Result<[f64; N], SimError>;
pub fn mass_matrix(..) -> [[f64; N]; N];
pub fn momentum(..) -> [f64; N];
pub fn energy(..) -> Energy;                       // { total, kinetic, potential }
pub fn velocity_from_momentum(sys, q, p, t, dq_guess) -> Result<[f64; N], SimError>;   // inverse Legendre transform
```
Exact names may be adjusted, but every capability must exist as a free function that takes plain `f64` arrays. Provide `impl<L: Lagrangian<N>, const N: usize> Lagrangian<N> for &L` so APIs can take systems by value or by reference, and the matching blanket impl of `Controlled<N, M>` for `&L` (without it, `WithInput { sys: &sys, .. }` does not compile).

`SimError` is `Copy`, `#[non_exhaustive]`, implements `core::fmt::Display` and `core::error::Error`, and carries the time of failure: `SingularMassMatrix { t }`, `StepSizeUnderflow { t }`, `MaxStepsExceeded { t }`, `NewtonDidNotConverge { t }`, `NonFiniteState { t }`.

### 2. IEEE-754 & AD Edge-Case Guards (CRITICAL — Do Not Skip!)
- **`powi(n: i32)` MUST use binary exponentiation by squaring** using `*` and `.recip()` on `Self`. NEVER implement `powi` on AD types using `n * (n - 1) * x.powi(n - 2)` or `exp(n * ln(x))`. Why: mechanical systems start from rest (`dq = 0.0`), and for `n = 1, x = 0.0`, `0.0 * 0.0_f64.powi(-1)` evaluates `0.0 * inf = NaN` in IEEE-754, while `ln(x)` produces `NaN` for negative velocities.
- Guard `hypot(x, y)` when both `x == 0` and `y == 0` so derivatives are `0.0` rather than `0.0 / 0.0 = NaN`. Apply the same convention to `atan2(0, 0)` and to `abs` at `0`.
- Do NOT mask `sqrt` at `0`: its derivative is infinite by mathematics. Document in rustdoc that norms must be written with `hypot`, and that speeds at rest should be kept squared.
- Implement arithmetic between `S` and `f64` (`Add<f64>`, `Sub<f64>`, `Mul<f64>`, `Div<f64>`, plus `From<f64>` and `Scalar::from_f64`) AND implement `Add<AdType> for f64`, `Mul<AdType> for f64`, etc. where possible; inside generic `<S: Scalar>`, supertraits allow `s * 0.5_f64` directly (document clearly in rustdoc & examples to write scalar constants on the RHS like `v_sq * (0.5 * self.m)` or `S::from(0.5) * ...`).
- `Scalar` also provides `ZERO`, `ONE`, `value() -> f64` and `PartialOrd` by value, so user code can branch on the primal.

### 3. Linear System Solver $M \ddot{q} = \text{RHS}$ (`linalg.rs`)
- Provide fast unrolled analytical solvers for `N = 1` (scalar division), `N = 2` (2x2 determinant/pivoted), and `N = 3` (3x3 direct/pivoted).
- For general `N` (and fallback), implement **stack-allocated Gaussian elimination with partial pivoting (`PA = LU`)** — NOT positive-definite-only Cholesky!
  - *Why:* Standard mechanical systems have SPD $M$, while relativistic geodesic Lagrangians (e.g., Schwarzschild metric with signature `(-, +, +, +)`) have **indefinite** symmetric $M$. Partial-pivoting LU handles both. Return a typed `SimError::SingularMassMatrix { t }` if a pivot magnitude drops below tolerance.
- Expose the factorization so one LU can solve several right-hand sides (needed by the variational Newton iteration and by `linearize`).

### 4. Exact Energy & Phase Space via Legendre Transform (`mechanics.rs`)
Users only provide `lagrangian(q, dq, t)`. Do NOT assume $L$ is strictly $\frac{1}{2}\dot{q}^T M \dot{q} - V(q)$ (rotating hoops have $\dot{q}^0$ centrifugal terms in $T$; magnetic fields have $\dot{q}^1$ vector-potential terms $A(q)\cdot\dot{q}$).
- **Generalized Momentum:** $p_i(q, \dot{q}, t) = \frac{\partial L}{\partial \dot{q}_i}$ (1 pass of `Grad<N>`).
- **Total Energy (Jacobi Integral / Hamiltonian):**
  $$E(q, \dot{q}, t) = p \cdot \dot{q} - L(q, \dot{q}, t)$$
  By Noether's theorem, $\frac{dE}{dt} = Q \cdot \dot{q} - \frac{\partial L}{\partial t}$, so $E$ is conserved by the continuous dynamics of EVERY autonomous conservative Lagrangian (including rotating frames and magnetic fields). Measured drift is therefore integrator error, and how small it is depends on the solver and tolerance.
- **Effective Potential & Kinetic Split:**
  $$V_{\text{eff}}(q, t) = -L(q, [0; N], t), \qquad T_{\text{eff}}(q, \dot{q}, t) = E(q, \dot{q}, t) - V_{\text{eff}}(q, t)$$
  Verify in tests that for natural systems $L = T_2(q, \dot{q}) - V(q)$, $T_{\text{eff}} == T_2$ and $V_{\text{eff}} == V$ to `1e-12` relative.
- **Relative drift** is `max_abs_drift / scale` with `scale = max(|E_0|, max_t T_eff)`. Never divide by `|E_0|` alone: a double pendulum released from horizontal has $E_0 = 0$.

### 5. Streaming `Stepper` and Two Integrator Families (`solver/`)

**`Stepper` (no `alloc`), shared by every solver:**
```rust
pub struct Stepper<L, const N: usize> { /* system, config, current knot (t, q, dq, ddq), controller state */ }

impl<L: Lagrangian<N>, const N: usize> Stepper<L, N> {
    pub fn new(sys: L, t0: f64, q0: [f64; N], dq0: [f64; N], config: SimConfig) -> Result<Self, SimError>;
    pub fn step(&mut self) -> Result<Segment<N>, SimError>;                  // one accepted step
    pub fn step_bounded(&mut self, t_max: f64) -> Result<Segment<N>, SimError>;   // never steps past t_max
    pub fn state(&self) -> State<N>;
    pub fn system(&self) -> &L;
    pub fn system_mut(&mut self) -> &mut L;                                  // marks the cached derivative stale
    pub fn set_state(&mut self, q: [f64; N], dq: [f64; N]) -> Result<(), SimError>;   // e.g. impact reset
}
```
- `step()` returns the `Segment<N>` covering exactly one accepted step. `Dopri5` retries rejected steps internally.
- `system_mut()` and `set_state()` invalidate the cached FSAL derivative; it is recomputed before the next step. $\ddot{q}$ may jump at that knot.
- `SimConfig` is `#[non_exhaustive]` with `Default` and builder-style setters.
- Integration is forward in time only in `v0.1`.
- With `alloc`: `simulate(&sys, (t0, t1), q0, dq0, &config) -> Result<Trajectory<N>, SimError>` loops `step_bounded(t1)`. `Trajectory::extend(&mut stepper, t_until)` grows an existing trajectory, for interactive apps that simulate ahead in chunks.

1. **`Solver::Dopri5` (Default — Adaptive Dormand-Prince 5(4) with FSAL):**
   - Classic 7-stage Dormand-Prince tableau with **FSAL** (stage 7 of an accepted step becomes stage 1 of the next step, giving exact $\ddot{q}_{k+1}$ at zero extra cost).
   - Adaptive PI step-size controller (`rtol = 1e-9`, `atol = 1e-11` defaults, configurable `min_step`, `max_step`, `max_steps`).
2. **`Solver::Variational4 { dt }` (True Symplectic Variational Integrator for Non-Separable Lagrangians):**
   - Implement the **Marsden-West Midpoint Discrete Euler-Lagrange** step $(q_k, p_k) \mapsto (q_{k+1}, p_{k+1})$ using:
     $$L_d(q_k, q_{k+1}, t_k, h) = h \, L\left(\frac{q_k + q_{k+1}}{2}, \frac{q_{k+1} - q_k}{h}, t_k + \frac{h}{2}\right)$$
     solving $p_k + D_1 L_d(q_k, q_{k+1}) + \frac{h}{2} Q_{\text{mid}} = 0$ via Newton-Raphson (computing the exact $N \times N$ Jacobian via our AD), then updating $p_{k+1} = D_2 L_d(q_k, q_{k+1}) + \frac{h}{2} Q_{\text{mid}}$.
   - **Newton's unknown is the increment $\Delta q = q_{k+1} - q_k$.** Evaluate $L$ at $q_k + \Delta q / 2$ and $\Delta q / h$. NEVER compute $(q_{k+1} - q_k)/h$ from stored positions: the cancellation amplifies roundoff by $|q|/h$ (measured: $8.5 \times 10^{-11}$ momentum drift instead of $4.5 \times 10^{-14}$ after $10^4$ steps).
   - Compose 3 symmetric substeps with **Yoshida's 4th-order coefficients**:
     $$w_1 = \frac{1}{2 - 2^{1/3}}, \qquad w_0 = 1 - 2 w_1$$
     (`step(w1 * dt)` -> `step(w0 * dt)` -> `step(w1 * dt)`). $w_0$ is negative, so the middle substep runs backward: advance $t$ by each substep's own size so time-dependent Lagrangians stay correct.
   - Recover $(\dot{q}_{k+1}, \ddot{q}_{k+1})$ at each recorded knot from $p_{k+1} = \nabla_{\dot{q}} L(q_{k+1}, \dot{q}_{k+1}, t_{k+1})$ via Newton iteration with Jacobian $M$, then `accelerations`.
   - Also offer `Solver::Variational2 { dt }` (single midpoint step) for fast 2nd-order symplectic stepping.

### 6. `Segment<N>` and Continuous $C^2$-Smooth Scrubbable `Trajectory<N>`
At every accepted step $[t_k, t_{k+1}]$ ($h = t_{k+1} - t_k$), we have exact $(q_k, \dot{q}_k, \ddot{q}_k)$ and $(q_{k+1}, \dot{q}_{k+1}, \ddot{q}_{k+1})$.

**`Segment<N>` (`segment.rs`, `Copy`, no `alloc`)** stores `t0`, `h` and a **quintic Hermite spline** on $q(\tau)$ ($\tau = (t - t_k)/h \in [0, 1]$) with 6 coefficients `[f64; 6]` per DOF:
- $c_0 = q_k$, $c_1 = h \dot{q}_k$, $c_2 = \frac{1}{2} h^2 \ddot{q}_k$
- Let $\Delta = q_{k+1} - c_0 - c_1 - c_2$, $\Delta v = h \dot{q}_{k+1} - c_1 - 2 c_2$, $\Delta a = h^2 \ddot{q}_{k+1} - 2 c_2$:
  - $c_3 = 10 \Delta - 4 \Delta v + \frac{1}{2} \Delta a$
  - $c_4 = -15 \Delta + 7 \Delta v - \Delta a$
  - $c_5 = 6 \Delta - 3 \Delta v + \frac{1}{2} \Delta a$
- Benefits: $\mathcal{O}(h^6)$ position accuracy, $C^2$-smooth position, $C^1$-smooth velocity $\dot{q}(t) \equiv q'(t)$, and **$C^0$-continuous acceleration $\ddot{q}(t) \equiv q''(t)$** across step boundaries.
- `Segment::sample(t) -> State<N>` (`{ t, q, dq, ddq }`) and `Segment::eval(tau)`, so heap-free consumers can interpolate inside the latest step.
- **Know the limit:** the step controller does not bound mid-step error. Mid-step velocity carries $\mathcal{O}(h^5)$ truncation error; on a chaotic double pendulum the mid-step energy error measured 10–25× the knot error. Say so in rustdoc.

**`Trajectory<N>` (`trajectory.rs`, behind `alloc`)** is a `Vec<Segment<N>>` with a scrubbing and analysis API:
  - `duration()`, `t_start()`, `t_end()`, `len_segments()`, `segments() -> &[Segment<N>]`
  - `sample(t) -> State<N>` via binary search (`O(log K)`), clamping outside `[t_start, t_end]` (plus `try_sample(t) -> Option<State<N>>`).
  - `Cursor` + `sample_with_cursor(t, &mut cursor)` for `O(1)` sequential forward/reverse playback.
  - Animation timeline helpers: `eval_normalized(alpha: f64)` ($\alpha \in [0, 1]$), `sample_looped(t)`, `sample_ping_pong(t)`.
  - Physics queries: `momentum(&sys, t)`, `phase_point(&sys, t) -> ([f64; N], [f64; N])`, `energy(&sys, t) -> Energy { total, kinetic, potential }`, `energy_drift(&sys, num_samples) -> EnergyReport { initial, min, max, scale, max_abs_drift, max_rel_drift, rms_drift }`, `action(&sys) -> f64` (Hamilton's principal action $S = \int L\,dt$ via 5-point Gauss-Legendre quadrature per segment).
  - **Event & Poincaré Section Finder:** `crossings(indicator: impl Fn(&State<N>) -> f64, direction: CrossingDirection) -> Vec<State<N>>` using sign-change detection + Brent's root-finding method to $10^{-13}$ on the quintic Hermite polynomial.
    - Detect sign changes at knots AND at a configurable number of interior sample points per segment (default 4). Knot-only detection misses an even number of crossings inside one step.
    - Document that $10^{-13}$ is accuracy with respect to the interpolant. Accuracy against the true solution is bounded by the integration tolerance.

### 7. Zero-Algebra Kinematics & Velocity Combinators (`mechanics.rs`)
1. **World-Space Kinematics (`SceneMap`):**
   ```rust
   pub trait SceneMap<const N: usize, const D: usize, const P: usize = 1> {
       fn points<S: Scalar>(&self, q: [S; N], t: S) -> [[S; D]; P];
   }
   ```
   By evaluating `points` once with `Taylor2` (`q_i = Taylor2::new(q[i], dq[i], 0.5 * ddq[i])`, `t = Taylor2::new(t, 1.0, 0.0)`), provide `kinematics(&sys, &state) -> [Kinematics<D>; P]` (no `alloc`) and `traj.kinematics(&sys, t)`, where `Kinematics<D> { pos: [f64; D], vel: [f64; D], acc: [f64; D] }` returns exact Cartesian positions, velocities, and accelerations with ZERO manual differentiation.
2. **Cartesian Velocity Helper inside `lagrangian` (`velocity_of`):**
   Provide `pub fn velocity_of<S: Scalar, const N: usize, const D: usize, F>(f: F, q: [S; N], dq: [S; N], t: S) -> ([S; D], [S; D])` (using `Dual<S>`) that takes a position closure `|q, t| [x, y]` and returns `(pos, vel)`. This lets users write a **Triple Pendulum** or **Kapitza's Vibrating Pendulum** using ONLY Cartesian coordinates without expanding $\dot{x}^2 + \dot{y}^2$ by hand.

### 8. Control Inputs & Linearization (`mechanics.rs`)
```rust
/// A system with M external inputs (motor torques, thrust, ...).
pub trait Controlled<const N: usize, const M: usize>: Lagrangian<N> {
    fn input_force<S: Scalar>(&self, q: [S; N], dq: [S; N], u: [S; M], t: S) -> [S; N];
}

/// Adapter: a controlled system with its input held constant is a plain `Lagrangian<N>`.
pub struct WithInput<L, const M: usize> { pub sys: L, pub u: [f64; M] }

pub struct Linearization<const N: usize, const M: usize> {
    pub a_q: [[f64; N]; N],     // ∂q̈_i / ∂q_j
    pub a_dq: [[f64; N]; N],    // ∂q̈_i / ∂q̇_j
    pub b: [[f64; M]; N],       // ∂q̈_i / ∂u_k
}
pub fn linearize<L: Controlled<N, M>, const N: usize, const M: usize>(
    sys: &L, q: [f64; N], dq: [f64; N], u: [f64; M], t: f64,
) -> Result<Linearization<N, M>, SimError>;
```
- These shapes (sealed `Scalar`, `Controlled`, `WithInput`, `Linearization`, `Stepper<L, N>` over owned and borrowed systems, and the README example) were compile-checked together on stable Rust with `#![no_std]` and clippy `-D warnings`.
- `WithInput` implements `Lagrangian<N>` with `generalized_force = sys.generalized_force + sys.input_force(.., u, ..)`. Every solver then works on controlled systems unchanged. Inputs are held constant across a step (zero-order hold): change them between steps through `stepper.system_mut().u`, which invalidates the FSAL derivative.
- With state $x = (\delta q, \delta\dot{q})$ the linear model is $\dot{x} = A x + B\,\delta u$ with $A = \begin{bmatrix} 0 & I \\ a_q & a_{\dot{q}} \end{bmatrix}$, $B = \begin{bmatrix} 0 \\ b \end{bmatrix}$. Return the blocks; stable Rust cannot express the dense $2N \times 2N$ type. With `alloc`, offer `to_dense()` returning row-major `Vec<f64>`s.
- Compute by differentiating $M \ddot{q} = \text{rhs}$: for each direction $d$ in $(q, \dot{q})$ space, $\delta\ddot{q} = M^{-1}(\delta\,\text{rhs} - \delta M\,\ddot{q})$, with $\delta M$ and $\delta\,\text{rhs}$ obtained by wrapping the extraction passes in one more `Dual` level and reusing the LU factors. For inputs, $\partial\ddot{q}/\partial u_k = M^{-1}\,\partial Q_{\text{in}}/\partial u_k$.
- Also provide the same for plain `Lagrangian<N>` systems (no `b` block).

---

## Public API Ergonomics Target (Must Compile Cleanly!)

Ensure the flagship `README.md` example compiles cleanly as shown:

```rust
use lagrangian::prelude::*;

struct DoublePendulum {
    m1: f64, m2: f64,
    l1: f64, l2: f64,
    g: f64,
}

impl Lagrangian<2> for DoublePendulum {
    fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], _t: S) -> S {
        let [th1, th2] = q;
        let [w1, w2] = dq;
        let v1_sq = (w1 * self.l1).powi(2);
        let v2_sq = v1_sq
            + (w2 * self.l2).powi(2)
            + (w1 * w2 * (th1 - th2).cos()) * (2.0 * self.l1 * self.l2);

        let kinetic = v1_sq * (0.5 * self.m1) + v2_sq * (0.5 * self.m2);
        let y1 = -th1.cos() * self.l1;
        let y2 = y1 - th2.cos() * self.l2;
        let potential = y1 * (self.m1 * self.g) + y2 * (self.m2 * self.g);

        kinetic - potential
    }
}
```

With optional non-conservative generalized forces on `Lagrangian<N>`:
```rust
fn generalized_force<S: Scalar>(&self, _q: [S; N], _dq: [S; N], _t: S) -> [S; N] {
    [S::ZERO; N]
}
```

The README must also show, in a few lines each: batch use (`simulate` then `sample`), streaming use (`Stepper::step` in a loop), and engine-only use (`accelerations` inside another crate's ODE solver).

---

## Workspace Structure & Showcase Recipes to Build

Create a Cargo **virtual** workspace at the repository root (`resolver = "3"`). Cargo does not build `examples/*.rs` at a virtual workspace root, so loose examples live inside a package and anything with external dependencies is its own member.

1. **`crates/lagrangian`**: The `#![no_std]` core library + unit tests + `tests/` physics verification suite + `systems` module containing ready-to-use benchmark systems:
   - `DoublePendulum` (with `SceneMap<2, 2, 2>`)
   - `BeadOnRotatingHoop` (demonstrates spontaneous symmetry breaking & Legendre Jacobi energy conservation)
   - `KapitzaPendulum` (inverted pendulum stabilized by high-frequency vertical pivot vibration using `velocity_of`)
   - `CartPole` (implements `Controlled<2, 1>` with the cart force as input; optional damping)
   - `KeplerOrbit` (planar central force in polar coordinates $(r, \theta)$; $\theta$ is cyclic, so $p_\theta$ tests discrete Noether conservation)
   - `SchwarzschildGeodesic` (`Lagrangian<4>` in $(t, r, \theta, \varphi)$, signature $(-, +, +, +)$; exercises the general-$N$ LU with an indefinite mass matrix)
   - Package metadata: description beginning "Lagrangian mechanics engine"; keywords `physics`, `mechanics`, `simulation`, `robotics`, `autodiff`; categories `science`, `simulation`, `mathematics`, `no-std`, `wasm`; `license = "MIT OR Apache-2.0"` (owner to confirm); an `include` list so `PROMPT.md` and `CLAUDE.md` are not shipped.
2. **`crates/lagrangian/examples/svg_showcase.rs`** (zero external dependencies):
   - Simulates `DoublePendulum`, `BeadOnRotatingHoop`, and `SchwarzschildGeodesic`, and outputs crisp, publication-ready dark/light **SVG** diagrams (multi-panel: physical trajectory + velocity/acceleration arrows + phase portrait + Poincaré section + `Dopri5` vs `Variational4` long-term energy conservation comparison). Run it to verify it generates valid SVGs.
3. **`examples/egui_playground`** (workspace member, `publish = false`, so `crates/lagrangian` stays dependency-free):
   - Built with `eframe` / `egui` + `egui_plot`, configured to compile BOTH natively (`cargo run -p egui_playground --release`) AND to `wasm32-unknown-unknown` (include `index.html` and `Trunk.toml` ready for 1-command GitHub Pages deployment).
   - Features: Preset selector (Double Pendulum, Rotating Hoop, Kapitza Inverted Pendulum, Cart-Pole, Kepler Orbit, Black Hole Geodesic), timeline **Scrubber** slider (drag forward/backward, slow-mo `0.1x`..`2x`, reverse playback), live toggles for glowing comet trails, green velocity vectors & red acceleration vectors (via `SceneMap` + `Taylor2`), live Energy Drift plot (`Dopri5` vs `Variational4`), live Phase Space `(q, p)` & Poincaré section, and an inline code snippet panel showing the 15-line `Lagrangian` definition.
   - A **live mode** driven by `Stepper`: the cart-pole takes a force from the keyboard or a slider, proving streaming and control inputs work.
4. **`examples/ode_adapters`** (workspace member, `publish = false`):
   - Uses `lagrangian::accelerations` as the right-hand side for `ode_solvers`, and for `diffsol` if its build is light enough. Check the current API of each on crates.io before coding. Each adapter integrates the double pendulum and asserts agreement with the built-in `Dopri5` to a stated tolerance.
5. **`benches`** (workspace member, `publish = false`, criterion):
   - AD-extracted `accelerations` vs hand-derived equations for the double pendulum.
   - Scaling of `eom_terms` and `accelerations` on an $N$-link planar chain for $N = 2, 4, 8, 16$.
   - Steps per second for `Dopri5` and `Variational4`. README performance statements quote these numbers only.
6. **`crates/bevy_lagrangian`** (published separately):
   - A plugin that owns a `Stepper` per entity, advances it in `FixedUpdate` to the fixed clock, exposes the current `State<N>`, and writes `SceneMap` output to `Transform`s.
   - Examples: a double pendulum, and a cart-pole pushed from the keyboard through `Controlled`.
   - Check the current Bevy version and API on crates.io before coding. Pin one Bevy minor version per plugin release and publish a compatibility table in its README.
7. **Flagship `README.md`**:
   - Lines 1–3: Sharp 1-sentence hook + badges + prominent link placeholder for the **Live WASM Playground**.
   - The 15-line `DoublePendulum` code block front and center.
   - Comparison table from `PLAN.md` §1. ODE solver crates are presented as complements. State that Rapier has reduced-coordinate multibody joints and that this crate has no contacts.
   - Mathematical intuition (Euler-Lagrange chain rule + Convective AD + Discrete Variational Symplectic integration).
   - Quickstart recipes for batch, streaming, engine-only, `egui`, Bevy and SVG use.
   - **Verified claims table:** each numeric claim next to the name of the test or benchmark that reproduces it.
   - **Limitations** (from `PLAN.md` §6) and **Prior art** (SICM / `scmutils`, Emmy, Lagrangian Neural Networks, GeometricIntegrators.jl).

The `ranim` recipe from the earlier plan is dropped from `v0.1` (see `PLAN.md` §5).

---

## Rigorous Physics Verification Suite (Must All Pass in `cargo test`)

Thresholds are *provisional*: each is about 10–30× a prototype measurement recorded in `PLAN.md` §7, except where marked *uncalibrated*. After implementing, measure, reset each to about 10× the measured value, and update that table. Never loosen a threshold silently.

"Tight" below means `rtol = 1e-12, atol = 1e-14`. "Default" means `rtol = 1e-9, atol = 1e-11`.

1. **AD Oracle Test:** Cross-check `Dual<Grad<N>>` extraction (`M`, `C`, `p`, `∇_q L`) against scalar `HyperDual` element-by-element across all built-in systems, to `1e-12` relative, including non-zero initial velocities AND `dq = [0.0; N]` rest states to prove zero `NaN`s.
2. **Linear Solver Test:** On random symmetric indefinite matrices, the unrolled `N = 1, 2, 3` paths agree with the general LU to `1e-12` relative. The general LU solves `N = 4, 6, 10` with relative residual below `1e-12` for well-conditioned inputs, and returns `SingularMassMatrix` for a singular one.
3. **Exact Analytical Trajectory Match:**
   - Coupled 2-DOF harmonic oscillator ($k = 1$, $k_c = 0.5$, $x_0 = (1, 0)$ at rest) against the normal-mode closed form over $t \in [0, 20]$, max error in $q$ including mid-step samples: `< 5e-8` at default tolerances, `< 5e-11` tight.
   - Large-angle pendulum ($l = 1$, $g = 9.81$, $\theta_0 = 2.4\text{ rad}$): the first period measured by `traj.crossings(...)` matches $T = 4\sqrt{l/g}\,K(\sin(\theta_0 / 2))$ (AGM algorithm in 6 lines) to `< 5e-8` relative at default tolerances and `< 5e-11` tight.
4. **Noether / Legendre Energy Conservation:**
   - `DoublePendulum` ($m = l = 1$, $g = 9.81$, $q_0 = (2.2, 2.8)$ at rest, $t \in [0, 20]$), `Dopri5` tight: relative drift `< 5e-11` at knots and `< 1e-9` at mid-step samples.
   - `BeadOnRotatingHoop` ($R = 1$, $\Omega = 5$, $\theta_0 = 0.3$ at rest, $t \in [0, 20]$), `Dopri5` tight: Jacobi integral $E = p_\theta \dot{\theta} - L$ drifts `< 3e-10`. Also verify the bifurcation equilibrium $\cos\theta^* = g / (R \Omega^2)$.
   - **Symplectic energy band:** the same double pendulum with `Solver::Variational4`, $10^4$ steps: at $dt = 0.005$ the energy band is `< 5e-4`; the maximum over the second half is at most `1.5×` the maximum over the first half (no secular drift); and the band at $dt = 0.005$ is less than `1/8` of the band at $dt = 0.01$ (4th order).
   - **Discrete Noether:** `KeplerOrbit` ($GM = 1$, $r_0 = 1$, $\dot{r}_0 = 0$, $\dot{\theta}_0 = 1.2$), `Variational4` with $dt = 0.02$ for $10^4$ steps: $p_\theta$ relative drift `< 1e-12`.
   - **Zero-energy case:** a double pendulum released from horizontal reports a finite `max_rel_drift`.
5. **$C^2$ Continuity & Kinematic Identity:**
   - At every knot, segment $k$ evaluated at $\tau = 1$ and segment $k + 1$ at $\tau = 0$ agree to `1e-13` in $q$ and $\dot{q}$ and `1e-11` in $\ddot{q}$, relative to $\max(1, |x|)$.
   - Inside segments, central finite differences of `sample(t).q` match `sample(t).dq` and `sample(t).ddq` within finite-difference error.
   - A crossing test with two roots inside one step finds both.
6. **Streaming Equivalence:** Segments produced by looping `Stepper::step_bounded` are bitwise identical to those inside the `Trajectory<N>` returned by `simulate()` for the same inputs.
7. **Control & Linearization:**
   - `linearize` on `CartPole` about the upright equilibrium matches the closed form to `1e-10` relative. With $\theta$ from upright, pole tip at $x + l\sin\theta$, cart mass $M_c$, point mass $m$: $\ddot{x} = (u - m g\,\theta)/M_c$ and $\ddot{\theta} = ((M_c + m)\,g\,\theta - u)/(M_c\,l)$.
   - `linearize` matches central finite differences of `accelerations` to `1e-6` relative at a random non-equilibrium state of the double pendulum. *Uncalibrated.*
   - Changing the input between steps keeps $q$ and $\dot{q}$ continuous, and the next segment's initial $\ddot{q}$ equals `accelerations` with the new input.
8. **Kapitza Dynamic Stabilization Test:**
   - With pivot amplitude $a$ and angular frequency $\nu$ chosen so $a^2\nu^2 \ge 4 g l$ (twice the stability bound $a^2\nu^2 > 2 g l$), start at $\theta = \pi + 0.08\text{ rad}$ and verify $|\theta - \pi| < 0.2$ for 500 drive periods.
   - Control run with $a^2\nu^2 = g l$ (below the bound): the pendulum falls, $|\theta - \pi| > 1$. *Uncalibrated.*
9. **Schwarzschild Black Hole Geodesic Test:**
   - Verify the circular photon orbit at $r = 1.5\,r_s$ is unstable.
   - Along an eccentric precessing orbit started in the equatorial plane, `Dopri5` tight: the cyclic momenta $p_t$ and $p_\varphi$ and the norm $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu$ drift `< 1e-9` relative, and $\theta$ stays at $\pi/2$. *Uncalibrated.*
10. **Feature & Target Builds:**
    - `cargo check -p lagrangian --no-default-features --features libm` (no `alloc`)
    - `cargo check -p lagrangian --no-default-features --features libm,alloc`
    - `cargo check -p lagrangian --target wasm32-unknown-unknown` (run `rustup target add wasm32-unknown-unknown` first; this check is mandatory)
    - `cargo test -p lagrangian --features serde`: a `Trajectory<2>` round-trips through `serde_json` (the one permitted dev-dependency) and samples identically.

## Execution Plan
Work methodically in phases, running the verification commands from `CLAUDE.md` at the end of each phase before moving to the next:
- **Phase 1:** Workspace scaffold + sealed `Scalar` + AD (`Dual`, `Grad<N>`, `Taylor2`) + `linalg` + AD oracle and linear solver tests.
- **Phase 2:** Engine (`Lagrangian<N>`, `eom_terms`, `accelerations`, Legendre energy/momentum, `SceneMap`, `velocity_of`) + `Segment<N>` + `Stepper` + `Dopri5` + `Trajectory<N>` (`Cursor`, `crossings`/Poincaré). Stop and tell the owner the name-reservation milestone is reached; publishing to crates.io is the owner's action.
- **Phase 3:** `Variational2` / `Variational4` + `Controlled<N, M>` / `WithInput` / `linearize` + built-in `systems` + the full verification suite. Re-baseline thresholds into `PLAN.md` §7.
- **Phase 4:** `benches`, `svg_showcase` (run it to generate sample SVGs), `ode_adapters`, `egui_playground` (native + WASM Trunk setup), and the flagship `README.md`.
- **Phase 5:** `bevy_lagrangian`, then prepare `v0.1.0`.
