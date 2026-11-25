# lagrangian

[![crates.io](https://img.shields.io/crates/v/lagrangian.svg)](https://crates.io/crates/lagrangian)
[![docs.rs](https://img.shields.io/docsrs/lagrangian)](https://docs.rs/lagrangian)
[![CI](https://github.com/vyncint/lagrangian/actions/workflows/ci.yml/badge.svg)](https://github.com/vyncint/lagrangian/actions/workflows/ci.yml)

**Write the Lagrangian. Get the motion.** A mechanics engine for Rust: you write `L = T − V` as an ordinary generic function, and the crate differentiates it into exact equations of motion, integrates them, and hands back a trajectory you can sample at any time.

No equation of motion, mass matrix, Coriolis term or Jacobian is ever derived by hand. No dependencies. Works without `std` and without an allocator.

> **Status: 0.1.0, the first release.** The core crate is implemented and verified. It is pre-1.0, so a minor release may change the API. The interactive playground and the Bevy plugin described in the [plan](https://github.com/vyncint/lagrangian/blob/main/PLAN.md) are not built yet.

```toml
[dependencies]
lagrangian = "0.1"
```

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

fn main() -> Result<(), SimError> {
    let pendulum = DoublePendulum { m1: 1.0, m2: 1.0, l1: 1.0, l2: 1.0, g: 9.81 };

    // Ten seconds from rest, both rods above horizontal: chaotic.
    let traj = simulate(&pendulum, (0.0, 10.0), [2.2, 2.8], [0.0, 0.0], &SimConfig::default())?;

    let state = traj.sample(3.7);                      // any time, any order
    println!("angles {:?}, velocities {:?}", state.q, state.dq);

    let drift = traj.energy_drift_at_knots(&pendulum); // energy by Legendre transform
    assert!(drift.max_rel_drift < 1e-7);
    Ok(())
}
```

That is the whole program. The same `lagrangian` function is evaluated with dual numbers to produce the mass matrix and every force term; nothing symbolic happens and nothing is allocated while integrating.

## Three ways to use it

### 1. Just the equations of motion

The engine is a set of plain functions of plain arrays. Use them with any ODE solver, in a game loop, or in a controller.

```rust
use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;

fn main() -> Result<(), SimError> {
    let pendulum = DoublePendulum::default();
    let (q, dq, t) = ([1.0, 2.0], [0.5, -0.3], 0.0);

    let ddq = accelerations(&pendulum, q, dq, t)?;   // the right-hand side for your solver
    let terms = eom_terms(&pendulum, q, dq, t);      // M, ∇L, the Coriolis vector, the momentum
    let e = energy(&pendulum, q, dq, t);             // total, kinetic, potential

    // M·q̈ = ∇_q L + Q − C
    let rhs = terms.rhs();
    for i in 0..2 {
        let m_ddq = terms.mass[i][0] * ddq[0] + terms.mass[i][1] * ddq[1];
        assert!((m_ddq - rhs[i]).abs() < 1e-12);
    }
    assert!((e.total - (e.kinetic + e.potential)).abs() < 1e-12);
    Ok(())
}
```

### 2. One step at a time, with no heap

`Stepper` returns each accepted step as a `Segment`: a small `Copy` polynomial you can sample anywhere inside the step. Change a parameter or an input between steps and the next one sees it. This is the whole simulation loop on a microcontroller.

```rust
use lagrangian::prelude::*;
use lagrangian::systems::CartPole;

fn main() -> Result<(), SimError> {
    let cart = CartPole::default();
    // The pole starts 0.1 rad from upright; the input is a force on the cart.
    let mut live = Stepper::new(
        WithInput { sys: &cart, u: [0.0] },
        0.0, [0.0, 0.1], [0.0, 0.0],
        SimConfig::default(),
    )?;

    while live.state().t < 0.5 {
        let s = live.state();
        // Any control law: here, push the cart under the falling pole.
        live.system_mut().u = [40.0 * s.q[1] + 8.0 * s.dq[1]];
        let segment = live.step_bounded(live.state().t + 0.01)?;
        let _halfway = segment.eval(0.5);              // sample inside the step
    }
    assert!(live.state().q[1].abs() < 0.1, "the pole is still up");
    Ok(())
}
```

### 3. A whole trajectory, scrubbable

With an allocator, `simulate` collects the segments into a `Trajectory`: one smooth function of time. Position, velocity and acceleration are continuous across steps, so nothing snaps when you drag a slider.

```rust
use lagrangian::prelude::*;
use lagrangian::systems::KeplerOrbit;

fn main() -> Result<(), SimError> {
    let orbit = KeplerOrbit::default();
    // A symplectic integrator: energy stays in a band however long the run.
    let config = SimConfig::variational4(0.02);
    let traj = simulate(&orbit, (0.0, 100.0), [1.0, 0.0], [0.0, 1.2], &config)?;

    let _frame = traj.eval_normalized(0.25);           // a quarter of the way: animation timelines
    let _loop = traj.sample_looped(1234.5);            // repeat forever
    let [body] = traj.kinematics(&orbit, 42.0);        // Cartesian position, velocity, acceleration
    assert!(body.pos[0].hypot(body.pos[1]) > 0.5);

    // Events without re-integrating: every closest approach to the centre.
    let closest = traj.crossings(|s| s.dq[0], CrossingDirection::Rising);
    assert!(closest.len() > 5);

    // The angle is a cyclic coordinate, so its momentum is conserved.
    let p0 = traj.momentum(&orbit, 0.0)[1];
    let p1 = traj.momentum(&orbit, 100.0)[1];
    assert!((p1 - p0).abs() < 1e-14);
    Ok(())
}
```

## Control: linearize about any state

`linearize` differentiates the equations of motion once more and returns the blocks of the state-space pair `(A, B)` that LQR and pole placement start from. It is exact, not a finite difference.

```rust
use lagrangian::prelude::*;
use lagrangian::systems::CartPole;

fn main() -> Result<(), SimError> {
    let cart = CartPole { m_cart: 1.0, m_pole: 0.1, length: 0.5, g: 9.81, ..Default::default() };
    let lin = linearize(&cart, [0.0, 0.0], [0.0, 0.0], [0.0], 0.0)?;   // upright, at rest

    // θ̈ = (M + m)·g/(M·l)·θ − u/(M·l): the unstable pole a controller must move.
    assert!((lin.a_q[1][1] - 1.1 * 9.81 / 0.5).abs() < 1e-12);
    assert!((lin.b[1][0] + 1.0 / 0.5).abs() < 1e-12);
    Ok(())
}
```

## What has been measured

Every number below is reproduced by a test that runs in CI, on the platform's math library and again on `libm`; where the two differ, the larger is shown. Each test asserts at about ten times the value it measured; the baselines and their setups are in [PLAN.md §7](https://github.com/vyncint/lagrangian/blob/main/PLAN.md). "Default" is `rtol = 1e-9, atol = 1e-11`; "tight" is `rtol = 1e-12, atol = 1e-14`.

| What | Measured | Test |
| --- | --- | --- |
| Engine against an independent scalar hyper-dual oracle, all six built-in systems, moving and at rest | ≤ 2.9e-14 relative | [`oracle.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/oracle.rs) |
| Coupled oscillators against their normal modes, `t ∈ [0, 20]`, mid-step samples included | 1.4e-9 default, 1.3e-12 tight | [`analytic.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/analytic.rs) |
| Period of a pendulum swinging to 2.4 rad against the elliptic integral | 1.2e-9 default, 1.1e-12 tight | [`analytic.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/analytic.rs) |
| Energy of a chaotic double pendulum, Dormand-Prince, `t ∈ [0, 20]`, at the ends of steps | 3.4e-9 default, 8.3e-13 tight | [`energy.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/energy.rs) |
| The same, sampled between the ends of steps | 1.7e-8 default, 1.9e-11 tight | [`energy.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/energy.rs) |
| Energy of the same pendulum, 4th-order variational, `dt = 0.005`, 10⁴ steps | within 2.2e-5, no growth | [`energy.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/energy.rs) |
| Angular momentum of a Kepler orbit, 4th-order variational, 10⁴ steps | 3.7e-16 (rounding) | [`energy.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/energy.rs) |
| The same with Dormand-Prince, tight, `t ∈ [0, 200]` | 2.5e-12 | [`energy.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/energy.rs) |
| Jump across a step boundary in position / velocity / acceleration | 1.4e-17 / 2.6e-16 / 2.2e-14 | [`trajectory.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/trajectory.rs) |
| Cart-pole linearization against its closed form | 4.8e-16 | [`control.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/control.rs) |
| Speed of a charge in a magnetic field after ten turns | 5.6e-12 Dormand-Prince tight, 2.8e-15 variational | [`forces.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/forces.rs) |
| Damped and driven oscillators against their closed forms, 4th-order variational, `dt = 0.01` | 1.9e-7 and 6.3e-8 | [`forces.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/forces.rs) |
| Engine against the oracle on chains of 3, 5 and 8 links | ≤ 1.7e-13 relative | [`chains.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/chains.rs) |
| Energy and angular momentum of a precessing orbit around a black hole, proper time 2000, tight | 8.9e-13 and 3.1e-12 | [`systems.rs`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/tests/systems.rs) |

Two things the table says that are easy to miss:

- **Between the ends of steps the error is larger**, five to twenty times here. The step controller bounds the error where a step ends; in between you are reading an interpolating polynomial with its own error. Tighten the tolerance if mid-step accuracy matters.
- **The two integrators are for different jobs.** Dormand-Prince is more accurate per unit of work, and its energy error grows with time. The variational integrator's error per step is larger, and its energy error does not grow at all.

**There are no performance numbers** because no benchmark has been written yet. Nothing here is a claim about speed.

## How it works

For coordinates `q`, the Euler-Lagrange equations expand to a linear system for the accelerations:

```text
M(q, q̇, t) · q̈ = ∇_q L + Q − C        M_ij = ∂²L/∂q̇_i∂q̇_j
                                       C_i  = Σ_j ∂²L/∂q̇_i∂q_j · q̇_j + ∂²L/∂q̇_i∂t
```

- **Derivatives come from dual numbers**, not symbols and not finite differences. Your function is called with a number type that carries derivatives through every `+`, `*`, `sin` and `powi`. The types live on the stack and are `Copy`.
- **The whole vector `C` is one directional derivative** of the momentum `∇_q̇ L` along `(q̇, 0, 1)`, so it costs one evaluation instead of one per entry. All terms together take `N + 2` evaluations. Each is `O(N)` wide, so the total grows as `N²`: a constant-factor saving, not a different order.
- **`M` is solved with pivoted elimination**, never Cholesky. A mass matrix need not be positive definite: for a geodesic it is the metric, and a spacetime metric has a negative eigenvalue.
- **Energy is the Jacobi integral `p·q̇ − L`**, which is what is conserved in a rotating frame or a magnetic field. `½q̇ᵀMq̇ + V` is wrong for both.
- **Each step is stored as a quintic** matching position, velocity and acceleration at both ends, which is what makes the trajectory twice continuously differentiable.
- **The variational integrator** discretizes the action rather than the equations, which makes it symplectic even when the mass matrix depends on `q`. A coordinate the Lagrangian does not depend on keeps its momentum to the last bit.

## Built-in systems

[`lagrangian::systems`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/src/systems.rs) holds the benchmarks the crate is verified against. Each is a few lines of Lagrangian and nothing else.

| System | Coordinates | Why it is there |
| --- | --- | --- |
| `DoublePendulum` | 2 | Chaos; a mass matrix that depends on the configuration |
| `BeadOnRotatingHoop` | 1 | A rotating frame: the conserved energy is not `T + V` |
| `KapitzaPendulum` | 1 | Explicit time dependence; an upside-down pendulum stabilized by shaking |
| `CartPole` | 2, 1 input | The control benchmark, with friction |
| `KeplerOrbit` | 2 | A cyclic coordinate and its conserved momentum |
| `SchwarzschildGeodesic` | 4 | An indefinite mass matrix: orbits and light around a black hole |

## Figures

[`svg_showcase`](https://github.com/vyncint/lagrangian/blob/main/crates/lagrangian/examples/svg_showcase.rs) draws phase portraits, Poincaré sections, velocity and acceleration arrows, light bending around a black hole and energy-error comparisons as SVG files, in dark and light themes, with no dependencies:

```sh
cargo run --release --example svg_showcase     # writes target/svg_showcase/
```

## Cargo features

| Feature | Default | What it gives |
| --- | --- | --- |
| `std` | yes | Float math from the standard library. Implies `alloc`. **No dependencies.** |
| `alloc` | via `std` | `Trajectory`, `simulate`, `crossings`. |
| `libm` | no | Float math for targets without `std`: the only dependency a `no_std` build has. |
| `serde` | no | `Serialize` / `Deserialize` for `State`, `Segment` and `Trajectory`. |

```toml
# Bare metal: no std, no allocator. The engine, Stepper and Segment all work.
lagrangian = { version = "0.1", default-features = false, features = ["libm"] }
```

Both claims are gates in CI: the dependency tree of each configuration is asserted exactly, and the crate is compiled for a bare-metal Cortex-M target and for `wasm32-unknown-unknown`. Minimum supported Rust version: 1.85.

With `serde_json`, enable its `float_roundtrip` feature if a trajectory must come back bit for bit.

## How it compares

| Tool | What it solves | Relationship |
| --- | --- | --- |
| `rapier`, `avian` | Rigid-body game physics with contacts; Rapier also has reduced-coordinate multibody joints | A different job. This crate has no contacts. It takes arbitrary generalized coordinates from one scalar function and gives exact energy, phase space and scrubbable trajectories. |
| `ode_solvers`, `diffsol`, `differential-equations` | General ODE solvers | Complements. They need `q̈ = f(q, q̇, t)` derived by hand; `accelerations` supplies it. |
| `num-dual` | Generic dual numbers | A lower-level building block. Automatic differentiation is internal here, to keep the crate dependency-free. |
| SymPy mechanics, ModelingToolkit | Symbolic derivation of equations of motion | The same idea with a symbolic engine and a heavier runtime. |

## Limitations

- **Minimal coordinates only.** No constraint forces or Lagrange multipliers, so no closed kinematic loops and no rolling without slipping.
- **No contacts, collisions or impacts.** An event that changes the state has to be applied by hand between steps (`Stepper::set_state`).
- **No stiff solver.** A very stiff spring makes the adaptive solver crawl. `accelerations` can be handed to an implicit solver from another crate.
- **The number of coordinates is fixed at compile time**, and cost grows as its square. This is for systems of a handful of degrees of freedom, not a fifty-joint arm or a cloth.
- **`f64` only.**
- **Forward in time only.**
- **Coordinate singularities are yours to avoid:** the pole of a sphere, the centre of a central force, the horizon of a black hole. They are reported as errors, not handled.
- **`sqrt` at zero has an infinite derivative** and is reported as an error. Write a norm as `x.hypot(y)`, which is defined at the origin.

## Prior art

The idea is old and this crate did not invent it.

- **Structure and Interpretation of Classical Mechanics** (Sussman and Wisdom) and its `scmutils` library turn a Lagrangian function into equations of motion; [Emmy](https://github.com/mentat-collective/emmy) is its Clojure descendant. This crate is that idea for Rust, using dual numbers in place of a symbolic engine.
- **Lagrangian Neural Networks** (Cranmer et al., 2020) obtain the same `q̈ = M⁻¹(…)` formula by automatic differentiation.
- **GeometricIntegrators.jl** provides variational integrators in Julia; the theory is Marsden and West's *Discrete mechanics and variational integrators* (2001).
- **MuJoCo, Pinocchio and Drake** simulate articulated bodies in reduced coordinates with algorithms linear in the number of joints. For a robot, use those.

## Contributing

See [CONTRIBUTING.md](https://github.com/vyncint/lagrangian/blob/main/CONTRIBUTING.md). Commits are signed off under the DCO, and every numeric claim in this file names the test that reproduces it.

## License

Licensed under either of
[Apache License, Version 2.0](https://github.com/vyncint/lagrangian/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/vyncint/lagrangian/blob/main/LICENSE-MIT)
at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
