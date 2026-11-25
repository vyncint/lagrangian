//! A Lagrangian mechanics engine: write `L = T − V` in plain generic Rust and
//! get exact equations of motion by forward-mode automatic differentiation.
//!
//! You implement one trait, [`Lagrangian`], with one method. You do not
//! derive Euler-Lagrange equations, a mass matrix, a Coriolis term or a
//! Jacobian; the crate differentiates the function you wrote.
//!
//! ```
//! use lagrangian::prelude::*;
//!
//! struct Pendulum { l: f64, g: f64 }
//!
//! impl Lagrangian<1> for Pendulum {
//!     fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
//!         (dq[0] * self.l).powi(2) * 0.5 + q[0].cos() * (self.g * self.l)
//!     }
//! }
//!
//! let pendulum = Pendulum { l: 1.0, g: 9.81 };
//! let (q, dq) = ([1.0], [0.0]);
//!
//! // The engine: plain functions of plain arrays.
//! let ddq = accelerations(&pendulum, q, dq, 0.0)?;      // θ̈ = −(g/l)·sin θ
//! assert!((ddq[0] + 9.81 * 1.0_f64.sin()).abs() < 1e-12);
//! let e0 = energy(&pendulum, q, dq, 0.0).total;
//!
//! // A solver built on it, one step at a time, with no allocation.
//! let mut stepper = Stepper::new(&pendulum, 0.0, q, dq, SimConfig::default())?;
//! while stepper.state().t < 2.0 {
//!     stepper.step_bounded(2.0)?;
//! }
//! let end = stepper.state();
//! assert!((energy(&pendulum, end.q, end.dq, end.t).total - e0).abs() < 1e-7);
//! # Ok::<(), SimError>(())
//! ```
//!
//! # Three layers
//!
//! 1. **The engine**: [`eom_terms`], [`accelerations`], [`mass_matrix`],
//!    [`momentum`], [`energy`], [`linearize`]. Functions of `f64` arrays
//!    that neither integrate nor allocate. Hand [`accelerations`] to any ODE
//!    solver; everything else in the crate is built from these.
//! 2. **The stepper**: [`Stepper`] advances one accepted step at a time and
//!    returns each as a [`Segment`], a polynomial that can be sampled
//!    anywhere inside the step. Two integrator families share it:
//!    adaptive Dormand-Prince ([`Solver::Dopri5`]) and symplectic
//!    variational integrators ([`Solver::Variational4`]).
//! 3. **The trajectory** (with the `alloc` feature): `simulate` collects the
//!    segments into a `Trajectory`, a `C²`-smooth function of time that can
//!    be scrubbed, looped, searched for events and asked for its energy.
//!
//! # Features
//!
//! | Feature | Default | What it gives |
//! | --- | --- | --- |
//! | `std` | yes | Float math from the standard library. Implies `alloc`. No dependencies. |
//! | `alloc` | via `std` | `Trajectory`, `simulate`, `crossings`. |
//! | `libm` | no | Float math for targets without `std`. The only dependency a `no_std` build has. |
//! | `serde` | no | `Serialize`/`Deserialize` for `State`, `Segment` and `Trajectory`. |
//!
//! Without `std` one of `libm` must be on:
//! `default-features = false, features = ["libm"]`. That build has no
//! allocator requirement; the engine, [`Stepper`] and [`Segment`] all work.
//!
//! # Performance
//!
//! [`accelerations`] evaluates your Lagrangian `N + 2` times with dual
//! numbers `O(N)` wide: 178 ns for the double pendulum in a release build,
//! and 20 µs for a chain of 16 coordinates (`cargo bench -p benches` in the
//! repository; the README has the table).
//!
//! The engine is generic over your Lagrangian, so it is compiled in your
//! crate, at your crate's optimization level, and a default debug build
//! runs it about 40 times slower than a release one. Raising only this
//! crate's level with `[profile.dev.package.lagrangian]` changes nothing.
//! Raising your own brings a debug build within a third of release speed:
//!
//! ```toml
//! [profile.dev]
//! opt-level = 1
//! ```
//!
//! # Stack
//!
//! Nothing here allocates, so everything the engine works on lives on the
//! stack, and the need grows with the square of `N`: every call to your
//! [`Lagrangian`] takes `q` and `q̇` as arrays of dual numbers `2(N + 1)`
//! floats wide, and `4(N + 1)` in [`linearize`]. The smallest thread stack
//! each call completes in, for a chain of `N` springs with one cosine per
//! coordinate, on `x86_64` Linux; 16 KiB is the least a Linux thread gets.
//!
//! | `N` | `accelerations` | `eom_terms` | Dopri5 step | Variational4 step | `linearize_free` |
//! | ---: | ---: | ---: | ---: | ---: | ---: |
//! | release, ≤ 8 | ≤ 16 KiB | ≤ 16 KiB | ≤ 16 KiB | ≤ 16 KiB | ≤ 16 KiB |
//! | release, 16 | 38 KiB | 32 KiB | 47 KiB | 64 KiB | 83 KiB |
//! | release, 32 | 189 KiB | 166 KiB | 206 KiB | 279 KiB | 470 KiB |
//! | release, 64 | 636 KiB | 583 KiB | 724 KiB | 981 KiB | 1.5 MiB |
//! | release, 128 | 2.5 MiB | 2.3 MiB | 2.7 MiB | 3.7 MiB | |
//! | debug, 4 | ≤ 16 KiB | ≤ 16 KiB | ≤ 16 KiB | ≤ 16 KiB | 29 KiB |
//! | debug, 8 | 33 KiB | 32 KiB | 41 KiB | 45 KiB | 64 KiB |
//! | debug, 16 | 87 KiB | 83 KiB | 99 KiB | 112 KiB | 197 KiB |
//! | debug, 32 | 292 KiB | 279 KiB | 304 KiB | 362 KiB | 693 KiB |
//! | debug, 64 | 1.0 MiB | 981 KiB | 1.0 MiB | 1.2 MiB | 2.5 MiB |
//!
//! `tests/stack.rs` runs each operation at `N = 8`, 16 and 32 in a thread
//! with half again the figure for its profile. A Lagrangian with more
//! temporaries needs more.
//!
//! What an overflow does depends on the target, and so does the remedy:
//!
//! - **Operating systems:** the process aborts with "thread … has
//!   overflowed its stack". A spawned thread gets 2 MiB unless told
//!   otherwise, and so does each test thread; give a thread more with
//!   `std::thread::Builder::stack_size`, and tests with `RUST_MIN_STACK`.
//! - **`wasm32-unknown-unknown`:** the stack is 1 MiB, placed below the
//!   static data, and an overflow traps with "memory access out of bounds"
//!   without touching that data. Raise it with
//!   `-C link-arg=-zstack-size=<bytes>`.
//! - **Bare metal:** with `cortex-m-rt`'s default memory layout the stack
//!   grows down into `.bss` and `.data` with no guard, so an overflow can
//!   corrupt memory silently: the one hazard `#![forbid(unsafe_code)]`
//!   cannot rule out. Size the stack from the table, and place it below the
//!   static data, with `flip-link` or a linker script, so that an overflow
//!   faults instead.
//!
//! # What it is not
//!
//! Coordinates must be minimal: there are no constraint forces or Lagrange
//! multipliers, no contacts and no collisions. The number of coordinates is
//! fixed at compile time and the cost grows as its square, so this is for
//! systems of a handful of degrees of freedom, not for a robot arm with
//! fifty joints or a cloth.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "lagrangian needs a source of float math: enable the default `std` feature, \
     or `libm` for targets without the standard library \
     (`default-features = false, features = [\"libm\"]`)"
);

#[cfg(feature = "alloc")]
extern crate alloc;

// With both `std` and `libm` on, `std` supplies the math and `libm` is built
// but never called. Feature unification makes that combination ordinary, so
// it must not look like a forgotten dependency.
#[cfg(all(feature = "std", feature = "libm"))]
use libm as _;

pub mod ad;
pub mod linalg;
pub mod systems;

mod error;
mod math;
mod mechanics;
#[cfg(feature = "alloc")]
mod roots;
mod scalar;
mod segment;
#[cfg(feature = "serde")]
mod serde_impl;
mod solver;
#[cfg(feature = "alloc")]
mod trajectory;

pub use error::SimError;
pub use mechanics::{
    Controlled, Energy, EomTerms, Kinematics, Lagrangian, Linearization, SceneMap, WithInput,
    accelerations, energy, eom_terms, kinematics, linearize, linearize_free, mass_matrix, momentum,
    velocity_from_momentum, velocity_of,
};
pub use scalar::Scalar;
pub use segment::{Segment, State};
pub use solver::{SimConfig, Solver, Stepper};
#[cfg(feature = "alloc")]
pub use trajectory::{CrossingDirection, Cursor, EnergyReport, Trajectory, simulate};

/// Everything a typical program needs, in one import.
pub mod prelude {
    pub use crate::{
        Controlled, Energy, Kinematics, Lagrangian, Scalar, SceneMap, Segment, SimConfig, SimError,
        Solver, State, Stepper, WithInput, accelerations, energy, eom_terms, kinematics, linearize,
        mass_matrix, momentum, velocity_of,
    };
    #[cfg(feature = "alloc")]
    pub use crate::{CrossingDirection, Cursor, EnergyReport, Trajectory, simulate};
}

/// The README's Rust blocks, compiled and run as doctests, so the front
/// page cannot drift from the crate. They use `simulate`, hence the feature.
#[cfg(all(doctest, feature = "std"))]
#[doc = include_str!("../README.md")]
mod readme {}
