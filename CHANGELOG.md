# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A breaking change is always listed under a **Changed** or **Removed**
heading, in a bullet that begins **Breaking:**.

## [Unreleased]

## [0.1.0] - 2026-11-25

The first release: the core crate.

### Added

- The `Lagrangian<N>` trait: a mechanical system is one generic function,
  `L(q, q̇, t)`, written with ordinary arithmetic on a sealed `Scalar`.
  Non-conservative forces are an optional second method.
- The equations-of-motion engine, as plain functions of `f64` arrays that
  neither integrate nor allocate: `eom_terms`, `accelerations`,
  `mass_matrix`, `momentum`, `energy` and `velocity_from_momentum`. All
  terms come from `N + 2` evaluations of the Lagrangian with stack-allocated
  forward-mode dual numbers.
- `Controlled<N, M>` and `WithInput` for systems with external inputs, and
  `linearize` / `linearize_free`, which return the exact blocks of the
  state-space pair `(A, B)` about any state.
- `SceneMap`, `kinematics` and `velocity_of`: the position, velocity and
  acceleration of any point of a mechanism from its position alone.
- `Stepper`, which advances one accepted step at a time without allocating
  and returns each step as a `Segment<N>`: a `Copy` quintic Hermite
  polynomial matching position, velocity and acceleration at both ends.
- Two integrator families behind it: adaptive Dormand-Prince 5(4)
  (`Solver::Dopri5`) and the midpoint variational integrator at second and
  fourth order (`Solver::Variational2`, `Solver::Variational4`), which is
  symplectic for non-separable Lagrangians and conserves the momentum of a
  cyclic coordinate exactly.
- With the `alloc` feature, `simulate` and `Trajectory<N>`: a `C²` function
  of time that can be sampled in any order, looped, played with a `Cursor`,
  searched for events and Poincaré sections with `crossings`, and asked for
  its energy drift and its action.
- Six built-in systems in `lagrangian::systems`: `DoublePendulum`,
  `BeadOnRotatingHoop`, `KapitzaPendulum`, `CartPole`, `KeplerOrbit` and
  `SchwarzschildGeodesic`.
- `lagrangian::linalg`: stack-allocated linear solves with partial pivoting,
  which accept the indefinite mass matrix of a spacetime metric.
- The `serde` feature: `Serialize` and `Deserialize` for `State`, `Segment`
  and `Trajectory`, refusing data that would break sampling.
- The `libm` feature for targets without `std`. The crate builds for a
  bare-metal Cortex-M target with no allocator, and has no dependencies at
  all with the default features.
- The `svg_showcase` example: phase portraits, Poincaré sections, light
  bending around a black hole and energy-error comparisons as dependency-free
  SVG.

[Unreleased]: https://github.com/vyncint/lagrangian/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vyncint/lagrangian/releases/tag/v0.1.0
