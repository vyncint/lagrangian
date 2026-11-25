# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A breaking change is always listed under a **Changed** or **Removed**
heading, in a bullet that begins **Breaking:**.

## [Unreleased]

### Fixed

- `Stepper::step` can no longer hang ([#4]). At rest the adaptive step grew
  tenfold per step until it was infinite, and the next call never returned;
  a step a few ulps short of a `step_bounded` target rounded back up to it
  after every rejection; and a first-step estimate that overflowed became an
  infinite step. The step is now capped at `f64::MAX`, landing is decided by
  comparing lengths rather than rounded times, a call makes at most 1000
  attempts, and a non-finite estimate falls back to a small step.
- A very short step gives a correct segment ([#7]). Below `h ≈ 1e-64` the
  segment was infinite or `NaN`, and a step a few ulps long reported
  velocities off by order one inside it. Segments are now built from the
  increments the solver added to the start knot, without forming powers of
  `h`, and a short step is represented by the lower-degree polynomial that
  still reaches its end to rounding. A variational step shorter than
  `16 ε` is taken as an explicit Taylor step, where Newton would divide by
  it.
- `atan2` and `hypot` on the dual-number types kept a derivative of zero, or
  lost it, for arguments below about `1e-154`, where squares underflow
  ([#16]). Both are now scaled before anything is squared.
- `linalg::solve`, and so `accelerations`, called invertible matrices
  singular when their entries spanned more than about `7e13`: the
  Schwarzschild metric within `1e-7·r_s` of the horizon, or a system that
  mixes very different units ([#16]). Rows and columns are now equilibrated
  by powers of two before elimination, which adds no rounding.
- With the `serde` feature, reading a trajectory refuses data that would
  sample to `NaN` or infinity: a duration that overflows, a polynomial that
  overflows between the ends of its segment, or an initial state that is not
  finite ([#8]). `Trajectory::push` holds a segment to the same rules.

### Changed

- `simulate` and `Trajectory::extend` refuse an end time that is not finite
  with `InvalidConfig` ([#5]). An infinite one ran until `max_steps`, and a
  `NaN` one returned at once having done nothing.
- `Trajectory::crossings_with` looks at no more than 2¹⁶ points inside each
  step, and `Trajectory::energy_drift` takes no more than 2²⁰ samples
  ([#5]). `usize::MAX` used to mean a call that never returned.
- `Trajectory::start` and the first of `Trajectory::knots` are where the
  first segment starts, when there is one ([#8]). A file whose initial state
  disagreed with its first segment sampled to one at its start time and to
  the other just after it.
- `Stepper::set_state` also resets the step size and the error history: the
  next step is the one a new stepper would take from that state.
- A fixed step too short for the clock to advance at the current time is
  reported as `StepSizeUnderflow`. The message of that error no longer
  mentions a tolerance.
- The documentation of `SimConfig::max_steps` and `simulate` gives the memory
  a call can hold, and SECURITY.md lists what bounds each resource ([#5]).

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
[#4]: https://github.com/vyncint/lagrangian/issues/4
[#5]: https://github.com/vyncint/lagrangian/issues/5
[#7]: https://github.com/vyncint/lagrangian/issues/7
[#8]: https://github.com/vyncint/lagrangian/issues/8
[#16]: https://github.com/vyncint/lagrangian/issues/16
[0.1.0]: https://github.com/vyncint/lagrangian/releases/tag/v0.1.0
