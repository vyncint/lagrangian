//! Integrating the equations of motion one step at a time.

use core::fmt;

use crate::error::SimError;
use crate::mechanics::{Lagrangian, accelerations, momentum};
use crate::segment::{Segment, State};

mod dopri5;
mod variational;

/// Which integrator advances the state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Solver {
    /// Dormand-Prince 5(4) with adaptive step size: the default. Accurate
    /// per step and efficient, but energy error accumulates over long runs.
    Dopri5,
    /// The midpoint variational integrator: second order, symplectic, fixed
    /// step `dt`. Energy error stays bounded however long the run.
    Variational2 {
        /// The step size.
        dt: f64,
    },
    /// Three midpoint steps composed with Yoshida's coefficients: fourth
    /// order, symplectic, fixed step `dt`.
    Variational4 {
        /// The step size.
        dt: f64,
    },
}

/// How a [`Stepper`] integrates.
///
/// Start from [`SimConfig::default`] or one of the named constructors and
/// adjust with the `with_*` methods:
///
/// ```
/// use lagrangian::SimConfig;
///
/// let tight = SimConfig::dopri5().with_tolerances(1e-12, 1e-14);
/// let symplectic = SimConfig::variational4(0.005);
/// # let _ = (tight, symplectic);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SimConfig {
    /// The integrator.
    pub solver: Solver,
    /// Relative error tolerance per step ([`Solver::Dopri5`]). Default `1e-9`.
    pub rtol: f64,
    /// Absolute error tolerance per step ([`Solver::Dopri5`]). Default `1e-11`.
    pub atol: f64,
    /// The smallest step the adaptive solver may take before giving up with
    /// [`SimError::StepSizeUnderflow`]. The default, `0`, leaves only the
    /// limit of floating point.
    pub min_step: f64,
    /// The largest step the adaptive solver may take. Default: unlimited.
    pub max_step: f64,
    /// The first step of the adaptive solver. Default: chosen from the
    /// initial state.
    pub initial_step: Option<f64>,
    /// How many accepted steps one call to `simulate` or
    /// `Trajectory::extend` may take before it stops with
    /// [`SimError::MaxStepsExceeded`]. A [`Stepper`] used directly has no
    /// such budget: it is the caller's loop. Default `1_000_000`.
    pub max_steps: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            solver: Solver::Dopri5,
            rtol: 1e-9,
            atol: 1e-11,
            min_step: 0.0,
            max_step: f64::INFINITY,
            initial_step: None,
            max_steps: 1_000_000,
        }
    }
}

impl SimConfig {
    /// The adaptive Dormand-Prince solver with default tolerances.
    pub fn dopri5() -> Self {
        Self::default()
    }

    /// The second-order variational integrator with step `dt`.
    pub fn variational2(dt: f64) -> Self {
        Self::default().with_solver(Solver::Variational2 { dt })
    }

    /// The fourth-order variational integrator with step `dt`.
    pub fn variational4(dt: f64) -> Self {
        Self::default().with_solver(Solver::Variational4 { dt })
    }

    /// Sets the integrator.
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Sets the relative and absolute tolerances of the adaptive solver.
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    /// Sets the smallest step the adaptive solver may take.
    pub fn with_min_step(mut self, min_step: f64) -> Self {
        self.min_step = min_step;
        self
    }

    /// Sets the largest step the adaptive solver may take.
    pub fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = max_step;
        self
    }

    /// Sets the first step of the adaptive solver.
    pub fn with_initial_step(mut self, initial_step: f64) -> Self {
        self.initial_step = Some(initial_step);
        self
    }

    /// Sets the step budget of one `simulate` or `Trajectory::extend` call.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Refuses values no integration can run with. The comparisons are
    /// written so that a `NaN` fails them.
    fn validate(&self) -> Result<(), SimError> {
        let bad = |reason| Err(SimError::InvalidConfig { reason });
        match self.solver {
            Solver::Dopri5 => {
                if !(self.rtol >= 0.0 && self.atol >= 0.0 && self.rtol + self.atol > 0.0) {
                    return bad("rtol and atol must be non-negative and not both zero");
                }
                if !(self.rtol.is_finite() && self.atol.is_finite()) {
                    return bad("rtol and atol must be finite");
                }
                if !(self.min_step >= 0.0 && self.min_step.is_finite()) {
                    return bad("min_step must be finite and non-negative");
                }
                if !(self.max_step > 0.0 && self.max_step >= self.min_step) {
                    return bad("max_step must be positive and at least min_step");
                }
                if let Some(h) = self.initial_step {
                    if !(h > 0.0 && h.is_finite()) {
                        return bad("initial_step must be positive and finite");
                    }
                }
            }
            Solver::Variational2 { dt } | Solver::Variational4 { dt } => {
                if !(dt > 0.0 && dt.is_finite()) {
                    return bad("dt must be positive and finite");
                }
            }
        }
        Ok(())
    }
}

/// Advances a system one accepted step at a time.
///
/// Each call to [`step`](Stepper::step) returns the [`Segment`] it covered:
/// a polynomial that can be sampled anywhere inside the step. Nothing is
/// allocated, so this is the whole simulation loop on a target with no heap,
/// and the natural shape for anything interactive: change a parameter or an
/// input between two steps and the next one sees it.
///
/// ```
/// use lagrangian::prelude::*;
/// use lagrangian::systems::DoublePendulum;
///
/// let pendulum = DoublePendulum::default();
/// let mut stepper = Stepper::new(&pendulum, 0.0, [1.0, 2.0], [0.0, 0.0], SimConfig::default())?;
/// let start = energy(&pendulum, [1.0, 2.0], [0.0, 0.0], 0.0).total;
/// while stepper.state().t < 1.0 {
///     let segment = stepper.step_bounded(1.0)?;
///     let mid = segment.eval(0.5);   // any point inside the step
///     assert!(mid.t > segment.t_start() && mid.t < segment.t_end());
/// }
/// let end = stepper.state();
/// assert_eq!(end.t, 1.0);
/// assert!((energy(&pendulum, end.q, end.dq, end.t).total - start).abs() < 1e-7);
/// # Ok::<(), SimError>(())
/// ```
///
/// The system is taken by value. Pass a reference (`&sys`) to keep using it
/// elsewhere, or move it in to own it.
pub struct Stepper<L, const N: usize> {
    sys: L,
    config: SimConfig,
    /// The knot the next step starts from.
    state: State<N>,
    /// The momentum at that knot.
    p: [f64; N],
    /// Whether `state.ddq` and `p` describe `sys` as it is now. A change to
    /// the system makes both stale; they are recomputed before the next
    /// step, which is also what keeps Dormand-Prince's reuse of the last
    /// stage honest.
    fresh: bool,
    /// The step the adaptive solver will try next; `0` until chosen.
    h: f64,
    /// The previous step's error, for the PI controller.
    err_prev: f64,
    /// Fixed-step solvers count steps from an anchor instead of adding `dt`
    /// to the time over and over, which would let rounding drift in.
    anchor_t: f64,
    anchor_steps: f64,
}

impl<L, const N: usize> fmt::Debug for Stepper<L, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stepper")
            .field("config", &self.config)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl<L: Lagrangian<N>, const N: usize> Stepper<L, N> {
    /// Starts at time `t0` from coordinates `q0` and velocities `dq0`.
    ///
    /// # Errors
    ///
    /// [`SimError::InvalidConfig`] for a configuration that cannot run, and
    /// whatever [`accelerations`] reports for the initial state.
    pub fn new(
        sys: L,
        t0: f64,
        q0: [f64; N],
        dq0: [f64; N],
        config: SimConfig,
    ) -> Result<Self, SimError> {
        config.validate()?;
        let mut stepper = Self {
            sys,
            config,
            state: State {
                t: t0,
                q: q0,
                dq: dq0,
                ddq: [0.0; N],
            },
            p: [0.0; N],
            fresh: false,
            h: 0.0,
            err_prev: 1e-4,
            anchor_t: t0,
            anchor_steps: 0.0,
        };
        stepper.refresh()?;
        Ok(stepper)
    }

    /// Advances by one accepted step and returns the segment it covered.
    ///
    /// # Errors
    ///
    /// Any [`SimError`]: the mass matrix became singular, the state stopped
    /// being finite, the adaptive step collapsed, or an implicit step did
    /// not converge. The stepper is left at the last good knot.
    pub fn step(&mut self) -> Result<Segment<N>, SimError> {
        self.advance(None)
    }

    /// Like [`step`](Stepper::step), but never past `t_max`: the step is
    /// shortened to land on it exactly.
    ///
    /// # Errors
    ///
    /// As [`step`](Stepper::step), and [`SimError::InvalidConfig`] if
    /// `t_max` is not after the current time.
    pub fn step_bounded(&mut self, t_max: f64) -> Result<Segment<N>, SimError> {
        if t_max.partial_cmp(&self.state.t) != Some(core::cmp::Ordering::Greater) {
            return Err(SimError::InvalidConfig {
                reason: "t_max must be after the current time",
            });
        }
        self.advance(Some(t_max))
    }

    fn advance(&mut self, t_max: Option<f64>) -> Result<Segment<N>, SimError> {
        self.refresh()?;
        match self.config.solver {
            Solver::Dopri5 => dopri5::step(self, t_max),
            Solver::Variational2 { dt } => variational::step(self, dt, false, t_max),
            Solver::Variational4 { dt } => variational::step(self, dt, true, t_max),
        }
    }

    /// Brings the acceleration and momentum of the current knot up to date
    /// with the system as it is now.
    ///
    /// Called automatically before every step. Call it yourself after
    /// [`system_mut`](Stepper::system_mut) if you want
    /// [`state`](Stepper::state) to show the new acceleration before
    /// stepping.
    ///
    /// # Errors
    ///
    /// Whatever [`accelerations`] reports for the current state.
    pub fn refresh(&mut self) -> Result<(), SimError> {
        if self.fresh {
            return Ok(());
        }
        let State { t, q, dq, .. } = self.state;
        if !(t.is_finite() && q.iter().chain(dq.iter()).all(|x| x.is_finite())) {
            return Err(SimError::NonFiniteState { t });
        }
        self.state.ddq = accelerations(&self.sys, q, dq, t)?;
        self.p = momentum(&self.sys, q, dq, t);
        self.fresh = true;
        Ok(())
    }

    /// The knot the next step starts from.
    pub fn state(&self) -> State<N> {
        self.state
    }

    /// The configuration.
    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    /// The system.
    pub fn system(&self) -> &L {
        &self.sys
    }

    /// The system, for changing a parameter or an input between steps.
    ///
    /// The change takes effect at the current knot: the next segment starts
    /// with the acceleration the changed system produces there, so the
    /// acceleration is discontinuous at that knot, as it is physically when
    /// a force is switched on.
    pub fn system_mut(&mut self) -> &mut L {
        self.fresh = false;
        &mut self.sys
    }

    /// Replaces the coordinates and velocities at the current time: an
    /// impact, a reset, a teleport.
    ///
    /// # Errors
    ///
    /// Whatever [`accelerations`] reports for the new state, in which case
    /// the stepper keeps its previous one.
    pub fn set_state(&mut self, q: [f64; N], dq: [f64; N]) -> Result<(), SimError> {
        let previous = (self.state, self.p, self.fresh);
        self.state.q = q;
        self.state.dq = dq;
        self.fresh = false;
        if let Err(e) = self.refresh() {
            (self.state, self.p, self.fresh) = previous;
            return Err(e);
        }
        // Fixed-step solvers count their steps from here on.
        self.anchor_t = self.state.t;
        self.anchor_steps = 0.0;
        Ok(())
    }

    /// The stepper's system, consuming it.
    pub fn into_system(self) -> L {
        self.sys
    }
}
