//! The one error type of the crate.

use core::fmt;

/// Why a computation could not be completed.
///
/// Every variant that happens at a moment of simulated time carries that
/// time, so a failure deep in a long run says where it happened.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SimError {
    /// The mass matrix `∂²L/∂q̇∂q̇` is singular to working precision: a
    /// coordinate with no kinetic term, or a coordinate singularity such as
    /// the pole of a sphere.
    SingularMassMatrix {
        /// The time at which it was evaluated.
        t: f64,
    },
    /// The adaptive solver could not meet its tolerance with any step it is
    /// allowed to take.
    StepSizeUnderflow {
        /// The time at which the step size collapsed.
        t: f64,
    },
    /// More accepted steps were needed than `SimConfig::max_steps` allows.
    MaxStepsExceeded {
        /// The time reached when the budget ran out.
        t: f64,
    },
    /// An implicit equation did not converge: the variational step, or the
    /// recovery of a velocity from a momentum.
    NewtonDidNotConverge {
        /// The time of the step being solved.
        t: f64,
    },
    /// A state, or something computed from it, is `NaN` or infinite: a
    /// division by zero or an unguarded `sqrt(0)` inside the Lagrangian, a
    /// solution that blew up, or coordinates on a horizon.
    NonFiniteState {
        /// The time at which it was noticed.
        t: f64,
    },
    /// A configuration value makes no sense: a tolerance or a step that is
    /// not positive, or an end time before the current time.
    InvalidConfig {
        /// Which value, and what is wrong with it.
        reason: &'static str,
    },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::SingularMassMatrix { t } => {
                write!(f, "the mass matrix is singular at t = {t}")
            }
            Self::StepSizeUnderflow { t } => write!(
                f,
                "the step size underflowed at t = {t}: the tolerance cannot be met"
            ),
            Self::MaxStepsExceeded { t } => {
                write!(f, "the step budget ran out at t = {t}")
            }
            Self::NewtonDidNotConverge { t } => {
                write!(f, "Newton's iteration did not converge at t = {t}")
            }
            Self::NonFiniteState { t } => {
                write!(f, "the state is NaN or infinite at t = {t}")
            }
            Self::InvalidConfig { reason } => write!(f, "invalid configuration: {reason}"),
        }
    }
}

impl core::error::Error for SimError {}
