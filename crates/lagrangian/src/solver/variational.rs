//! The midpoint variational integrator and its fourth-order composition.
//!
//! A variational integrator discretizes the action instead of the equations
//! of motion. With the midpoint rule the discrete Lagrangian is
//!
//! ```text
//! L_d(q₀, q₁) = h · L((q₀ + q₁)/2, (q₁ − q₀)/h, t + h/2)
//! ```
//!
//! and one step `(q₀, p₀) → (q₁, p₁)` solves the discrete Euler-Lagrange
//! equations
//!
//! ```text
//! p₀ + D₁L_d + (h/2)·Q = 0        p₁ = D₂L_d + (h/2)·Q
//! ```
//!
//! The map is symplectic for any Lagrangian with an invertible mass matrix,
//! including the non-separable ones (a mass matrix that depends on `q`)
//! where velocity Verlet is not.

// Filling a matrix column by column indexes two arrays with one counter.
#![allow(clippy::needless_range_loop)]

use core::array::from_fn;

use super::Stepper;
use crate::ad::{Dual, Grad};
use crate::error::SimError;
use crate::linalg;
use crate::mechanics::{
    Lagrangian, NewtonMonitor, accelerations, max_abs, solve_failure, velocity_from_momentum,
};
use crate::segment::{Segment, State};

/// Yoshida's weights: three second-order symmetric steps of sizes
/// `w1·h, w0·h, w1·h` make one fourth-order step. `w1 = 1/(2 − 2^⅓)` and
/// `w0 = 1 − 2·w1`, which is negative: the middle step runs backward.
const W1: f64 = 1.351_207_191_959_657_6;
const W0: f64 = -1.702_414_383_919_315_3;

/// One step of size `dt` (or less, to land on `t_max`).
pub(super) fn step<L: Lagrangian<N>, const N: usize>(
    s: &mut Stepper<L, N>,
    dt: f64,
    fourth_order: bool,
    t_max: Option<f64>,
) -> Result<Segment<N>, SimError> {
    let start = s.state;

    // Time is counted in steps from an anchor, not accumulated.
    let natural_end = s.anchor_t + (s.anchor_steps + 1.0) * dt;
    let (end_t, shortened) = match t_max {
        // Within rounding of the grid: land on `t_max` itself, so the
        // caller's loop ends, but stay on the grid. A run made in pieces
        // that end on step boundaries then takes the same steps as a run
        // made in one go.
        Some(tm) if natural_end >= tm - 1e-9 * dt => (tm, natural_end > tm + 1e-9 * dt),
        _ => (natural_end, false),
    };
    let h = end_t - start.t;

    let weights: &[f64] = if fourth_order { &[W1, W0, W1] } else { &[1.0] };
    let (mut q, mut p, mut t) = (start.q, s.p, start.t);
    let mut velocity = start.dq;
    for &w in weights {
        let sub = w * h;
        let next = discrete_step(&s.sys, q, p, t, sub, velocity)?;
        (q, p, t, velocity) = (next.q, next.p, t + sub, next.velocity);
    }

    // The knot needs a velocity and an acceleration; the integrator carries
    // a momentum. The inverse Legendre transform recovers the first, and the
    // equations of motion give the second.
    let dq = velocity_from_momentum(&s.sys, q, p, end_t, velocity)?;
    let ddq = accelerations(&s.sys, q, dq, end_t)?;
    let end = State {
        t: end_t,
        q,
        dq,
        ddq,
    };
    if !end.is_finite() {
        return Err(SimError::NonFiniteState { t: start.t });
    }

    if shortened {
        // Off the grid: count from here.
        s.anchor_t = end_t;
        s.anchor_steps = 0.0;
    } else {
        s.anchor_steps += 1.0;
    }
    s.state = end;
    s.p = p;
    Ok(Segment::from_knots(&start, &end))
}

/// Where one midpoint step ends.
struct Substep<const N: usize> {
    q: [f64; N],
    p: [f64; N],
    /// The mean velocity over the step: the guess for the next one.
    velocity: [f64; N],
}

/// The residual of the discrete Euler-Lagrange equation and what is needed
/// to finish the step once it is small.
struct Residual<const N: usize> {
    /// `p₀ + D₁L_d + (h/2)·Q`.
    value: [f64; N],
    /// Its derivative with respect to the increment.
    jacobian: [[f64; N]; N],
    /// `∇_q L + Q` at the midpoint.
    force: [f64; N],
    /// The size of the terms that were summed into `value`: what its
    /// rounding error is relative to.
    scale: f64,
}

/// One midpoint step of size `h`, which may be negative.
///
/// Newton's unknown is the **increment** `d = q₁ − q₀`. The midpoint is
/// `q₀ + d/2` and the velocity `d/h`, so nothing is ever formed by
/// subtracting two stored positions: that subtraction loses `|q|/h` digits
/// and was measured to cost three orders of magnitude in momentum
/// conservation (`PLAN.md` §7).
fn discrete_step<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    p: [f64; N],
    t: f64,
    h: f64,
    velocity_guess: [f64; N],
) -> Result<Substep<N>, SimError> {
    let t_mid = t + 0.5 * h;
    let mut d: [f64; N] = from_fn(|i| h * velocity_guess[i]);
    let mut monitor = NewtonMonitor::new();

    for _ in 0..NewtonMonitor::MAX_ITERATIONS {
        let r = residual(sys, q, p, d, t_mid, h);
        if !r.value.iter().all(|x| x.is_finite()) {
            return Err(SimError::NonFiniteState { t });
        }
        if monitor.done(max_abs(&r.value), r.scale) {
            // p₁ = D₂L_d + (h/2)·Q, rewritten with the equation just solved
            // as p₁ = p₀ + h·(∇_q L + Q). The two agree to the residual, and
            // this form leaves the momentum of a cyclic coordinate (one L
            // does not depend on) untouched to the last bit, which is the
            // discrete Noether theorem holding exactly rather than to
            // rounding.
            return Ok(Substep {
                q: from_fn(|i| q[i] + d[i]),
                p: from_fn(|i| p[i] + h * r.force[i]),
                velocity: d.map(|x| x / h),
            });
        }
        let negated = r.value.map(|x| -x);
        let update =
            linalg::solve(&r.jacobian, &negated).ok_or_else(|| solve_failure(&r.jacobian, t))?;
        for i in 0..N {
            d[i] += update[i];
        }
    }
    Err(SimError::NewtonDidNotConverge { t })
}

/// The residual and its exact Jacobian at increment `d`.
///
/// Write `L̃(q₀, d) = h·L(q₀ + d/2, d/h)`. Then `D₂L_d = ∂L̃/∂d` and
/// `D₁L_d = ∂L̃/∂q₀ − ∂L̃/∂d`, so two gradients give the residual, and
/// carrying one more derivative along each `d_j` gives its Jacobian:
/// `2N` evaluations of the Lagrangian in all.
fn residual<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    p: [f64; N],
    d: [f64; N],
    t_mid: f64,
    h: f64,
) -> Residual<N> {
    let mid: [f64; N] = from_fn(|i| q[i] + 0.5 * d[i]);
    let vel: [f64; N] = d.map(|x| x / h);
    let unit = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let time = Dual::constant(Grad::<N>::constant(t_mid));

    // by_d.d   = ∂L/∂d   = ½·∇_q L + ∇_q̇ L / h
    // by_q0.d  = ∂L/∂q₀  = ∇_q L
    let mut by_d = [0.0; N];
    let mut by_q0 = [0.0; N];
    let mut jacobian = [[0.0; N]; N];
    for j in 0..N {
        let along_d = sys.lagrangian(
            from_fn(|i| {
                let mut g = Grad::constant(mid[i]);
                g.d[i] = 0.5;
                Dual::new(g, Grad::constant(0.5 * unit(i, j)))
            }),
            from_fn(|i| {
                let mut g = Grad::constant(vel[i]);
                g.d[i] = 1.0 / h;
                Dual::new(g, Grad::constant(unit(i, j) / h))
            }),
            time,
        );
        let along_q0 = sys.lagrangian(
            from_fn(|i| Dual::new(Grad::variable(mid[i], i), Grad::constant(0.5 * unit(i, j)))),
            from_fn(|i| Dual::new(Grad::constant(vel[i]), Grad::constant(unit(i, j) / h))),
            time,
        );
        by_d = along_d.val.d;
        by_q0 = along_q0.val.d;
        for i in 0..N {
            jacobian[i][j] = h * (along_q0.eps.d[i] - along_d.eps.d[i]);
        }
    }

    // The non-conservative force and its derivative with respect to d.
    let external = sys.generalized_force(
        from_fn(|i| {
            let mut g = Grad::constant(mid[i]);
            g.d[i] = 0.5;
            g
        }),
        from_fn(|i| {
            let mut g = Grad::constant(vel[i]);
            g.d[i] = 1.0 / h;
            g
        }),
        Grad::<N>::constant(t_mid),
    );

    let mut value = [0.0; N];
    let mut force = [0.0; N];
    let mut scale = 0.0_f64;
    for i in 0..N {
        let conservative = h * (by_q0[i] - by_d[i]);
        let pushed = 0.5 * h * external[i].val;
        value[i] = p[i] + conservative + pushed;
        force[i] = by_q0[i] + external[i].val;
        scale = scale
            .max(crate::math::abs(p[i]))
            .max(crate::math::abs(h * by_q0[i]))
            .max(crate::math::abs(h * by_d[i]))
            .max(crate::math::abs(pushed));
        for j in 0..N {
            jacobian[i][j] += 0.5 * h * external[i].d[j];
        }
    }
    Residual {
        value,
        jacobian,
        force,
        scale,
    }
}
