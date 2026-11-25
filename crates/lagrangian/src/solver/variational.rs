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
use crate::linalg::Lu;
use crate::mechanics::{
    Lagrangian, NewtonMonitor, accelerations, max_abs, momentum, solve_failure,
    velocity_from_momentum,
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
    if h.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
        // `dt` is below the resolution of the clock at this time.
        return Err(SimError::StepSizeUnderflow { t: start.t });
    }

    let (end, p, increment, inc_dq) = if h <= TAYLOR_STEP {
        let increment: [f64; N] = from_fn(|i| h * (start.dq[i] + 0.5 * h * start.ddq[i]));
        let inc_dq: [f64; N] = from_fn(|i| h * start.ddq[i]);
        let q = from_fn(|i| start.q[i] + increment[i]);
        let dq = from_fn(|i| start.dq[i] + inc_dq[i]);
        let ddq = accelerations(&s.sys, q, dq, end_t)?;
        let end = State {
            t: end_t,
            q,
            dq,
            ddq,
        };
        (end, momentum(&s.sys, q, dq, end_t), increment, inc_dq)
    } else {
        let weights: &[f64] = if fourth_order { &[W1, W0, W1] } else { &[1.0] };
        let (mut q, mut p, mut t) = (start.q, s.p, start.t);
        let mut velocity = start.dq;
        // The whole step's position increment, summed from the substeps'
        // own: the segment is built from it, and the knot is the start plus
        // it, so the two agree to one rounding.
        let mut increment = [0.0; N];
        for &w in weights {
            let sub = w * h;
            let next = discrete_step(&s.sys, q, p, t, sub, velocity)?;
            for i in 0..N {
                increment[i] += next.increment[i];
            }
            (q, p, t, velocity) = (next.q, next.p, t + sub, next.velocity);
        }
        let q: [f64; N] = from_fn(|i| start.q[i] + increment[i]);

        // The knot needs a velocity and an acceleration; the integrator
        // carries a momentum. The inverse Legendre transform recovers the
        // first, and the equations of motion give the second.
        let dq = velocity_from_momentum(&s.sys, q, p, end_t, velocity)?;
        let ddq = accelerations(&s.sys, q, dq, end_t)?;
        let end = State {
            t: end_t,
            q,
            dq,
            ddq,
        };
        let inc_dq: [f64; N] = from_fn(|i| end.dq[i] - start.dq[i]);
        (end, p, increment, inc_dq)
    };
    if !end.is_finite() {
        return Err(SimError::NonFiniteState { t: start.t });
    }
    let segment = Segment::from_step(&start, &end, &increment, &inc_dq);
    if segment.defect().is_some() {
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
    Ok(segment)
}

/// The longest step taken as an explicit Taylor step instead of a
/// variational one: a landing step a few ulps long, or a `dt` far below any
/// physical time scale. Newton divides by the step, `1/h` overflows below
/// `h ≈ 5.6e-309`, and 0.1.0 returned an error or a wrong segment for such
/// steps (#7). The Taylor step's own error, `h³·q⃛/6 ≈ 7e-45·q⃛`, is far
/// below the rounding of any state. At `|t| ≥ 16` no step the clock can
/// resolve is this short, so every step there is variational.
const TAYLOR_STEP: f64 = 16.0 * f64::EPSILON;

/// Where one midpoint step ends.
struct Substep<const N: usize> {
    q: [f64; N],
    /// `q − q₀`, as solved for: not a difference of rounded positions.
    increment: [f64; N],
    p: [f64; N],
    /// The mean velocity over the step: the guess for the next one.
    velocity: [f64; N],
}

/// The residual of the discrete Euler-Lagrange equation and what is needed
/// to finish the step once it is small.
struct Residual<const N: usize> {
    /// `p₀ + D₁L_d + (h/2)·Q`.
    value: [f64; N],
    /// `∇_q L + Q` at the midpoint.
    force: [f64; N],
    /// The size of the terms that were summed into `value`: what its
    /// rounding error is relative to.
    scale: f64,
}

/// An update that shrinks the residual at least this much keeps the
/// Jacobian it was computed with; a slower one asks for a new one.
const KEEP_JACOBIAN: f64 = 0.25;

/// One midpoint step of size `h`, which may be negative.
///
/// Newton's unknown is the **increment** `d = q₁ − q₀`. The midpoint is
/// `q₀ + d/2` and the velocity `d/h`, so nothing is ever formed by
/// subtracting two stored positions: that subtraction loses `|q|/h` digits
/// and was measured to cost three orders of magnitude in momentum
/// conservation (`PLAN.md` §7).
///
/// The iteration is simplified Newton. A residual costs two gradient
/// passes of the Lagrangian and its Jacobian `2N` second-order ones, and
/// the Jacobian, `−M + O(h)`, barely moves while the increment converges.
/// So it is factored once and kept for as long as each update shrinks the
/// residual fourfold. 0.1.0 built a new one on every iteration, the last
/// included, where it was thrown away (#13).
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
    let mut factored: Option<Lu<N>> = None;
    let mut last = f64::INFINITY;

    for _ in 0..NewtonMonitor::MAX_ITERATIONS {
        let r = residual(sys, q, p, d, t_mid, h);
        if !r.value.iter().all(|x| x.is_finite()) {
            return Err(SimError::NonFiniteState { t });
        }
        let size = max_abs(&r.value);
        if monitor.done(size, r.scale) {
            // p₁ = D₂L_d + (h/2)·Q, rewritten with the equation just solved
            // as p₁ = p₀ + h·(∇_q L + Q). The two agree to the residual, and
            // this form leaves the momentum of a cyclic coordinate (one L
            // does not depend on) untouched to the last bit, which is the
            // discrete Noether theorem holding exactly rather than to
            // rounding.
            return Ok(Substep {
                q: from_fn(|i| q[i] + d[i]),
                increment: d,
                p: from_fn(|i| p[i] + h * r.force[i]),
                velocity: d.map(|x| x / h),
            });
        }
        let lu = match factored {
            Some(lu) if size <= KEEP_JACOBIAN * last => lu,
            _ => {
                let jacobian = jacobian(sys, q, d, t_mid, h);
                Lu::factor(&jacobian).ok_or_else(|| solve_failure(&jacobian, t))?
            }
        };
        factored = Some(lu);
        last = size;
        let update = lu.solve(&r.value.map(|x| -x));
        for i in 0..N {
            d[i] += update[i];
        }
    }
    Err(SimError::NewtonDidNotConverge { t })
}

/// The midpoint and the velocity of the step with increment `d`.
fn midpoint<const N: usize>(q: [f64; N], d: [f64; N], h: f64) -> ([f64; N], [f64; N]) {
    (from_fn(|i| q[i] + 0.5 * d[i]), d.map(|x| x / h))
}

/// The residual at increment `d`, from two gradient passes.
///
/// Write `L̃(q₀, d) = h·L(q₀ + d/2, d/h)`. Then `D₂L_d = ∂L̃/∂d` and
/// `D₁L_d = ∂L̃/∂q₀ − ∂L̃/∂d`.
fn residual<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    p: [f64; N],
    d: [f64; N],
    t_mid: f64,
    h: f64,
) -> Residual<N> {
    let (mid, vel) = midpoint(q, d, h);
    let time = Grad::<N>::constant(t_mid);
    // ∂L/∂d = ½·∇_q L + ∇_q̇ L / h
    let by_d = sys
        .lagrangian(
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
            time,
        )
        .d;
    // ∂L/∂q₀ = ∇_q L
    let by_q0 = sys
        .lagrangian(
            from_fn(|i| Grad::variable(mid[i], i)),
            vel.map(Grad::constant),
            time,
        )
        .d;
    let external = sys.generalized_force(mid, vel, t_mid);

    let mut value = [0.0; N];
    let mut force = [0.0; N];
    let mut scale = 0.0_f64;
    for i in 0..N {
        let conservative = h * (by_q0[i] - by_d[i]);
        let pushed = 0.5 * h * external[i];
        value[i] = p[i] + conservative + pushed;
        force[i] = by_q0[i] + external[i];
        scale = scale
            .max(crate::math::abs(p[i]))
            .max(crate::math::abs(h * by_q0[i]))
            .max(crate::math::abs(h * by_d[i]))
            .max(crate::math::abs(pushed));
    }
    Residual {
        value,
        force,
        scale,
    }
}

/// The residual's exact derivative with respect to `d`: the residual's two
/// gradients, each carried one derivative further along every `d_j`, `2N`
/// evaluations of the Lagrangian in all.
fn jacobian<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    d: [f64; N],
    t_mid: f64,
    h: f64,
) -> [[f64; N]; N] {
    let (mid, vel) = midpoint(q, d, h);
    let unit = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let time = Dual::constant(Grad::<N>::constant(t_mid));
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
        for i in 0..N {
            jacobian[i][j] = h * (along_q0.eps.d[i] - along_d.eps.d[i]);
        }
    }

    // The non-conservative force's derivative with respect to d.
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
    for i in 0..N {
        for j in 0..N {
            jacobian[i][j] += 0.5 * h * external[i].d[j];
        }
    }
    jacobian
}
