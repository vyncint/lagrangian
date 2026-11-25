//! Dormand-Prince 5(4) with first-same-as-last and a PI step controller.

use super::Stepper;
use crate::error::SimError;
use crate::math;
use crate::mechanics::{Lagrangian, accelerations};
use crate::segment::{Segment, State};

/// Stage times.
const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

/// Stage weights. The last row is also the fifth-order solution, which is
/// what makes the seventh stage of an accepted step the first of the next.
const A: [[f64; 6]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

/// Fifth-order weights minus fourth-order weights: the error estimate.
const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

// The controller of Hairer, Nørsett & Wanner's DOPRI5.
const SAFETY: f64 = 0.9;
const BETA: f64 = 0.04;
const EXPO: f64 = 0.2 - BETA * 0.75;
/// The step may shrink to a fifth…
const SHRINK_LIMIT: f64 = 5.0;
/// …and grow tenfold.
const GROW_LIMIT: f64 = 0.1;

/// One accepted step, retrying rejected ones.
pub(super) fn step<L: Lagrangian<N>, const N: usize>(
    s: &mut Stepper<L, N>,
    t_max: Option<f64>,
) -> Result<Segment<N>, SimError> {
    let cfg = s.config;
    let start = s.state;
    let t = start.t;

    if s.h == 0.0 {
        s.h = match cfg.initial_step {
            Some(h) => h,
            None => initial_step(s, t_max),
        };
    }

    // Below this a step no longer moves the clock by a meaningful amount.
    let floor = cfg
        .min_step
        .max(16.0 * f64::EPSILON * math::abs(t).max(1.0));

    let mut h = s.h;
    let mut after_rejection = false;
    let mut failure: Option<SimError> = None;

    loop {
        let mut h_try = h.min(cfg.max_step);
        // Land on t_max when it is within a hundredth of a step, rather
        // than leave a sliver for the next call.
        let lands = matches!(t_max, Some(tm) if t + 1.01 * h_try >= tm);
        if let (true, Some(tm)) = (lands, t_max) {
            h_try = tm - t;
        } else if h_try.partial_cmp(&floor) != Some(core::cmp::Ordering::Greater) {
            return Err(failure.unwrap_or(SimError::StepSizeUnderflow { t }));
        }

        match attempt(s, &start, h_try) {
            Ok((mut end, err)) if err <= 1.0 => {
                if let (true, Some(tm)) = (lands, t_max) {
                    end.t = tm;
                }
                let scaled = math::powf(err.max(1e-10), EXPO) / math::powf(s.err_prev, BETA);
                let factor = (scaled / SAFETY).clamp(GROW_LIMIT, SHRINK_LIMIT);
                let mut h_next = h_try / factor;
                if after_rejection {
                    h_next = h_next.min(h_try);
                }
                // A step shortened to land says nothing about the step the
                // dynamics would have accepted.
                s.h = if lands && h_try < h {
                    h_next.max(h)
                } else {
                    h_next
                };
                s.err_prev = err.max(1e-4);
                s.state = end;
                return Ok(Segment::from_knots(&start, &end));
            }
            Ok((_, err)) if err.is_finite() => {
                let factor = (math::powf(err, EXPO) / SAFETY).min(SHRINK_LIMIT);
                h = h_try / factor;
            }
            Ok(_) => {
                // The error estimate itself is not a number.
                failure = Some(SimError::NonFiniteState { t });
                h = h_try / SHRINK_LIMIT;
            }
            Err(e) => {
                // A stage could not be evaluated. A shorter step may avoid
                // whatever it ran into; if none does, this is the error.
                failure = Some(e);
                h = h_try / SHRINK_LIMIT;
            }
        }
        // Every branch above shrinks the step, so the loop ends: either a
        // step is accepted or `h` falls to the floor.
        after_rejection = true;
    }
}

/// One trial step of size `h` from `start`: the end state, and the scaled
/// error estimate (accept at or below one).
fn attempt<L: Lagrangian<N>, const N: usize>(
    s: &Stepper<L, N>,
    start: &State<N>,
    h: f64,
) -> Result<(State<N>, f64), SimError> {
    // The derivative of (q, q̇) at each stage is (kq, kv).
    let mut kq = [[0.0; N]; 7];
    let mut kv = [[0.0; N]; 7];
    kq[0] = start.dq;
    kv[0] = start.ddq;

    let mut q = start.q;
    let mut dq = start.dq;
    for stage in 1..7 {
        for i in 0..N {
            let mut sq = 0.0;
            let mut sv = 0.0;
            for j in 0..stage {
                sq += A[stage][j] * kq[j][i];
                sv += A[stage][j] * kv[j][i];
            }
            q[i] = start.q[i] + h * sq;
            dq[i] = start.dq[i] + h * sv;
        }
        kq[stage] = dq;
        kv[stage] = accelerations(&s.sys, q, dq, start.t + C[stage] * h)?;
    }

    // After the seventh stage (q, dq) is the fifth-order solution and
    // kv[6] its acceleration: the next step's first stage, for free.
    let mut sum = 0.0;
    for i in 0..N {
        let mut eq = 0.0;
        let mut ev = 0.0;
        for j in 0..7 {
            eq += E[j] * kq[j][i];
            ev += E[j] * kv[j][i];
        }
        let scale_q = s.config.atol + s.config.rtol * math::abs(start.q[i]).max(math::abs(q[i]));
        let scale_v = s.config.atol + s.config.rtol * math::abs(start.dq[i]).max(math::abs(dq[i]));
        let rq = h * eq / scale_q;
        let rv = h * ev / scale_v;
        sum += rq * rq + rv * rv;
    }
    let err = if N == 0 {
        0.0
    } else {
        math::sqrt(sum / (2 * N) as f64)
    };

    let end = State {
        t: start.t + h,
        q,
        dq,
        ddq: kv[6],
    };
    if !end.is_finite() {
        return Err(SimError::NonFiniteState { t: start.t });
    }
    Ok((end, err))
}

/// A first step from the size of the state and of its derivatives, after
/// Hairer, Nørsett & Wanner (II.4). One extra evaluation.
fn initial_step<L: Lagrangian<N>, const N: usize>(s: &Stepper<L, N>, t_max: Option<f64>) -> f64 {
    let cfg = &s.config;
    let st = &s.state;
    let cap = match t_max {
        Some(tm) => cfg.max_step.min(tm - st.t),
        None => cfg.max_step,
    };
    if N == 0 {
        return cap.min(1.0);
    }

    let scale = |y: f64| cfg.atol + cfg.rtol * math::abs(y);
    let rms = |sum: f64| math::sqrt(sum / (2 * N) as f64);

    let (mut d0, mut d1) = (0.0, 0.0);
    for i in 0..N {
        let (sq, sv) = (scale(st.q[i]), scale(st.dq[i]));
        d0 += (st.q[i] / sq) * (st.q[i] / sq) + (st.dq[i] / sv) * (st.dq[i] / sv);
        d1 += (st.dq[i] / sq) * (st.dq[i] / sq) + (st.ddq[i] / sv) * (st.ddq[i] / sv);
    }
    let (d0, d1) = (rms(d0), rms(d1));
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    }
    .min(cap);

    // One explicit Euler step, to see how fast the derivative changes.
    let q1: [f64; N] = core::array::from_fn(|i| st.q[i] + h0 * st.dq[i]);
    let dq1: [f64; N] = core::array::from_fn(|i| st.dq[i] + h0 * st.ddq[i]);
    let Ok(ddq1) = accelerations(&s.sys, q1, dq1, st.t + h0) else {
        return h0;
    };
    let mut d2 = 0.0;
    for i in 0..N {
        let (sq, sv) = (scale(st.q[i]), scale(st.dq[i]));
        let (aq, av) = ((dq1[i] - st.dq[i]) / sq, (ddq1[i] - st.ddq[i]) / sv);
        d2 += aq * aq + av * av;
    }
    let d2 = rms(d2) / h0;

    let largest = d1.max(d2);
    let h1 = if largest <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        math::powf(0.01 / largest, 0.2)
    };
    let h = (100.0 * h0).min(h1).min(cap);
    if h > 0.0 && h.is_finite() { h } else { h0 }
}
