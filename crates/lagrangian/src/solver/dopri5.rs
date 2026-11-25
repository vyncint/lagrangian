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

/// Attempts one call may make before it gives up with
/// [`SimError::StepSizeUnderflow`]. A rejection divides the trial step by
/// between 1.1 and 5, depending on its error, and a step at the floor ends
/// the call, so an ordinary step needs a handful; even a fall from
/// `f64::MAX` to the floor by fivefold cuts takes fewer than 470. The bound
/// is there so that no input, however its error estimates behave, can keep
/// the loop turning (#4).
const MAX_ATTEMPTS: u32 = 1000;

/// One trial step: where it ends, how wrong it may be, and the increments it
/// added to the start knot as they were computed, before being added. The
/// segment is built from the increments, which carry the rounding of the
/// step rather than of the positions.
struct Trial<const N: usize> {
    end: State<N>,
    err: f64,
    inc_q: [f64; N],
    inc_dq: [f64; N],
}

/// One accepted step, retrying rejected ones.
pub(super) fn step<L: Lagrangian<N>, const N: usize>(
    s: &mut Stepper<L, N>,
    t_max: Option<f64>,
) -> Result<Segment<N>, SimError> {
    let cfg = s.config;
    let start = s.state;
    let t = start.t;
    // The longest step the arithmetic can take. With the default unlimited
    // `max_step` a step that grows tenfold with every acceptance otherwise
    // reaches infinity after a few hundred steps at rest, and no rejection
    // can shrink infinity (#4).
    let max_step = cfg.max_step.min(f64::MAX);

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
    let remainder = t_max.map(|tm| tm - t);

    let mut h = s.h;
    let mut after_rejection = false;
    let mut failure: Option<SimError> = None;

    for _ in 0..MAX_ATTEMPTS {
        let mut h_try = h.min(max_step);
        // Land on t_max when it is within a hundredth of a step, rather than
        // leave a sliver for the next call. Lengths are compared, not times:
        // `t + 1.01·h >= t_max` rounds, and a few ulps short of t_max a step
        // shrunk after a rejection rounded back up and landed again, for
        // ever (#4). A landing step may be shorter than the floor, so that a
        // target just ahead can still be reached; once rejected, the shorter
        // steps that follow cannot land.
        let lands = matches!(remainder, Some(rem) if 1.01 * h_try >= rem);
        if let (true, Some(rem)) = (lands, remainder) {
            h_try = rem;
        } else if h_try.partial_cmp(&floor) != Some(core::cmp::Ordering::Greater) {
            return Err(failure.unwrap_or(SimError::StepSizeUnderflow { t }));
        }

        match attempt(s, &start, h_try) {
            Ok(trial) if trial.err <= 1.0 => {
                let mut end = trial.end;
                if let (true, Some(tm)) = (lands, t_max) {
                    end.t = tm;
                }
                let segment = Segment::from_step(&start, &end, &trial.inc_q, &trial.inc_dq);
                if segment.defect().is_some() {
                    // Accurate at its ends, but a polynomial that overflows
                    // in between: as unusable as a failed evaluation, and as
                    // likely to be cured by a shorter step.
                    failure = Some(SimError::NonFiniteState { t });
                    h = h_try / SHRINK_LIMIT;
                    after_rejection = true;
                    continue;
                }
                let scaled = math::powf(trial.err.max(1e-10), EXPO) / math::powf(s.err_prev, BETA);
                let factor = (scaled / SAFETY).clamp(GROW_LIMIT, SHRINK_LIMIT);
                let mut h_next = h_try / factor;
                if after_rejection {
                    h_next = h_next.min(h_try);
                }
                // A step shortened to land says nothing about the step the
                // dynamics would have accepted.
                let next = if lands && h_try < h {
                    h_next.max(h)
                } else {
                    h_next
                };
                s.h = next.min(max_step);
                s.err_prev = trial.err.max(1e-4);
                s.state = end;
                return Ok(segment);
            }
            Ok(trial) if trial.err.is_finite() => {
                let factor = (math::powf(trial.err, EXPO) / SAFETY).min(SHRINK_LIMIT);
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
        after_rejection = true;
    }
    Err(failure.unwrap_or(SimError::StepSizeUnderflow { t }))
}

/// One trial step of size `h` from `start`: the end state, the scaled error
/// estimate (accept at or below one), and the increments.
fn attempt<L: Lagrangian<N>, const N: usize>(
    s: &Stepper<L, N>,
    start: &State<N>,
    h: f64,
) -> Result<Trial<N>, SimError> {
    // The derivative of (q, q̇) at each stage is (kq, kv).
    let mut kq = [[0.0; N]; 7];
    let mut kv = [[0.0; N]; 7];
    kq[0] = start.dq;
    kv[0] = start.ddq;

    let mut q = start.q;
    let mut dq = start.dq;
    let mut inc_q = [0.0; N];
    let mut inc_dq = [0.0; N];
    for stage in 1..7 {
        for i in 0..N {
            let mut sq = 0.0;
            let mut sv = 0.0;
            for j in 0..stage {
                sq += A[stage][j] * kq[j][i];
                sv += A[stage][j] * kv[j][i];
            }
            inc_q[i] = h * sq;
            inc_dq[i] = h * sv;
            q[i] = start.q[i] + inc_q[i];
            dq[i] = start.dq[i] + inc_dq[i];
        }
        kq[stage] = dq;
        kv[stage] = accelerations(&s.sys, q, dq, start.t + C[stage] * h)?;
    }

    // After the seventh stage (q, dq) is the fifth-order solution, its
    // increments are the last ones computed, and kv[6] is its acceleration:
    // the next step's first stage, for free.
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
    Ok(Trial {
        end,
        err,
        inc_q,
        inc_dq,
    })
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
    // With extreme tolerances the squares above overflow and the ratio is
    // inf / inf. `NaN.min(cap)` is `cap`, which is infinite by default, and
    // an infinite first step is the hang of #4.
    let ratio = 0.01 * d0 / d1;
    let h0 = if d0 < 1e-5 || d1 < 1e-5 || !ratio.is_finite() {
        1e-6
    } else {
        ratio
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
