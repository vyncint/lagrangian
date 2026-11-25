//! One step of a trajectory as a polynomial that can be sampled anywhere
//! inside it.

/// A point of a trajectory: time, coordinates, velocities, accelerations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State<const N: usize> {
    /// Time.
    pub t: f64,
    /// Generalized coordinates.
    pub q: [f64; N],
    /// Generalized velocities.
    pub dq: [f64; N],
    /// Generalized accelerations.
    pub ddq: [f64; N],
}

impl<const N: usize> State<N> {
    /// `true` when every number in the state is finite.
    pub fn is_finite(&self) -> bool {
        self.t.is_finite()
            && self.q.iter().all(|x| x.is_finite())
            && self.dq.iter().all(|x| x.is_finite())
            && self.ddq.iter().all(|x| x.is_finite())
    }
}

/// One accepted integration step as a polynomial per coordinate.
///
/// The polynomial matches position, velocity **and acceleration** at both
/// ends of the step, so consecutive segments join with a continuous second
/// derivative, and the velocity and acceleration it reports anywhere inside
/// are exactly the first and second derivatives of the position it reports.
///
/// `Copy`, `6 N + 2` numbers, no allocation: the latest segment of a
/// [`Stepper`](crate::Stepper) can be sampled on a target with no heap.
///
/// # Exact at its start
///
/// The polynomial is stored in powers of the time since the start of the
/// step, so its first three coefficients are the starting position,
/// velocity and half the starting acceleration themselves.
/// [`start`](Segment::start) therefore returns the solver's knot bit for
/// bit, with one exception: an acceleration below `2·f64::MIN_POSITIVE`
/// in magnitude, about `4.5e-308`, has a subnormal half and can lose its
/// last bit to the halving. The
/// end of a segment agrees with the start of the next to rounding
/// (`PLAN.md` §7).
///
/// # Accuracy between the ends
///
/// The step-size controller bounds the error at the ends of a step. It does
/// **not** bound the error of this polynomial in between: the velocity it
/// reports mid-step carries its own `O(h⁵)` truncation error. On a chaotic
/// double pendulum that was measured at around twenty times the error at
/// the ends (`PLAN.md` §7). Tighten the tolerance if mid-step accuracy
/// matters.
///
/// # Very short steps
///
/// For an ordinary step the polynomial is the quintic through position,
/// velocity and acceleration at both ends. A step can be far shorter than
/// the solver would choose, because
/// [`Stepper::step_bounded`](crate::Stepper::step_bounded) lands on its
/// target however close it is. The ends of such a step differ by little
/// more than their rounding, which the quintic would divide by powers of the
/// step on its way to the acceleration. Instead the segment is a cubic
/// through the start and the end acceleration, or a quartic through the
/// velocities and accelerations at both ends, whichever reaches the end to
/// rounding, and the start's Taylor quadratic if no polynomial can be
/// represented at all. For steps from one ulp of the time to `1e-4`, the
/// acceleration inside agrees with the equations of motion to `4e-11`
/// (`short_steps_agree_with_the_equations_of_motion_inside`, in
/// `tests/stepper.rs`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment<const N: usize> {
    pub(crate) t0: f64,
    pub(crate) t1: f64,
    /// `q(t) = Σ c[i][k] · (t − t0)ᵏ`.
    pub(crate) c: [[f64; 6]; N],
}

/// The six coefficients of one coordinate over a step of length `h`, in
/// powers of the time since its start.
///
/// `inc_q` and `inc_v` are the step's position and velocity increments. A
/// solver passes them as it computed them, before adding them to the start
/// knot: they then carry the rounding of the step rather than of the state.
///
/// Every rate below is formed by successive division, so no power of `h` is
/// ever formed: `h⁵` underflows to zero below `h ≈ 1e-62`, and 0.1.0 divided
/// by it (#7). The candidates are tried from the one with the least noise:
///
/// 1. the cubic through the start knot and the end acceleration, which uses
///    no difference of positions or velocities at all;
/// 2. the quartic through the velocities and accelerations at both ends,
///    which loses nothing to the rounding of positions;
/// 3. the quintic through everything, for ordinary steps.
///
/// The first two are used only where they reach the end knot to rounding
/// anyway, which is exactly where the quintic's divisions by `h` turn that
/// rounding into noise.
#[allow(clippy::too_many_arguments)]
fn coefficients(
    h: f64,
    q0: f64,
    v0: f64,
    a0: f64,
    inc_q: f64,
    inc_v: f64,
    q1: f64,
    v1: f64,
    a1: f64,
) -> [f64; 6] {
    let taylor = [q0, v0, 0.5 * a0, 0.0, 0.0, 0.0];
    if !(h > 0.0 && h.is_finite()) {
        return taylor;
    }
    let q_rounding = 2.0 * f64::EPSILON * q0.abs().max(q1.abs()).max(inc_q.abs());
    let v_rounding = 4.0 * f64::EPSILON * v0.abs().max(v1.abs()).max(inc_v.abs());
    // The end acceleration's change, as a rate in units of q per time cubed.
    let a = (a1 - a0) / h;

    let c3 = a / 6.0;
    let reach_q = h * (v0 + h * (0.5 * a0 + h * c3));
    let reach_v = h * (a0 + 3.0 * h * c3);
    if c3.is_finite()
        && (reach_q - inc_q).abs() <= q_rounding
        && (reach_v - inc_v).abs() <= v_rounding
    {
        return [q0, v0, 0.5 * a0, c3, 0.0, 0.0];
    }

    let v = (inc_v - h * a0) / h / h;
    let c3 = v - a / 3.0;
    let c4 = (a - 2.0 * v) / 4.0 / h;
    let reach = h * (v0 + h * (0.5 * a0 + h * (c3 + h * c4)));
    if c3.is_finite() && c4.is_finite() && (reach - inc_q).abs() <= q_rounding {
        return [q0, v0, 0.5 * a0, c3, c4, 0.0];
    }

    let p = (inc_q - h * (v0 + 0.5 * h * a0)) / h / h / h;
    let quintic = [
        q0,
        v0,
        0.5 * a0,
        10.0 * p - 4.0 * v + 0.5 * a,
        (-15.0 * p + 7.0 * v - a) / h,
        (6.0 * p - 3.0 * v + 0.5 * a) / h / h,
    ];
    if quintic.iter().all(|c| c.is_finite()) {
        quintic
    } else {
        taylor
    }
}

impl<const N: usize> Segment<N> {
    /// The polynomial through `start` and `end`, matching `q`, `dq` and
    /// `ddq` at both (to rounding; see [Very short steps](Segment#very-short-steps)).
    ///
    /// # Panics
    ///
    /// In debug builds, if `end.t` is not after `start.t`.
    pub fn from_knots(start: &State<N>, end: &State<N>) -> Self {
        debug_assert!(end.t > start.t, "a segment must run forward in time");
        let inc_q = core::array::from_fn(|i| end.q[i] - start.q[i]);
        let inc_dq = core::array::from_fn(|i| end.dq[i] - start.dq[i]);
        Self::from_step(start, end, &inc_q, &inc_dq)
    }

    /// The same, from the increments a solver added to the start knot.
    pub(crate) fn from_step(
        start: &State<N>,
        end: &State<N>,
        inc_q: &[f64; N],
        inc_dq: &[f64; N],
    ) -> Self {
        let h = end.t - start.t;
        let c = core::array::from_fn(|i| {
            coefficients(
                h,
                start.q[i],
                start.dq[i],
                start.ddq[i],
                inc_q[i],
                inc_dq[i],
                end.q[i],
                end.dq[i],
                end.ddq[i],
            )
        });
        Self {
            t0: start.t,
            t1: end.t,
            c,
        }
    }

    /// Why this segment could not have come from a solver, if it could not:
    /// times that are not finite, a duration that is not positive and
    /// finite, or a polynomial that can overflow somewhere inside.
    ///
    /// The overflow test bounds `|q|`, `|q'|` and `|q''|` over the whole
    /// segment by evaluating each with the absolute values of the
    /// coefficients at the end. If the bounds are finite, every sample
    /// inside is.
    pub(crate) fn defect(&self) -> Option<&'static str> {
        if !(self.t0.is_finite() && self.t1.is_finite()) {
            return Some("a segment's start and end times must be finite");
        }
        let h = self.t1 - self.t0;
        if h.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
            return Some("a segment must run forward in time");
        }
        if !h.is_finite() {
            return Some("a segment's duration must be finite");
        }
        for c in &self.c {
            let c = c.map(f64::abs);
            let q = c[0] + h * (c[1] + h * (c[2] + h * (c[3] + h * (c[4] + h * c[5]))));
            let v = c[1] + h * (2.0 * c[2] + h * (3.0 * c[3] + h * (4.0 * c[4] + h * 5.0 * c[5])));
            let a = 2.0 * c[2] + h * (6.0 * c[3] + h * (12.0 * c[4] + h * 20.0 * c[5]));
            if !(q.is_finite() && v.is_finite() && a.is_finite()) {
                return Some("a segment's polynomial must stay finite between its ends");
            }
        }
        None
    }

    /// The time at which the segment starts.
    #[inline]
    pub fn t_start(&self) -> f64 {
        self.t0
    }

    /// The time at which the segment ends.
    #[inline]
    pub fn t_end(&self) -> f64 {
        self.t1
    }

    /// The length of the step.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.t1 - self.t0
    }

    /// The six polynomial coefficients of each coordinate, lowest order
    /// first, in powers of the time since [`t_start`](Segment::t_start):
    /// `q(t) = Σ c[k] · (t − t_start)ᵏ`.
    #[inline]
    pub fn coefficients(&self) -> &[[f64; 6]; N] {
        &self.c
    }

    /// The state a time `s` after the start of the segment, stamped `t`.
    fn at_offset(&self, s: f64, t: f64) -> State<N> {
        let mut q = [0.0; N];
        let mut dq = [0.0; N];
        let mut ddq = [0.0; N];
        for (i, c) in self.c.iter().enumerate() {
            q[i] = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))));
            dq[i] = c[1] + s * (2.0 * c[2] + s * (3.0 * c[3] + s * (4.0 * c[4] + s * 5.0 * c[5])));
            ddq[i] = 2.0 * c[2] + s * (6.0 * c[3] + s * (12.0 * c[4] + s * 20.0 * c[5]));
        }
        State { t, q, dq, ddq }
    }

    /// The state at normalized position `tau`, where `0` is the start of the
    /// segment and `1` its end. Not clamped: a value outside `[0, 1]`
    /// extrapolates the polynomial.
    pub fn eval(&self, tau: f64) -> State<N> {
        let s = tau * (self.t1 - self.t0);
        self.at_offset(s, self.t0 + s)
    }

    /// The state at time `t`, clamped to the segment.
    pub fn sample(&self, t: f64) -> State<N> {
        if t <= self.t0 {
            return self.start();
        }
        if t >= self.t1 {
            return self.end();
        }
        self.at_offset(t - self.t0, t)
    }

    /// The state at the start of the segment: the knot the step began from,
    /// exactly.
    pub fn start(&self) -> State<N> {
        self.at_offset(0.0, self.t0)
    }

    /// The state at the end of the segment.
    pub fn end(&self) -> State<N> {
        self.at_offset(self.t1 - self.t0, self.t1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knots() -> (State<2>, State<2>) {
        (
            State {
                t: 1.0,
                q: [0.3, -1.2],
                dq: [2.0, 0.5],
                ddq: [-4.0, 3.0],
            },
            State {
                t: 1.25,
                q: [0.7, -1.0],
                dq: [1.1, 1.4],
                ddq: [-2.5, 0.2],
            },
        )
    }

    #[test]
    fn the_quintic_reproduces_both_knots() {
        let (a, b) = knots();
        let seg = Segment::from_knots(&a, &b);
        // Bit for bit: the low coefficients are the knot.
        assert_eq!(seg.start(), a);
        assert_eq!(seg.eval(0.0), a);
        let end = seg.end();
        assert_eq!(end.t, b.t);
        for i in 0..2 {
            assert!((end.q[i] - b.q[i]).abs() <= 1e-15);
            assert!((end.dq[i] - b.dq[i]).abs() <= 1e-13);
            assert!((end.ddq[i] - b.ddq[i]).abs() <= 1e-11);
        }
    }

    #[test]
    fn the_start_is_the_knot_bit_for_bit() {
        // Over many awkward step sizes and values: a representation that
        // scales the velocity by the step and back is right for most of
        // them and one rounding off for the rest.
        let mut seed = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for _ in 0..2000 {
            let h = 10f64.powf(3.0 * next() - 3.0);
            let start = State {
                t: 10.0 * next(),
                q: [7.0 * next(), 0.3 * next()],
                dq: [5.0 * next(), 40.0 * next()],
                ddq: [90.0 * next(), 0.01 * next()],
            };
            let end = State {
                t: start.t + h,
                q: [start.q[0] + h * next(), start.q[1] + h * next()],
                dq: [5.0 * next(), 40.0 * next()],
                ddq: [90.0 * next(), 0.01 * next()],
            };
            let seg = Segment::from_knots(&start, &end);
            assert_eq!(seg.start(), start, "h = {h}");
            assert_eq!(seg.sample(start.t), start);
        }
    }

    #[test]
    fn a_quintic_motion_is_reproduced_exactly() {
        // q(t) = t⁵ − 2t³ + t: the interpolant has the freedom to match it.
        let f = |t: f64| t.powi(5) - 2.0 * t.powi(3) + t;
        let df = |t: f64| 5.0 * t.powi(4) - 6.0 * t * t + 1.0;
        let ddf = |t: f64| 20.0 * t.powi(3) - 12.0 * t;
        let at = |t: f64| State {
            t,
            q: [f(t)],
            dq: [df(t)],
            ddq: [ddf(t)],
        };
        let seg = Segment::from_knots(&at(0.2), &at(0.9));
        for k in 0..=20 {
            let t = 0.2 + 0.7 * k as f64 / 20.0;
            let s = seg.sample(t);
            assert!((s.q[0] - f(t)).abs() <= 1e-14);
            assert!((s.dq[0] - df(t)).abs() <= 1e-13);
            assert!((s.ddq[0] - ddf(t)).abs() <= 1e-12);
        }
    }

    #[test]
    fn velocity_and_acceleration_are_the_derivatives_of_position() {
        let (a, b) = knots();
        let seg = Segment::from_knots(&a, &b);
        let h = 1e-5;
        for &t in &[1.03, 1.1, 1.2] {
            let s = seg.sample(t);
            let (lo, hi) = (seg.sample(t - h), seg.sample(t + h));
            for i in 0..2 {
                let v = (hi.q[i] - lo.q[i]) / (2.0 * h);
                let acc = (hi.dq[i] - lo.dq[i]) / (2.0 * h);
                assert!((v - s.dq[i]).abs() <= 1e-7 * (1.0 + s.dq[i].abs()));
                assert!((acc - s.ddq[i]).abs() <= 1e-6 * (1.0 + s.ddq[i].abs()));
            }
        }
    }

    #[test]
    fn sampling_clamps_outside_the_segment() {
        let (a, b) = knots();
        let seg = Segment::from_knots(&a, &b);
        assert_eq!(seg.sample(0.0), seg.start());
        assert_eq!(seg.sample(99.0), seg.end());
        assert_eq!(seg.duration(), 0.25);
    }
}
