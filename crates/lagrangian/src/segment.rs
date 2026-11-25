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

/// One accepted integration step as a quintic Hermite polynomial per
/// coordinate.
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
/// bit. The end of a segment agrees with the start of the next to rounding
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment<const N: usize> {
    pub(crate) t0: f64,
    pub(crate) t1: f64,
    /// `q(t) = Σ c[i][k] · (t − t0)ᵏ`.
    pub(crate) c: [[f64; 6]; N],
}

impl<const N: usize> Segment<N> {
    /// The quintic through `start` and `end`, matching `q`, `dq` and `ddq`
    /// at both.
    ///
    /// # Panics
    ///
    /// In debug builds, if `end.t` is not after `start.t`.
    pub fn from_knots(start: &State<N>, end: &State<N>) -> Self {
        let h = end.t - start.t;
        debug_assert!(h > 0.0, "a segment must run forward in time");
        let (h3, h4, h5) = (h * h * h, h * h * h * h, h * h * h * h * h);
        let c = core::array::from_fn(|i| {
            // In the normalized variable τ = (t − t0)/h the low coefficients
            // are q, h·q̇ and ½h²·q̈ …
            let c1 = h * start.dq[i];
            let c2 = 0.5 * h * h * start.ddq[i];
            // … and this is what is left for the three high ones to account
            // for at τ = 1, in position, velocity and acceleration.
            let d = end.q[i] - start.q[i] - c1 - c2;
            let dv = h * end.dq[i] - c1 - 2.0 * c2;
            let da = h * h * end.ddq[i] - 2.0 * c2;
            // Stored in powers of (t − t0): the low three are then the knot
            // itself, with no rounding from scaling by h and back.
            [
                start.q[i],
                start.dq[i],
                0.5 * start.ddq[i],
                (10.0 * d - 4.0 * dv + 0.5 * da) / h3,
                (-15.0 * d + 7.0 * dv - da) / h4,
                (6.0 * d - 3.0 * dv + 0.5 * da) / h5,
            ]
        });
        Self {
            t0: start.t,
            t1: end.t,
            c,
        }
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
