//! A whole simulation as one continuous, scrubbable function of time.

use alloc::vec::Vec;

use crate::error::SimError;
use crate::math;
use crate::mechanics::{self, Energy, Kinematics, Lagrangian, SceneMap};
use crate::roots;
use crate::segment::{Segment, State};
use crate::solver::{SimConfig, Stepper};

/// The most samples [`Trajectory::energy_drift`] takes.
const MAX_DRIFT_SAMPLES: usize = 1 << 20;

/// The most interior points [`Trajectory::crossings_with`] looks at per step.
const MAX_INTERIOR_SAMPLES: usize = 1 << 16;

/// Which way an indicator must pass through zero to count as a crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrossingDirection {
    /// From negative to positive.
    Rising,
    /// From positive to negative.
    Falling,
    /// Either way.
    Either,
}

/// A position in a [`Trajectory`], remembered between samples so that
/// playing it forward or backward does not search from scratch each frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cursor {
    index: usize,
}

/// How well a trajectory conserved its energy.
///
/// The energy is the Jacobi integral `E = p · q̇ − L`, so this is meaningful
/// in rotating frames and magnetic fields as well.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyReport {
    /// The energy at the start.
    pub initial: f64,
    /// The smallest energy seen.
    pub min: f64,
    /// The largest energy seen.
    pub max: f64,
    /// What the relative drift is relative to: the larger of `|initial|` and
    /// the largest kinetic energy seen. Never the initial energy alone,
    /// which is exactly zero for a pendulum released from horizontal.
    pub scale: f64,
    /// The largest `|E − initial|`.
    pub max_abs_drift: f64,
    /// `max_abs_drift / scale`.
    pub max_rel_drift: f64,
    /// The root mean square of `E − initial`.
    pub rms_drift: f64,
}

/// The solution of the equations of motion over an interval, as a
/// `C²`-smooth function of time.
///
/// A trajectory is a sequence of [`Segment`]s, one per accepted step. It can
/// be sampled at any time in any order: forward, backward, looped, or by a
/// slider. Position, velocity and acceleration are all continuous across the
/// joins, and the velocity and acceleration it reports are exactly the first
/// and second derivatives of the position it reports.
///
/// ```
/// use lagrangian::prelude::*;
/// use lagrangian::systems::DoublePendulum;
///
/// let pendulum = DoublePendulum::default();
/// let traj = simulate(&pendulum, (0.0, 5.0), [2.0, 2.5], [0.0, 0.0], &SimConfig::default())?;
///
/// let a = traj.sample(1.234);            // any time, any order
/// let b = traj.eval_normalized(0.5);     // halfway, for an animation timeline
/// assert_eq!(b.t, 2.5);
/// assert!(a.q[0].is_finite());
///
/// let drift = traj.energy_drift_at_knots(&pendulum);
/// assert!(drift.max_rel_drift < 1e-7);
/// # Ok::<(), SimError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<const N: usize> {
    pub(crate) initial: State<N>,
    pub(crate) segments: Vec<Segment<N>>,
}

/// Integrates `sys` from `t_span.0` to `t_span.1` and returns the whole
/// solution.
///
/// This is a loop over [`Stepper::step_bounded`] that collects the segments;
/// nothing here is a separate code path.
///
/// # Memory
///
/// One [`Segment`] is kept per accepted step, so a call holds at most
/// [`SimConfig::max_steps`] `× size_of::<Segment<N>>()` bytes, `8·(6N + 2)`
/// per segment: 107 MiB for a double pendulum with the default budget of a
/// million steps, before it gives up with
/// [`MaxStepsExceeded`](SimError::MaxStepsExceeded). Lower the budget when
/// the span or the system comes from someone else.
///
/// # Errors
///
/// [`SimError::InvalidConfig`] if either end of the span is not finite or
/// the span runs backward, [`SimError::MaxStepsExceeded`] if it needs more
/// than [`SimConfig::max_steps`] steps, and anything [`Stepper::step`]
/// reports.
pub fn simulate<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    t_span: (f64, f64),
    q0: [f64; N],
    dq0: [f64; N],
    config: &SimConfig,
) -> Result<Trajectory<N>, SimError> {
    let (t0, t1) = t_span;
    // An infinite span can only end in `MaxStepsExceeded`, after a million
    // steps and their segments, or, at rest, in a step that never returned
    // (#4, #5).
    if !(t0.is_finite() && t1.is_finite()) {
        return Err(SimError::InvalidConfig {
            reason: "the time span must be finite",
        });
    }
    if t1
        .partial_cmp(&t0)
        .is_none_or(|o| o == core::cmp::Ordering::Less)
    {
        return Err(SimError::InvalidConfig {
            reason: "the time span must not run backward",
        });
    }
    let mut stepper = Stepper::new(sys, t0, q0, dq0, *config)?;
    let mut trajectory = Trajectory::new(stepper.state());
    trajectory.extend(&mut stepper, t1)?;
    Ok(trajectory)
}

impl<const N: usize> Trajectory<N> {
    /// An empty trajectory that starts, and for now ends, at `initial`.
    pub fn new(initial: State<N>) -> Self {
        Self {
            initial,
            segments: Vec::new(),
        }
    }

    /// Appends a segment.
    ///
    /// # Errors
    ///
    /// [`SimError::InvalidConfig`] if the segment does not start at the time
    /// the trajectory ends, does not run forward for a finite time, or has a
    /// polynomial that overflows between its ends.
    pub fn push(&mut self, segment: Segment<N>) -> Result<(), SimError> {
        if let Some(reason) = segment.defect() {
            return Err(SimError::InvalidConfig { reason });
        }
        if segment.t0 != self.t_end() {
            return Err(SimError::InvalidConfig {
                reason: "a segment must start where the trajectory ends",
            });
        }
        self.segments.push(segment);
        Ok(())
    }

    /// Steps `stepper` until `t_until`, appending every segment: how an
    /// interactive application simulates ahead a little at a time.
    ///
    /// With a fixed-step solver, pieces that end on step boundaries add up
    /// to exactly the trajectory one call would have produced. With the
    /// adaptive solver each piece ends with a step shortened to land on
    /// `t_until`, so the steps differ, though both are within tolerance.
    ///
    /// # Errors
    ///
    /// [`SimError::InvalidConfig`] if `t_until` is not finite or the stepper
    /// is not at the end of this trajectory, [`SimError::MaxStepsExceeded`]
    /// if this call needs more than [`SimConfig::max_steps`] steps, and
    /// anything [`Stepper::step`] reports. Segments appended before an error
    /// stay.
    pub fn extend<L: Lagrangian<N>>(
        &mut self,
        stepper: &mut Stepper<L, N>,
        t_until: f64,
    ) -> Result<(), SimError> {
        if !t_until.is_finite() {
            return Err(SimError::InvalidConfig {
                reason: "t_until must be finite",
            });
        }
        if stepper.state().t != self.t_end() {
            return Err(SimError::InvalidConfig {
                reason: "the stepper must be at the end of the trajectory",
            });
        }
        let budget = stepper.config().max_steps;
        let mut taken = 0;
        while stepper.state().t < t_until {
            if taken >= budget {
                return Err(SimError::MaxStepsExceeded {
                    t: stepper.state().t,
                });
            }
            let segment = stepper.step_bounded(t_until)?;
            self.segments.push(segment);
            taken += 1;
        }
        Ok(())
    }

    /// The time at which the trajectory starts.
    pub fn t_start(&self) -> f64 {
        self.initial.t
    }

    /// The time at which the trajectory ends.
    pub fn t_end(&self) -> f64 {
        self.segments.last().map_or(self.initial.t, |s| s.t1)
    }

    /// `t_end − t_start`.
    pub fn duration(&self) -> f64 {
        self.t_end() - self.t_start()
    }

    /// The number of accepted steps.
    pub fn len_segments(&self) -> usize {
        self.segments.len()
    }

    /// The segments, in order.
    pub fn segments(&self) -> &[Segment<N>] {
        &self.segments
    }

    /// The state at the start: where the first segment starts, or the state
    /// the trajectory was created with if it has no segment yet.
    ///
    /// The two differ only if the first step began from another state than
    /// the trajectory was created with: a stepper given a new state with
    /// [`Stepper::set_state`] before its first step, or data read from
    /// elsewhere. Every other knot reports the state the step after it began
    /// from, and so does the first.
    pub fn start(&self) -> State<N> {
        self.segments.first().map_or(self.initial, Segment::start)
    }

    /// The state at the end.
    pub fn end(&self) -> State<N> {
        self.segments.last().map_or(self.initial, Segment::end)
    }

    /// The states at the ends of the accepted steps, the start included:
    /// the points the step-size controller actually controlled.
    pub fn knots(&self) -> impl Iterator<Item = State<N>> + '_ {
        // A shared knot is taken from the segment that starts there, which
        // reproduces it exactly.
        core::iter::once(self.start())
            .chain(self.segments.iter().skip(1).map(Segment::start))
            .chain(self.segments.last().map(Segment::end))
    }

    /// The index of the segment containing `t`, which must be inside the
    /// trajectory and the trajectory not empty.
    fn locate(&self, t: f64) -> usize {
        self.segments
            .partition_point(|s| s.t1 <= t)
            .min(self.segments.len() - 1)
    }

    /// The state at time `t`, clamped to the trajectory. A `t` that is `NaN`
    /// gives a state of `NaN`s.
    pub fn sample(&self, t: f64) -> State<N> {
        if self.segments.is_empty() || t <= self.t_start() {
            return self.start();
        }
        if t >= self.t_end() {
            return self.end();
        }
        self.segments[self.locate(t)].sample(t)
    }

    /// The state at time `t`, or `None` outside the trajectory.
    pub fn try_sample(&self, t: f64) -> Option<State<N>> {
        (t >= self.t_start() && t <= self.t_end()).then(|| self.sample(t))
    }

    /// The state at time `t`, starting the search from where `cursor` last
    /// was. For playback, where each sample is next to the previous one,
    /// this is constant time in either direction.
    pub fn sample_with_cursor(&self, t: f64, cursor: &mut Cursor) -> State<N> {
        if self.segments.is_empty() || t <= self.t_start() {
            cursor.index = 0;
            return self.start();
        }
        let last = self.segments.len() - 1;
        if t >= self.t_end() {
            cursor.index = last;
            return self.end();
        }
        let mut i = cursor.index.min(last);
        let mut hops = 0;
        loop {
            let seg = &self.segments[i];
            if t < seg.t0 && i > 0 {
                i -= 1;
            } else if t >= seg.t1 && i < last {
                i += 1;
            } else {
                break;
            }
            hops += 1;
            if hops > 8 {
                // A jump, not playback: search instead of walking.
                i = self.locate(t);
                break;
            }
        }
        cursor.index = i;
        self.segments[i].sample(t)
    }

    /// The state at a fraction of the way through: `0` is the start and `1`
    /// the end. Values outside `[0, 1]` are clamped. The natural driver for
    /// an animation timeline.
    pub fn eval_normalized(&self, alpha: f64) -> State<N> {
        let alpha = if alpha.is_nan() {
            0.0
        } else {
            alpha.clamp(0.0, 1.0)
        };
        if alpha >= 1.0 {
            return self.end();
        }
        self.sample(self.t_start() + alpha * self.duration())
    }

    /// The state at time `t` with the trajectory repeated end to end.
    pub fn sample_looped(&self, t: f64) -> State<N> {
        let duration = self.duration();
        if duration.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) || !t.is_finite() {
            return self.start();
        }
        self.sample(self.t_start() + math::rem_euclid(t - self.t_start(), duration))
    }

    /// The state at time `t` with the trajectory played forward, then
    /// backward, then forward again.
    pub fn sample_ping_pong(&self, t: f64) -> State<N> {
        let duration = self.duration();
        if duration.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) || !t.is_finite() {
            return self.start();
        }
        let local = math::rem_euclid(t - self.t_start(), 2.0 * duration);
        let folded = if local > duration {
            2.0 * duration - local
        } else {
            local
        };
        self.sample(self.t_start() + folded)
    }

    /// The generalized momentum at time `t`.
    pub fn momentum<L: Lagrangian<N> + ?Sized>(&self, sys: &L, t: f64) -> [f64; N] {
        let s = self.sample(t);
        mechanics::momentum(sys, s.q, s.dq, s.t)
    }

    /// The point `(q, p)` of phase space at time `t`.
    pub fn phase_point<L: Lagrangian<N> + ?Sized>(&self, sys: &L, t: f64) -> ([f64; N], [f64; N]) {
        let s = self.sample(t);
        (s.q, mechanics::momentum(sys, s.q, s.dq, s.t))
    }

    /// The energy at time `t`.
    pub fn energy<L: Lagrangian<N> + ?Sized>(&self, sys: &L, t: f64) -> Energy {
        let s = self.sample(t);
        mechanics::energy(sys, s.q, s.dq, s.t)
    }

    /// Energy conservation at `num_samples` evenly spaced times, at least 2
    /// and at most 1 048 576 (2²⁰): a larger count is clamped, so that no
    /// argument can keep the call running. Each sample costs one energy
    /// evaluation.
    ///
    /// Evenly spaced times fall **between** the ends of steps, where the
    /// error is that of the interpolating polynomial and is larger than the
    /// solver's own (see [`Segment`]). Use
    /// [`energy_drift_at_knots`](Trajectory::energy_drift_at_knots) to
    /// measure the solver alone.
    pub fn energy_drift<L: Lagrangian<N> + ?Sized>(
        &self,
        sys: &L,
        num_samples: usize,
    ) -> EnergyReport {
        let n = num_samples.clamp(2, MAX_DRIFT_SAMPLES);
        let (t0, duration) = (self.t_start(), self.duration());
        energy_report(
            sys,
            (0..n).map(|k| self.sample(t0 + duration * (k as f64 / (n - 1) as f64))),
        )
    }

    /// Energy conservation at the ends of the accepted steps: the solver's
    /// own error, with no interpolation in it.
    pub fn energy_drift_at_knots<L: Lagrangian<N> + ?Sized>(&self, sys: &L) -> EnergyReport {
        energy_report(sys, self.knots())
    }

    /// Hamilton's action `S = ∫ L dt` along the trajectory, by five-point
    /// Gauss-Legendre quadrature on each segment.
    pub fn action<L: Lagrangian<N> + ?Sized>(&self, sys: &L) -> f64 {
        // Nodes and weights on [-1, 1].
        const NODES: [f64; 5] = [
            0.0,
            -0.538_469_310_105_683_1,
            0.538_469_310_105_683_1,
            -0.906_179_845_938_664,
            0.906_179_845_938_664,
        ];
        const WEIGHTS: [f64; 5] = [
            0.568_888_888_888_888_9,
            0.478_628_670_499_366_5,
            0.478_628_670_499_366_5,
            0.236_926_885_056_189_1,
            0.236_926_885_056_189_1,
        ];
        let mut action = 0.0;
        for seg in &self.segments {
            let mut sum = 0.0;
            for (x, w) in NODES.iter().zip(WEIGHTS) {
                let s = seg.eval(0.5 * (1.0 + x));
                sum += w * sys.lagrangian(s.q, s.dq, s.t);
            }
            action += 0.5 * seg.duration() * sum;
        }
        action
    }

    /// The position, velocity and acceleration of every point of `map` at
    /// time `t`.
    pub fn kinematics<Map, const D: usize, const P: usize>(
        &self,
        map: &Map,
        t: f64,
    ) -> [Kinematics<D>; P]
    where
        Map: SceneMap<N, D, P> + ?Sized,
    {
        mechanics::kinematics(map, &self.sample(t))
    }

    /// Every state at which `indicator` passes through zero in the given
    /// direction, in order of time.
    ///
    /// Events, periods and Poincaré sections all come from this: the
    /// indicator is any function of the state, such as a coordinate, a
    /// velocity or the height of a point. No re-integration is involved; the
    /// roots are found on the stored polynomials by Brent's method.
    ///
    /// ```
    /// use lagrangian::prelude::*;
    /// use lagrangian::systems::KeplerOrbit;
    ///
    /// let orbit = KeplerOrbit::default();
    /// let traj = simulate(&orbit, (0.0, 30.0), [1.0, 0.0], [0.0, 1.2], &SimConfig::default())?;
    /// // Closest approach is where the radial velocity turns from inward to outward.
    /// let perihelia = traj.crossings(|s| s.dq[0], CrossingDirection::Rising);
    /// assert!(perihelia.len() >= 2);
    /// # Ok::<(), SimError>(())
    /// ```
    ///
    /// # Accuracy
    ///
    /// A root is located to about `1e-13` of a step **on the interpolant**.
    /// How close that is to the true event depends on the solver tolerance,
    /// as everything sampled between knots does.
    ///
    /// A sign change is looked for at the ends of each step and at four
    /// points inside it, so two crossings inside one step are both found
    /// unless they fall between the same two of those points. Use
    /// [`crossings_with`](Trajectory::crossings_with) to look more finely.
    pub fn crossings(
        &self,
        indicator: impl Fn(&State<N>) -> f64,
        direction: CrossingDirection,
    ) -> Vec<State<N>> {
        self.crossings_with(indicator, direction, 4)
    }

    /// [`crossings`](Trajectory::crossings), looking for sign changes at
    /// `interior_samples` evenly spaced points inside each step as well as
    /// at its ends.
    ///
    /// `interior_samples` is clamped to 65 536 (2¹⁶), far finer than any
    /// quintic needs, so that no argument can keep the call running. The
    /// indicator is evaluated about `interior_samples + 1` times per step.
    pub fn crossings_with(
        &self,
        indicator: impl Fn(&State<N>) -> f64,
        direction: CrossingDirection,
        interior_samples: usize,
    ) -> Vec<State<N>> {
        // 0.1.0 computed `interior_samples.saturating_add(1)` here, so
        // `usize::MAX` asked for 2⁶⁴ evaluations per step and never returned.
        let pieces = interior_samples.min(MAX_INTERIOR_SAMPLES) + 1;
        let mut found = Vec::new();
        for (index, seg) in self.segments.iter().enumerate() {
            // The value at a shared knot is taken from the segment that
            // starts there on both sides of it, so the two never disagree
            // about the sign of a value that is zero to rounding.
            let end_value = match self.segments.get(index + 1) {
                Some(next) => indicator(&next.start()),
                None => indicator(&seg.end()),
            };
            let mut tau_a = 0.0;
            let mut g_a = indicator(&seg.start());
            for k in 1..=pieces {
                let (tau_b, g_b) = if k == pieces {
                    (1.0, end_value)
                } else {
                    let tau = k as f64 / pieces as f64;
                    (tau, indicator(&seg.eval(tau)))
                };
                let rising = g_a < 0.0 && g_b >= 0.0;
                let falling = g_a > 0.0 && g_b <= 0.0;
                let wanted = match direction {
                    CrossingDirection::Rising => rising,
                    CrossingDirection::Falling => falling,
                    CrossingDirection::Either => rising || falling,
                };
                if wanted {
                    let tau = if g_b == 0.0 {
                        tau_b
                    } else {
                        roots::brent(
                            |tau| indicator(&seg.eval(tau)),
                            tau_a,
                            tau_b,
                            g_a,
                            g_b,
                            1e-13,
                        )
                    };
                    found.push(seg.eval(tau));
                }
                (tau_a, g_a) = (tau_b, g_b);
            }
        }
        found
    }
}

fn energy_report<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    states: impl Iterator<Item = State<N>>,
) -> EnergyReport {
    let mut initial = 0.0;
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut kinetic_max, mut drift_max, mut sum_sq) = (0.0_f64, 0.0_f64, 0.0);
    let mut count = 0usize;
    for s in states {
        let e = mechanics::energy(sys, s.q, s.dq, s.t);
        if count == 0 {
            initial = e.total;
        }
        let drift = e.total - initial;
        min = min.min(e.total);
        max = max.max(e.total);
        kinetic_max = kinetic_max.max(math::abs(e.kinetic));
        drift_max = drift_max.max(math::abs(drift));
        sum_sq += drift * drift;
        count += 1;
    }
    let scale = math::abs(initial).max(kinetic_max);
    EnergyReport {
        initial,
        min,
        max,
        scale,
        max_abs_drift: drift_max,
        // With nothing moving and no energy there is nothing to be relative
        // to, and nothing drifted.
        max_rel_drift: if scale > 0.0 {
            drift_max / scale
        } else {
            drift_max
        },
        rms_drift: if count > 0 {
            math::sqrt(sum_sq / count as f64)
        } else {
            0.0
        },
    }
}
