//! Shared by the verification suite: a deterministic generator and two
//! systems simple enough to have closed-form solutions.
#![allow(dead_code)]

use lagrangian::EomTerms;
use lagrangian::ad::{Dual, HyperDual};
use lagrangian::prelude::*;

/// A small deterministic generator: the suite must not depend on a crate,
/// and must draw the same states on every machine.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    /// Uniform in `[-1, 1)`.
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Uniform in `[lo, hi)`.
    pub(crate) fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * 0.5 * (self.next() + 1.0)
    }
}

/// `|a − b|` relative to the larger of one and their magnitudes.
pub(crate) fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / 1.0_f64.max(a.abs()).max(b.abs())
}

/// A point mass on a rod: the angle is measured from straight down.
pub(crate) struct Pendulum {
    pub(crate) l: f64,
    pub(crate) g: f64,
}

impl Lagrangian<1> for Pendulum {
    fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
        (dq[0] * self.l).powi(2) * 0.5 + q[0].cos() * (self.g * self.l)
    }
}

impl Pendulum {
    /// The exact period for amplitude `theta0`:
    /// `4·sqrt(l/g)·K(sin(θ₀/2))`, with the complete elliptic integral from
    /// the arithmetic-geometric mean.
    pub(crate) fn period(&self, theta0: f64) -> f64 {
        let (mut a, mut b) = (1.0_f64, (theta0 / 2.0).cos());
        for _ in 0..40 {
            (a, b) = (0.5 * (a + b), (a * b).sqrt());
        }
        4.0 * (self.l / self.g).sqrt() * core::f64::consts::PI / (2.0 * a)
    }
}

/// Two unit masses, each tied to a wall by a spring `k` and to each other
/// by a spring `kc`.
pub(crate) struct CoupledOscillators {
    pub(crate) k: f64,
    pub(crate) kc: f64,
}

impl Lagrangian<2> for CoupledOscillators {
    fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], _t: S) -> S {
        let kinetic = (dq[0].powi(2) + dq[1].powi(2)) * 0.5;
        let potential = (q[0].powi(2) + q[1].powi(2)) * (0.5 * self.k)
            + (q[0] - q[1]).powi(2) * (0.5 * self.kc);
        kinetic - potential
    }
}

impl CoupledOscillators {
    /// The solution from `x = (1, 0)` at rest: the sum and difference of the
    /// two coordinates are independent oscillators.
    pub(crate) fn exact(&self, t: f64) -> [f64; 2] {
        let sum = 0.5 * (self.k.sqrt() * t).cos();
        let diff = 0.5 * ((self.k + 2.0 * self.kc).sqrt() * t).cos();
        [sum + diff, sum - diff]
    }
}

/// "Tight" tolerances: what the verification suite uses when it measures
/// the solver rather than the default configuration.
pub(crate) fn tight() -> SimConfig {
    SimConfig::dopri5().with_tolerances(1e-12, 1e-14)
}

/// Asserts a measured error against its limit, and prints both.
///
/// Every limit in the suite is about ten times a value measured on this
/// implementation (`PLAN.md` §7). Printing the measurement next to the limit
/// is what keeps that true: `cargo test -- --nocapture | grep measured`
/// re-reads every baseline at once, and a margin that has shrunk towards one
/// is visible before the test goes red.
#[track_caller]
pub(crate) fn check(name: &str, measured: f64, limit: f64) {
    println!(
        "measured  {name:<62} {measured:9.2e}   limit {limit:7.0e}   margin {:6.1}x",
        limit / measured
    );
    // Written so that a NaN fails.
    assert!(
        measured <= limit,
        "{name}: measured {measured:e} exceeds the limit {limit:e}"
    );
}

/// A list of small numbers, readably.
pub(crate) fn sci(values: &[f64]) -> String {
    values
        .iter()
        .map(|x| format!("{x:.2e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A hyperdual seed: the value, whether it is the inner variable, whether it
/// is the outer variable.
pub(crate) fn seed(value: f64, inner: bool, outer: bool) -> HyperDual {
    let one = |on: bool| if on { 1.0 } else { 0.0 };
    Dual::new(Dual::new(value, one(inner)), Dual::new(one(outer), 0.0))
}

/// Every term of the equations of motion, element by element: one hyperdual
/// evaluation per second derivative.
pub(crate) fn oracle<L: Lagrangian<N>, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> EomTerms<N> {
    // The Lagrangian with coordinate `a` of the velocities as the inner
    // variable and one of (velocity b, coordinate b, time) as the outer.
    let eval = |a: usize, outer_dq: Option<usize>, outer_q: Option<usize>, outer_t: bool| {
        sys.lagrangian(
            core::array::from_fn(|i| seed(q[i], false, outer_q == Some(i))),
            core::array::from_fn(|i| seed(dq[i], i == a, outer_dq == Some(i))),
            seed(t, false, outer_t),
        )
    };

    let mut terms = EomTerms {
        lagrangian: sys.lagrangian(q, dq, t),
        mass: [[0.0; N]; N],
        grad_q: [0.0; N],
        momentum: [0.0; N],
        convective: [0.0; N],
        force: sys.generalized_force(q, dq, t),
        dl_dt: 0.0,
    };
    for i in 0..N {
        for j in 0..N {
            terms.mass[i][j] = eval(i, Some(j), None, false).eps.eps;
            terms.convective[i] += eval(i, None, Some(j), false).eps.eps * dq[j];
        }
        let by_time = eval(i, None, None, true);
        terms.convective[i] += by_time.eps.eps;
        terms.momentum[i] = by_time.val.eps;
        terms.dl_dt = by_time.eps.val;
        terms.grad_q[i] = eval(0, None, Some(i), false).eps.val;
    }
    terms
}

/// The largest relative difference between the engine and the oracle.
pub(crate) fn mismatch<L: Lagrangian<N>, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> f64 {
    let engine = eom_terms(sys, q, dq, t);
    let reference = oracle(sys, q, dq, t);
    let mut worst =
        rel(engine.lagrangian, reference.lagrangian).max(rel(engine.dl_dt, reference.dl_dt));
    for i in 0..N {
        worst = worst
            .max(rel(engine.grad_q[i], reference.grad_q[i]))
            .max(rel(engine.momentum[i], reference.momentum[i]))
            .max(rel(engine.convective[i], reference.convective[i]))
            .max(rel(engine.force[i], reference.force[i]));
        for j in 0..N {
            worst = worst.max(rel(engine.mass[i][j], reference.mass[i][j]));
            assert_eq!(
                engine.mass[i][j], engine.mass[j][i],
                "the mass matrix is symmetric"
            );
        }
    }
    assert!(
        worst.is_finite(),
        "a term is not finite at q = {q:?}, dq = {dq:?}"
    );
    worst
}

/// The state at `t_end`, by stepping: no trajectory and no allocation, so
/// the suites that use it also run in the build without an allocator.
pub(crate) fn advance<L: Lagrangian<N>, const N: usize>(
    sys: &L,
    q0: [f64; N],
    dq0: [f64; N],
    t_end: f64,
    config: SimConfig,
) -> State<N> {
    let mut stepper = Stepper::new(sys, 0.0, q0, dq0, config).expect("a valid start");
    while stepper.state().t < t_end {
        stepper.step_bounded(t_end).expect("a step");
    }
    stepper.state()
}

/// Runs `f` on its own thread and returns its result, or fails the test if
/// it has not returned after `seconds`: a call that never returns would
/// otherwise hang the whole suite instead of failing one test. A thread that
/// is still spinning is left behind; the test binary's exit ends it.
#[track_caller]
pub(crate) fn within<T: Send + 'static>(
    seconds: u64,
    what: &str,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = send.send(f());
    });
    match receive.recv_timeout(std::time::Duration::from_secs(seconds)) {
        Ok(value) => value,
        Err(_) => panic!("{what}: did not return within {seconds} s"),
    }
}
