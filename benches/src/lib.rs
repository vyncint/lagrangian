//! What the benchmarks share: a timing loop with no dependencies, a system
//! whose number of coordinates is a type parameter, the double pendulum's
//! equations derived by hand, and a counter of Lagrangian evaluations.
//!
//! ```sh
//! cargo bench -p benches                 # everything, about a minute
//! cargo bench -p benches -- steppers     # only those whose name contains it
//! ```
//!
//! A time is the median of seven batches of at least 50 ms each, in a
//! release build. CI builds the benchmarks and runs none of them: a time
//! measured on a shared runner is not a baseline. The baselines are in
//! `PLAN.md` §7.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use lagrangian::Lagrangian;
use lagrangian::Scalar;
use lagrangian::systems::DoublePendulum;

/// Whether the benchmark called `name` was asked for: every one, unless an
/// argument that is not a flag names part of it.
pub fn wanted(name: &str) -> bool {
    std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .is_none_or(|filter| name.contains(&filter))
}

/// Median nanoseconds per call of `f`, over seven batches of at least
/// 50 ms each.
pub fn time<R>(mut f: impl FnMut() -> R) -> f64 {
    let mut iterations = 1_u64;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(f());
        }
        if start.elapsed().as_secs_f64() > 0.05 {
            break;
        }
        iterations *= 2;
    }
    let mut samples: Vec<f64> = (0..7)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(f());
            }
            start.elapsed().as_secs_f64() * 1e9 / iterations as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[3]
}

/// `N` masses in a line tied by springs. Each mass depends on its own
/// position, so the mass matrix is not constant and there is one cosine
/// per coordinate; each spring stiffens quartically.
#[derive(Debug, Clone, Copy, Default)]
pub struct Springs<const N: usize>;

impl<const N: usize> Lagrangian<N> for Springs<N> {
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], _t: S) -> S {
        let mut total = S::ZERO;
        let mut left = S::ZERO;
        for i in 0..N {
            let mass = q[i].cos() * 0.25 + 1.0;
            let stretch = q[i] - left;
            let s2 = stretch * stretch;
            total += dq[i] * dq[i] * mass - s2 - s2 * s2 * 0.1;
            left = q[i];
        }
        total * 0.5
    }
}

/// A state of [`Springs`] with every coordinate displaced and moving.
pub fn springs_state<const N: usize>() -> ([f64; N], [f64; N]) {
    (
        core::array::from_fn(|i| 0.3 + 0.01 * i as f64),
        core::array::from_fn(|i| 0.2 - 0.01 * i as f64),
    )
}

/// The double pendulum's accelerations from the closed form found in
/// textbooks: what the engine is compared with.
pub fn double_pendulum_by_hand(p: &DoublePendulum, q: [f64; 2], dq: [f64; 2]) -> [f64; 2] {
    let (m1, m2, l1, l2, g) = (p.m1, p.m2, p.l1, p.l2, p.g);
    let ([t1, t2], [w1, w2]) = (q, dq);
    let d = t1 - t2;
    let (sd, cd) = d.sin_cos();
    let den = 2.0 * m1 + m2 - m2 * (2.0 * d).cos();
    let a1 = (-g * (2.0 * m1 + m2) * t1.sin()
        - m2 * g * (t1 - 2.0 * t2).sin()
        - 2.0 * sd * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cd))
        / (l1 * den);
    let a2 =
        (2.0 * sd * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * t1.cos() + w2 * w2 * l2 * m2 * cd))
            / (l2 * den);
    [a1, a2]
}

/// A system that counts the evaluations of the one it wraps, by the scalar
/// type they were made with.
#[derive(Debug)]
pub struct Counted<L> {
    inner: L,
    calls: RefCell<BTreeMap<&'static str, u64>>,
}

impl<L> Counted<L> {
    /// Wraps `inner`, with every count at zero.
    pub fn new(inner: L) -> Self {
        Self {
            inner,
            calls: RefCell::new(BTreeMap::new()),
        }
    }

    /// The counts so far, which start again from zero.
    pub fn take(&self) -> Counts {
        Counts(core::mem::take(&mut *self.calls.borrow_mut()))
    }
}

impl<L: Lagrangian<N>, const N: usize> Lagrangian<N> for Counted<L> {
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> S {
        *self
            .calls
            .borrow_mut()
            .entry(std::any::type_name::<S>())
            .or_insert(0) += 1;
        self.inner.lagrangian(q, dq, t)
    }

    fn generalized_force<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> [S; N] {
        self.inner.generalized_force(q, dq, t)
    }
}

/// Evaluations of a Lagrangian, by scalar type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counts(BTreeMap<&'static str, u64>);

impl Counts {
    /// All of them.
    pub fn total(&self) -> u64 {
        self.0.values().sum()
    }

    /// `"Dual<Grad<2>> 18.5, Grad<2> 6.2"`: each count divided by `per`,
    /// the most expensive type first.
    pub fn per(&self, per: f64) -> String {
        let mut rows: Vec<(String, f64)> = self
            .0
            .iter()
            .map(|(name, n)| (name.replace("lagrangian::ad::", ""), *n as f64 / per))
            .collect();
        rows.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
        rows.iter()
            .map(|(name, n)| format!("{name} {n:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}
