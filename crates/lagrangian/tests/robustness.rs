//! Random and extreme inputs through every public entry point: no panic, no
//! call that fails to return, and no `Ok` that breaks a documented promise.
//!
//! The inputs are mostly ordinary and sometimes extreme: `NaN`, `±∞`,
//! subnormals, `±1e300`, and masses and lengths that are zero or negative.
//! Each seed draws a fixed sequence of cases, so a failure names the seed and
//! the case that reproduces it. Seeds are run in order, `CASES_PER_SEED`
//! cases each, until the time budget is spent:
//!
//! ```sh
//! cargo test -p lagrangian --all-features --test robustness                # the default budget
//! LAGRANGIAN_ROBUSTNESS_SECONDS=300 cargo test -p lagrangian --release --all-features --test robustness
//! LAGRANGIAN_ROBUSTNESS_SEED=17 cargo test -p lagrangian --all-features --test robustness  # start from seed 17
//! ```
//!
//! A watchdog thread fails the run if a case stops making progress, which is
//! how the hangs of #4 were found against 0.1.0. CI runs the default budget
//! on every change and a long one every week (`canary.yml`).
#![cfg(feature = "alloc")]

use std::any::Any;
use std::fmt::Debug;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lagrangian::prelude::*;
use lagrangian::systems::*;
use lagrangian::{linalg, linearize_free, mass_matrix, velocity_from_momentum};

/// The time budget when `LAGRANGIAN_ROBUSTNESS_SECONDS` is not set.
const DEFAULT_SECONDS: f64 = 2.0;
/// Cases drawn from one seed before the next seed starts.
const CASES_PER_SEED: u64 = 400;
/// How long one case may make no progress before the run fails. A debug
/// build of the slowest case allowed below takes about a second.
const WATCHDOG: Duration = Duration::from_secs(60);

const EXTREME: [f64; 21] = [
    0.0,
    -0.0,
    5e-324,
    -5e-324,
    1e-300,
    1e-160,
    -1e-160,
    1e-8,
    1.0,
    -1.0,
    1e8,
    -1e8,
    1e160,
    -1e160,
    1e300,
    -1e300,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

/// SplitMix64: small, and the same on every machine.
struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }
    fn extreme(&mut self) -> f64 {
        EXTREME[self.below(EXTREME.len())]
    }
    /// Mostly ordinary, sometimes extreme.
    fn value(&mut self) -> f64 {
        if self.chance(0.8) {
            self.range(-3.0, 3.0)
        } else {
            self.extreme()
        }
    }
    /// A physical parameter: mostly positive and moderate.
    fn param(&mut self) -> f64 {
        if self.chance(0.9) {
            self.range(0.1, 10.0)
        } else {
            self.extreme()
        }
    }
}

/// What every case reports to: a broken promise is a failure, collected
/// rather than asserted so that one run reports all of them.
#[derive(Default)]
struct Report {
    cases: u64,
    failures: Vec<String>,
}

/// The state the watchdog reads.
struct Shared {
    progress: AtomicU64,
    current: Mutex<String>,
    done: AtomicBool,
}

struct Harness {
    shared: Arc<Shared>,
    report: Report,
    seed: u64,
    index: u64,
}

impl Harness {
    /// Runs one case, catching a panic, and records anything it reports.
    fn case(
        &mut self,
        kind: &str,
        describe: impl FnOnce() -> String,
        f: impl FnOnce(&mut Vec<String>),
    ) {
        let label = format!("seed {} case {} ({kind})", self.seed, self.index);
        *self.shared.current.lock().unwrap() = label.clone();
        let mut broken = Vec::new();
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| f(&mut broken)));
        self.shared.progress.fetch_add(1, Ordering::Relaxed);
        self.report.cases += 1;
        if let Err(payload) = outcome {
            broken.push(format!("panicked: {}", message(payload.as_ref())));
        }
        if !broken.is_empty() {
            let what = describe();
            for b in broken {
                self.report
                    .failures
                    .push(format!("{label}: {b}\n      {what}"));
            }
        }
    }
}

fn message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".into())
}

#[derive(Debug, Clone, Copy)]
enum Sys {
    Dp(DoublePendulum),
    Hoop(BeadOnRotatingHoop),
    Kapitza(KapitzaPendulum),
    Cart(CartPole),
    Kepler(KeplerOrbit),
    Hole(SchwarzschildGeodesic),
}

macro_rules! each_system {
    ($sys:expr, $s:ident => $body:expr) => {
        match $sys {
            Sys::Dp($s) => $body,
            Sys::Hoop($s) => $body,
            Sys::Kapitza($s) => $body,
            Sys::Cart($s) => $body,
            Sys::Kepler($s) => $body,
            Sys::Hole($s) => $body,
        }
    };
}

fn random_system(rng: &mut Rng) -> Sys {
    match rng.below(6) {
        0 => Sys::Dp(DoublePendulum {
            m1: rng.param(),
            m2: rng.param(),
            l1: rng.param(),
            l2: rng.param(),
            g: rng.param(),
        }),
        1 => Sys::Hoop(BeadOnRotatingHoop {
            m: rng.param(),
            radius: rng.param(),
            omega: rng.param(),
            g: rng.param(),
        }),
        2 => Sys::Kapitza(KapitzaPendulum {
            m: rng.param(),
            length: rng.param(),
            g: rng.param(),
            amplitude: rng.param() * 0.01,
            frequency: rng.param() * 20.0,
        }),
        3 => Sys::Cart(CartPole {
            m_cart: rng.param(),
            m_pole: rng.param(),
            length: rng.param(),
            g: rng.param(),
            cart_damping: if rng.chance(0.5) { 0.0 } else { rng.param() },
            pole_damping: if rng.chance(0.5) { 0.0 } else { rng.param() },
        }),
        4 => Sys::Kepler(KeplerOrbit {
            mu: rng.param(),
            m: rng.param(),
        }),
        _ => Sys::Hole(SchwarzschildGeodesic { rs: rng.param() }),
    }
}

/// Coordinates: random, and most of the time where the system is regular.
fn coordinates<const N: usize>(tag: &Sys, rng: &mut Rng) -> [f64; N] {
    let mut q: [f64; N] = core::array::from_fn(|_| rng.value());
    if rng.chance(0.7) {
        match tag {
            Sys::Kepler(_) => q[0] = q[0].abs() + 0.1,
            Sys::Hole(h) => {
                q[1] = h.rs.abs() * (1.5 + q[1].abs());
                q[2] = rng.range(0.3, 2.8);
            }
            _ => {}
        }
    }
    q
}

fn finite<const N: usize>(x: &[f64; N]) -> bool {
    x.iter().all(|v| v.is_finite())
}

fn engine<L: Lagrangian<N> + Debug, const N: usize>(
    h: &mut Harness,
    sys: &L,
    tag: &Sys,
    rng: &mut Rng,
) {
    let q: [f64; N] = coordinates(tag, rng);
    let dq: [f64; N] = core::array::from_fn(|_| rng.value());
    let guess: [f64; N] = core::array::from_fn(|_| rng.value());
    let t = rng.value();
    h.case(
        "engine",
        || format!("{sys:?} q={q:?} dq={dq:?} t={t:?}"),
        |broken| {
            if let Ok(a) = accelerations(sys, q, dq, t) {
                if !finite(&a) {
                    broken.push("accelerations returned Ok with a value that is not finite".into());
                }
            }
            let terms = eom_terms(sys, q, dq, t);
            let _ = (
                energy(sys, q, dq, t),
                mass_matrix(sys, q, dq, t),
                momentum(sys, q, dq, t),
            );
            if let Ok(l) = linearize_free(sys, q, dq, t) {
                if !l.a_q.iter().chain(l.a_dq.iter()).all(finite) {
                    broken
                        .push("linearize_free returned Ok with a value that is not finite".into());
                }
            }
            if let Ok(v) = velocity_from_momentum(sys, q, terms.momentum, t, guess) {
                if !finite(&v) {
                    broken.push(
                        "velocity_from_momentum returned Ok with a value that is not finite".into(),
                    );
                }
            }
        },
    );
}

fn controlled(h: &mut Harness, cart: &CartPole, rng: &mut Rng) {
    let tag = Sys::Cart(*cart);
    let q: [f64; 2] = coordinates(&tag, rng);
    let dq = [rng.value(), rng.value()];
    let (u, t) = (rng.value(), rng.value());
    h.case(
        "control",
        || format!("{cart:?} q={q:?} dq={dq:?} u={u:?} t={t:?}"),
        |broken| {
            if let Ok(l) = linearize(cart, q, dq, [u], t) {
                if !(l.a_q.iter().chain(l.a_dq.iter()).all(finite) && l.b.iter().all(finite)) {
                    broken.push("linearize returned Ok with a value that is not finite".into());
                }
            }
            if let Ok(a) = accelerations(&WithInput { sys: cart, u: [u] }, q, dq, t) {
                if !finite(&a) {
                    broken.push(
                        "accelerations with an input returned Ok with a value that is not finite"
                            .into(),
                    );
                }
            }
        },
    );
}

fn scene_one<M, const N: usize, const D: usize, const P: usize>(
    h: &mut Harness,
    map: &M,
    tag: &Sys,
    rng: &mut Rng,
) where
    M: SceneMap<N, D, P> + Debug,
{
    let state = State {
        t: rng.value(),
        q: coordinates::<N>(tag, rng),
        dq: core::array::from_fn(|_| rng.value()),
        ddq: core::array::from_fn(|_| rng.value()),
    };
    h.case(
        "scene",
        || format!("{map:?} {state:?}"),
        |_| {
            let _ = kinematics(map, &state);
        },
    );
}

fn scene(h: &mut Harness, sys: Sys, rng: &mut Rng) {
    match sys {
        Sys::Dp(s) => scene_one::<_, 2, 2, 2>(h, &s, &sys, rng),
        Sys::Hoop(s) => scene_one::<_, 1, 3, 1>(h, &s, &sys, rng),
        Sys::Kapitza(s) => scene_one::<_, 1, 2, 2>(h, &s, &sys, rng),
        Sys::Cart(s) => scene_one::<_, 2, 2, 2>(h, &s, &sys, rng),
        Sys::Kepler(s) => scene_one::<_, 2, 2, 1>(h, &s, &sys, rng),
        Sys::Hole(s) => scene_one::<_, 4, 3, 1>(h, &s, &sys, rng),
    }
}

fn random_config(rng: &mut Rng) -> SimConfig {
    let dt = |rng: &mut Rng| {
        if rng.chance(0.85) {
            rng.range(1e-3, 0.1)
        } else {
            rng.extreme()
        }
    };
    let tol = |rng: &mut Rng| {
        if rng.chance(0.7) {
            10_f64.powf(rng.range(-14.0, -3.0))
        } else {
            rng.extreme()
        }
    };
    let step = |rng: &mut Rng| {
        if rng.chance(0.7) {
            10_f64.powf(rng.range(-8.0, 1.0))
        } else {
            rng.extreme()
        }
    };
    let mut c = match rng.below(10) {
        0..=5 => SimConfig::dopri5(),
        6..=7 => SimConfig::variational2(dt(rng)),
        _ => SimConfig::variational4(dt(rng)),
    };
    if rng.chance(0.3) {
        c = c.with_tolerances(tol(rng), tol(rng));
    }
    if rng.chance(0.2) {
        c = c.with_max_step(step(rng));
    }
    if rng.chance(0.15) {
        c = c.with_min_step(step(rng));
    }
    if rng.chance(0.2) {
        c = c.with_initial_step(step(rng));
    }
    c
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Step,
    Bounded(f64),
    Set(u64),
}

/// What a segment a step returned promises (`Segment`, `Stepper::step`).
fn check_segment<const N: usize>(
    seg: &Segment<N>,
    before: &State<N>,
    target: Option<f64>,
    after: &State<N>,
    broken: &mut Vec<String>,
) {
    if seg.t_end().partial_cmp(&seg.t_start()) != Some(core::cmp::Ordering::Greater) {
        broken.push("a step returned a segment that does not run forward".into());
    }
    if seg.t_start() != before.t || after.t != seg.t_end() {
        broken.push("a segment does not run from the stepper's time to its new one".into());
    }
    if target.is_some_and(|tm| seg.t_end() > tm) {
        broken.push("step_bounded stepped past its bound".into());
    }
    if !after.is_finite() {
        broken.push("a step returned Ok and left a state that is not finite".into());
    }
    // Exact at its start, but for the documented exception: an
    // acceleration whose half is subnormal can lose its last bit to the
    // halving. The harness found one just above `f64::MIN_POSITIVE`.
    let start = seg.start();
    let close =
        |a: f64, b: f64| a == b || (b.abs() < 2.0 * f64::MIN_POSITIVE && (a - b).abs() <= 5e-324);
    let exact = start.t == before.t
        && start.q == before.q
        && start.dq == before.dq
        && start.ddq.iter().zip(before.ddq).all(|(&a, b)| close(a, b));
    if !exact {
        broken.push("a segment's start is not the knot it started from".into());
    }
    for tau in [0.5, 1.0] {
        if !seg.eval(tau).is_finite() {
            broken.push("a segment is not finite between its ends".into());
        }
    }
}

fn stepper<L: Lagrangian<N> + Debug + Copy, const N: usize>(
    h: &mut Harness,
    sys: L,
    tag: &Sys,
    rng: &mut Rng,
) {
    let q: [f64; N] = coordinates(tag, rng);
    let dq: [f64; N] = core::array::from_fn(|_| rng.value());
    let t0 = if rng.chance(0.9) {
        rng.range(-10.0, 10.0)
    } else {
        rng.extreme()
    };
    let config = random_config(rng);
    let ops: Vec<Op> = (0..1 + rng.below(40))
        .map(|_| match rng.below(10) {
            0..=3 => Op::Step,
            4..=8 => Op::Bounded(if rng.chance(0.85) {
                10_f64.powf(rng.range(-6.0, 1.0))
            } else {
                rng.extreme()
            }),
            _ => Op::Set(rng.u64()),
        })
        .collect();
    let tag = *tag;
    h.case(
        "stepper",
        || format!("{sys:?} t0={t0:?} q={q:?} dq={dq:?} {config:?} ops={ops:?}"),
        |broken| {
            let Ok(mut s) = Stepper::new(sys, t0, q, dq, config) else {
                return;
            };
            if !s.state().is_finite() {
                broken.push("Stepper::new returned Ok with a state that is not finite".into());
            }
            for op in &ops {
                let before = s.state();
                let outcome = match *op {
                    Op::Step => s.step().map(|seg| Some((seg, None))),
                    Op::Bounded(d) => {
                        let target = before.t + d;
                        s.step_bounded(target).map(|seg| Some((seg, Some(target))))
                    }
                    Op::Set(seed) => {
                        let mut g = Rng(seed);
                        let q2: [f64; N] = coordinates(&tag, &mut g);
                        let dq2: [f64; N] = core::array::from_fn(|_| g.value());
                        s.set_state(q2, dq2).map(|()| None)
                    }
                };
                match outcome {
                    Ok(Some((seg, target))) => {
                        check_segment(&seg, &before, target, &s.state(), broken)
                    }
                    Ok(None) if !s.state().is_finite() => {
                        broken.push("set_state returned Ok with a state that is not finite".into());
                    }
                    Ok(None) => {}
                    Err(_) if s.state() != before => {
                        broken.push("an operation that returned an error moved the stepper".into());
                    }
                    Err(_) => {}
                }
            }
        },
    );
}

/// What a trajectory promises, whether `simulate` made it or it was read.
fn check_trajectory<L: Lagrangian<N>, const N: usize>(
    sys: &L,
    traj: &Trajectory<N>,
    ts: &[f64],
    rng: &mut Rng,
    broken: &mut Vec<String>,
) {
    if traj
        .segments()
        .windows(2)
        .any(|w| w[0].t_end() != w[1].t_start())
    {
        broken.push("segments are not contiguous".into());
    }
    if !traj.duration().is_finite() {
        broken.push("a trajectory's duration is not finite".into());
    }
    let knots_finite = traj.knots().all(|k| k.is_finite());
    let mut cursor = Cursor::default();
    for &t in ts {
        let s = traj.sample(t);
        if s.is_finite() && s != traj.sample_with_cursor(t, &mut cursor) {
            broken.push("sample_with_cursor disagrees with sample".into());
        }
        if knots_finite && t >= traj.t_start() && t <= traj.t_end() && !s.is_finite() {
            broken.push("a trajectory with finite knots samples to a state that is not".into());
        }
        let _ = (
            traj.try_sample(t),
            traj.sample_looped(t),
            traj.sample_ping_pong(t),
        );
        let _ = (
            traj.momentum(sys, t),
            traj.phase_point(sys, t),
            traj.energy(sys, t),
        );
    }
    for alpha in [0.0, 0.25, 1.0, -1.0, 2.0, f64::NAN] {
        let _ = traj.eval_normalized(alpha);
    }
    let samples = if rng.chance(0.1) {
        usize::MAX
    } else {
        2 + rng.below(60)
    };
    let _ = (
        traj.energy_drift(sys, samples),
        traj.energy_drift_at_knots(sys),
        traj.action(sys),
    );
    let direction = [
        CrossingDirection::Rising,
        CrossingDirection::Falling,
        CrossingDirection::Either,
    ][rng.below(3)];
    let level = traj.sample(0.5 * (traj.t_start() + traj.t_end())).q[0];
    let found = traj.crossings(|s| s.q[0] - level, direction);
    if found.windows(2).any(|w| w[0].t > w[1].t) {
        broken.push("crossings are not in order of time".into());
    }
    if found
        .iter()
        .any(|s| !(s.t >= traj.t_start() && s.t <= traj.t_end()))
    {
        broken.push("a crossing lies outside the trajectory".into());
    }
    let interior = if rng.chance(0.05) {
        usize::MAX
    } else {
        rng.below(12)
    };
    if traj.len_segments() <= 64 {
        let _ = traj.crossings_with(|s| s.dq[N - 1], direction, interior);
    }
}

fn simulate_case<L: Lagrangian<N> + Debug + Copy, const N: usize>(
    h: &mut Harness,
    sys: L,
    tag: &Sys,
    rng: &mut Rng,
) {
    let q: [f64; N] = coordinates(tag, rng);
    let dq: [f64; N] = core::array::from_fn(|_| rng.value());
    let t0 = if rng.chance(0.9) {
        rng.range(-5.0, 5.0)
    } else {
        rng.extreme()
    };
    let duration = if rng.chance(0.85) {
        rng.range(0.0, 5.0)
    } else {
        rng.extreme()
    };
    let t1 = t0 + duration;
    let config = random_config(rng).with_max_steps(3000);
    let ts: Vec<f64> = (0..12)
        .map(|_| {
            if rng.chance(0.6) {
                t0 + rng.unit() * duration
            } else {
                rng.extreme()
            }
        })
        .collect();
    let mut draws = Rng(rng.u64());
    h.case(
        "simulate",
        || format!("{sys:?} span=({t0:?}, {t1:?}) q={q:?} dq={dq:?} {config:?}"),
        |broken| {
            let Ok(traj) = simulate(&sys, (t0, t1), q, dq, &config) else {
                return;
            };
            if traj.t_end() != t1 {
                broken.push("simulate returned Ok short of its end time".into());
            }
            if !traj.knots().all(|k| k.is_finite()) {
                broken.push("simulate returned Ok with a knot that is not finite".into());
            }
            check_trajectory(&sys, &traj, &ts, &mut draws, broken);
        },
    );
}

fn linalg_case<const N: usize>(h: &mut Harness, rng: &mut Rng) {
    let scale = if rng.chance(0.2) {
        10_f64.powf(rng.range(-300.0, 300.0))
    } else {
        1.0
    };
    let a: [[f64; N]; N] = core::array::from_fn(|_| core::array::from_fn(|_| rng.value() * scale));
    let b: [f64; N] = core::array::from_fn(|_| rng.value());
    h.case(
        "linalg",
        || format!("a={a:?} b={b:?}"),
        |broken| {
            // The fixed-size paths and the general factorization do the same
            // arithmetic in the same order, so they agree to the bit.
            let fast = linalg::solve(&a, &b);
            let general = linalg::Lu::factor(&a).map(|lu| lu.solve(&b));
            let same = match (fast, general) {
                (Some(x), Some(y)) => x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()),
                (None, None) => true,
                _ => false,
            };
            if !same {
                broken.push(format!(
                    "linalg::solve and Lu disagree: {fast:?} against {general:?}"
                ));
            }
            let finite_input = a.iter().flatten().all(|x| x.is_finite());
            if !finite_input && fast.is_some() {
                broken.push("a matrix with a value that is not finite was solved".into());
            }
        },
    );
}

#[cfg(feature = "serde")]
mod files {
    use super::*;

    const REPLACEMENTS: [&str; 14] = [
        "0",
        "-0",
        "1",
        "-1",
        "0.5",
        "1e-15",
        "5e-324",
        "1e-300",
        "1e300",
        "1e308",
        "-1e308",
        "1.7976931348623157e308",
        "-1.7976931348623157e308",
        "123456789",
    ];

    /// `base` with one to three of its numbers replaced, and sometimes cut
    /// short, and what was done to it.
    fn mutate(base: &str, rng: &mut Rng) -> (String, String) {
        let bytes = base.as_bytes();
        let mut numbers = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'-' || bytes[i].is_ascii_digit() {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || b".eE+-".contains(&bytes[i]))
                {
                    i += 1;
                }
                numbers.push((start, i));
            } else {
                i += 1;
            }
        }
        let mut chosen: Vec<(usize, usize, &str)> = (0..1 + rng.below(3))
            .map(|_| {
                let (s, e) = numbers[rng.below(numbers.len())];
                (s, e, REPLACEMENTS[rng.below(REPLACEMENTS.len())])
            })
            .collect();
        chosen.sort_by_key(|c| std::cmp::Reverse(c.0));
        chosen.dedup_by_key(|c| c.0);
        let mut out = base.to_string();
        let mut what = Vec::new();
        for (s, e, with) in &chosen {
            what.push(format!("{} -> {with}", &base[*s..*e]));
            out.replace_range(*s..*e, with);
        }
        if rng.chance(0.1) {
            let at = rng.below(out.len());
            out.truncate(at);
            what.push(format!("cut at {at}"));
        }
        (out, what.join("; "))
    }

    pub(super) fn read(h: &mut Harness, base: &str, rng: &mut Rng) {
        let (json, what) = mutate(base, rng);
        let ts: Vec<f64> = (0..8)
            .map(|_| {
                if rng.chance(0.6) {
                    rng.range(0.0, 0.3)
                } else {
                    rng.extreme()
                }
            })
            .collect();
        let mut draws = Rng(rng.u64());
        h.case(
            "serde",
            || what.clone(),
            |broken| {
                let Ok(traj) = serde_json::from_str::<Trajectory<2>>(&json) else {
                    return;
                };
                if !traj
                    .segments()
                    .iter()
                    .all(|s| s.eval(0.5).is_finite() && s.end().is_finite())
                {
                    broken.push("a trajectory was read whose segments are not finite".into());
                }
                check_trajectory(&DoublePendulum::default(), &traj, &ts, &mut draws, broken);
            },
        );
    }
}

/// One seed's sequence of cases.
fn run_seed(h: &mut Harness, base: Option<&str>, deadline: Instant) {
    let mut rng = Rng(h.seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xDEAD_BEEF);
    for index in 0..CASES_PER_SEED {
        if Instant::now() >= deadline {
            return;
        }
        h.index = index;
        match rng.below(20) {
            0..=6 => {
                let sys = random_system(&mut rng);
                each_system!(sys, s => engine(h, &s, &sys, &mut rng));
                if let Sys::Cart(cart) = sys {
                    controlled(h, &cart, &mut rng);
                }
                scene(h, sys, &mut rng);
            }
            7..=11 => {
                let sys = random_system(&mut rng);
                each_system!(sys, s => stepper(h, s, &sys, &mut rng));
            }
            12..=14 => {
                let sys = random_system(&mut rng);
                each_system!(sys, s => simulate_case(h, s, &sys, &mut rng));
            }
            15..=16 => match rng.below(6) {
                0 => linalg_case::<1>(h, &mut rng),
                1 => linalg_case::<2>(h, &mut rng),
                2 => linalg_case::<3>(h, &mut rng),
                3 => linalg_case::<4>(h, &mut rng),
                4 => linalg_case::<5>(h, &mut rng),
                _ => linalg_case::<8>(h, &mut rng),
            },
            _ => {
                #[cfg(feature = "serde")]
                if let Some(base) = base {
                    files::read(h, base, &mut rng);
                }
                #[cfg(not(feature = "serde"))]
                let _ = base;
            }
        }
    }
}

fn env_number(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

#[test]
fn random_and_extreme_inputs_keep_every_promise() {
    let seconds = env_number("LAGRANGIAN_ROBUSTNESS_SECONDS").unwrap_or(DEFAULT_SECONDS);
    let first_seed = env_number("LAGRANGIAN_ROBUSTNESS_SEED").map_or(1, |s| s as u64);
    let shared = Arc::new(Shared {
        progress: AtomicU64::new(0),
        current: Mutex::new(String::new()),
        done: AtomicBool::new(false),
    });

    // A case that never returns cannot report itself. The watchdog does,
    // and ends the process, which fails the test.
    let watched = Arc::clone(&shared);
    let watchdog = std::thread::spawn(move || {
        let (mut last, mut since) = (u64::MAX, Instant::now());
        while !watched.done.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
            let now = watched.progress.load(Ordering::Relaxed);
            if now != last {
                (last, since) = (now, Instant::now());
            } else if since.elapsed() > WATCHDOG {
                eprintln!(
                    "robustness: {} made no progress for {} s, a hang",
                    watched.current.lock().unwrap(),
                    WATCHDOG.as_secs()
                );
                std::process::exit(101);
            }
        }
    });

    #[cfg(feature = "serde")]
    let base = {
        let traj = simulate(
            &DoublePendulum::default(),
            (0.0, 0.3),
            [1.0, 2.0],
            [0.0, 0.0],
            &SimConfig::default(),
        )
        .unwrap();
        Some(serde_json::to_string(&traj).unwrap())
    };
    #[cfg(not(feature = "serde"))]
    let base: Option<String> = None;

    // Caught panics are reported below with their case; the default hook
    // would print each of them a second time.
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let start = Instant::now();
    let deadline = start + Duration::from_secs_f64(seconds);
    let mut harness = Harness {
        shared: Arc::clone(&shared),
        report: Report::default(),
        seed: first_seed,
        index: 0,
    };
    while Instant::now() < deadline {
        run_seed(&mut harness, base.as_deref(), deadline);
        harness.seed += 1;
    }
    panic::set_hook(hook);
    shared.done.store(true, Ordering::Relaxed);
    watchdog.join().unwrap();

    let report = harness.report;
    println!(
        "robustness: {} cases from seeds {first_seed} to {} in {:.1} s",
        report.cases,
        harness.seed - 1,
        start.elapsed().as_secs_f64()
    );
    assert!(
        report.failures.is_empty(),
        "{} broken promise(s):\n  {}",
        report.failures.len(),
        report.failures.join("\n  ")
    );
}
