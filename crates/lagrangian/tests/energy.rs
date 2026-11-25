//! Conservation laws: the energy by Legendre transform, and the discrete
//! Noether theorem of the variational integrators.
#![cfg(feature = "alloc")]

mod common;

use common::{check, rel, sci, tight};
use lagrangian::prelude::*;
use lagrangian::systems::{BeadOnRotatingHoop, DoublePendulum, KeplerOrbit};

/// Relative energy drift at the ends of the steps, and at five points
/// inside each: the solver's error, and the interpolant's.
fn drift<L: Lagrangian<N>, const N: usize>(sys: &L, traj: &Trajectory<N>) -> (f64, f64) {
    let report = traj.energy_drift_at_knots(sys);
    let mut inside = 0.0_f64;
    for segment in traj.segments() {
        for tau in [0.2, 0.35, 0.5, 0.65, 0.8] {
            let s = segment.eval(tau);
            let e = energy(sys, s.q, s.dq, s.t).total;
            inside = inside.max((e - report.initial).abs() / report.scale);
        }
    }
    (report.max_rel_drift, inside)
}

/// The double pendulum every energy test uses: released from rest, high
/// enough for both rods to go over the top.
const CHAOTIC: ([f64; 2], [f64; 2]) = ([2.2, 2.8], [0.0, 0.0]);

#[test]
fn a_chaotic_double_pendulum_conserves_energy_under_dopri5() {
    let dp = DoublePendulum::default();
    let (q0, dq0) = CHAOTIC;

    let traj = simulate(&dp, (0.0, 20.0), q0, dq0, &tight()).unwrap();
    let (knots, inside) = drift(&dp, &traj);
    check(
        "double pendulum energy at knots, t in [0, 20] (tight)",
        knots,
        1e-11,
    );
    check(
        "double pendulum energy mid-step, t in [0, 20] (tight)",
        inside,
        2e-10,
    );

    let traj = simulate(&dp, (0.0, 20.0), q0, dq0, &SimConfig::default()).unwrap();
    let (knots, inside) = drift(&dp, &traj);
    check(
        "double pendulum energy at knots, t in [0, 20] (default)",
        knots,
        5e-8,
    );
    check(
        "double pendulum energy mid-step, t in [0, 20] (default)",
        inside,
        2e-7,
    );

    // What a user gets from evenly spaced samples: mostly mid-step points.
    let uniform = traj.energy_drift(&dp, 2000);
    check(
        "double pendulum energy at 2000 even samples (default)",
        uniform.max_rel_drift,
        2e-7,
    );
    assert!(uniform.rms_drift <= uniform.max_abs_drift);
    assert!(uniform.min <= uniform.initial && uniform.initial <= uniform.max);
}

#[test]
fn the_interpolant_is_less_accurate_than_the_knots() {
    // The claim the documentation makes about mid-step accuracy, held to
    // the code: if the two ever came out equal the warning would be stale.
    let dp = DoublePendulum::default();
    let (q0, dq0) = CHAOTIC;
    let traj = simulate(&dp, (0.0, 20.0), q0, dq0, &tight()).unwrap();
    let (knots, inside) = drift(&dp, &traj);
    println!(
        "measured  mid-step / knot energy error ratio (tight): {:.1}",
        inside / knots
    );
    assert!(inside > 3.0 * knots);
}

#[test]
fn the_rotating_hoop_conserves_its_jacobi_integral() {
    let hoop = BeadOnRotatingHoop::default();
    let traj = simulate(&hoop, (0.0, 20.0), [0.3], [0.0], &tight()).unwrap();
    let (knots, inside) = drift(&hoop, &traj);
    check(
        "rotating hoop Jacobi integral at knots (tight)",
        knots,
        1e-10,
    );
    check(
        "rotating hoop Jacobi integral mid-step (tight)",
        inside,
        1e-10,
    );

    // ½ q̇ᵀ M q̇ + V, the energy a naive implementation would report, is not
    // conserved here: it is off by twice the centrifugal term, which swings
    // by order one as the bead moves.
    let naive = |s: &State<1>| {
        let r2 = hoop.radius * hoop.radius;
        0.5 * hoop.m * r2 * s.dq[0] * s.dq[0]
            + 0.5 * hoop.m * r2 * hoop.omega * hoop.omega * s.q[0].sin().powi(2)
            - hoop.m * hoop.g * hoop.radius * s.q[0].cos()
    };
    let first = naive(&traj.start());
    let swing = traj
        .knots()
        .map(|s| (naive(&s) - first).abs())
        .fold(0.0, f64::max);
    println!(
        "measured  rotating hoop: naive energy swings by {swing:.2} while the Jacobi integral holds"
    );
    assert!(swing > 1.0);
}

#[test]
fn the_hoop_bifurcates_where_the_formula_says() {
    let hoop = BeadOnRotatingHoop::default();
    // cos θ* = g / (R Ω²): a bead placed there stays there.
    let theta = hoop.equilibrium_angle().expect("spinning fast enough");
    assert!(
        rel(
            theta.cos(),
            hoop.g / (hoop.radius * hoop.omega * hoop.omega)
        ) <= 1e-15
    );
    let ddq = accelerations(&hoop, [theta], [0.0], 0.0).unwrap()[0];
    check(
        "rotating hoop: acceleration at the tilted equilibrium",
        ddq.abs(),
        2e-14,
    );

    // The tilted equilibrium is stable and the bottom is not: small
    // oscillations about one, exponential departure from the other.
    let tilted = lagrangian::linearize_free(&hoop, [theta], [0.0], 0.0).unwrap();
    let bottom = lagrangian::linearize_free(&hoop, [0.0], [0.0], 0.0).unwrap();
    assert!(tilted.a_q[0][0] < 0.0, "restoring");
    assert!(bottom.a_q[0][0] > 0.0, "repelling");
    assert!(
        rel(
            bottom.a_q[0][0],
            hoop.omega * hoop.omega - hoop.g / hoop.radius
        ) <= 1e-14
    );
    assert!(
        rel(
            tilted.a_q[0][0],
            -hoop.omega * hoop.omega * theta.sin().powi(2)
        ) <= 1e-14
    );

    // Spinning slowly, the bottom is the only equilibrium and it is stable.
    let slow = BeadOnRotatingHoop { omega: 2.0, ..hoop };
    assert!(slow.equilibrium_angle().is_none());
    assert!(
        lagrangian::linearize_free(&slow, [0.0], [0.0], 0.0)
            .unwrap()
            .a_q[0][0]
            < 0.0
    );
}

/// The largest relative energy error in each quarter of a run.
fn quarters<L: Lagrangian<N>, const N: usize>(sys: &L, traj: &Trajectory<N>) -> [f64; 4] {
    let report = traj.energy_drift_at_knots(sys);
    let n = traj.len_segments() + 1;
    let mut worst = [0.0_f64; 4];
    for (i, s) in traj.knots().enumerate() {
        let e = energy(sys, s.q, s.dq, s.t).total;
        let quarter = (4 * i / n).min(3);
        worst[quarter] = worst[quarter].max((e - report.initial).abs() / report.scale);
    }
    worst
}

#[test]
fn the_variational_integrator_keeps_energy_in_a_band() {
    let dp = DoublePendulum::default();
    let (q0, dq0) = CHAOTIC;
    let run = |dt: f64| {
        let config = SimConfig::variational4(dt);
        simulate(&dp, (0.0, 10_000.0 * dt), q0, dq0, &config).unwrap()
    };

    let fine = run(0.005);
    assert_eq!(fine.len_segments(), 10_000);
    let band = quarters(&dp, &fine);
    let widest = band.iter().copied().fold(0.0, f64::max);
    check(
        "double pendulum energy band, variational4 dt=0.005, 10^4 steps",
        widest,
        3e-4,
    );

    // No secular drift: an error that grew linearly would be four times
    // larger in the last quarter of the run than in the first.
    println!("measured  energy band by quarter: {}", sci(&band));
    assert!(band[3] <= 2.0 * band[0], "the band widened: {band:?}");

    // Fourth order: twice the step, sixteen times the error.
    let coarse = quarters(&dp, &run(0.01))
        .iter()
        .copied()
        .fold(0.0, f64::max);
    println!(
        "measured  band(dt=0.01) / band(dt=0.005) = {:.1} (expect about 16)",
        coarse / widest
    );
    assert!(
        coarse / widest > 8.0,
        "not fourth order: ratio {}",
        coarse / widest
    );
}

#[test]
fn dopri5_drifts_where_the_variational_integrator_does_not() {
    // The reason to offer both. At a loose tolerance the adaptive solver's
    // energy error grows steadily; the symplectic one, with a comparable
    // error per step, stays in its band.
    let orbit = KeplerOrbit::default();
    let (q0, dq0) = ([1.0, 0.0], [0.0, 1.2]);
    let loose = SimConfig::dopri5().with_tolerances(1e-5, 1e-7);
    let adaptive = simulate(&orbit, (0.0, 2000.0), q0, dq0, &loose).unwrap();
    let symplectic = simulate(
        &orbit,
        (0.0, 2000.0),
        q0,
        dq0,
        &SimConfig::variational4(0.1),
    )
    .unwrap();
    let a = quarters(&orbit, &adaptive);
    let s = quarters(&orbit, &symplectic);
    println!(
        "measured  kepler energy error by quarter, dopri5 rtol=1e-5: {}",
        sci(&a)
    );
    println!(
        "measured  kepler energy error by quarter, variational4 dt=0.1: {}",
        sci(&s)
    );
    assert!(a[3] > 3.0 * a[0], "dopri5's error grows");
    assert!(s[3] <= 1.5 * s[0], "the variational band does not");
}

#[test]
fn a_cyclic_coordinate_has_a_conserved_momentum() {
    let orbit = KeplerOrbit::default();
    let (q0, dq0) = ([1.0, 0.0], [0.0, 1.2]);
    let p0 = momentum(&orbit, q0, dq0, 0.0)[1];
    let drift_of = |traj: &Trajectory<2>| {
        traj.knots()
            .map(|s| rel(momentum(&orbit, s.q, s.dq, s.t)[1], p0))
            .fold(0.0, f64::max)
    };

    let config = SimConfig::variational4(0.02);
    let traj = simulate(&orbit, (0.0, 200.0), q0, dq0, &config).unwrap();
    assert_eq!(traj.len_segments(), 10_000);
    check(
        "kepler p_theta, variational4 dt=0.02, 10^4 steps",
        drift_of(&traj),
        5e-15,
    );
    check(
        "kepler energy band, variational4 dt=0.02, 10^4 steps",
        traj.energy_drift_at_knots(&orbit).max_rel_drift,
        5e-8,
    );

    // The adaptive solver conserves it only as well as it solves the
    // equations: good, and three orders of magnitude worse.
    let adaptive = simulate(&orbit, (0.0, 200.0), q0, dq0, &tight()).unwrap();
    check(
        "kepler p_theta, dopri5 (tight), t in [0, 200]",
        drift_of(&adaptive),
        3e-11,
    );
}

#[test]
fn zero_initial_energy_does_not_poison_the_relative_drift() {
    // Released from horizontal, with the potential measured from the pivot:
    // the initial energy is zero to rounding, and a drift relative to it
    // would be meaningless.
    let dp = DoublePendulum::default();
    let level = core::f64::consts::FRAC_PI_2;
    let traj = simulate(&dp, (0.0, 5.0), [level, level], [0.0, 0.0], &tight()).unwrap();
    let report = traj.energy_drift_at_knots(&dp);
    assert!(
        report.initial.abs() < 1e-14,
        "initial energy {}",
        report.initial
    );
    assert!(
        report.scale > 1.0,
        "the scale is the kinetic energy reached"
    );
    check(
        "double pendulum from horizontal: energy relative to scale",
        report.max_rel_drift,
        1e-11,
    );
}

#[test]
fn a_system_at_rest_reports_no_drift() {
    // Nothing moves and nothing is stored: there is nothing to be relative
    // to, and the report must say zero rather than divide by it.
    struct Free;
    impl Lagrangian<1> for Free {
        fn lagrangian<S: Scalar>(&self, _q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5
        }
    }
    let traj = simulate(&Free, (0.0, 1.0), [0.3], [0.0], &SimConfig::default()).unwrap();
    let report = traj.energy_drift(&Free, 50);
    assert_eq!(
        (report.scale, report.max_abs_drift, report.max_rel_drift),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(traj.end().q, [0.3]);
}
